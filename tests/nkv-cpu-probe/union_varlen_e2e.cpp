// CPU-only end-to-end driver for the union/varlen binder.
//
// Two context arms share one model and one batch schedule, and both run real
// llama_decode() calls with a variable token width, a growing n_kv and a
// variable active output count.
//
//   main   n_seq_max = 1, kv_unified = true, one sequence. This is the
//          configuration src/llama-context.cpp documents as supported: the
//          LLAMA_EXPERIMENTAL_VARLEN_NTOK envelope requires kv_unified,
//          n_seq_max == 1, n_seqs_unq == 1, n_seq_tokens == 1 and
//          n_seqs == n_tokens (llama-context.cpp:4830-4834). The gate applies
//          to this arm.
//   multiseq
//          n_seq_max = 2, kv_unified = false, two interleaved sequences. The
//          binder refuses a multi-stream KV cache outright
//          (get_n_stream() > 1 -> rebuild fallback, llama-context.cpp:4177),
//          so this arm is expected to fall back. It is reported, not gated as
//          a pass: the report records per-step whether a bind applied, and a
//          bind here is a finding, never a success.
//
// The gate conditions rest on observations, not on the plan: the per-step
// llama_graph_build_count deltas, the ON arm's bound_tokens from
// llama_get_union_diagnostics(), the OFF arm's absence of an enabled binder,
// the observed KV occupancy, and byte equality of the requested logits rows.
//
// The JSON report is written before the exit code is decided, so a failing
// gate still leaves a complete record.
#include "llama.h"

#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int width_max     = 4;  // widest token batch the schedule submits
constexpr int steps_min     = 32; // gate floor for the requested step count
constexpr int steps_default = 64;

// Per-token sequence for batch positions 0..3 of a step. Widths 1..4 use the
// first `width` entries, so every step of width >= 2 carries both sequences.
constexpr int seq_of[4] = {0, 1, 1, 0};

// Token widths and active output counts, cycled by the step index.
constexpr int widths_of[4]  = {1, 3, 2, 4};
constexpr int outputs_of[4] = {1, 2, 1, 2};

// One step's requested work, fixed before either arm runs.
struct step_plan {
    int              width      = 0;
    int              n_outputs  = 0;
    int              pos[2]     = {}; // per-sequence first position
    int              n_kv       = 0;  // cumulative tokens resident in the cache
    std::vector<int> seq_of_token;    // sequence of each token, batch order
    // batch token indices requested as outputs, ascending. This is the row
    // identity of the step: both arms hash exactly these rows, in this order.
    std::vector<int> output_index;
};

// Everything the report emits for one executed step of one arm.
struct step_record {
    int         width      = 0;
    int         n_outputs  = 0;
    int         pos[2]     = {};
    int         n_kv       = 0;
    uint64_t    on_build   = 0; // ON arm: per-step llama_graph_build_count delta
    uint64_t    off_build  = 0; // OFF arm: per-step llama_graph_build_count delta
    uint64_t    on_total   = 0;
    uint64_t    off_total  = 0;
    uint64_t    binds      = 0;
    uint64_t    fallbacks  = 0;
    uint64_t    declines   = 0;
    bool        bound      = false; // ON decode built no graph this step
    int         rows       = 0;     // requested output rows (from the plan)
    int         equal_rows = 0;
    // observed KV occupancy after the decode: the largest position the memory
    // holds for each sequence, read from llama_memory_seq_pos_max(). -1 means
    // the sequence is empty. This is an observation, not the plan's count.
    int         kv_pos_max_on[2]  = {-1, -1};
    int         kv_pos_max_off[2] = {-1, -1};
    // tokens the plan submitted per sequence for this step. Beside
    // kv_pos_max this separates "the step never used sequence 1" (0 tokens
    // submitted) from "the sequence is used but empty" (-1 observed).
    int         seq_tokens[2]     = {0, 0};
    // observed logits exposure: how many of the width batch positions returned
    // a non-null llama_get_logits_ith() row, and how many of those were
    // requested. Rows that were not requested are not exposed by the API.
    int         logits_rows_exposed_on  = 0;
    int         logits_rows_exposed_off = 0;
    // ON-arm binder state after the decode, from llama_get_union_diagnostics():
    // the token width the live graph is bound at, the n_kv width it is bound
    // at, and whether the binder had applied a patch by this step.
    // `diag_have` is false when the context reports no binder state at all.
    bool        diag_have     = false;
    uint32_t    bound_tokens  = 0;
    // n_kv width the live graph is bound at. On the exact lane this is the
    // true used span, i.e. the cumulative token count after the step; the
    // stock padded width would show a 256-cell bucket instead.
    uint32_t    bound_n_kv    = 0;
    // row identity, one entry per requested output row, in plan order. The
    // batch token index of each row is row_index[i]; the hashes are indexed by
    // row position i, never by the batch index.
    std::vector<std::string> row_sha_on;
    std::vector<std::string> row_sha_off;
    std::vector<int>         row_index; // batch token index of each output row
    std::vector<bool>        row_equal;
    int         on_rc      = 0;
    int         off_rc     = 0;
};

// ---- SHA-256 (FIPS 180-4), used as the per-row identity ----------------

struct sha256_ctx {
    uint32_t state[8];
    uint64_t bits;
    uint8_t  block[64];
    size_t   used;
};

constexpr uint32_t sha256_k[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
};

inline uint32_t rotr32(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

void sha256_compress(sha256_ctx & c, const uint8_t * p) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
        w[i] = (uint32_t(p[4 * i]) << 24) | (uint32_t(p[4 * i + 1]) << 16) |
               (uint32_t(p[4 * i + 2]) << 8) | uint32_t(p[4 * i + 3]);
    }
    for (int i = 16; i < 64; ++i) {
        const uint32_t s0 = rotr32(w[i - 15], 7) ^ rotr32(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const uint32_t s1 = rotr32(w[i - 2], 17) ^ rotr32(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = c.state[0], b = c.state[1], cc = c.state[2], d = c.state[3];
    uint32_t e = c.state[4], f = c.state[5], g = c.state[6], h = c.state[7];
    for (int i = 0; i < 64; ++i) {
        const uint32_t s1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
        const uint32_t ch = (e & f) ^ (~e & g);
        const uint32_t t1 = h + s1 + ch + sha256_k[i] + w[i];
        const uint32_t s0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
        const uint32_t mj = (a & b) ^ (a & cc) ^ (b & cc);
        const uint32_t t2 = s0 + mj;
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = cc;
        cc = b;
        b = a;
        a = t1 + t2;
    }
    c.state[0] += a;
    c.state[1] += b;
    c.state[2] += cc;
    c.state[3] += d;
    c.state[4] += e;
    c.state[5] += f;
    c.state[6] += g;
    c.state[7] += h;
}

void sha256_init(sha256_ctx & c) {
    c.state[0] = 0x6a09e667u;
    c.state[1] = 0xbb67ae85u;
    c.state[2] = 0x3c6ef372u;
    c.state[3] = 0xa54ff53au;
    c.state[4] = 0x510e527fu;
    c.state[5] = 0x9b05688cu;
    c.state[6] = 0x1f83d9abu;
    c.state[7] = 0x5be0cd19u;
    c.bits = 0;
    c.used = 0;
}

void sha256_update(sha256_ctx & c, const void * data, size_t bytes) {
    const uint8_t * p = static_cast<const uint8_t *>(data);
    c.bits += uint64_t(bytes) * 8;
    while (bytes > 0) {
        const size_t take = bytes < 64 - c.used ? bytes : 64 - c.used;
        std::memcpy(c.block + c.used, p, take);
        c.used += take;
        p      += take;
        bytes  -= take;
        if (c.used == 64) {
            sha256_compress(c, c.block);
            c.used = 0;
        }
    }
}

std::string sha256_hex(const void * data, size_t bytes) {
    sha256_ctx c;
    sha256_init(c);
    sha256_update(c, data, bytes);
    const uint64_t bits = c.bits;
    // pad to 56 bytes mod 64, then append the 64-bit big-endian bit count
    c.block[c.used++] = 0x80;
    if (c.used > 56) {
        std::memset(c.block + c.used, 0, 64 - c.used);
        sha256_compress(c, c.block);
        c.used = 0;
    }
    std::memset(c.block + c.used, 0, 56 - c.used);
    for (int i = 0; i < 8; ++i) {
        c.block[56 + i] = uint8_t((bits >> (56 - 8 * i)) & 0xffu);
    }
    sha256_compress(c, c.block);
    char out[65];
    for (int i = 0; i < 8; ++i) {
        std::snprintf(out + 8 * i, 9, "%08x", c.state[i]);
    }
    return std::string(out, 64);
}

// ---- report helpers ----------------------------------------------------

std::string json_escape(const char * s) {
    std::string out;
    for (const char * p = s; p && *p; ++p) {
        const unsigned char c = static_cast<unsigned char>(*p);
        if (c == '"' || c == '\\') {
            out.push_back('\\');
            out.push_back(char(c));
        } else if (c < 0x20) {
            char buf[7];
            std::snprintf(buf, sizeof(buf), "\\u%04x", c);
            out += buf;
        } else {
            out.push_back(char(c));
        }
    }
    return out;
}

std::string json_bool(bool v) {
    return v ? "true" : "false";
}

std::string json_int_array(const int * v, size_t n) {
    std::string out = "[";
    for (size_t i = 0; v != nullptr && i < n; ++i) {
        char buf[16];
        std::snprintf(buf, sizeof(buf), "%s%d", i ? ", " : "", v[i]);
        out += buf;
    }
    return out + "]";
}

std::string json_str_array(const std::vector<std::string> & v) {
    std::string out = "[";
    for (size_t i = 0; i < v.size(); ++i) {
        out += i ? ", \"" : "\"";
        out += v[i];
        out += "\"";
    }
    return out + "]";
}

std::string json_bool_array(const std::vector<bool> & v) {
    std::string out = "[";
    for (size_t i = 0; i < v.size(); ++i) {
        out += i ? ", " : "";
        out += json_bool(v[i]);
    }
    return out + "]";
}

// Deterministic token id for a (sequence, position) pair. Stays inside the
// model vocabulary and never uses the special ids 0..2.
llama_token token_for(int seq, int pos, int vocab) {
    const int64_t span = int64_t(vocab) - 3;
    return llama_token(3 + ((int64_t(pos) * 7 + seq * 13 + 1) % span));
}

// The schedule is fixed: widths 1,3,2,4 and active output counts 1,2,1,2.
// `max_n_ctx` is the largest single-sequence position count any arm reaches:
// the single-sequence arm collapses every step's tokens onto sequence 0, so it
// needs the sum of all widths, not the per-sequence total.
bool build_plan(int steps, std::vector<step_plan> & plan, int & total_tokens, int & max_n_ctx) {
    plan.clear();
    plan.reserve(size_t(steps));
    int pos[2] = {};
    int n_kv = 0;
    int single_seq_pos = 0;
    for (int step = 0; step < steps; ++step) {
        step_plan p;
        p.width     = widths_of[step % 4];
        p.n_outputs = outputs_of[step % 4];
        if (p.n_outputs > p.width) {
            p.n_outputs = p.width; // the active count cannot exceed the token count
        }
        p.pos[0] = pos[0];
        p.pos[1] = pos[1];
        for (int j = 0; j < p.width; ++j) {
            p.seq_of_token.push_back(seq_of[j % 4]);
        }
        for (int j = 0; j < p.width; ++j) {
            ++pos[p.seq_of_token[size_t(j)]];
        }
        // the active output tokens are the last n_outputs tokens of the batch
        for (int j = p.width - p.n_outputs; j < p.width; ++j) {
            p.output_index.push_back(j);
        }
        single_seq_pos += p.width;
        n_kv += p.width;
        p.n_kv = n_kv;
        plan.push_back(std::move(p));
    }
    total_tokens = n_kv;
    max_n_ctx    = std::max(pos[0], std::max(pos[1], single_seq_pos));
    return true;
}

// Fill one batch. `single_seq` collapses every token onto sequence 0, which is
// what the n_seq_max == 1 arm needs; otherwise the plan's interleave is used.
bool fill_batch(llama_batch & batch, const step_plan & p, int vocab, bool single_seq) {
    if (!batch.token || !batch.pos || !batch.n_seq_id || !batch.seq_id || !batch.logits) {
        return false;
    }
    // single_seq: every token belongs to sequence 0, and the step's first
    // position is the number of tokens already submitted (pos[0] + pos[1]).
    // Otherwise each sequence keeps its own position from the plan.
    int pos_now[2];
    if (single_seq) {
        pos_now[0] = p.pos[0] + p.pos[1];
        pos_now[1] = pos_now[0];
    } else {
        pos_now[0] = p.pos[0];
        pos_now[1] = p.pos[1];
    }
    // the active output tokens are the last p.n_outputs tokens of the batch,
    // so the set of output rows moves with the step index
    const int first_output = p.width - p.n_outputs;
    for (int j = 0; j < p.width; ++j) {
        const int seq = single_seq ? 0 : p.seq_of_token[size_t(j)];
        batch.token[j]     = token_for(seq, pos_now[seq], vocab);
        batch.pos[j]       = pos_now[seq]++;
        batch.n_seq_id[j]  = 1;
        batch.seq_id[j][0] = seq;
        batch.logits[j]    = j >= first_output ? 1 : 0;
    }
    batch.n_tokens = p.width;
    return true;
}

// Hash the requested logits rows of one arm, one entry per requested row in
// plan order. llama_get_logits_ith(j) is indexed by the batch token index, so
// the plan's output_index names the rows; the hash vector is indexed by the
// row position, never by the batch index, so it is exactly as long as the
// row_index and row_equal arrays the report emits beside it.
//
// The arm fills its own hash vector and nothing else. A null logits row or a
// row count that disagrees with the plan returns false and the caller fails
// the step: a short capture can never pass.
bool capture_rows(llama_context * ctx, const step_plan & p, int vocab,
                  std::vector<std::string> & hashes) {
    hashes.assign(p.output_index.size(), std::string());
    for (size_t i = 0; i < p.output_index.size(); ++i) {
        const float * logits = llama_get_logits_ith(ctx, p.output_index[i]);
        if (!logits) {
            return false;
        }
        hashes[i] = sha256_hex(logits, size_t(vocab) * sizeof(float));
    }
    return hashes.size() == size_t(p.n_outputs);
}

// Observed KV occupancy: the largest position the memory holds for each
// sequence. llama_memory_seq_pos_max() returns -1 for an empty sequence.
// n_seq is 1 for the main arm and 2 for the multiseq arm; sequence 1 is only
// queried when the arm actually uses it, because the KV cache asserts that the
// sequence id is inside its own range.
void observe_kv(llama_context * ctx, int n_seq, int out[2]) {
    out[0] = -1;
    out[1] = -1;
    llama_memory_t mem = llama_get_memory(ctx);
    if (mem == nullptr) {
        return;
    }
    out[0] = int(llama_memory_seq_pos_max(mem, 0));
    if (n_seq > 1) {
        out[1] = int(llama_memory_seq_pos_max(mem, 1));
    }
}

// Observed logits exposure over the whole batch width. llama_get_logits_ith()
// returns null for a batch position that was not requested as an output (it
// resolves through output_ids), so this counts what the context actually
// exposes, not what the plan asked for.
int observe_logits_rows(llama_context * ctx, int width) {
    int exposed = 0;
    for (int j = 0; j < width; ++j) {
        if (llama_get_logits_ith(ctx, j) != nullptr) {
            ++exposed;
        }
    }
    return exposed;
}

// Compare every requested row by row position. rec.row_index and both hash
// vectors are built from the same plan list, so position i names one row on
// both arms. rec.rows is the plan's expected count: a length disagreement is a
// failed step, never a silent pass.
bool compare_rows(step_record & rec) {
    const size_t n = size_t(rec.rows);
    const bool lengths_ok = rec.row_index.size() == n && rec.row_sha_on.size() == n &&
                            rec.row_sha_off.size() == n;
    rec.row_equal.assign(n, false);
    rec.equal_rows = 0;
    for (size_t i = 0; i < n && lengths_ok; ++i) {
        const bool have = !rec.row_sha_on[i].empty() && !rec.row_sha_off[i].empty();
        rec.row_equal[i] = have && rec.row_sha_on[i] == rec.row_sha_off[i];
        rec.equal_rows += rec.row_equal[i] ? 1 : 0;
    }
    return lengths_ok && rec.equal_rows == rec.rows;
}

struct arm_diag {
    bool     have              = false;
    bool     enabled           = false;
    bool     envelope_disabled = false;
    bool     record_valid      = false;
    bool     token_map_valid   = false;
    uint64_t binds             = 0;
    uint64_t fallbacks         = 0;
    uint64_t declines          = 0;
    uint32_t bound_tokens      = 0;
    uint32_t bound_n_kv        = 0; // n_kv width the live graph is bound at
};

bool read_diag(llama_context * ctx, arm_diag & out) {
    llama_union_diagnostics d = {};
    out.have = llama_get_union_diagnostics(ctx, &d);
    if (!out.have) {
        return false;
    }
    out.enabled           = d.enabled;
    out.envelope_disabled = d.envelope_disabled;
    out.record_valid      = d.record_valid;
    out.token_map_valid   = d.token_map_valid;
    out.binds             = d.patch_binds;
    out.fallbacks         = d.rebuild_fallbacks;
    out.declines          = d.lane_declines;
    out.bound_tokens      = d.bound_tokens;
    out.bound_n_kv        = d.bound_n_kv;
    return true;
}

void append_step(std::string & out, size_t index, const step_record & r) {
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "      {\"step\": %zu, \"width\": %d, \"n_outputs\": %d, "
                  "\"pos\": %s, \"n_kv\": %d, "
                  "\"on_build\": %llu, \"off_build\": %llu, "
                  "\"on_total\": %llu, \"off_total\": %llu, "
                  "\"bound\": %s, \"binds\": %llu, \"fallbacks\": %llu, "
                  "\"declines\": %llu, \"rows\": %d, \"equal_rows\": %d, "
                  "\"bound_tokens\": %u, \"bound_n_kv\": %u, "
                  "\"row_index\": ",
                  index, r.width, r.n_outputs, json_int_array(r.pos, 2).c_str(), r.n_kv,
                  (unsigned long long) r.on_build, (unsigned long long) r.off_build,
                  (unsigned long long) r.on_total, (unsigned long long) r.off_total,
                  json_bool(r.bound).c_str(),
                  (unsigned long long) r.binds, (unsigned long long) r.fallbacks,
                  (unsigned long long) r.declines, r.rows, r.equal_rows,
                  r.bound_tokens, r.bound_n_kv, json_bool(r.diag_have).c_str());
    out += buf;
    out += json_int_array(r.row_index.data(), r.row_index.size());
    out += ", \"row_sha_on\": ";
    out += json_str_array(r.row_sha_on);
    out += ", \"row_sha_off\": ";
    out += json_str_array(r.row_sha_off);
    out += ", \"row_equal\": ";
    out += json_bool_array(r.row_equal);
    out += ", \"kv_pos_max_on\": ";
    out += json_int_array(r.kv_pos_max_on, 2);
    out += ", \"kv_pos_max_off\": ";
    out += json_int_array(r.kv_pos_max_off, 2);
    out += ", \"seq_tokens\": ";
    out += json_int_array(r.seq_tokens, 2);
    std::snprintf(buf, sizeof(buf),
                  ", \"logits_rows_exposed_on\": %d, \"logits_rows_exposed_off\": %d",
                  r.logits_rows_exposed_on, r.logits_rows_exposed_off);
    out += buf;
    std::snprintf(buf, sizeof(buf), ", \"on_rc\": %d, \"off_rc\": %d}", r.on_rc, r.off_rc);
    out += buf;
}

bool write_file(const char * path, const std::string & text) {
    FILE * f = std::fopen(path, "wb");
    if (!f) {
        return false;
    }
    const bool ok = std::fwrite(text.data(), 1, text.size(), f) == text.size();
    return std::fclose(f) == 0 && ok;
}

// Append the invocation as a JSON array of strings. argv is caller data and
// may be arbitrarily long, so the array is built element by element instead
// of through a fixed-size buffer.
void append_argv(std::string & out, int argc, char ** argv) {
    out += "  \"argv\": [";
    for (int i = 0; i < argc; ++i) {
        out += i ? ", \"" : "\"";
        out += json_escape(argv[i]);
        out += "\"";
    }
    out += "],\n";
}

// Report for a failure that happens before any decode. It carries the same
// header fields as a full report, so a failed run can still be tied to the
// command that produced it. `steps` is the requested step count, already
// parsed by the caller.
std::string early_failure_report(const char * reason, int argc, char ** argv, int steps) {
    std::string body = "{\n  \"schema\": \"union-varlen-e2e/4\",\n";
    body += "  \"verdict\": \"fail\",\n";
    body += "  \"failure\": \"" + json_escape(reason) + "\",\n";
    if (argc >= 2) {
        body += "  \"model_path\": \"" + json_escape(argv[1]) + "\",\n";
    } else {
        body += "  \"model_path\": null,\n";
    }
    char buf[128];
    std::snprintf(buf, sizeof(buf), "  \"steps_requested\": %d,\n", steps);
    body += buf;
    std::snprintf(buf, sizeof(buf), "  \"steps_default\": %d,\n", steps_default);
    body += buf;
    std::snprintf(buf, sizeof(buf), "  \"steps_min\": %d,\n", steps_min);
    body += buf;
    body += "  \"limits\": \"CPU only; no GPU, relay, pipeline or production claim. "
            "The run stopped before any decode, so no row identity, build count or "
            "binder observation exists.\",\n";
    body += "  \"not_run\": \"GPU/HIP/CUDA backends, relay, pipeline parallelism, "
            "recurrent or non-LLAMA architectures, production qualification.\",\n";
    append_argv(body, argc, argv);
    body += "  \"exit_code\": 1,\n";
    body += "  \"arms\": [\n    {\"arm\": \"main\", \"status\": \"not_run\", \"steps\": []},\n"
            "    {\"arm\": \"multiseq\", \"status\": \"not_run\", \"steps\": []}\n  ]\n}\n";
    return body;
}

// One context arm: the two contexts (binder ON and OFF), the schedule, and
// the per-step records. `single_seq` selects the supported configuration.
struct arm_result {
    const char * name        = nullptr;
    const char * expectation = nullptr;
    bool         single_seq  = true;
    bool         constructed = false;
    const char * construct_error = nullptr;
    uint32_t     n_ctx       = 0;
    uint32_t     n_ctx_kv    = 0;
    uint32_t     n_seq_max   = 0;
    bool         kv_unified  = false;
    uint64_t     on_init_builds  = 0;
    uint64_t     off_init_builds = 0;
    arm_diag     on_init;
    arm_diag     off_init;
    arm_diag     on_final;
    arm_diag     off_final;
    uint64_t     on_builds   = 0;
    uint64_t     off_builds  = 0;
    int          executed    = 0;
    int          equal_steps = 0;
    int          bound_steps = 0;
    // ON-arm token envelope evidence, per executed step: the observed
    // bound_tokens equals the step width, and the observed bound_tokens moved
    // away from the build width (a bind applied) at least once. Both rest on
    // llama_get_union_diagnostics().bound_tokens, never on the plan.
    int          tok_steps_checked   = 0;
    int          tok_steps_matching  = 0;
    bool         tok_moved           = false;
    // ON-arm exact-path evidence, per executed step, from
    // llama_get_union_diagnostics(): bound_n_kv equals the step's cumulative
    // token count, i.e. the true used span, not the stock padded width.
    int          exact_steps_checked = 0;
    int          exact_steps_matching = 0;
    int          exact_fail_step     = -1;
    uint32_t     bound_n_kv_max      = 0;
    // OFF-arm stock gate: every executed step reported no binder state, or a
    // state with enabled == false. A step that reports an enabled binder on
    // the OFF context fails the gate.
    int          off_stock_steps     = 0;
    int          off_stock_fail_step = -1;
    // observed n_kv growth invariant (main arm): kv_pos_max[0] must be
    // strictly increasing across steps and equal n_kv - 1 at each step
    bool         kv_growth_ok = false;
    int          kv_growth_fail_step = -1;
    int          kv_growth_fail_observed = -1;
    int          kv_growth_fail_expected = -1;
    const char * failure     = nullptr;
    std::vector<step_record> records;
};

// Run one arm. Returns false only on an internal failure; the gate decision is
// left to the caller because the two arms are gated differently.
//
// `max_n_ctx` sizes the context: the single-sequence arm needs one window long
// enough for every token the schedule submits, because it collapses all steps
// onto sequence 0. Without it the per-sequence total under-sizes the window and
// the stock KV cache would evict the oldest positions, which would make the
// observed KV growth invariant fail for a reason that has nothing to do with
// the binder.
bool run_arm(llama_model * model, const std::vector<step_plan> & plan, int total_tokens,
             int max_n_ctx, int steps, int vocab, arm_result & arm) {
    auto cp = llama_context_default_params();
    cp.n_ctx         = uint32_t(max_n_ctx);
    cp.n_ctx_kv      = uint32_t(max_n_ctx);
    cp.n_batch       = width_max;
    cp.n_ubatch      = width_max;
    cp.n_seq_max     = arm.single_seq ? 1 : 2;
    cp.n_outputs_max = width_max;
    cp.n_threads     = 1;
    cp.n_threads_batch = 1;
    // The supported lane needs a unified cache with one sequence. The
    // multi-sequence arm needs one stream per sequence, which is exactly the
    // cache the binder documents as a fail-closed refusal.
    cp.kv_unified    = arm.single_seq;
    cp.offload_kqv   = false;
    cp.op_offload    = false;
    cp.no_perf       = false;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.embeddings    = false;

    setenv("GGML_TP5_UNION_BINDER", "1", 1);
    setenv("LLAMA_EXPERIMENTAL_VARLEN_NTOK", "1", 1);
    unsetenv("GGML_TP5_UNION_BINDER_EXACT");
    unsetenv("LLAMA_GRAPH_REUSE_DISABLE");
    llama_context * on = llama_init_from_model(model, cp);

    setenv("GGML_TP5_UNION_BINDER", "0", 1);
    llama_context * off = llama_init_from_model(model, cp);

    if (!on || !off) {
        // an observed result, not a silent skip
        std::fprintf(stderr, "ARM_FAIL arm=%s context creation failed (on=%p off=%p)\n",
                     arm.name, (void *) on, (void *) off);
        if (on) llama_free(on);
        if (off) llama_free(off);
        arm.construct_error = "context creation failed";
        return false;
    }
    arm.constructed = true;
    arm.n_ctx       = llama_n_ctx(on);
    arm.n_ctx_kv    = llama_n_ctx_kv(on);
    arm.n_seq_max   = arm.single_seq ? 1 : 2;
    arm.kv_unified  = arm.single_seq;

    arm.on_init_builds  = llama_graph_build_count(on);
    arm.off_init_builds = llama_graph_build_count(off);
    read_diag(on, arm.on_init);
    read_diag(off, arm.off_init);

    std::fprintf(stderr,
                 "ARM_BEGIN arm=%s single_seq=%d n_ctx=%u n_ctx_kv=%u steps=%d total_tokens=%d "
                 "max_n_ctx=%d vocab=%d on_init_builds=%llu off_init_builds=%llu on_enabled=%d "
                 "on_envelope_disabled=%d\n",
                 arm.name, arm.single_seq ? 1 : 0, arm.n_ctx, arm.n_ctx_kv, steps,
                 total_tokens, max_n_ctx, vocab, (unsigned long long) arm.on_init_builds,
                 (unsigned long long) arm.off_init_builds,
                 arm.on_init.enabled ? 1 : 0, arm.on_init.envelope_disabled ? 1 : 0);

    // the ON-arm build width: bound_tokens starts here and a bind moves it
    const uint32_t on_init_bound_tokens = arm.on_init.bound_tokens;
    int kv_previous = -1;
    if (arm.single_seq) {
        arm.kv_growth_ok = true;
    }

    for (int step = 0; step < steps; ++step) {
        const step_plan & p = plan[size_t(step)];
        llama_batch batch = llama_batch_init(p.width, 0, 1);
        if (!fill_batch(batch, p, vocab, arm.single_seq)) {
            std::fprintf(stderr, "ARM_FAIL arm=%s batch allocation step=%d\n", arm.name, step);
            llama_batch_free(batch);
            arm.failure = "batch allocation failed";
            break;
        }

        step_record rec;
        rec.width     = p.width;
        rec.n_outputs = p.n_outputs;
        rec.rows      = p.n_outputs; // expected count from the plan
        rec.pos[0]    = p.pos[0];
        rec.pos[1]    = p.pos[1];
        rec.n_kv      = p.n_kv;
        // the row identity is fixed from the plan before either arm runs, so a
        // short capture cannot leave the two arms with different row lists
        rec.row_index = p.output_index;
        for (int j = 0; j < p.width; ++j) {
            ++rec.seq_tokens[p.seq_of_token[size_t(j)]];
        }

        const uint64_t on_before  = llama_graph_build_count(on);
        const uint64_t off_before = llama_graph_build_count(off);
        rec.on_rc  = llama_decode(on, batch);
        rec.off_rc = llama_decode(off, batch);
        rec.on_build  = llama_graph_build_count(on) - on_before;
        rec.off_build = llama_graph_build_count(off) - off_before;
        llama_batch_free(batch);

        // observed state after the decode: KV occupancy per sequence and how
        // many logits rows the context actually exposes over the full width
        const int arm_seqs = arm.single_seq ? 1 : 2;
        observe_kv(on, arm_seqs, rec.kv_pos_max_on);
        observe_kv(off, arm_seqs, rec.kv_pos_max_off);
        rec.logits_rows_exposed_on  = observe_logits_rows(on, p.width);
        rec.logits_rows_exposed_off = observe_logits_rows(off, p.width);

        arm.on_builds  += rec.on_build;
        arm.off_builds += rec.off_build;
        rec.on_total  = arm.on_init_builds + arm.on_builds;
        rec.off_total = arm.off_init_builds + arm.off_builds;
        rec.bound     = rec.on_build == 0;
        arm.bound_steps += rec.bound ? 1 : 0;

        // ON-arm binder state after this step: the cumulative counters, the
        // token width and the n_kv width the live graph is bound at
        arm_diag d;
        read_diag(on, d);
        rec.diag_have     = d.have;
        rec.binds         = d.binds;
        rec.fallbacks     = d.fallbacks;
        rec.declines      = d.declines;
        rec.bound_tokens  = d.bound_tokens;
        rec.bound_n_kv    = d.bound_n_kv;

        // OFF-arm stock gate: a stock context allocates no binder state, so
        // llama_get_union_diagnostics() returns false. A state that exists must
        // report enabled == false. Anything else means the OFF arm is bound,
        // and a bound OFF arm makes every row comparison meaningless.
        arm_diag off_d;
        read_diag(off, off_d);
        const bool off_stock = !off_d.have || !off_d.enabled;
        if (off_stock) {
            ++arm.off_stock_steps;
        } else if (arm.off_stock_fail_step < 0) {
            arm.off_stock_fail_step = step;
        }

        if (rec.on_rc != 0 || rec.off_rc != 0) {
            std::fprintf(stderr, "ARM_FAIL arm=%s decode step=%d on_rc=%d off_rc=%d\n",
                         arm.name, step, rec.on_rc, rec.off_rc);
            arm.failure = "llama_decode failed";
            arm.records.push_back(rec);
            ++arm.executed;
            break;
        }

        if (!capture_rows(on, p, vocab, rec.row_sha_on) ||
            !capture_rows(off, p, vocab, rec.row_sha_off)) {
            std::fprintf(stderr, "ARM_FAIL arm=%s logits capture step=%d\n", arm.name, step);
            arm.failure = "logits capture failed";
            arm.records.push_back(rec);
            ++arm.executed;
            break;
        }
        const bool equal = compare_rows(rec);
        arm.equal_steps += equal ? 1 : 0;

        // ON-arm token envelope evidence: the live graph must be bound at the
        // width this step submitted. A bind that leaves the graph at another
        // width is a failure, not a pass.
        if (rec.diag_have) {
            ++arm.tok_steps_checked;
            if (rec.bound_tokens == uint32_t(p.width)) {
                ++arm.tok_steps_matching;
            }
            if (rec.bound_tokens != on_init_bound_tokens) {
                arm.tok_moved = true;
            }
            // Exact-path evidence. The true used span after this step is the
            // plan's cumulative token count `p.n_kv`: every submitted token
            // stays resident on the single sequence. A padded lane would bind
            // a 256-cell bucket instead, so equality with p.n_kv is the
            // assertion; "not a multiple of 256" would be vacuous here because
            // the schedule never reaches 256.
            ++arm.exact_steps_checked;
            const bool nkv_exact = rec.bound_n_kv == uint32_t(p.n_kv);
            if (nkv_exact) {
                ++arm.exact_steps_matching;
            } else if (arm.exact_fail_step < 0) {
                arm.exact_fail_step = step;
            }
            if (rec.bound_n_kv > arm.bound_n_kv_max) {
                arm.bound_n_kv_max = rec.bound_n_kv;
            }
        }

        // Observed n_kv growth, main arm only, checked per step so the loop
        // stops at the first violation instead of spending the rest of the
        // schedule on a graph that already broke the invariant. One sequence,
        // contiguous 0-based positions: after step s the memory must hold
        // positions 0..n_kv-1, so the largest one is n_kv-1 and strictly
        // larger than the previous step's.
        if (arm.single_seq) {
            const int expected = rec.n_kv - 1;
            const int observed = rec.kv_pos_max_on[0];
            if (observed != expected || observed <= kv_previous) {
                arm.kv_growth_ok = false;
                arm.kv_growth_fail_step = step;
                arm.kv_growth_fail_observed = observed;
                arm.kv_growth_fail_expected = expected;
            } else {
                kv_previous = observed;
            }
        }

        std::fprintf(stderr,
                     "STEP arm=%s step=%d width=%d n_outputs=%d pos=[%d,%d] n_kv=%d "
                     "on_build=%llu off_build=%llu bound=%d binds=%llu fallbacks=%llu "
                     "declines=%llu bitwise_rows=%d/%d bound_tokens=%u bound_n_kv=%u "
                     "off_stock=%d\n",
                     arm.name, step, rec.width, rec.n_outputs, rec.pos[0], rec.pos[1],
                     rec.n_kv, (unsigned long long) rec.on_build,
                     (unsigned long long) rec.off_build, rec.bound ? 1 : 0,
                     (unsigned long long) rec.binds,
                     (unsigned long long) rec.fallbacks,
                     (unsigned long long) rec.declines, rec.equal_rows, rec.rows,
                     rec.bound_tokens, rec.bound_n_kv, off_stock ? 1 : 0);
        arm.records.push_back(rec);
        ++arm.executed;

        if (arm.single_seq && !arm.kv_growth_ok) {
            std::fprintf(stderr,
                         "ARM_FAIL arm=%s kv growth step=%d observed=%d expected=%d\n",
                         arm.name, step, arm.kv_growth_fail_observed,
                         arm.kv_growth_fail_expected);
            arm.failure = "kv growth invariant failed";
            break;
        }
    }

    read_diag(on, arm.on_final);
    read_diag(off, arm.off_final);

    std::fprintf(stderr,
                 "ARM_END arm=%s executed=%d bitwise_steps=%d bound_steps=%d "
                 "on_decode_builds=%llu off_decode_builds=%llu on_binds=%llu "
                 "on_fallbacks=%llu kv_growth_ok=%d kv_growth_fail_step=%d "
                 "kv_growth_observed=%d kv_growth_expected=%d "
                 "tok_steps_checked=%d tok_steps_matching=%d tok_moved=%d "
                 "off_stock_steps=%d off_stock_fail_step=%d\n",
                 arm.name, arm.executed, arm.equal_steps, arm.bound_steps,
                 (unsigned long long) arm.on_builds, (unsigned long long) arm.off_builds,
                 (unsigned long long) arm.on_final.binds,
                 (unsigned long long) arm.on_final.fallbacks,
                 arm.kv_growth_ok ? 1 : 0, arm.kv_growth_fail_step,
                 arm.kv_growth_fail_observed, arm.kv_growth_fail_expected,
                 arm.tok_steps_checked, arm.tok_steps_matching, arm.tok_moved ? 1 : 0,
                 arm.off_stock_steps, arm.off_stock_fail_step);

    llama_free(on);
    llama_free(off);
    return arm.failure == nullptr;
}

// Append one arm object. `main_ok` is the gate result for the main arm; the
// multi-sequence arm reports its observation and never claims a pass.
void append_arm(std::string & out, const arm_result & a, int steps, bool main_ok) {
    char buf[1024];
    out += "    {\n";
    std::snprintf(buf, sizeof(buf),
                  "      \"arm\": \"%s\",\n"
                  "      \"expectation\": \"%s\",\n"
                  "      \"single_seq\": %s,\n"
                  "      \"n_seq_max\": %u,\n"
                  "      \"kv_unified\": %s,\n"
                  "      \"constructed\": %s,\n",
                  a.name, a.expectation, json_bool(a.single_seq).c_str(), a.n_seq_max,
                  json_bool(a.kv_unified).c_str(), json_bool(a.constructed).c_str());
    out += buf;
    if (!a.constructed) {
        out += "      \"construct_error\": \"";
        out += json_escape(a.construct_error ? a.construct_error : "unknown");
        out += "\",\n";
        out += "      \"status\": \"construction_failed\",\n";
        out += "      \"steps\": []\n    }";
        return;
    }
    std::snprintf(buf, sizeof(buf),
                  "      \"n_ctx\": %u,\n"
                  "      \"n_ctx_kv\": %u,\n"
                  "      \"init_builds\": {\"on\": %llu, \"off\": %llu},\n"
                  "      \"decode_builds\": {\"on\": %llu, \"off\": %llu, \"expected_on\": 1, "
                  "\"expected_off\": %d},\n"
                  "      \"bitwise\": {\"steps_equal\": %d, \"steps_executed\": %d, "
                  "\"bound_steps\": %d},\n"
                  "      \"on_diagnostics\": {\"enabled\": %s, \"envelope_disabled\": %s, "
                  "\"record_valid\": %s, \"token_map_valid\": %s, "
                  "\"patch_binds\": %llu, \"rebuild_fallbacks\": %llu, "
                  "\"lane_declines\": %llu, \"bound_tokens\": %u},\n"
                  "      \"off_diagnostics\": {\"available\": %s, \"enabled\": %s, "
                  "\"envelope_disabled\": %s, \"record_valid\": %s, "
                  "\"token_map_valid\": %s, \"patch_binds\": %llu, "
                  "\"rebuild_fallbacks\": %llu, \"lane_declines\": %llu, "
                  "\"bound_tokens\": %u},\n",
                  a.n_ctx, a.n_ctx_kv,
                  (unsigned long long) a.on_init_builds,
                  (unsigned long long) a.off_init_builds,
                  (unsigned long long) a.on_builds, (unsigned long long) a.off_builds, steps,
                  a.equal_steps, a.executed, a.bound_steps,
                  json_bool(a.on_final.enabled).c_str(),
                  json_bool(a.on_final.envelope_disabled).c_str(),
                  json_bool(a.on_final.record_valid).c_str(),
                  json_bool(a.on_final.token_map_valid).c_str(),
                  (unsigned long long) a.on_final.binds,
                  (unsigned long long) a.on_final.fallbacks,
                  (unsigned long long) a.on_final.declines, a.on_final.bound_tokens,
                  json_bool(a.off_final.have).c_str(),
                  json_bool(a.off_final.enabled).c_str(),
                  json_bool(a.off_final.envelope_disabled).c_str(),
                  json_bool(a.off_final.record_valid).c_str(),
                  json_bool(a.off_final.token_map_valid).c_str(),
                  (unsigned long long) a.off_final.binds,
                  (unsigned long long) a.off_final.fallbacks,
                  (unsigned long long) a.off_final.declines, a.off_final.bound_tokens);
    out += buf;
    // gate conditions, each from a real observation. Both arms state them the
    // same way; only the main arm's result decides the exit code.
    const bool steps_ok  = a.executed == steps && steps >= steps_min;
    const bool rows_ok   = a.executed > 0 && a.equal_steps == a.executed;
    const bool on_ok     = a.on_builds == 1;
    const bool off_ok    = a.off_builds == uint64_t(a.executed);
    const bool diag_ok   = a.on_final.have && a.on_final.enabled &&
                           !a.on_final.envelope_disabled;
    // the OFF arm must be stock: no binder state, or a state that says
    // enabled == false, at every executed step
    const bool off_stock_ok = a.executed > 0 && a.off_stock_steps == a.executed;
    // the ON arm's live graph must be bound at the width the step submitted
    const bool tok_ok    = a.executed > 0 && a.tok_steps_checked == a.executed &&
                           a.tok_steps_matching == a.executed;
    // the exact path was selected and delivered: bound_n_kv is the true used
    // span at every step
    const bool exact_ok  = a.executed > 0 && a.exact_steps_checked == a.executed &&
                           a.exact_steps_matching == a.executed;
    if (a.single_seq) {
        // the gate arm: every condition is a real observation
        std::snprintf(buf, sizeof(buf),
                      "      \"gates\": {\n"
                      "        \"steps_executed_eq_requested\": %s,\n"
                      "        \"all_rows_bitwise_equal\": %s,\n"
                      "        \"on_decode_builds_eq_1\": %s,\n"
                      "        \"off_decode_builds_eq_steps\": %s,\n"
                      "        \"on_binder_enabled\": %s,\n"
                      "        \"off_binder_stock\": %s,\n"
                      "        \"on_bound_tokens_eq_width\": %s,\n"
                      "        \"kv_pos_max_eq_n_kv_minus_1\": %s,\n"
                      "        \"on_exact_path_selected\": %s\n"
                      "      },\n"
                      "      \"token_binding\": {\"steps_checked\": %d, "
                      "\"steps_matching_width\": %d, \"bound_tokens_moved\": %s},\n"
                      "      \"exact\": {\"steps_checked\": %d, "
                      "\"bound_n_kv_max\": %u, \"first_exact_miss_step\": %d},\n"
                      "      \"off_stock\": {\"steps_checked\": %d, \"steps_stock\": %d, "
                      "\"first_non_stock_step\": %d},\n"
                      "      \"kv_growth\": {\"ok\": %s, \"fail_step\": %d, "
                      "\"observed\": %d, \"expected\": %d},\n"
                      "      \"status\": \"%s\",\n",
                      json_bool(steps_ok).c_str(), json_bool(rows_ok).c_str(),
                      json_bool(on_ok).c_str(), json_bool(off_ok).c_str(),
                      json_bool(diag_ok).c_str(), json_bool(off_stock_ok).c_str(),
                      json_bool(tok_ok).c_str(), json_bool(a.kv_growth_ok).c_str(),
                      json_bool(exact_ok).c_str(),
                      a.tok_steps_checked, a.tok_steps_matching,
                      json_bool(a.tok_moved).c_str(),
                      a.exact_steps_checked, a.bound_n_kv_max, a.exact_fail_step,
                      a.executed, a.off_stock_steps, a.off_stock_fail_step,
                      json_bool(a.kv_growth_ok).c_str(), a.kv_growth_fail_step,
                      a.kv_growth_fail_observed, a.kv_growth_fail_expected,
                      main_ok ? "pass" : "fail");
        out += buf;
    } else {
        // the reported arm: a bind here is a finding, not a success.
        // Precedence: rows_differ outranks unexpected_bind. A bind that
        // produced different logits is a correctness failure; a bind with equal
        // logits is only a scope violation (the documented refusal did not
        // happen). Both flags stay in `observed`; the harder one names status.
        const bool any_bind = a.on_final.binds > 0 || a.bound_steps > 0;
        const char * status = !rows_ok ? "rows_differ" :
                              any_bind ? "unexpected_bind" :
                                         "fail_closed_refusal_observed";
        std::snprintf(buf, sizeof(buf),
                      "      \"gates\": {\n"
                      "        \"steps_executed_eq_requested\": %s,\n"
                      "        \"all_rows_bitwise_equal\": %s,\n"
                      "        \"off_decode_builds_eq_steps\": %s,\n"
                      "        \"off_binder_stock\": %s,\n"
                      "        \"on_bound_tokens_eq_width\": %s,\n"
                      "        \"on_exact_path_selected\": %s\n"
                      "      },\n"
                      "      \"observed\": {\"any_bind\": %s, \"all_rows_bitwise_equal\": %s, "
                      "\"fallback_every_step\": %s, \"off_stock\": %s},\n"
                      "      \"token_binding\": {\"steps_checked\": %d, "
                      "\"steps_matching_width\": %d, \"bound_tokens_moved\": %s},\n"
                      "      \"exact\": {\"steps_checked\": %d, "
                      "\"bound_n_kv_max\": %u, \"first_exact_miss_step\": %d},\n"
                      "      \"status\": \"%s\",\n",
                      json_bool(steps_ok).c_str(), json_bool(rows_ok).c_str(),
                      json_bool(off_ok).c_str(), json_bool(off_stock_ok).c_str(),
                      json_bool(tok_ok).c_str(),
                      json_bool(exact_ok).c_str(),
                      json_bool(any_bind).c_str(), json_bool(rows_ok).c_str(),
                      json_bool(a.executed > 0 && a.bound_steps == 0).c_str(),
                      json_bool(off_stock_ok).c_str(),
                      a.tok_steps_checked, a.tok_steps_matching,
                      json_bool(a.tok_moved).c_str(),
                      a.exact_steps_checked, a.bound_n_kv_max, a.exact_fail_step, status);
        out += buf;
    }
    if (a.failure) {
        out += "      \"failure\": \"";
        out += json_escape(a.failure);
        out += "\",\n";
    } else {
        out += "      \"failure\": null,\n";
    }

    out += "      \"steps\": [\n";
    for (size_t i = 0; i < a.records.size(); ++i) {
        append_step(out, i, a.records[i]);
        out += i + 1 < a.records.size() ? ",\n" : "\n";
    }
    out += "      ]\n    }";
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 3 || argc > 4) {
        std::fprintf(stderr,
                     "usage: %s <model.gguf> <out.json> [steps]\n"
                     "  steps defaults to %d and must be at least %d\n",
                     argv[0], steps_default, steps_min);
        return 2;
    }
    const char * model_path = argv[1];
    const char * out_path   = argv[2];
    int steps = steps_default;
    if (argc == 4) {
        char * end = nullptr;
        errno = 0;
        const long v = std::strtol(argv[3], &end, 10);
        if (errno || end == argv[3] || *end || v < 1 || v > INT_MAX / width_max) {
            std::fprintf(stderr, "invalid steps: %s\n", argv[3]);
            return 2;
        }
        steps = int(v);
    }

    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    ggml_backend_dev_t devices[] = {nullptr};
    mp.devices = devices;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) {
        std::fprintf(stderr, "FAIL: cannot load model %s\n", model_path);
        llama_backend_free();
        write_file(out_path, early_failure_report("model load failed", argc, argv, steps));
        return 1;
    }

    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<step_plan> plan;
    int total_tokens = 0;
    int max_n_ctx    = 0;
    if (vocab <= 3 || !build_plan(steps, plan, total_tokens, max_n_ctx)) {
        std::fprintf(stderr, "FAIL: unusable vocabulary (%d)\n", vocab);
        llama_model_free(model);
        llama_backend_free();
        write_file(out_path, early_failure_report("unusable vocabulary", argc, argv, steps));
        return 1;
    }

    arm_result main_arm;
    main_arm.name        = "main";
    main_arm.expectation = "supported_single_stream_lane";
    main_arm.single_seq  = true;

    arm_result multi_arm;
    multi_arm.name        = "multiseq";
    multi_arm.expectation = "fail_closed_refusal_documented";
    multi_arm.single_seq  = false;

    run_arm(model, plan, total_tokens, max_n_ctx, steps, vocab, main_arm);
    run_arm(model, plan, total_tokens, max_n_ctx, steps, vocab, multi_arm);

    // Gate: the main arm only. The multi-sequence arm is evidence, not a pass.
    const bool steps_ok = main_arm.constructed && main_arm.executed == steps &&
                          steps >= steps_min;
    const bool rows_ok  = main_arm.constructed && main_arm.executed > 0 &&
                          main_arm.equal_steps == main_arm.executed;
    const bool on_ok    = main_arm.constructed && main_arm.on_builds == 1;
    const bool off_ok   = main_arm.constructed &&
                          main_arm.off_builds == uint64_t(main_arm.executed);
    const bool diag_ok  = main_arm.constructed && main_arm.on_final.have &&
                          main_arm.on_final.enabled && !main_arm.on_final.envelope_disabled;
    // the OFF arm must report no enabled binder at every executed step
    const bool off_stock_ok = main_arm.constructed && main_arm.executed > 0 &&
                              main_arm.off_stock_steps == main_arm.executed;
    // the ON arm's live graph must be bound at the step width at every step
    const bool tok_ok   = main_arm.constructed && main_arm.executed > 0 &&
                          main_arm.tok_steps_checked == main_arm.executed &&
                          main_arm.tok_steps_matching == main_arm.executed;
    const bool kv_ok    = main_arm.constructed && main_arm.kv_growth_ok;
    // The exact path was selected AND delivered: bound_n_kv is the true used
    // span at every executed step, not the stock padded width.
    const bool exact_ok = main_arm.constructed && main_arm.executed > 0 &&
                          main_arm.exact_steps_checked == main_arm.executed &&
                          main_arm.exact_steps_matching == main_arm.executed;

    // the failure string names the step and both values when the observed KV
    // occupancy disagrees with the plan
    char kv_fail[256];
    kv_fail[0] = '\0';
    if (!kv_ok) {
        std::snprintf(kv_fail, sizeof(kv_fail),
                      "kv_pos_max invariant failed at step %d: observed %d, expected n_kv-1 = %d",
                      main_arm.kv_growth_fail_step, main_arm.kv_growth_fail_observed,
                      main_arm.kv_growth_fail_expected);
    }
    char off_fail[256];
    off_fail[0] = '\0';
    if (!off_stock_ok) {
        std::snprintf(off_fail, sizeof(off_fail),
                      "OFF arm reported an enabled binder at step %d",
                      main_arm.off_stock_fail_step);
    }
    char tok_fail[256];
    tok_fail[0] = '\0';
    if (!tok_ok) {
        std::snprintf(tok_fail, sizeof(tok_fail),
                      "ON arm bound_tokens != step width at %d of %d executed steps",
                      main_arm.executed - main_arm.tok_steps_matching, main_arm.executed);
    }
    char exact_fail[256];
    exact_fail[0] = '\0';
    if (!exact_ok) {
        std::snprintf(exact_fail, sizeof(exact_fail),
                      "ON arm bound_n_kv != true used span at step %d "
                      "(%d of %d executed steps matched)",
                      main_arm.exact_fail_step,
                      main_arm.exact_steps_matching, main_arm.executed);
    }
    const char * fail_reason = nullptr;
    if (!main_arm.constructed) {
        fail_reason = "main arm context creation failed";
    } else if (!steps_ok) {
        fail_reason = "steps executed != steps requested (or below 32)";
    } else if (!off_stock_ok) {
        fail_reason = off_fail;
    } else if (!rows_ok) {
        fail_reason = "logits rows differ between arms";
    } else if (!diag_ok) {
        fail_reason = "ON arm binder diagnostics unavailable or disabled";
    } else if (!tok_ok) {
        fail_reason = tok_fail;
    } else if (!kv_ok) {
        fail_reason = kv_fail;
    } else if (!exact_ok) {
        fail_reason = exact_fail;
    } else if (!on_ok) {
        fail_reason = "ON arm decode builds != 1";
    } else if (!off_ok) {
        fail_reason = "OFF arm decode builds != steps";
    }
    const bool ok = fail_reason == nullptr;

    std::string body;
    char buf[2048];
    body += "{\n";
    body += "  \"schema\": \"union-varlen-e2e/4\",\n";
    std::snprintf(buf, sizeof(buf), "  \"verdict\": \"%s\",\n", ok ? "pass" : "fail");
    body += buf;
    if (ok) {
        body += "  \"failure\": null,\n";
    } else {
        body += "  \"failure\": \"" + json_escape(fail_reason) + "\",\n";
    }
    // the path is caller data: append it directly so no fixed buffer truncates it
    body += "  \"model_path\": \"" + json_escape(model_path) + "\",\n";
    std::snprintf(buf, sizeof(buf), "  \"vocab\": %d,\n", vocab);
    body += buf;
    std::snprintf(buf, sizeof(buf), "  \"steps_requested\": %d,\n", steps);
    body += buf;
    std::snprintf(buf, sizeof(buf), "  \"steps_default\": %d,\n", steps_default);
    body += buf;
    std::snprintf(buf, sizeof(buf), "  \"steps_min\": %d,\n", steps_min);
    body += buf;
    std::snprintf(buf, sizeof(buf), "  \"total_tokens\": %d,\n", total_tokens);
    body += buf;
    std::snprintf(buf, sizeof(buf), "  \"max_n_ctx\": %d,\n", max_n_ctx);
    body += buf;
    body += "  \"width_schedule\": [1, 3, 2, 4],\n";
    body += "  \"outputs_schedule\": [1, 2, 1, 2],\n";
    body += "  \"gate_arm\": \"main\",\n";
    body += "  \"limits\": \"CPU only; no GPU, relay, pipeline or production claim. "
            "Row identity is the SHA-256 of the F32 logits row, one entry per requested "
            "output row in plan order, with the batch token index in row_index. The "
            "bound width of the output-gather closure is not observable through the "
            "public API, so the output-count envelope is evidenced only by the build "
            "counts and by bound_tokens. on_exact_path_selected rests on bound_n_kv "
            "equalling the step's cumulative token count, which is the true used span "
            "on the single sequence; the schedule never reaches 256, so the stock "
            "padded width cannot coincide with it. The key does not observe GPU FA "
            "widths, iSWA/MSA/DSV4 caches, or the rope long/short "
            "bucket, none of which this lane reaches. The multiseq arm is expected to "
            "fail closed on a multi-stream KV cache and is reported, not gated as a "
            "pass.\",\n";
    body += "  \"not_run\": \"GPU/HIP/CUDA backends, relay, pipeline parallelism, "
            "recurrent or non-LLAMA architectures, production qualification.\",\n";
    append_argv(body, argc, argv);
    // the report states the gate verdict. A write failure cannot be reported in
    // the file it failed to write, so it lands on stderr and in the process
    // exit code instead.
    std::snprintf(buf, sizeof(buf), "  \"exit_code\": %d,\n", ok ? 0 : 1);
    body += buf;
    body += "  \"arms\": [\n";
    append_arm(body, main_arm, steps, ok);
    body += ",\n";
    append_arm(body, multi_arm, steps, ok);
    body += "\n  ]\n}\n";

    bool wrote = write_file(out_path, body);
    if (!wrote) {
        std::fprintf(stderr, "FAIL: cannot write %s\n", out_path);
    }

    // the multiseq status follows the same precedence the arm object states:
    // rows_differ outranks unexpected_bind
    const char * multi_status = "construction_failed";
    if (multi_arm.constructed) {
        const bool multi_rows_ok = multi_arm.executed > 0 &&
                                   multi_arm.equal_steps == multi_arm.executed;
        const bool any_bind = multi_arm.on_final.binds > 0 || multi_arm.bound_steps > 0;
        multi_status = !multi_rows_ok ? "rows_differ" :
                       any_bind ? "unexpected_bind" : "fail_closed_refusal_observed";
    }
    // A failed JSON write leaves no artifact. The log must not say PASS for a
    // run that produced nothing, so the verdict carries the write result.
    const char * gate_verdict = !ok ? "FAIL" : wrote ? "PASS" : "FAIL reason=json_write";
    std::fprintf(stderr,
                 "GATE %s arm=main steps=%d/%d rows=%d/%d on_builds=%llu off_builds=%llu "
                 "kv_growth_ok=%d off_stock_ok=%d bound_tokens_ok=%d(%d/%d) "
                 "bound_tokens_moved=%d multiseq_status=%s\n",
                 gate_verdict, main_arm.executed, steps, main_arm.equal_steps,
                 main_arm.executed, (unsigned long long) main_arm.on_builds,
                 (unsigned long long) main_arm.off_builds, main_arm.kv_growth_ok ? 1 : 0,
                 off_stock_ok ? 1 : 0, tok_ok ? 1 : 0,
                 main_arm.tok_steps_matching, main_arm.tok_steps_checked,
                 main_arm.tok_moved ? 1 : 0, multi_status);

    llama_model_free(model);
    llama_backend_free();
    return ok && wrote ? 0 : 1;
}
