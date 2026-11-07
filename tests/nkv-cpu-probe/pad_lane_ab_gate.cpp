// CPU A/B gate for the two n_kv lanes the union binder can run.
//
// The binder has exactly two lanes, and they differ in the knob
// GGML_TP5_UNION_BINDER_EXACT:
//
//   exact  GGML_TP5_UNION_BINDER=1, GGML_TP5_UNION_BINDER_EXACT unset. The
//          binder records the graph and binds the true used span at every
//          step, and the context is created with the requested capacity (no
//          256-cell pad).
//   pad    GGML_TP5_UNION_BINDER=1, GGML_TP5_UNION_BINDER_EXACT=0. The binder
//          records NOTHING (llama-context.cpp:5532 requires nkv_exact_enabled
//          for a record build), so the graph runs stock and no bind happens.
//          The context capacity is the stock 256-cell padded value
//          (llama-context.cpp:667 pads n_ctx, :700 disarms
//          runtime_len_capacity).
//
// WHAT THE TWO LANES DIFFER IN, read from the merged tree:
//
//   capacity      YES. The pad lane pads n_ctx to GGML_PAD(n_ctx, 256)
//                 (llama-context.cpp:667). The exact lane keeps the requested
//                 value. This is a CAPACITY-pad site; the n_kv bucket floor
//                 that 8de2c4a deleted was a different site, in
//                 llama_kv_cache::get_n_kv().
//   binder state  YES. The pad lane builds no record, so
//                 llama_get_union_diagnostics() reports bound_n_kv == 0 and no
//                 patch_binds. The exact lane reports the bound span.
//   bound width   NO. Both lanes execute at the true used span. The pad lane
//                 reaches it because get_n_kv() rounds by n_pad, which is 1
//                 for a plain cache (llama-kv-cache.h:341), so the 256-cell
//                 bucket no longer exists to bind to.
//
// The driver therefore asserts, per step:
//
//   exact  bound_n_kv == the step's cumulative token count (the true used
//          span), and llama_n_ctx(ctx) == the requested n_ctx.
//   pad    no binder record: bound_n_kv == 0 (the field is member-wise, so 0
//          means "no record", never "bound at width 0"), and
//          llama_n_ctx(ctx) == the stock 256-padded n_ctx.
//
// The gate is falsifiable in the only way that matters here: the two arms must
// be distinguishable. The distinctness check is on the CAPACITY, which is what
// actually differs; it fails if the capacities agree, or if the pad arm
// starts reporting a binder record. It deliberately does NOT require the two
// bound widths to differ, because on this tree they do not -- asserting that
// would be asserting a deleted mechanism.
//
// The padded width must not change numerics: every step's requested logits row
// is hashed on both arms and the hashes must match bitwise.
//
// Public llama.h only. The report is written before the exit code is decided,
// so a failing gate still leaves a complete record.
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

constexpr int steps_default = 32; // enough steps to cross several widths
constexpr int steps_min     = 8;

// The requested capacity, deliberately not a multiple of 256: the exact arm
// keeps it and the pad arm rounds it up, so the capacity observation is a real
// difference and not a coincidence.
constexpr int n_ctx_requested = 1000;

// The stock 256-cell capacity pad the padded lane applies to n_ctx.
constexpr uint32_t n_ctx_stock_bucket = 256;

// One executed step of one arm.
struct step_record {
    int      step      = 0;
    int      n_kv      = 0; // plan: cumulative token count after the step
    int      rc        = 0;
    bool     diag_have = false;
    uint32_t bound_n_kv = 0;
    // SHA-256 of the requested logits row. A decode that returns no row gets
    // an empty hash, which never compares equal.
    std::string row_sha;
};

struct arm_result {
    const char * name            = nullptr;
    bool         pad_lane        = false; // false: exact, true: EXACT=0
    bool         constructed     = false;
    const char * construct_error = nullptr;
    uint32_t     n_ctx           = 0;
    int          steps_checked   = 0;
    int          bound_matching  = 0; // steps whose bound_n_kv met this lane's expectation
    int          steps_with_record = 0; // steps whose bound_n_kv was nonzero (a binder record)
    int          rows_captured   = 0;
    std::vector<step_record> records;
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

void append_argv(std::string & out, int argc, char ** argv) {
    out += "  \"argv\": [";
    for (int i = 0; i < argc; ++i) {
        out += i ? ", \"" : "\"";
        out += json_escape(argv[i]);
        out += "\"";
    }
    out += "],\n";
}

bool write_file(const char * path, const std::string & text) {
    FILE * f = std::fopen(path, "wb");
    if (!f) {
        return false;
    }
    const bool ok = std::fwrite(text.data(), 1, text.size(), f) == text.size();
    return std::fclose(f) == 0 && ok;
}

// Deterministic token id for a position, inside the vocabulary and away from
// the special ids 0..2.
llama_token token_for(int pos, int vocab) {
    const int64_t span = int64_t(vocab) - 3;
    return llama_token(3 + ((int64_t(pos) * 7 + 1) % span));
}

// One step: a single token at `pos`, requested as the only output.
bool decode_step(llama_context * ctx, int pos, int vocab, step_record & rec) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    if (!batch.token || !batch.pos || !batch.n_seq_id || !batch.seq_id || !batch.logits) {
        llama_batch_free(batch);
        return false;
    }
    batch.token[0]     = token_for(pos, vocab);
    batch.pos[0]       = pos;
    batch.n_seq_id[0]  = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0]    = 1;
    batch.n_tokens     = 1;
    rec.rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    return true;
}

// Run one arm over the schedule. The lane is selected by
// GGML_TP5_UNION_BINDER_EXACT, set here before the context is created because
// the knob is read once per context. Returns false only on an internal failure;
// the gate decision is the caller's.
bool run_arm(llama_model * model, int steps, int vocab, arm_result & arm) {
    auto cp = llama_context_default_params();
    cp.n_ctx           = uint32_t(n_ctx_requested);
    // n_ctx_kv stays 0: the physical KV capacity then follows n_ctx_seq, so
    // the exact arm allocates the requested 1000 cells and the padded arm the
    // stock 1024. That capacity difference is what separates the two lanes.
    cp.n_batch         = 1;
    cp.n_ubatch        = 1;
    cp.n_seq_max       = 1;
    cp.n_outputs_max   = 1;
    cp.n_threads       = 1;
    cp.n_threads_batch = 1;
    // the binder needs the single-stream unified cache it documents as
    // supported
    cp.kv_unified      = true;
    cp.offload_kqv     = false;
    cp.op_offload      = false;
    cp.no_perf         = false;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.embeddings      = false;

    setenv("GGML_TP5_UNION_BINDER", "1", 1);
    setenv("LLAMA_EXPERIMENTAL_VARLEN_NTOK", "1", 1);
    unsetenv("LLAMA_GRAPH_REUSE_DISABLE");
    // The lane is per arm, set before the context exists.
    if (arm.pad_lane) {
        setenv("GGML_TP5_UNION_BINDER_EXACT", "0", 1);
    } else {
        unsetenv("GGML_TP5_UNION_BINDER_EXACT");
    }

    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        arm.construct_error = "context creation failed";
        return false;
    }
    arm.constructed = true;
    arm.n_ctx       = llama_n_ctx(ctx);

    std::fprintf(stderr, "ARM_BEGIN arm=%s pad_lane=%d n_ctx=%u\n",
                 arm.name, arm.pad_lane ? 1 : 0, arm.n_ctx);

    for (int step = 0; step < steps; ++step) {
        step_record rec;
        rec.step = step;
        rec.n_kv = step + 1; // one token per step, one sequence, all resident

        if (!decode_step(ctx, step, vocab, rec)) {
            std::fprintf(stderr, "ARM_FAIL arm=%s batch allocation step=%d\n", arm.name, step);
            break;
        }

        llama_union_diagnostics d = {};
        rec.diag_have  = llama_get_union_diagnostics(ctx, &d);
        rec.bound_n_kv = d.bound_n_kv;

        const float * logits = llama_get_logits_ith(ctx, 0);
        if (logits) {
            rec.row_sha = sha256_hex(logits, size_t(vocab) * sizeof(float));
            ++arm.rows_captured;
        }

        // The lane's own expectation for this step.
        //
        //   exact  the binder recorded the graph and bound the used span.
        //   pad    the binder recorded NOTHING, so bound_n_kv is 0. The field
        //          is member-wise, so 0 reads as "no record", never as "bound
        //          at width 0"; the used span is not what this arm reports.
        const uint32_t want = arm.pad_lane ? 0u : uint32_t(rec.n_kv);
        const bool bound_ok = rec.diag_have && rec.bound_n_kv == want;
        if (rec.diag_have) {
            ++arm.steps_checked;
            if (bound_ok) {
                ++arm.bound_matching;
            }
            if (rec.bound_n_kv != 0) {
                ++arm.steps_with_record;
            }
        }

        std::fprintf(stderr,
                     "STEP arm=%s step=%d n_kv=%d rc=%d bound_n_kv=%u want=%u "
                     "diag_have=%d\n",
                     arm.name, rec.step, rec.n_kv, rec.rc, rec.bound_n_kv, want,
                     rec.diag_have ? 1 : 0);

        arm.records.push_back(rec);
        if (rec.rc != 0) {
            std::fprintf(stderr, "ARM_FAIL arm=%s decode step=%d rc=%d\n", arm.name, step, rec.rc);
            break;
        }
    }

    std::fprintf(stderr, "ARM_END arm=%s executed=%zu steps_checked=%d bound_matching=%d "
                 "rows_captured=%d n_ctx=%u\n",
                 arm.name, arm.records.size(), arm.steps_checked, arm.bound_matching,
                 arm.rows_captured, arm.n_ctx);

    llama_free(ctx);
    return true;
}

// Bitwise comparison of the two arms' rows, step by step. A step whose row was
// not captured on either arm is a mismatch, never a silent pass.
int count_equal_rows(const arm_result & a, const arm_result & b) {
    const size_t n = a.records.size() < b.records.size() ? a.records.size() : b.records.size();
    int equal = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!a.records[i].row_sha.empty() && a.records[i].row_sha == b.records[i].row_sha) {
            ++equal;
        }
    }
    return equal;
}

// The stock 256-cell pad the padded lane applies to a requested capacity.
uint32_t stock_n_ctx(uint32_t requested) {
    return (requested + n_ctx_stock_bucket - 1) / n_ctx_stock_bucket * n_ctx_stock_bucket;
}

void append_arm(std::string & out, const arm_result & a, int steps, bool bound_ok) {
    char buf[1024];
    out += "    {\n";
    std::snprintf(buf, sizeof(buf),
                  "      \"arm\": \"%s\",\n"
                  "      \"pad_lane\": %s,\n"
                  "      \"constructed\": %s,\n",
                  a.name, json_bool(a.pad_lane).c_str(), json_bool(a.constructed).c_str());
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
                  "      \"steps_executed\": %zu,\n"
                  "      \"steps_requested\": %d,\n"
                  "      \"steps_checked\": %d,\n"
                  "      \"bound_matching\": %d,\n"
                  "      \"rows_captured\": %d,\n"
                  "      \"status\": \"%s\",\n",
                  a.n_ctx, a.records.size(), steps, a.steps_checked, a.bound_matching,
                  a.rows_captured, bound_ok ? "pass" : "fail");
    out += buf;
    out += "      \"steps\": [\n";
    for (size_t i = 0; i < a.records.size(); ++i) {
        const step_record & r = a.records[i];
        std::snprintf(buf, sizeof(buf),
                      "        {\"step\": %d, \"n_kv\": %d, \"rc\": %d, \"diag_have\": %s, "
                      "\"bound_n_kv\": %u, \"row_sha\": \"%s\"}",
                      r.step, r.n_kv, r.rc, json_bool(r.diag_have).c_str(), r.bound_n_kv,
                      r.row_sha.c_str());
        out += buf;
        out += i + 1 < a.records.size() ? ",\n" : "\n";
    }
    out += "      ]\n    }";
}

// Report for a failure that happens before any decode.
std::string early_failure_report(const char * reason, int argc, char ** argv, int steps) {
    std::string body = "{\n  \"schema\": \"pad-lane-ab-gate/1\",\n";
    body += "  \"verdict\": \"fail\",\n";
    body += "  \"failure\": \"" + json_escape(reason) + "\",\n";
    if (argc >= 2) {
        body += "  \"model_path\": \"" + json_escape(argv[1]) + "\",\n";
    }
    char buf[256];
    std::snprintf(buf, sizeof(buf), "  \"steps_requested\": %d,\n", steps);
    body += buf;
    body += "  \"env_gate\": \"GGML_TP5_UNION_BINDER\",\n";
    body += "  \"env_lane\": \"GGML_TP5_UNION_BINDER_EXACT\",\n";
    body += "  \"not_run\": \"GPU/HIP/CUDA backends, relay, pipeline parallelism, "
            "recurrent or non-LLAMA architectures, production qualification.\",\n";
    append_argv(body, argc, argv);
    body += "  \"exit_code\": 1,\n";
    body += "  \"arms\": [\n"
            "    {\"arm\": \"exact\", \"status\": \"not_run\", \"steps\": []},\n"
            "    {\"arm\": \"pad\", \"status\": \"not_run\", \"steps\": []}\n  ]\n}\n";
    return body;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 3 || argc > 4) {
        std::fprintf(stderr,
                     "usage: %s <model.gguf> <out.json> [steps=%d]\n"
                     "  steps must be at least %d\n",
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
        if (errno || end == argv[3] || *end || v < 1 || v > INT_MAX / 2) {
            std::fprintf(stderr, "invalid steps: %s\n", argv[3]);
            return 2;
        }
        steps = int(v);
    }
    if (steps < steps_min) {
        std::fprintf(stderr, "FAIL: need steps >= %d (got %d)\n", steps_min, steps);
        return 2;
    }
    // The schedule must stay below the 256-cell capacity pad, so the padded
    // arm's capacity is the pad itself and not a larger multiple of it. This
    // bounds the CAPACITY observation, which is the one that separates the
    // lanes; it is not a bound on the bound width, because both lanes execute
    // at the used span.
    if (steps >= int(n_ctx_stock_bucket)) {
        std::fprintf(stderr, "FAIL: the schedule must stay below %u tokens (got %d)\n",
                     n_ctx_stock_bucket, steps);
        return 2;
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
    if (vocab <= 3) {
        std::fprintf(stderr, "FAIL: unusable vocabulary (%d)\n", vocab);
        llama_model_free(model);
        llama_backend_free();
        write_file(out_path, early_failure_report("unusable vocabulary", argc, argv, steps));
        return 1;
    }

    arm_result exact_arm;
    exact_arm.name     = "exact";
    exact_arm.pad_lane = false;

    arm_result pad_arm;
    pad_arm.name     = "pad";
    pad_arm.pad_lane = true;

    run_arm(model, steps, vocab, exact_arm);
    run_arm(model, steps, vocab, pad_arm);

    const int equal_rows = count_equal_rows(exact_arm, pad_arm);

    // Fixture properties, from observations only:
    //   the exact arm binds the used span at every step and keeps the
    //   requested capacity;
    //   the pad arm records nothing (bound_n_kv == 0 at every step) and pads
    //   the capacity;
    //   the two arms are distinguishable on the capacity, which is the
    //   observation that actually differs on this tree;
    //   the padded width does not change the numerics.
    const bool exact_bound_eq_used_span =
        exact_arm.constructed && exact_arm.steps_checked == steps &&
        exact_arm.bound_matching == steps;
    const bool pad_arm_has_no_record =
        pad_arm.constructed && pad_arm.steps_checked == steps &&
        pad_arm.bound_matching == steps && pad_arm.steps_with_record == 0;
    const bool capacity_differs =
        exact_arm.constructed && pad_arm.constructed &&
        exact_arm.n_ctx == uint32_t(n_ctx_requested) &&
        pad_arm.n_ctx == stock_n_ctx(uint32_t(n_ctx_requested)) &&
        exact_arm.n_ctx != pad_arm.n_ctx;
    // The distinctness check is the CAPACITY, not the bound width. On this
    // tree both lanes execute at the true used span, so requiring the bound
    // widths to differ would be asserting a mechanism that was deleted. This
    // still fails loudly if the lanes stop being distinguishable: equal
    // capacities fail it, and so does a pad arm that starts reporting a
    // binder record.
    const bool arms_distinct =
        exact_arm.constructed && pad_arm.constructed && capacity_differs &&
        exact_arm.steps_with_record == steps && pad_arm.steps_with_record == 0;
    const bool rows_bitwise_equal =
        equal_rows == steps && exact_arm.rows_captured == steps &&
        pad_arm.rows_captured == steps;

    const bool ok = exact_bound_eq_used_span && pad_arm_has_no_record && capacity_differs &&
                    arms_distinct && rows_bitwise_equal;

    char fail_reason_buf[512];
    fail_reason_buf[0] = '\0';
    const char * fail_reason = nullptr;
    if (!exact_arm.constructed || !pad_arm.constructed) {
        fail_reason = "context creation failed";
    } else if (!exact_bound_eq_used_span) {
        std::snprintf(fail_reason_buf, sizeof(fail_reason_buf),
                      "exact arm bound_n_kv != used span at %d of %d steps",
                      steps - exact_arm.bound_matching, steps);
        fail_reason = fail_reason_buf;
    } else if (!pad_arm_has_no_record) {
        std::snprintf(fail_reason_buf, sizeof(fail_reason_buf),
                      "pad arm reported a binder record at %d of %d steps "
                      "(bound_n_kv must be 0: EXACT=0 builds no record)",
                      pad_arm.steps_with_record, steps);
        fail_reason = fail_reason_buf;
    } else if (!capacity_differs) {
        std::snprintf(fail_reason_buf, sizeof(fail_reason_buf),
                      "capacity observation is not the lane difference (exact n_ctx %u, "
                      "pad n_ctx %u, requested %d)",
                      exact_arm.n_ctx, pad_arm.n_ctx, n_ctx_requested);
        fail_reason = fail_reason_buf;
    } else if (!arms_distinct) {
        std::snprintf(fail_reason_buf, sizeof(fail_reason_buf),
                      "arms are not distinct: exact record %d/%d steps, pad record %d/%d steps, "
                      "capacities %u and %u",
                      exact_arm.steps_with_record, steps, pad_arm.steps_with_record, steps,
                      exact_arm.n_ctx, pad_arm.n_ctx);
        fail_reason = fail_reason_buf;
    } else if (!rows_bitwise_equal) {
        std::snprintf(fail_reason_buf, sizeof(fail_reason_buf),
                      "rows differ at %d of %d steps (captured %d and %d): the padded width "
                      "changed the numerics",
                      steps - equal_rows, steps, exact_arm.rows_captured, pad_arm.rows_captured);
        fail_reason = fail_reason_buf;
    }

    std::string body = "{\n  \"schema\": \"pad-lane-ab-gate/1\",\n";
    body += ok ? "  \"verdict\": \"pass\",\n" : "  \"verdict\": \"fail\",\n";
    if (ok) {
        body += "  \"failure\": null,\n";
    } else {
        body += "  \"failure\": \"" + json_escape(fail_reason) + "\",\n";
    }
    body += "  \"model_path\": \"" + json_escape(model_path) + "\",\n";
    char buf[512];
    std::snprintf(buf, sizeof(buf), "  \"vocab\": %d,\n", vocab);
    body += buf;
    std::snprintf(buf, sizeof(buf), "  \"steps_requested\": %d,\n", steps);
    body += buf;
    std::snprintf(buf, sizeof(buf), "  \"steps_default\": %d,\n", steps_default);
    body += buf;
    std::snprintf(buf, sizeof(buf), "  \"steps_min\": %d,\n", steps_min);
    body += buf;
    std::snprintf(buf, sizeof(buf), "  \"n_ctx_requested\": %d,\n", n_ctx_requested);
    body += buf;
    std::snprintf(buf, sizeof(buf), "  \"n_ctx_stock\": %u,\n",
                  stock_n_ctx(uint32_t(n_ctx_requested)));
    body += buf;
    body += "  \"env_gate\": \"GGML_TP5_UNION_BINDER\",\n";
    body += "  \"env_lane\": \"GGML_TP5_UNION_BINDER_EXACT\",\n";
    body += "  \"gate_arm\": \"exact\",\n";
    body += "  \"fixture\": {\n";
    std::snprintf(buf, sizeof(buf),
                  "    \"arms_distinct\": %s,\n"
                  "    \"exact_record_steps\": %d,\n"
                  "    \"pad_record_steps\": %d,\n"
                  "    \"rows_bitwise_equal\": %s,\n"
                  "    \"rows_equal_steps\": %d,\n"
                  "    \"exact_bound_eq_used_span\": %s,\n"
                  "    \"pad_arm_has_no_record\": %s,\n"
                  "    \"capacity_differs\": %s\n"
                  "  },\n",
                  json_bool(arms_distinct).c_str(),
                  exact_arm.steps_with_record, pad_arm.steps_with_record,
                  json_bool(rows_bitwise_equal).c_str(), equal_rows,
                  json_bool(exact_bound_eq_used_span).c_str(),
                  json_bool(pad_arm_has_no_record).c_str(),
                  json_bool(capacity_differs).c_str());
    body += buf;
    body += "  \"limits\": \"CPU only; no GPU, relay, pipeline or production claim. "
            "Row identity is the SHA-256 of the requested F32 logits row per step. "
            "exact_bound_eq_used_span rests on bound_n_kv equalling the cumulative token "
            "count. pad_arm_has_no_record rests on bound_n_kv being 0 at every step: the "
            "EXACT=0 lane builds no record, so the field is member-wise 0 and reads as no "
            "record. capacity_differs rests on llama_n_ctx() being the requested value on "
            "the exact arm and its 256-cell pad on the padded arm. arms_distinct rests on "
            "the capacity differing AND on exactly one arm reporting records: the two "
            "lanes do NOT differ in bound width on this tree, because the 256-cell bucket "
            "floor was deleted and get_n_kv() rounds by n_pad == 1.\",\n";
    body += "  \"not_run\": \"GPU/HIP/CUDA backends, relay, pipeline parallelism, "
            "recurrent or non-LLAMA architectures, production qualification.\",\n";
    append_argv(body, argc, argv);
    std::snprintf(buf, sizeof(buf), "  \"exit_code\": %d,\n", ok ? 0 : 1);
    body += buf;
    body += "  \"arms\": [\n";
    append_arm(body, exact_arm, steps, exact_bound_eq_used_span);
    body += ",\n";
    append_arm(body, pad_arm, steps, pad_arm_has_no_record);
    body += "\n  ]\n}\n";

    const bool wrote = write_file(out_path, body);
    if (!wrote) {
        std::fprintf(stderr, "FAIL: cannot write %s\n", out_path);
    }

    // A failed JSON write leaves no artifact. The log must not say PASS for a
    // run that produced nothing, so the verdict carries the write result.
    const char * gate_verdict = !ok ? "FAIL" : wrote ? "PASS" : "FAIL reason=json_write";
    std::fprintf(stderr,
                 "GATE %s exact=%s(%d/%d bound) pad=%s(%d/%d no-record) distinct=%d "
                 "capacity_differs=%d rows_equal=%d/%d fixture=%d\n",
                 gate_verdict,
                 exact_bound_eq_used_span ? "pass" : "fail",
                 exact_arm.bound_matching, steps,
                 pad_arm_has_no_record ? "pass" : "fail",
                 pad_arm.bound_matching, steps,
                 arms_distinct ? 1 : 0, capacity_differs ? 1 : 0,
                 equal_rows, steps, ok ? 1 : 0);
    if (fail_reason) {
        std::fprintf(stderr, "GATE_FAILURE %s\n", fail_reason);
    }

    llama_model_free(model);
    llama_backend_free();
    return ok && wrote ? 0 : 1;
}
