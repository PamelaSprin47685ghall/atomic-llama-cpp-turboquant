// nkv_union_driver.cpp — production-path driver for the NKV union binder and
// the attention-view reader refusal (source tree: lead-fa-lower-20261030).
//
// Public llama.h API only: llama_model_load_from_file / llama_init_from_model /
// llama_decode / llama_memory_seq_rm / llama_attention_set_views /
// llama_attention_clear_views / llama_log_set. No internal headers, no mocks:
// every decode runs the real llama-graph / llama-kv-cache / union-binder code
// and the real ggml CPU backend of the linked tree (patched or stock f1c5faf).
//
// Per-scenario one process (see run-union-driver.sh); env gates:
//   GGML_TP5_NKV_TRUEWIDTH / GGML_TP5_NKV_COV / GGML_TP5_UNION_BINDER are read
//   per llama_context construction (llama-context.cpp:461-514) — the driver
//   sets them with setenv() before creating each context.
//   GGML_TP5_FA_RTLEN / GGML_TP5_FA_CHECKED_TAIL / GGML_TP5_FA_NKV_PARAM are
//   read once per process (llama-graph.cpp:36-64 static initializers) — the
//   fa-plain / fa-rtlen scenarios therefore run as separate processes.
//
// Output protocol:
//   SCENARIO <name> begin/end
//   DECODER step=<i> tag=<tag> rc=<r> n_tok=<n>
//   LOGITS step=<tag> sha256=<hex> min=<f> max=<f> finite=<n>/<n>
//   BINDER-LINE <captured NKV_BINDER_STATS line> | BINDER-LINE none
//   BINDER-COV  <captured NKV_BINDER_COV line>   | (absent when cov off)
//   CHECK <name> PASS | CHECK <name> FAIL <detail>
//   SCENARIO <name> checks=<n> failed=<n>          (final line; exit code 0/1)

#include "llama.h"

#include <dlfcn.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

// ---------------------------------------------------------------- logging --
static std::vector<std::string> g_log;
static bool g_capture = false;

static void driver_log_cb(ggml_log_level /*level*/, const char * text, void * /*ud*/) {
    if (g_capture && text != nullptr) {
        g_log.emplace_back(text);
    }
}

static void capture_begin() {
    g_log.clear();
    g_capture = true;
}

static void capture_end() {
    g_capture = false;
}

static bool find_line(const char * prefix, std::string & out) {
    for (const auto & l : g_log) {
        if (l.rfind(prefix, 0) == 0) {
            out = l;
            while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) {
                out.pop_back();
            }
            return true;
        }
    }
    return false;
}

// "k=v k=v ..." parser for the NKV_BINDER_STATS line
static std::map<std::string, std::string> parse_stats(const std::string & line) {
    std::map<std::string, std::string> kv;
    size_t pos = 0;
    while (pos < line.size()) {
        const size_t key0 = line.find_first_not_of(' ', pos);
        if (key0 == std::string::npos) {
            break;
        }
        const size_t eq = line.find('=', key0);
        if (eq == std::string::npos) {
            break;
        }
        const size_t sp = line.find(' ', eq);
        kv[line.substr(key0, eq - key0)] = line.substr(
                eq + 1, (sp == std::string::npos ? line.size() : sp) - eq - 1);
        pos = sp;
    }
    return kv;
}

// ----------------------------------------------------------------- sha256 --
// Compact FIPS 180-4 implementation for deterministic logits digests.
namespace {

struct Sha256 {
    uint32_t h[8];
    uint64_t len = 0;
    uint8_t  buf[64];
    size_t   buflen = 0;

    Sha256() {
        static const uint32_t iv[8] = {
            0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
            0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u };
        memcpy(h, iv, sizeof(h));
    }

    static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

    void block(const uint8_t * p) {
        static const uint32_t k[64] = {
            0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u,
            0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
            0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u,
            0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
            0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
            0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
            0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
            0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
            0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au,
            0x5b9cca4fu, 0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
            0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u };
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = (uint32_t(p[i*4]) << 24) | (uint32_t(p[i*4+1]) << 16) |
                   (uint32_t(p[i*4+2]) << 8) | uint32_t(p[i*4+3]);
        }
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i-15], 7) ^ rotr(w[i-15], 18) ^ (w[i-15] >> 3);
            const uint32_t s1 = rotr(w[i-2], 17) ^ rotr(w[i-2], 19) ^ (w[i-2] >> 10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
        uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t ch = (e & f) ^ (~e & g);
            const uint32_t t1 = hh + S1 + ch + k[i] + w[i];
            const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = S0 + mj;
            hh = g; g = f; f = e; e = d + t1;
            d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d;
        h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }

    void update(const void * data, size_t n) {
        const uint8_t * p = (const uint8_t *) data;
        len += n;
        while (n > 0) {
            const size_t take = 64 - buflen < n ? 64 - buflen : n;
            memcpy(buf + buflen, p, take);
            buflen += take;
            p += take;
            n -= take;
            if (buflen == 64) {
                block(buf);
                buflen = 0;
            }
        }
    }

    std::string hex() {
        uint8_t tail[128];
        memcpy(tail, buf, buflen);
        tail[buflen] = 0x80;
        const size_t zl = (buflen < 56 ? 56 : 120) - buflen;
        memset(tail + buflen + 1, 0, zl - 1);
        const uint64_t bits = len * 8;
        for (int i = 0; i < 8; ++i) {
            tail[buflen + zl + i] = uint8_t(bits >> (56 - 8*i));
        }
        block(tail);
        if (buflen >= 56) {
            block(tail + 64);
        }
        char out[65];
        for (int i = 0; i < 8; ++i) {
            snprintf(out + i*8, 9, "%08x", h[i]);
        }
        return std::string(out, 64);
    }
};

} // namespace

static std::string sha256_hex(const void * data, size_t nbytes) {
    Sha256 s;
    s.update(data, nbytes);
    return s.hex();
}
// vocabulary size of the LOADED model (set once per scenario after load);
// all logits rows are read at exactly this width
static int32_t g_n_vocab = 0;
// ----------------------------------------------------------------- checks --
struct Checks {
    int total = 0;
    int failed = 0;

    void check(const std::string & name, bool ok, const std::string & detail = "") {
        ++total;
        if (ok) {
            std::printf("CHECK %s PASS\n", name.c_str());
        } else {
            ++failed;
            std::printf("CHECK %s FAIL %s\n", name.c_str(), detail.c_str());
        }
    }
};

static Checks g_ck;

// optional cross-run logits reference (--dump-reflogits / --ref-logits)
static std::map<std::string, std::vector<float>> g_ref;
static FILE * g_refout = nullptr;

struct StepRec {
    std::string tag;
    int         rc = -999;
    int32_t     n_rows = 0;      // output rows of THIS decode (logits-flagged)
    std::vector<float> row0;     // output row 0 (digest basis; REFLOGITS has all rows)
    size_t      n_finite = 0;
    float       mn = 0.0f;
    float       mx = 0.0f;
};

static StepRec run_decode(llama_context * ctx, llama_batch & b, const std::string & tag) {
    StepRec r;
    r.tag = tag;
    r.rc = llama_decode(ctx, b);
    std::printf("DECODER step=%s rc=%d n_tok=%d\n", r.tag.c_str(), r.rc, b.n_tokens);
    // every step's rc is a hard check, walk steps included
    g_ck.check("decode-" + r.tag, r.rc == 0, "llama_decode failed");
    if (r.rc == 0) {
        // output rows of THIS decode = rows flagged for logits
        int32_t n_out = 0;
        for (int32_t i = 0; i < b.n_tokens; ++i) {
            n_out += b.logits[i] != 0;
        }
        r.n_rows = n_out;
        const float * all = llama_get_logits(ctx);   // [n_out * n_vocab]
        const float * lg = llama_get_logits_ith(ctx, 0);
        if (lg != nullptr && g_n_vocab > 0) {
            r.row0.assign(lg, lg + g_n_vocab);
            r.mn = r.mx = r.row0[0];
            for (float f : r.row0) {
                if (std::isfinite(f)) {
                    ++r.n_finite;
                    r.mn = f < r.mn ? f : r.mn;
                    r.mx = f > r.mx ? f : r.mx;
                }
            }
            // REFLOGITS: every output row of every decode (key tag#r[k]),
            // walk steps included; the per-step LOGITS digest covers row 0
            // only. NO fake fallback: if the full-row buffer is unavailable
            // we FAIL and emit NOTHING - later rows are never row-0 copies.
            if (g_refout != nullptr) {
                g_ck.check("refrows-buffer-" + r.tag, all != nullptr && n_out > 0, "llama_get_logits null or no output rows");
            }
            if (g_refout != nullptr) {
                for (int32_t row = 0; all != nullptr && row < n_out; ++row) {
                    const float * src = all + (size_t) row * g_n_vocab;
                    std::fprintf(g_refout, "REFLOGITS %s#r%d", r.tag.c_str(), row);
                    for (int32_t i = 0; i < g_n_vocab; ++i) {
                        std::fprintf(g_refout, " %.6e", (double) src[i]);
                    }
                    std::fprintf(g_refout, "\n");
                }
                std::fflush(g_refout);
            }
            // row-0 reference diff; a MISSING reference row (reference set
            // loaded but this tag absent) or a WIDTH mismatch must FAIL,
            // never silently skip
            const auto it = g_ref.find(r.tag + "#r0");
            if (it == g_ref.end()) {
                g_ck.check("ref-missing-row-" + r.tag, g_ref.empty(),
                           "reference set loaded but this tag has no REFLOGITS row");
            } else if (it->second.size() != r.row0.size()) {
                g_ck.check("ref-width-" + r.tag, false,
                           "reference row width mismatch");
            } else {
                float worst = 0.0f;
                for (size_t i = 0; i < r.row0.size(); ++i) {
                    const float d = fabsf(r.row0[i] - it->second[i]);
                    worst = d > worst ? d : worst;
                }
                std::printf("REFDIFF step=%s maxabs=%.6e\n", r.tag.c_str(), (double) worst);
                g_ck.check("refdiff-" + r.tag, worst <= 1e-4f, "logits differ from reference beyond 1e-4");
            }
        }
    }
    return r;
}

static void print_logits(const char * tag, const StepRec & r) {
    if (r.row0.empty()) {
        std::printf("LOGITS step=%s sha256=none finite=0/0\n", tag);
        return;
    }
    std::printf("LOGITS step=%s sha256=%s min=%.6f max=%.6f finite=%zu/%zu\n",
                tag, sha256_hex(r.row0.data(), r.row0.size() * sizeof(float)).c_str(),
                (double) r.mn, (double) r.mx, r.n_finite, r.row0.size());
}

// Reference logits are loaded BEFORE the model, so rows are parsed with their
// own recorded width (REFVOCAB header); consistency with the loaded model is
// checked at diff time (rows must match the loaded n_vocab).
static int32_t g_ref_vocab = 0;

// dynamic-length line reader: REFLOGITS rows scale with n_vocab (a real
// 128k-vocab row is ~1.5 MB), so no fixed buffer is used
static char * read_line_dyn(FILE * f, char ** buf, size_t * cap) {
    size_t len = 0;
    bool any = false;
    for (;;) {
        const int c = fgetc(f);
        if (c == EOF) {
            break;
        }
        any = true;
        if (len + 2 > *cap) {
            *cap = *cap == 0 ? 4096 : *cap * 2;
            *buf = (char *) std::realloc(*buf, *cap);
        }
        (*buf)[len++] = (char) c;
        if (c == '\n') {
            break;
        }
    }
    if (!any) {
        return nullptr;
    }
    if (len + 1 > *cap) {
        *cap = len + 1;
        *buf = (char *) std::realloc(*buf, *cap);
    }
    (*buf)[len] = '\0';
    return *buf;
}

static void load_ref(const char * path) {
    FILE * f = fopen(path, "r");
    if (f == nullptr) {
        g_ck.check("ref-load", false, std::string("cannot open ") + path);
        return;
    }
    char * line = nullptr;
    size_t cap = 0;
    while (read_line_dyn(f, &line, &cap) != nullptr) {
        if (std::strncmp(line, "REFVOCAB ", 9) == 0) {
            g_ref_vocab = (int32_t) std::strtol(line + 9, nullptr, 10);
            continue;
        }
        if (std::strncmp(line, "REFLOGITS ", 10) != 0) {
            if (line[0] != '\n' && line[0] != '\0') {   // blank lines are fine
                g_ck.check("ref-parse", false,
                           std::string("malformed line: ") + line);
            }
            continue;
        }
        const char * p = line + 10;
        char tag[256];
        size_t ti = 0;
        while (*p != '\0' && *p != ' ' && *p != '\n' && ti + 1 < sizeof(tag)) {
            tag[ti++] = *p++;
        }
        tag[ti] = '\0';
        if (ti == 0) {
            g_ck.check("ref-parse", false, "REFLOGITS row without tag");
            continue;
        }
        std::vector<float> v;
        char * end = nullptr;
        for (;;) {
            while (*p == ' ') {
                ++p;
            }
            if (*p == '\0' || *p == '\n') {
                break;
            }
            const double d = std::strtod(p, &end);
            if (end == p) {
                g_ck.check("ref-parse", false,
                           std::string("unparseable token in row ") + tag);
                break;
            }
            v.push_back((float) d);
            p = end;
        }
        if (v.empty()) {
            g_ck.check("ref-parse", false,
                       std::string("empty REFLOGITS row ") + tag);
            continue;
        }
        g_ref[tag] = std::move(v);
    }
    std::free(line);
    fclose(f);
    g_ck.check("ref-load", !g_ref.empty(), "no REFLOGITS rows parsed");
    if (!g_ref.empty()) {
        g_ck.check("ref-uniform-width", g_ref_vocab > 0, "REFVOCAB header missing");
    }
}



// ------------------------------------------------------------------ setup --
static void set_or_unset(const char * name, const char * val) {
    if (val != nullptr) {
        setenv(name, val, 1);
    } else {
        unsetenv(name);
    }
}

struct CtxCfg {
    uint32_t     n_ctx;
    bool         flash_attn;
    const char * gate_truewidth; // nullptr = unset; else "1"/"0"
    const char * gate_cov;
    const char * gate_union;
};

static llama_context * make_ctx(const llama_model * model, const CtxCfg & cfg) {
    // read per llama_context construction (llama-context.cpp:461-514)
    set_or_unset("GGML_TP5_NKV_TRUEWIDTH", cfg.gate_truewidth);
    set_or_unset("GGML_TP5_NKV_COV",       cfg.gate_cov);
    set_or_unset("GGML_TP5_UNION_BINDER",  cfg.gate_union);

    llama_context_params p = llama_context_default_params();
    p.n_ctx           = cfg.n_ctx;
    p.n_batch         = 512;
    p.n_ubatch        = 512;
    p.n_seq_max       = 2;
    p.n_threads       = 1;   // fixed thread count for reproducible logits
    p.n_threads_batch = 1;
    p.flash_attn_type = cfg.flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                                       : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    p.type_k          = GGML_TYPE_F32; // reader views require F32 K and V
    p.type_v          = GGML_TYPE_F32;
    p.kv_unified      = true;
    p.embeddings      = false;
    p.no_perf         = true;
    return llama_init_from_model(const_cast<llama_model *>(model), p);
}

// Reader-view set, validated against llama_kv_cache::prepare_attention_views:
//   V0: query (seq0, pos 4, virtual 4) — own pending key at pos 4 plus the
//       seq-1 key at pos 1, re-framed to effective rope position 3
//       (effective = query_virtual + key_pos - virtual = 4 + 1 - 2 = 3).
//   V1: query (seq1, pos 2, virtual 2) — own history 0..1 plus own key.
// Both include their own current key exactly once; every foreign entry
// exists in the cache at set time; virtual positions strictly increase and
// stay <= query_virtual_pos (prepare_attention_views constraints).
// Reader-view API resolved via dlsym so the same source builds and runs
// against a tree that lacks it (stock f1c5faf HAS the API — git-verified:
// include/llama.h is untouched by the cumulative patch — but a missing API
// must degrade to an explicit CHECK FAIL, never a wrong PASS).
static bool         (*g_fn_views_supported)(const llama_context *)                           = nullptr;
static const char * (*g_fn_views_reason)(const llama_context *)                              = nullptr;
static bool         (*g_fn_set_views)(llama_context *, const llama_attention_view *, size_t) = nullptr;
static void         (*g_fn_clear_views)(llama_context *)                                     = nullptr;

static void resolve_reader_api(Checks & ck) {
    g_fn_views_supported = (bool (*) (const llama_context *))
        dlsym(RTLD_DEFAULT, "llama_attention_views_supported");
    g_fn_views_reason = (const char * (*) (const llama_context *))
        dlsym(RTLD_DEFAULT, "llama_attention_views_unsupported_reason");
    g_fn_set_views = (bool (*) (llama_context *, const llama_attention_view *, size_t))
        dlsym(RTLD_DEFAULT, "llama_attention_set_views");
    g_fn_clear_views = (void (*) (llama_context *))
        dlsym(RTLD_DEFAULT, "llama_attention_clear_views");
    ck.check("reader-api-resolved",
             g_fn_views_supported != nullptr && g_fn_views_reason != nullptr &&
             g_fn_set_views != nullptr && g_fn_clear_views != nullptr,
             "reader-view API missing in the linked tree");
}

static void clear_views(llama_context * ctx, Checks & ck) {
    if (g_fn_clear_views != nullptr) {
        g_fn_clear_views(ctx);
        ck.check("clear-views", true, "");
    } else {
        ck.check("clear-views", false, "llama_attention_clear_views unavailable");
    }
}

static bool install_views(llama_context * ctx) {
    static const llama_attention_entry e0[2] = { {1, 1, 2}, {0, 4, 4} };
    static const llama_attention_entry e1[3] = { {1, 0, 0}, {1, 1, 1}, {1, 2, 2} };
    const llama_attention_view v[2] = {
        { 0, 4, 4, 2, e0 },
        { 1, 2, 2, 3, e1 },
    };
    if (g_fn_set_views != nullptr) {
        return g_fn_set_views(ctx, v, 2);
    }
    return false; // reader API missing: resolve_reader_api already reported
}

// batch rows on the stack; llama_batch_init allocates per-row seq slots
struct Rows {
    llama_token  tok[600];
    llama_pos    pos[600];
    llama_seq_id seq[600];
    int32_t      n = 0;
};

static void push_row(Rows & r, llama_seq_id seq, llama_pos pos, llama_token tok) {
    r.tok[r.n]  = tok;
    r.pos[r.n]  = pos;
    r.seq[r.n]  = seq;
    ++r.n;
}

static llama_batch make_batch(const Rows & r) {
    llama_batch b = llama_batch_init(r.n, 0, 1);
    b.n_tokens = r.n;
    for (int32_t i = 0; i < r.n; ++i) {
        b.token[i]     = r.tok[i];
        b.pos[i]       = r.pos[i];
        b.n_seq_id[i]  = 1;
        b.seq_id[i][0] = r.seq[i];
        b.logits[i]    = 1;
    }
    return b;
}

static llama_token tok_at(llama_pos p) {
    return (llama_token) ((7 * p + 3) % (g_n_vocab > 1 ? g_n_vocab : 64));
}

// ---------------------------------------------------------- trajectories --
using Steps = std::map<std::string, StepRec>;

static void logit_finite_check(Checks & ck, const char * tag, const StepRec & r) {
    ck.check(std::string("logits-finite-") + tag,
             r.rc == 0 && !r.row0.empty() && r.n_finite == r.row0.size(),
             "decode rc=" + std::to_string(r.rc) + " finite=" + std::to_string(r.n_finite) +
             " n=" + std::to_string(r.row0.size()));
}

// Plain trajectory on a 512-cell unified cache (no attention views):
//   prefill300: 300 tokens seq0 pos 0..299  -> fresh build, n_kv 512, mask 512
//   step1: 1 token pos 300                  -> out_ids 300 vs 1 -> binder entered,
//                                              nkv unchanged -> no-op fallback
//   step2: 1 token pos 301                  -> pure reuse
//   seq_rm(0, 100, -1)                      -> frontier move, n_kv 512 -> 256
//   step3: 1 token pos 100                  -> binder patch 512->256 APPLIED
//   step4: 1 token pos 101                  -> pure reuse at 256
//   walk pos 102..256                       -> at pos 256 used=257 -> n_kv 512
//                                              (<= record_cap) -> APPLIED
//   step6: 1 token pos 257                  -> pure reuse at 512
static void traj_plain(llama_context * ctx, Checks & ck, Steps & steps) {
    {
        Rows r;
        for (llama_pos p = 0; p < 300; ++p) {
            push_row(r, 0, p, tok_at(p));
        }
        llama_batch b = make_batch(r);
        steps["prefill300"] = run_decode(ctx, b, "prefill300");
        llama_batch_free(b);
        logit_finite_check(ck, "prefill300", steps["prefill300"]);
        print_logits("prefill300", steps["prefill300"]);
    }
    {
        Rows r;
        push_row(r, 0, 300, tok_at(300));
        llama_batch b = make_batch(r);
        steps["step1"] = run_decode(ctx, b, "step1");
        llama_batch_free(b);
        logit_finite_check(ck, "step1", steps["step1"]);
        print_logits("step1", steps["step1"]);
    }
    {
        Rows r;
        push_row(r, 0, 301, tok_at(301));
        llama_batch b = make_batch(r);
        steps["step2"] = run_decode(ctx, b, "step2");
        llama_batch_free(b);
        logit_finite_check(ck, "step2", steps["step2"]);
        print_logits("step2", steps["step2"]);
    }
    ck.check("seq-rm-frontier", llama_memory_seq_rm(llama_get_memory(ctx), 0, 100, -1),
             "llama_memory_seq_rm(seq0, 100, -1) returned false");
    {
        Rows r;
        push_row(r, 0, 100, tok_at(100));
        llama_batch b = make_batch(r);
        steps["step3"] = run_decode(ctx, b, "step3");
        llama_batch_free(b);
        logit_finite_check(ck, "step3", steps["step3"]);
        print_logits("step3", steps["step3"]);
    }
    {
        Rows r;
        push_row(r, 0, 101, tok_at(101));
        llama_batch b = make_batch(r);
        steps["step4"] = run_decode(ctx, b, "step4");
        llama_batch_free(b);
        logit_finite_check(ck, "step4", steps["step4"]);
        print_logits("step4", steps["step4"]);
    }
    for (llama_pos p = 102; p <= 256; ++p) {
        Rows r;
        push_row(r, 0, p, tok_at(p));
        llama_batch b = make_batch(r);
        const std::string tag = p == 256 ? "step5" : ("walk" + std::to_string(p));
        steps[tag] = run_decode(ctx, b, tag);
        llama_batch_free(b);
        logit_finite_check(ck, tag.c_str(), steps[tag]);   // every walk step checked
        print_logits(tag.c_str(), steps[tag]);
    }
    {
        Rows r;
        push_row(r, 0, 257, tok_at(257));
        llama_batch b = make_batch(r);
        steps["step6"] = run_decode(ctx, b, "step6");
        llama_batch_free(b);
        logit_finite_check(ck, "step6", steps["step6"]);
        print_logits("step6", steps["step6"]);
    }
}

static void drop_query_keys(llama_context * ctx, Checks & ck) {
    // remove the view query rows' own keys: the frontier moves (used cells
    // shrink) while the installed views keep referencing (seq,pos) pairs
    const bool r0 = llama_memory_seq_rm(llama_get_memory(ctx), 0, 4, -1);
    const bool r1 = llama_memory_seq_rm(llama_get_memory(ctx), 1, 2, -1);
    ck.check("seq-rm-query-own", r0, "llama_memory_seq_rm(seq0) returned false");
    ck.check("seq-rm-query-foreign", r1, "llama_memory_seq_rm(seq1) returned false");
}

static void decode_view_round(llama_context * ctx, Checks & ck, Steps & steps,
                              const char * tag) {
    ck.check(std::string("install-views-") + tag, install_views(ctx),
             "llama_attention_set_views=false");
    Rows r;
    push_row(r, 0, 4, tok_at(4));    // V0 query row
    push_row(r, 1, 2, tok_at(102));  // V1 query row (foreign entries + own)
    llama_batch b = make_batch(r);
    steps[tag] = run_decode(ctx, b, tag);
    llama_batch_free(b);
    logit_finite_check(ck, tag, steps[tag]);
    print_logits(tag, steps[tag]);
}

static void plain_one_token(llama_context * ctx, Checks & ck, Steps & steps,
                            const char * tag, llama_pos p) {
    Rows r;
    push_row(r, 0, p, tok_at(p));
    llama_batch b = make_batch(r);
    steps[tag] = run_decode(ctx, b, tag);
    llama_batch_free(b);
    logit_finite_check(ck, tag, steps[tag]);
    print_logits(tag, steps[tag]);
}

// -------------------------------------------------------- binder parsing --
struct BinderStats {
    bool present = false;
    std::map<std::string, std::string> kv;
};

static BinderStats binder_stats() {
    BinderStats s;
    std::string line;
    s.present = find_line("NKV_BINDER_STATS", line);
    if (s.present) {
        std::printf("BINDER-LINE %s\n", line.c_str());
        s.kv = parse_stats(line.substr(std::string("NKV_BINDER_STATS").size()));
        // missing fields must never silently read as zero-value successes
        g_ck.check("stats-has-attempts",  s.kv.count("attempts")  == 1, "field missing in stats line");
        g_ck.check("stats-has-fallbacks", s.kv.count("fallbacks") == 1, "field missing in stats line");
    } else {
        std::printf("BINDER-LINE none\n");
    }
    std::string cov;
    if (find_line("NKV_BINDER_COV", cov)) {
        std::printf("BINDER-COV %s\n", cov.c_str());
    }
    return s;
}

static unsigned long long stats_ull(const std::map<std::string, std::string> & kv,
                                    const char * key) {
    const auto it = kv.find(key);
    return it == kv.end() ? 0ull : strtoull(it->second.c_str(), nullptr, 10);
}

// ------------------------------------------------------------- scenarios --
// OFF: no binder gates at all -> no NKV_BINDER_STATS line, stock behavior
static void scenario_off_stock(llama_model * model, Checks & ck) {
    const CtxCfg cfg = { 512, false, nullptr, nullptr, nullptr };
    capture_begin();
    llama_context * ctx = make_ctx(model, cfg);
    ck.check("ctx-create", ctx != nullptr, "llama_init_from_model failed");
    if (ctx != nullptr) {
        Steps steps;
        traj_plain(ctx, ck, steps);
        llama_free(ctx);
    }
    capture_end();
    std::string line;
    ck.check("binder-silent", !find_line("NKV_BINDER_STATS", line),
             "unexpected line: " + line);
}

// plain-rebind: TRUEWIDTH (+COV), no attention views — the non-reader control.
// The binder runs, patches at the frontier moves and refuses nothing reader
// related: refused_view == 0.
static void scenario_plain_rebind(llama_model * model, Checks & ck) {
    const CtxCfg cfg = { 512, false, "1", "1", nullptr };
    capture_begin();
    llama_context * ctx = make_ctx(model, cfg);
    ck.check("ctx-create", ctx != nullptr, "llama_init_from_model failed");
    if (ctx != nullptr) {
        Steps steps;
        traj_plain(ctx, ck, steps);
        llama_free(ctx);
    }
    capture_end();
    const BinderStats st = binder_stats();
    ck.check("stats-present", st.present, "NKV_BINDER_STATS missing");
    if (st.present) {
        ck.check("attempts>=3",       stats_ull(st.kv, "attempts")       >= 3, "attempts too low");
        ck.check("applied>=2",        stats_ull(st.kv, "applied")        >= 2, "applied too low");
        ck.check("fallbacks>=1",      stats_ull(st.kv, "fallbacks")      >= 1, "fallbacks too low");
        ck.check("refused_field-present", st.kv.count("refused_view") == 1, "r8 format must carry refused_view (cov on)");
        ck.check("refused_view==0",   stats_ull(st.kv, "refused_view") == 0, "views refused without views installed");
    }
}

// refuse-views-live: views stay set across repeated frontier moves; every
// view decode is refused (cache-state detector). 3 view rounds expected.
static void scenario_views_live(llama_model * model, Checks & ck) {
    const CtxCfg cfg = { 256, false, "1", "1", nullptr };
    capture_begin();
    llama_context * ctx = make_ctx(model, cfg);
    ck.check("ctx-create", ctx != nullptr, "llama_init_from_model failed");
    if (ctx != nullptr) {
        Steps steps;
        // prefill seq0 (4 rows) then seq1 (2 rows) so both foreign and own
        // keys exist before the views are installed
        {
            Rows r;
            for (llama_pos p = 0; p < 4; ++p) { push_row(r, 0, p, tok_at(p)); }
            llama_batch b = make_batch(r);
            steps["prefill0"] = run_decode(ctx, b, "prefill0");
            llama_batch_free(b);
            print_logits("prefill0", steps["prefill0"]);
        }
        {
            Rows r;
            push_row(r, 1, 0, tok_at(100));
            push_row(r, 1, 1, tok_at(101));
            llama_batch b = make_batch(r);
            steps["prefill1"] = run_decode(ctx, b, "prefill1");
            llama_batch_free(b);
            print_logits("prefill1", steps["prefill1"]);
        }
        decode_view_round(ctx, ck, steps, "view1");
        drop_query_keys(ctx, ck);
        decode_view_round(ctx, ck, steps, "view2");
        drop_query_keys(ctx, ck);
        decode_view_round(ctx, ck, steps, "view3");
        llama_free(ctx);
    }
    capture_end();
    const BinderStats st = binder_stats();
    ck.check("stats-present", st.present, "NKV_BINDER_STATS missing");
    if (st.present) {
        ck.check("refused_view>=2", stats_ull(st.kv, "refused_view") >= 2, "reader refusals missing");
        ck.check("fallbacks>=2",    stats_ull(st.kv, "fallbacks")    >= 2, "fallbacks missing");
    }
}

// refuse-views-clear: after one view decode the views are CLEARED; the next
// frontier move + decode still must be refused — by the graph-side detector
// that walks the live reader graph (reader-audit fix). Evidence: refused_view
// exceeds the single pre-clear refusal.
static void scenario_views_clear(llama_model * model, Checks & ck) {
    const CtxCfg cfg = { 256, false, "1", "1", nullptr };
    capture_begin();
    llama_context * ctx = make_ctx(model, cfg);
    ck.check("ctx-create", ctx != nullptr, "llama_init_from_model failed");
    if (ctx != nullptr) {
        Steps steps;
        {
            Rows r;
            for (llama_pos p = 0; p < 4; ++p) { push_row(r, 0, p, tok_at(p)); }
            llama_batch b = make_batch(r);
            steps["prefill0"] = run_decode(ctx, b, "prefill0");
            llama_batch_free(b);
            print_logits("prefill0", steps["prefill0"]);
        }
        {
            Rows r;
            push_row(r, 1, 0, tok_at(100));
            push_row(r, 1, 1, tok_at(101));
            llama_batch b = make_batch(r);
            steps["prefill1"] = run_decode(ctx, b, "prefill1");
            llama_batch_free(b);
            print_logits("prefill1", steps["prefill1"]);
        }
        decode_view_round(ctx, ck, steps, "view1");
        // build-then-clear: the live graph is now the reader graph
        clear_views(ctx, ck);   // build-then-clear: the live graph is now the reader graph
        drop_query_keys(ctx, ck);
        // plain 1-row decode over the reader graph: the graph-side detector
        // must refuse this rebind (views are empty, the graph is not)
        plain_one_token(ctx, ck, steps, "plain-after-clear", 4);
        llama_free(ctx);
    }
    capture_end();
    const BinderStats st = binder_stats();
    ck.check("stats-present", st.present, "NKV_BINDER_STATS missing");
    if (st.present) {
        ck.check("refused_view>=2", stats_ull(st.kv, "refused_view") >= 2,
                 "graph-side reader refusal missing");
        ck.check("fallbacks>=2",    stats_ull(st.kv, "fallbacks")    >= 2, "fallbacks missing");
    }
}

// refuse-cov-off: identical trajectory to refuse-views-live's first two view
// rounds but WITHOUT GGML_TP5_NKV_COV: the refusal is unconditional now, and
// the stats line must print in the r7 format (no refused_view field).
static void scenario_cov_off(llama_model * model, Checks & ck) {
    const CtxCfg cfg = { 256, false, "1", nullptr, nullptr };
    capture_begin();
    llama_context * ctx = make_ctx(model, cfg);
    ck.check("ctx-create", ctx != nullptr, "llama_init_from_model failed");
    if (ctx != nullptr) {
        Steps steps;
        {
            Rows r;
            for (llama_pos p = 0; p < 4; ++p) { push_row(r, 0, p, tok_at(p)); }
            llama_batch b = make_batch(r);
            steps["prefill0"] = run_decode(ctx, b, "prefill0");
            llama_batch_free(b);
            print_logits("prefill0", steps["prefill0"]);
        }
        {
            Rows r;
            push_row(r, 1, 0, tok_at(100));
            push_row(r, 1, 1, tok_at(101));
            llama_batch b = make_batch(r);
            steps["prefill1"] = run_decode(ctx, b, "prefill1");
            llama_batch_free(b);
            print_logits("prefill1", steps["prefill1"]);
        }
        decode_view_round(ctx, ck, steps, "view1");
        drop_query_keys(ctx, ck);
        decode_view_round(ctx, ck, steps, "view2");
        llama_free(ctx);
    }
    capture_end();
    const BinderStats st = binder_stats();
    ck.check("stats-present", st.present, "NKV_BINDER_STATS missing");
    if (st.present) {
        ck.check("r7-format", st.kv.count("refused_view") == 0,
                 "refused_view must not be printed with the cov gate off");
        ck.check("fallbacks>=2", stats_ull(st.kv, "fallbacks") >= 2, "fallbacks missing");
    }
}

// refuse-outputs-lane: adds GGML_TP5_UNION_BINDER; after the reader graph is
// live, a plain 1-row decode (n_outputs 2 -> 1) must be refused by BOTH the
// nkv lane (graph-side reader walk) and the n_outputs lane.
static void scenario_outputs_lane(llama_model * model, Checks & ck) {
    const CtxCfg cfg = { 256, false, "1", "1", "1" };
    capture_begin();
    llama_context * ctx = make_ctx(model, cfg);
    ck.check("ctx-create", ctx != nullptr, "llama_init_from_model failed");
    if (ctx != nullptr) {
        Steps steps;
        {
            Rows r;
            for (llama_pos p = 0; p < 4; ++p) { push_row(r, 0, p, tok_at(p)); }
            llama_batch b = make_batch(r);
            steps["prefill0"] = run_decode(ctx, b, "prefill0");
            llama_batch_free(b);
            print_logits("prefill0", steps["prefill0"]);
        }
        {
            Rows r;
            push_row(r, 1, 0, tok_at(100));
            push_row(r, 1, 1, tok_at(101));
            llama_batch b = make_batch(r);
            steps["prefill1"] = run_decode(ctx, b, "prefill1");
            llama_batch_free(b);
            print_logits("prefill1", steps["prefill1"]);
        }
        decode_view_round(ctx, ck, steps, "view1");
        clear_views(ctx, ck);
        drop_query_keys(ctx, ck);
        plain_one_token(ctx, ck, steps, "plain-after-clear", 4);
        llama_free(ctx);
    }
    capture_end();
    const BinderStats st = binder_stats();
    ck.check("stats-present", st.present, "NKV_BINDER_STATS missing");
    if (st.present) {
        ck.check("refused_view>=2", stats_ull(st.kv, "refused_view") >= 2, "reader refusals missing");
        ck.check("fallbacks>=3",    stats_ull(st.kv, "fallbacks")    >= 3, "outputs-lane refusal missing");
    }
}

// FA scenarios: GGML_TP5_FA_* are process-static (llama-graph.cpp:36-64), so
// the runner starts one process per gate set. fa-rtlen additionally verifies
// the fail-closed src[5] lane end-to-end on the real CPU FA kernels.
static void scenario_fa(llama_model * model, Checks & ck, bool rtlen) {
    // FA gates are static per process; only the binder slot gates differ:
    // fa-rtlen keeps TRUEWIDTH+COV on (slot exists, binder blocked by
    // GGML_TP5_FA_NKV_PARAM), fa-plain runs gate-free (no slot, no line).
    const CtxCfg cfg = rtlen ? CtxCfg{ 512, true, "1", "1", nullptr }
                             : CtxCfg{ 512, true, nullptr, nullptr, nullptr };
    capture_begin();
    llama_context * ctx = make_ctx(model, cfg);
    ck.check("ctx-create", ctx != nullptr, "llama_init_from_model failed");
    if (ctx != nullptr) {
        Steps steps;
        traj_plain(ctx, ck, steps);
        llama_free(ctx);
    }
    capture_end();
    const BinderStats st = binder_stats();
    if (rtlen) {
        ck.check("stats-present", st.present, "NKV_BINDER_STATS missing");
        if (st.present) {
            ck.check("attempts==0",     stats_ull(st.kv, "attempts") == 0, "FA_NKV_PARAM must block every binder entry");
            ck.check("refused_view==0", stats_ull(st.kv, "refused_view") == 0, "no reader refusals here");
        }
    } else {
        std::string line;
        ck.check("binder-silent", !find_line("NKV_BINDER_STATS", line),
                 "unexpected line: " + line);
    }
}

// ------------------------------------------------------------------ main --
static int run_scenario(const std::string & sc, const std::string & model_path) {
    std::printf("SCENARIO %s begin\n", sc.c_str());
    Checks & ck = g_ck;   // SINGLE tally: run_decode/binder_stats checks and
                          // scenario checks land in the same failure state

    llama_log_set(driver_log_cb, nullptr);
    resolve_reader_api(ck);

    llama_model * model = llama_model_load_from_file(model_path.c_str(),
                                                     llama_model_default_params());
    ck.check("model-load", model != nullptr, "llama_model_load_from_file failed");
    if (model != nullptr) {
        g_n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        ck.check("vocab-nonempty", g_n_vocab > 0, "loaded model has no tokens");
        if (g_ref_vocab > 0) {
            ck.check("ref-vocab-match", g_ref_vocab == g_n_vocab,
                     "reference logits width " + std::to_string(g_ref_vocab) +
                     " vs model " + std::to_string(g_n_vocab));
        }
        if (g_refout != nullptr) {
            std::fprintf(g_refout, "REFVOCAB %d\n", g_n_vocab);
            std::fflush(g_refout);
        }
    }

    if (model != nullptr) {
        if      (sc == "off-stock")          scenario_off_stock(model, ck);
        else if (sc == "plain-rebind")       scenario_plain_rebind(model, ck);
        else if (sc == "refuse-views-live")  scenario_views_live(model, ck);
        else if (sc == "refuse-views-clear") scenario_views_clear(model, ck);
        else if (sc == "refuse-cov-off")     scenario_cov_off(model, ck);
        else if (sc == "refuse-outputs")     scenario_outputs_lane(model, ck);
        else if (sc == "fa-plain")           scenario_fa(model, ck, false);
        else if (sc == "fa-rtlen")           scenario_fa(model, ck, true);
        else {
            ck.check("scenario-name", false, "unknown scenario " + sc);
        }
        llama_model_free(model);
    }

    std::printf("SCENARIO %s checks=%d failed=%d\n", sc.c_str(), ck.total, ck.failed);
    return ck.failed == 0 ? 0 : 1;
}

int main(int argc, char ** argv) {
    std::string sc;
    std::string model_path;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--scenario") == 0 && i + 1 < argc) {
            sc = argv[++i];
        } else if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (std::strcmp(argv[i], "--dump-reflogits") == 0 && i + 1 < argc) {
            g_refout = std::fopen(argv[++i], "w");
        } else if (std::strcmp(argv[i], "--ref-logits") == 0 && i + 1 < argc) {
            load_ref(argv[++i]);
        } else {
            std::fprintf(stderr, "unknown arg %s\n", argv[i]);
            return 2;
        }
    }
    if (sc.empty() || model_path.empty()) {
        std::fprintf(stderr,
                     "usage: %s --model PATH.gguf --scenario NAME [--ref-logits F] [--dump-reflogits F]\n"
                     "scenarios: off-stock plain-rebind refuse-views-live refuse-views-clear\n"
                     "           refuse-cov-off refuse-outputs fa-plain fa-rtlen\n",
                     argv[0]);
        return 2;
    }
    return run_scenario(sc, model_path);
}
