// nout_floor1_driver.cpp
//
// 12022: real llama_context long-residency driver for the NOUT floor1 lane
// (env LLAMA_NOUT_FLOOR1_REUSE, strict "1"; see tree/src/llama-context.cpp
// :508-534 and tree/src/llama-nout-state.h:66-101 in the source tree).
//
// One process; two persistent llama_context instances (floor1 OFF reference +
// floor1 ON experiment) over the SAME llama_model. Arms run in per-step
// lockstep; per-step full-vocab logits rows are compared bit-exact (CPU
// single-thread) with an optional tolerance (--tol). Output-count grid per
// request: 0,1,2,3(k),0 plus a 1,0 segment; positions restart at 0 per
// request; per-request KV cleanup via llama_memory_seq_rm on BOTH arms.
// The dedicated sampler-transition phase (R4) attaches one sampler chain per
// arm (greedy on the OFF arm, greedy on the ON arm) so that both arms share
// an identical token conversion: logits to tokens via greedy argmax, first
// max on ties = the "sampler 转换" step.
//
// Correlation: every llama_decode is bracketed by JSONL events carrying
// (arm, request, step); `step` counts decodes per context and equals the
// runtime trace's dec= counter on the ON arm.

#include "llama.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include <unistd.h> // getpid, POSIX setenv/unsetenv

namespace {

// ---- options ---------------------------------------------------------------

struct Options {
    std::string model_path;
    std::string event_log_path = "-"; // "-" = stdout

    uint32_t n_ctx     = 512;  // whole script must stay inside one n_kv bucket
    uint32_t n_batch   = 8;    // >= grid_tokens: grid batches never split
    uint32_t n_ubatch  = 8;    // >= grid_tokens: exactly one ubatch per decode
    uint32_t n_seq_max = 8;    // seq_id room for every request + probe
    int32_t  threads   = 1;    // single-thread CPU path (bit-exact contract)

    int32_t grid_tokens = 4;   // tokens per grid phase ubatch (constant shape)
    int32_t k_out       = 3;   // k-output phase, 2 < k <= grid_tokens
    int32_t gen_tokens  = 8;   // greedy steps per request
    int32_t requests    = 3;   // R1, R2, R3 (R3 re-decodes R1's script)
    long    seed        = 42;  // token-script LCG seed

    double tol              = 0.0; // 0 = bit-exact (CPU single-thread default)
    bool   probe_empty_rows = false;
};

Options g_opt;

// fixed grid: report:50 counts 0,1,2,k,0; report:52 adds the 1,0 segment
// (adjacent 0->1, 1->0 and 1->2->k->0 coverage per request)
const int32_t kGridPhases[] = { 0, 1, 2, -3, 0, 1, 0 }; // -3 encodes k
const int32_t kNGridPhases  = (int32_t) (sizeof(kGridPhases) / sizeof(kGridPhases[0]));

// ---- tiny JSONL emitter ----------------------------------------------------

FILE * g_events = nullptr; // JSONL event log ('-' = stdout), flushed per line

void json_string(FILE * f, const std::string & s) {
    fputc('"', f);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  fputs("\\\"", f); break;
            case '\\': fputs("\\\\", f); break;
            case '\n': fputs("\\n", f);  break;
            case '\r': fputs("\\r", f);  break;
            default:
                if (c < 0x20) {
                    fprintf(f, "\\u%04x", c);
                } else {
                    fputc(c, f);
                }
        }
    }
    fputc('"', f);
}

void ev_open(const char * type) {
    fprintf(g_events, "{\"pid\":%d,\"ts_us\":%" PRId64 ",\"type\":", (int) getpid(),
            (int64_t) llama_time_us());
    json_string(g_events, type);
}
void ev_end() {
    fputc('}', g_events);
    fputc('\n', g_events);
    fflush(g_events);
}

void ev_raw(const char * type, const char * fields) { // fields = ",\"k\":v,..."
    fprintf(g_events, "{\"pid\":%d,\"ts_us\":%" PRId64 ",\"type\":", (int) getpid(),
            (int64_t) llama_time_us());
    json_string(g_events, type);
    fputs(fields, g_events);
    ev_end();
}

void ev_str(const char * type, const char * key, const std::string & val) {
    fprintf(g_events, "{\"pid\":%d,\"ts_us\":%" PRId64 ",\"type\":", (int) getpid(),
            (int64_t) llama_time_us());
    json_string(g_events, type);
    fprintf(g_events, ",\"%s\":", key);
    json_string(g_events, val);
    ev_end();
}
void ev_i(const char * key, long long v) {
    fprintf(g_events, ",\"%s\":%lld", key, v);
}

void ev_u(const char * key, unsigned long long v) {
    fprintf(g_events, ",\"%s\":%llu", key, v);
}

void ev_b(const char * key, bool v) {
    fprintf(g_events, ",\"%s\":%s", key, v ? "true" : "false");
}

void ev_f(const char * key, double v) {
    fprintf(g_events, ",\"%s\":%.9g", key, v);
}

void ev_s(const char * key, const std::string & val) {
    fprintf(g_events, ",\"%s\":", key);
    json_string(g_events, val);
}


// ---- runtime log tap -------------------------------------------------------

int g_cur_arm = 0; // 0 = OFF, 1 = ON; also set around llama_free (dtor stats)

struct TraceRec { // last NOUT1_FLOOR_TRACE line of the current arm
    bool               seen   = false;
    unsigned long long dec    = 0;
    int                reuse  = 0;
    unsigned           n_out  = 0;
    unsigned           key    = 0;
    unsigned           stored = 0;
    std::string        miss;
};

struct TraceStats { // NOUT1_FLOOR_TRACE_STATS (context destruction)
    bool               seen        = false;
    unsigned long long dec         = 0;
    unsigned long long reuse       = 0;
    unsigned long long miss_first  = 0;
    unsigned long long miss_dis    = 0;
    unsigned long long miss_ubatch = 0;
    unsigned long long miss_out    = 0;
    unsigned long long miss_smpl   = 0;
    unsigned long long miss_cpar   = 0;
    unsigned long long miss_inputs = 0;
    unsigned long long builds      = 0;
    unsigned long long builds_fb   = 0;
    unsigned long long bind_zero   = 0;
    unsigned long long skip        = 0;
    unsigned long long gen         = 0;
};

TraceRec   g_trace[2];
TraceStats g_trace_stats[2];

void forward_log(enum ggml_log_level /*level*/, const char * text, void * /*user*/) {
    if (text == nullptr) {
        return;
    }
    fprintf(stderr, "[nout-driver arm=%s] %s", g_cur_arm ? "ON" : "OFF", text);

    const char * p = strstr(text, "NOUT1_FLOOR_TRACE ");
    if (p != nullptr) {
        TraceRec t;
        char miss[16] = "-";
        if (sscanf(p, "NOUT1_FLOOR_TRACE dec=%llu reuse=%d miss=%15s n_out=%u key=%u stored=%u",
                   &t.dec, &t.reuse, miss, &t.n_out, &t.key, &t.stored) == 6) {
            t.seen = true;
            t.miss = miss;
            g_trace[g_cur_arm] = t;
        }
        return;
    }

    p = strstr(text, "NOUT1_FLOOR_TRACE_STATS ");
    if (p != nullptr) {
        TraceStats s;
        if (sscanf(p,
            "NOUT1_FLOOR_TRACE_STATS dec=%llu reuse=%llu miss(first=%llu disabled=%llu ubatch=%llu outputs=%llu samplers=%llu cparams=%llu inputs=%llu) builds=%llu builds_fb=%llu bind_zero=%llu compute_skip=%llu gen=%llu",
            &s.dec, &s.reuse, &s.miss_first, &s.miss_dis, &s.miss_ubatch, &s.miss_out,
            &s.miss_smpl, &s.miss_cpar, &s.miss_inputs, &s.builds, &s.builds_fb,
            &s.bind_zero, &s.skip, &s.gen) == 14) {
            s.seen = true;
            g_trace_stats[g_cur_arm] = s;
        }
    }
}

// ---- token script ----------------------------------------------------------

struct Lcg {
    uint64_t state;
    explicit Lcg(long seed) : state(((uint64_t) seed) ^ 0x9E3779B97F4A7C15ull) {}
    uint32_t next() {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        return (uint32_t) ((state >> 33) & 0x7fffffffu);
    }
};

// ---- per-step records ------------------------------------------------------

struct RowOut { // one output row of one arm for one step
    llama_token argmax   = -1;   // greedy conversion of this row
    uint64_t    checksum = 0;    // FNV-1a over the raw row bytes
    bool        finite   = true;
};

struct StepRec { // one decode step of one arm
    int                 arm       = 0;
    int                 step      = 0;   // 1-based decode ordinal per context
    int                 phase     = -1;  // grid phase index (-1 = gen segment)
    int32_t             n_tokens  = 0;
    int32_t             n_out     = 0;   // requested output count (flags)
    int32_t             n_out_seen = 0;  // rows actually readable (runtime truth)
    llama_pos           pos0      = 0;
    llama_seq_id        seq       = 0;
    uint32_t            n_kv_est  = 0;   // driver-side cell estimate
    bool                kv_bucket_crossed = false;
    std::vector<RowOut> rows;
    // runtime trace (ON arm only; parsed from NOUT1_FLOOR_TRACE)
    bool               t_seen    = false;
    int                t_reuse   = 0;
    unsigned           t_n_out   = 0;
    unsigned           t_stored  = 0;
    unsigned long long t_dec     = 0;
    unsigned long long t_key     = 0;
    std::string        t_miss;
};

struct CmpRec { // OFF-vs-ON comparison result for one step
    int   request     = 0;
    int   step        = 0;
    int   n_out       = 0;
    int   rows_off    = 0;
    int   rows_on     = 0;
    bool  rows_match  = true;
    bool  argmax_ok   = true;
    bool  finite      = true;
    bool  bitexact    = true;
    float max_abs     = 0.0f;
    float max_rel     = 0.0f;
    bool  pass        = true;
};

struct Verdict {
    long long steps        = 0;
    long long compares     = 0;
    long long mismatches   = 0;
    long long gen_div      = 0;
    long long decode_err   = 0;
    long long api_err      = 0;
    long long row_probe_bad = 0;   // 0-output step exposed a logits row
    long long bucket_cross = 0;
    long long kv_rm_miss   = 0;    // seq_rm removed nothing on either arm
    bool      pass         = true;
};

Verdict g_verdict;

uint64_t fnv1a(const float * row, int32_t n) {
    uint64_t h = 1469598103934665603ull;
    const unsigned char * p = (const unsigned char *) row;
    for (int32_t i = 0; i < n * (int32_t) sizeof(float); ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

llama_token argmax(const float * row, int32_t n_vocab) {
    int32_t best = 0;
    for (int32_t i = 1; i < n_vocab; ++i) {
        if (row[i] > row[best]) { // strictly greater: first max on ties
            best = i;
        }
    }
    return (llama_token) best;
}

// ---- arm state -------------------------------------------------------------

// per-request KV isolation lives on the context; the driver tracks one live
// request per arm at a time and mirrors the cell accounting for its n_kv
// estimate (the runtime's own n_kv comes from the trace/instrumentation merge)

struct Cells {
    std::unordered_map<llama_seq_id, int64_t> per_seq;
    uint32_t total() const {
        uint32_t n = 0;
        for (const auto & kv : per_seq) {
            n += (uint32_t) kv.second;
        }
        return n;
    }
};

struct ArmState {
    const char * name = nullptr; // "OFF" / "ON"
    int          arm  = 0;       // 0 / 1
    llama_context * ctx = nullptr;
    Cells        cells;          // driver-side per-seq cell estimate
    uint32_t     last_step = 0;  // per-context monotonic decode ordinal
    uint32_t     last_n_kv = 0;  // previous step's cell estimate (bucket check)
    std::vector<StepRec> steps;  // full per-step record (trace + rows)
};

ArmState g_arm[2];

llama_context * g_ctx[2] = { nullptr, nullptr };

int32_t g_n_vocab = 0;

// OFF-arm row stash for the same-step compare (the OFF decode always runs
// first, and the ON decode that follows invalidates nothing on the OFF
// context's own logits buffer - buffers are per context)

std::vector<std::vector<float>> g_off_rows;

// ---- context params + arm creation -----------------------------------------

llama_context_params make_cparams(const Options & opt) {
    llama_context_params p = llama_context_default_params();
    p.n_ctx           = opt.n_ctx;
    p.n_batch         = opt.n_batch;
    p.n_ubatch        = opt.n_ubatch;
    p.n_seq_max       = opt.n_seq_max;
    p.n_threads       = opt.threads;
    p.n_threads_batch = opt.threads;
    p.embeddings      = false;
    p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED; // CPU bit-exact path
    p.no_perf         = false;
    p.kv_unified      = true;  // shared cell pool; seq_id masks attention
    p.offload_kqv     = true;
    p.op_offload      = true;
    p.n_samplers      = 0;     // no backend sampler chain: floor1 precondition
    p.samplers        = nullptr;
    p.cb_eval         = nullptr;
    p.abort_callback  = nullptr;
    return p;
}

void ev_context_new(int arm) {
    char buf[512];
    snprintf(buf, sizeof(buf),
        ",\"arm\":\"%s\",\"n_ctx\":%u,\"n_batch\":%u,\"n_ubatch\":%u,"
        "\"n_seq_max\":%u,\"n_seq_max_actual\":%u,\"n_threads\":%d,"
        "\"flash_attn\":\"%s\",\"kv_unified\":true,\"n_samplers\":0,"
        "\"gpu_offload_supported\":%s",
        arm ? "ON" : "OFF", g_opt.n_ctx, g_opt.n_batch, g_opt.n_ubatch,
        g_opt.n_seq_max, (unsigned) llama_n_seq_max(g_ctx[arm]),
        g_opt.threads, llama_flash_attn_type_name(LLAMA_FLASH_ATTN_TYPE_DISABLED),
        llama_supports_gpu_offload() ? "true" : "false");
    ev_raw("context_new", buf);
}

bool arm_create(int arm, const struct llama_model * model) {
    // env is read inside the llama_context ctor (llama-context.cpp:518-534);
    // set immediately before this arm's creation, never touched afterwards
    char val[2] = { arm ? '1' : '0', 0 };
    setenv("LLAMA_NOUT_FLOOR1_REUSE", val, 1);
    setenv("LLAMA_NOUT_FLOOR1_TRACE", "1", 1); // OFF arm stays silent: lane disarmed

    g_cur_arm = arm;
    llama_context_params cparams = make_cparams(g_opt);
    g_ctx[arm] = llama_init_from_model(const_cast<llama_model *>(model), cparams);
    if (g_ctx[arm] == nullptr) {
        ev_str("context_new_failed", "arm", arm ? "ON" : "OFF");
        return false;
    }
    ev_context_new(arm);
    // instrumentation: tag this context so the runtime's (pid, ctx, step)
    // events merge with the driver's (arm, request, step) events 1:1; the
    // LLAMA_NOUT_INSTR patch must be applied to the tree BEFORE building
    llama_nout_instrument_set_tag(g_ctx[arm], arm ? "ON" : "OFF");
    return true;
}

// ---- request/step script ---------------------------------------------------

struct Request {
    int                id   = 0;
    llama_seq_id       seq  = 0;
    std::vector<llama_token> tokens; // grid phases + first gen token (script)
    int32_t            n_steps = 0;
};

struct StepSpec {
    int         step     = 0;   // 1-based decode ordinal per context
    int         phase    = -1;  // grid phase index (-1 = gen segment)
    int         req_step = 0;   // 1-based WITHIN-request step ordinal
    int32_t     n_tokens = 0;
    int32_t     n_out    = 0;   // requested output count (flags)
    llama_pos   pos0     = 0;
    llama_token  token    = 0;   // single-token (gen) steps
    const llama_token * tokens = nullptr; // multi-token (grid) steps
};

// R3 re-decodes R1's token script on a fresh seq_id after R1/R2 were removed
std::vector<Request> build_requests() {
    std::vector<Request> reqs;
    const int32_t n_grid_tokens = kNGridPhases * g_opt.grid_tokens;
    for (int32_t r = 0; r < g_opt.requests; ++r) {
        Request req;
        req.id  = r + 1;
        req.seq = (llama_seq_id) (r % (int32_t) g_opt.n_seq_max);
        // R3 re-uses R1's script (cross-request determinism check); R1/R2 get
        // their own draws so R2's steps cannot accidentally echo R1's
        const long seed = (r == 2) ? g_opt.seed : g_opt.seed + 7919L * r;
        Lcg lcg(seed);
        for (int32_t i = 0; i < n_grid_tokens + 1; ++i) {
            req.tokens.push_back((llama_token) (lcg.next() % (uint32_t) g_n_vocab));
        }
        req.n_steps = kNGridPhases + g_opt.gen_tokens;
        reqs.push_back(req);
    }
    // R3 = R1's script, verbatim (same tokens, same positions)
    if (reqs.size() >= 3) {
        reqs[2].tokens = reqs[0].tokens;
    }
    return reqs;
}

// fill one step's batch: n_tokens tokens of one sequence, flags on the last
// n_out tokens (report:50 "0 时显式全部 logits flags 为 0")
void fill_batch(llama_batch & batch, const Request & req, const StepSpec & st) {
    batch.n_tokens = st.n_tokens;
    for (int32_t i = 0; i < st.n_tokens; ++i) {
        batch.token[i]     = st.tokens ? st.tokens[i] : st.token;
        batch.pos[i]       = st.pos0 + i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = req.seq;
        batch.logits[i]    = (i >= st.n_tokens - st.n_out) ? 1 : 0;
    }
}

// decode one step on one arm; verifies the row contract; merges the
// NOUT1_FLOOR_TRACE line fired during this decode (log tap) into the record.
// `full_rows` (OFF arm only) additionally stashes the raw rows for the
// same-step compare against the ON arm. Returns false only on a decode/API
// error (the caller aborts the run).
bool decode_step(ArmState & arm, const Request & req, const StepSpec & st,
                 llama_batch & batch, std::vector<std::vector<float>> * full_rows) {
    fill_batch(batch, req, st);

    g_cur_arm = arm.arm;

    // per-context ordinal allocated BEFORE the step_begin event and used in
    // every event of this step (step is never a correlation key)
    const int step = ++arm.last_step;

    ev_open("step_begin");
    ev_i("arm", arm.arm);
    ev_i("request", req.id);
    ev_i("step", step);
    ev_i("phase", st.phase);
    ev_i("n_tokens", st.n_tokens);
    ev_i("n_out", st.n_out);
    ev_i("pos0", st.pos0);
    // instrumentation correlation: the runtime's (pid, ctx, step) events use
    // the same per-context id and per-context decode ordinal (ON arm only;
    // the OFF arm's floor1 lane is disarmed, so no trace fires there)
    ev_i("ctx_id", (long long)(uintptr_t) llama_nout_instrument_context_id(arm.ctx));
    ev_end();

    const int32_t ret = llama_decode(arm.ctx, batch);
    llama_synchronize(arm.ctx);

    if (ret != 0) {
        ++g_verdict.decode_err;
        g_verdict.pass = false;
        char b[160];
        snprintf(b, sizeof(b), ",\"arm\":\"%s\",\"request\":%d,\"step\":%d,\"ret\":%d",
                 arm.name, req.id, step, ret);
        ev_raw("decode_error", b);
        return false;
    }

    StepRec rec;
    rec.arm      = arm.arm;
    rec.step     = step;             // per-context monotonic decode ordinal
    rec.phase    = st.phase;
    rec.n_tokens = st.n_tokens;
    rec.n_out    = st.n_out;
    rec.pos0     = st.pos0;
    rec.seq      = req.seq;

    rec.n_kv_est          = arm.cells.total();
    rec.kv_bucket_crossed = (arm.last_n_kv / 256u) != (rec.n_kv_est / 256u);
    arm.last_n_kv         = rec.n_kv_est;
    if (rec.kv_bucket_crossed) {
        ++g_verdict.bucket_cross;
        char b[160];
        snprintf(b, sizeof(b), ",\"arm\":\"%s\",\"request\":%d,\"step\":%d,\"n_kv_est\":%u",
                 arm.name, req.id, step, rec.n_kv_est);
        ev_raw("kv_bucket_cross", b);
    }

    if (st.n_out == 0) {
        // report:60 "0 输出不得读取不存在的 logits 行" - no row read at all;
        // --probe-empty-rows opts into the negative API probe only (a probe
        // that must return nullptr and would otherwise violate report:60)
        if (g_opt.probe_empty_rows) {
            const float * row = llama_get_logits_ith(arm.ctx, 0);
            const bool non_null = row != nullptr;
            if (non_null) {
                ++g_verdict.row_probe_bad;
                g_verdict.pass = false;
            }
            char b[256];
            snprintf(b, sizeof(b), ",\"arm\":\"%s\",\"request\":%d,\"step\":%d,\"row\":0,\"non_null\":%s",
                     arm.name, req.id, step, non_null ? "true" : "false");
            ev_raw("row_probe", b);
        }
    } else {
        for (int32_t j = 0; j < st.n_out; ++j) {
            const float * row = llama_get_logits_ith(arm.ctx, st.n_tokens - st.n_out + j);
            if (row == nullptr) {
                // flagged row missing: the runtime truth contradicts the
                // request markers (report:48 "实际输出 count 必须由 runtime
                // 记录核实，不能仅信请求标记")
                ++g_verdict.api_err;
                g_verdict.pass = false;
                char b[256];
                snprintf(b, sizeof(b), ",\"arm\":\"%s\",\"request\":%d,\"step\":%d,\"row\":%d",
                         arm.name, req.id, step, j);
                ev_raw("row_missing", b);
                return false;
            }
            RowOut r;
            r.argmax   = argmax(row, g_n_vocab);
            r.checksum = fnv1a(row, g_n_vocab);
            for (int32_t d = 0; d < g_n_vocab; ++d) {
                if (!std::isfinite(row[d])) {
                    r.finite = false;
                    break;
                }
            }
            rec.rows.push_back(r);
            if (full_rows != nullptr) {
                full_rows->emplace_back(row, row + g_n_vocab);
            }
        }
    }
    rec.n_out_seen = (int32_t) rec.rows.size();

    // merge the runtime trace into the record (ON arm only: the OFF arm's
    // lane is disarmed, so no NOUT1_FLOOR_TRACE line can exist there - any
    // such line would be unexpected and is recorded verbatim)
    const TraceRec & t = g_trace[arm.arm];
    rec.t_seen   = t.seen;
    rec.t_reuse  = t.seen ? t.reuse : 0;
    rec.t_n_out  = t.seen ? t.n_out : 0;
    rec.t_stored = t.seen ? t.stored : 0;
    rec.t_dec    = t.seen ? t.dec : 0;
    rec.t_key    = t.seen ? t.key : 0;
    rec.t_miss   = t.seen ? t.miss : std::string();
    g_trace[arm.arm] = TraceRec(); // consumed

    arm.steps.push_back(rec);

    char b[768];
    snprintf(b, sizeof(b),
        ",\"arm\":\"%s\",\"request\":%d,\"step\":%d,\"n_out_seen\":%d,"
        "\"n_kv_est\":%u,\"bucket_crossed\":%s,"
        "\"trace_seen\":%s,\"trace_reuse\":%d,\"trace_miss\":\"%s\","
        "\"trace_dec\":%llu,\"trace_n_out\":%u,\"trace_stored\":%u",
        arm.name, req.id, step, rec.n_out_seen, rec.n_kv_est,
        rec.kv_bucket_crossed ? "true" : "false", rec.t_seen ? "true" : "false",
        rec.t_miss.c_str(), (unsigned long long) rec.t_dec, rec.t_n_out, rec.t_stored);
    ev_raw("step_end", b);
    ++g_verdict.steps;
    return true;
}

// ---- the compare -----------------------------------------------------------

// OFF-vs-ON comparison for one step: full-vocab rows per token index
// (report:60 "按 token index 通过 llama_get_logits_ith 取每个标记输出的完整
// vocab 行，对照 token ID、位置、输出行数、数据"), bit-exact expected on the
// single-thread CPU path; --tol relaxes to a recorded max abs/relative error
// budget. Missing or extra rows fail (report:60 "缺行或额外行失败").
CmpRec compare_step(const Request & req, const StepSpec & st, const ArmState & off, ArmState & on) {
    CmpRec c;
    c.request    = req.id;
    c.step       = off.steps.empty() ? 0 : off.steps.back().step;
    c.rows_off   = off.steps.empty() ? 0 : off.steps.back().n_out_seen;
    c.rows_on    = on.steps.empty() ? 0 : on.steps.back().n_out_seen;
    c.rows_match = c.rows_off == c.rows_on;
    c.argmax_ok  = true;
    c.finite     = true;
    c.bitexact   = true;

    if (c.rows_match && c.rows_off > 0) {
        const StepRec & ro = off.steps.back();
        for (int32_t j = 0; j < c.rows_off && j < (int32_t) ro.rows.size() &&
                            j < (int32_t) g_off_rows.size(); ++j) {
            c.argmax_ok = c.argmax_ok && ro.rows[j].argmax == g_arm[1].steps.back().rows[j].argmax;
            c.finite    = c.finite && ro.rows[j].finite && g_arm[1].steps.back().rows[j].finite;
            const std::vector<float> & a = g_off_rows[j];
            const float * b = llama_get_logits_ith(on.ctx, st.n_tokens - st.n_out + j);
            if (b == nullptr) {
                c.rows_match = false;
                break;
            }
            for (int32_t d = 0; d < g_n_vocab; ++d) {
                const float x = a[d];
                const float y = b[d];
                c.bitexact = c.bitexact && (memcmp(&x, &y, sizeof(float)) == 0);
                const float diff = std::fabs(x - y);
                c.max_abs = std::max(c.max_abs, diff);
                const float denom = std::max(std::fabs(x), std::fabs(y));
                if (denom > 0.0f) {
                    c.max_rel = std::max(c.max_rel, diff / denom);
                } else if (diff > 0.0f) {
                    c.max_rel = std::max(c.max_rel, diff);
                }
            }
        }
    }

    c.pass = c.rows_match && c.argmax_ok && c.finite &&
             (c.bitexact || (g_opt.tol > 0.0 && c.max_abs <= g_opt.tol));
    ++g_verdict.compares;
    if (!c.pass) {
        ++g_verdict.mismatches;
        g_verdict.pass = false;
    }

    char b[512];
    snprintf(b, sizeof(b),
        ",\"request\":%d,\"step\":%d,\"n_out\":%d,\"rows_off\":%d,\"rows_on\":%d,"
        "\"rows_match\":%s,\"argmax_ok\":%s,\"finite\":%s,\"bitexact\":%s,"
        "\"max_abs\":%.9g,\"max_rel\":%.9g,\"tol\":%.9g,\"pass\":%s",
        c.request, c.step, c.n_out, c.rows_off, c.rows_on,
        c.rows_match ? "true" : "false", c.argmax_ok ? "true" : "false",
        c.finite ? "true" : "false", c.bitexact ? "true" : "false",
        (double) c.max_abs, (double) c.max_rel, g_opt.tol, c.pass ? "true" : "false");
    ev_raw("compare", b);
    return c;
}

// ---- the run ---------------------------------------------------------------

// one request's step list: grid phases (n_tokens = grid_tokens, counts vary)
std::vector<StepSpec> build_phases(const Request & req) {
    std::vector<StepSpec> phases;
    int req_step = 0; // 1-based within-request step ordinal
    for (int32_t p = 0; p < kNGridPhases; ++p) {
        StepSpec st;
        st.phase    = p;
        st.n_tokens = g_opt.grid_tokens;
        st.n_out    = kGridPhases[p] < 0 ? g_opt.k_out : kGridPhases[p];
        st.pos0     = p * g_opt.grid_tokens;
        st.tokens   = req.tokens.data() + p * g_opt.grid_tokens;
        st.req_step = ++req_step;
        phases.push_back(st);
    }
    for (int32_t t = 0; t < g_opt.gen_tokens; ++t) {
        StepSpec st;
        st.phase    = -1;
        st.n_tokens = 1;
        st.n_out    = 1;
        st.pos0     = g_opt.grid_tokens * kNGridPhases + t;
        // the first gen token is the script's; the continuation is the OFF
        // arm's greedy argmax of the previous step (forced into BOTH arms)
        st.token    = (t == 0) ? req.tokens.back() : 0;
        st.req_step = ++req_step;
        phases.push_back(st);
    }
    return phases;
}

bool run_request(Request & req, llama_batch & batch) {
    char b[256];
    snprintf(b, sizeof(b), ",\"request\":%d,\"seq\":%d,\"steps\":%d", req.id,
             (int) req.seq, req.n_steps);
    ev_raw("request_begin", b);

    std::vector<StepSpec> phases = build_phases(req);
    bool ok = true;
    for (auto & st : phases) {
        g_off_rows.clear(); // per-step stash: stale rows would alias into the compare
        // OFF arm first (reference), then ON arm; the OFF rows are stashed
        // for the same-step compare
        if (!decode_step(g_arm[0], req, st, batch, &g_off_rows)) { ok = false; break; }
        if (!decode_step(g_arm[1], req, st, batch, nullptr))     { ok = false; break; }
        const CmpRec c = compare_step(req, st, g_arm[0], g_arm[1]);
        // R3-vs-R1 temporal check: same script tokens, same positions, same
        // context lifetime — the OFF arm's R3 record must equal R1's per row
        // (argmax + checksum); a mismatch is a real finding, not a benign
        // layout difference (both runs start from a cleaned cache)
        if (req.id == 3) {
            const size_t r1_idx = (size_t)(st.req_step - 1);
            if (r1_idx < g_arm[0].steps.size()) {
                const StepRec & r1 = g_arm[0].steps[r1_idx];
                const StepRec & r3 = g_arm[0].steps.back();
                bool rows_eq    = r1.rows.size() == r3.rows.size();
                bool argmax_eq  = rows_eq;
                bool checksum_eq = rows_eq;
                for (size_t j = 0; rows_eq && j < r3.rows.size(); ++j) {
                    argmax_eq   = argmax_eq   && r1.rows[j].argmax   == r3.rows[j].argmax;
                    checksum_eq = checksum_eq && r1.rows[j].checksum == r3.rows[j].checksum;
                }
                const bool pass3 = rows_eq && argmax_eq && checksum_eq;
                char bb[320];
                snprintf(bb, sizeof(bb), ",\"request\":3,\"req_step\":%d,\"step\":%d,\"phase\":%d,\"rows_eq\":%s,\"argmax_eq\":%s,\"checksum_eq\":%s", st.req_step, r3.step, st.phase, rows_eq ? "true" : "false", argmax_eq ? "true" : "false", checksum_eq ? "true" : "false");
                ev_raw("temporal_check", bb);
                if (!pass3) { ++g_verdict.mismatches; g_verdict.pass = false; }
            }
        }
    }

    // identical per-request KV cleanup on BOTH arms (report:48)
    for (int a = 0; a < 2; ++a) {
        const llama_memory_t mem = llama_get_memory(g_ctx[a]);
        const bool removed = llama_memory_seq_rm(mem, req.seq, 0, -1);
        g_arm[a].cells.per_seq[req.seq] = 0;
        char bb[256];
        snprintf(bb, sizeof(bb), ",\"arm\":\"%s\",\"request\":%d,\"seq\":%d,\"removed\":%s",
                 g_arm[a].name, req.id, (int) req.seq, removed ? "true" : "false");
        ev_raw("kv_cleanup", bb);
        if (!removed) {
            ++g_verdict.kv_rm_miss;
            g_verdict.pass = false;
        }
    }

    snprintf(b, sizeof(b), ",\"request\":%d,\"steps\":%zu,\"mismatches\":%lld",
             req.id, g_arm[0].steps.size(), (long long) g_verdict.mismatches);
    ev_raw("request_end", b);
    return ok;
}

// ---- the sampler-transition phase (report:17) ------------------------------
//
// Real backend-sampler state transitions on BOTH arms, in lockstep: add,
// replace (only after a successful add), remove, remove-again (no-op), each
// followed by one 1-token decode step (n_out = 1) on both arms. Per
// report:17, every state-changing transition triggers reserve
// (sched_need_reserve), which destroys the old result; the old NKV record is
// marked invalid before replacement so it cannot be reused across the
// switch. The reserve itself is internal: the driver records the
// transition's return value and the following decode; the instrumentation's
// reserve/lane events (merge key pid + ctx) prove the reserve fired and that
// the floor1 lane re-armed after the removal.
//
// Per-arm INDEPENDENT chain objects (no shared sampler state across
// contexts): g_chain[arm] holds this arm's own chain object. A shared
// sampler object would carry one sampler's mutable state across two
// contexts.
//
// Documented blocker (not worked around): on a CPU-only build a successful
// add/replace requires a chain whose elements all implement backend_init and
// return true (llama-sampler.cpp:700-726 chain_backend_init; can_offload at
// llama-context.cpp:2585-2586 also requires chain_n > 0). If the linked
// build has no backend-capable chain, every attach is rejected (return
// false; the sampler set stays empty, the lane stays armed) and the
// replace/reserve-firing transitions' preconditions never hold — the driver
// records the observed outcome per transition instead of claiming the
// success path ran.

// per-arm INDEPENDENT chain objects: [arm][0] = add-chain (greedy),
// [arm][1] = replace-chain (dist). Two contexts must never share one sampler
// object: the sampler carries its own mutable state per context.
llama_sampler * g_chain[2][2] = {};

bool run_sampler_phase(llama_batch & batch) {
    llama_sampler_chain_params sp = llama_sampler_chain_default_params();
    sp.no_perf = true;
    for (int a = 0; a < 2; ++a) {
        g_chain[a][0] = llama_sampler_chain_init(sp);
        llama_sampler_chain_add(g_chain[a][0], llama_sampler_init_greedy());
        g_chain[a][1] = llama_sampler_chain_init(sp);
        llama_sampler_chain_add(g_chain[a][1], llama_sampler_init_dist((uint32_t) g_opt.seed));
    }

    // a fresh request for the phase (a fresh seq_id, positions continuing
    // after the request scripts)
    Request req;
    req.id      = g_opt.requests + 1;
    req.seq     = (llama_seq_id) (req.id % (int32_t) g_opt.n_seq_max);
    req.n_steps = 4;
    Lcg lcg(g_opt.seed + 104729L);
    req.tokens.push_back((llama_token) (lcg.next() % (uint32_t) g_n_vocab));

    char b[256];
    snprintf(b, sizeof(b), ",\"request\":%d,\"seq\":%d,\"steps\":%d", req.id,
             (int) req.seq, req.n_steps);
    ev_raw("request_begin", b);

    bool ok = true;
    llama_pos pos = 0;

    // one decode step on both arms after a transition; the OFF rows are
    // stashed for the same-step compare
    auto step_after = [&](const char * action, int accepted) {
        StepSpec st;
        st.phase    = -2; // sampler-phase marker
        st.n_tokens = 1;
        st.n_out    = 1;
        st.pos0     = pos;
        st.token    = req.tokens.back();
        g_off_rows.clear();
        for (int a = 0; a < 2 && ok; ++a) {
            g_cur_arm = a;
            ev_open("sampler_phase");
            ev_i("arm", a);
            ev_i("request", req.id);
            ev_s("action", action);
            ev_i("accepted", accepted);
            ev_end();
            if (!decode_step(g_arm[a], req, st, batch, a == 0 ? &g_off_rows : nullptr)) {
                ok = false;
            }
        }
        if (ok) {
            compare_step(req, st, g_arm[0], g_arm[1]);
        }
        ++pos;
    };

    // S1 add
    const int add0 = llama_set_sampler(g_ctx[0], req.seq, g_chain[0][0]);
    const int add1 = llama_set_sampler(g_ctx[1], req.seq, g_chain[1][0]);
    step_after("add", add0 && add1);

    // S2 replace (only after an accepted add on BOTH arms); with the add
    // rejected, the replace precondition never holds and is skipped
    if (add0 && add1) {
        const int rep0 = llama_set_sampler(g_ctx[0], req.seq, g_chain[0][1]);
        const int rep1 = llama_set_sampler(g_ctx[1], req.seq, g_chain[1][1]);
        step_after("replace", rep0 && rep1);
    } else {
        ev_str("sampler_phase_skipped", "reason", "replace needs an accepted add (runbook blocker)");
    }

    // S3 remove (re-arms the floor1 lane)
    const int rm0 = llama_set_sampler(g_ctx[0], req.seq, nullptr);
    const int rm1 = llama_set_sampler(g_ctx[1], req.seq, nullptr);
    step_after("remove", rm0 && rm1);

    // S4 remove again: no-op, no state change, no reserve (report:17)
    const int rn0 = llama_set_sampler(g_ctx[0], req.seq, nullptr);
    const int rn1 = llama_set_sampler(g_ctx[1], req.seq, nullptr);
    step_after("remove_noop", rn0 && rn1);

    for (int a = 0; a < 2; ++a) {
        const llama_memory_t mem = llama_get_memory(g_ctx[a]);
        const bool removed = llama_memory_seq_rm(mem, req.seq, 0, -1);
        g_arm[a].cells.per_seq[req.seq] = 0;
        char bb[256];
        snprintf(bb, sizeof(bb), ",\"arm\":\"%s\",\"request\":%d,\"seq\":%d,\"removed\":%s",
                 g_arm[a].name, req.id, (int) req.seq, removed ? "true" : "false");
        ev_raw("kv_cleanup", bb);
        if (!removed) {
            ++g_verdict.kv_rm_miss;
            g_verdict.pass = false;
        }
    }

    snprintf(b, sizeof(b), ",\"request\":%d,\"steps\":%d,\"mismatches\":%lld",
             req.id, req.n_steps, (long long) g_verdict.mismatches);
    ev_raw("request_end", b);
    return ok;
}

bool run() {
    std::vector<Request> reqs = build_requests();
    llama_batch batch = llama_batch_init(std::max(g_opt.grid_tokens, 1), 0, 1);
    if (batch.token == nullptr || batch.pos == nullptr || batch.n_seq_id == nullptr ||
        batch.seq_id == nullptr || batch.logits == nullptr) {
        ev_str("batch_init_failed", "reason", "llama_batch_init returned null arrays");
        return false;
    }

    bool ok = true;
    for (auto & req : reqs) {
        if (!run_request(req, batch)) { ok = false; }
    }
    // the dedicated sampler-transition phase (report:17): real add / replace
    // / remove / remove-noop transitions on both arms, one decode per
    // transition, reserve + lane recovery observed via the instrumentation
    if (!run_sampler_phase(batch)) { ok = false; }
    llama_batch_free(batch);
    return ok;
}

} // namespace

// ---- usage + args ----------------------------------------------------------

void usage(FILE * f) {
    fprintf(f,
        "usage: nout_floor1_driver --model PATH [options]\n"
        "\n"
        "  --model PATH            real GGUF (required)\n"
        "  --event-log PATH        JSONL event log ('-' = stdout, default)\n"
        "  --n-ctx N               context size (default 512)\n"
        "  --n-batch N             logical batch cap (default 8)\n"
        "  --n-ubatch N            physical batch cap (default 8)\n"
        "  --n-seq-max N           sequence room (default 8)\n"
        "  --threads N             CPU threads, 1 = bit-exact path (default 1)\n"
        "  --grid-tokens N         tokens per grid phase ubatch (default 4)\n"
        "  --k N                   k-output phase, 2 < k <= grid-tokens (default 3)\n"
        "  --gen N                 greedy steps per request (default 8)\n"
        "  --requests N            requests R1..Rn (default 3; R3 re-decodes R1)\n"
        "  --seed N                token-script LCG seed (default 42)\n"
        "  --tol F                 max abs logits diff, 0 = bit-exact (default 0)\n"
        "  --probe-empty-rows      negative probe: row 0 must be unreadable at 0-out\n");
}

bool parse_args(int argc, char ** argv) {
    std::vector<std::string> a(argv + 1, argv + argc);
    for (size_t i = 0; i < a.size(); ++i) {
        const std::string & s = a[i];
        if (s == "--model" && i + 1 < a.size()) { g_opt.model_path = a[++i]; continue; }
        if (s == "--event-log" && i + 1 < a.size()) { g_opt.event_log_path = a[++i]; continue; }
        if (s == "--n-ctx" && i + 1 < a.size()) { g_opt.n_ctx = (uint32_t) atoi(a[++i].c_str()); continue; }
        if (s == "--n-batch" && i + 1 < a.size()) { g_opt.n_batch = (uint32_t) atoi(a[++i].c_str()); continue; }
        if (s == "--n-ubatch" && i + 1 < a.size()) { g_opt.n_ubatch = (uint32_t) atoi(a[++i].c_str()); continue; }
        if (s == "--n-seq-max" && i + 1 < a.size()) { g_opt.n_seq_max = (uint32_t) atoi(a[++i].c_str()); continue; }
        if (s == "--threads" && i + 1 < a.size()) { g_opt.threads = atoi(a[++i].c_str()); continue; }
        if (s == "--grid-tokens" && i + 1 < a.size()) { g_opt.grid_tokens = atoi(a[++i].c_str()); continue; }
        if (s == "--k" && i + 1 < a.size()) { g_opt.k_out = atoi(a[++i].c_str()); continue; }
        if (s == "--gen" && i + 1 < a.size()) { g_opt.gen_tokens = atoi(a[++i].c_str()); continue; }
        if (s == "--requests" && i + 1 < a.size()) { g_opt.requests = atoi(a[++i].c_str()); continue; }
        if (s == "--seed" && i + 1 < a.size()) { g_opt.seed = atol(a[++i].c_str()); continue; }
        if (s == "--tol" && i + 1 < a.size()) { g_opt.tol = atof(a[++i].c_str()); continue; }
        if (s == "--probe-empty-rows") { g_opt.probe_empty_rows = true; continue; }
        return false;
    }
    return !g_opt.model_path.empty();
}

int main(int argc, char ** argv) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(stdout);
            return 0;
        }
    }
    if (!parse_args(argc, argv)) {
        usage(stderr);
        fprintf(stderr, "error: --model is required\n");
        return 3;
    }

    // defensive env hygiene: the single-factor floor1 comparison must not
    // inherit any other experimental lane (report:58 "envelope 关闭")
    unsetenv("GGML_TP5_NOUT_REUSE");        // envelope lane
    unsetenv("LLAMA_GRAPH_UNION_NKV_RUNTIME");
    unsetenv("GGML_TP5_UNION_BINDER");
    unsetenv("GGML_TP5_NKV_TRUEWIDTH");
    unsetenv("LLAMA_GRAPH_REUSE_DISABLE");  // would kill reuse entirely

    g_events = (g_opt.event_log_path == "-") ? stdout : fopen(g_opt.event_log_path.c_str(), "w");
    if (g_events == nullptr) {
        fprintf(stderr, "error: cannot open event log '%s'\n", g_opt.event_log_path.c_str());
        return 3;
    }

    llama_backend_init();
    llama_log_set(forward_log, nullptr);

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0; // CPU-only mandate: never offload
    llama_model * model = llama_model_load_from_file(g_opt.model_path.c_str(), mparams);
    if (model == nullptr) {
        ev_str("model_load_failed", "path", g_opt.model_path);
        return 3;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    g_n_vocab = llama_vocab_n_tokens(vocab);

    // run meta: pid, model, n_vocab, params, gpu-offload support
    char b[640];
    snprintf(b, sizeof(b),
        ",\"model\":,\"n_vocab\":%d,\"n_ctx\":%u,\"n_batch\":%u,\"n_ubatch\":%u,"
        "\"n_seq_max\":%u,\"threads\":%d,\"grid_tokens\":%d,\"k\":%d,\"gen\":%d,"
        "\"requests\":%d,\"seed\":%ld,\"tol\":%.9g,\"probe_empty_rows\":%s",
        "", g_n_vocab, g_opt.n_ctx, g_opt.n_batch, g_opt.n_ubatch, g_opt.n_seq_max,
        g_opt.threads, g_opt.grid_tokens, g_opt.k_out, g_opt.gen_tokens,
        g_opt.requests, g_opt.seed, g_opt.tol,
        g_opt.probe_empty_rows ? "true" : "false");
    ev_open("run_meta");
    ev_raw("run_meta", b);

    if (!arm_create(0, model) || !arm_create(1, model)) {
        return 3;
    }

    const bool ok = run();
    g_cur_arm = 0; llama_free(g_ctx[0]);
    g_cur_arm = 1; llama_free(g_ctx[1]);
    // chains are freed only after the contexts: a chain still attached at
    // llama_free would leave the context's sampler map dangling
    for (int a = 0; a < 2; ++a) {
        for (int k = 0; k < 2; ++k) {
            llama_sampler_free(g_chain[a][k]);
        }
    }
    llama_model_free(model);
    llama_backend_free();

    // dtor stats arrived via the log tap during llama_free
    char sb[768];
    snprintf(sb, sizeof(sb),
        ",\"steps\":%lld,\"compares\":%lld,\"mismatches\":%lld,\"gen_div\":%lld,"
        "\"decode_err\":%lld,\"api_err\":%lld,\"row_probe_bad\":%lld,"
        "\"bucket_cross\":%lld,\"kv_rm_miss\":%lld,"
        "\"on_trace_dec\":%llu,\"on_trace_reuse\":%llu,\"on_builds\":%llu,"
        "\"on_builds_fb\":%llu,\"on_bind_zero\":%llu,\"on_compute_skip\":%llu,"
        "\"on_gen\":%llu,\"pass\":%s",
        (long long) g_verdict.steps, (long long) g_verdict.compares,
        (long long) g_verdict.mismatches, (long long) g_verdict.gen_div,
        (long long) g_verdict.decode_err, (long long) g_verdict.api_err,
        (long long) g_verdict.row_probe_bad, (long long) g_verdict.bucket_cross,
        (long long) g_verdict.kv_rm_miss, (unsigned long long) g_trace_stats[1].dec,
        (unsigned long long) g_trace_stats[1].reuse,
        (unsigned long long) g_trace_stats[1].builds,
        (unsigned long long) g_trace_stats[1].builds_fb,
        (unsigned long long) g_trace_stats[1].bind_zero,
        (unsigned long long) g_trace_stats[1].skip,
        (unsigned long long) g_trace_stats[1].gen,
        g_verdict.pass && ok ? "true" : "false");
    ev_raw("summary", sb);

    if (g_events != stdout) {
        fclose(g_events);
    }
    return (g_verdict.pass && ok) ? 0 : 1;
}
