// N3 verify capture: real MTP target, full-vocabulary logits per verify row, and the same
// rows decoded one at a time by an N1 loop on a private copy of the state.
//
// What this program measures, and what it deliberately does not
// ------------------------------------------------------------
// One window = one private target context over the trunk shards plus the trained NextN/MTP
// head (the three-shard pair produced by work/trunk-mtp-pair-20261031/pair_mtp.py), one
// private draft context over the same model (`LLAMA_CONTEXT_TYPE_MTP`, its own KV), one
// sequence.
//
// Per verify step at position p the target decodes [x, d0, d1] at positions [p, p+1, p+2]
// through `llama_decode_n3_verify_logits()`, i.e. the N3 shape the MTP lane executes.
// Acceptance is decided from those same rows exactly as the lane does (greedy: accept while
// the row's argmax equals the draft), and the next round's drafts come from the trained head
// fed with the accepted rows' wide NextN residuals.
//
// What the capture flag may and may not change. Both legs run the same entry point, the same
// rows, the same acceptance and the same state transitions; acceptance and every captured value
// are read from the context's own logits buffer in both legs. The flag only adds, after the
// graph retired, one full-vocabulary copy per row into the harness's buffers, and the harness
// then requires that copy to equal the context's row bit for bit. It never bypasses the
// publication path, so N3_CAPTURE=0 and N3_CAPTURE=1 runs of the same input must produce
// identical trajectories - and a disagreement between the copy and the context's row aborts the
// run instead of being captured as if it were data.
//
// The N1 reference runs BEFORE each N3 step, on a third context of the same window whose
// memory is loaded from a full sequence state snapshot of the target taken at the same
// position p. It then decodes the same rows one at a time (1x1 ubatches) and is discarded.
// Both sides therefore start from the same GDN/PLE/indexer/attention state and consume the
// same tokens; the only differences are the batch shape (3x1 vs 1x1) and the number of
// decode calls. Rows are labelled (window, seq, step, row, position, token) on both sides, so
// a comparison needs no guessing and no re-decode of the target.
//
// The reference is a *second decode* on a copied state, never a different-time N1 decode of
// the target: the target is not touched between the snapshot and the N3 step.
//
// Refusals (exit 2, nothing written): not exactly five ROCm devices; a model whose NextN
// block count is not 1 or whose token embeddings / output head are absent from the file (the
// two-shard trunk, which would make the head borrow the target's tables and is not the model
// this lane captures); an output directory that already exists; a relay/FA environment
// variable this capture cannot vouch for, including GGML_TP5_RSZ_RT and GGML_TP5_FA_MASK_KVLEN;
// and a capture run without N3_SNAPSHOT_REPLAY=1 (see the state contract below).
//
// STATE CONTRACT OF A ROUND (why N3_SNAPSHOT_REPLAY is mandatory for a capture).
//   The N3 step leaves the target's GDN cell at the state AFTER its last row, while a partially
//   accepted round must end at p + accepted + 1 with the GDN state after row `accepted`. A
//   context with n_rs_seq == 0 cannot remove a partial range of the recurrent cache
//   (llama_memory_recurrent::seq_rm refuses and mutates nothing), so the only exact way to reach
//   that state with public APIs is to restore the pre-step snapshot and re-decode the accepted
//   rows one at a time. That is what this flag does. It is a DIAGNOSTIC path: it costs a full
//   host state snapshot plus one restore and up to two single-row decodes per rejected round,
//   and nothing measured on it is a throughput result. It is also not the production runtime's
//   protocol (fork/tp5-compose's MTP loop).
//
//   The snapshot and the replay run in BOTH legs (capture and no-copy): they are what keeps the
//   target's state correct on a rejected round, and the no-copy leg must be able to run the same
//   rounds. Only the numeric output and the comparison are gated on N3_CAPTURE.
//
//   The replay is a one-real-row call, the verify row came from a three-row call: the two are
//   compared NUMERICALLY (KL, reported as replay_mean_kl / replay_max_kl), never byte for byte,
//   because a shape difference may legitimately reorder a reduction (ACCEPTANCE §4). Byte
//   equality is required only between the harness's captured copy and the context's own row of
//   the same graph, which is the same bytes by construction.
//
// Scope: this program is the *reference* implementation of the entry point and of the
// same-state N1 pair. It is not the production runtime (fork/tp5-compose's MTP loop) and no
// result of it may be quoted as a statement about that runtime or about the row-tail binary.
//
// This program is compiled into the test directory only. It is not a lane binary and must
// never run inside a timing leg. Its author executed nothing here: no build, no run, no GPU.

#include "llama.h"
#include "llama-ext.h"

#include <ggml-backend.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {

constexpr int      kN3Rows     = LLAMA_N3_VERIFY_MAX_ROWS; // current + two drafts
constexpr int      kMaxDrafts  = kN3Rows - 1;
constexpr uint64_t kStateBound = 512ull << 20;             // the lane's own snapshot bound

[[noreturn]] void refuse(const std::string & why) {
    std::fprintf(stderr, "N3_REFUSED %s\n", why.c_str());
    std::fflush(stderr);
    std::exit(2);
}

void require(bool ok, const std::string & why) {
    if (!ok) {
        std::fprintf(stderr, "N3_FAIL %s\n", why.c_str());
        std::fflush(stderr);
        std::exit(1);
    }
}

bool env_is(const char * name, const char * value) {
    const char * v = std::getenv(name);
    return v != nullptr && std::strcmp(v, value) == 0;
}

std::string env_str(const char * name, const char * fallback) {
    const char * v = std::getenv(name);
    return v && v[0] ? std::string(v) : std::string(fallback);
}

std::uint32_t env_u32(const char * name, std::uint32_t fallback) {
    const char * v = std::getenv(name);
    if (!v || !v[0]) {
        return fallback;
    }
    char * end = nullptr;
    const unsigned long parsed = std::strtoul(v, &end, 10);
    if (end == v || *end != '\0' || parsed > std::numeric_limits<std::uint32_t>::max()) {
        refuse(std::string(name) + " is not a decimal u32");
    }
    return std::uint32_t(parsed);
}

// ---------------------------------------------------------------------------------------
// capture output: rows.jsonl + logits.f32le in one exclusively created directory
// ---------------------------------------------------------------------------------------
struct capture {
    std::string dir;
    std::FILE * rows    = nullptr;
    std::FILE * binary  = nullptr;
    uint64_t    n_rows  = 0;
    uint64_t    n_bytes = 0;
    uint64_t    n_vocab = 0;

    void open(const std::string & path, uint64_t vocab) {
#if defined(__unix__) || defined(__APPLE__)
        if (::mkdir(path.c_str(), 0700) != 0) {
            refuse(std::string("output directory: ") + std::strerror(errno));
        }
#endif
        dir     = path;
        n_vocab = vocab;
        rows    = std::fopen((dir + "/rows.jsonl").c_str(), "wbx");
        binary  = std::fopen((dir + "/logits.f32le").c_str(), "wbx");
        require(rows != nullptr && binary != nullptr, "cannot create the capture files");
    }

    // The row order is request-major like the tp5-compose container: both sides of one step are
    // written together (n1 first, then n3), so the side-specific selection never needs a file
    // region of its own and the gate can check one tiling over the whole file.
    //
    // side: "prefill" | "n3" (this step's verify rows) | "n1" (reference rows of the same step)
    void row(const char * side, uint32_t window, int32_t seq, uint64_t step, int row,
             int64_t position, int32_t token, const float * logits) {
        std::fprintf(rows,
                "{\"side\":\"%s\",\"window\":%u,\"seq\":%d,\"step\":%llu,\"row\":%d,"
                "\"input_position\":%lld,\"input_token\":%d,\"vocabulary\":%llu,"
                "\"offset_bytes\":%llu,\"count\":%llu,\"dtype\":\"f32le\"}\n",
                side, window, seq, (unsigned long long) step, row, (long long) position, token,
                (unsigned long long) n_vocab, (unsigned long long) n_bytes,
                (unsigned long long) n_vocab);
        const auto written = std::fwrite(logits, sizeof(float), size_t(n_vocab), binary);
        require(written == size_t(n_vocab), "short write of a logits row");
        n_bytes += n_vocab * sizeof(float);
        ++n_rows;
    }

    void close() {
        if (rows) {
            require(std::fflush(rows) == 0, "rows.jsonl flush");
            require(std::fclose(rows) == 0, "rows.jsonl close");
            rows = nullptr;
        }
        if (binary) {
            require(std::fflush(binary) == 0, "logits flush");
#if defined(__unix__) || defined(__APPLE__)
            require(::fsync(::fileno(binary)) == 0, "logits fsync");
#endif
            require(std::fclose(binary) == 0, "logits close");
            binary = nullptr;
        }
    }
};

struct model_ptr {
    llama_model * p = nullptr;
    ~model_ptr() { if (p) llama_model_free(p); }
    llama_model * get() const { return p; }
};

struct context_ptr {
    llama_context * p = nullptr;
    ~context_ptr() { if (p) llama_free(p); }
    llama_context * get() const { return p; }
};

struct window {
    uint32_t            id  = 0;
    int32_t             seq = 0;
    context_ptr         target;   // N3 verify target
    context_ptr         draft;    // trained NextN head
    context_ptr         n1;       // N1 reference, restored from a snapshot per step
    std::vector<float>  head_h;   // wide residual the head consumes at the next position
    // The state snapshot is a BYTE blob: llama_state_seq_get_data_ext / set_data_ext take a
    // uint8_t* and a byte count, and their size function reports bytes. Holding it as floats
    // would both fail to compile and, if it were cast, describe a buffer four times too long.
    std::vector<uint8_t> state;   // target state snapshot at the next verify position
    uint64_t replays = 0;         // rounds that were restored and replayed (diagnostic counter)
    // Replay-vs-verify comparison, DIAGNOSTIC ONLY: the replayed row is produced at an
    // N1-shaped call, the verify row at the N3 shape, so the two are compared numerically
    // (KL), never byte for byte. These are reported, never gated.
    std::vector<double> replay_kl;
    uint64_t replay_rows      = 0;   // rows compared
    uint64_t replay_bits_diff = 0;   // rows that are not byte-identical (informational)
    uint64_t verify_rounds   = 0;
    uint64_t drafts_proposed = 0;
    uint64_t accepted_drafts = 0;
    uint64_t full_accept     = 0;
    uint64_t n1_rows         = 0;   // N1 reference rows decoded (prefill + verify steps, capture legs only)
    uint64_t prefill_rows    = 0;   // prompt rows the reference decoded before the first step
    uint64_t committed       = 0;
};

std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    std::vector<llama_token> out(64);
    auto n = llama_tokenize(vocab, text.data(), int32_t(text.size()), out.data(), int32_t(out.size()), true, false);
    if (n < 0) {
        out.resize(size_t(-n));
        n = llama_tokenize(vocab, text.data(), int32_t(text.size()), out.data(), int32_t(out.size()), true, false);
    }
    require(n > 0, "prompt tokenization failed");
    out.resize(size_t(n));
    return out;
}

// one single-row decode; returns that row's logits (or nullptr when not requested)
const float * decode_one(llama_context * ctx, int32_t seq, llama_pos pos, llama_token token,
                         bool want_logits) {
    llama_batch b = llama_batch_init(1, 0, 1);
    b.n_tokens     = 1;
    b.token[0]     = token;
    b.pos[0]       = pos;
    b.n_seq_id[0]  = 1;
    b.seq_id[0][0] = seq;
    b.logits[0]    = want_logits ? 1 : 0;
    const int rc = llama_decode(ctx, b);
    llama_batch_free(b);
    require(rc == 0, "single-row decode failed");
    return want_logits ? llama_get_logits_ith(ctx, 0) : nullptr;
}

int32_t greedy(const float * logits, uint32_t n_vocab) {
    int32_t best   = -1;
    float   best_v = -std::numeric_limits<float>::infinity();
    for (uint32_t i = 0; i < n_vocab; ++i) {
        const float v = logits[i];
        require(std::isfinite(v), "nonfinite logits");
        if (v > best_v) {
            best_v = v;
            best   = int32_t(i);
        }
    }
    require(best >= 0, "empty logits row");
    return best;
}

// One head step: input (token, wide residual of the previous row) at position q. Returns the
// head's argmax; when out_h is given, also the head's wide residual for the next chain step.
int32_t head_step(llama_context * dctx, llama_token token, llama_pos q, const float * h_wide,
                  int32_t wide, uint32_t n_vocab, std::vector<float> * out_h) {
    llama_batch b = llama_batch_init(1, wide, 1);
    b.n_tokens     = 1;
    b.token[0]     = token;
    b.pos[0]       = q;
    b.n_seq_id[0]  = 1;
    b.seq_id[0][0] = 0;
    b.logits[0]    = 1;
    std::memcpy(b.embd, h_wide, size_t(wide) * sizeof(float));
    const int rc = llama_decode(dctx, b);
    llama_batch_free(b);
    require(rc == 0, "head decode failed");
    const float * dl = llama_get_logits_ith(dctx, 0);
    require(dl != nullptr, "head logits missing");
    const int32_t id = greedy(dl, n_vocab);
    if (out_h) {
        const float * dh = llama_get_embeddings_nextn_ith(dctx, 0);
        require(dh != nullptr, "head wide residual missing");
        out_h->assign(dh, dh + wide);
    }
    return id;
}

// Decode exactly one real row (position p, token t) through the same N3 entry point, padded to
// the declared three rows, and copy that row's full-vocabulary logits into `out`.
// Returns the row's logits pointer inside the context (valid until the next decode).
const float * verify_one(llama_context * ctx, int32_t seq, llama_pos p, llama_token t,
                         int32_t n_vocab, std::vector<float> * copy) {
    std::array<llama_token, kN3Rows> toks{t, t, t};
    std::array<int32_t, kN3Rows>     needs{1, 0, 0};
    std::array<std::vector<float>, kN3Rows> rows;
    std::array<float *, kN3Rows> outs{};
    if (copy) {
        for (int k = 0; k < kN3Rows; ++k) {
            rows[size_t(k)].assign(size_t(n_vocab), 0.0f);
            outs[size_t(k)] = rows[size_t(k)].data();
        }
    }
    const int rc = llama_decode_n3_verify_logits(ctx, seq, p, toks.data(),
            needs.data(), copy ? outs.data() : nullptr);
    require(rc == 0, "single-row verify replay failed");
    if (copy) {
        copy->assign(rows[0].begin(), rows[0].end());
    }
    return llama_get_logits_ith(ctx, 0);
}

// KL(ref || cand) over the full vocabulary, double precision with a max shift. Same definition
// as n3_logits_gate.py, so a number printed here and a number computed there mean the same thing.
double replay_mean(const std::vector<double> & v) {
    if (v.empty()) {
        return 0.0;
    }
    double sum = 0.0;
    for (const double x : v) {
        sum += x;
    }
    return sum / double(v.size());
}

double soft_gate_row(const float * ref, const float * cand, uint32_t n_vocab) {
    double rmax = -std::numeric_limits<double>::infinity();
    double cmax = -std::numeric_limits<double>::infinity();
    for (uint32_t i = 0; i < n_vocab; ++i) {
        rmax = std::max(rmax, double(ref[i]));
        cmax = std::max(cmax, double(cand[i]));
    }
    double zr = 0.0, zc = 0.0, cross = 0.0;
    for (uint32_t i = 0; i < n_vocab; ++i) {
        const double a = double(ref[i]);
        const double b = double(cand[i]);
        if (!(std::isfinite(a) && std::isfinite(b))) {
            return std::numeric_limits<double>::infinity();
        }
        const double wa = std::exp(a - rmax);
        const double wb = std::exp(b - cmax);
        zr += wa;
        zc += wb;
        cross += wa * (a - b);
    }
    return cross / zr + (cmax + std::log(zc)) - (rmax + std::log(zr));
}

void head_trim(llama_context * dctx, llama_pos from) {
    require(llama_memory_seq_rm(llama_get_memory(dctx), 0, from, -1), "head KV tail trim failed");
}

// Snapshot the target's full sequence state (attention KV + GDN/PLE/indexer) at its frontier
// into `bytes`. `bytes` is a byte buffer sized by the size query, and the data call is given
// exactly that size, so nothing is written past it.
void snapshot(llama_context * ctx, int32_t seq, std::vector<uint8_t> & bytes) {
    const size_t sz = llama_state_seq_get_size_ext(ctx, seq, LLAMA_STATE_SEQ_FLAGS_NONE);
    // sz is a byte count; the bound is compared in 64 bits so no cast can narrow it
    require(sz > 0 && uint64_t(sz) <= kStateBound, "full target state snapshot unavailable");
    bytes.assign(sz, 0);   // exact size, fully initialised: a short write stays detectable
    const size_t got = llama_state_seq_get_data_ext(ctx, bytes.data(), bytes.size(), seq,
            LLAMA_STATE_SEQ_FLAGS_NONE);
    // A short write would leave the tail at its zero fill and restore as garbage; refuse it here
    // rather than at the reference context.
    require(got == bytes.size(), "full target state snapshot failed");
}

} // namespace

int main(int argc, char ** argv) {
    const std::string model_path = argc > 1 ? argv[1] : "";
    const std::string out_dir    = argc > 2 ? argv[2] : "";
    if (model_path.empty() || out_dir.empty()) {
        std::fprintf(stderr, "usage: %s <trunk+nextn model shard 1 of 3> <output dir>\n", argv[0]);
        return 64;
    }

    if (std::getenv("GGML_TP5_RSZ_RT")) {
        refuse("GGML_TP5_RSZ_RT is set (promotion path, HANDOFF section 10)");
    }
    if (std::getenv("GGML_TP5_FA_MASK_KVLEN")) {
        refuse("GGML_TP5_FA_MASK_KVLEN is set (ncols2>1 OOB risk until the FA kernel fix)");
    }

    const std::string prompt0 = env_str("N3_PROMPT0",
            "Please explain how to lose weight safely in eight practical steps.");
    const std::string prompt1 = env_str("N3_PROMPT1",
            "Explain how a CPU and memory work together in a computer.");
    const uint32_t predict    = env_u32("N3_PREDICT", 8);
    const bool     capture_on = !env_is("N3_CAPTURE", "0");
    const uint32_t threads    = env_u32("N3_THREADS", 22);
    const uint32_t ctx_size   = env_u32("N3_CTX", 262144);
    // Plan B: the diagnostic restore-and-replay path for partial acceptance. It is required
    // whenever a round is only partially accepted, and a partial acceptance without it is a
    // refusal (there is no recurrent rollback plane to fall back on).
    const bool snapshot_replay = env_is("N3_SNAPSHOT_REPLAY", "1");
    require(predict >= 2 && predict <= 512, "N3_PREDICT must be in [2, 512]");
    // The snapshot/replay path is independent of whether rows are written: it is what keeps the
    // target's state correct on a rejected round, so BOTH legs run it. N3_SNAPSHOT_REPLAY=0 is
    // refused later, before the first step (there is no other way to commit a partial round).

    std::vector<ggml_backend_dev_t> rocm;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto * dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }
        if (std::string(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev))) == "ROCm") {
            rocm.push_back(dev);
        }
    }
    if (rocm.size() != 5) {
        refuse("the TP5 relay profile requires exactly five ROCm devices");
    }
    std::array<ggml_backend_dev_t, 6> dev_list{};
    std::copy(rocm.begin(), rocm.end(), dev_list.begin());
    dev_list[5] = nullptr;

    auto mparams = llama_model_default_params();
    mparams.devices       = dev_list.data();
    mparams.n_gpu_layers  = 99;
    mparams.split_mode    = LLAMA_SPLIT_MODE_TENSOR;
    mparams.offload_input = true;

    model_ptr model;
    model.p = llama_model_load_from_file(model_path.c_str(), mparams);
    require(model.get() != nullptr, "model load failed (the trunk and the NextN head must be one 3-shard model)");
    if (llama_model_n_layer_nextn(model.get()) != 1) {
        refuse("the model does not carry exactly one NextN/MTP block; there is nothing to verify");
    }
    // The pair produced by pair_mtp.py is 49 blocks over three shards with the head's own tables
    // merged; a two-shard trunk has n_layer_nextn == 0 and never reaches this point, and a
    // 49-block file whose token_embd/output came from somewhere else is not the captured model.
    if (llama_model_n_layer(model.get()) != 48) {
        refuse("the model is not the 48-layer Swift trunk with one attached NextN block");
    }
    const llama_vocab * vocab = llama_model_get_vocab(model.get());
    const uint32_t n_vocab = uint32_t(llama_vocab_n_tokens(vocab));
    require(n_vocab > 0, "invalid vocabulary");
    const int32_t wide = llama_model_n_embd_out(model.get());
    require(wide > 0, "the model does not expose a wide output width");

    std::vector<std::vector<llama_token>> prompt_tokens;
    for (const auto & text : { prompt0, prompt1 }) {
        prompt_tokens.push_back(tokenize(vocab, text));
        require(prompt_tokens.back().size() + predict + kN3Rows < ctx_size,
                "prompt plus generation exceeds the context");
    }

    auto base_params = [&] {
        auto p = llama_context_default_params();
        p.n_ctx           = ctx_size;
        p.n_seq_max       = 1;
        p.n_batch         = kN3Rows;
        p.n_ubatch        = kN3Rows;
        p.n_threads       = p.n_threads_batch = threads;
        p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        p.type_k = p.type_v = GGML_TYPE_Q4_0; // Swift relay + FA contract
        p.no_perf         = true;
        return p;
    };

    std::array<window, 2> windows;
    for (uint32_t w = 0; w < 2; ++w) {
        auto & win = windows[w];
        win.id  = w;
        win.seq = 0;

        win.target.p = llama_init_from_model(model.get(), base_params());
        require(win.target.get() != nullptr, "target context creation failed");
        llama_set_embeddings_nextn(win.target.get(), true, /*masked=*/false);

        auto dp = base_params();
        dp.ctx_type  = LLAMA_CONTEXT_TYPE_MTP;
        dp.n_batch   = 1;
        dp.n_ubatch  = 1;
        dp.ctx_other = win.target.get();
        win.draft.p  = llama_init_from_model(model.get(), dp);
        require(win.draft.get() != nullptr, "draft (MTP) context creation failed");
        llama_set_embeddings_nextn(win.draft.get(), true, /*masked=*/false);

        auto np = base_params();
        np.n_batch  = 1;
        np.n_ubatch = 1;
        win.n1.p    = llama_init_from_model(model.get(), np);
        require(win.n1.get() != nullptr, "N1 reference context creation failed");
        llama_set_embeddings_nextn(win.n1.get(), true, /*masked=*/false);

        win.head_h.assign(size_t(wide), 0.0f);
    }

    // Plan B preflight. The reference loop cannot commit a partially accepted round without a
    // recurrent rollback plane, and this lane has none (n_rs_seq == 0). The only supported route
    // is the diagnostic restore-and-replay, so it is required up front rather than discovered
    // mid-run: without it the run would abort on its first rejected round, which is the common
    // case and not an error condition.
    // The snapshot and the replay are what keep the TARGET's state correct on a partially
    // accepted round; they are not a capture feature. Both legs therefore need them, otherwise
    // the no-copy leg cannot finish a round it must also be able to run. Only the numeric output
    // and the comparison depend on `capture_on`.
    {
        require(snapshot_replay,
                "N3_SNAPSHOT_REPLAY=1 is required: partial acceptance cannot be committed "
                "without a recurrent rollback plane (n_rs_seq=0)");
        // The replay path replaces the missing rollback plane. It is only legal if the context
        // really has none: if a caller enabled n_rs_seq > 0, the native partial rollback is the
        // mechanism under test and this harness must not silently shadow it with a restore.
        const uint32_t rs = llama_n_rs_seq(windows[0].target.get());
        require(rs == 0,
                "N3_SNAPSHOT_REPLAY assumes no recurrent rollback plane, but the target reports "
                "n_rs_seq != 0; run that configuration without N3_SNAPSHOT_REPLAY instead");
        const size_t probe = llama_state_seq_get_size_ext(windows[0].target.get(), 0,
                LLAMA_STATE_SEQ_FLAGS_NONE);
        require(probe > 0 && uint64_t(probe) <= kStateBound,
                "the target cannot produce a full sequence state snapshot");
        // the snapshot API must fill exactly the buffer its size query named; nothing is
        // restored here (restoring the empty prefill state would be pointless, and a restore is
        // only ever done from a snapshot taken at the same frontier)
        std::vector<uint8_t> roundtrip(probe, 0);
        require(llama_state_seq_get_data_ext(windows[0].target.get(), roundtrip.data(),
                    roundtrip.size(), 0, LLAMA_STATE_SEQ_FLAGS_NONE) == roundtrip.size(),
                "the target cannot produce a full sequence state snapshot");
        std::fprintf(stderr,
                "N3_PLAN vocab=%u predict=%u rows_per_round=%d replay=%d n_rs_seq=%u "
                "replay_cost=full_state_restore+accepted_rows snapshot_bytes=%zu\n",
                n_vocab, predict, kN3Rows, int(snapshot_replay), rs, probe);
    }

    capture cap;
    if (capture_on) {
        cap.open(out_dir, n_vocab);
    }

    // ---- prefill: prompt rows on the target and on the reference, the head's catch-up on the
    //      draft context. The head's input at position i is (prompt[i], target residual at i),
    //      which is the shift-by-one pairing the lane uses.
    // The target is prefilled ONE ROW PER DECODE. With a multi-row batch the NextN residual is
    // stored densely by token index (unmasked export) and every row is a different graph output,
    // so the head's catch-up would have to address row pt.size() - 1 of the last batch while the
    // earlier rows' residuals had already been overwritten. One row per decode makes each
    // residual exactly `llama_get_embeddings_nextn_ith(ctx, 0)` and keeps the loop honest.
    // (The lane prefills in one batch; that only changes how many decode calls are made, not
    // which tokens the sequence has seen before the first verify step.)
    for (uint32_t w = 0; w < 2; ++w) {
        auto &       win = windows[w];
        const auto & pt  = prompt_tokens[w];
        std::vector<float> prev(size_t(wide), 0.0f);   // head input for position 0
        for (size_t i = 0; i < pt.size(); ++i) {
            const bool last = (i + 1 == pt.size());
            const float * tgt = decode_one(win.target.get(), win.seq, llama_pos(i), pt[i], last);
            require(tgt != nullptr, "prompt logits missing");
            // this row's residual is the head's input for the NEXT row: fetch it before the next
            // decode overwrites the context's residual buffer
            const float * h = llama_get_embeddings_nextn_ith(win.target.get(), 0);
            require(h != nullptr, "target did not export the wide NextN residual");
            std::vector<float> next;
            (void) head_step(win.draft.get(), pt[i], llama_pos(i), prev.data(), wide, n_vocab, &next);
            prev = next;
            if (last) {
                const float * n1 = decode_one(win.n1.get(), win.seq, llama_pos(i), pt[i], true);
                require(n1 != nullptr, "reference prompt logits missing");
                if (capture_on) {
                    cap.row("prefill", w, win.seq, 0, 0, llama_pos(i), pt[i], tgt);
                    cap.row("prefill", w, win.seq, 0, 1, llama_pos(i), pt[i], n1);
                }
            }
        }
        win.head_h = prev;
        // two counters, because the same field used to mean "prefill" at one point and "every
        // reference row" at another; the prefill is counted separately and never inferred
        win.prefill_rows = capture_on ? pt.size() : 0;
        win.n1_rows      = win.prefill_rows;
    }

    // ---- the decode loops ----------------------------------------------------------------
    for (uint32_t w = 0; w < 2; ++w) {
        auto & win = windows[w];
        llama_token last_emitted = prompt_tokens[w].back();
        llama_pos   p            = llama_pos(prompt_tokens[w].size());
        std::vector<llama_token> drafts;
        drafts.reserve(kMaxDrafts);

        while (win.committed + 1 < predict) {
            // ---- drafts for this round, from the head state at p --------------------------
            drafts.clear();
            {
                std::vector<float> h   = win.head_h;
                llama_token        tok = last_emitted;
                // committed tokens are the emitted ones; the budget is `predict` of them, so the
                // rows this round may still commit is predict - committed, of which row 0
                // (the current token) commits one and each accepted draft one more
                const int want = int(std::min<uint64_t>(kMaxDrafts, predict - win.committed - 1));
                for (int k = 0; k < want; ++k) {
                    std::vector<float> next;
                    const int32_t d = head_step(win.draft.get(), tok, p + k, h.data(), wide, n_vocab, &next);
                    drafts.push_back(llama_token(d));
                    tok = llama_token(d);
                    h   = next;
                }
                head_trim(win.draft.get(), p); // proposals past p are not valid yet
            }
            const int n_rows = int(1 + drafts.size());
            require(n_rows >= 2 && n_rows <= kN3Rows, "verify row count out of range");

            // llama_decode_n3_verify_logits() is declared for exactly LLAMA_N3_VERIFY_MAX_ROWS
            // rows, so a round with fewer drafts (end of budget) pads the tail by repeating the
            // previous token. The padding rows are speculative rows of the same sequence: they
            // are never read (only k < n_rows is evaluated and captured) and the commit trim
            // below drops every position from p + accepted + 1 onward.
            std::array<llama_token, kN3Rows> toks{};
            toks[0] = last_emitted;
            for (int k = 0; k < kN3Rows - 1; ++k) {
                toks[size_t(k + 1)] = k < n_rows - 1 ? drafts[size_t(k)] : toks[size_t(k)];
            }

            // ---- N1 reference, from a snapshot of the target's state at p ----------------
            // Taken before the N3 step, so the reference and the verify start from exactly the
            // same state. The reference never writes back into the target.
            // The snapshot itself is taken in BOTH legs: the replay below needs it to restore
            // the target, independently of whether anything is written to disk.
            snapshot(win.target.get(), win.seq, win.state);
            if (capture_on) {
                // same byte buffer, same byte count: the restore is given exactly what was read
                const size_t restored = llama_state_seq_set_data_ext(win.n1.get(), win.state.data(),
                        win.state.size(), win.seq, LLAMA_STATE_SEQ_FLAGS_NONE);
                require(restored == win.state.size(), "N1 reference state restore failed");
                for (int k = 0; k < n_rows; ++k) {
                    const float * row = decode_one(win.n1.get(), win.seq, p + k, toks[size_t(k)], true);
                    require(row != nullptr, "N1 reference row logits missing");
                    cap.row("n1", w, win.seq, win.verify_rounds, k, p + k, toks[size_t(k)], row);
                    ++win.n1_rows;
                }
                require(llama_memory_seq_rm(llama_get_memory(win.n1.get()), win.seq, -1, -1),
                        "N1 reference state removal failed");
            }

            // ---- the N3 step --------------------------------------------------------------
            // Both legs call the same entry point with the same three rows. The ONLY difference
            // is what happens after the graph retired: the capture leg copies every row into
            // its own buffers, the no-copy leg copies nothing. `needs_logits` is therefore an
            // explicit per-row flag array in both legs - all ones for the capture, all zeros for
            // the no-copy leg - and never NULL, so the no-copy leg can never be mistaken for a
            // call that wanted rows and got no destination (which the API refuses).
            std::array<std::vector<float>, kN3Rows> all_rows;
            std::array<float *, kN3Rows> all_outs{};
            // Only the REAL rows of this round are requested and buffered. A padded row is not a
            // draft, its logits are never read and never compared, and asking for it would put
            // an unused buffer on the copy path. The array is still non-NULL in both legs (an
            // all-zero array means "copy nothing", which is what the no-copy leg wants).
            std::array<int32_t, kN3Rows> needs{};
            for (int k = 0; k < kN3Rows; ++k) {
                needs[size_t(k)] = (capture_on && k < n_rows) ? 1 : 0;
                if (needs[size_t(k)]) {
                    all_rows[size_t(k)].assign(size_t(n_vocab), 0.0f);
                    all_outs[size_t(k)] = all_rows[size_t(k)].data();
                }
            }
            const int rc = llama_decode_n3_verify_logits(win.target.get(), win.seq, p, toks.data(),
                    needs.data(), capture_on ? all_outs.data() : nullptr);
            require(rc == 0, "N3 verify decode failed");

            // The rows are read the same way in both legs: from the context's logits buffer,
            // which the entry point synchronized and reordered before it returned. The capture
            // leg additionally holds its own copy, and G3 requires the two to agree bit for bit;
            // that equality is what proves the copy is a bypass of the *publication*, not of
            // the computation.
            std::array<int32_t, kN3Rows> argmax{};
            for (int k = 0; k < kN3Rows; ++k) {
                // a padded row carries no verdict: it is not a draft and the round ends before it
                if (k >= n_rows) {
                    argmax[size_t(k)] = -1;
                    continue;
                }
                const float * row = llama_get_logits_ith(win.target.get(), k);
                require(row != nullptr, "verify row logits missing");
                if (capture_on) {
                    // copy-vs-context: the SAME output of the SAME graph, so this is byte equality
                    // by construction and a mismatch is a publication bug, not a numeric one.
                    const auto & copy = all_rows[size_t(k)];
                    require(copy.size() == size_t(n_vocab), "capture row width changed");
                    require(std::memcmp(copy.data(), row, size_t(n_vocab) * sizeof(float)) == 0,
                            "captured row differs from the context's row");
                    cap.row("n3", w, win.seq, win.verify_rounds, k, p + k, toks[size_t(k)], row);
                }
                argmax[size_t(k)] = greedy(row, n_vocab);
            }
            int accepted = 0;
            while (accepted + 1 < n_rows && argmax[size_t(accepted)] == toks[size_t(accepted + 1)]) {
                ++accepted;
            }
            ++win.verify_rounds;
            win.drafts_proposed += uint64_t(n_rows - 1);
            win.accepted_drafts += uint64_t(accepted);
            if (accepted + 1 == n_rows) {
                ++win.full_accept;
            }

            // Only the LAST accepted row's residual is the head's next input: the head chain is
            // (x_p, h_p) -> d0 -> d1, so advancing it means consuming that one pair, exactly as
            // the lane does when it feeds `real_h` of the last target row into the draft context.
            // It is read BEFORE any restore below, because the restore rewinds the target to p
            // and overwrites the residual buffer.
            const float * hr = llama_get_embeddings_nextn_ith(win.target.get(), accepted);
            require(hr != nullptr, "target wide residual missing for the last verified row");
            std::vector<float> last_h(hr, hr + wide);

            // ---- restore-and-replay (plan B, DIAGNOSTIC ONLY) ---------------------------
            // The step left the target's GDN cell at the state after the LAST row. The sequence
            // must end at p + accepted + 1 with the GDN state after row `accepted`, so whenever
            // the step wrote more rows than that, the pre-step state is restored and the
            // accepted rows are decoded again, one real row per call. Two cases reach this:
            //   * a partial acceptance (accepted + 1 < n_rows), and
            //   * a full acceptance of a PADDED round (n_rows < 3), where the padding rows are
            //     positions the sequence must not keep.
            // Both are the same partial recurrent removal, which n_rs_seq == 0 cannot do.
            // This is exact and uses only public APIs; it is not the production protocol and its
            // cost is not a throughput measurement.
            if (accepted + 1 < kN3Rows) {
                if (!snapshot_replay) {
                    refuse("partial acceptance requires N3_SNAPSHOT_REPLAY=1 "
                           "(no recurrent rollback plane at n_rs_seq=0)");
                }
                require(!win.state.empty(),
                        "partial acceptance needs the pre-step state snapshot");
                const size_t restored = llama_state_seq_set_data_ext(win.target.get(),
                        win.state.data(), win.state.size(), win.seq, LLAMA_STATE_SEQ_FLAGS_NONE);
                require(restored == win.state.size(),
                        "snapshot restore incomplete (state_drop may have run); the target must "
                        "not be used again");
                for (int k = 0; k <= accepted; ++k) {
                    // The replayed row is read from the context's getter in BOTH legs, exactly
                    // like the verify rows: the no-copy leg has no buffer of its own, and the
                    // state it hands to the head must be checked whether or not rows are written.
                    const float * row = verify_one(win.target.get(), win.seq, p + llama_pos(k),
                            toks[size_t(k)], int32_t(n_vocab), nullptr);
                    require(row != nullptr, "replayed row logits missing");
                    // Finiteness is a STATE invariant, not a measurement: a NaN/Inf row would be
                    // consumed by the head and by the next round's acceptance, so it stops the
                    // run in both legs. This is the same check greedy() applies to the verify
                    // rows, restated here because the replay path does not go through greedy().
                    for (uint32_t t = 0; t < n_vocab; ++t) {
                        require(std::isfinite(row[t]),
                                "replayed row holds a non-finite logit; refusing to continue");
                    }
                    // A replayed row is the same three-row entry point called at a DIFFERENT
                    // shape (one real row plus padding) than the verify step (three real rows),
                    // so it is an N1-shaped row against an N3-shaped row: different kernels,
                    // different reduction order. Byte equality is NOT required and must not be
                    // required - that is exactly the numeric-order difference ACCEPTANCE §4
                    // admits. It is measured and reported instead; the soft gate that decides
                    // anything lives in n3_logits_gate.py (G4), not here.
                    if (capture_on) {
                        const auto & orig = all_rows[size_t(k)];
                        require(orig.size() == size_t(n_vocab), "replayed row width changed");
                        const double d = soft_gate_row(orig.data(), row, n_vocab);
                        // A non-finite KL means a row held NaN/Inf. Such a row must not be fed to
                        // anything downstream (the head, the next round's acceptance), so the run
                        // stops here. The KL THRESHOLD is not applied in the harness: the class
                        // gate is n3_logits_gate.py (G4) and it decides, offline, on the captures.
                        require(std::isfinite(d),
                                "replayed row is not finite (nonfinite logits); refusing to continue");
                        win.replay_kl.push_back(d);
                        win.replay_rows += 1;
                        win.replay_bits_diff += (std::memcmp(orig.data(), row,
                                size_t(n_vocab) * sizeof(float)) == 0) ? 0 : 1;
                    }
                }
                ++win.replays;
                // the replay re-decoded row `accepted`; re-read its residual so the head is fed
                // the row that is actually in the context now
                const float * hr2 = llama_get_embeddings_nextn_ith(win.target.get(), accepted);
                require(hr2 != nullptr, "target wide residual missing after the replay");
                last_h.assign(hr2, hr2 + wide);
            }

            // ---- commit -------------------------------------------------------------------
            for (int k = 0; k < accepted; ++k) {
                last_emitted = toks[size_t(k + 1)];
                ++win.committed;
            }
            last_emitted = llama_token(argmax[size_t(accepted)]);
            ++win.committed;

            // keep the valid prefix p..p+accepted, drop the speculative rows
            // The two positions cannot cross: the accepted prefix is bounded by the budget and
            // the sequence can never exceed context minus the verify window (prefill check).
            const llama_pos keep_to = p + llama_pos(accepted) + 1;
            // The step wrote kN3Rows positions (padding included) and exactly 1 + accepted of
            // them are valid; everything from keep_to on is speculative and must not survive.
            require(keep_to + kN3Rows <= llama_pos(ctx_size), "verify window would leave the context");
            // The sequence must now end exactly at keep_to. After the replay (or after a full
            // acceptance of an unpadded round) it already does, so this is a no-op - but it is
            // ASSERTED, not assumed: a suffix left behind here would be a partial recurrent
            // removal waiting to fail, and the harness must not carry on as if it had worked.
            require(keep_to >= p, "commit position moved backwards");
            const llama_pos frontier = llama_memory_seq_pos_max(llama_get_memory(win.target.get()), win.seq);
            require(frontier + 1 == keep_to,
                    "target frontier is not at the committed position (a speculative suffix remains)");
            if (keep_to < p + llama_pos(kN3Rows)) {
                // Unreachable after the replay; kept so a future change cannot silently drop the
                // bound. It is a partial recurrent removal and would fail at n_rs_seq == 0.
                require(llama_memory_seq_rm(llama_get_memory(win.target.get()), win.seq, keep_to, -1),
                        "target speculative suffix trim failed");
            }

            // ---- advance the head over the accepted rows, at their own positions ------------
            std::vector<float> head_out;
            (void) head_step(win.draft.get(), toks[size_t(accepted)], p + llama_pos(accepted),
                    last_h.data(), wide, n_vocab, &head_out);
            // The head's row for the last accepted row is its residual at the next position; the
            // head has not seen the next token yet, so the next round's first draft step pairs
            // that residual with last_emitted, which is that row's argmax.
            win.head_h = head_out;
            p = keep_to;
            head_trim(win.draft.get(), p);
        }
        std::fprintf(stderr,
                "N3_WINDOW window=%u verify_rounds=%llu drafts=%llu accepted=%llu full_accept=%llu "
                "committed=%llu p=%lld prefill_rows=%llu n1_rows=%llu replays=%llu "
                "replay_rows=%llu replay_bits_diff=%llu replay_mean_kl=%.9g replay_max_kl=%.9g\n",
                w, (unsigned long long) win.verify_rounds, (unsigned long long) win.drafts_proposed,
                (unsigned long long) win.accepted_drafts, (unsigned long long) win.full_accept,
                (unsigned long long) win.committed, (long long) p,
                (unsigned long long) win.prefill_rows, (unsigned long long) win.n1_rows,
                (unsigned long long) win.replays,
                (unsigned long long) win.replay_rows, (unsigned long long) win.replay_bits_diff,
                win.replay_kl.empty() ? 0.0 : replay_mean(win.replay_kl),
                win.replay_kl.empty() ? 0.0 : *std::max_element(win.replay_kl.begin(), win.replay_kl.end()));
    }

    if (capture_on) {
        cap.close();
        std::FILE * rep = std::fopen((out_dir + "/report.json").c_str(), "wbx");
        require(rep != nullptr, "cannot create report.json");
        std::fprintf(rep, "{\"schema_version\":1,\"capture\":\"n3-verify\",\"model\":\"%s\","
                "\"vocabulary\":%u,\"rows\":%llu,\"bytes\":%llu,\"predict\":%u,"
                "\"prompt_tokens\":[%zu,%zu],\"windows\":[",
                model_path.c_str(), n_vocab, (unsigned long long) cap.n_rows,
                (unsigned long long) cap.n_bytes, predict,
                prompt_tokens[0].size(), prompt_tokens[1].size());
        for (size_t w = 0; w < windows.size(); ++w) {
            std::fprintf(rep, "%s{\"window\":%zu,\"verify_rounds\":%llu,\"drafts\":%llu,"
                    "\"accepted\":%llu,\"full_accept\":%llu,\"prefill_rows\":%llu,"
                    "\"n1_rows\":%llu,\"committed\":%llu,"
                    "\"replays\":%llu}",
                    w ? "," : "", w,
                    (unsigned long long) windows[w].verify_rounds,
                    (unsigned long long) windows[w].drafts_proposed,
                    (unsigned long long) windows[w].accepted_drafts,
                    (unsigned long long) windows[w].full_accept,
                    (unsigned long long) windows[w].prefill_rows,
                    (unsigned long long) windows[w].n1_rows,
                    (unsigned long long) windows[w].committed,
                    (unsigned long long) windows[w].replays);
        }
        std::fprintf(rep, "]}\n");
        require(std::fclose(rep) == 0, "report close");
    }

    std::fprintf(stderr, "N3_CAPTURE_OK rows=%llu capture=%d\n",
            (unsigned long long) cap.n_rows, int(capture_on));
    return 0;
}
