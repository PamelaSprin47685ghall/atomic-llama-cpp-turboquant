// N3 verify capture, QUALIFIED TWO-MODEL layout: trunk and MTP head loaded as two separate
// models, exactly as the qualified row-tail runtime does.
//
// Load shape (mirrors overlay/mtp-async-protocol-candidate-20261027/runtime.cpp:1032,925,1192,
// 1213,1215-1240 and nothing else):
//
//   trunk model   the two ORIGINAL shards, 48 blocks, n_layer_nextn == 0
//   head model    the separate trained NextN/MTP GGUF, n_layer_nextn == 1
//   per window    target context  = DEFAULT over the trunk, n_rs_seq as requested
//                 head context    = LLAMA_CONTEXT_TYPE_MTP over the HEAD model,
//                                   ctx_other = that window's trunk context
//   The two models are never merged, aliased, or substituted for one another. The head's input
//   is the trunk's wide NextN residual (llama_get_embeddings_nextn_ith on the trunk context),
//   which is the only link between the two and is checked, not assumed.
//
// Row identity of every captured row (identical to the single-model harness, plus model_role):
//   side, window, model_role ("target"), seq, step, row, input_position, input_token,
//   vocabulary, offset_bytes, count, dtype
// The N1 reference rows carry the same fields with side "n1".
//
// State after a round, and how it is reached. The three-row step leaves the target's GDN cell at
// the state after its LAST row; a round that accepted `a` drafts must end at p + a + 1 with the
// GDN state after row `a`. Two mechanisms exist and the harness asks, never assumes:
//
//   NATIVE   llama_n3_verify_rollback_steps(target) >= the rollback the round needs
//            (rollback = 2 - a, 1 <= rollback <= 2). The step's own speculative suffix is removed
//            from the recurrent cache through the same memory API the qualified runtime uses
//            (timed_seq_rm -> seq_rm(seq, p+1+a, -1)); the recurrent cache performs a bounded
//            per-token rollback when the context has a snapshot ring (n_rs_seq >= rollback).
//            This is the qualified behaviour and the harness does not shadow it.
//   REPLAY   the ring cannot cover the rollback. The pre-step snapshot is restored into the
//            target and the accepted rows are re-decoded one at a time. Diagnostic only.
//   neither  -> N3_REFUSED; a partially accepted round is never committed on a state the harness
//            cannot vouch for.
//
// WHICH OF THE TWO HAS BEEN OBSERVED is recorded per round as `state_path=native|replay` and per
// window as counters. The NATIVE path is a CODE FACT here (the memory call is made and its
// result is required); it is NOT a measured result of this program until Devops runs it. Nothing
// in this file may be quoted as evidence that the native trim works - only as evidence that the
// harness performs it and refuses when it fails.
//
// WHAT MAY NOT BE EXTRAPOLATED. This harness runs the three-row N3 step. The qualified runtime's
// relay path submits a FOUR-row batch (the tp5-relay profile requires n_ubatch >= 4 and the
// runtime's own ubatch width is 4), and its dispatch/scheduling is the relay's, not this
// program's. A result here therefore says nothing about the relay's four-row shape, its kernel
// selection, or its throughput; the two are not the same measurement and neither may be derived
// from the other.
//
// Scope: DIAGNOSTIC. No throughput, ms/tok or tok/s may be derived from a leg of this program.
// It is not the production runtime and does not implement the qualified row-tail protocol.
//
// Refusals (exit 2, nothing written): not exactly five ROCm devices; trunk not 48 blocks with
// no NextN block; head without exactly one NextN block; trunk and head vocabularies differ;
// trunk cannot export the wide residual; an output directory that already exists; a relay/FA
// variable this capture cannot vouch for; no state mechanism for a round it must commit.
//
// Author executed nothing: no compile, no syntax check, no script, no test, no GPU.

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

constexpr int      kN3Rows     = LLAMA_N3_VERIFY_PAIR_MAX_ROWS;   // current + two drafts
constexpr int      kMaxDrafts  = kN3Rows - 1;
constexpr uint64_t kStateBound = 512ull << 20;

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
// capture output
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

    void row(const char * side, const char * model_role, uint32_t window, int32_t seq,
             uint64_t step, int row, int64_t position, int32_t token, const float * logits) {
        std::fprintf(rows,
                "{\"side\":\"%s\",\"model_role\":\"%s\",\"window\":%u,\"seq\":%d,\"step\":%llu,"
                "\"row\":%d,\"input_position\":%lld,\"input_token\":%d,\"vocabulary\":%llu,"
                "\"offset_bytes\":%llu,\"count\":%llu,\"dtype\":\"f32le\"}\n",
                side, model_role, window, seq, (unsigned long long) step, row,
                (long long) position, token, (unsigned long long) n_vocab,
                (unsigned long long) n_bytes, (unsigned long long) n_vocab);
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
    bool owned = true;
    ~context_ptr() { if (p && owned) llama_free(p); }
    llama_context * get() const { return p; }
};

// One window: its own trunk context, its own head context over the separate head model, its own
// sequence and its own buffers. Nothing is shared with the other window except the read-only
// model weights.
struct window {
    uint32_t             id  = 0;
    int32_t              seq = 0;
    context_ptr          target;   // trunk model, DEFAULT context
    context_ptr          head;     // head model, MTP context, ctx_other = target
    context_ptr          n1;       // N1 reference: DEFAULT over the TRUNK model, n_batch 1,
                                   // re-filled from a snapshot of `target` before every step
    std::vector<float>   head_h;   // wide residual the head consumes at the next position
    std::vector<uint8_t> state;    // target snapshot (only used by the replay path)
    uint32_t             rollback_steps = 0;   // as reported by the TARGET context
    uint32_t             n1_rollback_steps = 0; // as reported by the REFERENCE context (must match)
    uint64_t verify_rounds   = 0;
    uint64_t drafts_proposed = 0;
    uint64_t accepted_drafts = 0;
    uint64_t full_accept     = 0;
    uint64_t committed       = 0;
    uint64_t native_rounds   = 0;   // rounds committed with the native bounded rollback
    uint64_t replay_rounds   = 0;   // rounds committed with snapshot + replay
    uint64_t partial_rounds  = 0;   // rounds that were not fully accepted
    uint64_t n1_rows         = 0;
    std::vector<double> replay_kl;
    uint64_t replay_rows      = 0;
    uint64_t replay_bits_diff = 0;
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

// One head step: the head MODEL's context, input (token, trunk residual) at position q.
int32_t head_step(llama_context * hctx, llama_token token, llama_pos q, const float * h_wide,
                  int32_t wide, uint32_t n_vocab, std::vector<float> * out_h) {
    // The width the head graph actually consumes, read from the model this context decodes on -
    // not inherited from the trunk. Cheap enough to state per call, and it is the only place a
    // mismatched buffer could be handed to the graph.
    require(int32_t(llama_model_n_embd_out(llama_get_model(hctx))) == wide,
            "head_step: the residual width does not match the head model's input width");
    llama_batch b = llama_batch_init(1, wide, 1);
    b.n_tokens     = 1;
    b.token[0]     = token;
    b.pos[0]       = q;
    b.n_seq_id[0]  = 1;
    b.seq_id[0][0] = 0;
    b.logits[0]    = 1;
    std::memcpy(b.embd, h_wide, size_t(wide) * sizeof(float));
    const int rc = llama_decode(hctx, b);
    llama_batch_free(b);
    require(rc == 0, "head decode failed");
    const float * dl = llama_get_logits_ith(hctx, 0);
    require(dl != nullptr, "head logits missing");
    const int32_t id = greedy(dl, n_vocab);
    if (out_h) {
        const float * dh = llama_get_embeddings_nextn_ith(hctx, 0);
        require(dh != nullptr, "head wide residual missing");
        out_h->assign(dh, dh + wide);
    }
    return id;
}

void head_trim(llama_context * hctx, llama_pos from) {
    require(llama_memory_seq_rm(llama_get_memory(hctx), 0, from, -1), "head KV tail trim failed");
}

void snapshot(llama_context * ctx, int32_t seq, std::vector<uint8_t> & bytes) {
    const size_t sz = llama_state_seq_get_size_ext(ctx, seq, LLAMA_STATE_SEQ_FLAGS_NONE);
    require(sz > 0 && uint64_t(sz) <= kStateBound, "target state snapshot unavailable");
    bytes.assign(sz, 0);
    require(llama_state_seq_get_data_ext(ctx, bytes.data(), bytes.size(), seq,
                LLAMA_STATE_SEQ_FLAGS_NONE) == bytes.size(),
            "target state snapshot incomplete");
}

// One real row through the two-model entry point, padded to the declared row count.
const float * verify_one(llama_context * ctx, int32_t seq, llama_pos p, llama_token t,
                         uint32_t n_vocab) {
    std::array<llama_token, kN3Rows> toks{t, t, t};
    std::array<int32_t, kN3Rows>     needs{1, 0, 0};
    const int rc = llama_decode_n3_verify_logits_pair(ctx, seq, p, toks.data(), needs.data(), nullptr);
    require(rc == 0, "single-row two-model replay failed");
    const float * row = llama_get_logits_ith(ctx, 0);
    require(row != nullptr, "replayed row logits missing");
    for (uint32_t i = 0; i < n_vocab; ++i) {
        require(std::isfinite(row[i]), "replayed row holds a non-finite logit");
    }
    return row;
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

double mean_of(const std::vector<double> & v) {
    if (v.empty()) {
        return 0.0;
    }
    double sum = 0.0;
    for (const double x : v) {
        sum += x;
    }
    return sum / double(v.size());
}

} // namespace

int main(int argc, char ** argv) {
    const std::string trunk_path = argc > 1 ? argv[1] : "";
    const std::string head_path  = argc > 2 ? argv[2] : "";
    const std::string out_dir    = argc > 3 ? argv[3] : "";
    if (trunk_path.empty() || head_path.empty() || out_dir.empty()) {
        std::fprintf(stderr,
                "usage: %s <trunk shard 1 of 2> <trained NextN head .gguf> <output dir>\n", argv[0]);
        return 64;
    }
    require(trunk_path != head_path, "the trunk and the head must be two different files");

    if (std::getenv("GGML_TP5_RSZ_RT")) {
        refuse("GGML_TP5_RSZ_RT is set (promotion path)");
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
    // The rollback ring the qualified runtime requests (runtime.cpp: params.n_rs_seq = 2 when
    // GGML_TP5_MTP_RS_RING=1). The harness asks the context for the number it actually got.
    const uint32_t want_rs_seq = env_u32("N3_RS_SEQ", 2);
    const bool     allow_replay = env_is("N3_SNAPSHOT_REPLAY", "1");
    require(predict >= 2 && predict <= 512, "N3_PREDICT must be in [2, 512]");

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

    // ---- two models, loaded exactly as the qualified runtime loads them --------------------
    model_ptr trunk;
    trunk.p = llama_model_load_from_file(trunk_path.c_str(), mparams);
    require(trunk.get() != nullptr, "trunk model load failed (the two original shards)");
    if (llama_model_n_layer_nextn(trunk.get()) != 0 || llama_model_n_layer(trunk.get()) != 48) {
        refuse("the trunk is not a 48-block model without a NextN block");
    }

    model_ptr head;
    auto hparams = mparams;
    hparams.load_mtp = true;
    head.p = llama_model_load_from_file(head_path.c_str(), hparams);
    require(head.get() != nullptr, "head model load failed (the separate trained NextN GGUF)");
    if (llama_model_n_layer_nextn(head.get()) != 1) {
        refuse("the head model does not carry exactly one NextN block");
    }

    const llama_vocab * tvocab = llama_model_get_vocab(trunk.get());
    const llama_vocab * hvocab = llama_model_get_vocab(head.get());
    const uint32_t n_vocab = uint32_t(llama_vocab_n_tokens(tvocab));
    require(n_vocab > 0, "invalid trunk vocabulary");
    // A row comparison across vocabularies is meaningless, and the head's tokens are indices into
    // its own vocabulary: the two must be the SAME vocabulary. Token 0's text proves nothing about
    // the other n_vocab-1 entries, so the whole table is compared token by token.
    require(uint32_t(llama_vocab_n_tokens(hvocab)) == n_vocab,
            "trunk and head vocabularies differ in size");
    for (uint32_t t = 0; t < n_vocab; ++t) {
        const char * a = llama_vocab_get_text(tvocab, llama_token(t));
        const char * b = llama_vocab_get_text(hvocab, llama_token(t));
        require(a != nullptr && b != nullptr,
                "a vocabulary has no text for a token id; the tables cannot be compared");
        if (std::strcmp(a, b) != 0) {
            char why[192];
            std::snprintf(why, sizeof(why),
                    "trunk and head vocabularies differ at token %u (%s vs %s)", t, a, b);
            refuse(why);
        }
    }

    // The head's input width is NOT the trunk's n_embd_out by assumption: graph_mtp builds its
    // input from ITS OWN hparams (`inp->h` has ne[0] = hc * n_embd) and
    // llm_graph_input_embd_h::set_input asserts n_embd == h->ne[0] before copying n_tokens*n_embd
    // floats out of the caller's buffer. Feeding a wider or narrower buffer would read past or
    // short of it, so the two widths must be equal and the harness refuses otherwise.
    const int32_t wide = llama_model_n_embd_out(trunk.get());
    const int32_t wide_head = llama_model_n_embd_out(head.get());
    require(wide > 0, "the trunk does not expose a wide output width");
    require(wide_head == wide,
            "the head expects a different residual width than the trunk exports "
            "(graph_mtp copies n_tokens * n_embd floats from the caller's buffer)");

    std::vector<std::vector<llama_token>> prompt_tokens;
    for (const auto & text : { prompt0, prompt1 }) {
        prompt_tokens.push_back(tokenize(tvocab, text));
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

        auto tp = base_params();
        tp.n_rs_seq = want_rs_seq;    // asked for, then verified below
        win.target.p = llama_init_from_model(trunk.get(), tp);
        require(win.target.get() != nullptr, "trunk context creation failed");
        llama_set_embeddings_nextn(win.target.get(), true, /*masked=*/false);
        win.rollback_steps = llama_n3_verify_rollback_steps(win.target.get());

        // the head context: its own model, its own KV, paired with THIS window's trunk context
        auto hp = base_params();
        hp.ctx_type  = LLAMA_CONTEXT_TYPE_MTP;
        hp.n_batch   = 1;
        hp.n_ubatch  = 1;
        hp.ctx_other = win.target.get();
        win.head.p   = llama_init_from_model(head.get(), hp);
        require(win.head.get() != nullptr, "head context creation failed");
        llama_set_embeddings_nextn(win.head.get(), true, /*masked=*/false);

        // The N1 reference is a context of the SAME model as the candidate (the trunk), so a
        // row comparison is between two decodes of one model and not between two models. It is
        // re-filled from a snapshot before every step and holds nothing of its own.
        // The reference must be the SAME configuration as the target, or a snapshot taken from
        // one may not describe the other: same model, same ctx size, same flash-attn, same KV
        // types, same n_rs_seq.
        //
        // Decode shape: the reference decodes ONE REAL ROW per call, and it passes exactly one
        // token (llama_batch_init(1, ...)), so the number of rows computed is 1. What n_rs_seq > 0
        // changes is the recurrent cache's CAPACITY: with a ring the cache splits its ubatch into
        // n_rs_seq + 1 chunks internally (llama-memory-recurrent.cpp: `split_equal(n_ubatch, true,
        // n_rs_seq > 0 ? n_rs_seq + 1 : 0)`), so the allocation is sized for up to 4 rows per
        // sequence at n_rs_seq = 2. It does NOT turn the reference into a multi-row decode: the
        // row count is what the batch says, and the batch says 1.
        auto np = base_params();
        np.n_batch   = 1;
        np.n_ubatch  = 1;
        np.n_rs_seq  = tp.n_rs_seq;
        if (w == 0) {
            win.n1.p     = llama_init_from_model(trunk.get(), np);
        } else {
            win.n1.p     = windows[0].n1.p;
            win.n1.owned = false;
        }
        require(win.n1.get() != nullptr, "N1 reference context creation failed");
        llama_set_embeddings_nextn(win.n1.get(), true, /*masked=*/false);
        // and the values must actually agree, not merely be requested
        require(llama_n_rs_seq(win.n1.get()) == llama_n_rs_seq(win.target.get()),
                "reference and target contexts disagree on n_rs_seq; a snapshot would not describe "
                "the reference");
        win.n1_rollback_steps = llama_n_rs_seq(win.n1.get());

        win.head_h.assign(size_t(wide), 0.0f);
    }

    // The rollback capability is REPORTED by each context, never assumed. The harness accepts a
    // context with no ring (it then needs the replay path) but refuses a context whose ring is
    // smaller than the largest rollback a round can need, because that round could not be
    // committed.
    for (const auto & win : windows) {
        // The context must report EXACTLY the ring that was requested. A silent downgrade would
        // turn "native rollback" into a claim about a configuration that never existed.
        require(win.rollback_steps == want_rs_seq,
                "the target reports n_rs_seq != the requested N3_RS_SEQ; refusing a downgraded "
                "configuration");
        require(win.rollback_steps == 0 || win.rollback_steps >= uint32_t(kN3Rows - 1),
                "the context reports a rollback ring too small for a three-row round");
        if (win.rollback_steps == 0 && !allow_replay) {
            refuse("no recurrent rollback ring and N3_SNAPSHOT_REPLAY=0: a partially accepted "
                   "round cannot be committed");
        }
    }

    // Prove, before the first round, that a state produced by the TARGET can be consumed by the
    // REFERENCE: snapshot the empty target and restore it into the empty reference. A size
    // mismatch here means the two contexts are not snapshot-compatible and every later
    // comparison would be meaningless, so it is a refusal rather than a first-round failure.
    if (capture_on) {
        std::vector<uint8_t> probe;
        snapshot(windows[0].target.get(), 0, probe);
        const size_t back = llama_state_seq_set_data_ext(windows[0].n1.get(), probe.data(),
                probe.size(), 0, LLAMA_STATE_SEQ_FLAGS_NONE);
        require(back == probe.size(),
                "target and reference contexts are not snapshot-compatible (restore size mismatch)");
        require(llama_memory_seq_rm(llama_get_memory(windows[0].n1.get()), 0, -1, -1),
                "reference probe state removal failed");
        std::fprintf(stderr, "N3_PLAN snapshot_probe_bytes=%zu compatible=1\n", probe.size());
    }

    capture cap;
    if (capture_on) {
        cap.open(out_dir, n_vocab);
    }
    std::fprintf(stderr,
            "N3_PLAN two_model=1 trunk=%s head=%s vocab=%u wide=%d predict=%u rows=%d "
            "rollback_steps=[%u,%u] replay_allowed=%d capture=%d\n",
            trunk_path.c_str(), head_path.c_str(), n_vocab, wide, predict, kN3Rows,
            windows[0].rollback_steps, windows[1].rollback_steps, int(allow_replay), int(capture_on));

    // ---- prefill: trunk rows one at a time, head catch-up over the same tokens ---------------
    for (uint32_t w = 0; w < 2; ++w) {
        auto &       win = windows[w];
        const auto & pt  = prompt_tokens[w];
        std::vector<float> prev(size_t(wide), 0.0f);
        for (size_t i = 0; i < pt.size(); ++i) {
            const bool last = (i + 1 == pt.size());
            const float * tgt = decode_one(win.target.get(), win.seq, llama_pos(i), pt[i], last);
            require(tgt != nullptr, "prompt logits missing");
            if (capture_on && !last) {
                (void) decode_one(win.n1.get(), win.seq, llama_pos(i), pt[i], false);
            }
            const float * h = llama_get_embeddings_nextn_ith(win.target.get(), 0);
            require(h != nullptr,
                    "the trunk did not export the wide NextN residual; the two-model layout "
                    "needs it as the head's input");
            std::vector<float> next;
            (void) head_step(win.head.get(), pt[i], llama_pos(i), prev.data(), wide, n_vocab, &next);
            prev = next;
            if (last && capture_on) {
                cap.row("prefill", "target", w, win.seq, 0, 0, llama_pos(i), pt[i], tgt);
                const float * n1 = decode_one(win.n1.get(), win.seq, llama_pos(i), pt[i], true);
                require(n1 != nullptr, "reference prompt logits missing");
                cap.row("prefill", "n1", w, win.seq, 0, 0, llama_pos(i), pt[i], n1);
            }
        }
        win.head_h = prev;
    }

    // ---- decode loops ----------------------------------------------------------------------
    for (uint32_t w = 0; w < 2; ++w) {
        auto & win = windows[w];
        llama_token last_emitted = prompt_tokens[w].back();
        llama_pos   p            = llama_pos(prompt_tokens[w].size());
        std::vector<llama_token> drafts;
        drafts.reserve(kMaxDrafts);

        while (win.committed + 1 < predict) {
            // drafts, from the head state at p
            drafts.clear();
            {
                std::vector<float> h   = win.head_h;
                llama_token        tok = last_emitted;
                const int want = int(std::min<uint64_t>(kMaxDrafts, predict - win.committed - 1));
                for (int k = 0; k < want; ++k) {
                    std::vector<float> next;
                    const int32_t d = head_step(win.head.get(), tok, p + k, h.data(), wide, n_vocab, &next);
                    drafts.push_back(llama_token(d));
                    tok = llama_token(d);
                    h   = next;
                }
                head_trim(win.head.get(), p);
            }
            const int n_rows = int(1 + drafts.size());
            require(n_rows >= 2 && n_rows <= kN3Rows, "verify row count out of range");

            std::array<llama_token, kN3Rows> toks{};
            toks[0] = last_emitted;
            for (int k = 0; k < kN3Rows - 1; ++k) {
                toks[size_t(k + 1)] = k < n_rows - 1 ? drafts[size_t(k)] : toks[size_t(k)];
            }

            // The state mechanism for this round is chosen BEFORE the step, from what the
            // context reports and from the rollback the round would need.
            const uint32_t need_rollback = uint32_t(kN3Rows - 1);   // worst case: 0 accepted
            const bool native_possible = win.rollback_steps >= need_rollback;
            if (!native_possible && !allow_replay) {
                refuse("this context cannot roll back a partially accepted round and the replay "
                       "path is disabled");
            }
            // The snapshot is taken in EVERY round and in both legs: the replay path needs it to
            // restore the target, and the N1 reference needs it to start from the SAME state as
            // the step. It is taken before the step, so nothing the step does can leak into it.
            snapshot(win.target.get(), win.seq, win.state);

            // ---- N1 reference: the same rows, one at a time, from the same state -----------
            // A one-row decode of the same sequence at some later time is NOT the same state
            // (the GDN cell, the PLE history window and the indexer frontier all differ), so the
            // reference is a context restored from the snapshot just taken and advanced row by
            // row with exactly these tokens at exactly these positions.
            if (capture_on) {
                // The restore returning the snapshot's exact size is the only evidence available
                // that the reference context can consume a state produced by the target. A
                // different size means the two layouts disagree and the reference is not a
                // reference; that is fatal, not a warning.
                const size_t restored = llama_state_seq_set_data_ext(win.n1.get(), win.state.data(),
                        win.state.size(), win.seq, LLAMA_STATE_SEQ_FLAGS_NONE);
                require(restored == win.state.size(),
                        "the reference context rejected the target's snapshot (different size); "
                        "the two layouts are not snapshot-compatible and no row may be compared");
                for (int k = 0; k < n_rows; ++k) {
                    const float * row = decode_one(win.n1.get(), win.seq, p + k, toks[size_t(k)], true);
                    require(row != nullptr, "N1 reference row logits missing");
                    cap.row("n1", "target", w, win.seq, win.verify_rounds, k, p + k,
                            toks[size_t(k)], row);
                    ++win.n1_rows;
                }
                require(llama_memory_seq_rm(llama_get_memory(win.n1.get()), win.seq, -1, -1),
                        "N1 reference state removal failed");
            }

            // ---- the N3 step (two-model entry point) --------------------------------------
            std::array<std::vector<float>, kN3Rows> all_rows;
            std::array<float *, kN3Rows> all_outs{};
            std::array<int32_t, kN3Rows> needs{};
            for (int k = 0; k < kN3Rows; ++k) {
                needs[size_t(k)] = (capture_on && k < n_rows) ? 1 : 0;
                if (needs[size_t(k)]) {
                    all_rows[size_t(k)].assign(size_t(n_vocab), 0.0f);
                    all_outs[size_t(k)] = all_rows[size_t(k)].data();
                }
            }
            const int rc = llama_decode_n3_verify_logits_pair(win.target.get(), win.seq, p,
                    toks.data(), needs.data(), capture_on ? all_outs.data() : nullptr);
            require(rc == 0, "two-model verify decode failed");

            std::array<int32_t, kN3Rows> argmax{};
            for (int k = 0; k < kN3Rows; ++k) {
                if (k >= n_rows) {
                    argmax[size_t(k)] = -1;
                    continue;
                }
                const float * row = llama_get_logits_ith(win.target.get(), k);
                require(row != nullptr, "verify row logits missing");
                if (capture_on) {
                    const auto & copy = all_rows[size_t(k)];
                    require(copy.size() == size_t(n_vocab), "capture row width changed");
                    require(std::memcmp(copy.data(), row, size_t(n_vocab) * sizeof(float)) == 0,
                            "captured row differs from the context's row");
                    cap.row("n3", "target", w, win.seq, win.verify_rounds, k, p + k,
                            toks[size_t(k)], row);
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

            // the head's input residual for the last accepted row, read before any state surgery
            const float * hr = llama_get_embeddings_nextn_ith(win.target.get(), accepted);
            require(hr != nullptr, "trunk wide residual missing for the last verified row");
            std::vector<float> last_h(hr, hr + wide);

            const llama_pos keep_to = p + llama_pos(accepted) + 1;
            const bool needs_rollback = keep_to < p + llama_pos(kN3Rows);

            // ---- commit the state --------------------------------------------------------
            const char * state_path = "none";
            if (!needs_rollback) {
                // the step's own state is exactly the committed state
                state_path = "none";
            } else if (native_possible) {
                // The qualified mechanism: remove the speculative suffix from the recurrent
                // cache, which performs a bounded per-token rollback when the context has the
                // ring. This is the same memory call the runtime makes; the harness only
                // requires that it succeeds.
                const llama_pos invalid_from = keep_to;
                require(llama_memory_seq_rm(llama_get_memory(win.target.get()), win.seq, invalid_from, -1),
                        "native bounded rollback of the speculative suffix failed");
                state_path = "native";
                ++win.native_rounds;
                ++win.partial_rounds;
            } else {
                // Diagnostic fallback: restore the pre-step snapshot and replay the accepted
                // rows one at a time.
                const size_t restored = llama_state_seq_set_data_ext(win.target.get(),
                        win.state.data(), win.state.size(), win.seq, LLAMA_STATE_SEQ_FLAGS_NONE);
                require(restored == win.state.size(),
                        "snapshot restore incomplete (state_drop may have run); the target must "
                        "not be used again");
                for (int k = 0; k <= accepted; ++k) {
                    const float * row = verify_one(win.target.get(), win.seq, p + llama_pos(k),
                            toks[size_t(k)], n_vocab);
                    require(row != nullptr, "replayed row logits missing");
                    if (capture_on) {
                        const auto & orig = all_rows[size_t(k)];
                        require(orig.size() == size_t(n_vocab), "replayed row width changed");
                        const double d = soft_gate_row(orig.data(), row, n_vocab);
                        require(std::isfinite(d), "replayed row is not finite");
                        win.replay_kl.push_back(d);
                        win.replay_rows += 1;
                        win.replay_bits_diff += (std::memcmp(orig.data(), row,
                                size_t(n_vocab) * sizeof(float)) == 0) ? 0 : 1;
                    }
                }
                const float * hr2 = llama_get_embeddings_nextn_ith(win.target.get(), accepted);
                require(hr2 != nullptr, "trunk wide residual missing after the replay");
                last_h.assign(hr2, hr2 + wide);
                state_path = "replay";
                ++win.replay_rounds;
                ++win.partial_rounds;
            }

            // the sequence must now end exactly at keep_to
            require(keep_to >= p, "commit position moved backwards");
            const llama_pos frontier = llama_memory_seq_pos_max(llama_get_memory(win.target.get()), win.seq);
            require(frontier + 1 == keep_to,
                    "target frontier is not at the committed position (a speculative suffix remains)");

            for (int k = 0; k < accepted; ++k) {
                last_emitted = toks[size_t(k + 1)];
                ++win.committed;
            }
            last_emitted = llama_token(argmax[size_t(accepted)]);
            ++win.committed;

            // ---- advance the head over the accepted rows ----------------------------------
            std::vector<float> head_out;
            (void) head_step(win.head.get(), toks[size_t(accepted)], p + llama_pos(accepted),
                    last_h.data(), wide, n_vocab, &head_out);
            win.head_h = head_out;
            p = keep_to;
            head_trim(win.head.get(), p);

            if (capture_on) {
                std::fprintf(stderr, "N3_STEP window=%u step=%llu accepted=%d rows=%d state_path=%s\n",
                        w, (unsigned long long) win.verify_rounds - 1, accepted, n_rows, state_path);
            }
        }

        std::fprintf(stderr,
                "N3_WINDOW window=%u verify_rounds=%llu drafts=%llu accepted=%llu full_accept=%llu "
                "committed=%llu p=%lld n1_rows=%llu partial=%llu native=%llu replay=%llu rollback_steps=%u "
                "replay_rows=%llu replay_bits_diff=%llu replay_mean_kl=%.9g replay_max_kl=%.9g\n",
                w, (unsigned long long) win.verify_rounds, (unsigned long long) win.drafts_proposed,
                (unsigned long long) win.accepted_drafts, (unsigned long long) win.full_accept,
                (unsigned long long) win.committed, (long long) p,
                (unsigned long long) win.n1_rows,
                (unsigned long long) win.partial_rounds, (unsigned long long) win.native_rounds,
                (unsigned long long) win.replay_rounds, win.rollback_steps,
                (unsigned long long) win.replay_rows, (unsigned long long) win.replay_bits_diff,
                win.replay_kl.empty() ? 0.0 : mean_of(win.replay_kl),
                win.replay_kl.empty() ? 0.0 : *std::max_element(win.replay_kl.begin(), win.replay_kl.end()));
    }

    if (capture_on) {
        cap.close();
        std::FILE * rep = std::fopen((out_dir + "/report.json").c_str(), "wbx");
        require(rep != nullptr, "cannot create report.json");
        // "capture" names the producer; the gate accepts both producers and requires the
        // reference rows to be present, which is what makes the pair comparable at all.
        std::fprintf(rep, "{\"schema_version\":1,\"capture\":\"n3-verify\","
                "\"layout\":\"two-model\","
                "\"trunk\":\"%s\",\"head\":\"%s\",\"vocabulary\":%u,\"rows\":%llu,\"bytes\":%llu,"
                "\"predict\":%u,\"prompt_tokens\":[%zu,%zu],\"windows\":[",
                trunk_path.c_str(), head_path.c_str(), n_vocab, (unsigned long long) cap.n_rows,
                (unsigned long long) cap.n_bytes, predict,
                prompt_tokens[0].size(), prompt_tokens[1].size());
        for (size_t w = 0; w < windows.size(); ++w) {
            // `native` counts rounds committed by the context's own bounded rollback (the ring);
            // `replay` counts rounds committed by snapshot+restore+replay. They are reported
            // separately and never summed into a "ring" claim: a leg whose partial rounds are all
            // replay has used NO ring at all, whatever n_rs_seq reports.
            std::fprintf(rep, "%s{\"window\":%zu,\"verify_rounds\":%llu,\"drafts\":%llu,"
                    "\"accepted\":%llu,\"full_accept\":%llu,\"committed\":%llu,\"n1_rows\":%llu,"
                    "\"partial\":%llu,"
                    "\"native\":%llu,\"replay\":%llu,\"rollback_steps\":%u,"
                    "\"requested_rollback_steps\":%u,\"replay_rows\":%llu}",
                    w ? "," : "", w,
                    (unsigned long long) windows[w].verify_rounds,
                    (unsigned long long) windows[w].drafts_proposed,
                    (unsigned long long) windows[w].accepted_drafts,
                    (unsigned long long) windows[w].full_accept,
                    (unsigned long long) windows[w].committed,
                    (unsigned long long) windows[w].n1_rows,
                    (unsigned long long) windows[w].partial_rounds,
                    (unsigned long long) windows[w].native_rounds,
                    (unsigned long long) windows[w].replay_rounds,
                    windows[w].rollback_steps,
                    want_rs_seq,
                    (unsigned long long) windows[w].replay_rows);
        }
        std::fprintf(rep, "]}\n");
        require(std::fclose(rep) == 0, "report close");
    }

    std::fprintf(stderr, "N3_CAPTURE_OK rows=%llu capture=%d\n",
            (unsigned long long) cap.n_rows, int(capture_on));
    return 0;
}
