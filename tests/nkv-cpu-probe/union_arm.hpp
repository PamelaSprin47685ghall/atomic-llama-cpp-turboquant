#pragma once

//
// union_arm.hpp
//
// Shared arm machinery for the CPU probes that compare a union-binder context
// against a stock context on one model. Header-only, no new translation unit:
// every probe that uses it stays a single .cpp and links `llama` only.
//
// What lives here, and why it is here instead of in each probe:
//
//   - `arm_env`  : which environment variables select which arm, and the one
//                  function that applies them. The variables are read by the
//                  llama_context CONSTRUCTOR, so an arm is selected by setting
//                  them before the context is built and never by unsetting them
//                  afterwards. Three probes had their own copy of this block,
//                  and two of them disagreed about which variables they own.
//   - `open_model` / `close_model`, `make_context` : model and context setup
//                  with the CPU-only parameters the binder requires.
//   - `build_delta` : `llama_graph_build_count()` delta across a scope. This is
//                  the assertion that makes two arms different paths: the
//                  binder arm must report 0 and the stock arm at least one per
//                  cell. A probe that guessed a count instead of measuring it
//                  would be a gate that cannot fail.
//   - `rows_equal` : byte comparison of two captured logits blocks.
//   - `fail` / `arm_report` : the one place a gate failure is printed, so a
//                  reader and a shell gate see the same `FAIL:` shape.
//
// What does NOT live here: any probe's schedule, its gate list, or its JSON
// report. Those are the probe's own contract and stay in the probe.
//
// Every function is `inline` so the header can be included by more than one
// probe without an ODR problem, and nothing here allocates on its own.
//

#include "llama.h"

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace union_arm {

// ---------------------------------------------------------------------------
// environment: which variable selects which arm
// ---------------------------------------------------------------------------
//
// Each variable is read by the llama_context constructor, never by the probe.
// Setting a variable therefore selects the arm of the NEXT context that is
// built; unsetting it after the build changes nothing.
//
// GGML_TP5_UNION_PAD_BUCKET and GGML_TP5_NKV_FORCE_PAD_FALLBACK are NOT here:
// the padded lane is no longer a terminal state, so both knobs and the
// pad-fallback gate that drove them were deleted. The A/B control lane is now
// GGML_TP5_UNION_BINDER_EXACT=0, which selects the stock padded width without
// a bucket floor.
struct arm_env {
    // GGML_TP5_UNION_BINDER: "1" opts the binder in, "0" disables it, unset
    // leaves the CPU default. The binder arm sets "1"; the stock arm sets "0".
    const char * binder = nullptr;
    // LLAMA_GRAPH_REUSE_DISABLE: "1" makes every decode a real graph build.
    // The stock arm that must rebuild sets it; the binder arm unsets it.
    const char * reuse_disable = nullptr;
    // GGML_TP5_UNION_BINDER_EXACT: "0" selects the stock padded control lane
    // (the A/B arm); unset leaves the runtime-length lane, the default.
    const char * binder_exact = nullptr;
    // LLAMA_EXPERIMENTAL_VARLEN_NTOK: "1" enables the token lane.
    const char * varlen_ntok = nullptr;
    // LLAMA_NKV_MASK_OVERRIDE: a strictly positive decimal width that replaces
    // the real width in the KQ-mask fill, so the refusal path can be driven.
    const char * mask_override = nullptr;

    // Apply the block. A null member UNSETS that variable, which is the only
    // way to say "this arm does not own it": every arm states all five, so no
    // arm can inherit a variable from the arm before it.
    void apply() const {
        set_or_unset("GGML_TP5_UNION_BINDER",             binder);
        set_or_unset("LLAMA_GRAPH_REUSE_DISABLE",         reuse_disable);
        set_or_unset("GGML_TP5_UNION_BINDER_EXACT",       binder_exact);
        set_or_unset("LLAMA_EXPERIMENTAL_VARLEN_NTOK",    varlen_ntok);
        set_or_unset("LLAMA_NKV_MASK_OVERRIDE",           mask_override);
    }

    // The two arms every probe in this directory compares. `binder` is the
    // fixed-topology arm; `stock` rebuilds. They differ in exactly two
    // variables, and the build-count delta is what proves it.
    static arm_env binder_arm() {
        arm_env e;
        e.binder = "1";
        return e;
    }
    static arm_env stock_arm() {
        arm_env e;
        e.binder        = "0";
        e.reuse_disable = "1";
        return e;
    }

private:
    static void set_or_unset(const char * name, const char * value) {
        if (value != nullptr) {
            setenv(name, value, 1);
        } else {
            unsetenv(name);
        }
    }
};

// ---------------------------------------------------------------------------
// model and context
// ---------------------------------------------------------------------------

struct model_handle {
    llama_model * model = nullptr;
    int           vocab = 0;
};

// Load a model on the CPU only. Returns false and prints a FAIL line when the
// file cannot be loaded or its vocabulary is unusable.
inline bool open_model(const char * path, model_handle & m) {
    llama_backend_init();

    auto mp = llama_model_default_params();
    ggml_backend_dev_t devices[] = { nullptr };
    mp.devices      = devices;
    mp.n_gpu_layers = 0;

    m.model = llama_model_load_from_file(path, mp);
    if (m.model == nullptr) {
        fail("model load: %s", path);
        llama_backend_free();
        return false;
    }
    m.vocab = llama_vocab_n_tokens(llama_model_get_vocab(m.model));
    if (m.vocab <= 3) {
        std::fprintf(stderr, "FAIL: vocab=%d (need more than the 3 special ids)\n", m.vocab);
        llama_model_free(m.model);
        m.model = nullptr;
        llama_backend_free();
        return false;
    }
    return true;
}

inline void close_model(model_handle & m) {
    if (m.model != nullptr) {
        llama_model_free(m.model);
        m.model = nullptr;
    }
    llama_backend_free();
}

// The CPU context the binder requires: one unified KV stream, one sequence,
// no flash attention, no offload. `n_outputs_max` is the legal active-output
// bound; a request above it must be refused, which is one of the negative
// cases.
inline llama_context_params cpu_params(uint32_t n_ctx, uint32_t n_batch, uint32_t n_outputs_max) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = n_ctx;
    cp.n_batch         = n_batch;
    cp.n_ubatch        = n_batch;
    cp.n_seq_max       = 1;
    cp.n_outputs_max   = n_outputs_max;
    cp.n_threads       = 1;
    cp.n_threads_batch = 1;
    cp.kv_unified      = true;
    cp.offload_kqv     = false;
    cp.op_offload      = false;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.embeddings      = false;
    return cp;
}

// Build a context under an explicit arm. The arm's environment is applied
// first, so the constructor sees it. The caller owns the returned context.
inline llama_context * make_context(const model_handle & m, const arm_env & arm,
                                    const llama_context_params & cp) {
    arm.apply();
    return llama_init_from_model(m.model, cp);
}

// ---------------------------------------------------------------------------
// measurements
// ---------------------------------------------------------------------------

// The build counter as a scope. `delta()` is the number of graphs the context
// built since the object was created. A binder arm that reports 0 did not
// rebuild; a stock arm that reports less than one per cell did not take the
// stock path.
struct build_delta {
    const llama_context * ctx = nullptr;
    uint64_t              at  = 0;

    explicit build_delta(const llama_context * c) : ctx(c), at(c != nullptr ? llama_graph_build_count(c) : 0) {}

    uint64_t delta() const {
        return ctx != nullptr ? llama_graph_build_count(ctx) - at : 0;
    }
};

// Byte comparison of two captured logits blocks. Sizes must match: a block
// captured at a different row count is never "equal", it is a capture defect.
inline bool rows_equal(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && !a.empty() &&
           std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

// ---------------------------------------------------------------------------
// reporting
// ---------------------------------------------------------------------------

// The one failure printer. Every probe prints through it, so a shell gate can
// match `^FAIL:` without knowing which probe produced the line.
inline void fail(const char * fmt, ...) {
    std::fprintf(stderr, "FAIL: ");
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(stderr, fmt, ap);
    va_end(ap);
    std::fprintf(stderr, "\n");
}

// A gate failure that names the gate and carries its observed values. The
// `GATE_FAIL=<name>` prefix is what the gate scripts grep for; the rest is for
// a reader who has to decide whether the gate or the code is wrong.
inline void gate_fail(const char * gate, const char * fmt, ...) {
    std::fprintf(stderr, "GATE_FAIL=%s ", gate);
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(stderr, fmt, ap);
    va_end(ap);
    std::fprintf(stderr, "\n");
}

} // namespace union_arm
