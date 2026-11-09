// CPU probe for the padmax runtime-length lane (union binder n_kv + capacity).
//
// Purpose: prove, on a CPU-only model, that the runtime-length lane produces
// bitwise-identical logits to the stock path and that it does not add graph
// rebuilds.  The probe also records the context capacities (n_ctx, n_ctx_seq,
// n_ctx_kv) so a hidden 256-cell pad/headroom cannot pass unnoticed.
//
// Public llama.h only.  n_gpu_layers = 0 and an explicitly empty device list;
// the process must be run with HIP_VISIBLE_DEVICES= and ROCR_VISIBLE_DEVICES=.
//
// Modes (requested values, no added headroom):
//   nkv : n_ctx=512, n_ctx_kv=0 (capacity follows n_ctx_seq), kv_unified=true,
//         512 steps x 1 token = 512 positions.  Crosses the 256 bucket and
//         reaches the full capacity.  n_kv widths: 1,2,...,512.
//   cap : n_ctx=300, n_ctx_kv=300, kv_unified=true,
//         300 steps x 1 token = 300 positions.  Exact physical capacity 300;
//         n_kv widths: 1,...,300.
//   seq : n_ctx=1001, n_seq_max=2, kv_unified=false. The unset default
//         must reject this multi-stream cache; explicit opt-in must still
//         hit the binder's multi-stream refusal and fall back to stock.
//
// A fixed one-token step keeps the ubatch shape constant, so the only moving
// reuse key is n_kv.  A variable step size would make allow_reuse() fail on
// the token count and confound the rebuild count.
//
// Record layout (little-endian), compatible with compare_records.py:
//   uint32 n_tokens, vocab, pos_after, step, n_outputs, seq_id
//   float32 logits[n_outputs * vocab]
// n_outputs is always 1; the recorded row is the last token of the step, so
// the attention span covers the whole step and the exact n_kv width.
//
// Arms are selected with the environment outside the probe:
//   unset : no GGML_TP5_UNION_BINDER
//   off   : GGML_TP5_UNION_BINDER=0
//   on    : GGML_TP5_UNION_BINDER=1
//
// The probe never sets, unsets or overrides those variables. It reads them
// once at process start and prints them as
//   ARM binder=<unset|0|1|other> ntok=<unset|0|1|other> reuse_disable=<unset|0|1|other>
// so the arm a run actually took is in its own log. run_probe.sh injects the
// arm with `env`; check_probe.py refuses an arm whose log does not match the
// arm the runner asked for.
//
// The probe also observes the three n_kv runtime roots on every step through
// the eval callback (see shape_capture below): `attn_inp_kq_mask` and the
// `union_nkv_k` / `union_nkv_v` views. Their n_kv dimension must equal the
// 1-based position after the step. A root the callback never sees is a named
// failure, never a skip.
#include "llama.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {
// The arm this process was started with. Read once, never written: the
// runner owns the environment, the probe only reports it.
struct arm_env {
    const char * binder;
    const char * ntok;
    const char * reuse_disable;
};

const char * arm_value(const char * value) {
    return value == nullptr ? "unset" : value;
}

arm_env read_arm_env() {
    return arm_env{
        /*.binder        =*/ arm_value(std::getenv("GGML_TP5_UNION_BINDER")),
        /*.ntok          =*/ arm_value(std::getenv("LLAMA_EXPERIMENTAL_VARLEN_NTOK")),
        /*.reuse_disable =*/ arm_value(std::getenv("LLAMA_GRAPH_REUSE_DISABLE")),
    };
}

// What the eval callback observed for one decode step.
//
// Two of the three n_kv roots are nodes and one is an input, so the callback
// reaches them by two different routes:
//
//   union_nkv_k  llama-kv-cache.cpp:2184  llama_kv_cache::get_k
//   union_nkv_v  llama-kv-cache.cpp:2215,2226  llama_kv_cache::get_v
//                ggml_view_4d results, i.e. GGML_OP_VIEW nodes; the callback
//                is called with them directly.
//
//   attn_inp_kq_mask  llama-graph.cpp:97 (name at :111), and
//                     dsv4_build_raw_kq_mask at :976 (name at :992)
//                a graph INPUT (ggml_new_tensor_4d + ggml_set_input), never a
//                node. ggml_backend_sched_compute_splits
//                (ggml/src/ggml-backend.cpp:2039) calls the eval callback over
//                `split->graph.nodes` only; an input travels in
//                `split->inputs[]` (:1873) and the callback never sees it.
//                The mask is therefore observed on its CONSUMERS: SOFT_MAX
//                src[1] (ggml.c:4183), FLASH_ATTN_EXT and
//                FLASH_ATTN_EXT_BANDED src[3] (ggml.c:5550, :5597), MUL_MAT
//                src[2] when an attention flag is set (ggml.c:3394, :3408).
//                This mirrors llama-context.cpp:4721.
//
// The callback runs after the binder patched the graph and after the mask
// fill, so these are the executed shapes.
struct shape_capture {
    bool seen_mask = false;
    bool seen_k    = false;
    bool seen_v    = false;

    int64_t mask_ne0 = -1;
    int64_t mask_ne1 = -1;
    int64_t k_nkv    = -1;
    int64_t v_nkv    = -1;

    // The mask tensor itself, so the row can be read AFTER llama_decode
    // returned. Valid only while the graph is alive; the callback that sets
    // it runs inside llama_decode.
    ggml_tensor * mask = nullptr;
    enum ggml_type mask_type = GGML_TYPE_COUNT;

    // The pad-tail width of the node that carries the flag, or -1 when no
    // node of the graph carries GGML_TENSOR_FLAG_REDUCE_PAD_TAIL. On this
    // probe's graph (flash attention off, ordered attention lane) that is
    // the expected state: the producer skips a softmax that already carries
    // ATTN_SOFT_MAX (llama-context.cpp:4720).
    int32_t pad_tail = -1;

    void reset() {
        seen_mask = false;
        seen_k    = false;
        seen_v    = false;
        mask_ne0  = -1;
        mask_ne1  = -1;
        k_nkv     = -1;
        v_nkv     = -1;
        mask      = nullptr;
        mask_type = GGML_TYPE_COUNT;
        pad_tail  = -1;
    }
};

// The n_kv dimension of the V view depends on the cache layout: get_v builds
// ggml_view_4d(head_v_eff, n_head_kv, n_kv, ns) when !v_trans and
// ggml_view_4d(n_kv, n_head_kv, head_v_eff, ns) when v_trans. The two cases
// differ in which stride is the largest, so read the layout rather than
// assume it (llama-context.cpp:3526 uses the same test).
int64_t v_nkv_dim(const ggml_tensor * t) {
    return t->nb[1] > t->nb[2] ? t->ne[0] : t->ne[2];
}

// The generic mask a node consumes, or nullptr. Only the shape is read here:
// the callback fires before the consuming kernel runs and another thread may
// be reading the mask buffer at that moment.
ggml_tensor * mask_of_node(ggml_tensor * t) {
    if (t == nullptr) {
        return nullptr;
    }
    int slot = -1;
    switch (t->op) {
        case GGML_OP_SOFT_MAX:              slot = 1; break;
        case GGML_OP_FLASH_ATTN_EXT:        slot = 3; break;
        case GGML_OP_FLASH_ATTN_EXT_BANDED: slot = 3; break;
        case GGML_OP_MUL_MAT:
            if (t->flags & (GGML_TENSOR_FLAG_ATTN_MASK | GGML_TENSOR_FLAG_ATTN_SCORES)) {
                slot = 2;
            }
            break;
        default: break;
    }
    if (slot < 0) {
        return nullptr;
    }
    ggml_tensor * m = t->src[slot];
    if (m != nullptr && std::strcmp(ggml_get_name(m), "attn_inp_kq_mask") == 0) {
        return m;
    }
    return nullptr;
}

bool capture_shape(ggml_tensor * t, bool ask, void * user_data) {
    const char * name = ggml_get_name(t);
    ggml_tensor * mask = mask_of_node(t);
    // the pad tail is observed on the node that carries the flag, not on the
    // mask: the producer writes x->op_params[2] (llama-context.cpp:4723)
    const bool has_pad_tail = (t->flags & GGML_TENSOR_FLAG_REDUCE_PAD_TAIL) != 0;
    const bool wanted =
        mask != nullptr ||
        has_pad_tail ||
        std::strcmp(name, "union_nkv_k") == 0 ||
        std::strcmp(name, "union_nkv_v") == 0;

    if (ask) {
        return wanted;
    }
    if (!wanted) {
        return true;
    }

    auto & out = *static_cast<shape_capture *>(user_data);

    if (has_pad_tail) {
        out.pad_tail = t->op_params[2];
    }

    if (mask != nullptr) {
        out.seen_mask = true;
        out.mask_ne0  = mask->ne[0];
        out.mask_ne1  = mask->ne[1];
        out.mask_type = mask->type;
        out.mask      = mask;
    } else if (std::strcmp(name, "union_nkv_k") == 0) {
        // get_k: ggml_view_4d(head_k_eff, n_head_kv, n_kv, ns)
        out.seen_k = true;
        out.k_nkv  = t->ne[2];
    } else if (std::strcmp(name, "union_nkv_v") == 0) {
        out.seen_v = true;
        out.v_nkv  = v_nkv_dim(t);
    }

    return true;
}

// The pad-tail field as the log prints it: pad_tail=<n> when a node carried
// GGML_TENSOR_FLAG_REDUCE_PAD_TAIL, pad_tail=none when none did. Same token
// as tests/test-nkv-exact-width.cpp, so one grep reads both logs.
const char * pad_tail_text(const shape_capture & cap) {
    static char buf[16];
    if (cap.pad_tail < 0) {
        return "none";
    }
    std::snprintf(buf, sizeof(buf), "%d", (int) cap.pad_tail);
    return buf;
}

// Mask properties of the row the step's last token reads, checked AFTER
// llama_decode returned so no other thread can be inside the consuming
// kernel. Same two properties as mask_row_props_ok in
// tests/test-nkv-exact-width.cpp:
//
//   * every column j < width holds exactly 0.0f or -INFINITY;
//   * at least one column is visible.
//
// Which column is visible is not asserted: the physical cell index is not
// exposed by the public API. Columns >= width are not read: the fill loop
// never writes them.
bool mask_row_ok(const shape_capture & cap, int width, int64_t row) {
    if (cap.mask == nullptr || cap.mask->data == nullptr) {
        std::fprintf(stderr, "FAIL: mask row not readable (row=%lld)\n",
                     (long long) row);
        return false;
    }
    if (cap.mask_type != GGML_TYPE_F32) {
        std::fprintf(stderr, "FAIL: mask type=%d, want F32 (%d)\n",
                     int(cap.mask_type), int(GGML_TYPE_F32));
        return false;
    }
    if (width > cap.mask_ne0) {
        std::fprintf(stderr, "FAIL: width=%d exceeds mask ne0=%lld\n",
                     width, (long long) cap.mask_ne0);
        return false;
    }
    if (row < 0 || row >= cap.mask_ne1*cap.mask->ne[3]) {
        std::fprintf(stderr, "FAIL: mask row %lld outside [0, %lld)\n",
                     (long long) row, (long long) (cap.mask_ne1*cap.mask->ne[3]));
        return false;
    }
    const float * data = (const float *) cap.mask->data;
    const float * r    = data + row*cap.mask_ne0;
    int visible = 0;
    for (int j = 0; j < width; ++j) {
        const bool keep = r[j] == 0.0f;
        const bool drop = r[j] == -INFINITY;
        if (!keep && !drop) {
            std::fprintf(stderr,
                         "FAIL: mask column %d is neither 0 nor -inf (%f)\n",
                         j, (double) r[j]);
            return false;
        }
        if (keep) {
            ++visible;
        }
    }
    if (visible == 0) {
        std::fprintf(stderr,
                     "FAIL: no visible column at width %d "
                     "(at least one column must be visible)\n", width);
        return false;
    }
    return true;
}



struct step_spec {
    int seq;
    int n_tokens;
};

struct mode_spec {
    const char * name;
    uint32_t n_ctx;
    uint32_t n_ctx_kv;
    uint32_t n_seq_max;
    bool kv_unified;
    std::vector<step_spec> steps;
};

uint32_t g_step = 0;  // successful records written

bool write_record(FILE * out, const float * logits, uint32_t vocab,
                  uint32_t n_tokens, uint32_t pos_after, uint32_t seq,
                  uint32_t n_outputs) {
    const uint32_t hdr[6] = { n_tokens, vocab, pos_after, g_step, n_outputs, seq };
    if (std::fwrite(hdr, sizeof(hdr), 1, out) != 1) {
        std::perror("fwrite header");
        return false;
    }
    if (n_outputs != 0) {
        const size_t n = static_cast<size_t>(n_outputs) * vocab;
        if (std::fwrite(logits, sizeof(float), n, out) != n) {
            std::perror("fwrite logits");
            return false;
        }
    }
    return true;
}

llama_token token_for(int pos, int vocab) {
    if (vocab > 3) {
        return static_cast<llama_token>(3 + (pos * 7 + 1) % (vocab - 3));
    }
    return 0;
}

mode_spec make_mode(const std::string & name) {
    mode_spec spec;
    spec.n_ctx     = 0;
    spec.n_ctx_kv  = 0;
    spec.n_seq_max = 1;
    spec.kv_unified = true;

    if (name == "nkv") {
        spec.name      = "nkv";
        spec.n_ctx     = 512;
        spec.n_ctx_kv  = 0;
        spec.n_seq_max = 1;
        spec.kv_unified = true;
        for (int i = 0; i < 512; ++i) {
            spec.steps.push_back({ 0, 1 });
        }
    } else if (name == "cap") {
        spec.name      = "cap";
        spec.n_ctx     = 300;
        spec.n_ctx_kv  = 300;
        spec.n_seq_max = 1;
        spec.kv_unified = true;
        for (int i = 0; i < 300; ++i) {
            spec.steps.push_back({ 0, 1 });
        }
    } else if (name == "seq") {
        spec.name      = "seq";
        spec.n_ctx     = 1001;
        spec.n_ctx_kv  = 0;
        spec.n_seq_max = 2;
        spec.kv_unified = false;
        for (int i = 0; i < 501; ++i) {
            spec.steps.push_back({ 0, 1 });
        }
    } else {
        spec.name = nullptr;
    }
    return spec;
}

bool run_plan(llama_model * model, FILE * out, int vocab, const mode_spec & spec) {
    shape_capture cap;
    auto cp = llama_context_default_params();
    cp.n_ctx           = spec.n_ctx;
    cp.n_ctx_kv        = spec.n_ctx_kv;
    cp.n_batch         = spec.n_seq_max;
    cp.n_ubatch        = spec.n_seq_max;
    cp.n_seq_max       = spec.n_seq_max;
    cp.kv_unified      = spec.kv_unified;
    cp.n_outputs_max   = spec.n_seq_max;
    cp.n_threads       = 1;
    cp.n_threads_batch = 1;
    cp.offload_kqv     = false;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.type_k          = GGML_TYPE_F32;
    cp.type_v          = GGML_TYPE_F32;
    cp.no_perf         = false;
    cp.op_offload      = false;
    // the eval callback observes the executed n_kv roots; it never changes
    // what the graph computes
    cp.cb_eval           = capture_shape;
    cp.cb_eval_user_data = &cap;
    // no samplers: the n_outputs envelope requires an empty sampler chain
    llama_context * ctx = llama_init_from_model(model, cp);
    if (ctx == nullptr) {
        std::fprintf(stderr, "CONTEXT_FAIL MODE=%s\n", spec.name);
        return false;
    }
    llama_memory_t mem = llama_get_memory(ctx);
    if (mem == nullptr) {
        std::fprintf(stderr, "NO_MEMORY MODE=%s\n", spec.name);
        llama_free(ctx);
        return false;
    }

    uint32_t total_tokens = 0;
    for (const step_spec & st : spec.steps) {
        total_tokens += (uint32_t) st.n_tokens;
    }
    std::fprintf(stderr,
                 "MODE_SPEC MODE=%s N_CTX=%u N_CTX_KV=%u N_SEQ_MAX=%u KV_UNIFIED=%d "
                 "STEPS=%zu TOKENS=%u\n",
                 spec.name, spec.n_ctx, spec.n_ctx_kv, spec.n_seq_max,
                 spec.kv_unified ? 1 : 0, spec.steps.size(), total_tokens);
    std::fprintf(stderr,
                 "CTX_API MODE=%s N_CTX=%u N_CTX_SEQ=%u N_CTX_KV=%u\n",
                 spec.name, llama_n_ctx(ctx), llama_n_ctx_seq(ctx), llama_n_ctx_kv(ctx));
    std::fprintf(stderr,
                 "RUN_BEGIN MODE=%s N_CTX=%u N_CTX_KV=%u N_SEQ_MAX=%u KV_UNIFIED=%d "
                 "N_OUTPUTS_MAX=%u\n",
                 spec.name, spec.n_ctx, spec.n_ctx_kv, spec.n_seq_max,
                 spec.kv_unified ? 1 : 0, cp.n_outputs_max);

    bool ok = true;
    uint32_t decode_calls = 0;
    uint32_t anomalies    = 0;

    for (size_t i = 0; i < spec.steps.size(); ++i) {
        const step_spec & st = spec.steps[i];
        const int pos_before = llama_memory_seq_pos_max(mem, st.seq);
        const int start      = pos_before + 1;
        const int expect_pos = start + st.n_tokens - 1;

        cap.reset();

        llama_batch batch = llama_batch_init(st.n_tokens, 0, 1);
        if (!llama_batch_is_complete(&batch, st.n_tokens, false, true)) {
            std::fprintf(stderr, "BATCH_ALLOC_FAIL MODE=%s STEP=%u\n",
                         spec.name, (uint32_t) i);
            llama_batch_free(batch);
            batch = llama_batch{};
            ok = false;
            break;
        }
        for (int j = 0; j < st.n_tokens; ++j) {
            batch.token[j]     = token_for(start + j, vocab);
            batch.pos[j]       = start + j;
            batch.n_seq_id[j]  = 1;
            batch.seq_id[j][0] = st.seq;
            // record the LAST token: its attention span covers the full step
            // and therefore the exact n_kv width under test.
            batch.logits[j]    = (j == st.n_tokens - 1) ? 1 : 0;
        }
        batch.n_tokens = st.n_tokens;

        const llama_perf_context_data before = llama_perf_context(ctx);
        const int rc = llama_decode(ctx, batch);
        const llama_perf_context_data after = llama_perf_context(ctx);
        llama_batch_free(batch);

        const int32_t d_reused = after.n_reused - before.n_reused;
        const int pos_after    = llama_memory_seq_pos_max(mem, st.seq);
        const int built        = (rc == 0 && d_reused == 0) ? 1 : 0;
        if (rc == 0) {
            ++decode_calls;
            if (d_reused < 0 || d_reused > 1) {
                ++anomalies;
                std::fprintf(stderr,
                             "UBATCH_ANOMALY MODE=%s STEP=%u d_reused=%d\n",
                             spec.name, (uint32_t) i, d_reused);
            }
            if (pos_after != expect_pos) {
                ++anomalies;
                std::fprintf(stderr,
                             "POS_ANOMALY MODE=%s STEP=%u SEQ=%d POS_AFTER=%d EXPECT=%d\n",
                             spec.name, (uint32_t) i, st.seq, pos_after, expect_pos);
            }
        }

        std::fprintf(stderr,
                     "STEP=%u ATTEMPT=%u MODE=%s SEQ=%d NTOK=%d POS_BEFORE=%d "
                     "POS_AFTER=%d NOUT=1 RC=%d BUILD=%d BUILD_TOTAL=%u REUSED_TOTAL=%d\n",
                     (uint32_t) i, (uint32_t) i, spec.name, st.seq, st.n_tokens,
                     pos_before, pos_after, rc, built, decode_calls - (uint32_t) after.n_reused,
                     after.n_reused);

        if (rc != 0) {
            std::fprintf(stderr, "DECODE_FAIL MODE=%s STEP=%u RC=%d\n",
                         spec.name, (uint32_t) i, rc);
            ok = false;
            break;
        }
        if (pos_after != expect_pos) {
            ok = false;
            break;
        }

        // The three n_kv runtime roots must be observed on every step, and
        // their n_kv dimension must be the 1-based position after the step.
        // The mask is ne[0] (llama-graph.cpp:109), the K view ne[2]
        // (llama-kv-cache.cpp:2177) and the V view the layout-dependent
        // dimension v_nkv_dim() reads.
        if (!cap.seen_mask || !cap.seen_k || !cap.seen_v) {
            std::fprintf(stderr,
                         "FAIL: missing tensor at step=%u mask=%d k=%d v=%d "
                         "(the shape callback must observe attn_inp_kq_mask, "
                         "union_nkv_k and union_nkv_v)\n",
                         (uint32_t) i, int(cap.seen_mask), int(cap.seen_k),
                         int(cap.seen_v));
            ok = false;
            break;
        }
        std::fprintf(stderr,
                     "NKV_EXACT MODE=%s step=%u pos=%d mask_ne0=%lld k_ne2=%lld "
                     "v_nkv=%lld pad_tail=%s\n",
                     spec.name, (uint32_t) i, pos_after,
                     (long long) cap.mask_ne0, (long long) cap.k_nkv,
                     (long long) cap.v_nkv, pad_tail_text(cap));
        if (cap.mask_ne0 != expect_pos || cap.k_nkv != expect_pos ||
                cap.v_nkv != expect_pos) {
            std::fprintf(stderr,
                         "FAIL: executed width at step=%u pos=%d want=%d "
                         "mask=%lld k=%lld v=%lld\n",
                         (uint32_t) i, pos_after, expect_pos,
                         (long long) cap.mask_ne0, (long long) cap.k_nkv,
                         (long long) cap.v_nkv);
            ok = false;
            break;
        }

        // The mask row this step's last token reads, read AFTER the decode so
        // no other thread is inside the consuming kernel. One token per step
        // and one stream per mode, so the row is the last row of the mask:
        // row = ne[1]*ne[3] - 1, which is 0 for a one-token step.
        {
            const int64_t row = cap.mask_ne1*cap.mask->ne[3] - 1;
            const bool row_ok = mask_row_ok(cap, expect_pos, row);
            int visible = 0;
            if (row_ok) {
                const float * data = (const float *) cap.mask->data;
                const float * r    = data + row*cap.mask_ne0;
                for (int j = 0; j < expect_pos; ++j) {
                    if (r[j] == 0.0f) {
                        ++visible;
                    }
                }
            }
            std::fprintf(stderr,
                         "MASK_ROW MODE=%s step=%u width=%d row=%lld visible=%d ok=%d\n",
                         spec.name, (uint32_t) i, expect_pos, (long long) row,
                         visible, int(row_ok));
            if (!row_ok) {
                ok = false;
                break;
            }
        }

        const float * logits = llama_get_logits_ith(ctx, st.n_tokens - 1);
        if (logits == nullptr) {
            std::fprintf(stderr, "MISSING_LOGITS MODE=%s STEP=%u IDX=%d\n",
                         spec.name, (uint32_t) i, st.n_tokens - 1);
            ok = false;
            break;
        }
        if (!write_record(out, logits, (uint32_t) vocab, (uint32_t) st.n_tokens,
                          (uint32_t) pos_after, (uint32_t) st.seq, 1)) {
            ok = false;
            break;
        }
        ++g_step;
    }

    const llama_perf_context_data p = llama_perf_context(ctx);
    const int32_t builds = (int32_t) decode_calls - p.n_reused;
    std::fprintf(stderr,
                 "RUN_PERF MODE=%s DECODE_CALLS=%u N_REUSED=%d BUILDS=%d "
                 "REBUILDS_AFTER_FIRST=%d ANOMALIES=%u\n",
                 spec.name, decode_calls, p.n_reused, builds,
                 builds > 1 ? builds - 1 : 0, anomalies);

    // NKV_BINDER_STATS is printed by llama_free(); bracket it so the raw log
    // keeps an unambiguous section even when other INFO lines interleave.
    std::fprintf(stderr, "RUN_STATS_BEGIN MODE=%s\n", spec.name);
    std::fflush(stderr);
    llama_free(ctx);
    std::fprintf(stderr, "RUN_STATS_END MODE=%s\n", spec.name);
    std::fprintf(stderr, "RUN_END MODE=%s OK=%d\n", spec.name, ok ? 1 : 0);
    return ok;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s MODEL.gguf LOGITS.bin nkv|cap|seq\n", argv[0]);
        return 2;
    }
    const std::string mode = argv[3];
    const mode_spec spec = make_mode(mode);
    if (spec.name == nullptr) {
        std::fprintf(stderr, "unknown mode: %s\n", mode.c_str());
        return 2;
    }

    FILE * out = std::fopen(argv[2], "wb");
    if (out == nullptr) {
        std::perror("fopen");
        return 2;
    }

    // The arm this process was started with, read once and never written.
    // check_probe.py refuses a log whose binder= value is not the arm the
    // runner asked for, so the three arms cannot silently collapse.
    {
        const arm_env arm = read_arm_env();
        std::fprintf(stderr, "ARM binder=%s ntok=%s reuse_disable=%s\n",
                     arm.binder, arm.ntok, arm.reuse_disable);
    }

    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    ggml_backend_dev_t devices[] = { nullptr };
    mp.devices = devices;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (model == nullptr) {
        std::fprintf(stderr, "MODEL_LOAD_FAIL\n");
        std::fclose(out);
        llama_backend_free();
        return 1;
    }
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    if (vocab <= 0) {
        std::fprintf(stderr, "BAD_VOCAB\n");
        llama_model_free(model);
        std::fclose(out);
        llama_backend_free();
        return 1;
    }
    std::fprintf(stderr, "HARNESS_BEGIN MODE=%s VOCAB=%d\n", spec.name, vocab);

    const bool ok = run_plan(model, out, vocab, spec);

    std::fprintf(stderr, "HARNESS_END OK=%d RECORDS=%u\n", ok ? 1 : 0, g_step);
    llama_model_free(model);
    llama_backend_free();
    if (std::fclose(out) != 0) {
        std::perror("fclose");
        return 1;
    }
    return ok ? 0 : 1;
}
