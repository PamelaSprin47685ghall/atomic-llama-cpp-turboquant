#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"

#include "tp5-router-scratch-diag.h"

// The real-graph identity of the executed decode graph is not reachable from
// this example: the graph lives in llama_context (src/llama-context.h, struct
// not in llama.h), and get_gf_res_reserve() returns the reserve-sized graph, not
// the executed one. The join is made by the owner instead: with
// GGML_TP5_MOE_ROUTER_SCRATCH=1 and GGML_TP5_ROUTER_SCRATCH_AUDIT=1,
// llama_context::graph_compute prints `router_scratch plan identity ... match=`
// on stderr and the allocator prints `retention audit ... graph_uid=... plan_id=...`
// on the same stream. This tool only reports that it made no such join itself
// (see the VERIFY_REAL line below); the driver grades the two owner lines.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Real-consumer CPU verification of the TP5 MoE router scratch retention
// (GGML_TP5_MOE_ROUTER_SCRATCH). Nothing here is a stand-in for the model: the
// model leg decodes a real Qwen4Exp checkpoint through llama_decode (the real
// graph builder, scheduler, allocator and ops), and the plan leg rebuilds one
// layer's real MoE node sequence around the production gallocr.
//
// The tool is report-only: it prints evidence and exits non-zero only when a
// decode fails. Interpreting the lines is the caller's job.
//
// Two legs, same binary (the environment selects the behaviour):
//   L1 model leg   one llama_decode per row count (1, 2, 3, 4 ~ the retention
//                  window and its N1 exclusion), every logits row hashed. OFF
//                  vs ON must be bit-identical; sha lines are directly
//                  comparable across trees and runs.
//   L2 plan leg    one trunk layer's MoE, exactly the nodes build_layer_ffn
//                  emits (router MUL_MAT [512, rows] -> SOFT_MAX -> TOP_K ->
//                  MUL_MAT_ID + shared expert), with the retained router
//                  marked NO_REUSE. Reserves a real gallocr plan and audits
//                  that every *other* allocation in that plan lies outside the
//                  retained extent, then computes the graph and hashes the
//                  output. Without the mark, computes the same graph.
//
// The L1 leg needs a llama context over the fixture. That path is currently
// blocked by a production defect that is outside the router candidate:
// llm_graph_input_mem_hybrid::set_input (llama-graph.cpp:1129-1131) feeds the
// hybrid ATTENTION kv-cache context (llama-memory.h:143-152) to
// llama_kv_cache::set_input_k_idxs, whose GGML_ASSERT(buffer) aborts because a
// qwen4exp decode graph has no attention layer (both trunk blocks are recurrent
// and the NextN block is outside the trunk graph). --no-model skips that leg and
// reports it as blocked instead of pretending the decode ran.
//
// Usage:
//   llama-tp5-router-verify -m fixture.gguf [--rows 1,2,3,4] [--ctx 32]
//                           [--threads 1] [--plan-rows 3] [--no-model]

namespace {

constexpr int64_t N_EXPERT      = 512; // production router width
constexpr int64_t N_EXPERT_USED = 10;

uint64_t fnv1a64(const void * data, size_t len) {
    const uint8_t * p = (const uint8_t *) data;
    uint64_t hash = 1469598103934665603ull;
    for (size_t i = 0; i < len; i++) {
        hash ^= p[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

llama_batch make_batch(const std::vector<llama_token> & toks, llama_pos pos0) {
    llama_batch batch = llama_batch_init((int32_t) toks.size(), 0, 1);
    for (int32_t i = 0; i < (int32_t) toks.size(); i++) {
        batch.token[i]     = toks[i];
        batch.pos[i]       = pos0 + i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = 1;
    }
    batch.n_tokens = (int32_t) toks.size();
    return batch;
}

int64_t meta_u32(const llama_model * model, const char * key, int64_t fallback) {
    char buf[32] = {0};
    if (llama_model_meta_val_str(model, key, buf, sizeof(buf)) < 0) {
        return fallback;
    }
    return strtoll(buf, nullptr, 10);
}

// One trunk layer's MoE, as build_layer_ffn emits it for a 512/top-10 router.
// NOTE: this is a SYNTHETIC graph. It mirrors build_moe_ffn's node and consumer set
// (ids straight from top_k, same join shape), but it is not the model's graph, so a
// clean audit here is supporting evidence only. The executed decode graph is not
// reachable from here (see the VERIFY_REAL line), so nothing in this file can
// promote retention on its own; that join is printed by llama_context.
// The expert reduction mirrors build_moe_ffn (permute -> sum_rows -> 2-D), so the
// join with the shared branch is 2-D + 2-D and the output is the real [n_ff, rows].
struct plan_graph {
    ggml_context *   ctx     = nullptr;
    ggml_cgraph *    graph   = nullptr;
    int64_t          rows    = 0;
    ggml_tensor *    logits  = nullptr; // router projection [512, rows]
    ggml_tensor *    out     = nullptr;
    ggml_tensor *    x       = nullptr;
    ggml_tensor *    w_gate  = nullptr;
    ggml_tensor *    w_exps  = nullptr;
    ggml_tensor *    w_shexp = nullptr;
    std::vector<ggml_tensor *> nodes;
};

plan_graph build_plan_graph(int64_t rows, int64_t n_embd, int64_t n_ff_exp, bool retain) {
    plan_graph pg;

    ggml_init_params ip = {};
    ip.mem_size   = 64ull * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    pg.ctx = ggml_init(ip);

    ggml_context * ctx = pg.ctx;

    // production MoE input layout: build_moe_ffn reshapes to [n_embd, 1, n_tokens]
    // (llama-graph.cpp:2203) before MUL_MAT_ID, whose ids are [n_used, n_tokens]
    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, 1, rows);
    ggml_set_name(x, "moe_input");
    ggml_set_input(x);

    ggml_tensor * w_gate = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, N_EXPERT);
    ggml_set_name(w_gate, "ffn_gate_inp");
    ggml_set_input(w_gate);

    ggml_tensor * w_gate_exps = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, n_ff_exp, N_EXPERT);
    ggml_set_name(w_gate_exps, "ffn_gate_exps");
    ggml_set_input(w_gate_exps);

    ggml_tensor * w_gate_shexp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_ff_exp);
    ggml_set_name(w_gate_shexp, "ffn_gate_shexp");
    ggml_set_input(w_gate_shexp);

    // router projection: the tensor the model marks GGML_TENSOR_FLAG_NO_REUSE.
    // ggml_mul_mat wants b 2-D, so the 3-D mixtral layout is flattened exactly
    // the way build_moe_ffn does it (rows * 1 columns collapse)
    ggml_tensor * x_2d   = ggml_reshape_2d(ctx, x, n_embd, rows);
    ggml_tensor * logits = ggml_mul_mat(ctx, w_gate, x_2d);
    ggml_set_name(logits, "ffn_moe_logits");
    if (retain) {
        logits->flags |= GGML_TENSOR_FLAG_NO_REUSE;
    }

    ggml_tensor * probs   = ggml_soft_max(ctx, logits);
    // build_moe_ffn feeds ggml_top_k's tensor directly to MUL_MAT_ID: no argsort,
    // no view, no unused node. The plan must have the same consumer set as the
    // real trunk layer or the audit would describe a different graph.
    ggml_tensor * ids     = ggml_top_k(ctx, probs, N_EXPERT_USED);              // [n_used, rows], a plain tensor
    ggml_set_name(ids, "ffn_moe_topk");

    // Routed experts keep the production shape [n_ff, n_used, rows]; build_moe_ffn
    // reduces them with sum_rows over the expert axis after a permute, giving
    // [1, n_ff, rows] -> [n_ff, rows]. Both sides of the join are then 2-D
    // [n_ff, rows], so ggml_can_repeat(shared, moe_out) holds; adding the 3-D
    // routed tensor and the 2-D shared one (the previous code) cannot repeat
    // because ne[1] = 10 vs 1 in the wrong order.
    ggml_tensor * routed = ggml_mul_mat_id(ctx, w_gate_exps, x, ids);           // [n_ff, 10, rows]
    ggml_tensor * routed_pe = ggml_cont(ctx, ggml_permute(ctx, routed, 1, 0, 2, 3));
    ggml_tensor * summed    = ggml_sum_rows(ctx, routed_pe);                     // [1, n_ff, rows]
    ggml_tensor * moe_out   = ggml_reshape_2d(ctx, summed, n_ff_exp, rows);      // [n_ff, rows]

    ggml_tensor * shared = ggml_mul_mat(ctx, w_gate_shexp, x_2d);                // [n_ff, rows]
    ggml_tensor * out    = ggml_add(ctx, moe_out, shared);                       // [n_ff, rows]
    ggml_set_name(out, "ffn_moe_out");

    pg.rows   = rows;
    pg.logits = logits;
    pg.out    = out;
    pg.x      = x;
    pg.w_gate  = w_gate;
    pg.w_exps  = w_gate_exps;
    pg.w_shexp = w_gate_shexp;

    pg.nodes = { logits, probs, ids, routed, shared, out };

    pg.graph = ggml_new_graph_custom(ctx, 256, false);
    ggml_build_forward_expand(pg.graph, out);
    return pg;
}

void fill_plan_inputs(const plan_graph & pg) {
    // deterministic F32 data: the plan does not depend on the values, and the
    // compute leg just needs non-degenerate router logits. Sizes come from
    // ggml_nelements, so the 3-D [n_embd, 1, rows] input is filled completely.
    auto fill = [](ggml_tensor * t, float scale, float bias) {
        std::vector<float> buf((size_t) ggml_nelements(t));
        for (size_t i = 0; i < buf.size(); ++i) {
            const float phase = (float) ((i * 2654435761ull) % 1000) / 1000.0f;
            buf[i] = bias + scale * (phase - 0.5f);
        }
        ggml_backend_tensor_set(t, buf.data(), 0, buf.size() * sizeof(float));
    };
    fill(pg.x,      1.0f,  0.0f);
    fill(pg.w_gate, 0.08f, 0.0f);
    fill(pg.w_exps, 0.2f,  0.0f);
}

struct plan_report {
    bool  reserved     = false;
    bool  materialized = false;
    bool  reuses_retained = false;
    bool  audit_clean  = false;
    bool  overlap      = false;
    long  retained_size = 0;
    long  other_overlaps = 0;
    long  n_allocs     = 0;
    uint64_t out_hash  = 0;
    bool  computed     = false;
};

plan_report run_plan_leg(const plan_graph & pg, bool retain) {
    plan_report rep;

    // public registry initialisation; examples do not include ggml-cpu.h
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (backend == nullptr) {
        printf("VERIFY_PLAN note no CPU backend\n");
        return rep;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
    ggml_gallocr_t galloc = ggml_gallocr_new(buft);

    // ---- 1) measure the plan with the real API, without allocating ----
    // ggml_gallocr_reserve_n_size runs reserve_n_impl(no_alloc=true) and reports the
    // exact size per galloc buffer. That size covers nodes AND leafs, so no manual
    // padding or 64-byte assumption is needed (and none is used below).
    size_t plan_size = 0;
    ggml_gallocr_reserve_n_size(galloc, pg.graph, nullptr, nullptr, &plan_size);
    printf("VERIFY_PLAN plan_size=%zu bytes (nodes+leafs, measured with reserve_n_size)\n", plan_size);
    if (plan_size == 0) {
        printf("VERIFY_PLAN reserved=0 (size measurement failed)\n");
        ggml_gallocr_free(galloc);
        ggml_backend_free(backend);
        return rep;
    }

    // ---- 2) reserve for real ----
    // gallocr owns its buffers (ggml_gallocr_new takes a buffer *type* only): the
    // plan and every leaf live in galloc->buffers[0], created by this call. The
    // measurement above is therefore informational (it proves the graph fits a
    // single buffer and reports its size); nothing here assumes a second buffer or
    // stacks inputs after the plan.
    rep.reserved = ggml_gallocr_reserve_n(galloc, pg.graph, nullptr, nullptr);
    if (!rep.reserved) {
        printf("VERIFY_PLAN reserved=0 (reserve_n failed)\n");
        ggml_gallocr_free(galloc);
        ggml_backend_free(backend);
        return rep;
    }

    // one placement pass for nodes and leafs; view chains resolve because gallocr
    // allocates the whole graph in a single call
    if (!ggml_gallocr_alloc_graph(galloc, pg.graph)) {
        printf("VERIFY_PLAN allocated=0 (alloc_graph failed)\n");
        ggml_gallocr_free(galloc);
        ggml_backend_free(backend);
        return rep;
    }

    // ---- 3) audit the real graph, not a hand-written node list ----
    // The retained tensor must exist, own storage, and the graph's own nodes must
    // not share that extent. Any sharing at all -- identical start or partial
    // overlap -- violates NO_REUSE and fails the leg. The allocator's own audit
    // (ggml_gallocr_check_no_reuse_plan, reserve_n/GGML_TP5_ROUTER_SCRATCH_AUDIT=1)
    // is the fail-closed backstop; this is the physical check on the same plan.
    rep.retained_size = (long) ggml_nbytes(pg.logits);
    rep.materialized  = pg.logits->data != nullptr && pg.logits->view_src == nullptr;
    if (!rep.materialized) {
        printf("VERIFY_PLAN note retained tensor has no storage (view_src=%d data=%p): retention not materialized\n",
               pg.logits->view_src != nullptr ? 1 : 0, pg.logits->data);
        ggml_gallocr_free(galloc);
        ggml_backend_free(backend);
        return rep;
    }

    const uint8_t * lo = (const uint8_t *) pg.logits->data;
    const uint8_t * hi = lo + ggml_nbytes(pg.logits);
    printf("VERIFY_PLAN retain=%d retained_ptr=%p bytes=%ld no_reuse=%d buffer=%s nodes=%d\n",
           retain ? 1 : 0, pg.logits->data, rep.retained_size,
           (pg.logits->flags & GGML_TENSOR_FLAG_NO_REUSE) ? 1 : 0,
           pg.logits->buffer ? "own" : "none", ggml_graph_n_nodes(pg.graph));

    for (int i = 0; i < ggml_graph_n_nodes(pg.graph); ++i) {
        ggml_tensor * node = ggml_graph_node(pg.graph, i);
        if (node == pg.logits || node->view_src != nullptr || node->data == nullptr) {
            continue;
        }
        rep.n_allocs++;
        const uint8_t * a = (const uint8_t *) node->data;
        const uint8_t * b = a + ggml_nbytes(node);
        if (a < hi && lo < b) {
            printf("VERIFY_PLAN alloc '%s' op=%s [%p,+%zu] %s retained block\n",
                   node->name, ggml_op_name(node->op), node->data, ggml_nbytes(node),
                   a == lo ? "REUSES" : "OVERLAPS");
            rep.reuses_retained = true;
            rep.overlap = true; // NO_REUSE forbids sharing with ANY other allocation
            rep.other_overlaps++;
        }
    }
    // a retained tensor that some non-view node shares storage with is a failure,
    // whether the share is exact (in-place continuation) or partial
    rep.audit_clean = !rep.overlap && pg.logits->view_src == nullptr;

    // ---- 4) compute the same graph on the plan ----
    fill_plan_inputs(pg);
    rep.computed = ggml_backend_graph_compute(backend, pg.graph) == GGML_STATUS_SUCCESS;
    if (rep.computed && pg.out->data != nullptr) {
        rep.out_hash = fnv1a64(pg.out->data, ggml_nbytes(pg.out));
    }

    printf("VERIFY_PLAN retain=%d rows=%lld reserved=%d materialized=%d audit_clean=%d "
           "overlaps=%ld reuses_retained=%d retained_bytes=%ld other_allocs=%ld computed=%d "
           "sha=%016llx\n",
           retain ? 1 : 0, (long long) pg.rows,
           rep.reserved ? 1 : 0, rep.materialized ? 1 : 0, rep.audit_clean ? 1 : 0,
           rep.other_overlaps, rep.reuses_retained ? 1 : 0,
           rep.retained_size, rep.n_allocs, rep.computed ? 1 : 0,
           (unsigned long long) rep.out_hash);

    ggml_gallocr_free(galloc);
    ggml_backend_free(backend);
    return rep;
}

} // namespace

int main(int argc, char ** argv) {
    std::string model_path;
    std::string rows_arg = "1,2,3,4";
    int         n_ctx     = 32;
    int         n_threads = 1;
    int64_t     plan_rows = 3;
    bool        no_model  = false;

    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        const bool has_next = i + 1 < argc;
        if (arg == "-m" && has_next) {
            model_path = argv[++i];
        } else if (arg == "--rows" && has_next) {
            rows_arg = argv[++i];
        } else if (arg == "--ctx" && has_next) {
            n_ctx = atoi(argv[++i]);
        } else if (arg == "--threads" && has_next) {
            n_threads = atoi(argv[++i]);
        } else if (arg == "--plan-rows" && has_next) {
            plan_rows = strtoll(argv[++i], nullptr, 10);
        } else if (arg == "--no-model") {
            no_model = true;
        } else {
            fprintf(stderr, "usage: %s -m model.gguf [--rows 1,2,3,4] [--ctx 32] "
                            "[--threads 1] [--plan-rows 3] [--no-model]\n", argv[0]);
            return 1;
        }
    }
    if (model_path.empty()) {
        fprintf(stderr, "usage: %s -m model.gguf [--rows 1,2,3,4] [--ctx 32] "
                        "[--threads 1] [--plan-rows 3] [--no-model]\n", argv[0]);
        return 1;
    }

    std::vector<int> rows;
    {
        const char * p = rows_arg.c_str();
        while (*p) {
            const long v = strtol(p, (char **) &p, 10);
            rows.push_back((int) v);
            if (*p == ',') {
                p++;
            } else if (*p != '\0') {
                fprintf(stderr, "%s: bad --rows '%s'\n", argv[0], rows_arg.c_str());
                return 1;
            }
        }
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "%s: failed to load '%s'\n", argv[0], model_path.c_str());
        llama_backend_free();
        return 1;
    }

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);
    const int64_t       n_embd  = meta_u32(model, "qwen4exp.embedding_length", 8);
    const int64_t       n_ff_ex = meta_u32(model, "qwen4exp.expert_feed_forward_length", 2);
    const int64_t       n_exp   = meta_u32(model, "qwen4exp.expert_count", 0);
    const int64_t       n_used  = meta_u32(model, "qwen4exp.expert_used_count", 0);
    const int64_t       n_blk   = meta_u32(model, "qwen4exp.block_count", 0);

    const char * env_retain = getenv("GGML_TP5_MOE_ROUTER_SCRATCH");
    const char * env_audit  = getenv("GGML_TP5_ROUTER_SCRATCH_AUDIT");
    const char * env_debug  = getenv("GGML_TP5_ROUTER_SCRATCH_DEBUG");

    printf("VERIFY_ENV retain=%s audit=%s debug=%s\n",
           env_retain ? env_retain : "(unset)",
           env_audit  ? env_audit  : "(unset)",
           env_debug  ? env_debug  : "(unset)");
    printf("VERIFY_MODEL n_embd=%lld n_vocab=%d n_block=%lld n_expert=%lld n_expert_used=%lld n_ff_exp=%lld\n",
           (long long) n_embd, n_vocab, (long long) n_blk,
           (long long) n_exp, (long long) n_used, (long long) n_ff_ex);

    if (n_exp != N_EXPERT || n_used != N_EXPERT_USED) {
        printf("VERIFY_NOTE router gate needs n_expert==%lld && n_expert_used==%lld; the model leg "
               "will not reach the retention path\n", (long long) N_EXPERT, (long long) N_EXPERT_USED);
    }

    // ---------------- L1: model leg ----------------
    int rc = 0;
    int max_rows = 0;
    for (int r : rows) {
        max_rows = r > max_rows ? r : max_rows;
    }
    if (max_rows <= 0 || max_rows + 4 > n_ctx) {
        fprintf(stderr, "%s: --rows must be >= 1 and --ctx must leave room for the decode\n", argv[0]);
        rc = 1;
    }

    if (no_model) {
        printf("VERIFY_MODEL_LEG skipped=1 reason=llm_graph_input_mem_hybrid_set_input_assert\n");
    } else {
        llama_context_params cparams = llama_context_default_params();
        // Diagnostic-only observation of the executed graph; enabled by
        // GGML_TP5_ROUTER_SCRATCH_DIAG=1, otherwise a no-op callback.
        tp5_router_scratch_diag diag;
        tp5_router_scratch_diag_init(diag);
        cparams.cb_eval           = tp5_router_scratch_diag_eval;
        cparams.cb_eval_user_data = &diag;
        cparams.n_ctx           = (uint32_t) n_ctx;
        cparams.n_batch         = (uint32_t) n_ctx;
        cparams.n_ubatch        = (uint32_t) n_ctx;
        cparams.n_threads       = n_threads;
        cparams.n_threads_batch = n_threads;
        cparams.offload_kqv     = false;

        llama_context * ctx = llama_init_from_model(model, cparams);
        if (!ctx) {
            fprintf(stderr, "%s: failed to create the context\n", argv[0]);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }

        std::vector<llama_token> tokens;
        for (int i = 0; i < max_rows; ++i) {
            tokens.push_back((llama_token) ((7 * i + 3) % n_vocab));
        }

        for (int r : rows) {
            if (rc != 0) {
                break;
            }
            // single-sequence, logits for every row
            llama_memory_clear(llama_get_memory(ctx), true);
            llama_batch batch = make_batch(std::vector<llama_token>(tokens.begin(), tokens.begin() + r), 0);
            const int dec = (int) llama_decode(ctx, batch);
            llama_batch_free(batch);
            // audit what the scheduler executed in this step
            tp5_router_scratch_diag_end_step(diag, dec);
            if (dec != 0) {
                printf("VERIFY_ROWS rows=%d decode=%d\n", r, dec);
                rc = 1;
                break;
            }
            uint64_t hash = 1469598103934665603ull;
            bool finite = true;
            int top1 = -1;
            double best = -INFINITY;
            for (int i = 0; i < r; ++i) {
                const float * logits = llama_get_logits_ith(ctx, i);
                if (logits == nullptr) {
                    fprintf(stderr, "VERIFY_ROWS rows=%d row=%d no logits\n", r, i);
                    rc = 1;
                    break;
                }
                const uint64_t h = fnv1a64(logits, (size_t) n_vocab * sizeof(float));
                if (i == 0) {
                    hash = h;
                    for (int j = 0; j < n_vocab; ++j) {
                        if (!std::isfinite(logits[j])) {
                            finite = false;
                        }
                        if (logits[j] > best) {
                            best  = logits[j];
                            top1  = j;
                        }
                    }
                }
                printf("VERIFY_ROW rows=%d row=%d sha=%016llx\n", r, i, (unsigned long long) h);
            }
            printf("VERIFY_ROWS rows=%d decode=%d finite=%d top1=%d sha=%016llx\n",
                   r, dec, finite ? 1 : 0, top1, (unsigned long long) hash);
        }

        // ---- real-graph gate: the owner prints it, this tool cannot ----
        // Source check (llama_context ownership):
        //   * the executed graph is gf_res_prev (src/llama-context.cpp, process_ubatch),
        //     rebuilt per ubatch;
        //   * gf_res_reserve is the reserve-sized graph built by graph_reserve
        //     solely to size the scheduler buffers; it is a different
        //     llm_graph_result with its own ubatch (1 output per sequence) and its
        //     own ggml context.
        // Auditing gf_res_reserve here would count marks in a graph that is not run.
        // The join needs the executed graph and the allocator plan at once, which
        // only llama_context has: with both knobs on it prints
        //   `llama_context::graph_compute: router_scratch plan identity ... match=<0|1>`
        // and the allocator prints `retention audit ... graph_uid=... plan_id=... fp=...`
        // on the same stderr. The driver grades those two owner lines (C4b). No
        // public accessor was added for this test, and this line states only that
        // the tool itself makes no claim.
        printf("VERIFY_REAL unavailable=1 reason=owner_side_evidence "
               "join=router_scratch_plan_identity_line "
               "note=reserve_graph_is_sizing_only_not_executed "
               "promotion=graded_by_driver_C4b\n");
        // OBSERVATION ONLY — not promotion evidence. This line comes from the
        // scheduler eval callback, which (a) is unavailable on the tp5_relay
        // path (llama rejects a non-null cb_eval), (b) disables the
        // single-split defer_input_flush fast path and synchronizes per batched
        // range, so it is not the production schedule, and (c) sees only the
        // executed split's nodes, never sched->graph input copies. A DISJOINT
        // count here therefore does not prove plan-wide disjointness and must
        // not be used on its own to pass the retention gate. The production
        // allocator audit (GGML_TP5_ROUTER_SCRATCH_AUDIT=1 + GGML_TP5_MOE_ROUTER_SCRATCH=1,
        // `retention audit clean (reserve): N retained allocations, ..., graph_uid=...,
        // plan_id=...`) is the stronger leg; its plan identity is joined to the
        // submitted graph by the owner line above, which is what C4b grades.
        printf("VERIFY_REAL_DIAG enabled=%d scope=cpu_non_relay_executed_nodes_only "
               "retain_hits=%d retain_steps=%d disjoint=%d overlaps=%d "
               "unresolved=%d n1_shape=%d fire_steps=%d promotion_evidence=0\n",
               diag.enabled ? 1 : 0, diag.retains, diag.retain_steps, diag.disjoint,
               diag.overlaps, diag.unresolved, diag.retain_n1, diag.fired_steps);

        tp5_router_scratch_diag_summary(diag);
        tp5_router_scratch_diag_close(diag);
        llama_free(ctx);
    } // !no_model

    // ---------------- L2: plan leg ----------------
    for (bool retain : { false, true }) {
        plan_graph pg = build_plan_graph(plan_rows, n_embd, n_ff_ex, retain);
        const plan_report rep = run_plan_leg(pg, retain);
        printf("VERIFY_PLAN_SUMMARY retain=%d rows=%lld reserved=%d materialized=%d "
               "audit_clean=%d overlaps=%ld reuses_retained=%d retained_bytes=%ld "
               "other_allocs=%ld computed=%d sha=%016llx\n",
               retain ? 1 : 0, (long long) plan_rows, rep.reserved ? 1 : 0,
               rep.materialized ? 1 : 0, rep.audit_clean ? 1 : 0, rep.other_overlaps,
               rep.reuses_retained ? 1 : 0, rep.retained_size,
               rep.n_allocs, rep.computed ? 1 : 0, (unsigned long long) rep.out_hash);
        ggml_free(pg.ctx);
    }

    llama_model_free(model);
    llama_backend_free();
    return rc;
}
