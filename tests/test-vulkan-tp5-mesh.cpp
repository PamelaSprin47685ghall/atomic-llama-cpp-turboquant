// test-vulkan-tp5-mesh: TP5 mesh AllReduce over N Vulkan GPUs (TP5.md 22.2).
//
// Exercises the production collective transport (ggml-vulkan-collective.cpp)
// through real Vulkan backends — no POC-style private VkDevice creation.
//
// Checks:
//   1. constant inputs: rank i fills its tensor with i+1 -> every element == sum(1..P)
//   2. varying inputs: value = f(rank, element, round) with exact integer
//      representation; CPU reference sums in the same fixed rank order
//   3. multi-round dependency chain: next round's input derives from the
//      previous result (bounded recurrence), catching stale-slot reads
//   4. delayed producer: one rank runs a long dummy kernel first; consumers
//      must still read this round's data (SYNC_FD real-wait proof, 13.8)
//   5. resource accounting: no fd leak across rounds (count /proc/self/fd)
//
// Usage: test-vulkan-tp5-mesh [--devices 0,1,2,3,4] [--elements 2560]
//                              [--rounds 96] [--wire f32|f16] [--sync host|syncfd|timeline|star|l3_star|gpuflag]
//                              [--check-all] [--vary-input] [--delay-producer]
//                              [--adversarial] [--epochs 1] [--chain-stages N]
//                              [--run-chain-regression] [--benchmark-chain]

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-vulkan.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <dirent.h>

// from ggml-vulkan-collective.cpp (registry-resolved)
typedef void * (*comm_init_t)(ggml_backend_t *, size_t);
typedef void   (*comm_free_t)(void *);
typedef bool   (*allreduce_t)(void *, ggml_tensor **);
typedef void   (*tp5_flush_async_t)(ggml_backend_t);
typedef bool (*submit_epoch_chain_t)(void *,
                                     const std::vector<std::vector<std::vector<void *>>> &,
                                     const std::vector<std::vector<ggml_tensor *>> &);
typedef bool (*get_cached_cmd_bufs_t)(ggml_backend_t, ggml_cgraph *, std::vector<void *> &);
typedef bool (*prepare_graph_t)(void *, size_t, ggml_cgraph *, bool);

static int g_failures = 0;
#define TEST_ASSERT(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        g_failures++; \
        return; \
    } \
} while (0)

static size_t count_open_fds() {
    DIR * d = opendir("/proc/self/fd");
    if (!d) return 0;
    size_t n = 0;
    while (readdir(d) != nullptr) ++n;
    closedir(d);
    return n; // includes '.' '..' and the DIR handle itself; stable for deltas
}

static std::vector<ggml_backend_t> g_backends;
static comm_init_t  g_comm_init  = nullptr;
static comm_free_t  g_comm_free  = nullptr;
static allreduce_t  g_allreduce  = nullptr;
static tp5_flush_async_t g_flush_async = nullptr;
static submit_epoch_chain_t               g_submit_epoch_chain  = nullptr;
static get_cached_cmd_bufs_t              g_get_cached_cmd_bufs = nullptr;
static prepare_graph_t                    g_prepare_graph       = nullptr;
static std::vector<ggml_context *> g_allocated_contexts;
static std::vector<ggml_backend_buffer_t> g_allocated_buffers;

static void setup(const std::vector<int> & dev_ids) {
    size_t n_reg = ggml_backend_reg_dev_count(ggml_backend_vk_reg());
    for (int id : dev_ids) {
        if ((size_t) id >= n_reg) {
            fprintf(stderr, "device index %d out of range (%zu Vulkan devices)\n", id, n_reg);
            exit(2);
        }
        ggml_backend_dev_t dev = ggml_backend_reg_dev_get(ggml_backend_vk_reg(), id);
        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        if (!backend) {
            fprintf(stderr, "failed to init Vulkan device %d\n", id);
            exit(2);
        }
        g_backends.push_back(backend);
    }
    g_comm_init = (comm_init_t) ggml_backend_reg_get_proc_address(
        ggml_backend_vk_reg(), "ggml_backend_comm_init");
    g_comm_free = (comm_free_t) ggml_backend_reg_get_proc_address(
        ggml_backend_vk_reg(), "ggml_backend_comm_free");
    g_allreduce = (allreduce_t) ggml_backend_reg_get_proc_address(
        ggml_backend_vk_reg(), "ggml_backend_comm_allreduce_tensor");
    g_flush_async = (tp5_flush_async_t) ggml_backend_reg_get_proc_address(
        ggml_backend_vk_reg(), "ggml_backend_vk_flush_async");
    g_submit_epoch_chain = (submit_epoch_chain_t) ggml_backend_reg_get_proc_address(
        ggml_backend_vk_reg(), "ggml_backend_comm_submit_epoch_chain");
    g_get_cached_cmd_bufs = (get_cached_cmd_bufs_t) ggml_backend_reg_get_proc_address(
        ggml_backend_vk_reg(), "ggml_backend_vk_get_cached_cmd_bufs");
    g_prepare_graph =
        (prepare_graph_t) ggml_backend_reg_get_proc_address(ggml_backend_vk_reg(), "ggml_backend_comm_prepare_graph");
    if (!g_comm_init || !g_comm_free || !g_allreduce) {
        fprintf(stderr, "Vulkan registry does not expose the comm interface\n");
        exit(2);
    }
}

// Allocate an F32 tensor on backend j.
static ggml_tensor * alloc_tensor(ggml_backend_t backend, int64_t ne) {
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    // build a tiny context to create the tensor
    ggml_init_params ip{4096, nullptr, true};
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne);
    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(t));
    t->buffer = buf;
    t->data = ggml_backend_buffer_get_base(buf);
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    g_allocated_contexts.push_back(ctx);
    g_allocated_buffers.push_back(buf);
    return t;
}

// Create a 1D view with byte offset into parent tensor (must be storage buffer aligned, e.g. 256 bytes)
static ggml_tensor * alloc_tensor_view(ggml_tensor * parent, int64_t ne, size_t offset_bytes) {
    ggml_init_params ip{4096, nullptr, true};
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * view = ggml_view_1d(ctx, parent, ne, offset_bytes);
    enum ggml_status status = ggml_backend_view_init(view);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "ggml_backend_view_init failed for view (status %d)\n", (int) status);
        exit(2);
    }
    g_allocated_contexts.push_back(ctx);
    return view;
}

// Fill via staging (host-visible default buffer type path).
static void fill_tensor(ggml_tensor * t, const std::vector<float> & values) {
    ggml_backend_tensor_set(t, values.data(), 0, values.size() * sizeof(float));
}

static void read_tensor(ggml_tensor * t, std::vector<float> & out) {
    ggml_backend_tensor_get(t, out.data(), 0, ggml_nbytes(t));
}

struct rank_delay_workload {
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_cgraph *         gf  = nullptr;
    ggml_tensor *         out = nullptr;
};

// Create bounded small GPU-only delay workload on a backend (e.g. 32 dependent scale/add ops over 4096 floats).
// Flushed asynchronously to the GPU queue before collective submission — no host wait, no spin shaders.
static rank_delay_workload create_rank_delay_workload(ggml_backend_t backend, size_t n_elems = 4096, int n_ops = 32) {
    rank_delay_workload dw;
    size_t              mem_needed =
        ggml_tensor_overhead() * (n_ops * 2 + 4) + ggml_graph_overhead_custom(n_ops * 2 + 8, false) + 8192;
    ggml_init_params ip{ mem_needed, nullptr, true };
    dw.ctx              = ggml_init(ip);
    ggml_tensor * cur   = ggml_new_tensor_1d(dw.ctx, GGML_TYPE_F32, (int64_t) n_elems);
    ggml_tensor * input = cur;
    for (int op = 0; op < n_ops; ++op) {
        ggml_tensor * scale = ggml_scale(dw.ctx, cur, 1.0001f);
        cur                 = ggml_add(dw.ctx, scale, cur);
    }
    dw.out = cur;
    dw.gf  = ggml_new_graph_custom(dw.ctx, n_ops * 2 + 8, false);
    ggml_build_forward_expand(dw.gf, dw.out);

    dw.buf = ggml_backend_alloc_ctx_tensors(dw.ctx, backend);
    if (!dw.buf) {
        fprintf(stderr, "failed to alloc tensors for rank delay workload\n");
        exit(2);
    }
    fill_tensor(input, std::vector<float>(n_elems, 0.0f));
    return dw;
}

static void execute_rank_delay_workload(ggml_backend_t backend, const rank_delay_workload & dw) {
    enum ggml_status status = ggml_backend_graph_compute_async(backend, dw.gf);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "execute_rank_delay_workload failed async compute (status %d)\n", (int) status);
        g_failures++;
        return;
    }
    if (g_flush_async) {
        g_flush_async(backend);
    }
}

static float generate_test_val(int round, int binding_id, size_t rank, size_t elem, bool vary, bool fractional) {
    if (!vary) {
        return (float) (rank + 1);
    }
    int int_part = ((int)(rank + 1) * 7 + (int)(elem % 13) * 3 + (round * 11) + (binding_id * 5)) % 19 - 9;
    if (!fractional) {
        return (float) int_part;
    }
    float frac = 0.0f;
    switch (elem % 8) {
        case 0: frac = 0.1f * (float)(rank + 1); break;
        case 1: frac = 0.25f; break;
        case 2: frac = 0.0001f * (float)(rank + 1); break;
        case 3: frac = 0.5f; break;
        case 4: frac = 0.125f * (float)((int)rank - 2); break;
        case 5: frac = 0.03125f; break;
        case 6: frac = 0.0078125f * (float)(rank + 1); break;
        case 7: frac = (float)((elem + round + binding_id) % 7) * 0.1f; break;
    }
    return (float) int_part + frac;
}

static void run_round(void *                                   comm,
                      std::vector<ggml_tensor *> &             tensors,
                      size_t                                   n_elems,
                      int                                      round,
                      bool                                     vary,
                      bool                                     delay_producer,
                      int                                      binding_id,
                      bool                                     fractional,
                      std::vector<std::vector<float>> &        all_rank_results,
                      const std::vector<rank_delay_workload> * delay_workloads = nullptr) {
    const size_t P = tensors.size();
    all_rank_results.resize(P);
    for (size_t j = 0; j < P; ++j) {
        std::vector<float> in(n_elems);
        for (size_t e = 0; e < n_elems; ++e) {
            in[e] = generate_test_val(round, binding_id, j, e, vary, fractional);
        }
        fill_tensor(tensors[j], in);
    }
    // Ensure all staging writes from fill_tensor have finished on all backends
    for (size_t j = 0; j < P; ++j) {
        ggml_backend_synchronize(g_backends[j]);
    }

    if (delay_producer && delay_workloads && !delay_workloads->empty()) {
        // Delay selected rank (e.g. rank (round % P)) with bounded small GPU-only compute
        // flushed to GPU queue before collective submit, without host synchronization
        const size_t delayed_rank = (size_t) (round % (int) P);
        execute_rank_delay_workload(g_backends[delayed_rank], (*delay_workloads)[delayed_rank]);
    }

    if (!g_allreduce(comm, tensors.data())) {
        fprintf(stderr, "allreduce returned false (round %d)\n", round);
        g_failures++;
        return;
    }

    // Ensure reduction output is visible before host read
    for (size_t j = 0; j < P; ++j) {
        ggml_backend_synchronize(g_backends[j]);
    }
    for (size_t j = 0; j < P; ++j) {
        all_rank_results[j].resize(n_elems);
        read_tensor(tensors[j], all_rank_results[j]);
    }
}

static void check_result(const std::vector<std::vector<float>> & got_ranks, size_t n_elems,
                         size_t P, int round, bool vary, int binding_id, bool fractional, bool is_f16_wire) {
    for (size_t e = 0; e < n_elems; ++e) {
        float expect = 0.0f;
        if (vary) {
            for (size_t j = 0; j < P; ++j) {
                float v = generate_test_val(round, binding_id, j, e, vary, fractional);
                if (is_f16_wire) {
                    expect += ggml_fp16_to_fp32(ggml_fp32_to_fp16(v));
                } else {
                    expect += v;
                }
            }
        } else {
            for (size_t j = 0; j < P; ++j) expect += (float) (j + 1);
        }
        for (size_t j = 0; j < P; ++j) {
            float got = got_ranks[j][e];
            if (!std::isfinite(got) || !std::isfinite(expect)) {
                if (g_failures < 10) {
                    fprintf(stderr, "NON-FINITE rank=%zu round=%d elem=%zu got=%f expect=%f (binding=%d wire=%s)\n",
                            j, round, e, got, expect, binding_id, is_f16_wire ? "f16" : "f32");
                }
                g_failures++;
            } else
            if (got != expect) {
                if (g_failures < 10) {
                    fprintf(stderr, "MISMATCH rank=%zu round=%d elem=%zu got=%f expect=%f diff=%e (binding=%d wire=%s)\n",
                            j, round, e, got, expect, (double)(got - expect), binding_id, is_f16_wire ? "f16" : "f32");
                }
                g_failures++;
            }
        }
    }
}

struct rank_producer_graph {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor * in = nullptr;
    ggml_tensor * bias = nullptr;
    ggml_tensor * scaled = nullptr;
    ggml_tensor * out = nullptr;
    ggml_cgraph * gf = nullptr;
};

static rank_producer_graph create_rank_producer(ggml_backend_t backend, size_t n_elems) {
    rank_producer_graph rpg;
    // Compute graph: out = scale(in, 2.0f) + bias
    // Compute memory required: 4 tensors + cgraph overhead for 16 nodes
    size_t mem_needed = ggml_tensor_overhead() * 8 + ggml_graph_overhead_custom(16, false) + 4096;
    if (mem_needed < 131072) mem_needed = 131072; // bounded 128 KiB
    ggml_init_params ip{mem_needed, nullptr, true};
    rpg.ctx = ggml_init(ip);
    rpg.in = ggml_new_tensor_1d(rpg.ctx, GGML_TYPE_F32, (int64_t) n_elems);
    rpg.bias = ggml_new_tensor_1d(rpg.ctx, GGML_TYPE_F32, (int64_t) n_elems);
    rpg.scaled = ggml_scale(rpg.ctx, rpg.in, 2.0f);
    rpg.out = ggml_add(rpg.ctx, rpg.scaled, rpg.bias);

    rpg.gf = ggml_new_graph_custom(rpg.ctx, 16, false);
    ggml_build_forward_expand(rpg.gf, rpg.out);

    rpg.buf = ggml_backend_alloc_ctx_tensors(rpg.ctx, backend);
    if (!rpg.buf) {
        fprintf(stderr, "failed to alloc tensors for rank producer graph\n");
        exit(2);
    }
    return rpg;
}

static void run_producer_round(void * comm, const std::vector<rank_producer_graph> & producers,
                               size_t n_elems, int round, bool vary, bool fractional,
                               std::vector<std::vector<float>> & all_rank_results) {
    const size_t P = producers.size();
    all_rank_results.resize(P);

    for (size_t j = 0; j < P; ++j) {
        std::vector<float> in_vals(n_elems);
        std::vector<float> bias_vals(n_elems);
        for (size_t e = 0; e < n_elems; ++e) {
            float base = generate_test_val(round, 42, j, e, vary, fractional);
            in_vals[e] = base * 0.5f;
            bias_vals[e] = (float)((int)(j + 1) * 3 + round % 7);
        }
        fill_tensor(producers[j].in, in_vals);
        fill_tensor(producers[j].bias, bias_vals);
    }

    // Queue asynchronous compute on all backends with NO host sync
    for (size_t j = 0; j < P; ++j) {
        enum ggml_status status = ggml_backend_graph_compute_async(g_backends[j], producers[j].gf);
        if (status != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "ggml_backend_graph_compute_async failed on rank %zu (status %d)\n", j, (int) status);
            g_failures++;
            return;
        }
    }

    // Flush async to GPU queues before collective submit
    for (size_t j = 0; j < P; ++j) {
        if (g_flush_async) {
            g_flush_async(g_backends[j]);
        }
    }

    std::vector<ggml_tensor *> out_tensors(P);
    for (size_t j = 0; j < P; ++j) out_tensors[j] = producers[j].out;

    if (!g_allreduce(comm, out_tensors.data())) {
        fprintf(stderr, "allreduce returned false in producer round %d\n", round);
        g_failures++;
        return;
    }

    for (size_t j = 0; j < P; ++j) {
        ggml_backend_synchronize(g_backends[j]);
        all_rank_results[j].resize(n_elems);
        read_tensor(producers[j].out, all_rank_results[j]);
    }
}

static void check_producer_result(const std::vector<std::vector<float>> & got_ranks, size_t n_elems,
                                  size_t P, int round, bool vary, bool fractional, bool is_f16_wire) {
    for (size_t e = 0; e < n_elems; ++e) {
        float expect = 0.0f;
        for (size_t j = 0; j < P; ++j) {
            float base = generate_test_val(round, 42, j, e, vary, fractional);
            float in_val = base * 0.5f;
            float bias_val = (float)((int)(j + 1) * 3 + round % 7);
            float out_val = in_val * 2.0f + bias_val;
            if (is_f16_wire) {
                expect += ggml_fp16_to_fp32(ggml_fp32_to_fp16(out_val));
            } else {
                expect += out_val;
            }
        }
        for (size_t j = 0; j < P; ++j) {
            float got = got_ranks[j][e];
            if (!std::isfinite(got) || !std::isfinite(expect)) {
                if (g_failures < 10) {
                    fprintf(stderr, "NON-FINITE (producer) rank=%zu round=%d elem=%zu got=%f expect=%f (wire=%s)\n",
                            j, round, e, got, expect, is_f16_wire ? "f16" : "f32");
                }
                g_failures++;
            } else
            if (got != expect) {
                if (g_failures < 10) {
                    fprintf(stderr, "MISMATCH (producer) rank=%zu round=%d elem=%zu got=%f expect=%f diff=%e (wire=%s)\n",
                            j, round, e, got, expect, (double)(got - expect), is_f16_wire ? "f16" : "f32");
                }
                g_failures++;
            }
        }
    }
}

struct rank_step_graph {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor * in = nullptr;
    ggml_tensor * bias = nullptr;
    ggml_tensor * scaled = nullptr;
    ggml_tensor * added = nullptr;
    ggml_tensor * out = nullptr;
    ggml_cgraph * gf = nullptr;
};

static rank_step_graph create_rank_step_graph(ggml_backend_t backend,
                                              ggml_tensor *  in_tensor,
                                              ggml_tensor *  dst_tensor,
                                              size_t         n_elems,
                                              size_t         rank,
                                              bool           is_odd,
                                              bool           replay_matmul = false) {
    rank_step_graph rsg;
    size_t mem_needed = ggml_tensor_overhead() * 8 + ggml_graph_overhead_custom(16, false) + 4096;
    ggml_init_params ip{mem_needed, nullptr, true};
    rsg.ctx = ggml_init(ip);
    rsg.in = in_tensor;
    rsg.bias = ggml_new_tensor_1d(rsg.ctx, GGML_TYPE_F32, (int64_t) n_elems);
    ggml_tensor * weight =
        replay_matmul ? ggml_new_tensor_2d(rsg.ctx, GGML_TYPE_F32, (int64_t) n_elems, (int64_t) n_elems) : nullptr;
    rsg.scaled = weight ? ggml_mul_mat(rsg.ctx, weight, rsg.in) : ggml_scale(rsg.ctx, rsg.in, 0.125f);
    rsg.added = ggml_add(rsg.ctx, rsg.scaled, rsg.bias);
    rsg.out = ggml_cpy(rsg.ctx, rsg.added, dst_tensor);

    rsg.gf = ggml_new_graph_custom(rsg.ctx, 16, false);
    ggml_build_forward_expand(rsg.gf, rsg.out);

    rsg.buf = ggml_backend_alloc_ctx_tensors(rsg.ctx, backend);
    if (!rsg.buf) {
        fprintf(stderr, "failed to alloc tensors for rank step graph\n");
        exit(2);
    }

    std::vector<float> bias_vals(n_elems);
    for (size_t e = 0; e < n_elems; ++e) {
        bias_vals[e] = is_odd
            ? (float)((int)(rank + 1) * 5 + (int)(e % 11) * 2 - 9) * 0.125f
            : (float)((int)(rank + 1) * 3 + (int)(e % 7) * 2 - 5) * 0.125f;
    }
    fill_tensor(rsg.bias, bias_vals);
    if (weight) {
        std::vector<float> diagonal(n_elems * n_elems, 0.0f);
        for (size_t e = 0; e < n_elems; ++e)
            diagonal[e * n_elems + e] = 0.125f;
        fill_tensor(weight, diagonal);
    }
    ggml_backend_synchronize(backend);
    return rsg;
}

static void compute_cpu_timeline_oracle(size_t P, size_t n_elems, int steps, bool is_f16_wire,
                                        std::vector<std::vector<float>> & oracle_ranks) {
    oracle_ranks.assign(P, std::vector<float>(n_elems));
    std::vector<std::vector<float>> state(P, std::vector<float>(n_elems));
    for (size_t j = 0; j < P; ++j) {
        for (size_t e = 0; e < n_elems; ++e) {
            state[j][e] = (float)((int)(j + 1) * 7 + (int)(e % 13) * 3 - 11) * 0.25f;
        }
    }

    for (int s = 0; s < steps; ++s) {
        std::vector<std::vector<float>> computed(P, std::vector<float>(n_elems));
        for (size_t j = 0; j < P; ++j) {
            for (size_t e = 0; e < n_elems; ++e) {
                float bias_val = (s % 2 == 0)
                    ? (float)((int)(j + 1) * 3 + (int)(e % 7) * 2 - 5) * 0.125f
                    : (float)((int)(j + 1) * 5 + (int)(e % 11) * 2 - 9) * 0.125f;
                float val = state[j][e] * 0.125f + bias_val;
                if (is_f16_wire) {
                    // In Phase 1, pack_pipe converts input tensor to wire_buf (float16_t).
                    // Both local push (own mailbox slot) and peer push (peer mailbox slots) copy from wire_buf.
                    // Phase 2 shader reads float16_t from all mailbox slots into FP32 accumulator.
                    val = ggml_fp16_to_fp32(ggml_fp32_to_fp16(val));
                }
                computed[j][e] = val;
            }
        }
        // Fixed-rank-order reduction across all mailbox slots (identical across all ranks)
        for (size_t e = 0; e < n_elems; ++e) {
            float acc = 0.0f;
            for (size_t j = 0; j < P; ++j) {
                acc += computed[j][e];
            }
            for (size_t j = 0; j < P; ++j) {
                state[j][e] = acc;
            }
        }
    }
    oracle_ranks = std::move(state);
}

static void check_timeline_overlap_result(const std::vector<std::vector<float>> & got_ranks,
                                         const std::vector<std::vector<float>> & oracle_ranks,
                                         size_t n_elems, size_t P, int steps, bool is_f16_wire) {
    for (size_t e = 0; e < n_elems; ++e) {
        for (size_t j = 0; j < P; ++j) {
            float expect = oracle_ranks[j][e];
            float got = got_ranks[j][e];
            float tol = (float) steps * 1e-5f * std::max(1.0f, std::abs(expect)) + 1e-4f;
            if (!std::isfinite(got) || !std::isfinite(expect)) {
                if (g_failures < 10) {
                    fprintf(stderr, "NON-FINITE (timeline overlap) rank=%zu elem=%zu got=%f expect=%f (wire=%s)\n",
                            j, e, got, expect, is_f16_wire ? "f16" : "f32");
                }
                g_failures++;
            } else {
                float diff = std::abs(got - expect);
                if (diff > tol) {
                    if (g_failures < 10) {
                        fprintf(stderr, "MISMATCH (timeline overlap) rank=%zu elem=%zu got=%f expect=%f diff=%e > tol=%e (wire=%s)\n",
                                j, e, got, expect, (double) diff, (double) tol, is_f16_wire ? "f16" : "f32");
                    }
                    g_failures++;
                }
            }
        }
    }
}

static void run_timeline_overlap_regression(void * comm,
                                           const std::vector<rank_step_graph> & step_graphs_0,
                                           const std::vector<rank_step_graph> & step_graphs_1,
                                           std::vector<ggml_tensor *> & tensors_a,
                                           std::vector<ggml_tensor *> & tensors_b,
                                           size_t n_elems, int steps, bool is_f16_wire) {
    const size_t P = tensors_a.size();

    // Fill distinct skewed per-rank initial inputs into tensors_b (nonzero offset view)
    for (size_t j = 0; j < P; ++j) {
        std::vector<float> init_vals(n_elems);
        for (size_t e = 0; e < n_elems; ++e) {
            init_vals[e] = (float)((int)(j + 1) * 7 + (int)(e % 13) * 3 - 11) * 0.25f;
        }
        fill_tensor(tensors_b[j], init_vals);
    }
    for (size_t j = 0; j < P; ++j) {
        ggml_backend_synchronize(g_backends[j]);
    }

    // Bounded sequence: GPU producer -> AllReduce -> GPU consumer/nextproducer
    // Strictly ZERO host read or synchronize between individual collectives!
    for (int s = 0; s < steps; ++s) {
        const bool is_even = (s % 2 == 0);
        const auto & graphs = is_even ? step_graphs_0 : step_graphs_1;
        auto & out_tensors = is_even ? tensors_a : tensors_b;

        for (size_t j = 0; j < P; ++j) {
            enum ggml_status status = ggml_backend_graph_compute_async(g_backends[j], graphs[j].gf);
            if (status != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "ggml_backend_graph_compute_async failed in timeline overlap step %d rank %zu (status %d)\n",
                        s, j, (int) status);
                g_failures++;
                return;
            }
        }

        for (size_t j = 0; j < P; ++j) {
            if (g_flush_async) {
                g_flush_async(g_backends[j]);
            }
        }

        if (!g_allreduce(comm, out_tensors.data())) {
            fprintf(stderr, "allreduce returned false in timeline overlap step %d\n", s);
            g_failures++;
            return;
        }
    }

    for (size_t j = 0; j < P; ++j) {
        ggml_backend_synchronize(g_backends[j]);
    }

    const bool final_even = (steps % 2 == 1);
    auto & final_tensors = final_even ? tensors_a : tensors_b;

    std::vector<std::vector<float>> got_ranks(P, std::vector<float>(n_elems));
    for (size_t j = 0; j < P; ++j) {
        read_tensor(final_tensors[j], got_ranks[j]);
    }

    std::vector<std::vector<float>> oracle_ranks;
    compute_cpu_timeline_oracle(P, n_elems, steps, is_f16_wire, oracle_ranks);
    check_timeline_overlap_result(got_ranks, oracle_ranks, n_elems, P, steps, is_f16_wire);
}

// A repeated compute -> AR -> compute chain whose outputs depend on every
// preceding reduction. Matmul makes these tiny graphs genuinely replay eligible.
static void run_cached_compute_chain_regression(void * comm, bool is_f16_wire) {
    TEST_ASSERT(g_submit_epoch_chain && g_get_cached_cmd_bufs);
    const size_t                 P        = g_backends.size();
    const size_t                 elements = 32;
    std::vector<ggml_tensor *>   a(P), b(P);
    std::vector<rank_step_graph> even, odd, alt;
    for (size_t rank = 0; rank < P; ++rank) {
        a[rank] = alloc_tensor(g_backends[rank], elements);
        b[rank] = alloc_tensor(g_backends[rank], elements);
        fill_tensor(a[rank], std::vector<float>(elements, 0.0f));
        fill_tensor(b[rank], std::vector<float>(elements, 0.0f));
        even.push_back(create_rank_step_graph(g_backends[rank], b[rank], a[rank], elements, rank, false, true));
        odd.push_back(create_rank_step_graph(g_backends[rank], a[rank], b[rank], elements, rank, true, true));
        // Alternative tail compute graph: uses a different rank-scaled bias formula to produce distinct observable values
        alt.push_back(create_rank_step_graph(g_backends[rank], b[rank], a[rank], elements, rank, false, true));
        std::vector<float> alt_bias(elements);
        for (size_t e = 0; e < elements; ++e) {
            alt_bias[e] = float(int(rank + 1) * 11 + int(e % 5) * 4 + 7) * 0.125f;
        }
        fill_tensor(alt[rank].bias, alt_bias);
        // First-ever scratch growth can cancel recording; the second pass
        // records both graphs in the final scratch generation.
        for (int warmup = 0; warmup < 2; ++warmup) {
            TEST_ASSERT(ggml_backend_graph_compute(g_backends[rank], even[rank].gf) == GGML_STATUS_SUCCESS);
            TEST_ASSERT(ggml_backend_graph_compute(g_backends[rank], odd[rank].gf) == GGML_STATUS_SUCCESS);
            TEST_ASSERT(ggml_backend_graph_compute(g_backends[rank], alt[rank].gf) == GGML_STATUS_SUCCESS);
        }
    }
    for (int steps : { 65, 66 }) {
        std::vector<std::vector<void *>> even_cbs(P), odd_cbs(P), alt_cbs(P);
        for (size_t rank = 0; rank < P; ++rank) {
            std::vector<float> input(elements);
            for (size_t e = 0; e < elements; ++e)
                input[e] = float(int(rank + 1) * 7 + int(e % 13) * 3 - 11) * 0.25f;
            fill_tensor(b[rank], input);
            TEST_ASSERT(g_get_cached_cmd_bufs(g_backends[rank], even[rank].gf, even_cbs[rank]));
            TEST_ASSERT(g_get_cached_cmd_bufs(g_backends[rank], odd[rank].gf, odd_cbs[rank]));
            TEST_ASSERT(g_get_cached_cmd_bufs(g_backends[rank], alt[rank].gf, alt_cbs[rank]));
        }
        std::vector<std::vector<std::vector<void *>>> cbs(steps + 1);
        std::vector<std::vector<ggml_tensor *>>       tensors(steps);
        for (int step = 0; step < steps; ++step) {
            cbs[step]     = step % 2 == 0 ? even_cbs : odd_cbs;
            tensors[step] = step % 2 == 0 ? a : b;
        }
        cbs.back()       = steps % 2 == 0 ? even_cbs : odd_cbs;
        // A malformed final CB must fail BEFORE any reduction is submitted.
        auto invalid_cbs = cbs;
        invalid_cbs.back()[0].push_back(nullptr);
        TEST_ASSERT(!g_submit_epoch_chain(comm, invalid_cbs, tensors));

        std::vector<std::vector<float>> oracle;
        compute_cpu_timeline_oracle(P, elements, steps, is_f16_wire, oracle);

        auto run_pass = [&](const std::vector<std::vector<std::vector<void *>>> & pass_cbs, bool has_tail,
                            bool is_alt_tail) {
            for (size_t rank = 0; rank < P; ++rank) {
                std::vector<float> input(elements);
                for (size_t e = 0; e < elements; ++e) {
                    input[e] = float(int(rank + 1) * 7 + int(e % 13) * 3 - 11) * 0.25f;
                }
                fill_tensor(b[rank], input);
            }
            for (auto backend : g_backends) {
                ggml_backend_synchronize(backend);
            }
            TEST_ASSERT(g_submit_epoch_chain(comm, pass_cbs, tensors));
            for (auto backend : g_backends) {
                ggml_backend_synchronize(backend);
            }

            // The last AllReduce writes stage tensor[steps - 1] (steps % 2 == 0 ? b : a).
            // When tail compute is present, it computes from tensor[steps - 1] into destination (steps % 2 == 0 ? a : b).
            // When tail compute is empty, tensor[steps - 1] holds the final reduction output.
            const auto & target_tensors = has_tail ? (steps % 2 == 0 ? a : b) : (steps % 2 == 0 ? b : a);
            for (size_t rank = 0; rank < P; ++rank) {
                std::vector<float> output(elements);
                read_tensor(target_tensors[rank], output);
                for (size_t e = 0; e < elements; ++e) {
                    float expected = oracle[rank][e];
                    if (has_tail) {
                        const float tail_bias =
                            is_alt_tail ? float(int(rank + 1) * 11 + int(e % 5) * 4 + 7) * 0.125f :
                                          (steps % 2 == 0 ? float(int(rank + 1) * 3 + int(e % 7) * 2 - 5) * 0.125f :
                                                            float(int(rank + 1) * 5 + int(e % 11) * 2 - 9) * 0.125f);
                        expected = expected * 0.125f + tail_bias;
                    }
                    TEST_ASSERT(std::isfinite(output[e]) && output[e] == expected);
                }
            }
        };

        // Initial cold execution
        run_pass(cbs, true, false);
        // Replay with warm template
        run_pass(cbs, true, false);

        // Transition to empty tail compute CBs
        auto cbs_empty = cbs;
        for (size_t rank = 0; rank < P; ++rank) {
            cbs_empty.back()[rank].clear();
        }
        run_pass(cbs_empty, false, false);
        // Replay empty tail template
        run_pass(cbs_empty, false, false);

        // Transition to equal-count different-handle compute tail (distinguishes CB identity vs count)
        auto cbs_alt = cbs;
        for (size_t rank = 0; rank < P; ++rank) {
            cbs_alt.back()[rank] = alt_cbs[rank];
        }
        // The alternate graph reads b and writes a, matching the even chain tail.
        if (steps % 2 == 0) {
            for (size_t rank = 0; rank < P; ++rank) {
                TEST_ASSERT(alt_cbs[rank].size() == even_cbs[rank].size());
            }
            run_pass(cbs_alt, true, true);
            run_pass(cbs_alt, true, true);
        }

        // Transition back to original tail compute CBs
        run_pass(cbs, true, false);
        run_pass(cbs, true, false);

        fprintf(stderr,
                "  cached GPU compute -> AR -> final compute (%d reductions, replay, empty/restored tail): OK\n",
                steps);
    }
    for (auto * graphs : { &even, &odd, &alt }) {
        for (auto & graph : *graphs) {
            ggml_backend_buffer_free(graph.buf);
            ggml_free(graph.ctx);
        }
    }
}

// Structure representing a stage in an epoch chain
struct chain_stage_data {
    size_t                          n_elems;
    std::vector<ggml_tensor *>      tensors;     // per-rank tensors (P tensors)
    std::vector<std::vector<float>> cpu_inputs;  // [rank][elem]
    std::vector<float>              cpu_expect;  // [elem]
    // Optional view parent tensors and guard bytes for offset view stages
    std::vector<ggml_tensor *>      view_parents;
    size_t                          head_guard_bytes = 0;
    size_t                          tail_guard_bytes = 0;
};

// Runs a comprehensive regression test over ggml_backend_comm_submit_epoch_chain:
// - Bounded regression through actual epoch-chain registry for both F16 and F32 wire contracts
// - Bank wrap (>= 2 stages per chain and multiple chained submits)
// - Alternating offsets / shape growth across calls (e.g. 2560 -> 5120 -> 2573 odd length)
// - Changed data across stages and calls
// - >128 total epochs across calls and starting parities
// - Multi-plan stability (cold plans added, vector stability, memory retention)
// - Pure collective chain with empty compute vectors across distinct per-stage tensors
// - Checks ALL stage outputs against CPU oracle
// - Guard bytes on offset views verified untouched
// - Pre-submit safety failure verification for malformed input (>128 stages)
static void run_epoch_chain_regression(void * comm, const std::vector<ggml_backend_t> & backends, bool is_f16_wire) {
    TEST_ASSERT(g_submit_epoch_chain);
    const size_t P = backends.size();
    fprintf(stderr, "  running epoch-chain regression (wire=%s, P=%zu)...\n", is_f16_wire ? "f16" : "f32", P);

    // Pre-flight check: malformed/excessive stages (>128 stages) MUST safely fail closed (return false) before submission
    {
        const size_t                                  invalid_stage_count = 129;
        std::vector<std::vector<std::vector<void *>>> invalid_cbs(invalid_stage_count + 1,
                                                                  std::vector<std::vector<void *>>(P));
        std::vector<std::vector<ggml_tensor *>> invalid_tensors(invalid_stage_count, std::vector<ggml_tensor *>(P));
        // Allocate a dummy 1-element tensor per rank for stage 0
        std::vector<ggml_tensor *>              dummy_tensors;
        for (size_t j = 0; j < P; ++j) {
            dummy_tensors.push_back(alloc_tensor(backends[j], 1));
        }
        for (size_t s = 0; s < invalid_stage_count; ++s) {
            for (size_t j = 0; j < P; ++j) {
                invalid_tensors[s][j] = dummy_tensors[j];
            }
        }
        bool preflight_res = g_submit_epoch_chain(comm, invalid_cbs, invalid_tensors);
        if (preflight_res) {
            fprintf(stderr, "FAIL: submit_epoch_chain accepted >128 stages (%zu); expected pre-submit false!\n",
                    invalid_stage_count);
            g_failures++;
            return;
        }
        fprintf(stderr, "    preflight >128 stages rejection: OK\n");
    }

    // Prepare test calls exercising:
    // Call 0: 65 stages, base shape (2560 elements), pure collective (empty compute CBs)
    // Call 1: 33 stages, workspace grow (5120 elements), alternating standard & offset views with guard bytes
    // Call 2: 48 stages, odd shape > 2560 (2573 elements), changed data, cold plan addition
    // Odd chain lengths change the starting bank; 146 total epochs exceed the owner window.
    const std::vector<size_t> call_stage_counts = { 65, 33, 48 };
    const std::vector<size_t> call_elem_counts  = { 2560, 5120, 2573 };

    int call_idx = 0;
    for (size_t c_idx = 0; c_idx < call_stage_counts.size(); ++c_idx) {
        const size_t N_STAGES = call_stage_counts[c_idx];
        const size_t n_elems  = call_elem_counts[c_idx];
        fprintf(stderr, "    call %d: submitting chain with %zu stages (elements=%zu)...\n", call_idx, N_STAGES,
                n_elems);

        std::vector<chain_stage_data>                 stage_data(N_STAGES);
        std::vector<std::vector<std::vector<void *>>> stage_cbs(N_STAGES + 1, std::vector<std::vector<void *>>(P));
        std::vector<std::vector<ggml_tensor *>>       stage_tensors(N_STAGES, std::vector<ggml_tensor *>(P));

        for (size_t s = 0; s < N_STAGES; ++s) {
            stage_data[s].n_elems = n_elems;
            stage_data[s].tensors.resize(P);
            stage_data[s].cpu_inputs.resize(P, std::vector<float>(n_elems));
            stage_data[s].cpu_expect.assign(n_elems, 0.0f);

            const bool   is_view_stage     = (s % 4 == 2);  // Exercise offset views with guard bytes
            const size_t head_guard_bytes  = is_view_stage ? 256 : 0;
            const size_t tail_guard_bytes  = is_view_stage ? 256 : 0;
            stage_data[s].head_guard_bytes = head_guard_bytes;
            stage_data[s].tail_guard_bytes = tail_guard_bytes;
            if (is_view_stage) {
                stage_data[s].view_parents.resize(P);
            }

            for (size_t j = 0; j < P; ++j) {
                if (is_view_stage) {
                    const size_t  parent_elems    = n_elems + (head_guard_bytes + tail_guard_bytes) / sizeof(float);
                    ggml_tensor * parent          = alloc_tensor(backends[j], (int64_t) parent_elems);
                    stage_data[s].view_parents[j] = parent;
                    stage_data[s].tensors[j]      = alloc_tensor_view(parent, (int64_t) n_elems, head_guard_bytes);

                    // Fill parent with guard pattern 0xAA in head and 0x55 in tail
                    std::vector<uint8_t> head_guard(head_guard_bytes, 0xAA);
                    std::vector<uint8_t> tail_guard(tail_guard_bytes, 0x55);
                    ggml_backend_tensor_set(parent, head_guard.data(), 0, head_guard_bytes);
                    ggml_backend_tensor_set(parent, tail_guard.data(), head_guard_bytes + n_elems * sizeof(float),
                                            tail_guard_bytes);
                } else {
                    stage_data[s].tensors[j] = alloc_tensor(backends[j], (int64_t) n_elems);
                }
                stage_tensors[s][j] = stage_data[s].tensors[j];

                // Fill distinct inputs for stage s, rank j
                for (size_t e = 0; e < n_elems; ++e) {
                    float val = generate_test_val((int) (call_idx * 100 + s), (int) s, j, e, true, true);
                    stage_data[s].cpu_inputs[j][e] = val;
                }
                fill_tensor(stage_data[s].tensors[j], stage_data[s].cpu_inputs[j]);
            }

            // Compute CPU oracle for stage s
            for (size_t e = 0; e < n_elems; ++e) {
                float sum = 0.0f;
                for (size_t j = 0; j < P; ++j) {
                    float v = stage_data[s].cpu_inputs[j][e];
                    if (is_f16_wire) {
                        sum += ggml_fp16_to_fp32(ggml_fp32_to_fp16(v));
                    } else {
                        sum += v;
                    }
                }
                stage_data[s].cpu_expect[e] = sum;
            }
        }

        // Ensure all host-staging writes completed on all backends before chain submission
        for (size_t j = 0; j < P; ++j) {
            ggml_backend_synchronize(backends[j]);
        }

        // Submit chain
        bool ok = g_submit_epoch_chain(comm, stage_cbs, stage_tensors);
        if (!ok) {
            fprintf(stderr, "FAIL: submit_epoch_chain returned false for call %d (%zu stages)\n", call_idx, N_STAGES);
            g_failures++;
            return;
        }

        // Synchronize all devices once after full chain completes
        for (size_t j = 0; j < P; ++j) {
            ggml_backend_synchronize(backends[j]);
        }

        // Verify outputs of ALL stages across ALL ranks against CPU oracle
        for (size_t s = 0; s < N_STAGES; ++s) {
            for (size_t j = 0; j < P; ++j) {
                std::vector<float> got(n_elems);
                read_tensor(stage_data[s].tensors[j], got);
                for (size_t e = 0; e < n_elems; ++e) {
                    float expect = stage_data[s].cpu_expect[e];
                    float actual = got[e];
                    if (!std::isfinite(actual) || !std::isfinite(expect)) {
                        if (g_failures < 10) {
                            fprintf(
                                stderr,
                                "NON-FINITE (chain call %d stage %zu) rank=%zu elem=%zu got=%f expect=%f (wire=%s)\n",
                                call_idx, s, j, e, actual, expect, is_f16_wire ? "f16" : "f32");
                        }
                        g_failures++;
                    } else if (actual != expect) {
                        if (g_failures < 10) {
                            fprintf(stderr,
                                    "MISMATCH (chain call %d stage %zu) rank=%zu elem=%zu got=%f expect=%f (wire=%s)\n",
                                    call_idx, s, j, e, actual, expect, is_f16_wire ? "f16" : "f32");
                        }
                        g_failures++;
                    }
                }

                // Verify guard bytes for offset view stages
                if (stage_data[s].head_guard_bytes > 0 && stage_data[s].view_parents[j]) {
                    std::vector<uint8_t> head_check(stage_data[s].head_guard_bytes);
                    ggml_backend_tensor_get(stage_data[s].view_parents[j], head_check.data(), 0,
                                            stage_data[s].head_guard_bytes);
                    for (size_t b = 0; b < stage_data[s].head_guard_bytes; ++b) {
                        if (head_check[b] != 0xAA) {
                            fprintf(stderr,
                                    "FAIL: head guard byte corruption at byte %zu in call %d stage %zu rank %zu (got "
                                    "0x%02x != 0xAA)\n",
                                    b, call_idx, s, j, head_check[b]);
                            g_failures++;
                            break;
                        }
                    }
                    std::vector<uint8_t> tail_check(stage_data[s].tail_guard_bytes);
                    const size_t         tail_offset = stage_data[s].head_guard_bytes + n_elems * sizeof(float);
                    ggml_backend_tensor_get(stage_data[s].view_parents[j], tail_check.data(), tail_offset,
                                            stage_data[s].tail_guard_bytes);
                    for (size_t b = 0; b < stage_data[s].tail_guard_bytes; ++b) {
                        if (tail_check[b] != 0x55) {
                            fprintf(stderr,
                                    "FAIL: tail guard byte corruption at byte %zu in call %d stage %zu rank %zu (got "
                                    "0x%02x != 0x55)\n",
                                    b, call_idx, s, j, tail_check[b]);
                            g_failures++;
                            break;
                        }
                    }
                }
            }
        }

        // Immediate unchanged replay on warm compiled-chain template:
        // Refill stage tensors with fresh values, recompute CPU oracle, and verify template replay
        for (size_t s = 0; s < N_STAGES; ++s) {
            stage_data[s].cpu_expect.assign(n_elems, 0.0f);
            const bool is_view_stage = (s % 4 == 2);
            for (size_t j = 0; j < P; ++j) {
                if (is_view_stage && stage_data[s].view_parents[j]) {
                    std::vector<uint8_t> head_guard(stage_data[s].head_guard_bytes, 0xAA);
                    std::vector<uint8_t> tail_guard(stage_data[s].tail_guard_bytes, 0x55);
                    ggml_backend_tensor_set(stage_data[s].view_parents[j], head_guard.data(), 0,
                                            stage_data[s].head_guard_bytes);
                    ggml_backend_tensor_set(stage_data[s].view_parents[j], tail_guard.data(),
                                            stage_data[s].head_guard_bytes + n_elems * sizeof(float),
                                            stage_data[s].tail_guard_bytes);
                }
                for (size_t e = 0; e < n_elems; ++e) {
                    float val = generate_test_val((int) (call_idx * 100 + s + 50), (int) (s + 7), j, e, true, true);
                    stage_data[s].cpu_inputs[j][e] = val;
                }
                fill_tensor(stage_data[s].tensors[j], stage_data[s].cpu_inputs[j]);
            }
            for (size_t e = 0; e < n_elems; ++e) {
                float sum = 0.0f;
                for (size_t j = 0; j < P; ++j) {
                    float v = stage_data[s].cpu_inputs[j][e];
                    sum += is_f16_wire ? ggml_fp16_to_fp32(ggml_fp32_to_fp16(v)) : v;
                }
                stage_data[s].cpu_expect[e] = sum;
            }
        }
        for (size_t j = 0; j < P; ++j) {
            ggml_backend_synchronize(backends[j]);
        }
        ok = g_submit_epoch_chain(comm, stage_cbs, stage_tensors);
        if (!ok) {
            fprintf(stderr, "FAIL: submit_epoch_chain replay returned false for call %d (%zu stages)\n", call_idx,
                    N_STAGES);
            g_failures++;
            return;
        }
        for (size_t j = 0; j < P; ++j) {
            ggml_backend_synchronize(backends[j]);
        }
        for (size_t s = 0; s < N_STAGES; ++s) {
            for (size_t j = 0; j < P; ++j) {
                std::vector<float> got(n_elems);
                read_tensor(stage_data[s].tensors[j], got);
                for (size_t e = 0; e < n_elems; ++e) {
                    float expect = stage_data[s].cpu_expect[e];
                    float actual = got[e];
                    if (!std::isfinite(actual) || !std::isfinite(expect) || actual != expect) {
                        if (g_failures < 10) {
                            fprintf(stderr,
                                    "MISMATCH (chain replay call %d stage %zu) rank=%zu elem=%zu got=%f expect=%f\n",
                                    call_idx, s, j, e, actual, expect);
                        }
                        g_failures++;
                    }
                }
                if (stage_data[s].head_guard_bytes > 0 && stage_data[s].view_parents[j]) {
                    std::vector<uint8_t> head_check(stage_data[s].head_guard_bytes);
                    ggml_backend_tensor_get(stage_data[s].view_parents[j], head_check.data(), 0,
                                            stage_data[s].head_guard_bytes);
                    for (size_t b = 0; b < stage_data[s].head_guard_bytes; ++b) {
                        if (head_check[b] != 0xAA) {
                            fprintf(stderr,
                                    "FAIL: head guard byte corruption on replay in call %d stage %zu rank %zu\n",
                                    call_idx, s, j);
                            g_failures++;
                            break;
                        }
                    }
                    std::vector<uint8_t> tail_check(stage_data[s].tail_guard_bytes);
                    const size_t         tail_offset = stage_data[s].head_guard_bytes + n_elems * sizeof(float);
                    ggml_backend_tensor_get(stage_data[s].view_parents[j], tail_check.data(), tail_offset,
                                            stage_data[s].tail_guard_bytes);
                    for (size_t b = 0; b < stage_data[s].tail_guard_bytes; ++b) {
                        if (tail_check[b] != 0x55) {
                            fprintf(stderr,
                                    "FAIL: tail guard byte corruption on replay in call %d stage %zu rank %zu\n",
                                    call_idx, s, j);
                            g_failures++;
                            break;
                        }
                    }
                }
            }
        }
        fprintf(stderr, "    call %d (%zu stages, elements=%zu): OK\n", call_idx, N_STAGES, n_elems);
        call_idx++;
    }
    fprintf(stderr, "  epoch-chain regression complete: OK\n");
}

// Paired warm measurements. Both paths execute and validate the same work;
// input reset and readback are outside the timed submission + completion span.
static void run_paired_chain_benchmark(void *                              comm,
                                       const std::vector<ggml_backend_t> & backends,
                                       size_t                              n_elems,
                                       size_t                              n_stages,
                                       bool                                is_f16_wire) {
    TEST_ASSERT(g_submit_epoch_chain);
    const size_t                            P = backends.size();
    std::vector<std::vector<ggml_tensor *>> tensors(n_stages, std::vector<ggml_tensor *>(P));
    for (size_t s = 0; s < n_stages; ++s) {
        for (size_t j = 0; j < P; ++j)
            tensors[s][j] = alloc_tensor(backends[j], (int64_t) n_elems);
    }
    std::vector<std::vector<std::vector<void *>>> cbs(n_stages + 1, std::vector<std::vector<void *>>(P));
    auto                                          run = [&](bool batched, double & elapsed) -> bool {
        for (size_t s = 0; s < n_stages; ++s) {
            for (size_t j = 0; j < P; ++j)
                fill_tensor(tensors[s][j], std::vector<float>(n_elems, float(s + j + 1)));
        }
        for (auto backend : backends)
            ggml_backend_synchronize(backend);
        if (batched) {
            std::vector<float> pre0(n_elems);
            read_tensor(tensors[0][0], pre0);
            fprintf(stderr, "TEST_RUN pre0[0]=%f\n", pre0[0]);
        }
        const auto start = std::chrono::steady_clock::now();
        if (batched) {
            if (!g_submit_epoch_chain(comm, cbs, tensors))
                return false;
        } else {
            for (auto & stage : tensors) {
                if (!g_allreduce(comm, stage.data()))
                    return false;
            }
        }
        for (auto backend : backends)
            ggml_backend_synchronize(backend);
        elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        for (size_t s = 0; s < n_stages; ++s) {
            float expected = 0;
            for (size_t j = 0; j < P; ++j) {
                const float input = float(s + j + 1);
                expected += is_f16_wire ? ggml_fp16_to_fp32(ggml_fp32_to_fp16(input)) : input;
            }
            for (size_t j = 0; j < P; ++j) {
                std::vector<float> output(n_elems);
                read_tensor(tensors[s][j], output);
                for (float value : output) {
                    if (!std::isfinite(value) || value != expected) {
                        fprintf(stderr, "paired benchmark mismatch: batched=%d stage=%zu rank=%zu got=%f expected=%f\n",
                                                                         int(batched), s, j, value, expected);
                        return false;
                    }
                }
            }
        }
        return true;
    };
    double warmup = 0;
    TEST_ASSERT(run(false, warmup));
    TEST_ASSERT(run(true, warmup));
    std::vector<double> ordinary(5), batched(5), speedups(5);
    for (size_t pair = 0; pair < ordinary.size(); ++pair) {
        if (pair % 2 == 0) {
            TEST_ASSERT(run(false, ordinary[pair]));
            TEST_ASSERT(run(true, batched[pair]));
        } else {
            TEST_ASSERT(run(true, batched[pair]));
            TEST_ASSERT(run(false, ordinary[pair]));
        }
        speedups[pair] = ordinary[pair] / batched[pair];
        fprintf(stderr, "  paired run %zu: ordinary=%.3f ms chain=%.3f ms speedup=%.3fx\n", pair, ordinary[pair],
                batched[pair], speedups[pair]);
    }
    std::sort(ordinary.begin(), ordinary.end());
    std::sort(batched.begin(), batched.end());
    std::sort(speedups.begin(), speedups.end());
    fprintf(stderr,
            "  paired median: %zu stages, %zu elements, wire=%s, ordinary=%.3f ms chain=%.3f ms speedup=%.3fx "
            "(includes completion)\n",
            n_stages, n_elems, is_f16_wire ? "f16" : "f32", ordinary[2], batched[2], speedups[2]);
}

// Two distinct terminal MoE producers share each backend's packed companion.
// Compare against independently rounded F32 producers, then replay both stages
// without host waits. A second reduction without compute must not reuse the
// original companion: its input is now the first reduction's F32 result.
static void run_producer_wire_regression(void * comm) {
    TEST_ASSERT(g_prepare_graph && g_get_cached_cmd_bufs && g_submit_epoch_chain);
    constexpr int width = 256, k = 128, selected = 10, experts = 16, rounds = 5;
    const size_t  ranks = g_backends.size();

    struct producer {
        ggml_cgraph * graph;
        ggml_tensor * input;
        ggml_tensor * hidden;
        ggml_tensor * ids;
        ggml_tensor * routing;
        ggml_tensor * output;
    };

    std::vector<std::vector<producer>> producers(2, std::vector<producer>(ranks));
    for (size_t stage = 0; stage < 2; ++stage) {
        for (size_t rank = 0; rank < ranks; ++rank) {
            auto * ctx = ggml_init({ 1024 * 1024, nullptr, true });
            TEST_ASSERT(ctx);
            g_allocated_contexts.push_back(ctx);
            auto * down   = ggml_new_tensor_3d(ctx, GGML_TYPE_IQ4_NL, k, width, experts);
            auto * gate   = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, width, k);
            auto * up     = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, width, k);
            auto * shared = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, k, width);
            auto * scalar = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, 1);
            auto & p      = producers[stage][rank];
            p.input       = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, width);
            p.hidden      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, selected);
            p.ids         = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, selected);
            p.routing     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, selected);
            for (auto * t : { p.input, p.hidden, p.ids, p.routing })
                ggml_set_input(t);
            auto * input = ggml_scale(ctx, p.input, 1.0f);
            auto * weighted =
                ggml_mul(ctx, ggml_mul_mat_id(ctx, down, ggml_scale(ctx, p.hidden, 1.0f), p.ids), p.routing);
            p.graph = ggml_new_graph_custom(ctx, 128, false);
            ggml_build_forward_expand(p.graph, input);
            ggml_tensor * views[selected];
            for (int e = 0; e < selected; ++e) {
                views[e] = ggml_view_1d(ctx, weighted, width, e * width * sizeof(float));
                ggml_build_forward_expand(p.graph, views[e]);
            }
            auto * sum = ggml_add(ctx, views[0], views[1]);
            for (int e = 2; e < selected; ++e)
                sum = ggml_add(ctx, sum, views[e]);
            auto * shared_value = ggml_mul_mat(
                ctx, shared, ggml_swiglu_split(ctx, ggml_mul_mat(ctx, gate, input), ggml_mul_mat(ctx, up, input)));
            auto * shared_gate = ggml_sigmoid(ctx, ggml_mul_mat(ctx, scalar, input));
            p.output           = ggml_add(ctx, sum, ggml_mul(ctx, shared_value, shared_gate));
            if (stage == 1)
                p.output = ggml_reshape_2d(ctx, ggml_reshape_1d(ctx, p.output, width), width, 1);
            ggml_set_output(p.output);
            ggml_build_forward_expand(p.graph, p.output);
            auto buffer = ggml_backend_alloc_ctx_tensors(ctx, g_backends[rank]);
            TEST_ASSERT(buffer);
            g_allocated_buffers.push_back(buffer);
            int seed = 0;
            for (auto * weight : { down, gate, up, shared, scalar }) {
                std::vector<float> values(ggml_nelements(weight));
                for (size_t i = 0; i < values.size(); ++i) {
                    values[i] = std::sin(float(i % 997) * 0.037f + seed + stage * 0.31f + rank * 0.43f) * 0.125f;
                }
                std::vector<unsigned char> packed(ggml_nbytes(weight));
                TEST_ASSERT(ggml_quantize_chunk(weight->type, values.data(), packed.data(), 0,
                                                ggml_nelements(weight) / weight->ne[0], weight->ne[0],
                                                nullptr) == packed.size());
                ggml_backend_tensor_set(weight, packed.data(), 0, packed.size());
                ++seed;
            }
        }
    }
    auto inputs = [&](producer & p, int round, size_t rank, size_t stage) {
        int32_t ids[selected];
        for (int e = 0; e < selected; ++e)
            ids[e] = (e * 7 + round * 3 + int(rank)) % experts;
        ggml_backend_tensor_set(p.ids, ids, 0, sizeof(ids));
        for (auto * t : { p.input, p.hidden, p.routing }) {
            std::vector<float> values(ggml_nelements(t));
            for (size_t i = 0; i < values.size(); ++i) {
                values[i] = t == p.routing ?
                                0.01f + (i + round) * 0.017f :
                                std::sin(float(i % 127) * 0.171f + round * 0.33f + rank * 0.21f + stage * 0.57f) * 1.5f;
            }
            fill_tensor(t, values);
        }
    };
    std::vector<std::vector<std::vector<float>>> expected(
        rounds, std::vector<std::vector<float>>(2, std::vector<float>(width, 0.0f)));
    std::vector<std::vector<std::vector<float>>> raw_reference(
        2, std::vector<std::vector<float>>(ranks, std::vector<float>(width)));
    for (int round = 0; round < rounds; ++round) {
        for (size_t stage = 0; stage < 2; ++stage) {
            for (size_t rank = 0; rank < ranks; ++rank) {
                auto & p = producers[stage][rank];
                inputs(p, round, rank, stage);
                TEST_ASSERT(g_prepare_graph(comm, rank, p.graph, false));
                TEST_ASSERT(ggml_backend_graph_compute(g_backends[rank], p.graph) == GGML_STATUS_SUCCESS);
                std::vector<float> values(width);
                read_tensor(p.output, values);
                if (round == 0)
                    raw_reference[stage][rank] = values;
                for (int e = 0; e < width; ++e)
                    expected[round][stage][e] += ggml_fp16_to_fp32(ggml_fp32_to_fp16(values[e]));
            }
        }
    }
    std::vector<std::vector<ggml_tensor *>> outputs(2, std::vector<ggml_tensor *>(ranks));
    for (size_t stage = 0; stage < 2; ++stage) {
        for (size_t rank = 0; rank < ranks; ++rank)
            outputs[stage][rank] = producers[stage][rank].output;
    }
    for (int round = 0; round < rounds; ++round) {
        std::vector<std::vector<std::vector<void *>>> cbs(3, std::vector<std::vector<void *>>(ranks));
        for (size_t stage = 0; stage < 2; ++stage) {
            for (size_t rank = 0; rank < ranks; ++rank) {
                auto & p = producers[stage][rank];
                inputs(p, round, rank, stage);
                TEST_ASSERT(g_prepare_graph(comm, rank, p.graph, true));
                if (round >= 2) {
                    TEST_ASSERT(g_get_cached_cmd_bufs(g_backends[rank], p.graph, cbs[stage][rank]));
                } else {
                    TEST_ASSERT(ggml_backend_graph_compute_async(g_backends[rank], p.graph) == GGML_STATUS_SUCCESS);
                }
            }
            if (round == 0) {
                for (size_t rank = 0; rank < ranks; ++rank) {
                    ggml_backend_synchronize(g_backends[rank]);
                    std::vector<float> values(width);
                    read_tensor(outputs[stage][rank], values);
                    TEST_ASSERT(std::memcmp(values.data(), raw_reference[stage][rank].data(), width * sizeof(float)) ==
                                0);
                }
            }
            if (round < 2)
                TEST_ASSERT(g_allreduce(comm, outputs[stage].data()));
        }
        if (round >= 2)
            TEST_ASSERT(g_submit_epoch_chain(comm, cbs, outputs));
        for (size_t stage = 0; stage < 2; ++stage) {
            for (size_t rank = 0; rank < ranks; ++rank) {
                ggml_backend_synchronize(g_backends[rank]);
                std::vector<float> values(width);
                read_tensor(outputs[stage][rank], values);
                for (int e = 0; e < width; ++e) {
                    if (!std::isfinite(values[e]) || values[e] != expected[round][stage][e]) {
                        fprintf(stderr,
                                "producer-wire mismatch: round=%d stage=%zu rank=%zu element=%d got=%a expected=%a\n",
                                round, stage, rank, e, values[e], expected[round][stage][e]);
                    }
                    TEST_ASSERT(std::isfinite(values[e]) && values[e] == expected[round][stage][e]);
                }
            }
        }
        // Taking the companion must consume it even when its plan is cached.
        TEST_ASSERT(g_allreduce(comm, outputs[1].data()));
        for (size_t rank = 0; rank < ranks; ++rank) {
            ggml_backend_synchronize(g_backends[rank]);
            std::vector<float> values(width);
            read_tensor(outputs[1][rank], values);
            for (int e = 0; e < width; ++e) {
                float sum = 0.0f;
                for (size_t peer = 0; peer < ranks; ++peer)
                    sum += ggml_fp16_to_fp32(ggml_fp32_to_fp16(expected[round][1][e]));
                TEST_ASSERT(std::isfinite(values[e]) && values[e] == sum);
            }
        }
    }
    fprintf(
        stderr,
        "  producer-wire regression: mutable routes, shared scratch, serial/replay chain and repeated reduction: OK\n");
}

static void run_hc_sum_regression(void * comm) {
    constexpr int width = 256, streams = 4, low_rank = 32, rounds = 6;
    const size_t  ranks = g_backends.size();

    struct consumer {
        ggml_cgraph * producer_graph;
        ggml_cgraph * graph;
        ggml_cgraph * ineligible_graph;
        ggml_tensor * source;
        ggml_tensor * block;
        ggml_tensor * residual;
        ggml_tensor * previous;
        ggml_tensor * gamma;
        ggml_tensor * rms;
        ggml_tensor * combined;
        ggml_tensor * normalized;
        ggml_tensor * output;
        ggml_tensor * ineligible_output;
    };

    std::vector<consumer>                   consumers(ranks);
    std::vector<std::vector<ggml_tensor *>> outputs(2, std::vector<ggml_tensor *>(ranks));
    for (size_t rank = 0; rank < ranks; ++rank) {
        auto * ctx = ggml_init({ 1024 * 1024, nullptr, true });
        TEST_ASSERT(ctx);
        g_allocated_contexts.push_back(ctx);
        auto & p   = consumers[rank];
        p.source   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, width);
        p.block    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, width);
        p.residual = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, streams);
        p.previous = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, width * streams);
        p.gamma    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, width * streams);
        for (auto * t : { p.source, p.block, p.residual, p.previous, p.gamma })
            ggml_set_input(t);
        auto * inject_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, width * streams, streams);
        auto * down_weight   = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, width * streams, low_rank);
        auto * up_weight     = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, low_rank, width * streams);
        // The block stays a leaf in the consumer graph. A real GPU producer
        // writes its allocation, exactly as two separately recorded stages do.
        p.producer_graph     = ggml_new_graph_custom(ctx, 128, false);
        ggml_build_forward_expand(p.producer_graph, ggml_cpy(ctx, ggml_scale(ctx, p.source, 1.0f), p.block));
        auto * injected = ggml_mul_mat(ctx, inject_weight, p.previous);
        auto * inject = ggml_reshape_3d(ctx, ggml_scale(ctx, ggml_sigmoid(ctx, ggml_scale(ctx, injected, 0.25f)), 2.0f),
                                        1, streams, 1);
        auto * expanded = ggml_repeat_4d(ctx, p.block, width, streams, 1, 1);
        p.combined      = ggml_add(ctx, p.residual, ggml_mul(ctx, expanded, inject));
        p.rms           = ggml_rms_norm(ctx, p.combined, 1e-6f);
        p.normalized    = ggml_mul(ctx, ggml_reshape_2d(ctx, p.rms, width * streams, 1), p.gamma);
        auto * lo       = ggml_silu(ctx, ggml_scale(ctx, ggml_mul_mat(ctx, down_weight, p.normalized), 0.25f));
        auto * gated    = ggml_mul(ctx, p.normalized, ggml_sigmoid(ctx, ggml_mul_mat(ctx, up_weight, lo)));
        auto * shaped   = ggml_reshape_3d(ctx, gated, width, streams, 1);
        auto * sum      = ggml_cont(ctx, ggml_view_2d(ctx, shaped, width, 1, shaped->nb[2], 0));
        for (int i = 1; i < streams; ++i) {
            sum = ggml_add(ctx, sum, ggml_view_2d(ctx, shaped, width, 1, shaped->nb[2], i * shaped->nb[1]));
        }
        p.output = ggml_scale(ctx, sum, 0.25f);
        for (auto * t : { p.combined, p.normalized, p.output })
            ggml_set_output(t);
        p.graph = ggml_new_graph_custom(ctx, 128, false);
        ggml_build_forward_expand(p.graph, p.output);

        // Alternate consumer graph for HC eligible -> ineligible transition regression:
        // Takes the SAME producer wire block tensor (outputs[0] == p.block) from stage 0,
        // but uses a replay-eligible decode matmul (weight * block + bias) that lacks
        // the 6-tensor HC pattern, so ggml_vk_tp5_hc_consumer returns false.
        auto * inel_weight  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, width);
        auto * inel_bias    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, width);
        p.ineligible_output = ggml_add(ctx, ggml_mul_mat(ctx, inel_weight, p.block), inel_bias);
        ggml_set_output(p.ineligible_output);
        p.ineligible_graph = ggml_new_graph_custom(ctx, 32, false);
        ggml_build_forward_expand(p.ineligible_graph, p.ineligible_output);

        auto buffer = ggml_backend_alloc_ctx_tensors(ctx, g_backends[rank]);
        TEST_ASSERT(buffer);
        g_allocated_buffers.push_back(buffer);

        std::vector<float> inel_w_diag(width * width, 0.0f);
        for (int i = 0; i < width; ++i) {
            inel_w_diag[i * width + i] = 2.0f;
        }
        ggml_backend_tensor_set(inel_weight, inel_w_diag.data(), 0, inel_w_diag.size() * sizeof(float));
        std::vector<float> inel_b(width);
        for (int i = 0; i < width; ++i) {
            inel_b[i] = float((int(rank + 1) * 7 + i % 19) % 23 - 11) * 0.125f;
        }
        ggml_backend_tensor_set(inel_bias, inel_b.data(), 0, inel_b.size() * sizeof(float));

        int seed = 0;
        for (auto * weight : { inject_weight, down_weight, up_weight }) {
            std::vector<float> values(ggml_nelements(weight));
            for (size_t i = 0; i < values.size(); ++i)
                values[i] = 0.03125f * std::sin(float(i % 991) * 0.073f + seed + rank * 0.13f);
            std::vector<unsigned char> packed(ggml_nbytes(weight));
            TEST_ASSERT(ggml_quantize_chunk(weight->type, values.data(), packed.data(), 0,
                                            ggml_nelements(weight) / weight->ne[0], weight->ne[0],
                                            nullptr) == packed.size());
            ggml_backend_tensor_set(weight, packed.data(), 0, packed.size());
            ++seed;
        }
        outputs[0][rank] = p.block;
        outputs[1][rank] = p.output;
    }
    auto inputs = [&](int round) {
        const float epsilon = round < 3 ? 1e-6f : 1e-3f;
        for (size_t rank = 0; rank < ranks; ++rank) {
            auto & p = consumers[rank];
            std::memcpy(p.rms->op_params, &epsilon, sizeof(epsilon));
            int seed = 0;
            for (auto * t : { p.source, p.residual, p.previous, p.gamma }) {
                std::vector<float> values(ggml_nelements(t));
                for (size_t i = 0; i < values.size(); ++i)
                    values[i] = std::sin(float(i % 199) * 0.13f + round * 0.37f + rank * 0.23f + seed) * 0.75f +
                                (t == p.gamma ? 1.0f : 0.0f);
                fill_tensor(t, values);
                ++seed;
            }
        }
    };
    std::vector<std::vector<std::vector<float>>> intermediate(
        rounds, std::vector<std::vector<float>>(ranks, std::vector<float>(2 * streams * width)));
    std::vector<std::vector<float>> expected(rounds, std::vector<float>(width, 0.0f));
    for (int round = 0; round < rounds; ++round) {
        inputs(round);
        for (size_t rank = 0; rank < ranks; ++rank) {
            auto & p = consumers[rank];
            TEST_ASSERT(g_prepare_graph(comm, rank, p.producer_graph, false));
            TEST_ASSERT(ggml_backend_graph_compute_async(g_backends[rank], p.producer_graph) == GGML_STATUS_SUCCESS);
        }
        TEST_ASSERT(g_allreduce(comm, outputs[0].data()));
        for (size_t rank = 0; rank < ranks; ++rank) {
            auto & p = consumers[rank];
            TEST_ASSERT(g_prepare_graph(comm, rank, p.graph, false));
            TEST_ASSERT(ggml_backend_graph_compute(g_backends[rank], p.graph) == GGML_STATUS_SUCCESS);
            auto & data = intermediate[round][rank];
            ggml_backend_tensor_get(p.combined, data.data(), 0, streams * width * sizeof(float));
            ggml_backend_tensor_get(p.normalized, data.data() + streams * width, 0, streams * width * sizeof(float));
            std::vector<float> values(width);
            read_tensor(p.output, values);
            for (int e = 0; e < width; ++e)
                expected[round][e] += ggml_fp16_to_fp32(ggml_fp32_to_fp16(values[e]));
        }
    }
    for (int round = 0; round < rounds; ++round) {
        inputs(round);
        std::vector<std::vector<std::vector<void *>>> cbs(3, std::vector<std::vector<void *>>(ranks));
        bool                                          cached = round > 0;
        for (size_t stage = 0; stage < 2; ++stage) {
            for (size_t rank = 0; rank < ranks; ++rank) {
                auto * graph = stage == 0 ? consumers[rank].producer_graph : consumers[rank].graph;
                TEST_ASSERT(g_prepare_graph(comm, rank, graph, true));
                if (cached && !g_get_cached_cmd_bufs(g_backends[rank], graph, cbs[stage][rank]))
                    cached = false;
            }
        }
        if (cached) {
            TEST_ASSERT(g_submit_epoch_chain(comm, cbs, outputs));
        } else {
            for (size_t stage = 0; stage < 2; ++stage) {
                for (size_t rank = 0; rank < ranks; ++rank) {
                    auto * graph = stage == 0 ? consumers[rank].producer_graph : consumers[rank].graph;
                    TEST_ASSERT(g_prepare_graph(comm, rank, graph, true));
                    TEST_ASSERT(ggml_backend_graph_compute_async(g_backends[rank], graph) == GGML_STATUS_SUCCESS);
                }
                TEST_ASSERT(g_allreduce(comm, outputs[stage].data()));
            }
        }
        for (size_t rank = 0; rank < ranks; ++rank) {
            ggml_backend_synchronize(g_backends[rank]);
            std::vector<float> data(2 * streams * width), values(width);
            auto &             p = consumers[rank];
            ggml_backend_tensor_get(p.combined, data.data(), 0, streams * width * sizeof(float));
            ggml_backend_tensor_get(p.normalized, data.data() + streams * width, 0, streams * width * sizeof(float));
            TEST_ASSERT(std::memcmp(data.data(), intermediate[round][rank].data(), data.size() * sizeof(float)) == 0);
            read_tensor(p.output, values);
            for (int e = 0; e < width; ++e)
                TEST_ASSERT(std::isfinite(values[e]) && values[e] == expected[round][e]);
        }
    }

    {
        std::vector<std::vector<ggml_tensor *>> inel_outputs(2, std::vector<ggml_tensor *>(ranks));
        for (size_t rank = 0; rank < ranks; ++rank) {
            inel_outputs[0][rank] = consumers[rank].block;
            inel_outputs[1][rank] = consumers[rank].ineligible_output;
        }

        // Precompute references with prepare_graph(false) then warm cache entries with prepare_graph(true)
        const int                       inel_rounds = 2;
        std::vector<std::vector<float>> inel_expected(inel_rounds, std::vector<float>(width, 0.0f));
        for (int r = 0; r < inel_rounds; ++r) {
            inputs(10 + r);
            for (size_t rank = 0; rank < ranks; ++rank) {
                auto & p = consumers[rank];
                TEST_ASSERT(g_prepare_graph(comm, rank, p.producer_graph, false));
                TEST_ASSERT(ggml_backend_graph_compute_async(g_backends[rank], p.producer_graph) ==
                            GGML_STATUS_SUCCESS);
            }
            TEST_ASSERT(g_allreduce(comm, inel_outputs[0].data()));
            for (size_t rank = 0; rank < ranks; ++rank) {
                auto & p = consumers[rank];
                TEST_ASSERT(g_prepare_graph(comm, rank, p.ineligible_graph, false));
                TEST_ASSERT(ggml_backend_graph_compute(g_backends[rank], p.ineligible_graph) == GGML_STATUS_SUCCESS);
                std::vector<float> values(width);
                read_tensor(p.ineligible_output, values);
                for (int e = 0; e < width; ++e) {
                    inel_expected[r][e] += ggml_fp16_to_fp32(ggml_fp32_to_fp16(values[e]));
                }
            }
            TEST_ASSERT(g_allreduce(comm, inel_outputs[1].data()));
        }

        // Record both graphs with the same producer-wire identity used by the chain.
        for (int warmup = 0; warmup < 2; ++warmup) {
            for (size_t rank = 0; rank < ranks; ++rank) {
                for (auto * graph : { consumers[rank].producer_graph, consumers[rank].ineligible_graph }) {
                    TEST_ASSERT(g_prepare_graph(comm, rank, graph, true));
                    TEST_ASSERT(ggml_backend_graph_compute(g_backends[rank], graph) == GGML_STATUS_SUCCESS);
                }
            }
        }

        auto run_ineligible_pass = [&](int r) {
            inputs(10 + r);
            std::vector<std::vector<std::vector<void *>>> cbs(3, std::vector<std::vector<void *>>(ranks));
            for (size_t rank = 0; rank < ranks; ++rank) {
                TEST_ASSERT(g_prepare_graph(comm, rank, consumers[rank].producer_graph, true));
                TEST_ASSERT(g_get_cached_cmd_bufs(g_backends[rank], consumers[rank].producer_graph, cbs[0][rank]));
                TEST_ASSERT(g_prepare_graph(comm, rank, consumers[rank].ineligible_graph, true));
                TEST_ASSERT(g_get_cached_cmd_bufs(g_backends[rank], consumers[rank].ineligible_graph, cbs[1][rank]));
            }
            TEST_ASSERT(g_submit_epoch_chain(comm, cbs, inel_outputs));
            for (size_t rank = 0; rank < ranks; ++rank) {
                ggml_backend_synchronize(g_backends[rank]);
                std::vector<float> values(width);
                read_tensor(consumers[rank].ineligible_output, values);
                for (int e = 0; e < width; ++e) {
                    TEST_ASSERT(std::isfinite(values[e]) && values[e] == inel_expected[r][e]);
                }
            }
        };

        run_ineligible_pass(0);
        run_ineligible_pass(1);

        const int                                    restore_rounds = 2;
        std::vector<std::vector<float>>              hc_expected(restore_rounds, std::vector<float>(width, 0.0f));
        std::vector<std::vector<std::vector<float>>> hc_intermediate(
            restore_rounds, std::vector<std::vector<float>>(ranks, std::vector<float>(2 * streams * width)));
        for (int r = 0; r < restore_rounds; ++r) {
            inputs(20 + r);
            for (size_t rank = 0; rank < ranks; ++rank) {
                auto & p = consumers[rank];
                TEST_ASSERT(g_prepare_graph(comm, rank, p.producer_graph, false));
                TEST_ASSERT(ggml_backend_graph_compute_async(g_backends[rank], p.producer_graph) ==
                            GGML_STATUS_SUCCESS);
            }
            TEST_ASSERT(g_allreduce(comm, outputs[0].data()));
            for (size_t rank = 0; rank < ranks; ++rank) {
                auto & p = consumers[rank];
                TEST_ASSERT(g_prepare_graph(comm, rank, p.graph, false));
                TEST_ASSERT(ggml_backend_graph_compute(g_backends[rank], p.graph) == GGML_STATUS_SUCCESS);
                auto & data = hc_intermediate[r][rank];
                ggml_backend_tensor_get(p.combined, data.data(), 0, streams * width * sizeof(float));
                ggml_backend_tensor_get(p.normalized, data.data() + streams * width, 0,
                                        streams * width * sizeof(float));
                std::vector<float> values(width);
                read_tensor(p.output, values);
                for (int e = 0; e < width; ++e) {
                    hc_expected[r][e] += ggml_fp16_to_fp32(ggml_fp32_to_fp16(values[e]));
                }
            }
            TEST_ASSERT(g_allreduce(comm, outputs[1].data()));
        }

        for (int warmup = 0; warmup < 2; ++warmup) {
            for (size_t rank = 0; rank < ranks; ++rank) {
                for (auto * graph : { consumers[rank].producer_graph, consumers[rank].graph }) {
                    TEST_ASSERT(g_prepare_graph(comm, rank, graph, true));
                    TEST_ASSERT(ggml_backend_graph_compute(g_backends[rank], graph) == GGML_STATUS_SUCCESS);
                }
            }
        }

        auto run_hc_restore_pass = [&](int r) {
            inputs(20 + r);
            std::vector<std::vector<std::vector<void *>>> cbs(3, std::vector<std::vector<void *>>(ranks));
            for (size_t stage = 0; stage < 2; ++stage) {
                for (size_t rank = 0; rank < ranks; ++rank) {
                    auto * graph = stage == 0 ? consumers[rank].producer_graph : consumers[rank].graph;
                    TEST_ASSERT(g_prepare_graph(comm, rank, graph, true));
                    TEST_ASSERT(g_get_cached_cmd_bufs(g_backends[rank], graph, cbs[stage][rank]));
                }
            }
            TEST_ASSERT(g_submit_epoch_chain(comm, cbs, outputs));
            for (size_t rank = 0; rank < ranks; ++rank) {
                ggml_backend_synchronize(g_backends[rank]);
                std::vector<float> data(2 * streams * width), values(width);
                auto &             p = consumers[rank];
                ggml_backend_tensor_get(p.combined, data.data(), 0, streams * width * sizeof(float));
                ggml_backend_tensor_get(p.normalized, data.data() + streams * width, 0,
                                        streams * width * sizeof(float));
                TEST_ASSERT(std::memcmp(data.data(), hc_intermediate[r][rank].data(), data.size() * sizeof(float)) ==
                            0);
                read_tensor(p.output, values);
                for (int e = 0; e < width; ++e) {
                    TEST_ASSERT(std::isfinite(values[e]) && values[e] == hc_expected[r][e]);
                }
            }
        };

        run_hc_restore_pass(0);
        run_hc_restore_pass(1);
    }

    // Retained shared P1CB lifetime & HC plan churn regression:
    // Generate >256 distinct HC sum plan identities (MAX_CACHED_PLANS = 256) while
    // reusing the SAME producer wire input binding (outputs[0] == p.block).
    // Each plan gets a cold record followed by an unchanged replay, forcing
    // HC plan creation, capacity saturation, and plan cache clearing / reclamation.
    // When epoch-chain cache capacity is exceeded, cache overflow clears all cached plans;
    // surviving and rebuilt plans are replayed and verified against the CPU half-sum oracle.
    constexpr int churn_plans       = 288;
    auto          fill_churn_inputs = [&](int p_idx, float eps, float seed_offset) {
        for (size_t rank = 0; rank < ranks; ++rank) {
            auto & p = consumers[rank];
            std::memcpy(p.rms->op_params, &eps, sizeof(eps));
            int seed = 0;
            for (auto * t : { p.source, p.residual, p.previous, p.gamma }) {
                std::vector<float> values(ggml_nelements(t));
                for (size_t i = 0; i < values.size(); ++i) {
                    values[i] =
                        std::sin(float(i % 199) * 0.13f + float(p_idx) * 0.071f + rank * 0.23f + seed + seed_offset) *
                            0.75f +
                        (t == p.gamma ? 1.0f : 0.0f);
                }
                fill_tensor(t, values);
                ++seed;
            }
        }
    };

    auto exercise_churn_case = [&](int p_idx, float seed_offset) {
        const float                     eps = 1e-5f + float(p_idx + 1) * 1e-7f;
        std::vector<std::vector<float>> ref_intermediate(ranks, std::vector<float>(2 * streams * width));
        std::vector<float>              ref_expected(width, 0.0f);

        // 1) Reference pass & recording: builds/records graph in backend cgraph_cmd_cache
        fill_churn_inputs(p_idx, eps, seed_offset);
        for (size_t rank = 0; rank < ranks; ++rank) {
            auto & p = consumers[rank];
            TEST_ASSERT(g_prepare_graph(comm, rank, p.producer_graph, true));
            TEST_ASSERT(ggml_backend_graph_compute_async(g_backends[rank], p.producer_graph) == GGML_STATUS_SUCCESS);
        }
        TEST_ASSERT(g_allreduce(comm, outputs[0].data()));
        for (size_t rank = 0; rank < ranks; ++rank) {
            auto & p = consumers[rank];
            TEST_ASSERT(g_prepare_graph(comm, rank, p.graph, true));
            TEST_ASSERT(ggml_backend_graph_compute(g_backends[rank], p.graph) == GGML_STATUS_SUCCESS);
            auto & data = ref_intermediate[rank];
            ggml_backend_tensor_get(p.combined, data.data(), 0, streams * width * sizeof(float));
            ggml_backend_tensor_get(p.normalized, data.data() + streams * width, 0, streams * width * sizeof(float));
            std::vector<float> values(width);
            read_tensor(p.output, values);
            for (int e = 0; e < width; ++e)
                ref_expected[e] += ggml_fp16_to_fp32(ggml_fp32_to_fp16(values[e]));
        }
        TEST_ASSERT(g_allreduce(comm, outputs[1].data()));

        // 2) Chained execution: retrieves cached CBs and submits epoch chain (creates/replays HC plan in communicator)
        fill_churn_inputs(p_idx, eps, seed_offset);
        std::vector<std::vector<std::vector<void *>>> cbs(3, std::vector<std::vector<void *>>(ranks));
        for (size_t stage = 0; stage < 2; ++stage) {
            for (size_t rank = 0; rank < ranks; ++rank) {
                auto * graph = stage == 0 ? consumers[rank].producer_graph : consumers[rank].graph;
                TEST_ASSERT(g_prepare_graph(comm, rank, graph, true));
                TEST_ASSERT(g_get_cached_cmd_bufs(g_backends[rank], graph, cbs[stage][rank]));
            }
        }
        TEST_ASSERT(g_submit_epoch_chain(comm, cbs, outputs));
        for (size_t rank = 0; rank < ranks; ++rank) {
            ggml_backend_synchronize(g_backends[rank]);
            std::vector<float> data(2 * streams * width), values(width);
            auto &             p = consumers[rank];
            ggml_backend_tensor_get(p.combined, data.data(), 0, streams * width * sizeof(float));
            ggml_backend_tensor_get(p.normalized, data.data() + streams * width, 0, streams * width * sizeof(float));
            TEST_ASSERT(std::memcmp(data.data(), ref_intermediate[rank].data(), data.size() * sizeof(float)) == 0);
            read_tensor(p.output, values);
            for (int e = 0; e < width; ++e)
                TEST_ASSERT(std::isfinite(values[e]) && values[e] == ref_expected[e]);
        }

        // 3) Immediate unchanged replay: exercises collective-plan cache hit for this HC identity
        fill_churn_inputs(p_idx, eps, seed_offset);
        TEST_ASSERT(g_submit_epoch_chain(comm, cbs, outputs));
        for (size_t rank = 0; rank < ranks; ++rank) {
            ggml_backend_synchronize(g_backends[rank]);
            std::vector<float> values(width);
            read_tensor(consumers[rank].output, values);
            for (int e = 0; e < width; ++e)
                TEST_ASSERT(std::isfinite(values[e]) && values[e] == ref_expected[e]);
        }
    };

    for (int p_idx = 0; p_idx < churn_plans; ++p_idx) {
        exercise_churn_case(p_idx, 0.0f);
        if (g_failures > 0)
            return;
    }

    // Revisit early cleared/evicted plan (0) and latest plan (287) with modified data to verify clean rebuild / continued reuse
    exercise_churn_case(0, 0.5f);
    if (g_failures > 0)
        return;
    exercise_churn_case(churn_plans - 1, 0.5f);
    if (g_failures > 0)
        return;

    fprintf(stderr,
            "  HC sum regression: GPU producer, combined/normalized outputs, mutable inputs and RMS epsilon, "
            "replay/fallback, %d-plan churn/reclamation: OK\n",
            churn_plans);
}

int main(int argc, char ** argv) {
    std::vector<int> devices;
    size_t elements = 2560;
    int rounds = 96;
    bool check_all = false, vary = false, delay = false;
    bool adversarial = false;
    int epochs = 1;
    size_t           chain_stages    = 32;
    bool             run_chain_reg   = false;
    bool             benchmark_chain = false;
    std::string wire_str = "f16"; // default matches production collective
    std::string sync_str = "timeline";
    std::string relay_str = "off";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--devices") {
            std::string list = next();
            size_t pos = 0;
            while (pos < list.size()) {
                size_t comma = list.find(',', pos);
                if (comma == std::string::npos) comma = list.size();
                devices.push_back(atoi(list.substr(pos, comma - pos).c_str()));
                pos = comma + 1;
            }
        } else if (a == "--elements") elements = atoll(next().c_str());
        else if (a == "--rounds")     rounds = atoi(next().c_str());
        else if (a == "--wire")       { wire_str = next(); setenv("GGML_TP5_WIRE", wire_str.c_str(), 1); }
        else if (a == "--sync") {
            sync_str = next();
            if (sync_str == "gpu") {
                sync_str = "gpuflag";
            } else if (sync_str == "l3_star") {
                sync_str = "star";
            }
            if (sync_str != "host" && sync_str != "syncfd" && sync_str != "timeline" && sync_str != "star" && sync_str != "gpuflag" && sync_str != "drm") {
                fprintf(stderr, "test-vulkan-tp5-mesh: invalid sync mode '%s'; use 'host', 'syncfd', 'timeline', 'star', 'l3_star', 'gpuflag', or 'drm'\n", sync_str.c_str());
                return 2;
            }
            if (sync_str == "gpuflag") {
                fprintf(stderr, "test-vulkan-tp5-mesh: WARNING: gpuflag is experimental GPU-flag spin sync\n");
            }
            setenv("GGML_TP5_SYNC", sync_str.c_str(), 1);
        }
        else if (a == "--relay") {
            relay_str = next();
            if (relay_str != "off") {
                fprintf(stderr, "test-vulkan-tp5-mesh: invalid relay mode '%s'; host-relay is prohibited, only '--relay off' is allowed\n", relay_str.c_str());
                return 2;
            }
            setenv("GGML_TP5_RELAY", relay_str.c_str(), 1);
        }
        else if (a == "--check-all")  check_all = true;
        else if (a == "--vary-input") vary = true;
        else if (a == "--delay-producer") delay = true;
        else if (a == "--adversarial") adversarial = true;
        else if (a == "--epochs")     epochs = atoi(next().c_str());
        else if (a == "--chain-stages")
            chain_stages = atoll(next().c_str());
        else if (a == "--run-chain-regression")
            run_chain_reg = true;
        else if (a == "--benchmark-chain")
            benchmark_chain = true;
    }
    if (chain_stages == 0 || chain_stages > 128 || elements == 0 || rounds < 0) {
        fprintf(stderr, "test-vulkan-tp5-mesh: require 1..128 chain stages, positive elements, nonnegative rounds\n");
        return 2;
    }
    if ((sync_str == "timeline" || sync_str == "star") && (run_chain_reg || adversarial)) {
        // The cached-compute fixture explicitly exercises replay, which is
        // otherwise opt-in independently of collective-plan replay.
        setenv("GGML_VK_CMD_REPLAY", "1", 1);
    }
    const bool is_f16_wire = (wire_str != "f32");
    if (devices.empty()) {
        for (int i = 0; i < 5; ++i) devices.push_back(i);
    }

    setup(devices);
    const size_t P = devices.size();
    fprintf(stderr, "test-vulkan-tp5-mesh: %zu devices, %zu elements, %d rounds\n",
            P, elements, rounds);

    void * comm = g_comm_init(g_backends.data(), P);
    if (!comm) {
        fprintf(stderr, "comm_init failed\n");
        return 2;
    }

    std::vector<ggml_tensor *> tensors;
    for (size_t j = 0; j < P; ++j) {
        tensors.push_back(alloc_tensor(g_backends[j], (int64_t) elements));
    }

    // Binding B tensors (same element count, independent allocations)
    std::vector<ggml_tensor *> tensors_b;
    for (size_t j = 0; j < P; ++j) {
        tensors_b.push_back(alloc_tensor(g_backends[j], (int64_t) elements));
    }

    // Binding C tensors: views with nonzero aligned byte offset (256 bytes = 64 floats)
    const size_t view_offset_bytes = 256;
    const size_t view_offset_elems = view_offset_bytes / sizeof(float);
    std::vector<ggml_tensor *> tensors_view_parent;
    std::vector<ggml_tensor *> tensors_view;
    for (size_t j = 0; j < P; ++j) {
        ggml_tensor * parent = alloc_tensor(g_backends[j], (int64_t)(elements + view_offset_elems + 64));
        tensors_view_parent.push_back(parent);
        tensors_view.push_back(alloc_tensor_view(parent, (int64_t) elements, view_offset_bytes));
    }

    // Bounded workspace grow / shrink tensors (2x elements and shrink back)
    const size_t elements_large = elements * 2;
    std::vector<ggml_tensor *> tensors_large;
    for (size_t j = 0; j < P; ++j) {
        tensors_large.push_back(alloc_tensor(g_backends[j], (int64_t) elements_large));
    }

    // Real GPU graph producer contexts (out = scale(in, 2.0) + bias)
    std::vector<rank_producer_graph> producers;
    for (size_t j = 0; j < P; ++j) {
        producers.push_back(create_rank_producer(g_backends[j], elements));
    }

    // Bounded GPU delay workloads for --delay-producer
    std::vector<rank_delay_workload> delay_workloads;
    if (delay) {
        for (size_t j = 0; j < P; ++j) {
            delay_workloads.push_back(create_rank_delay_workload(g_backends[j]));
        }
    }

    // Bounded asynchronous timeline overlap step graphs (ping-pong between tensors and tensors_view)
    std::vector<rank_step_graph> step_graphs_0;
    std::vector<rank_step_graph> step_graphs_1;
    if (sync_str == "timeline" || sync_str == "star" || sync_str == "drm") {
        for (size_t j = 0; j < P; ++j) {
            step_graphs_0.push_back(create_rank_step_graph(g_backends[j], tensors_view[j], tensors[j], elements, j, false));
            step_graphs_1.push_back(create_rank_step_graph(g_backends[j], tensors[j], tensors_view[j], elements, j, true));
        }
    }

    const size_t fds_before = count_open_fds();

    // 1) constant inputs
    {
        std::vector<std::vector<float>> rank_results;
        run_round(comm, tensors, elements, 0, false, false, 0, false, rank_results);
        check_result(rank_results, elements, P, 0, false, 0, false, is_f16_wire);
        if (g_failures == 0) fprintf(stderr, "  constant-input round: OK (all == %.1f)\n", rank_results[0].empty() ? -1.f : rank_results[0][0]);
    }

    // 2) varying inputs + multi-round chain
    {
        for (auto b : g_backends) ggml_backend_synchronize(b);
        auto t_start = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < rounds; ++r) {
            std::vector<std::vector<float>> rank_results;
            run_round(comm, tensors, elements, r, vary, delay, 0, false, rank_results, &delay_workloads);
            if (check_all || r == 0 || r == rounds - 1) {
                check_result(rank_results, elements, P, r, vary, 0, false, is_f16_wire);
            }
        }
        for (auto b : g_backends) ggml_backend_synchronize(b);
        auto t_end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        fprintf(stderr, "\n>>> [BENCHMARK] %d AllReduces total: %.3f ms (avg: %.2f us, throughput: %.2f tok/s) <<<\n",
                rounds, ms, (ms / rounds) * 1000.0, 1000.0 / ms);
        if (g_failures == 0) fprintf(stderr, "  %d rounds (vary=%d): OK\n", rounds, (int) vary);
    }

    // 3) real GPU graph-producer rounds (queued compute -> flush_async -> allreduce -> CPU compare)
    // Run at least two successive iterations with changed inputs to catch stale compute_ctx
    // or submit_pending reuse across graph evaluations.
    {
        const int producer_rounds = adversarial ? 8 : 4;
        for (int pr = 0; pr < producer_rounds; ++pr) {
            std::vector<std::vector<float>> rank_results;
            run_producer_round(comm, producers, elements, pr, true, true, rank_results);
            if (g_failures > 0) {
                fprintf(stderr, "  producer round %d failed; aborting producer rounds\n", pr);
                break;
            }
            check_producer_result(rank_results, elements, P, pr, true, true, is_f16_wire);
        }
        if (g_failures == 0) fprintf(stderr, "  %d real GPU graph-producer rounds (async compute->flush->AR): OK\n", producer_rounds);
    }

    // 3.5) asynchronous timeline overlap consumer regression:
    // Bounded sequence of GPU producer -> AllReduce -> GPU consumer/nextproducer with
    // zero host read or synchronize between individual collectives.
    // Reuses existing rank tensors (tensors and nonzero-offset tensors_view) and backend graphs.
    // Final readback compares against explicit CPU oracle across all ranks and elements.
    if (sync_str == "timeline" || sync_str == "star" || sync_str == "drm") {
        const int timeline_steps = 8;
        fprintf(stderr, "  running asynchronous timeline overlap regression (%d dependent steps, %zu elements, zero host-sync)...\n",
                timeline_steps, elements);
        run_timeline_overlap_regression(comm, step_graphs_0, step_graphs_1, tensors, tensors_view, elements, timeline_steps, is_f16_wire);
        if (g_failures == 0) {
            fprintf(stderr, "  asynchronous timeline overlap regression (%d dependent steps): OK\n", timeline_steps);
        }
    }

    // 4) adversarial / durable regression rounds:
    //    - Alternate independent bindings (A, B, and C nonzero offset view) with identical length but distinct data
    //    - Workspace grow (to 2x elements) and shrink (back to 1x elements)
    //    - Exact fp16 vs fp32 wire rounding contract with fractional data (no tolerance loosening)
    //    - Tested across epochs and >256 rounds to detect cmd buffer/descriptor cache reuse corruption
    if (adversarial) {
        fprintf(stderr, "  running adversarial regression suite (%d epochs, alternating bindings A/B/view, grow/shrink, >256 rounds)...\n", epochs);
        const int adv_rounds = (rounds > 288) ? rounds : 288;
        for (int ep = 0; ep < epochs; ++ep) {
            for (int r = 0; r < adv_rounds; ++r) {
                int binding_mode = r % 5;
                std::vector<std::vector<float>> rank_results;
                switch (binding_mode) {
                    case 0: // Binding A (standard buffer)
                        run_round(comm, tensors, elements, r, true, delay, 100 + binding_mode, true, rank_results,
                                  &delay_workloads);
                        check_result(rank_results, elements, P, r, true, 100 + binding_mode, true, is_f16_wire);
                        break;
                    case 1: // Binding B (different independent buffer, same length)
                        run_round(comm, tensors_b, elements, r, true, delay, 100 + binding_mode, true, rank_results,
                                  &delay_workloads);
                        check_result(rank_results, elements, P, r, true, 100 + binding_mode, true, is_f16_wire);
                        break;
                    case 2: // Binding C (view tensor with nonzero 256-byte aligned offset)
                        run_round(comm, tensors_view, elements, r, true, delay, 100 + binding_mode, true, rank_results,
                                  &delay_workloads);
                        check_result(rank_results, elements, P, r, true, 100 + binding_mode, true, is_f16_wire);
                        break;
                    case 3: // Workspace GROW: double elements
                        run_round(comm, tensors_large, elements_large, r, true, delay, 100 + binding_mode, true,
                                  rank_results, &delay_workloads);
                        check_result(rank_results, elements_large, P, r, true, 100 + binding_mode, true, is_f16_wire);
                        break;
                    case 4: // Workspace SHRINK: return back to Binding A
                        run_round(comm, tensors, elements, r, true, delay, 100 + binding_mode, true, rank_results,
                                  &delay_workloads);
                        check_result(rank_results, elements, P, r, true, 100 + binding_mode, true, is_f16_wire);
                        break;
                }
            }

            // Additional graph producer pass within adversarial epoch
            for (int pr = 0; pr < 4; ++pr) {
                std::vector<std::vector<float>> rank_results;
                run_producer_round(comm, producers, elements, ep * 10 + pr, true, true, rank_results);
                check_producer_result(rank_results, elements, P, ep * 10 + pr, true, true, is_f16_wire);
            }
        }
        if (g_failures == 0) fprintf(stderr, "  adversarial regression suite (%d rounds x %d epochs): OK\n", adv_rounds, epochs);
    }

    // 4.5) Epoch-chain regression & paired benchmark:
    // Run explicitly or as part of the adversarial suite.
    if ((sync_str == "timeline" || sync_str == "star" || sync_str == "drm") && (run_chain_reg || adversarial)) {
        fprintf(stderr, "\n--- Starting Epoch-Chain Regression Suite ---\n");
        run_epoch_chain_regression(comm, g_backends, is_f16_wire);
        if (g_failures == 0)
            run_cached_compute_chain_regression(comm, is_f16_wire);
        if (g_failures == 0 && is_f16_wire)
            run_producer_wire_regression(comm);
        if (g_failures == 0 && is_f16_wire)
            run_hc_sum_regression(comm);
    }

    if (benchmark_chain && (sync_str == "timeline" || sync_str == "star" || sync_str == "drm") && g_failures == 0) {
        run_paired_chain_benchmark(comm, g_backends, elements, chain_stages, is_f16_wire);
    }

    // 5) resource accounting
    {
        const size_t fds_after = count_open_fds();
        fprintf(stderr, "  open fds: before=%zu after=%zu (delta=%zd)\n",
                fds_before, fds_after, (ssize_t)(fds_after - fds_before));
        if ((ssize_t)(fds_after - fds_before) > 64) {
            fprintf(stderr, "FAIL: fd leak suspected\n");
            g_failures++;
        }
    }

    // Synchronize all backends before freeing buffers and contexts to ensure no GPU work is inflight
    for (auto b : g_backends) {
        ggml_backend_synchronize(b);
    }

    // Cleanup all test-allocated contexts, buffers, and producers before destroying comm and backends
    for (auto & rpg : producers) {
        if (rpg.buf) ggml_backend_buffer_free(rpg.buf);
        if (rpg.ctx) ggml_free(rpg.ctx);
    }
    producers.clear();

    for (auto & rsg : step_graphs_0) {
        if (rsg.buf) ggml_backend_buffer_free(rsg.buf);
        if (rsg.ctx) ggml_free(rsg.ctx);
    }
    step_graphs_0.clear();

    for (auto & rsg : step_graphs_1) {
        if (rsg.buf) ggml_backend_buffer_free(rsg.buf);
        if (rsg.ctx) ggml_free(rsg.ctx);
    }
    step_graphs_1.clear();

    for (auto & dw : delay_workloads) {
        if (dw.buf)
            ggml_backend_buffer_free(dw.buf);
        if (dw.ctx)
            ggml_free(dw.ctx);
    }
    delay_workloads.clear();

    for (auto buf : g_allocated_buffers) {
        ggml_backend_buffer_free(buf);
    }
    g_allocated_buffers.clear();

    for (auto ctx : g_allocated_contexts) {
        ggml_free(ctx);
    }
    g_allocated_contexts.clear();

    g_comm_free(comm);
    for (auto b : g_backends) ggml_backend_free(b);

    if (g_failures > 0) {
        fprintf(stderr, "test-vulkan-tp5-mesh: %d FAILURES\n", g_failures);
        return 1;
    }
    printf("test-vulkan-tp5-mesh: all passed\n");
    return 0;
}
