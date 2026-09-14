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
//                              [--rounds 96] [--wire f32|f16] [--sync host|syncfd|timeline]
//                              [--check-all] [--vary-input] [--delay-producer]
//                              [--adversarial] [--epochs 1]

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


static void run_round(void * comm, std::vector<ggml_tensor *> & tensors,
                      size_t n_elems, int round, bool vary, bool delay_producer,
                      int binding_id, bool fractional,
                      std::vector<std::vector<float>> & all_rank_results) {
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

    (void) delay_producer;
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

static rank_step_graph create_rank_step_graph(ggml_backend_t backend, ggml_tensor * in_tensor,
                                             ggml_tensor * dst_tensor, size_t n_elems,
                                             size_t rank, bool is_odd) {
    rank_step_graph rsg;
    size_t mem_needed = ggml_tensor_overhead() * 8 + ggml_graph_overhead_custom(16, false) + 4096;
    ggml_init_params ip{mem_needed, nullptr, true};
    rsg.ctx = ggml_init(ip);
    rsg.in = in_tensor;
    rsg.bias = ggml_new_tensor_1d(rsg.ctx, GGML_TYPE_F32, (int64_t) n_elems);
    rsg.scaled = ggml_scale(rsg.ctx, rsg.in, 0.125f);
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

int main(int argc, char ** argv) {
    std::vector<int> devices;
    size_t elements = 2560;
    int rounds = 96;
    bool check_all = false, vary = false, delay = false;
    bool adversarial = false;
    int epochs = 1;
    std::string wire_str = "f16"; // default matches production collective
    std::string sync_str = "host";
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
            if (sync_str == "gpu" || sync_str == "gpuflag") {
                fprintf(stderr, "test-vulkan-tp5-mesh: sync mode '%s' (GPU spin polling) is unsafe and rejected; use 'host', 'syncfd', or 'timeline'\n", sync_str.c_str());
                return 2;
            }
            if (sync_str != "host" && sync_str != "syncfd" && sync_str != "timeline") {
                fprintf(stderr, "test-vulkan-tp5-mesh: invalid sync mode '%s'; only 'host', 'syncfd', or 'timeline' allowed\n", sync_str.c_str());
                return 2;
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

    // Bounded asynchronous timeline overlap step graphs (ping-pong between tensors and tensors_view)
    std::vector<rank_step_graph> step_graphs_0;
    std::vector<rank_step_graph> step_graphs_1;
    if (sync_str == "timeline") {
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
        for (int r = 0; r < rounds; ++r) {
            std::vector<std::vector<float>> rank_results;
            run_round(comm, tensors, elements, r, vary, delay, 0, false, rank_results);
            if (check_all || r == 0 || r == rounds - 1) {
                check_result(rank_results, elements, P, r, vary, 0, false, is_f16_wire);
            }
        }
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
            check_producer_result(rank_results, elements, P, pr, true, true, is_f16_wire);
        }
        if (g_failures == 0) fprintf(stderr, "  %d real GPU graph-producer rounds (async compute->flush->AR): OK\n", producer_rounds);
    }

    // 3.5) asynchronous timeline overlap consumer regression:
    // Bounded sequence of GPU producer -> AllReduce -> GPU consumer/nextproducer with
    // zero host read or synchronize between individual collectives.
    // Reuses existing rank tensors (tensors and nonzero-offset tensors_view) and backend graphs.
    // Final readback compares against explicit CPU oracle across all ranks and elements.
    if (sync_str == "timeline") {
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
                        run_round(comm, tensors, elements, r, true, delay, 100 + binding_mode, true, rank_results);
                        check_result(rank_results, elements, P, r, true, 100 + binding_mode, true, is_f16_wire);
                        break;
                    case 1: // Binding B (different independent buffer, same length)
                        run_round(comm, tensors_b, elements, r, true, delay, 100 + binding_mode, true, rank_results);
                        check_result(rank_results, elements, P, r, true, 100 + binding_mode, true, is_f16_wire);
                        break;
                    case 2: // Binding C (view tensor with nonzero 256-byte aligned offset)
                        run_round(comm, tensors_view, elements, r, true, delay, 100 + binding_mode, true, rank_results);
                        check_result(rank_results, elements, P, r, true, 100 + binding_mode, true, is_f16_wire);
                        break;
                    case 3: // Workspace GROW: double elements
                        run_round(comm, tensors_large, elements_large, r, true, delay, 100 + binding_mode, true, rank_results);
                        check_result(rank_results, elements_large, P, r, true, 100 + binding_mode, true, is_f16_wire);
                        break;
                    case 4: // Workspace SHRINK: return back to Binding A
                        run_round(comm, tensors, elements, r, true, delay, 100 + binding_mode, true, rank_results);
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
