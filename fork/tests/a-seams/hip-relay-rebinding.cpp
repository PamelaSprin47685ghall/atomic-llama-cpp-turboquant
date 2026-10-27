#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-relay-impl.h"

#include <array>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
using clock_type = std::chrono::steady_clock;
constexpr size_t ranks = GGML_RELAY_RANKS;
static_assert(ranks == 5, "this fixture requires the five-rank provider ABI");
constexpr size_t width = 257; // Exercise the relay's vector body and scalar tail.
constexpr uint32_t all_ranks = (uint32_t(1) << ranks) - 1;

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool same_float(float a, float b) {
    uint32_t ua, ub;
    std::memcpy(&ua, &a, sizeof(ua));
    std::memcpy(&ub, &b, sizeof(ub));
    return ua == ub;
}

float input_value(size_t epoch, size_t rank, size_t element) {
    return float(100 * (epoch + 1) + 10 * (rank + 1) + element % 7);
}

float terminal_value(size_t epoch, size_t rank, size_t element, bool select_b) {
    return input_value(epoch, rank, element) * (select_b ? 20.0f : 2.0f);
}

float ordered_sum(const ggml_relay_event & event, size_t element) {
    float sum = event.inputs[0][element];
    for (size_t r = 1; r < ranks; ++r) {
        volatile float next = sum + event.inputs[r][element];
        sum = next;
    }
    return sum;
}

// The watchdog never reads provider pointers. A stuck native call must not
// unwind and free buffers whose last native user has not been proved complete.
class deadline_guard {
    std::mutex mutex_;
    std::condition_variable changed_;
    bool done_ = false;
    std::thread thread_;
public:
    explicit deadline_guard(clock_type::time_point deadline) : thread_([this, deadline] {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!changed_.wait_until(lock, deadline, [this] { return done_; })) {
            // Do not wait for an iostream/stdio lock or a full diagnostic pipe.
            std::_Exit(124);
        }
    }) {}
    ~deadline_guard() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            done_ = true;
        }
        changed_.notify_one();
        thread_.join();
    }
};

struct rank_data {
    ggml_backend_t backend = nullptr;
    ggml_context * context = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * a = nullptr;
    ggml_tensor * b = nullptr;
    ggml_cgraph * first = nullptr;
    ggml_cgraph * last = nullptr;
};

struct fixture {
    std::array<rank_data, ranks> rank{};
    std::array<ggml_backend_t, ranks> backends{};
    std::array<ggml_cgraph *, 2 * ranks> graphs{};
    const ggml_relay_provider_i * api = nullptr;
    void * provider = nullptr;
    bool create_attempted = false;
    bool borrowed = false;
    ggml_relay_event borrow{};
    size_t input_checks = 0;
    size_t selected_checks = 0;
    size_t other_checks = 0;

    void open() {
        ggml_backend_reg_t chosen = nullptr;
        size_t count = 0;
        for (size_t i = 0; i < ggml_backend_dev_count() && count < ranks; ++i) {
            auto * device = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_GPU) {
                continue;
            }
            auto * registry = ggml_backend_dev_backend_reg(device);
            if (chosen && registry != chosen) {
                continue;
            }
            auto getter = reinterpret_cast<ggml_backend_relay_get_interface_t>(
                ggml_backend_reg_get_proc_address(registry, "ggml_backend_relay_get_interface"));
            if (!getter) {
                continue;
            }
            const auto * candidate = getter();
            require(candidate && candidate->abi_version == GGML_RELAY_PROVIDER_ABI_VERSION,
                    "provider ABI mismatch");
            if (!chosen) {
                chosen = registry;
                api = candidate;
            }
            require(candidate == api, "devices returned different provider interfaces");
            auto & r = rank[count];
            r.backend = ggml_backend_dev_init(device, nullptr);
            require(r.backend != nullptr, "creating a fresh HIP backend failed");
            backends[count] = r.backend;
            std::cerr << "rank " << count << ": " << ggml_backend_dev_name(device) << '\n';
            ++count;
        }
        require(count == ranks, "this opt-in A test requires five qualified HIP devices");
        require(api && api->create && api->prepare && api->submit && api->poll &&
                api->publish && api->snapshot && api->abort && api->destroy, "incomplete provider interface");

        for (size_t i = 0; i < ranks; ++i) {
            auto & r = rank[i];
            ggml_init_params params{};
            params.mem_size = 16 * ggml_tensor_overhead() + 2 * ggml_graph_overhead_custom(16, false);
            params.no_alloc = true;
            r.context = ggml_init(params);
            require(r.context != nullptr, "creating tensor metadata failed");
            r.input = ggml_new_tensor_1d(r.context, GGML_TYPE_F32, width);
            ggml_set_input(r.input);
            ggml_set_name(r.input, "rebind_input");
            r.a = ggml_scale(r.context, r.input, 2.0f);
            r.b = ggml_scale(r.context, r.input, 20.0f);
            ggml_set_name(r.a, "rebind_a");
            ggml_set_name(r.b, "rebind_b");
            ggml_set_output(r.a);
            ggml_set_output(r.b);
            r.a->flags |= GGML_TENSOR_FLAG_COMPUTE;
            r.b->flags |= GGML_TENSOR_FLAG_COMPUTE;
            r.first = ggml_new_graph_custom(r.context, 16, false);
            ggml_build_forward_expand(r.first, r.a);
            ggml_build_forward_expand(r.first, r.b);
            r.last = ggml_new_graph_custom(r.context, 16, false);
            require(ggml_graph_n_nodes(r.first) == 2 && ggml_graph_n_nodes(r.last) == 0,
                    "expected one fixed two-output graph and an empty final stage");
            r.buffer = ggml_backend_alloc_ctx_tensors(r.context, r.backend);
            require(r.buffer && r.a->data && r.b->data && r.a->data != r.b->data,
                    "allocating distinct rank-local terminals failed");
            graphs[i] = r.first;
            graphs[ranks + i] = r.last;
        }

        // Tensor allocation did not execute a graph or use a child scratch pool.
        std::cerr << "phase provider_create\n";
        create_attempted = true;
        provider = api->create(backends.data(), ranks);
        require(provider != nullptr, "creating the five-rank HIP provider failed");
        std::cerr << "phase provider_created\n";
    }

    void prepare(bool select_b) {
        std::cerr << "phase prepare " << (select_b ? "B" : "A") << '\n';
        std::array<ggml_tensor *, ranks> terminals{};
        for (size_t r = 0; r < ranks; ++r) {
            terminals[r] = select_b ? rank[r].b : rank[r].a;
        }
        require(api->prepare(provider, terminals.data(), 1), "preparing terminal binding failed");
    }

    void run_epoch(size_t epoch, bool select_b, clock_type::time_point deadline) {
        std::cerr << "phase epoch " << epoch + 1 << " inputs\n";
        std::array<float, width> values{};
        std::array<float, width> expected{};
        for (size_t r = 0; r < ranks; ++r) {
            for (size_t j = 0; j < width; ++j) {
                values[j] = input_value(epoch, r, j);
            }
            // Contents alone change. Graphs and all tensor metadata are fixed.
            ggml_backend_tensor_set(rank[r].input, values.data(), 0, sizeof(values));
        }
        std::cerr << "phase epoch " << epoch + 1 << " submit\n";
        require(api->submit(provider, graphs.data(), 2), "submitting a complete epoch failed");
        std::cerr << "phase epoch " << epoch + 1 << " service\n";
        bool published = false;
        bool released = false;
        ggml_relay_status final_status{};
        for (;;) {
            require(clock_type::now() < deadline, "epoch deadline exceeded");
            ggml_relay_event event{};
            const int result = api->poll(provider, &event);
            require(result >= 0, "provider poll reported failure");
            if (result == 1) {
                if (event.kind == GGML_RELAY_INPUT_READY) {
                    borrow = event;
                    borrowed = true;
                    require(!published && event.abi_version == GGML_RELAY_ABI_VERSION &&
                            event.epoch == epoch + 1 && event.channel == 0 &&
                            event.ranks == ranks && event.elements == width && event.output,
                            "unexpected or repeated input-ready view");
                    for (size_t r = 0; r < ranks; ++r) {
                        require(event.inputs[r] != nullptr, "missing rank input");
                        for (size_t j = 0; j < width; ++j) {
                            const float wanted = terminal_value(epoch, r, j, select_b);
                            if (!same_float(event.inputs[r][j], wanted)) {
                                throw std::runtime_error(
                                    "binding input mismatch: epoch=" + std::to_string(epoch + 1) +
                                    " selected=" + (select_b ? "B" : "A") +
                                    " rank=" + std::to_string(r) + " element=" + std::to_string(j));
                            }
                            ++input_checks;
                        }
                    }
                    for (size_t j = 0; j < width; ++j) {
                        expected[j] = ordered_sum(event, j);
                        event.output[j] = expected[j];
                    }
                    const bool ok = api->publish(provider, &event);
                    // Matching publish returns the borrow even after abort.
                    borrowed = false;
                    require(ok, "publishing the CPU ordered F32 fold failed");
                    published = true;
                } else if (event.kind == GGML_RELAY_OUTPUT_RELEASED) {
                    require(published && !released && event.epoch == epoch + 1 &&
                            event.channel == 0 && event.ranks == ranks && event.elements == width,
                            "unexpected or repeated output release");
                    released = true;
                } else {
                    throw std::runtime_error("unknown provider event");
                }
            }

            ggml_relay_status status{};
            if (api->snapshot(provider, &status)) {
                require(!status.failed, std::string("provider failure: ") + status.error);
                if (status.finished && published && released) {
                    final_status = status;
                    break;
                }
            }
            if (result == 0) {
                std::this_thread::yield();
            }
        }
        require(final_status.epoch == epoch + 1 && final_status.channels == 1 &&
                final_status.sealed && final_status.finished &&
                final_status.attempted_mask == all_ranks && final_status.accepted_mask == all_ranks &&
                final_status.completed_mask == all_ranks && final_status.unknown_mask == 0 &&
                final_status.cpu_reductions == epoch + 1, "incomplete native/CPU completion proof");

        // Native completion and the CPU borrow return were proved before reads.
        std::cerr << "phase epoch " << epoch + 1 << " readback\n";
        for (size_t r = 0; r < ranks; ++r) {
            ggml_backend_tensor_get(select_b ? rank[r].b : rank[r].a, values.data(), 0, sizeof(values));
            for (size_t j = 0; j < width; ++j) {
                require(same_float(values[j], expected[j]), "selected terminal was not written with the fold");
                ++selected_checks;
            }
            ggml_backend_tensor_get(select_b ? rank[r].a : rank[r].b, values.data(), 0, sizeof(values));
            for (size_t j = 0; j < width; ++j) {
                require(same_float(values[j], terminal_value(epoch, r, j, !select_b)),
                        "unselected terminal was overwritten");
                ++other_checks;
            }
        }
    }

    void close() {
        // The last output event has already been delivered. The idle-poll fast
        // path must still observe an abort and retain the native drain proof.
        require(!borrowed, "idle abort still has a CPU borrow");
        api->abort(provider, ECANCELED);
        ggml_relay_event event{};
        require(api->poll(provider, &event) < 0, "idle poll hid the abort");
        ggml_relay_status status{};
        require(api->snapshot(provider, &status) && status.failed && status.finished &&
                status.completed_mask == all_ranks && status.unknown_mask == 0,
                "idle abort lost the existing native drain proof");
        require(api->destroy(provider), "provider refused completed destruction");
        provider = nullptr;
        free_consumers();
    }

    void free_consumers() {
        for (auto & r : rank) {
            if (r.buffer) {
                ggml_backend_buffer_free(r.buffer);
                r.buffer = nullptr;
            }
            if (r.context) {
                ggml_free(r.context);
                r.context = nullptr;
            }
            if (r.backend) {
                ggml_backend_free(r.backend);
                r.backend = nullptr;
            }
        }
    }

    void fail_cleanup() {
        if (create_attempted && !provider) {
            // A failed create may retain a partially initialized native owner.
            std::fputs("provider creation/cleanup failed; exiting without freeing possible native owners\n", stderr);
            std::fflush(stderr);
            std::_Exit(1);
        }
        if (provider) {
            api->abort(provider, EIO);
            if (borrowed) {
                (void) api->publish(provider, &borrow);
                borrowed = false;
            }
            const auto deadline = clock_type::now() + std::chrono::seconds(5);
            bool destroyed = false;
            while (clock_type::now() < deadline) {
                ggml_relay_status status{};
                if (api->snapshot(provider, &status) && status.finished &&
                    status.unknown_mask == 0 &&
                    (status.completed_mask & status.attempted_mask) == status.attempted_mask &&
                    api->destroy(provider)) {
                    provider = nullptr;
                    destroyed = true;
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            if (!destroyed) {
                std::fputs("native drain unproved; exiting without freeing live owners\n", stderr);
                std::fflush(stderr);
                std::_Exit(1);
            }
        }
        free_consumers();
    }
};
} // namespace

int main(int argc, char ** argv) {
    int seconds = 120;
    const char * backend_dir = nullptr;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--timeout-seconds" && i + 1 < argc) {
            char * end = nullptr;
            const long value = std::strtol(argv[++i], &end, 10);
            if (!end || *end || value < 1 || value > 600) {
                std::cerr << "invalid timeout\n";
                return 2;
            }
            seconds = int(value);
        } else if (arg == "--backend-dir" && i + 1 < argc) {
            backend_dir = argv[++i];
        } else {
            std::cerr << "usage: test-hip-relay-rebinding [--backend-dir PATH] [--timeout-seconds 1..600]\n";
            return 2;
        }
    }
    const auto deadline = clock_type::now() + std::chrono::seconds(seconds);
    deadline_guard watchdog(deadline + std::chrono::seconds(6));
    fixture test;
    try {
        if (backend_dir) {
            ggml_backend_load_all_from_path(backend_dir);
        } else {
            ggml_backend_load_all();
        }
        test.open();
        size_t epoch = 0;
        for (const bool select_b : {false, true, false}) {
            test.prepare(select_b);
            for (int repeat = 0; repeat < 3; ++repeat) {
                test.run_epoch(epoch++, select_b, deadline);
            }
        }
        test.close();
        std::cout << "{\"status\":\"PASS\",\"ranks\":5,\"channels\":1,\"epochs\":9,"
                     "\"binding_order\":\"A-B-A\",\"epochs_per_binding\":3,\"elements\":257,"
                     "\"input_bitwise_checks\":" << test.input_checks <<
                     ",\"selected_terminal_bitwise_checks\":" << test.selected_checks <<
                     ",\"other_terminal_bitwise_checks\":" << test.other_checks <<
                     ",\"cache_hit_evidence\":\"not_exposed_by_provider_api\","
                     "\"scope\":\"internal_A_provider_terminal_rebinding\"}\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "hip-relay-rebinding FAIL: " << e.what() << '\n';
        test.fail_cleanup();
        return 1;
    }
}
