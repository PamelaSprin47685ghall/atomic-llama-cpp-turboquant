// Narrow CPU behavior for llama_kv_cache::update's status protocol.
//
// No production fault hook. A real CPU context and a dense or hybrid GGUF
// drive three observable paths:
//   * a decode with no pending shift -> NO_UPDATE path (decode succeeds);
//   * seq_add followed by decode -> APPLIED path (K-shift graph runs, decode
//     succeeds);
//   * a controllable CPU backend device fails one K-shift graph_compute ->
//     FAILED path, decode returns -3, the failed ubatch positions are removed
//     by the decode cleanup, state get/set are refused, and clear(true)
//     recovers. State buffers are sized by llama_state_seq_get_size, not a
//     fixed 1 MiB, so hybrid recurrent state is not truncated.
//
// The enum values are observed through decode/state behavior; the test does
// not call a status accessor.
//
// The backend wrapper is a real CPU backend substitute for the model/context,
// not a source-level test hook.
//
// usage: test-kv-update-status <dense-or-hybrid.gguf>

#include "llama.h"

#include "llama-context.h"
#include "llama-ext.h"

#include "ggml-backend-impl.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int g_checks = 0;
static int g_fail   = 0;
static bool g_skip  = false;

#define CHECK(cond, ...) do { \
    ++g_checks; \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
        std::fprintf(stderr, __VA_ARGS__); \
        std::fprintf(stderr, "\n"); \
        ++g_fail; \
    } \
} while (0)

static llama_context * make_ctx(llama_model * model) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = 64;
    cp.n_batch         = 4;
    cp.n_ubatch        = 4;
    cp.n_outputs_max   = 4;
    cp.n_seq_max       = 1;
    cp.n_threads       = 1;
    cp.n_threads_batch = 1;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.offload_kqv     = false;
    cp.op_offload      = false;
    cp.no_perf         = true;
    cp.kv_unified      = true;
    return llama_init_from_model(model, cp);
}

static llama_context * make_ctx_split(llama_model * model) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = 64;
    cp.n_batch         = 4;
    cp.n_ubatch        = 4;
    cp.n_outputs_max   = 4;
    cp.n_seq_max       = 2;
    cp.n_threads       = 1;
    cp.n_threads_batch = 1;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.offload_kqv     = false;
    cp.op_offload      = false;
    cp.no_perf         = true;
    cp.kv_unified      = false;
    return llama_init_from_model(model, cp);
}

static int decode_one_rc(llama_context * ctx, llama_token tok) {
    llama_batch b = llama_batch_get_one(&tok, 1);
    return llama_decode(ctx, b);
}

static bool decode_one(llama_context * ctx, llama_token tok) {
    return decode_one_rc(ctx, tok) == 0;
}

static bool logits_finite(llama_context * ctx, int n_vocab) {
    const float * logits = llama_get_logits_ith(ctx, 0);
    if (logits == nullptr) {
        return false;
    }
    for (int i = 0; i < n_vocab; ++i) {
        if (!std::isfinite(logits[i])) {
            return false;
        }
    }
    return true;
}


// A real CPU backend wrapped in a controllable device/backend. The model and
// context use the production llama_context path; only graph_compute can be
// told to fail once, so the K-shift FAILED branch is exercised without a
// production hook or a mock echo.
struct mock_device_state {
    ggml_backend_dev_t cpu_dev = nullptr;
    bool fail_next_compute = false;
};

struct mock_backend_state {
    ggml_backend_t cpu = nullptr;
};

static mock_device_state  g_mock_dev_state;
static ggml_backend_device g_mock_device;
static ggml_backend_reg    g_mock_reg;

static const char * mock_dev_name(ggml_backend_dev_t) { return "mock-cpu"; }
static const char * mock_dev_desc(ggml_backend_dev_t) { return "mock CPU wrapper"; }
static void mock_dev_memory(ggml_backend_dev_t, size_t * free, size_t * total) {
    *free = 0; *total = 0;
}
static enum ggml_backend_dev_type mock_dev_type(ggml_backend_dev_t) {
    return GGML_BACKEND_DEVICE_TYPE_CPU;
}
static void mock_dev_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    ggml_backend_dev_get_props(g_mock_dev_state.cpu_dev, props);
    props->name        = mock_dev_name(dev);
    props->description = mock_dev_desc(dev);
}
static const char * mock_be_name(ggml_backend_t) { return "mock-cpu-be"; }
static void mock_be_free(ggml_backend_t backend) {
    auto * st = static_cast<mock_backend_state *>(backend->context);
    if (st != nullptr) {
        if (st->cpu != nullptr) ggml_backend_free(st->cpu);
        delete st;
    }
    std::free(backend);
}
static void mock_be_set_async(ggml_backend_t, ggml_tensor * t, const void * data, size_t off, size_t size) {
    ggml_backend_tensor_set(t, data, off, size);
}
static void mock_be_get_async(ggml_backend_t, const ggml_tensor * t, void * data, size_t off, size_t size) {
    ggml_backend_tensor_get(t, data, off, size);
}
static void mock_be_sync(ggml_backend_t backend) {
    auto * st = static_cast<mock_backend_state *>(backend->context);
    ggml_backend_synchronize(st->cpu);
}
static enum ggml_status mock_be_compute(ggml_backend_t backend, ggml_cgraph * gf) {
    auto * st = static_cast<mock_backend_state *>(backend->context);
    if (g_mock_dev_state.fail_next_compute) {
        g_mock_dev_state.fail_next_compute = false;
        return GGML_STATUS_FAILED;
    }
    return ggml_backend_graph_compute(st->cpu, gf);
}
static ggml_backend_t mock_dev_init(ggml_backend_dev_t dev, const char *) {
    auto * st = new mock_backend_state();
    st->cpu = ggml_backend_cpu_init();
    if (st->cpu == nullptr) {
        delete st;
        return nullptr;
    }
    auto * backend = static_cast<ggml_backend_t>(std::calloc(1, sizeof(ggml_backend)));
    if (backend == nullptr) {
        ggml_backend_free(st->cpu);
        delete st;
        return nullptr;
    }
    static ggml_guid guid = { 0 };
    backend->guid = &guid;
    backend->device = dev;
    backend->context = st;
    backend->iface.get_name = mock_be_name;
    backend->iface.free = mock_be_free;
    backend->iface.set_tensor_async = mock_be_set_async;
    backend->iface.get_tensor_async = mock_be_get_async;
    backend->iface.synchronize = mock_be_sync;
    backend->iface.graph_compute = mock_be_compute;
    return backend;
}
static ggml_backend_buffer_type_t mock_dev_buft(ggml_backend_dev_t) {
    return ggml_backend_dev_buffer_type(g_mock_dev_state.cpu_dev);
}
static ggml_backend_buffer_type_t mock_dev_host_buft(ggml_backend_dev_t) {
    return ggml_backend_dev_host_buffer_type(g_mock_dev_state.cpu_dev);
}
static ggml_backend_buffer_t mock_dev_buffer_from_host(ggml_backend_dev_t, void * ptr, size_t size, size_t max) {
    return ggml_backend_dev_buffer_from_host_ptr(g_mock_dev_state.cpu_dev, ptr, size, max);
}
static bool mock_dev_supports_op(ggml_backend_dev_t, const ggml_tensor * op) {
    return ggml_backend_dev_supports_op(g_mock_dev_state.cpu_dev, op);
}
static bool mock_dev_supports_buft(ggml_backend_dev_t, ggml_backend_buffer_type_t buft) {
    return ggml_backend_dev_supports_buft(g_mock_dev_state.cpu_dev, buft);
}
static bool mock_dev_offload_op(ggml_backend_dev_t, const ggml_tensor *) { return false; }

static void init_mock_device() {
    g_mock_dev_state.cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    g_mock_reg = {};
    g_mock_reg.api_version = GGML_BACKEND_API_VERSION;
    g_mock_device = {};
    g_mock_device.reg = &g_mock_reg;
    g_mock_device.context = &g_mock_dev_state;
    g_mock_device.iface.get_name = mock_dev_name;
    g_mock_device.iface.get_description = mock_dev_desc;
    g_mock_device.iface.get_memory = mock_dev_memory;
    g_mock_device.iface.get_type = mock_dev_type;
    g_mock_device.iface.get_props = mock_dev_props;
    g_mock_device.iface.init_backend = mock_dev_init;
    g_mock_device.iface.get_buffer_type = mock_dev_buft;
    g_mock_device.iface.get_host_buffer_type = mock_dev_host_buft;
    g_mock_device.iface.buffer_from_host_ptr = mock_dev_buffer_from_host;
    g_mock_device.iface.supports_op = mock_dev_supports_op;
    g_mock_device.iface.supports_buft = mock_dev_supports_buft;
    g_mock_device.iface.offload_op = mock_dev_offload_op;
}

static void run_backend_failure(const char * model_path) {
    init_mock_device();
    if (g_mock_dev_state.cpu_dev == nullptr) {
        CHECK(false, "CPU device unavailable");
        return;
    }

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    ggml_backend_dev_t devices[] = { &g_mock_device, nullptr };
    mp.devices = devices;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    CHECK(model != nullptr, "mock-device model load");
    if (model == nullptr) {
        return;
    }

    llama_context * ctx = make_ctx_split(model);
    CHECK(ctx != nullptr, "mock-device split ctx");
    if (ctx == nullptr) {
        llama_model_free(model);
        return;
    }

    const auto initial_host = llama_get_host_memory_breakdown(ctx);
    CHECK(initial_host.journal == 0, "host account starts with an empty journal");

    if (!llama_memory_can_shift(llama_get_memory(ctx))) {
        CHECK(decode_one(ctx, 1), "host-account hybrid decode");
        const auto host = llama_get_host_memory_breakdown(ctx);
        CHECK(host.journal > 0, "host account aggregates the hybrid journal");
        // A model whose cache cannot shift cannot reach the K-shift FAILED
        // path. Do not run the core checks or print PASS for it; report a
        // skip so a dense regression cannot hide behind this fixture.
        std::fprintf(stderr, "SKIP: memory cannot shift; K-shift test not applicable\n");
        g_skip = true;
        llama_free(ctx);
        llama_model_free(model);
        return;
    }

    CHECK(decode_one(ctx, 1), "mock decode 1");
    const auto host = llama_get_host_memory_breakdown(ctx);
    CHECK(host.journal > 0, "host account exposes the dense journal");
    CHECK(decode_one(ctx, 2), "mock decode 2");
    const llama_pos pos_before = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
    // Size every buffer by the actual sequence state. A fixed 1 MiB buffer is
    // smaller than a qwen4exp hybrid recurrent state and would fail at the
    // baseline, before the K-shift failure under test.
    const size_t valid_cap = llama_state_seq_get_size(ctx, 0);
    std::vector<uint8_t> valid_state(valid_cap > 0 ? valid_cap : 1);
    const size_t valid_size = llama_state_seq_get_data(ctx, valid_state.data(), valid_state.size(), 0);
    CHECK(valid_size > 0, "baseline state get");

    // seq 1 has no cells; capture its own empty-state blob so the post-failure
    // set_data would succeed without the entry guard and must fail with it.
    const size_t empty_cap = llama_state_seq_get_size(ctx, 1);
    std::vector<uint8_t> empty_state(empty_cap > 0 ? empty_cap : 1);
    const size_t empty_size = llama_state_seq_get_data(ctx, empty_state.data(), empty_state.size(), 1);
    CHECK(empty_size > 0, "baseline empty-stream state get");

    llama_memory_seq_add(llama_get_memory(ctx), 0, 0, -1, 1);
    const llama_pos pos_after_add = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
    CHECK(pos_after_add != pos_before, "shift-capable fixture must move positions");
    if (pos_after_add == pos_before) {
        llama_free(ctx);
        llama_model_free(model);
        return;
    }

    g_mock_dev_state.fail_next_compute = true;
    const int rc = decode_one_rc(ctx, 3);
    CHECK(rc == -3, "K-shift compute failure returns -3 (got %d)", rc);

    CHECK(llama_memory_seq_pos_max(llama_get_memory(ctx), 0) == pos_before,
          "cells restored after K-shift compute failure");

    const size_t state_cap = valid_cap > empty_cap ? valid_cap : empty_cap;
    std::vector<uint8_t> state(state_cap > 0 ? state_cap : 1);
    CHECK(llama_state_seq_get_data(ctx, state.data(), state.size(), 0) == 0,
          "nonempty state write refused after failed update");

    CHECK(llama_state_seq_set_data(ctx, valid_state.data(), valid_size, 0) == 0,
          "nonempty state read refused after failed update");

    // seq 1 has no cells: cell_count == 0 skips the meta guard, so the entry
    // guard in state_write/state_read_sinfo must refuse it.
    CHECK(llama_state_seq_get_data(ctx, state.data(), state.size(), 1) == 0,
          "empty-stream state write refused after failed update");
    CHECK(llama_state_seq_set_data(ctx, empty_state.data(), empty_size, 1) == 0,
          "empty-stream state read refused after failed update");

    llama_memory_clear(llama_get_memory(ctx), true);
    g_mock_dev_state.fail_next_compute = false;
    CHECK(decode_one(ctx, 3), "decode recovers after clear(true)");

    llama_free(ctx);
    llama_model_free(model);
}

static void run_no_update_and_applied(llama_model * model) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    CHECK(n_vocab > 3, "vocab too small");
    if (n_vocab <= 3) {
        return;
    }

    llama_context * ctx = make_ctx(model);
    CHECK(ctx != nullptr, "ctx");
    if (ctx == nullptr) {
        return;
    }

    // NO_UPDATE: two ordinary decodes, no pending shift/stream copy.
    CHECK(decode_one(ctx, 1), "decode 1");
    CHECK(decode_one(ctx, 2), "decode 2");
    CHECK(logits_finite(ctx, n_vocab), "logits finite");

    // APPLIED: shift positions >= 0 by +1, then decode. The next memory_update
    // must run the K-shift graph and return APPLIED, not a bool no-op.
    llama_memory_seq_add(llama_get_memory(ctx), 0, 0, -1, 1);
    CHECK(decode_one(ctx, 3), "decode after seq_add");
    CHECK(logits_finite(ctx, n_vocab), "logits finite after shift");

    llama_free(ctx);
}


int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <dense-llama.gguf>\n", argv[0]);
        return 2;
    }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    ggml_backend_dev_t devices[] = { nullptr };
    mp.devices = devices;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (model == nullptr) {
        std::fprintf(stderr, "FAIL: model load: %s\n", argv[1]);
        llama_backend_free();
        return 1;
    }

    run_backend_failure(argv[1]);
    if (g_skip) {
        llama_model_free(model);
        llama_backend_free();
        if (g_fail != 0) {
            std::fprintf(stderr, "KV_UPDATE_STATUS_TEST FAIL checks=%d failures=%d\n", g_checks, g_fail);
            return 1;
        }
        std::fprintf(stderr, "KV_UPDATE_STATUS_TEST SKIP: K-shift test not applicable\n");
        return 77;
    }
    run_no_update_and_applied(model);

    llama_model_free(model);
    llama_backend_free();

    if (g_fail != 0) {
        std::fprintf(stderr, "KV_UPDATE_STATUS_TEST FAIL checks=%d failures=%d\n", g_checks, g_fail);
        return 1;
    }
    std::fprintf(stderr, "KV_UPDATE_STATUS_TEST PASS checks=%d\n", g_checks);
    return 0;
}
