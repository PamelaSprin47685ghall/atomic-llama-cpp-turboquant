// TP5 HC replicated-compute switch regression (CPU only, no GPU).
//
// TP5_HC_REPLICATED_COMPUTE selects the historical mirrored local-compute HC
// storage instead of the default static-owner layout:
//   * unset/0: an HC pair keeps one complete 320-wide low-rank owner,
//   * 1: the pair is MIRRORED on all five ranks and the load logs
//     "mirrored local-compute",
//   * any other value: the load throws instead of silently ignoring the flag.
//
// The test drives the model's Meta split-state callback directly with a mock
// Qwen4Exp model and Q8_0 HC pair tensors, so it needs no GGUF and no GPU.
// CTest runs three separate processes because the switch is cached for the
// process lifetime.
//
// Build and run:
//   cmake --build <build> -j --target test-tp5-hc-replicated-compute
//   ctest --test-dir <build> -R test-tp5-hc-replicated-compute

#include "llama.h"
#include "llama-ext.h"
#include "ggml.h"
#include "ggml-backend.h"

#include "../src/llama-model.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

namespace {

std::string g_log;

void capture_log(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    if (text != nullptr) {
        g_log += text;
        std::fputs(text, stderr);
    }
}

struct fixture {
    llama_model * model = nullptr;
    ggml_context * ctx  = nullptr;
    ggml_tensor * down  = nullptr;
    ggml_tensor * up    = nullptr;

    ~fixture() {
        if (model != nullptr) {
            model->tensors_by_name.clear();
        }
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
        if (model != nullptr) {
            llama_model_free(model);
        }
    }
};

bool make_fixture(fixture & f) {
    llama_quant_model_desc desc = {};
    desc.architecture  = "qwen4exp";
    desc.n_embd        = 2560;
    desc.n_ff          = 2560;
    desc.n_layer       = 48;
    desc.n_head        = 40;
    desc.n_head_kv     = 8;
    desc.n_expert      = 0;
    desc.n_embd_head_k = 64;
    desc.n_embd_head_v = 64;

    f.model = llama_quant_model_from_metadata(&desc);
    if (f.model == nullptr) {
        return false;
    }
    f.model->hparams.n_layer_nextn = 0;
    f.model->hparams.dsv4_hc_mult  = 4;
    f.model->hparams.hc_low_rank   = 320;

    const ggml_init_params ip = {
        /*.mem_size   =*/ 1u << 20,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    f.ctx = ggml_init(ip);
    if (f.ctx == nullptr) {
        return false;
    }
    f.down = ggml_new_tensor_2d(f.ctx, GGML_TYPE_Q8_0, 10240, 320);
    f.up   = ggml_new_tensor_2d(f.ctx, GGML_TYPE_Q8_0, 320, 10240);
    ggml_set_name(f.down, "blk.0.hc_attn_down.weight");
    ggml_set_name(f.up,   "blk.0.hc_attn_up.weight");
    f.model->tensors_by_name.emplace_back("blk.0.hc_attn_down.weight", f.down);
    f.model->tensors_by_name.emplace_back("blk.0.hc_attn_up.weight",   f.up);
    return true;
}

bool has(const char * needle) {
    return g_log.find(needle) != std::string::npos;
}

int test_off(fixture & f) {
    llama_meta_device_get_split_state_userdata ud = { 5, f.model };

    const auto down = llama_meta_device_get_split_state(f.down, &ud);
    if (down.axis != GGML_BACKEND_SPLIT_AXIS_1) {
        std::fprintf(stderr, "FAIL off: down axis=%d, want axis 1\n", int(down.axis));
        return 1;
    }
    size_t down_owners = 0;
    for (size_t rank = 0; rank < 5; ++rank) {
        down_owners += down.ne[rank] == 320;
    }
    if (down_owners != 1) {
        std::fprintf(stderr, "FAIL off: down owners=%zu, want exactly one 320-wide owner\n", down_owners);
        return 1;
    }

    const auto up = llama_meta_device_get_split_state(f.up, &ud);
    if (up.axis != GGML_BACKEND_SPLIT_AXIS_0) {
        std::fprintf(stderr, "FAIL off: up axis=%d, want axis 0\n", int(up.axis));
        return 1;
    }
    size_t up_owners = 0;
    for (size_t rank = 0; rank < 5; ++rank) {
        up_owners += up.ne[rank] == 320;
    }
    if (up_owners != 1) {
        std::fprintf(stderr, "FAIL off: up owners=%zu, want exactly one 320-wide owner\n", up_owners);
        return 1;
    }

    if (has("mirrored local-compute")) {
        std::fprintf(stderr, "FAIL off: the mirrored branch was selected while the switch is off\n");
        return 1;
    }
    return 0;
}

int test_on(fixture & f) {
    llama_meta_device_get_split_state_userdata ud = { 5, f.model };

    const auto down = llama_meta_device_get_split_state(f.down, &ud);
    if (down.axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
        std::fprintf(stderr, "FAIL on: down axis=%d, want MIRRORED\n", int(down.axis));
        return 1;
    }
    for (size_t rank = 0; rank < 5; ++rank) {
        if (down.ne[rank] != 0) {
            std::fprintf(stderr, "FAIL on: down rank %zu has slice %lld, want MIRRORED\n",
                    rank, (long long) down.ne[rank]);
            return 1;
        }
    }

    const auto up = llama_meta_device_get_split_state(f.up, &ud);
    if (up.axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
        std::fprintf(stderr, "FAIL on: up axis=%d, want MIRRORED\n", int(up.axis));
        return 1;
    }

    if (!has("TP5_HC_REPLICATED_COMPUTE=1")) {
        std::fprintf(stderr, "FAIL on: the switch log line is missing\n");
        return 1;
    }
    if (!has("mirrored local-compute")) {
        std::fprintf(stderr, "FAIL on: the mirrored local-compute label is missing\n");
        return 1;
    }
    return 0;
}

int test_invalid(fixture & f) {
    llama_meta_device_get_split_state_userdata ud = { 5, f.model };
    bool threw = false;
    try {
        (void) llama_meta_device_get_split_state(f.down, &ud);
    } catch (const std::runtime_error & e) {
        threw = true;
        if (std::string(e.what()).find("TP5_HC_REPLICATED_COMPUTE requires 0 or 1") == std::string::npos) {
            std::fprintf(stderr, "FAIL invalid: unexpected error: %s\n", e.what());
            return 1;
        }
        std::printf("observed error: %s\n", e.what());
    }
    if (!threw) {
        std::fprintf(stderr, "FAIL invalid: TP5_HC_REPLICATED_COMPUTE=2 was silently ignored\n");
        return 1;
    }
    if (has("mirrored local-compute")) {
        std::fprintf(stderr, "FAIL invalid: the mirrored branch was selected for an invalid value\n");
        return 1;
    }
    return 0;
}

int test_conflict(fixture & f) {
    llama_meta_device_get_split_state_userdata ud = { 5, f.model };
    bool threw = false;
    try {
        (void) llama_meta_device_get_split_state(f.down, &ud);
    } catch (const std::runtime_error & e) {
        threw = true;
        if (std::string(e.what()).find("requires context-owner OFF") == std::string::npos) {
            std::fprintf(stderr, "FAIL conflict: unexpected error: %s\n", e.what());
            return 1;
        }
        std::printf("observed error: %s\n", e.what());
    }
    if (!threw) {
        std::fprintf(stderr, "FAIL conflict: replicated compute and context-owner were both accepted\n");
        return 1;
    }
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s off|on|invalid|conflict\n", argv[0]);
        return 2;
    }
    const std::string mode = argv[1];
    if (mode != "off" && mode != "on" && mode != "invalid" && mode != "conflict") {
        std::fprintf(stderr, "unknown mode: %s\n", mode.c_str());
        return 2;
    }

    llama_log_set(capture_log, nullptr);

    fixture f;
    if (!make_fixture(f)) {
        std::fprintf(stderr, "FAIL: cannot build the mock Qwen4Exp fixture\n");
        return 1;
    }

    int rc = 0;
    try {
        rc = mode == "off"      ? test_off(f) :
             mode == "on"       ? test_on(f)  :
             mode == "invalid"  ? test_invalid(f) : test_conflict(f);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL %s: uncaught exception: %s\n", mode.c_str(), e.what());
        return 1;
    }
    if (rc != 0) {
        return rc;
    }
    std::printf("TP5_HC_REPLICATED_COMPUTE_TEST PASS mode=%s\n", mode.c_str());
    return 0;
}
