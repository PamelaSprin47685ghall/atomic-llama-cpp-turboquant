// TP5 HC replicated-compute switch regression (CPU only, no GPU).
//
// TP5_HC_REPLICATED_COMPUTE selects the historical mirrored local-compute HC
// storage instead of the default static-owner layout:
//   * unset/0: an HC pair keeps one complete 320-wide low-rank owner,
//   * 1: the pair is MIRRORED on all five ranks; the switch logs its
//     selection, but the full-model storage-label line is not asserted here,
//   * any other value: the load throws instead of silently ignoring the flag.
//
// The split-state tests drive the Meta callback with mock Qwen4Exp models and
// Q8_0/BF16 HC pairs. The policy/validate tests exercise the pure validation
// policy and a synthetic loader; they are unit tests, not real GGUF loads.
// CTest runs separate processes because the switch is cached for the process
// lifetime.
//
// Build and run:
//   cmake --build <build> -j --target test-tp5-hc-replicated-compute
//   ctest --test-dir <build> -R test-tp5-hc-replicated-compute

#include "llama.h"
#include "llama-ext.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include "../src/llama-model.h"
#include "../src/llama-model-loader.h"

#include <cstdio>
#include <cstring>
#include <memory>
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

llama_model * make_model(uint32_t n_layer, uint32_t n_layer_nextn) {
    llama_quant_model_desc desc = {};
    desc.architecture  = "qwen4exp";
    desc.n_embd        = 2560;
    desc.n_ff          = 2560;
    desc.n_layer       = n_layer;
    desc.n_head        = 40;
    desc.n_head_kv     = 8;
    desc.n_expert      = 0;
    desc.n_embd_head_k = 64;
    desc.n_embd_head_v = 64;

    llama_model * model = llama_quant_model_from_metadata(&desc);
    if (model == nullptr) {
        return nullptr;
    }
    model->hparams.n_layer_nextn = n_layer_nextn;
    model->hparams.dsv4_hc_mult  = 4;
    model->hparams.hc_low_rank   = 320;
    return model;
}

bool make_tensor_pair(fixture & f, ggml_type type, const char * down_name, const char * up_name) {
    const ggml_init_params ip = {
        /*.mem_size   =*/ 1u << 20,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    f.ctx = ggml_init(ip);
    if (f.ctx == nullptr) {
        return false;
    }
    f.down = ggml_new_tensor_2d(f.ctx, type, 10240, 320);
    f.up   = ggml_new_tensor_2d(f.ctx, type, 320, 10240);
    ggml_set_name(f.down, down_name);
    ggml_set_name(f.up,   up_name);
    f.model->tensors_by_name.emplace_back(down_name, f.down);
    f.model->tensors_by_name.emplace_back(up_name,   f.up);
    return true;
}

bool make_fixture(fixture & f) {
    f.model = make_model(48, 0);
    if (f.model == nullptr) {
        return false;
    }
    return make_tensor_pair(f, GGML_TYPE_Q8_0,
            "blk.0.hc_attn_down.weight", "blk.0.hc_attn_up.weight");
}

bool make_mtp_fixture(fixture & f) {
    // The private MTP-only model has one nextn layer and deliberately keeps
    // its own BF16 HC block; the compose loader temporarily sets
    // GGML_TP5_HC_Q8_WEIGHTS=0 for this load.
    f.model = make_model(49, 1);
    if (f.model == nullptr) {
        return false;
    }
    return make_tensor_pair(f, GGML_TYPE_BF16,
            "blk.48.nextn.hc_head_down.weight", "blk.48.nextn.hc_head_up.weight");
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

    if (has("TP5_HC_REPLICATED_COMPUTE=1")) {
        std::fprintf(stderr, "FAIL off: the switch log appeared while the switch is off\n");
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
    if (has("TP5_HC_REPLICATED_COMPUTE=1")) {
        std::fprintf(stderr, "FAIL invalid: the switch was selected for an invalid value\n");
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

int test_mtp(fixture & f) {
    // The private MTP-only model keeps one BF16 nextn.hc_head pair. The
    // replicated flag is set, but the pair is outside hc_pair_profile(), so
    // it must stay on the static-owner path and load successfully.
    llama_meta_device_get_split_state_userdata ud = { 5, f.model };

    const auto down = llama_meta_device_get_split_state(f.down, &ud);
    if (down.axis != GGML_BACKEND_SPLIT_AXIS_1) {
        std::fprintf(stderr, "FAIL mtp: down axis=%d, want static-owner axis 1\n", int(down.axis));
        return 1;
    }
    size_t down_owners = 0;
    for (size_t rank = 0; rank < 5; ++rank) {
        down_owners += down.ne[rank] == 320;
    }
    if (down_owners != 1) {
        std::fprintf(stderr, "FAIL mtp: down owners=%zu, want exactly one 320-wide owner\n", down_owners);
        return 1;
    }

    const auto up = llama_meta_device_get_split_state(f.up, &ud);
    if (up.axis != GGML_BACKEND_SPLIT_AXIS_0) {
        std::fprintf(stderr, "FAIL mtp: up axis=%d, want static-owner axis 0\n", int(up.axis));
        return 1;
    }
    size_t up_owners = 0;
    for (size_t rank = 0; rank < 5; ++rank) {
        up_owners += up.ne[rank] == 320;
    }
    if (up_owners != 1) {
        std::fprintf(stderr, "FAIL mtp: up owners=%zu, want exactly one 320-wide owner\n", up_owners);
        return 1;
    }

    if (down.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED ||
            up.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
        std::fprintf(stderr, "FAIL mtp: the MTP-only model entered the mirrored branch\n");
        return 1;
    }
    return 0;
}

int test_policy() {
    llama_model * main_model = make_model(48, 0);
    llama_model * mtp_model  = make_model(49, 1);
    if (main_model == nullptr || mtp_model == nullptr) {
        std::fprintf(stderr, "FAIL policy: cannot build the mock models\n");
        llama_model_free(main_model);
        llama_model_free(mtp_model);
        return 1;
    }

    struct policy_case {
        const char * name;
        const llama_model * model;
        ggml_type weight_type;
        bool context_owner;
        bool replicated;
        bool mtp_only;
        llama_hc_layout_validation want;
    };
    const policy_case cases[] = {
        { "main q8 replicated",                main_model, GGML_TYPE_Q8_0,  false, true,  false, LLAMA_HC_LAYOUT_VALIDATE_REPLICATED },
        { "main bf16 replicated",              main_model, GGML_TYPE_BF16, false, true,  false, LLAMA_HC_LAYOUT_REJECT_WEIGHT_TYPE },
        { "trailing mtp full trunk bf16 repl", mtp_model,  GGML_TYPE_BF16, false, true,  false, LLAMA_HC_LAYOUT_VALIDATE_DEFAULT },
        { "trailing mtp full trunk q8 repl",   mtp_model,  GGML_TYPE_Q8_0, false, true,  false, LLAMA_HC_LAYOUT_VALIDATE_DEFAULT },
        { "mtp-only bf16 replicated",          mtp_model,  GGML_TYPE_BF16, false, true,  true,  LLAMA_HC_LAYOUT_SKIP },
        { "mtp-only q8 replicated",            mtp_model,  GGML_TYPE_Q8_0, false, true,  true,  LLAMA_HC_LAYOUT_VALIDATE_DEFAULT },
        { "main q8 default",                   main_model, GGML_TYPE_Q8_0,  false, false, false, LLAMA_HC_LAYOUT_VALIDATE_DEFAULT },
        { "main bf16 default",                 main_model, GGML_TYPE_BF16, false, false, false, LLAMA_HC_LAYOUT_SKIP },
        { "mtp-only bf16 default",             mtp_model,  GGML_TYPE_BF16, false, false, true,  LLAMA_HC_LAYOUT_SKIP },
        { "trailing mtp full trunk bf16 def",  mtp_model,  GGML_TYPE_BF16, false, false, false, LLAMA_HC_LAYOUT_SKIP },
        { "trailing mtp full trunk q8 def",    mtp_model,  GGML_TYPE_Q8_0, false, false, false, LLAMA_HC_LAYOUT_VALIDATE_DEFAULT },
    };

    int rc = 0;
    for (const auto & c : cases) {
        const llama_hc_layout_validation got = llama_hc_layout_validation_for(
                *c.model, c.weight_type, c.context_owner, c.replicated, c.mtp_only);
        if (got != c.want) {
            std::fprintf(stderr, "FAIL policy %s: got %d, want %d\n",
                    c.name, int(got), int(c.want));
            rc = 1;
            break;
        }
    }

    llama_model_free(main_model);
    llama_model_free(mtp_model);
    return rc;
}

struct loader_fixture {
    gguf_context * meta = nullptr;
    FILE * file = nullptr;
    std::unique_ptr<llama_model_loader> ml;

    explicit loader_fixture(ggml_type hc_weight_type, bool with_trunk_tensor = false) {
        std::vector<std::string> splits;
        if (with_trunk_tensor) {
            // A tiny metadata-only GGUF that carries the trunk marker
            // blk.0.hc_attn_norm.weight. The loader only needs the weight map
            // to distinguish a full trunk with a trailing MTP head from a true
            // MTP-only checkpoint.
            ggml_init_params ip = { 1u << 20, nullptr, false };
            ggml_context * ctx = ggml_init(ip);
            if (ctx == nullptr) {
                throw std::runtime_error("cannot create the tiny GGUF context");
            }
            ggml_tensor * marker = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
            ggml_set_name(marker, "blk.0.hc_attn_norm.weight");
            *static_cast<float *>(marker->data) = 0.0f;

            gguf_context * g = gguf_init_empty();
            if (g == nullptr) {
                ggml_free(ctx);
                throw std::runtime_error("cannot create the tiny GGUF metadata");
            }
            gguf_add_tensor(g, marker);

            file = tmpfile();
            if (file == nullptr || !gguf_write_to_file_ptr(g, file, /*only_meta*/ false)) {
                if (file != nullptr) {
                    std::fclose(file);
                    file = nullptr;
                }
                gguf_free(g);
                ggml_free(ctx);
                throw std::runtime_error("cannot write the tiny trunk GGUF");
            }
            std::fflush(file);
            std::rewind(file);
            gguf_free(g);
            ggml_free(ctx);

            try {
                ml = std::make_unique<llama_model_loader>(
                        /*metadata*/ nullptr, /*set_tensor_data*/ nullptr,
                        /*set_tensor_data_ud*/ nullptr, /*fname*/ std::string(),
                        splits, file, LLAMA_LOAD_MODE_NONE, /*check_tensors*/ false,
                        /*no_alloc*/ true, /*load_mtp*/ false,
                        /*param_overrides_p*/ nullptr,
                        /*param_tensor_buft_overrides_p*/ nullptr);
            } catch (...) {
                std::fclose(file);
                file = nullptr;
                throw;
            }
        } else {
            // The empty metadata path needs no file; it is enough for the
            // policy paths that never consult the weight map.
            meta = gguf_init_empty();
            if (meta == nullptr) {
                throw std::runtime_error("cannot create the empty GGUF metadata");
            }
            try {
                ml = std::make_unique<llama_model_loader>(
                        meta, /*set_tensor_data*/ nullptr, /*set_tensor_data_ud*/ nullptr,
                        /*fname*/ std::string(), splits, /*file*/ nullptr,
                        LLAMA_LOAD_MODE_NONE, /*check_tensors*/ false, /*no_alloc*/ true,
                        /*load_mtp*/ false, /*param_overrides_p*/ nullptr,
                        /*param_tensor_buft_overrides_p*/ nullptr);
            } catch (...) {
                gguf_free(meta);
                meta = nullptr;
                throw;
            }
        }
        ml->hc_weight_type = hc_weight_type;
    }

    ~loader_fixture() {
        ml.reset();
        if (meta != nullptr) {
            gguf_free(meta);
        }
        if (file != nullptr) {
            std::fclose(file);
        }
    }
};

int test_validate() {
    llama_model * main_model = make_model(48, 0);
    llama_model * mtp_model  = make_model(49, 1);
    if (main_model == nullptr || mtp_model == nullptr) {
        std::fprintf(stderr, "FAIL validate: cannot build the mock models\n");
        llama_model_free(main_model);
        llama_model_free(mtp_model);
        return 1;
    }

    int rc = 0;

    // (c) The private MTP-only model loads its BF16 nextn HC block while the
    // replicated flag is on. It must pass the validation policy.
    {
        loader_fixture lf(GGML_TYPE_BF16);
        try {
            llama_hc_context_owner_validate_layout(*mtp_model, *lf.ml);
        } catch (const std::exception & e) {
            std::fprintf(stderr, "FAIL validate mtp bf16: unexpected error: %s\n", e.what());
            rc = 1;
        }
    }

    // A full trunk with a trailing MTP head is not MTP-only: the trunk marker
    // is present, so the policy must not skip it into a late allocation error.
    // It must fail the exact-profile check before any buffer allocation.
    if (rc == 0) {
        loader_fixture lf(GGML_TYPE_BF16, /*with_trunk_tensor*/ true);
        bool threw = false;
        try {
            llama_hc_context_owner_validate_layout(*mtp_model, *lf.ml);
        } catch (const std::runtime_error & e) {
            threw = true;
            if (std::string(e.what()).find("no MTP") == std::string::npos) {
                std::fprintf(stderr, "FAIL validate trailing mtp: unexpected error: %s\n", e.what());
                rc = 1;
            } else {
                std::printf("observed error: %s\n", e.what());
            }
        }
        if (!threw && rc == 0) {
            std::fprintf(stderr, "FAIL validate trailing mtp: the full trunk was skipped instead of rejected early\n");
            rc = 1;
        }
    }

    // (b) The main trunk with the flag on but Q8 conversion off must fail.
    if (rc == 0) {
        loader_fixture lf(GGML_TYPE_BF16);
        bool threw = false;
        try {
            llama_hc_context_owner_validate_layout(*main_model, *lf.ml);
        } catch (const std::runtime_error & e) {
            threw = true;
            if (std::string(e.what()).find("requires Q8_0 HC weights") == std::string::npos) {
                std::fprintf(stderr, "FAIL validate main bf16: unexpected error: %s\n", e.what());
                rc = 1;
            } else {
                std::printf("observed error: %s\n", e.what());
            }
        }
        if (!threw && rc == 0) {
            std::fprintf(stderr, "FAIL validate main bf16: Q8 requirement was not enforced\n");
            rc = 1;
        }
    }

    // (a) The main trunk with Q8 must pass the weight-type guard. The mock
    // model has no Meta device, so the next guard is the expected stop.
    if (rc == 0) {
        loader_fixture lf(GGML_TYPE_Q8_0);
        bool threw = false;
        std::string message;
        try {
            llama_hc_context_owner_validate_layout(*main_model, *lf.ml);
        } catch (const std::runtime_error & e) {
            threw = true;
            message = e.what();
        }
        if (message.find("requires Q8_0 HC weights") != std::string::npos) {
            std::fprintf(stderr, "FAIL validate main q8: the Q8 guard rejected a Q8 model\n");
            rc = 1;
        } else if (!threw || message.find("requires one Meta device") == std::string::npos) {
            std::fprintf(stderr, "FAIL validate main q8: unexpected result: %s\n",
                    threw ? message.c_str() : "no error");
            rc = 1;
        } else {
            std::printf("main q8 next guard: %s\n", message.c_str());
        }
    }

    llama_model_free(main_model);
    llama_model_free(mtp_model);
    return rc;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s off|on|invalid|conflict|mtp|policy|validate\n", argv[0]);
        return 2;
    }
    const std::string mode = argv[1];
    if (mode != "off" && mode != "on" && mode != "invalid" &&
            mode != "conflict" && mode != "mtp" && mode != "policy" &&
            mode != "validate") {
        std::fprintf(stderr, "unknown mode: %s\n", mode.c_str());
        return 2;
    }

    llama_log_set(capture_log, nullptr);

    if (mode == "policy" || mode == "validate") {
        const int rc = mode == "policy" ? test_policy() : test_validate();
        if (rc != 0) {
            return rc;
        }
        std::printf("TP5_HC_REPLICATED_COMPUTE_TEST PASS mode=%s\n", mode.c_str());
        return 0;
    }

    fixture f;
    if (mode == "mtp" ? !make_mtp_fixture(f) : !make_fixture(f)) {
        std::fprintf(stderr, "FAIL: cannot build the mock Qwen4Exp fixture\n");
        return 1;
    }

    int rc = 0;
    try {
        rc = mode == "off"      ? test_off(f) :
             mode == "on"       ? test_on(f)  :
             mode == "invalid"  ? test_invalid(f) :
             mode == "conflict" ? test_conflict(f) : test_mtp(f);
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
