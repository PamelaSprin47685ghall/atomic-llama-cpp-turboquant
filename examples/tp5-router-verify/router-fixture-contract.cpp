#include "router-fixture-contract.h"

#include "ggml.h"
#include "gguf.h"
#include "llama.h"

// in-tree loader contract: qwen4exp's own load_arch_* code, not a copy of it
#include "llama-arch.h"
#include "llama-model-loader.h"
#include "llama-model.h"
#include "models/models.h" // llama_model_qwen4exp, the model the fixture claims to be

#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void check_array_len(const gguf_context * ctx, const char * key, uint64_t want, fixture_contract_diff & diff) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0) {
        diff.problems.push_back(std::string("missing key ") + key);
        return;
    }
    diff.n_meta_checked++;
    if (gguf_get_kv_type(ctx, id) != GGUF_TYPE_ARRAY) {
        diff.problems.push_back(std::string(key) + " is not an array");
        return;
    }
    const uint64_t got = gguf_get_arr_n(ctx, id);
    if (got != want) {
        diff.problems.push_back(std::string(key) + " has " + std::to_string(got) +
                " entries, the loader wants " + std::to_string(want));
    }
}

void check_scalar_u32(const gguf_context * ctx, const char * key, uint32_t want, fixture_contract_diff & diff) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0) {
        diff.problems.push_back(std::string("missing key ") + key);
        return;
    }
    diff.n_meta_checked++;
    if (gguf_get_kv_type(ctx, id) != GGUF_TYPE_UINT32) {
        diff.problems.push_back(std::string(key) + " is not a u32");
        return;
    }
    if (gguf_get_val_u32(ctx, id) != want) {
        diff.problems.push_back(std::string(key) + " = " + std::to_string(gguf_get_val_u32(ctx, id)) +
                ", expected " + std::to_string(want));
    }
}

} // namespace

fixture_contract_diff fixture_check_contract(const std::string & model_path) {
    fixture_contract_diff diff;

    gguf_init_params ip = {};
    ip.no_alloc = true;
    ip.ctx      = nullptr;
    gguf_context * ctx = gguf_init_from_file(model_path.c_str(), ip);
    if (ctx == nullptr) {
        throw std::runtime_error("cannot read the fixture metadata");
    }

    const int64_t arch_id = gguf_find_key(ctx, "general.architecture");
    if (arch_id < 0) {
        diff.problems.push_back("missing key general.architecture");
        gguf_free(ctx);
        return diff;
    }
    const std::string arch_name = gguf_get_val_str(ctx, arch_id);
    if (llm_arch_from_string(arch_name) != LLM_ARCH_QWEN4EXP) {
        diff.problems.push_back("general.architecture is '" + arch_name + "', expected qwen4exp");
        gguf_free(ctx);
        return diff;
    }

    std::vector<std::string> splits;
    // load_mtp true: the checkpoint carries the NextN block (block_count 3,
    // nextn_predict_layers 1) and load_arch_tensors marks its tensors
    // TENSOR_SKIP otherwise (qwen4exp.cpp:306-307, 314). The loader and the
    // params used to create the model must agree on that flag.
    llama_model_loader ml(ctx, [](struct ggml_tensor *, void *) {}, nullptr, model_path, splits, nullptr,
            LLAMA_LOAD_MODE_NONE, /*check_tensors =*/ false, /*no_alloc =*/ true, /*load_mtp =*/ true,
            nullptr, nullptr);

    llama_model_params mparams = llama_model_default_params();
    mparams.no_alloc = true;
    mparams.vocab_only = false;
    mparams.use_extra_bufts = false;
    // same flag as the loader above: the NextN block is part of this checkpoint
    mparams.load_mtp = true;

    // Create the model through the production factory: llama_model_create(arch)
    // is what sets model->arch, and llama_model_qwen4exp's constructor goes
    // through llama_model_base -> llama_model(params), which leaves arch at
    // LLM_ARCH_UNKNOWN. Constructing the subclass directly would hand
    // load_hparams a model whose arch is unknown, and llama_model_rope_type()
    // would then abort. Same call the real load path makes
    // (src/llama.cpp:316 llama_model_create(ml, params)).
    llama_model * model_raw = llama_model_create(LLM_ARCH_QWEN4EXP, mparams);
    if (model_raw == nullptr) {
        diff.problems.push_back("contract: no model implementation for qwen4exp");
        gguf_free(ctx);
        return diff;
    }
    if (model_raw->arch != LLM_ARCH_QWEN4EXP) {
        // the factory must have stamped the arch; without it every rope lookup
        // below would be an "unknown architecture" abort
        diff.problems.push_back("contract: the factory did not set arch to qwen4exp");
        delete model_raw;
        gguf_free(ctx);
        return diff;
    }
    auto * qwen4exp = dynamic_cast<llama_model_qwen4exp *>(model_raw);
    if (qwen4exp == nullptr) {
        diff.problems.push_back("contract: the qwen4exp factory returned another model type");
        delete model_raw;
        gguf_free(ctx);
        return diff;
    }
    llama_model & model = *qwen4exp;

    // 1) metadata: llama_model_base::load_hparams runs the common keys and then
    //    qwen4exp's load_arch_hparams, and ends with
    //    hparams.rope_type = llama_model_rope_type(this). With the arch set by
    //    the factory that switch resolves QWEN4EXP -> IMROPE; a missing key, a
    //    wrong type or a wrong array length throws here, and the check fails
    //    closed on it -- there is no fallback state to continue from.
    bool hparams_ok = true;
    bool no_problems_before = diff.problems.empty();
    try {
        model.load_hparams(ml);
    } catch (const std::exception & e) {
        hparams_ok = false;
        diff.problems.push_back(std::string("hparams: ") + e.what());
    }
    if (hparams_ok && !no_problems_before) {
        // an earlier step already failed: do not treat this pass as usable
        hparams_ok = false;
    }
    if (hparams_ok) {
        const llama_hparams & hp = model.hparams;
        check_array_len(ctx, "qwen4exp.rope.dimension_sections", 4, diff);
        check_array_len(ctx, "qwen4exp.attention.compress_ratios", hp.n_layer_all, diff);
        check_array_len(ctx, "qwen4exp.attention.recurrent_layers", hp.n_layer_all, diff);
        check_scalar_u32(ctx, "qwen4exp.expert_count",      512, diff);
        check_scalar_u32(ctx, "qwen4exp.expert_used_count", 10,  diff);

        // recurrent_layers drives is_recr() and therefore which token-mixer
        // tensors every block must carry (qwen4exp.cpp:235-243): trunk blocks
        // recurrent, the NextN block full attention
        uint32_t n_recr = 0;
        for (uint32_t il = 0; il < hp.n_layer_all; ++il) {
            if (model.hparams.is_recr(il)) {
                n_recr++;
            }
        }
        if (n_recr != hp.n_layer()) {
            diff.problems.push_back("hparams: " + std::to_string(n_recr) + " recurrent blocks, expected " +
                    std::to_string(hp.n_layer()) + " (NextN must be full attention)");
        }
        if (hp.rope_type != LLAMA_ROPE_TYPE_IMROPE) {
            diff.problems.push_back("hparams: rope_type is " + std::to_string((int) hp.rope_type) +
                    ", qwen4exp's IMROPE case says " + std::to_string((int) LLAMA_ROPE_TYPE_IMROPE));
        }
    }

    // 2) vocabulary, in the production order (llama.cpp:337-356): load_hparams
    //    -> load_vocab -> load_stats -> load_tensors. The vocab pass is what
    //    reads qwen4exp.vocab_size, so without it the model keeps n_vocab == 0
    //    and load_arch_tensors asks for token_embd [8, 0].
    bool vocab_ok = false;
    if (hparams_ok) {
        try {
            model.load_vocab(ml);
            model.load_stats(ml);
            vocab_ok = true;
        } catch (const std::exception & e) {
            diff.problems.push_back(std::string("vocab: ") + e.what());
        }
        if (vocab_ok && llama_vocab_n_tokens(llama_model_get_vocab(&model)) == 0) {
            diff.problems.push_back("vocab: n_vocab is 0 after load_vocab");
            vocab_ok = false;
        }
    }

    // 3) tensors: load_arch_tensors asks the loader for each weight with its
    //    shape; a missing required tensor or a shape mismatch throws with the
    //    name and both shapes, before any data is read. Only run once the
    //    hparams and the vocabulary are complete -- a failed earlier pass means
    //    no model exists and the shapes below would be meaningless.
    bool tensors_ok = false;
    if (vocab_ok) {
        try {
            tensors_ok = model.load_tensors(ml);
        } catch (const std::exception & e) {
            diff.problems.push_back(std::string("tensors: ") + e.what());
        }
    }
    if (tensors_ok) {
        // what the model bound, which is what the graph will read
        const std::pair<const char *, const struct ggml_tensor *> bound[] = {
            { "token_embd",     model.tok_embd     },
            { "output_hc_norm", model.hc_head_norm },
            { "output_hc_down", model.hc_head_down },
            { "output_hc_up",   model.hc_head_up   },
            { "output",         model.output       },
        };
        for (const auto & entry : bound) {
            if (entry.second == nullptr) {
                diff.problems.push_back(std::string(entry.first) + " is unbound");
            }
            diff.n_tensors_checked++;
        }
        struct layer_field { const char * name; struct ggml_tensor * llama_layer::* ptr; };
        static const layer_field per_layer[] = {
            { "hc_attn_norm",      &llama_layer::hc_attn_norm      },
            { "hc_attn_down",      &llama_layer::hc_attn_down      },
            { "hc_attn_up",        &llama_layer::hc_attn_up        },
            { "hc_attn_inject",    &llama_layer::hc_attn_inject    },
            { "hc_ffn_norm",       &llama_layer::hc_ffn_norm       },
            { "hc_ffn_down",       &llama_layer::hc_ffn_down       },
            { "hc_ffn_up",         &llama_layer::hc_ffn_up         },
            { "hc_ffn_inject",     &llama_layer::hc_ffn_inject     },
            { "ffn_gate_inp",      &llama_layer::ffn_gate_inp      },
            { "ffn_down_exps",     &llama_layer::ffn_down_exps     },
            { "ffn_gate_inp_shexp",&llama_layer::ffn_gate_inp_shexp},
            { "ffn_gate_shexp",    &llama_layer::ffn_gate_shexp    },
            { "ffn_up_shexp",      &llama_layer::ffn_up_shexp      },
            { "ffn_down_shexp",    &llama_layer::ffn_down_shexp    },
        };
        // the token mixer depends on is_recr(il): trunk blocks are recurrent and
        // bind attn_qkv/attn_gate/ssm_*, the NextN block is not and binds the
        // full-attention set (qwen4exp.cpp:337-360)
        static const layer_field recr_only[] = {
            { "attn_qkv",      &llama_layer::wqkv           },
            { "attn_gate",     &llama_layer::wqkv_gate      },
            { "ssm_conv1d",    &llama_layer::ssm_conv1d     },
            { "ssm_dt",        &llama_layer::ssm_dt         },
            { "ssm_a",         &llama_layer::ssm_a          },
            { "ssm_beta",      &llama_layer::ssm_beta       },
            { "ssm_alpha",     &llama_layer::ssm_alpha      },
            { "ssm_norm",      &llama_layer::ssm_norm       },
            { "ssm_out",       &llama_layer::ssm_out        },
        };
        static const layer_field full_attn_only[] = {
            { "attn_qkv",        &llama_layer::wqkv         }, // create_tensor_qkv
            { "attn_output",     &llama_layer::wo           },
            { "attn_q_norm",     &llama_layer::attn_q_norm  },
            { "attn_k_norm",     &llama_layer::attn_k_norm  },
            { "indexer.q_proj",  &llama_layer::index_q_proj },
            { "indexer.k_proj",  &llama_layer::index_k_proj },
            { "indexer.q_norm",  &llama_layer::index_q_norm },
            { "indexer.k_norm",  &llama_layer::index_k_norm },
        };
        // nextn is an embedded llama_layer_nextn value, not a base subobject, so
        // its tensors take a member pointer into llama_layer_nextn, applied to
        // layer.nextn
        struct nextn_field { const char * name; struct ggml_tensor * llama_layer_nextn::* ptr; };
        // (a pointer to llama_layer::* would be ill-formed here: nextn is not a
        //  base of llama_layer)
        static const nextn_field nextn_only[] = {
            { "nextn.enorm",   &llama_layer_nextn::enorm   },
            { "nextn.hnorm",   &llama_layer_nextn::hnorm   },
            { "nextn.eh_proj", &llama_layer_nextn::eh_proj },
        };
        for (size_t il = 0; il < model.layers.size(); ++il) {
            const llama_layer & layer = model.layers[il];
            for (const layer_field & f : per_layer) {
                if (layer.*(f.ptr) == nullptr) {
                    diff.problems.push_back("blk." + std::to_string(il) + "." + f.name + " is unbound");
                }
                diff.n_tensors_checked++;
            }
            const bool recr = model.hparams.is_recr((uint32_t) il);
            const bool nextn = il >= model.hparams.n_layer();
            if (recr) {
                for (const layer_field & f : recr_only) {
                    if (layer.*(f.ptr) == nullptr) {
                        diff.problems.push_back("blk." + std::to_string(il) + "." + f.name + " is unbound (recurrent block)");
                    }
                    diff.n_tensors_checked++;
                }
            } else {
                for (const layer_field & f : full_attn_only) {
                    if (layer.*(f.ptr) == nullptr) {
                        diff.problems.push_back("blk." + std::to_string(il) + "." + f.name + " is unbound (full-attention block)");
                    }
                    diff.n_tensors_checked++;
                }
            }
            if (nextn) {
                for (const nextn_field & f : nextn_only) {
                    if (layer.nextn.*(f.ptr) == nullptr) {
                        diff.problems.push_back("blk." + std::to_string(il) + "." + f.name + " is unbound (NextN head)");
                    }
                    diff.n_tensors_checked++;
                }
            }
            if (layer.ffn_gate_exps == nullptr && layer.ffn_gate_up_exps == nullptr) {
                diff.problems.push_back("blk." + std::to_string(il) + ": neither ffn_gate_exps nor ffn_gate_up_exps is bound");
            }
        }
    }

    // 4) every stored tensor must be a non-empty dense F32 block (checked on the
    //    file, not on the model, so it runs even when a loader pass failed)
    for (int64_t i = 0; i < gguf_get_n_tensors(ctx); ++i) {
        if (gguf_get_tensor_type(ctx, i) != GGML_TYPE_F32) {
            diff.problems.push_back(std::string("tensor ") + gguf_get_tensor_name(ctx, i) + " is not F32");
        }
        if (gguf_get_tensor_size(ctx, i) == 0) {
            diff.problems.push_back(std::string("tensor ") + gguf_get_tensor_name(ctx, i) + " is empty");
        }
    }

    // teardown order matters: the model owns the tensor descriptors it created
    // (released by its virtual destructor, no tensors were materialized), and
    // `ml` holds a non-owning pointer to ctx, so the gguf context goes last.
    delete model_raw;
    gguf_free(ctx);
    return diff;
}
