#pragma once

#include "ggml-backend.h"
#include "../../ggml/src/ggml-impl.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

// Opt-in observation, not a fallback or a change to the graph's equations.
// Sampling a bounded number of SMALL activation tensors locates where Lane
// representations converge without copying the persistent recurrent matrices.
struct server_rerot_audit {
    std::unordered_map<std::string, size_t> seen;
    size_t blocks = 0;
    size_t attention_blocks = 0;
    size_t attention_seen = 0;

    bool dump_block(ggml_tensor * tensor) {
        const char * directory = std::getenv("LLAMA_REROT_AUDIT_DIR");
        if (!directory || blocks >= 6) return true;
        try {
            const auto dir = std::filesystem::path(directory) / ("block-" + std::to_string(blocks++));
            std::filesystem::create_directories(dir);
            std::ofstream(dir / "density-mode.txt") << ggml_get_op_params_i32(tensor, 2) << '\n';
            std::ofstream meta(dir / "tensors.tsv");
            for (int i = 0; i < 8; ++i) {
                ggml_tensor * t = i == 7 ? tensor : tensor->src[i];
                const std::string name = i == 7 ? "result" : "src" + std::to_string(i);
                meta << name << '\t' << ggml_type_name(t->type);
                for (int d = 0; d < 4; ++d) meta << '\t' << t->ne[d];
                for (int d = 0; d < 4; ++d) meta << '\t' << t->nb[d];
                meta << '\t' << ggml_nbytes(t) << '\n';
                std::vector<uint8_t> bytes(ggml_nbytes(t));
                ggml_backend_tensor_get(t, bytes.data(), 0, bytes.size());
                std::ofstream out(dir / (name + ".bin"), std::ios::binary);
                out.write(reinterpret_cast<const char *>(bytes.data()), std::streamsize(bytes.size()));
                if (!out) return false;
            }
            std::fprintf(stderr, "rerot.audit.block path=%s writers=%lld\n", dir.c_str(), (long long) tensor->src[2]->ne[3]);
            return bool(meta);
        } catch (const std::exception & error) {
            std::fprintf(stderr, "rerot.audit.block failed: %s\n", error.what());
            return false;
        }
    }

    bool dump_attention(ggml_tensor * tensor, size_t observation) {
        const char * directory = std::getenv("LLAMA_REROT_AUDIT_DIR");
        if (!directory || attention_blocks >= 12) return true;
        try {
            const auto dir = std::filesystem::path(directory) /
                ("attention-" + std::to_string(observation));
            ++attention_blocks;
            std::filesystem::create_directories(dir);
            std::ofstream meta(dir / "tensors.tsv");
            for (int i = 0; i < 7; ++i) {
                ggml_tensor * t = i == 6 ? tensor : tensor->src[i];
                if (!t) continue;
                const std::string tname = i == 6 ? "result" : "src" + std::to_string(i);
                meta << tname << '\t' << ggml_type_name(t->type);
                for (int d = 0; d < 4; ++d) meta << '\t' << t->ne[d];
                for (int d = 0; d < 4; ++d) meta << '\t' << t->nb[d];
                meta << '\t' << ggml_nbytes(t) << '\n';
                std::vector<uint8_t> bytes(ggml_nbytes(t));
                ggml_backend_tensor_get(t, bytes.data(), 0, bytes.size());
                std::ofstream out(dir / (tname + ".bin"), std::ios::binary);
                out.write(reinterpret_cast<const char *>(bytes.data()), std::streamsize(bytes.size()));
                if (!out) return false;
            }
            std::ofstream params(dir / "op-params.bin", std::ios::binary);
            params.write(reinterpret_cast<const char *>(tensor->op_params), GGML_MAX_OP_PARAMS);
            std::fprintf(stderr, "rerot.audit.attention path=%s queries=%lld groups=%lld keys=%lld\n",
                dir.c_str(), (long long) tensor->ne[2],
                tensor->src[0] ? (long long) tensor->src[0]->ne[1] : -1LL,
                tensor->src[1] ? (long long) tensor->src[1]->ne[1] : -1LL);
            return bool(meta) && bool(params);
        } catch (const std::exception & error) {
            std::fprintf(stderr, "rerot.audit.attention failed: %s\n", error.what());
            return false;
        }
    }

    static bool observe(ggml_tensor * tensor, bool ask, void * userdata) {
        auto & audit = *static_cast<server_rerot_audit *>(userdata);
        const std::string name = ggml_get_name(tensor);
        if (name == "rerot_block_result-4" && tensor->src[2] && tensor->src[2]->ne[3] > 1 &&
            std::getenv("LLAMA_REROT_AUDIT_DIR") != nullptr) {
            return ask ? audit.blocks < 6 : audit.dump_block(tensor);
        }
        if (name == "rerot_indexed_attn-3" && tensor->src[0] && tensor->src[1] &&
            tensor->src[3] && tensor->src[4] && tensor->ne[2] > 1 &&
            std::getenv("LLAMA_REROT_AUDIT_DIR") != nullptr) {
            if (ask) {
                return audit.attention_seen <= 256;
            }
            const size_t observation = audit.attention_seen++;
            const bool capture = observation <= 2 || observation == 4 || observation == 8 ||
                observation == 16 || observation == 32 || observation == 64 || observation == 96 ||
                observation == 128 || observation == 192 || observation == 256;
            return !capture || audit.dump_attention(tensor, observation);
        }
        const size_t split = name.rfind('-');
        const std::string kind = name.substr(0, split);
        const std::string layer = split == std::string::npos ? "" : name.substr(split + 1);
        const bool selected_layer = layer == "0" || layer == "2" || layer == "3" ||
            layer == "4" || layer == "7" || layer == "8" || layer == "39";
        const bool selected_kind = kind == "attn_norm" || kind == "linear_attn_out" ||
            kind == "attn_pregate" || kind == "l_out";
        const int64_t rows = tensor->ne[0] > 0 ? ggml_nelements(tensor) / tensor->ne[0] : 0;
        const bool selected = selected_layer && selected_kind && tensor->type == GGML_TYPE_F32 &&
            ggml_is_contiguous(tensor) && rows > 1 && rows <= 32 && ggml_nbytes(tensor) <= 1024 * 1024;
        if (ask) {
            return selected && audit.seen[name] < 96;
        }
        if (!selected) {
            return true;
        }
        const size_t step = audit.seen[name]++;
        std::vector<float> data(size_t(ggml_nelements(tensor)));
        ggml_backend_tensor_get(tensor, data.data(), 0, data.size() * sizeof(float));
        double sum_sq = 0.0, max_abs = 0.0, cos_sum = 0.0;
        size_t nonfinite = 0, pairs = 0;
        for (const float value : data) {
            if (!std::isfinite(value)) {
                ++nonfinite;
            } else {
                sum_sq += double(value) * value;
                max_abs = std::max(max_abs, std::abs(double(value)));
            }
        }
        const size_t width = size_t(tensor->ne[0]);
        for (int64_t i = 0; i < rows; ++i) {
            for (int64_t j = i + 1; j < rows; ++j) {
                double dot = 0.0, ni = 0.0, nj = 0.0;
                for (size_t k = 0; k < width; ++k) {
                    const double x = data[size_t(i) * width + k];
                    const double y = data[size_t(j) * width + k];
                    dot += x * y;
                    ni += x * x;
                    nj += y * y;
                }
                if (ni > 0.0 && nj > 0.0 && std::isfinite(dot)) {
                    cos_sum += dot / std::sqrt(ni * nj);
                    ++pairs;
                }
            }
        }
        std::fprintf(stderr, "rerot.audit.activation name=%s obs=%zu shape=%lld,%lld,%lld,%lld rms=%g max=%g cosine_mean=%g nonfinite=%zu\n",
            name.c_str(), step, (long long) tensor->ne[0], (long long) tensor->ne[1],
            (long long) tensor->ne[2], (long long) tensor->ne[3], std::sqrt(sum_sq / data.size()),
            max_abs, pairs ? cos_sum / pairs : 0.0, nonfinite);
        return true;
    }
};
