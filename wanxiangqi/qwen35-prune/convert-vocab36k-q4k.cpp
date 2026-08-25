#include "ggml.h"
#include "ggml-quants.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <omp.h>

struct gguf_context_deleter { void operator()(gguf_context * p) const { if (p) gguf_free(p); } };
using gguf_ptr = std::unique_ptr<gguf_context, gguf_context_deleter>;

static bool ends_with(const std::string & str, const std::string & suffix) {
    return str.size() >= suffix.size() &&
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

class output_writer {
public:
    output_writer(const std::string & path, gguf_context * meta) : path_(path), meta_(meta) {
        if (!gguf_write_to_file(meta, path.c_str(), true)) throw std::runtime_error("failed to write GGUF metadata");
        file_ = std::fopen(path.c_str(), "r+b");
        if (!file_) throw std::runtime_error("failed to reopen output GGUF");
        base_ = gguf_get_meta_size(meta_);
        const int64_t n_tensors = gguf_get_n_tensors(meta_);
        if (n_tensors <= 0) throw std::runtime_error("cannot preallocate GGUF with no tensors");
        const int64_t last = n_tensors - 1;
        const uint64_t final_size = (uint64_t) base_ + gguf_get_tensor_offset(meta_, last) + gguf_get_tensor_size(meta_, last);
        const int rc = ::posix_fallocate(fileno(file_), 0, (off_t) final_size);
        if (rc != 0) {
            throw std::runtime_error(std::string("posix_fallocate failed: ") + std::strerror(rc));
        }
    }
    ~output_writer() { if (file_) std::fclose(file_); }

    void write_tensor(int64_t tensor_id, size_t rel, const void * data, size_t size) {
        const size_t total = gguf_get_tensor_size(meta_, tensor_id);
        if (rel + size > total) throw std::runtime_error("output tensor write out of range");
        const size_t off = base_ + gguf_get_tensor_offset(meta_, tensor_id) + rel;
        const uint8_t * p = static_cast<const uint8_t *>(data);
        size_t done = 0;
        const int fd = fileno(file_);
        while (done < size) {
            const ssize_t n = ::pwrite(fd, p + done, size - done, (off_t) (off + done));
            if (n < 0) {
                if (errno == EINTR) continue;
                throw std::runtime_error(std::string("pwrite failed: ") + std::strerror(errno));
            }
            if (n == 0) throw std::runtime_error("short write");
            done += (size_t) n;
        }
    }

private:
    std::string path_;
    gguf_context * meta_ = nullptr;
    FILE * file_ = nullptr;
    size_t base_ = 0;
};

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " INPUT_192x256.gguf VOCAB_FIXTURE.gguf OUTPUT.gguf\n";
        return 1;
    }

    const std::string input_path = argv[1];
    const std::string vocab_fixture_path = argv[2];
    const std::string output_path = argv[3];

    std::cout << "Loading input model: " << input_path << "\n";
    gguf_init_params in_params { /*.no_alloc =*/ false, /*.ctx =*/ nullptr };
    gguf_ptr in_ctx(gguf_init_from_file(input_path.c_str(), in_params));
    if (!in_ctx) throw std::runtime_error("failed to open input GGUF");

    std::cout << "Loading vocab fixture: " << vocab_fixture_path << "\n";
    gguf_init_params vocab_params { /*.no_alloc =*/ false, /*.ctx =*/ nullptr };
    gguf_ptr vocab_ctx(gguf_init_from_file(vocab_fixture_path.c_str(), vocab_params));
    if (!vocab_ctx) throw std::runtime_error("failed to open vocab fixture GGUF");

    FILE * fin = std::fopen(input_path.c_str(), "rb");
    if (!fin) throw std::runtime_error("failed to open input file for reading");

    const int64_t n_tensors = gguf_get_n_tensors(in_ctx.get());
    std::cout << "Input tensor count: " << n_tensors << "\n";

    // Extract mapping from vocab fixture
    const int64_t f_token_key = gguf_find_key(vocab_ctx.get(), "tokenizer.ggml.tokens");
    const size_t n_vocab_dst = gguf_get_arr_n(vocab_ctx.get(), f_token_key);
    std::cout << "Target vocab size: " << n_vocab_dst << " tokens\n";

    const int64_t in_token_key = gguf_find_key(in_ctx.get(), "tokenizer.ggml.tokens");
    const size_t n_vocab_src = gguf_get_arr_n(in_ctx.get(), in_token_key);

    std::unordered_map<std::string, int> src_token_to_id;
    src_token_to_id.reserve(n_vocab_src * 2);
    for (size_t i = 0; i < n_vocab_src; ++i) {
        src_token_to_id.emplace(gguf_get_arr_str(in_ctx.get(), in_token_key, i), (int) i);
    }

    std::vector<int> vocab_mapping(n_vocab_dst);
    for (size_t i = 0; i < n_vocab_dst; ++i) {
        const std::string token = gguf_get_arr_str(vocab_ctx.get(), f_token_key, i);
        auto it = src_token_to_id.find(token);
        if (it == src_token_to_id.end()) {
            throw std::runtime_error("vocab token missing in source: " + token);
        }
        vocab_mapping[i] = it->second;
    }

    // Prepare output GGUF metadata
    gguf_ptr out_ctx(gguf_init_empty());
    // Copy all metadata from input
    gguf_set_kv(out_ctx.get(), in_ctx.get());

    // Overwrite tokenizer keys with vocab fixture keys
    gguf_set_kv(out_ctx.get(), vocab_ctx.get());

    // Re-apply model-specific architectural overrides from input
    for (const char * key : {"qwen35moe.expert_count",
                             "qwen35moe.expert_feed_forward_length",
                             "qwen35moe.block_count",
                             "qwen35moe.nextn_predict_layers"}) {
        const int64_t k = gguf_find_key(in_ctx.get(), key);
        if (k >= 0) {
            gguf_set_val_u32(out_ctx.get(), key, gguf_get_val_u32(in_ctx.get(), k));
        }
    }

    const int64_t vocab_size_key = gguf_find_key(out_ctx.get(), "qwen35moe.vocab_size");
    if (vocab_size_key >= 0) gguf_set_val_u32(out_ctx.get(), "qwen35moe.vocab_size", (uint32_t) n_vocab_dst);

    // Build tensor descriptors
    ggml_init_params desc_params { 8u * 1024u * 1024u, nullptr, true };
    std::unique_ptr<ggml_context, void(*)(ggml_context*)> desc_ctx(ggml_init(desc_params), ggml_free);

    int q6k_converted = 0;
    int shexp_converted = 0;
    int vocab_converted = 0;

    for (int64_t id = 0; id < n_tensors; ++id) {
        const std::string name = gguf_get_tensor_name(in_ctx.get(), id);
        const int64_t * ne = gguf_get_tensor_ne(in_ctx.get(), id);
        ggml_type type = (ggml_type) gguf_get_tensor_type(in_ctx.get(), id);

        std::array<int64_t, GGML_MAX_DIMS> out_ne {1, 1, 1, 1};
        for (int d = 0; d < GGML_MAX_DIMS; ++d) out_ne[d] = ne[d];

        int n_dims = 1;
        if (ne[3] > 1) n_dims = 4;
        else if (ne[2] > 1) n_dims = 3;
        else if (ne[1] > 1) n_dims = 2;

        if (name == "token_embd.weight" || name == "output.weight") {
            out_ne[1] = n_vocab_dst;
            if (name == "output.weight") {
                type = GGML_TYPE_Q4_K;
            }
            ++vocab_converted;
        } else if (type == GGML_TYPE_Q6_K) {
            type = GGML_TYPE_Q4_K;
            ++q6k_converted;
        }

        ggml_tensor * t = ggml_new_tensor(desc_ctx.get(), type, n_dims, out_ne.data());
        ggml_set_name(t, name.c_str());
        gguf_add_tensor(out_ctx.get(), t);
    }

    std::cout << "Output GGUF: " << vocab_converted << " vocab tensors resized, "
              << q6k_converted << " Q6_K tensors converted to Q4_K (shared experts retained in Q8_0)\n";

    std::cout << "Initializing output writer: " << output_path << "\n";
    output_writer writer(output_path, out_ctx.get());

    const size_t in_data_offset = gguf_get_data_offset(in_ctx.get());
    auto t0 = std::chrono::steady_clock::now();

    for (int64_t id = 0; id < n_tensors; ++id) {
        const std::string name = gguf_get_tensor_name(in_ctx.get(), id);
        const int64_t * in_ne = gguf_get_tensor_ne(in_ctx.get(), id);
        int in_dims = 1;
        if (in_ne[3] > 1) in_dims = 4;
        else if (in_ne[2] > 1) in_dims = 3;
        else if (in_ne[1] > 1) in_dims = 2;

        const ggml_type in_type = (ggml_type) gguf_get_tensor_type(in_ctx.get(), id);
        const size_t in_offset = in_data_offset + gguf_get_tensor_offset(in_ctx.get(), id);
        const size_t in_size = gguf_get_tensor_size(in_ctx.get(), id);

        const int64_t out_id = gguf_find_tensor(out_ctx.get(), name.c_str());
        const ggml_type out_type = (ggml_type) gguf_get_tensor_type(out_ctx.get(), out_id);
        const size_t out_size = gguf_get_tensor_size(out_ctx.get(), out_id);

        if (name == "token_embd.weight" || name == "output.weight") {
            const int64_t hidden = in_ne[0];
            const size_t src_row_size = ggml_row_size(in_type, hidden);
            std::vector<uint8_t> in_data(in_size);
            std::fseek(fin, (long) in_offset, SEEK_SET);
            if (std::fread(in_data.data(), 1, in_size, fin) != in_size) {
                throw std::runtime_error("failed to read " + name);
            }

            if (out_type == in_type) {
                // Direct row copy for token_embd (Q3_K)
                std::vector<uint8_t> out_data(out_size);
                for (size_t j = 0; j < n_vocab_dst; ++j) {
                    const int old_id = vocab_mapping[j];
                    std::memcpy(out_data.data() + j * src_row_size,
                                in_data.data() + (size_t) old_id * src_row_size,
                                src_row_size);
                }
                writer.write_tensor(out_id, 0, out_data.data(), out_size);
            } else {
                // Requantize from Q6_K to Q4_K row by row
                const size_t dst_row_size = ggml_row_size(out_type, hidden);
                std::vector<uint8_t> out_data(out_size);
                const auto * in_tt = ggml_get_type_traits(in_type);
                std::vector<float> decoded(hidden);
                for (size_t j = 0; j < n_vocab_dst; ++j) {
                    const int old_id = vocab_mapping[j];
                    in_tt->to_float(in_data.data() + (size_t) old_id * src_row_size, decoded.data(), hidden);
                    quantize_row_q4_K_ref(decoded.data(), (block_q4_K *)(out_data.data() + j * dst_row_size), hidden);
                }
                writer.write_tensor(out_id, 0, out_data.data(), out_size);
            }
        } else if (in_type == GGML_TYPE_Q6_K && out_type == GGML_TYPE_Q4_K) {
            // Convert Q6_K tensor to Q4_K
            const int64_t n_rows = (in_dims == 1) ? 1 : (in_dims == 2 ? in_ne[1] : in_ne[1] * in_ne[2]);
            const int64_t row_len = in_ne[0];
            const size_t src_row_size = ggml_row_size(in_type, row_len);
            const size_t dst_row_size = ggml_row_size(out_type, row_len);

            std::vector<uint8_t> in_data(in_size);
            std::vector<uint8_t> out_data(out_size);
            std::fseek(fin, (long) in_offset, SEEK_SET);
            if (std::fread(in_data.data(), 1, in_size, fin) != in_size) {
                throw std::runtime_error("failed to read " + name);
            }

            #pragma omp parallel for schedule(static)
            for (int64_t r = 0; r < n_rows; ++r) {
                std::vector<float> decoded(row_len);
                dequantize_row_q6_K((const block_q6_K *)(in_data.data() + r * src_row_size), decoded.data(), row_len);
                quantize_row_q4_K_ref(decoded.data(), (block_q4_K *)(out_data.data() + r * dst_row_size), row_len);
            }

            writer.write_tensor(out_id, 0, out_data.data(), out_size);
        } else {
            // Direct exact copy
            std::vector<uint8_t> buf(in_size);
            std::fseek(fin, (long) in_offset, SEEK_SET);
            if (std::fread(buf.data(), 1, in_size, fin) != in_size) {
                throw std::runtime_error("failed to read " + name);
            }
            writer.write_tensor(out_id, 0, buf.data(), out_size);
        }

        if ((id + 1) % 100 == 0 || id == n_tensors - 1) {
            std::cout << "Progress: " << (id + 1) << "/" << n_tensors << " tensors written\n";
        }
    }

    std::fclose(fin);

    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();
    std::cout << "\nSUCCESS: Created " << output_path << " in " << (elapsed_ms / 1000.0) << " seconds!\n";
    return 0;
}
