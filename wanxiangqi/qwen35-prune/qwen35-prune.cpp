#include "q2-opt-signed.h"
#include "enp-geometric-coreset.h"
#include "expert-replacement.h"
#include "gdn-geometry-plan.h"
#include "gdn-joint-replay.h"
#include "gdn-function-projection.h"
#include "gdn-observable-risk.h"

#include "build-info.h"
#include "common.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "imatrix-loader.h"
#include "llama.h"
#include "llama-model.h"
#include "nlohmann/json.hpp"
#if defined(QWEN35_PRUNE_USE_OPENSSL_SHA)
#include <openssl/evp.h>
#endif
extern "C" {
#include "sha256/sha256.h"
}

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

using json = nlohmann::ordered_json;
using namespace qwen35_prune;

namespace {

constexpr int N_LAYER_ALL = 41;
constexpr int N_LAYER_MAIN = 40;
constexpr int N_EXPERT = 256;
constexpr int N_EXPERT_DST = 192;
// With exactly 256-192=64 experts removed and Top-8 routing retained, the
// post-prune Top-8 must lie inside the original Top-(64+8)=Top-72 for every
// token, for every possible 192-expert survivor set.  This makes the offline
// router reconstruction exact on the sampled hidden states without imposing a
// candidate-coverage restriction on the optimizer.
constexpr int EXPERT_REPLACEMENT_CANDIDATES = (N_EXPERT - N_EXPERT_DST) + 8;
constexpr int EXPERT_WIDTH_SRC = 512;
constexpr int EXPERT_WIDTH_DST = 256;
constexpr int HIDDEN = 2048;
constexpr int GDN_DIM_SRC = 128;
constexpr int GDN_DIM_DST = 64;
constexpr int GDN_QK_HEADS = 16;
constexpr int GDN_V_HEADS = 32;
constexpr int VOCAB_DST = 36096;

const std::array<int, 30> RECURRENT_LAYERS = {
    0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18,
    20, 21, 22, 24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38,
};

struct gguf_context_deleter { void operator()(gguf_context * p) const { if (p) gguf_free(p); } };
struct ggml_context_deleter { void operator()(ggml_context * p) const { if (p) ggml_free(p); } };
using gguf_ptr = std::unique_ptr<gguf_context, gguf_context_deleter>;
using ggml_ptr = std::unique_ptr<ggml_context, ggml_context_deleter>;

class advisory_file_lock {
public:
    explicit advisory_file_lock(const std::string & path) : path_(path) {
        fd_ = ::open(path.c_str(), O_CREAT | O_RDWR, 0644);
        if (fd_ < 0) throw std::runtime_error("failed to open lock file: " + path);
        if (::flock(fd_, LOCK_EX | LOCK_NB) != 0) {
            const int saved_errno = errno;
            ::close(fd_);
            fd_ = -1;
            if (saved_errno == EWOULDBLOCK || saved_errno == EAGAIN) {
                throw std::runtime_error("another qwen35-prune process already owns lock: " + path);
            }
            throw std::runtime_error("failed to lock file: " + path + ": " + std::strerror(saved_errno));
        }
    }

    ~advisory_file_lock() {
        if (fd_ >= 0) {
            ::flock(fd_, LOCK_UN);
            ::close(fd_);
        }
    }

    advisory_file_lock(const advisory_file_lock &) = delete;
    advisory_file_lock & operator=(const advisory_file_lock &) = delete;

private:
    std::string path_;
    int fd_ = -1;
};

static json file_identity(const std::string & path) {
    struct stat st {};
    if (::stat(path.c_str(), &st) != 0) {
        throw std::runtime_error("stat failed for " + path + ": " + std::strerror(errno));
    }
    return {
        {"dev", (uint64_t) st.st_dev},
        {"ino", (uint64_t) st.st_ino},
        {"size", (uint64_t) st.st_size},
        {"mtime_sec", (int64_t) st.st_mtim.tv_sec},
        {"mtime_nsec", (int64_t) st.st_mtim.tv_nsec},
    };
}

static void require_file_identity(
        const json & expected,
        const std::string & path,
        const char * label) {
    if (expected != file_identity(path)) {
        throw std::runtime_error(std::string(label) + " file identity changed since checkpoint: " + path);
    }
}

static std::string hex_digest(const unsigned char digest[SHA256_DIGEST_SIZE]) {
    static const char * hex = "0123456789abcdef";
    std::string s(SHA256_DIGEST_SIZE * 2, '0');
    for (int i = 0; i < SHA256_DIGEST_SIZE; ++i) {
        s[2*i + 0] = hex[digest[i] >> 4];
        s[2*i + 1] = hex[digest[i] & 15];
    }
    return s;
}

static std::string sha256_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open for SHA-256: " + path);
#if defined(QWEN35_PRUNE_USE_OPENSSL_SHA)
    struct evp_md_ctx_deleter {
        void operator()(EVP_MD_CTX * p) const { if (p) EVP_MD_CTX_free(p); }
    };
    std::unique_ptr<EVP_MD_CTX, evp_md_ctx_deleter> sha(EVP_MD_CTX_new());
    if (!sha || EVP_DigestInit_ex(sha.get(), EVP_sha256(), nullptr) != 1) {
        throw std::runtime_error("OpenSSL SHA-256 initialization failed");
    }
#else
    sha256_t sha;
    sha256_init(&sha);
#endif
    std::array<unsigned char, 4 << 20> buf {};
    uint64_t hashed = 0;
    uint64_t next_report = uint64_t(4) << 30;
    while (in) {
        in.read((char *) buf.data(), buf.size());
        const std::streamsize n = in.gcount();
        if (n > 0) {
#if defined(QWEN35_PRUNE_USE_OPENSSL_SHA)
            if (EVP_DigestUpdate(sha.get(), buf.data(), (size_t) n) != 1) {
                throw std::runtime_error("OpenSSL SHA-256 update failed: " + path);
            }
#else
            sha256_update(&sha, buf.data(), (size_t) n);
#endif
            hashed += (uint64_t) n;
            if (hashed >= next_report) {
                std::cerr << "sha256: " << path << " " << (hashed >> 30) << " GiB read\n";
                next_report += uint64_t(4) << 30;
            }
        }
    }
    unsigned char digest[SHA256_DIGEST_SIZE];
#if defined(QWEN35_PRUNE_USE_OPENSSL_SHA)
    unsigned int digest_size = 0;
    if (EVP_DigestFinal_ex(sha.get(), digest, &digest_size) != 1 || digest_size != SHA256_DIGEST_SIZE) {
        throw std::runtime_error("OpenSSL SHA-256 finalization failed: " + path);
    }
#else
    sha256_final(&sha, digest);
#endif
    return hex_digest(digest);
}

static std::string current_executable_sha256() {
    // Linux resolves /proc/self/exe to the exact mapped executable. Binding
    // checkpoints to this hash prevents a rebuild from silently resuming a
    // partially generated plan/output with different surgery code.
    return sha256_file("/proc/self/exe");
}

static std::string sha256_fd_range(int fd, uint64_t offset, uint64_t size) {
#if defined(QWEN35_PRUNE_USE_OPENSSL_SHA)
    struct evp_md_ctx_deleter {
        void operator()(EVP_MD_CTX * p) const { if (p) EVP_MD_CTX_free(p); }
    };
    std::unique_ptr<EVP_MD_CTX, evp_md_ctx_deleter> sha(EVP_MD_CTX_new());
    if (!sha || EVP_DigestInit_ex(sha.get(), EVP_sha256(), nullptr) != 1) {
        throw std::runtime_error("OpenSSL SHA-256 initialization failed");
    }
#else
    sha256_t sha;
    sha256_init(&sha);
#endif
    std::array<unsigned char, 1 << 20> buf {};
    uint64_t done = 0;
    while (done < size) {
        const size_t want = (size_t) std::min<uint64_t>(buf.size(), size - done);
        ssize_t n = ::pread(fd, buf.data(), want, (off_t) (offset + done));
        if (n < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error(std::string("pread during SHA-256 failed: ") + std::strerror(errno));
        }
        if (n == 0) throw std::runtime_error("short read during SHA-256");
#if defined(QWEN35_PRUNE_USE_OPENSSL_SHA)
        if (EVP_DigestUpdate(sha.get(), buf.data(), (size_t) n) != 1) {
            throw std::runtime_error("OpenSSL SHA-256 update failed");
        }
#else
        sha256_update(&sha, buf.data(), (size_t) n);
#endif
        done += (uint64_t) n;
    }
    unsigned char digest[SHA256_DIGEST_SIZE];
#if defined(QWEN35_PRUNE_USE_OPENSSL_SHA)
    unsigned int digest_size = 0;
    if (EVP_DigestFinal_ex(sha.get(), digest, &digest_size) != 1 || digest_size != SHA256_DIGEST_SIZE) {
        throw std::runtime_error("OpenSSL SHA-256 finalization failed");
    }
#else
    sha256_final(&sha, digest);
#endif
    return hex_digest(digest);
}

static std::string sha256_bytes(const void * data, size_t size) {
    unsigned char digest[SHA256_DIGEST_SIZE];
    sha256_hash(digest, (const unsigned char *) data, size);
    return hex_digest(digest);
}

static bool starts_with(const std::string & s, const std::string & p) {
    return s.size() >= p.size() && std::equal(p.begin(), p.end(), s.begin());
}

static bool ends_with(const std::string & s, const std::string & p) {
    return s.size() >= p.size() && std::equal(p.rbegin(), p.rend(), s.rbegin());
}

static std::string type_name(ggml_type t) {
    const auto * tt = ggml_get_type_traits(t);
    return tt ? tt->type_name : "unknown";
}

class source_gguf {
public:
    explicit source_gguf(const std::string & path) : path_(path) {
        ggml_context * raw_ctx = nullptr;
        gguf_init_params params { true, &raw_ctx };
        gguf_.reset(gguf_init_from_file(path.c_str(), params));
        ctx_.reset(raw_ctx);
        if (!gguf_ || !ctx_) throw std::runtime_error("failed to parse GGUF: " + path);
        file_ = std::fopen(path.c_str(), "rb");
        if (!file_) throw std::runtime_error("failed to open GGUF payload: " + path);
        data_offset_ = gguf_get_data_offset(gguf_.get());
        alignment_ = gguf_get_alignment(gguf_.get());
        const int64_t n = gguf_get_n_tensors(gguf_.get());
        for (int64_t i = 0; i < n; ++i) tensor_ids_.emplace(gguf_get_tensor_name(gguf_.get(), i), i);
    }

    ~source_gguf() { if (file_) std::fclose(file_); }
    source_gguf(const source_gguf &) = delete;
    source_gguf & operator=(const source_gguf &) = delete;

    gguf_context * meta() const { return gguf_.get(); }
    ggml_context * tensors() const { return ctx_.get(); }
    size_t alignment() const { return alignment_; }
    int64_t tensor_count() const { return gguf_get_n_tensors(gguf_.get()); }

    ggml_tensor * tensor(const std::string & name) const {
        ggml_tensor * t = ggml_get_tensor(ctx_.get(), name.c_str());
        if (!t) throw std::runtime_error("missing tensor: " + name);
        return t;
    }

    int64_t tensor_id(const std::string & name) const {
        auto it = tensor_ids_.find(name);
        if (it == tensor_ids_.end()) throw std::runtime_error("missing tensor id: " + name);
        return it->second;
    }

    size_t tensor_offset(const std::string & name) const {
        return data_offset_ + gguf_get_tensor_offset(gguf_.get(), tensor_id(name));
    }

    size_t tensor_size(const std::string & name) const {
        return gguf_get_tensor_size(gguf_.get(), tensor_id(name));
    }

    void read_at(size_t offset, void * dst, size_t size) const {
        uint8_t * p = static_cast<uint8_t *>(dst);
        size_t done = 0;
        const int fd = fileno(file_);
        while (done < size) {
            const ssize_t n = ::pread(fd, p + done, size - done, (off_t) (offset + done));
            if (n < 0) {
                if (errno == EINTR) continue;
                throw std::runtime_error(std::string("pread failed: ") + std::strerror(errno));
            }
            if (n == 0) throw std::runtime_error("short read");
            done += (size_t) n;
        }
    }

    void read_tensor_bytes(const std::string & name, size_t rel, void * dst, size_t size) const {
        if (rel + size > tensor_size(name)) throw std::runtime_error("tensor read out of range: " + name);
        read_at(tensor_offset(name) + rel, dst, size);
    }

    uint32_t get_u32(const char * key) const {
        const int64_t id = gguf_find_key(gguf_.get(), key);
        if (id < 0) throw std::runtime_error(std::string("missing metadata: ") + key);
        switch (gguf_get_kv_type(gguf_.get(), id)) {
            case GGUF_TYPE_UINT32: return gguf_get_val_u32(gguf_.get(), id);
            case GGUF_TYPE_INT32:  return (uint32_t) gguf_get_val_i32(gguf_.get(), id);
            default: throw std::runtime_error(std::string("metadata is not u32/i32: ") + key);
        }
    }

    float get_f32(const char * key) const {
        const int64_t id = gguf_find_key(gguf_.get(), key);
        if (id < 0) throw std::runtime_error(std::string("missing metadata: ") + key);
        switch (gguf_get_kv_type(gguf_.get(), id)) {
            case GGUF_TYPE_FLOAT32: return gguf_get_val_f32(gguf_.get(), id);
            case GGUF_TYPE_FLOAT64: return (float) gguf_get_val_f64(gguf_.get(), id);
            default: throw std::runtime_error(std::string("metadata is not f32/f64: ") + key);
        }
    }

    std::string get_str(const char * key) const {
        const int64_t id = gguf_find_key(gguf_.get(), key);
        if (id < 0 || gguf_get_kv_type(gguf_.get(), id) != GGUF_TYPE_STRING) {
            throw std::runtime_error(std::string("missing string metadata: ") + key);
        }
        return gguf_get_val_str(gguf_.get(), id);
    }

private:
    std::string path_;
    gguf_ptr gguf_;
    ggml_ptr ctx_;
    FILE * file_ = nullptr;
    size_t data_offset_ = 0;
    size_t alignment_ = 0;
    std::unordered_map<std::string, int64_t> tensor_ids_;
};

static void require_source_contract(const source_gguf & src) {
    if (src.get_str("general.architecture") != "qwen35moe") throw std::runtime_error("architecture is not qwen35moe");
    const uint32_t block_count = src.get_u32("qwen35moe.block_count");
    if (block_count != 40 && block_count != 41) throw std::runtime_error("block_count must be 40 or 41");
    const int64_t expected_tensors = (block_count == 41) ? 753 : 733;
    if (src.tensor_count() != expected_tensors) throw std::runtime_error("expected exactly " + std::to_string(expected_tensors) + " tensors");
    if (src.get_u32("qwen35moe.embedding_length") != HIDDEN) throw std::runtime_error("embedding_length != 2048");
    const uint32_t expected_nextn = (block_count == 41) ? 1 : 0;
    if (src.get_u32("qwen35moe.nextn_predict_layers") != expected_nextn) throw std::runtime_error("nextn_predict_layers != " + std::to_string(expected_nextn));
    if (src.get_u32("qwen35moe.expert_count") != 256) throw std::runtime_error("expert_count != 256");
    if (src.get_u32("qwen35moe.expert_used_count") != 8) throw std::runtime_error("expert_used_count != 8");
    if (src.get_u32("qwen35moe.expert_feed_forward_length") != 512) throw std::runtime_error("expert width != 512");
    if (src.get_u32("qwen35moe.expert_shared_feed_forward_length") != 512) throw std::runtime_error("shared expert width != 512");
    if (src.get_u32("qwen35moe.ssm.state_size") != 128) throw std::runtime_error("ssm state_size != 128");
    if (src.get_u32("qwen35moe.ssm.inner_size") != 4096) throw std::runtime_error("ssm inner_size != 4096");
    if (src.get_u32("qwen35moe.ssm.group_count") != 16) throw std::runtime_error("ssm group_count != 16");
    if (src.get_u32("qwen35moe.ssm.time_step_rank") != 32) throw std::runtime_error("ssm time_step_rank != 32");

    int experts = 0;
    int ssm_out_count = 0;
    int vocab_dim_tensors = 0;
    for (int64_t i = 0; i < src.tensor_count(); ++i) {
        const std::string name = gguf_get_tensor_name(src.meta(), i);
        if (name.find("_exps.weight") != std::string::npos) ++experts;
        if (ends_with(name, "ssm_out.weight")) ++ssm_out_count;
        const ggml_tensor * t = src.tensor(name);
        for (int d = 0; d < ggml_n_dims(t); ++d) {
            if (t->ne[d] == 248320) {
                ++vocab_dim_tensors;
                break;
            }
        }
    }
    const int expected_experts = (int) block_count * 3;
    if (experts != expected_experts) throw std::runtime_error("expected exactly " + std::to_string(expected_experts) + " routed expert tensors");
    if (ssm_out_count != (int) RECURRENT_LAYERS.size()) throw std::runtime_error("expected exactly 30 recurrent ssm_out tensors");
    if (vocab_dim_tensors != 2) throw std::runtime_error("expected exactly two tensors carrying the 248320 vocab dimension");

    auto require_shape = [&](const std::string & name, std::initializer_list<int64_t> shape) {
        const ggml_tensor * t = src.tensor(name);
        if (ggml_n_dims(t) != (int) shape.size()) throw std::runtime_error("unexpected tensor rank: " + name);
        int d = 0;
        for (int64_t n : shape) {
            if (t->ne[d++] != n) throw std::runtime_error("unexpected tensor shape: " + name);
        }
    };

    for (int layer = 0; layer < (int) block_count; ++layer) {
        const std::string p = "blk." + std::to_string(layer) + ".";
        require_shape(p + "ffn_gate_exps.weight", {HIDDEN, EXPERT_WIDTH_SRC, N_EXPERT});
        require_shape(p + "ffn_up_exps.weight",   {HIDDEN, EXPERT_WIDTH_SRC, N_EXPERT});
        require_shape(p + "ffn_down_exps.weight", {EXPERT_WIDTH_SRC, HIDDEN, N_EXPERT});
    }

    for (int layer : RECURRENT_LAYERS) {
        const std::string p = "blk." + std::to_string(layer) + ".";
        require_shape(p + "attn_qkv.weight", {HIDDEN, 8192});
        require_shape(p + "attn_gate.weight", {HIDDEN, 4096});
        require_shape(p + "ssm_conv1d.weight", {4, 8192});
        require_shape(p + "ssm_norm.weight", {GDN_DIM_SRC});
        require_shape(p + "ssm_out.weight", {4096, HIDDEN});
    }

    require_shape("token_embd.weight", {HIDDEN, 248320});
    require_shape("output.weight", {HIDDEN, 248320});
}

// Expert-count pruning is a second-stage operation on the already materialized
// 384-wide model.  Keep this contract separate from require_source_contract():
// the original planner must continue rejecting anything other than the 512-wide
// teacher, while this path must preserve every axis except expert_count.
static void require_expert_count_source_contract(const source_gguf & src) {
    if (src.get_str("general.architecture") != "qwen35moe") throw std::runtime_error("architecture is not qwen35moe");
    const uint32_t block_count = src.get_u32("qwen35moe.block_count");
    if (block_count != 40 && block_count != 41) throw std::runtime_error("block_count must be 40 or 41");
    const int64_t expected_tensors = (block_count == 41) ? 753 : 733;
    if (src.tensor_count() != expected_tensors) throw std::runtime_error("expected exactly " + std::to_string(expected_tensors) + " tensors");
    if (src.get_u32("qwen35moe.embedding_length") != HIDDEN) throw std::runtime_error("embedding_length != 2048");
    const uint32_t expected_nextn = (block_count == 41) ? 1 : 0;
    if (src.get_u32("qwen35moe.nextn_predict_layers") != expected_nextn) throw std::runtime_error("nextn_predict_layers != " + std::to_string(expected_nextn));
    if (src.get_u32("qwen35moe.expert_count") != N_EXPERT) throw std::runtime_error("expert_count != 256");
    if (src.get_u32("qwen35moe.expert_used_count") != 8) throw std::runtime_error("expert_used_count != 8");
    if (src.get_u32("qwen35moe.expert_feed_forward_length") != EXPERT_WIDTH_DST) throw std::runtime_error("expert width != " + std::to_string(EXPERT_WIDTH_DST));
    if (src.get_u32("qwen35moe.expert_shared_feed_forward_length") != 512) throw std::runtime_error("shared expert width != 512");

    auto require_shape = [&](const std::string & name, std::initializer_list<int64_t> shape) {
        const ggml_tensor * t = src.tensor(name);
        if (ggml_n_dims(t) != (int) shape.size()) throw std::runtime_error("unexpected tensor rank: " + name);
        int d = 0;
        for (int64_t n : shape) {
            if (t->ne[d++] != n) throw std::runtime_error("unexpected tensor shape: " + name);
        }
    };
    for (int layer = 0; layer < (int) block_count; ++layer) {
        const std::string p = "blk." + std::to_string(layer) + ".";
        require_shape(p + "ffn_gate_inp.weight", {HIDDEN, N_EXPERT});
        require_shape(p + "ffn_gate_exps.weight", {HIDDEN, EXPERT_WIDTH_DST, N_EXPERT});
        require_shape(p + "ffn_up_exps.weight", {HIDDEN, EXPERT_WIDTH_DST, N_EXPERT});
        require_shape(p + "ffn_down_exps.weight", {EXPERT_WIDTH_DST, HIDDEN, N_EXPERT});
    }
}

static const common_imatrix_entry & require_entry(const common_imatrix & imat, const std::string & name) {
    auto it = imat.entries.find(name);
    if (it == imat.entries.end()) throw std::runtime_error("imatrix missing entry: " + name);
    return it->second;
}

static std::vector<float> mean_expert_stats(
        const common_imatrix_entry & e, int expert, int width, bool require_nonzero = true) {
    if ((int) e.counts.size() != N_EXPERT || (int) e.sums.size() != N_EXPERT * width) {
        throw std::runtime_error("unexpected expert imatrix dimensions");
    }
    const int64_t count = e.counts[(size_t) expert];
    if (count <= 0 && require_nonzero) throw std::runtime_error("expert has zero calibration routing count");
    std::vector<float> result(width, 0.0f);
    if (count > 0) {
        for (int i = 0; i < width; ++i) result[i] = e.sums[(size_t) expert * width + i] / count;
    }
    return result;
}

struct standard_enp_sample {
    int n = 0;
    std::vector<float> x;       // actual hidden-state medoids [n, HIDDEN]
    std::vector<double> weight;
    std::vector<double> transport;
    std::vector<double> radius;
    int64_t population = 0;
    double max_norm2 = 0.0;
    bool geometric = false;
    std::string method;
};

struct standard_enp_certification {
    standard_enp_sample remainder;
    standard_enp_sample tail;
    int64_t full_population = 0;
    bool tail_split = false;
};

static standard_enp_sample load_standard_enp_sample(const common_imatrix & imat, int layer) {
    standard_enp_sample out;
    const std::string coreset_name = "prune.enp.coreset.blk." + std::to_string(layer);
    const auto coreset_it = imat.entries.find(coreset_name);
    if (coreset_it != imat.entries.end()) {
        const auto & e = coreset_it->second;
        if (e.counts.empty() || e.sums.size() != e.counts.size() * (size_t) HIDDEN) {
            throw std::runtime_error("unexpected standard ENP coreset dimensions: " + coreset_name);
        }
        const auto & weights = require_entry(imat, "prune.enp.coreset_weight.blk." + std::to_string(layer));
        const auto & transport = require_entry(imat, "prune.enp.coreset_transport.blk." + std::to_string(layer));
        const auto & radius = require_entry(imat, "prune.enp.coreset_radius.blk." + std::to_string(layer));
        out.n = (int) e.counts.size();
        out.x = e.sums;
        if ((int) weights.sums.size() != out.n || (int) transport.sums.size() != out.n ||
            (int) radius.sums.size() != out.n) {
            throw std::runtime_error("standard ENP coreset auxiliary dimensions mismatch");
        }
        out.weight.resize((size_t) out.n);
        out.transport.resize((size_t) out.n);
        out.radius.resize((size_t) out.n);
        for (int r = 0; r < out.n; ++r) {
            if (e.counts[(size_t) r] != 1 || !std::isfinite(weights.sums[(size_t) r]) ||
                weights.sums[(size_t) r] <= 0.0f || !std::isfinite(transport.sums[(size_t) r]) ||
                transport.sums[(size_t) r] < 0.0f || !std::isfinite(radius.sums[(size_t) r]) ||
                radius.sums[(size_t) r] < 0.0f) {
                throw std::runtime_error("invalid standard ENP coreset metadata");
            }
            out.weight[(size_t) r] = weights.sums[(size_t) r];
            out.transport[(size_t) r] = transport.sums[(size_t) r];
            out.radius[(size_t) r] = radius.sums[(size_t) r];
        }
        out.geometric = true;
        out.method = "task-blind-streaming-medoid-original-l2-v1";
    } else {
        const std::string name = "prune.enp.sample.blk." + std::to_string(layer);
        const auto & e = require_entry(imat, name);
        if (e.counts.empty() || e.sums.size() != e.counts.size() * (size_t) HIDDEN) {
            throw std::runtime_error("unexpected standard ENP sample dimensions: " + name);
        }
        out.n = (int) e.counts.size();
        out.x = e.sums;
        out.weight.assign((size_t) out.n, 1.0);
        out.transport.assign((size_t) out.n, 0.0);
        out.radius.assign((size_t) out.n, 0.0);
        for (int r = 0; r < out.n; ++r) {
            if (e.counts[(size_t) r] != 1) throw std::runtime_error("non-unit standard ENP sample count: " + name);
        }
        out.geometric = false;
        out.method = "bottom-k-uniform-without-replacement";
    }
    for (int r = 0; r < out.n; ++r) {
        for (int i = 0; i < HIDDEN; ++i) {
            if (!std::isfinite(out.x[(size_t) r * HIDDEN + i])) {
                throw std::runtime_error("non-finite ENP hidden state at layer " + std::to_string(layer));
            }
        }
    }

    const std::string population_name = "prune.enp.population.blk." + std::to_string(layer);
    const auto & population = require_entry(imat, population_name);
    if (population.sums.size() != 1 || population.counts.size() != 1 || population.counts[0] != 1 ||
        !std::isfinite(population.sums[0]) || population.sums[0] <= 0.0f) {
        throw std::runtime_error("invalid standard ENP population entry: " + population_name);
    }
    out.population = (int64_t) std::llround(population.sums[0]);
    if (out.population < out.n) {
        throw std::runtime_error("standard ENP representatives exceed population at layer " + std::to_string(layer));
    }
    const std::string max_norm_name = "prune.enp.max_norm2.blk." + std::to_string(layer);
    const auto & max_norm = require_entry(imat, max_norm_name);
    if (max_norm.sums.size() != 1 || max_norm.counts.size() != 1 || max_norm.counts[0] != 1 ||
        !std::isfinite(max_norm.sums[0]) || max_norm.sums[0] <= 0.0f) {
        throw std::runtime_error("invalid standard ENP max-norm entry: " + max_norm_name);
    }
    out.max_norm2 = max_norm.sums[0];
    const double total_weight = std::accumulate(out.weight.begin(), out.weight.end(), 0.0);
    if (out.geometric && std::fabs(total_weight - (double) out.population) > 0.5) {
        throw std::runtime_error("standard ENP coreset weight does not equal population");
    }
    return out;
}

static standard_enp_certification load_standard_enp_cert_sample(const common_imatrix & imat, int layer) {
    standard_enp_certification cert;
    auto & out = cert.remainder;
    const std::string name = "prune.enp.cert_sample.blk." + std::to_string(layer);
    const auto & e = require_entry(imat, name);
    if (e.counts.empty() || e.sums.size() != e.counts.size() * (size_t) HIDDEN) {
        throw std::runtime_error("unexpected standard ENP certification sample dimensions: " + name);
    }
    out.n = (int) e.counts.size();
    out.x = e.sums;
    out.weight.assign((size_t) out.n, 1.0);
    out.transport.assign((size_t) out.n, 0.0);
    out.radius.assign((size_t) out.n, 0.0);
    for (int r = 0; r < out.n; ++r) {
        if (e.counts[(size_t) r] != 1) {
            throw std::runtime_error("non-unit ENP certification sample count: " + name);
        }
        for (int i = 0; i < HIDDEN; ++i) {
            if (!std::isfinite(out.x[(size_t) r * HIDDEN + i])) {
                throw std::runtime_error("non-finite ENP certification hidden state at layer " + std::to_string(layer));
            }
        }
    }
    const auto & population = require_entry(imat, "prune.enp.population.blk." + std::to_string(layer));
    if (population.sums.size() != 1 || population.counts.size() != 1 || population.counts[0] != 1 ||
        !std::isfinite(population.sums[0]) || population.sums[0] <= 0.0f) {
        throw std::runtime_error("invalid ENP certification population entry");
    }
    cert.full_population = (int64_t) std::llround(population.sums[0]);
    if (cert.full_population < out.n) throw std::runtime_error("ENP certification sample exceeds population");

    const std::string tail_name = "prune.enp.cert_tail.blk." + std::to_string(layer);
    const auto tail_it = imat.entries.find(tail_name);
    cert.tail_split = tail_it != imat.entries.end();
    if (cert.tail_split) {
        const auto & rem_population = require_entry(imat, "prune.enp.cert_remainder_population.blk." + std::to_string(layer));
        const auto & rem_max_norm = require_entry(imat, "prune.enp.cert_remainder_max_norm2.blk." + std::to_string(layer));
        if (rem_population.sums.size() != 1 || rem_population.counts.size() != 1 || rem_population.counts[0] != 1 ||
            !std::isfinite(rem_population.sums[0]) || rem_population.sums[0] < 0.0f ||
            rem_max_norm.sums.size() != 1 || rem_max_norm.counts.size() != 1 || rem_max_norm.counts[0] != 1 ||
            !std::isfinite(rem_max_norm.sums[0]) || rem_max_norm.sums[0] < 0.0f) {
            throw std::runtime_error("invalid ENP tail-split certification remainder metadata");
        }
        out.population = (int64_t) std::llround(rem_population.sums[0]);
        if (out.population < out.n || (out.population > 0 && rem_max_norm.sums[0] <= 0.0f)) {
            throw std::runtime_error("invalid ENP tail-split certification remainder population");
        }
        out.max_norm2 = rem_max_norm.sums[0];

        const auto & tail_entry = tail_it->second;
        auto & tail = cert.tail;
        if (tail_entry.sums.size() != tail_entry.counts.size() * (size_t) HIDDEN) {
            throw std::runtime_error("unexpected ENP certification tail dimensions");
        }
        tail.n = (int) tail_entry.counts.size();
        tail.x = tail_entry.sums;
        tail.weight.assign((size_t) tail.n, 1.0);
        tail.transport.assign((size_t) tail.n, 0.0);
        tail.radius.assign((size_t) tail.n, 0.0);
        for (int r = 0; r < tail.n; ++r) {
            if (tail_entry.counts[(size_t) r] != 1) throw std::runtime_error("non-unit ENP certification tail count");
            for (int i = 0; i < HIDDEN; ++i) {
                if (!std::isfinite(tail.x[(size_t) r * HIDDEN + i])) {
                    throw std::runtime_error("non-finite ENP certification tail hidden state");
                }
            }
        }
        tail.population = tail.n;
        const auto & full_max_norm = require_entry(imat, "prune.enp.max_norm2.blk." + std::to_string(layer));
        if (full_max_norm.sums.size() != 1 || full_max_norm.counts.size() != 1 || full_max_norm.counts[0] != 1 ||
            !std::isfinite(full_max_norm.sums[0]) || full_max_norm.sums[0] <= 0.0f) {
            throw std::runtime_error("invalid ENP certification full max-norm entry");
        }
        tail.max_norm2 = full_max_norm.sums[0];
        tail.geometric = false;
        tail.method = "exact-high-hidden-norm-tail";
        if (out.population + tail.n != cert.full_population) {
            throw std::runtime_error("ENP tail-split certification population does not add up");
        }
    } else {
        out.population = cert.full_population;
        const auto & max_norm = require_entry(imat, "prune.enp.max_norm2.blk." + std::to_string(layer));
        if (max_norm.sums.size() != 1 || max_norm.counts.size() != 1 || max_norm.counts[0] != 1 ||
            !std::isfinite(max_norm.sums[0]) || max_norm.sums[0] <= 0.0f) {
            throw std::runtime_error("invalid ENP certification max-norm entry");
        }
        out.max_norm2 = max_norm.sums[0];
    }
    out.geometric = false;
    out.method = cert.tail_split
        ? "independent-bottom-k-uniform-without-replacement-on-nontail"
        : "independent-bottom-k-uniform-without-replacement";
    return cert;
}

static standard_enp_sample coarsen_standard_enp_sample(const standard_enp_sample & input, int target) {
    if (!input.geometric || target >= input.n) return input;
    std::vector<enp_geometric_cluster> clusters;
    clusters.reserve((size_t) input.n);
    for (int r = 0; r < input.n; ++r) {
        enp_geometric_cluster c;
        c.center.assign(input.x.begin() + (size_t) r * HIDDEN, input.x.begin() + (size_t) (r + 1) * HIDDEN);
        c.weight = (int64_t) std::llround(input.weight[(size_t) r]);
        c.transport = input.transport[(size_t) r];
        c.radius = input.radius[(size_t) r];
        c.source_id = (uint64_t) r;
        clusters.push_back(std::move(c));
    }
    clusters = enp_geometric_coarsen(clusters, target);
    standard_enp_sample out;
    out.n = (int) clusters.size();
    out.population = input.population;
    out.max_norm2 = input.max_norm2;
    out.geometric = true;
    out.method = input.method;
    out.x.resize((size_t) out.n * HIDDEN);
    out.weight.resize((size_t) out.n);
    out.transport.resize((size_t) out.n);
    out.radius.resize((size_t) out.n);
    for (int r = 0; r < out.n; ++r) {
        std::copy(clusters[(size_t) r].center.begin(), clusters[(size_t) r].center.end(),
                  out.x.begin() + (size_t) r * HIDDEN);
        out.weight[(size_t) r] = clusters[(size_t) r].weight;
        out.transport[(size_t) r] = clusters[(size_t) r].transport;
        out.radius[(size_t) r] = clusters[(size_t) r].radius;
    }
    return out;
}

static void read_quant_row(
        const source_gguf & src,
        const std::string & name,
        size_t row_index,
        std::vector<uint8_t> & packed,
        std::vector<float> & decoded) {
    ggml_tensor * t = src.tensor(name);
    const size_t row_size = ggml_row_size(t->type, t->ne[0]);
    packed.resize(row_size);
    decoded.resize((size_t) t->ne[0]);
    src.read_tensor_bytes(name, row_index * row_size, packed.data(), row_size);
    const auto * tt = ggml_get_type_traits(t->type);
    if (!tt || !tt->to_float) throw std::runtime_error("type cannot be dequantized: " + type_name(t->type));
    tt->to_float(packed.data(), decoded.data(), t->ne[0]);
}

struct gdn_geometry_prior {
    std::array<std::vector<double>, GDN_QK_HEADS> qk;
    std::vector<double> v;
};

static void decode_packed_row(
        const ggml_tensor * t,
        const uint8_t * packed,
        std::vector<float> & decoded) {
    decoded.resize((size_t) t->ne[0]);
    if (t->type == GGML_TYPE_F32) {
        std::memcpy(decoded.data(), packed, (size_t) t->ne[0] * sizeof(float));
        return;
    }
    const auto * traits = ggml_get_type_traits(t->type);
    if (!traits || !traits->to_float) {
        throw std::runtime_error("GDN geometry prior cannot decode type: " + type_name(t->type));
    }
    traits->to_float(packed, decoded.data(), t->ne[0]);
}

static double vector_norm2(const std::vector<float> & x) {
    long double sum = 0.0;
    for (float v : x) sum += (long double) v * v;
    return (double) sum;
}

// A deliberately low-freedom prior: each coordinate gets only a diagonal
// weight-space energy.  Empirical off-diagonal geometry comes from the one
// teacher pass; calibration therefore cannot invent support for a direction
// that is absent from the stock projection+FIR operator.
static gdn_geometry_prior compute_gdn_geometry_prior(const source_gguf & src, int layer) {
    const std::string prefix = "blk." + std::to_string(layer) + ".";
    const std::string qkv_name = prefix + "attn_qkv.weight";
    const std::string conv_name = prefix + "ssm_conv1d.weight";
    const std::string gate_name = prefix + "attn_gate.weight";
    const ggml_tensor * qkv = src.tensor(qkv_name);
    const ggml_tensor * conv = src.tensor(conv_name);
    const ggml_tensor * gate = src.tensor(gate_name);
    if (qkv->ne[0] != HIDDEN || qkv->ne[1] != 8192 ||
        conv->ne[0] != 4 || conv->ne[1] != 8192 ||
        gate->ne[0] != HIDDEN || gate->ne[1] != 4096) {
        throw std::runtime_error("unexpected GDN geometry-prior tensor shape at layer " + std::to_string(layer));
    }

    const size_t qkv_row_size = ggml_row_size(qkv->type, qkv->ne[0]);
    const size_t conv_row_size = ggml_row_size(conv->type, conv->ne[0]);
    const size_t gate_row_size = ggml_row_size(gate->type, gate->ne[0]);
    std::vector<uint8_t> qkv_packed(src.tensor_size(qkv_name));
    std::vector<uint8_t> conv_packed(src.tensor_size(conv_name));
    std::vector<uint8_t> gate_packed(src.tensor_size(gate_name));
    src.read_tensor_bytes(qkv_name, 0, qkv_packed.data(), qkv_packed.size());
    src.read_tensor_bytes(conv_name, 0, conv_packed.data(), conv_packed.size());
    src.read_tensor_bytes(gate_name, 0, gate_packed.data(), gate_packed.size());

    gdn_geometry_prior out;
    for (auto & h : out.qk) h.assign(GDN_DIM_SRC, 0.0);
    out.v.assign(GDN_DIM_SRC, 0.0);
    std::vector<float> w, c;

    for (int row = 0; row < 8192; ++row) {
        decode_packed_row(qkv, qkv_packed.data() + (size_t) row * qkv_row_size, w);
        decode_packed_row(conv, conv_packed.data() + (size_t) row * conv_row_size, c);
        const double energy = vector_norm2(w) * vector_norm2(c);
        if (!std::isfinite(energy) || energy < 0.0) {
            throw std::runtime_error("non-finite GDN geometry-prior energy");
        }
        if (row < 2048) {
            const int h = row / GDN_DIM_SRC;
            const int i = row % GDN_DIM_SRC;
            out.qk[(size_t) h][(size_t) i] += 0.5 * energy;
        } else if (row < 4096) {
            const int krow = row - 2048;
            const int h = krow / GDN_DIM_SRC;
            const int i = krow % GDN_DIM_SRC;
            out.qk[(size_t) h][(size_t) i] += 0.5 * energy;
        } else {
            const int vrow = row - 4096;
            const int i = vrow % GDN_DIM_SRC;
            out.v[(size_t) i] += energy / GDN_V_HEADS;
        }
    }
    for (int row = 0; row < 4096; ++row) {
        decode_packed_row(gate, gate_packed.data() + (size_t) row * gate_row_size, w);
        const double energy = vector_norm2(w);
        if (!std::isfinite(energy) || energy < 0.0) {
            throw std::runtime_error("non-finite GDN gate geometry-prior energy");
        }
        const int i = row % GDN_DIM_SRC;
        out.v[(size_t) i] += energy / GDN_V_HEADS;
    }
    return out;
}

struct decoded_expert_slice {
    int ne0 = 0;
    int ne1 = 0;
    std::vector<float> values;

    const float * row(int r) const {
        return values.data() + (size_t) r * ne0;
    }
};

// All routed-expert tensors are laid out with expert as ne2, so a complete
// expert is one contiguous packed slice. Reading/dequantizing it once avoids
// millions of small pread() calls during production Q2-aware planning.
static decoded_expert_slice decode_expert_slice(
        const source_gguf & src,
        const std::string & name,
        int expert,
        int expected_ne0,
        int expected_ne1) {
    ggml_tensor * t = src.tensor(name);
    if (t->ne[0] != expected_ne0 || t->ne[1] != expected_ne1 || t->ne[2] != N_EXPERT) {
        throw std::runtime_error("unexpected expert shape: " + name);
    }
    const auto * tt = ggml_get_type_traits(t->type);
    if (!tt || !tt->to_float) throw std::runtime_error("cannot dequantize expert tensor: " + name);

    const size_t row_size = ggml_row_size(t->type, t->ne[0]);
    const size_t expert_bytes = row_size * (size_t) t->ne[1];
    std::vector<uint8_t> packed(expert_bytes);
    src.read_tensor_bytes(name, (size_t) expert * expert_bytes, packed.data(), packed.size());

    decoded_expert_slice out;
    out.ne0 = expected_ne0;
    out.ne1 = expected_ne1;
    out.values.resize((size_t) expected_ne0 * expected_ne1);
    for (int r = 0; r < expected_ne1; ++r) {
        tt->to_float(packed.data() + (size_t) r * row_size,
                     out.values.data() + (size_t) r * expected_ne0,
                     expected_ne0);
    }
    return out;
}

static inline float standard_enp_silu(float x) {
    if (x >= 0.0f) return x / (1.0f + std::exp(-x));
    const float e = std::exp(x);
    return x * e / (1.0f + e);
}

// Exact ENP projection formula evaluated on task-blind hidden-state
// representatives.  Legacy imatrices use a simple random sample without
// replacement.  New imatrices use a weighted geometric medoid coreset and
// carry deterministic original-space transport/radius bounds.
struct standard_enp_score_stats {
    std::vector<double> mean;
    std::vector<double> variance;
    std::vector<double> abs_bound;
    std::vector<double> error_bound;
    double transport_total = 0.0;
    double max_radius = 0.0;
    double min_local_output_margin = std::numeric_limits<double>::infinity();
    bool geometric_certificate_finite = true;
};

static std::vector<double> down_column_norm2(const decoded_expert_slice & down);

static standard_enp_score_stats score_standard_enp_on_sample_decoded(
        const decoded_expert_slice & gate,
        const decoded_expert_slice & up,
        const decoded_expert_slice & down,
        const standard_enp_sample & sample,
        int layer,
        int expert) {
    (void) layer;
    (void) expert;
    standard_enp_score_stats out;
    out.mean.assign(EXPERT_WIDTH_SRC, 0.0);
    out.variance.assign(EXPERT_WIDTH_SRC, 0.0);
    out.abs_bound.assign(EXPERT_WIDTH_SRC, 0.0);
    out.error_bound.assign(EXPERT_WIDTH_SRC, 0.0);
    std::vector<double> sumsq(EXPERT_WIDTH_SRC, 0.0);
    std::array<double, EXPERT_WIDTH_SRC> gate_l2 {};
    std::array<double, EXPERT_WIDTH_SRC> up_l2 {};
    std::array<double, EXPERT_WIDTH_SRC> down_l2 {};
    std::array<float, EXPERT_WIDTH_SRC> g_pre {};
    std::array<float, EXPERT_WIDTH_SRC> u_pre {};
    std::array<float, EXPERT_WIDTH_SRC> m {};
    std::array<float, EXPERT_WIDTH_SRC> dty {};
    std::array<float, HIDDEN> y {};
    std::array<double, EXPERT_WIDTH_SRC> local_lip_c {};
    std::array<double, EXPERT_WIDTH_SRC> local_sup_c {};

    if (sample.n <= 0 || sample.population <= 0 || sample.max_norm2 <= 0.0 ||
        sample.weight.size() != (size_t) sample.n || sample.transport.size() != (size_t) sample.n ||
        sample.radius.size() != (size_t) sample.n) {
        throw std::runtime_error("standard ENP sample is empty or malformed");
    }
    const double normalizer = sample.geometric
        ? (double) sample.population
        : std::accumulate(sample.weight.begin(), sample.weight.end(), 0.0);
    if (!(normalizer > 0.0)) throw std::runtime_error("standard ENP sample has zero total weight");

    const auto down_norm2 = down_column_norm2(down);
    for (int j = 0; j < EXPERT_WIDTH_SRC; ++j) {
        double gate_norm2 = 0.0;
        double up_norm2 = 0.0;
        const float * wg = gate.row(j);
        const float * wu = up.row(j);
        for (int i = 0; i < HIDDEN; ++i) {
            gate_norm2 += (double) wg[i] * wg[i];
            up_norm2 += (double) wu[i] * wu[i];
        }
        gate_l2[(size_t) j] = std::sqrt(std::max(0.0, gate_norm2));
        up_l2[(size_t) j] = std::sqrt(std::max(0.0, up_norm2));
        down_l2[(size_t) j] = std::sqrt(std::max(0.0, down_norm2[(size_t) j]));
        // For f_j=<d_j m_j, y>/||y||, Cauchy gives |f_j|<=||d_j m_j||.
        // Since |SiLU(z)|<=|z|,
        //   |m_j(x)| <= ||g_j|| ||u_j|| ||x||^2.
        // sample.max_norm2 is the exact maximum ||x||^2 over the full finite
        // calibration population, hence this is a population-wide range bound.
        out.abs_bound[(size_t) j] = sample.max_norm2 *
            std::sqrt(std::max(0.0, gate_norm2 * up_norm2 * down_norm2[(size_t) j]));
    }

    for (int r = 0; r < sample.n; ++r) {
        const float * x = sample.x.data() + (size_t) r * HIDDEN;
        for (int j = 0; j < EXPERT_WIDTH_SRC; ++j) {
            const float * wg = gate.row(j);
            const float * wu = up.row(j);
            float g = 0.0f;
            float u = 0.0f;
            #pragma omp simd reduction(+:g,u)
            for (int i = 0; i < HIDDEN; ++i) {
                g += wg[i] * x[i];
                u += wu[i] * x[i];
            }
            g_pre[(size_t) j] = g;
            u_pre[(size_t) j] = u;
            m[(size_t) j] = standard_enp_silu(g) * u;
        }

        double norm2 = 0.0;
        for (int i = 0; i < HIDDEN; ++i) {
            const float * wd = down.row(i);
            float yi = 0.0f;
            #pragma omp simd reduction(+:yi)
            for (int j = 0; j < EXPERT_WIDTH_SRC; ++j) yi += wd[j] * m[(size_t) j];
            y[(size_t) i] = yi;
            norm2 += (double) yi * yi;
        }

        std::fill(dty.begin(), dty.end(), 0.0f);
        for (int i = 0; i < HIDDEN; ++i) {
            const float * wd = down.row(i);
            const float yi = y[(size_t) i];
            #pragma omp simd
            for (int j = 0; j < EXPERT_WIDTH_SRC; ++j) dty[(size_t) j] += wd[j] * yi;
        }

        const double output_norm = std::sqrt(norm2);
        const double denom = output_norm + 1.0e-8;
        const double w = sample.weight[(size_t) r];
        for (int j = 0; j < EXPERT_WIDTH_SRC; ++j) {
            const double value = (double) m[(size_t) j] * dty[(size_t) j] / denom;
            out.mean[(size_t) j] += w * value;
            sumsq[(size_t) j] += w * value * value;
        }

        if (sample.geometric && sample.transport[(size_t) r] > 0.0) {
            // A rigorous local SiLU Lipschitz bound: for z>=0,
            // z*sigma(z)*(1-sigma(z)) <= z*exp(-z) <= 1/e; the z<0 case is
            // symmetric after taking absolute values.  Therefore
            // |SiLU'| <= 1 + 1/e everywhere.
            constexpr double SILU_LIP = 1.3678794411714423216;
            const double R = sample.radius[(size_t) r];
            double lip_y = 0.0;
            for (int j = 0; j < EXPERT_WIDTH_SRC; ++j) {
                const double sup_silu = std::fabs((double) standard_enp_silu(g_pre[(size_t) j])) +
                    SILU_LIP * gate_l2[(size_t) j] * R;
                const double sup_u = std::fabs((double) u_pre[(size_t) j]) + up_l2[(size_t) j] * R;
                const double lip_m = sup_silu * up_l2[(size_t) j] +
                    sup_u * SILU_LIP * gate_l2[(size_t) j];
                local_lip_c[(size_t) j] = down_l2[(size_t) j] * lip_m;
                local_sup_c[(size_t) j] = down_l2[(size_t) j] *
                    (std::fabs((double) m[(size_t) j]) + lip_m * R);
                lip_y += local_lip_c[(size_t) j];
            }
            const double gamma = output_norm - lip_y * R;
            out.min_local_output_margin = std::min(out.min_local_output_margin, gamma);
            const double transport_mass = sample.transport[(size_t) r] / normalizer;
            const double probability_mass = w / normalizer;
            for (int j = 0; j < EXPERT_WIDTH_SRC; ++j) {
                // Even if the cell crosses an expert-output zero and no useful
                // directional Lipschitz constant exists, ENP still has the
                // population-wide magnitude bound |f_j| <= abs_bound_j.  Thus
                // replacing an arbitrary point in this cell by its medoid can
                // change the cell contribution by at most 2*B_j*mass.  This
                // keeps the certificate rigorous and finite instead of
                // serializing an artificial infinity/null interval.
                const double range_fallback = 2.0 * out.abs_bound[(size_t) j] * probability_mass;
                double cell_error = range_fallback;
                if (gamma > 0.0 && std::isfinite(gamma)) {
                    // q(y)=y/(||y||+eps) has operator-Lipschitz constant at
                    // most 2/(gamma+eps) on this cell.  With
                    // f_j=<c_j,q(y)>, product Lipschitz gives the bound below.
                    const double lip_f = local_lip_c[(size_t) j] +
                        2.0 * local_sup_c[(size_t) j] * lip_y / (gamma + 1.0e-8);
                    cell_error = std::min(cell_error, transport_mass * lip_f);
                }
                out.error_bound[(size_t) j] += cell_error;
            }
        }
    }
    for (int j = 0; j < EXPERT_WIDTH_SRC; ++j) {
        out.mean[(size_t) j] /= normalizer;
        out.variance[(size_t) j] = std::max(0.0,
            sumsq[(size_t) j] / normalizer - out.mean[(size_t) j] * out.mean[(size_t) j]);
    }
    if (sample.geometric) {
        out.transport_total = std::accumulate(sample.transport.begin(), sample.transport.end(), 0.0);
        out.max_radius = *std::max_element(sample.radius.begin(), sample.radius.end());
        if (!std::isfinite(out.min_local_output_margin)) out.min_local_output_margin = 0.0;
    }
    return out;
}

static std::vector<double> score_standard_enp_mean_only_decoded(
        const decoded_expert_slice & gate,
        const decoded_expert_slice & up,
        const decoded_expert_slice & down,
        const standard_enp_sample & sample) {
    if (sample.n <= 0 || sample.weight.size() != (size_t) sample.n) {
        throw std::runtime_error("standard ENP mean-only sample is empty or malformed");
    }
    const double normalizer = sample.geometric
        ? (double) sample.population
        : std::accumulate(sample.weight.begin(), sample.weight.end(), 0.0);
    if (!(normalizer > 0.0)) throw std::runtime_error("standard ENP mean-only sample has zero total weight");

    // The downstream planner needs only the ENP mean.  Precompute the expert's
    // down Gram matrix once so each representative can use
    //   D^T(Dm) = (D^T D)m,  ||Dm||^2 = m^T(D^T D)m
    // instead of traversing the 2048x512 down matrix twice per sample.
    std::vector<double> gram((size_t) EXPERT_WIDTH_SRC * EXPERT_WIDTH_SRC, 0.0);
    for (int i = 0; i < HIDDEN; ++i) {
        const float * wd = down.row(i);
        for (int j = 0; j < EXPERT_WIDTH_SRC; ++j) {
            double * grow = gram.data() + (size_t) j * EXPERT_WIDTH_SRC;
            const double wj = wd[j];
            #pragma omp simd
            for (int k = 0; k <= j; ++k) grow[k] += wj * (double) wd[k];
        }
    }
    for (int j = 0; j < EXPERT_WIDTH_SRC; ++j) {
        for (int k = 0; k < j; ++k) {
            gram[(size_t) k * EXPERT_WIDTH_SRC + j] =
                gram[(size_t) j * EXPERT_WIDTH_SRC + k];
        }
    }

    std::vector<double> mean(EXPERT_WIDTH_SRC, 0.0);
    std::array<float, EXPERT_WIDTH_SRC> m {};
    std::array<double, EXPERT_WIDTH_SRC> dty {};
    for (int r = 0; r < sample.n; ++r) {
        const float * x = sample.x.data() + (size_t) r * HIDDEN;
        for (int j = 0; j < EXPERT_WIDTH_SRC; ++j) {
            const float * wg = gate.row(j);
            const float * wu = up.row(j);
            float g = 0.0f;
            float u = 0.0f;
            #pragma omp simd reduction(+:g,u)
            for (int i = 0; i < HIDDEN; ++i) {
                g += wg[i] * x[i];
                u += wu[i] * x[i];
            }
            m[(size_t) j] = standard_enp_silu(g) * u;
        }

        double norm2 = 0.0;
        for (int j = 0; j < EXPERT_WIDTH_SRC; ++j) {
            const double * grow = gram.data() + (size_t) j * EXPERT_WIDTH_SRC;
            double v = 0.0;
            #pragma omp simd reduction(+:v)
            for (int k = 0; k < EXPERT_WIDTH_SRC; ++k) {
                v += grow[k] * (double) m[(size_t) k];
            }
            dty[(size_t) j] = v;
            norm2 += (double) m[(size_t) j] * v;
        }
        const double denom = std::sqrt(std::max(0.0, norm2)) + 1.0e-8;
        const double w = sample.weight[(size_t) r];
        for (int j = 0; j < EXPERT_WIDTH_SRC; ++j) {
            mean[(size_t) j] += w * (double) m[(size_t) j] * dty[(size_t) j] / denom;
        }
    }
    for (double & v : mean) v /= normalizer;
    return mean;
}

static std::vector<double> standard_enp_bernstein_serfling_intervals(
        const standard_enp_score_stats & score,
        const standard_enp_sample & sample,
        double & confidence_log) {
    constexpr double confidence_delta_total = 1.0e-6;
    constexpr double kappa = 7.0 / 3.0 + 3.0 / std::sqrt(2.0);
    constexpr uint64_t n_score_functions = (uint64_t) N_LAYER_MAIN * N_EXPERT * EXPERT_WIDTH_SRC;
    confidence_log = std::log(10.0 * (double) n_score_functions / confidence_delta_total);
    std::vector<double> interval(EXPERT_WIDTH_SRC, 0.0);
    if (sample.n == sample.population) return interval;
    const double n = (double) sample.n;
    const double N = (double) sample.population;
    const double rho = sample.n <= sample.population / 2
        ? 1.0 - (n - 1.0) / N
        : (1.0 - n / N) * (1.0 + 1.0 / n);
    for (int j = 0; j < EXPERT_WIDTH_SRC; ++j) {
        const double sigma_hat = std::sqrt(std::max(0.0, score.variance[(size_t) j]));
        const double range = 2.0 * score.abs_bound[(size_t) j];
        interval[(size_t) j] = sigma_hat * std::sqrt(2.0 * rho * confidence_log / n) +
                               kappa * range * confidence_log / n;
        if (!std::isfinite(interval[(size_t) j])) {
            throw std::runtime_error("non-finite ENP Bernstein-Serfling confidence radius");
        }
    }
    return interval;
}

static std::vector<double> down_column_norm2(const decoded_expert_slice & down) {
    if (down.ne0 != EXPERT_WIDTH_SRC || down.ne1 != HIDDEN) {
        throw std::runtime_error("unexpected decoded down expert dimensions");
    }
    std::vector<double> norms(EXPERT_WIDTH_SRC, 0.0);
    for (int r = 0; r < HIDDEN; ++r) {
        const float * row = down.row(r);
        for (int j = 0; j < EXPERT_WIDTH_SRC; ++j) norms[j] += (double) row[j] * row[j];
    }
    return norms;
}

using survivor_group = std::array<int, Q2_OPT_BLOCK_SIZE>;

struct down_group_cache {
    survivor_group ids {};
    std::vector<q2_opt_block> blocks;
    std::array<double, Q2_OPT_BLOCK_SIZE> contribution {};
    double error = 0.0;
};

struct neuron_fingerprint {
    double log_rms = 0.0;
    std::array<double, 5> v {};
};

static double fixed_scale_weighted_error(float w, float h, float d) {
    if (h <= 0.0f) return 0.0;
    if (d == 0.0f) return (double) h * w * w;
    const double sign = std::signbit(d) ? -1.0 : 1.0;
    const double y = sign * (double) w;
    const double a = std::fabs((double) d);
    int c;
    if (y <= -0.5 * a) c = -1;
    else if (y <= 0.5 * a) c = 0;
    else if (y <= 1.5 * a) c = 1;
    else c = 2;
    const double diff = y - c * a;
    return (double) h * diff * diff;
}

static down_group_cache make_down_group_cache(
        const decoded_expert_slice & down,
        const std::vector<float> & ha,
        const survivor_group & ids) {
    if (down.ne0 != EXPERT_WIDTH_SRC || down.ne1 != HIDDEN || ha.size() != EXPERT_WIDTH_SRC) {
        throw std::runtime_error("down group dimensions mismatch");
    }
    down_group_cache out;
    out.ids = ids;
    out.blocks.resize(HIDDEN);
    float w[Q2_OPT_BLOCK_SIZE];
    float h[Q2_OPT_BLOCK_SIZE];
    for (int p = 0; p < Q2_OPT_BLOCK_SIZE; ++p) h[p] = ha[(size_t) ids[(size_t) p]];

    for (int r = 0; r < HIDDEN; ++r) {
        const float * src_row = down.row(r);
        for (int p = 0; p < Q2_OPT_BLOCK_SIZE; ++p) w[p] = src_row[ids[(size_t) p]];
        const auto metrics = encode_q2_opt(w, h, out.blocks[(size_t) r], q2_opt_mode::signed_scale);
        out.error += metrics.weighted_sse;
        const float d = ggml_fp16_to_fp32(out.blocks[(size_t) r].d);
        for (int p = 0; p < Q2_OPT_BLOCK_SIZE; ++p) {
            const uint8_t code = (out.blocks[(size_t) r].qs[p / 4] >> (2 * (p % 4))) & 3u;
            const double recon = (double) ((int) code - 1) * d;
            const double diff = (double) w[p] - recon;
            out.contribution[(size_t) p] += (double) h[p] * diff * diff;
        }
    }
    return out;
}

// Exact weighted objective for a candidate 64-neuron down block without
// materializing Q2 bytes or per-position contribution caches. Shortlisted
// local-search candidates only need this scalar; the winner is rebuilt once
// with make_down_group_cache() when it is committed to the next round.
static double down_group_error_exact(
        const decoded_expert_slice & down,
        const std::vector<float> & ha,
        const survivor_group & ids) {
    if (down.ne0 != EXPERT_WIDTH_SRC || down.ne1 != HIDDEN || ha.size() != EXPERT_WIDTH_SRC) {
        throw std::runtime_error("down group dimensions mismatch");
    }
    float w[Q2_OPT_BLOCK_SIZE];
    float h[Q2_OPT_BLOCK_SIZE];
    for (int p = 0; p < Q2_OPT_BLOCK_SIZE; ++p) h[p] = ha[(size_t) ids[(size_t) p]];

    double error = 0.0;
    for (int r = 0; r < HIDDEN; ++r) {
        const float * src_row = down.row(r);
        for (int p = 0; p < Q2_OPT_BLOCK_SIZE; ++p) w[p] = src_row[ids[(size_t) p]];
        error += score_q2_opt(w, h, q2_opt_mode::signed_scale).weighted_sse;
    }
    return error;
}

static int position_in_group(const survivor_group & group, int neuron) {
    for (int p = 0; p < Q2_OPT_BLOCK_SIZE; ++p) {
        if (group[(size_t) p] == neuron) return p;
    }
    return -1;
}

static double fixed_scale_insertion_error(
        const decoded_expert_slice & down,
        const std::vector<float> & ha,
        int neuron,
        const down_group_cache & group) {
    double error = 0.0;
    const float h = ha[(size_t) neuron];
    for (int r = 0; r < HIDDEN; ++r) {
        const float d = ggml_fp16_to_fp32(group.blocks[(size_t) r].d);
        error += fixed_scale_weighted_error(down.row(r)[neuron], h, d);
    }
    return error;
}

static neuron_fingerprint make_neuron_fingerprint(
        const decoded_expert_slice & down,
        const std::vector<double> & norm2,
        int neuron) {
    neuron_fingerprint fp;
    std::array<float, HIDDEN> abs_values {};
    float max_abs = 0.0f;
    for (int r = 0; r < HIDDEN; ++r) {
        const float a = std::fabs(down.row(r)[neuron]);
        abs_values[(size_t) r] = a;
        max_abs = std::max(max_abs, a);
    }
    auto percentile = [&](size_t k) {
        std::nth_element(abs_values.begin(), abs_values.begin() + k, abs_values.end());
        return (double) abs_values[k];
    };
    const double rms = std::sqrt(std::max(0.0, norm2[(size_t) neuron]) / HIDDEN);
    fp.log_rms = std::log(std::max(rms, 1e-30));
    fp.v = {
        fp.log_rms,
        std::log(std::max((double) max_abs, 1e-30)),
        percentile(HIDDEN / 4),
        percentile(HIDDEN / 2),
        percentile(3 * HIDDEN / 4),
    };
    return fp;
}

static double fingerprint_distance2(const neuron_fingerprint & fp, const std::array<double, 5> & centroid) {
    double d2 = 0.0;
    for (int i = 0; i < 5; ++i) {
        const double d = fp.v[(size_t) i] - centroid[(size_t) i];
        d2 += d * d;
    }
    return d2;
}

struct survivor_optimization_result {
    std::vector<int> mapping;
    int packing_swaps = 0;
};

static survivor_optimization_result optimize_q2_packing(
        const decoded_expert_slice & down,
        const std::vector<float> & ha,
        const std::vector<double> & norm2,
        const std::vector<int> & initial_selected) {
    if (initial_selected.size() != EXPERT_WIDTH_DST || norm2.size() != EXPERT_WIDTH_SRC) {
        throw std::runtime_error("Q2 packing dimensions mismatch");
    }

    // BAOMU.md initial packing: stable sort by down-column log RMS, then split
    // the 256 survivors into four contiguous 64-neuron Q2 blocks.
    std::vector<int> ordered = initial_selected;
    std::stable_sort(ordered.begin(), ordered.end(), [&](int a, int b) {
        const double ra = std::log(std::max(std::sqrt(std::max(0.0, norm2[(size_t) a]) / HIDDEN), 1e-30));
        const double rb = std::log(std::max(std::sqrt(std::max(0.0, norm2[(size_t) b]) / HIDDEN), 1e-30));
        if (ra != rb) return ra < rb;
        return a < b;
    });

    std::array<down_group_cache, 4> groups;
    for (int g = 0; g < 4; ++g) {
        survivor_group ids {};
        for (int p = 0; p < Q2_OPT_BLOCK_SIZE; ++p) {
            ids[(size_t) p] = ordered[(size_t) g * Q2_OPT_BLOCK_SIZE + p];
        }
        groups[(size_t) g] = make_down_group_cache(down, ha, ids);
    }

    survivor_optimization_result result;

    // Fingerprints are only needed for the currently selected 256 neurons.
    std::array<neuron_fingerprint, EXPERT_WIDTH_SRC> fingerprints {};
    std::array<bool, EXPERT_WIDTH_SRC> have_fingerprint {};
    auto fingerprint = [&](int id) -> const neuron_fingerprint & {
        if (!have_fingerprint[(size_t) id]) {
            fingerprints[(size_t) id] = make_neuron_fingerprint(down, norm2, id);
            have_fingerprint[(size_t) id] = true;
        }
        return fingerprints[(size_t) id];
    };

    struct packing_pair {
        double proxy = 0.0;
        int group_a = -1;
        int group_b = -1;
        int neuron_a = -1;
        int neuron_b = -1;
        int pos_a = -1;
        int pos_b = -1;
    };

    // Two group-swap sweeps. Each group contributes the 8 fingerprint points
    // farthest from its centroid; fixed-scale proxy ranks all cross-group pairs
    // and only the best four receive exact two-block evaluation.
    for (int sweep = 0; sweep < 2; ++sweep) {
        std::array<std::vector<int>, 4> edge;
        for (int g = 0; g < 4; ++g) {
            std::array<double, 5> centroid {};
            for (int id : groups[(size_t) g].ids) {
                const auto & fp = fingerprint(id);
                for (int k = 0; k < 5; ++k) centroid[(size_t) k] += fp.v[(size_t) k];
            }
            for (double & x : centroid) x /= Q2_OPT_BLOCK_SIZE;

            std::vector<std::pair<double,int>> dist;
            dist.reserve(Q2_OPT_BLOCK_SIZE);
            for (int id : groups[(size_t) g].ids) dist.emplace_back(fingerprint_distance2(fingerprint(id), centroid), id);
            std::stable_sort(dist.begin(), dist.end(), [](const auto & a, const auto & b) {
                if (a.first != b.first) return a.first > b.first;
                return a.second < b.second;
            });
            for (int i = 0; i < 8; ++i) edge[(size_t) g].push_back(dist[(size_t) i].second);
        }

        std::vector<std::array<double, 4>> insertion(EXPERT_WIDTH_SRC);
        for (auto & x : insertion) x.fill(std::numeric_limits<double>::quiet_NaN());
        for (int ga = 0; ga < 4; ++ga) {
            for (int id : edge[(size_t) ga]) {
                for (int gb = 0; gb < 4; ++gb) {
                    if (ga != gb && std::isnan(insertion[(size_t) id][(size_t) gb])) {
                        insertion[(size_t) id][(size_t) gb] = fixed_scale_insertion_error(down, ha, id, groups[(size_t) gb]);
                    }
                }
            }
        }

        std::vector<packing_pair> pairs;
        for (int ga = 0; ga < 4; ++ga) {
            for (int gb = ga + 1; gb < 4; ++gb) {
                for (int a : edge[(size_t) ga]) {
                    const int pa = position_in_group(groups[(size_t) ga].ids, a);
                    for (int b : edge[(size_t) gb]) {
                        const int pb = position_in_group(groups[(size_t) gb].ids, b);
                        const double proxy =
                            insertion[(size_t) a][(size_t) gb] - groups[(size_t) ga].contribution[(size_t) pa] +
                            insertion[(size_t) b][(size_t) ga] - groups[(size_t) gb].contribution[(size_t) pb];
                        pairs.push_back({proxy, ga, gb, a, b, pa, pb});
                    }
                }
            }
        }
        std::stable_sort(pairs.begin(), pairs.end(), [](const packing_pair & a, const packing_pair & b) {
            if (a.proxy != b.proxy) return a.proxy < b.proxy;
            return std::tie(a.group_a, a.group_b, a.neuron_a, a.neuron_b) <
                   std::tie(b.group_a, b.group_b, b.neuron_a, b.neuron_b);
        });

        bool improved = false;
        double best_delta = 0.0;
        packing_pair best_pair;
        const int exact_n = std::min<int>(4, pairs.size());
        for (int i = 0; i < exact_n; ++i) {
            const auto & pair = pairs[(size_t) i];
            survivor_group ida = groups[(size_t) pair.group_a].ids;
            survivor_group idb = groups[(size_t) pair.group_b].ids;
            std::swap(ida[(size_t) pair.pos_a], idb[(size_t) pair.pos_b]);
            const double ea = down_group_error_exact(down, ha, ida);
            const double eb = down_group_error_exact(down, ha, idb);
            const double delta = ea + eb - groups[(size_t) pair.group_a].error - groups[(size_t) pair.group_b].error;
            if (delta < best_delta || (delta == best_delta && improved &&
                    std::tie(pair.group_a, pair.group_b, pair.neuron_a, pair.neuron_b) <
                    std::tie(best_pair.group_a, best_pair.group_b, best_pair.neuron_a, best_pair.neuron_b))) {
                improved = delta < 0.0;
                best_delta = delta;
                best_pair = pair;
            }
        }
        if (!improved) break;
        survivor_group committed_a = groups[(size_t) best_pair.group_a].ids;
        survivor_group committed_b = groups[(size_t) best_pair.group_b].ids;
        std::swap(committed_a[(size_t) best_pair.pos_a], committed_b[(size_t) best_pair.pos_b]);
        auto best_a = make_down_group_cache(down, ha, committed_a);
        auto best_b = make_down_group_cache(down, ha, committed_b);
        groups[(size_t) best_pair.group_a] = std::move(best_a);
        groups[(size_t) best_pair.group_b] = std::move(best_b);
        ++result.packing_swaps;
    }

    // Canonical final order: groups by median log RMS, then source id inside a
    // group. This makes output_to_input deterministic and is the sole mapping
    // consumed by gate/up row gather and down column gather.
    std::array<int, 4> group_order {0, 1, 2, 3};
    std::array<double, 4> median_rms {};
    for (int g = 0; g < 4; ++g) {
        std::array<double, Q2_OPT_BLOCK_SIZE> x {};
        for (int p = 0; p < Q2_OPT_BLOCK_SIZE; ++p) {
            const int id = groups[(size_t) g].ids[(size_t) p];
            x[(size_t) p] = std::log(std::max(std::sqrt(std::max(0.0, norm2[(size_t) id]) / HIDDEN), 1e-30));
        }
        std::sort(x.begin(), x.end());
        median_rms[(size_t) g] = 0.5 * (x[31] + x[32]);
    }
    std::stable_sort(group_order.begin(), group_order.end(), [&](int a, int b) {
        if (median_rms[(size_t) a] != median_rms[(size_t) b]) return median_rms[(size_t) a] < median_rms[(size_t) b];
        return a < b;
    });
    result.mapping.reserve(EXPERT_WIDTH_DST);
    for (int g : group_order) {
        auto ids = groups[(size_t) g].ids;
        std::sort(ids.begin(), ids.end());
        result.mapping.insert(result.mapping.end(), ids.begin(), ids.end());
    }
    return result;
}

static json make_expert_record(
        const source_gguf & src,
        const common_imatrix & imat,
        const standard_enp_sample & enp_sample,
        const standard_enp_certification * cert_sample,
        int layer,
        int expert,
        bool quick,
        bool allow_uncovered) {
    (void) quick;
    (void) allow_uncovered;
    const std::string down_name = "blk." + std::to_string(layer) + ".ffn_down_exps.weight";
    int64_t routing_count = -1;
    bool routing_count_available = false;
    if (const auto it = imat.entries.find(down_name); it != imat.entries.end()) {
        if (it->second.counts.size() != N_EXPERT) {
            throw std::runtime_error("expert routing count dimension mismatch: " + down_name);
        }
        routing_count = it->second.counts[(size_t) expert];
        routing_count_available = true;
    }
    const std::string prefix = "blk." + std::to_string(layer) + ".";
    const auto gate = decode_expert_slice(src, prefix + "ffn_gate_exps.weight", expert, HIDDEN, EXPERT_WIDTH_SRC);
    const auto up   = decode_expert_slice(src, prefix + "ffn_up_exps.weight",   expert, HIDDEN, EXPERT_WIDTH_SRC);
    const auto down = decode_expert_slice(src, prefix + "ffn_down_exps.weight", expert, EXPERT_WIDTH_SRC, HIDDEN);

    standard_enp_score_stats enp;
    std::vector<double> interval(EXPERT_WIDTH_SRC, 0.0);
    std::vector<std::pair<double, int>> ranking;
    std::vector<int> attempted_points;
    json adaptive_stability = json::array();
    std::vector<int> previous_top256;
    int points_used = enp_sample.n;
    std::string confidence_method;
    double confidence_delta_total_value = 0.0;
    double confidence_log_value = 0.0;
    bool top256_certified = false;
    bool cert_sample_evaluated = false;
    bool cert_sample_certified = false;
    int cert_rank_disagreement = -1;
    int cert_sample_size = 0;
    int cert_tail_size = 0;
    int64_t cert_remainder_population = 0;
    double cert_remainder_max_norm2 = 0.0;
    bool cert_projection_mass_evaluated = false;
    double cert_full_projection_mass = 0.0;
    double cert_selected_projection_mass = 0.0;
    double cert_oracle_projection_mass = 0.0;
    double cert_selected_retained_fraction = 0.0;
    double cert_oracle_retained_fraction = 0.0;
    double cert_excess_loss_fraction = 0.0;
    double cert_selected384_projection_mass = 0.0;
    double cert_oracle384_projection_mass = 0.0;
    double cert_selected384_retained_fraction = 0.0;
    double cert_oracle384_retained_fraction = 0.0;
    double cert_excess384_loss_fraction = 0.0;
    double min_selected_lcb = -std::numeric_limits<double>::infinity();
    double max_pruned_ucb = std::numeric_limits<double>::infinity();

    auto rank_and_certify = [&](const standard_enp_score_stats & score, const std::vector<double> & err) {
        ranking.clear();
        ranking.reserve(EXPERT_WIDTH_SRC);
        for (int j = 0; j < EXPERT_WIDTH_SRC; ++j) ranking.emplace_back(score.mean[(size_t) j], j);
        std::stable_sort(ranking.begin(), ranking.end(), [](const auto & a, const auto & b) {
            if (a.first != b.first) return a.first > b.first;
            return a.second < b.second;
        });
        min_selected_lcb = std::numeric_limits<double>::infinity();
        max_pruned_ucb = -std::numeric_limits<double>::infinity();
        for (int i = 0; i < EXPERT_WIDTH_SRC; ++i) {
            const int j = ranking[(size_t) i].second;
            if (i < EXPERT_WIDTH_DST) {
                min_selected_lcb = std::min(min_selected_lcb, score.mean[(size_t) j] - err[(size_t) j]);
            } else {
                max_pruned_ucb = std::max(max_pruned_ucb, score.mean[(size_t) j] + err[(size_t) j]);
            }
        }
        return min_selected_lcb > max_pruned_ucb;
    };

    if (enp_sample.geometric) {
        // Adaptive R uses nested deterministic coarsenings of the stored
        // maximum medoid coreset.  The representative score is always the
        // original ENP projection score; geometry only supplies an integration
        // error interval and never changes the survivor objective.
        std::vector<int> levels;
        for (int requested = 64; requested < enp_sample.n; requested *= 2) {
            levels.push_back(requested);
            if (requested > std::numeric_limits<int>::max() / 2) break;
        }
        if (levels.empty() || levels.back() != enp_sample.n) levels.push_back(enp_sample.n);
        for (int requested : levels) {
            const auto level = coarsen_standard_enp_sample(enp_sample, requested);
            attempted_points.push_back(level.n);
            enp = score_standard_enp_on_sample_decoded(gate, up, down, level, layer, expert);
            interval = enp.error_bound;
            points_used = level.n;
            top256_certified = rank_and_certify(enp, interval);
            std::vector<int> current_top256;
            current_top256.reserve(EXPERT_WIDTH_DST);
            for (int i = 0; i < EXPERT_WIDTH_DST; ++i) {
                current_top256.push_back(ranking[(size_t) i].second);
            }
            std::sort(current_top256.begin(), current_top256.end());
            int changed_from_previous = 0;
            if (!previous_top256.empty()) {
                std::vector<int> symmetric_difference;
                symmetric_difference.reserve(EXPERT_WIDTH_DST * 2);
                std::set_symmetric_difference(
                    previous_top256.begin(), previous_top256.end(),
                    current_top256.begin(), current_top256.end(),
                    std::back_inserter(symmetric_difference));
                changed_from_previous = (int) symmetric_difference.size() / 2;
            }
            double level_max_interval = 0.0;
            for (double v : interval) level_max_interval = std::max(level_max_interval, v);
            adaptive_stability.push_back({
                {"points", level.n},
                {"changed_survivors_from_previous", changed_from_previous},
                {"score_margin_256_257",
                    ranking[(size_t) EXPERT_WIDTH_DST - 1].first - ranking[(size_t) EXPERT_WIDTH_DST].first},
                {"max_confidence_radius", level_max_interval},
                {"top256_certified", top256_certified},
            });
            previous_top256 = std::move(current_top256);
            if (top256_certified) break;
        }
        confidence_method = "geometric-transport-local-lipschitz-v1";
    } else {
        // Legacy compatibility: Bardenet-Maillard empirical
        // Bernstein-Serfling confidence radius for the old uniform-WOR sample.
        enp = score_standard_enp_on_sample_decoded(gate, up, down, enp_sample, layer, expert);
        attempted_points.push_back(enp_sample.n);
        interval = standard_enp_bernstein_serfling_intervals(enp, enp_sample, confidence_log_value);
        confidence_method = "empirical-bernstein-serfling-two-sided-global-union-v1";
        confidence_delta_total_value = 1.0e-6;
        top256_certified = rank_and_certify(enp, interval);
    }

    // Geometric coreset scores propose the survivor set.  When an independent
    // uniform-WOR certification reservoir is present, use it only to certify
    // that fixed set against the full finite calibration population.  The
    // certification sample never changes the ranking or survivor identity.
    if (enp_sample.geometric && cert_sample && !top256_certified) {
        if (cert_sample->full_population != enp_sample.population || cert_sample->remainder.geometric) {
            throw std::runtime_error("invalid independent ENP certification sample");
        }
        cert_sample_evaluated = true;
        cert_sample_size = cert_sample->remainder.n;
        cert_tail_size = cert_sample->tail_split ? cert_sample->tail.n : 0;
        cert_remainder_population = cert_sample->remainder.population;
        cert_remainder_max_norm2 = cert_sample->remainder.max_norm2;
        const auto remainder_score = score_standard_enp_on_sample_decoded(
            gate, up, down, cert_sample->remainder, layer, expert);
        double cert_log = 0.0;
        auto cert_interval = standard_enp_bernstein_serfling_intervals(
            remainder_score, cert_sample->remainder, cert_log);
        std::vector<double> cert_mean = remainder_score.mean;
        const double remainder_mass = (double) cert_sample->remainder.population /
            (double) cert_sample->full_population;
        for (double & radius : cert_interval) radius *= remainder_mass;
        for (double & mean : cert_mean) mean *= remainder_mass;
        if (cert_sample->tail_split && cert_sample->tail.n > 0) {
            const auto tail_score = score_standard_enp_on_sample_decoded(
                gate, up, down, cert_sample->tail, layer, expert);
            const double tail_mass = (double) cert_sample->tail.n /
                (double) cert_sample->full_population;
            for (int j = 0; j < EXPERT_WIDTH_SRC; ++j) {
                cert_mean[(size_t) j] += tail_mass * tail_score.mean[(size_t) j];
            }
        }

        std::array<uint8_t, EXPERT_WIDTH_SRC> selected_mask {};
        for (int i = 0; i < EXPERT_WIDTH_DST; ++i) {
            selected_mask[(size_t) ranking[(size_t) i].second] = 1;
        }
        min_selected_lcb = std::numeric_limits<double>::infinity();
        max_pruned_ucb = -std::numeric_limits<double>::infinity();
        std::vector<std::pair<double, int>> cert_ranking;
        cert_ranking.reserve(EXPERT_WIDTH_SRC);
        for (int j = 0; j < EXPERT_WIDTH_SRC; ++j) {
            cert_ranking.emplace_back(cert_mean[(size_t) j], j);
            if (selected_mask[(size_t) j]) {
                min_selected_lcb = std::min(
                    min_selected_lcb, cert_mean[(size_t) j] - cert_interval[(size_t) j]);
            } else {
                max_pruned_ucb = std::max(
                    max_pruned_ucb, cert_mean[(size_t) j] + cert_interval[(size_t) j]);
            }
        }
        std::stable_sort(cert_ranking.begin(), cert_ranking.end(), [](const auto & a, const auto & b) {
            if (a.first != b.first) return a.first > b.first;
            return a.second < b.second;
        });
        cert_full_projection_mass = std::accumulate(cert_mean.begin(), cert_mean.end(), 0.0);
        for (int j = 0; j < EXPERT_WIDTH_SRC; ++j) {
            if (selected_mask[(size_t) j]) cert_selected_projection_mass += cert_mean[(size_t) j];
        }
        for (int i = 0; i < EXPERT_WIDTH_DST; ++i) {
            cert_oracle_projection_mass += cert_ranking[(size_t) i].first;
        }
        constexpr int ENP_DIAGNOSTIC_WIDTH_384 = 384;
        for (int i = 0; i < ENP_DIAGNOSTIC_WIDTH_384; ++i) {
            cert_selected384_projection_mass += cert_mean[(size_t) ranking[(size_t) i].second];
            cert_oracle384_projection_mass += cert_ranking[(size_t) i].first;
        }
        if (cert_full_projection_mass > 0.0 && std::isfinite(cert_full_projection_mass)) {
            cert_projection_mass_evaluated = true;
            cert_selected_retained_fraction = cert_selected_projection_mass / cert_full_projection_mass;
            cert_oracle_retained_fraction = cert_oracle_projection_mass / cert_full_projection_mass;
            cert_excess_loss_fraction =
                (cert_oracle_projection_mass - cert_selected_projection_mass) / cert_full_projection_mass;
            cert_selected384_retained_fraction = cert_selected384_projection_mass / cert_full_projection_mass;
            cert_oracle384_retained_fraction = cert_oracle384_projection_mass / cert_full_projection_mass;
            cert_excess384_loss_fraction =
                (cert_oracle384_projection_mass - cert_selected384_projection_mass) / cert_full_projection_mass;
        }
        int disagreement = 0;
        for (int i = 0; i < EXPERT_WIDTH_DST; ++i) {
            disagreement += selected_mask[(size_t) cert_ranking[(size_t) i].second] ? 0 : 1;
        }
        cert_rank_disagreement = disagreement;
        cert_sample_certified = min_selected_lcb > max_pruned_ucb;
        top256_certified = cert_sample_certified;
        interval = cert_interval;
        confidence_method = cert_sample->tail_split
            ? "exact-high-norm-tail-plus-independent-uniform-wor-bernstein-serfling-fixed-set-v1"
            : "independent-uniform-wor-bernstein-serfling-fixed-set-v1";
        confidence_delta_total_value = 1.0e-6;
        confidence_log_value = cert_log;
    }

    int score_positive = 0;
    int score_zero = 0;
    int score_negative = 0;
    double score_min = std::numeric_limits<double>::infinity();
    for (double v : enp.mean) {
        if (v > 0.0) ++score_positive;
        else if (v < 0.0) ++score_negative;
        else ++score_zero;
        score_min = std::min(score_min, v);
    }

    std::vector<int> selected;
    selected.reserve(EXPERT_WIDTH_DST);
    for (int i = 0; i < EXPERT_WIDTH_DST; ++i) selected.push_back(ranking[(size_t) i].second);
    double max_interval = 0.0;
    for (double v : interval) max_interval = std::max(max_interval, v);
    const double score_margin = ranking[(size_t) EXPERT_WIDTH_DST - 1].first - ranking[(size_t) EXPERT_WIDTH_DST].first;
    const bool full_population_exact = enp_sample.geometric &&
        enp_sample.n == enp_sample.population &&
        enp.transport_total == 0.0 && enp.max_radius == 0.0;
    // "Unique" is a statement about the full calibration population, not
    // merely about the approximate coreset ranking.  A certified interval
    // separation proves uniqueness; otherwise only an exact full-population
    // evaluation may make that claim.
    const bool top256_unique = top256_certified || (full_population_exact && score_margin > 0.0);

    // Output order is semantically free as long as gate/up/down share it.
    // Source-id order preserves the teacher's local neuron adjacency and makes
    // the Q4_K_M ablation deterministic without a quantizer-aware packing step.
    std::sort(selected.begin(), selected.end());

    return {
        {"layer", layer},
        {"expert", expert},
        {"output_to_input", selected},
        {"routing_count", routing_count},
        {"routing_count_available", routing_count_available},
        {"zero_routing_weight_only_fallback", false},
        {"selection_method", enp_sample.geometric
            ? "standard-enp-geometric-coreset-projection-topk"
            : "standard-enp-uniform-wor-projection-topk"},
        {"enp_population_tokens", enp_sample.population},
        {"enp_sample_size", points_used},
        {"enp_coreset_max_points", enp_sample.geometric ? enp_sample.n : 0},
        {"enp_coreset_method", enp_sample.geometric ? enp_sample.method : ""},
        {"enp_adaptive_points", attempted_points},
        {"enp_adaptive_stability", adaptive_stability},
        {"enp_confidence_method", confidence_method},
        {"enp_confidence_delta_total", confidence_delta_total_value},
        {"enp_confidence_log", confidence_log_value},
        {"enp_cert_sample_evaluated", cert_sample_evaluated},
        {"enp_cert_sample_size", cert_sample_size},
        {"enp_cert_tail_size", cert_tail_size},
        {"enp_cert_remainder_population", cert_remainder_population},
        {"enp_cert_remainder_max_norm2", cert_remainder_max_norm2},
        {"enp_cert_projection_mass_evaluated", cert_projection_mass_evaluated},
        {"enp_cert_full_projection_mass", cert_full_projection_mass},
        {"enp_cert_selected_projection_mass", cert_selected_projection_mass},
        {"enp_cert_oracle_projection_mass", cert_oracle_projection_mass},
        {"enp_cert_selected_retained_fraction", cert_selected_retained_fraction},
        {"enp_cert_oracle_retained_fraction", cert_oracle_retained_fraction},
        {"enp_cert_excess_loss_fraction", cert_excess_loss_fraction},
        {"enp_cert_selected384_projection_mass", cert_selected384_projection_mass},
        {"enp_cert_oracle384_projection_mass", cert_oracle384_projection_mass},
        {"enp_cert_selected384_retained_fraction", cert_selected384_retained_fraction},
        {"enp_cert_oracle384_retained_fraction", cert_oracle384_retained_fraction},
        {"enp_cert_excess384_loss_fraction", cert_excess384_loss_fraction},
        {"enp_cert_sample_certified", cert_sample_certified},
        {"enp_cert_rank_disagreement", cert_rank_disagreement},
        {"enp_top256_certified", top256_certified},
        {"enp_top256_unique", top256_unique},
        {"enp_full_population_exact", full_population_exact},
        {"enp_exact_boundary_tie", full_population_exact && !top256_unique},
        {"enp_full_population_fallback_required", enp_sample.geometric && !top256_certified && !full_population_exact},
        {"enp_min_selected_lcb", min_selected_lcb},
        {"enp_max_pruned_ucb", max_pruned_ucb},
        {"enp_max_confidence_radius", max_interval},
        {"enp_transport_total", enp.transport_total},
        {"enp_transport_mean", enp_sample.geometric ? enp.transport_total / enp_sample.population : 0.0},
        {"enp_max_cluster_radius", enp.max_radius},
        {"enp_min_local_output_margin", enp.min_local_output_margin},
        {"enp_score_max", ranking.front().first},
        {"enp_score_min", score_min},
        {"enp_score_positive", score_positive},
        {"enp_score_zero", score_zero},
        {"enp_score_negative", score_negative},
        {"enp_score_cutoff", ranking[(size_t) EXPERT_WIDTH_DST - 1].first},
        {"enp_score_margin_256_257", score_margin},
        {"packing_swaps", 0},
    };
}

static json make_mtp_fallback_expert_record(int expert) {
    std::vector<int> selected(EXPERT_WIDTH_DST);
    std::iota(selected.begin(), selected.end(), 0);
    return {
        {"layer", N_LAYER_MAIN},
        {"expert", expert},
        {"output_to_input", selected},
        {"routing_count", 0},
        {"zero_routing_weight_only_fallback", true},
        {"selection_method", "mtp-nontarget-firstk-fallback"},
        {"enp_population_tokens", 0},
        {"enp_sample_size", 0},
        {"enp_top256_certified", false},
        {"packing_swaps", 0},
    };
}

static void validate_partial_expert_plan(const json & records, int n_layer_all = N_LAYER_ALL) {
    if (!records.is_array() || records.size() > (size_t) n_layer_all * N_EXPERT) {
        throw std::runtime_error("partial expert plan record count is invalid");
    }
    for (size_t i = 0; i < records.size(); ++i) {
        const int expected_layer = (int) (i / N_EXPERT);
        const int expected_expert = (int) (i % N_EXPERT);
        const auto & r = records[i];
        if (r.at("layer").get<int>() != expected_layer || r.at("expert").get<int>() != expected_expert) {
            throw std::runtime_error("partial expert plan is not in canonical layer/expert order");
        }
        const auto mapping = r.at("output_to_input").get<std::vector<int>>();
        if (mapping.size() != EXPERT_WIDTH_DST) throw std::runtime_error("partial expert mapping length mismatch");
        std::array<uint8_t, EXPERT_WIDTH_SRC> seen {};
        for (int x : mapping) {
            if (x < 0 || x >= EXPERT_WIDTH_SRC || seen[(size_t) x]) {
                throw std::runtime_error("partial expert mapping is not a unique 0..511 selection");
            }
            seen[(size_t) x] = 1;
        }
    }
}

static void extend_expert_plan(
        const source_gguf & src,
        const common_imatrix & imat,
        bool quick,
        bool allow_uncovered,
        int n_threads,
        json & records,
        const std::function<void(size_t)> & checkpoint) {
    const int n_layer_all = (int) src.get_u32("qwen35moe.block_count");
    validate_partial_expert_plan(records, n_layer_all);
    n_threads = std::max(1, std::min(n_threads, N_EXPERT));
    constexpr int CHECKPOINT_EXPERTS = 64;
    const size_t total = (size_t) n_layer_all * N_EXPERT;
    size_t completed = records.size();
    while (completed < total) {
        const int layer = (int) (completed / N_EXPERT);
        const int first_expert = (int) (completed % N_EXPERT);
        const int chunk_end = std::min(N_EXPERT, first_expert + CHECKPOINT_EXPERTS);

        if (layer == N_LAYER_MAIN) {
            for (int expert = first_expert; expert < chunk_end; ++expert) {
                records.push_back(make_mtp_fallback_expert_record(expert));
            }
            completed = records.size();
            checkpoint(completed);
            std::cerr << "plan experts: blk." << layer << " experts [" << first_expert << "," << chunk_end
                      << ") complete using non-target MTP fallback; total=" << completed << "/" << total << "\n";
            continue;
        }

        const standard_enp_sample enp_sample = load_standard_enp_sample(imat, layer);
        standard_enp_certification cert_sample;
        const standard_enp_certification * cert_sample_ptr = nullptr;
        if (imat.entries.find("prune.enp.cert_sample.blk." + std::to_string(layer)) != imat.entries.end()) {
            cert_sample = load_standard_enp_cert_sample(imat, layer);
            cert_sample_ptr = &cert_sample;
        }
        std::array<json, N_EXPERT> chunk_records;
        std::atomic<int> next_expert {first_expert};
        std::atomic<bool> failed {false};
        std::exception_ptr error;
        std::mutex error_mutex;

        auto worker = [&]() {
            try {
                while (!failed.load(std::memory_order_relaxed)) {
                    const int expert = next_expert.fetch_add(1);
                    if (expert >= chunk_end) break;
                    chunk_records[(size_t) expert] = make_expert_record(
                        src, imat, enp_sample, cert_sample_ptr, layer, expert, quick, allow_uncovered);
                }
            } catch (...) {
                failed.store(true, std::memory_order_relaxed);
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!error) error = std::current_exception();
            }
        };

        const int worker_count = std::min(n_threads, chunk_end - first_expert);
        std::vector<std::thread> workers;
        workers.reserve((size_t) worker_count);
        for (int i = 0; i < worker_count; ++i) workers.emplace_back(worker);
        for (auto & worker_thread : workers) worker_thread.join();
        if (error) std::rethrow_exception(error);

        for (int expert = first_expert; expert < chunk_end; ++expert) {
            records.push_back(std::move(chunk_records[(size_t) expert]));
        }
        completed = records.size();
        checkpoint(completed);
        std::cerr << "plan experts: blk." << layer << " experts [" << first_expert << "," << chunk_end
                  << ") complete; total=" << completed << "/" << total
                  << " (threads=" << worker_count << ")\n";
    }
}

static json make_gdn_plan(const source_gguf &, const common_imatrix &) {
    json out = json::array();
    std::vector<int> seed(GDN_DIM_DST);
    std::iota(seed.begin(), seed.end(), 0);
    for (int layer : RECURRENT_LAYERS) {
        json heads = json::array();
        for (int h = 0; h < GDN_QK_HEADS; ++h) {
            heads.push_back(seed);
        }
        out.push_back({
            {"layer", layer},
            {"v_indices", seed},
            {"qk_indices_by_head", heads},
            {"initialization", "exact-teacher-atom-gather-v1"},
            {"initialization_note", "No activation-space mixing or internal quadratic surrogate is used."},
        });
        std::cerr << "plan GDN exact atom seed: layer " << layer << " coordinates=0..63\n";
    }
    return out;
}

static json make_keep_source_gdn_plan() {
    json out = json::array();
    std::vector<int> seed(GDN_DIM_DST);
    std::iota(seed.begin(), seed.end(), 0);
    for (int layer : RECURRENT_LAYERS) {
        json heads = json::array();
        for (int h = 0; h < GDN_QK_HEADS; ++h) heads.push_back(seed);
        out.push_back({
            {"layer", layer},
            {"v_indices", seed},
            {"qk_indices_by_head", heads},
            {"initialization", "inactive-keep-source"},
            {"initialization_note", "Expert-only plan: apply must preserve all GDN tensors byte-for-byte."},
        });
    }
    return out;
}

static json make_keep_source_vocab_plan() {
    std::vector<int> placeholder(VOCAB_DST);
    std::iota(placeholder.begin(), placeholder.end(), 0);
    return {
        {"output_to_input", placeholder},
        {"byte_fallback_count", 256},
        {"protected_count", 0},
        {"mode", "inactive-keep-source"},
        {"note", "Expert-only plan: apply must preserve tokenizer and vocab tensors byte-for-byte."},
    };
}

static bool metadata_token_id_key(const std::string & key) {
    return starts_with(key, "tokenizer.ggml.") &&
           (ends_with(key, "_token_id") || ends_with(key, "_id"));
}

static bool decode_single_utf8_codepoint(const std::string & s, uint32_t & cp) {
    if (s.empty()) return false;
    const auto * p = (const uint8_t *) s.data();
    const size_t n = s.size();

    if (p[0] < 0x80) {
        if (n != 1) return false;
        cp = p[0];
        return true;
    }
    if ((p[0] & 0xE0) == 0xC0) {
        if (n != 2 || (p[1] & 0xC0) != 0x80) return false;
        cp = ((uint32_t) (p[0] & 0x1F) << 6) | (p[1] & 0x3F);
        return cp >= 0x80;
    }
    if ((p[0] & 0xF0) == 0xE0) {
        if (n != 3 || (p[1] & 0xC0) != 0x80 || (p[2] & 0xC0) != 0x80) return false;
        cp = ((uint32_t) (p[0] & 0x0F) << 12) |
             ((uint32_t) (p[1] & 0x3F) << 6) |
             (p[2] & 0x3F);
        return cp >= 0x800 && !(cp >= 0xD800 && cp <= 0xDFFF);
    }
    if ((p[0] & 0xF8) == 0xF0) {
        if (n != 4 || (p[1] & 0xC0) != 0x80 || (p[2] & 0xC0) != 0x80 || (p[3] & 0xC0) != 0x80) return false;
        cp = ((uint32_t) (p[0] & 0x07) << 18) |
             ((uint32_t) (p[1] & 0x3F) << 12) |
             ((uint32_t) (p[2] & 0x3F) << 6) |
             (p[3] & 0x3F);
        return cp >= 0x10000 && cp <= 0x10FFFF;
    }
    return false;
}

// GPT-2 byte-level BPE maps all 256 input bytes onto a fixed Unicode alphabet:
// printable Latin-1 codepoints stay as-is, and the remaining 68 bytes map to
// U+0100..U+0143. Keeping this complete alphabet guarantees that pruning merges
// cannot make arbitrary byte strings (and therefore arbitrary UTF-8) unencodable.
static bool is_gpt2_byte_fallback_token(const std::string & token) {
    uint32_t cp = 0;
    if (!decode_single_utf8_codepoint(token, cp)) return false;
    return (cp >= 33 && cp <= 126) ||
           (cp >= 161 && cp <= 172) ||
           (cp >= 174 && cp <= 255) ||
           (cp >= 256 && cp <= 323);
}

static bool gpt2_byte_from_token(const std::string & token, uint8_t & byte) {
    uint32_t cp = 0;
    if (!decode_single_utf8_codepoint(token, cp)) return false;
    if ((cp >= 33 && cp <= 126) || (cp >= 161 && cp <= 172) || (cp >= 174 && cp <= 255)) {
        byte = (uint8_t) cp;
        return true;
    }
    if (cp < 256 || cp > 323) return false;
    static const std::array<uint8_t, 68> escaped = [] {
        std::array<uint8_t, 68> out {};
        int k = 0;
        for (int b = 0; b < 256; ++b) {
            const bool direct = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
            if (!direct) out[(size_t) k++] = (uint8_t) b;
        }
        return out;
    }();
    byte = escaped[(size_t) (cp - 256)];
    return true;
}

static size_t utf8_codepoint_count(const std::string & s) {
    size_t n = 0;
    for (uint8_t c : s) {
        if ((c & 0xc0u) != 0x80u) ++n;
    }
    return n;
}

static bool gpt2_token_decodes_to_zh_cn(const std::string & token) {
    std::string raw;
    raw.reserve(token.size());
    for (size_t i = 0; i < token.size();) {
        const unsigned char c = (unsigned char) token[i];
        size_t n = c < 0x80 ? 1 : ((c & 0xe0) == 0xc0 ? 2 : ((c & 0xf0) == 0xe0 ? 3 : 4));
        if (i + n > token.size()) return false;
        uint8_t b = 0;
        if (!gpt2_byte_from_token(token.substr(i, n), b)) return false;
        raw.push_back((char) b);
        i += n;
    }

    for (size_t i = 0; i < raw.size();) {
        const unsigned char c = (unsigned char) raw[i];
        uint32_t cp = 0;
        size_t n = 0;
        if (c < 0x80) {
            cp = c; n = 1;
        } else if ((c & 0xe0) == 0xc0 && i + 2 <= raw.size()) {
            cp = ((uint32_t) (c & 0x1f) << 6) | ((unsigned char) raw[i + 1] & 0x3f); n = 2;
        } else if ((c & 0xf0) == 0xe0 && i + 3 <= raw.size()) {
            cp = ((uint32_t) (c & 0x0f) << 12) |
                 ((uint32_t) ((unsigned char) raw[i + 1] & 0x3f) << 6) |
                 ((unsigned char) raw[i + 2] & 0x3f); n = 3;
        } else if ((c & 0xf8) == 0xf0 && i + 4 <= raw.size()) {
            cp = ((uint32_t) (c & 0x07) << 18) |
                 ((uint32_t) ((unsigned char) raw[i + 1] & 0x3f) << 12) |
                 ((uint32_t) ((unsigned char) raw[i + 2] & 0x3f) << 6) |
                 ((unsigned char) raw[i + 3] & 0x3f); n = 4;
        } else {
            return false;
        }
        // Simplified-Chinese corpora are overwhelmingly covered by unified Han
        // ideographs plus the two common extension ranges. This is a ranking hint,
        // not a tokenizer-validity rule.
        if ((cp >= 0x3400 && cp <= 0x4dbf) ||
            (cp >= 0x4e00 && cp <= 0x9fff) ||
            (cp >= 0x20000 && cp <= 0x2ebef)) {
            return true;
        }
        i += n;
    }
    return false;
}

static double gpt2_ascii_english_priority(const std::string & token, size_t token_id) {
    std::string raw;
    raw.reserve(token.size());
    for (size_t i = 0; i < token.size();) {
        const unsigned char c = (unsigned char) token[i];
        const size_t n = c < 0x80 ? 1 : ((c & 0xe0) == 0xc0 ? 2 : ((c & 0xf0) == 0xe0 ? 3 : 4));
        if (i + n > token.size()) return -1.0;
        uint8_t b = 0;
        if (!gpt2_byte_from_token(token.substr(i, n), b) || b >= 0x80) return -1.0;
        raw.push_back((char) b);
        i += n;
    }
    size_t alpha = 0;
    for (unsigned char c : raw) {
        if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) ++alpha;
    }
    if (alpha == 0) return -1.0;
    // Source merge rank is a useful commonness prior, while alpha length favors
    // whole technical/common words over tiny fragments. This is used only for
    // zero-frequency BAOMU tokens, never ahead of an observed token.
    return std::pow((double) alpha, 1.5) / std::sqrt((double) token_id + 256.0);
}

static json make_vocab_plan(const source_gguf & src, const common_imatrix & imat) {
    const gguf_context * meta = src.meta();
    const int64_t token_key = gguf_find_key(meta, "tokenizer.ggml.tokens");
    const int64_t type_key  = gguf_find_key(meta, "tokenizer.ggml.token_type");
    if (token_key < 0 || type_key < 0) throw std::runtime_error("tokenizer arrays missing");
    const size_t n_vocab = gguf_get_arr_n(meta, token_key);
    if (n_vocab != 248320 || gguf_get_arr_n(meta, type_key) != n_vocab) throw std::runtime_error("unexpected source vocab size");
    const int32_t * types = (const int32_t *) gguf_get_arr_data(meta, type_key);
    const std::string freq_name = "prune.vocab";
    const auto & freq_e = require_entry(imat, freq_name);
    if (freq_e.sums.size() != n_vocab) throw std::runtime_error("vocab histogram size mismatch");

    std::vector<uint8_t> protected_id(n_vocab, 0);
    int byte_fallback_count = 0;
    for (size_t i = 0; i < n_vocab; ++i) {
        const std::string token = gguf_get_arr_str(meta, token_key, i);
        const int32_t ty = types[i];
        if (ty == 3 || ty == 4 || ty == 6) protected_id[i] = 1; // CONTROL, USER_DEFINED, BYTE
        if (is_gpt2_byte_fallback_token(token)) {
            protected_id[i] = 1;
            ++byte_fallback_count;
        }
    }
    if (byte_fallback_count != 256) {
        throw std::runtime_error("expected exactly 256 GPT-2 byte fallback tokens, found " + std::to_string(byte_fallback_count));
    }
    for (int64_t k = 0; k < gguf_get_n_kv(meta); ++k) {
        const std::string key = gguf_get_key(meta, k);
        if (!metadata_token_id_key(key)) continue;
        uint64_t id = std::numeric_limits<uint64_t>::max();
        switch (gguf_get_kv_type(meta, k)) {
            case GGUF_TYPE_UINT32: id = gguf_get_val_u32(meta, k); break;
            case GGUF_TYPE_INT32:  id = (uint64_t) std::max(0, gguf_get_val_i32(meta, k)); break;
            case GGUF_TYPE_UINT64: id = gguf_get_val_u64(meta, k); break;
            case GGUF_TYPE_INT64:  id = (uint64_t) std::max<int64_t>(0, gguf_get_val_i64(meta, k)); break;
            default: break;
        }
        if (id < n_vocab) protected_id[(size_t) id] = 1;
    }

    // Build one deterministic BPE ancestry edge per result token. Keeping a
    // high-value result token without the operands needed to construct it makes
    // that token unreachable after merge filtering and wastes vocabulary budget.
    std::unordered_map<std::string, int> token_id;
    token_id.reserve(n_vocab * 2);
    for (size_t i = 0; i < n_vocab; ++i) token_id.emplace(gguf_get_arr_str(meta, token_key, i), (int) i);
    std::vector<std::array<int, 2>> parents(n_vocab, std::array<int,2>{-1, -1});
    const int64_t merge_key = gguf_find_key(meta, "tokenizer.ggml.merges");
    if (merge_key < 0) throw std::runtime_error("tokenizer merges missing");
    const size_t n_merges = gguf_get_arr_n(meta, merge_key);
    for (size_t i = 0; i < n_merges; ++i) {
        const std::string merge = gguf_get_arr_str(meta, merge_key, i);
        const size_t sp = merge.find(' ');
        if (sp == std::string::npos) continue;
        const std::string a = merge.substr(0, sp);
        const std::string b = merge.substr(sp + 1);
        auto ia = token_id.find(a), ib = token_id.find(b), ir = token_id.find(a + b);
        if (ia == token_id.end() || ib == token_id.end() || ir == token_id.end()) continue;
        auto & p = parents[(size_t) ir->second];
        if (p[0] < 0) p = {ia->second, ib->second}; // earliest merge rank wins deterministically
    }

    std::vector<int> keep;
    std::vector<uint8_t> kept(n_vocab, 0);
    for (size_t i = 0; i < n_vocab; ++i) {
        if (protected_id[i]) {
            kept[i] = 1;
            keep.push_back((int) i);
        }
    }
    const int protected_count = (int) keep.size();

    // BAOMU is intentionally a narrow calibration corpus. Reserve part of the
    // zero-frequency remainder for likely zh-CN and English merges so that the
    // 36k vocabulary does not become a pure "whatever happened to occur in the
    // document" vocabulary. Observed BAOMU tokens always outrank these priors.
    std::vector<uint8_t> zh_zero_priority(n_vocab, 0);
    std::vector<double> en_zero_priority(n_vocab, 0.0);
    size_t zh_zero_kept = 0;
    constexpr size_t ZH_ZERO_BUDGET = 16500;
    constexpr size_t EN_ZERO_BUDGET = 12000;
    for (size_t i = 0; i < n_vocab && zh_zero_kept < ZH_ZERO_BUDGET; ++i) {
        if (freq_e.sums[i] == 0.0f && !protected_id[i]) {
            const std::string token = gguf_get_arr_str(meta, token_key, i);
            if (gpt2_token_decodes_to_zh_cn(token)) {
                zh_zero_priority[i] = 1;
                ++zh_zero_kept;
            }
        }
    }
    std::vector<std::pair<double, int>> en_zero_ranked;
    en_zero_ranked.reserve(n_vocab / 2);
    for (size_t i = 0; i < n_vocab; ++i) {
        if (freq_e.sums[i] != 0.0f || protected_id[i]) continue;
        const std::string token = gguf_get_arr_str(meta, token_key, i);
        const double p = gpt2_ascii_english_priority(token, i);
        if (p >= 0.0) en_zero_ranked.emplace_back(p, (int) i);
    }
    std::stable_sort(en_zero_ranked.begin(), en_zero_ranked.end(), [](const auto & a, const auto & b) {
        if (a.first != b.first) return a.first > b.first;
        return a.second < b.second;
    });
    const size_t en_budget = std::min(EN_ZERO_BUDGET, en_zero_ranked.size());
    for (size_t i = 0; i < en_budget; ++i) {
        // Preserve the heuristic order when these zero-frequency candidates are
        // mixed with closure dependencies. A flat score would fall back to token
        // id and discard useful whole words near the tail of the reserved set.
        en_zero_priority[(size_t) en_zero_ranked[i].second] =
            0.50 + 0.20 * (double) (en_budget - i) / (double) en_budget;
    }

    std::vector<std::pair<double,int>> scored;
    for (size_t i = 0; i < n_vocab; ++i) {
        if (!protected_id[i]) {
            const std::string token = gguf_get_arr_str(meta, token_key, i);
            const size_t byte_len = utf8_codepoint_count(token); // one GPT-2 alphabet codepoint == one raw byte
            const double gain = std::max<size_t>(1, byte_len > 0 ? byte_len - 1 : 1);
            // BAOMU's deployment target is intentionally English + zh-CN. Bias
            // scarce 36k vocabulary capacity toward Han-token merges; other
            // languages are allowed to degrade rather than stealing this budget.
            const double zh_boost = gpt2_token_decodes_to_zh_cn(token) ? 24.0 : 1.0;
            double score = (double) freq_e.sums[i] * gain * zh_boost;
            if (freq_e.sums[i] == 0.0f) {
                if (zh_zero_priority[i]) score = 0.75;
                else if (en_zero_priority[i] > 0.0) score = en_zero_priority[i];
            }
            scored.emplace_back(score, (int) i);
        }
    }
    if ((int) keep.size() > VOCAB_DST) throw std::runtime_error("protected vocabulary exceeds target size");
    std::stable_sort(scored.begin(), scored.end(), [](const auto & a, const auto & b) {
        if (a.first != b.first) return a.first > b.first;
        return a.second < b.second;
    });

    size_t dependency_added = 0;
    auto missing_closure = [&](int root) {
        std::vector<int> missing;
        std::vector<uint8_t> staged(n_vocab, 0);
        std::vector<uint8_t> visiting(n_vocab, 0);
        std::function<void(int)> visit = [&](int id) {
            if (kept[(size_t) id] || staged[(size_t) id]) return;
            if (visiting[(size_t) id]) throw std::runtime_error("cycle in tokenizer merge ancestry");
            visiting[(size_t) id] = 1;
            const auto p = parents[(size_t) id];
            if (p[0] >= 0) {
                visit(p[0]);
                visit(p[1]);
            }
            visiting[(size_t) id] = 0;
            staged[(size_t) id] = 1;
            missing.push_back(id);
        };
        visit(root);
        return missing;
    };

    for (const auto & [score, id] : scored) {
        GGML_UNUSED(score);
        if ((int) keep.size() >= VOCAB_DST) break;
        if (kept[(size_t) id]) continue;
        auto missing = missing_closure(id);
        if (keep.size() + missing.size() > VOCAB_DST) continue;
        for (int x : missing) {
            if (!kept[(size_t) x]) {
                kept[(size_t) x] = 1;
                keep.push_back(x);
                if (x != id) ++dependency_added;
            }
        }
    }
    if ((int) keep.size() != VOCAB_DST) {
        throw std::runtime_error("failed to fill target vocabulary size without violating BPE ancestry closure");
    }
    std::sort(keep.begin(), keep.end());

    json specials = json::object();
    std::vector<int> inverse(n_vocab, -1);
    for (int i = 0; i < (int) keep.size(); ++i) inverse[(size_t) keep[i]] = i;
    for (int64_t k = 0; k < gguf_get_n_kv(meta); ++k) {
        const std::string key = gguf_get_key(meta, k);
        if (!metadata_token_id_key(key)) continue;
        int64_t old = -1;
        switch (gguf_get_kv_type(meta, k)) {
            case GGUF_TYPE_UINT32: old = gguf_get_val_u32(meta, k); break;
            case GGUF_TYPE_INT32: old = gguf_get_val_i32(meta, k); break;
            case GGUF_TYPE_UINT64: old = (int64_t) gguf_get_val_u64(meta, k); break;
            case GGUF_TYPE_INT64: old = gguf_get_val_i64(meta, k); break;
            default: break;
        }
        if (old >= 0 && old < (int64_t) n_vocab && inverse[(size_t) old] >= 0) specials[key] = inverse[(size_t) old];
    }

    return {
        {"output_to_input", keep},
        {"protected_count", protected_count},
        {"byte_fallback_count", byte_fallback_count},
        {"merge_dependency_tokens_added", dependency_added},
        {"zh_zero_priority_budget", ZH_ZERO_BUDGET},
        {"en_zero_priority_budget", EN_ZERO_BUDGET},
        {"special_id_remap", specials},
    };
}

static void verify_plan_json(const json & plan) {
    if (plan.value("format", "") != "qwen35-combined-prune-v2") throw std::runtime_error("wrong plan format");
    const std::string expert_selection = plan.value("expert_selection", "");
    const bool geometric_enp = expert_selection == "standard-enp-geometric-coreset-projection-topk-v1";
    const bool uniform_enp = expert_selection == "standard-enp-uniform-wor-projection-topk-v1";
    const bool downstream_lens = expert_selection == "standard-enp-times-post-moe-to-final-256d-ridge-survival-v1";
    if (!geometric_enp && !uniform_enp && !downstream_lens) {
        throw std::runtime_error("plan does not use a supported standard ENP projection Top-K calibration");
    }
    if (!plan.contains("source_sha256") || !plan.contains("imatrix_sha256")) throw std::runtime_error("plan hashes missing");
    const auto & experts = plan.at("experts");
    const int n_layer_all = plan.value("source_block_count", (int)(experts.size() / N_EXPERT));
    if (!experts.is_array() || experts.size() != (size_t) n_layer_all * N_EXPERT) throw std::runtime_error("expert plan record count mismatch");
    std::set<std::pair<int,int>> expert_keys;
    for (const auto & r : experts) {
        const int l = r.at("layer"), e = r.at("expert");
        if (l < 0 || l >= n_layer_all || e < 0 || e >= N_EXPERT || !expert_keys.emplace(l,e).second) throw std::runtime_error("duplicate/bad expert plan key");
        const auto m = r.at("output_to_input").get<std::vector<int>>();
        if (m.size() != EXPERT_WIDTH_DST) throw std::runtime_error("expert mapping length mismatch");
        std::set<int> s(m.begin(), m.end());
        if (s.size() != m.size() || *s.begin() < 0 || *s.rbegin() >= EXPERT_WIDTH_SRC) throw std::runtime_error("expert mapping invalid");
        const std::string method = r.value("selection_method", std::string());
        if (l < N_LAYER_MAIN) {
            const std::string expected_method = downstream_lens
                ? "standard-enp-times-post-moe-to-final-256d-ridge-survival-v1"
                : (geometric_enp
                    ? "standard-enp-geometric-coreset-projection-topk"
                    : "standard-enp-uniform-wor-projection-topk");
            if (method != expected_method ||
                r.value("enp_population_tokens", int64_t(0)) <= 0 ||
                r.value("enp_sample_size", 0) <= 0 ||
                r.value("zero_routing_weight_only_fallback", false)) {
                throw std::runtime_error("main-model expert is not selected by the declared standard ENP projection Top-K method");
            }
            if (geometric_enp) {
                const bool certified = r.value("enp_top256_certified", false);
                const bool exact = r.value("enp_full_population_exact", false);
                const bool unresolved = r.value("enp_full_population_fallback_required", false);
                if (unresolved == (certified || exact)) {
                    throw std::runtime_error("geometric ENP resolution metadata is inconsistent");
                }
                if (r.value("enp_top256_unique", false) && !certified && !exact) {
                    throw std::runtime_error("geometric ENP claims a unique Top-256 without certification or exact evaluation");
                }
                if (exact && (r.value("enp_transport_total", -1.0) != 0.0 ||
                              r.value("enp_max_cluster_radius", -1.0) != 0.0)) {
                    throw std::runtime_error("full-population ENP record has nonzero geometric approximation error");
                }
            }
        } else {
            if ((method != "mtp-nontarget-first256-fallback" &&
                 method != "mtp-nontarget-firstk-fallback") ||
                r.value("enp_population_tokens", int64_t(-1)) != 0 ||
                r.value("enp_sample_size", -1) != 0 ||
                !r.value("zero_routing_weight_only_fallback", false)) {
                throw std::runtime_error("MTP non-target expert fallback metadata is invalid");
            }
            for (int i = 0; i < EXPERT_WIDTH_DST; ++i) {
                if (m[(size_t) i] != i) throw std::runtime_error("MTP non-target fallback is not canonical first-K");
            }
        }
    }
    const auto & gdn = plan.at("gdn");
    if (!gdn.is_array() || gdn.size() != RECURRENT_LAYERS.size()) throw std::runtime_error("GDN plan record count mismatch");
    for (const auto & r : gdn) {
        const auto vg = r.at("v_indices").get<std::vector<int>>();
        if (vg.size() != GDN_DIM_DST || std::set<int>(vg.begin(), vg.end()).size() != GDN_DIM_DST ||
            *std::min_element(vg.begin(), vg.end()) < 0 || *std::max_element(vg.begin(), vg.end()) >= GDN_DIM_SRC) {
            throw std::runtime_error("V coordinate selection invalid");
        }
        const auto & qk = r.at("qk_indices_by_head");
        if (qk.size() != GDN_QK_HEADS) throw std::runtime_error("QK head mask count invalid");
        for (const auto & coords : qk) {
            auto p = coords.get<std::vector<int>>();
            if (p.size() != GDN_DIM_DST || std::set<int>(p.begin(), p.end()).size() != GDN_DIM_DST ||
                *std::min_element(p.begin(), p.end()) < 0 || *std::max_element(p.begin(), p.end()) >= GDN_DIM_SRC) {
                throw std::runtime_error("QK coordinate selection invalid");
            }
        }
    }
    const auto vocab = plan.at("vocab").at("output_to_input").get<std::vector<int>>();
    if (vocab.size() != VOCAB_DST || std::set<int>(vocab.begin(), vocab.end()).size() != VOCAB_DST) throw std::runtime_error("vocab mapping invalid");
    if (plan.at("vocab").value("byte_fallback_count", 0) != 256) throw std::runtime_error("vocab plan does not preserve the complete 256-byte fallback alphabet");
}

static size_t unresolved_geometric_enp_experts(const json & plan) {
    if (plan.value("expert_selection", std::string()) != "standard-enp-geometric-coreset-projection-topk-v1") {
        return 0;
    }
    size_t unresolved = 0;
    for (const auto & r : plan.at("experts")) {
        if (r.at("layer").get<int>() >= N_LAYER_MAIN) continue;
        unresolved += r.value("enp_full_population_fallback_required", false) ? 1u : 0u;
    }
    return unresolved;
}

static void require_materializable_enp_plan(const json & plan) {
    const size_t unresolved = unresolved_geometric_enp_experts(plan);
    if (unresolved == 0) return;
    throw std::runtime_error(
        "refusing to materialize geometric ENP plan with " + std::to_string(unresolved) +
        " unresolved experts; rerun calibration with a larger --prune-enp-coreset-points "
        "budget (up to the full calibration population) until every expert is certified or exact");
}

static json expert_plan_diagnostics(const json & plan) {
    const bool routing_stats_available = !plan.value("expert_only", false);
    const int n_layer_all = plan.value("source_block_count", (int)(plan.at("experts").size() / N_EXPERT));
    std::vector<std::vector<int64_t>> routing((size_t) n_layer_all);
    uint64_t packing_swaps = 0;
    uint64_t zero_routing = 0;
    uint64_t standard_enp = 0;
    uint64_t certified_enp = 0;
    uint64_t unresolved_enp = 0;
    uint64_t weight_only_fallback = 0;
    for (const auto & r : plan.at("experts")) {
        const int layer = r.at("layer").get<int>();
        const int64_t count = r.value("routing_count", int64_t(0));
        if (routing_stats_available && layer >= 0 && layer < n_layer_all) routing[(size_t) layer].push_back(count);
        packing_swaps += (uint64_t) r.value("packing_swaps", 0);
        if (routing_stats_available && count <= 0) ++zero_routing;
        const std::string method = r.value("selection_method", std::string());
        const bool is_standard_enp = method == "standard-enp-uniform-wor-projection-topk" ||
            method == "standard-enp-geometric-coreset-projection-topk";
        standard_enp += is_standard_enp;
        certified_enp += is_standard_enp && r.value("enp_top256_certified", false);
        unresolved_enp += is_standard_enp && r.value("enp_full_population_fallback_required", false);
        weight_only_fallback += method == "mtp-nontarget-first256-fallback";
    }

    json per_layer = json::array();
    for (int layer = 0; routing_stats_available && layer < n_layer_all; ++layer) {
        auto v = routing[(size_t) layer];
        if (v.size() != N_EXPERT) throw std::runtime_error("routing diagnostics expert count mismatch");
        std::sort(v.begin(), v.end());
        auto pct = [&](int num, int den) -> int64_t {
            const size_t idx = (size_t) ((N_EXPERT - 1) * num / den);
            return v[idx];
        };
        per_layer.push_back({
            {"layer", layer},
            {"min", v.front()},
            {"p05", pct(5, 100)},
            {"p25", pct(1, 4)},
            {"p50", pct(1, 2)},
            {"p75", pct(3, 4)},
            {"max", v.back()},
            {"zero", std::count_if(v.begin(), v.end(), [](int64_t x) { return x <= 0; })},
            {"below_32", std::count_if(v.begin(), v.end(), [](int64_t x) { return x < 32; })},
        });
    }
    return {
        {"routing_stats_available", routing_stats_available},
        {"routing_count_percentiles", per_layer},
        {"zero_routing_experts", zero_routing},
        {"standard_enp_experts", standard_enp},
        {"certified_enp_experts", certified_enp},
        {"unresolved_enp_experts", unresolved_enp},
        {"zero_routing_weight_only_fallback_experts", weight_only_fallback},
        {"packing_swaps", packing_swaps},
    };
}

static void require_full_expert_coverage(const common_imatrix & imat) {
    std::vector<std::string> missing;
    for (int layer = 0; layer < N_LAYER_MAIN; ++layer) {
        const std::string name = "blk." + std::to_string(layer) + ".ffn_down_exps.weight";
        const auto & e = require_entry(imat, name);
        if ((int) e.counts.size() != N_EXPERT) throw std::runtime_error("expert routing count dimension mismatch: " + name);
        for (int expert = 0; expert < N_EXPERT; ++expert) {
            if (e.counts[(size_t) expert] <= 0) {
                if (missing.size() < 16) missing.push_back("blk." + std::to_string(layer) + "/e" + std::to_string(expert));
            }
        }
    }
    if (!missing.empty()) {
        std::string msg = "production plan requires nonzero routing coverage for every expert; first missing:";
        for (const auto & x : missing) msg += " " + x;
        msg += ". Provide a broader calibration corpus, or explicitly use --allow-uncovered when the corpus is intentionally fixed.";
        throw std::runtime_error(msg);
    }
}

static json validate_calibration_contract(const common_imatrix & imat, bool allow_uncovered) {
    (void) allow_uncovered;
    if (imat.is_legacy || !imat.has_metadata || imat.chunk_count <= 0 || imat.chunk_size <= 0) {
        throw std::runtime_error("production planning requires GGUF imatrix metadata with positive chunk_count/chunk_size");
    }
    uint64_t expected_tokens = (uint64_t) imat.chunk_count * (uint64_t) imat.chunk_size;
    bool exact_token_count_entry = false;
    if (auto it = imat.entries.find("prune.token_count"); it != imat.entries.end()) {
        const auto & token_count = it->second;
        if (token_count.sums.size() != 1 || token_count.counts.size() != 1 ||
            !std::isfinite(token_count.sums[0]) || token_count.sums[0] <= 0.0f) {
            throw std::runtime_error("bad prune.token_count entry");
        }
        const double rounded = std::round((double) token_count.sums[0]);
        if (std::fabs((double) token_count.sums[0] - rounded) > 1e-3) {
            throw std::runtime_error("prune.token_count is not integral");
        }
        expected_tokens = (uint64_t) rounded;
        exact_token_count_entry = true;
    }
    json routing = json::array();
    uint64_t zero_total = 0;
    uint64_t below32_total = 0;

    const bool geometric_enp = imat.entries.find("prune.enp.coreset.blk.0") != imat.entries.end();
    if (!geometric_enp) {
        const std::string enp_seed_name = "prune.enp.sample_seed";
        const auto & enp_seed = require_entry(imat, enp_seed_name);
        if (enp_seed.counts.size() != 1 || enp_seed.counts[0] != 1 || enp_seed.sums.size() != 4) {
            throw std::runtime_error("bad standard ENP random sample seed metadata");
        }
        for (float x : enp_seed.sums) {
            if (!std::isfinite(x) || x < 0.0f || x > 65535.0f || x != std::floor(x)) {
                throw std::runtime_error("bad standard ENP random sample seed component");
            }
        }
    }

    for (int layer = 0; layer < N_LAYER_MAIN; ++layer) {
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        const std::string gate_name = prefix + "ffn_gate_exps.weight";
        const std::string up_name   = prefix + "ffn_up_exps.weight";
        const std::string down_name = prefix + "ffn_down_exps.weight";
        const std::string enp_name  = (geometric_enp ? "prune.enp.coreset.blk." : "prune.enp.sample.blk.") + std::to_string(layer);
        const std::string enp_population_name = "prune.enp.population.blk." + std::to_string(layer);
        const std::string enp_max_norm_name = "prune.enp.max_norm2.blk." + std::to_string(layer);
        const auto & gate = require_entry(imat, gate_name);
        const auto & up   = require_entry(imat, up_name);
        const auto & down = require_entry(imat, down_name);
        const auto & enp  = require_entry(imat, enp_name);
        const auto & enp_population = require_entry(imat, enp_population_name);
        const auto & enp_max_norm = require_entry(imat, enp_max_norm_name);
        if (gate.counts.size() != N_EXPERT || up.counts.size() != N_EXPERT || down.counts.size() != N_EXPERT ||
            gate.sums.size() != (size_t) N_EXPERT * HIDDEN || up.sums.size() != (size_t) N_EXPERT * HIDDEN ||
            down.sums.size() != (size_t) N_EXPERT * EXPERT_WIDTH_SRC ||
            enp.counts.empty() || enp.sums.size() != enp.counts.size() * (size_t) HIDDEN) {
            throw std::runtime_error("bad expert calibration dimensions at layer " + std::to_string(layer));
        }

        if (geometric_enp) {
            const auto & weights = require_entry(imat, "prune.enp.coreset_weight.blk." + std::to_string(layer));
            const auto & transport = require_entry(imat, "prune.enp.coreset_transport.blk." + std::to_string(layer));
            const auto & radius = require_entry(imat, "prune.enp.coreset_radius.blk." + std::to_string(layer));
            if (weights.sums.size() != enp.counts.size() || transport.sums.size() != enp.counts.size() ||
                radius.sums.size() != enp.counts.size()) {
                throw std::runtime_error("bad standard ENP geometric coreset auxiliary dimensions at layer " + std::to_string(layer));
            }
            double weight_sum = 0.0;
            for (size_t r = 0; r < enp.counts.size(); ++r) {
                if (!std::isfinite(weights.sums[r]) || weights.sums[r] <= 0.0f ||
                    !std::isfinite(transport.sums[r]) || transport.sums[r] < 0.0f ||
                    !std::isfinite(radius.sums[r]) || radius.sums[r] < 0.0f) {
                    throw std::runtime_error("bad standard ENP geometric coreset metadata at layer " + std::to_string(layer));
                }
                weight_sum += weights.sums[r];
            }
            if (std::fabs(weight_sum - (double) expected_tokens) > 0.5) {
                throw std::runtime_error("standard ENP geometric coreset weight sum mismatch at layer " + std::to_string(layer));
            }
        }

        if (enp_population.sums.size() != 1 || enp_population.counts.size() != 1 || enp_population.counts[0] != 1 ||
            !std::isfinite(enp_population.sums[0])) {
            throw std::runtime_error("bad standard ENP population entry at layer " + std::to_string(layer));
        }
        const uint64_t enp_population_tokens = (uint64_t) std::llround(enp_population.sums[0]);
        if (enp_population_tokens != expected_tokens) {
            throw std::runtime_error("standard ENP sample population mismatch at layer " + std::to_string(layer) +
                                     ": got " + std::to_string(enp_population_tokens) +
                                     ", expected " + std::to_string(expected_tokens));
        }
        for (size_t r = 0; r < enp.counts.size(); ++r) {
            if (enp.counts[r] != 1) throw std::runtime_error("non-unit standard ENP sample count at layer " + std::to_string(layer));
            for (int i = 0; i < HIDDEN; ++i) {
                if (!std::isfinite(enp.sums[r * HIDDEN + (size_t) i])) {
                    throw std::runtime_error("non-finite standard ENP sample value at layer " + std::to_string(layer));
                }
            }
        }
        if (enp_max_norm.counts.size() != 1 || enp_max_norm.sums.size() != 1 || enp_max_norm.counts[0] != 1 ||
            !std::isfinite(enp_max_norm.sums[0]) || enp_max_norm.sums[0] <= 0.0f) {
            throw std::runtime_error("bad standard ENP max-norm entry at layer " + std::to_string(layer));
        }

        std::vector<int64_t> counts(N_EXPERT);
        int zero = 0;
        int below32 = 0;
        int64_t routing_sum = 0;
        for (int expert = 0; expert < N_EXPERT; ++expert) {
            const int64_t cg = gate.counts[(size_t) expert];
            const int64_t cu = up.counts[(size_t) expert];
            const int64_t cd = down.counts[(size_t) expert];
            if (cg != cu || cg != cd) {
                throw std::runtime_error("gate/up/down routing count mismatch at blk." +
                                         std::to_string(layer) + "/e" + std::to_string(expert));
            }
            counts[(size_t) expert] = cd;
            routing_sum += cd;
            zero += cd <= 0;
            below32 += cd < 32;
            for (int i = 0; i < HIDDEN; ++i) {
                const double sg = gate.sums[(size_t) expert * HIDDEN + i];
                const double su = up.sums[(size_t) expert * HIDDEN + i];
                if (!std::isfinite(sg) || !std::isfinite(su) || sg < 0.0 || su < 0.0) {
                    throw std::runtime_error("invalid gate/up sum2 at blk." + std::to_string(layer) +
                                             "/e" + std::to_string(expert));
                }
                if (cd > 0) {
                    const double mg = sg / (double) cg;
                    const double mu = su / (double) cu;
                    const double scale = std::max({1.0, std::fabs(mg), std::fabs(mu)});
                    if (std::fabs(mg - mu) > 1e-6 * scale) {
                        throw std::runtime_error("gate/up hidden-input mean mismatch at blk." +
                                                 std::to_string(layer) + "/e" + std::to_string(expert));
                    }
                }
            }
            for (int j = 0; j < EXPERT_WIDTH_SRC; ++j) {
                const double sd = down.sums[(size_t) expert * EXPERT_WIDTH_SRC + j];
                if (!std::isfinite(sd) || sd < 0.0) {
                    throw std::runtime_error("invalid down sum2 at blk." + std::to_string(layer) +
                                             "/e" + std::to_string(expert));
                }
            }
        }
        if ((uint64_t) routing_sum != expected_tokens * 8u) {
            throw std::runtime_error("expert routing total mismatch at layer " + std::to_string(layer) +
                                     ": got " + std::to_string(routing_sum) +
                                     ", expected " + std::to_string(expected_tokens * 8u));
        }
        std::sort(counts.begin(), counts.end());
        const auto pct = [&](int num, int den) -> int64_t {
            return counts[(size_t) ((N_EXPERT - 1) * num / den)];
        };
        std::cerr << "calibration routing: blk." << layer
                  << " min=" << counts.front()
                  << " p05=" << pct(5, 100)
                  << " median=" << pct(1, 2)
                  << " zero=" << zero
                  << " <32=" << below32 << "\n";
        routing.push_back({
            {"layer", layer},
            {"min", counts.front()},
            {"p05", pct(5, 100)},
            {"p50", pct(1, 2)},
            {"max", counts.back()},
            {"zero", zero},
            {"below_32", below32},
            {"enp_sample_size", enp.counts.size()},
            {"enp_sampling_method", geometric_enp
                ? "task-blind-streaming-medoid-original-l2-v1"
                : "bottom-k-uniform-without-replacement"},
            {"enp_max_norm2", enp_max_norm.sums[0]},
            {"enp_population_tokens", enp_population_tokens},
        });
        zero_total += (uint64_t) zero;
        below32_total += (uint64_t) below32;
    }

    const std::string vocab_name = "prune.vocab";
    const auto & vocab = require_entry(imat, vocab_name);
    if (vocab.sums.size() != 248320) throw std::runtime_error("bad prune.vocab histogram size");
    double vocab_total = 0.0;
    for (float x : vocab.sums) {
        if (!std::isfinite(x) || x < 0.0f) throw std::runtime_error("invalid prune.vocab count");
        vocab_total += x;
    }
    if (std::fabs(vocab_total - (double) expected_tokens) > 0.5) {
        throw std::runtime_error("prune.vocab count sum does not match imatrix chunk_count*chunk_size");
    }

    return {
        {"routing", routing},
        {"zero_routing_experts", zero_total},
        {"experts_below_32", below32_total},
        {"vocab_count_sum", vocab_total},
        {"expected_calibration_tokens", expected_tokens},
        {"exact_token_count_entry", exact_token_count_entry},
        {"expert_neuron_importance", geometric_enp
            ? "standard-enp-projection-eq9-eq10-on-task-blind-geometric-hidden-coreset"
            : "standard-enp-projection-eq9-eq10-on-task-agnostic-uniform-hidden-sample-without-replacement"},
        {"imatrix_chunk_count", imat.chunk_count},
        {"imatrix_chunk_size", imat.chunk_size},
    };
}

static json validate_expert_only_calibration_contract(const common_imatrix & imat) {
    if (imat.is_legacy || !imat.has_metadata || imat.chunk_count <= 0 || imat.chunk_size <= 0) {
        throw std::runtime_error("expert-only planning requires GGUF imatrix metadata with positive chunk_count/chunk_size");
    }
    const auto & token_count = require_entry(imat, "prune.token_count");
    if (token_count.sums.size() != 1 || token_count.counts.size() != 1 ||
        !std::isfinite(token_count.sums[0]) || token_count.sums[0] <= 0.0f) {
        throw std::runtime_error("expert-only calibration requires an exact prune.token_count entry");
    }
    const double rounded = std::round((double) token_count.sums[0]);
    if (std::fabs((double) token_count.sums[0] - rounded) > 1e-3) {
        throw std::runtime_error("expert-only prune.token_count is not integral");
    }
    const uint64_t expected_tokens = (uint64_t) rounded;
    if (imat.entries.find("prune.enp.coreset.blk.0") == imat.entries.end()) {
        throw std::runtime_error("expert-only planning requires the geometric ENP coreset, not the legacy uniform sample");
    }
    const bool has_cert_sample = imat.entries.find("prune.enp.cert_sample.blk.0") != imat.entries.end();
    const bool has_cert_tail = imat.entries.find("prune.enp.cert_tail.blk.0") != imat.entries.end();
    int cert_requested = 0;
    int cert_tail_requested = 0;
    if (has_cert_sample) {
        const auto & seed = require_entry(imat, "prune.enp.cert_sample_seed");
        if (seed.sums.size() != 4 || seed.counts.size() != 1 || seed.counts[0] != 1) {
            throw std::runtime_error("bad ENP certification sample seed metadata");
        }
        for (float x : seed.sums) {
            if (!std::isfinite(x) || x < 0.0f || x > 65535.0f || x != std::floor(x)) {
                throw std::runtime_error("bad ENP certification sample seed component");
            }
        }
        const auto & requested = require_entry(imat, "prune.enp.cert_sample_requested");
        if (requested.sums.size() != 1 || requested.counts.size() != 1 || requested.counts[0] != 1 ||
            !std::isfinite(requested.sums[0]) || requested.sums[0] <= 0.0f ||
            requested.sums[0] != std::floor(requested.sums[0])) {
            throw std::runtime_error("bad ENP certification requested-size metadata");
        }
        cert_requested = (int) requested.sums[0];
        if (has_cert_tail) {
            const auto & tail_requested = require_entry(imat, "prune.enp.cert_tail_requested");
            if (tail_requested.sums.size() != 1 || tail_requested.counts.size() != 1 || tail_requested.counts[0] != 1 ||
                !std::isfinite(tail_requested.sums[0]) || tail_requested.sums[0] < 0.0f ||
                tail_requested.sums[0] != std::floor(tail_requested.sums[0])) {
                throw std::runtime_error("bad ENP certification tail requested-size metadata");
            }
            cert_tail_requested = (int) tail_requested.sums[0];
        }
    }

    json layers = json::array();
    for (int layer = 0; layer < N_LAYER_MAIN; ++layer) {
        const std::string suffix = std::to_string(layer);
        const auto & enp = require_entry(imat, "prune.enp.coreset.blk." + suffix);
        const auto & weights = require_entry(imat, "prune.enp.coreset_weight.blk." + suffix);
        const auto & transport = require_entry(imat, "prune.enp.coreset_transport.blk." + suffix);
        const auto & radius = require_entry(imat, "prune.enp.coreset_radius.blk." + suffix);
        const auto & population = require_entry(imat, "prune.enp.population.blk." + suffix);
        const auto & max_norm = require_entry(imat, "prune.enp.max_norm2.blk." + suffix);
        if (enp.counts.empty() || enp.sums.size() != enp.counts.size() * (size_t) HIDDEN ||
            weights.sums.size() != enp.counts.size() || transport.sums.size() != enp.counts.size() ||
            radius.sums.size() != enp.counts.size()) {
            throw std::runtime_error("bad expert-only ENP coreset dimensions at layer " + suffix);
        }
        double weight_sum = 0.0;
        double transport_sum = 0.0;
        double max_radius = 0.0;
        for (size_t r = 0; r < enp.counts.size(); ++r) {
            if (enp.counts[r] != 1 || !std::isfinite(weights.sums[r]) || weights.sums[r] <= 0.0f ||
                !std::isfinite(transport.sums[r]) || transport.sums[r] < 0.0f ||
                !std::isfinite(radius.sums[r]) || radius.sums[r] < 0.0f) {
                throw std::runtime_error("invalid expert-only ENP coreset metadata at layer " + suffix);
            }
            weight_sum += weights.sums[r];
            transport_sum += transport.sums[r];
            max_radius = std::max<double>(max_radius, radius.sums[r]);
            for (int i = 0; i < HIDDEN; ++i) {
                if (!std::isfinite(enp.sums[r * HIDDEN + (size_t) i])) {
                    throw std::runtime_error("non-finite expert-only ENP hidden state at layer " + suffix);
                }
            }
        }
        if (std::fabs(weight_sum - (double) expected_tokens) > 0.5) {
            throw std::runtime_error("expert-only ENP coreset weight sum mismatch at layer " + suffix);
        }
        if (population.sums.size() != 1 || population.counts.size() != 1 || population.counts[0] != 1 ||
            !std::isfinite(population.sums[0]) ||
            (uint64_t) std::llround(population.sums[0]) != expected_tokens) {
            throw std::runtime_error("expert-only ENP population mismatch at layer " + suffix);
        }
        if (max_norm.sums.size() != 1 || max_norm.counts.size() != 1 || max_norm.counts[0] != 1 ||
            !std::isfinite(max_norm.sums[0]) || max_norm.sums[0] <= 0.0f) {
            throw std::runtime_error("bad expert-only ENP max norm at layer " + suffix);
        }
        int cert_sample_size = 0;
        int cert_tail_size = 0;
        if (has_cert_sample) {
            const auto cert = load_standard_enp_cert_sample(imat, layer);
            cert_sample_size = cert.remainder.n;
            cert_tail_size = cert.tail_split ? cert.tail.n : 0;
            if (cert.tail_split != has_cert_tail || (uint64_t) cert.full_population != expected_tokens) {
                throw std::runtime_error("ENP certification split/population mismatch at layer " + suffix);
            }
            const uint64_t expected_tail = has_cert_tail
                ? std::min<uint64_t>((uint64_t) cert_tail_requested, expected_tokens)
                : 0;
            const uint64_t expected_remainder = expected_tokens - expected_tail;
            const int expected_cert_size = (int) std::min<uint64_t>((uint64_t) cert_requested, expected_remainder);
            if ((uint64_t) cert_tail_size != expected_tail || cert_sample_size != expected_cert_size ||
                (uint64_t) cert.remainder.population != expected_remainder) {
                throw std::runtime_error("ENP certification sample size/population mismatch at layer " + suffix);
            }
        } else if (imat.entries.find("prune.enp.cert_sample.blk." + suffix) != imat.entries.end()) {
            throw std::runtime_error("partial ENP certification sample set");
        }
        layers.push_back({
            {"layer", layer},
            {"enp_sample_size", enp.counts.size()},
            {"enp_cert_sample_size", cert_sample_size},
            {"enp_cert_tail_size", cert_tail_size},
            {"enp_population_tokens", expected_tokens},
            {"enp_transport_total", transport_sum},
            {"enp_max_cluster_radius", max_radius},
        });
    }
    return {
        {"expert_only", true},
        {"routing_stats_available", false},
        {"enp_certification_available", has_cert_sample},
        {"enp_certification_requested", cert_requested},
        {"enp_certification_tail_requested", cert_tail_requested},
        {"expected_calibration_tokens", expected_tokens},
        {"exact_token_count_entry", true},
        {"expert_neuron_importance", "standard-enp-projection-eq9-eq10-on-task-blind-geometric-hidden-coreset"},
        {"layers", layers},
        {"imatrix_chunk_count", imat.chunk_count},
        {"imatrix_chunk_size", imat.chunk_size},
    };
}

static json read_json(const std::string & path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open JSON: " + path);
    json j;
    in >> j;
    return j;
}

static void write_json(const std::string & path, const json & j) {
    const std::string tmp = path + ".tmp";
    {
        std::ofstream out(tmp, std::ios::trunc);
        if (!out) throw std::runtime_error("cannot write JSON: " + tmp);
        out << std::setw(2) << j << '\n';
        out.flush();
        if (!out) throw std::runtime_error("failed writing JSON: " + tmp);
    }
    if (std::rename(tmp.c_str(), path.c_str()) != 0) throw std::runtime_error("rename failed for JSON output");
}

static void command_inspect(const std::string & model) {
    source_gguf src(model);
    require_source_contract(src);
    int q2 = 0, expert = 0, gdn = 0;
    std::map<std::string,int> expert_types;
    for (int64_t i = 0; i < src.tensor_count(); ++i) {
        const std::string name = gguf_get_tensor_name(src.meta(), i);
        const ggml_type t = gguf_get_tensor_type(src.meta(), i);
        if (name.find("_exps.weight") != std::string::npos) {
            ++expert;
            ++expert_types[type_name(t)];
            if (t == GGML_TYPE_Q2_0) ++q2;
        }
        if (ends_with(name, "ssm_out.weight")) ++gdn;
    }
    std::cout << "architecture: qwen35moe\n"
              << "tensors: " << src.tensor_count() << "\n"
              << "blocks: 41 (40 main + 1 MTP)\n"
              << "routed expert tensors: " << expert << " (Q2_0=" << q2 << ")\n"
              << "recurrent layers: " << gdn << "\n"
              << "vocab rows: " << src.tensor("token_embd.weight")->ne[1] << "\n"
              << "source SHA-256: " << sha256_file(model) << "\n";
    for (const auto & [k,v] : expert_types) std::cout << "expert type " << k << ": " << v << "\n";
}

static void command_plan(
        const std::string & model,
        const std::string & imatrix_path,
        const std::string & output,
        bool quick,
        bool allow_uncovered,
        bool expert_only,
        int n_threads) {
    if (expert_only && allow_uncovered) {
        throw std::runtime_error("--expert-only does not use router coverage; do not combine it with --allow-uncovered");
    }
    const std::string checkpoint_path = output + ".checkpoint.json";
    advisory_file_lock checkpoint_lock(checkpoint_path + ".lock");
    source_gguf src(model);
    require_source_contract(src);
    common_imatrix imat;
    if (!common_imatrix_load(imatrix_path, imat)) throw std::runtime_error("failed to load imatrix");
    const json calibration = expert_only
        ? validate_expert_only_calibration_contract(imat)
        : validate_calibration_contract(imat, quick || allow_uncovered);
    const bool geometric_enp = calibration.value("expert_neuron_importance", std::string()) ==
        "standard-enp-projection-eq9-eq10-on-task-blind-geometric-hidden-coreset";
    const std::string expert_selection = geometric_enp
        ? "standard-enp-geometric-coreset-projection-topk-v1"
        : "standard-enp-uniform-wor-projection-topk-v1";

    const std::string tool_sha = current_executable_sha256();
    const std::string robustness = quick ? "smoke-r=1" : "all-32-blocks";
    const json source_identity = file_identity(model);
    const json imatrix_identity = file_identity(imatrix_path);

    json plan;
    {
        std::ifstream checkpoint_in(checkpoint_path);
        if (checkpoint_in.good()) {
            checkpoint_in.close();
            plan = read_json(checkpoint_path);
            if (plan.value("checkpoint_format", std::string()) != "qwen35-plan-checkpoint-v2") {
                throw std::runtime_error("unsupported plan checkpoint format");
            }
            if (plan.value("format", std::string()) != "qwen35-combined-prune-v2" ||
                plan.value("tool_exe_sha256", std::string()) != tool_sha ||
                plan.value("plan_q2_robustness", std::string()) != robustness ||
                plan.value("allow_uncovered_experts", false) != allow_uncovered ||
                plan.value("expert_only", false) != expert_only ||
                plan.value("expert_selection", std::string()) != expert_selection) {
                throw std::runtime_error("plan checkpoint input/options mismatch");
            }
            if (!plan.contains("source_file_identity") || !plan.contains("imatrix_file_identity")) {
                throw std::runtime_error("plan checkpoint predates file-identity resume support; restart it with this binary");
            }
            require_file_identity(plan.at("source_file_identity"), model, "source");
            require_file_identity(plan.at("imatrix_file_identity"), imatrix_path, "imatrix");
            if (!plan.contains("experts")) throw std::runtime_error("plan checkpoint has no expert records");
            validate_partial_expert_plan(plan.at("experts"));
            std::cerr << "resuming plan at expert record " << plan.at("experts").size()
                      << "/" << (N_LAYER_ALL * N_EXPERT) << "\n";
        } else {
            const std::string source_sha = sha256_file(model);
            const std::string imatrix_sha = sha256_file(imatrix_path);
            plan["format"] = "qwen35-combined-prune-v2";
            plan["source_block_count"] = (int) src.get_u32("qwen35moe.block_count");
            plan["target_block_count"] = (int) src.get_u32("qwen35moe.block_count");
            plan["source_sha256"] = source_sha;
            plan["imatrix_sha256"] = imatrix_sha;
            plan["source_file_identity"] = source_identity;
            plan["imatrix_file_identity"] = imatrix_identity;
            plan["tool_exe_sha256"] = tool_sha;
            plan["llama_cpp_commit"] = llama_commit();
            plan["llama_cpp_build_info"] = llama_build_info();
            plan["compiler"] = llama_compiler();
            plan["build_target"] = llama_build_target();
            plan["cxx_flags"] = QWEN35_PRUNE_CXX_FLAGS;
            plan["targets"] = {
                {"expert_width", EXPERT_WIDTH_DST},
                {"expert_encoder", "source-quant-preserve-ablation"},
                {"gdn_dim", expert_only ? GDN_DIM_SRC : GDN_DIM_DST},
                {"vocab_size", expert_only ? 248320 : VOCAB_DST},
            };
            plan["expert_selection"] = expert_selection;
            plan["plan_q2_robustness"] = robustness;
            plan["allow_uncovered_experts"] = allow_uncovered;
            plan["expert_only"] = expert_only;
            plan["calibration"] = calibration;
            plan["experts"] = json::array();
        }
    }

    plan["checkpoint_format"] = "qwen35-plan-checkpoint-v2";
    auto checkpoint = [&](size_t completed) {
        plan["checkpoint_completed_experts"] = completed;
        write_json(checkpoint_path, plan);
    };
    // Persist the authenticated source/imatrix hashes before the first costly
    // expert chunk. A killed fresh run can therefore resume without hashing the
    // 20+ GiB source again; final plan publication still re-hashes both inputs.
    checkpoint(plan.at("experts").size());
    extend_expert_plan(src, imat, quick, allow_uncovered, n_threads, plan["experts"], checkpoint);

    plan.erase("checkpoint_format");
    plan.erase("checkpoint_completed_experts");
    plan["gdn"] = expert_only ? make_keep_source_gdn_plan() : make_gdn_plan(src, imat);
    plan["vocab"] = expert_only ? make_keep_source_vocab_plan() : make_vocab_plan(src, imat);
    verify_plan_json(plan);
    // Checkpoint resumes use a cheap inode/size/mtime identity so repeated
    // short-lived invocations do not re-read a 20+ GiB source. Before the plan
    // becomes authoritative, re-authenticate both immutable inputs with full
    // SHA-256. This preserves the BAOMU hash contract even if a file changed
    // and was maliciously restored to the same metadata between resumes.
    require_file_identity(plan.at("source_file_identity"), model, "source");
    require_file_identity(plan.at("imatrix_file_identity"), imatrix_path, "imatrix");
    if (sha256_file(model) != plan.at("source_sha256").get<std::string>()) {
        throw std::runtime_error("source SHA-256 changed during plan generation");
    }
    if (sha256_file(imatrix_path) != plan.at("imatrix_sha256").get<std::string>()) {
        throw std::runtime_error("imatrix SHA-256 changed during plan generation");
    }
    write_json(output, plan);
    std::remove(checkpoint_path.c_str());
    std::cout << "wrote verified plan: " << output << "\n";
}

static void command_probe_expert(
        const std::string & model,
        const std::string & imatrix_path,
        int layer,
        int expert) {
    if (layer < 0 || layer >= N_LAYER_ALL || expert < 0 || expert >= N_EXPERT) {
        throw std::runtime_error("probe-expert layer/expert out of range");
    }
    source_gguf src(model);
    require_source_contract(src);
    common_imatrix imat;
    if (!common_imatrix_load(imatrix_path, imat)) throw std::runtime_error("failed to load imatrix");
    const standard_enp_sample enp_sample = load_standard_enp_sample(imat, layer);
    standard_enp_certification cert_sample;
    const standard_enp_certification * cert_sample_ptr = nullptr;
    if (imat.entries.find("prune.enp.cert_sample.blk." + std::to_string(layer)) != imat.entries.end()) {
        cert_sample = load_standard_enp_cert_sample(imat, layer);
        cert_sample_ptr = &cert_sample;
    }
    json record = make_expert_record(src, imat, enp_sample, cert_sample_ptr, layer, expert,
                                     /*quick=*/ false,
                                     /*allow_uncovered=*/ true);

    const std::string prefix = "blk." + std::to_string(layer) + ".";
    const auto gate = decode_expert_slice(src, prefix + "ffn_gate_exps.weight", expert, HIDDEN, EXPERT_WIDTH_SRC);
    const auto up   = decode_expert_slice(src, prefix + "ffn_up_exps.weight",   expert, HIDDEN, EXPERT_WIDTH_SRC);
    const auto down = decode_expert_slice(src, prefix + "ffn_down_exps.weight", expert, EXPERT_WIDTH_SRC, HIDDEN);
    const auto down_norm2 = down_column_norm2(down);
    int gate_zero = 0, up_zero = 0, down_zero = 0;
    double min_gate_norm2 = std::numeric_limits<double>::infinity();
    double min_up_norm2 = std::numeric_limits<double>::infinity();
    double min_down_norm2 = std::numeric_limits<double>::infinity();
    for (int j = 0; j < EXPERT_WIDTH_SRC; ++j) {
        double gn = 0.0, un = 0.0;
        for (int i = 0; i < HIDDEN; ++i) {
            const double g = gate.row(j)[i];
            const double u = up.row(j)[i];
            gn += g*g;
            un += u*u;
        }
        gate_zero += gn == 0.0;
        up_zero += un == 0.0;
        down_zero += down_norm2[(size_t) j] == 0.0;
        if (gn > 0.0) min_gate_norm2 = std::min(min_gate_norm2, gn);
        if (un > 0.0) min_up_norm2 = std::min(min_up_norm2, un);
        if (down_norm2[(size_t) j] > 0.0) min_down_norm2 = std::min(min_down_norm2, down_norm2[(size_t) j]);
    }
    record["weight_diagnostics"] = {
        {"gate_zero_rows", gate_zero},
        {"up_zero_rows", up_zero},
        {"down_zero_columns", down_zero},
        {"min_positive_gate_norm2", min_gate_norm2},
        {"min_positive_up_norm2", min_up_norm2},
        {"min_positive_down_norm2", min_down_norm2},
    };
    std::cout << std::setw(2) << record << '\n';
}

static double enp_quantile(std::vector<double> values, double q) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const double pos = std::clamp(q, 0.0, 1.0) * (values.size() - 1);
    const size_t lo = (size_t) std::floor(pos);
    const size_t hi = (size_t) std::ceil(pos);
    if (lo == hi) return values[lo];
    return values[lo] * (hi - pos) + values[hi] * (pos - lo);
}

static double enp_worst_decile_mean(std::vector<double> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end(), std::greater<double>());
    const size_t n = std::max<size_t>(1, (values.size() + 9) / 10);
    return std::accumulate(values.begin(), values.begin() + n, 0.0) / n;
}

static constexpr int ENP_DOWNSTREAM_LENS_DIM = 256;
static constexpr uint64_t ENP_DOWNSTREAM_LENS_SKETCH_SEED = 0x3c6ef372fe94f82bULL;

static uint64_t enp_downstream_mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static std::array<double, ENP_DOWNSTREAM_LENS_DIM> enp_downstream_sketch_down_column(
        const decoded_expert_slice & down,
        int neuron) {
    std::array<double, ENP_DOWNSTREAM_LENS_DIM> out {};
    for (int i = 0; i < HIDDEN; ++i) {
        const uint64_t h = enp_downstream_mix64(ENP_DOWNSTREAM_LENS_SKETCH_SEED ^ (uint64_t) i);
        const int bucket = (int) (h & (ENP_DOWNSTREAM_LENS_DIM - 1));
        const double sign = ((h >> 8) & 1u) ? 1.0 : -1.0;
        out[(size_t) bucket] += sign * down.row(i)[neuron];
    }
    return out;
}

struct enp_downstream_lens_sample {
    int n = 0;
    std::vector<float> x;
    std::vector<float> y;
};

static enp_downstream_lens_sample load_enp_downstream_lens_sample(
        const common_imatrix & imat,
        int layer) {
    const auto & dim = require_entry(imat, "prune.enp.lens.dim");
    if (dim.sums.size() != 1 || dim.counts.size() != 1 || dim.counts[0] != 1 ||
            (int) std::llround(dim.sums[0]) != ENP_DOWNSTREAM_LENS_DIM) {
        throw std::runtime_error("unexpected downstream ENP lens dimension");
    }
    const auto & x = require_entry(imat, "prune.enp.lens.post_moe.blk." + std::to_string(layer));
    const auto & y = require_entry(imat, "prune.enp.lens.final");
    if (x.counts.empty() || x.counts.size() != y.counts.size() ||
            x.sums.size() != x.counts.size() * ENP_DOWNSTREAM_LENS_DIM ||
            y.sums.size() != y.counts.size() * ENP_DOWNSTREAM_LENS_DIM) {
        throw std::runtime_error("malformed downstream ENP lens sample");
    }
    for (size_t i = 0; i < x.counts.size(); ++i) {
        if (x.counts[i] != 1 || y.counts[i] != 1) {
            throw std::runtime_error("non-unit downstream ENP lens sample count");
        }
    }
    enp_downstream_lens_sample out;
    out.n = (int) x.counts.size();
    out.x = x.sums;
    out.y = y.sums;
    return out;
}

struct enp_downstream_lens_fit {
    std::vector<double> map;
    double ridge = 0.0;
    double train_relative_mse = 0.0;
};

struct expert_replacement_sample {
    int n = 0;
    int64_t population = 0;
    std::vector<float> x;
};

static expert_replacement_sample load_expert_replacement_sample(
        const common_imatrix & imat,
        int layer) {
    const auto & dim = require_entry(imat, "prune.expert_replace.hidden_dim");
    if (dim.sums.size() != 1 || dim.counts.size() != 1 || dim.counts[0] != 1 ||
            (int) std::llround(dim.sums[0]) != HIDDEN) {
        throw std::runtime_error("unexpected expert-replacement hidden dimension");
    }
    const auto & e = require_entry(imat, "prune.expert_replace.input.blk." + std::to_string(layer));
    if (e.counts.empty() || e.sums.size() != e.counts.size() * HIDDEN) {
        throw std::runtime_error("malformed expert-replacement sample at layer " + std::to_string(layer));
    }
    for (int64_t c : e.counts) {
        if (c != 1) throw std::runtime_error("expert-replacement sample is not an exact unit-count reservoir");
    }
    const auto & population = require_entry(imat, "prune.expert_replace.population.blk." + std::to_string(layer));
    if (population.sums.size() != 1 || population.counts.size() != 1 || population.counts[0] != 1 ||
            population.sums[0] <= 0.0f) {
        throw std::runtime_error("malformed expert-replacement population");
    }
    expert_replacement_sample out;
    out.n = (int) e.counts.size();
    out.population = (int64_t) std::llround(population.sums[0]);
    out.x = e.sums;
    return out;
}

static std::vector<float> decode_matrix_rows(
        const source_gguf & src,
        const std::string & name,
        int expected_ne0,
        int expected_ne1) {
    const ggml_tensor * t = src.tensor(name);
    if (ggml_n_dims(t) != 2 || t->ne[0] != expected_ne0 || t->ne[1] != expected_ne1) {
        throw std::runtime_error("unexpected matrix shape: " + name);
    }
    const size_t row_size = ggml_row_size(t->type, t->ne[0]);
    std::vector<uint8_t> packed(src.tensor_size(name));
    src.read_tensor_bytes(name, 0, packed.data(), packed.size());
    std::vector<float> out((size_t) expected_ne0 * expected_ne1);
    const auto * traits = ggml_get_type_traits(t->type);
    for (int r = 0; r < expected_ne1; ++r) {
        const uint8_t * row = packed.data() + (size_t) r * row_size;
        float * dst = out.data() + (size_t) r * expected_ne0;
        if (t->type == GGML_TYPE_F32) {
            std::memcpy(dst, row, (size_t) expected_ne0 * sizeof(float));
        } else {
            if (!traits || !traits->to_float) throw std::runtime_error("cannot decode matrix: " + name);
            traits->to_float(row, dst, expected_ne0);
        }
    }
    return out;
}

static enp_downstream_lens_fit fit_enp_downstream_lens(const enp_downstream_lens_sample & sample) {
    const int d = ENP_DOWNSTREAM_LENS_DIM;
    if (sample.n <= d || sample.x.size() != (size_t) sample.n * d ||
            sample.y.size() != (size_t) sample.n * d) {
        throw std::runtime_error("downstream ENP lens sample is too small");
    }
    std::vector<double> mean_x((size_t) d, 0.0);
    std::vector<double> mean_y((size_t) d, 0.0);
    for (int r = 0; r < sample.n; ++r) {
        const float * x = sample.x.data() + (size_t) r * d;
        const float * y = sample.y.data() + (size_t) r * d;
        for (int j = 0; j < d; ++j) {
            mean_x[(size_t) j] += x[j];
            mean_y[(size_t) j] += y[j];
        }
    }
    for (int j = 0; j < d; ++j) {
        mean_x[(size_t) j] /= sample.n;
        mean_y[(size_t) j] /= sample.n;
    }

    std::vector<double> a((size_t) d * d, 0.0);
    std::vector<double> c((size_t) d * d, 0.0);
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < d; ++j) {
        for (int k = 0; k < d; ++k) {
            double ax = 0.0;
            double cy = 0.0;
            for (int r = 0; r < sample.n; ++r) {
                const double xj = sample.x[(size_t) r * d + j] - mean_x[(size_t) j];
                ax += xj * (sample.x[(size_t) r * d + k] - mean_x[(size_t) k]);
                cy += xj * (sample.y[(size_t) r * d + k] - mean_y[(size_t) k]);
            }
            a[(size_t) j * d + k] = ax / sample.n;
            c[(size_t) j * d + k] = cy / sample.n;
        }
    }
    double trace = 0.0;
    for (int j = 0; j < d; ++j) trace += a[(size_t) j * d + j];
    const double ridge = std::max(1.0e-8, 1.0e-3 * trace / d);
    for (int j = 0; j < d; ++j) a[(size_t) j * d + j] += ridge;

    std::vector<double> l((size_t) d * d, 0.0);
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum = a[(size_t) i * d + j];
            for (int k = 0; k < j; ++k) sum -= l[(size_t) i * d + k] * l[(size_t) j * d + k];
            if (i == j) {
                if (!(sum > 0.0) || !std::isfinite(sum)) {
                    throw std::runtime_error("downstream ENP lens ridge matrix is not positive definite");
                }
                l[(size_t) i * d + j] = std::sqrt(sum);
            } else {
                l[(size_t) i * d + j] = sum / l[(size_t) j * d + j];
            }
        }
    }

    std::vector<double> map((size_t) d * d, 0.0);
    std::vector<double> tmp((size_t) d, 0.0);
    for (int col = 0; col < d; ++col) {
        for (int i = 0; i < d; ++i) {
            double sum = c[(size_t) i * d + col];
            for (int k = 0; k < i; ++k) sum -= l[(size_t) i * d + k] * tmp[(size_t) k];
            tmp[(size_t) i] = sum / l[(size_t) i * d + i];
        }
        for (int i = d - 1; i >= 0; --i) {
            double sum = tmp[(size_t) i];
            for (int k = i + 1; k < d; ++k) sum -= l[(size_t) k * d + i] * map[(size_t) k * d + col];
            map[(size_t) i * d + col] = sum / l[(size_t) i * d + i];
        }
    }

    double err2 = 0.0;
    double target2 = 0.0;
    #pragma omp parallel for reduction(+:err2,target2) schedule(static)
    for (int r = 0; r < sample.n; ++r) {
        std::array<double, ENP_DOWNSTREAM_LENS_DIM> pred {};
        for (int j = 0; j < d; ++j) {
            const double xj = sample.x[(size_t) r * d + j] - mean_x[(size_t) j];
            const double * row = map.data() + (size_t) j * d;
            #pragma omp simd
            for (int k = 0; k < d; ++k) pred[(size_t) k] += xj * row[k];
        }
        for (int k = 0; k < d; ++k) {
            const double target = sample.y[(size_t) r * d + k] - mean_y[(size_t) k];
            const double e = target - pred[(size_t) k];
            err2 += e * e;
            target2 += target * target;
        }
    }
    return {std::move(map), ridge, err2 / std::max(1.0e-30, target2)};
}

static std::vector<double> enp_downstream_survival(
        const decoded_expert_slice & down,
        const enp_downstream_lens_fit & lens) {
    const int d = ENP_DOWNSTREAM_LENS_DIM;
    if (lens.map.size() != (size_t) d * d) throw std::runtime_error("malformed downstream ENP lens fit");
    std::vector<double> out(EXPERT_WIDTH_SRC, 0.0);
    #pragma omp parallel for schedule(static)
    for (int neuron = 0; neuron < EXPERT_WIDTH_SRC; ++neuron) {
        const auto s = enp_downstream_sketch_down_column(down, neuron);
        double in2 = 0.0;
        double out2 = 0.0;
        std::array<double, ENP_DOWNSTREAM_LENS_DIM> mapped {};
        for (int j = 0; j < d; ++j) in2 += s[(size_t) j] * s[(size_t) j];
        for (int j = 0; j < d; ++j) {
            const double sj = s[(size_t) j];
            const double * row = lens.map.data() + (size_t) j * d;
            #pragma omp simd
            for (int k = 0; k < d; ++k) mapped[(size_t) k] += sj * row[k];
        }
        for (double v : mapped) out2 += v * v;
        out[(size_t) neuron] = std::sqrt(out2 / std::max(1.0e-30, in2));
    }
    return out;
}

static json make_downstream_lens_expert_record(
        const source_gguf & src,
        const standard_enp_sample & enp_sample,
        const enp_downstream_lens_fit & lens,
        int layer,
        int expert) {
    const std::string prefix = "blk." + std::to_string(layer) + ".";
    const auto gate = decode_expert_slice(src, prefix + "ffn_gate_exps.weight", expert, HIDDEN, EXPERT_WIDTH_SRC);
    const auto up   = decode_expert_slice(src, prefix + "ffn_up_exps.weight",   expert, HIDDEN, EXPERT_WIDTH_SRC);
    const auto down = decode_expert_slice(src, prefix + "ffn_down_exps.weight", expert, EXPERT_WIDTH_SRC, HIDDEN);
    const auto enp = score_standard_enp_mean_only_decoded(gate, up, down, enp_sample);
    const auto survival = enp_downstream_survival(down, lens);

    std::vector<std::pair<double, int>> ranking;
    ranking.reserve(EXPERT_WIDTH_SRC);
    for (int j = 0; j < EXPERT_WIDTH_SRC; ++j) {
        ranking.emplace_back(enp[(size_t) j] * survival[(size_t) j], j);
    }
    std::stable_sort(ranking.begin(), ranking.end(), [](const auto & a, const auto & b) {
        if (a.first != b.first) return a.first > b.first;
        return a.second < b.second;
    });

    std::vector<int> selected;
    selected.reserve(EXPERT_WIDTH_DST);
    for (int i = 0; i < EXPERT_WIDTH_DST; ++i) selected.push_back(ranking[(size_t) i].second);
    std::sort(selected.begin(), selected.end());

    std::vector<double> sorted_survival = survival;
    std::sort(sorted_survival.begin(), sorted_survival.end());
    return {
        {"layer", layer},
        {"expert", expert},
        {"output_to_input", selected},
        {"routing_count", -1},
        {"routing_count_available", false},
        {"zero_routing_weight_only_fallback", false},
        {"selection_method", "standard-enp-times-post-moe-to-final-256d-ridge-survival-v1"},
        {"enp_population_tokens", enp_sample.population},
        {"enp_sample_size", enp_sample.n},
        {"enp_coreset_max_points", enp_sample.geometric ? enp_sample.n : 0},
        {"enp_coreset_method", enp_sample.geometric ? enp_sample.method : ""},
        {"downstream_lens_ridge", lens.ridge},
        {"downstream_lens_train_relative_mse", lens.train_relative_mse},
        {"downstream_survival_min", sorted_survival.front()},
        {"downstream_survival_median", sorted_survival[sorted_survival.size() / 2]},
        {"downstream_survival_max", sorted_survival.back()},
        {"downstream_score_cutoff", ranking[(size_t) EXPERT_WIDTH_DST - 1].first},
        {"downstream_score_margin", ranking[(size_t) EXPERT_WIDTH_DST - 1].first - ranking[(size_t) EXPERT_WIDTH_DST].first},
        {"enp_top256_certified", false},
        {"enp_top256_unique", false},
        {"enp_full_population_exact", false},
        {"enp_full_population_fallback_required", false},
        {"packing_swaps", 0},
    };
}

static double enp_downstream_mapped_norm2(
        const enp_downstream_lens_fit & lens,
        const double * v,
        const double * subtract = nullptr) {
    const int d = ENP_DOWNSTREAM_LENS_DIM;
    std::array<double, ENP_DOWNSTREAM_LENS_DIM> sketch {};
    for (int i = 0; i < HIDDEN; ++i) {
        const uint64_t h = enp_downstream_mix64(ENP_DOWNSTREAM_LENS_SKETCH_SEED ^ (uint64_t) i);
        const int bucket = (int) (h & (ENP_DOWNSTREAM_LENS_DIM - 1));
        const double sign = ((h >> 8) & 1u) ? 1.0 : -1.0;
        sketch[(size_t) bucket] += sign * (v[i] - (subtract ? subtract[i] : 0.0));
    }
    std::array<double, ENP_DOWNSTREAM_LENS_DIM> mapped {};
    for (int j = 0; j < d; ++j) {
        const double sj = sketch[(size_t) j];
        const double * row = lens.map.data() + (size_t) j * d;
        #pragma omp simd
        for (int k = 0; k < d; ++k) mapped[(size_t) k] += sj * row[k];
    }
    double norm2 = 0.0;
    for (double v : mapped) norm2 += v * v;
    return norm2;
}

static json heldout_energy_report_decoded(
        const decoded_expert_slice & gate,
        const decoded_expert_slice & up,
        const decoded_expert_slice & down,
        const standard_enp_sample & heldout,
        const std::vector<int> & proposal_order,
        const enp_downstream_lens_fit * downstream_lens = nullptr) {
    static constexpr std::array<int, 4> WIDTHS = {256, 320, 384, 416};
    static constexpr int N_SHARDS = 8;
    if (heldout.n <= 0 || heldout.x.size() != (size_t) heldout.n * HIDDEN ||
        heldout.weight.size() != (size_t) heldout.n || proposal_order.size() != EXPERT_WIDTH_SRC) {
        throw std::runtime_error("malformed held-out ENP sample/order");
    }

    std::array<std::vector<double>, WIDTHS.size()> projection_loss;
    std::array<std::vector<double>, WIDTHS.size()> l2_loss;
    std::array<double, WIDTHS.size()> selected_projection_mass {};
    std::array<std::array<double, N_SHARDS>, WIDTHS.size()> shard_projection_sum {};
    std::array<std::array<double, N_SHARDS>, WIDTHS.size()> shard_l2_sum {};
    std::array<std::array<int, N_SHARDS>, WIDTHS.size()> shard_count {};
    std::vector<double> neuron_projection(EXPERT_WIDTH_SRC, 0.0);
    double full_projection_mass = 0.0;
    double total_weight = 0.0;
    int valid_ratio_tokens = 0;
    double downstream_full_mass = 0.0;
    double downstream_error_mass_384 = 0.0;
    std::vector<double> downstream_relative_error_384;
    downstream_relative_error_384.reserve((size_t) heldout.n);

    std::array<float, EXPERT_WIDTH_SRC> m {};
    std::array<double, HIDDEN> y {};
    std::array<std::array<double, HIDDEN>, WIDTHS.size()> yk {};
    std::array<double, EXPERT_WIDTH_SRC> dty {};

    for (auto & v : projection_loss) v.reserve((size_t) heldout.n);
    for (auto & v : l2_loss) v.reserve((size_t) heldout.n);

    for (int r = 0; r < heldout.n; ++r) {
        const float * x = heldout.x.data() + (size_t) r * HIDDEN;
        const double w = heldout.weight[(size_t) r];
        total_weight += w;
        for (int j = 0; j < EXPERT_WIDTH_SRC; ++j) {
            const float * wg = gate.row(j);
            const float * wu = up.row(j);
            float g = 0.0f;
            float u = 0.0f;
            #pragma omp simd reduction(+:g,u)
            for (int i = 0; i < HIDDEN; ++i) {
                g += wg[i] * x[i];
                u += wu[i] * x[i];
            }
            m[(size_t) j] = standard_enp_silu(g) * u;
        }

        double norm2 = 0.0;
        for (int i = 0; i < HIDDEN; ++i) {
            const float * wd = down.row(i);
            double yi = 0.0;
            size_t wi = 0;
            for (int rank = 0; rank < EXPERT_WIDTH_SRC; ++rank) {
                const int j = proposal_order[(size_t) rank];
                yi += (double) wd[j] * m[(size_t) j];
                if (wi < WIDTHS.size() && rank + 1 == WIDTHS[wi]) {
                    yk[wi][(size_t) i] = yi;
                    ++wi;
                }
            }
            y[(size_t) i] = yi;
            norm2 += yi * yi;
        }

        std::fill(dty.begin(), dty.end(), 0.0);
        for (int i = 0; i < HIDDEN; ++i) {
            const float * wd = down.row(i);
            const double yi = y[(size_t) i];
            #pragma omp simd
            for (int j = 0; j < EXPERT_WIDTH_SRC; ++j) dty[(size_t) j] += (double) wd[j] * yi;
        }
        const double output_norm = std::sqrt(norm2);
        const double denom = output_norm + 1.0e-8;
        std::array<double, EXPERT_WIDTH_SRC> p {};
        double full_p = 0.0;
        for (int j = 0; j < EXPERT_WIDTH_SRC; ++j) {
            p[(size_t) j] = (double) m[(size_t) j] * dty[(size_t) j] / denom;
            full_p += p[(size_t) j];
            neuron_projection[(size_t) j] += w * p[(size_t) j];
        }
        full_projection_mass += w * full_p;

        if (downstream_lens) {
            const double full_downstream = enp_downstream_mapped_norm2(*downstream_lens, y.data());
            const double err_downstream = enp_downstream_mapped_norm2(
                *downstream_lens, y.data(), yk[2].data());
            downstream_full_mass += w * full_downstream;
            downstream_error_mass_384 += w * err_downstream;
            if (full_downstream > 1.0e-20) {
                downstream_relative_error_384.push_back(err_downstream / full_downstream);
            }
        }

        if (norm2 <= 1.0e-20 || std::fabs(full_p) <= 1.0e-20) continue;
        ++valid_ratio_tokens;
        const int shard = std::min(N_SHARDS - 1, r * N_SHARDS / heldout.n);
        for (size_t wi = 0; wi < WIDTHS.size(); ++wi) {
            double selected_p = 0.0;
            for (int rank = 0; rank < WIDTHS[wi]; ++rank) {
                selected_p += p[(size_t) proposal_order[(size_t) rank]];
            }
            selected_projection_mass[wi] += w * selected_p;
            const double proj_loss = 1.0 - selected_p / full_p;
            double err2 = 0.0;
            for (int i = 0; i < HIDDEN; ++i) {
                const double e = y[(size_t) i] - yk[wi][(size_t) i];
                err2 += e * e;
            }
            const double rel_l2 = err2 / norm2;
            projection_loss[wi].push_back(proj_loss);
            l2_loss[wi].push_back(rel_l2);
            shard_projection_sum[wi][(size_t) shard] += proj_loss;
            shard_l2_sum[wi][(size_t) shard] += rel_l2;
            shard_count[wi][(size_t) shard] += 1;
        }
    }
    if (!(total_weight > 0.0) || !(full_projection_mass > 0.0) || valid_ratio_tokens <= 0) {
        throw std::runtime_error("held-out expert output/projection mass is degenerate");
    }

    for (double & v : neuron_projection) v /= total_weight;
    std::vector<std::pair<double, int>> oracle;
    oracle.reserve(EXPERT_WIDTH_SRC);
    for (int j = 0; j < EXPERT_WIDTH_SRC; ++j) oracle.emplace_back(neuron_projection[(size_t) j], j);
    std::stable_sort(oracle.begin(), oracle.end(), [](const auto & a, const auto & b) {
        if (a.first != b.first) return a.first > b.first;
        return a.second < b.second;
    });
    const double full_mean_projection = std::accumulate(neuron_projection.begin(), neuron_projection.end(), 0.0);

    json widths = json::array();
    for (size_t wi = 0; wi < WIDTHS.size(); ++wi) {
        const double proj_mean = std::accumulate(projection_loss[wi].begin(), projection_loss[wi].end(), 0.0) /
            projection_loss[wi].size();
        const double l2_mean = std::accumulate(l2_loss[wi].begin(), l2_loss[wi].end(), 0.0) / l2_loss[wi].size();
        double oracle_mass = 0.0;
        for (int rank = 0; rank < WIDTHS[wi]; ++rank) oracle_mass += oracle[(size_t) rank].first;
        json shard_proj = json::array();
        json shard_l2 = json::array();
        double shard_proj_min = std::numeric_limits<double>::infinity();
        double shard_proj_max = -std::numeric_limits<double>::infinity();
        double shard_l2_min = std::numeric_limits<double>::infinity();
        double shard_l2_max = -std::numeric_limits<double>::infinity();
        for (int s = 0; s < N_SHARDS; ++s) {
            if (shard_count[wi][(size_t) s] == 0) continue;
            const double sp = shard_projection_sum[wi][(size_t) s] / shard_count[wi][(size_t) s];
            const double sl = shard_l2_sum[wi][(size_t) s] / shard_count[wi][(size_t) s];
            shard_proj.push_back(sp);
            shard_l2.push_back(sl);
            shard_proj_min = std::min(shard_proj_min, sp);
            shard_proj_max = std::max(shard_proj_max, sp);
            shard_l2_min = std::min(shard_l2_min, sl);
            shard_l2_max = std::max(shard_l2_max, sl);
        }
        widths.push_back({
            {"width", WIDTHS[wi]},
            {"projection_retained_ratio_of_means", selected_projection_mass[wi] / full_projection_mass},
            {"heldout_oracle_projection_retained_fraction", oracle_mass / full_mean_projection},
            {"heldout_excess_projection_loss_fraction",
                oracle_mass / full_mean_projection - selected_projection_mass[wi] / full_projection_mass},
            {"projection_loss_token_mean", proj_mean},
            {"projection_loss_token_p50", enp_quantile(projection_loss[wi], 0.50)},
            {"projection_loss_token_p90", enp_quantile(projection_loss[wi], 0.90)},
            {"projection_loss_token_p95", enp_quantile(projection_loss[wi], 0.95)},
            {"projection_loss_token_worst10_mean", enp_worst_decile_mean(projection_loss[wi])},
            {"projection_loss_shard_means", shard_proj},
            {"projection_loss_shard_range", shard_proj_max - shard_proj_min},
            {"relative_l2_token_mean", l2_mean},
            {"relative_l2_token_p50", enp_quantile(l2_loss[wi], 0.50)},
            {"relative_l2_token_p90", enp_quantile(l2_loss[wi], 0.90)},
            {"relative_l2_token_p95", enp_quantile(l2_loss[wi], 0.95)},
            {"relative_l2_token_worst10_mean", enp_worst_decile_mean(l2_loss[wi])},
            {"relative_l2_shard_means", shard_l2},
            {"relative_l2_shard_range", shard_l2_max - shard_l2_min},
        });
    }
    json out = {
        {"heldout_sample_size", heldout.n},
        {"heldout_population", heldout.population},
        {"valid_ratio_tokens", valid_ratio_tokens},
        {"deterministic_random_shards", N_SHARDS},
        {"widths", widths},
    };
    if (downstream_lens) {
        if (!(downstream_full_mass > 0.0) || downstream_relative_error_384.empty()) {
            throw std::runtime_error("held-out downstream-lens expert energy is degenerate");
        }
        out["downstream_lens_384"] = {
            {"predicted_final_hidden_relative_error_ratio_of_means",
                downstream_error_mass_384 / downstream_full_mass},
            {"predicted_final_hidden_relative_error_token_mean",
                std::accumulate(downstream_relative_error_384.begin(), downstream_relative_error_384.end(), 0.0) /
                    downstream_relative_error_384.size()},
            {"predicted_final_hidden_relative_error_token_p90",
                enp_quantile(downstream_relative_error_384, 0.90)},
            {"predicted_final_hidden_relative_error_token_p95",
                enp_quantile(downstream_relative_error_384, 0.95)},
            {"predicted_final_hidden_relative_error_worst10_mean",
                enp_worst_decile_mean(downstream_relative_error_384)},
        };
    }
    return out;
}

static void command_probe_expert_heldout(
        const std::string & model,
        const std::string & selection_imatrix_path,
        const std::string & heldout_imatrix_path,
        int layer,
        int expert) {
    if (layer < 0 || layer >= N_LAYER_MAIN || expert < 0 || expert >= N_EXPERT) {
        throw std::runtime_error("probe-expert-heldout layer/expert out of range");
    }
    source_gguf src(model);
    require_source_contract(src);
    common_imatrix selection_imat;
    common_imatrix heldout_imat;
    if (!common_imatrix_load(selection_imatrix_path, selection_imat)) {
        throw std::runtime_error("failed to load selection imatrix");
    }
    if (!common_imatrix_load(heldout_imatrix_path, heldout_imat)) {
        throw std::runtime_error("failed to load held-out imatrix");
    }
    const auto selection_sample = load_standard_enp_sample(selection_imat, layer);
    const auto heldout_cert = load_standard_enp_cert_sample(heldout_imat, layer);
    if (heldout_cert.tail.n != 0) {
        throw std::runtime_error("probe-expert-heldout currently requires --prune-enp-cert-tail-points 0");
    }

    const std::string prefix = "blk." + std::to_string(layer) + ".";
    const auto gate = decode_expert_slice(src, prefix + "ffn_gate_exps.weight", expert, HIDDEN, EXPERT_WIDTH_SRC);
    const auto up   = decode_expert_slice(src, prefix + "ffn_up_exps.weight",   expert, HIDDEN, EXPERT_WIDTH_SRC);
    const auto down = decode_expert_slice(src, prefix + "ffn_down_exps.weight", expert, EXPERT_WIDTH_SRC, HIDDEN);
    const auto proposal_score = score_standard_enp_on_sample_decoded(
        gate, up, down, selection_sample, layer, expert);
    std::vector<int> proposal_order(EXPERT_WIDTH_SRC);
    std::iota(proposal_order.begin(), proposal_order.end(), 0);
    std::stable_sort(proposal_order.begin(), proposal_order.end(), [&](int a, int b) {
        if (proposal_score.mean[(size_t) a] != proposal_score.mean[(size_t) b]) {
            return proposal_score.mean[(size_t) a] > proposal_score.mean[(size_t) b];
        }
        return a < b;
    });

    json out = heldout_energy_report_decoded(gate, up, down, heldout_cert.remainder, proposal_order);
    out["layer"] = layer;
    out["expert"] = expert;
    out["selection_method"] = "geometric-coreset-ranking-fixed-before-heldout";
    std::cout << std::setw(2) << out << '\n';
}

static void command_probe_expert_heldout_lens(
        const std::string & model,
        const std::string & selection_imatrix_path,
        const std::string & lens_imatrix_path,
        const std::string & heldout_imatrix_path,
        int layer,
        int expert) {
    if (layer < 0 || layer >= N_LAYER_MAIN || expert < 0 || expert >= N_EXPERT) {
        throw std::runtime_error("probe-expert-heldout-lens layer/expert out of range");
    }
    source_gguf src(model);
    require_source_contract(src);
    common_imatrix selection_imat;
    common_imatrix lens_imat;
    common_imatrix heldout_imat;
    if (!common_imatrix_load(selection_imatrix_path, selection_imat)) {
        throw std::runtime_error("failed to load selection imatrix");
    }
    if (!common_imatrix_load(lens_imatrix_path, lens_imat)) {
        throw std::runtime_error("failed to load downstream-lens imatrix");
    }
    if (!common_imatrix_load(heldout_imatrix_path, heldout_imat)) {
        throw std::runtime_error("failed to load held-out imatrix");
    }
    const auto selection_sample = load_standard_enp_sample(selection_imat, layer);
    const auto heldout_cert = load_standard_enp_cert_sample(heldout_imat, layer);
    if (heldout_cert.tail.n != 0) {
        throw std::runtime_error("probe-expert-heldout-lens requires held-out tail points = 0");
    }
    const auto lens_sample = load_enp_downstream_lens_sample(lens_imat, layer);
    const auto lens_fit = fit_enp_downstream_lens(lens_sample);

    const std::string prefix = "blk." + std::to_string(layer) + ".";
    const auto gate = decode_expert_slice(src, prefix + "ffn_gate_exps.weight", expert, HIDDEN, EXPERT_WIDTH_SRC);
    const auto up   = decode_expert_slice(src, prefix + "ffn_up_exps.weight",   expert, HIDDEN, EXPERT_WIDTH_SRC);
    const auto down = decode_expert_slice(src, prefix + "ffn_down_exps.weight", expert, EXPERT_WIDTH_SRC, HIDDEN);
    const auto enp_mean = score_standard_enp_mean_only_decoded(gate, up, down, selection_sample);
    const auto survival = enp_downstream_survival(down, lens_fit);

    std::vector<int> old_order(EXPERT_WIDTH_SRC);
    std::vector<int> lens_order(EXPERT_WIDTH_SRC);
    std::iota(old_order.begin(), old_order.end(), 0);
    std::iota(lens_order.begin(), lens_order.end(), 0);
    std::stable_sort(old_order.begin(), old_order.end(), [&](int a, int b) {
        if (enp_mean[(size_t) a] != enp_mean[(size_t) b]) return enp_mean[(size_t) a] > enp_mean[(size_t) b];
        return a < b;
    });
    std::stable_sort(lens_order.begin(), lens_order.end(), [&](int a, int b) {
        const double sa = enp_mean[(size_t) a] * survival[(size_t) a];
        const double sb = enp_mean[(size_t) b] * survival[(size_t) b];
        if (sa != sb) return sa > sb;
        return a < b;
    });

    std::array<uint8_t, EXPERT_WIDTH_SRC> old384 {};
    int changed384 = 0;
    for (int rank = 0; rank < 384; ++rank) old384[(size_t) old_order[(size_t) rank]] = 1;
    for (int rank = 0; rank < 384; ++rank) changed384 += old384[(size_t) lens_order[(size_t) rank]] ? 0 : 1;

    std::vector<double> sorted_survival = survival;
    std::sort(sorted_survival.begin(), sorted_survival.end());
    json out;
    out["layer"] = layer;
    out["expert"] = expert;
    out["lens_sample_size"] = lens_sample.n;
    out["lens_ridge"] = lens_fit.ridge;
    out["lens_train_relative_mse"] = lens_fit.train_relative_mse;
    out["survival_min"] = sorted_survival.front();
    out["survival_median"] = sorted_survival[sorted_survival.size() / 2];
    out["survival_max"] = sorted_survival.back();
    out["changed_survivors_at_384"] = changed384;
    out["old_enp"] = heldout_energy_report_decoded(
        gate, up, down, heldout_cert.remainder, old_order, &lens_fit);
    out["downstream_lens"] = heldout_energy_report_decoded(
        gate, up, down, heldout_cert.remainder, lens_order, &lens_fit);
    out["selection_method"] = "standard-enp-times-post-moe-to-final-256d-ridge-survival-v1";
    std::cout << std::setw(2) << out << '\n';
}

static void command_plan_expert_downstream_384(
        const std::string & model,
        const std::string & selection_imatrix_path,
        const std::string & lens_imatrix_path,
        const std::string & output,
        int n_threads) {
    static_assert(EXPERT_WIDTH_DST == 384 || EXPERT_WIDTH_DST == 256, "downstream planner requires expert width 384 or 256");
    const std::string checkpoint_path = output + ".checkpoint.json";
    advisory_file_lock checkpoint_lock(checkpoint_path + ".lock");
    source_gguf src(model);
    require_source_contract(src);
    common_imatrix selection_imat;
    common_imatrix lens_imat;
    if (!common_imatrix_load(selection_imatrix_path, selection_imat)) {
        throw std::runtime_error("failed to load downstream-384 selection imatrix");
    }
    if (!common_imatrix_load(lens_imatrix_path, lens_imat)) {
        throw std::runtime_error("failed to load downstream-384 lens imatrix");
    }
    json calibration = validate_expert_only_calibration_contract(selection_imat);
    const std::string tool_sha = current_executable_sha256();
    const json source_identity = file_identity(model);
    const json imatrix_identity = file_identity(selection_imatrix_path);
    const json lens_identity = file_identity(lens_imatrix_path);

    json plan;
    {
        std::ifstream checkpoint_in(checkpoint_path);
        if (checkpoint_in.good()) {
            checkpoint_in.close();
            plan = read_json(checkpoint_path);
            if (plan.value("checkpoint_format", std::string()) != "qwen35-downstream-384-checkpoint-v1" ||
                plan.value("format", std::string()) != "qwen35-combined-prune-v2" ||
                plan.value("tool_exe_sha256", std::string()) != tool_sha ||
                plan.value("expert_selection", std::string()) !=
                    "standard-enp-times-post-moe-to-final-256d-ridge-survival-v1" ||
                plan.value("expert_only", false) != true) {
                throw std::runtime_error("downstream-384 plan checkpoint input/options mismatch");
            }
            require_file_identity(plan.at("source_file_identity"), model, "source");
            require_file_identity(plan.at("imatrix_file_identity"), selection_imatrix_path, "imatrix");
            require_file_identity(plan.at("downstream_lens_file_identity"), lens_imatrix_path, "downstream lens");
            validate_partial_expert_plan(plan.at("experts"));
            if (plan.at("experts").size() % N_EXPERT != 0) {
                throw std::runtime_error("downstream-384 checkpoint is not layer-aligned");
            }
            std::cerr << "resuming downstream-384 plan at expert record " << plan.at("experts").size()
                      << "/" << (N_LAYER_ALL * N_EXPERT) << "\n";
        } else {
            plan["format"] = "qwen35-combined-prune-v2";
            plan["source_sha256"] = sha256_file(model);
            plan["imatrix_sha256"] = sha256_file(selection_imatrix_path);
            plan["downstream_lens_sha256"] = sha256_file(lens_imatrix_path);
            plan["source_file_identity"] = source_identity;
            plan["imatrix_file_identity"] = imatrix_identity;
            plan["downstream_lens_file_identity"] = lens_identity;
            plan["tool_exe_sha256"] = tool_sha;
            plan["llama_cpp_commit"] = llama_commit();
            plan["llama_cpp_build_info"] = llama_build_info();
            plan["compiler"] = llama_compiler();
            plan["build_target"] = llama_build_target();
            plan["cxx_flags"] = QWEN35_PRUNE_CXX_FLAGS;
            plan["targets"] = {
                {"expert_width", EXPERT_WIDTH_DST},
                {"expert_encoder", "source-q4km-gate-up-q4_0-down-ablation"},
                {"gdn_dim", GDN_DIM_SRC},
                {"vocab_size", 248320},
            };
            plan["expert_selection"] = "standard-enp-times-post-moe-to-final-256d-ridge-survival-v1";
            plan["plan_q2_robustness"] = "downstream-lens-384-single-pass-v1";
            plan["allow_uncovered_experts"] = false;
            plan["expert_only"] = true;
            calibration["downstream_lens_sha256"] = plan["downstream_lens_sha256"];
            calibration["downstream_lens_dim"] = ENP_DOWNSTREAM_LENS_DIM;
            calibration["downstream_selection_coreset_points"] = 1024;
            calibration["downstream_lens_method"] = "post-moe-to-final-feature-hash-paired-reservoir-v1";
            plan["calibration"] = calibration;
            plan["experts"] = json::array();
        }
    }

    n_threads = std::max(1, std::min(n_threads, N_EXPERT));
    size_t completed = plan.at("experts").size();
    for (int layer = (int) (completed / N_EXPERT); layer < N_LAYER_MAIN; ++layer) {
        if (completed != (size_t) layer * N_EXPERT) {
            throw std::runtime_error("downstream-384 checkpoint layer alignment mismatch");
        }
        const auto enp_full_sample = load_standard_enp_sample(selection_imat, layer);
        const auto enp_sample = coarsen_standard_enp_sample(enp_full_sample, std::min(1024, enp_full_sample.n));
        const auto lens_sample = load_enp_downstream_lens_sample(lens_imat, layer);
        const auto lens_fit = fit_enp_downstream_lens(lens_sample);
        std::array<json, N_EXPERT> records;
        std::atomic<int> next_expert {0};
        std::atomic<bool> failed {false};
        std::exception_ptr error;
        std::mutex error_mutex;
        auto worker = [&]() {
            try {
                while (!failed.load(std::memory_order_relaxed)) {
                    const int expert = next_expert.fetch_add(1);
                    if (expert >= N_EXPERT) break;
                    records[(size_t) expert] = make_downstream_lens_expert_record(
                        src, enp_sample, lens_fit, layer, expert);
                }
            } catch (...) {
                failed.store(true, std::memory_order_relaxed);
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!error) error = std::current_exception();
            }
        };
        std::vector<std::thread> workers;
        workers.reserve((size_t) n_threads);
        for (int i = 0; i < n_threads; ++i) workers.emplace_back(worker);
        for (auto & th : workers) th.join();
        if (error) std::rethrow_exception(error);
        for (int expert = 0; expert < N_EXPERT; ++expert) {
            plan["experts"].push_back(std::move(records[(size_t) expert]));
        }
        completed = plan.at("experts").size();
        plan["checkpoint_format"] = "qwen35-downstream-384-checkpoint-v1";
        plan["checkpoint_completed_experts"] = completed;
        write_json(checkpoint_path, plan);
        std::cerr << "downstream-384 plan: blk." << layer << " complete; total="
                  << completed << "/" << (N_LAYER_ALL * N_EXPERT)
                  << " lens_mse=" << lens_fit.train_relative_mse << "\n";
    }
    if (plan.at("experts").size() == (size_t) N_LAYER_MAIN * N_EXPERT) {
        for (int expert = 0; expert < N_EXPERT; ++expert) {
            plan["experts"].push_back(make_mtp_fallback_expert_record(expert));
        }
    }
    plan.erase("checkpoint_format");
    plan.erase("checkpoint_completed_experts");
    plan["gdn"] = make_keep_source_gdn_plan();
    plan["vocab"] = make_keep_source_vocab_plan();
    verify_plan_json(plan);
    require_file_identity(plan.at("source_file_identity"), model, "source");
    require_file_identity(plan.at("imatrix_file_identity"), selection_imatrix_path, "imatrix");
    require_file_identity(plan.at("downstream_lens_file_identity"), lens_imatrix_path, "downstream lens");
    if (sha256_file(model) != plan.at("source_sha256").get<std::string>() ||
        sha256_file(selection_imatrix_path) != plan.at("imatrix_sha256").get<std::string>() ||
        sha256_file(lens_imatrix_path) != plan.at("downstream_lens_sha256").get<std::string>()) {
        throw std::runtime_error("downstream-384 immutable input changed during planning");
    }
    write_json(output, plan);
    std::remove(checkpoint_path.c_str());
    std::cout << "wrote verified downstream-aware 384 plan: " << output << "\n";
}

static std::vector<float> expert_replacement_mapped_down_columns(
        const decoded_expert_slice & down,
        const enp_downstream_lens_fit & lens) {
    const int d = ENP_DOWNSTREAM_LENS_DIM;
    if (down.ne0 != EXPERT_WIDTH_DST || down.ne1 != HIDDEN || lens.map.size() != (size_t) d * d) {
        throw std::runtime_error("bad expert-replacement down/lens dimensions");
    }
    std::vector<float> sketch((size_t) EXPERT_WIDTH_DST * d, 0.0f);
    for (int i = 0; i < HIDDEN; ++i) {
        const uint64_t h = enp_downstream_mix64(ENP_DOWNSTREAM_LENS_SKETCH_SEED ^ (uint64_t) i);
        const int bucket = (int) (h & (d - 1));
        const float sign = ((h >> 8) & 1u) ? 1.0f : -1.0f;
        const float * row = down.row(i);
        for (int j = 0; j < EXPERT_WIDTH_DST; ++j) {
            sketch[(size_t) j * d + bucket] += sign * row[j];
        }
    }

    std::vector<float> mapped((size_t) EXPERT_WIDTH_DST * d, 0.0f);
    for (int j = 0; j < EXPERT_WIDTH_DST; ++j) {
        const float * s = sketch.data() + (size_t) j * d;
        float * dst = mapped.data() + (size_t) j * d;
        for (int b = 0; b < d; ++b) {
            const double sb = s[b];
            const double * row = lens.map.data() + (size_t) b * d;
            for (int k = 0; k < d; ++k) dst[k] += (float) (sb * row[k]);
        }
    }
    return mapped;
}

static expert_replacement_problem build_expert_replacement_problem(
        const source_gguf & src,
        const expert_replacement_sample & sample,
        const enp_downstream_lens_fit & lens,
        int layer,
        int n_threads) {
    expert_replacement_problem p;
    p.n_expert = N_EXPERT;
    p.n_keep = N_EXPERT_DST;
    p.top_k = 8;
    p.candidate_depth = EXPERT_REPLACEMENT_CANDIDATES;
    p.n_tokens = sample.n;
    p.output_dim = ENP_DOWNSTREAM_LENS_DIM;
    if (sample.n <= 0 || sample.x.size() != (size_t) sample.n * HIDDEN) {
        throw std::runtime_error("empty expert-replacement calibration sample");
    }

    const std::string prefix = "blk." + std::to_string(layer) + ".";
    const auto router = decode_matrix_rows(src, prefix + "ffn_gate_inp.weight", HIDDEN, N_EXPERT);
    p.candidate_ids.resize((size_t) sample.n * p.candidate_depth);
    p.candidate_logits.resize((size_t) sample.n * p.candidate_depth);
    p.outputs.resize((size_t) sample.n * p.candidate_depth * p.output_dim);

    // Router Top-72 is sufficient and exact: Qwen3.5 uses monotone softmax before Top-K, so
    // sorting logits reproduces the candidate ordering.  The final optimizer
    // removes exactly 64 experts globally, hence at least 8 of the original
    // Top-72 must survive and the post-prune Top-8 is guaranteed to remain
    // inside this recorded set for every possible 192-expert survivor set.
    for (int t = 0; t < sample.n; ++t) {
        const float * x = sample.x.data() + (size_t) t * HIDDEN;
        std::vector<std::pair<float,int>> ranking((size_t) N_EXPERT);
        for (int e = 0; e < N_EXPERT; ++e) {
            const float * w = router.data() + (size_t) e * HIDDEN;
            float dot = 0.0f;
            #pragma omp simd reduction(+:dot)
            for (int i = 0; i < HIDDEN; ++i) dot += w[i] * x[i];
            ranking[(size_t) e] = {dot, e};
        }
        std::partial_sort(ranking.begin(), ranking.begin() + p.candidate_depth, ranking.end(),
            [](const auto & a, const auto & b) {
                if (a.first != b.first) return a.first > b.first;
                return a.second < b.second;
            });
        for (int r = 0; r < p.candidate_depth; ++r) {
            const size_t q = (size_t) t * p.candidate_depth + r;
            p.candidate_logits[q] = ranking[(size_t) r].first;
            p.candidate_ids[q] = ranking[(size_t) r].second;
        }
    }

    std::array<std::vector<std::pair<int,int>>, N_EXPERT> work;
    for (int t = 0; t < sample.n; ++t) {
        for (int r = 0; r < p.candidate_depth; ++r) {
            const int e = p.candidate_ids[(size_t) t * p.candidate_depth + r];
            work[(size_t) e].push_back({t, r});
        }
    }

    n_threads = std::max(1, std::min(n_threads, N_EXPERT));
    std::atomic<int> next_expert {0};
    std::atomic<bool> failed {false};
    std::exception_ptr error;
    std::mutex error_mutex;
    auto worker = [&]() {
        try {
            std::array<float, EXPERT_WIDTH_DST> m {};
            while (!failed.load(std::memory_order_relaxed)) {
                const int expert = next_expert.fetch_add(1);
                if (expert >= N_EXPERT) break;
                if (work[(size_t) expert].empty()) continue;
                const auto gate = decode_expert_slice(src, prefix + "ffn_gate_exps.weight", expert, HIDDEN, EXPERT_WIDTH_DST);
                const auto up   = decode_expert_slice(src, prefix + "ffn_up_exps.weight", expert, HIDDEN, EXPERT_WIDTH_DST);
                const auto down = decode_expert_slice(src, prefix + "ffn_down_exps.weight", expert, EXPERT_WIDTH_DST, HIDDEN);
                const auto mapped_down = expert_replacement_mapped_down_columns(down, lens);

                for (const auto & [token, rank] : work[(size_t) expert]) {
                    const float * x = sample.x.data() + (size_t) token * HIDDEN;
                    for (int j = 0; j < EXPERT_WIDTH_DST; ++j) {
                        const float * wg = gate.row(j);
                        const float * wu = up.row(j);
                        float g = 0.0f, u = 0.0f;
                        #pragma omp simd reduction(+:g,u)
                        for (int i = 0; i < HIDDEN; ++i) {
                            g += wg[i] * x[i];
                            u += wu[i] * x[i];
                        }
                        m[(size_t) j] = standard_enp_silu(g) * u;
                    }
                    float * dst = p.outputs.data() +
                        ((size_t) token * p.candidate_depth + rank) * p.output_dim;
                    std::fill(dst, dst + p.output_dim, 0.0f);
                    for (int j = 0; j < EXPERT_WIDTH_DST; ++j) {
                        const float mj = m[(size_t) j];
                        const float * col = mapped_down.data() + (size_t) j * p.output_dim;
                        #pragma omp simd
                        for (int k = 0; k < p.output_dim; ++k) dst[k] += mj * col[k];
                    }
                }
            }
        } catch (...) {
            failed.store(true, std::memory_order_relaxed);
            std::lock_guard<std::mutex> lock(error_mutex);
            if (!error) error = std::current_exception();
        }
    };
    std::vector<std::thread> workers;
    workers.reserve((size_t) n_threads);
    for (int i = 0; i < n_threads; ++i) workers.emplace_back(worker);
    for (auto & th : workers) th.join();
    if (error) std::rethrow_exception(error);
    return p;
}

static void verify_expert_replacement_plan(const json & plan) {
    if (plan.value("format", std::string()) != "qwen35-expert-replacement-v1" ||
        plan.value("source_expert_count", 0) != N_EXPERT ||
        plan.value("target_expert_count", 0) != N_EXPERT_DST ||
        plan.value("expert_used_count", 0) != 8 ||
        plan.value("expert_width", 0) != EXPERT_WIDTH_DST ||
        plan.value("candidate_depth", 0) != EXPERT_REPLACEMENT_CANDIDATES ||
        plan.value("mtp_removed", false) != true ||
        plan.value("target_block_count", 0) != N_LAYER_MAIN) {
        throw std::runtime_error("bad expert-replacement plan header");
    }
    const auto & layers = plan.at("layers");
    if (!layers.is_array() || layers.size() != N_LAYER_MAIN) throw std::runtime_error("expert-replacement layer count mismatch");
    for (int layer = 0; layer < N_LAYER_MAIN; ++layer) {
        const auto & rec = layers[(size_t) layer];
        if (rec.value("layer", -1) != layer) throw std::runtime_error("expert-replacement layer order mismatch");
        const auto mapping = rec.at("output_to_input").get<std::vector<int>>();
        if (mapping.size() != N_EXPERT_DST || !std::is_sorted(mapping.begin(), mapping.end())) {
            throw std::runtime_error("expert-replacement mapping size/order mismatch");
        }
        std::set<int> unique(mapping.begin(), mapping.end());
        if (unique.size() != mapping.size() || *unique.begin() < 0 || *unique.rbegin() >= N_EXPERT) {
            throw std::runtime_error("expert-replacement mapping invalid");
        }
        if (rec.value("minimum_candidate_coverage", 0) < 8) {
            throw std::runtime_error("expert-replacement plan violates Top-8 candidate coverage");
        }
    }
}

static void command_plan_expert_replacement_192(
        const std::string & model,
        const std::string & calibration_path,
        const std::string & output,
        int n_threads) {
    source_gguf src(model);
    require_expert_count_source_contract(src);
    common_imatrix imat;
    if (!common_imatrix_load(calibration_path, imat)) throw std::runtime_error("failed to load expert-replacement calibration");
    const std::string checkpoint_path = output + ".checkpoint.json";
    advisory_file_lock checkpoint_lock(checkpoint_path + ".lock");
    const std::string tool_sha = current_executable_sha256();
    const std::string source_sha = sha256_file(model);
    const std::string calibration_sha = sha256_file(calibration_path);

    json plan;
    int first_layer = 0;
    {
        std::ifstream in(checkpoint_path);
        if (in.good()) {
            in.close();
            plan = read_json(checkpoint_path);
            if (plan.value("checkpoint_format", std::string()) != "qwen35-expert-replacement-checkpoint-v1" ||
                plan.value("tool_exe_sha256", std::string()) != tool_sha ||
                plan.value("source_sha256", std::string()) != source_sha ||
                plan.value("calibration_sha256", std::string()) != calibration_sha) {
                throw std::runtime_error("expert-replacement checkpoint input/options mismatch");
            }
            first_layer = (int) plan.at("layers").size();
            if (first_layer < 0 || first_layer > N_LAYER_MAIN) throw std::runtime_error("bad expert-replacement checkpoint layer count");
            std::cerr << "resuming expert-replacement plan at layer " << first_layer << "/" << N_LAYER_MAIN << "\n";
        } else {
            plan["format"] = "qwen35-expert-replacement-v1";
            plan["checkpoint_format"] = "qwen35-expert-replacement-checkpoint-v1";
            plan["source_sha256"] = source_sha;
            plan["calibration_sha256"] = calibration_sha;
            plan["source_file_identity"] = file_identity(model);
            plan["calibration_file_identity"] = file_identity(calibration_path);
            plan["tool_exe_sha256"] = tool_sha;
            plan["llama_cpp_commit"] = llama_commit();
            plan["source_expert_count"] = N_EXPERT;
            plan["target_expert_count"] = N_EXPERT_DST;
            plan["expert_used_count"] = 8;
            plan["expert_width"] = EXPERT_WIDTH_DST;
            plan["candidate_depth"] = EXPERT_REPLACEMENT_CANDIDATES;
            plan["mtp_removed"] = true;
            plan["target_block_count"] = N_LAYER_MAIN;
            plan["selection_method"] = "downstream-replacement-exact-top72-reverse-greedy-plus-1swap-v2";
            plan["objective"] = "mean-absolute-squared-256d-final-hidden-lens-perturbation";
            plan["layers"] = json::array();
        }
    }

    for (int layer = first_layer; layer < N_LAYER_MAIN; ++layer) {
        const auto sample = load_expert_replacement_sample(imat, layer);
        const auto lens_sample = load_enp_downstream_lens_sample(imat, layer);
        const auto lens = fit_enp_downstream_lens(lens_sample);
        const auto problem = build_expert_replacement_problem(src, sample, lens, layer, n_threads);
        const auto result = optimize_expert_replacement(problem, 16, 1e-10);
        json rec;
        rec["layer"] = layer;
        rec["output_to_input"] = result.survivors;
        rec["removed"] = result.removed;
        rec["baseline_top8_count"] = result.baseline_topk_count;
        rec["objective"] = result.objective;
        rec["objective_mean"] = result.objective_mean;
        rec["minimum_candidate_coverage"] = result.minimum_candidate_coverage;
        rec["greedy_steps"] = result.greedy_steps;
        rec["swap_steps"] = result.swap_steps;
        rec["sample_size"] = sample.n;
        rec["population"] = sample.population;
        rec["lens_train_relative_mse"] = lens.train_relative_mse;
        rec["selection_method"] = plan["selection_method"];
        plan["layers"].push_back(std::move(rec));
        write_json(checkpoint_path, plan);
        std::cerr << "expert-replacement: blk." << layer
                  << " keep=" << result.survivors.size()
                  << " objective_mean=" << result.objective_mean
                  << " coverage_min=" << result.minimum_candidate_coverage
                  << " swaps=" << result.swap_steps << "\n";
    }

    plan.erase("checkpoint_format");
    verify_expert_replacement_plan(plan);
    require_file_identity(plan.at("source_file_identity"), model, "expert-replacement source");
    require_file_identity(plan.at("calibration_file_identity"), calibration_path, "expert-replacement calibration");
    if (sha256_file(model) != source_sha || sha256_file(calibration_path) != calibration_sha) {
        throw std::runtime_error("expert-replacement immutable input changed during planning");
    }
    write_json(output, plan);
    std::remove(checkpoint_path.c_str());
    std::cout << "wrote replacement-aware 192-expert plan: " << output << "\n";
}

struct q2_manifest_stats {
    uint64_t blocks = 0;
    uint64_t negative_d = 0;
    uint64_t zero_weight_fallbacks = 0;
    std::array<uint64_t, 4> code_hist {};
    long double weighted_sse = 0.0;
    long double unweighted_sse = 0.0;
    long double positive_weighted_sse = 0.0;
    long double stock_weighted_sse = 0.0;

    void add(
            const q2_opt_block & b,
            const q2_opt_metrics & m,
            const q2_opt_metrics & positive,
            const q2_opt_metrics & stock,
            bool zero_weight) {
        ++blocks;
        const float d = ggml_fp16_to_fp32(b.d);
        if (std::signbit(d) && d != 0.0f) ++negative_d;
        if (zero_weight) ++zero_weight_fallbacks;
        weighted_sse += m.weighted_sse;
        unweighted_sse += m.unweighted_sse;
        positive_weighted_sse += positive.weighted_sse;
        stock_weighted_sse += stock.weighted_sse;
        for (int i = 0; i < 64; ++i) ++code_hist[(b.qs[i/4] >> (2*(i%4))) & 3u];
    }

    void merge(const q2_manifest_stats & other) {
        blocks += other.blocks;
        negative_d += other.negative_d;
        zero_weight_fallbacks += other.zero_weight_fallbacks;
        for (int i = 0; i < 4; ++i) code_hist[i] += other.code_hist[i];
        weighted_sse += other.weighted_sse;
        unweighted_sse += other.unweighted_sse;
        positive_weighted_sse += other.positive_weighted_sse;
        stock_weighted_sse += other.stock_weighted_sse;
    }
};

static json q2_stats_to_json(const q2_manifest_stats & s) {
    return {
        {"blocks", s.blocks},
        {"negative_d", s.negative_d},
        {"zero_weight_fallbacks", s.zero_weight_fallbacks},
        {"code_hist", s.code_hist},
        {"weighted_sse", (double) s.weighted_sse},
        {"unweighted_sse", (double) s.unweighted_sse},
        {"positive_weighted_sse", (double) s.positive_weighted_sse},
        {"stock_weighted_sse", (double) s.stock_weighted_sse},
    };
}

static q2_manifest_stats q2_stats_from_json(const json & j) {
    q2_manifest_stats s;
    if (j.is_null()) return s;
    s.blocks = j.value("blocks", uint64_t(0));
    s.negative_d = j.value("negative_d", uint64_t(0));
    s.zero_weight_fallbacks = j.value("zero_weight_fallbacks", uint64_t(0));
    if (j.contains("code_hist")) s.code_hist = j.at("code_hist").get<std::array<uint64_t,4>>();
    s.weighted_sse = j.value("weighted_sse", 0.0);
    s.unweighted_sse = j.value("unweighted_sse", 0.0);
    s.positive_weighted_sse = j.value("positive_weighted_sse", 0.0);
    s.stock_weighted_sse = j.value("stock_weighted_sse", 0.0);
    return s;
}

static int parse_block_id(const std::string & name) {
    if (!starts_with(name, "blk.")) return -1;
    const char * p = name.c_str() + 4;
    char * end = nullptr;
    const long v = std::strtol(p, &end, 10);
    if (end == p || *end != '.') return -1;
    return (int) v;
}

static bool is_expert_tensor(const std::string & name) {
    return ends_with(name, "ffn_gate_exps.weight") ||
           ends_with(name, "ffn_up_exps.weight") ||
           ends_with(name, "ffn_down_exps.weight");
}

static bool is_gdn_target(const std::string & name) {
    return ends_with(name, "attn_qkv.weight") ||
           ends_with(name, "attn_gate.weight") ||
           ends_with(name, "ssm_conv1d.weight") ||
           ends_with(name, "ssm_norm.weight") ||
           ends_with(name, "ssm_out.weight");
}

static bool is_recurrent_layer(int layer) {
    return std::find(RECURRENT_LAYERS.begin(), RECURRENT_LAYERS.end(), layer) != RECURRENT_LAYERS.end();
}

static bool is_vocab_tensor(const std::string & name, const ggml_tensor * t) {
    if (name == "token_embd.weight" || name == "output.weight") return true;
    return name.find("nextn") != std::string::npos && ggml_n_dims(t) >= 2 && t->ne[1] == 248320;
}

enum class tensor_policy { expert_q2, gdn, vocab, copy };

static tensor_policy classify_tensor(const std::string & name, const ggml_tensor * t) {
    if (is_expert_tensor(name)) return tensor_policy::expert_q2;
    const int layer = parse_block_id(name);
    if (layer >= 0 && is_recurrent_layer(layer) && is_gdn_target(name)) return tensor_policy::gdn;
    if (is_vocab_tensor(name, t)) return tensor_policy::vocab;
    return tensor_policy::copy;
}

struct plan_index {
    std::array<std::array<std::vector<int>, N_EXPERT>, N_LAYER_ALL> expert;
    std::unordered_map<int, json> gdn;
    std::vector<int> vocab;
};

static plan_index index_plan(const json & plan) {
    verify_plan_json(plan);
    plan_index idx;
    for (const auto & r : plan.at("experts")) {
        idx.expert[(size_t) r.at("layer").get<int>()][(size_t) r.at("expert").get<int>()] =
            r.at("output_to_input").get<std::vector<int>>();
    }
    for (const auto & r : plan.at("gdn")) idx.gdn.emplace(r.at("layer").get<int>(), r);
    idx.vocab = plan.at("vocab").at("output_to_input").get<std::vector<int>>();
    return idx;
}

static std::vector<int> gdn_v_positions(const json & rec) {
    return rec.at("v_indices").get<std::vector<int>>();
}

static std::vector<int> gdn_v_channel_mapping(const json & rec) {
    const auto pos = gdn_v_positions(rec);
    std::vector<int> rows;
    rows.reserve(GDN_V_HEADS * GDN_DIM_DST);
    for (int h = 0; h < GDN_V_HEADS; ++h) for (int p : pos) rows.push_back(h * GDN_DIM_SRC + p);
    return rows;
}

static std::vector<int> gdn_qk_channel_mapping(const json & rec) {
    std::vector<int> q;
    q.reserve(GDN_QK_HEADS * GDN_DIM_DST);
    const auto & heads = rec.at("qk_indices_by_head");
    for (int h = 0; h < GDN_QK_HEADS; ++h) {
        const auto coords = heads[(size_t) h].get<std::vector<int>>();
        for (int p : coords) q.push_back(h * GDN_DIM_SRC + p);
    }
    return q;
}

static std::vector<int> gdn_qkv_mapping(const json & rec) {
    const auto q = gdn_qk_channel_mapping(rec);
    const auto v = gdn_v_channel_mapping(rec);
    std::vector<int> out;
    out.reserve(4096);
    out.insert(out.end(), q.begin(), q.end());
    for (int x : q) out.push_back(2048 + x);
    for (int x : v) out.push_back(4096 + x);
    return out;
}

static void set_special_id(gguf_context * out, const gguf_context * src, const std::string & key, uint64_t value) {
    const int64_t id = gguf_find_key(src, key.c_str());
    if (id < 0) return;
    switch (gguf_get_kv_type(src, id)) {
        case GGUF_TYPE_UINT32: gguf_set_val_u32(out, key.c_str(), (uint32_t) value); break;
        case GGUF_TYPE_INT32:  gguf_set_val_i32(out, key.c_str(), (int32_t) value); break;
        case GGUF_TYPE_UINT64: gguf_set_val_u64(out, key.c_str(), value); break;
        case GGUF_TYPE_INT64:  gguf_set_val_i64(out, key.c_str(), (int64_t) value); break;
        default: throw std::runtime_error("unsupported special token metadata type: " + key);
    }
}

static size_t rebuild_tokenizer_metadata(
        gguf_context * out,
        const source_gguf & src,
        const json & plan,
        const std::vector<int> & vocab) {
    const gguf_context * meta = src.meta();
    const int64_t token_key = gguf_find_key(meta, "tokenizer.ggml.tokens");
    const int64_t type_key  = gguf_find_key(meta, "tokenizer.ggml.token_type");
    const int64_t score_key = gguf_find_key(meta, "tokenizer.ggml.scores");
    const int64_t merge_key = gguf_find_key(meta, "tokenizer.ggml.merges");
    if (token_key < 0 || type_key < 0 || merge_key < 0) throw std::runtime_error("tokenizer metadata incomplete");

    std::vector<std::string> token_store;
    std::vector<const char *> tokens;
    std::vector<int32_t> types;
    std::vector<float> scores;
    token_store.reserve(vocab.size());
    tokens.reserve(vocab.size());
    types.reserve(vocab.size());
    if (score_key >= 0) scores.reserve(vocab.size());

    const int32_t * src_types = (const int32_t *) gguf_get_arr_data(meta, type_key);
    const float * src_scores = score_key >= 0 ? (const float *) gguf_get_arr_data(meta, score_key) : nullptr;
    std::unordered_set<std::string> kept_text;
    kept_text.reserve(vocab.size() * 2);
    for (int old : vocab) {
        token_store.emplace_back(gguf_get_arr_str(meta, token_key, (size_t) old));
        types.push_back(src_types[old]);
        if (src_scores) scores.push_back(src_scores[old]);
        kept_text.insert(token_store.back());
    }
    for (auto & s : token_store) tokens.push_back(s.c_str());
    gguf_set_arr_str(out, "tokenizer.ggml.tokens", tokens.data(), tokens.size());
    gguf_set_arr_data(out, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types.data(), types.size());
    if (src_scores) gguf_set_arr_data(out, "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32, scores.data(), scores.size());

    std::vector<std::string> merge_store;
    const size_t n_merges = gguf_get_arr_n(meta, merge_key);
    merge_store.reserve(std::min<size_t>(n_merges, vocab.size() * 2));
    for (size_t i = 0; i < n_merges; ++i) {
        const std::string m = gguf_get_arr_str(meta, merge_key, i);
        const size_t sp = m.find(' ');
        if (sp == std::string::npos) continue;
        const std::string a = m.substr(0, sp);
        const std::string b = m.substr(sp + 1);
        if (kept_text.count(a) && kept_text.count(b) && kept_text.count(a + b)) merge_store.push_back(m);
    }
    std::vector<const char *> merges;
    merges.reserve(merge_store.size());
    for (auto & s : merge_store) merges.push_back(s.c_str());
    gguf_set_arr_str(out, "tokenizer.ggml.merges", merges.data(), merges.size());

    const json & special = plan.at("vocab").at("special_id_remap");
    for (auto it = special.begin(); it != special.end(); ++it) set_special_id(out, meta, it.key(), it.value().get<uint64_t>());

    const int64_t vocab_size_key = gguf_find_key(meta, "qwen35moe.vocab_size");
    if (vocab_size_key >= 0) gguf_set_val_u32(out, "qwen35moe.vocab_size", VOCAB_DST);
    return merge_store.size();
}

static void command_vocab_fixture(
        const std::string & model,
        const std::string & imatrix_path,
        const std::string & output) {
    source_gguf src(model);
    require_source_contract(src);
    common_imatrix imat;
    if (!common_imatrix_load(imatrix_path, imat)) throw std::runtime_error("failed to load imatrix");
    const json vocab_plan = make_vocab_plan(src, imat);
    json plan;
    plan["vocab"] = vocab_plan;

    gguf_ptr out(gguf_init_empty());
    if (!out) throw std::runtime_error("failed to create vocab fixture GGUF");
    gguf_set_kv(out.get(), src.meta());
    const auto mapping = vocab_plan.at("output_to_input").get<std::vector<int>>();
    const size_t merges = rebuild_tokenizer_metadata(out.get(), src, plan, mapping);
    if (!gguf_write_to_file(out.get(), output.c_str(), true)) {
        throw std::runtime_error("failed to write vocab-only fixture: " + output);
    }
    write_json(output + ".vocab.json", vocab_plan);
    std::cout << "wrote vocab-only fixture: " << output << "\n"
              << "tokens: " << mapping.size() << " merges: " << merges
              << " dependencies: " << vocab_plan.value("merge_dependency_tokens_added", size_t(0)) << "\n";
}

static ggml_tensor * make_output_descriptor(
        ggml_context * ctx,
        const ggml_tensor * src_t,
        const std::string & name,
        tensor_policy policy,
        bool expert_source_quant = false) {
    ggml_type type = src_t->type;
    int n_dims = ggml_n_dims(src_t);
    std::array<int64_t, GGML_MAX_DIMS> ne {1,1,1,1};
    for (int i = 0; i < GGML_MAX_DIMS; ++i) ne[i] = src_t->ne[i];

    if (policy == tensor_policy::expert_q2) {
        if (expert_source_quant) {
            const bool down = ends_with(name, "ffn_down_exps.weight");
            type = down && EXPERT_WIDTH_DST % ggml_blck_size(src_t->type) != 0
                ? GGML_TYPE_Q4_0
                : src_t->type;
        } else {
            type = GGML_TYPE_Q2_0;
        }
        if (ends_with(name, "ffn_down_exps.weight")) ne[0] = EXPERT_WIDTH_DST;
        else ne[1] = EXPERT_WIDTH_DST;
    } else if (policy == tensor_policy::vocab) {
        ne[1] = VOCAB_DST;
    } else if (policy == tensor_policy::gdn) {
        if (ends_with(name, "attn_qkv.weight")) ne[1] = 4096;
        else if (ends_with(name, "attn_gate.weight")) ne[1] = 2048;
        else if (ends_with(name, "ssm_conv1d.weight")) ne[1] = 4096;
        else if (ends_with(name, "ssm_norm.weight")) ne[0] = GDN_DIM_DST;
        else if (ends_with(name, "ssm_out.weight")) ne[0] = 2048;
    }

    ggml_tensor * out_t = ggml_new_tensor(ctx, type, n_dims, ne.data());
    ggml_set_name(out_t, name.c_str());
    return out_t;
}

class output_writer {
public:
    output_writer(const std::string & path, gguf_context * meta, bool initialize) : path_(path), meta_(meta) {
        if (initialize && !gguf_write_to_file(meta, path.c_str(), true)) throw std::runtime_error("failed to write GGUF metadata");
        file_ = std::fopen(path.c_str(), "r+b");
        if (!file_) throw std::runtime_error("failed to reopen output GGUF");
        // For a freshly constructed GGUF context, ctx->offset is not populated
        // the way it is after parsing a file. The public writer API explicitly
        // defines gguf_get_meta_size() as header + KV + tensor-info + alignment
        // padding, i.e. the byte offset at which tensor payloads begin.
        base_ = gguf_get_meta_size(meta_);
        if (initialize) {
            const int64_t n_tensors = gguf_get_n_tensors(meta_);
            if (n_tensors <= 0) throw std::runtime_error("cannot preallocate GGUF with no tensors");
            const int64_t last = n_tensors - 1;
            const uint64_t final_size = (uint64_t) base_ + gguf_get_tensor_offset(meta_, last) + gguf_get_tensor_size(meta_, last);
            const int rc = ::posix_fallocate(fileno(file_), 0, (off_t) final_size);
            if (rc != 0) {
                throw std::runtime_error(std::string("posix_fallocate failed: ") + std::strerror(rc));
            }
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

    void finish() {
        if (std::fflush(file_) != 0) throw std::runtime_error("fflush failed");
        if (::fsync(fileno(file_)) != 0) throw std::runtime_error("fsync failed");
    }

    void sync_data() {
        if (std::fflush(file_) != 0) throw std::runtime_error("fflush failed");
        if (::fdatasync(fileno(file_)) != 0) throw std::runtime_error("fdatasync failed");
    }

    std::string tensor_sha256(int64_t tensor_id) const {
        if (tensor_id < 0 || tensor_id >= gguf_get_n_tensors(meta_)) {
            throw std::runtime_error("output tensor hash id out of range");
        }
        const uint64_t offset = (uint64_t) base_ + gguf_get_tensor_offset(meta_, tensor_id);
        const uint64_t size = gguf_get_tensor_size(meta_, tensor_id);
        return sha256_fd_range(fileno(file_), offset, size);
    }

private:
    std::string path_;
    gguf_context * meta_;
    FILE * file_ = nullptr;
    size_t base_ = 0;
};

static std::string copy_tensor_exact(
        const source_gguf & src, output_writer & out, int64_t id, const std::string & name) {
    const size_t size = src.tensor_size(name);
    std::vector<uint8_t> buf(1 << 20);
    sha256_t sha;
    sha256_init(&sha);
    for (size_t pos = 0; pos < size; pos += buf.size()) {
        const size_t n = std::min(buf.size(), size - pos);
        src.read_tensor_bytes(name, pos, buf.data(), n);
        sha256_update(&sha, buf.data(), n);
        out.write_tensor(id, pos, buf.data(), n);
    }
    unsigned char digest[SHA256_DIGEST_SIZE];
    sha256_final(&sha, digest);
    return hex_digest(digest);
}

static void copy_selected_rows(
        const source_gguf & src,
        output_writer & out,
        int64_t id,
        const std::string & name,
        const std::vector<int> & rows) {
    const ggml_tensor * t = src.tensor(name);
    const size_t row_size = ggml_row_size(t->type, t->ne[0]);
    std::vector<uint8_t> row(row_size);
    for (size_t i = 0; i < rows.size(); ++i) {
        const int r = rows[i];
        if (r < 0 || r >= t->ne[1]) throw std::runtime_error("selected row out of range: " + name);
        src.read_tensor_bytes(name, (size_t) r * row_size, row.data(), row_size);
        out.write_tensor(id, i * row_size, row.data(), row_size);
    }
}

static bool is_routed_expert_payload(const std::string & name) {
    return ends_with(name, "ffn_gate_exps.weight") ||
           ends_with(name, "ffn_up_exps.weight") ||
           ends_with(name, "ffn_down_exps.weight");
}

static bool is_routed_expert_router(const std::string & name) {
    return ends_with(name, "ffn_gate_inp.weight");
}

static bool is_mtp_tensor(const std::string & name) {
    return starts_with(name, "blk.40.");
}

static std::array<std::vector<int>, N_LAYER_ALL> index_expert_replacement_plan(const json & plan) {
    verify_expert_replacement_plan(plan);
    std::array<std::vector<int>, N_LAYER_ALL> out;
    for (const auto & rec : plan.at("layers")) {
        const int layer = rec.at("layer").get<int>();
        out[(size_t) layer] = rec.at("output_to_input").get<std::vector<int>>();
    }
    return out;
}

static ggml_tensor * make_expert_count_output_descriptor(
        ggml_context * ctx,
        const ggml_tensor * src_t,
        const std::string & name) {
    const int n_dims = ggml_n_dims(src_t);
    std::array<int64_t, GGML_MAX_DIMS> ne {1,1,1,1};
    for (int i = 0; i < GGML_MAX_DIMS; ++i) ne[i] = src_t->ne[i];
    if (is_routed_expert_payload(name)) {
        if (n_dims != 3 || ne[2] != N_EXPERT) throw std::runtime_error("bad routed-expert payload descriptor: " + name);
        ne[2] = N_EXPERT_DST;
    } else if (is_routed_expert_router(name)) {
        if (n_dims != 2 || ne[0] != HIDDEN || ne[1] != N_EXPERT) throw std::runtime_error("bad routed-expert router descriptor: " + name);
        ne[1] = N_EXPERT_DST;
    }
    ggml_tensor * out_t = ggml_new_tensor(ctx, src_t->type, n_dims, ne.data());
    ggml_set_name(out_t, name.c_str());
    return out_t;
}

static void copy_selected_expert_slices(
        const source_gguf & src,
        output_writer & out,
        int64_t id,
        const std::string & name,
        const std::vector<int> & selected) {
    const ggml_tensor * t = src.tensor(name);
    if (ggml_n_dims(t) != 3 || t->ne[2] != N_EXPERT || selected.size() != N_EXPERT_DST) {
        throw std::runtime_error("bad expert-slice gather request: " + name);
    }
    const size_t slice_size = src.tensor_size(name) / N_EXPERT;
    if (slice_size * N_EXPERT != src.tensor_size(name)) throw std::runtime_error("expert tensor is not slice-aligned: " + name);
    std::vector<uint8_t> buf(slice_size);
    for (size_t dst = 0; dst < selected.size(); ++dst) {
        const int src_expert = selected[dst];
        if (src_expert < 0 || src_expert >= N_EXPERT) throw std::runtime_error("expert gather index out of range");
        src.read_tensor_bytes(name, (size_t) src_expert * slice_size, buf.data(), buf.size());
        out.write_tensor(id, dst * slice_size, buf.data(), buf.size());
    }
}

static bool byte_ranges_equal(
        const source_gguf & a,
        const std::string & aname,
        size_t aoff,
        const source_gguf & b,
        const std::string & bname,
        size_t boff,
        size_t size) {
    std::vector<uint8_t> abuf(1 << 20), bbuf(1 << 20);
    for (size_t pos = 0; pos < size; pos += abuf.size()) {
        const size_t n = std::min(abuf.size(), size - pos);
        a.read_tensor_bytes(aname, aoff + pos, abuf.data(), n);
        b.read_tensor_bytes(bname, boff + pos, bbuf.data(), n);
        if (std::memcmp(abuf.data(), bbuf.data(), n) != 0) return false;
    }
    return true;
}

static void command_apply_expert_replacement_192(
        const std::string & model,
        const std::string & plan_path,
        const std::string & output) {
    if (!ends_with(output, ".tmp")) throw std::runtime_error("expert-replacement apply output must end in .tmp");
    source_gguf src(model);
    require_expert_count_source_contract(src);
    const json plan = read_json(plan_path);
    verify_expert_replacement_plan(plan);
    if (sha256_file(model) != plan.at("source_sha256").get<std::string>()) {
        throw std::runtime_error("expert-replacement source SHA-256 mismatch");
    }
    const auto mapping = index_expert_replacement_plan(plan);

    gguf_ptr out_meta(gguf_init_empty());
    gguf_set_kv(out_meta.get(), src.meta());
    gguf_set_val_u32(out_meta.get(), "qwen35moe.expert_count", N_EXPERT_DST);
    gguf_set_val_u32(out_meta.get(), "qwen35moe.block_count", N_LAYER_MAIN);
    gguf_set_val_u32(out_meta.get(), "qwen35moe.nextn_predict_layers", 0);
    // expert_used_count intentionally remains 8; this is parameter-capacity
    // pruning, not active-compute pruning.

    ggml_init_params desc_params { 8u * 1024u * 1024u, nullptr, true };
    ggml_ptr desc_ctx(ggml_init(desc_params));
    if (!desc_ctx) throw std::runtime_error("failed to allocate expert-count descriptor context");
    int changed_payloads = 0;
    int changed_routers = 0;
    int dropped_mtp_tensors = 0;
    for (int64_t id = 0; id < src.tensor_count(); ++id) {
        const std::string name = gguf_get_tensor_name(src.meta(), id);
        if (is_mtp_tensor(name)) {
            ++dropped_mtp_tensors;
            continue;
        }
        const ggml_tensor * st = src.tensor(name);
        gguf_add_tensor(out_meta.get(), make_expert_count_output_descriptor(desc_ctx.get(), st, name));
        changed_payloads += is_routed_expert_payload(name) ? 1 : 0;
        changed_routers += is_routed_expert_router(name) ? 1 : 0;
    }
    if (changed_payloads != N_LAYER_MAIN * 3 || changed_routers != N_LAYER_MAIN || (src.tensor_count() == 753 && dropped_mtp_tensors <= 0)) {
        throw std::runtime_error("expert-count tensor inventory mismatch");
    }

    output_writer writer(output, out_meta.get(), true);
    json manifest;
    manifest["format"] = "qwen35-expert-replacement-apply-v1";
    manifest["source_sha256"] = plan.at("source_sha256");
    manifest["plan_sha256"] = sha256_file(plan_path);
    manifest["source_expert_count"] = N_EXPERT;
    manifest["target_expert_count"] = N_EXPERT_DST;
    manifest["expert_used_count"] = 8;
    manifest["expert_width"] = EXPERT_WIDTH_DST;
    manifest["selection_method"] = plan.at("selection_method");
    manifest["changed_payload_tensors"] = changed_payloads;
    manifest["changed_router_tensors"] = changed_routers;
    manifest["dropped_mtp_tensors"] = dropped_mtp_tensors;
    manifest["mtp_removed"] = true;
    manifest["changed"] = json::array();

    for (int64_t id = 0; id < src.tensor_count(); ++id) {
        const auto started = std::chrono::steady_clock::now();
        const std::string name = gguf_get_tensor_name(src.meta(), id);
        if (is_mtp_tensor(name)) continue;
        const int64_t oid = gguf_find_tensor(out_meta.get(), name.c_str());
        if (oid < 0) throw std::runtime_error("missing output descriptor: " + name);
        const int layer = parse_block_id(name);
        if ((is_routed_expert_payload(name) || is_routed_expert_router(name)) &&
            (layer < 0 || layer >= N_LAYER_MAIN)) {
            throw std::runtime_error("cannot parse expert-count tensor layer: " + name);
        }
        if (is_routed_expert_payload(name)) {
            copy_selected_expert_slices(src, writer, oid, name, mapping[(size_t) layer]);
        } else if (is_routed_expert_router(name)) {
            copy_selected_rows(src, writer, oid, name, mapping[(size_t) layer]);
        } else {
            copy_tensor_exact(src, writer, oid, name);
        }
        const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - started).count();
        if (is_routed_expert_payload(name) || is_routed_expert_router(name)) {
            const int64_t * ne = gguf_get_tensor_ne(out_meta.get(), oid);
            json shape = json::array();
            for (int d = 0; d < ggml_n_dims(src.tensor(name)); ++d) shape.push_back(ne[d]);
            manifest["changed"].push_back({
                {"name", name}, {"new_shape", shape},
                {"new_nbytes", gguf_get_tensor_size(out_meta.get(), oid)},
                {"elapsed_ms", elapsed_ms},
            });
        }
        std::cerr << "expert-count apply: " << name << " elapsed_ms=" << elapsed_ms << "\n";
    }
    writer.finish();
    write_json(output + ".manifest.json", manifest);
    std::cout << "wrote 192-expert GGUF tmp: " << output << "\n";
}

static void command_verify_expert_replacement_192(
        const std::string & source,
        const std::string & output,
        const std::string & plan_path) {
    source_gguf src(source), out(output);
    require_expert_count_source_contract(src);
    const json plan = read_json(plan_path);
    verify_expert_replacement_plan(plan);
    if (sha256_file(source) != plan.at("source_sha256").get<std::string>()) {
        throw std::runtime_error("expert-replacement verify source hash mismatch");
    }
    int dropped_mtp_tensors = 0;
    for (int64_t id = 0; id < src.tensor_count(); ++id) {
        if (is_mtp_tensor(gguf_get_tensor_name(src.meta(), id))) ++dropped_mtp_tensors;
    }
    if (out.tensor_count() != src.tensor_count() - dropped_mtp_tensors ||
        out.get_str("general.architecture") != "qwen35moe" ||
        out.get_u32("qwen35moe.block_count") != N_LAYER_MAIN ||
        out.get_u32("qwen35moe.nextn_predict_layers") != 0 ||
        out.get_u32("qwen35moe.expert_count") != N_EXPERT_DST ||
        out.get_u32("qwen35moe.expert_used_count") != 8 ||
        out.get_u32("qwen35moe.expert_feed_forward_length") != EXPERT_WIDTH_DST) {
        throw std::runtime_error("expert-replacement output metadata mismatch");
    }
    const auto mapping = index_expert_replacement_plan(plan);

    int verified_payloads = 0;
    int verified_routers = 0;
    for (int64_t id = 0; id < src.tensor_count(); ++id) {
        const std::string name = gguf_get_tensor_name(src.meta(), id);
        if (is_mtp_tensor(name)) {
            if (gguf_find_tensor(out.meta(), name.c_str()) >= 0) {
                throw std::runtime_error("MTP tensor unexpectedly retained: " + name);
            }
            continue;
        }
        const ggml_tensor * st = src.tensor(name);
        const ggml_tensor * ot = out.tensor(name);
        if (st->type != ot->type || ggml_n_dims(st) != ggml_n_dims(ot)) {
            throw std::runtime_error("expert-replacement tensor type/rank changed: " + name);
        }
        const int layer = parse_block_id(name);
        if (is_routed_expert_payload(name)) {
            if (layer < 0 || layer >= N_LAYER_MAIN || ot->ne[2] != N_EXPERT_DST ||
                ot->ne[0] != st->ne[0] || ot->ne[1] != st->ne[1]) {
                throw std::runtime_error("expert-replacement payload shape mismatch: " + name);
            }
            const size_t src_slice = src.tensor_size(name) / N_EXPERT;
            const size_t dst_slice = out.tensor_size(name) / N_EXPERT_DST;
            if (src_slice != dst_slice) throw std::runtime_error("expert payload slice byte size changed: " + name);
            for (int dst = 0; dst < N_EXPERT_DST; ++dst) {
                const int se = mapping[(size_t) layer][(size_t) dst];
                if (!byte_ranges_equal(src, name, (size_t) se * src_slice,
                                       out, name, (size_t) dst * dst_slice, src_slice)) {
                    throw std::runtime_error("expert payload gather mismatch: " + name + " dst=" + std::to_string(dst));
                }
            }
            ++verified_payloads;
        } else if (is_routed_expert_router(name)) {
            if (layer < 0 || layer >= N_LAYER_MAIN || ot->ne[0] != HIDDEN || ot->ne[1] != N_EXPERT_DST) {
                throw std::runtime_error("expert-replacement router shape mismatch: " + name);
            }
            const size_t row_size = ggml_row_size(st->type, HIDDEN);
            for (int dst = 0; dst < N_EXPERT_DST; ++dst) {
                const int se = mapping[(size_t) layer][(size_t) dst];
                if (!byte_ranges_equal(src, name, (size_t) se * row_size,
                                       out, name, (size_t) dst * row_size, row_size)) {
                    throw std::runtime_error("expert router gather mismatch: " + name + " dst=" + std::to_string(dst));
                }
            }
            ++verified_routers;
        } else {
            for (int d = 0; d < ggml_n_dims(st); ++d) {
                if (st->ne[d] != ot->ne[d]) throw std::runtime_error("non-expert tensor shape changed: " + name);
            }
            if (src.tensor_size(name) != out.tensor_size(name) ||
                !byte_ranges_equal(src, name, 0, out, name, 0, src.tensor_size(name))) {
                throw std::runtime_error("non-expert tensor payload changed: " + name);
            }
        }
    }
    if (verified_payloads != N_LAYER_MAIN * 3 || verified_routers != N_LAYER_MAIN) {
        throw std::runtime_error("expert-replacement verify tensor inventory mismatch");
    }
    std::cout << "expert-replacement verify OK: 256->192 experts, Top-8 preserved, MTP removed, "
              << verified_payloads << " expert payload tensors + " << verified_routers
              << " router tensors gathered byte-exactly\n";
}

static void command_finalize_expert_replacement_192(
        const std::string & source,
        const std::string & tmp,
        const std::string & plan_path,
        const std::string & final_path) {
    if (!ends_with(tmp, ".tmp")) throw std::runtime_error("expert-replacement finalize input must end in .tmp");
    command_verify_expert_replacement_192(source, tmp, plan_path);
    if (::rename(tmp.c_str(), final_path.c_str()) != 0) {
        throw std::runtime_error(std::string("expert-replacement finalize rename failed: ") + std::strerror(errno));
    }
    const std::string tmp_manifest = tmp + ".manifest.json";
    const std::string final_manifest = final_path + ".manifest.json";
    if (::rename(tmp_manifest.c_str(), final_manifest.c_str()) != 0) {
        throw std::runtime_error(std::string("expert-replacement manifest rename failed: ") + std::strerror(errno));
    }
    json manifest = read_json(final_manifest);
    manifest["final_sha256"] = sha256_file(final_path);
    manifest["final_size_bytes"] = file_identity(final_path).at("size");
    write_json(final_manifest, manifest);
    std::cout << "finalized 192-expert GGUF: " << final_path << "\n"
              << "sha256: " << manifest.at("final_sha256").get<std::string>() << "\n";
}

static std::vector<float> expert_hx(const common_imatrix & imat, int layer, int expert) {
    const std::string prefix = "blk." + std::to_string(layer) + ".";
    const std::string gname = prefix + "ffn_gate_exps.weight";
    const std::string uname = prefix + "ffn_up_exps.weight";
    const auto & g = require_entry(imat, gname);
    const auto & u = require_entry(imat, uname);
    if ((int) g.counts.size() != N_EXPERT || (int) u.counts.size() != N_EXPERT) {
        throw std::runtime_error("missing gate/up counts in apply");
    }
    if (g.counts[(size_t) expert] <= 0 || u.counts[(size_t) expert] <= 0) {
        // Matches --quick plan's explicit zero-routing fallback. Unit importance
        // keeps q2-opt-signed weight-aware without inventing activation statistics.
        return std::vector<float>(HIDDEN, 1.0f);
    }
    const double denom = (double) g.counts[(size_t) expert] + u.counts[(size_t) expert];
    std::vector<float> h(HIDDEN);
    for (int i = 0; i < HIDDEN; ++i) h[i] = (float) ((g.sums[(size_t) expert * HIDDEN + i] + u.sums[(size_t) expert * HIDDEN + i]) / denom);
    return h;
}

static std::vector<float> expert_ha(const common_imatrix & imat, int layer, int expert) {
    const std::string name = "blk." + std::to_string(layer) + ".ffn_down_exps.weight";
    const auto & e = require_entry(imat, name);
    if ((int) e.counts.size() != N_EXPERT) throw std::runtime_error("missing down counts in apply");
    if (e.counts[(size_t) expert] <= 0) return std::vector<float>(EXPERT_WIDTH_SRC, 1.0f);
    return mean_expert_stats(e, expert, EXPERT_WIDTH_SRC);
}

static void encode_q2_row(
        const float * values,
        const float * importance,
        int n,
        std::vector<uint8_t> & packed,
        q2_manifest_stats & stats) {
    if (n % 64 != 0) throw std::runtime_error("Q2 row width is not a multiple of 64");
    packed.resize((size_t) n / 64 * sizeof(q2_opt_block));
    for (int b = 0; b < n / 64; ++b) {
        q2_opt_block block {};
        q2_opt_block stock_block {};
        bool zero_weight = true;
        for (int i = 0; i < 64; ++i) zero_weight &= importance[b*64+i] == 0.0f;
        q2_opt_metrics positive {};
        const auto m = encode_q2_opt_signed_compare(values + b*64, importance + b*64, block, positive);
        const auto stock = encode_q2_stock(values + b*64, importance + b*64, stock_block);
        const double tol = 1e-10 * std::max(1.0, stock.weighted_sse);
        if (m.weighted_sse > positive.weighted_sse + tol ||
            positive.weighted_sse > stock.weighted_sse + tol) {
            throw std::runtime_error("Q2 block SSE inclusion violated: signed <= positive <= stock");
        }
        std::memcpy(packed.data() + (size_t) b * sizeof(block), &block, sizeof(block));
        stats.add(block, m, positive, stock, zero_weight);
    }
}

static bool expert_mapping_packed_slice(
        const std::vector<int> & mapping,
        ggml_type type,
        size_t & byte_offset) {
    if (mapping.size() != EXPERT_WIDTH_DST) return false;
    const int first = mapping.front();
    if (first < 0 || first + EXPERT_WIDTH_DST > EXPERT_WIDTH_SRC) return false;
    for (int j = 0; j < EXPERT_WIDTH_DST; ++j) {
        if (mapping[(size_t) j] != first + j) return false;
    }
    const int64_t block_size = ggml_blck_size(type);
    if (block_size <= 0 || first % block_size != 0 || EXPERT_WIDTH_DST % block_size != 0) return false;
    byte_offset = (size_t) (first / block_size) * ggml_type_size(type);
    return true;
}

static void write_expert_tensor(
        const source_gguf & src,
        output_writer & out,
        int64_t id,
        const std::string & name,
        const plan_index & plan,
        const common_imatrix & imat,
        q2_manifest_stats & stats,
        int n_threads,
        bool expert_energy_match,
        bool expert_source_quant) {
    const int layer = parse_block_id(name);
    const ggml_tensor * t = src.tensor(name);
    const bool down = ends_with(name, "ffn_down_exps.weight");
    const auto * tt = ggml_get_type_traits(t->type);
    if (!expert_source_quant || down) {
        if (!tt || !tt->to_float) {
            throw std::runtime_error("expert source type cannot be decoded/requantized: " + name);
        }
    }
    const size_t src_row_size = ggml_row_size(t->type, t->ne[0]);
    const size_t src_rows_per_expert = (size_t) t->ne[1];
    const size_t src_expert_bytes = src_row_size * src_rows_per_expert;
    const ggml_type dst_type = expert_source_quant
        ? (down && EXPERT_WIDTH_DST % ggml_blck_size(t->type) != 0 ? GGML_TYPE_Q4_0 : t->type)
        : GGML_TYPE_Q2_0;
    const auto * dst_tt = ggml_get_type_traits(dst_type);
    if (!expert_source_quant || (down && dst_type != t->type)) {
        if (!dst_tt || !dst_tt->from_float_ref) {
            throw std::runtime_error("expert destination type cannot be quantized: " + name);
        }
    }
    const size_t dst_row_size = ggml_row_size(dst_type, down ? EXPERT_WIDTH_DST : HIDDEN);
    const size_t dst_rows_per_expert = down ? HIDDEN : EXPERT_WIDTH_DST;
    const size_t dst_expert_bytes = dst_row_size * dst_rows_per_expert;

    std::array<q2_manifest_stats, N_EXPERT> per_expert_stats {};
    std::atomic<int> next_expert {0};
    std::atomic<bool> failed {false};
    std::exception_ptr error;
    std::mutex error_mutex;

    auto worker = [&]() {
        try {
            std::vector<uint8_t> slice(src_expert_bytes);
            std::vector<uint8_t> dst_slice(dst_expert_bytes);
            std::vector<float> decoded((size_t) t->ne[0]);
            std::vector<float> gathered(EXPERT_WIDTH_DST);
            std::vector<float> gathered_h(EXPERT_WIDTH_DST);
            std::vector<uint8_t> qrow;
            while (!failed.load(std::memory_order_relaxed)) {
                const int expert = next_expert.fetch_add(1);
                if (expert >= N_EXPERT) break;
                src.read_tensor_bytes(name, (size_t) expert * src_expert_bytes, slice.data(), slice.size());
                const auto & mapping = plan.expert[(size_t) layer][(size_t) expert];
                auto & local_stats = per_expert_stats[(size_t) expert];
                if (down) {
                    // Pure source-quant pruning only gathers/requantizes teacher
                    // weights.  It must not require activation statistics (in
                    // particular for the non-target MTP block, whose plan uses
                    // the explicit first-256 fallback and has no ENP imatrix
                    // entry).  ha is needed only by Q2 importance weighting or
                    // the separate energy-match experiment.
                    const auto ha = (!expert_source_quant || expert_energy_match)
                        ? expert_ha(imat, layer, expert)
                        : std::vector<float>();
                    if (!expert_source_quant) {
                        for (int j = 0; j < EXPERT_WIDTH_DST; ++j) {
                            gathered_h[j] = ha[(size_t) mapping[(size_t) j]];
                        }
                    }
                    double down_scale = 1.0;
                    if (expert_energy_match) {
                        // The current pruning objective keeps only diagonal activation
                        // second moments.  Under that same approximation, the expected
                        // squared expert output is
                        //   sum_j E[z_j^2] * ||down[:, j]||^2.
                        // Match this quantity before/after 512->256 pruning with one
                        // deterministic per-expert scalar.  This cannot reconstruct
                        // cross-neuron covariance, but it avoids the systematic output
                        // energy shrinkage caused by simply deleting half the neurons.
                        long double total_energy = 0.0L;
                        long double kept_energy = 0.0L;
                        for (int r = 0; r < HIDDEN; ++r) {
                            tt->to_float(slice.data() + (size_t) r * src_row_size, decoded.data(), EXPERT_WIDTH_SRC);
                            for (int j = 0; j < EXPERT_WIDTH_SRC; ++j) {
                                const long double w = decoded[(size_t) j];
                                total_energy += (long double) std::max(0.0f, ha[(size_t) j]) * w * w;
                            }
                            for (int j = 0; j < EXPERT_WIDTH_DST; ++j) {
                                const int old = mapping[(size_t) j];
                                const long double w = decoded[(size_t) old];
                                kept_energy += (long double) std::max(0.0f, ha[(size_t) old]) * w * w;
                            }
                        }
                        if (kept_energy > 0.0L && total_energy > 0.0L) {
                            down_scale = std::sqrt((double) (total_energy / kept_energy));
                        }
                        if (!std::isfinite(down_scale) || down_scale < 1.0) {
                            throw std::runtime_error("invalid expert energy-match scale");
                        }
                    }
                    size_t packed_slice_offset = 0;
                    const bool packed_slice = expert_source_quant && dst_type == t->type && !expert_energy_match &&
                        expert_mapping_packed_slice(mapping, t->type, packed_slice_offset);
                    if (packed_slice && packed_slice_offset + dst_row_size > src_row_size) {
                        throw std::runtime_error("source-quant expert packed slice is out of row bounds");
                    }
                    for (int r = 0; r < HIDDEN; ++r) {
                        if (packed_slice) {
                            std::memcpy(
                                dst_slice.data() + (size_t) r * dst_row_size,
                                slice.data() + (size_t) r * src_row_size + packed_slice_offset,
                                dst_row_size);
                            continue;
                        }
                        tt->to_float(slice.data() + (size_t) r * src_row_size, decoded.data(), EXPERT_WIDTH_SRC);
                        for (int j = 0; j < EXPERT_WIDTH_DST; ++j) {
                            gathered[j] = (float) ((double) decoded[(size_t) mapping[(size_t) j]] * down_scale);
                        }
                        if (expert_source_quant) {
                            qrow.resize(dst_row_size);
                            dst_tt->from_float_ref(gathered.data(), qrow.data(), EXPERT_WIDTH_DST);
                        } else {
                            encode_q2_row(gathered.data(), gathered_h.data(), EXPERT_WIDTH_DST, qrow, local_stats);
                        }
                        std::memcpy(dst_slice.data() + (size_t) r * dst_row_size, qrow.data(), qrow.size());
                    }
                } else {
                    const auto hx = expert_source_quant ? std::vector<float>() : expert_hx(imat, layer, expert);
                    for (int j = 0; j < EXPERT_WIDTH_DST; ++j) {
                        const int old = mapping[(size_t) j];
                        if (expert_source_quant) {
                            // Gate/up keep their full 2048-wide row.  With the
                            // source quantizer retained, selecting a neuron can
                            // therefore be bit-exact: no requantization at all.
                            if (src_row_size != dst_row_size) {
                                throw std::runtime_error("source-quant gate/up row size changed unexpectedly");
                            }
                            std::memcpy(dst_slice.data() + (size_t) j * dst_row_size,
                                        slice.data() + (size_t) old * src_row_size,
                                        src_row_size);
                        } else {
                            tt->to_float(slice.data() + (size_t) old * src_row_size, decoded.data(), HIDDEN);
                            encode_q2_row(decoded.data(), hx.data(), HIDDEN, qrow, local_stats);
                            std::memcpy(dst_slice.data() + (size_t) j * dst_row_size, qrow.data(), qrow.size());
                        }
                    }
                }
                out.write_tensor(id, (size_t) expert * dst_expert_bytes, dst_slice.data(), dst_slice.size());
            }
        } catch (...) {
            failed.store(true, std::memory_order_relaxed);
            std::lock_guard<std::mutex> lock(error_mutex);
            if (!error) error = std::current_exception();
        }
    };

    const int workers = std::max(1, std::min(n_threads, N_EXPERT));
    std::vector<std::thread> threads;
    threads.reserve((size_t) workers);
    for (int i = 0; i < workers; ++i) threads.emplace_back(worker);
    for (auto & th : threads) th.join();
    if (error) std::rethrow_exception(error);
    for (int expert = 0; expert < N_EXPERT; ++expert) stats.merge(per_expert_stats[(size_t) expert]);
}

static void write_ssm_out(
        const source_gguf & src,
        output_writer & out,
        int64_t id,
        const std::string & name,
        const json & rec,
        uint64_t & q4_repack,
        uint64_t & q8_copy) {
    const ggml_tensor * t = src.tensor(name);
    const auto channels = gdn_v_channel_mapping(rec);
    if (channels.size() != 2048) throw std::runtime_error("ssm_out channel map length mismatch");
    const size_t src_row_size = ggml_row_size(t->type, 4096);
    const size_t dst_row_size = ggml_row_size(t->type, 2048);
    std::vector<uint8_t> src_row(src_row_size), dst_row(dst_row_size);
    const auto * traits = ggml_get_type_traits(t->type);
    if (!traits || !traits->to_float || !traits->from_float_ref ||
        (t->type != GGML_TYPE_Q8_0 && t->type != GGML_TYPE_Q4_K)) {
        throw std::runtime_error("unsupported ssm_out type: " + type_name(t->type));
    }
    std::vector<float> decoded(4096), gathered(2048);
    for (int r = 0; r < HIDDEN; ++r) {
        src.read_tensor_bytes(name, (size_t) r * src_row_size, src_row.data(), src_row.size());
        traits->to_float(src_row.data(), decoded.data(), 4096);
        for (int j = 0; j < 2048; ++j) gathered[(size_t) j] = decoded[(size_t) channels[(size_t) j]];
        traits->from_float_ref(gathered.data(), dst_row.data(), 2048);
        out.write_tensor(id, (size_t) r * dst_row_size, dst_row.data(), dst_row.size());
        if (t->type == GGML_TYPE_Q4_K) q4_repack += 8;   // eight 256-value output blocks
        else q8_copy += 64;                              // historical counter: 64 output blocks requantized
    }
}

static void write_gdn_tensor(
        const source_gguf & src,
        output_writer & out,
        int64_t id,
        const std::string & name,
        const json & rec,
        uint64_t & q4_repack,
        uint64_t & q8_copy) {
    if (ends_with(name, "attn_qkv.weight")) {
        copy_selected_rows(src, out, id, name, gdn_qkv_mapping(rec));
    } else if (ends_with(name, "ssm_conv1d.weight")) {
        copy_selected_rows(src, out, id, name, gdn_qkv_mapping(rec));
    } else if (ends_with(name, "attn_gate.weight")) {
        copy_selected_rows(src, out, id, name, gdn_v_channel_mapping(rec));
    } else if (ends_with(name, "ssm_norm.weight")) {
        const ggml_tensor * t = src.tensor(name);
        if (ggml_n_dims(t) != 1 || t->type != GGML_TYPE_F32 || t->ne[0] != GDN_DIM_SRC) {
            throw std::runtime_error("unexpected ssm_norm layout: " + name);
        }
        const auto positions = gdn_v_positions(rec);
        if ((int) positions.size() != GDN_DIM_DST) throw std::runtime_error("ssm_norm position map mismatch");
        std::array<float, GDN_DIM_SRC> src_values {};
        std::array<float, GDN_DIM_DST> dst_values {};
        src.read_tensor_bytes(name, 0, src_values.data(), sizeof(src_values));
        for (int i = 0; i < GDN_DIM_DST; ++i) {
            dst_values[(size_t) i] = src_values[(size_t) positions[(size_t) i]];
        }
        out.write_tensor(id, 0, dst_values.data(), sizeof(dst_values));
    } else if (ends_with(name, "ssm_out.weight")) {
        write_ssm_out(src, out, id, name, rec, q4_repack, q8_copy);
    } else {
        throw std::runtime_error("unexpected GDN target: " + name);
    }
}

static void command_rewrite_gdn_inplace(
        const std::string & source_path,
        const std::string & plan_path,
        const std::string & target_path,
        int only_layer = -1) {
    if (!ends_with(target_path, ".tmp")) {
        throw std::runtime_error("rewrite-gdn-inplace target must end in .tmp");
    }

    source_gguf src(source_path);
    source_gguf target(target_path);
    require_source_contract(src);
    const json plan = read_json(plan_path);
    verify_plan_json(plan);
    if (sha256_file(source_path) != plan.at("source_sha256").get<std::string>()) {
        throw std::runtime_error("source SHA-256 does not match GDN rewrite plan");
    }
    if (target.tensor_count() != src.tensor_count() ||
        target.get_str("general.architecture") != "qwen35moe" ||
        target.get_u32("qwen35moe.expert_feed_forward_length") != EXPERT_WIDTH_DST ||
        target.get_u32("qwen35moe.ssm.state_size") != GDN_DIM_DST ||
        target.get_u32("qwen35moe.ssm.inner_size") != 2048) {
        throw std::runtime_error("rewrite-gdn-inplace target is not a full pruned diagnostic GGUF");
    }

    const plan_index idx = index_plan(plan);
    output_writer writer(target_path, target.meta(), /*initialize=*/ false);
    uint64_t q4_repack = 0;
    uint64_t q8_copy = 0;
    std::unordered_map<std::string, std::string> rewritten_hash;

    for (int64_t sid = 0; sid < src.tensor_count(); ++sid) {
        const std::string name = gguf_get_tensor_name(src.meta(), sid);
        const ggml_tensor * st = src.tensor(name);
        if (classify_tensor(name, st) != tensor_policy::gdn) continue;

        const int layer = parse_block_id(name);
        if (only_layer >= 0 && layer != only_layer) continue;
        const auto it = idx.gdn.find(layer);
        if (it == idx.gdn.end()) throw std::runtime_error("missing GDN rewrite plan record: " + name);
        const int64_t tid = gguf_find_tensor(target.meta(), name.c_str());
        if (tid < 0) throw std::runtime_error("target missing GDN tensor: " + name);

        const ggml_tensor * tt = target.tensor(name);
        if (tt->type != st->type) throw std::runtime_error("GDN rewrite target type mismatch: " + name);
        if (ends_with(name, "attn_qkv.weight") || ends_with(name, "ssm_conv1d.weight")) {
            if (tt->ne[1] != 4096) throw std::runtime_error("GDN rewrite qkv/conv target shape mismatch: " + name);
        } else if (ends_with(name, "attn_gate.weight")) {
            if (tt->ne[1] != 2048) throw std::runtime_error("GDN rewrite gate target shape mismatch: " + name);
        } else if (ends_with(name, "ssm_norm.weight")) {
            if (tt->ne[0] != GDN_DIM_DST) throw std::runtime_error("GDN rewrite norm target shape mismatch: " + name);
        } else if (ends_with(name, "ssm_out.weight")) {
            if (tt->ne[0] != 2048) throw std::runtime_error("GDN rewrite output target shape mismatch: " + name);
        }

        write_gdn_tensor(src, writer, tid, name, it->second, q4_repack, q8_copy);
        rewritten_hash[name] = writer.tensor_sha256(tid);
        std::cerr << "rewrite GDN: " << name << " sha256=" << rewritten_hash[name].substr(0, 12) << "\n";
    }
    const size_t expected_rewrites = only_layer >= 0 ? 5 : 150;
    if (rewritten_hash.size() != expected_rewrites) {
        throw std::runtime_error("GDN rewrite touched unexpected tensor count");
    }
    writer.finish();

    const std::string manifest_path = target_path + ".manifest.json";
    json manifest = json::object();
    {
        std::ifstream in(manifest_path);
        if (in.good()) {
            in.close();
            manifest = read_json(manifest_path);
        }
    }
    manifest["plan_sha256"] = sha256_file(plan_path);
    manifest["gdn_plan"] = plan.at("gdn");
    manifest["gdn_rewrite_inplace"] = true;
    if (only_layer >= 0) manifest["gdn_rewrite_last_layer"] = only_layer;
    manifest["gdn_rewrite_tool_exe_sha256"] = current_executable_sha256();
    manifest["gdn_rewrite_q4_repack_blocks"] = q4_repack;
    manifest["gdn_rewrite_q8_block_copies"] = q8_copy;
    if (manifest.contains("changed") && manifest["changed"].is_array()) {
        for (auto & rec : manifest["changed"]) {
            const std::string name = rec.value("name", std::string());
            const auto it = rewritten_hash.find(name);
            if (it != rewritten_hash.end()) rec["output_sha256"] = it->second;
        }
    }
    manifest["output_sha256"] = sha256_file(target_path);
    write_json(manifest_path, manifest);
    std::cout << "rewrote " << rewritten_hash.size() << " GDN tensors in place: " << target_path << "\n"
              << "manifest: " << manifest_path << "\n";
}

static void command_apply(
        const std::string & model,
        const std::string & imatrix_path,
        const std::string & plan_path,
        const std::string & output,
        int n_threads,
        int max_tensors,
        bool keep_gdn,
        bool keep_vocab,
        bool keep_experts,
        bool expert_energy_match,
        bool expert_source_quant) {
    if (!ends_with(output, ".tmp")) throw std::runtime_error("apply output must end in .tmp");
    const std::string checkpoint_path = output + ".checkpoint.json";
    advisory_file_lock checkpoint_lock(checkpoint_path + ".lock");
    source_gguf src(model);
    require_source_contract(src);
    const json plan = read_json(plan_path);
    verify_plan_json(plan);
    if (plan.value("expert_only", false) &&
        (!keep_gdn || !keep_vocab || !expert_source_quant || expert_energy_match)) {
        throw std::runtime_error(
            "expert-only plan requires --keep-gdn --keep-vocab --expert-source-quant and forbids --expert-energy-match");
    }
    if (!keep_experts) require_materializable_enp_plan(plan);
    // Every apply invocation, including a checkpoint resume, authenticates all
    // immutable inputs before writing another byte. A checkpoint proves what a
    // previous invocation saw; it must not silently bless a source file changed
    // between segments.
    if (sha256_file(model) != plan.at("source_sha256").get<std::string>()) throw std::runtime_error("source SHA-256 does not match plan");
    if (sha256_file(imatrix_path) != plan.at("imatrix_sha256").get<std::string>()) throw std::runtime_error("imatrix SHA-256 does not match plan");
    common_imatrix imat;
    if (!common_imatrix_load(imatrix_path, imat)) throw std::runtime_error("failed to load imatrix");
    const plan_index idx = index_plan(plan);
    const std::string plan_sha = sha256_file(plan_path);
    const std::string apply_tool_sha = current_executable_sha256();

    gguf_ptr out_meta(gguf_init_empty());
    gguf_set_kv(out_meta.get(), src.meta());
    if (!keep_experts) {
        gguf_set_val_u32(out_meta.get(), "qwen35moe.expert_feed_forward_length", EXPERT_WIDTH_DST);
    }
    if (!keep_gdn) {
        gguf_set_val_u32(out_meta.get(), "qwen35moe.ssm.state_size", GDN_DIM_DST);
        gguf_set_val_u32(out_meta.get(), "qwen35moe.ssm.inner_size", 2048);
    }
    size_t merge_count = 0;
    if (!keep_vocab) {
        merge_count = rebuild_tokenizer_metadata(out_meta.get(), src, plan, idx.vocab);
    } else {
        const int64_t merge_key = gguf_find_key(src.meta(), "tokenizer.ggml.merges");
        if (merge_key < 0) throw std::runtime_error("source tokenizer merges missing");
        merge_count = gguf_get_arr_n(src.meta(), merge_key);
    }

    ggml_init_params desc_params { 8u * 1024u * 1024u, nullptr, true };
    ggml_ptr desc_ctx(ggml_init(desc_params));
    if (!desc_ctx) throw std::runtime_error("failed to allocate descriptor context");
    std::array<int,4> policy_counts {};
    for (int64_t id = 0; id < src.tensor_count(); ++id) {
        const std::string name = gguf_get_tensor_name(src.meta(), id);
        const ggml_tensor * st = src.tensor(name);
        const tensor_policy p = classify_tensor(name, st);
        ++policy_counts[(int) p];
        tensor_policy output_policy = p;
        if ((keep_gdn && p == tensor_policy::gdn) ||
            (keep_vocab && p == tensor_policy::vocab) ||
            (keep_experts && p == tensor_policy::expert_q2)) {
            output_policy = tensor_policy::copy;
        }
        gguf_add_tensor(out_meta.get(), make_output_descriptor(
            desc_ctx.get(), st, name, output_policy,
            expert_source_quant && output_policy == tensor_policy::expert_q2));
    }
    const uint32_t block_count = src.get_u32("qwen35moe.block_count");
    const int exp_experts = (int) block_count * 3;
    const int exp_gdn = (int) RECURRENT_LAYERS.size() * 5;
    const int exp_vocab = 2;
    const int exp_copy = (int) src.tensor_count() - (exp_experts + exp_gdn + exp_vocab);
    if (policy_counts[(int) tensor_policy::expert_q2] != exp_experts ||
        policy_counts[(int) tensor_policy::gdn] != exp_gdn ||
        policy_counts[(int) tensor_policy::vocab] != exp_vocab ||
        policy_counts[(int) tensor_policy::copy] != exp_copy) {
        throw std::runtime_error("tensor policy inventory mismatch");
    }

    json manifest;
    q2_manifest_stats q2stats;
    uint64_t q4_repack = 0, q8_copy = 0;
    int64_t start_id = 0;
    bool resume = false;

    {
        std::ifstream chk(checkpoint_path);
        resume = chk.good();
    }
    if (resume) {
        const json checkpoint = read_json(checkpoint_path);
        if (checkpoint.value("format", std::string()) != "qwen35-apply-checkpoint-v1") {
            throw std::runtime_error("unsupported apply checkpoint format");
        }
        if (checkpoint.at("source_sha256") != plan.at("source_sha256") ||
            checkpoint.at("imatrix_sha256") != plan.at("imatrix_sha256") ||
            checkpoint.at("plan_sha256").get<std::string>() != plan_sha ||
            checkpoint.value("tool_exe_sha256", std::string()) != apply_tool_sha ||
            checkpoint.value("keep_gdn", false) != keep_gdn ||
            checkpoint.value("keep_vocab", false) != keep_vocab ||
            checkpoint.value("keep_experts", false) != keep_experts ||
            checkpoint.value("expert_energy_match", false) != expert_energy_match ||
            checkpoint.value("expert_source_quant", false) != expert_source_quant) {
            throw std::runtime_error("apply checkpoint input hash mismatch");
        }
        std::ifstream existing(output, std::ios::binary);
        if (!existing.good()) throw std::runtime_error("checkpoint exists but output .tmp is missing");
        start_id = checkpoint.at("next_tensor").get<int64_t>();
        if (start_id < 0 || start_id > src.tensor_count()) throw std::runtime_error("checkpoint next_tensor out of range");
        manifest = checkpoint.at("manifest");
        q2stats = q2_stats_from_json(checkpoint.at("q2"));
        q4_repack = checkpoint.value("q4_repack_blocks", uint64_t(0));
        q8_copy = checkpoint.value("q8_block_copies", uint64_t(0));
        std::cerr << "resuming apply at tensor " << start_id << "/" << src.tensor_count() << "\n";
    } else {
        manifest["source_sha256"] = plan.at("source_sha256");
        manifest["imatrix_sha256"] = plan.at("imatrix_sha256");
        manifest["plan_sha256"] = plan_sha;
        manifest["policy_counts"] = {exp_experts, exp_gdn, exp_vocab, exp_copy};
        manifest["tokenizer_merges"] = merge_count;
        manifest["threads_requested"] = n_threads;
        manifest["diagnostic_keep_gdn"] = keep_gdn;
        manifest["diagnostic_keep_vocab"] = keep_vocab;
        manifest["diagnostic_keep_experts"] = keep_experts;
        manifest["expert_energy_match"] = expert_energy_match;
        manifest["expert_source_quant"] = expert_source_quant;
        manifest["expert_quantization"] = expert_source_quant
            ? (EXPERT_WIDTH_DST % 256 == 0
                ? "source-q4km-mix"
                : "source-q4km-gate-up-q4_0-down")
            : "q2-opt-signed";
        manifest["planner_tool_exe_sha256"] = plan.value("tool_exe_sha256", std::string());
        manifest["apply_tool_exe_sha256"] = apply_tool_sha;
        manifest["llama_cpp_commit"] = plan.value("llama_cpp_commit", std::string(llama_commit()));
        manifest["llama_cpp_build_info"] = plan.value("llama_cpp_build_info", std::string(llama_build_info()));
        manifest["compiler"] = plan.value("compiler", std::string(llama_compiler()));
        manifest["build_target"] = plan.value("build_target", std::string(llama_build_target()));
        manifest["cxx_flags"] = plan.value("cxx_flags", std::string(QWEN35_PRUNE_CXX_FLAGS));
        manifest["expert_plan"] = expert_plan_diagnostics(plan);
        manifest["gdn_plan"] = plan.at("gdn");
        manifest["protected_tokens"] = plan.at("vocab").value("protected_count", 0);
        manifest["byte_fallback_tokens"] = plan.at("vocab").value("byte_fallback_count", 0);
#if defined(NDEBUG)
        manifest["build_mode"] = "Release/NDEBUG";
#else
        manifest["build_mode"] = "Debug";
#endif
        manifest["policy_d"] = json::array();
        manifest["changed"] = json::array();
    }

    output_writer writer(output, out_meta.get(), /*initialize=*/ !resume);
    const int64_t end_id = max_tensors > 0
        ? std::min<int64_t>(src.tensor_count(), start_id + max_tensors)
        : src.tensor_count();

    auto save_checkpoint = [&](int64_t next_tensor, bool sync_payload) {
        if (sync_payload) writer.sync_data();
        json checkpoint;
        checkpoint["format"] = "qwen35-apply-checkpoint-v1";
        checkpoint["source_sha256"] = plan.at("source_sha256");
        checkpoint["imatrix_sha256"] = plan.at("imatrix_sha256");
        checkpoint["plan_sha256"] = plan_sha;
        checkpoint["tool_exe_sha256"] = apply_tool_sha;
        checkpoint["keep_gdn"] = keep_gdn;
        checkpoint["keep_vocab"] = keep_vocab;
        checkpoint["keep_experts"] = keep_experts;
        checkpoint["expert_energy_match"] = expert_energy_match;
        checkpoint["expert_source_quant"] = expert_source_quant;
        checkpoint["next_tensor"] = next_tensor;
        checkpoint["q2"] = q2_stats_to_json(q2stats);
        checkpoint["q4_repack_blocks"] = q4_repack;
        checkpoint["q8_block_copies"] = q8_copy;
        checkpoint["manifest"] = manifest;
        write_json(checkpoint_path, checkpoint);
    };

    for (int64_t id = start_id; id < end_id; ++id) {
        const auto tensor_started = std::chrono::steady_clock::now();
        const std::string name = gguf_get_tensor_name(src.meta(), id);
        const ggml_tensor * st = src.tensor(name);
        const tensor_policy p = classify_tensor(name, st);
        tensor_policy effective_p = p;
        if ((keep_gdn && p == tensor_policy::gdn) ||
            (keep_vocab && p == tensor_policy::vocab) ||
            (keep_experts && p == tensor_policy::expert_q2)) {
            effective_p = tensor_policy::copy;
        }
        std::string payload_sha256;
        if (effective_p == tensor_policy::copy) {
            payload_sha256 = copy_tensor_exact(src, writer, id, name);
        } else if (effective_p == tensor_policy::vocab) {
            copy_selected_rows(src, writer, id, name, idx.vocab);
        } else if (effective_p == tensor_policy::gdn) {
            const int layer = parse_block_id(name);
            auto it = idx.gdn.find(layer);
            if (it == idx.gdn.end()) throw std::runtime_error("missing GDN plan record");
            write_gdn_tensor(src, writer, id, name, it->second, q4_repack, q8_copy);
        } else {
            write_expert_tensor(src, writer, id, name, idx, imat, q2stats, n_threads,
                                expert_energy_match, expert_source_quant);
        }
        if (effective_p != tensor_policy::copy) payload_sha256 = writer.tensor_sha256(id);
        const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - tensor_started).count();

        if (effective_p == tensor_policy::copy) {
            manifest["policy_d"].push_back({
                {"name", name},
                {"sha256", payload_sha256},
                {"source_sha256", payload_sha256},
                {"output_sha256", payload_sha256},
                {"elapsed_ms", elapsed_ms},
            });
        }
        if (effective_p != tensor_policy::copy) {
            const int64_t oid = gguf_find_tensor(out_meta.get(), name.c_str());
            const int64_t * one = gguf_get_tensor_ne(out_meta.get(), oid);
            json old_shape = json::array(), new_shape = json::array();
            for (int d = 0; d < ggml_n_dims(st); ++d) { old_shape.push_back(st->ne[d]); new_shape.push_back(one[d]); }
            manifest["changed"].push_back({
                {"name",name}, {"old_type",type_name(st->type)},
                {"new_type",type_name(gguf_get_tensor_type(out_meta.get(),oid))},
                {"old_shape",old_shape}, {"new_shape",new_shape},
                {"old_nbytes",src.tensor_size(name)}, {"new_nbytes",gguf_get_tensor_size(out_meta.get(),oid)},
                {"output_sha256", payload_sha256}, {"elapsed_ms", elapsed_ms},
            });
        }
        std::cerr << "apply " << (id+1) << "/" << src.tensor_count() << ": " << name
                  << " sha256=" << payload_sha256.substr(0, 12)
                  << " elapsed_ms=" << elapsed_ms << "\n";
        if ((id + 1) < end_id && (id + 1) % 16 == 0) {
            save_checkpoint(id + 1, /*sync_payload=*/ true);
        }
    }
    if (end_id < src.tensor_count()) {
        writer.finish();
        save_checkpoint(end_id, /*sync_payload=*/ false);
        std::cout << "apply segment complete: " << end_id << "/" << src.tensor_count()
                  << " tensors; checkpoint: " << checkpoint_path << "\n";
        return;
    }

    writer.finish();

    manifest["q2"] = q2_stats_to_json(q2stats);
    manifest["q2"]["negative_d_ratio"] = q2stats.blocks ? (double) q2stats.negative_d/q2stats.blocks : 0.0;
    {
        const long double tol = 1e-10L * std::max<long double>(1.0L, q2stats.stock_weighted_sse);
        if (q2stats.weighted_sse > q2stats.positive_weighted_sse + tol ||
            q2stats.positive_weighted_sse > q2stats.stock_weighted_sse + tol) {
            throw std::runtime_error("aggregate Q2 SSE inclusion violated: signed <= positive <= stock");
        }
        manifest["q2"]["signed_le_positive_le_stock"] = true;
    }
    manifest["q4_repack_blocks"] = q4_repack;
    manifest["q8_block_copies"] = q8_copy;
    manifest["output_sha256"] = sha256_file(output);
    write_json(output + ".manifest.json", manifest);
    std::remove(checkpoint_path.c_str());
    std::cout << "wrote mixed GGUF tmp: " << output << "\nmanifest: " << output << ".manifest.json\n";
}

static bool tensors_equal(const source_gguf & a, const source_gguf & b, const std::string & name) {
    if (a.tensor_size(name) != b.tensor_size(name)) return false;
    const size_t size = a.tensor_size(name);
    std::vector<uint8_t> x(1<<20), y(1<<20);
    for (size_t p = 0; p < size; p += x.size()) {
        const size_t n = std::min(x.size(), size-p);
        a.read_tensor_bytes(name,p,x.data(),n);
        b.read_tensor_bytes(name,p,y.data(),n);
        if (std::memcmp(x.data(),y.data(),n) != 0) return false;
    }
    return true;
}

static int64_t read_integer_metadata(const gguf_context * meta, const std::string & key) {
    const int64_t k = gguf_find_key(meta, key.c_str());
    if (k < 0) throw std::runtime_error("missing integer metadata: " + key);
    switch (gguf_get_kv_type(meta, k)) {
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(meta, k);
        case GGUF_TYPE_INT32:  return gguf_get_val_i32(meta, k);
        case GGUF_TYPE_UINT64: return (int64_t) gguf_get_val_u64(meta, k);
        case GGUF_TYPE_INT64:  return gguf_get_val_i64(meta, k);
        default: throw std::runtime_error("metadata is not integer: " + key);
    }
}

static void verify_tokenizer(const source_gguf & src, const source_gguf & out, const json & plan) {
    const gguf_context * sm = src.meta();
    const gguf_context * m = out.meta();
    const int64_t tk = gguf_find_key(m,"tokenizer.ggml.tokens");
    const int64_t ty = gguf_find_key(m,"tokenizer.ggml.token_type");
    const int64_t mk = gguf_find_key(m,"tokenizer.ggml.merges");
    if (tk < 0 || ty < 0 || mk < 0 || gguf_get_arr_n(m,tk) != VOCAB_DST || gguf_get_arr_n(m,ty) != VOCAB_DST) throw std::runtime_error("output tokenizer array size mismatch");

    const int64_t stk = gguf_find_key(sm, "tokenizer.ggml.tokens");
    const int64_t sty = gguf_find_key(sm, "tokenizer.ggml.token_type");
    const int64_t ssc = gguf_find_key(sm, "tokenizer.ggml.scores");
    const int64_t osc = gguf_find_key(m,  "tokenizer.ggml.scores");
    if (stk < 0 || sty < 0) throw std::runtime_error("source tokenizer arrays missing during verify");
    const int32_t * src_types = (const int32_t *) gguf_get_arr_data(sm, sty);
    const int32_t * out_types = (const int32_t *) gguf_get_arr_data(m, ty);
    const float * src_scores = ssc >= 0 ? (const float *) gguf_get_arr_data(sm, ssc) : nullptr;
    const float * out_scores = osc >= 0 ? (const float *) gguf_get_arr_data(m, osc) : nullptr;
    if ((src_scores == nullptr) != (out_scores == nullptr)) throw std::runtime_error("tokenizer score-array presence changed");
    const auto mapping = plan.at("vocab").at("output_to_input").get<std::vector<int>>();
    for (int i = 0; i < VOCAB_DST; ++i) {
        const int old = mapping[(size_t) i];
        if (std::string(gguf_get_arr_str(m, tk, (size_t) i)) != gguf_get_arr_str(sm, stk, (size_t) old)) {
            throw std::runtime_error("tokenizer token mapping mismatch at new id " + std::to_string(i));
        }
        if (out_types[i] != src_types[old]) throw std::runtime_error("tokenizer type mapping mismatch at new id " + std::to_string(i));
        if (src_scores && std::memcmp(&out_scores[i], &src_scores[old], sizeof(float)) != 0) {
            throw std::runtime_error("tokenizer score mapping mismatch at new id " + std::to_string(i));
        }
    }

    std::unordered_set<std::string> tokens;
    for (int i=0;i<VOCAB_DST;++i) tokens.insert(gguf_get_arr_str(m,tk,(size_t)i));
    for (size_t i=0;i<gguf_get_arr_n(m,mk);++i) {
        const std::string merge=gguf_get_arr_str(m,mk,i); const size_t sp=merge.find(' ');
        if (sp==std::string::npos) throw std::runtime_error("malformed output BPE merge");
        const std::string a=merge.substr(0,sp), b=merge.substr(sp+1);
        if (!tokens.count(a)||!tokens.count(b)||!tokens.count(a+b)) throw std::runtime_error("BPE merge closure violation");
    }
    for (int64_t k=0;k<gguf_get_n_kv(m);++k) {
        const std::string key=gguf_get_key(m,k); if(!metadata_token_id_key(key)) continue;
        int64_t v=-1; switch(gguf_get_kv_type(m,k)) {
            case GGUF_TYPE_UINT32:v=gguf_get_val_u32(m,k);break; case GGUF_TYPE_INT32:v=gguf_get_val_i32(m,k);break;
            case GGUF_TYPE_UINT64:v=(int64_t)gguf_get_val_u64(m,k);break; case GGUF_TYPE_INT64:v=gguf_get_val_i64(m,k);break; default:continue;
        }
        if(v<0||v>=VOCAB_DST) throw std::runtime_error("special token id out of range: "+key);
    }
    for (auto it = plan.at("vocab").at("special_id_remap").begin();
         it != plan.at("vocab").at("special_id_remap").end(); ++it) {
        if (read_integer_metadata(m, it.key()) != it.value().get<int64_t>()) {
            throw std::runtime_error("special token id remap mismatch: " + it.key());
        }
    }
}

static void verify_tokenizer_unchanged(const source_gguf & src, const source_gguf & out) {
    const gguf_context * sm = src.meta();
    const gguf_context * om = out.meta();
    const int64_t stk = gguf_find_key(sm, "tokenizer.ggml.tokens");
    const int64_t otk = gguf_find_key(om, "tokenizer.ggml.tokens");
    const int64_t sty = gguf_find_key(sm, "tokenizer.ggml.token_type");
    const int64_t oty = gguf_find_key(om, "tokenizer.ggml.token_type");
    const int64_t smk = gguf_find_key(sm, "tokenizer.ggml.merges");
    const int64_t omk = gguf_find_key(om, "tokenizer.ggml.merges");
    if (stk < 0 || otk < 0 || sty < 0 || oty < 0 || smk < 0 || omk < 0) {
        throw std::runtime_error("keep-vocab tokenizer arrays missing");
    }
    const size_t n_tokens = gguf_get_arr_n(sm, stk);
    if (gguf_get_arr_n(om, otk) != n_tokens ||
        gguf_get_arr_n(sm, sty) != n_tokens || gguf_get_arr_n(om, oty) != n_tokens) {
        throw std::runtime_error("keep-vocab tokenizer array size changed");
    }
    const int32_t * src_types = (const int32_t *) gguf_get_arr_data(sm, sty);
    const int32_t * out_types = (const int32_t *) gguf_get_arr_data(om, oty);
    for (size_t i = 0; i < n_tokens; ++i) {
        if (std::string(gguf_get_arr_str(sm, stk, i)) != gguf_get_arr_str(om, otk, i) ||
            src_types[i] != out_types[i]) {
            throw std::runtime_error("keep-vocab tokenizer token/type changed at id " + std::to_string(i));
        }
    }
    const int64_t ssc = gguf_find_key(sm, "tokenizer.ggml.scores");
    const int64_t osc = gguf_find_key(om, "tokenizer.ggml.scores");
    if ((ssc < 0) != (osc < 0)) throw std::runtime_error("keep-vocab tokenizer score presence changed");
    if (ssc >= 0) {
        if (gguf_get_arr_n(sm, ssc) != n_tokens || gguf_get_arr_n(om, osc) != n_tokens) {
            throw std::runtime_error("keep-vocab tokenizer score size changed");
        }
        const float * ss = (const float *) gguf_get_arr_data(sm, ssc);
        const float * os = (const float *) gguf_get_arr_data(om, osc);
        if (std::memcmp(ss, os, n_tokens * sizeof(float)) != 0) {
            throw std::runtime_error("keep-vocab tokenizer scores changed");
        }
    }
    const size_t n_merges = gguf_get_arr_n(sm, smk);
    if (gguf_get_arr_n(om, omk) != n_merges) throw std::runtime_error("keep-vocab tokenizer merge count changed");
    for (size_t i = 0; i < n_merges; ++i) {
        if (std::string(gguf_get_arr_str(sm, smk, i)) != gguf_get_arr_str(om, omk, i)) {
            throw std::runtime_error("keep-vocab tokenizer merge changed at id " + std::to_string(i));
        }
    }
    for (int64_t k = 0; k < gguf_get_n_kv(sm); ++k) {
        const std::string key = gguf_get_key(sm, k);
        if (!metadata_token_id_key(key)) continue;
        if (read_integer_metadata(sm, key) != read_integer_metadata(om, key)) {
            throw std::runtime_error("keep-vocab special token id changed: " + key);
        }
    }
}

struct llama_model_deleter {
    void operator()(llama_model * model) const { if (model) llama_model_free(model); }
};
using llama_model_ptr = std::unique_ptr<llama_model, llama_model_deleter>;

struct llama_context_deleter {
    void operator()(llama_context * ctx) const { if (ctx) llama_free(ctx); }
};
using llama_context_ptr = std::unique_ptr<llama_context, llama_context_deleter>;

static llama_model_ptr load_vocab_only(const std::string & path) {
    llama_model_params params = llama_model_default_params();
    params.vocab_only = true;
    llama_model_ptr model(llama_model_load_from_file(path.c_str(), params));
    if (!model) throw std::runtime_error("vocab-only loader failed: " + path);
    return model;
}

static std::vector<llama_token> tokenize_bytes(const llama_vocab * vocab, const std::string & text, bool parse_special = false) {
    int32_t n = llama_tokenize(vocab, text.data(), (int32_t) text.size(), nullptr, 0, false, parse_special);
    if (n == 0) return {};
    if (n > 0) throw std::runtime_error("llama_tokenize unexpectedly succeeded with zero-capacity output");
    std::vector<llama_token> tokens((size_t) -n);
    n = llama_tokenize(vocab, text.data(), (int32_t) text.size(), tokens.data(), (int32_t) tokens.size(), false, parse_special);
    if (n < 0) throw std::runtime_error("llama_tokenize failed after sizing");
    tokens.resize((size_t) n);
    return tokens;
}

static std::string detokenize_bytes(const llama_vocab * vocab, const std::vector<llama_token> & tokens) {
    if (tokens.empty()) return {};
    int32_t n = llama_detokenize(vocab, tokens.data(), (int32_t) tokens.size(), nullptr, 0, false, true);
    if (n == 0) return {};
    if (n > 0) throw std::runtime_error("llama_detokenize unexpectedly succeeded with zero-capacity output");
    std::string text((size_t) -n, '\0');
    n = llama_detokenize(vocab, tokens.data(), (int32_t) tokens.size(), text.data(), (int32_t) text.size(), false, true);
    if (n < 0) throw std::runtime_error("llama_detokenize failed after sizing");
    text.resize((size_t) n);
    return text;
}

static std::string read_file_bytes(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read tokenizer audit corpus: " + path);
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

struct gdn_layer_output_capture {
    int layer = -1;
    std::vector<float> final_output;
    std::vector<float> linear_output;
};

static bool parse_layer_named_node(const char * name, const char * prefix, int & layer) {
    if (!name || !prefix) return false;
    const size_t n = std::strlen(prefix);
    if (std::strncmp(name, prefix, n) != 0) return false;
    const char * p = name + n;
    char * end = nullptr;
    const long value = std::strtol(p, &end, 10);
    if (end == p || *end != '\0' || value < 0 || value > INT_MAX) return false;
    layer = (int) value;
    return true;
}

struct gdn_geometry_shard_stats {
    std::array<std::vector<double>, GDN_QK_HEADS> qk;
    std::vector<double> v;
    uint64_t samples = 0;

    void initialize() {
        for (auto & m : qk) m.assign((size_t) GDN_DIM_SRC * GDN_DIM_SRC, 0.0);
        v.assign((size_t) GDN_DIM_SRC * GDN_DIM_SRC, 0.0);
    }
};

struct gdn_geometry_layer_stats {
    std::vector<gdn_geometry_shard_stats> shards;
    gdn_joint_replay_data replay;
    std::array<int,5> replay_seen_tokens {};
};

struct gdn_geometry_capture {
    int total_tokens = 0;
    int current_start = 0;
    int n_shards = 1;
    int replay_tokens = 0;
    std::array<gdn_geometry_layer_stats, N_LAYER_ALL> layer;

    gdn_geometry_capture(int tokens, int shards, int replay = 0)
        : total_tokens(tokens), n_shards(shards), replay_tokens(std::min(tokens, std::max(0, replay))) {
        for (int il : RECURRENT_LAYERS) {
            layer[(size_t) il].shards.resize((size_t) n_shards);
            for (auto & shard : layer[(size_t) il].shards) shard.initialize();
            auto & r = layer[(size_t) il].replay;
            r.tokens = replay_tokens;
            if (replay_tokens > 0) {
                r.conv_silu.assign((size_t) replay_tokens * 8192, 0.0f);
                r.beta.assign((size_t) replay_tokens * GDN_V_HEADS, 0.0f);
                r.gate_log_decay.assign((size_t) replay_tokens * GDN_V_HEADS, 0.0f);
                r.z.assign((size_t) replay_tokens * GDN_V_HEADS * GDN_DIM_SRC, 0.0f);
                r.teacher_final.assign((size_t) replay_tokens * GDN_V_HEADS * GDN_DIM_SRC, 0.0f);
            }
        }
    }
};

static bool gdn_capture_replay_tensor(
        gdn_geometry_capture & cap,
        ggml_tensor * t,
        int layer,
        int per_token,
        std::vector<float> & dst,
        int & seen_tokens) {
    if (cap.replay_tokens <= 0) return true;
    if (t->type != GGML_TYPE_F32 || per_token <= 0 || ggml_nelements(t) % per_token != 0) {
        throw std::runtime_error("unexpected GDN replay capture shape at layer " + std::to_string(layer));
    }
    const int n_tokens = (int) (ggml_nelements(t) / per_token);
    std::vector<float> values((size_t) n_tokens * per_token);
    ggml_backend_tensor_get(t, values.data(), 0, values.size() * sizeof(float));
    for (int tok = 0; tok < n_tokens; ++tok) {
        const int global = cap.current_start + tok;
        if (global < 0 || global >= cap.total_tokens) {
            throw std::runtime_error("GDN replay token position out of range");
        }
        if (global >= cap.replay_tokens) continue;
        std::memcpy(dst.data() + (size_t) global * per_token,
                    values.data() + (size_t) tok * per_token,
                    (size_t) per_token * sizeof(float));
        ++seen_tokens;
    }
    return true;
}

static void gdn_accumulate_outer_upper(
        std::vector<double> & dst,
        const float * x,
        double scale) {
    for (int i = 0; i < GDN_DIM_SRC; ++i) {
        const double xi = x[i] * scale;
        for (int j = i; j < GDN_DIM_SRC; ++j) {
            dst[(size_t) i * GDN_DIM_SRC + j] += xi * x[j];
        }
    }
}

static bool gdn_geometry_capture_cb(ggml_tensor * t, bool ask, void * user_data) {
    auto & cap = *static_cast<gdn_geometry_capture *>(user_data);
    int layer = -1;
    if (parse_layer_named_node(t->name, "conv_output_silu-", layer) &&
        layer >= 0 && layer < N_LAYER_ALL && !cap.layer[(size_t) layer].shards.empty()) {
        if (ask) return cap.replay_tokens > 0;
        return gdn_capture_replay_tensor(cap, t, layer, 8192, cap.layer[(size_t) layer].replay.conv_silu,
                                         cap.layer[(size_t) layer].replay_seen_tokens[0]);
    }
    if (parse_layer_named_node(t->name, "beta_sigmoid-", layer) &&
        layer >= 0 && layer < N_LAYER_ALL && !cap.layer[(size_t) layer].shards.empty()) {
        if (ask) return cap.replay_tokens > 0;
        return gdn_capture_replay_tensor(cap, t, layer, GDN_V_HEADS, cap.layer[(size_t) layer].replay.beta,
                                         cap.layer[(size_t) layer].replay_seen_tokens[1]);
    }
    if (parse_layer_named_node(t->name, "gate-", layer) &&
        layer >= 0 && layer < N_LAYER_ALL && !cap.layer[(size_t) layer].shards.empty()) {
        if (ask) return cap.replay_tokens > 0;
        return gdn_capture_replay_tensor(cap, t, layer, GDN_V_HEADS, cap.layer[(size_t) layer].replay.gate_log_decay,
                                         cap.layer[(size_t) layer].replay_seen_tokens[2]);
    }
    if (parse_layer_named_node(t->name, "z-", layer) &&
        layer >= 0 && layer < N_LAYER_ALL && !cap.layer[(size_t) layer].shards.empty()) {
        if (ask) return cap.replay_tokens > 0;
        return gdn_capture_replay_tensor(cap, t, layer, GDN_V_HEADS * GDN_DIM_SRC, cap.layer[(size_t) layer].replay.z,
                                         cap.layer[(size_t) layer].replay_seen_tokens[3]);
    }
    if (parse_layer_named_node(t->name, "final_output-", layer) &&
        layer >= 0 && layer < N_LAYER_ALL && !cap.layer[(size_t) layer].shards.empty()) {
        if (ask) return cap.replay_tokens > 0;
        return gdn_capture_replay_tensor(cap, t, layer, GDN_V_HEADS * GDN_DIM_SRC, cap.layer[(size_t) layer].replay.teacher_final,
                                         cap.layer[(size_t) layer].replay_seen_tokens[4]);
    }

    layer = -1;
    enum class kind { none, q, k, v } capture_kind = kind::none;
    if (parse_layer_named_node(t->name, "q_conv_predelta-", layer)) capture_kind = kind::q;
    else if (parse_layer_named_node(t->name, "k_conv_predelta-", layer)) capture_kind = kind::k;
    else if (parse_layer_named_node(t->name, "v_conv_predelta-", layer)) capture_kind = kind::v;
    if (capture_kind == kind::none || layer < 0 || layer >= N_LAYER_ALL ||
        cap.layer[(size_t) layer].shards.empty()) {
        return false;
    }
    if (ask) return true;
    if (t->type != GGML_TYPE_F32) {
        throw std::runtime_error("unexpected GDN geometry capture shape at layer " + std::to_string(layer));
    }
    if (t->ne[0] != GDN_DIM_SRC) {
        throw std::runtime_error("unexpected GDN q/k/v capture shape at layer " + std::to_string(layer));
    }
    const int actual_heads = (int) t->ne[1];
    if ((capture_kind == kind::v && actual_heads != GDN_V_HEADS) ||
        (capture_kind != kind::v && actual_heads != GDN_QK_HEADS && actual_heads != GDN_V_HEADS)) {
        throw std::runtime_error("unexpected GDN geometry capture head count at layer " + std::to_string(layer));
    }
    const int64_t n_elem = ggml_nelements(t);
    const int64_t per_token = (int64_t) GDN_DIM_SRC * actual_heads;
    if (n_elem % per_token != 0) throw std::runtime_error("GDN geometry capture element count mismatch");
    const int n_tokens = (int) (n_elem / per_token);
    std::vector<float> values((size_t) n_elem);
    ggml_backend_tensor_get(t, values.data(), 0, values.size() * sizeof(float));

    for (int tok = 0; tok < n_tokens; ++tok) {
        const int global = cap.current_start + tok;
        if (global < 0 || global >= cap.total_tokens) {
            throw std::runtime_error("GDN geometry capture token position out of range");
        }
        const int shard_id = std::min(cap.n_shards - 1,
                (int) (((int64_t) global * cap.n_shards) / cap.total_tokens));
        auto & shard = cap.layer[(size_t) layer].shards[(size_t) shard_id];
        const float * row = values.data() + (size_t) tok * actual_heads * GDN_DIM_SRC;
        if (capture_kind == kind::v) {
            ++shard.samples;
            for (int h = 0; h < GDN_V_HEADS; ++h) {
                const float * v = row + (size_t) h * GDN_DIM_SRC;
                gdn_accumulate_outer_upper(shard.v, v, 1.0 / GDN_V_HEADS);
            }
        } else {
            // In unfused builds q/k can be repeated from 16 to 32 heads before
            // this callback.  The repeat pattern is periodic; the first 16 are
            // the original normalized Q/K heads, so use only those.
            for (int h = 0; h < GDN_QK_HEADS; ++h) {
                const float * x = row + (size_t) h * GDN_DIM_SRC;
                gdn_accumulate_outer_upper(shard.qk[(size_t) h], x, 0.5);
            }
        }
    }
    return true;
}

static llama_context_ptr make_gdn_geometry_capture_context(
        llama_model * model,
        gdn_geometry_capture & capture,
        int n_ctx,
        int n_batch) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) n_ctx;
    cp.n_batch = (uint32_t) n_batch;
    cp.n_ubatch = (uint32_t) n_batch;
    cp.n_seq_max = 1;
    cp.n_threads = std::max(1, std::min(8, (int) std::thread::hardware_concurrency()));
    cp.n_threads_batch = cp.n_threads;
    cp.cb_eval = gdn_geometry_capture_cb;
    cp.cb_eval_user_data = &capture;
    llama_context_ptr ctx(llama_init_from_model(model, cp));
    if (!ctx) throw std::runtime_error("failed to create GDN geometry capture context");
    return ctx;
}

static bool gdn_layer_output_capture_cb(ggml_tensor * t, bool ask, void * user_data) {
    auto & cap = *static_cast<gdn_layer_output_capture *>(user_data);
    int layer = -1;
    bool final = parse_layer_named_node(t->name, "final_output-", layer);
    bool linear = !final && parse_layer_named_node(t->name, "linear_attn_out-", layer);
    if ((!final && !linear) || layer != cap.layer) return false;
    if (ask) return true;
    if (t->type != GGML_TYPE_F32 || (final && t->ne[0] % GDN_DIM_DST != 0) ||
        (linear && t->ne[0] != HIDDEN)) {
        throw std::runtime_error("unexpected GDN layer-output capture shape at layer " + std::to_string(layer));
    }
    const int64_t n_elem = ggml_nelements(t);
    auto & dst = final ? cap.final_output : cap.linear_output;
    dst.resize((size_t) n_elem);
    ggml_backend_tensor_get(t, dst.data(), 0, (size_t) n_elem * sizeof(float));
    return true;
}

static llama_model_ptr load_full_model(const std::string & path) {
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = -1;
    llama_model_ptr model(llama_model_load_from_file(path.c_str(), mp));
    if (!model) throw std::runtime_error("failed to load calibration model: " + path);
    return model;
}

static llama_context_ptr make_layer_output_capture_context(
        llama_model * model,
        gdn_layer_output_capture & capture,
        int n_ctx,
        int n_batch) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) n_ctx;
    cp.n_batch = (uint32_t) n_batch;
    cp.n_ubatch = (uint32_t) n_batch;
    cp.n_seq_max = 1;
    cp.n_threads = std::max(1, std::min(8, (int) std::thread::hardware_concurrency()));
    cp.n_threads_batch = cp.n_threads;
    cp.cb_eval = gdn_layer_output_capture_cb;
    cp.cb_eval_user_data = &capture;
    llama_context_ptr ctx(llama_init_from_model(model, cp));
    if (!ctx) throw std::runtime_error("failed to create GDN direct-output capture context");
    return ctx;
}

static void decode_capture_batch(
        llama_context * ctx,
        const std::vector<llama_token> & tokens,
        int start,
        int n) {
    llama_batch batch = llama_batch_init(n, 0, 1);
    common_batch_clear(batch);
    for (int i = 0; i < n; ++i) {
        common_batch_add(batch, tokens[(size_t) start + i], start + i, {0}, i + 1 == n);
    }
    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) throw std::runtime_error("trajectory calibration llama_decode failed: " + std::to_string(rc));
}

static std::vector<double> gdn_geometry_materialize_moment(
        const gdn_geometry_layer_stats & stats,
        int qk_head,
        int only_shard,
        uint64_t & samples) {
    std::vector<double> out((size_t) GDN_DIM_SRC * GDN_DIM_SRC, 0.0);
    samples = 0;
    for (int s = 0; s < (int) stats.shards.size(); ++s) {
        if (only_shard >= 0 && s != only_shard) continue;
        const auto & shard = stats.shards[(size_t) s];
        samples += shard.samples;
        const auto & src = qk_head >= 0 ? shard.qk[(size_t) qk_head] : shard.v;
        for (int i = 0; i < GDN_DIM_SRC; ++i) {
            for (int j = i; j < GDN_DIM_SRC; ++j) {
                out[(size_t) i * GDN_DIM_SRC + j] += src[(size_t) i * GDN_DIM_SRC + j];
            }
        }
    }
    for (int i = 0; i < GDN_DIM_SRC; ++i) {
        for (int j = i + 1; j < GDN_DIM_SRC; ++j) {
            out[(size_t) j * GDN_DIM_SRC + i] = out[(size_t) i * GDN_DIM_SRC + j];
        }
    }
    return out;
}

static std::vector<int> gdn_geometry_select(
        const std::vector<double> & moment,
        uint64_t samples,
        const std::vector<double> & prior,
        const std::string & selection_method,
        gdn_geometry_shrink_result * shrink_out = nullptr) {
    auto shrink = gdn_geometry_shrink_second_moment(moment, samples, prior, GDN_DIM_SRC);
    std::vector<int> selected;
    if (selection_method == "topk") {
        selected = gdn_geometry_topk_diagonal_select(shrink.matrix, GDN_DIM_SRC, GDN_DIM_DST);
    } else if (selection_method == "pivoted") {
        selected = gdn_geometry_pivoted_cholesky_select(shrink.matrix, GDN_DIM_SRC, GDN_DIM_DST);
    } else {
        throw std::runtime_error("unknown GDN geometry selection method: " + selection_method);
    }
    std::sort(selected.begin(), selected.end());
    if (shrink_out) *shrink_out = std::move(shrink);
    return selected;
}

static json gdn_geometry_stability_json(
        const gdn_geometry_layer_stats & stats,
        int qk_head,
        const std::vector<double> & prior,
        const std::string & selection_method,
        const std::vector<int> & full_selection) {
    json shard_rows = json::array();
    double min_j = 1.0;
    double sum_j = 0.0;
    int count = 0;
    for (int s = 0; s < (int) stats.shards.size(); ++s) {
        uint64_t n = 0;
        auto moment = gdn_geometry_materialize_moment(stats, qk_head, s, n);
        if (n == 0) continue;
        gdn_geometry_shrink_result shrink;
        const auto sel = gdn_geometry_select(moment, n, prior, selection_method, &shrink);
        const double j = gdn_geometry_jaccard(full_selection, sel);
        min_j = std::min(min_j, j);
        sum_j += j;
        ++count;
        shard_rows.push_back({
            {"shard", s},
            {"samples", n},
            {"shrinkage", shrink.shrinkage},
            {"jaccard_vs_full", j},
        });
    }
    return {
        {"min_jaccard_vs_full", count ? min_j : 1.0},
        {"mean_jaccard_vs_full", count ? sum_j / count : 1.0},
        {"shards", shard_rows},
    };
}

static std::vector<int> gdn_blend_subset(
        const std::vector<int> & base,
        const std::vector<int> & target,
        const std::vector<double> & importance,
        double fraction) {
    if (base.size() != GDN_DIM_DST || target.size() != GDN_DIM_DST || importance.size() != GDN_DIM_SRC ||
        !(fraction >= 0.0 && fraction <= 1.0)) {
        throw std::runtime_error("invalid GDN synchronized blend request");
    }
    std::array<uint8_t,GDN_DIM_SRC> in_base {}, in_target {};
    for (int x : base) {
        if (x < 0 || x >= GDN_DIM_SRC || in_base[(size_t) x]) throw std::runtime_error("invalid GDN blend base subset");
        in_base[(size_t) x] = 1;
    }
    for (int x : target) {
        if (x < 0 || x >= GDN_DIM_SRC || in_target[(size_t) x]) throw std::runtime_error("invalid GDN blend target subset");
        in_target[(size_t) x] = 1;
    }
    std::vector<int> add, remove;
    for (int i = 0; i < GDN_DIM_SRC; ++i) {
        if (in_target[(size_t) i] && !in_base[(size_t) i]) add.push_back(i);
        if (in_base[(size_t) i] && !in_target[(size_t) i]) remove.push_back(i);
    }
    if (add.size() != remove.size()) throw std::runtime_error("GDN blend symmetric-difference mismatch");
    std::stable_sort(add.begin(), add.end(), [&](int a, int b) {
        if (importance[(size_t) a] != importance[(size_t) b]) return importance[(size_t) a] > importance[(size_t) b];
        return a < b;
    });
    std::stable_sort(remove.begin(), remove.end(), [&](int a, int b) {
        if (importance[(size_t) a] != importance[(size_t) b]) return importance[(size_t) a] < importance[(size_t) b];
        return a < b;
    });
    const size_t swaps = (size_t) std::llround(fraction * add.size());
    std::array<uint8_t,GDN_DIM_SRC> keep = in_base;
    for (size_t i = 0; i < swaps; ++i) {
        keep[(size_t) remove[i]] = 0;
        keep[(size_t) add[i]] = 1;
    }
    std::vector<int> out;
    out.reserve(GDN_DIM_DST);
    for (int i = 0; i < GDN_DIM_SRC; ++i) if (keep[(size_t) i]) out.push_back(i);
    if (out.size() != GDN_DIM_DST) throw std::runtime_error("GDN blend output size mismatch");
    return out;
}

static std::vector<double> gdn_diag_importance(const gdn_geometry_shrink_result & shrink) {
    std::vector<double> out(GDN_DIM_SRC);
    for (int i = 0; i < GDN_DIM_SRC; ++i) out[(size_t) i] = shrink.matrix[(size_t) i * GDN_DIM_SRC + i];
    return out;
}

static gdn_joint_candidate gdn_candidate_from_record(const json & rec) {
    gdn_joint_candidate out;
    const auto qk = rec.at("qk_indices_by_head").get<std::vector<std::vector<int>>>();
    if (qk.size() != GDN_QK_HEADS) throw std::runtime_error("GDN candidate record Q/K head mismatch");
    for (int h = 0; h < GDN_QK_HEADS; ++h) out.qk[(size_t) h] = qk[(size_t) h];
    out.v = rec.at("v_indices").get<std::vector<int>>();
    return out;
}

static json gdn_candidate_to_record_fields(const gdn_joint_candidate & candidate) {
    json qk = json::array();
    for (const auto & h : candidate.qk) qk.push_back(h);
    return {{"qk_indices_by_head", qk}, {"v_indices", candidate.v}};
}

static gdn_joint_replay_weights load_gdn_joint_replay_weights(const source_gguf & src, int layer) {
    gdn_joint_replay_weights out;
    out.rms_epsilon = src.get_f32("qwen35moe.attention.layer_norm_rms_epsilon");
    out.l2_epsilon = out.rms_epsilon;
    const std::string prefix = "blk." + std::to_string(layer) + ".";
    const std::string norm_name = prefix + "ssm_norm.weight";
    const ggml_tensor * norm = src.tensor(norm_name);
    if (norm->type != GGML_TYPE_F32 || norm->ne[0] != GDN_DIM_SRC || ggml_n_dims(norm) != 1) {
        throw std::runtime_error("unexpected GDN replay norm tensor");
    }
    src.read_tensor_bytes(norm_name, 0, out.norm_gamma.data(), sizeof(out.norm_gamma));

    const std::string sout_name = prefix + "ssm_out.weight";
    const ggml_tensor * sout = src.tensor(sout_name);
    if (sout->ne[0] != 4096 || sout->ne[1] != HIDDEN) throw std::runtime_error("unexpected GDN replay ssm_out shape");
    std::vector<uint8_t> packed;
    std::vector<float> decoded;
    for (int row = 0; row < HIDDEN; ++row) {
        read_quant_row(src, sout_name, (size_t) row, packed, decoded);
        for (int c = 0; c < 4096; ++c) {
            out.ssm_out_column_energy[(size_t) c] += (double) decoded[(size_t) c] * decoded[(size_t) c];
        }
    }
    return out;
}

static json gdn_joint_score_json(const std::string & name, double fraction, const gdn_joint_replay_score & score) {
    return {
        {"name", name},
        {"fraction", fraction},
        {"weighted_sse", score.weighted_sse},
        {"relative_error", score.relative_error()},
        {"retained_dynamic_sse", score.retained_dynamic_sse},
        {"omitted_teacher_energy", score.omitted_teacher_energy},
        {"teacher_energy", score.teacher_energy},
    };
}

struct gdn_joint_target_family {
    std::string name;
    gdn_joint_candidate target;
};

static gdn_joint_candidate gdn_blend_candidate(
        const gdn_joint_candidate & base,
        const gdn_joint_candidate & target,
        const std::array<std::vector<double>,GDN_QK_HEADS> & qk_importance,
        const std::vector<double> & v_importance,
        double fraction) {
    gdn_joint_candidate out;
    for (int h = 0; h < GDN_QK_HEADS; ++h) {
        out.qk[(size_t) h] = gdn_blend_subset(base.qk[(size_t) h], target.qk[(size_t) h],
                                              qk_importance[(size_t) h], fraction);
    }
    out.v = gdn_blend_subset(base.v, target.v, v_importance, fraction);
    return out;
}

static void command_plan_gdn_joint_v1(
        const std::string & teacher_path,
        const std::string & base_plan_path,
        const std::string & corpus_path,
        const std::string & output_plan_path,
        int max_tokens,
        int batch_size,
        int requested_shards,
        int requested_replay_tokens,
        int only_layer) {
    if (max_tokens < 64 || batch_size <= 0 || requested_shards <= 0 || requested_shards > 4 ||
        requested_replay_tokens < 16) {
        throw std::runtime_error("invalid GDN joint-v1 calibration arguments");
    }
    source_gguf src(teacher_path);
    require_source_contract(src);
    json plan = read_json(base_plan_path);
    verify_plan_json(plan);
    const std::string source_sha = sha256_file(teacher_path);
    if (plan.at("source_sha256").get<std::string>() != source_sha) {
        throw std::runtime_error("base plan source SHA-256 does not match teacher model");
    }

    llama_backend_init();
    auto teacher_model = load_full_model(teacher_path);
    const llama_vocab * vocab = llama_model_get_vocab(teacher_model.get());
    if (!vocab) throw std::runtime_error("GDN joint-v1 teacher has no vocabulary");
    auto tokens = tokenize_bytes(vocab, read_file_bytes(corpus_path));
    if ((int) tokens.size() > max_tokens) tokens.resize((size_t) max_tokens);
    if (tokens.size() < 64) throw std::runtime_error("GDN joint-v1 corpus tokenized to fewer than 64 tokens");
    const int n_tokens = (int) tokens.size();
    const int replay_tokens = std::min(requested_replay_tokens, n_tokens);
    const int n_shards = std::max(1, std::min(requested_shards, n_tokens / 64));
    batch_size = std::min(batch_size, n_tokens);
    gdn_geometry_capture capture(n_tokens, n_shards, replay_tokens);
    auto ctx = make_gdn_geometry_capture_context(teacher_model.get(), capture, n_tokens, batch_size);
    for (int start = 0; start < n_tokens; start += batch_size) {
        const int n = std::min(batch_size, n_tokens - start);
        capture.current_start = start;
        decode_capture_batch(ctx.get(), tokens, start, n);
        std::cerr << "GDN joint v1 teacher pass: " << (start + n) << "/" << n_tokens << " tokens\n";
    }
    ctx.reset();
    teacher_model.reset();
    llama_backend_free();

    json diagnostics = json::array();
    for (auto & rec : plan.at("gdn")) {
        const int layer = rec.at("layer").get<int>();
        if (only_layer >= 0 && layer != only_layer) continue;
        auto & ls = capture.layer[(size_t) layer];
        uint64_t observed = 0;
        for (const auto & shard : ls.shards) observed += shard.samples;
        if (observed != (uint64_t) n_tokens) throw std::runtime_error("GDN joint-v1 moment capture missed tokens");
        for (int seen : ls.replay_seen_tokens) {
            if (seen != replay_tokens) throw std::runtime_error("GDN joint-v1 replay capture missed tokens at layer " + std::to_string(layer));
        }

        const auto prior = compute_gdn_geometry_prior(src, layer);
        std::array<gdn_geometry_shrink_result,GDN_QK_HEADS> qk_shrink;
        std::array<std::vector<int>,GDN_QK_HEADS> qk_topk, qk_pivot;
        std::array<std::vector<double>,GDN_QK_HEADS> qk_importance;
        for (int h = 0; h < GDN_QK_HEADS; ++h) {
            uint64_t samples = 0;
            auto moment = gdn_geometry_materialize_moment(ls, h, -1, samples);
            qk_shrink[(size_t) h] = gdn_geometry_shrink_second_moment(moment, samples, prior.qk[(size_t) h], GDN_DIM_SRC);
            qk_topk[(size_t) h] = gdn_geometry_topk_diagonal_select(qk_shrink[(size_t) h].matrix, GDN_DIM_SRC, GDN_DIM_DST);
            qk_pivot[(size_t) h] = gdn_geometry_pivoted_cholesky_select(qk_shrink[(size_t) h].matrix, GDN_DIM_SRC, GDN_DIM_DST);
            std::sort(qk_topk[(size_t) h].begin(), qk_topk[(size_t) h].end());
            std::sort(qk_pivot[(size_t) h].begin(), qk_pivot[(size_t) h].end());
            qk_importance[(size_t) h] = gdn_diag_importance(qk_shrink[(size_t) h]);
        }
        uint64_t v_samples = 0;
        auto v_moment = gdn_geometry_materialize_moment(ls, -1, -1, v_samples);
        auto v_shrink = gdn_geometry_shrink_second_moment(v_moment, v_samples, prior.v, GDN_DIM_SRC);
        auto v_topk = gdn_geometry_topk_diagonal_select(v_shrink.matrix, GDN_DIM_SRC, GDN_DIM_DST);
        auto v_pivot = gdn_geometry_pivoted_cholesky_select(v_shrink.matrix, GDN_DIM_SRC, GDN_DIM_DST);
        std::sort(v_topk.begin(), v_topk.end());
        std::sort(v_pivot.begin(), v_pivot.end());
        const auto v_importance = gdn_diag_importance(v_shrink);

        const auto base = gdn_candidate_from_record(rec);
        auto make_target = [&](bool pivot_qk, bool pivot_v) {
            gdn_joint_candidate c;
            for (int h = 0; h < GDN_QK_HEADS; ++h) c.qk[(size_t) h] = pivot_qk ? qk_pivot[(size_t) h] : qk_topk[(size_t) h];
            c.v = pivot_v ? v_pivot : v_topk;
            return c;
        };
        std::array<gdn_joint_target_family,4> family {{
            {"topk-qk_topk-v", make_target(false, false)},
            {"topk-qk_pivot-v", make_target(false, true)},
            {"pivot-qk_topk-v", make_target(true, false)},
            {"pivot-qk_pivot-v", make_target(true, true)},
        }};
        const auto weights = load_gdn_joint_replay_weights(src, layer);
        json scores = json::array();

        // Stage 1: choose the synchronized target family at full strength.
        int best_family = 0;
        double best_family_sse = std::numeric_limits<double>::infinity();
        for (int f = 0; f < (int) family.size(); ++f) {
            const auto score = gdn_score_joint_candidate(ls.replay, family[(size_t) f].target, weights);
            scores.push_back(gdn_joint_score_json(family[(size_t) f].name, 1.0, score));
            if (score.weighted_sse < best_family_sse) {
                best_family_sse = score.weighted_sse;
                best_family = f;
            }
        }

        // Stage 2: one global trust fraction moves Q/K and V together toward
        // the selected target family. This is intentionally only one scalar
        // degree of freedom, not per-head/per-coordinate calibration fitting.
        gdn_joint_candidate chosen = base;
        std::string chosen_name = "base";
        double chosen_fraction = 0.0;
        auto base_score = gdn_score_joint_candidate(ls.replay, base, weights);
        double chosen_sse = base_score.weighted_sse;
        scores.push_back(gdn_joint_score_json("base", 0.0, base_score));
        for (double fraction : {0.25, 0.50, 0.75, 1.0}) {
            const auto candidate = gdn_blend_candidate(base, family[(size_t) best_family].target,
                                                       qk_importance, v_importance, fraction);
            const auto score = gdn_score_joint_candidate(ls.replay, candidate, weights);
            scores.push_back(gdn_joint_score_json(family[(size_t) best_family].name + "-trust", fraction, score));
            if (score.weighted_sse < chosen_sse) {
                chosen_sse = score.weighted_sse;
                chosen = candidate;
                chosen_name = family[(size_t) best_family].name;
                chosen_fraction = fraction;
            }
        }

        const auto fields = gdn_candidate_to_record_fields(chosen);
        rec["qk_indices_by_head"] = fields.at("qk_indices_by_head");
        rec["v_indices"] = fields.at("v_indices");
        rec["initialization"] = "teacher-one-pass-joint-replay-v1";
        rec["joint_family"] = chosen_name;
        rec["joint_trust_fraction"] = chosen_fraction;
        diagnostics.push_back({
            {"layer", layer},
            {"chosen_family", chosen_name},
            {"chosen_fraction", chosen_fraction},
            {"scores", scores},
        });
        std::cerr << "GDN joint v1 planned layer " << layer << ": " << chosen_name
                  << " fraction=" << chosen_fraction << "\n";
    }

    plan["gdn_joint_v1"] = {
        {"method", "one-pass-shrunk-geometry-plus-synchronized-local-recurrence-replay-v1"},
        {"teacher_forward_passes", 1},
        {"student_forward_passes", 0},
        {"tokens", n_tokens},
        {"replay_tokens", replay_tokens},
        {"only_layer", only_layer},
        {"stability_shards", n_shards},
        {"candidate_families", {"topk/topk", "topk/pivot", "pivot/topk", "pivot/pivot"}},
        {"trust_fractions", {0.25, 0.50, 0.75, 1.0}},
        {"objective", "downstream-column-energy-weighted teacher-feature error; omitted V energy + retained 64D recurrent/RMSNorm/gate replay error"},
        {"diagnostics", diagnostics},
    };
    verify_plan_json(plan);
    write_json(output_plan_path, plan);
    std::cout << "wrote one-pass synchronized GDN joint plan: " << output_plan_path
              << " (tokens=" << n_tokens << ", replay=" << replay_tokens << ")\n";
}

static void command_plan_gdn_geometry_v1(
        const std::string & teacher_path,
        const std::string & base_plan_path,
        const std::string & corpus_path,
        const std::string & output_plan_path,
        int max_tokens,
        int batch_size,
        int requested_shards,
        const std::string & selection_method) {
    if (max_tokens < 64) throw std::runtime_error("GDN geometry v1 requires at least 64 calibration tokens");
    if (batch_size <= 0) throw std::runtime_error("GDN geometry v1 batch size must be positive");
    if (requested_shards <= 0 || requested_shards > 4) {
        throw std::runtime_error("GDN geometry v1 shard count must be in [1,4]");
    }
    if (selection_method != "topk" && selection_method != "pivoted") {
        throw std::runtime_error("GDN geometry v1 selection must be topk or pivoted");
    }

    source_gguf src(teacher_path);
    require_source_contract(src);
    json plan = read_json(base_plan_path);
    verify_plan_json(plan);
    const std::string source_sha = sha256_file(teacher_path);
    if (plan.at("source_sha256").get<std::string>() != source_sha) {
        throw std::runtime_error("base plan source SHA-256 does not match teacher model");
    }

    llama_backend_init();
    auto teacher_model = load_full_model(teacher_path);
    const llama_vocab * vocab = llama_model_get_vocab(teacher_model.get());
    if (!vocab) throw std::runtime_error("GDN geometry v1 teacher has no vocabulary");
    auto tokens = tokenize_bytes(vocab, read_file_bytes(corpus_path));
    if ((int) tokens.size() > max_tokens) tokens.resize((size_t) max_tokens);
    if (tokens.size() < 64) throw std::runtime_error("GDN geometry v1 corpus tokenized to fewer than 64 tokens");

    const int n_tokens = (int) tokens.size();
    const int n_shards = std::max(1, std::min(requested_shards, n_tokens / 64));
    batch_size = std::min(batch_size, n_tokens);
    gdn_geometry_capture capture(n_tokens, n_shards);
    auto ctx = make_gdn_geometry_capture_context(teacher_model.get(), capture, n_tokens, batch_size);
    for (int start = 0; start < n_tokens; start += batch_size) {
        const int n = std::min(batch_size, n_tokens - start);
        capture.current_start = start;
        decode_capture_batch(ctx.get(), tokens, start, n);
        std::cerr << "GDN geometry v1 teacher pass: " << (start + n) << "/" << n_tokens << " tokens\n";
    }
    ctx.reset();
    teacher_model.reset();
    llama_backend_free();

    json layer_diagnostics = json::array();
    for (auto & rec : plan.at("gdn")) {
        const int layer = rec.at("layer").get<int>();
        auto & stats = capture.layer[(size_t) layer];
        uint64_t observed = 0;
        for (const auto & shard : stats.shards) observed += shard.samples;
        if (observed != (uint64_t) n_tokens) {
            throw std::runtime_error("GDN geometry v1 callback missed tokens at layer " + std::to_string(layer));
        }

        const auto prior = compute_gdn_geometry_prior(src, layer);
        json qk_diag = json::array();
        json qk_indices = json::array();
        for (int h = 0; h < GDN_QK_HEADS; ++h) {
            uint64_t samples = 0;
            auto moment = gdn_geometry_materialize_moment(stats, h, -1, samples);
            gdn_geometry_shrink_result shrink;
            auto selected = gdn_geometry_select(moment, samples, prior.qk[(size_t) h], selection_method, &shrink);
            qk_indices.push_back(selected);
            qk_diag.push_back({
                {"head", h},
                {"samples", samples},
                {"shrinkage", shrink.shrinkage},
                {"data_trace", shrink.data_trace},
                {"prior_trace_before_trace_match", shrink.prior_trace},
                {"stability", gdn_geometry_stability_json(stats, h, prior.qk[(size_t) h], selection_method, selected)},
            });
        }

        uint64_t v_samples = 0;
        auto v_moment = gdn_geometry_materialize_moment(stats, -1, -1, v_samples);
        gdn_geometry_shrink_result v_shrink;
        auto v_selected = gdn_geometry_select(v_moment, v_samples, prior.v, selection_method, &v_shrink);

        rec["qk_indices_by_head"] = qk_indices;
        rec["v_indices"] = v_selected;
        rec["initialization"] = "teacher-one-pass-shrunk-geometry-v1-" + selection_method;
        rec["initialization_note"] =
                "One teacher pass; post-SiLU second moments; diagonal weight/FIR prior; "
                "automatic n_dim/(n+n_dim) shrinkage; deterministic coordinate selection. "
                "No student replay, output regression, iterative search, or OOS feedback.";
        layer_diagnostics.push_back({
            {"layer", layer},
            {"qk", qk_diag},
            {"v", {
                {"samples", v_samples},
                {"shrinkage", v_shrink.shrinkage},
                {"data_trace", v_shrink.data_trace},
                {"prior_trace_before_trace_match", v_shrink.prior_trace},
                {"stability", gdn_geometry_stability_json(stats, -1, prior.v, selection_method, v_selected)},
            }},
        });
        std::cerr << "GDN geometry v1 planned layer " << layer << "\n";
    }

    plan["gdn_geometry_v1"] = {
        {"method", "teacher-one-pass-shrunk-second-moment-v1"},
        {"teacher_forward_passes", 1},
        {"student_forward_passes", 0},
        {"calibration_corpus_sha256", sha256_file(corpus_path)},
        {"tokens", n_tokens},
        {"batch_size", batch_size},
        {"stability_shards", n_shards},
        {"shrinkage_rule", "lambda = 128 / (samples + 128); diagonal prior trace-matched to empirical second moment"},
        {"qk_statistic", "0.5 * E[qhat qhat^T + khat khat^T] per Q/K head, captured directly from stock q_conv_predelta/k_conv_predelta"},
        {"v_statistic", "mean over 32 V heads of E[v v^T], captured from stock v_conv_predelta; gate importance is included only through the static diagonal weight prior to reduce calibration sensitivity"},
        {"selection", selection_method},
        {"diagnostics", layer_diagnostics},
    };
    verify_plan_json(plan);
    write_json(output_plan_path, plan);
    std::cout << "wrote one-pass GDN geometry plan: " << output_plan_path
              << " (tokens=" << n_tokens << ", shards=" << n_shards << ")\n";
}

static llama_context_ptr make_function_risk_context(llama_model * model, int n_ctx, int n_batch) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) n_ctx;
    cp.n_batch = (uint32_t) n_batch;
    cp.n_ubatch = (uint32_t) n_batch;
    cp.n_seq_max = 1;
    cp.n_threads = std::max(1, std::min(8, (int) std::thread::hardware_concurrency()));
    cp.n_threads_batch = cp.n_threads;
    llama_context_ptr ctx(llama_init_from_model(model, cp));
    if (!ctx) throw std::runtime_error("failed to create global function-risk context");
    return ctx;
}

static void decode_all_logits_batch(
        llama_context * ctx,
        const std::vector<llama_token> & tokens,
        int start,
        int n) {
    llama_batch batch = llama_batch_init(n, 0, 1);
    common_batch_clear(batch);
    for (int i = 0; i < n; ++i) {
        common_batch_add(batch, tokens[(size_t) start + i], start + i, {0}, true);
    }
    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) throw std::runtime_error("global function-risk llama_decode failed: " + std::to_string(rc));
}

struct ppl_score_result {
    uint64_t samples = 0;
    double nll = 0.0;
    double ppl = 0.0;
};

static double logits_nll(const float * logits, int vocab_size, int target) {
    if (!logits || vocab_size <= 0 || target < 0 || target >= vocab_size) {
        throw std::runtime_error("invalid logits for PPL scoring");
    }
    double max_logit = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < vocab_size; ++i) {
        const double x = logits[i];
        if (!std::isfinite(x)) throw std::runtime_error("non-finite logit in PPL scoring");
        max_logit = std::max(max_logit, x);
    }
    long double sum = 0.0;
    for (int i = 0; i < vocab_size; ++i) sum += std::exp((double) logits[i] - max_logit);
    return max_logit + std::log((double) sum) - (double) logits[target];
}

static ppl_score_result measure_model_ppl(
        llama_context * ctx,
        const llama_model * model,
        const std::vector<llama_token> & tokens,
        int score_from_position,
        int n_batch) {
    if (!ctx || !model || tokens.size() < 2 || n_batch <= 0 ||
        score_from_position < 0 || score_from_position >= (int) tokens.size() - 1) {
        throw std::runtime_error("invalid PPL scoring request");
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    if (!vocab) throw std::runtime_error("PPL model has no vocabulary");
    const int vocab_size = llama_vocab_n_tokens(vocab);
    std::vector<llama_token> decode_tokens = tokens;
    if (llama_vocab_get_add_bos(vocab)) decode_tokens[0] = llama_vocab_bos(vocab);
    llama_memory_clear(llama_get_memory(ctx), true);

    long double nll_sum = 0.0;
    uint64_t samples = 0;
    for (int start = 0; start < (int) tokens.size(); start += n_batch) {
        const int n = std::min(n_batch, (int) tokens.size() - start);
        decode_all_logits_batch(ctx, decode_tokens, start, n);
        for (int i = 0; i < n; ++i) {
            const int pos = start + i;
            if (pos + 1 >= (int) tokens.size()) break;
            if (pos < score_from_position) continue;
            const float * logits = llama_get_logits_ith(ctx, i);
            if (!logits) throw std::runtime_error("PPL logits are unavailable");
            nll_sum += logits_nll(logits, vocab_size, tokens[(size_t) pos + 1]);
            ++samples;
        }
    }
    if (samples == 0) throw std::runtime_error("PPL scorer produced no samples");
    ppl_score_result out;
    out.samples = samples;
    out.nll = (double) (nll_sum / samples);
    out.ppl = std::exp(std::min(700.0, out.nll));
    return out;
}

static void command_score_ppl(
        const std::string & model_path,
        const std::string & corpus_path,
        int max_tokens,
        int score_from_position,
        int batch_size) {
    if (max_tokens < 2) throw std::runtime_error("score-ppl needs at least two tokens");
    if (batch_size <= 0) throw std::runtime_error("score-ppl batch size must be positive");
    llama_backend_init();
    auto model = load_full_model(model_path);
    const llama_vocab * vocab = llama_model_get_vocab(model.get());
    if (!vocab) throw std::runtime_error("score-ppl model has no vocabulary");
    auto tokens = tokenize_bytes(vocab, read_file_bytes(corpus_path));
    if ((int) tokens.size() > max_tokens) tokens.resize((size_t) max_tokens);
    if (tokens.size() < 2) throw std::runtime_error("score-ppl corpus tokenized to fewer than two tokens");
    if (score_from_position < 0) score_from_position = std::min(32, (int) tokens.size() / 4);
    if (score_from_position >= (int) tokens.size() - 1) {
        throw std::runtime_error("score-ppl score-from leaves no targets");
    }
    batch_size = std::min(batch_size, (int) tokens.size());
    auto ctx = make_function_risk_context(model.get(), (int) tokens.size(), batch_size);
    const auto score = measure_model_ppl(ctx.get(), model.get(), tokens, score_from_position, batch_size);
    ctx.reset();
    model.reset();
    llama_backend_free();
    std::cout << "PPL score: samples=" << score.samples
              << " nll_nats=" << score.nll
              << " ppl=" << score.ppl
              << " tokens=" << tokens.size()
              << " score_from=" << score_from_position << "\n";
}

struct gdn_global_function_risk_result {
    uint64_t samples = 0;
    double student_nll = 0.0;
    double student_ppl = 0.0;
    double teacher_conditional_kl = 0.0;
    double teacher_conditional_entropy = 0.0;
    double teacher_retained_mass = 0.0;
    double top1_agreement = 0.0;
};

struct gdn_teacher_risk_cache {
    int student_vocab_size = 0;
    int score_from_position = -1;
    double distill_temperature = 1.0;
    std::vector<int> positions;
    std::vector<gdn_prepared_observable_teacher> sample;
};

static gdn_teacher_risk_cache prepare_gdn_teacher_risk_cache(
        llama_context * teacher_ctx,
        const llama_model * teacher_model,
        const llama_model * student_model,
        const std::vector<llama_token> & teacher_tokens,
        const std::vector<llama_token> & student_tokens,
        const std::vector<int> & vocab_map,
        double distill_temperature,
        int score_from_position,
        int n_batch) {
    if (!teacher_ctx || !teacher_model || !student_model || teacher_tokens.size() != student_tokens.size() ||
        student_tokens.size() < 2 || score_from_position < 0 ||
        score_from_position >= (int) student_tokens.size() - 1 || n_batch <= 0) {
        throw std::runtime_error("invalid teacher observable-risk cache request");
    }
    const llama_vocab * teacher_vocab = llama_model_get_vocab(teacher_model);
    const llama_vocab * student_vocab = llama_model_get_vocab(student_model);
    if (!teacher_vocab || !student_vocab) throw std::runtime_error("teacher observable-risk cache model has no vocabulary");
    const int teacher_vocab_size = llama_vocab_n_tokens(teacher_vocab);
    const int student_vocab_size = llama_vocab_n_tokens(student_vocab);
    if ((int) vocab_map.size() != student_vocab_size) {
        throw std::runtime_error("teacher observable-risk cache vocabulary map size mismatch");
    }

    std::vector<llama_token> decode_tokens = teacher_tokens;
    if (llama_vocab_get_add_bos(teacher_vocab)) decode_tokens[0] = llama_vocab_bos(teacher_vocab);
    llama_memory_clear(llama_get_memory(teacher_ctx), true);

    gdn_teacher_risk_cache out;
    out.student_vocab_size = student_vocab_size;
    out.score_from_position = score_from_position;
    out.distill_temperature = distill_temperature;
    out.positions.reserve(student_tokens.size() - (size_t) score_from_position);
    out.sample.reserve(student_tokens.size() - (size_t) score_from_position);
    for (int start = 0; start < (int) student_tokens.size(); start += n_batch) {
        const int n = std::min(n_batch, (int) student_tokens.size() - start);
        decode_all_logits_batch(teacher_ctx, decode_tokens, start, n);
        for (int i = 0; i < n; ++i) {
            const int pos = start + i;
            if (pos + 1 >= (int) student_tokens.size()) break;
            if (pos < score_from_position) continue;
            const float * teacher_logits = llama_get_logits_ith(teacher_ctx, i);
            if (!teacher_logits) throw std::runtime_error("teacher observable-risk cache logits are unavailable");
            out.positions.push_back(pos);
            out.sample.push_back(gdn_prepare_observable_teacher(
                    teacher_logits, teacher_vocab_size, student_vocab_size,
                    vocab_map.data(), distill_temperature));
        }
    }
    if (out.sample.empty()) throw std::runtime_error("teacher observable-risk cache has no samples");
    return out;
}

static gdn_global_function_risk_result measure_gdn_global_function_risk_cached_teacher(
        llama_context * student_ctx,
        const llama_model * student_model,
        const std::vector<llama_token> & student_tokens,
        const gdn_teacher_risk_cache & teacher_cache,
        int n_batch) {
    if (!student_ctx || !student_model || student_tokens.size() < 2 || n_batch <= 0 ||
        teacher_cache.sample.size() != teacher_cache.positions.size() || teacher_cache.sample.empty()) {
        throw std::runtime_error("invalid cached-teacher observable-risk request");
    }
    const llama_vocab * student_vocab = llama_model_get_vocab(student_model);
    if (!student_vocab || llama_vocab_n_tokens(student_vocab) != teacher_cache.student_vocab_size) {
        throw std::runtime_error("cached-teacher observable-risk student vocabulary mismatch");
    }

    std::vector<llama_token> decode_tokens = student_tokens;
    if (llama_vocab_get_add_bos(student_vocab)) decode_tokens[0] = llama_vocab_bos(student_vocab);
    llama_memory_clear(llama_get_memory(student_ctx), true);
    gdn_observable_risk_sum total;
    size_t cache_index = 0;
    for (int start = 0; start < (int) student_tokens.size(); start += n_batch) {
        const int n = std::min(n_batch, (int) student_tokens.size() - start);
        decode_all_logits_batch(student_ctx, decode_tokens, start, n);
        for (int i = 0; i < n; ++i) {
            const int pos = start + i;
            if (pos + 1 >= (int) student_tokens.size()) break;
            if (pos < teacher_cache.score_from_position) continue;
            if (cache_index >= teacher_cache.sample.size() || teacher_cache.positions[cache_index] != pos) {
                throw std::runtime_error("cached-teacher observable-risk position mismatch");
            }
            const float * student_logits = llama_get_logits_ith(student_ctx, i);
            if (!student_logits) throw std::runtime_error("cached-teacher student logits are unavailable");
            total.add(gdn_measure_observable_risk_prepared(
                    teacher_cache.sample[cache_index], student_logits,
                    student_tokens[(size_t) pos + 1]));
            ++cache_index;
        }
    }
    if (cache_index != teacher_cache.sample.size() || total.samples == 0) {
        throw std::runtime_error("cached-teacher observable-risk sample count mismatch");
    }
    gdn_global_function_risk_result result;
    result.samples = total.samples;
    result.student_nll = total.student_nll / total.samples;
    result.student_ppl = std::exp(std::min(700.0, result.student_nll));
    result.teacher_conditional_kl = total.teacher_conditional_kl / total.samples;
    result.teacher_conditional_entropy = total.teacher_conditional_entropy / total.samples;
    result.teacher_retained_mass = total.teacher_retained_mass / total.samples;
    result.top1_agreement = (double) total.top1_agree / total.samples;
    return result;
}

static gdn_global_function_risk_result measure_gdn_global_function_risk(
        llama_context * teacher_ctx,
        llama_context * student_ctx,
        const llama_model * teacher_model,
        const llama_model * student_model,
        const std::vector<llama_token> & teacher_tokens,
        const std::vector<llama_token> & student_tokens,
        const std::vector<int> & vocab_map,
        double distill_temperature,
        int score_from_position,
        int n_batch) {
    if (teacher_tokens.size() != student_tokens.size() || student_tokens.size() < 2) {
        throw std::runtime_error("global function-risk token sequences must have equal length >= 2");
    }
    if (score_from_position < 0 || score_from_position >= (int) student_tokens.size() - 1) {
        throw std::runtime_error("global function-risk score support is empty or invalid");
    }
    const llama_vocab * teacher_vocab = llama_model_get_vocab(teacher_model);
    const llama_vocab * student_vocab = llama_model_get_vocab(student_model);
    if (!teacher_vocab || !student_vocab) throw std::runtime_error("global function-risk model has no vocabulary");
    const int teacher_vocab_size = llama_vocab_n_tokens(teacher_vocab);
    const int student_vocab_size = llama_vocab_n_tokens(student_vocab);
    if ((int) vocab_map.size() != student_vocab_size) {
        throw std::runtime_error("global function-risk plan vocabulary map size does not match student model");
    }

    // Match llama-perplexity's sequence convention: tokenization itself does
    // not add BOS, but if a model requests one the first *input* token in a
    // context window is replaced by BOS. Targets remain the original corpus
    // tokens. This is part of the empirical function measure, not merely an
    // implementation detail for recurrent models.
    std::vector<llama_token> teacher_decode_tokens = teacher_tokens;
    std::vector<llama_token> student_decode_tokens = student_tokens;
    if (llama_vocab_get_add_bos(teacher_vocab)) {
        teacher_decode_tokens[0] = llama_vocab_bos(teacher_vocab);
    }
    if (llama_vocab_get_add_bos(student_vocab)) {
        student_decode_tokens[0] = llama_vocab_bos(student_vocab);
    }

    llama_memory_clear(llama_get_memory(teacher_ctx), true);
    llama_memory_clear(llama_get_memory(student_ctx), true);
    gdn_observable_risk_sum total;
    for (int start = 0; start < (int) student_tokens.size(); start += n_batch) {
        const int n = std::min(n_batch, (int) student_tokens.size() - start);
        decode_all_logits_batch(teacher_ctx, teacher_decode_tokens, start, n);
        decode_all_logits_batch(student_ctx, student_decode_tokens, start, n);
        for (int i = 0; i < n; ++i) {
            const int pos = start + i;
            if (pos + 1 >= (int) student_tokens.size()) break;
            if (pos < score_from_position) continue;
            const float * teacher_logits = llama_get_logits_ith(teacher_ctx, i);
            const float * student_logits = llama_get_logits_ith(student_ctx, i);
            if (!teacher_logits || !student_logits) {
                throw std::runtime_error("global function-risk logits are unavailable");
            }
            total.add(gdn_measure_observable_risk_sample(
                    teacher_logits, teacher_vocab_size,
                    student_logits, student_vocab_size,
                    vocab_map.data(), student_tokens[(size_t) pos + 1], distill_temperature));
        }
    }
    if (total.samples == 0) throw std::runtime_error("global function-risk produced no prediction samples");
    gdn_global_function_risk_result result;
    result.samples = total.samples;
    result.student_nll = total.student_nll / total.samples;
    result.student_ppl = std::exp(std::min(700.0, result.student_nll));
    result.teacher_conditional_kl = total.teacher_conditional_kl / total.samples;
    result.teacher_conditional_entropy = total.teacher_conditional_entropy / total.samples;
    result.teacher_retained_mass = total.teacher_retained_mass / total.samples;
    result.top1_agreement = (double) total.top1_agree / total.samples;
    return result;
}

static json gdn_global_function_risk_json(const gdn_global_function_risk_result & risk) {
    return {
        {"samples", risk.samples},
        {"student_nll_nats", risk.student_nll},
        {"student_ppl", risk.student_ppl},
        {"teacher_conditional_kl_nats", risk.teacher_conditional_kl},
        {"teacher_conditional_entropy_nats", risk.teacher_conditional_entropy},
        {"teacher_retained_mass", risk.teacher_retained_mass},
        {"teacher_student_top1_agreement", risk.top1_agreement},
    };
}

static void command_score_gdn_function_risk(
        const std::string & teacher_path,
        const std::string & student_path,
        const std::string & plan_path,
        const std::string & corpus_path,
        int max_tokens,
        double distill_temperature,
        int score_from_position) {
    if (max_tokens < 2) throw std::runtime_error("global function-risk scorer needs at least two tokens");
    const json plan = read_json(plan_path);
    verify_plan_json(plan);
    const auto vocab_map = plan.at("vocab").at("output_to_input").get<std::vector<int>>();

    llama_backend_init();
    auto teacher_model = load_full_model(teacher_path);
    auto student_model = load_full_model(student_path);
    const llama_vocab * student_vocab = llama_model_get_vocab(student_model.get());
    if (!student_vocab) throw std::runtime_error("student model has no vocabulary");
    auto student_tokens = tokenize_bytes(student_vocab, read_file_bytes(corpus_path));
    if ((int) student_tokens.size() > max_tokens) student_tokens.resize((size_t) max_tokens);
    if (student_tokens.size() < 2) throw std::runtime_error("global function-risk corpus tokenized to fewer than two tokens");
    if (score_from_position < 0) score_from_position = (int) student_tokens.size() / 2;
    std::vector<llama_token> teacher_tokens(student_tokens.size());
    for (size_t i = 0; i < student_tokens.size(); ++i) {
        const llama_token id = student_tokens[i];
        if (id < 0 || (size_t) id >= vocab_map.size()) throw std::runtime_error("student token id missing from plan vocab map");
        teacher_tokens[i] = vocab_map[(size_t) id];
    }
    const int batch_size = std::min<int>(32, (int) student_tokens.size());
    auto teacher_ctx = make_function_risk_context(teacher_model.get(), (int) student_tokens.size(), batch_size);
    auto student_ctx = make_function_risk_context(student_model.get(), (int) student_tokens.size(), batch_size);
    const auto risk = measure_gdn_global_function_risk(
            teacher_ctx.get(), student_ctx.get(), teacher_model.get(), student_model.get(),
            teacher_tokens, student_tokens, vocab_map, distill_temperature, score_from_position, batch_size);
    json out = gdn_global_function_risk_json(risk);
    out["objective"] = "observable-next-token-function-risk-v1";
    out["teacher_distribution"] = "teacher-softmax-conditioned-on-retained-student-vocabulary";
    out["distill_temperature"] = distill_temperature;
    out["score_from_position"] = score_from_position;
    out["empirical_measure"] = "uniform-over-selected-next-token-prefixes";
    out["corpus"] = corpus_path;
    std::cout << std::setw(2) << out << '\n';
    llama_backend_free();
}

struct gdn_function_projection_regression {
    int layer = -1;
    uint64_t train_samples = 0;
    uint64_t validation_samples = 0;
    double ridge_relative = 0.0;
    double ridge_absolute = 0.0;
    double baseline_train_residual_ratio = 0.0;
    double fitted_train_residual_ratio = 0.0;
    double baseline_validation_residual_ratio = 0.0;
    double fitted_validation_residual_ratio = 0.0;
    std::vector<float> weight;
};

static void write_gdn_function_projection_regression(
        const std::string & path,
        const gdn_function_projection_regression & reg) {
    if (reg.weight.size() != (size_t) HIDDEN * GDN_V_HEADS * GDN_DIM_DST) {
        throw std::runtime_error("bad GDN function-projection weight size");
    }
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot write GDN function-projection file: " + path);
    const char magic[8] = {'G','D','N','F','P','R','J','1'};
    const uint32_t version = 1;
    const int32_t layer = reg.layer;
    const uint32_t n_out = HIDDEN;
    const uint32_t n_in = GDN_V_HEADS * GDN_DIM_DST;
    out.write(magic, sizeof(magic));
    out.write((const char *) &version, sizeof(version));
    out.write((const char *) &layer, sizeof(layer));
    out.write((const char *) &n_out, sizeof(n_out));
    out.write((const char *) &n_in, sizeof(n_in));
    out.write((const char *) &reg.train_samples, sizeof(reg.train_samples));
    out.write((const char *) &reg.validation_samples, sizeof(reg.validation_samples));
    out.write((const char *) &reg.ridge_relative, sizeof(reg.ridge_relative));
    out.write((const char *) &reg.ridge_absolute, sizeof(reg.ridge_absolute));
    out.write((const char *) &reg.baseline_train_residual_ratio, sizeof(reg.baseline_train_residual_ratio));
    out.write((const char *) &reg.fitted_train_residual_ratio, sizeof(reg.fitted_train_residual_ratio));
    out.write((const char *) &reg.baseline_validation_residual_ratio, sizeof(reg.baseline_validation_residual_ratio));
    out.write((const char *) &reg.fitted_validation_residual_ratio, sizeof(reg.fitted_validation_residual_ratio));
    out.write((const char *) reg.weight.data(), reg.weight.size() * sizeof(float));
    if (!out) throw std::runtime_error("short write in GDN function-projection file");
}

static gdn_function_projection_regression read_gdn_function_projection_regression(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read GDN function-projection file: " + path);
    char magic[8] = {};
    uint32_t version = 0, n_out = 0, n_in = 0;
    int32_t layer = -1;
    gdn_function_projection_regression reg;
    in.read(magic, sizeof(magic));
    in.read((char *) &version, sizeof(version));
    in.read((char *) &layer, sizeof(layer));
    in.read((char *) &n_out, sizeof(n_out));
    in.read((char *) &n_in, sizeof(n_in));
    in.read((char *) &reg.train_samples, sizeof(reg.train_samples));
    in.read((char *) &reg.validation_samples, sizeof(reg.validation_samples));
    in.read((char *) &reg.ridge_relative, sizeof(reg.ridge_relative));
    in.read((char *) &reg.ridge_absolute, sizeof(reg.ridge_absolute));
    in.read((char *) &reg.baseline_train_residual_ratio, sizeof(reg.baseline_train_residual_ratio));
    in.read((char *) &reg.fitted_train_residual_ratio, sizeof(reg.fitted_train_residual_ratio));
    in.read((char *) &reg.baseline_validation_residual_ratio, sizeof(reg.baseline_validation_residual_ratio));
    in.read((char *) &reg.fitted_validation_residual_ratio, sizeof(reg.fitted_validation_residual_ratio));
    if (std::memcmp(magic, "GDNFPRJ1", sizeof(magic)) != 0 || version != 1 ||
        n_out != HIDDEN || n_in != GDN_V_HEADS * GDN_DIM_DST) {
        throw std::runtime_error("unsupported GDN function-projection file format");
    }
    reg.layer = layer;
    reg.weight.resize((size_t) n_out * n_in);
    in.read((char *) reg.weight.data(), reg.weight.size() * sizeof(float));
    if (!in) throw std::runtime_error("short read in GDN function-projection file");
    return reg;
}

struct gdn_function_capture_set {
    int samples = 0;
    std::vector<float> teacher_output;
    std::vector<float> student_output;
    std::vector<float> student_feature;
};

struct gdn_layer_replay_set {
    int samples = 0;
    std::vector<float> linear_output;
    std::vector<float> final_output;
};

static std::vector<llama_token> gdn_decode_tokens_for_model(
        const llama_model * model,
        const std::vector<llama_token> & tokens) {
    std::vector<llama_token> out = tokens;
    if (out.empty()) return out;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    if (!vocab) throw std::runtime_error("GDN replay model has no vocabulary");
    if (llama_vocab_get_add_bos(vocab)) out[0] = llama_vocab_bos(vocab);
    return out;
}

static gdn_layer_replay_set replay_gdn_layer_sequence(
        llama_context * ctx,
        gdn_layer_output_capture & capture,
        const std::vector<llama_token> & decode_tokens,
        int batch_size,
        int expected_final_width = GDN_V_HEADS * GDN_DIM_DST) {
    if (!ctx || decode_tokens.empty() || batch_size <= 0) {
        throw std::runtime_error("invalid GDN layer replay request");
    }
    llama_memory_clear(llama_get_memory(ctx), true);
    gdn_layer_replay_set out;
    out.samples = (int) decode_tokens.size();
    out.linear_output.reserve((size_t) out.samples * HIDDEN);
    out.final_output.reserve((size_t) out.samples * GDN_V_HEADS * GDN_DIM_DST);
    for (int start = 0; start < out.samples; start += batch_size) {
        const int n = std::min(batch_size, out.samples - start);
        capture.linear_output.clear();
        capture.final_output.clear();
        decode_capture_batch(ctx, decode_tokens, start, n);
        if (!capture.linear_output.empty()) {
            if (capture.linear_output.size() != (size_t) n * HIDDEN) {
                throw std::runtime_error("GDN replay linear-output callback size mismatch");
            }
            out.linear_output.insert(
                    out.linear_output.end(), capture.linear_output.begin(), capture.linear_output.end());
        }
        if (expected_final_width > 0 && !capture.final_output.empty()) {
            if (capture.final_output.size() != (size_t) n * expected_final_width) {
                throw std::runtime_error("GDN replay final-output callback size mismatch");
            }
            out.final_output.insert(
                    out.final_output.end(), capture.final_output.begin(), capture.final_output.end());
        }
    }
    return out;
}

static gdn_function_capture_set capture_gdn_function_set(
        llama_model * teacher_model,
        llama_model * student_model,
        const std::vector<int> & vocab_map,
        const std::string & corpus_path,
        int layer,
        int max_tokens) {
    if (max_tokens <= 0) throw std::runtime_error("GDN function-projection token limit must be positive");
    const llama_vocab * student_vocab = llama_model_get_vocab(student_model);
    if (!student_vocab) throw std::runtime_error("student model has no vocab");
    auto student_tokens = tokenize_bytes(student_vocab, read_file_bytes(corpus_path));
    if (student_tokens.empty()) throw std::runtime_error("GDN function-projection corpus tokenized to empty sequence: " + corpus_path);
    if ((int) student_tokens.size() > max_tokens) student_tokens.resize((size_t) max_tokens);
    std::vector<llama_token> teacher_tokens(student_tokens.size());
    for (size_t i = 0; i < student_tokens.size(); ++i) {
        const llama_token id = student_tokens[i];
        if (id < 0 || (size_t) id >= vocab_map.size()) {
            throw std::runtime_error("student token id missing from plan vocab map");
        }
        teacher_tokens[i] = vocab_map[(size_t) id];
    }

    gdn_layer_output_capture teacher_capture, student_capture;
    teacher_capture.layer = layer;
    student_capture.layer = layer;
    const int batch_size = std::min<int>(32, (int) student_tokens.size());
    auto teacher_ctx = make_layer_output_capture_context(
            teacher_model, teacher_capture, (int) student_tokens.size(), batch_size);
    auto student_ctx = make_layer_output_capture_context(
            student_model, student_capture, (int) student_tokens.size(), batch_size);

    gdn_function_capture_set out;
    out.samples = (int) student_tokens.size();
    out.teacher_output.reserve((size_t) out.samples * HIDDEN);
    out.student_output.reserve((size_t) out.samples * HIDDEN);
    out.student_feature.reserve((size_t) out.samples * GDN_V_HEADS * GDN_DIM_DST);
    for (int start = 0; start < out.samples; start += batch_size) {
        const int n = std::min(batch_size, out.samples - start);
        teacher_capture.final_output.clear();
        teacher_capture.linear_output.clear();
        student_capture.final_output.clear();
        student_capture.linear_output.clear();
        decode_capture_batch(teacher_ctx.get(), teacher_tokens, start, n);
        decode_capture_batch(student_ctx.get(), student_tokens, start, n);
        if (teacher_capture.linear_output.size() != (size_t) n * HIDDEN ||
            student_capture.linear_output.size() != (size_t) n * HIDDEN ||
            student_capture.final_output.size() != (size_t) n * GDN_V_HEADS * GDN_DIM_DST) {
            throw std::runtime_error("GDN function-projection callback size mismatch");
        }
        out.teacher_output.insert(
                out.teacher_output.end(), teacher_capture.linear_output.begin(), teacher_capture.linear_output.end());
        out.student_output.insert(
                out.student_output.end(), student_capture.linear_output.begin(), student_capture.linear_output.end());
        out.student_feature.insert(
                out.student_feature.end(), student_capture.final_output.begin(), student_capture.final_output.end());
        std::cerr << "GDN function-space capture: " << corpus_path << " "
                  << (start + n) << "/" << out.samples << " tokens\n";
    }
    return out;
}

static double relative_residual_ratio(
        const std::vector<float> & target,
        const std::vector<float> & prediction) {
    if (target.size() != prediction.size()) throw std::runtime_error("relative residual shape mismatch");
    long double target_energy = 0.0;
    long double residual_energy = 0.0;
    for (size_t i = 0; i < target.size(); ++i) {
        const long double y = target[i];
        const long double e = y - prediction[i];
        target_energy += y * y;
        residual_energy += e * e;
    }
    return target_energy > 0.0 ? (double) (residual_energy / target_energy) : 0.0;
}

static void command_fit_gdn_function_projection(
        const std::string & teacher_path,
        const std::string & student_path,
        const std::string & plan_path,
        const std::string & train_corpus_path,
        const std::string & validation_corpus_path,
        const std::string & output_path,
        int layer,
        int train_tokens,
        int validation_tokens,
        double ridge_center,
        int ridge_decades,
        int n_threads) {
    if (!is_recurrent_layer(layer)) throw std::runtime_error("GDN function-projection layer is not recurrent");
    if (train_tokens <= 0 || validation_tokens <= 0) {
        throw std::runtime_error("GDN function-projection train/validation token limits must be positive");
    }
    if (!(ridge_center > 0.0) || !std::isfinite(ridge_center)) {
        throw std::runtime_error("GDN function-projection ridge must be finite and positive");
    }
    if (ridge_decades < 0 || ridge_decades > 12) {
        throw std::runtime_error("GDN function-projection ridge decades must be in [0,12]");
    }

    const json plan = read_json(plan_path);
    verify_plan_json(plan);
    const auto vocab_map = plan.at("vocab").at("output_to_input").get<std::vector<int>>();

    llama_backend_init();
    auto teacher_model = load_full_model(teacher_path);
    auto student_model = load_full_model(student_path);
    auto train = capture_gdn_function_set(
            teacher_model.get(), student_model.get(), vocab_map, train_corpus_path, layer, train_tokens);
    auto validation = capture_gdn_function_set(
            teacher_model.get(), student_model.get(), vocab_map, validation_corpus_path, layer, validation_tokens);

    std::vector<double> ridge_path;
    ridge_path.reserve((size_t) 2 * ridge_decades + 1);
    for (int decade = -ridge_decades; decade <= ridge_decades; ++decade) {
        const double ridge = ridge_center * std::pow(10.0, decade);
        if (ridge > 0.0 && std::isfinite(ridge)) ridge_path.push_back(ridge);
    }
    if (ridge_path.empty()) throw std::runtime_error("GDN function-projection ridge path is empty");

    const int n_features = GDN_V_HEADS * GDN_DIM_DST;
    const auto path_fit = gdn_fit_function_projection_dual_path(
            train.student_feature, train.teacher_output, train.samples,
            validation.student_feature, validation.teacher_output, validation.samples,
            n_features, HIDDEN, ridge_path, n_threads);
    const auto & fit = path_fit.best;

    gdn_function_projection_regression reg;
    reg.layer = layer;
    reg.train_samples = (uint64_t) train.samples;
    reg.validation_samples = (uint64_t) validation.samples;
    reg.ridge_relative = fit.ridge_relative;
    reg.ridge_absolute = fit.ridge_absolute;
    reg.baseline_train_residual_ratio = relative_residual_ratio(train.teacher_output, train.student_output);
    reg.fitted_train_residual_ratio = fit.target_energy > 0.0 ? fit.residual_energy / fit.target_energy : 0.0;
    reg.baseline_validation_residual_ratio = relative_residual_ratio(validation.teacher_output, validation.student_output);
    reg.fitted_validation_residual_ratio = path_fit.validation_target_energy > 0.0 ?
            path_fit.validation_residual_energy / path_fit.validation_target_energy : 0.0;
    reg.weight = fit.weight;
    write_gdn_function_projection_regression(output_path, reg);
    json ridge_diagnostics = json::array();
    for (const auto & point : path_fit.path) {
        ridge_diagnostics.push_back({
            {"ridge_relative", point.ridge_relative},
            {"ridge_absolute", point.ridge_absolute},
            {"train_residual_ratio", point.train_residual_ratio},
            {"validation_residual_ratio", point.validation_residual_ratio},
        });
    }
    write_json(output_path + ".json", {
        {"format", "gdn-function-space-variable-projection-v1"},
        {"objective", "empirical-L2-best-stock-linear-readout-of-fixed-student-feature-map"},
        {"layer", layer},
        {"train_samples", reg.train_samples},
        {"validation_samples", reg.validation_samples},
        {"ridge_relative", reg.ridge_relative},
        {"ridge_absolute", reg.ridge_absolute},
        {"baseline_train_residual_ratio", reg.baseline_train_residual_ratio},
        {"fitted_train_residual_ratio", reg.fitted_train_residual_ratio},
        {"train_explained_fraction", 1.0 - reg.fitted_train_residual_ratio},
        {"baseline_validation_residual_ratio", reg.baseline_validation_residual_ratio},
        {"fitted_validation_residual_ratio", reg.fitted_validation_residual_ratio},
        {"validation_explained_fraction", 1.0 - reg.fitted_validation_residual_ratio},
        {"ridge_center", ridge_center},
        {"ridge_decades", ridge_decades},
        {"ridge_path", ridge_diagnostics},
        {"plan_sha256", sha256_file(plan_path)},
        {"teacher_identity", file_identity(teacher_path)},
        {"student_identity", file_identity(student_path)},
    });
    llama_backend_free();
    std::cout << "GDN function projection layer " << layer
              << ": train " << reg.baseline_train_residual_ratio << " -> " << reg.fitted_train_residual_ratio
              << ", validation " << reg.baseline_validation_residual_ratio << " -> "
              << reg.fitted_validation_residual_ratio << "\n"
              << "wrote GDN function projection: " << output_path << "\n";
}

static void command_rewrite_gdn_function_projection(
        const std::string & reg_path,
        const std::string & target_path,
        int n_threads) {
    if (!ends_with(target_path, ".tmp")) throw std::runtime_error("GDN function-projection target must end in .tmp");
    const auto reg = read_gdn_function_projection_regression(reg_path);
    source_gguf target(target_path);
    const std::string name = "blk." + std::to_string(reg.layer) + ".ssm_out.weight";
    const ggml_tensor * t = target.tensor(name);
    const auto * traits = ggml_get_type_traits(t->type);
    if (!traits || !traits->from_float_ref || t->ne[0] != GDN_V_HEADS * GDN_DIM_DST || t->ne[1] != HIDDEN) {
        throw std::runtime_error("GDN function-projection target ssm_out cannot be rewritten");
    }
    const int64_t tid = gguf_find_tensor(target.meta(), name.c_str());
    output_writer writer(target_path, target.meta(), /*initialize=*/ false);
    const size_t row_size = ggml_row_size(t->type, t->ne[0]);
    std::atomic<int> next_row {0};
    std::atomic<bool> failed {false};
    std::exception_ptr error;
    std::mutex error_mutex;
    auto worker = [&]() {
        try {
            std::vector<uint8_t> packed(row_size);
            while (!failed.load(std::memory_order_relaxed)) {
                const int row = next_row.fetch_add(1);
                if (row >= HIDDEN) break;
                const float * src = reg.weight.data() + (size_t) row * t->ne[0];
                traits->from_float_ref(src, packed.data(), t->ne[0]);
                writer.write_tensor(tid, (size_t) row * row_size, packed.data(), packed.size());
            }
        } catch (...) {
            failed.store(true, std::memory_order_relaxed);
            std::lock_guard<std::mutex> lock(error_mutex);
            if (!error) error = std::current_exception();
        }
    };
    const int workers = std::max(1, std::min(n_threads, HIDDEN));
    std::vector<std::thread> threads;
    threads.reserve((size_t) workers);
    for (int i = 0; i < workers; ++i) threads.emplace_back(worker);
    for (auto & th : threads) th.join();
    if (error) std::rethrow_exception(error);
    writer.finish();

    const std::string manifest_path = target_path + ".manifest.json";
    json manifest = json::object();
    std::ifstream in(manifest_path);
    if (in.good()) {
        in.close();
        manifest = read_json(manifest_path);
    }
    manifest["gdn_function_projection"] = "empirical-L2-variable-projection-v1";
    manifest["gdn_function_projection_last_layer"] = reg.layer;
    manifest["gdn_function_projection_file_sha256"] = sha256_file(reg_path);
    manifest["gdn_function_projection_train_samples"] = reg.train_samples;
    manifest["gdn_function_projection_validation_samples"] = reg.validation_samples;
    manifest["gdn_function_projection_train_residual_ratio"] = reg.fitted_train_residual_ratio;
    manifest["gdn_function_projection_validation_residual_ratio"] = reg.fitted_validation_residual_ratio;
    manifest["output_sha256"] = sha256_file(target_path);
    write_json(manifest_path, manifest);
    std::cout << "rewrote GDN function-space readout layer " << reg.layer << ": " << target_path << "\n";
}

static std::vector<uint8_t> pack_gdn_loaded_ssm_out(
        const ggml_tensor * tensor,
        const std::vector<float> & weight,
        int n_threads) {
    const int n_features = GDN_V_HEADS * GDN_DIM_DST;
    if (!tensor || tensor->ne[0] != n_features || tensor->ne[1] != HIDDEN ||
        weight.size() != (size_t) HIDDEN * n_features) {
        throw std::runtime_error("loaded GDN ssm_out shape mismatch");
    }
    const auto * traits = ggml_get_type_traits(tensor->type);
    if (!traits || !traits->from_float_ref) {
        throw std::runtime_error("loaded GDN ssm_out type cannot be quantized");
    }
    const size_t row_size = ggml_row_size(tensor->type, n_features);
    std::vector<uint8_t> packed(row_size * HIDDEN);
    std::atomic<int> next_row {0};
    const int workers = std::max(1, std::min(n_threads, HIDDEN));
    std::vector<std::thread> threads;
    threads.reserve((size_t) workers);
    for (int w = 0; w < workers; ++w) {
        threads.emplace_back([&]() {
            for (;;) {
                const int row = next_row.fetch_add(1);
                if (row >= HIDDEN) break;
                traits->from_float_ref(
                        weight.data() + (size_t) row * n_features,
                        packed.data() + (size_t) row * row_size,
                        n_features);
            }
        });
    }
    for (auto & th : threads) th.join();
    return packed;
}

struct backend_tensor_restore_guard {
    ggml_tensor * tensor = nullptr;
    std::vector<uint8_t> bytes;

    explicit backend_tensor_restore_guard(ggml_tensor * t) : tensor(t) {
        if (!tensor) throw std::runtime_error("cannot guard null backend tensor");
        bytes.resize(ggml_nbytes(tensor));
        ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
    }

    ~backend_tensor_restore_guard() {
        restore_now();
    }

    void restore_now() noexcept {
        if (tensor && !bytes.empty()) {
            ggml_backend_tensor_set(tensor, bytes.data(), 0, bytes.size());
            tensor = nullptr;
            bytes.clear();
        }
    }

    backend_tensor_restore_guard(const backend_tensor_restore_guard &) = delete;
    backend_tensor_restore_guard & operator=(const backend_tensor_restore_guard &) = delete;
};

static std::vector<int> gdn_qk_indices_for_head(const json & rec, int head) {
    if (head < 0 || head >= GDN_QK_HEADS) throw std::runtime_error("GDN Q/K head out of range");
    const auto out = rec.at("qk_indices_by_head")[(size_t) head].get<std::vector<int>>();
    if (out.size() != GDN_DIM_DST || std::set<int>(out.begin(), out.end()).size() != GDN_DIM_DST ||
        *std::min_element(out.begin(), out.end()) < 0 || *std::max_element(out.begin(), out.end()) >= GDN_DIM_SRC) {
        throw std::runtime_error("invalid explicit GDN Q/K coordinate set");
    }
    return out;
}

struct gdn_loaded_qk_atom_editor {
    const source_gguf & teacher;
    ggml_tensor * student_qkv = nullptr;
    ggml_tensor * student_conv = nullptr;
    std::string qkv_name;
    std::string conv_name;
    int head = -1;
    size_t qkv_row_size = 0;
    size_t conv_row_size = 0;
    std::vector<uint8_t> zero_qkv_row;
    std::vector<uint8_t> zero_conv_row;

    gdn_loaded_qk_atom_editor(
            const source_gguf & teacher_,
            llama_model * student_model,
            int layer,
            int head_)
        : teacher(teacher_),
          qkv_name("blk." + std::to_string(layer) + ".attn_qkv.weight"),
          conv_name("blk." + std::to_string(layer) + ".ssm_conv1d.weight"),
          head(head_) {
        if (!student_model || !is_recurrent_layer(layer) || head < 0 || head >= GDN_QK_HEADS) {
            throw std::runtime_error("invalid GDN Q/K atom editor target");
        }
        student_qkv = student_model->layers[(size_t) layer].wqkv;
        student_conv = student_model->layers[(size_t) layer].ssm_conv1d;
        if (!student_qkv || !student_conv) throw std::runtime_error("loaded student is missing Q/K atom tensors");

        const ggml_tensor * teacher_qkv = teacher.tensor(qkv_name);
        const ggml_tensor * teacher_conv = teacher.tensor(conv_name);
        if (teacher_qkv->ne[0] != HIDDEN || teacher_qkv->ne[1] != 8192 ||
            student_qkv->ne[0] != HIDDEN || student_qkv->ne[1] != 4096 ||
            teacher_qkv->type != student_qkv->type) {
            throw std::runtime_error("teacher/student QKV layouts are not exact-row compatible");
        }
        if (teacher_conv->ne[0] != 4 || teacher_conv->ne[1] != 8192 ||
            student_conv->ne[0] != 4 || student_conv->ne[1] != 4096 ||
            teacher_conv->type != student_conv->type) {
            throw std::runtime_error("teacher/student depthwise-conv layouts are not exact-row compatible");
        }
        qkv_row_size = ggml_row_size(student_qkv->type, HIDDEN);
        conv_row_size = ggml_row_size(student_conv->type, 4);
        if (student_qkv->nb[1] != qkv_row_size || teacher_qkv->nb[1] != qkv_row_size ||
            student_conv->nb[1] != conv_row_size || teacher_conv->nb[1] != conv_row_size) {
            throw std::runtime_error("GDN Q/K atom tensors are not row contiguous");
        }

        const auto * qkv_traits = ggml_get_type_traits(student_qkv->type);
        if (!qkv_traits || !qkv_traits->from_float_ref) {
            throw std::runtime_error("GDN QKV type cannot represent a zero atom");
        }
        std::vector<float> zero_projection(HIDDEN, 0.0f);
        zero_qkv_row.resize(qkv_row_size);
        qkv_traits->from_float_ref(zero_projection.data(), zero_qkv_row.data(), HIDDEN);
        zero_conv_row.assign(conv_row_size, 0);
    }

    void set_slot_from_teacher(int slot, int teacher_coordinate) const {
        if (slot < 0 || slot >= GDN_DIM_DST || teacher_coordinate < 0 || teacher_coordinate >= GDN_DIM_SRC) {
            throw std::runtime_error("GDN Q/K atom slot/coordinate out of range");
        }
        std::vector<uint8_t> qkv(qkv_row_size);
        std::vector<uint8_t> conv(conv_row_size);
        const int student_q_row = head * GDN_DIM_DST + slot;
        const int student_k_row = GDN_QK_HEADS * GDN_DIM_DST + head * GDN_DIM_DST + slot;
        const int teacher_q_row = head * GDN_DIM_SRC + teacher_coordinate;
        const int teacher_k_row = GDN_QK_HEADS * GDN_DIM_SRC + head * GDN_DIM_SRC + teacher_coordinate;

        teacher.read_tensor_bytes(qkv_name, (size_t) teacher_q_row * qkv_row_size, qkv.data(), qkv.size());
        ggml_backend_tensor_set(student_qkv, qkv.data(), (size_t) student_q_row * qkv_row_size, qkv.size());
        teacher.read_tensor_bytes(qkv_name, (size_t) teacher_k_row * qkv_row_size, qkv.data(), qkv.size());
        ggml_backend_tensor_set(student_qkv, qkv.data(), (size_t) student_k_row * qkv_row_size, qkv.size());

        teacher.read_tensor_bytes(conv_name, (size_t) teacher_q_row * conv_row_size, conv.data(), conv.size());
        ggml_backend_tensor_set(student_conv, conv.data(), (size_t) student_q_row * conv_row_size, conv.size());
        teacher.read_tensor_bytes(conv_name, (size_t) teacher_k_row * conv_row_size, conv.data(), conv.size());
        ggml_backend_tensor_set(student_conv, conv.data(), (size_t) student_k_row * conv_row_size, conv.size());
    }

    void zero_slot(int slot) const {
        if (slot < 0 || slot >= GDN_DIM_DST) throw std::runtime_error("GDN Q/K atom slot out of range");
        const int student_q_row = head * GDN_DIM_DST + slot;
        const int student_k_row = GDN_QK_HEADS * GDN_DIM_DST + head * GDN_DIM_DST + slot;
        ggml_backend_tensor_set(student_qkv, zero_qkv_row.data(), (size_t) student_q_row * qkv_row_size, zero_qkv_row.size());
        ggml_backend_tensor_set(student_qkv, zero_qkv_row.data(), (size_t) student_k_row * qkv_row_size, zero_qkv_row.size());
        ggml_backend_tensor_set(student_conv, zero_conv_row.data(), (size_t) student_q_row * conv_row_size, zero_conv_row.size());
        ggml_backend_tensor_set(student_conv, zero_conv_row.data(), (size_t) student_k_row * conv_row_size, zero_conv_row.size());
    }

    void set_head_from_teacher(const std::vector<int> & coordinates) const {
        if (coordinates.size() != GDN_DIM_DST) throw std::runtime_error("GDN Q/K atom head size mismatch");
        for (int slot = 0; slot < GDN_DIM_DST; ++slot) {
            set_slot_from_teacher(slot, coordinates[(size_t) slot]);
        }
    }
};

static void set_loaded_gdn_ssm_out(
        ggml_tensor * tensor,
        const std::vector<float> & weight,
        int n_threads) {
    auto packed = pack_gdn_loaded_ssm_out(tensor, weight, n_threads);
    if (packed.size() != ggml_nbytes(tensor)) {
        throw std::runtime_error("packed loaded GDN ssm_out byte size mismatch");
    }
    ggml_backend_tensor_set(tensor, packed.data(), 0, packed.size());
}

static void prepare_gdn_global_tokens(
        const llama_model * student_model,
        const std::vector<int> & vocab_map,
        const std::string & corpus_path,
        int max_tokens,
        std::vector<llama_token> & student_tokens,
        std::vector<llama_token> & teacher_tokens) {
    const llama_vocab * student_vocab = llama_model_get_vocab(student_model);
    if (!student_vocab) throw std::runtime_error("student model has no vocabulary");
    student_tokens = tokenize_bytes(student_vocab, read_file_bytes(corpus_path));
    if ((int) student_tokens.size() > max_tokens) student_tokens.resize((size_t) max_tokens);
    if (student_tokens.size() < 2) throw std::runtime_error("global function-risk corpus tokenized to fewer than two tokens");
    teacher_tokens.resize(student_tokens.size());
    for (size_t i = 0; i < student_tokens.size(); ++i) {
        const llama_token id = student_tokens[i];
        if (id < 0 || (size_t) id >= vocab_map.size()) {
            throw std::runtime_error("student token id missing from plan vocab map");
        }
        teacher_tokens[i] = vocab_map[(size_t) id];
    }
}

static double gdn_global_selection_value(
        const gdn_global_function_risk_result & risk,
        const std::string & objective) {
    if (objective == "kl") return risk.teacher_conditional_kl;
    if (objective == "nll") return risk.student_nll;
    throw std::runtime_error("global function objective must be 'kl' or 'nll'");
}

struct gdn_reduced_core_risk_result {
    gdn_global_function_risk_result global;
    double selection_value = std::numeric_limits<double>::infinity();
    bool selected_current_readout = true;
    double ridge_relative = 0.0;
    double ridge_absolute = 0.0;
    double baseline_train_residual_ratio = 0.0;
    double baseline_validation_residual_ratio = 0.0;
    double selected_train_residual_ratio = 0.0;
    double selected_validation_residual_ratio = 0.0;
    json readout_candidates = json::array();
};

struct gdn_reduced_readout_oracle {
    llama_model * student_model = nullptr;
    llama_context * student_train_ctx = nullptr;
    llama_context * student_validation_ctx = nullptr;
    llama_context * student_risk_ctx = nullptr;
    gdn_layer_output_capture * student_train_capture = nullptr;
    gdn_layer_output_capture * student_validation_capture = nullptr;
    ggml_tensor * student_ssm_out = nullptr;
    const std::vector<float> * teacher_train_output = nullptr;
    const std::vector<float> * teacher_validation_output = nullptr;
    const std::vector<llama_token> * student_train_decode_tokens = nullptr;
    const std::vector<llama_token> * student_validation_decode_tokens = nullptr;
    const std::vector<llama_token> * student_risk_tokens = nullptr;
    const gdn_teacher_risk_cache * teacher_risk_cache = nullptr;
    const std::vector<uint8_t> * current_readout_bytes = nullptr;
    const std::vector<double> * ridge_path = nullptr;
    std::string objective;
    double distill_temperature = 1.0;
    int score_from_position = -1;
    int train_batch_size = 0;
    int validation_batch_size = 0;
    int risk_batch_size = 0;
    int n_threads = 1;

    void restore_current_readout() const {
        if (!student_ssm_out || !current_readout_bytes ||
            current_readout_bytes->size() != ggml_nbytes(student_ssm_out)) {
            throw std::runtime_error("GDN reduced oracle has invalid current readout bytes");
        }
        ggml_backend_tensor_set(
                student_ssm_out, current_readout_bytes->data(), 0, current_readout_bytes->size());
    }

    gdn_reduced_core_risk_result evaluate() const {
        if (!student_model || !student_train_ctx || !student_validation_ctx ||
            !student_risk_ctx || !student_train_capture ||
            !student_validation_capture || !student_ssm_out || !teacher_train_output ||
            !teacher_validation_output || !student_train_decode_tokens ||
            !student_validation_decode_tokens || !student_risk_tokens || !teacher_risk_cache ||
            !current_readout_bytes || !ridge_path || ridge_path->empty()) {
            throw std::runtime_error("GDN reduced oracle is incomplete");
        }
        const int n_features = GDN_V_HEADS * GDN_DIM_DST;
        restore_current_readout();
        try {
            const auto train = replay_gdn_layer_sequence(
                    student_train_ctx, *student_train_capture,
                    *student_train_decode_tokens, train_batch_size);
            const auto validation = replay_gdn_layer_sequence(
                    student_validation_ctx, *student_validation_capture,
                    *student_validation_decode_tokens, validation_batch_size);
            if (train.final_output.size() != (size_t) train.samples * n_features ||
                validation.final_output.size() != (size_t) validation.samples * n_features ||
                train.linear_output.size() != (size_t) train.samples * HIDDEN ||
                validation.linear_output.size() != (size_t) validation.samples * HIDDEN ||
                teacher_train_output->size() != (size_t) train.samples * HIDDEN ||
                teacher_validation_output->size() != (size_t) validation.samples * HIDDEN) {
                throw std::runtime_error("GDN reduced oracle capture shape mismatch");
            }

            gdn_reduced_core_risk_result out;
            out.baseline_train_residual_ratio = relative_residual_ratio(
                    *teacher_train_output, train.linear_output);
            out.baseline_validation_residual_ratio = relative_residual_ratio(
                    *teacher_validation_output, validation.linear_output);

            const auto current_global = measure_gdn_global_function_risk_cached_teacher(
                    student_risk_ctx, student_model, *student_risk_tokens,
                    *teacher_risk_cache, risk_batch_size);
            out.global = current_global;
            out.selection_value = gdn_global_selection_value(current_global, objective);
            out.selected_current_readout = true;
            out.selected_train_residual_ratio = out.baseline_train_residual_ratio;
            out.selected_validation_residual_ratio = out.baseline_validation_residual_ratio;
            json current_json = gdn_global_function_risk_json(current_global);
            current_json["readout"] = "current-student";
            current_json["selection_value"] = out.selection_value;
            current_json["local_train_residual_ratio"] = out.baseline_train_residual_ratio;
            current_json["local_validation_residual_ratio"] = out.baseline_validation_residual_ratio;
            out.readout_candidates.push_back(std::move(current_json));

            const auto path_fit = gdn_fit_function_projection_dual_path(
                    train.final_output, *teacher_train_output, train.samples,
                    validation.final_output, *teacher_validation_output, validation.samples,
                    n_features, HIDDEN, *ridge_path, n_threads);
            for (size_t i = 0; i < path_fit.path.size(); ++i) {
                auto weight = gdn_materialize_function_projection_weight(
                        train.final_output, path_fit.dual_path.at(i),
                        train.samples, n_features, HIDDEN, n_threads);
                set_loaded_gdn_ssm_out(student_ssm_out, weight, n_threads);
                const auto global = measure_gdn_global_function_risk_cached_teacher(
                        student_risk_ctx, student_model, *student_risk_tokens,
                        *teacher_risk_cache, risk_batch_size);
                const double value = gdn_global_selection_value(global, objective);
                const auto & local = path_fit.path[i];
                json row = gdn_global_function_risk_json(global);
                row["readout"] = "exact-ridge-variable-projection";
                row["ridge_relative"] = local.ridge_relative;
                row["ridge_absolute"] = local.ridge_absolute;
                row["local_train_residual_ratio"] = local.train_residual_ratio;
                row["local_validation_residual_ratio"] = local.validation_residual_ratio;
                row["selection_value"] = value;
                out.readout_candidates.push_back(std::move(row));
                if (value < out.selection_value) {
                    out.global = global;
                    out.selection_value = value;
                    out.selected_current_readout = false;
                    out.ridge_relative = local.ridge_relative;
                    out.ridge_absolute = local.ridge_absolute;
                    out.selected_train_residual_ratio = local.train_residual_ratio;
                    out.selected_validation_residual_ratio = local.validation_residual_ratio;
                }
            }
            restore_current_readout();
            return out;
        } catch (...) {
            try {
                restore_current_readout();
            } catch (...) {
            }
            throw;
        }
    }
};

static json gdn_reduced_core_risk_json(const gdn_reduced_core_risk_result & risk) {
    json out = gdn_global_function_risk_json(risk.global);
    out["selection_value"] = risk.selection_value;
    out["selected_readout"] = risk.selected_current_readout ?
        "current-student" : "exact-ridge-variable-projection";
    out["selected_ridge_relative"] = risk.ridge_relative;
    out["selected_ridge_absolute"] = risk.ridge_absolute;
    out["baseline_local_train_residual_ratio"] = risk.baseline_train_residual_ratio;
    out["baseline_local_validation_residual_ratio"] = risk.baseline_validation_residual_ratio;
    out["selected_local_train_residual_ratio"] = risk.selected_train_residual_ratio;
    out["selected_local_validation_residual_ratio"] = risk.selected_validation_residual_ratio;
    out["readout_candidates"] = risk.readout_candidates;
    return out;
}

static void command_select_gdn_function_projection_global(
        const std::string & teacher_path,
        const std::string & student_path,
        const std::string & plan_path,
        const std::string & train_corpus_path,
        const std::string & validation_corpus_path,
        const std::string & output_path,
        int layer,
        int train_tokens,
        int validation_tokens,
        double ridge_center,
        int ridge_decades,
        double distill_temperature,
        const std::string & objective,
        int score_from_position,
        int n_threads) {
    if (!is_recurrent_layer(layer)) throw std::runtime_error("global function-projection layer is not recurrent");
    if (train_tokens <= 0 || validation_tokens < 2) {
        throw std::runtime_error("global function-projection token limits are invalid");
    }
    if (!(ridge_center > 0.0) || !std::isfinite(ridge_center) || ridge_decades < 0 || ridge_decades > 12) {
        throw std::runtime_error("global function-projection ridge path is invalid");
    }
    if (!(distill_temperature > 0.0) || !std::isfinite(distill_temperature)) {
        throw std::runtime_error("global function-projection temperature is invalid");
    }
    (void) gdn_global_selection_value({}, objective); // validate spelling before expensive model load

    const json plan = read_json(plan_path);
    verify_plan_json(plan);
    const auto vocab_map = plan.at("vocab").at("output_to_input").get<std::vector<int>>();

    llama_backend_init();
    auto teacher_model = load_full_model(teacher_path);
    auto student_model = load_full_model(student_path);

    // Inner approximation problem: for the current nonlinear 64D recurrent
    // feature map h_eta, construct a ridge path of exact empirical-L2 best
    // linear readouts. This is only a candidate generator for the outer
    // observable function problem below.
    auto train = capture_gdn_function_set(
            teacher_model.get(), student_model.get(), vocab_map, train_corpus_path, layer, train_tokens);
    auto validation = capture_gdn_function_set(
            teacher_model.get(), student_model.get(), vocab_map, validation_corpus_path, layer, validation_tokens);
    std::vector<double> ridge_path;
    for (int decade = -ridge_decades; decade <= ridge_decades; ++decade) {
        const double ridge = ridge_center * std::pow(10.0, decade);
        if (ridge > 0.0 && std::isfinite(ridge)) ridge_path.push_back(ridge);
    }
    const int n_features = GDN_V_HEADS * GDN_DIM_DST;
    const auto path_fit = gdn_fit_function_projection_dual_path(
            train.student_feature, train.teacher_output, train.samples,
            validation.student_feature, validation.teacher_output, validation.samples,
            n_features, HIDDEN, ridge_path, n_threads);

    std::vector<llama_token> student_tokens, teacher_tokens;
    prepare_gdn_global_tokens(
            student_model.get(), vocab_map, validation_corpus_path, validation_tokens,
            student_tokens, teacher_tokens);
    if (score_from_position < 0) score_from_position = (int) student_tokens.size() / 2;
    const int batch_size = std::min<int>(32, (int) student_tokens.size());
    auto teacher_ctx = make_function_risk_context(teacher_model.get(), (int) student_tokens.size(), batch_size);
    auto student_ctx = make_function_risk_context(student_model.get(), (int) student_tokens.size(), batch_size);
    const auto teacher_risk_cache = prepare_gdn_teacher_risk_cache(
            teacher_ctx.get(), teacher_model.get(), student_model.get(),
            teacher_tokens, student_tokens, vocab_map,
            distill_temperature, score_from_position, batch_size);
    const auto baseline_global = measure_gdn_global_function_risk_cached_teacher(
            student_ctx.get(), student_model.get(), student_tokens, teacher_risk_cache, batch_size);

    const std::string tensor_name = "blk." + std::to_string(layer) + ".ssm_out.weight";
    ggml_tensor * loaded_ssm_out = const_cast<ggml_tensor *>(student_model->get_tensor(tensor_name.c_str()));
    if (!loaded_ssm_out) throw std::runtime_error("loaded student is missing " + tensor_name);
    backend_tensor_restore_guard restore(loaded_ssm_out);

    double best_value = std::numeric_limits<double>::infinity();
    size_t best_index = 0;
    std::vector<float> best_weight;
    gdn_global_function_risk_result best_global;
    json candidate_json = json::array();
    for (size_t i = 0; i < path_fit.path.size(); ++i) {
        auto weight = gdn_materialize_function_projection_weight(
                train.student_feature, path_fit.dual_path.at(i),
                train.samples, n_features, HIDDEN, n_threads);
        set_loaded_gdn_ssm_out(loaded_ssm_out, weight, n_threads);
        const auto global = measure_gdn_global_function_risk_cached_teacher(
                student_ctx.get(), student_model.get(), student_tokens, teacher_risk_cache, batch_size);
        const double value = gdn_global_selection_value(global, objective);
        const auto & local = path_fit.path[i];
        json row = gdn_global_function_risk_json(global);
        row["ridge_relative"] = local.ridge_relative;
        row["ridge_absolute"] = local.ridge_absolute;
        row["local_train_residual_ratio"] = local.train_residual_ratio;
        row["local_validation_residual_ratio"] = local.validation_residual_ratio;
        row["selection_value"] = value;
        candidate_json.push_back(std::move(row));
        std::cerr << "global function projection layer " << layer
                  << " ridge=" << local.ridge_relative
                  << " local_val=" << local.validation_residual_ratio
                  << " KL=" << global.teacher_conditional_kl
                  << " NLL=" << global.student_nll << "\n";
        if (value < best_value) {
            best_value = value;
            best_index = i;
            best_weight = std::move(weight);
            best_global = global;
        }
    }
    if (best_weight.empty()) throw std::runtime_error("global function-projection selector produced no candidate");

    const auto & best_local = path_fit.path.at(best_index);
    gdn_function_projection_regression reg;
    reg.layer = layer;
    reg.train_samples = (uint64_t) train.samples;
    reg.validation_samples = (uint64_t) validation.samples;
    reg.ridge_relative = best_local.ridge_relative;
    reg.ridge_absolute = best_local.ridge_absolute;
    reg.baseline_train_residual_ratio = relative_residual_ratio(train.teacher_output, train.student_output);
    reg.fitted_train_residual_ratio = best_local.train_residual_ratio;
    reg.baseline_validation_residual_ratio = relative_residual_ratio(validation.teacher_output, validation.student_output);
    reg.fitted_validation_residual_ratio = best_local.validation_residual_ratio;
    reg.weight = std::move(best_weight);
    write_gdn_function_projection_regression(output_path, reg);

    json out = {
        {"format", "gdn-global-function-approximation-v1"},
        {"theory", "outer-observable-function-risk-with-inner-variable-projection-candidate-family"},
        {"layer", layer},
        {"selection_objective", objective == "kl" ? "teacher-conditional-KL" : "data-NLL"},
        {"teacher_distribution", "teacher-softmax-conditioned-on-retained-student-vocabulary"},
        {"distill_temperature", distill_temperature},
        {"score_from_position", score_from_position},
        {"empirical_measure", "uniform-over-selected-next-token-prefixes"},
        {"train_samples", train.samples},
        {"validation_samples", validation.samples},
        {"selected_ridge_relative", best_local.ridge_relative},
        {"selected_ridge_absolute", best_local.ridge_absolute},
        {"selected_local_train_residual_ratio", best_local.train_residual_ratio},
        {"selected_local_validation_residual_ratio", best_local.validation_residual_ratio},
        {"baseline_global", gdn_global_function_risk_json(baseline_global)},
        {"selected_global", gdn_global_function_risk_json(best_global)},
        {"candidates", candidate_json},
        {"plan_sha256", sha256_file(plan_path)},
        {"teacher_identity", file_identity(teacher_path)},
        {"student_identity", file_identity(student_path)},
    };
    write_json(output_path + ".json", out);
    // Restore the loaded model before tearing down the backend. The persisted
    // student file is intentionally untouched by this selector.
    restore.restore_now();
    std::cout << "GDN global function projection layer " << layer
              << ": selected ridge=" << best_local.ridge_relative
              << " " << objective << " " << gdn_global_selection_value(baseline_global, objective)
              << " -> " << best_value << "\n"
              << "wrote globally-selected GDN function projection: " << output_path << "\n";
    llama_backend_free();
}

static void command_select_gdn_qk_atom_exchange_global(
        const std::string & teacher_path,
        const std::string & student_path,
        const std::string & plan_path,
        const std::string & train_corpus_path,
        const std::string & validation_corpus_path,
        const std::string & output_plan_path,
        int layer,
        int head,
        int train_tokens,
        int validation_tokens,
        double ridge_center,
        int ridge_decades,
        double distill_temperature,
        const std::string & objective,
        int score_from_position,
        int n_threads) {
    if (!is_recurrent_layer(layer) || head < 0 || head >= GDN_QK_HEADS) {
        throw std::runtime_error("GDN Q/K atom exchange layer/head is invalid");
    }
    if (train_tokens < 2 || validation_tokens < 2 ||
        !(ridge_center > 0.0) || !std::isfinite(ridge_center) ||
        ridge_decades < 0 || ridge_decades > 12 ||
        !(distill_temperature > 0.0) || !std::isfinite(distill_temperature) || n_threads <= 0) {
        throw std::runtime_error("GDN Q/K atom exchange scoring parameters are invalid");
    }
    (void) gdn_global_selection_value({}, objective);

    json plan = read_json(plan_path);
    verify_plan_json(plan);
    json * rec = nullptr;
    for (auto & candidate : plan.at("gdn")) {
        if (candidate.at("layer").get<int>() == layer) {
            rec = &candidate;
            break;
        }
    }
    if (!rec) throw std::runtime_error("GDN Q/K atom exchange plan is missing layer");
    auto current = gdn_qk_indices_for_head(*rec, head);
    std::array<bool, GDN_DIM_SRC> selected {};
    for (int x : current) selected[(size_t) x] = true;

    const auto vocab_map = plan.at("vocab").at("output_to_input").get<std::vector<int>>();
    source_gguf teacher_src(teacher_path);
    require_source_contract(teacher_src);

    llama_backend_init();
    auto teacher_model = load_full_model(teacher_path);
    auto student_model = load_full_model(student_path);

    std::vector<llama_token> student_train_tokens, teacher_train_tokens;
    prepare_gdn_global_tokens(
            student_model.get(), vocab_map, train_corpus_path, train_tokens,
            student_train_tokens, teacher_train_tokens);
    std::vector<llama_token> student_validation_tokens, teacher_validation_tokens;
    prepare_gdn_global_tokens(
            student_model.get(), vocab_map, validation_corpus_path, validation_tokens,
            student_validation_tokens, teacher_validation_tokens);
    if (score_from_position < 0) score_from_position = (int) student_validation_tokens.size() / 2;
    if (score_from_position < 0 || score_from_position >= (int) student_validation_tokens.size() - 1) {
        throw std::runtime_error("GDN Q/K atom exchange score support is empty");
    }

    const int train_batch_size = std::min<int>(32, (int) student_train_tokens.size());
    const int validation_batch_size = std::min<int>(32, (int) student_validation_tokens.size());
    const auto teacher_train_decode_tokens = gdn_decode_tokens_for_model(
            teacher_model.get(), teacher_train_tokens);
    const auto teacher_validation_decode_tokens = gdn_decode_tokens_for_model(
            teacher_model.get(), teacher_validation_tokens);
    const auto student_train_decode_tokens = gdn_decode_tokens_for_model(
            student_model.get(), student_train_tokens);
    const auto student_validation_decode_tokens = gdn_decode_tokens_for_model(
            student_model.get(), student_validation_tokens);

    gdn_layer_output_capture teacher_train_capture, teacher_validation_capture;
    teacher_train_capture.layer = layer;
    teacher_validation_capture.layer = layer;
    auto teacher_train_ctx = make_layer_output_capture_context(
            teacher_model.get(), teacher_train_capture,
            (int) teacher_train_tokens.size(), train_batch_size);
    auto teacher_validation_ctx = make_layer_output_capture_context(
            teacher_model.get(), teacher_validation_capture,
            (int) teacher_validation_tokens.size(), validation_batch_size);
    const auto teacher_train = replay_gdn_layer_sequence(
            teacher_train_ctx.get(), teacher_train_capture,
            teacher_train_decode_tokens, train_batch_size, 0);
    const auto teacher_validation = replay_gdn_layer_sequence(
            teacher_validation_ctx.get(), teacher_validation_capture,
            teacher_validation_decode_tokens, validation_batch_size, 0);
    if (teacher_train.linear_output.size() != (size_t) teacher_train.samples * HIDDEN ||
        teacher_validation.linear_output.size() != (size_t) teacher_validation.samples * HIDDEN) {
        throw std::runtime_error("GDN Q/K atom exchange teacher target capture failed");
    }

    gdn_layer_output_capture student_train_capture, student_validation_capture;
    student_train_capture.layer = layer;
    student_validation_capture.layer = layer;
    auto student_train_ctx = make_layer_output_capture_context(
            student_model.get(), student_train_capture,
            (int) student_train_tokens.size(), train_batch_size);
    auto student_validation_ctx = make_layer_output_capture_context(
            student_model.get(), student_validation_capture,
            (int) student_validation_tokens.size(), validation_batch_size);
    auto teacher_risk_ctx = make_function_risk_context(
            teacher_model.get(), (int) student_validation_tokens.size(), validation_batch_size);
    auto student_risk_ctx = make_function_risk_context(
            student_model.get(), (int) student_validation_tokens.size(), validation_batch_size);
    const auto teacher_risk_cache = prepare_gdn_teacher_risk_cache(
            teacher_risk_ctx.get(), teacher_model.get(), student_model.get(),
            teacher_validation_tokens, student_validation_tokens, vocab_map,
            distill_temperature, score_from_position, validation_batch_size);

    std::vector<double> ridge_path;
    ridge_path.reserve((size_t) 2 * ridge_decades + 1);
    for (int decade = -ridge_decades; decade <= ridge_decades; ++decade) {
        const double ridge = ridge_center * std::pow(10.0, decade);
        if (ridge > 0.0 && std::isfinite(ridge)) ridge_path.push_back(ridge);
    }
    if (ridge_path.empty()) throw std::runtime_error("GDN Q/K atom exchange ridge path is empty");

    ggml_tensor * qkv = student_model->layers[(size_t) layer].wqkv;
    ggml_tensor * conv = student_model->layers[(size_t) layer].ssm_conv1d;
    ggml_tensor * ssm_out = student_model->layers[(size_t) layer].ssm_out;
    backend_tensor_restore_guard restore_qkv(qkv);
    backend_tensor_restore_guard restore_conv(conv);
    backend_tensor_restore_guard restore_ssm_out(ssm_out);
    gdn_loaded_qk_atom_editor editor(teacher_src, student_model.get(), layer, head);

    gdn_reduced_readout_oracle oracle;
    oracle.student_model = student_model.get();
    oracle.student_train_ctx = student_train_ctx.get();
    oracle.student_validation_ctx = student_validation_ctx.get();
    oracle.student_risk_ctx = student_risk_ctx.get();
    oracle.student_train_capture = &student_train_capture;
    oracle.student_validation_capture = &student_validation_capture;
    oracle.student_ssm_out = ssm_out;
    oracle.teacher_train_output = &teacher_train.linear_output;
    oracle.teacher_validation_output = &teacher_validation.linear_output;
    oracle.student_train_decode_tokens = &student_train_decode_tokens;
    oracle.student_validation_decode_tokens = &student_validation_decode_tokens;
    oracle.student_risk_tokens = &student_validation_tokens;
    oracle.teacher_risk_cache = &teacher_risk_cache;
    oracle.current_readout_bytes = &restore_ssm_out.bytes;
    oracle.ridge_path = &ridge_path;
    oracle.objective = objective;
    oracle.distill_temperature = distill_temperature;
    oracle.score_from_position = score_from_position;
    oracle.train_batch_size = train_batch_size;
    oracle.validation_batch_size = validation_batch_size;
    oracle.risk_batch_size = validation_batch_size;
    oracle.n_threads = n_threads;

    // Define the baseline inside the exact teacher-atom function family,
    // independent of whatever Q/K bytes happen to be present in the input
    // student. The score is the reduced objective over an exact ridge readout
    // candidate family plus the current stock readout.
    editor.set_head_from_teacher(current);
    const auto baseline = oracle.evaluate();
    const double baseline_value = baseline.selection_value;
    std::cerr << "GDN Q/K reduced baseline layer=" << layer << " head=" << head
              << " " << objective << "=" << baseline_value
              << " readout=" << (baseline.selected_current_readout ? "current" : "ridge") << "\n";

    int replace_slot = -1;
    double best_deletion_value = std::numeric_limits<double>::infinity();
    json deletion_json = json::array();
    for (int slot = 0; slot < GDN_DIM_DST; ++slot) {
        editor.zero_slot(slot);
        const auto risk = oracle.evaluate();
        const double value = risk.selection_value;
        json row = gdn_reduced_core_risk_json(risk);
        row["slot"] = slot;
        row["teacher_coordinate"] = current[(size_t) slot];
        deletion_json.push_back(std::move(row));
        editor.set_slot_from_teacher(slot, current[(size_t) slot]);
        std::cerr << "GDN Q/K reduced deletion layer=" << layer << " head=" << head
                  << " slot=" << slot << " coord=" << current[(size_t) slot]
                  << " " << objective << "=" << value << "\n";
        if (value < best_deletion_value ||
            (value == best_deletion_value && (replace_slot < 0 || slot < replace_slot))) {
            best_deletion_value = value;
            replace_slot = slot;
        }
    }
    if (replace_slot < 0) throw std::runtime_error("GDN Q/K atom exchange found no deletion slot");

    const int removed_coordinate = current[(size_t) replace_slot];
    int best_coordinate = removed_coordinate;
    double best_value = baseline_value;
    gdn_reduced_core_risk_result best_risk = baseline;
    json replacement_json = json::array();
    for (int coordinate = 0; coordinate < GDN_DIM_SRC; ++coordinate) {
        if (selected[(size_t) coordinate]) continue;
        editor.set_slot_from_teacher(replace_slot, coordinate);
        const auto risk = oracle.evaluate();
        const double value = risk.selection_value;
        json row = gdn_reduced_core_risk_json(risk);
        row["candidate_teacher_coordinate"] = coordinate;
        replacement_json.push_back(std::move(row));
        std::cerr << "GDN Q/K reduced replacement layer=" << layer << " head=" << head
                  << " slot=" << replace_slot << " " << removed_coordinate << "->" << coordinate
                  << " " << objective << "=" << value << "\n";
        if (value < best_value || (value == best_value && coordinate < best_coordinate)) {
            best_value = value;
            best_coordinate = coordinate;
            best_risk = risk;
        }
    }

    const double improvement_tol = 1e-12 * std::max(1.0, std::fabs(baseline_value));
    const bool accepted = best_coordinate != removed_coordinate && best_value < baseline_value - improvement_tol;
    if (accepted) {
        current[(size_t) replace_slot] = best_coordinate;
        (*rec)["qk_indices_by_head"][(size_t) head] = current;
    }
    (*rec)["initialization"] = "exact-teacher-atom-subset";
    (*rec)["initialization_note"] = "All selected Q/K channels are exact teacher projection+depthwise-conv atoms; no internal surrogate defines the selection.";

    json search_record = {
        {"algorithm", "reduced-observable-risk-greedy-paired-qk-atom-exchange-v2"},
        {"reduced_objective", "min-over-current-readout-plus-exact-ridge-variable-projection-candidates"},
        {"readout_candidate_generator", "exact-empirical-L2-ridge-variable-projection"},
        {"readout_candidate_acceptance", "observable-whole-model-risk-only"},
        {"layer", layer},
        {"head", head},
        {"objective", objective == "kl" ? "teacher-conditional-KL" : "data-NLL"},
        {"train_corpus", train_corpus_path},
        {"validation_corpus", validation_corpus_path},
        {"train_tokens", student_train_tokens.size()},
        {"validation_tokens", student_validation_tokens.size()},
        {"score_from_position", score_from_position},
        {"distill_temperature", distill_temperature},
        {"ridge_center", ridge_center},
        {"ridge_decades", ridge_decades},
        {"ridge_path", ridge_path},
        {"baseline", gdn_reduced_core_risk_json(baseline)},
        {"deletion_selected_slot", replace_slot},
        {"deletion_selected_teacher_coordinate", removed_coordinate},
        {"deletion_selected_value", best_deletion_value},
        {"accepted", accepted},
        {"replacement_teacher_coordinate", accepted ? best_coordinate : removed_coordinate},
        {"selected", gdn_reduced_core_risk_json(accepted ? best_risk : baseline)},
        {"deletion_candidates", deletion_json},
        {"replacement_candidates", replacement_json},
    };
    if (!plan.contains("gdn_function_search_history")) plan["gdn_function_search_history"] = json::array();
    plan["gdn_function_search_history"].push_back(search_record);
    plan["derived_from_plan_sha256"] = sha256_file(plan_path);
    plan["gdn_function_search_tool_exe_sha256"] = current_executable_sha256();
    verify_plan_json(plan);
    write_json(output_plan_path, plan);

    restore_ssm_out.restore_now();
    restore_conv.restore_now();
    restore_qkv.restore_now();
    std::cout << "GDN Q/K atom exchange layer " << layer << " head " << head
              << ": " << (accepted ? "accepted " : "kept ")
              << removed_coordinate << (accepted ? " -> " + std::to_string(best_coordinate) : std::string())
              << ", reduced " << objective << " " << baseline_value << " -> "
              << (accepted ? best_value : baseline_value) << "\n"
              << "wrote function-search plan: " << output_plan_path << "\n";
    llama_backend_free();
}

static json audit_tokenizers(
        const std::string & source_path,
        const std::string & output_path,
        const std::string & corpus_path,
        bool enforce_hard_limit = true) {
    llama_backend_init();
    auto source_model = load_vocab_only(source_path);
    auto output_model = load_vocab_only(output_path);
    const llama_vocab * source_vocab = llama_model_get_vocab(source_model.get());
    const llama_vocab * output_vocab = llama_model_get_vocab(output_model.get());
    if (!source_vocab || !output_vocab) throw std::runtime_error("vocab-only model has no vocabulary");
    const int32_t source_vocab_size = llama_vocab_n_tokens(source_vocab);
    const int32_t output_vocab_size = llama_vocab_n_tokens(output_vocab);
    const bool keep_vocab = output_vocab_size == source_vocab_size;
    if (!keep_vocab && output_vocab_size != VOCAB_DST) {
        throw std::runtime_error("vocab-only loader sees wrong output vocabulary size");
    }

    std::vector<std::pair<std::string,std::string>> domains;
    domains.emplace_back("zh",
        "量化后的模型必须保持中文表达自然，工具调用、长上下文和数学推理都不能因为词表裁剪而明显膨胀。"
        "这是一个用于检查中文分词效率与字节级无损往返的固定样本。\n");
    domains.emplace_back("en",
        "A tokenizer pruning pass should preserve common English words, punctuation, numbers, and technical prose without excessive token inflation. "
        "The resulting model must remain efficient for ordinary instructions and long-form generation.\n");
    domains.emplace_back("code",
        "template <typename T> T dot(const std::vector<T>& a, const std::vector<T>& b) { T s{}; for (size_t i=0;i<a.size();++i) s += a[i]*b[i]; return s; }\n"
        "def quick_check(x: bytes) -> str:\n    return x.hex() if x else '<empty>'\n");
    domains.emplace_back("math",
        "For A\\in\\mathbb{R}^{m\\times n}, minimize \\|Ax-b\\|_2^2+\\lambda\\|x\\|_1. "
        "If f(x)=x^3-2x+1, then f'(x)=3x^2-2 and \\int_0^1 x^2 dx=1/3.\n");
    domains.emplace_back("tool",
        "<tool_call>{\"name\":\"search\",\"arguments\":{\"query\":\"qwen35 mtp q2_0\",\"limit\":8}}</tool_call>\n"
        "{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{\"id\":\"call_01\",\"type\":\"function\"}]}\n");
    std::string unicode_text;
    unicode_text.push_back('\0');
    unicode_text.push_back('\1');
    unicode_text.push_back('\x1f');
    unicode_text += " ASCII café Ελληνικά русский العربية हिन्दी 中文 日本語 한국어 🙂🚀 𝛑 ∑ ∫\n";
    domains.emplace_back("unicode", std::move(unicode_text));
    std::string long_text;
    for (int i = 0; i < 64; ++i) {
        long_text += "Long-context tokenizer audit paragraph " + std::to_string(i) +
            ": cache locality, recurrent state, speculative decoding, multilingual 文本, and 0123456789 symbols.\n";
    }
    domains.emplace_back("long", std::move(long_text));
    if (!corpus_path.empty()) domains.emplace_back("corpus", read_file_bytes(corpus_path));

    // Qwen uses a GPT-2 byte alphabet. Feeding arbitrary invalid UTF-8 into
    // llama_tokenize is not a meaningful pretokenizer contract, so verify the
    // actual byte fallback directly: all 256 alphabet tokens must exist and the
    // stock token_to_piece decoder must map each one back to its exact raw byte.
    auto byte_fallback_ids = [&](const llama_vocab * vocab) {
        std::array<llama_token, 256> ids {};
        ids.fill(-1);
        for (llama_token id = 0; id < llama_vocab_n_tokens(vocab); ++id) {
            const char * text = llama_vocab_get_text(vocab, id);
            if (!text) continue;
            uint8_t b = 0;
            if (!gpt2_byte_from_token(text, b)) continue;
            if (ids[(size_t) b] != -1) throw std::runtime_error("duplicate GPT-2 byte fallback token");
            ids[(size_t) b] = id;
        }
        for (int b = 0; b < 256; ++b) {
            if (ids[(size_t) b] < 0) throw std::runtime_error("missing GPT-2 byte fallback token for byte " + std::to_string(b));
            char piece[8] = {};
            const int32_t n = llama_token_to_piece(vocab, ids[(size_t) b], piece, sizeof(piece), 0, false);
            if (n != 1 || (uint8_t) piece[0] != (uint8_t) b) {
                throw std::runtime_error("GPT-2 byte fallback token decodes to wrong raw byte " + std::to_string(b));
            }
        }
        return ids;
    };
    const auto src_byte_ids = byte_fallback_ids(source_vocab);
    const auto out_byte_ids = byte_fallback_ids(output_vocab);
    std::vector<llama_token> src_byte_seq(src_byte_ids.begin(), src_byte_ids.end());
    std::vector<llama_token> out_byte_seq(out_byte_ids.begin(), out_byte_ids.end());
    std::string all_bytes(256, '\0');
    for (int b = 0; b < 256; ++b) all_bytes[(size_t) b] = (char) b;
    if (detokenize_bytes(source_vocab, src_byte_seq) != all_bytes) throw std::runtime_error("source 256-byte fallback sequence decode failed");
    if (detokenize_bytes(output_vocab, out_byte_seq) != all_bytes) throw std::runtime_error("output 256-byte fallback sequence decode failed");

    size_t control_literal_count = 0;
    for (llama_token id = 0; id < llama_vocab_n_tokens(output_vocab); ++id) {
        if (!llama_vocab_is_control(output_vocab, id)) continue;
        const char * text = llama_vocab_get_text(output_vocab, id);
        if (!text || !*text) continue;
        const auto parsed = tokenize_bytes(output_vocab, text, /*parse_special=*/ true);
        if (parsed.size() != 1 || parsed[0] != id) {
            throw std::runtime_error("control-token literal does not parse back to its token id: " + std::string(text));
        }
        ++control_literal_count;
    }

    json ratios = json::array();
    double worst = 0.0;
    std::string worst_domain;
    double priority_worst = 0.0;
    std::string priority_worst_domain;
    for (const auto & [name, text] : domains) {
        const auto src_tokens = tokenize_bytes(source_vocab, text);
        const auto out_tokens = tokenize_bytes(output_vocab, text);
        if (detokenize_bytes(source_vocab, src_tokens) != text) throw std::runtime_error("source tokenizer round-trip failed in domain: " + name);
        if (detokenize_bytes(output_vocab, out_tokens) != text) throw std::runtime_error("output tokenizer round-trip failed in domain: " + name);
        const double ratio = src_tokens.empty() ? (out_tokens.empty() ? 1.0 : std::numeric_limits<double>::infinity()) :
                             (double) out_tokens.size() / src_tokens.size();
        if (ratio > worst) {
            worst = ratio;
            worst_domain = name;
        }
        if ((name == "zh" || name == "en") && ratio > priority_worst) {
            priority_worst = ratio;
            priority_worst_domain = name;
        }
        ratios.push_back({
            {"domain", name},
            {"source_tokens", src_tokens.size()},
            {"output_tokens", out_tokens.size()},
            {"ratio", ratio},
        });
    }

    json result = {
        {"vocab_mode", keep_vocab ? "unchanged" : "pruned"},
        {"source_vocab_size", source_vocab_size},
        {"output_vocab_size", output_vocab_size},
        {"byte_fallback_roundtrip", true},
        {"byte_fallback_tokens", 256},
        {"control_literal_tokens", control_literal_count},
        {"domains", ratios},
        {"worst_ratio", worst},
        {"worst_domain", worst_domain},
        {"hard_limit", 1.10},
        {"hard_limit_domains", {"zh", "en"}},
        {"priority_worst_ratio", priority_worst},
        {"priority_worst_domain", priority_worst_domain},
        {"hard_limit_pass", priority_worst < 1.10},
    };
    if (enforce_hard_limit && !(priority_worst < 1.10)) {
        throw std::runtime_error("priority token inflation hard limit violated in domain '" + priority_worst_domain +
                                 "'; r=" + std::to_string(priority_worst) +
                                 " (only zh/en are hard-gated; other domains are diagnostic)");
    }
    return result;
}

static void verify_selected_rows_exact(
        const source_gguf & src,
        const source_gguf & out,
        const std::string & name,
        const std::vector<int> & rows) {
    const ggml_tensor * st = src.tensor(name);
    const ggml_tensor * ot = out.tensor(name);
    if (st->type != ot->type || st->ne[0] != ot->ne[0] || ot->ne[1] != (int64_t) rows.size()) {
        throw std::runtime_error("row-copy descriptor mismatch: " + name);
    }
    const size_t row_size = ggml_row_size(st->type, st->ne[0]);
    std::vector<uint8_t> a(row_size), b(row_size);
    for (size_t r = 0; r < rows.size(); ++r) {
        const int sr = rows[r];
        if (sr < 0 || sr >= st->ne[1]) throw std::runtime_error("row-copy source row out of range: " + name);
        src.read_tensor_bytes(name, (size_t) sr * row_size, a.data(), row_size);
        out.read_tensor_bytes(name, r * row_size, b.data(), row_size);
        if (std::memcmp(a.data(), b.data(), row_size) != 0) {
            throw std::runtime_error("packed row-copy payload mismatch: " + name + " row " + std::to_string(r));
        }
    }
}

static void verify_source_quant_expert_tensor(
        const source_gguf & src,
        const source_gguf & out,
        const std::string & name,
        const plan_index & plan) {
    const int layer = parse_block_id(name);
    if (layer < 0 || layer >= N_LAYER_ALL) {
        throw std::runtime_error("source-quant expert tensor has invalid layer: " + name);
    }

    const ggml_tensor * st = src.tensor(name);
    const ggml_tensor * ot = out.tensor(name);
    const bool down = ends_with(name, "ffn_down_exps.weight");
    if ((!down && st->type != ot->type) || st->ne[2] != N_EXPERT || ot->ne[2] != N_EXPERT) {
        throw std::runtime_error("source-quant expert descriptor mismatch: " + name);
    }

    const size_t src_row_size = ggml_row_size(st->type, st->ne[0]);
    const size_t dst_row_size = ggml_row_size(ot->type, ot->ne[0]);
    const size_t src_rows_per_expert = (size_t) st->ne[1];
    const size_t dst_rows_per_expert = (size_t) ot->ne[1];
    const size_t src_expert_bytes = src_row_size * src_rows_per_expert;
    const size_t dst_expert_bytes = dst_row_size * dst_rows_per_expert;

    if (!down) {
        if (st->ne[0] != HIDDEN || st->ne[1] != EXPERT_WIDTH_SRC ||
            ot->ne[0] != HIDDEN || ot->ne[1] != EXPERT_WIDTH_DST ||
            src_row_size != dst_row_size) {
            throw std::runtime_error("source-quant gate/up expert shape mismatch: " + name);
        }
        std::vector<uint8_t> expected(src_row_size), actual(dst_row_size);
        for (int expert = 0; expert < N_EXPERT; ++expert) {
            const auto & mapping = plan.expert[(size_t) layer][(size_t) expert];
            for (int j = 0; j < EXPERT_WIDTH_DST; ++j) {
                const int old = mapping[(size_t) j];
                if (old < 0 || old >= EXPERT_WIDTH_SRC) {
                    throw std::runtime_error("source-quant expert mapping out of range: " + name);
                }
                src.read_tensor_bytes(
                    name,
                    (size_t) expert * src_expert_bytes + (size_t) old * src_row_size,
                    expected.data(), expected.size());
                out.read_tensor_bytes(
                    name,
                    (size_t) expert * dst_expert_bytes + (size_t) j * dst_row_size,
                    actual.data(), actual.size());
                if (expected != actual) {
                    throw std::runtime_error("source-quant expert row-copy mismatch: " + name +
                                             " expert " + std::to_string(expert) +
                                             " row " + std::to_string(j));
                }
            }
        }
        return;
    }

    if (st->ne[0] != EXPERT_WIDTH_SRC || st->ne[1] != HIDDEN ||
        ot->ne[0] != EXPERT_WIDTH_DST || ot->ne[1] != HIDDEN) {
        throw std::runtime_error("source-quant down expert shape mismatch: " + name);
    }
    const auto * src_traits = ggml_get_type_traits(st->type);
    const auto * dst_traits = ggml_get_type_traits(ot->type);
    if (!src_traits || !src_traits->to_float || !dst_traits || !dst_traits->from_float_ref) {
        throw std::runtime_error("source-quant down expert type cannot be reproduced: " + name);
    }

    std::vector<uint8_t> src_row(src_row_size), actual(dst_row_size), expected(dst_row_size);
    std::vector<float> decoded(EXPERT_WIDTH_SRC), gathered(EXPERT_WIDTH_DST);
    for (int expert = 0; expert < N_EXPERT; ++expert) {
        const auto & mapping = plan.expert[(size_t) layer][(size_t) expert];
        size_t packed_slice_offset = 0;
        const bool packed_slice = st->type == ot->type &&
            expert_mapping_packed_slice(mapping, st->type, packed_slice_offset);
        if (packed_slice && packed_slice_offset + dst_row_size > src_row_size) {
            throw std::runtime_error("source-quant expert packed slice is out of row bounds: " + name);
        }
        for (int r = 0; r < HIDDEN; ++r) {
            src.read_tensor_bytes(
                name,
                (size_t) expert * src_expert_bytes + (size_t) r * src_row_size,
                src_row.data(), src_row.size());
            if (packed_slice) {
                std::memcpy(expected.data(), src_row.data() + packed_slice_offset, dst_row_size);
            } else {
                src_traits->to_float(src_row.data(), decoded.data(), EXPERT_WIDTH_SRC);
                for (int j = 0; j < EXPERT_WIDTH_DST; ++j) {
                    const int old = mapping[(size_t) j];
                    if (old < 0 || old >= EXPERT_WIDTH_SRC) {
                        throw std::runtime_error("source-quant expert mapping out of range: " + name);
                    }
                    gathered[(size_t) j] = decoded[(size_t) old];
                }
                dst_traits->from_float_ref(gathered.data(), expected.data(), EXPERT_WIDTH_DST);
            }
            out.read_tensor_bytes(
                name,
                (size_t) expert * dst_expert_bytes + (size_t) r * dst_row_size,
                actual.data(), actual.size());
            if (expected != actual) {
                throw std::runtime_error("source-quant down expert requantization mismatch: " + name +
                                         " expert " + std::to_string(expert) +
                                         " row " + std::to_string(r));
            }
        }
    }
}

static void verify_ssm_norm_gather(
        const source_gguf & src,
        const source_gguf & out,
        const std::string & name,
        const json & rec) {
    const auto positions = gdn_v_positions(rec);
    std::array<float, GDN_DIM_SRC> a {};
    std::array<float, GDN_DIM_DST> b {};
    src.read_tensor_bytes(name, 0, a.data(), sizeof(a));
    out.read_tensor_bytes(name, 0, b.data(), sizeof(b));
    const double scale = rec.value("ssm_norm_scale", 1.0);
    for (int i = 0; i < GDN_DIM_DST; ++i) {
        const float expected = (float) (a[(size_t) positions[(size_t) i]] * scale);
        if (std::memcmp(&expected, &b[(size_t) i], sizeof(float)) != 0) {
            throw std::runtime_error("ssm_norm scaled gather mismatch: " + name + " position " + std::to_string(i));
        }
    }
}

struct ssm_out_requant_error {
    uint64_t values = 0;
    long double target_energy = 0.0;
    long double error_energy = 0.0;
    double max_abs_error = 0.0;

    double relative_rms() const {
        return target_energy > 0.0 ? std::sqrt((double) (error_energy / target_energy)) : 0.0;
    }

    double rms() const {
        return values ? std::sqrt((double) (error_energy / values)) : 0.0;
    }
};

static ssm_out_requant_error measure_ssm_out_requantize(
        const source_gguf & src,
        const std::string & name,
        const json & rec) {
    const ggml_tensor * st = src.tensor(name);
    if (st->ne[0] != 4096 || st->ne[1] != HIDDEN) {
        throw std::runtime_error("ssm_out source descriptor mismatch: " + name);
    }
    const auto channels = gdn_v_channel_mapping(rec);
    if (channels.size() != 2048) throw std::runtime_error("ssm_out score channel map mismatch");
    const auto * traits = ggml_get_type_traits(st->type);
    if (!traits || !traits->to_float || !traits->from_float_ref ||
        (st->type != GGML_TYPE_Q8_0 && st->type != GGML_TYPE_Q4_K)) {
        throw std::runtime_error("unsupported ssm_out score type: " + type_name(st->type));
    }

    const size_t src_row_size = ggml_row_size(st->type, 4096);
    const size_t dst_row_size = ggml_row_size(st->type, 2048);
    std::vector<uint8_t> src_row(src_row_size), packed(dst_row_size);
    std::vector<float> src_decoded(4096), gathered(2048), requantized(2048);
    ssm_out_requant_error stats;
    for (int r = 0; r < HIDDEN; ++r) {
        src.read_tensor_bytes(name, (size_t) r * src_row_size, src_row.data(), src_row.size());
        traits->to_float(src_row.data(), src_decoded.data(), 4096);
        for (int j = 0; j < 2048; ++j) {
            gathered[(size_t) j] = src_decoded[(size_t) channels[(size_t) j]];
        }
        traits->from_float_ref(gathered.data(), packed.data(), 2048);
        traits->to_float(packed.data(), requantized.data(), 2048);
        for (int j = 0; j < 2048; ++j) {
            const double target = gathered[(size_t) j];
            const double error = (double) requantized[(size_t) j] - target;
            stats.target_energy += (long double) target * target;
            stats.error_energy += (long double) error * error;
            stats.max_abs_error = std::max(stats.max_abs_error, std::fabs(error));
            ++stats.values;
        }
    }
    return stats;
}

static void command_score_gdn_requantization(
        const std::string & source,
        const std::string & plan_path,
        int only_layer) {
    source_gguf src(source);
    require_source_contract(src);
    const json plan = read_json(plan_path);
    verify_plan_json(plan);
    const plan_index idx = index_plan(plan);
    ssm_out_requant_error total;
    for (int layer : RECURRENT_LAYERS) {
        if (only_layer >= 0 && layer != only_layer) continue;
        const auto it = idx.gdn.find(layer);
        if (it == idx.gdn.end()) throw std::runtime_error("missing GDN requant score plan record");
        const std::string name = "blk." + std::to_string(layer) + ".ssm_out.weight";
        const auto stats = measure_ssm_out_requantize(src, name, it->second);
        total.values += stats.values;
        total.target_energy += stats.target_energy;
        total.error_energy += stats.error_energy;
        total.max_abs_error = std::max(total.max_abs_error, stats.max_abs_error);
        std::cout << "GDN ssm_out requant layer " << layer
                  << ": rel_rms=" << stats.relative_rms()
                  << " rms=" << stats.rms()
                  << " max_abs=" << stats.max_abs_error << "\n";
    }
    std::cout << "GDN ssm_out requant total: rel_rms=" << total.relative_rms()
              << " rms=" << total.rms()
              << " max_abs=" << total.max_abs_error << "\n";
    std::cerr << "note: score-gdn-requantization is read-only and intentionally does not rescan source SHA-256\n";
}

static ssm_out_requant_error verify_ssm_out_requantize(
        const source_gguf & src,
        const source_gguf & out,
        const std::string & name,
        const json & rec) {
    const ggml_tensor * st = src.tensor(name);
    const ggml_tensor * ot = out.tensor(name);
    if (st->type != ot->type || st->ne[0] != 4096 || ot->ne[0] != 2048 || st->ne[1] != HIDDEN || ot->ne[1] != HIDDEN) {
        throw std::runtime_error("ssm_out descriptor mismatch: " + name);
    }
    const auto channels = gdn_v_channel_mapping(rec);
    if (channels.size() != 2048) throw std::runtime_error("ssm_out verify channel map mismatch");

    const size_t src_row_size = ggml_row_size(st->type, 4096);
    const size_t dst_row_size = ggml_row_size(st->type, 2048);
    const auto * traits = ggml_get_type_traits(st->type);
    if (!traits || !traits->to_float || !traits->from_float_ref ||
        (st->type != GGML_TYPE_Q8_0 && st->type != GGML_TYPE_Q4_K)) {
        throw std::runtime_error("unsupported ssm_out verify type: " + type_name(st->type));
    }
    std::vector<uint8_t> src_row(src_row_size), dst_row(dst_row_size), expected(dst_row_size);
    std::vector<float> src_decoded(4096), gathered(2048), dst_decoded(2048);
    ssm_out_requant_error stats;
    for (int r = 0; r < HIDDEN; ++r) {
        src.read_tensor_bytes(name, (size_t) r * src_row_size, src_row.data(), src_row.size());
        out.read_tensor_bytes(name, (size_t) r * dst_row_size, dst_row.data(), dst_row.size());
        traits->to_float(src_row.data(), src_decoded.data(), 4096);
        for (int j = 0; j < 2048; ++j) {
            gathered[(size_t) j] = src_decoded[(size_t) channels[(size_t) j]];
        }
        traits->from_float_ref(gathered.data(), expected.data(), 2048);
        if (expected != dst_row) {
            throw std::runtime_error("ssm_out arbitrary-gather requantization mismatch: " + name +
                                     " row " + std::to_string(r));
        }
        traits->to_float(dst_row.data(), dst_decoded.data(), 2048);
        for (int j = 0; j < 2048; ++j) {
            const double target = gathered[(size_t) j];
            const double error = (double) dst_decoded[(size_t) j] - target;
            stats.target_energy += (long double) target * target;
            stats.error_energy += (long double) error * error;
            stats.max_abs_error = std::max(stats.max_abs_error, std::fabs(error));
            ++stats.values;
        }
    }
    return stats;
}

static void command_verify_gdn_layer(
        const std::string & source,
        const std::string & output,
        const std::string & plan_path,
        int layer) {
    if (!is_recurrent_layer(layer)) throw std::runtime_error("verify-gdn-layer layer is not recurrent");
    source_gguf src(source), out(output);
    require_source_contract(src);
    const json plan = read_json(plan_path);
    verify_plan_json(plan);
    const plan_index idx = index_plan(plan);
    const auto it = idx.gdn.find(layer);
    if (it == idx.gdn.end()) throw std::runtime_error("verify-gdn-layer plan record missing");

    const std::string prefix = "blk." + std::to_string(layer) + ".";
    const std::string qkv  = prefix + "attn_qkv.weight";
    const std::string gate = prefix + "attn_gate.weight";
    const std::string conv = prefix + "ssm_conv1d.weight";
    const std::string norm = prefix + "ssm_norm.weight";
    const std::string sout = prefix + "ssm_out.weight";
    const json & rec = it->second;
    verify_selected_rows_exact(src, out, qkv, gdn_qkv_mapping(rec));
    verify_selected_rows_exact(src, out, gate, gdn_v_channel_mapping(rec));
    verify_selected_rows_exact(src, out, conv, gdn_qkv_mapping(rec));
    verify_ssm_norm_gather(src, out, norm, rec);
    const auto requant = verify_ssm_out_requantize(src, out, sout, rec);
    std::cout << "verify GDN layer " << layer << " OK (exact row gather + stock ssm_out requantization)\n"
              << "  ssm_out requant: rel_rms=" << requant.relative_rms()
              << " rms=" << requant.rms()
              << " max_abs=" << requant.max_abs_error << "\n";
}

static void command_verify(
        const std::string & source,
        const std::string & output,
        const std::string & plan_path,
        const std::string & corpus_path) {
    source_gguf src(source), out(output);
    require_source_contract(src);
    const json plan=read_json(plan_path); verify_plan_json(plan);
    const plan_index idx = index_plan(plan);
    const std::string manifest_path = output + ".manifest.json";
    bool keep_gdn = false;
    bool keep_vocab = false;
    bool expert_source_quant = false;
    bool expert_energy_match = false;
    {
        std::ifstream manifest_in(manifest_path);
        if (manifest_in.good()) {
            manifest_in.close();
            const json manifest = read_json(manifest_path);
            keep_gdn = manifest.value("diagnostic_keep_gdn", false);
            keep_vocab = manifest.value("diagnostic_keep_vocab", false);
            expert_source_quant = manifest.value("expert_source_quant", false);
            expert_energy_match = manifest.value("expert_energy_match", false);
        }
    }
    if (plan.value("expert_only", false) &&
        (!keep_gdn || !keep_vocab || !expert_source_quant || expert_energy_match)) {
        throw std::runtime_error(
            "expert-only verification requires keep-GDN, keep-vocab, source-quant experts, and no energy matching");
    }
    if(sha256_file(source)!=plan.at("source_sha256").get<std::string>()) throw std::runtime_error("source hash mismatch");
    if(out.tensor_count()!=753) throw std::runtime_error("output tensor count mismatch");
    if(out.get_str("general.architecture")!="qwen35moe" || out.get_u32("qwen35moe.block_count")!=41 || out.get_u32("qwen35moe.nextn_predict_layers")!=1) throw std::runtime_error("architecture/MTP metadata changed");
    if(out.get_u32("qwen35moe.expert_feed_forward_length")!=EXPERT_WIDTH_DST) throw std::runtime_error("target expert metadata mismatch");
    if (keep_gdn) {
        if (read_integer_metadata(src.meta(), "qwen35moe.ssm.state_size") != read_integer_metadata(out.meta(), "qwen35moe.ssm.state_size") ||
            read_integer_metadata(src.meta(), "qwen35moe.ssm.inner_size") != read_integer_metadata(out.meta(), "qwen35moe.ssm.inner_size")) {
            throw std::runtime_error("keep-GDN metadata changed");
        }
    } else if(out.get_u32("qwen35moe.ssm.state_size")!=64 || out.get_u32("qwen35moe.ssm.inner_size")!=2048) {
        throw std::runtime_error("target GDN metadata mismatch");
    }
    for (const char * key : {
            "qwen35moe.embedding_length",
            "qwen35moe.expert_count",
            "qwen35moe.expert_used_count",
            "qwen35moe.expert_shared_feed_forward_length",
            "qwen35moe.ssm.group_count",
            "qwen35moe.ssm.time_step_rank",
            "general.file_type"}) {
        if (read_integer_metadata(src.meta(), key) != read_integer_metadata(out.meta(), key)) {
            throw std::runtime_error(std::string("unexpected metadata change: ") + key);
        }
    }

    std::array<int,4> counts{};
    uint64_t q2blocks=0, neg=0; std::array<uint64_t,4> hist{};
    uint64_t expert_payload_bytes = 0;
    ssm_out_requant_error gdn_requant_total;
    for(int64_t id=0;id<src.tensor_count();++id){
        const std::string name=gguf_get_tensor_name(src.meta(),id); const ggml_tensor * st=src.tensor(name); const tensor_policy p=classify_tensor(name,st); ++counts[(int)p];
        const ggml_tensor * ot=out.tensor(name);
        if(p==tensor_policy::copy){ if(st->type!=ot->type||ggml_n_dims(st)!=ggml_n_dims(ot)||src.tensor_size(name)!=out.tensor_size(name)||!tensors_equal(src,out,name)) throw std::runtime_error("policy-D mismatch: "+name); }
        else if(p==tensor_policy::expert_q2){
            expert_payload_bytes += out.tensor_size(name);
            if(ends_with(name,"ffn_down_exps.weight")){ if(ot->ne[0]!=EXPERT_WIDTH_DST||ot->ne[1]!=2048||ot->ne[2]!=N_EXPERT) throw std::runtime_error("down expert shape mismatch"); }
            else if(ot->ne[0]!=2048||ot->ne[1]!=EXPERT_WIDTH_DST||ot->ne[2]!=N_EXPERT) throw std::runtime_error("gate/up expert shape mismatch");
            if (expert_source_quant) {
                if (!ends_with(name,"ffn_down_exps.weight") && ot->type != st->type) {
                    throw std::runtime_error("source-quant gate/up expert type changed: " + name);
                }
                // With energy matching disabled, the source-quant ablation has
                // a byte-reproducible contract: gate/up rows are copied exactly
                // and down rows are gather-then-requantized with the source
                // tensor type. Verify that contract rather than accepting only
                // the expected shape/type.
                if (!expert_energy_match) {
                    verify_source_quant_expert_tensor(src, out, name, idx);
                }
            } else {
                if(ot->type!=GGML_TYPE_Q2_0) throw std::runtime_error("expert not Q2_0: "+name);
                if(out.tensor_size(name)!=37748736ULL) throw std::runtime_error("expert Q2 payload size mismatch: "+name);
                std::vector<uint8_t> data(out.tensor_size(name)); out.read_tensor_bytes(name,0,data.data(),data.size());
                if(data.size()%sizeof(q2_opt_block)) throw std::runtime_error("Q2 payload alignment mismatch");
                for(size_t off=0;off<data.size();off+=sizeof(q2_opt_block)){
                    q2_opt_block b{}; std::memcpy(&b,data.data()+off,sizeof(b)); const float d=ggml_fp16_to_fp32(b.d);
                    if(!std::isfinite(d)||(d==0.0f&&std::signbit(d))) throw std::runtime_error("invalid Q2 scale in "+name);
                    ++q2blocks; if(std::signbit(d)&&d!=0.0f)++neg; for(int i=0;i<64;++i)++hist[(b.qs[i/4]>>(2*(i%4)))&3u];
                }
            }
        } else if(p==tensor_policy::vocab){
            if (keep_vocab) {
                if (st->type != ot->type || ggml_n_dims(st) != ggml_n_dims(ot) ||
                    src.tensor_size(name) != out.tensor_size(name) || !tensors_equal(src, out, name)) {
                    throw std::runtime_error("keep-vocab tensor mismatch: " + name);
                }
            } else {
                if(ot->ne[1]!=VOCAB_DST||ot->type!=st->type) throw std::runtime_error("vocab tensor mismatch: "+name);
                verify_selected_rows_exact(src, out, name, idx.vocab);
            }
        }
        else {
            if (keep_gdn) {
                if(st->type!=ot->type || ggml_n_dims(st)!=ggml_n_dims(ot) ||
                   src.tensor_size(name)!=out.tensor_size(name) || !tensors_equal(src,out,name)) {
                    throw std::runtime_error("keep-GDN tensor mismatch: "+name);
                }
                continue;
            }
            if(ot->type!=st->type) throw std::runtime_error("GDN type changed: "+name);
            const int layer = parse_block_id(name);
            const auto it = idx.gdn.find(layer);
            if (it == idx.gdn.end()) throw std::runtime_error("missing GDN verify plan record: " + name);
            if (ends_with(name, "attn_qkv.weight")) {
                verify_selected_rows_exact(src, out, name, gdn_qkv_mapping(it->second));
            } else if (ends_with(name, "ssm_conv1d.weight")) {
                verify_selected_rows_exact(src, out, name, gdn_qkv_mapping(it->second));
            } else if (ends_with(name, "attn_gate.weight")) {
                verify_selected_rows_exact(src, out, name, gdn_v_channel_mapping(it->second));
            } else if (ends_with(name, "ssm_norm.weight")) {
                verify_ssm_norm_gather(src, out, name, it->second);
            } else if (ends_with(name, "ssm_out.weight")) {
                const auto requant = verify_ssm_out_requantize(src, out, name, it->second);
                gdn_requant_total.values += requant.values;
                gdn_requant_total.target_energy += requant.target_energy;
                gdn_requant_total.error_energy += requant.error_energy;
                gdn_requant_total.max_abs_error = std::max(
                        gdn_requant_total.max_abs_error, requant.max_abs_error);
            }
        }
    }
    const uint32_t block_count = src.get_u32("qwen35moe.block_count");
    const int exp_experts = (int) block_count * 3;
    const int exp_gdn = (int) RECURRENT_LAYERS.size() * 5;
    const int exp_vocab = 2;
    const int exp_copy = (int) src.tensor_count() - (exp_experts + exp_gdn + exp_vocab);
    if(counts!=std::array<int,4>{exp_experts,exp_gdn,exp_vocab,exp_copy}) throw std::runtime_error("policy counts mismatch");
    if(!expert_source_quant && expert_payload_bytes!=4643094528ULL) throw std::runtime_error("aggregate expert Q2 payload size mismatch");
    if (!keep_gdn) {
        for(int layer:RECURRENT_LAYERS){
            if(out.tensor("blk."+std::to_string(layer)+".attn_qkv.weight")->ne[1]!=4096||out.tensor("blk."+std::to_string(layer)+".attn_gate.weight")->ne[1]!=2048||out.tensor("blk."+std::to_string(layer)+".ssm_conv1d.weight")->ne[1]!=4096||out.tensor("blk."+std::to_string(layer)+".ssm_norm.weight")->ne[0]!=64||out.tensor("blk."+std::to_string(layer)+".ssm_out.weight")->ne[0]!=2048) throw std::runtime_error("GDN shape mismatch at layer "+std::to_string(layer));
        }
    }
    if (keep_vocab) verify_tokenizer_unchanged(src, out);
    else verify_tokenizer(src, out, plan);
    if(expert_source_quant) {
        if(q2blocks != 0) throw std::runtime_error("Q2 blocks present in source-quant expert ablation");
    } else if(q2blocks==0||std::accumulate(hist.begin(),hist.end(),uint64_t(0))!=q2blocks*64) {
        throw std::runtime_error("Q2 histogram invalid");
    }
    {
        std::ifstream manifest_in(manifest_path);
        if (manifest_in.good()) {
            manifest_in.close();
            const json manifest = read_json(manifest_path);
            if (expert_source_quant) {
                const std::string expected_quant = EXPERT_WIDTH_DST % 256 == 0
                    ? "source-q4km-mix"
                    : "source-q4km-gate-up-q4_0-down";
                if (manifest.value("expert_quantization", std::string()) != expected_quant) {
                    throw std::runtime_error("manifest does not declare expected source-derived expert quantization");
                }
            } else {
                if (!manifest.contains("q2")) throw std::runtime_error("manifest Q2 stats missing");
                const json & mq2 = manifest.at("q2");
                if (mq2.value("blocks", uint64_t(0)) != q2blocks) throw std::runtime_error("manifest/output Q2 block count mismatch");
                if (mq2.value("negative_d", uint64_t(0)) != neg) throw std::runtime_error("manifest/output negative-d count mismatch");
                if (!mq2.contains("code_hist") || mq2.at("code_hist").get<std::array<uint64_t,4>>() != hist) {
                    throw std::runtime_error("manifest/output Q2 code histogram mismatch");
                }
            }
        }
    }
    std::cout<<"verify OK\npolicy A/B/C/D: "<<exp_experts<<"/"<<exp_gdn<<"/"<<exp_vocab<<"/"<<exp_copy<<"\n";
    if (expert_source_quant) {
        std::cout << "expert quantization: source-derived gate/up with compatible down requantization (Q2 ablated), payload_bytes="
                  << expert_payload_bytes << "\n";
    } else {
        std::cout << "Q2 blocks: "<<q2blocks<<" negative-d: "<<neg<<" ("<<(100.0*neg/q2blocks)<<"%)\ncodes: "<<hist[0]<<","<<hist[1]<<","<<hist[2]<<","<<hist[3]<<"\n";
    }
    std::cout <<"GDN ssm_out requant: rel_rms="<<gdn_requant_total.relative_rms()
             <<" rms="<<gdn_requant_total.rms()
             <<" max_abs="<<gdn_requant_total.max_abs_error<<"\n";
    if (!corpus_path.empty()) {
        const bool strict_tokenizer = plan.value("plan_q2_robustness", std::string()) != "smoke-r=1";
        const json tokenizer = audit_tokenizers(source, output, corpus_path, strict_tokenizer);
        std::cout << "tokenizer audit:\n" << std::setw(2) << tokenizer << '\n';
        std::ifstream manifest_in(manifest_path);
        if (manifest_in.good()) {
            manifest_in.close();
            json manifest = read_json(manifest_path);
            manifest["tokenizer_audit"] = tokenizer;
            write_json(manifest_path, manifest);
        }
        llama_backend_free();
    }
}

static void command_finalize(
        const std::string & source,
        const std::string & tmp,
        const std::string & plan_path,
        const std::string & corpus_path,
        const std::string & final_path) {
    if (!ends_with(tmp, ".tmp")) throw std::runtime_error("finalize input must end in .tmp");
    const json plan = read_json(plan_path);
    verify_plan_json(plan);
    require_materializable_enp_plan(plan);
    if (plan.value("plan_q2_robustness", std::string()) == "smoke-r=1") {
        throw std::runtime_error("refusing to finalize a --quick smoke plan; use a fully covered strict calibration/plan");
    }
    if (plan.value("allow_uncovered_experts", false)) {
        // Production exception: the one MTP block is not part of the target
        // model's next-token path and can legitimately leave some router
        // experts unobserved even when all 40 main blocks are fully covered.
        // Only that exact case may use the existing weight-only fallback.
        size_t mtp_zero = 0;
        for (const auto & r : plan.at("experts")) {
            if (!r.value("zero_routing_weight_only_fallback", false)) continue;
            if (r.at("layer").get<int>() != N_LAYER_MAIN) {
                throw std::runtime_error("refusing to finalize: uncovered expert exists in a main-model block");
            }
            if (r.value("routing_count", int64_t(-1)) != 0) {
                throw std::runtime_error("refusing to finalize: MTP fallback record has nonzero routing_count");
            }
            ++mtp_zero;
        }
        const size_t declared_zero = plan.at("calibration").value("zero_routing_experts", size_t(0));
        if (mtp_zero == 0 || mtp_zero != declared_zero) {
            throw std::runtime_error("refusing to finalize: uncovered-expert allowance is not exactly MTP-only");
        }
        std::cerr << "finalize: accepting " << mtp_zero
                  << " MTP-only zero-routing experts via weight-only fallback; main blocks remain fully covered\n";
    }

    // This is deliberately the only path that turns a .tmp into a final model:
    // structural bytes, Q2/GDN invariants and tokenizer hard limits all gate it.
    command_verify(source, tmp, plan_path, corpus_path);

    const std::string tmp_manifest = tmp + ".manifest.json";
    const std::string final_manifest = final_path + ".manifest.json";
    {
        std::ifstream manifest(tmp_manifest);
        if (!manifest.good()) throw std::runtime_error("finalize manifest missing: " + tmp_manifest);
    }
    {
        std::ifstream existing(final_path);
        if (existing.good()) throw std::runtime_error("refusing to overwrite existing final model: " + final_path);
    }
    {
        std::ifstream existing(final_manifest);
        if (existing.good()) throw std::runtime_error("refusing to overwrite existing final manifest: " + final_manifest);
    }

    if (std::rename(tmp_manifest.c_str(), final_manifest.c_str()) != 0) {
        throw std::runtime_error("failed to rename manifest during finalize");
    }
    if (std::rename(tmp.c_str(), final_path.c_str()) != 0) {
        std::rename(final_manifest.c_str(), tmp_manifest.c_str());
        throw std::runtime_error("failed to atomically rename model during finalize");
    }
    std::cout << "finalized GGUF: " << final_path << "\n"
              << "sha256: " << sha256_file(final_path) << "\n"
              << "manifest: " << final_manifest << "\n";
}

static void usage(const char * argv0) {
    std::cerr
        << "usage:\n"
        << "  " << argv0 << " inspect MODEL.gguf\n"
        << "  " << argv0 << " plan MODEL.gguf IMATRIX.gguf PLAN.json [--quick] [--allow-uncovered] [--expert-only] [--threads N]\n"
        << "  " << argv0 << " plan-gdn-geometry-v1 TEACHER.gguf BASE_PLAN.json CORPUS OUT_PLAN.json [TOKENS=512] [BATCH=32] [SHARDS=2] [SELECTION=topk]\n"
        << "  " << argv0 << " plan-gdn-joint-v1 TEACHER.gguf BASE_PLAN.json CORPUS OUT_PLAN.json [TOKENS=512] [BATCH=32] [SHARDS=2] [REPLAY=128] [ONLY_LAYER=-1]\n"
        << "  " << argv0 << " score-ppl MODEL.gguf CORPUS [TOKENS=256] [SCORE_FROM=-1] [BATCH=32]\n"
        << "  " << argv0 << " score-gdn-requantization SOURCE.gguf PLAN.json [LAYER]\n"
        << "  " << argv0 << " rewrite-gdn-inplace SOURCE.gguf PLAN.json TARGET.gguf.tmp [LAYER]\n"
        << "  " << argv0 << " probe-expert MODEL.gguf IMATRIX.gguf LAYER EXPERT\n"
        << "  " << argv0 << " probe-expert-heldout MODEL.gguf SELECTION_IMATRIX.gguf HELDOUT_IMATRIX.gguf LAYER EXPERT\n"
        << "  " << argv0 << " probe-expert-heldout-lens MODEL.gguf SELECTION_IMATRIX.gguf LENS_IMATRIX.gguf HELDOUT_IMATRIX.gguf LAYER EXPERT\n"
        << "  " << argv0 << " plan-expert-downstream-384 MODEL.gguf SELECTION_IMATRIX.gguf LENS_IMATRIX.gguf PLAN.json [THREADS]\n"
        << "  " << argv0 << " plan-expert-replacement-192 MODEL384.gguf CALIBRATION.gguf PLAN.json [THREADS]\n"
        << "  " << argv0 << " apply-expert-replacement-192 MODEL384.gguf PLAN.json OUT.gguf.tmp\n"
        << "  " << argv0 << " verify-expert-replacement-192 MODEL384.gguf OUT.gguf PLAN.json\n"
        << "  " << argv0 << " finalize-expert-replacement-192 MODEL384.gguf OUT.gguf.tmp PLAN.json FINAL.gguf\n"
        << "  " << argv0 << " fit-gdn-function-projection TEACHER.gguf STUDENT.gguf PLAN.json TRAIN_CORPUS VALID_CORPUS OUT.gdnfp LAYER [TRAIN_TOKENS] [VALID_TOKENS] [RIDGE_CENTER] [RIDGE_DECADES] [THREADS]\n"
        << "  " << argv0 << " rewrite-gdn-function-projection REG.gdnfp TARGET.gguf.tmp [THREADS]\n"
        << "  " << argv0 << " select-gdn-function-projection-global TEACHER.gguf STUDENT.gguf PLAN.json TRAIN_CORPUS VALID_CORPUS OUT.gdnfp LAYER [TRAIN_TOKENS] [VALID_TOKENS] [RIDGE_CENTER] [RIDGE_DECADES] [TEMPERATURE] [kl|nll] [SCORE_FROM] [THREADS]\n"
        << "  " << argv0 << " select-gdn-qk-atom-exchange-global TEACHER.gguf STUDENT.gguf PLAN.json TRAIN_CORPUS VALID_CORPUS OUT_PLAN.json LAYER HEAD [TRAIN_TOKENS] [VALID_TOKENS] [RIDGE_CENTER] [RIDGE_DECADES] [TEMPERATURE] [kl|nll] [SCORE_FROM] [THREADS]\n"
        << "  " << argv0 << " score-gdn-function-risk TEACHER.gguf STUDENT.gguf PLAN.json CORPUS [TOKENS] [TEMPERATURE] [SCORE_FROM]\n"
        << "  " << argv0 << " vocab-fixture MODEL.gguf IMATRIX.gguf OUT.gguf\n"
        << "  " << argv0 << " audit-tokenizer SOURCE.gguf OUTPUT.gguf CORPUS [--no-hard-limit]\n"
        << "  " << argv0 << " apply MODEL.gguf IMATRIX.gguf PLAN.json OUT.gguf.tmp [--threads N] [--max-tensors N] [--keep-gdn] [--keep-vocab] [--keep-experts] [--expert-energy-match] [--expert-source-quant]\n"
        << "  " << argv0 << " verify SOURCE.gguf OUTPUT.gguf PLAN.json [CORPUS]\n"
        << "  " << argv0 << " verify-gdn-layer SOURCE.gguf OUTPUT.gguf PLAN.json LAYER\n"
        << "  " << argv0 << " finalize SOURCE.gguf OUT.gguf.tmp PLAN.json CORPUS FINAL.gguf\n";
}

// apply/verify implementation is below; kept in one TU so the GGUF inventory,
// tokenizer surgery and quantized GDN rewriter share exactly the same layout helpers.

} // namespace

int main(int argc, char ** argv) {
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        ggml_quantize_init((ggml_type) t);
    }
    try {
        if (argc < 3) { usage(argv[0]); return 2; }
        const std::string cmd = argv[1];
        if (cmd == "inspect" && argc == 3) {
            command_inspect(argv[2]);
            return 0;
        }
        if (cmd == "plan" && argc >= 5) {
            bool quick = false;
            bool allow_uncovered = false;
            bool expert_only = false;
            int threads = std::max(1, (int) std::thread::hardware_concurrency());
            for (int i = 5; i < argc; ) {
                const std::string opt = argv[i++];
                if (opt == "--quick") {
                    quick = true;
                } else if (opt == "--allow-uncovered") {
                    allow_uncovered = true;
                } else if (opt == "--expert-only") {
                    expert_only = true;
                } else if (opt == "--threads") {
                    if (i >= argc) throw std::runtime_error("missing --threads value");
                    threads = std::max(1, std::stoi(argv[i++]));
                } else {
                    throw std::runtime_error("unknown plan option: " + opt);
                }
            }
            command_plan(argv[2], argv[3], argv[4], quick, allow_uncovered, expert_only, threads);
            return 0;
        }
        if (cmd == "plan-gdn-geometry-v1" && argc >= 6 && argc <= 10) {
            command_plan_gdn_geometry_v1(
                    argv[2], argv[3], argv[4], argv[5],
                    argc >= 7 ? std::stoi(argv[6]) : 512,
                    argc >= 8 ? std::stoi(argv[7]) : 32,
                    argc >= 9 ? std::stoi(argv[8]) : 2,
                    argc >= 10 ? argv[9] : "topk");
            return 0;
        }
        if (cmd == "plan-gdn-joint-v1" && argc >= 6 && argc <= 11) {
            command_plan_gdn_joint_v1(
                    argv[2], argv[3], argv[4], argv[5],
                    argc >= 7 ? std::stoi(argv[6]) : 512,
                    argc >= 8 ? std::stoi(argv[7]) : 32,
                    argc >= 9 ? std::stoi(argv[8]) : 2,
                    argc >= 10 ? std::stoi(argv[9]) : 128,
                    argc >= 11 ? std::stoi(argv[10]) : -1);
            return 0;
        }
        if (cmd == "score-ppl" && argc >= 4 && argc <= 7) {
            command_score_ppl(
                    argv[2], argv[3],
                    argc >= 5 ? std::stoi(argv[4]) : 256,
                    argc >= 6 ? std::stoi(argv[5]) : -1,
                    argc >= 7 ? std::stoi(argv[6]) : 32);
            return 0;
        }
        if (cmd == "score-gdn-requantization" && (argc == 4 || argc == 5)) {
            command_score_gdn_requantization(argv[2], argv[3], argc == 5 ? std::stoi(argv[4]) : -1);
            return 0;
        }
        if (cmd == "rewrite-gdn-inplace" && (argc == 5 || argc == 6)) {
            command_rewrite_gdn_inplace(argv[2], argv[3], argv[4],
                                        argc == 6 ? std::stoi(argv[5]) : -1);
            return 0;
        }
        if (cmd == "probe-expert" && argc == 6) {
            command_probe_expert(argv[2], argv[3], std::stoi(argv[4]), std::stoi(argv[5]));
            return 0;
        }
        if (cmd == "probe-expert-heldout" && argc == 7) {
            command_probe_expert_heldout(
                argv[2], argv[3], argv[4], std::stoi(argv[5]), std::stoi(argv[6]));
            return 0;
        }
        if (cmd == "probe-expert-heldout-lens" && argc == 8) {
            command_probe_expert_heldout_lens(
                argv[2], argv[3], argv[4], argv[5], std::stoi(argv[6]), std::stoi(argv[7]));
            return 0;
        }
        if (cmd == "plan-expert-downstream-384" && (argc == 6 || argc == 7)) {
            command_plan_expert_downstream_384(
                argv[2], argv[3], argv[4], argv[5],
                argc == 7 ? std::max(1, std::stoi(argv[6])) :
                            std::max(1, (int) std::thread::hardware_concurrency()));
            return 0;
        }
        if (cmd == "plan-expert-replacement-192" && (argc == 5 || argc == 6)) {
            command_plan_expert_replacement_192(
                argv[2], argv[3], argv[4],
                argc == 6 ? std::max(1, std::stoi(argv[5])) :
                            std::max(1, (int) std::thread::hardware_concurrency()));
            return 0;
        }
        if (cmd == "apply-expert-replacement-192" && argc == 5) {
            command_apply_expert_replacement_192(argv[2], argv[3], argv[4]);
            return 0;
        }
        if (cmd == "verify-expert-replacement-192" && argc == 5) {
            command_verify_expert_replacement_192(argv[2], argv[3], argv[4]);
            return 0;
        }
        if (cmd == "finalize-expert-replacement-192" && argc == 6) {
            command_finalize_expert_replacement_192(argv[2], argv[3], argv[4], argv[5]);
            return 0;
        }
        if (cmd == "fit-gdn-function-projection" && (argc >= 9 && argc <= 14)) {
            command_fit_gdn_function_projection(
                    argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], std::stoi(argv[8]),
                    argc >= 10 ? std::stoi(argv[9]) : 256,
                    argc >= 11 ? std::stoi(argv[10]) : 256,
                    argc >= 12 ? std::stod(argv[11]) : 1e-2,
                    argc >= 13 ? std::stoi(argv[12]) : 4,
                    argc >= 14 ? std::max(1, std::stoi(argv[13])) :
                                 std::max(1, (int) std::thread::hardware_concurrency()));
            return 0;
        }
        if (cmd == "rewrite-gdn-function-projection" && (argc == 4 || argc == 5)) {
            command_rewrite_gdn_function_projection(
                    argv[2], argv[3],
                    argc == 5 ? std::max(1, std::stoi(argv[4])) :
                                std::max(1, (int) std::thread::hardware_concurrency()));
            return 0;
        }
        if (cmd == "score-gdn-function-risk" && (argc >= 6 && argc <= 9)) {
            command_score_gdn_function_risk(
                    argv[2], argv[3], argv[4], argv[5],
                    argc >= 7 ? std::stoi(argv[6]) : 256,
                    argc >= 8 ? std::stod(argv[7]) : 1.0,
                    argc >= 9 ? std::stoi(argv[8]) : -1);
            return 0;
        }
        if (cmd == "select-gdn-function-projection-global" && (argc >= 9 && argc <= 17)) {
            command_select_gdn_function_projection_global(
                    argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], std::stoi(argv[8]),
                    argc >= 10 ? std::stoi(argv[9]) : 256,
                    argc >= 11 ? std::stoi(argv[10]) : 256,
                    argc >= 12 ? std::stod(argv[11]) : 1e-2,
                    argc >= 13 ? std::stoi(argv[12]) : 4,
                    argc >= 14 ? std::stod(argv[13]) : 1.0,
                    argc >= 15 ? argv[14] : "kl",
                    argc >= 16 ? std::stoi(argv[15]) : -1,
                    argc >= 17 ? std::max(1, std::stoi(argv[16])) :
                                 std::max(1, (int) std::thread::hardware_concurrency()));
            return 0;
        }
        if (cmd == "select-gdn-qk-atom-exchange-global" && (argc >= 10 && argc <= 18)) {
            command_select_gdn_qk_atom_exchange_global(
                    argv[2], argv[3], argv[4], argv[5], argv[6], argv[7],
                    std::stoi(argv[8]), std::stoi(argv[9]),
                    argc >= 11 ? std::stoi(argv[10]) : 128,
                    argc >= 12 ? std::stoi(argv[11]) : 128,
                    argc >= 13 ? std::stod(argv[12]) : 1e-2,
                    argc >= 14 ? std::stoi(argv[13]) : 2,
                    argc >= 15 ? std::stod(argv[14]) : 1.0,
                    argc >= 16 ? argv[15] : "kl",
                    argc >= 17 ? std::stoi(argv[16]) : -1,
                    argc >= 18 ? std::max(1, std::stoi(argv[17])) :
                                 std::max(1, (int) std::thread::hardware_concurrency()));
            return 0;
        }
        if (cmd == "vocab-fixture" && argc == 5) {
            command_vocab_fixture(argv[2], argv[3], argv[4]);
            return 0;
        }
        if (cmd == "audit-tokenizer" && (argc == 5 || argc == 6)) {
            const bool enforce = argc == 5 || std::string(argv[5]) != "--no-hard-limit";
            if (argc == 6 && enforce) throw std::runtime_error("unknown audit-tokenizer option");
            const json tokenizer = audit_tokenizers(argv[2], argv[3], argv[4], enforce);
            std::cout << std::setw(2) << tokenizer << '\n';
            llama_backend_free();
            return 0;
        }
        if (cmd == "apply" && argc >= 6) {
            int threads = (int) std::thread::hardware_concurrency();
            int max_tensors = 0;
            bool keep_gdn = false;
            bool keep_vocab = false;
            bool keep_experts = false;
            bool expert_energy_match = false;
            bool expert_source_quant = false;
            for (int i = 6; i < argc; ) {
                const std::string opt = argv[i++];
                if (opt == "--keep-gdn") {
                    keep_gdn = true;
                    continue;
                }
                if (opt == "--keep-vocab") {
                    keep_vocab = true;
                    continue;
                }
                if (opt == "--keep-experts") {
                    keep_experts = true;
                    continue;
                }
                if (opt == "--expert-energy-match") {
                    expert_energy_match = true;
                    continue;
                }
                if (opt == "--expert-source-quant") {
                    expert_source_quant = true;
                    continue;
                }
                if (i >= argc) throw std::runtime_error("missing apply option value");
                const int value = std::stoi(argv[i++]);
                if (opt == "--threads") threads = std::max(1, value);
                else if (opt == "--max-tensors") max_tensors = std::max(1, value);
                else throw std::runtime_error("unknown apply option: " + opt);
            }
            command_apply(argv[2], argv[3], argv[4], argv[5], threads, max_tensors,
                          keep_gdn, keep_vocab, keep_experts, expert_energy_match, expert_source_quant);
            return 0;
        }
        if (cmd == "verify" && (argc == 5 || argc == 6)) {
            command_verify(argv[2], argv[3], argv[4], argc == 6 ? argv[5] : "");
            return 0;
        }
        if (cmd == "verify-gdn-layer" && argc == 6) {
            command_verify_gdn_layer(argv[2], argv[3], argv[4], std::stoi(argv[5]));
            return 0;
        }
        if (cmd == "finalize" && argc == 7) {
            command_finalize(argv[2], argv[3], argv[4], argv[5], argv[6]);
            return 0;
        }
        usage(argv[0]);
        return 2;
    } catch (const std::exception & e) {
        std::cerr << "llama-qwen35-prune: " << e.what() << '\n';
        return 1;
    }
}
