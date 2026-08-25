#include "arg.h"
#include "common.h"
#include "enp-geometric-coreset.h"
#include "imatrix-loader.h"
#include "log.h"
#include "llama.h"
#include "../common/speculative.h"
#include "gguf.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <functional>
#include <thread>
#include <mutex>
#include <vector>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <limits>
#include <regex>
#include <numeric>
#include <set>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

static void print_usage(int, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n    %s \\\n"
            "       -m model.gguf -f some-text.txt [-o imatrix.gguf] [--output-format {gguf,dat}] [--no-ppl] \\\n"
            "       [--process-output] [--chunk 123] [--save-frequency 0] [--output-frequency 10] \\\n"
            "       [--in-file imatrix-prev-0.gguf --in-file imatrix-prev-1.gguf ...] [--parse-special] \\\n"
            "       [--show-statistics] [...]\n" , argv[0]);
    LOG("\n");
}

struct request_token_trie_node {
    uint64_t parent = 0;
    llama_token token = 0;
    bool present = false;
    bool live = false;
};

struct request_token_trie {
    std::vector<request_token_trie_node> nodes;
    std::vector<std::vector<uint64_t>> children;
    std::vector<uint64_t> leaves;
    uint64_t request_records = 0;
    uint64_t unique_tokens = 0;
    size_t min_leaf_tokens = 0;
    size_t max_leaf_tokens = 0;
};

static uint32_t read_u32_le(const unsigned char * p) {
    return (uint32_t) p[0] |
           ((uint32_t) p[1] << 8) |
           ((uint32_t) p[2] << 16) |
           ((uint32_t) p[3] << 24);
}

static uint64_t read_u64_le(const unsigned char * p) {
    return (uint64_t) read_u32_le(p) | ((uint64_t) read_u32_le(p + 4) << 32);
}

static std::vector<std::filesystem::path> request_token_trie_shards(
        const std::filesystem::path & dir,
        const std::string & prefix) {
    std::vector<std::filesystem::path> out;
    for (const auto & entry : std::filesystem::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        const std::string name = entry.path().filename().string();
        if (name.size() >= prefix.size() + 4 &&
            name.compare(0, prefix.size(), prefix) == 0 &&
            name.compare(name.size() - 4, 4, ".bin") == 0) {
            out.push_back(entry.path());
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

static std::vector<unsigned char> read_binary_file(const std::filesystem::path & path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("cannot open request-token trie shard: " + path.string());
    const std::streamoff size = in.tellg();
    if (size < 0) throw std::runtime_error("cannot size request-token trie shard: " + path.string());
    std::vector<unsigned char> bytes((size_t) size);
    in.seekg(0);
    if (!bytes.empty()) in.read((char *) bytes.data(), bytes.size());
    if (!in) throw std::runtime_error("cannot read request-token trie shard: " + path.string());
    return bytes;
}

static void validate_request_token_trie_header(
        const std::vector<unsigned char> & bytes,
        const char magic[9],
        uint32_t record_size,
        const std::filesystem::path & path) {
    if (bytes.size() < 24 || std::memcmp(bytes.data(), magic, 8) != 0 ||
        read_u32_le(bytes.data() + 8) != 1 || read_u32_le(bytes.data() + 12) != record_size ||
        (bytes.size() - 24) % record_size != 0) {
        throw std::runtime_error("invalid request-token trie shard: " + path.string());
    }
}

static request_token_trie load_request_token_trie(const std::string & directory, int32_t n_vocab = -1) {
    const std::filesystem::path dir(directory);
    if (!std::filesystem::is_directory(dir)) {
        throw std::runtime_error("request-token trie path is not a directory: " + directory);
    }

    const auto node_shards = request_token_trie_shards(dir, "nodes-");
    const auto request_shards = request_token_trie_shards(dir, "requests-");
    if (node_shards.empty() || request_shards.empty()) {
        throw std::runtime_error("request-token trie is missing node/request shards: " + directory);
    }

    std::vector<request_token_trie_node> nodes(1);
    nodes[0].present = true;
    for (const auto & path : node_shards) {
        const auto bytes = read_binary_file(path);
        validate_request_token_trie_header(bytes, "CLTNOD01", 16, path);
        const uint64_t first = read_u64_le(bytes.data() + 16);
        const uint64_t count = (bytes.size() - 24) / 16;
        if (first == 0 || first > UINT64_MAX - count || first + count > SIZE_MAX) {
            throw std::runtime_error("invalid request-token trie node id range: " + path.string());
        }
        if (nodes.size() < first + count) nodes.resize((size_t) (first + count));
        for (uint64_t i = 0; i < count; ++i) {
            const unsigned char * rec = bytes.data() + 24 + i * 16;
            const uint64_t id = first + i;
            const uint64_t parent = read_u64_le(rec);
            const int32_t token = (int32_t) read_u32_le(rec + 8);
            if (id == 0 || parent >= id || nodes[(size_t) id].present) {
                throw std::runtime_error("invalid/overlapping request-token trie node: " + path.string());
            }
            if (token < 0 || (n_vocab > 0 && token >= n_vocab)) {
                throw std::runtime_error("request-token trie contains token outside model vocabulary: " + std::to_string(token));
            }
            nodes[(size_t) id].parent = parent;
            nodes[(size_t) id].token = token;
            nodes[(size_t) id].present = true;
        }
    }

    std::vector<uint32_t> depth(nodes.size(), 0);
    for (size_t id = 1; id < nodes.size(); ++id) {
        if (!nodes[id].present) continue;
        const uint64_t parent = nodes[id].parent;
        if (parent >= nodes.size() || !nodes[(size_t) parent].present) {
            throw std::runtime_error("request-token trie node references missing parent");
        }
        depth[id] = depth[(size_t) parent] + 1;
    }

    std::unordered_map<uint64_t, uint32_t> terminals;
    uint64_t request_records = 0;
    for (const auto & path : request_shards) {
        const auto bytes = read_binary_file(path);
        validate_request_token_trie_header(bytes, "CLTREQ01", 24, path);
        const uint64_t count = (bytes.size() - 24) / 24;
        request_records += count;
        for (uint64_t i = 0; i < count; ++i) {
            const unsigned char * rec = bytes.data() + 24 + i * 24;
            const uint64_t leaf = read_u64_le(rec);
            const uint32_t n_tokens = read_u32_le(rec + 8);
            const uint32_t kind = read_u32_le(rec + 12);
            if (leaf == 0 || leaf >= nodes.size() || !nodes[(size_t) leaf].present ||
                n_tokens == 0 || kind < 1 || kind > 4) {
                throw std::runtime_error("invalid request-token trie request record: " + path.string());
            }
            auto [it, inserted] = terminals.emplace(leaf, n_tokens);
            if (!inserted && it->second != n_tokens) {
                throw std::runtime_error("request-token trie duplicate terminal has inconsistent token count");
            }
            if (depth[(size_t) leaf] != n_tokens) {
                throw std::runtime_error("request-token trie terminal depth does not match recorded token count");
            }
        }
    }

    // A calibration leaf is a maximal recorded request, not merely a physical
    // trie leaf. Interrupted writes can leave orphan node suffixes that have no
    // request record and must never become calibration samples.
    std::vector<uint32_t> terminal_descendants(nodes.size(), 0);
    for (const auto & [terminal, n_tokens] : terminals) {
        GGML_UNUSED(n_tokens);
        terminal_descendants[(size_t) terminal] = 1;
    }
    for (size_t id = nodes.size(); id-- > 1;) {
        if (!nodes[id].present) continue;
        terminal_descendants[(size_t) nodes[id].parent] += terminal_descendants[id];
    }

    std::vector<uint64_t> leaves;
    leaves.reserve(terminals.size());
    for (const auto & [leaf, n_tokens] : terminals) {
        GGML_UNUSED(n_tokens);
        if (terminal_descendants[(size_t) leaf] == 1) leaves.push_back(leaf);
    }
    std::sort(leaves.begin(), leaves.end());
    if (leaves.empty()) throw std::runtime_error("request-token trie has no terminal leaves");

    size_t min_tokens = SIZE_MAX;
    size_t max_tokens = 0;
    for (uint64_t leaf : leaves) {
        min_tokens = std::min(min_tokens, (size_t) depth[(size_t) leaf]);
        max_tokens = std::max(max_tokens, (size_t) depth[(size_t) leaf]);
        for (uint64_t id = leaf; id != 0 && !nodes[(size_t) id].live; id = nodes[(size_t) id].parent) {
            nodes[(size_t) id].live = true;
        }
    }
    nodes[0].live = true;

    request_token_trie trie;
    trie.nodes = std::move(nodes);
    trie.children.resize(trie.nodes.size());
    trie.leaves = std::move(leaves);
    trie.request_records = request_records;
    trie.min_leaf_tokens = min_tokens;
    trie.max_leaf_tokens = max_tokens;
    for (size_t id = 1; id < trie.nodes.size(); ++id) {
        if (!trie.nodes[id].live) continue;
        ++trie.unique_tokens;
        trie.children[(size_t) trie.nodes[id].parent].push_back(id);
    }

    LOG_INF("request-token trie: %llu request records, %zu unique terminals, %zu terminal leaves\n",
            (unsigned long long) trie.request_records, terminals.size(), trie.leaves.size());
    LOG_INF("request-token trie: leaf lengths min=%zu max=%zu, unique live token nodes=%llu\n",
            trie.min_leaf_tokens, trie.max_leaf_tokens, (unsigned long long) trie.unique_tokens);
    return trie;
}

struct Stats {
    std::vector<float>   values;
    std::vector<int64_t> counts;
};

struct tensor_statistics {
    std::string tensor;
    Stats stats;
    float total_sqract = 0.0f;
    float mean_sqract  = 0.0f;
    float max_sqract   = 0.0f;
    float min_sqract   = 0.0f;
    int elements       = 0;
    float stddev       = 0.0f;
    float active       = 0.0f;
    float entropy      = 0.0f;
    float zd           = 0.0f;
    float cossim       = 0.0f;
};

// Published legacy sampler retained only for byte-reproduction of the v2
// artifact. New calibration uses the geometric coreset below.
static constexpr int ENP_LEGACY_SAMPLE_SIZE = 512;
static constexpr uint64_t ENP_LEGACY_SAMPLE_SEED = 0x2fe24846bc52ca37ULL;
static constexpr uint64_t ENP_CERT_SAMPLE_SEED = 0x6a09e667f3bcc909ULL;
static constexpr int ENP_LENS_DIM = 256;
static constexpr int ENP_LENS_SAMPLE_SIZE = 2048;
static constexpr uint64_t ENP_LENS_SAMPLE_SEED = 0xbb67ae8584caa73bULL;
static constexpr uint64_t ENP_LENS_SKETCH_SEED = 0x3c6ef372fe94f82bULL;
static constexpr uint64_t EXPERT_REPLACEMENT_SAMPLE_SEED = 0xa54ff53a5f1d36f1ULL;

static uint64_t enp_legacy_mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static void enp_lens_sketch(const float * x, int width, float * out) {
    std::fill(out, out + ENP_LENS_DIM, 0.0f);
    for (int i = 0; i < width; ++i) {
        const uint64_t h = enp_legacy_mix64(ENP_LENS_SKETCH_SEED ^ (uint64_t) i);
        const int bucket = (int) (h & (ENP_LENS_DIM - 1));
        const float sign = ((h >> 8) & 1u) ? 1.0f : -1.0f;
        out[bucket] += sign * x[i];
    }
}

struct enp_legacy_sample_state {
    int width = 0;
    uint64_t seen = 0;
    double max_norm2 = 0.0;
    std::vector<uint64_t> keys;
    std::vector<float> samples;
    int max_slot = -1;
    uint64_t max_key = 0;

    int n_sample() const { return (int) keys.size(); }

    void initialize(int width_) {
        width = width_;
        keys.reserve(ENP_LEGACY_SAMPLE_SIZE);
        samples.resize((size_t) ENP_LEGACY_SAMPLE_SIZE * width);
    }

    void recompute_max() {
        max_slot = 0;
        max_key = keys[0];
        for (int i = 1; i < n_sample(); ++i) {
            if (keys[(size_t) i] > max_key) {
                max_key = keys[(size_t) i];
                max_slot = i;
            }
        }
    }

    void add(const float * x) {
        double norm2 = 0.0;
        for (int i = 0; i < width; ++i) norm2 += (double) x[i] * x[i];
        max_norm2 = std::max(max_norm2, norm2);
        const uint64_t key = enp_legacy_mix64(ENP_LEGACY_SAMPLE_SEED ^ seen++);
        if (n_sample() < ENP_LEGACY_SAMPLE_SIZE) {
            const int slot = n_sample();
            keys.push_back(key);
            std::copy_n(x, width, samples.data() + (size_t) slot * width);
            if (max_slot < 0 || key > max_key) {
                max_key = key;
                max_slot = slot;
            }
            return;
        }
        if (key >= max_key) return;
        keys[(size_t) max_slot] = key;
        std::copy_n(x, width, samples.data() + (size_t) max_slot * width);
        recompute_max();
    }

    void sort_by_priority() {
        std::vector<int> order((size_t) n_sample());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            if (keys[(size_t) a] != keys[(size_t) b]) return keys[(size_t) a] < keys[(size_t) b];
            return a < b;
        });
        std::vector<uint64_t> sorted_keys(keys.size());
        std::vector<float> sorted_samples((size_t) n_sample() * width);
        for (int dst = 0; dst < n_sample(); ++dst) {
            const int src = order[(size_t) dst];
            sorted_keys[(size_t) dst] = keys[(size_t) src];
            std::copy_n(samples.data() + (size_t) src * width, width,
                        sorted_samples.data() + (size_t) dst * width);
        }
        keys.swap(sorted_keys);
        samples.swap(sorted_samples);
        recompute_max();
    }
};

struct enp_cert_sample_state {
    int width = 0;
    int sample_capacity = 0;
    int tail_points = 0;
    uint64_t seen = 0;
    std::vector<uint64_t> keys;
    std::vector<uint64_t> ids;
    std::vector<float> samples;
    int max_slot = -1;
    uint64_t max_key = 0;
    std::vector<double> tail_norm2;
    std::vector<uint64_t> tail_ids;
    std::vector<float> tail_samples;
    int tail_worst_slot = -1;

    int n_sample() const { return (int) keys.size(); }
    int n_tail_candidates() const { return (int) tail_norm2.size(); }
    int random_capacity() const { return sample_capacity + tail_points; }
    int tail_capacity() const { return tail_points + 1; }

    void initialize(int width_, int sample_capacity_, int tail_points_) {
        if (width_ <= 0 || sample_capacity_ <= 0 || tail_points_ < 0) {
            throw std::runtime_error("invalid ENP certification sample dimensions");
        }
        width = width_;
        sample_capacity = sample_capacity_;
        tail_points = tail_points_;
        keys.reserve((size_t) random_capacity());
        ids.reserve((size_t) random_capacity());
        samples.resize((size_t) random_capacity() * width);
        tail_norm2.reserve((size_t) tail_capacity());
        tail_ids.reserve((size_t) tail_capacity());
        tail_samples.resize((size_t) tail_capacity() * width);
    }

    void recompute_max() {
        max_slot = 0;
        max_key = keys[0];
        for (int i = 1; i < n_sample(); ++i) {
            if (keys[(size_t) i] > max_key) {
                max_key = keys[(size_t) i];
                max_slot = i;
            }
        }
    }

    bool tail_candidate_worse(int a, int b) const {
        if (tail_norm2[(size_t) a] != tail_norm2[(size_t) b]) {
            return tail_norm2[(size_t) a] < tail_norm2[(size_t) b];
        }
        return tail_ids[(size_t) a] > tail_ids[(size_t) b];
    }

    void recompute_tail_worst() {
        tail_worst_slot = 0;
        for (int i = 1; i < n_tail_candidates(); ++i) {
            if (tail_candidate_worse(i, tail_worst_slot)) tail_worst_slot = i;
        }
    }

    void add(const float * x) {
        const uint64_t id = seen++;
        double norm2 = 0.0;
        for (int i = 0; i < width; ++i) norm2 += (double) x[i] * x[i];

        const uint64_t key = enp_legacy_mix64(ENP_CERT_SAMPLE_SEED ^ id);
        if (n_sample() < random_capacity()) {
            const int slot = n_sample();
            keys.push_back(key);
            ids.push_back(id);
            std::copy_n(x, width, samples.data() + (size_t) slot * width);
            if (max_slot < 0 || key > max_key) {
                max_key = key;
                max_slot = slot;
            }
        } else if (key < max_key) {
            keys[(size_t) max_slot] = key;
            ids[(size_t) max_slot] = id;
            std::copy_n(x, width, samples.data() + (size_t) max_slot * width);
            recompute_max();
        }

        if (n_tail_candidates() < tail_capacity()) {
            const int slot = n_tail_candidates();
            tail_norm2.push_back(norm2);
            tail_ids.push_back(id);
            std::copy_n(x, width, tail_samples.data() + (size_t) slot * width);
            if (tail_worst_slot < 0 || tail_candidate_worse(slot, tail_worst_slot)) {
                tail_worst_slot = slot;
            }
        } else {
            const int w = tail_worst_slot;
            const bool better = norm2 > tail_norm2[(size_t) w] ||
                (norm2 == tail_norm2[(size_t) w] && id < tail_ids[(size_t) w]);
            if (better) {
                tail_norm2[(size_t) w] = norm2;
                tail_ids[(size_t) w] = id;
                std::copy_n(x, width, tail_samples.data() + (size_t) w * width);
                recompute_tail_worst();
            }
        }
    }

    void sort_by_priority() {
        std::vector<int> order((size_t) n_sample());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            if (keys[(size_t) a] != keys[(size_t) b]) return keys[(size_t) a] < keys[(size_t) b];
            return a < b;
        });
        std::vector<uint64_t> sorted_keys(keys.size());
        std::vector<uint64_t> sorted_ids(ids.size());
        std::vector<float> sorted_samples((size_t) n_sample() * width);
        for (int dst = 0; dst < n_sample(); ++dst) {
            const int src = order[(size_t) dst];
            sorted_keys[(size_t) dst] = keys[(size_t) src];
            sorted_ids[(size_t) dst] = ids[(size_t) src];
            std::copy_n(samples.data() + (size_t) src * width, width,
                        sorted_samples.data() + (size_t) dst * width);
        }
        keys.swap(sorted_keys);
        ids.swap(sorted_ids);
        samples.swap(sorted_samples);
        recompute_max();
    }

    void sort_tail_by_norm() {
        std::vector<int> order((size_t) n_tail_candidates());
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
            if (tail_norm2[(size_t) a] != tail_norm2[(size_t) b]) {
                return tail_norm2[(size_t) a] > tail_norm2[(size_t) b];
            }
            return tail_ids[(size_t) a] < tail_ids[(size_t) b];
        });
        std::vector<double> sorted_norm2(tail_norm2.size());
        std::vector<uint64_t> sorted_ids(tail_ids.size());
        std::vector<float> sorted_samples((size_t) n_tail_candidates() * width);
        for (int dst = 0; dst < n_tail_candidates(); ++dst) {
            const int src = order[(size_t) dst];
            sorted_norm2[(size_t) dst] = tail_norm2[(size_t) src];
            sorted_ids[(size_t) dst] = tail_ids[(size_t) src];
            std::copy_n(tail_samples.data() + (size_t) src * width, width,
                        sorted_samples.data() + (size_t) dst * width);
        }
        tail_norm2.swap(sorted_norm2);
        tail_ids.swap(sorted_ids);
        tail_samples.swap(sorted_samples);
        recompute_tail_worst();
    }
};

struct enp_downstream_lens_state {
    uint64_t seen = 0;
    std::vector<uint64_t> keys;
    std::vector<float> final_samples;
    std::map<int, std::vector<float>> layer_samples;
    int max_slot = -1;
    uint64_t max_key = 0;

    int n_sample() const { return (int) keys.size(); }

    bool would_accept(uint64_t id) const {
        if (n_sample() < ENP_LENS_SAMPLE_SIZE) return true;
        return enp_legacy_mix64(ENP_LENS_SAMPLE_SEED ^ id) < max_key;
    }

    void ensure_layers(const std::map<int, std::vector<float>> & pending, int n_active) {
        for (const auto & [layer, values] : pending) {
            if (values.size() != (size_t) n_active * ENP_LENS_DIM) {
                throw std::runtime_error("invalid downstream-lens pending sample dimensions");
            }
            auto & dst = layer_samples[layer];
            if (dst.empty()) dst.resize((size_t) ENP_LENS_SAMPLE_SIZE * ENP_LENS_DIM);
        }
    }

    void recompute_max() {
        max_slot = 0;
        max_key = keys[0];
        for (int i = 1; i < n_sample(); ++i) {
            if (keys[(size_t) i] > max_key) {
                max_key = keys[(size_t) i];
                max_slot = i;
            }
        }
    }

    void add_batch(
            const std::map<int, std::vector<float>> & pending,
            const std::vector<float> & final,
            const std::vector<int> & active_tokens,
            int n_tokens) {
        if (pending.empty() || final.size() != active_tokens.size() * ENP_LENS_DIM) {
            throw std::runtime_error("invalid downstream-lens paired batch");
        }
        ensure_layers(pending, (int) active_tokens.size());
        if (final_samples.empty()) final_samples.resize((size_t) ENP_LENS_SAMPLE_SIZE * ENP_LENS_DIM);
        const uint64_t base_id = seen;
        seen += (uint64_t) n_tokens;
        for (size_t active = 0; active < active_tokens.size(); ++active) {
            const uint64_t id = base_id + (uint64_t) active_tokens[active];
            const uint64_t key = enp_legacy_mix64(ENP_LENS_SAMPLE_SEED ^ id);
            int slot = -1;
            if (n_sample() < ENP_LENS_SAMPLE_SIZE) {
                slot = n_sample();
                keys.push_back(key);
                if (max_slot < 0 || key > max_key) {
                    max_key = key;
                    max_slot = slot;
                }
            } else if (key < max_key) {
                slot = max_slot;
                keys[(size_t) slot] = key;
            }
            if (slot < 0) continue;

            std::copy_n(final.data() + active * ENP_LENS_DIM, ENP_LENS_DIM,
                        final_samples.data() + (size_t) slot * ENP_LENS_DIM);
            for (const auto & [layer, values] : pending) {
                auto & dst = layer_samples.at(layer);
                std::copy_n(values.data() + active * ENP_LENS_DIM, ENP_LENS_DIM,
                            dst.data() + (size_t) slot * ENP_LENS_DIM);
            }
            if (n_sample() == ENP_LENS_SAMPLE_SIZE) recompute_max();
        }
    }

    void sort_by_priority() {
        if (keys.empty()) return;
        std::vector<int> order(keys.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            if (keys[(size_t) a] != keys[(size_t) b]) return keys[(size_t) a] < keys[(size_t) b];
            return a < b;
        });
        std::vector<uint64_t> sorted_keys(keys.size());
        std::vector<float> sorted_final((size_t) n_sample() * ENP_LENS_DIM);
        std::map<int, std::vector<float>> sorted_layers;
        for (const auto & [layer, values] : layer_samples) {
            GGML_UNUSED(values);
            sorted_layers[layer].resize((size_t) n_sample() * ENP_LENS_DIM);
        }
        for (int dst = 0; dst < n_sample(); ++dst) {
            const int src = order[(size_t) dst];
            sorted_keys[(size_t) dst] = keys[(size_t) src];
            std::copy_n(final_samples.data() + (size_t) src * ENP_LENS_DIM, ENP_LENS_DIM,
                        sorted_final.data() + (size_t) dst * ENP_LENS_DIM);
            for (const auto & [layer, values] : layer_samples) {
                std::copy_n(values.data() + (size_t) src * ENP_LENS_DIM, ENP_LENS_DIM,
                            sorted_layers[layer].data() + (size_t) dst * ENP_LENS_DIM);
            }
        }
        keys.swap(sorted_keys);
        final_samples.swap(sorted_final);
        layer_samples.swap(sorted_layers);
        recompute_max();
    }
};

// Independent uniform bottom-k reservoir of exact MoE input hidden states.
// This is intentionally separate from the downstream-lens reservoir: the lens
// wants more points to fit a stable 256D map, while replacement-aware expert
// pruning only needs enough exact token states to reconstruct router Top-K and
// candidate expert outputs offline.  Keeping the reservoirs independent also
// prevents the expert-pruning sample from affecting the fitted lens.
struct expert_replacement_sample_state {
    int width = 0;
    int capacity = 0;
    uint64_t seen = 0;
    std::vector<uint64_t> keys;
    std::vector<float> samples;
    int max_slot = -1;
    uint64_t max_key = 0;

    int n_sample() const { return (int) keys.size(); }

    void initialize(int width_, int capacity_) {
        if (width_ <= 0 || capacity_ <= 0) {
            throw std::runtime_error("invalid expert-replacement reservoir dimensions");
        }
        width = width_;
        capacity = capacity_;
        keys.reserve((size_t) capacity);
        samples.resize((size_t) capacity * width);
    }

    void recompute_max() {
        max_slot = 0;
        max_key = keys[0];
        for (int i = 1; i < n_sample(); ++i) {
            if (keys[(size_t) i] > max_key) {
                max_key = keys[(size_t) i];
                max_slot = i;
            }
        }
    }

    void add(const float * x) {
        const uint64_t key = enp_legacy_mix64(EXPERT_REPLACEMENT_SAMPLE_SEED ^ seen++);
        if (n_sample() < capacity) {
            const int slot = n_sample();
            keys.push_back(key);
            std::copy_n(x, width, samples.data() + (size_t) slot * width);
            if (max_slot < 0 || key > max_key) {
                max_key = key;
                max_slot = slot;
            }
            return;
        }
        if (key >= max_key) return;
        keys[(size_t) max_slot] = key;
        std::copy_n(x, width, samples.data() + (size_t) max_slot * width);
        recompute_max();
    }

    void sort_by_priority() {
        if (keys.empty()) return;
        std::vector<int> order(keys.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            if (keys[(size_t) a] != keys[(size_t) b]) return keys[(size_t) a] < keys[(size_t) b];
            return a < b;
        });
        std::vector<uint64_t> sorted_keys(keys.size());
        std::vector<float> sorted_samples((size_t) n_sample() * width);
        for (int dst = 0; dst < n_sample(); ++dst) {
            const int src = order[(size_t) dst];
            sorted_keys[(size_t) dst] = keys[(size_t) src];
            std::copy_n(samples.data() + (size_t) src * width, width,
                        sorted_samples.data() + (size_t) dst * width);
        }
        keys.swap(sorted_keys);
        samples.swap(sorted_samples);
        recompute_max();
    }
};

// Task-agnostic geometric coreset for standard ENP.  The collector never sees
// task labels, router frequency, or downstream quantization error.  It streams
// every expert-input hidden state through a deterministic medoid coreset and
// records original-space transport/radius upper bounds.  The planner then
// evaluates the unmodified ENP projection function on these representatives.
static int32_t imatrix_accounting_chunk_size(const common_params & params) {
    // Trie calibration is not chunked for model-state purposes. Keep a small
    // accounting unit only for stock GGUF imatrix metadata; prune.token_count
    // remains the exact token total consumed by qwen35-prune.
    return params.imatrix_request_token_trie.empty()
        ? params.n_ctx / params.n_parallel
        : 128;
}

class IMatrixCollector {
public:
    IMatrixCollector() = default;
    void set_params(common_params params) { m_params = std::move(params); }
    bool collect_imatrix(struct ggml_tensor * t, bool ask, void * user_data);
    void collect_token_counts(const std::vector<llama_token> & tokens, int32_t n_vocab, size_t n_tokens);
    void save_imatrix_legacy(int32_t ncall = -1) const;
    void save_imatrix(int32_t n_chunk = -1) const;
    bool load_imatrix(const char * file_name);
    void finalize_enp_samples();
    const std::unordered_map<std::string, Stats> & get_mstats() const { return m_stats; }
private:
    std::unordered_map<std::string, Stats> m_stats;
    // Q/K Gram matrices are the only large reduction where FP32 accumulation
    // error is avoidable at negligible memory cost (~63 MiB for Qwen3.5 35B).
    // Keep the running sums in FP64 and only round once when serializing the
    // stock-compatible F32 imatrix GGUF payload.
    std::unordered_map<std::string, std::vector<double>> m_qk_gram64;
    common_params                          m_params;
    std::mutex                             m_mutex;
    std::vector<std::string>               m_datasets;
    int32_t                                m_last_chunk = 0;
    std::vector<char>                      m_src1_data;
    std::vector<char>                      m_ids; // the expert ids from ggml_mul_mat_id
    std::vector<char>                      m_node_data;
    std::unordered_map<int, qwen35_prune::enp_geometric_stream> m_enp_coresets;
    std::unordered_map<int, enp_legacy_sample_state> m_enp_legacy_samples;
    std::unordered_map<int, enp_cert_sample_state> m_enp_cert_samples;
    enp_downstream_lens_state m_enp_lens;
    std::unordered_map<int, expert_replacement_sample_state> m_expert_replacement_samples;
    std::map<int, std::vector<float>> m_enp_lens_pending;
    std::vector<int> m_enp_lens_active_tokens;
};

// remove any prefix and suffixes from the name
// CUDA0#blk.0.attn_k.weight#0 => blk.0.attn_k.weight
static std::string filter_tensor_name(const char * name) {
    std::string wname;
    const char * p = strchr(name, '#');
    if (p != NULL) {
        p = p + 1;
        const char * q = strchr(p, '#');
        if (q != NULL) {
            wname = std::string(p, q - p);
        } else {
            wname = p;
        }
    } else {
        wname = name;
    }
    return wname;
}

static void process_tensor_name(const std::string & input, std::string & layer, std::string & tensor) {
    std::vector<std::string> name;
    std::istringstream stream(input);
    std::string item;

    while (std::getline(stream, item, '.')) {
        name.push_back(item);
    }
    for (size_t i = 0; i < name.size(); ++i) {
        if (name[i] == "blk" && i + 1 < name.size()) {
            layer = name[i + 1];
            break;
        }
    }
    for (size_t i = 0; i < name.size(); ++i) {
        if (name[i] == "weight" && i > 0) {
            tensor = name[i - 1];
            break;
        }
    }

    if (tensor.empty()) {
        tensor = input;
    }
    if (layer.empty()) {
        layer = "-";
    }
}

static void compute_statistics(std::vector<tensor_statistics> & tstats, const std::string & name, const Stats & e) {
    if (e.values.size() % e.counts.size() != 0) {
        LOG_ERR("%s: activation size mismatch for tensor %s (%zu vs %zu)\n", __func__, name.c_str(), e.counts.size(), e.values.size());
        return;
    }
    if (e.counts.empty()) {
        LOG_ERR("%s: there are no activations for tensor %s. The imatrix may be suboptimal\n", __func__, name.c_str());
        return;
    }

    const int n_mat = e.counts.size();
    const int row_size = e.values.size() / n_mat;

    std::vector<float> activations;
    activations.reserve(e.values.size());

    for (int i = 0; i < n_mat; ++i) {
        if (e.counts[i] == 0) {
            LOG_DBG("%s: skipping tensor %s due to zero count at index %d\n", __func__, name.c_str(), i);
            continue;
        }
        for (int j = 0; j < row_size; ++j) {
            activations.push_back(e.values[i*row_size + j] / e.counts[i]);
        }
    }

    if (activations.empty()) {
        LOG_ERR("%s: all counts are zero for tensor %s, skipping statistics computation\n", __func__, name.c_str());
        return;
    }

    const float act_total     = std::accumulate(activations.begin(), activations.end(), 0.0f);
    const float act_max       = *std::max_element(activations.begin(), activations.end());
    const float act_min       = *std::min_element(activations.begin(), activations.end());
    const float act_mean      = act_total / activations.size();
    const float act_sqr_total = std::inner_product(activations.begin(), activations.end(), activations.begin(), 0.0f);
    const float act_var       = (act_sqr_total / activations.size()) - (act_mean * act_mean);
    const float act_dev       = std::sqrt(std::max(0.0f, act_var));
    float threshold           = 1e-5f;
    const int inactive_count  = std::count_if(activations.begin(), activations.end(),
                                               [threshold](const float v) { return fabsf(v) <= threshold; });
    const float active_ratio  = 1 - static_cast<float>(inactive_count) / activations.size();

    float entropy = 0;
    if (act_total > 0) {
        for (const auto act : activations) {
            if (const float p = act / act_total; p > 0) {
                entropy -= p * std::log2(p);
            }
        }
    }

    int z_score = 0;
    if (act_dev > 0.0f) {
        for (const auto act : activations) {
            if (const float p = (act - act_mean) / act_dev; p > 1) {
                z_score++;
            }
        }
    }

    auto & ts = tstats.emplace_back();
    ts.tensor     = name;
    ts.stats      = e;
    ts.total_sqract = act_total;
    ts.mean_sqract  = act_mean;
    ts.max_sqract   = act_max;
    ts.min_sqract   = act_min;
    ts.elements   = static_cast<int>(activations.size());
    ts.stddev     = act_dev;
    ts.active     = active_ratio;
    ts.entropy    = entropy;
    ts.zd         = static_cast<float>(z_score) / ts.elements;
}

static void compute_cossim(std::vector<tensor_statistics> & tstats) {
    static const std::regex pattern(R"(blk\.(\d+)\.)");
    for (auto & ts : tstats) {
        if (std::smatch match; std::regex_search(ts.tensor, match, pattern)) {
            const int blk = std::stoi(match[1]);
            std::string tname(ts.tensor);
            tname.replace(match.position(1), match.length(1), std::to_string(blk-1));
            auto prev = std::find_if(tstats.begin(), tstats.end(),
                [tname](const tensor_statistics & t) { return t.tensor == tname; });
            if (prev != tstats.end()) {
                const float dp = std::inner_product(ts.stats.values.begin(), ts.stats.values.end(),
                    prev->stats.values.begin(), 0.0f);
                const float curr_mag = std::sqrt(std::inner_product(ts.stats.values.begin(), ts.stats.values.end(),
                    ts.stats.values.begin(), 0.0f));
                const float prev_mag = std::sqrt(std::inner_product(prev->stats.values.begin(), prev->stats.values.end(),
                    prev->stats.values.begin(), 0.0f));
                const float cs = dp / (curr_mag * prev_mag);
                ts.cossim = cs;
            }
        } else {
            ts.cossim = 0;
        }
    }
}

template <typename T>
static bool all_finite(const T * v, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(v[i])) {
            return false;
        }
    }
    return true;
}

static bool parse_gdn_gram_node(const char * name, int & layer, bool & is_q, bool & pre_norm) {
    if (!name) return false;
    const char * prefix = nullptr;
    if (std::strncmp(name, "q_conv_predelta-", 16) == 0) {
        prefix = name + 16;
        is_q = true;
        pre_norm = false;
    } else if (std::strncmp(name, "k_conv_predelta-", 16) == 0) {
        prefix = name + 16;
        is_q = false;
        pre_norm = false;
    } else if (std::strncmp(name, "q_conv-", 7) == 0) {
        prefix = name + 7;
        is_q = true;
        pre_norm = true;
    } else if (std::strncmp(name, "k_conv-", 7) == 0) {
        prefix = name + 7;
        is_q = false;
        pre_norm = true;
    } else {
        return false;
    }
    char * end = nullptr;
    const long parsed = std::strtol(prefix, &end, 10);
    if (end == prefix || *end != '\0' || parsed < 0 || parsed > INT_MAX) return false;
    layer = (int) parsed;
    return true;
}

static bool parse_gdn_v_energy_node(const char * name, int & layer) {
    if (!name || std::strncmp(name, "v_conv_predelta-", 16) != 0) return false;
    const char * prefix = name + 16;
    char * end = nullptr;
    const long parsed = std::strtol(prefix, &end, 10);
    if (end == prefix || *end != '\0' || parsed < 0 || parsed > INT_MAX) return false;
    layer = (int) parsed;
    return true;
}

static bool parse_gdn_final_output_node(const char * name, int & layer) {
    if (!name || std::strncmp(name, "final_output-", 13) != 0) return false;
    const char * prefix = name + 13;
    char * end = nullptr;
    const long parsed = std::strtol(prefix, &end, 10);
    if (end == prefix || *end != '\0' || parsed < 0 || parsed > INT_MAX) return false;
    layer = (int) parsed;
    return true;
}

static bool parse_indexed_node(const char * name, const char * prefix, int & layer) {
    if (!name || !prefix) return false;
    const size_t n = std::strlen(prefix);
    if (std::strncmp(name, prefix, n) != 0) return false;
    const char * p = name + n;
    char * end = nullptr;
    const long parsed = std::strtol(p, &end, 10);
    if (end == p || *end != '\0' || parsed < 0 || parsed > INT_MAX) return false;
    layer = (int) parsed;
    return true;
}

static bool is_gdn_gram_entry(const std::string & name) {
    return name.rfind("prune.gdn.blk.", 0) == 0 && string_ends_with(name, "_gram");
}

void IMatrixCollector::collect_token_counts(
        const std::vector<llama_token> & tokens, int32_t n_vocab, size_t n_tokens) {
    if (m_params.prune_enp_cert_only) return;
    if (!m_params.prune_vocab_counts && !m_params.prune_enp_only) return;
    std::lock_guard<std::mutex> lock(m_mutex);
    n_tokens = std::min(n_tokens, tokens.size());
    if (m_params.prune_vocab_counts) {
        auto & e = m_stats["prune.vocab"];
        if (e.values.empty()) {
            e.values.resize((size_t) n_vocab, 0.0f);
            e.counts.resize(1, 0);
        }
        for (size_t i = 0; i < n_tokens; ++i) {
            const llama_token id = tokens[i];
            if (id >= 0 && id < n_vocab) e.values[(size_t) id] += 1.0f;
        }
        e.counts[0] += 1;
    }

    // Keep an exact, mergeable token total for prune calibration. Stock
    // imatrix.chunk_count * chunk_size cannot represent a final partial chunk,
    // while a regular imatrix entry naturally preserves additive shard merge
    // semantics through --in-file.
    auto & total = m_stats["prune.token_count"];
    if (total.values.empty()) {
        total.values.resize(1, 0.0f);
        total.counts.resize(1, 0);
    }
    total.values[0] += (float) n_tokens;
    total.counts[0] += 1;
}

void IMatrixCollector::finalize_enp_samples() {
    if (!m_params.prune_enp && !m_params.prune_enp_downstream_lens && !m_params.prune_expert_replacement) return;
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_params.prune_enp_downstream_lens) {
        if (!m_enp_lens_pending.empty()) {
            throw std::runtime_error("downstream ENP lens finalized with an unmatched decoder batch");
        }
        m_enp_lens.sort_by_priority();
        const int ns = m_enp_lens.n_sample();
        if (ns <= 0 || m_enp_lens.seen == 0 || m_enp_lens.layer_samples.empty()) {
            throw std::runtime_error("downstream ENP lens collected no paired hidden states");
        }
        auto & final = m_stats["prune.enp.lens.final"];
        final.values.assign(m_enp_lens.final_samples.begin(),
                            m_enp_lens.final_samples.begin() + (size_t) ns * ENP_LENS_DIM);
        final.counts.assign((size_t) ns, 1);
        for (const auto & [layer, values] : m_enp_lens.layer_samples) {
            auto & e = m_stats["prune.enp.lens.post_moe.blk." + std::to_string(layer)];
            e.values.assign(values.begin(), values.begin() + (size_t) ns * ENP_LENS_DIM);
            e.counts.assign((size_t) ns, 1);
        }
        auto & population = m_stats["prune.enp.lens.population"];
        population.values = {(float) m_enp_lens.seen};
        population.counts = {1};
        auto & dim = m_stats["prune.enp.lens.dim"];
        dim.values = {(float) ENP_LENS_DIM};
        dim.counts = {1};
        auto & sample_seed = m_stats["prune.enp.lens.sample_seed"];
        sample_seed.values = {
            (float) ((ENP_LENS_SAMPLE_SEED >>  0) & 0xffffu),
            (float) ((ENP_LENS_SAMPLE_SEED >> 16) & 0xffffu),
            (float) ((ENP_LENS_SAMPLE_SEED >> 32) & 0xffffu),
            (float) ((ENP_LENS_SAMPLE_SEED >> 48) & 0xffffu),
        };
        sample_seed.counts = {1};
        auto & sketch_seed = m_stats["prune.enp.lens.sketch_seed"];
        sketch_seed.values = {
            (float) ((ENP_LENS_SKETCH_SEED >>  0) & 0xffffu),
            (float) ((ENP_LENS_SKETCH_SEED >> 16) & 0xffffu),
            (float) ((ENP_LENS_SKETCH_SEED >> 32) & 0xffffu),
            (float) ((ENP_LENS_SKETCH_SEED >> 48) & 0xffffu),
        };
        sketch_seed.counts = {1};
        if (m_params.prune_enp_only) {
            const int32_t chunk_size = imatrix_accounting_chunk_size(m_params);
            const uint64_t chunks = (m_enp_lens.seen + (uint64_t) chunk_size - 1) / (uint64_t) chunk_size;
            m_last_chunk = std::max<int32_t>(m_last_chunk, (int32_t) chunks);
        }
        LOG_INF("ENP downstream lens: samples=%d population=%llu layers=%zu dim=%d "
                "method=post-moe-to-final-feature-hash-paired-reservoir-v1\n",
                ns, (unsigned long long) m_enp_lens.seen, m_enp_lens.layer_samples.size(), ENP_LENS_DIM);
        if (!m_params.prune_enp && !m_params.prune_expert_replacement) return;
    }
    if (m_params.prune_expert_replacement) {
        uint64_t max_population = 0;
        int layers = 0;
        for (auto & [layer, sample] : m_expert_replacement_samples) {
            sample.sort_by_priority();
            const int ns = sample.n_sample();
            if (ns <= 0 || sample.width != 2048 || sample.seen == 0) continue;
            auto & e = m_stats["prune.expert_replace.input.blk." + std::to_string(layer)];
            e.values.assign(sample.samples.begin(), sample.samples.begin() + (size_t) ns * sample.width);
            e.counts.assign((size_t) ns, 1);
            auto & population = m_stats["prune.expert_replace.population.blk." + std::to_string(layer)];
            population.values = {(float) sample.seen};
            population.counts = {1};
            max_population = std::max<uint64_t>(max_population, sample.seen);
            ++layers;
        }
        auto & dim = m_stats["prune.expert_replace.hidden_dim"];
        dim.values = {2048.0f};
        dim.counts = {1};
        auto & capacity = m_stats["prune.expert_replace.sample_capacity"];
        capacity.values = {(float) m_params.prune_expert_replacement_points};
        capacity.counts = {1};
        auto & seed = m_stats["prune.expert_replace.sample_seed"];
        seed.values = {
            (float) ((EXPERT_REPLACEMENT_SAMPLE_SEED >>  0) & 0xffffu),
            (float) ((EXPERT_REPLACEMENT_SAMPLE_SEED >> 16) & 0xffffu),
            (float) ((EXPERT_REPLACEMENT_SAMPLE_SEED >> 32) & 0xffffu),
            (float) ((EXPERT_REPLACEMENT_SAMPLE_SEED >> 48) & 0xffffu),
        };
        seed.counts = {1};
        if (m_params.prune_enp_only && max_population > 0) {
            const int32_t chunk_size = imatrix_accounting_chunk_size(m_params);
            const uint64_t chunks = (max_population + (uint64_t) chunk_size - 1) / (uint64_t) chunk_size;
            m_last_chunk = std::max<int32_t>(m_last_chunk, (int32_t) chunks);
        }
        LOG_INF("expert replacement reservoir: samples<=%d population=%llu layers=%d hidden=%d "
                "method=independent-uniform-bottom-k-exact-moe-input-v1\n",
                m_params.prune_expert_replacement_points,
                (unsigned long long) max_population, layers, 2048);
        if (!m_params.prune_enp) return;
    }
    if (m_params.prune_enp_cert_only) {
        uint64_t population = 0;
        for (auto & [layer, sample] : m_enp_cert_samples) {
            sample.sort_by_priority();
            sample.sort_tail_by_norm();
            if (sample.n_sample() <= 0 || sample.width <= 0 || sample.seen == 0) continue;

            const int tail_n = std::min<int>((int) sample.seen, sample.tail_points);
            const int64_t remainder_population = (int64_t) sample.seen - tail_n;
            std::unordered_set<uint64_t> tail_ids;
            tail_ids.reserve((size_t) tail_n * 2 + 1);
            for (int r = 0; r < tail_n; ++r) tail_ids.insert(sample.tail_ids[(size_t) r]);

            std::vector<int> cert_slots;
            cert_slots.reserve((size_t) std::min<int64_t>(sample.sample_capacity, remainder_population));
            for (int r = 0; r < sample.n_sample() && (int64_t) cert_slots.size() < remainder_population &&
                    (int) cert_slots.size() < sample.sample_capacity; ++r) {
                if (tail_ids.find(sample.ids[(size_t) r]) == tail_ids.end()) cert_slots.push_back(r);
            }
            const int cert_n = (int) cert_slots.size();
            const int expected_cert_n = (int) std::min<int64_t>(sample.sample_capacity, remainder_population);
            if (cert_n != expected_cert_n) {
                throw std::runtime_error("ENP tail-split certification reservoir did not retain enough remainder points");
            }

            auto & e = m_stats["prune.enp.cert_sample.blk." + std::to_string(layer)];
            e.values.resize((size_t) cert_n * sample.width);
            e.counts.assign((size_t) cert_n, 1);
            for (int dst = 0; dst < cert_n; ++dst) {
                const int src = cert_slots[(size_t) dst];
                std::copy_n(sample.samples.data() + (size_t) src * sample.width, sample.width,
                            e.values.data() + (size_t) dst * sample.width);
            }

            auto & tail = m_stats["prune.enp.cert_tail.blk." + std::to_string(layer)];
            tail.values.assign(sample.tail_samples.begin(), sample.tail_samples.begin() + (size_t) tail_n * sample.width);
            tail.counts.assign((size_t) tail_n, 1);

            auto & tail_norm = m_stats["prune.enp.cert_tail_norm2.blk." + std::to_string(layer)];
            tail_norm.values.resize((size_t) tail_n);
            tail_norm.counts.assign((size_t) tail_n, 1);
            for (int r = 0; r < tail_n; ++r) tail_norm.values[(size_t) r] = (float) sample.tail_norm2[(size_t) r];

            auto & rem_population = m_stats["prune.enp.cert_remainder_population.blk." + std::to_string(layer)];
            rem_population.values = {(float) remainder_population};
            rem_population.counts = {1};

            auto & rem_max_norm = m_stats["prune.enp.cert_remainder_max_norm2.blk." + std::to_string(layer)];
            if (remainder_population > 0) {
                if (sample.n_tail_candidates() <= tail_n) {
                    throw std::runtime_error("ENP tail-split certification is missing the remainder norm ceiling");
                }
                float bound = (float) sample.tail_norm2[(size_t) tail_n];
                if ((double) bound < sample.tail_norm2[(size_t) tail_n]) {
                    bound = std::nextafterf(bound, std::numeric_limits<float>::infinity());
                }
                rem_max_norm.values = {bound};
            } else {
                rem_max_norm.values = {0.0f};
            }
            rem_max_norm.counts = {1};

            // Standalone certification/held-out files also need a population
            // identity for qwen35-prune to interpret the reservoir without a
            // geometric --in-file.  Preserve an existing geometric population
            // when appending certification to a production imatrix.
            const std::string population_name = "prune.enp.population.blk." + std::to_string(layer);
            if (m_stats.find(population_name) == m_stats.end()) {
                auto & full_population = m_stats[population_name];
                full_population.values = {(float) sample.seen};
                full_population.counts = {1};
            }
            const std::string max_norm_name = "prune.enp.max_norm2.blk." + std::to_string(layer);
            if (m_stats.find(max_norm_name) == m_stats.end()) {
                if (sample.n_tail_candidates() <= 0) {
                    throw std::runtime_error("ENP certification-only traversal is missing full max norm");
                }
                auto & full_max_norm = m_stats[max_norm_name];
                float bound = (float) sample.tail_norm2.front();
                if ((double) bound < sample.tail_norm2.front()) {
                    bound = std::nextafterf(bound, std::numeric_limits<float>::infinity());
                }
                full_max_norm.values = {bound};
                full_max_norm.counts = {1};
            }

            population = std::max<uint64_t>(population, sample.seen);
            LOG_INF("ENP certification sample: blk.%d tail=%d samples=%d remainder=%lld population=%llu "
                    "method=exact-high-norm-tail-plus-independent-bottom-k-uniform-without-replacement\n",
                    layer, tail_n, cert_n, (long long) remainder_population, (unsigned long long) sample.seen);
        }
        auto & seed = m_stats["prune.enp.cert_sample_seed"];
        seed.values = {
            (float) ((ENP_CERT_SAMPLE_SEED >>  0) & 0xffffu),
            (float) ((ENP_CERT_SAMPLE_SEED >> 16) & 0xffffu),
            (float) ((ENP_CERT_SAMPLE_SEED >> 32) & 0xffffu),
            (float) ((ENP_CERT_SAMPLE_SEED >> 48) & 0xffffu),
        };
        seed.counts = {1};
        auto & requested = m_stats["prune.enp.cert_sample_requested"];
        requested.values = {(float) m_params.prune_enp_cert_points};
        requested.counts = {1};
        auto & tail_requested = m_stats["prune.enp.cert_tail_requested"];
        tail_requested.values = {(float) m_params.prune_enp_cert_tail_points};
        tail_requested.counts = {1};
        if (population == 0) throw std::runtime_error("ENP certification-only traversal collected no hidden states");
        return;
    }
    if (m_params.prune_enp_legacy_uniform) {
        for (auto & [layer, sample] : m_enp_legacy_samples) {
            sample.sort_by_priority();
            const int ns = sample.n_sample();
            if (ns <= 0 || sample.width <= 0 || sample.seen == 0) continue;
            auto & e = m_stats["prune.enp.sample.blk." + std::to_string(layer)];
            e.values.assign(sample.samples.begin(), sample.samples.begin() + (size_t) ns * sample.width);
            e.counts.assign((size_t) ns, 1);

            auto & population = m_stats["prune.enp.population.blk." + std::to_string(layer)];
            population.values = {(float) sample.seen};
            population.counts = {1};

            auto & max_norm = m_stats["prune.enp.max_norm2.blk." + std::to_string(layer)];
            max_norm.values = {(float) sample.max_norm2};
            max_norm.counts = {1};

            LOG_INF("ENP quadrature sample: blk.%d samples=%d population=%llu max_norm2=%.6g "
                    "method=bottom-k-uniform-without-replacement\n",
                    layer, ns, (unsigned long long) sample.seen, sample.max_norm2);
        }
        auto & seed = m_stats["prune.enp.sample_seed"];
        seed.values = {
            (float) ((ENP_LEGACY_SAMPLE_SEED >>  0) & 0xffffu),
            (float) ((ENP_LEGACY_SAMPLE_SEED >> 16) & 0xffffu),
            (float) ((ENP_LEGACY_SAMPLE_SEED >> 32) & 0xffffu),
            (float) ((ENP_LEGACY_SAMPLE_SEED >> 48) & 0xffffu),
        };
        seed.counts = {1};
        return;
    }
    auto upper_float = [](double x) {
        float f = (float) x;
        if ((double) f < x) f = std::nextafterf(f, std::numeric_limits<float>::infinity());
        return f;
    };
    uint64_t max_population = 0;
    for (auto & [layer, stream] : m_enp_coresets) {
        auto clusters = stream.finalize();
        const int ns = (int) clusters.size();
        if (ns <= 0 || stream.width() <= 0 || stream.population() == 0) continue;
        auto & centers = m_stats["prune.enp.coreset.blk." + std::to_string(layer)];
        centers.values.resize((size_t) ns * stream.width());
        centers.counts.assign((size_t) ns, 1);
        auto & weights = m_stats["prune.enp.coreset_weight.blk." + std::to_string(layer)];
        weights.values.resize((size_t) ns);
        weights.counts.assign((size_t) ns, 1);
        auto & transport = m_stats["prune.enp.coreset_transport.blk." + std::to_string(layer)];
        transport.values.resize((size_t) ns);
        transport.counts.assign((size_t) ns, 1);
        auto & radius = m_stats["prune.enp.coreset_radius.blk." + std::to_string(layer)];
        radius.values.resize((size_t) ns);
        radius.counts.assign((size_t) ns, 1);
        for (int r = 0; r < ns; ++r) {
            const auto & c = clusters[(size_t) r];
            std::copy(c.center.begin(), c.center.end(),
                      centers.values.begin() + (size_t) r * stream.width());
            weights.values[(size_t) r] = (float) c.weight; // exact for current trie populations (< 2^24)
            transport.values[(size_t) r] = upper_float(c.transport);
            radius.values[(size_t) r] = upper_float(c.radius);
        }

        auto & population = m_stats["prune.enp.population.blk." + std::to_string(layer)];
        population.values = {(float) stream.population()};
        population.counts = {1};

        auto & max_norm = m_stats["prune.enp.max_norm2.blk." + std::to_string(layer)];
        max_norm.values = {upper_float(stream.max_norm2())};
        max_norm.counts = {1};

        LOG_INF("ENP geometric coreset: blk.%d representatives=%d population=%llu transport=%.6g radius=%.6g "
                "method=task-blind-streaming-medoid-original-l2-v1\n",
                layer, ns, (unsigned long long) stream.population(),
                qwen35_prune::enp_geometric_total_transport(clusters),
                qwen35_prune::enp_geometric_max_radius(clusters));
        max_population = std::max<uint64_t>(max_population, stream.population());
    }
    if (m_params.prune_enp_only && max_population > 0) {
        const int32_t chunk_size = imatrix_accounting_chunk_size(m_params);
        const uint64_t chunks = (max_population + (uint64_t) chunk_size - 1) / (uint64_t) chunk_size;
        m_last_chunk = std::max<int32_t>(m_last_chunk, (int32_t) chunks);
    }
}

bool IMatrixCollector::collect_imatrix(struct ggml_tensor * t, bool ask, void * user_data) {
    GGML_UNUSED(user_data);

    int lens_layer = -1;
    const bool is_lens_hidden = m_params.prune_enp_downstream_lens &&
            parse_indexed_node(t->name, "l_out-", lens_layer);
    const bool is_lens_final = m_params.prune_enp_downstream_lens &&
            std::strcmp(t->name, "result_norm") == 0;
    if (is_lens_hidden || is_lens_final) {
        if (ask) return true;
        if (t->type != GGML_TYPE_F32 || t->ne[0] != 2048 || t->ne[2] != 1 || t->ne[3] != 1 ||
                !ggml_is_contiguous(t)) {
            LOG_ERR("%s: unexpected downstream-lens tensor %s type/shape/stride\n", __func__, t->name);
            return false;
        }
        std::lock_guard<std::mutex> lock(m_mutex);
        const bool is_host = ggml_backend_buffer_is_host(t->buffer);
        if (!is_host) {
            m_node_data.resize(ggml_nbytes(t));
            ggml_backend_tensor_get(t, m_node_data.data(), 0, m_node_data.size());
        }
        const char * data = is_host ? (const char *) t->data : m_node_data.data();
        const int n_tokens = (int) t->ne[1];
        if (is_lens_hidden) {
            if (m_enp_lens_pending.empty()) {
                m_enp_lens_active_tokens.clear();
                m_enp_lens_active_tokens.reserve((size_t) n_tokens);
                for (int token = 0; token < n_tokens; ++token) {
                    if (m_enp_lens.would_accept(m_enp_lens.seen + (uint64_t) token)) {
                        m_enp_lens_active_tokens.push_back(token);
                    }
                }
            }
            auto & pending = m_enp_lens_pending[lens_layer];
            pending.resize(m_enp_lens_active_tokens.size() * ENP_LENS_DIM);
            for (size_t active = 0; active < m_enp_lens_active_tokens.size(); ++active) {
                const int token = m_enp_lens_active_tokens[active];
                const float * x = (const float *) (data + (size_t) token * t->nb[1]);
                if (!all_finite(x, (size_t) t->ne[0])) {
                    LOG_ERR("%s: non-finite downstream-lens post-MoE residual at layer %d token %d\n",
                            __func__, lens_layer, token);
                    return false;
                }
                enp_lens_sketch(x, (int) t->ne[0], pending.data() + active * ENP_LENS_DIM);
            }
        } else if (!m_enp_lens_pending.empty()) {
            std::vector<float> final(m_enp_lens_active_tokens.size() * ENP_LENS_DIM);
            for (size_t active = 0; active < m_enp_lens_active_tokens.size(); ++active) {
                const int token = m_enp_lens_active_tokens[active];
                const float * x = (const float *) (data + (size_t) token * t->nb[1]);
                if (!all_finite(x, (size_t) t->ne[0])) {
                    LOG_ERR("%s: non-finite downstream-lens final hidden at token %d\n", __func__, token);
                    return false;
                }
                enp_lens_sketch(x, (int) t->ne[0], final.data() + active * ENP_LENS_DIM);
            }
            for (const auto & [layer, pending] : m_enp_lens_pending) {
                if (pending.size() != final.size()) {
                    LOG_ERR("%s: downstream-lens token mismatch at layer %d\n", __func__, layer);
                    return false;
                }
            }
            m_enp_lens.add_batch(m_enp_lens_pending, final, m_enp_lens_active_tokens, n_tokens);
            m_enp_lens_pending.clear();
            m_enp_lens_active_tokens.clear();
        }
        return true;
    }

    int replacement_layer = -1;
    const bool is_replacement_hidden = m_params.prune_expert_replacement &&
            parse_indexed_node(t->name, "attn_post_norm-", replacement_layer);
    if (is_replacement_hidden) {
        if (ask) return true;
        if (t->type != GGML_TYPE_F32 || t->ne[0] != 2048 || t->ne[2] != 1 || t->ne[3] != 1 ||
                !ggml_is_contiguous(t)) {
            LOG_ERR("%s: unexpected expert-replacement input tensor %s type/shape/stride\n", __func__, t->name);
            return false;
        }
        std::lock_guard<std::mutex> lock(m_mutex);
        const bool is_host = ggml_backend_buffer_is_host(t->buffer);
        if (!is_host) {
            m_node_data.resize(ggml_nbytes(t));
            ggml_backend_tensor_get(t, m_node_data.data(), 0, m_node_data.size());
        }
        const char * data = is_host ? (const char *) t->data : m_node_data.data();
        auto & sample = m_expert_replacement_samples[replacement_layer];
        if (sample.width == 0) {
            sample.initialize((int) t->ne[0], m_params.prune_expert_replacement_points);
        }
        if (sample.width != t->ne[0]) {
            LOG_ERR("%s: expert-replacement hidden width changed at layer %d\n", __func__, replacement_layer);
            return false;
        }
        for (int64_t token = 0; token < t->ne[1]; ++token) {
            const float * x = (const float *) (data + token * t->nb[1]);
            if (!all_finite(x, (size_t) t->ne[0])) {
                LOG_ERR("%s: non-finite expert-replacement hidden at layer %d token %lld\n",
                        __func__, replacement_layer, (long long) token);
                return false;
            }
            sample.add(x);
        }
        // The same node may also be requested by ENP.  Do not return early in
        // that combined mode; let the ENP collector below consume it too.
        if (!m_params.prune_enp) return true;
    }

    int enp_layer = -1;
    // Qwen3.5 names the exact post-attention-normalized tensor passed into
    // build_layer_ffn() as attn_post_norm.  Observing that existing callback
    // node gives standard ENP the exact expert input without adding a fork-only
    // tap to llama core / public context parameters.
    const bool is_enp_hidden = m_params.prune_enp &&
            parse_indexed_node(t->name, "attn_post_norm-", enp_layer);
    if (is_enp_hidden) {
        if (ask) return true;
        if (t->type != GGML_TYPE_F32 || t->ne[0] != 2048 || t->ne[2] != 1 || t->ne[3] != 1 ||
                !ggml_is_contiguous(t)) {
            LOG_ERR("%s: unexpected ENP hidden tensor %s type/shape/stride\n", __func__, t->name);
            return false;
        }
        std::lock_guard<std::mutex> lock(m_mutex);
        const bool is_host = ggml_backend_buffer_is_host(t->buffer);
        if (!is_host) {
            m_node_data.resize(ggml_nbytes(t));
            ggml_backend_tensor_get(t, m_node_data.data(), 0, m_node_data.size());
        }
        const char * data = is_host ? (const char *) t->data : m_node_data.data();
        for (int64_t token = 0; token < t->ne[1]; ++token) {
            const float * x = (const float *) (data + token * t->nb[1]);
            if (!all_finite(x, (size_t) t->ne[0])) {
                LOG_ERR("%s: non-finite ENP hidden at layer %d token %lld\n",
                        __func__, enp_layer, (long long) token);
                return false;
            }
            if (m_params.prune_enp_legacy_uniform) {
                auto & sample = m_enp_legacy_samples[enp_layer];
                if (sample.width == 0) sample.initialize((int) t->ne[0]);
                if (sample.width != t->ne[0]) {
                    LOG_ERR("%s: ENP hidden width changed at layer %d\n", __func__, enp_layer);
                    return false;
                }
                sample.add(x);
            } else if (m_params.prune_enp_cert_only) {
                auto & sample = m_enp_cert_samples[enp_layer];
                if (sample.width == 0) {
                    sample.initialize((int) t->ne[0], m_params.prune_enp_cert_points, m_params.prune_enp_cert_tail_points);
                }
                if (sample.width != t->ne[0]) {
                    LOG_ERR("%s: ENP certification hidden width changed at layer %d\n", __func__, enp_layer);
                    return false;
                }
                sample.add(x);
            } else {
                auto & stream = m_enp_coresets[enp_layer];
                if (stream.width() == 0) stream.initialize((int) t->ne[0], m_params.prune_enp_coreset_points);
                if (stream.width() != t->ne[0]) {
                    LOG_ERR("%s: ENP hidden width changed at layer %d\n", __func__, enp_layer);
                    return false;
                }
                stream.add(x);
            }
        }
        return true;
    }

    // Expert-only calibration deliberately bypasses the stock imatrix
    // MUL_MAT/MUL_MAT_ID reductions and all fork-side GDN taps. Standard ENP
    // needs only the exact expert-input hidden-state distribution above.
    if (m_params.prune_enp_only) return false;

    int gram_layer = -1;
    bool gram_is_q = false;
    bool gram_pre_norm = false;
    const bool is_gram = m_params.prune_qk_gram &&
            parse_gdn_gram_node(t->name, gram_layer, gram_is_q, gram_pre_norm);
    if (is_gram) {
        if (ask) return true;
        if (t->type != GGML_TYPE_F32 || t->ne[0] != 128 || t->ne[1] < 16) {
            LOG_ERR("%s: unexpected GDN Gram tensor %s type/shape: type=%s ne=[%lld,%lld,%lld,%lld]\n",
                    __func__, t->name, ggml_type_name(t->type),
                    (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);
            return false;
        }

        std::lock_guard<std::mutex> lock(m_mutex);
        const bool is_host = ggml_backend_buffer_is_host(t->buffer);
        if (!is_host) {
            const size_t nbytes = ggml_nbytes(t);
            m_node_data.resize(nbytes);
            ggml_backend_tensor_get(t, m_node_data.data(), 0, nbytes);
        }
        const char * data = is_host ? (const char *) t->data : m_node_data.data();

        const std::string prefix = "prune.gdn.blk." + std::to_string(gram_layer);
        const std::string separate_gram_name = prefix + (gram_is_q
                ? (gram_pre_norm ? ".q_pre_norm_gram" : ".q_gram")
                : (gram_pre_norm ? ".k_pre_norm_gram" : ".k_gram"));
        auto & separate_gram_entry = m_stats[separate_gram_name];
        constexpr int n_head_qk = 16;
        constexpr int head_dim = 128;
        if (separate_gram_entry.values.empty()) {
            separate_gram_entry.values.resize((size_t) n_head_qk * head_dim * head_dim, 0.0f);
            separate_gram_entry.counts.resize(1, 0);
        }
        auto & separate_gram64 = m_qk_gram64[separate_gram_name];
        if (separate_gram64.empty()) {
            separate_gram64.resize((size_t) n_head_qk * head_dim * head_dim, 0.0);
        } else if (separate_gram64.size() != separate_gram_entry.values.size()) {
            LOG_ERR("%s: inconsistent FP64 separate GDN Gram size for layer %d\n", __func__, gram_layer);
            return false;
        }

        Stats * combined_entry = nullptr;
        Stats * energy_entry = nullptr;
        std::vector<double> * combined_gram64 = nullptr;
        if (!gram_pre_norm) {
            const std::string entry_name = prefix + ".qk_gram";
            auto & e = m_stats[entry_name];
            if (e.values.empty()) {
                e.values.resize((size_t) n_head_qk * head_dim * head_dim, 0.0f);
                e.counts.resize(1, 0);
            }
            auto & gram64 = m_qk_gram64[entry_name];
            if (gram64.empty()) {
                gram64.resize((size_t) n_head_qk * head_dim * head_dim, 0.0);
            } else if (gram64.size() != e.values.size()) {
                LOG_ERR("%s: inconsistent FP64 GDN Gram size for layer %d\n", __func__, gram_layer);
                return false;
            }
            const std::string energy_name = prefix + (gram_is_q ? ".q_energy" : ".k_energy");
            auto & energy = m_stats[energy_name];
            if (energy.values.empty()) {
                energy.values.resize((size_t) n_head_qk * head_dim, 0.0f);
                energy.counts.resize(1, 0);
            }
            combined_entry = &e;
            energy_entry = &energy;
            combined_gram64 = &gram64;
        }

        for (int64_t i3 = 0; i3 < t->ne[3]; ++i3) {
            for (int64_t i2 = 0; i2 < t->ne[2]; ++i2) {
                for (int head = 0; head < n_head_qk; ++head) {
                    const float * x = (const float *) (data + head * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3]);
                    double * sg = separate_gram64.data() + (size_t) head * head_dim * head_dim;
                    double * g = combined_gram64
                            ? combined_gram64->data() + (size_t) head * head_dim * head_dim
                            : nullptr;
                    float * he = energy_entry
                            ? energy_entry->values.data() + (size_t) head * head_dim
                            : nullptr;
                    for (int r = 0; r < head_dim; ++r) {
                        const double xr = x[r];
                        if (he) he[r] += (float) (xr * xr);
                        double * grow = g ? g + (size_t) r * head_dim : nullptr;
                        double * sgrow = sg + (size_t) r * head_dim;
                        for (int c = 0; c < head_dim; ++c) {
                            const double v = xr * (double) x[c];
                            if (grow) grow[c] += v;
                            sgrow[c] += v;
                        }
                    }
                }
                // Count callback token-positions. With the current predelta
                // scheduling this accumulates to one Q/K vector per QK head,
                // i.e. 2 * calibration_tokens * n_head_qk after both Q and K
                // callbacks. The Gram payload itself is a raw additive sum;
                // this scalar is diagnostic and is also additive across shards.
                if (combined_entry) combined_entry->counts[0] += n_head_qk;
                separate_gram_entry.counts[0] += n_head_qk;
                if (energy_entry) energy_entry->counts[0] += n_head_qk;
            }
        }
        if ((combined_gram64 && !all_finite(combined_gram64->data(), combined_gram64->size())) ||
            !all_finite(separate_gram64.data(), separate_gram64.size())) {
            LOG_ERR("%s: non-finite GDN Gram values in layer %d\n", __func__, gram_layer);
            std::exit(1);
        }
        return true;
    }

    int v_energy_layer = -1;
    const bool is_v_energy = m_params.prune_qk_gram && parse_gdn_v_energy_node(t->name, v_energy_layer);
    if (is_v_energy) {
        if (ask) return true;
        constexpr int n_head_v = 32;
        constexpr int head_dim = 128;
        if (t->type != GGML_TYPE_F32 || t->ne[0] != head_dim || t->ne[1] != n_head_v) {
            LOG_ERR("%s: unexpected GDN V-energy tensor %s type/shape: type=%s ne=[%lld,%lld,%lld,%lld]\n",
                    __func__, t->name, ggml_type_name(t->type),
                    (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);
            return false;
        }

        std::lock_guard<std::mutex> lock(m_mutex);
        const bool is_host = ggml_backend_buffer_is_host(t->buffer);
        if (!is_host) {
            const size_t nbytes = ggml_nbytes(t);
            m_node_data.resize(nbytes);
            ggml_backend_tensor_get(t, m_node_data.data(), 0, nbytes);
        }
        const char * data = is_host ? (const char *) t->data : m_node_data.data();
        const std::string entry_name = "prune.gdn.blk." + std::to_string(v_energy_layer) + ".v_energy";
        auto & e = m_stats[entry_name];
        const std::string v_gram_name = "prune.gdn.blk." + std::to_string(v_energy_layer) + ".v_gram";
        auto & v_gram_entry = m_stats[v_gram_name];
        if (e.values.empty()) {
            e.values.resize((size_t) n_head_v * head_dim, 0.0f);
            e.counts.resize(1, 0);
        }
        if (v_gram_entry.values.empty()) {
            v_gram_entry.values.resize((size_t) head_dim * head_dim, 0.0f);
            v_gram_entry.counts.resize(1, 0);
        }
        auto & v_gram64 = m_qk_gram64[v_gram_name];
        if (v_gram64.empty()) v_gram64.resize((size_t) head_dim * head_dim, 0.0);
        for (int64_t i3 = 0; i3 < t->ne[3]; ++i3) {
            for (int64_t i2 = 0; i2 < t->ne[2]; ++i2) {
                for (int head = 0; head < n_head_v; ++head) {
                    const float * x = (const float *) (data + head * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3]);
                    float * he = e.values.data() + (size_t) head * head_dim;
                    for (int p = 0; p < head_dim; ++p) {
                        he[p] += x[p] * x[p];
                        const double xp = x[p];
                        double * grow = v_gram64.data() + (size_t) p * head_dim;
                        for (int c = 0; c < head_dim; ++c) grow[c] += xp * (double) x[c];
                    }
                    v_gram_entry.counts[0] += 1;
                }
                e.counts[0] += n_head_v;
            }
        }
        if (!all_finite(e.values.data(), e.values.size()) || !all_finite(v_gram64.data(), v_gram64.size())) {
            LOG_ERR("%s: non-finite GDN V energy in layer %d\n", __func__, v_energy_layer);
            std::exit(1);
        }
        return true;
    }

    int final_output_layer = -1;
    const bool is_final_output = m_params.prune_qk_gram && parse_gdn_final_output_node(t->name, final_output_layer);
    if (is_final_output) {
        if (ask) return true;
        constexpr int n_head_v = 32;
        constexpr int head_dim = 128;
        constexpr int value_dim = n_head_v * head_dim;
        if (t->type != GGML_TYPE_F32 || t->ne[0] != value_dim) {
            LOG_ERR("%s: unexpected GDN final-output tensor %s type/shape: type=%s ne=[%lld,%lld,%lld,%lld]\n",
                    __func__, t->name, ggml_type_name(t->type),
                    (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);
            return false;
        }

        std::lock_guard<std::mutex> lock(m_mutex);
        const bool is_host = ggml_backend_buffer_is_host(t->buffer);
        if (!is_host) {
            const size_t nbytes = ggml_nbytes(t);
            m_node_data.resize(nbytes);
            ggml_backend_tensor_get(t, m_node_data.data(), 0, nbytes);
        }
        const char * data = is_host ? (const char *) t->data : m_node_data.data();
        const std::string name = "prune.gdn.blk." + std::to_string(final_output_layer) + ".final_gram";
        auto & e = m_stats[name];
        if (e.values.empty()) {
            e.values.resize((size_t) n_head_v * head_dim * head_dim, 0.0f);
            e.counts.resize(1, 0);
        }
        auto & gram64 = m_qk_gram64[name];
        if (gram64.empty()) gram64.resize(e.values.size(), 0.0);
        if (gram64.size() != e.values.size()) {
            LOG_ERR("%s: inconsistent GDN final-output Gram size for layer %d\n", __func__, final_output_layer);
            return false;
        }

        for (int64_t i2 = 0; i2 < t->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
                const float * row = (const float *) (data + i1 * t->nb[1] + i2 * t->nb[2]);
                for (int head = 0; head < n_head_v; ++head) {
                    const float * x = row + (size_t) head * head_dim;
                    double * g = gram64.data() + (size_t) head * head_dim * head_dim;
                    for (int r = 0; r < head_dim; ++r) {
                        const double xr = x[r];
                        double * grow = g + (size_t) r * head_dim;
                        for (int c = 0; c < head_dim; ++c) grow[c] += xr * (double) x[c];
                    }
                }
                e.counts[0] += n_head_v;
            }
        }
        if (!all_finite(gram64.data(), gram64.size())) {
            LOG_ERR("%s: non-finite GDN final-output Gram in layer %d\n", __func__, final_output_layer);
            std::exit(1);
        }
        return true;
    }

    const struct ggml_tensor * src0 = t->src[0];
    const struct ggml_tensor * src1 = t->src[1];
    std::string wname = filter_tensor_name(src0->name);

    const int32_t chunk_size = imatrix_accounting_chunk_size(m_params);

    // when ask is true, the scheduler wants to know if we are interested in data from this tensor
    // if we return true, a follow-up call will be made with ask=false in which we can do the actual collection
    if (ask) {
        if (t->op == GGML_OP_MUL_MAT_ID) return true; // collect all indirect matrix multiplications
        if (t->op != GGML_OP_MUL_MAT) return false;
        // why are small batches ignored (<16 tokens)?
        if (src1->ne[1] < 16 || src1->type != GGML_TYPE_F32) return false;
        if (!(wname.substr(0, 4) == "blk." || (m_params.process_output && wname == "output.weight"))) return false;
        return true;
    }

    std::lock_guard<std::mutex> lock(m_mutex);

    // copy the data from the GPU memory if needed
    const bool is_host = ggml_backend_buffer_is_host(src1->buffer);

    if (!is_host) {
        const size_t src1_nbytes = ggml_nbytes(src1);
        m_src1_data.resize(src1_nbytes);
        ggml_backend_tensor_get(src1, m_src1_data.data(), 0, src1_nbytes);
    }

    const char * data = is_host ? (const char *) src1->data : m_src1_data.data();
    GGML_ASSERT(src1->nb[0] == ggml_element_size(src1));

    // this has been adapted to the new format of storing merged experts in a single 3d tensor
    // ref: https://github.com/ggml-org/llama.cpp/pull/6387
    if (t->op == GGML_OP_MUL_MAT_ID) {
        //   ids  -> [n_experts_used, n_tokens]
        //   src1 -> [cols, n_expert_used, n_tokens]
        const ggml_tensor * ids = t->src[2];
        const int64_t n_as = src0->ne[2];
        const int64_t n_ids = ids->ne[0];

        // the top-k selected expert ids are stored in the ids tensor
        // for simplicity, always copy ids to host, because it is small
        // take into account that ids is not contiguous!

        GGML_ASSERT(ids->ne[1] == src1->ne[2]);

        // the extra dimension would need to be stored somewhere to be reflected in the imatrix file
        if (ggml_nrows(src1) != src1->ne[1] * src1->ne[2]) {
            LOG_ERR("%s: tensor has more than 3 dimensions: %s", __func__, wname.c_str());
            GGML_ASSERT(false);
        }

        m_ids.resize(ggml_nbytes(ids));
        ggml_backend_tensor_get(ids, m_ids.data(), 0, ggml_nbytes(ids));

        auto & e = m_stats[wname];

        if (e.counts.size() == 1 && n_as > 1) {
            // broadcast, when loading an old imatrix
            e.counts.resize(n_as, e.counts[0]);
        }
        if (e.values.empty()) {
            e.values.resize(src1->ne[0]*n_as, 0);
            e.counts.resize(n_as, 0);
        }
        else if (e.values.size() != (size_t)src1->ne[0]*n_as) {
            LOG_ERR("%s: inconsistent size for %s (%d vs %d)\n", __func__, wname.c_str(), (int)e.values.size(), (int)(src1->ne[0]*n_as));
            exit(1); //GGML_ABORT("fatal error");
        }
        else if (e.counts.size() != (size_t)n_as) {
            LOG_ERR("%s: inconsistent expert count for %s (%d vs %d)\n", __func__, wname.c_str(), (int)e.counts.size(), (int)n_as);
            exit(1); //GGML_ABORT("fatal error");
        }
        LOG_DBGV(2, "%s[%d]: %32s, %s, %5d x %5d, %d\n", __func__, m_last_chunk, wname.c_str(), ggml_op_name(t->op), (int)src1->ne[0], (int)src1->ne[2], (int)src1->type);

        const int64_t ne0      = src1->ne[0];
        const int64_t n_tokens = src1->ne[2];

        // single pass over the routing ids
        std::vector<uint8_t> touched(n_as, 0);
        for (int64_t idx = 0; idx < n_ids; ++idx) {
            for (int64_t row = 0; row < n_tokens; ++row) {
                const int32_t ex = *(const int32_t *) (m_ids.data() + row * ids->nb[1] + idx * ids->nb[0]);

                GGML_ASSERT(ex >= 0 && ex < n_as);  // sanity check

                const int64_t i11 = idx % src1->ne[1];
                const float * x   = (const float *) (data + i11 * src1->nb[1] + row * src1->nb[2]);
                float *       acc = e.values.data() + ex * ne0;

                e.counts[ex]++;
                touched[ex] = 1;
                for (int64_t j = 0; j < ne0; ++j) {
                    acc[j] += x[j] * x[j];
                }
            }
        }

        // check for non-finite values, only checking experts that were routed to and touched
        for (int64_t ex = 0; ex < n_as; ++ex) {
            if (touched[ex] && !all_finite(e.values.data() + ex * ne0, ne0)) {
                LOG_ERR("%s: non-finite values detected in %s\n", __func__, wname.c_str());
                exit(1);
            }
        }

        for (int64_t ex = 0; ex < n_as; ++ex) {
            const int32_t n_chunk = e.counts[ex] / chunk_size;
            if (n_chunk > m_last_chunk) {
                const int32_t chunk_step = n_chunk - m_last_chunk;
                m_last_chunk = n_chunk;
                if (m_params.n_out_freq > 0 && (m_last_chunk % m_params.n_out_freq) / chunk_step == 0) {
                    save_imatrix();
                }
                if (m_params.n_save_freq > 0 && (m_last_chunk % m_params.n_save_freq) / chunk_step == 0) {
                    save_imatrix(m_last_chunk);
                }
            }
        }
    } else {
        auto & e = m_stats[wname];
        const int64_t n_mat = src0->ne[2] * src0->ne[3];

        // use a single count per dense tensor
        // (necessary when merging older GGUF-imatrix files with 3d tensors)
        if (e.counts.size() > 1) {
            bool all_equal = true;
            for (size_t i = 1; i < e.counts.size(); ++i) {
                if (e.counts[0] != e.counts[i]) {
                    all_equal = false;
                    break;
                }
            }
            if (all_equal) {
                e.counts.resize(1);
            }
        }
        if (e.values.empty()) {
            e.values.resize(src1->ne[0] * n_mat, 0);
            e.counts.resize(1, 0);
        }
        else if (e.values.size() != (size_t)(src1->ne[0] * n_mat)) {
            LOG_ERR("%s: inconsistent size for %s (%d vs %d)\n", __func__, wname.c_str(), (int)e.values.size(), (int)(src1->ne[0] * n_mat));
            exit(1); //GGML_ABORT("fatal error");
        }
        LOG_DBGV(2, "%s[%d]: %32s, %s, %5d x %5d x %5d, %d\n", __func__, m_last_chunk, wname.c_str(), ggml_op_name(t->op), (int)src1->ne[0], (int)src1->ne[1], (int)src1->ne[2], (int)src1->type);

        const int64_t ne0 = src1->ne[0];

        for (int64_t i3 = 0; i3 < src1->ne[3]; ++i3) {
            for (int64_t i2 = 0; i2 < src1->ne[2]; ++i2) {
                // handle 3D+ tensors, but flatten 3D+ activations when model tensor is 2D
                const int64_t mat_id = (i3 % src0->ne[3]) * src0->ne[2] + (i2 % src0->ne[2]);
                float *       acc    = e.values.data() + mat_id * ne0;

                for (int64_t row = 0; row < src1->ne[1]; ++row) {
                    const float * x = (const float *) (data + row * src1->nb[1] + i2 * src1->nb[2] + i3 * src1->nb[3]);
                    for (int64_t j = 0; j < ne0; ++j) {
                        acc[j] += x[j] * x[j];
                    }
                }
            }
        }

        // check for non-finite values
        if (!all_finite(e.values.data(), e.values.size())) {
            LOG_ERR("%s: non-finite values detected in %s\n", __func__, wname.c_str());
            exit(1);
        }
        // only 1 count in practice, except when a tensor is used for both MUL_MAT_ID and MUL_MAT
        for (size_t i = 0; i < e.counts.size(); ++i) {
            e.counts[i] += ggml_nrows(src1) / n_mat;
            const int32_t n_chunk = e.counts[i] / chunk_size;
            if (n_chunk > m_last_chunk) {
                const int32_t chunk_step = n_chunk - m_last_chunk;
                m_last_chunk = n_chunk;
                if (m_params.n_out_freq > 0 && (m_last_chunk % m_params.n_out_freq) / chunk_step == 0) {
                    save_imatrix();
                }
                if (m_params.n_save_freq > 0 && (m_last_chunk % m_params.n_save_freq) / chunk_step == 0) {
                    save_imatrix(m_last_chunk);
                }
            }
        }
    }

    return true;
}

void IMatrixCollector::save_imatrix_legacy(int32_t ncall) const {
    auto fname = m_params.out_file;

    if (ncall > 0) {
        fname += ".at_";
        fname += std::to_string(ncall);
    }

    // warn when writing imatrix entries that do not have full data
    // this can happen with MoE models where some of the experts end up not being exercised by the provided training data

    int n_entries = 0;
    std::vector<std::string> to_store;

    bool is_first = true; // for printing
    for (const auto & kv : m_stats) {
        const int n_all = kv.second.counts.size();

        if (n_all == 0) {
            continue;
        }

        int n_zeros = 0;
        for (const int c : kv.second.counts) {
            if (c == 0) {
                n_zeros++;
            }
        }

        if (n_zeros != 0 && is_first) {
            LOG_INF("\n");
            is_first = false;
        }

        if (n_zeros == n_all) {
            LOG_WRN("%s: entry '%40s' has no data - skipping\n", __func__, kv.first.c_str());
            continue;
        }

        if (n_zeros > 0) {
            LOG_WRN("%s: entry '%40s' has partial data (%.2f%%)\n", __func__, kv.first.c_str(), 100.0f * (n_all - n_zeros) / n_all);
        }

        n_entries++;
        to_store.push_back(kv.first);
    }

    if (to_store.size() < m_stats.size()) {
        LOG_WRN("%s: storing only %zu out of %zu entries\n", __func__, to_store.size(), m_stats.size());
    }

    // deterministic tensor name order
    std::sort(to_store.begin(), to_store.end());

    const int32_t chunk_size = imatrix_accounting_chunk_size(m_params);

    std::ofstream out(fname, std::ios::binary);
    out.write((const char *) &n_entries, sizeof(n_entries));
    for (const auto & name : to_store) {
        const auto & stat = m_stats.at(name);
        const int32_t len = name.size();
        out.write((const char *) &len, sizeof(len));
        out.write(name.c_str(), len);
        // ceiling division to avoid accidental zeros
        const int32_t ncall = (*std::max_element(stat.counts.begin(), stat.counts.end()) + (chunk_size - 1)) / chunk_size;
        out.write((const char *) &ncall, sizeof(ncall));
        const int32_t nval = stat.values.size();
        const int32_t nmat = stat.counts.size();
        out.write((const char *) &nval, sizeof(nval));
        if (nval > 0 && nmat > 0) {
            const auto gram_it = m_qk_gram64.find(name);
            const bool has_gram64 = gram_it != m_qk_gram64.end();
            if (has_gram64 && gram_it->second.size() != (size_t) nval) {
                throw std::runtime_error("FP64 Gram size mismatch while saving legacy imatrix: " + name);
            }
            std::vector<float> tmp(nval);
            for (int32_t i = 0; i < nval; i++) {
                float count = static_cast<float>(stat.counts[i / (nval / nmat)]);
                float value = has_gram64 ? (float) gram_it->second[(size_t) i] : stat.values[i];
                if (count == 0.0f) {
                    // store 1 for partial data
                    value = 1.0f;
                    count = 1.0f;
                }
                tmp[i] = (value / count) * static_cast<float>(ncall);
            }
            out.write((const char *) tmp.data(), nval * sizeof(float));
        }
    }

    // Write the number of call the matrix was computed with
    out.write((const char *) &m_last_chunk, sizeof(m_last_chunk));

    // Write the input filename at the end of the file to later on specify it in quantize
    {
        const char * dataset_file = m_params.prompt_file.c_str();
        int32_t len = m_params.prompt_file.size();
        // When there is no prompt but there were other imatrix files loaded, use the last dataset
        if (m_params.prompt_file.empty() && !m_datasets.empty()) {
            const std::string & dataset_str = m_datasets[m_datasets.size() - 1];
            dataset_file = dataset_str.c_str();
            len = dataset_str.size();
        } else if (m_params.prompt_file.empty() && !m_params.imatrix_request_token_trie.empty()) {
            dataset_file = m_params.imatrix_request_token_trie.c_str();
            len = m_params.imatrix_request_token_trie.size();
        }
        out.write((const char *) &len, sizeof(len));
        out.write(dataset_file, len);
    }

    LOGV(1, "\n");
    LOG_DBGV(1, "%s: stored collected data after %d chunks in %s\n", __func__, m_last_chunk, fname.c_str());
}

void IMatrixCollector::save_imatrix(int32_t n_chunk) const {
    auto fname = m_params.out_file;
    int8_t use_legacy_format = m_params.imat_dat;

    if (use_legacy_format > 0) {
        this->save_imatrix_legacy(n_chunk);
        return;
    }
    // only warn when `--output-format gguf` is not specified
    if (use_legacy_format == 0 && !string_ends_with(fname, ".gguf")) {
        LOG_WRN("\n%s: saving imatrix using GGUF format with a different suffix than .gguf\n", __func__);
        LOG_WRN("%s: if you want the previous imatrix format, use --output-format dat\n", __func__);
    }

    if (n_chunk > 0) {
        fname += ".at_";
        fname += std::to_string(n_chunk);
    }

    // write imatrix entries even if they don't have full data. (can be corrected when reading)
    // this can happen with MoE models where some of the experts end up not being exercised by the provided training data

    std::vector<std::string> to_store;
    size_t data_size = 0;

    bool is_first = true; // for printing
    for (const auto & kv : m_stats) {
        const int n_all = kv.second.counts.size();

        int n_zeros = 0;
        for (const auto c : kv.second.counts) {
            if (c == 0) {
                n_zeros++;
            }
        }

        if (n_zeros != 0 && is_first) {
            LOG_INF("\n");
            is_first = false;
        }

        if (n_zeros > 0) {
            LOG_WRN("%s: entry '%40s' has partial data (%.2f%%)\n", __func__, kv.first.c_str(), 100.0f * (n_all - n_zeros) / n_all);
        }

        to_store.push_back(kv.first);
        data_size += GGML_PAD(ggml_tensor_overhead() + sizeof(float) * kv.second.values.size(), GGML_MEM_ALIGN);
        data_size += GGML_PAD(ggml_tensor_overhead() + sizeof(float) * kv.second.counts.size(), GGML_MEM_ALIGN);
    }

    // deterministic tensor name order
    std::sort(to_store.begin(), to_store.end());

    struct ggml_init_params params = {
        /* .mem_size   = */ data_size,
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ false,
    };
    struct ggml_context * ctx = ggml_init(params);
    struct gguf_context * ctx_gguf = gguf_init_empty();

    {
        std::vector<const char *> datasets;
        datasets.reserve(m_datasets.size() + 1);
        for (size_t i = 0; i < m_datasets.size(); ++i) {
            datasets.push_back(m_datasets[i].c_str());
        }
        if (!m_params.prompt_file.empty()) {
            datasets.push_back(m_params.prompt_file.c_str());
        }
        if (!m_params.imatrix_request_token_trie.empty()) {
            datasets.push_back(m_params.imatrix_request_token_trie.c_str());
        }

        gguf_set_val_str(ctx_gguf, "general.type", "imatrix");
        // Write the dataset paths
        gguf_set_arr_str(ctx_gguf, LLM_KV_IMATRIX_DATASETS, datasets.data(), datasets.size());
        // Write the number of chunks the matrix was computed with
        gguf_set_val_u32(ctx_gguf, LLM_KV_IMATRIX_CHUNK_COUNT, m_last_chunk);
        gguf_set_val_u32(ctx_gguf, LLM_KV_IMATRIX_CHUNK_SIZE, imatrix_accounting_chunk_size(m_params));
    }

    for (const auto & name : to_store) {
        const auto & stat = m_stats.at(name);
        const int32_t nval = (int32_t) stat.values.size();
        const int32_t nmat = (int32_t) stat.counts.size();
        if (nval > 0 && nmat > 0) {
            const auto gram_it = m_qk_gram64.find(name);
            const bool has_gram64 = gram_it != m_qk_gram64.end();
            if (has_gram64 && gram_it->second.size() != (size_t) nval) {
                throw std::runtime_error("FP64 Gram size mismatch while saving imatrix: " + name);
            }
            struct ggml_tensor * in_sum2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nval / nmat, nmat);
            struct ggml_tensor * counts  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, nmat);
            ggml_format_name(in_sum2, "%s.in_sum2", name.c_str());
            ggml_format_name(counts, "%s.counts", name.c_str());

            for (int32_t j = 0; j < nval; ++j) {
                ((float *) in_sum2->data)[j] = has_gram64 ? (float) gram_it->second[(size_t) j] : stat.values[j];
            }
            for (int32_t j = 0; j < nmat; ++j) {
                ((float *) counts->data)[j] = (float) stat.counts[j];
            }

            gguf_add_tensor(ctx_gguf, in_sum2);
            gguf_add_tensor(ctx_gguf, counts);
        }
    }

    gguf_write_to_file(ctx_gguf, fname.c_str(), false);

    LOGV(1, "\n");
    LOG_DBGV(1, "%s: stored collected data after %d chunks in %s\n", __func__, m_last_chunk, fname.c_str());

    gguf_free(ctx_gguf);
    ggml_free(ctx);
}

bool IMatrixCollector::load_imatrix(const char * file_name) {
    common_imatrix loaded;
    if (!common_imatrix_load(file_name, loaded)) {
        return false;
    }

    const int32_t chunk_size = imatrix_accounting_chunk_size(m_params);
    const bool is_legacy = loaded.is_legacy;

    for (auto & [name, entry] : loaded.entries) {
        auto & e = m_stats[name];
        const bool is_gram = is_gdn_gram_entry(name);

        if (is_legacy) {
            // Legacy format: sums contain (raw_sum/raw_count)*ncall, counts contain {ncall}
            // Reconstruct raw form by multiplying by chunk_size
            if (e.values.empty()) {
                e.values.resize(entry.sums.size(), 0.0f);
                e.counts.resize(1, 0);
            }
            std::vector<double> * gram64 = nullptr;
            if (is_gram) {
                auto & g = m_qk_gram64[name];
                if (g.empty()) g.resize(entry.sums.size(), 0.0);
                if (g.size() != entry.sums.size()) {
                    LOG_ERR("%s: mismatched FP64 Gram size for %s\n", __func__, name.c_str());
                    return false;
                }
                gram64 = &g;
            }
            for (size_t j = 0; j < entry.sums.size(); ++j) {
                const double raw = (double) entry.sums[j] * chunk_size;
                if (gram64) {
                    (*gram64)[j] += raw;
                } else {
                    e.values[j] += (float) raw;
                }
            }
            for (size_t j = 0; j < e.counts.size(); ++j) {
                e.counts[j] += entry.counts[0] * chunk_size;
            }
        } else {
            // GGUF format: raw sums and counts, accumulate directly
            const int64_t nval    = entry.sums.size();
            const int64_t ncounts = entry.counts.size();

            if (e.values.empty()) {
                e.values.resize(nval, 0.0f);
            } else if ((size_t) nval != e.values.size()) {
                LOG_ERR("%s: mismatched sums size for %s: %zu != %zu\n", __func__, name.c_str(), (size_t) nval, e.values.size());
                return false;
            }

            if (e.counts.empty()) {
                e.counts.resize(ncounts, 0);
            } else if (e.counts.size() == 1 && ncounts > 1) {
                e.counts.resize(ncounts, e.counts[0]);
            } else if ((size_t) ncounts != e.counts.size()) {
                LOG_ERR("%s: mismatched counts size for %s: %zu != %zu\n", __func__, name.c_str(), (size_t) ncounts, e.counts.size());
                return false;
            }

            if (is_gram) {
                auto & gram64 = m_qk_gram64[name];
                if (gram64.empty()) gram64.resize((size_t) nval, 0.0);
                if (gram64.size() != (size_t) nval) {
                    LOG_ERR("%s: mismatched FP64 Gram size for %s\n", __func__, name.c_str());
                    return false;
                }
                for (int64_t j = 0; j < nval; ++j) {
                    gram64[(size_t) j] += (double) entry.sums[(size_t) j];
                }
            } else {
                for (int64_t j = 0; j < nval; ++j) {
                    e.values[(size_t) j] += entry.sums[(size_t) j];
                }
            }
            for (int64_t j = 0; j < ncounts; ++j) {
                e.counts[j] += entry.counts[j];
            }
        }
    }

    m_datasets.insert(m_datasets.end(), loaded.datasets.begin(), loaded.datasets.end());

    if (!is_legacy && loaded.has_metadata && loaded.chunk_size > 0) {
        if (loaded.chunk_size != chunk_size) {
            LOG_ERR("%s: imatrix chunk size mismatch: file=%d current=%d\n",
                    __func__, loaded.chunk_size, chunk_size);
            return false;
        }
        // GGUF imatrix already carries the exact chunk count. This is safer
        // than inferring it from custom Gram counts or sparse MoE routing.
        m_last_chunk += loaded.chunk_count;
    } else {
        // Legacy fallback: infer from stock entries only. prune.* entries have
        // different count semantics and must never drive chunk accounting.
        int64_t max_count = 0;
        for (const auto & [name, stats] : m_stats) {
            if (name.rfind("prune.", 0) == 0) continue;
            for (int64_t count : stats.counts) {
                max_count = std::max(max_count, count);
            }
        }
        m_last_chunk = max_count / chunk_size;
    }

    return true;
}

static IMatrixCollector g_collector;

static bool ik_collect_imatrix(struct ggml_tensor * t, bool ask, void * user_data) {
    return g_collector.collect_imatrix(t, ask, user_data);
}

struct results_log_softmax {
    double log_softmax;
    float  logit;
    float  prob;
};

static std::vector<float> softmax(const std::vector<float> & logits) {
    std::vector<float> probs(logits.size());
    float max_logit = logits[0];
    for (float v : logits) {
        max_logit = std::max(max_logit, v);
    }
    double sum_exp = 0.0;
    for (size_t i = 0; i < logits.size(); i++) {
        // Subtract the maximum logit value from the current logit value for numerical stability
        const float logit = logits[i] - max_logit;
        const float exp_logit = expf(logit);
        sum_exp += exp_logit;
        probs[i] = exp_logit;
    }
    for (size_t i = 0; i < probs.size(); i++) {
        probs[i] /= sum_exp;
    }
    return probs;
}

static results_log_softmax log_softmax(int n_vocab, const float * logits, int tok) {
    float max_logit = logits[0];
    for (int i = 1; i < n_vocab; ++i) {
        max_logit = std::max(max_logit, logits[i]);
    }
    double sum_exp = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        sum_exp += expf(logits[i] - max_logit);
    }
    return {logits[tok] - max_logit - log(sum_exp), logits[tok], expf(logits[tok] - max_logit) / (float) sum_exp};
}

static void process_logits(
    int n_vocab, const float * logits, const int * tokens, int n_token, std::vector<std::thread> & workers,
    double & nll, double & nll2, float * logit_history, float * prob_history) {
    std::mutex mutex;
    int counter = 0;
    auto compute = [&mutex, &counter, &nll, &nll2, logit_history, prob_history, n_vocab, logits, tokens, n_token] () {
        double local_nll  = 0;
        double local_nll2 = 0;
        while (true) {
            std::unique_lock<std::mutex> lock(mutex);
            int i = counter++;
            if (i >= n_token) {
                nll += local_nll; nll2 += local_nll2;
                break;
            }
            lock.unlock();
            const results_log_softmax results = log_softmax(n_vocab, logits + i*n_vocab, tokens[i+1]);
            const double v = -results.log_softmax;
            local_nll += v;
            local_nll2 += v*v;

            logit_history[i] = results.logit;
            prob_history[i]  = results.prob;
        }
    };
    for (auto & w : workers) {
        w = std::thread(compute);
    }
    compute();
    for (auto & w : workers) {
        w.join();
    }
}

static bool compute_imatrix_request_trie(
        llama_context * ctx,
        llama_context * ctx_mtp,
        common_speculative * mtp_spec,
        const common_params & params,
        const request_token_trie & trie,
        int32_t n_ctx) {
    if (params.compute_ppl) {
        LOG_ERR("%s: request-token trie calibration requires --no-ppl\n", __func__);
        return false;
    }
    if (params.i_chunk != 0) {
        LOG_ERR("%s: --chunk/--from-chunk is not supported with --request-token-trie\n", __func__);
        return false;
    }
    if (trie.leaves.empty() || trie.unique_tokens == 0) {
        LOG_ERR("%s: request-token trie has no live calibration tree\n", __func__);
        return false;
    }

    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const int32_t n_batch = params.n_batch;
    if (n_batch <= 0) {
        LOG_ERR("%s: invalid batch size %d\n", __func__, n_batch);
        return false;
    }

    if (trie.max_leaf_tokens > (size_t) n_ctx) {
        LOG_ERR("%s: longest request-token trie leaf has %zu tokens, exceeding --ctx-size %d\n",
                __func__, trie.max_leaf_tokens, n_ctx);
        return false;
    }

    LOG_INF("%s: traversing canonical request trie once (%llu unique token edges, %zu leaves), "
            "single-sequence GPU context n_ctx=%d, batch=%d; branch states are checkpointed on CPU\n",
            __func__, (unsigned long long) trie.unique_tokens, trie.leaves.size(), n_ctx, n_batch);

    llama_batch batch = llama_batch_init(std::min<int32_t>(n_batch, n_ctx), 0, 1);
    auto started = std::chrono::high_resolution_clock::now();
    uint64_t done_tokens = 0;

    std::vector<llama_token> unique_tokens;
    unique_tokens.reserve((size_t) trie.unique_tokens);
    for (size_t id = 1; id < trie.nodes.size(); ++id) {
        if (trie.nodes[id].live) unique_tokens.push_back(trie.nodes[id].token);
    }
    g_collector.collect_token_counts(unique_tokens, n_vocab, unique_tokens.size());

    llama_memory_clear(llama_get_memory(ctx), true);
    if (mtp_spec) common_speculative_reset(mtp_spec);

    auto synchronize_branch_state = [&]() {
        llama_synchronize(ctx);
        if (ctx_mtp && ctx_mtp != ctx) llama_synchronize(ctx_mtp);
    };

    struct branch_checkpoint {
        std::vector<uint8_t> target;
        std::vector<uint8_t> mtp_context;
        std::vector<uint8_t> mtp_driver;
        size_t depth = 0;
    };

    const bool mtp_has_separate_memory = ctx_mtp && llama_get_memory(ctx_mtp) != llama_get_memory(ctx);

    auto save_seq_state = [&](llama_context * state_ctx, std::vector<uint8_t> & out, const char * label) -> bool {
        if (!state_ctx) {
            out.clear();
            return true;
        }
        const size_t size = llama_state_seq_get_size(state_ctx, 0);
        if (size == 0) {
            LOG_ERR("%s: failed to size %s branch checkpoint\n", __func__, label);
            return false;
        }
        out.resize(size);
        const size_t written = llama_state_seq_get_data(state_ctx, out.data(), out.size(), 0);
        if (written != out.size()) {
            LOG_ERR("%s: failed to save %s branch checkpoint: wrote %zu of %zu bytes\n",
                    __func__, label, written, out.size());
            return false;
        }
        return true;
    };

    auto load_seq_state = [&](llama_context * state_ctx, const std::vector<uint8_t> & in, const char * label) -> bool {
        if (!state_ctx || in.empty()) return true;
        const size_t read = llama_state_seq_set_data(state_ctx, in.data(), in.size(), 0);
        if (read != in.size()) {
            LOG_ERR("%s: failed to restore %s branch checkpoint: read %zu of %zu bytes\n",
                    __func__, label, read, in.size());
            return false;
        }
        return true;
    };

    size_t checkpoint_stack_bytes = 0;
    size_t checkpoint_stack_peak = 0;
    size_t checkpoint_single_peak = 0;
    uint64_t checkpoint_saves = 0;
    uint64_t checkpoint_restores = 0;

    auto save_checkpoint = [&](branch_checkpoint & cp, size_t depth) -> bool {
        synchronize_branch_state();
        cp.depth = depth;
        if (!save_seq_state(ctx, cp.target, "target")) return false;
        if (mtp_has_separate_memory && !save_seq_state(ctx_mtp, cp.mtp_context, "MTP context")) return false;
        if (mtp_spec) {
            common_speculative_get_state(mtp_spec, 0, cp.mtp_driver);
        }
        const size_t bytes = cp.target.size() + cp.mtp_context.size() + cp.mtp_driver.size();
        checkpoint_stack_bytes += bytes;
        checkpoint_stack_peak = std::max(checkpoint_stack_peak, checkpoint_stack_bytes);
        checkpoint_single_peak = std::max(checkpoint_single_peak, bytes);
        ++checkpoint_saves;
        LOG_INF("%s: saved CPU branch checkpoint at depth %zu: %.2f MiB "
                "(checkpoint stack %.2f MiB)\n",
                __func__, depth, bytes / (1024.0 * 1024.0), checkpoint_stack_bytes / (1024.0 * 1024.0));
        return true;
    };

    auto restore_checkpoint = [&](const branch_checkpoint & cp) -> bool {
        if (!load_seq_state(ctx, cp.target, "target")) return false;
        if (mtp_has_separate_memory && !load_seq_state(ctx_mtp, cp.mtp_context, "MTP context")) return false;
        if (mtp_spec && !cp.mtp_driver.empty()) {
            common_speculative_set_state(mtp_spec, 0, cp.mtp_driver);
        }
        ++checkpoint_restores;
        return true;
    };

    auto release_checkpoint = [&](const branch_checkpoint & cp) {
        const size_t bytes = cp.target.size() + cp.mtp_context.size() + cp.mtp_driver.size();
        GGML_ASSERT(checkpoint_stack_bytes >= bytes);
        checkpoint_stack_bytes -= bytes;
    };

    std::function<bool(uint64_t, size_t)> visit;
    visit = [&](uint64_t start_node, size_t depth_before) -> bool {
        uint64_t current = start_node;
        size_t depth = depth_before;

        if (current != 0) {
            std::vector<uint64_t> chain;
            while (true) {
                chain.push_back(current);
                const auto & children = trie.children[(size_t) current];
                if (children.size() != 1) break;
                current = children[0];
            }

            for (size_t offset = 0; offset < chain.size(); offset += (size_t) n_batch) {
                const int32_t batch_size = (int32_t) std::min((size_t) n_batch, chain.size() - offset);
                common_batch_clear(batch);
                for (int32_t i = 0; i < batch_size; ++i) {
                    const uint64_t node_id = chain[offset + (size_t) i];
                    common_batch_add(batch, trie.nodes[(size_t) node_id].token,
                                     (llama_pos) depth + (llama_pos) offset + i, { 0 }, true);
                }
                if (llama_decode(ctx, batch)) {
                    LOG_ERR("%s: failed to eval trie edge at depth %zu\n", __func__, depth + offset);
                    return false;
                }
                if (mtp_spec && !common_speculative_process(mtp_spec, batch)) {
                    LOG_ERR("%s: failed to execute MTP graph at trie depth %zu\n", __func__, depth + offset);
                    return false;
                }
                done_tokens += (uint64_t) batch_size;
                if ((done_tokens % 4096) < (uint64_t) batch_size || done_tokens == trie.unique_tokens) {
                    const double fraction = (double) done_tokens / trie.unique_tokens;
                    const auto now = std::chrono::high_resolution_clock::now();
                    const double seconds = std::chrono::duration<double>(now - started).count();
                    const double eta = fraction > 0.0 ? seconds * (1.0 - fraction) / fraction : 0.0;
                    LOG_INF("%s: %llu/%llu unique trie tokens (%.1f%%, ETA %.1f min)\n",
                            __func__, (unsigned long long) done_tokens,
                            (unsigned long long) trie.unique_tokens, 100.0 * fraction, eta / 60.0);
                }
            }
            depth += chain.size();
        }

        const auto & children = trie.children[(size_t) current];
        if (children.empty()) return true;
        if (children.size() == 1) return visit(children[0], depth);

        branch_checkpoint cp;
        if (!save_checkpoint(cp, depth)) return false;
        for (size_t i = 0; i < children.size(); ++i) {
            // The first child can continue from the live parent state. Every
            // subsequent child restores the exact same target + recurrent +
            // MTP state from host memory. The imatrix accumulator is never
            // restored, so each trie edge contributes exactly once.
            if (i > 0 && !restore_checkpoint(cp)) {
                release_checkpoint(cp);
                return false;
            }
            if (!visit(children[i], depth)) {
                release_checkpoint(cp);
                return false;
            }
        }
        release_checkpoint(cp);
        return true;
    };

    const bool ok = visit(0, 0);
    synchronize_branch_state();

    llama_batch_free(batch);
    if (!ok) return false;
    if (done_tokens != trie.unique_tokens) {
        LOG_ERR("%s: trie traversal token mismatch: decoded=%llu expected=%llu\n",
                __func__, (unsigned long long) done_tokens, (unsigned long long) trie.unique_tokens);
        return false;
    }
    LOG_INF("%s: CPU checkpoint summary: saves=%llu restores=%llu, largest=%.2f MiB, peak stack=%.2f MiB\n",
            __func__, (unsigned long long) checkpoint_saves, (unsigned long long) checkpoint_restores,
            checkpoint_single_peak / (1024.0 * 1024.0), checkpoint_stack_peak / (1024.0 * 1024.0));
    return true;
}

static bool compute_imatrix(
        llama_context * ctx,
        common_speculative * mtp_spec,
        const common_params & params,
        const int32_t n_ctx) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const bool add_bos = llama_vocab_get_add_bos(vocab);

    if (llama_pooling_type(ctx) != LLAMA_POOLING_TYPE_LAST) {
        GGML_ASSERT(!llama_vocab_get_add_eos(vocab));
    }

    auto tim1 = std::chrono::high_resolution_clock::now();
    LOG_INF("%s: tokenizing the input ..\n", __func__);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, true, params.parse_special);

    auto tim2 = std::chrono::high_resolution_clock::now();
    LOG_INF("%s: tokenization took %g ms\n",__func__,1e-3*std::chrono::duration_cast<std::chrono::microseconds>(tim2-tim1).count());

    if (params.i_chunk > 0) {
        if (size_t((params.i_chunk + 2)*n_ctx) >= tokens.size()) {
            LOG_ERR("%s: there will be not enough tokens left after removing %d chunks\n", __func__, params.i_chunk);
            return false;
        }
        LOG_INF("%s: removing initial %d chunks (%d tokens)\n", __func__, params.i_chunk, params.i_chunk*n_ctx);
        tokens.erase(tokens.begin(), tokens.begin() + params.i_chunk*n_ctx);
    }

    if (int(tokens.size()) < 2*n_ctx) {
        LOG_ERR("%s: you need at least %d tokens for a context of %d tokens\n", __func__, 2*n_ctx, n_ctx);
        LOG_ERR("%s: the data file you provided tokenizes to only %zu tokens\n", __func__, tokens.size());
        return false;
    }

    std::vector<float> logit_history;
    std::vector<float> prob_history;

    if (params.compute_ppl) {
        logit_history.resize(tokens.size());
        prob_history.resize(tokens.size());
    }

    const int n_chunk_max = tokens.size() / n_ctx;

    const int n_chunk = params.n_chunks < 0 ? n_chunk_max : std::min(params.n_chunks, n_chunk_max);
    const bool prune_partial_enabled = params.n_chunks < 0 && !params.compute_ppl &&
        (params.prune_qk_gram || params.prune_vocab_counts || params.prune_mtp);
    const int tail_tokens = prune_partial_enabled
        ? (int) (tokens.size() - (size_t) n_chunk * n_ctx)
        : 0;
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int n_batch = params.n_batch;

    int count = 0;
    double nll = 0.0;
    double nll2 = 0.0;

    const int num_batches = (n_ctx + n_batch - 1) / n_batch;
    const int n_seq = std::max(1, n_batch / n_ctx);

    GGML_ASSERT(n_batch < n_ctx || n_batch % n_ctx == 0);
    GGML_ASSERT(params.n_ctx == n_seq * n_ctx);

    g_collector.collect_token_counts(tokens, n_vocab, (size_t) n_chunk * n_ctx + tail_tokens);

    llama_batch batch = llama_batch_init(std::min(n_batch, n_ctx*n_seq), 0, 1);

    std::vector<float> logits;
    if (params.compute_ppl && num_batches > 1) {
        logits.reserve((size_t)n_ctx * n_vocab);
    }

    LOG_INF("%s: computing over %d full chunks%s, n_ctx=%d, batch_size=%d, n_seq=%d\n",
            __func__, n_chunk, tail_tokens > 0 ? " + partial tail" : "", n_ctx, n_batch, n_seq);

    std::vector<std::thread> workers(std::thread::hardware_concurrency() - 1);

    for (int i = 0; i < n_chunk; i += n_seq) {
        const int start =     i * n_ctx;
        const int end   = start + n_ctx;

        const int n_seq_batch = std::min(n_seq, n_chunk - i);

        const auto t_start = std::chrono::high_resolution_clock::now();

        // clear the KV cache
        llama_memory_clear(llama_get_memory(ctx), true);
        if (mtp_spec) {
            // Each imatrix chunk is an independent sequence whose positions restart at 0.
            // Keep the draft-MTP context in lock-step with the target reset, including
            // pending_h from the previous chunk.
            common_speculative_reset(mtp_spec);
        }

        for (int j = 0; j < num_batches; ++j) {
            const int batch_start = start + j * n_batch;
            const int batch_size  = std::min(end - batch_start, n_batch);

            // clear the batch
            common_batch_clear(batch);

            for (int seq = 0; seq < n_seq_batch; seq++) {
                int seq_start = batch_start + seq*n_ctx;

                // save original token and restore it after eval
                const auto token_org = tokens[seq_start];

                // add BOS token for the first batch of each chunk
                if (add_bos && j == 0) {
                    tokens[seq_start] = llama_vocab_bos(vocab);
                }
                for (int k = 0; k < batch_size; ++k) {
                    // NOTE: specifying all logits to get activations for the output.weight tensor
                    //       and also for the perplexity calculation.
                    // TODO: only get outputs when (params.process_output || params.compute_ppl)
                    //       (not possible when this skips FFN computation of the last layer)
                    common_batch_add(batch, tokens[seq_start + k], j*n_batch + k, { seq }, true);
                }

                // restore the original token in case it was set to BOS
                tokens[seq_start] = token_org;
            }

            if (llama_decode(ctx, batch)) {
                LOG_ERR("%s : failed to eval\n", __func__);
                llama_batch_free(batch);
                return false;
            }

            if (mtp_spec && !common_speculative_process(mtp_spec, batch)) {
                LOG_ERR("%s: failed to execute MTP calibration graph\n", __func__);
                llama_batch_free(batch);
                return false;
            }

            if (params.compute_ppl && num_batches > 1) {
                const auto * batch_logits = llama_get_logits(ctx);
                logits.insert(logits.end(), batch_logits, batch_logits + batch_size * n_vocab);
            }
        }


        if (i == 0) {
            llama_synchronize(ctx);
            const auto t_end = std::chrono::high_resolution_clock::now();
            const float t_total = std::chrono::duration<float>(t_end - t_start).count();
            LOG_INF("%s: %.2f seconds per pass - ETA ", __func__, t_total);
            int total_seconds = (int)(t_total * n_chunk / n_seq);
            if (total_seconds >= 60*60) {
                LOG("%d hours ", total_seconds / (60*60));
                total_seconds = total_seconds % (60*60);
            }
            LOG("%.2f minutes\n", total_seconds / 60.0);
        }

        if (params.compute_ppl) {
            const int first = n_ctx/2;
            for (int seq = 0; seq < n_seq_batch; seq++) {
                const float * all_logits = num_batches > 1 ? logits.data() : llama_get_logits_ith(ctx, seq*n_ctx);

                llama_token * tokens_data = tokens.data() + start + seq*n_ctx + first;

                process_logits(n_vocab, all_logits + first*n_vocab,
                        tokens_data, n_ctx - 1 - first,
                        workers, nll, nll2,
                        logit_history.data() + start + seq*n_ctx + first,
                        prob_history.data()  + start + seq*n_ctx + first);

                count += n_ctx - first - 1;

                LOG("[%d]%.4lf,", i + seq + 1, std::exp(nll / count));
            }
            fflush(stdout);

            logits.clear();
        }
    }

    if (tail_tokens > 0) {
        const int tail_start = n_chunk * n_ctx;
        LOG_INF("%s: processing final partial prune-calibration tail: %d tokens\n", __func__, tail_tokens);

        llama_memory_clear(llama_get_memory(ctx), true);
        if (mtp_spec) common_speculative_reset(mtp_spec);

        for (int offset = 0; offset < tail_tokens; offset += n_batch) {
            const int batch_size = std::min(n_batch, tail_tokens - offset);
            common_batch_clear(batch);

            const int first_index = tail_start + offset;
            const auto token_org = tokens[first_index];
            if (add_bos && offset == 0) tokens[first_index] = llama_vocab_bos(vocab);
            for (int k = 0; k < batch_size; ++k) {
                common_batch_add(batch, tokens[first_index + k], offset + k, { 0 }, true);
            }
            tokens[first_index] = token_org;

            if (llama_decode(ctx, batch)) {
                LOG_ERR("%s: failed to eval final partial prune-calibration tail\n", __func__);
                llama_batch_free(batch);
                return false;
            }
            if (mtp_spec && !common_speculative_process(mtp_spec, batch)) {
                LOG_ERR("%s: failed to execute MTP graph for final partial prune-calibration tail\n", __func__);
                llama_batch_free(batch);
                return false;
            }
        }
        // Ensure callback-side Gram/expert accumulators have consumed the last
        // partial decode before the caller serializes the imatrix.
        llama_synchronize(ctx);
    }

    LOG("\n");

    if (params.compute_ppl) {
        nll2 /= count;
        nll /= count;
        const double ppl = exp(nll);
        nll2 -= nll * nll;
        if (nll2 > 0) {
            nll2 = sqrt(nll2/(count-1));
            LOG("Final estimate: PPL = %.4lf +/- %.5lf\n", ppl, nll2*ppl);
        } else {
            LOG("Unexpected negative standard deviation of log(prob)\n");
        }
    }

    llama_batch_free(batch);

    return true;
}

static bool show_statistics(const common_params & params) {
    std::vector<tensor_statistics> ts;
    if (params.in_files.empty() || params.in_files.size() > 1) {
        LOG_ERR("\nError: a single imatrix file is required to compute tensor statistics\n\n");
        return false;
    }
    if (g_collector.load_imatrix(params.in_files[0].c_str())) {
        for (const auto & [name, stats] :g_collector.get_mstats()) {
            compute_statistics(ts, name, stats);
        }
    } else {
        LOG_ERR("\nError: %s is not a valid imatrix file\n\n", params.in_files[0].c_str());
        return false;
    }
    if (!ts.empty()) {
        compute_cossim(ts);
    } else {
        LOG_ERR("Error: cannot compute statistics for %s\n\n", params.in_files[0].c_str());
        return false;
    }

    struct tensor_comparer {
        bool operator()(const tensor_statistics & a, const tensor_statistics & b) const {
            std::string layer, name_a, name_b;
            ;
            process_tensor_name(a.tensor, layer, name_a);
            process_tensor_name(b.tensor, layer, name_b);
            return name_a < name_b || (name_a == name_b && a.total_sqract > b.total_sqract);
        }
    };
    std::sort(ts.begin(), ts.end(), tensor_comparer());

    struct weighted_stats {
        float weighted_bias   = 0.0f;
        float weighted_zd     = 0.0f;
        float weighted_cossim = 0.0f;
        int   total_elements  = 0;
    };
    std::map<int, weighted_stats> ws;

    LOG_INF("\nComputing statistics for %s (%d tensors)\n", params.in_files[0].c_str(), static_cast<int>(ts.size()));
    LOG_INF("\n%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", " Layer", "       Tensor", "          Σ(Act²)",
            "  Min", "            Max", "           μ", "   σ", " % Active", "N", "   Entropy", "E (norm)", "ZD",
            "  CosSim");
    LOG_INF(
        "=============================================================================================================="
        "===========================================================\n");
    for (const auto & tstat : ts) {
        std::string layer, name;
        process_tensor_name(tstat.tensor, layer, name);

        int blk;
        try {
            blk = std::stoi(layer);
        } catch (const std::exception & e) {
            blk = -1;  // not a block layer
        }

        const float entropy_norm = (tstat.elements > 0) ? 100.0f * (tstat.entropy / std::log2(tstat.elements)) : 0.0f;

        LOG_INF("%5s\t%-20s\t%10.2f\t%8.4f\t%11.4f\t%6.2f\t%6.2f\t%8.2f%%\t%6d\t%10.4f\t%6.2f%%\t%10.2f%%\t%8.4f\n",
                layer.c_str(), name.c_str(), tstat.total_sqract, tstat.min_sqract, tstat.max_sqract, tstat.mean_sqract,
                tstat.stddev, tstat.active * 100.0f, tstat.elements, tstat.entropy,
                entropy_norm, 100.0f * tstat.zd, tstat.cossim);

        const float weighted_bias   = tstat.elements * tstat.total_sqract;
        const float weighted_zd     = tstat.elements * tstat.zd;
        const float weighted_cossim = tstat.elements * tstat.cossim;

        if (ws.find(blk) != ws.end()) {
            ws[blk].weighted_bias += weighted_bias;
            ws[blk].weighted_zd += weighted_zd;
            ws[blk].weighted_cossim += weighted_cossim;
            ws[blk].total_elements += tstat.elements;
        } else {
            weighted_stats temp_ws;
            temp_ws.weighted_bias   = weighted_bias;
            temp_ws.weighted_zd     = weighted_zd;
            temp_ws.weighted_cossim = weighted_cossim;
            temp_ws.total_elements  = tstat.elements;
            ws[blk]                 = temp_ws;
        }
    }

    const int layers = std::count_if(ws.begin(), ws.end(), [](const auto & kv) { return kv.first >= 0; });
    LOG_INF("\nComputing weighted average statistics per layer (%d layers)\n", layers);
    LOG_INF("\n%s\t%s\t%s\t%s\n", "  Layer", "     μΣ(Act²)", "      μZD", "μCosSim");
    LOG_INF("================================================\n");
    for (const auto & [first, second] : ws) {
        const auto & layer = first;
        const auto & stats = second;

        if (stats.total_elements == 0) {
            continue;
        }

        if (layer >= 0) {
            const float bias   = stats.weighted_bias / stats.total_elements;
            const float zd     = stats.weighted_zd / stats.total_elements;
            const float cossim = stats.weighted_cossim / stats.total_elements;

            LOG_INF("%5d\t%14.2f\t%10.4f%%\t%6.4f\n", layer, bias, 100.0f * zd, cossim);
        }
    }
    LOG_INF("\n");

    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;

    params.out_file = "imatrix.gguf";

    params.n_ctx = 512;
    params.escape = false;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_IMATRIX, print_usage)) {
        return 1;
    }

    if (params.prune_mtp) {
        params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
    }

    if (!params.prompt.empty() && !params.imatrix_request_token_trie.empty()) {
        LOG_ERR("Error: --request-token-trie cannot be combined with -f/--file or -p/--prompt.\n");
        return 1;
    }

    std::unique_ptr<request_token_trie> request_trie;
    if (!params.imatrix_request_token_trie.empty()) {
        try {
            request_trie = std::make_unique<request_token_trie>(
                    load_request_token_trie(params.imatrix_request_token_trie));
        } catch (const std::exception & e) {
            LOG_ERR("%s: %s\n", __func__, e.what());
            return 1;
        }
    }

    // set_params before show_statistics so load_imatrix has valid n_ctx/n_parallel
    g_collector.set_params(params);

    if (params.show_statistics) {
        if (!show_statistics(params)) {
            return 1;
        }
        return 0;
    }

    if (request_trie) {
        constexpr int32_t request_trie_ctx_cap = 262144;
        if (request_trie->max_leaf_tokens > (size_t) request_trie_ctx_cap) {
            LOG_ERR("Error: request-token trie longest leaf is %zu tokens, exceeding the fixed GPU context cap %d.\n",
                    request_trie->max_leaf_tokens, request_trie_ctx_cap);
            return 1;
        }
        if (params.n_ctx > request_trie_ctx_cap) {
            LOG_WRN("request-token trie calibration caps GPU context at %d tokens (requested %d)\n",
                    request_trie_ctx_cap, params.n_ctx);
        }
        params.n_ctx = std::min<int32_t>(
                request_trie_ctx_cap,
                std::max<int32_t>(params.n_ctx, (int32_t) request_trie->max_leaf_tokens));
        // Trie branches are serialized through one live sequence. Branch
        // points are checkpointed to host memory and restored into seq 0, so
        // GPU context usage depends only on the longest path, never leaf count.
        params.n_parallel = 1;
        params.kv_unified = false;
    }

    const int32_t n_ctx = params.n_ctx;

    if (n_ctx <= 0) {
        LOG_ERR("%s: imatrix tool requires '--ctx-size' > 0\n", __func__);
        return 1;
    }

    if (!request_trie) {
        const int32_t n_seq = std::max(1, params.n_batch / n_ctx);
        const int32_t n_kv = n_seq * n_ctx;

        params.n_parallel = n_seq;
        params.n_ctx      = n_kv;

        params.n_batch = std::min(params.n_batch, n_kv);
    } else {
        params.n_batch = std::min(params.n_batch, n_ctx);
    }

    g_collector.set_params(params);

    for (const auto & in_file : params.in_files) {
        LOG_INF("%s : loading imatrix from '%s'\n", __func__, in_file.c_str());
        if (!g_collector.load_imatrix(in_file.c_str())) {
            LOG_ERR("%s : failed to load %s\n", __func__, in_file.c_str());
            return 1;
        }
    }

    if (params.prompt.empty() && params.imatrix_request_token_trie.empty()) {
        LOG_INF("No prompt provided; combining precomputed matrices only.\n");

        if (params.in_files.empty()) {
            LOG_ERR("Error: No prompt provided and no precomputed matrices (--in-file) to combine.\n");
            return 1;
        }

        if (params.in_files.size() == 1) {
            LOG_INF("%s : saving imatrix to '%s'\n", __func__, params.out_file.c_str());
        } else if (params.in_files.size() > 1) {
            LOG_INF("%s : saving combined imatrix to '%s'\n", __func__, params.out_file.c_str());
        }

        g_collector.save_imatrix();

        return 0;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    // pass the callback to the backend scheduler
    // it will be executed for each node during the graph computation
    params.cb_eval = ik_collect_imatrix;
    params.cb_eval_user_data = NULL;
    params.warmup = false;

    // init
    auto llama_init = common_init_from_params(params);

    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s : failed to init\n", __func__);
        return 1;
    }

    common_speculative_init_result_ptr mtp_init;
    common_speculative_ptr mtp_spec;
    llama_context * ctx_mtp = nullptr;
    if (params.prune_mtp) {
        mtp_init = common_speculative_init_from_params(params, model, ctx);
        ctx_mtp = mtp_init ? mtp_init->context() : nullptr;
        if (!ctx_mtp) {
            LOG_ERR("%s: failed to create MTP calibration context\n", __func__);
            return 1;
        }
        params.speculative.draft.ctx_tgt = ctx;
        params.speculative.draft.ctx_dft = ctx_mtp;
        mtp_spec.reset(common_speculative_init(params.speculative, params.n_parallel));
        if (!mtp_spec) {
            LOG_ERR("%s: failed to initialize MTP calibration driver\n", __func__);
            return 1;
        }
        LOG_INF("%s: MTP calibration enabled; blk.%d will execute through the stock draft-MTP graph\n",
                __func__, llama_model_n_layer(model));
    }

    const int n_ctx_train = llama_model_n_ctx_train(model);
    if (params.n_ctx > n_ctx_train) {
        LOG_WRN("%s: model was trained on only %d context tokens (%d specified)\n",
                __func__, n_ctx_train, params.n_ctx);
    }

    // print system information
    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
    }

    if (request_trie) {
        const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        for (size_t id = 1; id < request_trie->nodes.size(); ++id) {
            if (!request_trie->nodes[id].live) continue;
            const llama_token token = request_trie->nodes[id].token;
            if (token < 0 || token >= n_vocab) {
                LOG_ERR("%s: request-token trie token %d is outside model vocabulary [0,%d)\n",
                        __func__, token, n_vocab);
                return 1;
            }
        }
        if (!compute_imatrix_request_trie(ctx, ctx_mtp, mtp_spec.get(), params, *request_trie, n_ctx)) {
            return 1;
        }
    } else if (!compute_imatrix(ctx, mtp_spec.get(), params, n_ctx)) {
        return 1;
    }

    g_collector.finalize_enp_samples();
    g_collector.save_imatrix();

    LOG("\n");
    llama_perf_context_print(ctx);

    llama_backend_free();

    return 0;
}
