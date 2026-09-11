#include "server-corpus-dump.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace {

static_assert(sizeof(llama_token) == sizeof(int32_t), "request corpus format requires 32-bit llama_token");

constexpr uint32_t FORMAT_VERSION = 1;
constexpr uint32_t HEADER_SIZE = 24;
constexpr uint32_t NODE_RECORD_SIZE = 16;
constexpr uint32_t REQUEST_RECORD_SIZE = 24;

constexpr char NODE_MAGIC[8] = {'C', 'L', 'T', 'N', 'O', 'D', '0', '1'};
constexpr char REQUEST_MAGIC[8] = {'C', 'L', 'T', 'R', 'E', 'Q', '0', '1'};

void write_u32(std::ostream & out, uint32_t value) {
    char buf[4];
    for (int i = 0; i < 4; ++i) {
        buf[i] = (char) ((value >> (8 * i)) & 0xffU);
    }
    out.write(buf, sizeof(buf));
}

void write_u64(std::ostream & out, uint64_t value) {
    char buf[8];
    for (int i = 0; i < 8; ++i) {
        buf[i] = (char) ((value >> (8 * i)) & 0xffU);
    }
    out.write(buf, sizeof(buf));
}

uint32_t read_u32(std::istream & in) {
    unsigned char buf[4];
    in.read((char *) buf, sizeof(buf));
    if (!in) {
        throw std::runtime_error("unexpected EOF while reading u32");
    }
    uint32_t value = 0;
    for (int i = 0; i < 4; ++i) {
        value |= (uint32_t) buf[i] << (8 * i);
    }
    return value;
}

uint64_t read_u64(std::istream & in) {
    unsigned char buf[8];
    in.read((char *) buf, sizeof(buf));
    if (!in) {
        throw std::runtime_error("unexpected EOF while reading u64");
    }
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
        value |= (uint64_t) buf[i] << (8 * i);
    }
    return value;
}

llama_token read_token(std::istream & in) {
    const uint32_t raw = read_u32(in);
    const int64_t value = raw <= (uint32_t) std::numeric_limits<int32_t>::max()
        ? (int64_t) raw
        : (int64_t) raw - (INT64_C(1) << 32);
    return (llama_token) value;
}

void write_header(
        std::ostream & out,
        const char magic[8],
        uint32_t record_size,
        uint64_t start_id) {
    out.write(magic, 8);
    write_u32(out, FORMAT_VERSION);
    write_u32(out, record_size);
    write_u64(out, start_id);
}

uint64_t read_header(
        std::istream & in,
        const char expected_magic[8],
        uint32_t expected_record_size) {
    char magic[8];
    in.read(magic, sizeof(magic));
    if (!in || std::memcmp(magic, expected_magic, sizeof(magic)) != 0) {
        throw std::runtime_error("invalid token-dump shard magic");
    }
    const uint32_t version = read_u32(in);
    const uint32_t record_size = read_u32(in);
    const uint64_t start_id = read_u64(in);
    if (version != FORMAT_VERSION) {
        throw std::runtime_error("unsupported token-dump shard version");
    }
    if (record_size != expected_record_size) {
        throw std::runtime_error("unexpected token-dump record size");
    }
    return start_id;
}

std::string make_shard_name(const char * prefix, uint64_t index) {
    std::ostringstream name;
    name << prefix << '-' << std::setw(6) << std::setfill('0') << index << ".bin";
    return name.str();
}

bool parse_shard_index(const std::string & name, const std::string & prefix, uint64_t & index) {
    const std::string start = prefix + '-';
    constexpr const char * suffix = ".bin";
    if (name.size() <= start.size() + std::strlen(suffix) || name.compare(0, start.size(), start) != 0) {
        return false;
    }
    if (name.compare(name.size() - std::strlen(suffix), std::strlen(suffix), suffix) != 0) {
        return false;
    }
    const std::string digits = name.substr(start.size(), name.size() - start.size() - std::strlen(suffix));
    if (digits.empty() || !std::all_of(digits.begin(), digits.end(), [](char c) { return c >= '0' && c <= '9'; })) {
        return false;
    }
    try {
        index = std::stoull(digits);
    } catch (...) {
        return false;
    }
    return true;
}

std::vector<std::pair<uint64_t, std::filesystem::path>> list_shards(
        const std::filesystem::path & dir,
        const std::string & prefix) {
    std::vector<std::pair<uint64_t, std::filesystem::path>> result;
    for (const auto & entry : std::filesystem::directory_iterator(dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        uint64_t index = 0;
        const std::string name = entry.path().filename().string();
        if (parse_shard_index(name, prefix, index)) {
            result.emplace_back(index, entry.path());
        }
    }
    std::sort(result.begin(), result.end(), [](const auto & a, const auto & b) {
        return a.first < b.first;
    });
    for (size_t i = 0; i < result.size(); ++i) {
        if (result[i].first != i) {
            throw std::runtime_error("token-dump shard index gap for " + prefix);
        }
    }
    return result;
}

uint64_t payload_records(const std::filesystem::path & path, uint32_t record_size, bool repair_tail) {
    const uint64_t size = std::filesystem::file_size(path);
    if (size < HEADER_SIZE) {
        if (repair_tail) {
            std::filesystem::remove(path);
            return UINT64_MAX;
        }
        throw std::runtime_error("token-dump shard is shorter than its header: " + path.string());
    }
    const uint64_t payload = size - HEADER_SIZE;
    if (payload % record_size != 0) {
        if (repair_tail) {
            const uint64_t repaired_size = HEADER_SIZE + (payload / record_size) * record_size;
            std::filesystem::resize_file(path, repaired_size);
            return payload / record_size;
        }
        throw std::runtime_error("token-dump shard has a partial trailing record: " + path.string());
    }
    return payload / record_size;
}

uint64_t now_ms() {
    using namespace std::chrono;
    return (uint64_t) duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

} // namespace

size_t server_token_dump::edge_hash::operator()(const edge_key & key) const {
    uint64_t x = key.parent;
    x ^= (uint64_t) (uint32_t) key.token + UINT64_C(0x9e3779b97f4a7c15) + (x << 6) + (x >> 2);
    x ^= x >> 33;
    x *= UINT64_C(0xff51afd7ed558ccd);
    x ^= x >> 33;
    return (size_t) x;
}

server_token_dump::server_token_dump(const std::string & path, uint64_t records_per_shard)
        : path_(path), records_per_shard_(records_per_shard) {
    if (path_.empty()) {
        throw std::invalid_argument("request token dump path must not be empty");
    }
    if (records_per_shard_ == 0) {
        throw std::invalid_argument("request token dump shard size must be greater than zero");
    }

    std::filesystem::create_directories(path_);
    if (!std::filesystem::is_directory(path_)) {
        throw std::runtime_error("request token dump path is not a directory: " + path_);
    }

    // Node 0 is implicit and never persisted.
    nodes_.push_back({0, 0});
    load_existing();
    write_manifest_if_missing();
}

server_token_dump::~server_token_dump() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (node_stream_.is_open()) {
        node_stream_.flush();
    }
    if (request_stream_.is_open()) {
        request_stream_.flush();
    }
}

void server_token_dump::load_existing() {
    const std::filesystem::path dir(path_);

    const auto node_shards = list_shards(dir, "nodes");
    for (size_t shard_pos = 0; shard_pos < node_shards.size(); ++shard_pos) {
        const auto & [shard_index, shard_path] = node_shards[shard_pos];
        const bool is_last = shard_pos + 1 == node_shards.size();
        const uint64_t n_records = payload_records(shard_path, NODE_RECORD_SIZE, is_last);
        if (n_records == UINT64_MAX) {
            break;
        }
        std::ifstream in(shard_path, std::ios::binary);
        if (!in) {
            throw std::runtime_error("failed to open token-dump node shard: " + shard_path.string());
        }
        const uint64_t start_id = read_header(in, NODE_MAGIC, NODE_RECORD_SIZE);
        if (start_id != nodes_.size()) {
            throw std::runtime_error("token-dump node shard has a non-contiguous start id: " + shard_path.string());
        }

        for (uint64_t i = 0; i < n_records; ++i) {
            const uint64_t parent = read_u64(in);
            const llama_token token = read_token(in);
            const uint32_t reserved = read_u32(in);
            const uint64_t id = nodes_.size();
            if (reserved != 0) {
                throw std::runtime_error("token-dump node record has non-zero reserved bits");
            }
            if (parent >= id) {
                throw std::runtime_error("token-dump node parent does not precede child");
            }

            const edge_key key{parent, token};
            if (edge_index_.find(key) != edge_index_.end()) {
                throw std::runtime_error("token-dump contains duplicate trie edge");
            }
            nodes_.push_back({parent, token});
            edge_index_.emplace(key, id);
        }

        node_shard_index_ = shard_index;
        node_shard_records_ = n_records;
    }

    const auto request_shards = list_shards(dir, "requests");
    for (size_t shard_pos = 0; shard_pos < request_shards.size(); ++shard_pos) {
        const auto & [shard_index, shard_path] = request_shards[shard_pos];
        const bool is_last = shard_pos + 1 == request_shards.size();
        const uint64_t n_records = payload_records(shard_path, REQUEST_RECORD_SIZE, is_last);
        if (n_records == UINT64_MAX) {
            break;
        }
        std::ifstream in(shard_path, std::ios::binary);
        if (!in) {
            throw std::runtime_error("failed to open token-dump request shard: " + shard_path.string());
        }
        const uint64_t start_id = read_header(in, REQUEST_MAGIC, REQUEST_RECORD_SIZE);
        if (start_id != request_count_) {
            throw std::runtime_error("token-dump request shard has a non-contiguous start id: " + shard_path.string());
        }

        for (uint64_t i = 0; i < n_records; ++i) {
            const uint64_t leaf = read_u64(in);
            const uint32_t n_tokens = read_u32(in);
            const uint32_t kind = read_u32(in);
            (void) read_u64(in); // timestamp_ms

            if (leaf >= nodes_.size()) {
                throw std::runtime_error("token-dump request references an unknown leaf node");
            }
            if (n_tokens > 0 && leaf == 0) {
                throw std::runtime_error("non-empty token-dump request references the root node");
            }
            if (kind < (uint32_t) server_token_dump_kind::completion ||
                    kind > (uint32_t) server_token_dump_kind::rerank) {
                throw std::runtime_error("token-dump request has an unknown kind");
            }
            request_count_++;
        }

        request_shard_index_ = shard_index;
        request_shard_records_ = n_records;
    }
}

void server_token_dump::write_manifest_if_missing() const {
    const std::filesystem::path manifest = std::filesystem::path(path_) / "format.json";
    if (std::filesystem::exists(manifest)) {
        return;
    }

    std::ofstream out(manifest, std::ios::out | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("failed to create token-dump format manifest: " + manifest.string());
    }
    out
        << "{\n"
        << "  \"format\": \"wanxiangshu-request-token-trie\",\n"
        << "  \"version\": 1,\n"
        << "  \"endianness\": \"little\",\n"
        << "  \"root_node_id\": 0,\n"
        << "  \"token_type\": \"int32\",\n"
        << "  \"node_record\": [\"parent_id:u64\", \"token_id:i32\", \"reserved:u32\"],\n"
        << "  \"request_record\": [\"leaf_id:u64\", \"n_tokens:u32\", \"kind:u32\", \"timestamp_ms:u64\"]\n"
        << "}\n";
    if (!out) {
        throw std::runtime_error("failed to write token-dump format manifest: " + manifest.string());
    }
}

std::string server_token_dump::node_shard_path(uint64_t index) const {
    return (std::filesystem::path(path_) / make_shard_name("nodes", index)).string();
}

std::string server_token_dump::request_shard_path(uint64_t index) const {
    return (std::filesystem::path(path_) / make_shard_name("requests", index)).string();
}

void server_token_dump::ensure_node_stream() {
    if (node_stream_.is_open()) {
        return;
    }

    if (node_shard_records_ >= records_per_shard_) {
        node_shard_index_++;
        node_shard_records_ = 0;
    }

    const std::string path = node_shard_path(node_shard_index_);
    if (!std::filesystem::exists(path)) {
        std::ofstream create(path, std::ios::binary | std::ios::trunc);
        if (!create) {
            throw std::runtime_error("failed to create token-dump node shard: " + path);
        }
        write_header(create, NODE_MAGIC, NODE_RECORD_SIZE, nodes_.size());
        create.flush();
        if (!create) {
            throw std::runtime_error("failed to write token-dump node shard header: " + path);
        }
    }

    node_stream_.open(path, std::ios::binary | std::ios::app);
    if (!node_stream_) {
        throw std::runtime_error("failed to append token-dump node shard: " + path);
    }
}

void server_token_dump::ensure_request_stream() {
    if (request_stream_.is_open()) {
        return;
    }

    if (request_shard_records_ >= records_per_shard_) {
        request_shard_index_++;
        request_shard_records_ = 0;
    }

    const std::string path = request_shard_path(request_shard_index_);
    if (!std::filesystem::exists(path)) {
        std::ofstream create(path, std::ios::binary | std::ios::trunc);
        if (!create) {
            throw std::runtime_error("failed to create token-dump request shard: " + path);
        }
        write_header(create, REQUEST_MAGIC, REQUEST_RECORD_SIZE, request_count_);
        create.flush();
        if (!create) {
            throw std::runtime_error("failed to write token-dump request shard header: " + path);
        }
    }

    request_stream_.open(path, std::ios::binary | std::ios::app);
    if (!request_stream_) {
        throw std::runtime_error("failed to append token-dump request shard: " + path);
    }
}

void server_token_dump::write_node_record(uint64_t parent, llama_token token) {
    ensure_node_stream();
    write_u64(node_stream_, parent);
    write_u32(node_stream_, (uint32_t) token);
    write_u32(node_stream_, 0);
    if (!node_stream_) {
        throw std::runtime_error("failed to write token-dump node record");
    }
    node_shard_records_++;

    if (node_shard_records_ >= records_per_shard_) {
        node_stream_.flush();
        if (!node_stream_) {
            throw std::runtime_error("failed to flush token-dump node shard");
        }
        node_stream_.close();
    }
}

void server_token_dump::write_request_record(
        uint64_t leaf,
        uint32_t n_tokens,
        server_token_dump_kind kind) {
    ensure_request_stream();
    write_u64(request_stream_, leaf);
    write_u32(request_stream_, n_tokens);
    write_u32(request_stream_, (uint32_t) kind);
    write_u64(request_stream_, now_ms());
    if (!request_stream_) {
        throw std::runtime_error("failed to write token-dump request record");
    }
    request_shard_records_++;
}

bool server_token_dump::append(
        const std::vector<llama_token> & tokens,
        server_token_dump_kind kind,
        std::string * error) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (failed_) {
        if (error) {
            *error = failure_message_;
        }
        return false;
    }
    if (tokens.size() > std::numeric_limits<uint32_t>::max()) {
        if (error) {
            *error = "request contains more than UINT32_MAX tokens";
        }
        return false;
    }

    try {
        uint64_t parent = 0;
        for (llama_token token : tokens) {
            const edge_key key{parent, token};
            const auto it = edge_index_.find(key);
            if (it != edge_index_.end()) {
                parent = it->second;
                continue;
            }

            const uint64_t id = nodes_.size();
            write_node_record(parent, token);
            nodes_.push_back({parent, token});
            edge_index_.emplace(key, id);
            parent = id;
        }

        // A request record must never become durable before the nodes it
        // references. Orphan nodes after a crash are harmless; dangling leaves
        // are not.
        if (node_stream_.is_open()) {
            node_stream_.flush();
            if (!node_stream_) {
                throw std::runtime_error("failed to flush token-dump nodes before request commit");
            }
        }

        write_request_record(parent, (uint32_t) tokens.size(), kind);
        request_stream_.flush();
        if (!request_stream_) {
            throw std::runtime_error("failed to flush token-dump request record");
        }
        if (request_shard_records_ >= records_per_shard_) {
            request_stream_.close();
        }
        request_count_++;
        return true;
    } catch (const std::exception & e) {
        failed_ = true;
        failure_message_ = e.what();
        if (error) {
            *error = failure_message_;
        }
        return false;
    }
}

uint64_t server_token_dump::node_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return nodes_.size() - 1;
}

uint64_t server_token_dump::request_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return request_count_;
}
