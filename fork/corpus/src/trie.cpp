#include <tq/corpus/trie.h>

#include <algorithm>
#include <array>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace tq::corpus {
namespace {

constexpr std::size_t header_size = 24;

std::uint32_t read_u32(const std::uint8_t * bytes) {
    std::uint32_t value = 0;
    for (unsigned i = 0; i < 4; ++i) {
        value |= std::uint32_t(bytes[i]) << (8 * i);
    }
    return value;
}

std::uint64_t read_u64(const std::uint8_t * bytes) {
    std::uint64_t value = 0;
    for (unsigned i = 0; i < 8; ++i) {
        value |= std::uint64_t(bytes[i]) << (8 * i);
    }
    return value;
}

void write_u32(std::uint8_t * bytes, std::uint32_t value) {
    for (unsigned i = 0; i < 4; ++i) {
        bytes[i] = static_cast<std::uint8_t>(value >> (8 * i));
    }
}

void write_u64(std::uint8_t * bytes, std::uint64_t value) {
    for (unsigned i = 0; i < 8; ++i) {
        bytes[i] = static_cast<std::uint8_t>(value >> (8 * i));
    }
}

template <typename T>
void reserve_size(std::vector<T> & values, std::size_t needed) {
    if (needed <= values.capacity()) {
        return;
    }
    const auto capacity = values.capacity();
    const auto growth = std::min(capacity / 2 + 1, values.max_size() - capacity);
    values.reserve(std::max(needed, capacity + growth));
}

class record_reader {
public:
    record_reader(const std::filesystem::path & path, const char * magic,
                  std::uint32_t stride)
        : input_(path, std::ios::binary), path_(path.string()) {
        if (!input_) {
            throw std::runtime_error("cannot open corpus file: " + path_);
        }
        const auto size = std::filesystem::file_size(path);
        if (size < header_size || (size - header_size) % stride != 0) {
            throw std::runtime_error("truncated corpus file: " + path_);
        }
        std::array<std::uint8_t, header_size> header{};
        read(header.data(), header.size());
        if (!std::equal(header.begin(), header.begin() + 8, magic)) {
            throw std::runtime_error("invalid corpus magic: " + path_);
        }
        if (read_u32(header.data() + 8) != 1 ||
            read_u32(header.data() + 12) != stride) {
            throw std::runtime_error("unsupported corpus format: " + path_);
        }
        count_ = (size - header_size) / stride;
    }

    std::uintmax_t count() const noexcept {
        return count_;
    }

    void read(std::uint8_t * bytes, std::size_t size) {
        if (!input_.read(reinterpret_cast<char *>(bytes),
                         static_cast<std::streamsize>(size))) {
            throw std::runtime_error("cannot read complete corpus record: " + path_);
        }
    }

    void finish() {
        if (input_.peek() != std::char_traits<char>::eof() || input_.bad()) {
            throw std::runtime_error("corpus file changed while reading: " + path_);
        }
    }

private:
    std::ifstream input_;
    std::string path_;
    std::uintmax_t count_ = 0;
};

void write_header(std::ofstream & output, const char * magic,
                  std::uint32_t stride) {
    std::array<std::uint8_t, header_size> bytes{};
    std::copy_n(magic, 8, bytes.begin());
    write_u32(bytes.data() + 8, 1);
    write_u32(bytes.data() + 12, stride);
    output.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
}

std::ofstream open_output(const std::filesystem::path & path) {
    std::ofstream output;
    output.exceptions(std::ios::badbit | std::ios::failbit);
    output.open(path, std::ios::binary);
    return output;
}

} // namespace

token_trie::token_trie(std::int32_t vocabulary_size)
    : vocabulary_size_(vocabulary_size), children_(1), depths_(1, 0) {
    if (vocabulary_size <= 0) {
        throw std::invalid_argument("vocabulary size must be positive");
    }
}

void token_trie::check_id(std::uint64_t id, bool allow_root) const {
    if ((!allow_root && id == 0) || id > nodes_.size()) {
        throw std::out_of_range("corpus node id is out of range");
    }
}

trie_node token_trie::node(std::uint64_t id) const {
    check_id(id, false);
    return nodes_[static_cast<std::size_t>(id - 1)];
}

std::vector<std::uint64_t> token_trie::children(std::uint64_t id) const {
    check_id(id, true);
    return children_[static_cast<std::size_t>(id)];
}

std::vector<std::uint64_t> token_trie::request_leaves() const {
    return leaves_;
}

std::vector<std::int32_t> token_trie::path_to(std::uint64_t id) const {
    check_id(id, true);
    std::vector<std::int32_t> result;
    result.reserve(static_cast<std::size_t>(depths_[static_cast<std::size_t>(id)]));
    while (id != 0) {
        const auto item = nodes_[static_cast<std::size_t>(id - 1)];
        result.push_back(item.token);
        id = item.parent;
    }
    std::reverse(result.begin(), result.end());
    return result;
}

std::uint64_t token_trie::node_count() const noexcept {
    return nodes_.size();
}

std::uint64_t token_trie::request_count() const noexcept {
    return leaves_.size();
}

std::uint64_t token_trie::leaf_positions() const noexcept {
    return leaf_positions_;
}

std::uint64_t token_trie::append(const std::vector<std::int32_t> & tokens) {
    if (tokens.empty()) {
        throw std::invalid_argument("a corpus request must contain a token");
    }
    if (tokens.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::length_error("a corpus request exceeds the format depth limit");
    }
    for (const auto token : tokens) {
        if (token < 0 || token >= vocabulary_size_) {
            throw std::invalid_argument("corpus token is outside the vocabulary");
        }
    }
    if (tokens.size() > std::numeric_limits<std::uint64_t>::max() - leaf_positions_) {
        throw std::overflow_error("corpus token count overflow");
    }

    std::uint64_t parent = 0;
    std::size_t prefix = 0;
    for (; prefix < tokens.size(); ++prefix) {
        const auto & edges = children_[static_cast<std::size_t>(parent)];
        const auto found = std::find_if(edges.begin(), edges.end(), [&](std::uint64_t id) {
            return nodes_[static_cast<std::size_t>(id - 1)].token == tokens[prefix];
        });
        if (found == edges.end()) {
            break;
        }
        parent = *found;
    }

    const auto added = tokens.size() - prefix;
    const auto max_nodes = std::min({
        nodes_.max_size(), children_.max_size() - 1, depths_.max_size() - 1});
    if (added > max_nodes - nodes_.size() || leaves_.size() == leaves_.max_size()) {
        throw std::length_error("corpus capacity limit exceeded");
    }
    const auto first_id = static_cast<std::uint64_t>(nodes_.size()) + 1;

    // Prepare all storage before changing the tree.
    std::vector<std::vector<std::uint64_t>> new_children(added);
    for (std::size_t i = 0; i + 1 < added; ++i) {
        new_children[i].push_back(first_id + i + 1);
    }
    reserve_size(nodes_, nodes_.size() + added);
    reserve_size(children_, children_.size() + added);
    reserve_size(depths_, depths_.size() + added);
    reserve_size(leaves_, leaves_.size() + 1);
    if (added != 0) {
        auto & edges = children_[static_cast<std::size_t>(parent)];
        reserve_size(edges, edges.size() + 1);
        edges.push_back(first_id);
    }

    for (std::size_t i = 0; i < added; ++i) {
        const auto id = first_id + i;
        nodes_.push_back({parent, tokens[prefix + i]});
        children_.push_back(std::move(new_children[i]));
        depths_.push_back(prefix + i + 1);
        parent = id;
    }
    leaves_.push_back(parent);
    leaf_positions_ += tokens.size();
    return parent;
}

token_trie token_trie::load(const std::filesystem::path & directory,
                           std::int32_t vocabulary_size) {
    token_trie result(vocabulary_size);
    record_reader nodes(directory / "nodes-000000.bin", "CLTNOD01", 16);
    const auto max_nodes = std::min({
        result.nodes_.max_size(), result.children_.max_size() - 1,
        result.depths_.max_size() - 1});
    if (nodes.count() > max_nodes) {
        throw std::length_error("corpus node capacity limit exceeded");
    }
    const auto count = static_cast<std::size_t>(nodes.count());
    result.nodes_.reserve(count);
    result.children_.resize(count + 1);
    result.depths_.resize(count + 1, 0);
    for (std::size_t i = 0; i < count; ++i) {
        std::array<std::uint8_t, 16> bytes{};
        nodes.read(bytes.data(), bytes.size());
        const auto parent = read_u64(bytes.data());
        const auto token = read_u32(bytes.data() + 8);
        if (parent >= i + 1) {
            throw std::runtime_error("corpus parent must precede its child");
        }
        if (token >= static_cast<std::uint32_t>(vocabulary_size)) {
            throw std::runtime_error("corpus token is outside the vocabulary");
        }
        const auto depth = result.depths_[static_cast<std::size_t>(parent)] + 1;
        if (depth > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("corpus node exceeds the format depth limit");
        }
        result.nodes_.push_back({parent, static_cast<std::int32_t>(token)});
        result.children_[static_cast<std::size_t>(parent)].push_back(i + 1);
        result.depths_[i + 1] = depth;
    }
    nodes.finish();

    record_reader requests(directory / "requests-000000.bin", "CLTREQ01", 24);
    if (requests.count() > result.leaves_.max_size()) {
        throw std::length_error("corpus request capacity limit exceeded");
    }
    result.leaves_.reserve(static_cast<std::size_t>(requests.count()));
    for (std::uintmax_t i = 0; i < requests.count(); ++i) {
        std::array<std::uint8_t, 24> bytes{};
        requests.read(bytes.data(), bytes.size());
        const auto leaf = read_u64(bytes.data());
        const auto depth = read_u32(bytes.data() + 8);
        if (leaf == 0 || leaf > result.nodes_.size()) {
            throw std::runtime_error("corpus request references a missing leaf");
        }
        if (result.depths_[static_cast<std::size_t>(leaf)] != depth) {
            throw std::runtime_error("corpus request depth does not match its leaf");
        }
        if (depth > std::numeric_limits<std::uint64_t>::max() - result.leaf_positions_) {
            throw std::overflow_error("corpus token count overflow");
        }
        result.leaves_.push_back(leaf);
        result.leaf_positions_ += depth;
    }
    requests.finish();
    return result;
}

void token_trie::save(const std::filesystem::path & directory) const {
    if (!std::filesystem::create_directory(directory)) {
        throw std::runtime_error("corpus target already exists: " + directory.string());
    }
    const auto nodes_path = directory / "nodes-000000.bin";
    const auto requests_path = directory / "requests-000000.bin";
    try {
        {
            auto output = open_output(nodes_path);
            write_header(output, "CLTNOD01", 16);
            for (const auto item : nodes_) {
                std::array<std::uint8_t, 16> bytes{};
                write_u64(bytes.data(), item.parent);
                write_u32(bytes.data() + 8, static_cast<std::uint32_t>(item.token));
                output.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
            }
            output.close();
        }
        {
            auto output = open_output(requests_path);
            write_header(output, "CLTREQ01", 24);
            for (const auto leaf : leaves_) {
                std::array<std::uint8_t, 24> bytes{};
                write_u64(bytes.data(), leaf);
                write_u32(bytes.data() + 8,
                          static_cast<std::uint32_t>(depths_[static_cast<std::size_t>(leaf)]));
                output.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
            }
            output.close();
        }
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(nodes_path, ignored);
        std::filesystem::remove(requests_path, ignored);
        std::filesystem::remove(directory, ignored);
        throw;
    }
}

} // namespace tq::corpus
