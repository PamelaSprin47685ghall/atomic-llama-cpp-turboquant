#include <tq/corpus/trie.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using tq::corpus::token_trie;
using bytes = std::vector<std::uint8_t>;

void check(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void must_fail(const std::function<void()> & action, const char * message) {
    try {
        action();
    } catch (const std::exception &) {
        return;
    }
    throw std::runtime_error(message);
}

bytes read_file(const fs::path & path) {
    std::ifstream input(path, std::ios::binary);
    input.exceptions(std::ios::badbit);
    check(input.is_open(), "cannot open test file");
    return bytes(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

void write_file(const fs::path & path, const bytes & data) {
    std::ofstream output;
    output.exceptions(std::ios::badbit | std::ios::failbit);
    output.open(path, std::ios::binary);
    output.write(reinterpret_cast<const char *>(data.data()), data.size());
    output.close();
}

struct test_directory {
    fs::path path;

    test_directory() {
        const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
        path = fs::temp_directory_path() / ("tq-corpus-test-" + std::to_string(tick));
        check(fs::create_directory(path), "test directory already exists");
    }

    ~test_directory() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

void run_tests() {
    must_fail([] { token_trie invalid(0); }, "zero vocabulary was accepted");
    must_fail([] { token_trie invalid(-1); }, "negative vocabulary was accepted");

    token_trie trie(8);
    check(trie.node_count() == 0 && trie.request_count() == 0, "new trie is not empty");
    check(trie.path_to(0).empty() && trie.children(0).empty(), "root has content");
    check(trie.append({1, 2, 3}) == 3, "first leaf is wrong");
    check(trie.append({1, 2, 4}) == 4, "shared prefix was lost");
    check(trie.append({5}) == 5, "second root edge is wrong");
    check(trie.append({1, 2, 3}) == 3, "duplicate request added nodes");
    check(trie.node_count() == 5, "node count is wrong");
    check(trie.request_count() == 4 && trie.leaf_positions() == 10,
          "request counts are wrong");
    check(trie.children(0) == std::vector<std::uint64_t>({1, 5}), "root edges are wrong");
    check(trie.path_to(4) == std::vector<std::int32_t>({1, 2, 4}), "path is wrong");

    auto returned_node = trie.node(1);
    returned_node.token = 7;
    auto returned_edges = trie.children(0);
    returned_edges.clear();
    auto returned_requests = trie.request_leaves();
    returned_requests[0] = 5;
    auto returned_path = trie.path_to(3);
    returned_path[0] = 7;
    check(returned_node.token == 7 && returned_edges.empty(), "test mutation failed");
    check(trie.node(1).token == 1 && trie.children(0).size() == 2,
          "node or edge value shared storage");
    check(trie.request_leaves()[0] == 3 && trie.path_to(3)[0] == 1,
          "request or path value shared storage");
    const auto detached = [] {
        token_trie local(8);
        local.append({2, 3});
        return local.path_to(2);
    }();
    check(detached == std::vector<std::int32_t>({2, 3}), "path needs a live owner");

    token_trie other(8);
    other.append({7, 6});
    check(trie.node_count() == 5, "separate instances shared state");
    token_trie copy = trie;
    copy.append({6});
    check(trie.node_count() == 5 && copy.node_count() == 6, "copy shared state");

    token_trie input_owner(8);
    std::vector<std::int32_t> input{3, 4, 5};
    const auto input_leaf = input_owner.append(input);
    input[1] = 7;
    input.clear();
    check(input_owner.path_to(input_leaf) == std::vector<std::int32_t>({3, 4, 5}),
          "append retained mutable input storage");
    check(input_owner.append({3, 4}) == 2, "prefix request endpoint is wrong");
    check(input_owner.node_count() == 3 && input_owner.leaf_positions() == 5 &&
          input_owner.request_leaves() == std::vector<std::uint64_t>({3, 2}),
          "an internal node cannot be a request endpoint");

    must_fail([&] { trie.append({1, 2, 8}); }, "high token was accepted");
    must_fail([&] { trie.append({1, -1}); }, "negative token was accepted");
    must_fail([&] { trie.append({}); }, "empty request was accepted");
    check(trie.node_count() == 5 && trie.request_count() == 4, "failed append changed data");
    must_fail([&] { trie.node(0); }, "root returned a token");
    must_fail([&] { trie.node(6); }, "missing node was accepted");
    must_fail([&] { trie.children(6); }, "missing children were accepted");
    must_fail([&] { trie.path_to(6); }, "missing path was accepted");

    test_directory temporary;
    input_owner.save(temporary.path / "prefix-request");
    const auto prefix_loaded = token_trie::load(temporary.path / "prefix-request", 8);
    check(prefix_loaded.request_leaves() == std::vector<std::uint64_t>({3, 2}) &&
          prefix_loaded.children(2) == std::vector<std::uint64_t>({3}),
          "internal request endpoint was lost in serialization");
    const auto good = temporary.path / "good";
    trie.save(good);
    auto loaded = token_trie::load(good, 8);
    check(loaded.node_count() == trie.node_count(), "node round trip failed");
    check(loaded.request_leaves() == trie.request_leaves(), "request round trip failed");
    check(loaded.leaf_positions() == 10, "position round trip failed");
    for (std::uint64_t id = 0; id <= trie.node_count(); ++id) {
        check(loaded.path_to(id) == trie.path_to(id), "path round trip failed");
        check(loaded.children(id) == trie.children(id), "children round trip failed");
    }
    loaded.append({1, 2, 6});
    check(loaded.node_count() == 6 && trie.node_count() == 5, "loaded trie cannot append");

    const auto node_data = read_file(good / "nodes-000000.bin");
    const auto request_data = read_file(good / "requests-000000.bin");
    check(node_data.size() == 24 + 5 * 16 && request_data.size() == 24 + 4 * 24,
          "record sizes are wrong");
    check(node_data[8] == 1 && node_data[9] == 0 && node_data[12] == 16,
          "node header is not little endian");
    check(node_data[24 + 16] == 1 && node_data[24 + 16 + 1] == 0 &&
          node_data[24 + 8] == 1, "node record is not little endian");
    check(request_data[24] == 3 && request_data[25] == 0 &&
          request_data[24 + 8] == 3, "request record is not little endian");
    token_trie wide(70000);
    wide.append({0x1234, 0x10002});
    const auto wide_path = temporary.path / "wide";
    wide.save(wide_path);
    const auto wide_bytes = read_file(wide_path / "nodes-000000.bin");
    check(wide_bytes[32] == 0x34 && wide_bytes[33] == 0x12 &&
          wide_bytes[48] == 2 && wide_bytes[50] == 1,
          "multibyte token encoding is wrong");
    check(token_trie::load(wide_path, 70000).path_to(2) ==
          std::vector<std::int32_t>({0x1234, 0x10002}), "multibyte token read failed");

    must_fail([&] { trie.save(good); }, "save overwrote an existing directory");
    check(read_file(good / "nodes-000000.bin") == node_data, "save changed existing nodes");
    check(read_file(good / "requests-000000.bin") == request_data,
          "save changed existing requests");
    const auto empty_directory = temporary.path / "already-empty";
    fs::create_directory(empty_directory);
    must_fail([&] { trie.save(empty_directory); }, "save reused an existing directory");
    const auto existing_file = temporary.path / "already-file";
    write_file(existing_file, bytes{1, 2});
    must_fail([&] { trie.save(existing_file); }, "save replaced a file");
    check(read_file(existing_file) == bytes({1, 2}), "save changed a file");

    std::size_t fixture_id = 0;
    const auto rejects = [&](bytes nodes, bytes requests, const char * message) {
        const auto fixture = temporary.path / ("bad-" + std::to_string(fixture_id++));
        fs::create_directory(fixture);
        write_file(fixture / "nodes-000000.bin", nodes);
        write_file(fixture / "requests-000000.bin", requests);
        must_fail([&] { token_trie::load(fixture, 8); }, message);
    };
    auto changed = node_data;
    changed[0] = 'X';
    rejects(changed, request_data, "bad node magic was accepted");
    changed = node_data;
    changed[8] = 2;
    rejects(changed, request_data, "bad version was accepted");
    changed = node_data;
    changed[12] = 24;
    rejects(changed, request_data, "bad record size was accepted");
    changed = node_data;
    changed.resize(23);
    rejects(changed, request_data, "short node header was accepted");
    changed = node_data;
    changed.pop_back();
    rejects(changed, request_data, "short node record was accepted");
    changed = node_data;
    changed[24] = 1;
    rejects(changed, request_data, "self cycle was accepted");
    changed = node_data;
    changed[24] = 2;
    rejects(changed, request_data, "forward parent was accepted");
    changed = node_data;
    changed[24 + 8] = 8;
    rejects(changed, request_data, "out of vocabulary token was accepted");
    changed = node_data;
    for (int i = 0; i < 4; ++i) {
        changed[24 + 8 + i] = 255;
    }
    rejects(changed, request_data, "negative serialized token was accepted");

    changed = request_data;
    changed[0] = 'X';
    rejects(node_data, changed, "bad request magic was accepted");
    changed = request_data;
    changed[24] = 0;
    rejects(node_data, changed, "root request was accepted");
    changed = request_data;
    changed[24] = 6;
    rejects(node_data, changed, "missing request leaf was accepted");
    changed = request_data;
    changed[24 + 8] = 2;
    rejects(node_data, changed, "incorrect request depth was accepted");
    changed = request_data;
    changed.pop_back();
    rejects(node_data, changed, "short request record was accepted");
    changed = request_data;
    changed.resize(23);
    rejects(node_data, changed, "short request header was accepted");

    must_fail([&] { token_trie::load(good, 5); }, "smaller vocabulary was accepted");
    must_fail([&] { token_trie::load(temporary.path / "missing", 8); },
              "missing files were accepted");
    token_trie empty(8);
    empty.save(temporary.path / "empty");
    const auto empty_loaded = token_trie::load(temporary.path / "empty", 8);
    check(empty_loaded.node_count() == 0 && empty_loaded.request_count() == 0,
          "empty trie round trip failed");
}

int main() {
    try {
        run_tests();
        std::cout << "corpus tests passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
