#include "o200k-codec.h"

#include "encoding.h"

#include <algorithm>
#include <atomic>
#include <exception>
#include <thread>
#include <utility>

namespace {

struct text_range {
    std::size_t begin;
    std::size_t end;
};

bool safe_split_position(const std::string & text, std::size_t position) {
    if (position == 0 || position >= text.size() || text[position - 1] != '\n') {
        return false;
    }

    const unsigned char next = static_cast<unsigned char>(text[position]);
    return next >= 0x21 && next <= 0x7e && next != '/';
}

std::vector<text_range> chunk_ranges(
        const std::string & text,
        std::size_t worker_count) {
    if (worker_count <= 1 || text.empty()) {
        return {{0, text.size()}};
    }

    std::vector<text_range> chunks;
    chunks.reserve(worker_count);
    std::size_t begin = 0;

    for (std::size_t i = 1; i < worker_count; ++i) {
        std::size_t position = std::max(begin + 1, text.size() * i / worker_count);
        while (position < text.size() && !safe_split_position(text, position)) {
            ++position;
        }
        if (position == text.size()) {
            break;
        }

        chunks.push_back({begin, position});
        begin = position;
    }

    chunks.push_back({begin, text.size()});
    return chunks;
}

} // namespace

struct wanxiangqi_o200k_codec::impl {
    std::shared_ptr<GptEncoding> encoding = GptEncoding::get_encoding(LanguageModel::O200K_BASE);
};

wanxiangqi_o200k_codec::wanxiangqi_o200k_codec()
    : pimpl(std::make_unique<impl>()) {
}

wanxiangqi_o200k_codec::~wanxiangqi_o200k_codec() = default;

wanxiangqi_o200k_codec::wanxiangqi_o200k_codec(wanxiangqi_o200k_codec &&) noexcept = default;

wanxiangqi_o200k_codec & wanxiangqi_o200k_codec::operator=(wanxiangqi_o200k_codec &&) noexcept = default;

std::vector<int> wanxiangqi_o200k_codec::encode(const std::string & text) const {
    return pimpl->encoding->encode(text, {}, {});
}

std::vector<int> wanxiangqi_o200k_codec::encode_parallel(const std::string & text, std::size_t worker_count) const {
    if (text.empty()) {
        return {};
    }

    if (worker_count == 0) {
        const unsigned int hardware_workers = std::thread::hardware_concurrency();
        worker_count = hardware_workers == 0 ? 1 : hardware_workers;
    }
    if (worker_count <= 1) {
        return encode(text);
    }

    const auto chunks = chunk_ranges(text, worker_count * 8);
    if (chunks.size() <= 1) {
        return encode(text);
    }

    const std::size_t thread_count = std::min(worker_count, chunks.size());
    std::vector<std::unique_ptr<wanxiangqi_o200k_codec>> worker_codecs;
    worker_codecs.reserve(thread_count - 1);
    for (std::size_t i = 1; i < thread_count; ++i) {
        worker_codecs.push_back(std::make_unique<wanxiangqi_o200k_codec>());
    }

    std::vector<std::vector<int>> results(chunks.size());
    std::vector<std::exception_ptr> failures(chunks.size());
    std::atomic<std::size_t> next_chunk{0};
    std::vector<std::thread> threads;
    threads.reserve(thread_count);

    for (std::size_t worker = 0; worker < thread_count; ++worker) {
        threads.emplace_back([&, worker]() {
            const auto & codec = worker == 0 ? *this : *worker_codecs[worker - 1];
            while (true) {
                const std::size_t i = next_chunk.fetch_add(1, std::memory_order_relaxed);
                if (i >= chunks.size()) {
                    return;
                }

                try {
                    const auto & chunk = chunks[i];
                    results[i] = codec.encode(text.substr(chunk.begin, chunk.end - chunk.begin));
                } catch (...) {
                    failures[i] = std::current_exception();
                }
            }
        });
    }

    for (auto & thread : threads) {
        thread.join();
    }
    for (const auto & failure : failures) {
        if (failure) {
            std::rethrow_exception(failure);
        }
    }

    std::size_t token_count = 0;
    for (const auto & result : results) {
        token_count += result.size();
    }

    std::vector<int> tokens;
    tokens.reserve(token_count);
    for (auto & result : results) {
        tokens.insert(tokens.end(), result.begin(), result.end());
    }
    return tokens;
}

std::size_t wanxiangqi_o200k_codec::vocabulary_size() const {
    return pimpl->encoding->get_byte_pair_token_map().size();
}
