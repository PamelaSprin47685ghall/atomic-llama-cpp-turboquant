#pragma once

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace tq_logits {

inline std::system_error io_error(const char * operation) {
    return std::system_error(errno, std::generic_category(), operation);
}

class new_file {
public:
    explicit new_file(const std::filesystem::path & path) {
        const auto parent = path.has_parent_path() ? path.parent_path() : std::filesystem::path(".");
        name_ = path.filename().string();
        if (name_.empty() || name_ == "." || name_ == "..") {
            throw std::invalid_argument("output must name a new file");
        }
        directory_ = ::open(parent.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (directory_ < 0) { throw io_error("open output directory"); }
        try {
            struct stat status;
            if (::fstatat(directory_, name_.c_str(), &status, AT_SYMLINK_NOFOLLOW) == 0) {
                throw std::runtime_error("output already exists");
            }
            if (errno != ENOENT) { throw io_error("inspect output path"); }
            for (unsigned attempt = 0; attempt < 256; ++attempt) {
                temporary_ = ".tq-logits-" + std::to_string(::getpid()) + "-" + std::to_string(attempt) + ".tmp";
                if (temporary_ == name_) { continue; }
                descriptor_ = ::openat(directory_, temporary_.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
                if (descriptor_ >= 0) { return; }
                if (errno != EEXIST) { throw io_error("create output temporary"); }
            }
            throw std::runtime_error("cannot reserve an output temporary");
        } catch (...) {
            ::close(directory_);
            directory_ = -1;
            throw;
        }
    }

    new_file(const new_file &) = delete;
    new_file & operator=(const new_file &) = delete;

    ~new_file() {
        if (descriptor_ >= 0) { ::close(descriptor_); }
        if (directory_ >= 0) {
            ::unlinkat(directory_, temporary_.c_str(), 0);
            ::close(directory_);
        }
    }

    void write(const void * data, std::size_t size) {
        if (descriptor_ < 0 || failed_) { throw std::logic_error("output is closed or failed"); }
        const auto * next = static_cast<const char *>(data);
        while (size) {
            const auto count = ::write(descriptor_, next, size);
            if (count < 0 && errno == EINTR) { continue; }
            if (count < 0) { failed_ = true; throw io_error("write output temporary"); }
            if (count == 0) { failed_ = true; throw std::runtime_error("output write made no progress"); }
            next += count;
            size -= static_cast<std::size_t>(count);
        }
    }

    void publish() {
        if (descriptor_ < 0 || failed_) { throw std::logic_error("output is closed or failed"); }
        if (::fsync(descriptor_) < 0) { failed_ = true; throw io_error("sync output temporary"); }
        const int descriptor = descriptor_;
        descriptor_ = -1;
        if (::close(descriptor) < 0) { throw io_error("close output temporary"); }
        if (::linkat(directory_, temporary_.c_str(), directory_, name_.c_str(), 0) < 0) {
            throw io_error("publish new output (existing paths are refused)");
        }
    }

private:
    int directory_ = -1;
    int descriptor_ = -1;
    bool failed_ = false;
    std::string name_;
    std::string temporary_;
};

inline void put_u32(unsigned char * out, std::uint32_t value) {
    for (unsigned i = 0; i < 4; ++i) { out[i] = static_cast<unsigned char>(value >> (8 * i)); }
}

class tape_writer {
public:
    tape_writer(new_file & output, std::uint32_t vocabulary, std::uint32_t prefill,
                const std::vector<std::int32_t> & tokens, std::uint64_t max_bytes)
        : output_(output), vocabulary_(vocabulary) {
        static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559, "TP5LOG2 requires IEEE binary32");
        if (!vocabulary || vocabulary > 2000000 || tokens.empty() || tokens.size() > 2000000 ||
            !prefill || prefill > tokens.size()) {
            throw std::invalid_argument("invalid TP5LOG2 dimensions");
        }
        for (const auto token : tokens) {
            if (token < 0 || std::uint32_t(token) >= vocabulary) {
                throw std::invalid_argument("input token outside the vocabulary");
            }
        }
        rows_ = static_cast<std::uint32_t>(tokens.size()) - prefill + 1;
        bytes_ = 24 + std::uint64_t(tokens.size()) * 4 + std::uint64_t(rows_) * vocabulary * 4;
        if (bytes_ > max_bytes) { throw std::invalid_argument("tape exceeds --max-output-bytes"); }
        unsigned char header[24] = {'T', 'P', '5', 'L', 'O', 'G', '2', 0};
        put_u32(header + 8, vocabulary);
        put_u32(header + 12, prefill);
        put_u32(header + 16, rows_ - 1);
        put_u32(header + 20, static_cast<std::uint32_t>(tokens.size()));
        output_.write(header, sizeof(header));
        for (const auto token : tokens) { write_u32(static_cast<std::uint32_t>(token)); }
    }

    tape_writer(const tape_writer &) = delete;
    tape_writer & operator=(const tape_writer &) = delete;
    tape_writer(tape_writer &&) = delete;
    tape_writer & operator=(tape_writer &&) = delete;

    void append(const float * logits) {
        if (!logits || written_ == rows_) { throw std::logic_error("invalid logits row"); }
        for (std::uint32_t i = 0; i < vocabulary_; ++i) {
            if (!std::isfinite(logits[i])) { throw std::runtime_error("nonfinite logit at row " + std::to_string(written_) + ", token " + std::to_string(i)); }
        }
        std::array<unsigned char, 16384> buffer;
        for (std::uint32_t first = 0; first < vocabulary_;) {
            const auto count = std::min<std::uint32_t>(vocabulary_ - first, buffer.size() / 4);
            for (std::uint32_t i = 0; i < count; ++i) {
                std::uint32_t bits;
                std::memcpy(&bits, logits + first + i, sizeof(bits));
                put_u32(buffer.data() + 4 * i, bits);
            }
            output_.write(buffer.data(), count * 4);
            first += count;
        }
        ++written_;
    }

    void finish() {
        if (written_ != rows_) { throw std::logic_error("incomplete logits tape"); }
        output_.publish();
    }

    std::uint64_t bytes() const { return bytes_; }

private:
    void write_u32(std::uint32_t value) {
        unsigned char bytes[4];
        put_u32(bytes, value);
        output_.write(bytes, sizeof(bytes));
    }
    new_file & output_;
    std::uint32_t vocabulary_;
    std::uint32_t rows_ = 0;
    std::uint32_t written_ = 0;
    std::uint64_t bytes_ = 0;
};

} // namespace tq_logits
