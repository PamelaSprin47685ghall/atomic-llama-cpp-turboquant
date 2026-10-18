#pragma once

#include <cerrno>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace tq_imatrix {

inline std::system_error io_error(const char * action) {
    return std::system_error(errno, std::generic_category(), action);
}

class private_file {
public:
    private_file(int directory, const std::string & reserved) : directory_(directory) {
        for (unsigned attempt = 0; attempt < 256; ++attempt) {
            name_ = ".tq-imatrix-" + std::to_string(::getpid()) + "-" + std::to_string(attempt) + ".tmp";
            if (name_ == reserved) { continue; }
            descriptor_ = ::openat(directory_, name_.c_str(), O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
            if (descriptor_ >= 0) { return; }
            if (errno != EEXIST) { throw io_error("create private output file"); }
        }
        throw std::runtime_error("cannot reserve private output file");
    }

    private_file(const private_file &) = delete;
    private_file & operator=(const private_file &) = delete;
    ~private_file() {
        if (descriptor_ >= 0) { ::close(descriptor_); }
        ::unlinkat(directory_, name_.c_str(), 0);
    }

    void write(const void * data, std::size_t count) {
        if (failed_ || descriptor_ < 0) { throw std::logic_error("private output has failed or closed"); }
        const auto * bytes = static_cast<const char *>(data);
        while (count) {
            const auto written = ::write(descriptor_, bytes, count);
            if (written < 0 && errno == EINTR) { continue; }
            if (written <= 0) {
                failed_ = true;
                if (written < 0) { throw io_error("write private output"); }
                throw std::runtime_error("private output write made no progress");
            }
            count -= static_cast<std::size_t>(written);
            bytes += written;
        }
    }

    void rewind() {
        if (failed_ || descriptor_ < 0) { throw std::logic_error("private output has failed or closed"); }
        if (::lseek(descriptor_, 0, SEEK_SET) < 0) { failed_ = true; throw io_error("rewind payload spool"); }
    }

    std::size_t read(void * data, std::size_t count) {
        if (failed_ || descriptor_ < 0) { throw std::logic_error("private output has failed or closed"); }
        ssize_t received;
        do { received = ::read(descriptor_, data, count); } while (received < 0 && errno == EINTR);
        if (received < 0) { failed_ = true; throw io_error("read payload spool"); }
        return static_cast<std::size_t>(received);
    }

    void publish(const std::string & destination) {
        if (failed_ || descriptor_ < 0) { throw std::logic_error("private output has failed or closed"); }
        if (::fsync(descriptor_) < 0) { failed_ = true; throw io_error("sync completed output"); }
        const int descriptor = descriptor_;
        descriptor_ = -1;
        if (::close(descriptor) < 0) { throw io_error("close completed output"); }
        if (::linkat(directory_, name_.c_str(), directory_, destination.c_str(), 0) < 0) {
            throw io_error("publish output (existing entries are refused)");
        }
    }

private:
    int directory_;
    int descriptor_ = -1;
    bool failed_ = false;
    std::string name_;
};

class output_files {
public:
    explicit output_files(const std::filesystem::path & output) {
        name_ = output.filename().string();
        if (name_.empty() || name_ == "." || name_ == "..") { throw std::invalid_argument("output must name a new file"); }
        const auto parent = output.has_parent_path() ? output.parent_path() : std::filesystem::path(".");
        directory_ = ::open(parent.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (directory_ < 0) { throw io_error("open output directory"); }
        try {
            struct stat status;
            if (::fstatat(directory_, name_.c_str(), &status, AT_SYMLINK_NOFOLLOW) == 0) { throw std::runtime_error("output already exists"); }
            if (errno != ENOENT) { throw io_error("inspect output destination"); }
            output_ = std::make_unique<private_file>(directory_, name_);
            spool_ = std::make_unique<private_file>(directory_, name_);
        } catch (...) {
            spool_.reset();
            output_.reset();
            ::close(directory_);
            throw;
        }
    }
    output_files(const output_files &) = delete;
    output_files & operator=(const output_files &) = delete;
    ~output_files() {
        spool_.reset();
        output_.reset();
        ::close(directory_);
    }
    private_file & spool() { return *spool_; }
    private_file & output() { return *output_; }
    void publish() { output_->publish(name_); }

private:
    int directory_ = -1;
    std::string name_;
    std::unique_ptr<private_file> output_;
    std::unique_ptr<private_file> spool_;
};

} // namespace tq_imatrix
