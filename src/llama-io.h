#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

struct ggml_tensor;

class llama_io_write_i {
public:
    llama_io_write_i() = default;
    virtual ~llama_io_write_i() = default;

    virtual void write(const void * src, size_t size) = 0;
    virtual void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) = 0;

    // bytes written so far
    virtual size_t n_bytes() = 0;

    // Finish deferred work (for example, device tensor copies). Returns false
    // when a copy submission is unproven and its resources must be retained.
    // The default is true because host and file streams finish their work
    // synchronously or in the destructor.
    virtual bool finalize() { return true; }

    void write_string(const std::string & str);
};

class llama_io_read_i {
public:
    llama_io_read_i() = default;
    virtual ~llama_io_read_i() = default;

    virtual void read(void * dst, size_t size) = 0;
    virtual void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) = 0;

    // bytes read so far
    virtual size_t n_bytes() = 0;

    // Finish deferred work. See llama_io_write_i::finalize().
    virtual bool finalize() { return true; }

    void read_string(std::string & str);
};
