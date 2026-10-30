#pragma once

#include "options.h"

#include <llama.h>
#include <tq/rerot/view.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace tq::rerot_compose {

struct input_row {
    tq::rerot::StageId stage;
    llama_pos position;
    llama_token token;
    bool logits;
    tq::rerot::ReadView view;
};

struct output_row {
    tq::rerot::StageId stage;
    std::vector<float> logits;
};

// This C object owns one native A context, its inputs, and submission facts.
// It uses public A APIs. It does not claim that llama_decode is reentrant or
// that one native call gives it access to A's internal per-device streams.
class native_execution {
public:
    // Native loading and supported compute backends query cancellation, as do
    // synchronized C boundaries. The callback must be safe on A's calling
    // thread. Cancellation is sticky. Exceptions are returned to C only after
    // the native call has returned and its submitted work has drained.
    explicit native_execution(const options & config, std::uint32_t stages, std::uint32_t pens,
                              std::function<bool()> stopped = {});
    ~native_execution();
    native_execution(const native_execution &) = delete;
    native_execution & operator=(const native_execution &) = delete;

    const llama_model * model() const noexcept;
    const llama_vocab * vocab() const noexcept;
    std::uint32_t context_size() const noexcept;
    std::uint32_t kv_capacity() const noexcept;
    std::uint32_t sequence_capacity() const noexcept;
    std::uint64_t compute_bytes() const noexcept;
    bool drained() const noexcept;
    bool pipeline_parallel_enabled() const noexcept;
    std::string device_summary() const;
    std::string context_summary() const;

    void prefill(const std::vector<llama_token> & tokens);
    void start(tq::rerot::StageId stage);
    void seal(tq::rerot::StageId stage);
    void release_compute();
    std::vector<output_row> forward(std::uint64_t frontier, const std::vector<input_row> & rows);

private:
    struct impl;
    std::unique_ptr<impl> state_;
};

} // namespace tq::rerot_compose
