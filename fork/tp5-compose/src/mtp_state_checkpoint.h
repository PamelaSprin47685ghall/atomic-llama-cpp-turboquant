#pragma once

// Checkpoint storage for one MTP target sequence state. With host state flags
// the retained storage owns the complete snapshot (attention KV plus the hybrid
// GDN recurrent state). With LLAMA_STATE_SEQ_FLAGS_ON_DEVICE the retained
// storage owns only the metadata written by llama_state_seq_get_data_ext; the
// tensor bytes live in ctx->mem_storage[seq_id] and are owned by the context.
// In device mode this class cannot see or validate that device copy: the caller
// must keep the context alive and must not overwrite the device snapshot
// between capture and restore. A second ON_DEVICE capture for the same seq
// replaces mem_storage[seq]; the device metadata carries a generation, so
// restoring the older metadata returns 0 from llama_state_seq_set_data_ext
// instead of silently copying the newer tensors. The current size and valid
// flag identify the bytes in this object's storage only; they do not identify
// the device snapshot.
//
// The storage survives across verify epochs. The current snapshot size and an
// explicit valid flag identify the bytes that belong to this epoch, so a buffer
// that still holds an older, larger snapshot is never restored as the new one.
// The storage grows only when the new snapshot is larger; a same-size or
// smaller snapshot reuses it without zero-filling.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tq::tp5 {

class mtp_state_checkpoint {
public:
    // Capture size bytes into the retained storage. The reader returns the
    // number of bytes it wrote; only an exact match marks the checkpoint valid.
    // A short, failed or throwing read leaves it invalid, even when an older
    // snapshot is still present.
    template <typename Capture>
    bool capture(std::size_t size, Capture && capture) {
        valid = false;
        size_ = 0;

        if (size == 0) {
            return false;
        }

        // Grow only. The new tail is zero-initialized by resize and then fully
        // overwritten by the reader; existing bytes are retained.
        if (size > data.size()) {
            data.resize(size);
        }

        const std::size_t written = capture(data.data(), size);
        if (written != size) {
            return false;
        }

        size_ = size;
        valid = true;
        return true;
    }

    // Restore the current snapshot through reader. The reader returns the
    // number of bytes it consumed. The checkpoint is consumed either way, so a
    // failed restore cannot be retried with the same bytes. size() is zero
    // afterwards; the storage itself is retained.
    template <typename Restore>
    std::size_t restore(Restore && restore) {
        if (!valid) {
            return 0;
        }

        const std::size_t size = size_;
        valid = false;
        size_ = 0;

        const std::size_t restored = restore(data.data(), size);
        return restored == size ? restored : 0;
    }

    void invalidate() {
        valid = false;
        size_ = 0;
    }

    bool is_valid() const {
        return valid;
    }

    // Bytes of the current snapshot, not the retained storage size.
    std::size_t size() const {
        return size_;
    }

    // Retained storage bytes. Never smaller than size().
    std::size_t capacity() const {
        return data.size();
    }

private:
    std::vector<std::uint8_t> data;
    std::size_t size_ = 0;
    bool valid = false;
};

} // namespace tq::tp5
