#pragma once

// Host snapshot of one MTP target sequence state (attention KV plus the hybrid
// GDN recurrent state). The storage survives across verify epochs. The current
// snapshot size and an explicit valid flag identify the bytes that belong to
// this epoch, so a buffer that still holds an older, larger snapshot is never
// restored as the new one. The storage grows only when the new snapshot is
// larger; a same-size or smaller snapshot reuses it without zero-filling.

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
