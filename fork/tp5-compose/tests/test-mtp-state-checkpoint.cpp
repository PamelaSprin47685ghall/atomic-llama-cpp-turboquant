// MTP target state checkpoint: retained storage with an explicit validity and
// snapshot size. Covers same-size reuse without zero-filling, growth, shrink,
// short capture, short restore, invalidation, zero size and exception unwinding.

#include "mtp_state_checkpoint.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace {

int g_failures = 0;

void check(bool ok, const char * name) {
    if (!ok) {
        ++g_failures;
    }
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", name);
}

std::size_t fill(std::uint8_t * dst, std::size_t size, std::uint8_t value) {
    std::memset(dst, value, size);
    return size;
}

} // namespace

int main() {
    using tq::tp5::mtp_state_checkpoint;

    {
        mtp_state_checkpoint checkpoint;
        check(!checkpoint.is_valid() && checkpoint.size() == 0 && checkpoint.capacity() == 0,
              "checkpoint: starts invalid");

        bool called = false;
        const std::size_t restored = checkpoint.restore(
            [&](const std::uint8_t *, std::size_t) {
                called = true;
                return std::size_t(1);
            });
        check(restored == 0 && !called, "checkpoint: invalid restore does not read");
    }

    {
        mtp_state_checkpoint checkpoint;
        const bool first = checkpoint.capture(16,
            [](std::uint8_t * dst, std::size_t size) { return fill(dst, size, 0xAB); });
        check(first && checkpoint.is_valid() && checkpoint.size() == 16 && checkpoint.capacity() == 16,
              "checkpoint: first capture");

        // A same-size capture must see the retained bytes, not a zero fill.
        bool retained = false;
        const bool second = checkpoint.capture(16,
            [&](std::uint8_t * dst, std::size_t size) {
                retained = size == 16;
                for (std::size_t i = 0; i < size; ++i) {
                    retained = retained && dst[i] == 0xAB;
                }
                return fill(dst, size, 0xCD);
            });
        check(second && retained, "checkpoint: same size reuses the bytes");
        check(checkpoint.size() == 16 && checkpoint.capacity() == 16,
              "checkpoint: same size keeps storage");
    }

    {
        mtp_state_checkpoint checkpoint;
        checkpoint.capture(16, [](std::uint8_t * dst, std::size_t size) {
            return fill(dst, size, 0xAB);
        });

        bool prefix_ok = false;
        const bool shrunk = checkpoint.capture(8,
            [&](std::uint8_t * dst, std::size_t size) {
                prefix_ok = size == 8;
                for (std::size_t i = 0; i < size; ++i) {
                    prefix_ok = prefix_ok && dst[i] == 0xAB;
                }
                return fill(dst, size, 0xCD);
            });
        check(shrunk && prefix_ok, "checkpoint: shrink retains the prefix");
        check(checkpoint.size() == 8 && checkpoint.capacity() == 16,
              "checkpoint: shrink keeps capacity");

        std::size_t read_size = 0;
        const std::size_t restored = checkpoint.restore(
            [&](const std::uint8_t * src, std::size_t size) {
                read_size = size;
                bool bytes = true;
                for (std::size_t i = 0; i < size; ++i) {
                    bytes = bytes && src[i] == 0xCD;
                }
                return bytes ? size : 0;
            });
        check(restored == 8 && read_size == 8,
              "checkpoint: restore uses the snapshot size, not the storage size");
        check(!checkpoint.is_valid() && checkpoint.size() == 0,
              "checkpoint: restore consumes the checkpoint");
    }

    {
        mtp_state_checkpoint checkpoint;
        checkpoint.capture(8, [](std::uint8_t * dst, std::size_t size) {
            return fill(dst, size, 0xAB);
        });

        bool prefix_ok = false;
        const bool grown = checkpoint.capture(24,
            [&](std::uint8_t * dst, std::size_t size) {
                prefix_ok = size == 24;
                for (std::size_t i = 0; i < 8; ++i) {
                    prefix_ok = prefix_ok && dst[i] == 0xAB;
                }
                return fill(dst, size, 0xCD);
            });
        check(grown && prefix_ok, "checkpoint: grow retains the prefix");
        check(checkpoint.size() == 24 && checkpoint.capacity() == 24,
              "checkpoint: grow updates storage");
    }

    {
        mtp_state_checkpoint checkpoint;
        checkpoint.capture(16, [](std::uint8_t * dst, std::size_t size) {
            return fill(dst, size, 0xAB);
        });

        const bool short_capture = checkpoint.capture(16,
            [](std::uint8_t * dst, std::size_t size) {
                return fill(dst, size, 0xCD) - 1;
            });
        check(!short_capture && !checkpoint.is_valid() && checkpoint.size() == 0,
              "checkpoint: short capture invalidates");

        bool called = false;
        const std::size_t restored = checkpoint.restore(
            [&](const std::uint8_t *, std::size_t size) {
                called = true;
                return size;
            });
        check(restored == 0 && !called, "checkpoint: short capture is not restored");
    }

    {
        mtp_state_checkpoint checkpoint;
        checkpoint.capture(16, [](std::uint8_t * dst, std::size_t size) {
            return fill(dst, size, 0xAB);
        });

        std::size_t calls = 0;
        const std::size_t short_restore = checkpoint.restore(
            [&](const std::uint8_t *, std::size_t size) {
                ++calls;
                return size - 1;
            });
        check(short_restore == 0 && !checkpoint.is_valid(), "checkpoint: short restore fails");

        const std::size_t retried = checkpoint.restore(
            [&](const std::uint8_t *, std::size_t size) {
                ++calls;
                return size;
            });
        check(retried == 0 && calls == 1, "checkpoint: failed restore cannot be retried");
    }

    {
        mtp_state_checkpoint checkpoint;
        checkpoint.capture(16, [](std::uint8_t * dst, std::size_t size) {
            return fill(dst, size, 0xAB);
        });
        checkpoint.invalidate();

        bool called = false;
        const std::size_t restored = checkpoint.restore(
            [&](const std::uint8_t *, std::size_t size) {
                called = true;
                return size;
            });
        check(restored == 0 && !called, "checkpoint: invalidated snapshot is not restored");
        check(checkpoint.size() == 0, "checkpoint: invalidation clears the valid length");
    }

    {
        // Epoch 1 captures 16 bytes and is consumed by full acceptance.
        // Epoch 2's state shrank to 8: the restore must pass 8 bytes, not the
        // retained 16, and must never see epoch 1's snapshot.
        mtp_state_checkpoint checkpoint;
        checkpoint.capture(16, [](std::uint8_t * dst, std::size_t size) {
            return fill(dst, size, 0xAB);
        });
        checkpoint.invalidate();

        const bool captured = checkpoint.capture(8, [](std::uint8_t * dst, std::size_t size) {
            return fill(dst, size, 0xCD);
        });

        std::size_t read_size = 0;
        bool epoch1_absent = true;
        const std::size_t restored = checkpoint.restore(
            [&](const std::uint8_t * src, std::size_t size) {
                read_size = size;
                for (std::size_t i = 0; i < size; ++i) {
                    epoch1_absent = epoch1_absent && src[i] == 0xCD;
                }
                return size;
            });
        check(captured && restored == 8 && read_size == 8,
              "checkpoint: full accept then shrink restores the new length");
        check(epoch1_absent, "checkpoint: full accept does not restore the old snapshot");
    }

    {
        // Epoch 1 captures 16. Epoch 2's 8-byte read is partial: it must not
        // leave the epoch-1 snapshot usable. A retry with 8 must restore 8.
        mtp_state_checkpoint checkpoint;
        checkpoint.capture(16, [](std::uint8_t * dst, std::size_t size) {
            return fill(dst, size, 0xAB);
        });
        const bool partial = checkpoint.capture(8, [](std::uint8_t * dst, std::size_t size) {
            fill(dst, size, 0xCD);
            return size - 1;
        });

        bool called_after_partial = false;
        const std::size_t rejected = checkpoint.restore(
            [&](const std::uint8_t *, std::size_t size) {
                called_after_partial = true;
                return size;
            });

        const bool retried = checkpoint.capture(8, [](std::uint8_t * dst, std::size_t size) {
            return fill(dst, size, 0xEF);
        });
        std::size_t read_size = 0;
        const std::size_t restored = checkpoint.restore(
            [&](const std::uint8_t * src, std::size_t size) {
                read_size = size;
                bool bytes = true;
                for (std::size_t i = 0; i < size; ++i) {
                    bytes = bytes && src[i] == 0xEF;
                }
                return bytes ? size : 0;
            });
        check(!partial && rejected == 0 && !called_after_partial,
              "checkpoint: partial capture rejects the old snapshot");
        check(retried && restored == 8 && read_size == 8,
              "checkpoint: retry after a partial capture restores the new length");
    }

    {
        // Epoch 1 captures 16. A throwing restore invalidates it. Epoch 2's
        // state shrank to 8: the next restore must pass 8, not 16.
        mtp_state_checkpoint checkpoint;
        checkpoint.capture(16, [](std::uint8_t * dst, std::size_t size) {
            return fill(dst, size, 0xAB);
        });
        bool threw = false;
        try {
            checkpoint.restore([](const std::uint8_t *, std::size_t) -> std::size_t {
                throw std::runtime_error("restore");
            });
        } catch (const std::runtime_error &) {
            threw = true;
        }

        const bool captured = checkpoint.capture(8, [](std::uint8_t * dst, std::size_t size) {
            return fill(dst, size, 0xCD);
        });
        std::size_t read_size = 0;
        const std::size_t restored = checkpoint.restore(
            [&](const std::uint8_t * src, std::size_t size) {
                read_size = size;
                bool bytes = true;
                for (std::size_t i = 0; i < size; ++i) {
                    bytes = bytes && src[i] == 0xCD;
                }
                return bytes ? size : 0;
            });
        check(threw && captured && restored == 8 && read_size == 8,
              "checkpoint: exception then shrink restores the new length");
    }

    {
        mtp_state_checkpoint checkpoint;
        bool called = false;
        const bool captured = checkpoint.capture(0,
            [&](std::uint8_t *, std::size_t) {
                called = true;
                return std::size_t(0);
            });
        check(!captured && !called && !checkpoint.is_valid(), "checkpoint: zero size is invalid");
    }

    {
        mtp_state_checkpoint checkpoint;
        checkpoint.capture(16, [](std::uint8_t * dst, std::size_t size) {
            return fill(dst, size, 0xAB);
        });

        bool threw = false;
        try {
            checkpoint.capture(16, [](std::uint8_t *, std::size_t) -> std::size_t {
                throw std::runtime_error("capture");
            });
        } catch (const std::runtime_error &) {
            threw = true;
        }
        check(threw && !checkpoint.is_valid() && checkpoint.size() == 0,
              "checkpoint: throwing capture invalidates");

        bool called = false;
        const std::size_t restored = checkpoint.restore(
            [&](const std::uint8_t *, std::size_t size) {
                called = true;
                return size;
            });
        check(restored == 0 && !called, "checkpoint: throwing capture is not restored");
    }

    {
        mtp_state_checkpoint checkpoint;
        checkpoint.capture(16, [](std::uint8_t * dst, std::size_t size) {
            return fill(dst, size, 0xAB);
        });

        bool threw = false;
        try {
            checkpoint.restore([](const std::uint8_t *, std::size_t) -> std::size_t {
                throw std::runtime_error("restore");
            });
        } catch (const std::runtime_error &) {
            threw = true;
        }
        check(threw && !checkpoint.is_valid(), "checkpoint: throwing restore invalidates");

        std::size_t calls = 0;
        const std::size_t retried = checkpoint.restore(
            [&](const std::uint8_t *, std::size_t size) {
                ++calls;
                return size;
            });
        check(retried == 0 && calls == 0, "checkpoint: throwing restore cannot be retried");
    }

    std::printf("%s failures=%d\n", g_failures ? "FAILED" : "OK", g_failures);
    return g_failures ? 1 : 0;
}
