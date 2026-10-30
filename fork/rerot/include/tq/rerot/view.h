#pragma once

#include "tq/rerot/plan.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tq::rerot {

enum class SegmentKind : std::uint8_t { prefix, frame, body, source_end, control };
enum class Visibility : std::uint8_t { public_kv, private_kv };

struct ReadStamp {
    std::uint64_t episode = 0;
    std::uint64_t frontier = 0;
    std::uint64_t revision = 0;
    StageId stage = 0;
};

struct QueryPoint {
    RunId run = 0;
    std::uint64_t offset = 0;
};

struct WriteSpan {
    RunId run = 0;
    StageId stage = 0;
    std::uint64_t offset = 0;
    std::uint64_t count = 0;
    std::int64_t storage_pos = 0;
    SegmentKind kind = SegmentKind::body;
    Visibility visibility = Visibility::public_kv;
};

struct Span : WriteSpan {
    std::int64_t virtual_pos = 0;
};

struct Entry {
    RunId run = 0;
    StageId stage = 0;
    std::uint64_t offset = 0;
    std::int64_t storage_pos = 0;
    std::int64_t virtual_pos = 0;
    std::int64_t effective_pos = 0;
};

class Episode;

// This value owns all visible prefix lengths. Later commits cannot change it.
// There are no physical key indices, pointers, or backend handles in a view.
class ReadView {
public:
    const ReadStamp & stamp() const noexcept;
    const std::vector<Span> & spans() const noexcept;
    std::int64_t query_virtual_pos() const noexcept;
    std::uint64_t key_count() const noexcept;

private:
    ReadStamp stamp_;
    std::vector<Span> spans_;
    std::int64_t query_virtual_pos_ = 0;
    std::uint64_t key_count_ = 0;
    friend class Episode;
};

// The caller supplies a hard allocation limit. Position arithmetic is checked.
std::vector<Entry> expand(const ReadView & view, std::size_t max_entries);

} // namespace tq::rerot
