#include "tq/rerot/view.h"

#include <limits>
#include <stdexcept>

namespace tq::rerot {

const ReadStamp & ReadView::stamp() const noexcept { return stamp_; }
const std::vector<Span> & ReadView::spans() const noexcept { return spans_; }
std::int64_t ReadView::query_virtual_pos() const noexcept { return query_virtual_pos_; }
std::uint64_t ReadView::key_count() const noexcept { return key_count_; }

std::vector<Entry> expand(const ReadView & view, std::size_t max_entries) {
    if (view.key_count() > max_entries || view.key_count() > std::numeric_limits<std::size_t>::max()) {
        throw std::length_error("RERoT view exceeds its entry limit");
    }
    std::vector<Entry> result;
    result.reserve(static_cast<std::size_t>(view.key_count()));
    const auto limit = std::numeric_limits<std::int64_t>::max();
    for (const auto & span : view.spans()) {
        // All positions in a view are non-negative. Subtract before adding so
        // a valid signed phase does not require an overflowing intermediate.
        const std::int64_t delta = span.storage_pos - span.virtual_pos;
        if (delta > 0 && view.query_virtual_pos() > limit - delta) {
            throw std::overflow_error("RERoT effective position overflows");
        }
        const std::int64_t effective = view.query_virtual_pos() + delta;
        for (std::uint64_t i = 0; i < span.count; ++i) {
            result.push_back({span.run, span.stage, span.offset + i,
                span.storage_pos + static_cast<std::int64_t>(i),
                span.virtual_pos + static_cast<std::int64_t>(i), effective});
        }
    }
    return result;
}

} // namespace tq::rerot
