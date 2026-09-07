#include "llama-xkv-reader.h"
#include "llama-xkv-factor.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace llama_xkv {

// ============================================================================
// Validation implementations
// ============================================================================
// Validation implementations
// ============================================================================

const xkv_factor_group_payload * xkv_segment_read_view::resolve_group(std::string * err) const {
    const auto * seg = get_segment();
    if (!seg) {
        if (err) *err = "xkv_segment_read_view: missing owning xkv_reader_pin";
        return nullptr;
    }

    const xkv_factor_group_payload * by_index = nullptr;
    if (factor_group_index != UINT32_MAX) {
        by_index = seg->find_group(factor_group_index);
        if (!by_index) {
            if (err) *err = "xkv_segment_read_view: factor group index " + std::to_string(factor_group_index) + " not found in segment bundle";
            return nullptr;
    }
    }

    const xkv_factor_group_payload * by_layer = nullptr;
    if (owning_layer != UINT32_MAX) {
        by_layer = seg->find_group_for_layer(owning_layer);
        if (!by_layer) {
            if (err) *err = "xkv_segment_read_view: owning_layer " + std::to_string(owning_layer) + " not found in any factor group";
            return nullptr;
    }
    }

    if (by_index && by_layer && by_index != by_layer) {
        if (err) *err = "xkv_segment_read_view: factor_group_index and owning_layer conflict; resolve to different groups";
        return nullptr;
    }

    const xkv_factor_group_payload * g = by_index ? by_index : by_layer;
    if (!g) {
        if (err) *err = "xkv_segment_read_view: neither valid factor_group_index nor owning_layer specified";
        return nullptr;
    }
    return g;
    }

uint32_t xkv_segment_read_view::get_b_feature_offset_k(uint32_t head_dim_k, std::string * err) const {
    const auto * g = resolve_group(err);
    if (!g) return UINT32_MAX;
    if (owning_layer == UINT32_MAX) {
        if (err) *err = "xkv_segment_read_view: owning_layer required to derive B feature offset";
        return UINT32_MAX;
    }
    int32_t l_idx = g->find_owning_layer_index(owning_layer);
    if (l_idx < 0) {
        if (err) *err = "xkv_segment_read_view: owning_layer not in resolved factor group";
        return UINT32_MAX;
    }
    uint64_t layer_base = g->layer_feature_offsets_k[l_idx];
    uint64_t layer_dim = g->layer_feature_dims_k[l_idx];
    uint64_t head_offset = static_cast<uint64_t>(kv_head) * static_cast<uint64_t>(head_dim_k);
    if (head_offset + head_dim_k > layer_dim) {
        if (err) *err = "xkv_segment_read_view: kv_head * head_dim_k exceeds layer_feature_dim_k";
        return UINT32_MAX;
    }
    uint64_t total_off = layer_base + head_offset;
    if (total_off > UINT32_MAX) {
        if (err) *err = "xkv_segment_read_view: B feature offset overflow";
        return UINT32_MAX;
    }
    return static_cast<uint32_t>(total_off);
    }

uint32_t xkv_segment_read_view::get_b_feature_offset_v(uint32_t head_dim_v, std::string * err) const {
    const auto * g = resolve_group(err);
    if (!g) return UINT32_MAX;
    if (owning_layer == UINT32_MAX) {
        if (err) *err = "xkv_segment_read_view: owning_layer required to derive B feature offset";
        return UINT32_MAX;
    }
    int32_t l_idx = g->find_owning_layer_index(owning_layer);
    if (l_idx < 0) {
        if (err) *err = "xkv_segment_read_view: owning_layer not in resolved factor group";
        return UINT32_MAX;
    }
    uint64_t layer_base = g->layer_feature_offsets_v[l_idx];
    uint64_t layer_dim = g->layer_feature_dims_v[l_idx];
    uint64_t head_offset = static_cast<uint64_t>(kv_head) * static_cast<uint64_t>(head_dim_v);
    if (head_offset + head_dim_v > layer_dim) {
        if (err) *err = "xkv_segment_read_view: kv_head * head_dim_v exceeds layer_feature_dim_v";
        return UINT32_MAX;
    }
    uint64_t total_off = layer_base + head_offset;
    if (total_off > UINT32_MAX) {
        if (err) *err = "xkv_segment_read_view: B feature offset overflow";
        return UINT32_MAX;
    }
    return static_cast<uint32_t>(total_off);
    }

bool xkv_segment_read_view::validate(std::string * err) const {
    const auto * seg = get_segment();
    if (!seg) {
        if (err) *err = "xkv_segment_read_view: missing owning xkv_reader_pin";
        return false;
    }

    const auto * g = resolve_group(err);
    if (!g) {
        return false;
    }

    if (!g->b_k || !g->b_v) {
        if (err) *err = "xkv_segment_read_view: segment factor group missing immutable B_K or B_V code stream handle";
        return false;
    }

    // Validate factor rank equality between factor streams A and B
    if (g->a_k.desc.logical_shape.cols != g->b_k->desc.logical_shape.cols) {
        if (err) *err = "xkv_segment_read_view: A_K rank (" +
            std::to_string(g->a_k.desc.logical_shape.cols) + ") != B_K rank (" +
            std::to_string(g->b_k->desc.logical_shape.cols) + ")";
        return false;
    }
    if (g->a_v.desc.logical_shape.cols != g->b_v->desc.logical_shape.cols) {
        if (err) *err = "xkv_segment_read_view: A_V rank (" +
            std::to_string(g->a_v.desc.logical_shape.cols) + ") != B_V rank (" +
            std::to_string(g->b_v->desc.logical_shape.cols) + ")";
        return false;
    }

    if (selected_rows.empty()) {
        if (err) *err = "xkv_segment_read_view: selected_rows is empty";
        return false;
    }

    if (storage_positions.size() != selected_rows.size()) {
        if (err) *err = "xkv_segment_read_view: storage_positions size (" +
            std::to_string(storage_positions.size()) + ") != selected_rows size (" +
            std::to_string(selected_rows.size()) + ")";
        return false;
    }

    if (!row_generations.empty() && row_generations.size() != selected_rows.size()) {
        if (err) *err = "xkv_segment_read_view: row_generations size mismatch";
        return false;
    }

    if (!group_indices.empty() && group_indices.size() != selected_rows.size()) {
        if (err) *err = "xkv_segment_read_view: group_indices size mismatch";
        return false;
    }

    if (!membership_mask.empty() && membership_mask.size() != selected_rows.size()) {
        if (err) *err = "xkv_segment_read_view: membership_mask size mismatch";
        return false;
    }

    uint64_t seg_rows = g->a_k.desc.logical_shape.rows;
    for (size_t i = 0; i < selected_rows.size(); ++i) {
        uint32_t r = selected_rows[i];
        if (r >= seg_rows) {
            if (err) *err = "xkv_segment_read_view: selected row " + std::to_string(r) +
                " out of segment bounds (" + std::to_string(seg_rows) + ")";
            return false;
        }
        if (r < seg->live_rows.size() && !seg->live_rows[r]) {
            if (err) *err = "xkv_segment_read_view: selected row " + std::to_string(r) + " is not live in segment";
            return false;
        }
        // Every selected row must carry a generation pin: an explicit per-row generation
        // when row_generations is provided, otherwise the segment-level storage_generation.
        // A zero effective generation means a stale/unbound row and is rejected before decode.
        uint64_t row_gen = !row_generations.empty() ? row_generations[i] : storage_generation;
        if (row_gen == 0) {
            if (err) *err = "xkv_segment_read_view: selected row " + std::to_string(r) +
                " has no generation pin (stale/unbound)";
            return false;
        }
    }

    // Validate owning_layer index inside resolved factor group
    if (owning_layer != UINT32_MAX) {
        int32_t l_idx = g->find_owning_layer_index(owning_layer);
        if (l_idx < 0) {
            if (err) *err = "xkv_segment_read_view: owning_layer " + std::to_string(owning_layer) + " not found in resolved factor group";
            return false;
        }
    }

    return true;
}

bool xkv_query_input::validate(std::string * err) const {
    if (head_dim_k == 0 || head_dim_v == 0 || n_q_heads == 0) {
        if (err) *err = "xkv_query_input: dimensions or head counts cannot be 0";
        return false;
    }

    // Scale 0.0 means "default to 1/sqrt(head_dim_k)" downstream. Any non-finite scale
    // (NaN, +Inf, -Inf) and any negative scale are rejected here, before any allocation.
    if (!std::isfinite(scale)) {
        if (err) *err = "xkv_query_input: non-finite scale";
        return false;
    }
    if (scale < 0.0f) {
        if (err) *err = "xkv_query_input: negative scale";
        return false;
    }

    // Huge dimensions are rejected before any buffer can be sized by them.
    if (n_q_heads > xkv_reader_max_dim || head_dim_k > xkv_reader_max_dim ||
        head_dim_v > xkv_reader_max_dim) {
        if (err) *err = "xkv_query_input: dimension exceeds sanity cap";
        return false;
    }

    if (logit_softcap < 0.0f || !std::isfinite(logit_softcap)) {
        if (err) *err = "xkv_query_input: invalid logit_softcap";
        return false;
    }

    for (float s : sink_logits) {
        if (std::isnan(s)) {
            if (err) *err = "xkv_query_input: sink logit is NaN";
            return false;
        }
    }

    size_t expected_q_len = 0;
    if (!safe_mul(static_cast<size_t>(n_q_heads), static_cast<size_t>(head_dim_k), expected_q_len)) {
        if (err) *err = "xkv_query_input: overflow computing expected query length";
        return false;
    }

    if (!q_vec.empty()) {
        if (q_vec.size() != expected_q_len) {
            if (err) *err = "xkv_query_input: q_vec size mismatch";
            return false;
        }
        for (float v : q_vec) {
            if (!std::isfinite(v)) {
                if (err) *err = "xkv_query_input: q_vec contains non-finite element";
                return false;
            }
        }
    }

    if (!q_groups.empty()) {
        for (size_t g = 0; g < q_groups.size(); ++g) {
            if (q_groups[g].size() != expected_q_len) {
                if (err) *err = "xkv_query_input: q_groups[" + std::to_string(g) + "] size mismatch";
                return false;
            }
            for (float v : q_groups[g]) {
                if (!std::isfinite(v)) {
                    if (err) *err = "xkv_query_input: q_groups contains non-finite element";
                    return false;
                }
            }
        }
    }

    if (q_vec.empty() && q_groups.empty()) {
        if (err) *err = "xkv_query_input: neither q_vec nor q_groups provided";
        return false;
    }

    return true;
}

// ============================================================================
// xkv_online_softmax_state
// ============================================================================

void xkv_online_softmax_state::reset(uint32_t dv) {
    max_score = -INFINITY;
    sum_exp = 0.0f;
    if (accum_ptr) {
        std::memset(accum_ptr, 0, dv * sizeof(float));
    } else {
        accum.assign(dv, 0.0f);
    }
}

void xkv_online_softmax_state::update_single(float score, const float * v, uint32_t dv) {
    if (!std::isfinite(score) || score <= -1e30f) {
        return; // Masked out
    }

    float * acc = get_accum();
    if (max_score <= -INFINITY) {
        max_score = score;
        sum_exp = 1.0f;
        for (uint32_t i = 0; i < dv; ++i) {
            acc[i] = v[i];
        }
    } else if (score > max_score) {
        float factor = std::exp(max_score - score);
        max_score = score;
        sum_exp = sum_exp * factor + 1.0f;
        for (uint32_t i = 0; i < dv; ++i) {
            acc[i] = acc[i] * factor + v[i];
        }
    } else {
        float factor = std::exp(score - max_score);
        sum_exp += factor;
        for (uint32_t i = 0; i < dv; ++i) {
            acc[i] += factor * v[i];
        }
    }
}

void xkv_online_softmax_state::update_sink(float sink_score, uint32_t dv) {
    if (!std::isfinite(sink_score) || sink_score <= -1e30f) {
        return; // No sink
    }

    float * acc = get_accum();
    if (max_score <= -INFINITY) {
        max_score = sink_score;
        sum_exp = 1.0f;
        if (accum_ptr) {
            std::memset(accum_ptr, 0, dv * sizeof(float));
        } else {
            accum.assign(dv, 0.0f);
        }
    } else if (sink_score > max_score) {
        float factor = std::exp(max_score - sink_score);
        max_score = sink_score;
        sum_exp = sum_exp * factor + 1.0f;
        for (uint32_t i = 0; i < dv; ++i) {
            acc[i] *= factor;
        }
    } else {
        float factor = std::exp(sink_score - max_score);
        sum_exp += factor;
    }
}

void xkv_online_softmax_state::merge_block(float block_max, float block_sum, const float * block_accum, uint32_t dv) {
    if (!std::isfinite(block_max) || block_sum <= 0.0f) {
        return;
    }

    float * acc = get_accum();
    if (max_score <= -INFINITY) {
        max_score = block_max;
        sum_exp = block_sum;
        for (uint32_t i = 0; i < dv; ++i) {
            acc[i] = block_accum[i];
        }
    } else if (block_max > max_score) {
        float factor = std::exp(max_score - block_max);
        max_score = block_max;
        sum_exp = sum_exp * factor + block_sum;
        for (uint32_t i = 0; i < dv; ++i) {
            acc[i] = acc[i] * factor + block_accum[i];
        }
    } else {
        float factor = std::exp(block_max - max_score);
        sum_exp += factor * block_sum;
        for (uint32_t i = 0; i < dv; ++i) {
            acc[i] += factor * block_accum[i];
        }
    }
}

void xkv_online_softmax_state::finalize(float * dst, uint32_t dv) const {
    const float * acc = get_accum();
    if (sum_exp > 0.0f && std::isfinite(sum_exp)) {
        float inv_sum = 1.0f / sum_exp;
        for (uint32_t i = 0; i < dv; ++i) {
            dst[i] = acc[i] * inv_sum;
        }
    } else {
        // Empty or all-masked: finite zero output without NaN
        for (uint32_t i = 0; i < dv; ++i) {
            dst[i] = 0.0f;
        }
    }
}

// ============================================================================
// xkv_b_tile_lease static factory
// ============================================================================

xkv_b_tile_lease xkv_b_tile_lease::uncached(std::shared_ptr<b_tile_record> rec) {
    return xkv_b_tile_lease(std::move(rec), nullptr);
}

xkv_b_tile_lease xkv_b_tile_lease::from_buffer(const float * ptr, size_t count) {
    xkv_b_tile_lease lease;
    lease.ptr_ = ptr;
    lease.count_ = count;
    return lease;
}

// ============================================================================
// xkv_b_tile_lease & xkv_b_tile_cache
// ============================================================================

xkv_b_tile_lease::xkv_b_tile_lease(std::shared_ptr<b_tile_record> rec, xkv_b_tile_cache * owner)
    : record_(std::move(rec)), owner_(owner) {
    if (record_) {
        ptr_ = record_->data.data();
        count_ = record_->data.size();
        record_->active_leases++;
    }
}

xkv_b_tile_lease::~xkv_b_tile_lease() {
    release();
}

xkv_b_tile_lease::xkv_b_tile_lease(xkv_b_tile_lease && o) noexcept
    : record_(std::move(o.record_)), ptr_(o.ptr_), count_(o.count_), owner_(o.owner_) {
    o.ptr_ = nullptr;
    o.count_ = 0;
    o.owner_ = nullptr;
}

xkv_b_tile_lease & xkv_b_tile_lease::operator=(xkv_b_tile_lease && o) noexcept {
    if (this != &o) {
        release();
        record_ = std::move(o.record_);
        ptr_ = o.ptr_;
        count_ = o.count_;
        owner_ = o.owner_;
        o.ptr_ = nullptr;
        o.count_ = 0;
        o.owner_ = nullptr;
    }
    return *this;
}

void xkv_b_tile_lease::release() {
    if (record_) {
        if (owner_) {
            owner_->on_lease_released(record_);
        } else {
            record_->active_leases--;
        }
    }
    record_.reset();
    ptr_ = nullptr;
    count_ = 0;
    owner_ = nullptr;
}

xkv_b_tile_cache::xkv_b_tile_cache(size_t max_bytes)
    : max_bytes_(max_bytes) {}

size_t xkv_b_tile_cache::compute_entry_bytes(size_t n_elements) {
    size_t data_bytes = 0;
    if (!safe_mul(n_elements, sizeof(float), data_bytes)) return SIZE_MAX;
    size_t entry_bytes = 0;
    if (!safe_add(sizeof(b_tile_record), data_bytes, entry_bytes) ||
        !safe_add(entry_bytes, sizeof(entry), entry_bytes)) {
        return SIZE_MAX;
    }
    return entry_bytes;
}

xkv_b_tile_lease xkv_b_tile_cache::get(const b_tile_cache_key & key) {
    if (max_bytes_ == 0) return xkv_b_tile_lease();

    std::lock_guard<std::mutex> lock(mtx_);
    auto it = map_.find(key);
    if (it == map_.end()) {
        misses_++;
        return xkv_b_tile_lease();
    }

    hits_++;
    // Move to MRU front
    lru_list_.splice(lru_list_.begin(), lru_list_, it->second);
    return xkv_b_tile_lease(it->second->record, this);
}

xkv_b_tile_lease xkv_b_tile_cache::put_and_lease_copy(const b_tile_cache_key & key, const float * data, size_t n_elements) {
    if (n_elements > 0 && data == nullptr) {
        return xkv_b_tile_lease();
    }
    size_t entry_bytes = compute_entry_bytes(n_elements);
    if (entry_bytes == SIZE_MAX) {
        return xkv_b_tile_lease(); // overflow: refuse without allocating
    }
    // Over-budget entries never touch the heap: the caller keeps using its own buffer.
    if (max_bytes_ == 0 || entry_bytes > max_bytes_) {
        return xkv_b_tile_lease::from_buffer(data, n_elements);
    }

    // Stage the record BEFORE taking the lock: allocation and fill happen outside,
    // so a throw here leaves the cache completely untouched. The bounded reader must
    // never fail semantically because this optional cache cannot allocate: every
    // failure below returns a borrow of the caller workspace buffer instead.
    std::shared_ptr<b_tile_record> staged;
    try {
        staged = std::make_shared<b_tile_record>();
        staged->data.assign(data, data + n_elements);
        staged->allocated_bytes = entry_bytes;
        staged->in_cache = true;
    } catch (...) {
        return xkv_b_tile_lease::from_buffer(data, n_elements);
    }

    std::lock_guard<std::mutex> lock(mtx_);
    auto it = map_.find(key);
    if (it != map_.end()) {
        // Immutable tile already cached: promote and lease the existing record.
        // The staged record is discarded; no counters were touched.
        hits_++;
        lru_list_.splice(lru_list_.begin(), lru_list_, it->second);
        return xkv_b_tile_lease(it->second->record, this);
    }
    // Evict unleased entries with checked fitting arithmetic; an overflow can never
    // read as "fits".
    while (true) {
        size_t projected = 0;
        if (!safe_add(total_bytes_, entry_bytes, projected) || projected <= max_bytes_) {
            break;
        }
        if (!evict_unleased_entry_locked()) {
            break; // all remaining entries leased: refuse insert
        }
    }
    {
        size_t projected = 0;
        if (!safe_add(total_bytes_, entry_bytes, projected) || projected > max_bytes_) {
            // Eviction could not free enough unleased space: borrow, no heap, untouched.
            return xkv_b_tile_lease::from_buffer(data, n_elements);
        }
    }
    // Insert with rollback: list and map mutations can throw (allocation), so the
    // counters commit only after both succeed. Any throw restores internal
    // consistency and returns the caller borrow.
    bool pushed = false;
    try {
        lru_list_.push_front(entry{key, staged});
        pushed = true;
        map_[key] = lru_list_.begin();
    } catch (...) {
        if (pushed) {
            lru_list_.pop_front();
        }
        return xkv_b_tile_lease::from_buffer(data, n_elements);
    }
    size_t next_total = 0;
    size_t next_active = 0;
    if (!safe_add(total_bytes_, entry_bytes, next_total) ||
        !safe_add(active_cache_bytes_, entry_bytes, next_active)) {
        // Unreachable after the fit check above, but never corrupt counters.
        map_.erase(key);
        lru_list_.pop_front();
        return xkv_b_tile_lease::from_buffer(data, n_elements);
    }
    total_bytes_ = next_total;
    active_cache_bytes_ = next_active;
    return xkv_b_tile_lease(staged, this);
}

xkv_b_tile_lease xkv_b_tile_cache::put_and_lease(const b_tile_cache_key & key, std::vector<float> tile_data) {
    size_t entry_bytes = compute_entry_bytes(tile_data.size());
    if (entry_bytes == SIZE_MAX) {
        auto rec = std::make_shared<b_tile_record>();
        rec->data = std::move(tile_data);
        return xkv_b_tile_lease(rec, nullptr);
    }

    if (max_bytes_ == 0 || entry_bytes > max_bytes_) {
        // Uncached single-lease fallback
        auto rec = std::make_shared<b_tile_record>();
        rec->data = std::move(tile_data);
        rec->allocated_bytes = entry_bytes;
        rec->in_cache = false;
        return xkv_b_tile_lease(rec, nullptr);
    }

    std::lock_guard<std::mutex> lock(mtx_);

    auto it = map_.find(key);
    if (it != map_.end()) {
        // B factor tiles are immutable for a given descriptor and cache key.
        // If the entry already exists, do NOT mutate the underlying record or replace data,
        // which would cause use-after-free for any active leaseholder.
        // Instead, promote to MRU front and return a new lease to the existing record.
        hits_++;
        lru_list_.splice(lru_list_.begin(), lru_list_, it->second);
        return xkv_b_tile_lease(it->second->record, this);
    }

    // Try to evict unleased entries until total_bytes_ + entry_bytes <= max_bytes_
    while (total_bytes_ + entry_bytes > max_bytes_) {
        if (!evict_unleased_entry_locked()) {
            // Cannot evict further (all entries currently leased!) -> refuse cache insert
            break;
        }
    }

    if (total_bytes_ + entry_bytes > max_bytes_) {
        // Refuse insertion into cache; return uncached lease
        auto rec = std::make_shared<b_tile_record>();
        rec->data = std::move(tile_data);
        rec->allocated_bytes = entry_bytes;
        rec->in_cache = false;
        return xkv_b_tile_lease(rec, nullptr);
    }

    auto rec = std::make_shared<b_tile_record>();
    rec->data = std::move(tile_data);
    rec->allocated_bytes = entry_bytes;
    rec->in_cache = true;

    entry e;
    e.key = key;
    e.record = rec;

    active_cache_bytes_ += entry_bytes;
    total_bytes_ += entry_bytes;

    lru_list_.push_front(std::move(e));
    map_[key] = lru_list_.begin();

    return xkv_b_tile_lease(rec, this);
}

bool xkv_b_tile_cache::evict_unleased_entry_locked() {
    // Traverse from LRU back to find first entry without active leases
    for (auto it = lru_list_.rbegin(); it != lru_list_.rend(); ++it) {
        if (it->record->active_leases == 0) {
            auto forward_it = std::next(it).base(); // converts reverse_iterator to forward iterator
            const size_t freed = forward_it->record->allocated_bytes;
            active_cache_bytes_ = (active_cache_bytes_ >= freed) ? (active_cache_bytes_ - freed) : 0;
            total_bytes_ = (total_bytes_ >= freed) ? (total_bytes_ - freed) : 0;
            forward_it->record->in_cache = false;

            map_.erase(forward_it->key);
            lru_list_.erase(forward_it);
            evictions_++;
            return true;
        }
    }
    return false;
}

void xkv_b_tile_cache::on_lease_released(const std::shared_ptr<b_tile_record> & rec) {
    std::lock_guard<std::mutex> lock(mtx_);
    rec->active_leases--;
    if (!rec->in_cache && rec->active_leases == 0) {
        // Leased record that was evicted or cleared is now fully released
        if (total_bytes_ >= rec->allocated_bytes) {
            total_bytes_ -= rec->allocated_bytes;
        } else {
            total_bytes_ = 0;
        }
    }
}

void xkv_b_tile_cache::invalidate_segment(uint64_t segment_id, uint64_t segment_version) {
    std::lock_guard<std::mutex> lock(mtx_);
    for (auto it = lru_list_.begin(); it != lru_list_.end();) {
        if (it->key.segment_id == segment_id && (segment_version == 0 || it->key.segment_version == segment_version)) {
            active_cache_bytes_ -= it->record->allocated_bytes;
            it->record->in_cache = false;
            if (it->record->active_leases == 0) {
                total_bytes_ -= it->record->allocated_bytes;
            }
            map_.erase(it->key);
            it = lru_list_.erase(it);
        } else {
            ++it;
        }
    }
}

void xkv_b_tile_cache::invalidate_epoch(uint64_t content_epoch, uint64_t codec_epoch) {
    // B tiles are content/bind-epoch independent: only a codec teardown evicts.
    // content_epoch is accepted for wrapper ABI stability and ignored.
    (void) content_epoch;
    std::lock_guard<std::mutex> lock(mtx_);
    for (auto it = lru_list_.begin(); it != lru_list_.end();) {
        if (it->key.codec_epoch != codec_epoch) {
            active_cache_bytes_ -= it->record->allocated_bytes;
            it->record->in_cache = false;
            if (it->record->active_leases == 0) {
                total_bytes_ -= it->record->allocated_bytes;
            }
            map_.erase(it->key);
            it = lru_list_.erase(it);
        } else {
            ++it;
        }
    }
}

void xkv_b_tile_cache::clear() {
    std::lock_guard<std::mutex> lock(mtx_);
    for (auto & e : lru_list_) {
        e.record->in_cache = false;
        if (e.record->active_leases == 0) {
            if (total_bytes_ >= e.record->allocated_bytes) {
                total_bytes_ -= e.record->allocated_bytes;
            } else {
                total_bytes_ = 0;
            }
        }
    }
    lru_list_.clear();
    map_.clear();
    active_cache_bytes_ = 0;
}

size_t xkv_b_tile_cache::total_allocated_bytes() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return total_bytes_;
}

size_t xkv_b_tile_cache::active_cache_bytes() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return active_cache_bytes_;
}

size_t xkv_b_tile_cache::leased_bytes() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return (total_bytes_ >= active_cache_bytes_) ? (total_bytes_ - active_cache_bytes_) : 0;
}

size_t xkv_b_tile_cache::max_bytes() const {
    return max_bytes_;
}

uint64_t xkv_b_tile_cache::hit_count() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return hits_;
}

uint64_t xkv_b_tile_cache::miss_count() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return misses_;
}

uint64_t xkv_b_tile_cache::eviction_count() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return evictions_;
}

// ============================================================================
// xkv_workspace_scratch_lease & xkv_reader_workspace
// ============================================================================

void xkv_workspace_scratch_lease::release() {
    if (ws_ && scratch_.valid) {
        ws_->release(scratch_);
        ws_ = nullptr;
        scratch_.valid = false;
    }
}

xkv_reader_workspace::xkv_reader_workspace(size_t capacity_bytes, bool preallocate)
    : capacity_bytes_(capacity_bytes) {
    if (preallocate && capacity_bytes_ > 0) {
        std::string err;
        if (!warmup(capacity_bytes_, &err)) {
            throw std::invalid_argument("xkv_reader_workspace: warmup failed: " + err);
        }
    }
}

xkv_reader_workspace::xkv_reader_workspace(const llama_cparams & cparams)
    : capacity_bytes_((size_t) cparams.xkv_workspace_mib * 1024 * 1024) {
    xkv_reader_workspace_config cfg;
    cfg.capacity_bytes = capacity_bytes_;
    cfg.max_queries = 64;
    cfg.max_q_heads = 32;
    cfg.max_head_dim_k = 256;
    cfg.max_head_dim_v = 256;
    cfg.max_tile_size = 128;
    cfg.max_rank_k = cparams.xkv_rank_k > 0 ? xkv_compute_padded_rank(cparams.xkv_rank_k) : 512;
    cfg.max_rank_v = cparams.xkv_rank_v > 0 ? xkv_compute_padded_rank(cparams.xkv_rank_v) : 512;
    cfg.max_csr_entries = 8192;
    std::string err;
    if (!warmup(cfg, &err)) {
        throw std::invalid_argument("xkv_reader_workspace: warmup failed: " + err);
    }
}

xkv_reader_workspace::xkv_reader_workspace(const llama_xkv_cache_store & store)
    : xkv_reader_workspace(store.get_cparams()) {}

// Pure checked layout computation shared by setup_layout_locked and the public planner
// (definition follows in the preflight section; must stay in lockstep with acquire()).
struct xkv_workspace_layout {
    size_t o_states_accum = 0;
    size_t o_tile_a_k = 0;
    size_t o_tile_a_v = 0;
    size_t o_tile_b_k = 0;
    size_t o_tile_b_v = 0;
    size_t o_tile_k = 0;
    size_t o_tile_v = 0;
    size_t o_transformed_k = 0;
    size_t o_tile_row_indices = 0;
    size_t o_b_row_indices = 0;
    size_t o_csr_entries = 0;
    size_t o_csr_refs = 0;
    size_t o_decode_tmp = 0;
    size_t decode_tmp_bytes = 0;
    size_t total = 0;
};
static bool compute_workspace_layout(const xkv_reader_workspace_config & cfg, xkv_workspace_layout & out, std::string * err);

bool xkv_reader_workspace::setup_layout_locked(const xkv_reader_workspace_config & cfg, std::string * err) {
    if (active_lease_) {
        if (err) *err = "xkv_reader_workspace: cannot reconfigure workspace while lease is active";
        return false;
    }

    size_t req_capacity = cfg.capacity_bytes > 0 ? cfg.capacity_bytes : capacity_bytes_;
    if (req_capacity == 0) {
        if (err) *err = "xkv_reader_workspace: capacity cannot be 0";
        return false;
    }

    // Single source of truth with xkv_estimate_workspace_layout: identical checked math.
    xkv_workspace_layout lay;
    if (!compute_workspace_layout(cfg, lay, err)) {
        return false;
    }
    size_t cur = lay.total;

    // HARD REJECT if required layout exceeds requested capacity: NEVER silently enlarge!
    if (cur > req_capacity) {
        if (err) {
            *err = "xkv_reader_workspace: layout requirement (" + std::to_string(cur) +
                   " bytes) exceeds configured capacity (" + std::to_string(req_capacity) + " bytes)";
        }
        return false;
    }

    size_t total_heads = 0;
    if (!safe_mul(static_cast<size_t>(cfg.max_queries), static_cast<size_t>(cfg.max_q_heads), total_heads)) {
        if (err) *err = "xkv_reader_workspace: overflow computing total_heads";
        return false;
    }

    // Prepare all temporary buffers FIRST so setup is atomic:
    // If allocation throws or fails, prior workspace state is left completely untouched.
    std::vector<uint8_t> new_buffer(req_capacity, 0);
    std::vector<xkv_online_softmax_state> new_states(total_heads);
    std::vector<xkv_csr_lookup_span> new_csr_spans(cfg.max_queries);

    // Atomically commit all fields
    cfg_ = cfg;
    capacity_bytes_ = req_capacity;
    fixed_layout_total_bytes_ = cur;
    offset_states_accum_ = lay.o_states_accum;
    offset_tile_a_k_ = lay.o_tile_a_k;
    offset_tile_a_v_ = lay.o_tile_a_v;
    offset_tile_b_k_ = lay.o_tile_b_k;
    offset_tile_b_v_ = lay.o_tile_b_v;
    offset_tile_k_ = lay.o_tile_k;
    offset_tile_v_ = lay.o_tile_v;
    offset_transformed_k_ = lay.o_transformed_k;
    offset_tile_row_indices_ = lay.o_tile_row_indices;
    offset_b_row_indices_ = lay.o_b_row_indices;
    offset_csr_entries_ = lay.o_csr_entries;
    offset_csr_refs_ = lay.o_csr_refs;

    // Owned warmup replaces any external carve.
    ext_base_ = nullptr;
    ext_size_ = 0;
    offset_decode_tmp_ = lay.o_decode_tmp;
    decode_tmp_bytes_ = lay.decode_tmp_bytes;

    buffer_ = std::move(new_buffer);
    state_storage_ = std::move(new_states);
    csr_spans_storage_ = std::move(new_csr_spans);
    warmed_up_ = true;
    return true;
}

bool xkv_reader_workspace::warmup(const xkv_reader_workspace_config & cfg, std::string * err) {
    std::lock_guard<std::mutex> lock(mtx_);
    return setup_layout_locked(cfg, err);
}

bool xkv_reader_workspace::warmup(size_t capacity_bytes, std::string * err) {
    std::lock_guard<std::mutex> lock(mtx_);
    xkv_reader_workspace_config cfg;
    if (capacity_bytes > 0) {
        cfg.capacity_bytes = capacity_bytes;
    } else if (capacity_bytes_ > 0) {
        cfg.capacity_bytes = capacity_bytes_;
    } else {
        cfg.capacity_bytes = 16 * 1024 * 1024;
    }
    return setup_layout_locked(cfg, err);
}

bool xkv_reader_workspace::warmup_external(void * backing, size_t backing_bytes,
                                           const xkv_reader_workspace_config & cfg,
                                           std::string * err) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (active_lease_) {
        if (err) *err = "xkv_reader_workspace: cannot reconfigure workspace while lease is active";
        return false;
    }
    if (backing == nullptr || backing_bytes == 0) {
        if (err) *err = "xkv_reader_workspace: external backing cannot be null or empty";
        return false;
    }
    if (reinterpret_cast<uintptr_t>(backing) % 64 != 0) {
        if (err) *err = "xkv_reader_workspace: external backing must be 64-byte aligned";
        return false;
    }
    // Same checked math as the owned path; never silently enlarge or truncate.
    xkv_workspace_layout lay;
    if (!compute_workspace_layout(cfg, lay, err)) {
        return false;
    }
    if (lay.total > backing_bytes) {
        if (err) {
            *err = "xkv_reader_workspace: layout requirement (" + std::to_string(lay.total) +
                   " bytes) exceeds external backing (" + std::to_string(backing_bytes) + " bytes)";
        }
        return false;
    }
    size_t total_heads = 0;
    if (!safe_mul(static_cast<size_t>(cfg.max_queries), static_cast<size_t>(cfg.max_q_heads), total_heads)) {
        if (err) *err = "xkv_reader_workspace: overflow computing total_heads";
        return false;
    }
    // Fallible side-table allocation first: prior state stays untouched on failure.
    std::vector<xkv_online_softmax_state> new_states(total_heads);
    std::vector<xkv_csr_lookup_span> new_csr_spans(cfg.max_queries);
    // Atomically commit the carve; the owned buffer (if any) is released.
    cfg_ = cfg;
    capacity_bytes_ = backing_bytes;
    fixed_layout_total_bytes_ = lay.total;
    offset_states_accum_ = lay.o_states_accum;
    offset_tile_a_k_ = lay.o_tile_a_k;
    offset_tile_a_v_ = lay.o_tile_a_v;
    offset_tile_b_k_ = lay.o_tile_b_k;
    offset_tile_b_v_ = lay.o_tile_b_v;
    offset_tile_k_ = lay.o_tile_k;
    offset_tile_v_ = lay.o_tile_v;
    offset_transformed_k_ = lay.o_transformed_k;
    offset_tile_row_indices_ = lay.o_tile_row_indices;
    offset_b_row_indices_ = lay.o_b_row_indices;
    offset_csr_entries_ = lay.o_csr_entries;
    offset_csr_refs_ = lay.o_csr_refs;
    buffer_.clear();
    buffer_.shrink_to_fit();
    offset_decode_tmp_ = lay.o_decode_tmp;
    decode_tmp_bytes_ = lay.decode_tmp_bytes;
    ext_base_ = static_cast<uint8_t *>(backing);
    ext_size_ = backing_bytes;
    state_storage_ = std::move(new_states);
    csr_spans_storage_ = std::move(new_csr_spans);
    warmed_up_ = true;
    return true;
}

bool xkv_reader_workspace::is_external() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return ext_base_ != nullptr;
}

bool xkv_reader_workspace::preflight(size_t needed_bytes) const {
    std::lock_guard<std::mutex> lock(mtx_);
    if (capacity_bytes_ == 0) return false;
    if (active_lease_) return false; // Enforce exclusive single lease
    size_t total = 0;
    if (!safe_add(live_bytes_, needed_bytes, total)) {
        return false;
    }
    return total <= capacity_bytes_;
}

bool xkv_reader_workspace::acquire(size_t needed_bytes, xkv_reader_scratch & out_scratch) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (capacity_bytes_ == 0) return false;
    if (active_lease_) {
        // Enforce exclusive single lease: reject simultaneous second lease to prevent data races/aliasing
        out_scratch = {};
        return false;
    }

    size_t total = 0;
    if (!safe_add(live_bytes_, needed_bytes, total) || total > capacity_bytes_) {
        out_scratch = {};
        return false;
    }

    if (ext_base_ != nullptr) {
        // Externally carved: explicit warmup_external is required; never silently
        // switch back to an owned buffer (that would reallocate and double-count).
        if (!warmed_up_) {
            out_scratch = {};
            return false;
        }
    } else if (!warmed_up_ || buffer_.size() < capacity_bytes_) {
        if (!setup_layout_locked(cfg_)) {
            out_scratch = {};
            return false;
        }
    }

    live_bytes_ = total;
    active_lease_ = true;
    if (live_bytes_ > peak_bytes_) {
        peak_bytes_ = live_bytes_;
    }

    uint8_t * base = backing_base_locked();
    out_scratch.states_accum = reinterpret_cast<float *>(base + offset_states_accum_);
    out_scratch.states_accum_capacity = (size_t) cfg_.max_queries * cfg_.max_q_heads * cfg_.max_head_dim_v;

    out_scratch.tile_a_k = reinterpret_cast<float *>(base + offset_tile_a_k_);
    out_scratch.tile_a_k_capacity = (size_t) cfg_.max_tile_size * cfg_.max_rank_k;

    out_scratch.tile_a_v = reinterpret_cast<float *>(base + offset_tile_a_v_);
    out_scratch.tile_a_v_capacity = (size_t) cfg_.max_tile_size * cfg_.max_rank_v;

    out_scratch.tile_b_k = reinterpret_cast<float *>(base + offset_tile_b_k_);
    out_scratch.tile_b_k_capacity = (size_t) cfg_.max_head_dim_k * cfg_.max_rank_k;

    out_scratch.tile_b_v = reinterpret_cast<float *>(base + offset_tile_b_v_);
    out_scratch.tile_b_v_capacity = (size_t) cfg_.max_head_dim_v * cfg_.max_rank_v;

    out_scratch.tile_k = reinterpret_cast<float *>(base + offset_tile_k_);
    out_scratch.tile_k_capacity = (size_t) cfg_.max_tile_size * cfg_.max_head_dim_k;

    out_scratch.tile_v = reinterpret_cast<float *>(base + offset_tile_v_);
    out_scratch.tile_v_capacity = (size_t) cfg_.max_tile_size * cfg_.max_head_dim_v;

    out_scratch.transformed_k = reinterpret_cast<float *>(base + offset_transformed_k_);
    out_scratch.transformed_k_capacity = (size_t) cfg_.max_head_dim_k;

    out_scratch.tile_row_indices = reinterpret_cast<uint64_t *>(base + offset_tile_row_indices_);
    out_scratch.tile_row_indices_capacity = (size_t) cfg_.max_tile_size;

    out_scratch.b_row_indices = reinterpret_cast<uint64_t *>(base + offset_b_row_indices_);
    out_scratch.b_row_indices_capacity = (size_t) std::max(cfg_.max_head_dim_k, cfg_.max_head_dim_v);

    out_scratch.csr_entries = reinterpret_cast<xkv_csr_entry *>(base + offset_csr_entries_);
    out_scratch.csr_entries_capacity = (size_t) cfg_.max_csr_entries;

    out_scratch.csr_refs = reinterpret_cast<segment_row_ref *>(base + offset_csr_refs_);
    out_scratch.csr_refs_capacity = (size_t) cfg_.max_csr_entries;

    out_scratch.decode_tmp = base + offset_decode_tmp_;
    out_scratch.decode_tmp_capacity = decode_tmp_bytes_;

    out_scratch.leased_bytes = needed_bytes;
    out_scratch.valid = true;
    return true;
}

xkv_workspace_scratch_lease xkv_reader_workspace::acquire_lease(size_t needed_bytes) {
    xkv_reader_scratch scratch;
    if (acquire(needed_bytes, scratch)) {
        return xkv_workspace_scratch_lease(this, scratch);
    }
    return xkv_workspace_scratch_lease(nullptr, {});
}

void xkv_reader_workspace::release(const xkv_reader_scratch & scratch) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (live_bytes_ >= scratch.leased_bytes) {
        live_bytes_ -= scratch.leased_bytes;
    } else {
        live_bytes_ = 0;
    }
    active_lease_ = false;
}

size_t xkv_reader_workspace::capacity_bytes() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return capacity_bytes_;
}

size_t xkv_reader_workspace::allocated_bytes() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return buffer_.capacity() * sizeof(uint8_t) +
           state_storage_.capacity() * sizeof(xkv_online_softmax_state) +
           csr_spans_storage_.capacity() * sizeof(xkv_csr_lookup_span);
}

size_t xkv_reader_workspace::peak_bytes() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return peak_bytes_;
}

size_t xkv_reader_workspace::live_bytes() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return live_bytes_;
}

bool xkv_reader_workspace::is_warmed_up() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return warmed_up_;
}

bool xkv_reader_workspace::has_active_lease() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return active_lease_;
}

const xkv_reader_workspace_config & xkv_reader_workspace::config() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return cfg_;
}

void xkv_reader_workspace::reset_peak() {
    std::lock_guard<std::mutex> lock(mtx_);
    peak_bytes_ = live_bytes_;
}

void xkv_reader_workspace::clear() {
    std::lock_guard<std::mutex> lock(mtx_);
    live_bytes_ = 0;
    peak_bytes_ = 0;
    warmed_up_ = false;
    active_lease_ = false;
    buffer_.clear();
    buffer_.shrink_to_fit();
    // Drop any external carve without freeing caller memory.
    ext_base_ = nullptr;
    ext_size_ = 0;
    decode_tmp_bytes_ = 0;
    state_storage_.clear();
    state_storage_.shrink_to_fit();
    csr_spans_storage_.clear();
    csr_spans_storage_.shrink_to_fit();
}

const void * xkv_reader_workspace::scratch_ptr() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return backing_base_locked();
}

const float * xkv_reader_workspace::tile_k_buffer() const {
    std::lock_guard<std::mutex> lock(mtx_);
    const uint8_t * base = backing_base_locked();
    return base ? reinterpret_cast<const float *>(base + offset_tile_k_) : nullptr;
}

const float * xkv_reader_workspace::tile_v_buffer() const {
    std::lock_guard<std::mutex> lock(mtx_);
    const uint8_t * base = backing_base_locked();
    return base ? reinterpret_cast<const float *>(base + offset_tile_v_) : nullptr;
}

const float * xkv_reader_workspace::tile_a_k_buffer() const {
    std::lock_guard<std::mutex> lock(mtx_);
    const uint8_t * base = backing_base_locked();
    return base ? reinterpret_cast<const float *>(base + offset_tile_a_k_) : nullptr;
}

const float * xkv_reader_workspace::tile_a_v_buffer() const {
    std::lock_guard<std::mutex> lock(mtx_);
    const uint8_t * base = backing_base_locked();
    return base ? reinterpret_cast<const float *>(base + offset_tile_a_v_) : nullptr;
}

const float * xkv_reader_workspace::tile_b_k_buffer() const {
    std::lock_guard<std::mutex> lock(mtx_);
    const uint8_t * base = backing_base_locked();
    return base ? reinterpret_cast<const float *>(base + offset_tile_b_k_) : nullptr;
}

const float * xkv_reader_workspace::tile_b_v_buffer() const {
    std::lock_guard<std::mutex> lock(mtx_);
    const uint8_t * base = backing_base_locked();
    return base ? reinterpret_cast<const float *>(base + offset_tile_b_v_) : nullptr;
}

const xkv_csr_entry * xkv_reader_workspace::csr_entries_buffer() const {
    std::lock_guard<std::mutex> lock(mtx_);
    const uint8_t * base = backing_base_locked();
    return base ? reinterpret_cast<const xkv_csr_entry *>(base + offset_csr_entries_) : nullptr;
}

const segment_row_ref * xkv_reader_workspace::csr_refs_buffer() const {
    std::lock_guard<std::mutex> lock(mtx_);
    const uint8_t * base = backing_base_locked();
    return base ? reinterpret_cast<const segment_row_ref *>(base + offset_csr_refs_) : nullptr;
}

xkv_online_softmax_state * xkv_reader_workspace::get_states_storage() {
    std::lock_guard<std::mutex> lock(mtx_);
    return state_storage_.empty() ? nullptr : state_storage_.data();
}

size_t xkv_reader_workspace::get_states_storage_capacity() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return state_storage_.size();
}

xkv_csr_lookup_span * xkv_reader_workspace::get_csr_spans_storage() {
    std::lock_guard<std::mutex> lock(mtx_);
    return csr_spans_storage_.empty() ? nullptr : csr_spans_storage_.data();
}

size_t xkv_reader_workspace::get_csr_spans_storage_capacity() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return csr_spans_storage_.size();
}

// ============================================================================
// GEMM tile reconstruction & helper
// ============================================================================

static void gemm_reconstruct_tile(
    const float * a_data,
    size_t padded_rank_a,
    const float * b_feat_data,
    size_t padded_rank_b,
    uint32_t logical_rank,
    size_t n_rows,
    size_t n_features,
    float * dst
) {
    for (size_t r = 0; r < n_rows; ++r) {
        const float * a_row = a_data + r * padded_rank_a;
        float * out_row = dst + r * n_features;
        for (size_t f = 0; f < n_features; ++f) {
            const float * b_row = b_feat_data + f * padded_rank_b;
            double sum = 0.0;
            for (size_t k = 0; k < logical_rank; ++k) {
                sum += static_cast<double>(a_row[k]) * static_cast<double>(b_row[k]);
            }
            out_row[f] = static_cast<float>(sum);
        }
    }
}

static const float * get_q_ptr(
    const xkv_query_input & query,
    uint32_t group_index,
    uint32_t qh
) {
    uint32_t offset = qh * query.head_dim_k;
    if (!query.q_groups.empty()) {
        if (group_index < query.q_groups.size() && !query.q_groups[group_index].empty()) {
            return query.q_groups[group_index].data() + offset;
        }
        return nullptr; // Out of bounds DDVR group index is invalid
    }
    if (!query.q_vec.empty()) {
        if (group_index != 0) {
            return nullptr; // No q_groups provided, only group 0 is valid
        }
        return query.q_vec.data() + offset;
    }
    return nullptr;
}

static float dot_product(const float * a, const float * b, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return static_cast<float>(sum);
}

// ============================================================================
// Checked Workspace Preflight (single and batch)
// ============================================================================

// ============================================================================
// Strict bounded-mode helpers: fail closed before output or cache mutation
// ============================================================================

namespace {

// Strict bounded mode: a workspace is configured and the caller did not opt into heap
// fallbacks. Every insufficient slice fails closed with workspace_exhausted; fallback heap
// vectors are never allocated. Reads without a configured workspace keep the legacy oracle
// behavior (heap scratch allowed) so unwired callers are unaffected.
inline bool strict_mode(const xkv_reader_config & config) {
    return config.workspace != nullptr && !config.reference_allow_heap;
}

// Bounded factor-row decode: strict mode uses the heap-free 8-arg overload with the
// workspace decode_tmp slice (exact needs verified upfront by check_strict_decode_tmp);
// legacy oracle mode keeps the 6-arg form with its internal scratch. In strict mode
// scratch is always non-null (lease verified before any decode).
inline void decode_rows_bounded(const encoded_matrix & em,
                                const uint64_t * row_indices,
                                size_t n_rows,
                                float * dst,
                                size_t dst_capacity_elements,
                                const xkv_reader_config & config,
                                const xkv_reader_scratch * scratch) {
    if (strict_mode(config)) {
        decode_rows(em, row_indices, n_rows, dst, dst_capacity_elements,
                    scratch->decode_tmp, scratch->decode_tmp_capacity,
                    value_domain::canonical);
    } else {
        decode_rows(em, row_indices, n_rows, dst, dst_capacity_elements,
                    value_domain::canonical);
    }
}

inline bool checked_u64_to_size(uint64_t v, size_t & out) {
    if (v > static_cast<uint64_t>(SIZE_MAX)) return false;
    out = static_cast<size_t>(v);
    return true;
}

// need = a * b must be exactly representable and fit in cap (elements).
inline bool slice_mul_le(size_t a, size_t b, size_t cap, size_t & out_need) {
    if (!safe_mul(a, b, out_need)) return false;
    return out_need <= cap;
}

inline uint32_t effective_tile_size(const xkv_reader_config & config) {
    return (config.tile_size == 0) ? 32 : config.tile_size;
}

} // namespace

// Pure checked layout computation shared by setup_layout_locked and the public planner.
// Must stay in lockstep with the scratch capacities published by acquire().
// (Struct declared ahead of setup_layout_locked; definition follows.)

static bool compute_workspace_layout(const xkv_reader_workspace_config & cfg, xkv_workspace_layout & out, std::string * err) {
    xkv_workspace_layout lay;
    size_t cur = 0;
    size_t sz = 0;

    lay.o_states_accum = cur;
    if (!safe_mul(static_cast<size_t>(cfg.max_queries), static_cast<size_t>(cfg.max_q_heads), sz) ||
        !safe_mul(sz, static_cast<size_t>(cfg.max_head_dim_v), sz) ||
        !safe_mul(sz, sizeof(float), sz) ||
        !safe_add(cur, sz, cur) ||
        !safe_align(cur, 64, cur)) {
        if (err) *err = "xkv_reader_workspace: overflow computing states_accum layout";
        return false;
    }
    lay.o_tile_a_k = cur;
    if (!safe_mul(static_cast<size_t>(cfg.max_tile_size), static_cast<size_t>(cfg.max_rank_k), sz) ||
        !safe_mul(sz, sizeof(float), sz) ||
        !safe_add(cur, sz, cur) ||
        !safe_align(cur, 64, cur)) {
        if (err) *err = "xkv_reader_workspace: overflow computing tile_a_k layout";
        return false;
    }
    lay.o_tile_a_v = cur;
    if (!safe_mul(static_cast<size_t>(cfg.max_tile_size), static_cast<size_t>(cfg.max_rank_v), sz) ||
        !safe_mul(sz, sizeof(float), sz) ||
        !safe_add(cur, sz, cur) ||
        !safe_align(cur, 64, cur)) {
        if (err) *err = "xkv_reader_workspace: overflow computing tile_a_v layout";
        return false;
    }
    lay.o_tile_b_k = cur;
    if (!safe_mul(static_cast<size_t>(cfg.max_head_dim_k), static_cast<size_t>(cfg.max_rank_k), sz) ||
        !safe_mul(sz, sizeof(float), sz) ||
        !safe_add(cur, sz, cur) ||
        !safe_align(cur, 64, cur)) {
        if (err) *err = "xkv_reader_workspace: overflow computing tile_b_k layout";
        return false;
    }
    lay.o_tile_b_v = cur;
    if (!safe_mul(static_cast<size_t>(cfg.max_head_dim_v), static_cast<size_t>(cfg.max_rank_v), sz) ||
        !safe_mul(sz, sizeof(float), sz) ||
        !safe_add(cur, sz, cur) ||
        !safe_align(cur, 64, cur)) {
        if (err) *err = "xkv_reader_workspace: overflow computing tile_b_v layout";
        return false;
    }
    lay.o_tile_k = cur;
    if (!safe_mul(static_cast<size_t>(cfg.max_tile_size), static_cast<size_t>(cfg.max_head_dim_k), sz) ||
        !safe_mul(sz, sizeof(float), sz) ||
        !safe_add(cur, sz, cur) ||
        !safe_align(cur, 64, cur)) {
        if (err) *err = "xkv_reader_workspace: overflow computing tile_k layout";
        return false;
    }
    lay.o_tile_v = cur;
    if (!safe_mul(static_cast<size_t>(cfg.max_tile_size), static_cast<size_t>(cfg.max_head_dim_v), sz) ||
        !safe_mul(sz, sizeof(float), sz) ||
        !safe_add(cur, sz, cur) ||
        !safe_align(cur, 64, cur)) {
        if (err) *err = "xkv_reader_workspace: overflow computing tile_v layout";
        return false;
    }
    lay.o_transformed_k = cur;
    if (!safe_mul(static_cast<size_t>(cfg.max_head_dim_k), sizeof(float), sz) ||
        !safe_add(cur, sz, cur) ||
        !safe_align(cur, 64, cur)) {
        if (err) *err = "xkv_reader_workspace: overflow computing transformed_k layout";
        return false;
    }
    lay.o_tile_row_indices = cur;
    if (!safe_mul(static_cast<size_t>(cfg.max_tile_size), sizeof(uint64_t), sz) ||
        !safe_add(cur, sz, cur) ||
        !safe_align(cur, 64, cur)) {
        if (err) *err = "xkv_reader_workspace: overflow computing tile_row_indices layout";
        return false;
    }
    lay.o_b_row_indices = cur;
    {
        size_t max_b_dim = std::max(cfg.max_head_dim_k, cfg.max_head_dim_v);
        if (!safe_mul(max_b_dim, sizeof(uint64_t), sz) ||
            !safe_add(cur, sz, cur) ||
            !safe_align(cur, 64, cur)) {
            if (err) *err = "xkv_reader_workspace: overflow computing b_row_indices layout";
            return false;
        }
    }
    lay.o_csr_refs = cur;
    if (!safe_mul(static_cast<size_t>(cfg.max_csr_entries), sizeof(segment_row_ref), sz) ||
        !safe_add(cur, sz, cur) ||
        !safe_align(cur, 64, cur)) {
        if (err) *err = "xkv_reader_workspace: overflow computing csr_refs layout";
        return false;
    }
    lay.o_csr_entries = cur;
    if (!safe_mul(static_cast<size_t>(cfg.max_csr_entries), sizeof(xkv_csr_entry), sz) ||
        !safe_add(cur, sz, cur) ||
        !safe_align(cur, 64, cur)) {
        if (err) *err = "xkv_reader_workspace: overflow computing csr_entries layout";
        return false;
    }
    // 12. decode_tmp: explicit cap, or one derived max-rank FP32 row.
    lay.o_decode_tmp = cur;
    {
        size_t tmp_cap = static_cast<size_t>(cfg.max_decode_tmp_bytes);
        if (tmp_cap == 0) {
            size_t max_rank = std::max(cfg.max_rank_k, cfg.max_rank_v);
            if (!safe_mul(static_cast<size_t>(max_rank), sizeof(float), tmp_cap)) {
                if (err) *err = "xkv_reader_workspace: overflow computing decode_tmp layout";
                return false;
            }
        }
        if (!safe_add(cur, tmp_cap, cur) ||
            !safe_align(cur, 64, cur)) {
            if (err) *err = "xkv_reader_workspace: overflow computing decode_tmp layout";
            return false;
        }
        lay.decode_tmp_bytes = tmp_cap;
    }
    lay.total = cur;
    out = lay;
    return true;
}

bool xkv_estimate_workspace_layout(const xkv_reader_workspace_config & cfg, size_t & out_total_bytes, std::string * err) {
    out_total_bytes = 0;
    xkv_workspace_layout lay;
    if (!compute_workspace_layout(cfg, lay, err)) {
        return false;
    }
    out_total_bytes = lay.total;
    return true;
}

// Resolved per-view padded widths (size_t) for slice checks. Returns false only when a
// resolvable view carries widths that do not fit in size_t; unresolvable views are skipped
// by the caller exactly like the main decode loop skips them.
struct xkv_view_widths {
    size_t pad_rank_k = 0;
    size_t pad_rank_v = 0;
    size_t pad_b_k = 0;
    size_t pad_b_v = 0;
};

static bool resolve_view_widths(const xkv_segment_read_view & view, xkv_view_widths & out) {
    const auto * g = view.resolve_group(nullptr);
    if (!g || !g->b_k || !g->b_v) return false;
    return checked_u64_to_size(g->a_k.desc.padded_shape.cols, out.pad_rank_k) &&
           checked_u64_to_size(g->a_v.desc.padded_shape.cols, out.pad_rank_v) &&
           checked_u64_to_size(g->b_k->desc.padded_shape.cols, out.pad_b_k) &&
           checked_u64_to_size(g->b_v->desc.padded_shape.cols, out.pad_b_v);
}

// Verify every scratch slice a strict single read will touch. Must run before any decode,
// cache mutation, or output so a short slice fails closed deterministically.
static bool check_single_strict_slices(
    const xkv_query_input & query,
    const std::vector<xkv_segment_read_view> & segment_views,
    const xkv_reader_config & config,
    const xkv_reader_scratch & scratch,
    std::string * err) {
    const xkv_reader_workspace * ws = config.workspace;
    auto fail = [err](const std::string & msg) -> bool {
        if (err) *err = msg;
        return false;
    };
    size_t need = 0;
    if (query.n_q_heads > ws->get_states_storage_capacity()) {
        return fail("xkv_read_attention: workspace states storage short for n_q_heads");
    }
    if (!slice_mul_le(query.n_q_heads, query.head_dim_v, scratch.states_accum_capacity, need)) {
        return fail("xkv_read_attention: workspace states_accum short (need " + std::to_string(need) + ")");
    }
    if (std::max(query.head_dim_k, query.head_dim_v) > scratch.b_row_indices_capacity) {
        return fail("xkv_read_attention: workspace b_row_indices short");
    }
    if (query.head_dim_k > scratch.transformed_k_capacity) {
        return fail("xkv_read_attention: workspace transformed_k short");
    }
    const uint32_t tile_size = effective_tile_size(config);
    for (const auto & view : segment_views) {
        if (view.get_segment() == nullptr || view.selected_rows.empty()) continue;
        xkv_view_widths w;
        if (!resolve_view_widths(view, w)) {
            const auto * g = view.resolve_group(nullptr);
            if (!g || !g->b_k || !g->b_v) continue; // skipped by the decode loop as well
            return fail("xkv_read_attention: segment rank exceeds size_t");
        }
        if (!slice_mul_le(query.head_dim_k, w.pad_b_k, scratch.tile_b_k_capacity, need)) {
            return fail("xkv_read_attention: workspace tile_b_k short (need " + std::to_string(need) + ")");
        }
        if (!slice_mul_le(query.head_dim_v, w.pad_b_v, scratch.tile_b_v_capacity, need)) {
            return fail("xkv_read_attention: workspace tile_b_v short (need " + std::to_string(need) + ")");
        }
        const uint32_t cur_tile = std::min(tile_size, static_cast<uint32_t>(view.selected_rows.size()));
        if (!slice_mul_le(cur_tile, w.pad_rank_k, scratch.tile_a_k_capacity, need)) {
            return fail("xkv_read_attention: workspace tile_a_k short (need " + std::to_string(need) + ")");
        }
        if (!slice_mul_le(cur_tile, w.pad_rank_v, scratch.tile_a_v_capacity, need)) {
            return fail("xkv_read_attention: workspace tile_a_v short (need " + std::to_string(need) + ")");
        }
        if (!slice_mul_le(cur_tile, query.head_dim_k, scratch.tile_k_capacity, need)) {
            return fail("xkv_read_attention: workspace tile_k short (need " + std::to_string(need) + ")");
        }
        if (!slice_mul_le(cur_tile, query.head_dim_v, scratch.tile_v_capacity, need)) {
            return fail("xkv_read_attention: workspace tile_v short (need " + std::to_string(need) + ")");
        }
        if (cur_tile > scratch.tile_row_indices_capacity) {
            return fail("xkv_read_attention: workspace tile_row_indices short");
        }
    }
    return true;
}

// Batch variant: uniform head dims (validated by the caller), summed states, union CSR refs.
static bool check_batch_strict_slices(
    const std::vector<xkv_query_input> & queries,
    const std::vector<xkv_segment_read_view> & segment_views,
    const sr_batch_selection_result & batch_selection,
    uint32_t batch_head_dim_k,
    uint32_t batch_head_dim_v,
    const xkv_reader_config & config,
    const xkv_reader_scratch & scratch,
    std::string * err) {
    const xkv_reader_workspace * ws = config.workspace;
    auto fail = [err](const std::string & msg) -> bool {
        if (err) *err = msg;
        return false;
    };
    size_t need = 0;
    size_t total_heads = 0;
    for (const auto & q : queries) {
        if (!safe_add(total_heads, static_cast<size_t>(q.n_q_heads), total_heads)) {
            return fail("xkv_read_attention_batch: overflow summing batch heads");
        }
    }
    if (total_heads > ws->get_states_storage_capacity()) {
        return fail("xkv_read_attention_batch: workspace states storage short for batch heads");
    }
    if (!slice_mul_le(total_heads, batch_head_dim_v, scratch.states_accum_capacity, need)) {
        return fail("xkv_read_attention_batch: workspace states_accum short (need " + std::to_string(need) + ")");
    }
    if (batch_selection.csr_indices.size() > scratch.csr_refs_capacity) {
        return fail("xkv_read_attention_batch: workspace csr_refs short");
    }
    if (batch_selection.csr_indices.size() > scratch.csr_entries_capacity) {
        return fail("xkv_read_attention_batch: workspace csr_entries short");
    }
    if (std::max(batch_head_dim_k, batch_head_dim_v) > scratch.b_row_indices_capacity) {
        return fail("xkv_read_attention_batch: workspace b_row_indices short");
    }
    if (batch_head_dim_k > scratch.transformed_k_capacity) {
        return fail("xkv_read_attention_batch: workspace transformed_k short");
    }
    const uint32_t tile_size = effective_tile_size(config);
    for (const auto & view : segment_views) {
        if (view.get_segment() == nullptr || view.selected_rows.empty()) continue;
        xkv_view_widths w;
        if (!resolve_view_widths(view, w)) {
            const auto * g = view.resolve_group(nullptr);
            if (!g || !g->b_k || !g->b_v) continue;
            return fail("xkv_read_attention_batch: segment rank exceeds size_t");
        }
        if (!slice_mul_le(batch_head_dim_k, w.pad_b_k, scratch.tile_b_k_capacity, need)) {
            return fail("xkv_read_attention_batch: workspace tile_b_k short (need " + std::to_string(need) + ")");
        }
        if (!slice_mul_le(batch_head_dim_v, w.pad_b_v, scratch.tile_b_v_capacity, need)) {
            return fail("xkv_read_attention_batch: workspace tile_b_v short (need " + std::to_string(need) + ")");
        }
        const uint32_t cur_tile = std::min(tile_size, static_cast<uint32_t>(view.selected_rows.size()));
        if (!slice_mul_le(cur_tile, w.pad_rank_k, scratch.tile_a_k_capacity, need)) {
            return fail("xkv_read_attention_batch: workspace tile_a_k short (need " + std::to_string(need) + ")");
        }
        if (!slice_mul_le(cur_tile, w.pad_rank_v, scratch.tile_a_v_capacity, need)) {
            return fail("xkv_read_attention_batch: workspace tile_a_v short (need " + std::to_string(need) + ")");
        }
        if (!slice_mul_le(cur_tile, batch_head_dim_k, scratch.tile_k_capacity, need)) {
            return fail("xkv_read_attention_batch: workspace tile_k short (need " + std::to_string(need) + ")");
        }
        if (!slice_mul_le(cur_tile, batch_head_dim_v, scratch.tile_v_capacity, need)) {
            return fail("xkv_read_attention_batch: workspace tile_v short (need " + std::to_string(need) + ")");
        }
        if (cur_tile > scratch.tile_row_indices_capacity) {
            return fail("xkv_read_attention_batch: workspace tile_row_indices short");
        }
    }
    return true;
}

// Exact decode-tmp needs for every segment touched by a strict read, checked BEFORE
// any decode. Returns invalid_argument for bad descriptors, workspace_exceeded when a
// need exceeds the decode_tmp region, success otherwise. F32/Turbo need 0 bytes.
static xkv_read_status check_strict_decode_tmp(
    const std::vector<xkv_segment_read_view> & segment_views,
    const xkv_reader_scratch & scratch,
    std::string * err) {
    static const char * kNames[4] = {"A_K", "A_V", "B_K", "B_V"};
    for (const auto & view : segment_views) {
        if (view.get_segment() == nullptr || view.selected_rows.empty()) continue;
        const auto * g = view.resolve_group(nullptr);
        if (!g || !g->b_k || !g->b_v) continue; // skipped by the decode loop as well
        const codec_desc * descs[4] = {&g->a_k.desc, &g->a_v.desc, &g->b_k->desc, &g->b_v->desc};
        for (int d = 0; d < 4; ++d) {
            size_t need = 0;
            std::string derr;
            if (!decode_rows_scratch_bytes(*descs[d], need, &derr)) {
                if (err) *err = std::string("decode tmp size query failed for ") + kNames[d] + ": " + derr;
                return xkv_read_status::invalid_argument;
            }
            if (need > scratch.decode_tmp_capacity) {
                if (err) *err = std::string("workspace decode_tmp short for ") + kNames[d] +
                    " (need " + std::to_string(need) + ")";
                return xkv_read_status::workspace_exceeded;
            }
        }
    }
    return xkv_read_status::success;
}

bool xkv_preflight_workspace(
    const xkv_query_input & query,
    const std::vector<xkv_segment_read_view> & segment_views,
    const xkv_reader_config & config,
    size_t & out_peak_workspace,
    std::string * err
) {
    out_peak_workspace = 0;

    size_t dv_bytes = 0;
    if (!safe_mul(query.head_dim_v, sizeof(float), dv_bytes)) {
        if (err) *err = "xkv_preflight_workspace: overflow computing state dv_bytes";
        return false;
    }

    size_t state_one = 0;
    if (!safe_add(sizeof(xkv_online_softmax_state), dv_bytes, state_one)) {
        if (err) *err = "xkv_preflight_workspace: overflow computing state_one";
        return false;
    }

    size_t state_bytes = 0;
    if (!safe_mul(query.n_q_heads, state_one, state_bytes)) {
        if (err) *err = "xkv_preflight_workspace: overflow computing state_bytes";
        return false;
    }

    size_t out_bytes = 0;
    if (!safe_mul(query.n_q_heads, dv_bytes, out_bytes)) {
        if (err) *err = "xkv_preflight_workspace: overflow computing out_bytes";
        return false;
    }

    size_t max_tile_ws = 0;
    const uint32_t tile_size = (config.tile_size == 0) ? 32 : config.tile_size;

    for (const auto & view : segment_views) {
        if (view.selected_rows.empty()) continue;
        const auto * g = view.resolve_group(nullptr);
        if (!g || !g->b_k || !g->b_v) continue;

        uint32_t cur_tile = std::min(tile_size, static_cast<uint32_t>(view.selected_rows.size()));
        uint64_t pad_rank_k = g->a_k.desc.padded_shape.cols;
        uint64_t pad_rank_v = g->a_v.desc.padded_shape.cols;
        uint64_t pad_b_k = g->b_k->desc.padded_shape.cols;
        uint64_t pad_b_v = g->b_v->desc.padded_shape.cols;

        size_t b_k_bytes = 0, b_v_bytes = 0, a_k_bytes = 0, a_v_bytes = 0;
        size_t k_tile_bytes = 0, v_tile_bytes = 0, phase_bytes = 0, indices_bytes = 0;

        // All factor products use checked arithmetic: padded widths come from codec
        // descriptors and must never overflow before the checked multiply.
        size_t pad_b_k_bytes = 0, pad_b_v_bytes = 0, pad_rank_k_bytes = 0, pad_rank_v_bytes = 0;
        size_t dk_bytes = 0, dv_bytes_tile = 0;
        if (!safe_mul(pad_b_k, sizeof(float), pad_b_k_bytes) ||
            !safe_mul(pad_b_v, sizeof(float), pad_b_v_bytes) ||
            !safe_mul(pad_rank_k, sizeof(float), pad_rank_k_bytes) ||
            !safe_mul(pad_rank_v, sizeof(float), pad_rank_v_bytes) ||
            !safe_mul(static_cast<size_t>(query.head_dim_k), sizeof(float), dk_bytes) ||
            !safe_mul(static_cast<size_t>(query.head_dim_v), sizeof(float), dv_bytes_tile)) {
            if (err) *err = "xkv_preflight_workspace: overflow computing tile workspace terms";
            return false;
        }

        if (!safe_mul(static_cast<size_t>(query.head_dim_k), pad_b_k_bytes, b_k_bytes) ||
            !safe_mul(static_cast<size_t>(query.head_dim_v), pad_b_v_bytes, b_v_bytes) ||
            !safe_mul(static_cast<size_t>(cur_tile), pad_rank_k_bytes, a_k_bytes) ||
            !safe_mul(static_cast<size_t>(cur_tile), pad_rank_v_bytes, a_v_bytes) ||
            !safe_mul(static_cast<size_t>(cur_tile), dk_bytes, k_tile_bytes) ||
            !safe_mul(static_cast<size_t>(cur_tile), dv_bytes_tile, v_tile_bytes) ||
            !safe_mul(static_cast<size_t>(query.head_dim_k), sizeof(float), phase_bytes) ||
            !safe_mul(static_cast<size_t>(cur_tile), sizeof(uint64_t), indices_bytes)) {
            if (err) *err = "xkv_preflight_workspace: overflow computing tile workspace terms";
            return false;
        }

        size_t view_ws = 0;
        if (!safe_add(b_k_bytes, b_v_bytes, view_ws) ||
            !safe_add(view_ws, a_k_bytes, view_ws) ||
            !safe_add(view_ws, a_v_bytes, view_ws) ||
            !safe_add(view_ws, k_tile_bytes, view_ws) ||
            !safe_add(view_ws, v_tile_bytes, view_ws) ||
            !safe_add(view_ws, phase_bytes, view_ws) ||
            !safe_add(view_ws, indices_bytes, view_ws)) {
            if (err) *err = "xkv_preflight_workspace: overflow summing tile workspace";
            return false;
        }

        max_tile_ws = std::max(max_tile_ws, view_ws);
    }

    if (!safe_add(state_bytes, out_bytes, out_peak_workspace) ||
        !safe_add(out_peak_workspace, max_tile_ws, out_peak_workspace)) {
        if (err) *err = "xkv_preflight_workspace: overflow computing total peak workspace";
        return false;
    }

    if (config.workspace_budget_bytes > 0 && out_peak_workspace > config.workspace_budget_bytes) {
        if (err) {
            *err = "xkv_preflight_workspace: estimated peak workspace (" +
                std::to_string(out_peak_workspace) + " bytes) exceeds configured budget (" +
                std::to_string(config.workspace_budget_bytes) + " bytes)";
        }
        return false;
    }

    if (config.workspace != nullptr && !config.workspace->preflight(out_peak_workspace)) {
        if (err) {
            *err = "xkv_preflight_workspace: peak workspace (" +
                std::to_string(out_peak_workspace) + " bytes) exceeds reusable workspace capacity (" +
                std::to_string(config.workspace->capacity_bytes()) + " bytes)";
        }
        return false;
    }

    if (config.workspace != nullptr) {
        const auto & wcfg = config.workspace->config();
        if (query.n_q_heads > wcfg.max_q_heads) {
            if (err) *err = "xkv_preflight_workspace: query.n_q_heads exceeds workspace max_q_heads";
            return false;
        }
        if (query.head_dim_k > wcfg.max_head_dim_k) {
            if (err) *err = "xkv_preflight_workspace: query.head_dim_k exceeds workspace max_head_dim_k";
            return false;
        }
        if (query.head_dim_v > wcfg.max_head_dim_v) {
            if (err) *err = "xkv_preflight_workspace: query.head_dim_v exceeds workspace max_head_dim_v";
            return false;
        }
        uint32_t tile_sz = (config.tile_size == 0) ? 32 : config.tile_size;
        if (tile_sz > wcfg.max_tile_size) {
            if (err) *err = "xkv_preflight_workspace: config.tile_size exceeds workspace max_tile_size";
            return false;
        }
        for (const auto & view : segment_views) {
            const auto * g = view.resolve_group(err);
            if (!g || !g->b_k || !g->b_v) continue;
            if (g->a_k.desc.padded_shape.cols > wcfg.max_rank_k || g->b_k->desc.padded_shape.cols > wcfg.max_rank_k) {
                if (err) *err = "xkv_preflight_workspace: segment rank_k exceeds workspace max_rank_k";
                return false;
            }
            if (g->a_v.desc.padded_shape.cols > wcfg.max_rank_v || g->b_v->desc.padded_shape.cols > wcfg.max_rank_v) {
                if (err) *err = "xkv_preflight_workspace: segment rank_v exceeds workspace max_rank_v";
                return false;
            }
        }
    }

    return true;
}

bool xkv_preflight_batch_workspace(
    const std::vector<xkv_query_input> & queries,
    const std::vector<xkv_segment_read_view> & segment_views,
    const sr_batch_selection_result & batch_selection,
    const xkv_reader_config & config,
    size_t & out_peak_workspace,
    std::string * err
) {
    out_peak_workspace = 0;

    if (queries.empty()) {
        return true;
    }

    // 1. Sum of all concurrent query states and output buffers
    size_t all_states_bytes = 0;
    for (const auto & q : queries) {
        size_t dv_bytes = 0;
        if (!safe_mul(q.head_dim_v, sizeof(float), dv_bytes)) {
            if (err) *err = "xkv_preflight_batch_workspace: overflow in dv_bytes";
            return false;
        }

        size_t state_one = 0;
        if (!safe_add(sizeof(xkv_online_softmax_state), dv_bytes, state_one)) {
            if (err) *err = "xkv_preflight_batch_workspace: overflow in state_one";
            return false;
        }

        size_t q_state_bytes = 0;
        if (!safe_mul(q.n_q_heads, state_one, q_state_bytes)) {
            if (err) *err = "xkv_preflight_batch_workspace: overflow in q_state_bytes";
            return false;
        }

        size_t q_out_bytes = 0;
        if (!safe_mul(q.n_q_heads, dv_bytes, q_out_bytes)) {
            if (err) *err = "xkv_preflight_batch_workspace: overflow in q_out_bytes";
            return false;
        }

        size_t q_total = 0;
        if (!safe_add(q_state_bytes, q_out_bytes, q_total) ||
            !safe_add(all_states_bytes, q_total, all_states_bytes)) {
            if (err) *err = "xkv_preflight_batch_workspace: overflow summing query states";
            return false;
        }
    }

    // 2. CSR lookup sets overhead
    size_t csr_overhead = 0;
    if (!batch_selection.csr_indices.empty()) {
        // Upper bound covers the paired (ref, group) entries plus sort slack.
        if (!safe_mul(batch_selection.csr_indices.size(), sizeof(xkv_csr_entry) + 16, csr_overhead)) {
            if (err) *err = "xkv_preflight_batch_workspace: overflow computing csr_overhead";
            return false;
        }
    }

    // 3. Tile workspace peak for the streaming decode
    size_t max_tile_ws = 0;
    const uint32_t tile_size = (config.tile_size == 0) ? 32 : config.tile_size;
    uint32_t head_dim_k = queries[0].head_dim_k;
    uint32_t head_dim_v = queries[0].head_dim_v;

    for (const auto & view : segment_views) {
        if (view.selected_rows.empty()) continue;
        const auto * g = view.resolve_group(nullptr);
        if (!g || !g->b_k || !g->b_v) continue;

        uint32_t cur_tile = std::min(tile_size, static_cast<uint32_t>(view.selected_rows.size()));
        uint64_t pad_rank_k = g->a_k.desc.padded_shape.cols;
        uint64_t pad_rank_v = g->a_v.desc.padded_shape.cols;
        uint64_t pad_b_k = g->b_k->desc.padded_shape.cols;
        uint64_t pad_b_v = g->b_v->desc.padded_shape.cols;

        size_t b_k_bytes = 0, b_v_bytes = 0, a_k_bytes = 0, a_v_bytes = 0;
        size_t k_tile_bytes = 0, v_tile_bytes = 0, phase_bytes = 0, indices_bytes = 0;

        size_t pad_b_k_bytes = 0, pad_b_v_bytes = 0, pad_rank_k_bytes = 0, pad_rank_v_bytes = 0;
        size_t dk_bytes = 0, dv_bytes_tile = 0;
        if (!safe_mul(pad_b_k, sizeof(float), pad_b_k_bytes) ||
            !safe_mul(pad_b_v, sizeof(float), pad_b_v_bytes) ||
            !safe_mul(pad_rank_k, sizeof(float), pad_rank_k_bytes) ||
            !safe_mul(pad_rank_v, sizeof(float), pad_rank_v_bytes) ||
            !safe_mul(static_cast<size_t>(head_dim_k), sizeof(float), dk_bytes) ||
            !safe_mul(static_cast<size_t>(head_dim_v), sizeof(float), dv_bytes_tile)) {
            if (err) *err = "xkv_preflight_batch_workspace: overflow computing tile workspace terms";
            return false;
        }

        if (!safe_mul(static_cast<size_t>(head_dim_k), pad_b_k_bytes, b_k_bytes) ||
            !safe_mul(static_cast<size_t>(head_dim_v), pad_b_v_bytes, b_v_bytes) ||
            !safe_mul(static_cast<size_t>(cur_tile), pad_rank_k_bytes, a_k_bytes) ||
            !safe_mul(static_cast<size_t>(cur_tile), pad_rank_v_bytes, a_v_bytes) ||
            !safe_mul(static_cast<size_t>(cur_tile), dk_bytes, k_tile_bytes) ||
            !safe_mul(static_cast<size_t>(cur_tile), dv_bytes_tile, v_tile_bytes) ||
            !safe_mul(static_cast<size_t>(head_dim_k), sizeof(float), phase_bytes) ||
            !safe_mul(static_cast<size_t>(cur_tile), sizeof(uint64_t), indices_bytes)) {
            if (err) *err = "xkv_preflight_batch_workspace: overflow computing tile workspace terms";
            return false;
        }

        size_t view_ws = 0;
        if (!safe_add(b_k_bytes, b_v_bytes, view_ws) ||
            !safe_add(view_ws, a_k_bytes, view_ws) ||
            !safe_add(view_ws, a_v_bytes, view_ws) ||
            !safe_add(view_ws, k_tile_bytes, view_ws) ||
            !safe_add(view_ws, v_tile_bytes, view_ws) ||
            !safe_add(view_ws, phase_bytes, view_ws) ||
            !safe_add(view_ws, indices_bytes, view_ws)) {
            if (err) *err = "xkv_preflight_batch_workspace: overflow summing tile workspace";
            return false;
        }

        max_tile_ws = std::max(max_tile_ws, view_ws);
    }

    if (!safe_add(all_states_bytes, csr_overhead, out_peak_workspace) ||
        !safe_add(out_peak_workspace, max_tile_ws, out_peak_workspace)) {
        if (err) *err = "xkv_preflight_batch_workspace: overflow summing batch workspace";
        return false;
    }

    if (config.workspace_budget_bytes > 0 && out_peak_workspace > config.workspace_budget_bytes) {
        if (err) {
            *err = "xkv_preflight_batch_workspace: batch peak workspace (" +
                std::to_string(out_peak_workspace) + " bytes) exceeds budget (" +
                std::to_string(config.workspace_budget_bytes) + " bytes)";
        }
        return false;
    }

    if (config.workspace != nullptr && !config.workspace->preflight(out_peak_workspace)) {
        if (err) {
            *err = "xkv_preflight_batch_workspace: batch peak workspace (" +
                std::to_string(out_peak_workspace) + " bytes) exceeds reusable workspace capacity (" +
                std::to_string(config.workspace->capacity_bytes()) + " bytes)";
        }
        return false;
    }

    if (config.workspace != nullptr) {
        const auto & wcfg = config.workspace->config();
        if (queries.size() > wcfg.max_queries) {
            if (err) *err = "xkv_preflight_batch_workspace: queries.size() exceeds workspace max_queries";
            return false;
        }
        for (const auto & q : queries) {
            if (q.n_q_heads > wcfg.max_q_heads) {
                if (err) *err = "xkv_preflight_batch_workspace: query.n_q_heads exceeds workspace max_q_heads";
                return false;
            }
            if (q.head_dim_k > wcfg.max_head_dim_k) {
                if (err) *err = "xkv_preflight_batch_workspace: query.head_dim_k exceeds workspace max_head_dim_k";
                return false;
            }
            if (q.head_dim_v > wcfg.max_head_dim_v) {
                if (err) *err = "xkv_preflight_batch_workspace: query.head_dim_v exceeds workspace max_head_dim_v";
                return false;
            }
        }
        if (batch_selection.csr_indices.size() > wcfg.max_csr_entries) {
            if (err) *err = "xkv_preflight_batch_workspace: CSR indices count exceeds workspace max_csr_entries";
            return false;
        }
        uint32_t tile_sz = (config.tile_size == 0) ? 32 : config.tile_size;
        if (tile_sz > wcfg.max_tile_size) {
            if (err) *err = "xkv_preflight_batch_workspace: config.tile_size exceeds workspace max_tile_size";
            return false;
        }
        for (const auto & view : segment_views) {
            const auto * g = view.resolve_group(err);
            if (!g || !g->b_k || !g->b_v) continue;
            if (g->a_k.desc.padded_shape.cols > wcfg.max_rank_k || g->b_k->desc.padded_shape.cols > wcfg.max_rank_k) {
                if (err) *err = "xkv_preflight_batch_workspace: segment rank_k exceeds workspace max_rank_k";
                return false;
            }
            if (g->a_v.desc.padded_shape.cols > wcfg.max_rank_v || g->b_v->desc.padded_shape.cols > wcfg.max_rank_v) {
                if (err) *err = "xkv_preflight_batch_workspace: segment rank_v exceeds workspace max_rank_v";
                return false;
            }
        }
    }

    return true;
}

// ============================================================================
// Single query attention reading: xkv_read_attention
// ============================================================================

xkv_read_result xkv_read_attention(
    const xkv_query_input & query,
    const std::vector<xkv_hot_row> & hot_rows,
    const std::vector<xkv_segment_read_view> & segment_views,
    const phase_transform_fn & phase_tx,
    const xkv_snapshot_stamp & expected_stamp,
    llama_xkv_cache_store * store,
    const xkv_reader_config & config
) {
    xkv_read_result result;

    std::string err;
    if (!query.validate(&err)) {
        result.status = xkv_read_status::invalid_argument;
        result.error_message = err;
        return result;
    }

    for (const auto & v : segment_views) {
        if (!v.validate(&err)) {
            result.status = xkv_read_status::invalid_argument;
            result.error_message = err;
            return result;
        }
        // Validate that each row's group_idx is valid for this query
        for (size_t r = 0; r < v.selected_rows.size(); ++r) {
            uint32_t group_idx = (!v.group_indices.empty()) ? v.group_indices[r] : 0;
            if (!query.q_groups.empty()) {
                if (group_idx >= query.q_groups.size()) {
                    result.status = xkv_read_status::invalid_argument;
                    result.error_message = "xkv_read_attention: row group_index (" +
                        std::to_string(group_idx) + ") exceeds query q_groups size (" +
                        std::to_string(query.q_groups.size()) + ")";
                    return result;
                }
            } else if (group_idx != 0) {
                result.status = xkv_read_status::invalid_argument;
                result.error_message = "xkv_read_attention: non-zero group_index without q_groups";
                return result;
            }
        }
    }

    for (const auto & hr : hot_rows) {
        if (!hr.is_valid) continue;
        // Positional per-query mapping: a single read resolves position 0.
        uint32_t hr_group = hr.group_index;
        if (!hr.query_group_indices.empty()) {
            if (hr.query_group_indices.size() != 1) {
                result.status = xkv_read_status::invalid_argument;
                result.error_message = "xkv_read_attention: hot_row query_group_indices size mismatch";
                return result;
            }
            hr_group = hr.query_group_indices[0];
            if (!hr.is_visible_to_query(query.query_index) && hr_group == UINT32_MAX) {
                continue;
            }
            if (hr_group == UINT32_MAX) {
                result.status = xkv_read_status::invalid_argument;
                result.error_message = "xkv_read_attention: hot_row visible query has invalid group";
                return result;
            }
        }
        if (!query.q_groups.empty()) {
            if (hr_group >= query.q_groups.size()) {
                result.status = xkv_read_status::invalid_argument;
                result.error_message = "xkv_read_attention: hot_row group_index exceeds query q_groups size";
                return result;
            }
        } else if (hr_group != 0) {
            result.status = xkv_read_status::invalid_argument;
            result.error_message = "xkv_read_attention: non-zero hot_row group_index without q_groups";
            return result;
        }
    }

    // 1. Snapshot stamp validation BEFORE reading
    if (store != nullptr) {
        xkv_snapshot_stamp stamp_before = store->current_stamp();
        if (stamp_before != expected_stamp) {
            result.status = xkv_read_status::retry_stale_stamp;
            result.error_message = "xkv_read_attention: snapshot stamp stale before read";
            return result;
        }
    }

    // 2. Preflight checked workspace check BEFORE any allocation
    size_t peak_ws = 0;
    if (!xkv_preflight_workspace(query, segment_views, config, peak_ws, &err)) {
        result.status = xkv_read_status::workspace_exceeded;
        result.error_message = err;
        return result;
    }

    // Acquire reader scratch lease if workspace is configured
    xkv_workspace_scratch_lease ws_lease;
    if (config.workspace != nullptr) {
        ws_lease = config.workspace->acquire_lease(peak_ws);
        if (!ws_lease.valid()) {
            result.status = xkv_read_status::workspace_exceeded;
            result.error_message = "xkv_read_attention: reusable workspace scratch acquisition failed";
            return result;
        }
    }
    const xkv_reader_scratch * scratch = ws_lease.valid() ? &ws_lease.get() : nullptr;

    // Strict bounded mode: verify every slice BEFORE any decode, arena reservation,
    // cache mutation, or output so a short slice fails closed deterministically.
    if (strict_mode(config)) {
        std::string slice_err;
        if (!check_single_strict_slices(query, segment_views, config, *scratch, &slice_err)) {
            result.status = xkv_read_status::workspace_exceeded;
            result.error_message = slice_err;
            return result;
        }
        xkv_read_status tmp_st = check_strict_decode_tmp(segment_views, *scratch, &slice_err);
        if (tmp_st != xkv_read_status::success) {
            result.status = tmp_st;
            result.error_message = slice_err;
            return result;
        }
    }

    // Acquire store workspace arena lease if available. Skipped for externally
    // carved workspaces: the arena already counts the carve for its whole lifetime,
    // so a per-read acquire would double-count the same bytes and fail spuriously.
    // Owned workspaces still need this per-read admission check.
    xkv_arena_lease arena_lease;
    const bool carved = config.workspace != nullptr && config.workspace->is_external();
    if (store != nullptr && !carved) {
        arena_lease = store->acquire_workspace_lease(peak_ws);
        if (!arena_lease.valid() && store->get_arena().get_capacity_bytes() > 0) {
            result.status = xkv_read_status::workspace_exceeded;
            result.error_message = "xkv_read_attention: store workspace arena capacity exceeded";
            return result;
        }
    }

    float scale = query.scale;
    if (scale <= 0.0f) {
        scale = 1.0f / std::sqrt(static_cast<float>(query.head_dim_k));
    }

    std::vector<xkv_online_softmax_state> fallback_states;
    xkv_online_softmax_state * states_ptr = nullptr;
    if (scratch && scratch->states_accum && config.workspace &&
        query.n_q_heads <= config.workspace->get_states_storage_capacity() &&
        query.n_q_heads * query.head_dim_v <= scratch->states_accum_capacity) {
        states_ptr = config.workspace->get_states_storage();
        for (uint32_t qh = 0; qh < query.n_q_heads; ++qh) {
            states_ptr[qh].init(scratch->states_accum + qh * query.head_dim_v, query.head_dim_v);
        }
    } else if (strict_mode(config)) {
        // Unreachable after the upfront slice check; defense in depth.
        result.status = xkv_read_status::workspace_exceeded;
        result.error_message = "xkv_read_attention: workspace states slice short";
        return result;
    } else {
        fallback_states.resize(query.n_q_heads);
        states_ptr = fallback_states.data();
        for (uint32_t qh = 0; qh < query.n_q_heads; ++qh) {
            states_ptr[qh].reset(query.head_dim_v);
        }
    }
    auto & states = states_ptr;

    try {
        // 3. Stream Hot Flat Rows
        for (const auto & hr : hot_rows) {
            if (!hr.is_valid || hr.k_ptr == nullptr || hr.v_ptr == nullptr) {
                continue;
            }
            if (query.causal_limit_pos >= 0 && hr.storage_pos > query.causal_limit_pos) {
                continue;
            }
            if (!hr.is_visible_to_query(query.query_index)) {
                continue;
            }

            // Positional per-query DDVR group for this hot row (validated upfront).
            const uint32_t hr_group =
                hr.query_group_indices.empty() ? hr.group_index : hr.query_group_indices[0];

            for (uint32_t qh = 0; qh < query.n_q_heads; ++qh) {
                const float * q_ptr = get_q_ptr(query, hr_group, qh);
                if (!q_ptr) continue;

                float score = dot_product(q_ptr, hr.k_ptr, query.head_dim_k) * scale;
                if (query.logit_softcap > 0.0f) {
                    score = query.logit_softcap * std::tanh(score / query.logit_softcap);
                }

                states[qh].update_single(score, hr.v_ptr, query.head_dim_v);
            }
        }

        // 4. Stream Factored Segments in Bounded Tiles
        const uint32_t tile_size = (config.tile_size == 0) ? 32 : config.tile_size;

        for (const auto & view : segment_views) {
            const auto * seg = view.get_segment();
            if (!seg || view.selected_rows.empty()) continue;

            const auto * g = view.resolve_group(nullptr);
            if (!g || !g->b_k || !g->b_v) continue;

            uint32_t n_selected = static_cast<uint32_t>(view.selected_rows.size());
            uint32_t rank_k = static_cast<uint32_t>(g->a_k.desc.logical_shape.cols);
            uint32_t rank_v = static_cast<uint32_t>(g->a_v.desc.logical_shape.cols);
            uint32_t pad_rank_k = static_cast<uint32_t>(g->a_k.desc.padded_shape.cols);
            uint32_t pad_rank_v = static_cast<uint32_t>(g->a_v.desc.padded_shape.cols);
            uint32_t pad_rank_b_k = static_cast<uint32_t>(g->b_k->desc.padded_shape.cols);
            uint32_t pad_rank_b_v = static_cast<uint32_t>(g->b_v->desc.padded_shape.cols);

            uint64_t seg_version = view.segment_version_id != 0 ? view.segment_version_id : seg->segment_version;

            xkv_b_tile_lease lease_b_k;
            xkv_b_tile_lease lease_b_v;

            std::string off_err;
            uint32_t feat_off_k = view.get_b_feature_offset_k(query.head_dim_k, &off_err);
            uint32_t feat_off_v = view.get_b_feature_offset_v(query.head_dim_v, &off_err);
            if (feat_off_k == UINT32_MAX || feat_off_v == UINT32_MAX) {
                throw std::runtime_error("xkv_read_attention: failed to derive B feature offset: " + off_err);
            }

            // B_K
            b_tile_cache_key key_k;
            key_k.segment_id = seg->segment_id;
            key_k.segment_version = seg_version;
            key_k.role = factor_role::b_k;
            key_k.owning_layer = view.owning_layer;
            key_k.kv_head = view.kv_head;
            key_k.tile_index = 0;
            key_k.codec_epoch = expected_stamp.codec_epoch;
            key_k.feature_offset = feat_off_k;
            key_k.feature_dim = query.head_dim_k;
            key_k.device_id = config.device_id;
            key_k.domain = value_domain::canonical;
            key_k.desc_fingerprint = g->b_k->desc.fingerprint();

            uint64_t * ptr_b_rows_k = nullptr;
            std::vector<uint64_t> fb_b_rows_k;
            if (scratch && query.head_dim_k <= scratch->b_row_indices_capacity) {
                ptr_b_rows_k = scratch->b_row_indices;
            } else if (strict_mode(config)) {
                result.status = xkv_read_status::workspace_exceeded;
                result.error_message = "xkv_read_attention: workspace b_row_indices short";
                return result;
            } else {
                fb_b_rows_k.resize(query.head_dim_k);
                ptr_b_rows_k = fb_b_rows_k.data();
            }
            for (uint32_t i = 0; i < query.head_dim_k; ++i) {
                ptr_b_rows_k[i] = feat_off_k + i;
            }

            // Bounded B tile: a cache hit uses the cached lease with no decode; a miss decodes
            // straight into the caller workspace slice and offers a counted copy to the cache
            // for future reads while this read keeps using its own slice.
            size_t need_b_k = 0;
            if (!safe_mul(static_cast<size_t>(query.head_dim_k), static_cast<size_t>(pad_rank_b_k), need_b_k)) {
                result.status = xkv_read_status::invalid_argument;
                result.error_message = "xkv_read_attention: overflow computing B_K tile size";
                return result;
            }
            if (config.b_cache != nullptr) {
                lease_b_k = config.b_cache->get(key_k);
                if (lease_b_k.valid()) {
                    result.b_cache_hit = true;
                } else if (scratch && need_b_k <= scratch->tile_b_k_capacity) {
                    decode_rows_bounded(*g->b_k, ptr_b_rows_k, query.head_dim_k, scratch->tile_b_k, need_b_k, config, scratch);
                    config.b_cache->put_and_lease_copy(key_k, scratch->tile_b_k, need_b_k);
                    lease_b_k = xkv_b_tile_lease::from_buffer(scratch->tile_b_k, need_b_k);
                } else if (strict_mode(config)) {
                    result.status = xkv_read_status::workspace_exceeded;
                    result.error_message = "xkv_read_attention: workspace tile_b_k short";
                    return result;
                } else {
                    std::vector<float> local_b_k(need_b_k);
                    decode_rows(*g->b_k, ptr_b_rows_k, query.head_dim_k, local_b_k.data(), local_b_k.size(), value_domain::canonical);
                    lease_b_k = config.b_cache->put_and_lease(key_k, std::move(local_b_k));
                }
            } else if (scratch && need_b_k <= scratch->tile_b_k_capacity) {
                decode_rows_bounded(*g->b_k, ptr_b_rows_k, query.head_dim_k, scratch->tile_b_k, need_b_k, config, scratch);
                lease_b_k = xkv_b_tile_lease::from_buffer(scratch->tile_b_k, need_b_k);
            } else if (strict_mode(config)) {
                result.status = xkv_read_status::workspace_exceeded;
                result.error_message = "xkv_read_attention: workspace tile_b_k short";
                return result;
            } else {
                std::vector<float> local_b_k(need_b_k);
                decode_rows(*g->b_k, ptr_b_rows_k, query.head_dim_k, local_b_k.data(), local_b_k.size(), value_domain::canonical);
                auto rec = std::make_shared<b_tile_record>();
                rec->data = std::move(local_b_k);
                rec->in_cache = false;
                lease_b_k = xkv_b_tile_lease::uncached(rec);
            }

            // B_V
            b_tile_cache_key key_v;
            key_v.segment_id = seg->segment_id;
            key_v.segment_version = seg_version;
            key_v.role = factor_role::b_v;
            key_v.owning_layer = view.owning_layer;
            key_v.kv_head = view.kv_head;
            key_v.tile_index = 0;
            key_v.codec_epoch = expected_stamp.codec_epoch;
            key_v.feature_offset = feat_off_v;
            key_v.feature_dim = query.head_dim_v;
            key_v.device_id = config.device_id;
            key_v.domain = value_domain::canonical;
            key_v.desc_fingerprint = g->b_v->desc.fingerprint();

            uint64_t * ptr_b_rows_v = nullptr;
            std::vector<uint64_t> fb_b_rows_v;
            if (scratch && query.head_dim_v <= scratch->b_row_indices_capacity) {
                ptr_b_rows_v = scratch->b_row_indices;
            } else if (strict_mode(config)) {
                result.status = xkv_read_status::workspace_exceeded;
                result.error_message = "xkv_read_attention: workspace b_row_indices short";
                return result;
            } else {
                fb_b_rows_v.resize(query.head_dim_v);
                ptr_b_rows_v = fb_b_rows_v.data();
            }
            for (uint32_t i = 0; i < query.head_dim_v; ++i) {
                ptr_b_rows_v[i] = feat_off_v + i;
            }

            size_t need_b_v = 0;
            if (!safe_mul(static_cast<size_t>(query.head_dim_v), static_cast<size_t>(pad_rank_b_v), need_b_v)) {
                result.status = xkv_read_status::invalid_argument;
                result.error_message = "xkv_read_attention: overflow computing B_V tile size";
                return result;
            }
            if (config.b_cache != nullptr) {
                lease_b_v = config.b_cache->get(key_v);
                if (!lease_b_v.valid()) {
                    if (scratch && need_b_v <= scratch->tile_b_v_capacity) {
                        decode_rows_bounded(*g->b_v, ptr_b_rows_v, query.head_dim_v, scratch->tile_b_v, need_b_v, config, scratch);
                        config.b_cache->put_and_lease_copy(key_v, scratch->tile_b_v, need_b_v);
                        lease_b_v = xkv_b_tile_lease::from_buffer(scratch->tile_b_v, need_b_v);
                    } else if (strict_mode(config)) {
                        result.status = xkv_read_status::workspace_exceeded;
                        result.error_message = "xkv_read_attention: workspace tile_b_v short";
                        return result;
                    } else {
                        std::vector<float> local_b_v(need_b_v);
                        decode_rows(*g->b_v, ptr_b_rows_v, query.head_dim_v, local_b_v.data(), local_b_v.size(), value_domain::canonical);
                        lease_b_v = config.b_cache->put_and_lease(key_v, std::move(local_b_v));
                    }
                }
            } else if (scratch && need_b_v <= scratch->tile_b_v_capacity) {
                decode_rows_bounded(*g->b_v, ptr_b_rows_v, query.head_dim_v, scratch->tile_b_v, need_b_v, config, scratch);
                lease_b_v = xkv_b_tile_lease::from_buffer(scratch->tile_b_v, need_b_v);
            } else if (strict_mode(config)) {
                result.status = xkv_read_status::workspace_exceeded;
                result.error_message = "xkv_read_attention: workspace tile_b_v short";
                return result;
            } else {
                std::vector<float> local_b_v(need_b_v);
                decode_rows(*g->b_v, ptr_b_rows_v, query.head_dim_v, local_b_v.data(), local_b_v.size(), value_domain::canonical);
                auto rec = std::make_shared<b_tile_record>();
                rec->data = std::move(local_b_v);
                rec->in_cache = false;
                lease_b_v = xkv_b_tile_lease::uncached(rec);
            }

            const float * b_k_ptr = lease_b_k.data();
            const float * b_v_ptr = lease_b_v.data();

            uint32_t cur_tile_size = std::min(tile_size, n_selected);

            // Use scratch buffers when available and sufficient, else fallback
            float * ptr_tile_a_k = nullptr;
            std::vector<float> fb_tile_a_k;
            // Checked tile sizes (size_t, no truncation) shared by slice selection below.
            size_t need_a_k = 0, need_a_v = 0, need_tk = 0, need_tv = 0;
            if (!safe_mul(static_cast<size_t>(cur_tile_size), static_cast<size_t>(pad_rank_k), need_a_k) ||
                !safe_mul(static_cast<size_t>(cur_tile_size), static_cast<size_t>(pad_rank_v), need_a_v) ||
                !safe_mul(static_cast<size_t>(cur_tile_size), static_cast<size_t>(query.head_dim_k), need_tk) ||
                !safe_mul(static_cast<size_t>(cur_tile_size), static_cast<size_t>(query.head_dim_v), need_tv)) {
                result.status = xkv_read_status::invalid_argument;
                result.error_message = "xkv_read_attention: overflow computing tile sizes";
                return result;
            }
            if (scratch && need_a_k <= scratch->tile_a_k_capacity) {
                ptr_tile_a_k = scratch->tile_a_k;
            } else if (strict_mode(config)) {
                result.status = xkv_read_status::workspace_exceeded;
                result.error_message = "xkv_read_attention: workspace tile_a_k short";
                return result;
            } else {
                fb_tile_a_k.resize(need_a_k);
                ptr_tile_a_k = fb_tile_a_k.data();
            }

            float * ptr_tile_a_v = nullptr;
            std::vector<float> fb_tile_a_v;
            if (scratch && need_a_v <= scratch->tile_a_v_capacity) {
                ptr_tile_a_v = scratch->tile_a_v;
            } else if (strict_mode(config)) {
                result.status = xkv_read_status::workspace_exceeded;
                result.error_message = "xkv_read_attention: workspace tile_a_v short";
                return result;
            } else {
                fb_tile_a_v.resize(need_a_v);
                ptr_tile_a_v = fb_tile_a_v.data();
            }

            float * ptr_tile_k = nullptr;
            std::vector<float> fb_tile_k;
            if (scratch && need_tk <= scratch->tile_k_capacity) {
                ptr_tile_k = scratch->tile_k;
            } else if (strict_mode(config)) {
                result.status = xkv_read_status::workspace_exceeded;
                result.error_message = "xkv_read_attention: workspace tile_k short";
                return result;
            } else {
                fb_tile_k.resize(need_tk);
                ptr_tile_k = fb_tile_k.data();
            }

            float * ptr_tile_v = nullptr;
            std::vector<float> fb_tile_v;
            if (scratch && need_tv <= scratch->tile_v_capacity) {
                ptr_tile_v = scratch->tile_v;
            } else if (strict_mode(config)) {
                result.status = xkv_read_status::workspace_exceeded;
                result.error_message = "xkv_read_attention: workspace tile_v short";
                return result;
            } else {
                fb_tile_v.resize(need_tv);
                ptr_tile_v = fb_tile_v.data();
            }

            float * ptr_transformed_k = nullptr;
            std::vector<float> fb_transformed_k;
            if (scratch && query.head_dim_k <= scratch->transformed_k_capacity) {
                ptr_transformed_k = scratch->transformed_k;
            } else if (strict_mode(config)) {
                result.status = xkv_read_status::workspace_exceeded;
                result.error_message = "xkv_read_attention: workspace transformed_k short";
                return result;
            } else {
                fb_transformed_k.resize(query.head_dim_k);
                ptr_transformed_k = fb_transformed_k.data();
            }

            uint64_t * ptr_tile_row_indices = nullptr;
            std::vector<uint64_t> fb_tile_row_indices;
            if (scratch && cur_tile_size <= scratch->tile_row_indices_capacity) {
                ptr_tile_row_indices = scratch->tile_row_indices;
            } else if (strict_mode(config)) {
                result.status = xkv_read_status::workspace_exceeded;
                result.error_message = "xkv_read_attention: workspace tile_row_indices short";
                return result;
            } else {
                fb_tile_row_indices.resize(cur_tile_size);
                ptr_tile_row_indices = fb_tile_row_indices.data();
            }

            for (uint32_t t_begin = 0; t_begin < n_selected; t_begin += tile_size) {
                uint32_t t_count = std::min(tile_size, n_selected - t_begin);

                for (uint32_t i = 0; i < t_count; ++i) {
                    ptr_tile_row_indices[i] = view.selected_rows[t_begin + i];
                }

                decode_rows_bounded(g->a_k, ptr_tile_row_indices, t_count, ptr_tile_a_k, t_count * pad_rank_k, config, scratch);
                decode_rows_bounded(g->a_v, ptr_tile_row_indices, t_count, ptr_tile_a_v, t_count * pad_rank_v, config, scratch);

                gemm_reconstruct_tile(ptr_tile_a_k, pad_rank_k, b_k_ptr, pad_rank_b_k, rank_k, t_count, query.head_dim_k, ptr_tile_k);
                gemm_reconstruct_tile(ptr_tile_a_v, pad_rank_v, b_v_ptr, pad_rank_b_v, rank_v, t_count, query.head_dim_v, ptr_tile_v);

                for (uint32_t i = 0; i < t_count; ++i) {
                    uint32_t global_idx = t_begin + i;

                    if (!view.membership_mask.empty() && !view.membership_mask[global_idx]) {
                        continue;
                    }

                    int64_t storage_pos = view.storage_positions[global_idx];
                    if (query.causal_limit_pos >= 0 && storage_pos > query.causal_limit_pos) {
                        continue;
                    }

                    uint32_t group_idx = (!view.group_indices.empty()) ? view.group_indices[global_idx] : 0;

                    const float * raw_k_ptr = ptr_tile_k + i * query.head_dim_k;
                    const float * cur_v_ptr = ptr_tile_v + i * query.head_dim_v;

                    const float * final_k_ptr = raw_k_ptr;
                    if (phase_tx) {
                        phase_tx(raw_k_ptr, storage_pos, view.kv_head, ptr_transformed_k);
                        final_k_ptr = ptr_transformed_k;
                    }

                    for (uint32_t qh = 0; qh < query.n_q_heads; ++qh) {
                        const float * q_ptr = get_q_ptr(query, group_idx, qh);
                        if (!q_ptr) continue;

                        float score = dot_product(q_ptr, final_k_ptr, query.head_dim_k) * scale;
                        if (query.logit_softcap > 0.0f) {
                            score = query.logit_softcap * std::tanh(score / query.logit_softcap);
                        }

                        states[qh].update_single(score, cur_v_ptr, query.head_dim_v);
                    }
                }
            }
        }

        // 5. Incorporate Single Sink Contribution
        if (!query.sink_logits.empty()) {
            for (uint32_t qh = 0; qh < query.n_q_heads; ++qh) {
                if (qh < query.sink_logits.size()) {
                    states[qh].update_sink(query.sink_logits[qh], query.head_dim_v);
                }
            }
        }

        // 6. Finalize Outputs
        result.output.resize(query.n_q_heads * query.head_dim_v);
        for (uint32_t qh = 0; qh < query.n_q_heads; ++qh) {
            states[qh].finalize(result.output.data() + qh * query.head_dim_v, query.head_dim_v);
        }

    } catch (const std::exception & e) {
        result.status = xkv_read_status::codec_error;
        result.output.clear();
        result.error_message = std::string("xkv_read_attention exception: ") + e.what();
        return result;
    }

    // 7. Snapshot stamp validation AFTER reading
    if (store != nullptr) {
        xkv_snapshot_stamp stamp_after = store->current_stamp();
        if (stamp_after != expected_stamp) {
            result.status = xkv_read_status::retry_stale_stamp;
            result.output.clear(); // No partial output on stale stamp
            result.error_message = "xkv_read_attention: snapshot stamp changed during read";
            return result;
        }
    }

    result.status = xkv_read_status::success;
    result.peak_workspace_bytes = peak_ws;
    return result;
}

// ============================================================================
// Batch query attention reading with single union decode & CSR isolation
// ============================================================================

xkv_batch_read_result xkv_read_attention_batch(
    const std::vector<xkv_query_input> & queries,
    const std::vector<xkv_hot_row> & hot_rows,
    const std::vector<xkv_segment_read_view> & segment_views,
    const sr_batch_selection_result & batch_selection,
    const phase_transform_fn & phase_tx,
    const xkv_snapshot_stamp & expected_stamp,
    llama_xkv_cache_store * store,
    const xkv_reader_config & config
) {
    xkv_batch_read_result batch_res;

    if (queries.empty()) {
        batch_res.status = xkv_read_status::success;
        return batch_res;
    }

    std::string err;
    uint32_t batch_head_dim_k = queries[0].head_dim_k;
    uint32_t batch_head_dim_v = queries[0].head_dim_v;

    for (size_t q = 0; q < queries.size(); ++q) {
        if (!queries[q].validate(&err)) {
            batch_res.status = xkv_read_status::invalid_argument;
            batch_res.error_message = "query " + std::to_string(q) + ": " + err;
            return batch_res;
        }
        // In batch attention, queries must share identical head dimensions
        if (queries[q].head_dim_k != batch_head_dim_k || queries[q].head_dim_v != batch_head_dim_v) {
            batch_res.status = xkv_read_status::invalid_argument;
            batch_res.error_message = "xkv_read_attention_batch: query " + std::to_string(q) +
                " head dimensions differ from batch query 0";
            return batch_res;
        }
    }

    for (size_t v = 0; v < segment_views.size(); ++v) {
        if (!segment_views[v].validate(&err)) {
            batch_res.status = xkv_read_status::invalid_argument;
            batch_res.error_message = "segment_view " + std::to_string(v) + ": " + err;
            return batch_res;
        }
        for (size_t r = 0; r < segment_views[v].selected_rows.size(); ++r) {
            uint32_t group_idx = (!segment_views[v].group_indices.empty()) ? segment_views[v].group_indices[r] : 0;
            for (size_t q = 0; q < queries.size(); ++q) {
                if (!queries[q].q_groups.empty()) {
                    if (group_idx >= queries[q].q_groups.size()) {
                        batch_res.status = xkv_read_status::invalid_argument;
                        batch_res.error_message = "xkv_read_attention_batch: row group_index exceeds query " +
                            std::to_string(q) + " q_groups size";
                        return batch_res;
                    }
                } else if (group_idx != 0) {
                    batch_res.status = xkv_read_status::invalid_argument;
                    batch_res.error_message = "xkv_read_attention_batch: non-zero group_index without q_groups";
                    return batch_res;
                }
            }
        }
    }

    // Validate hot-row per-query visibility and group mapping. A non-empty
    // query_group_indices must match the batch query count exactly; every visible
    // query needs a valid (non-UINT32_MAX) group. With an empty mapping the single
    // group_index must be valid for every visible query.
    for (size_t h = 0; h < hot_rows.size(); ++h) {
        const auto & hr = hot_rows[h];
        if (!hr.is_valid) continue;
        if (!hr.query_group_indices.empty() && hr.query_group_indices.size() != queries.size()) {
            batch_res.status = xkv_read_status::invalid_argument;
            batch_res.error_message = "xkv_read_attention_batch: hot_row query_group_indices size mismatch";
            return batch_res;
        }
        for (size_t q = 0; q < queries.size(); ++q) {
            if (!hr.is_visible_to_query(queries[q].query_index)) continue;
            uint32_t g = hr.group_index;
            if (!hr.query_group_indices.empty()) {
                g = hr.query_group_indices[q];
                if (g == UINT32_MAX) {
                    batch_res.status = xkv_read_status::invalid_argument;
                    batch_res.error_message = "xkv_read_attention_batch: hot_row visible query has invalid group";
                    return batch_res;
                }
            }
            if (!queries[q].q_groups.empty()) {
                if (g >= queries[q].q_groups.size()) {
                    batch_res.status = xkv_read_status::invalid_argument;
                    batch_res.error_message = "xkv_read_attention_batch: hot_row group_index exceeds query " +
                        std::to_string(q) + " q_groups size";
                    return batch_res;
                }
            } else if (g != 0) {
                batch_res.status = xkv_read_status::invalid_argument;
                batch_res.error_message = "xkv_read_attention_batch: non-zero hot_row group_index without q_groups";
                return batch_res;
            }
        }
    }

    // Validate CSR structure if provided
    bool has_csr = !batch_selection.csr_ptrs.empty();
    // Per-(query, row) group indices are present only when the parallel array is
    // non-empty; otherwise every query uses the view group for matched rows.
    const bool has_csr_groups = has_csr && !batch_selection.csr_group_indices.empty();
    if (has_csr) {
        if (batch_selection.csr_ptrs.size() != queries.size() + 1) {
            batch_res.status = xkv_read_status::invalid_argument;
            batch_res.error_message = "batch CSR ptrs size mismatch";
            return batch_res;
        }
        if (batch_selection.csr_ptrs[0] != 0) {
            batch_res.status = xkv_read_status::invalid_argument;
            batch_res.error_message = "batch CSR ptrs[0] must be 0";
            return batch_res;
        }
        for (size_t q = 0; q < queries.size(); ++q) {
            if (batch_selection.csr_ptrs[q] > batch_selection.csr_ptrs[q + 1]) {
                batch_res.status = xkv_read_status::invalid_argument;
                batch_res.error_message = "batch CSR ptrs must be monotonic";
                return batch_res;
            }
        }
        if (batch_selection.csr_ptrs.back() != batch_selection.csr_indices.size()) {
            batch_res.status = xkv_read_status::invalid_argument;
            batch_res.error_message = "batch CSR indices size mismatch";
            return batch_res;
        }
        for (uint32_t idx : batch_selection.csr_indices) {
            if (idx >= batch_selection.gather_rows.size()) {
                batch_res.status = xkv_read_status::invalid_argument;
                batch_res.error_message = "batch CSR index out of gather_rows range";
                return batch_res;
            }
        }
        // Validate per-(query, row) group indices when the parallel array is present:
        // exact length match; UINT32_MAX means untracked (view-group fallback); any
        // other value must be a valid group for the owning query.
        if (has_csr_groups) {
            if (batch_selection.csr_group_indices.size() != batch_selection.csr_indices.size()) {
                batch_res.status = xkv_read_status::invalid_argument;
                batch_res.error_message = "batch CSR group indices size mismatch";
                return batch_res;
            }
            for (size_t q = 0; q < queries.size(); ++q) {
                const uint32_t c_begin = batch_selection.csr_ptrs[q];
                const uint32_t c_end = batch_selection.csr_ptrs[q + 1];
                for (uint32_t c = c_begin; c < c_end; ++c) {
                    const uint32_t g = batch_selection.csr_group_indices[c];
                    if (g == UINT32_MAX) continue;
                    if (!queries[q].q_groups.empty()) {
                        if (g >= queries[q].q_groups.size()) {
                            batch_res.status = xkv_read_status::invalid_argument;
                            batch_res.error_message = "batch CSR group index exceeds query " +
                                std::to_string(q) + " q_groups size";
                            return batch_res;
                        }
                    } else if (g != 0) {
                        batch_res.status = xkv_read_status::invalid_argument;
                        batch_res.error_message = "batch CSR non-zero group index without q_groups";
                        return batch_res;
                    }
                }
            }
        }
    }

    // Stamp validation before
    if (store != nullptr) {
        xkv_snapshot_stamp stamp_before = store->current_stamp();
        if (stamp_before != expected_stamp) {
            batch_res.status = xkv_read_status::retry_stale_stamp;
            batch_res.error_message = "xkv_read_attention_batch: snapshot stamp stale before read";
            return batch_res;
        }
    }

    // Checked batch workspace preflight
    size_t peak_batch_ws = 0;
    if (!xkv_preflight_batch_workspace(queries, segment_views, batch_selection, config, peak_batch_ws, &err)) {
        batch_res.status = xkv_read_status::workspace_exceeded;
        batch_res.error_message = err;
        return batch_res;
    }

    // Acquire reader scratch lease if workspace is configured.
    xkv_workspace_scratch_lease ws_lease;
    if (config.workspace != nullptr) {
        ws_lease = config.workspace->acquire_lease(peak_batch_ws);
        if (!ws_lease.valid()) {
            batch_res.status = xkv_read_status::workspace_exceeded;
            batch_res.error_message = "xkv_read_attention_batch: reusable workspace scratch acquisition failed";
            return batch_res;
        }
    }
    const xkv_reader_scratch * scratch = ws_lease.valid() ? &ws_lease.get() : nullptr;

    // Strict bounded mode: verify every slice BEFORE any decode, arena reservation,
    // cache mutation, or output so a short slice fails closed deterministically.
    if (strict_mode(config)) {
        std::string slice_err;
        if (!check_batch_strict_slices(queries, segment_views, batch_selection,
                                       batch_head_dim_k, batch_head_dim_v,
                                       config, *scratch, &slice_err)) {
            batch_res.status = xkv_read_status::workspace_exceeded;
            batch_res.error_message = slice_err;
            return batch_res;
        }
        xkv_read_status tmp_st = check_strict_decode_tmp(segment_views, *scratch, &slice_err);
        if (tmp_st != xkv_read_status::success) {
            batch_res.status = tmp_st;
            batch_res.error_message = slice_err;
            return batch_res;
        }
    }

    // Acquire store workspace arena lease if available (skipped for carved
    // workspaces for the same no-double-count reason as the single-read path).
    xkv_arena_lease arena_lease;
    const bool carved_batch = config.workspace != nullptr && config.workspace->is_external();
    if (store != nullptr && !carved_batch) {
        arena_lease = store->acquire_workspace_lease(peak_batch_ws);
        if (!arena_lease.valid() && store->get_arena().get_capacity_bytes() > 0) {
            batch_res.status = xkv_read_status::workspace_exceeded;
            batch_res.error_message = "xkv_read_attention_batch: store workspace arena capacity exceeded";
            return batch_res;
        }
    }

    bool any_b_cache_hit = false;

    // Initialize per-query online softmax states
    size_t n_queries = queries.size();
    batch_res.per_query.resize(n_queries);

    size_t total_batch_heads = 0;
    for (size_t q = 0; q < n_queries; ++q) {
        if (!safe_add(total_batch_heads, static_cast<size_t>(queries[q].n_q_heads), total_batch_heads)) {
            batch_res.status = xkv_read_status::invalid_argument;
            batch_res.error_message = "xkv_read_attention_batch: overflow summing batch heads";
            return batch_res;
        }
    }

    // Flat state array for every batch head. Strict mode uses the workspace-owned states
    // plus the states_accum slice (verified upfront); legacy oracle mode uses one exact heap
    // vector. Heads are addressed via running per-query bases in the query-inner loops
    // below, so no pointer table is allocated on any path and update order is unchanged.
    std::vector<xkv_online_softmax_state> fallback_all_states;
    xkv_online_softmax_state * base_states = nullptr;
    if (scratch && scratch->states_accum && config.workspace &&
        total_batch_heads <= config.workspace->get_states_storage_capacity()) {
        size_t need_accum = 0;
        if (safe_mul(total_batch_heads, static_cast<size_t>(batch_head_dim_v), need_accum) &&
            need_accum <= scratch->states_accum_capacity) {
            base_states = config.workspace->get_states_storage();
            size_t head_offset = 0;
            for (size_t q = 0; q < n_queries; ++q) {
                for (uint32_t qh = 0; qh < queries[q].n_q_heads; ++qh) {
                    base_states[head_offset + qh].init(
                        scratch->states_accum + (head_offset + qh) * batch_head_dim_v, batch_head_dim_v);
                }
                head_offset += queries[q].n_q_heads;
            }
        } else if (strict_mode(config)) {
            batch_res.status = xkv_read_status::workspace_exceeded;
            batch_res.error_message = "xkv_read_attention_batch: workspace states_accum short";
            return batch_res;
        }
    }
    if (base_states == nullptr) {
        if (strict_mode(config)) {
            batch_res.status = xkv_read_status::workspace_exceeded;
            batch_res.error_message = "xkv_read_attention_batch: workspace states slice short";
            return batch_res;
        }
        fallback_all_states.resize(total_batch_heads);
        base_states = fallback_all_states.data();
        size_t head_offset = 0;
        for (size_t q = 0; q < n_queries; ++q) {
            for (uint32_t qh = 0; qh < queries[q].n_q_heads; ++qh) {
                base_states[head_offset + qh].reset(queries[q].head_dim_v);
            }
            head_offset += queries[q].n_q_heads;
        }
    }

    // Per-query CSR membership as sorted ranges over one flat array. Without
    // per-query groups the legacy ref array is used (workspace csr_refs slice in
    // strict mode, one exact heap vector in legacy oracle mode) with binary-search
    // gating. With per-query groups, paired (ref, group) entries are used (workspace
    // csr_entries slice or fallback vector) so sorting keeps each row's group
    // attached; gating is an exact lower_bound per decoded row. Either way the union
    // decode remains a pure physical-gather optimization. No hash sets on any path.
    segment_row_ref * csr_base = nullptr;
    std::vector<segment_row_ref> fallback_csr_refs;
    xkv_csr_entry * csr_entries = nullptr;
    std::vector<xkv_csr_entry> fallback_csr_entries;
    if (has_csr) {
        const size_t total_csr_indices = batch_selection.csr_indices.size();
        if (has_csr_groups) {
            if (scratch && scratch->csr_entries && total_csr_indices <= scratch->csr_entries_capacity) {
                csr_entries = scratch->csr_entries;
            } else if (strict_mode(config)) {
                batch_res.status = xkv_read_status::workspace_exceeded;
                batch_res.error_message = "xkv_read_attention_batch: workspace csr_entries short";
                return batch_res;
            } else {
                fallback_csr_entries.resize(total_csr_indices);
                csr_entries = fallback_csr_entries.data();
            }
            for (size_t q = 0; q < n_queries; ++q) {
                uint32_t begin = batch_selection.csr_ptrs[q];
                uint32_t end   = batch_selection.csr_ptrs[q + 1];
                for (uint32_t c = begin; c < end; ++c) {
                    uint32_t g_idx = batch_selection.csr_indices[c];
                    csr_entries[c].ref = batch_selection.gather_rows[g_idx];
                    csr_entries[c].group_index = batch_selection.csr_group_indices[c];
                }
                if (end > begin) {
                    std::sort(csr_entries + begin, csr_entries + end);
                }
            }
        } else {
            if (scratch && scratch->csr_refs && total_csr_indices <= scratch->csr_refs_capacity) {
                csr_base = scratch->csr_refs;
            } else if (strict_mode(config)) {
                batch_res.status = xkv_read_status::workspace_exceeded;
                batch_res.error_message = "xkv_read_attention_batch: workspace csr_refs short";
                return batch_res;
            } else {
                fallback_csr_refs.resize(total_csr_indices);
                csr_base = fallback_csr_refs.data();
            }
            for (size_t q = 0; q < n_queries; ++q) {
                uint32_t begin = batch_selection.csr_ptrs[q];
                uint32_t end   = batch_selection.csr_ptrs[q + 1];
                for (uint32_t c = begin; c < end; ++c) {
                    uint32_t g_idx = batch_selection.csr_indices[c];
                    csr_base[c] = batch_selection.gather_rows[g_idx];
                }
                if (end > begin) {
                    std::sort(csr_base + begin, csr_base + end);
                }
            }
        }
    }

    try {
        // 1. Process Hot Flat Rows across queries with strict causal and membership isolation
        for (const auto & hr : hot_rows) {
            if (!hr.is_valid || hr.k_ptr == nullptr || hr.v_ptr == nullptr) continue;

            // Running per-query head base into base_states; per-query update order is
            // unchanged (hot rows in order), so numerics are bitwise identical.
            size_t head_base = 0;
            for (size_t q = 0; q < n_queries; ++q) {
                const auto & query = queries[q];
                // Causal isolation: future hot rows cannot leak to query q
                if (query.causal_limit_pos >= 0 && hr.storage_pos > query.causal_limit_pos) {
                    head_base += query.n_q_heads;
                    continue;
                }
                if (!hr.is_visible_to_query(query.query_index)) {
                    head_base += query.n_q_heads;
                    continue;
                }

                // Positional per-query DDVR group for this hot row (validated upfront).
                const uint32_t hr_group =
                    hr.query_group_indices.empty() ? hr.group_index : hr.query_group_indices[q];

                float scale = query.scale > 0.0f ? query.scale : 1.0f / std::sqrt(static_cast<float>(query.head_dim_k));

                for (uint32_t qh = 0; qh < query.n_q_heads; ++qh) {
                    const float * q_ptr = get_q_ptr(query, hr_group, qh);
                    if (!q_ptr) continue;

                    float score = dot_product(q_ptr, hr.k_ptr, query.head_dim_k) * scale;
                    if (query.logit_softcap > 0.0f) {
                        score = query.logit_softcap * std::tanh(score / query.logit_softcap);
                    }

                    base_states[head_base + qh].update_single(score, hr.v_ptr, query.head_dim_v);
                }
                head_base += query.n_q_heads;
            }
        }

        // 2. Stream Factored Segments: Single union decode across queries
        const uint32_t tile_size = (config.tile_size == 0) ? 32 : config.tile_size;

        for (const auto & view : segment_views) {
            const auto * seg = view.get_segment();
            if (!seg || view.selected_rows.empty()) continue;

            const auto * g = view.resolve_group(nullptr);
            if (!g || !g->b_k || !g->b_v) continue;

            uint32_t n_selected = static_cast<uint32_t>(view.selected_rows.size());
            uint32_t rank_k = static_cast<uint32_t>(g->a_k.desc.logical_shape.cols);
            uint32_t rank_v = static_cast<uint32_t>(g->a_v.desc.logical_shape.cols);
            uint32_t pad_rank_k = static_cast<uint32_t>(g->a_k.desc.padded_shape.cols);
            uint32_t pad_rank_v = static_cast<uint32_t>(g->a_v.desc.padded_shape.cols);
            uint32_t pad_rank_b_k = static_cast<uint32_t>(g->b_k->desc.padded_shape.cols);
            uint32_t pad_rank_b_v = static_cast<uint32_t>(g->b_v->desc.padded_shape.cols);

            uint64_t seg_version = view.segment_version_id != 0 ? view.segment_version_id : seg->segment_version;

            xkv_b_tile_lease lease_b_k;
            xkv_b_tile_lease lease_b_v;

            std::string batch_off_err;
            uint32_t b_feat_off_k = view.get_b_feature_offset_k(batch_head_dim_k, &batch_off_err);
            uint32_t b_feat_off_v = view.get_b_feature_offset_v(batch_head_dim_v, &batch_off_err);
            if (b_feat_off_k == UINT32_MAX || b_feat_off_v == UINT32_MAX) {
                throw std::runtime_error("xkv_read_attention_batch: failed to derive B feature offset: " + batch_off_err);
            }

            // B_K
            b_tile_cache_key key_k;
            key_k.segment_id = seg->segment_id;
            key_k.segment_version = seg_version;
            key_k.role = factor_role::b_k;
            key_k.owning_layer = view.owning_layer;
            key_k.kv_head = view.kv_head;
            key_k.tile_index = 0;
            key_k.codec_epoch = expected_stamp.codec_epoch;
            key_k.feature_offset = b_feat_off_k;
            key_k.feature_dim = batch_head_dim_k;
            key_k.device_id = config.device_id;
            key_k.domain = value_domain::canonical;
            key_k.desc_fingerprint = g->b_k->desc.fingerprint();

            // B_V
            b_tile_cache_key key_v;
            key_v.segment_id = seg->segment_id;
            key_v.segment_version = seg_version;
            key_v.role = factor_role::b_v;
            key_v.owning_layer = view.owning_layer;
            key_v.kv_head = view.kv_head;
            key_v.tile_index = 0;
            key_v.codec_epoch = expected_stamp.codec_epoch;
            key_v.feature_offset = b_feat_off_v;
            key_v.feature_dim = batch_head_dim_v;
            key_v.device_id = config.device_id;
            key_v.domain = value_domain::canonical;
            key_v.desc_fingerprint = g->b_v->desc.fingerprint();

            uint64_t * ptr_b_rows_k = nullptr;
            std::vector<uint64_t> fb_b_rows_k;
            if (scratch && batch_head_dim_k <= scratch->b_row_indices_capacity) {
                ptr_b_rows_k = scratch->b_row_indices;
            } else if (strict_mode(config)) {
                batch_res.status = xkv_read_status::workspace_exceeded;
                batch_res.error_message = "xkv_read_attention_batch: workspace b_row_indices short";
                return batch_res;
            } else {
                fb_b_rows_k.resize(batch_head_dim_k);
                ptr_b_rows_k = fb_b_rows_k.data();
            }
            for (uint32_t i = 0; i < batch_head_dim_k; ++i) ptr_b_rows_k[i] = b_feat_off_k + i;

            size_t need_b_k = 0;
            if (!safe_mul(static_cast<size_t>(batch_head_dim_k), static_cast<size_t>(pad_rank_b_k), need_b_k)) {
                batch_res.status = xkv_read_status::invalid_argument;
                batch_res.error_message = "xkv_read_attention_batch: overflow computing B_K tile size";
                return batch_res;
            }
            if (config.b_cache != nullptr) {
                lease_b_k = config.b_cache->get(key_k);
                if (lease_b_k.valid()) {
                    any_b_cache_hit = true;
                } else if (scratch && need_b_k <= scratch->tile_b_k_capacity) {
                    decode_rows_bounded(*g->b_k, ptr_b_rows_k, batch_head_dim_k, scratch->tile_b_k, need_b_k, config, scratch);
                    config.b_cache->put_and_lease_copy(key_k, scratch->tile_b_k, need_b_k);
                    lease_b_k = xkv_b_tile_lease::from_buffer(scratch->tile_b_k, need_b_k);
                } else if (strict_mode(config)) {
                    batch_res.status = xkv_read_status::workspace_exceeded;
                    batch_res.error_message = "xkv_read_attention_batch: workspace tile_b_k short";
                    return batch_res;
                } else {
                    std::vector<float> local_b_k(need_b_k);
                    decode_rows(*g->b_k, ptr_b_rows_k, batch_head_dim_k, local_b_k.data(), local_b_k.size(), value_domain::canonical);
                    lease_b_k = config.b_cache->put_and_lease(key_k, std::move(local_b_k));
                }
            } else if (scratch && need_b_k <= scratch->tile_b_k_capacity) {
                decode_rows_bounded(*g->b_k, ptr_b_rows_k, batch_head_dim_k, scratch->tile_b_k, need_b_k, config, scratch);
                lease_b_k = xkv_b_tile_lease::from_buffer(scratch->tile_b_k, need_b_k);
            } else if (strict_mode(config)) {
                batch_res.status = xkv_read_status::workspace_exceeded;
                batch_res.error_message = "xkv_read_attention_batch: workspace tile_b_k short";
                return batch_res;
            } else {
                std::vector<float> local_b_k(need_b_k);
                decode_rows(*g->b_k, ptr_b_rows_k, batch_head_dim_k, local_b_k.data(), local_b_k.size(), value_domain::canonical);
                auto rec = std::make_shared<b_tile_record>();
                rec->data = std::move(local_b_k);
                rec->in_cache = false;
                lease_b_k = xkv_b_tile_lease::uncached(rec);
            }

            uint64_t * ptr_b_rows_v = nullptr;
            std::vector<uint64_t> fb_b_rows_v;
            if (scratch && batch_head_dim_v <= scratch->b_row_indices_capacity) {
                ptr_b_rows_v = scratch->b_row_indices;
            } else if (strict_mode(config)) {
                batch_res.status = xkv_read_status::workspace_exceeded;
                batch_res.error_message = "xkv_read_attention_batch: workspace b_row_indices short";
                return batch_res;
            } else {
                fb_b_rows_v.resize(batch_head_dim_v);
                ptr_b_rows_v = fb_b_rows_v.data();
            }
            for (uint32_t i = 0; i < batch_head_dim_v; ++i) ptr_b_rows_v[i] = b_feat_off_v + i;

            size_t need_b_v = 0;
            if (!safe_mul(static_cast<size_t>(batch_head_dim_v), static_cast<size_t>(pad_rank_b_v), need_b_v)) {
                batch_res.status = xkv_read_status::invalid_argument;
                batch_res.error_message = "xkv_read_attention_batch: overflow computing B_V tile size";
                return batch_res;
            }
            if (config.b_cache != nullptr) {
                lease_b_v = config.b_cache->get(key_v);
                if (lease_b_v.valid()) {
                    any_b_cache_hit = true;
                } else if (scratch && need_b_v <= scratch->tile_b_v_capacity) {
                    decode_rows_bounded(*g->b_v, ptr_b_rows_v, batch_head_dim_v, scratch->tile_b_v, need_b_v, config, scratch);
                    config.b_cache->put_and_lease_copy(key_v, scratch->tile_b_v, need_b_v);
                    lease_b_v = xkv_b_tile_lease::from_buffer(scratch->tile_b_v, need_b_v);
                } else if (strict_mode(config)) {
                    batch_res.status = xkv_read_status::workspace_exceeded;
                    batch_res.error_message = "xkv_read_attention_batch: workspace tile_b_v short";
                    return batch_res;
                } else {
                    std::vector<float> local_b_v(need_b_v);
                    decode_rows(*g->b_v, ptr_b_rows_v, batch_head_dim_v, local_b_v.data(), local_b_v.size(), value_domain::canonical);
                    lease_b_v = config.b_cache->put_and_lease(key_v, std::move(local_b_v));
                }
            } else if (scratch && need_b_v <= scratch->tile_b_v_capacity) {
                decode_rows_bounded(*g->b_v, ptr_b_rows_v, batch_head_dim_v, scratch->tile_b_v, need_b_v, config, scratch);
                lease_b_v = xkv_b_tile_lease::from_buffer(scratch->tile_b_v, need_b_v);
            } else if (strict_mode(config)) {
                batch_res.status = xkv_read_status::workspace_exceeded;
                batch_res.error_message = "xkv_read_attention_batch: workspace tile_b_v short";
                return batch_res;
            } else {
                std::vector<float> local_b_v(need_b_v);
                decode_rows(*g->b_v, ptr_b_rows_v, batch_head_dim_v, local_b_v.data(), local_b_v.size(), value_domain::canonical);
                auto rec = std::make_shared<b_tile_record>();
                rec->data = std::move(local_b_v);
                rec->in_cache = false;
                lease_b_v = xkv_b_tile_lease::uncached(rec);
            }

            const float * b_k_ptr = lease_b_k.data();
            const float * b_v_ptr = lease_b_v.data();

            uint32_t cur_tile_size = std::min(tile_size, n_selected);

            float * ptr_tile_a_k = nullptr;
            std::vector<float> fb_tile_a_k;
            // Checked tile sizes (size_t, no truncation) shared by slice selection below.
            size_t need_a_k = 0, need_a_v = 0, need_tk = 0, need_tv = 0;
            if (!safe_mul(static_cast<size_t>(cur_tile_size), static_cast<size_t>(pad_rank_k), need_a_k) ||
                !safe_mul(static_cast<size_t>(cur_tile_size), static_cast<size_t>(pad_rank_v), need_a_v) ||
                !safe_mul(static_cast<size_t>(cur_tile_size), static_cast<size_t>(batch_head_dim_k), need_tk) ||
                !safe_mul(static_cast<size_t>(cur_tile_size), static_cast<size_t>(batch_head_dim_v), need_tv)) {
                batch_res.status = xkv_read_status::invalid_argument;
                batch_res.error_message = "xkv_read_attention_batch: overflow computing tile sizes";
                return batch_res;
            }
            if (scratch && need_a_k <= scratch->tile_a_k_capacity) {
                ptr_tile_a_k = scratch->tile_a_k;
            } else if (strict_mode(config)) {
                batch_res.status = xkv_read_status::workspace_exceeded;
                batch_res.error_message = "xkv_read_attention_batch: workspace tile_a_k short";
                return batch_res;
            } else {
                fb_tile_a_k.resize(need_a_k);
                ptr_tile_a_k = fb_tile_a_k.data();
            }

            float * ptr_tile_a_v = nullptr;
            std::vector<float> fb_tile_a_v;
            if (scratch && need_a_v <= scratch->tile_a_v_capacity) {
                ptr_tile_a_v = scratch->tile_a_v;
            } else if (strict_mode(config)) {
                batch_res.status = xkv_read_status::workspace_exceeded;
                batch_res.error_message = "xkv_read_attention_batch: workspace tile_a_v short";
                return batch_res;
            } else {
                fb_tile_a_v.resize(need_a_v);
                ptr_tile_a_v = fb_tile_a_v.data();
            }

            float * ptr_tile_k = nullptr;
            std::vector<float> fb_tile_k;
            if (scratch && need_tk <= scratch->tile_k_capacity) {
                ptr_tile_k = scratch->tile_k;
            } else if (strict_mode(config)) {
                batch_res.status = xkv_read_status::workspace_exceeded;
                batch_res.error_message = "xkv_read_attention_batch: workspace tile_k short";
                return batch_res;
            } else {
                fb_tile_k.resize(need_tk);
                ptr_tile_k = fb_tile_k.data();
            }

            float * ptr_tile_v = nullptr;
            std::vector<float> fb_tile_v;
            if (scratch && need_tv <= scratch->tile_v_capacity) {
                ptr_tile_v = scratch->tile_v;
            } else if (strict_mode(config)) {
                batch_res.status = xkv_read_status::workspace_exceeded;
                batch_res.error_message = "xkv_read_attention_batch: workspace tile_v short";
                return batch_res;
            } else {
                fb_tile_v.resize(need_tv);
                ptr_tile_v = fb_tile_v.data();
            }

            float * ptr_transformed_k = nullptr;
            std::vector<float> fb_transformed_k;
            if (scratch && batch_head_dim_k <= scratch->transformed_k_capacity) {
                ptr_transformed_k = scratch->transformed_k;
            } else if (strict_mode(config)) {
                batch_res.status = xkv_read_status::workspace_exceeded;
                batch_res.error_message = "xkv_read_attention_batch: workspace transformed_k short";
                return batch_res;
            } else {
                fb_transformed_k.resize(batch_head_dim_k);
                ptr_transformed_k = fb_transformed_k.data();
            }

            uint64_t * ptr_tile_row_indices = nullptr;
            std::vector<uint64_t> fb_tile_row_indices;
            if (scratch && cur_tile_size <= scratch->tile_row_indices_capacity) {
                ptr_tile_row_indices = scratch->tile_row_indices;
            } else if (strict_mode(config)) {
                batch_res.status = xkv_read_status::workspace_exceeded;
                batch_res.error_message = "xkv_read_attention_batch: workspace tile_row_indices short";
                return batch_res;
            } else {
                fb_tile_row_indices.resize(cur_tile_size);
                ptr_tile_row_indices = fb_tile_row_indices.data();
            }

            for (uint32_t t_begin = 0; t_begin < n_selected; t_begin += tile_size) {
                uint32_t t_count = std::min(tile_size, n_selected - t_begin);

                for (uint32_t i = 0; i < t_count; ++i) {
                    ptr_tile_row_indices[i] = view.selected_rows[t_begin + i];
                }

                decode_rows_bounded(g->a_k, ptr_tile_row_indices, t_count, ptr_tile_a_k, t_count * pad_rank_k, config, scratch);
                decode_rows_bounded(g->a_v, ptr_tile_row_indices, t_count, ptr_tile_a_v, t_count * pad_rank_v, config, scratch);

                gemm_reconstruct_tile(ptr_tile_a_k, pad_rank_k, b_k_ptr, pad_rank_b_k, rank_k, t_count, batch_head_dim_k, ptr_tile_k);
                gemm_reconstruct_tile(ptr_tile_a_v, pad_rank_v, b_v_ptr, pad_rank_b_v, rank_v, t_count, batch_head_dim_v, ptr_tile_v);

                for (uint32_t i = 0; i < t_count; ++i) {
                    uint32_t global_idx = t_begin + i;
                    int64_t storage_pos = view.storage_positions[global_idx];
                    uint32_t group_idx = (!view.group_indices.empty()) ? view.group_indices[global_idx] : 0;
                    segment_row_ref row_ref = view.get_row_ref(global_idx);

                    const float * raw_k_ptr = ptr_tile_k + i * batch_head_dim_k;
                    const float * cur_v_ptr = ptr_tile_v + i * batch_head_dim_v;

                    const float * final_k_ptr = raw_k_ptr;
                    if (phase_tx) {
                        phase_tx(raw_k_ptr, storage_pos, view.kv_head, ptr_transformed_k);
                        final_k_ptr = ptr_transformed_k;
                    }

                    // Running per-query head base into base_states; per-query update order is
                    // unchanged (tile rows in global order), so numerics are bitwise identical.
                    size_t head_base = 0;
                    for (size_t q = 0; q < n_queries; ++q) {
                        const auto & query = queries[q];

                        if (has_csr) {
                            // Exact per-query membership over this query's sorted CSR range.
                            // The union decode above is only a physical gather optimization.
                            // With per-query groups the matched entry additionally supplies
                            // this query's DDVR group (UINT32_MAX keeps the view group).
                            uint32_t c_begin = batch_selection.csr_ptrs[q];
                            uint32_t c_end   = batch_selection.csr_ptrs[q + 1];
                            bool member = false;
                            if (has_csr_groups) {
                                if (csr_entries != nullptr && c_end > c_begin) {
                                    const xkv_csr_entry * eb = csr_entries + c_begin;
                                    const xkv_csr_entry * ee = eb + (c_end - c_begin);
                                    const xkv_csr_entry * it = std::lower_bound(eb, ee, row_ref);
                                    if (it != ee && !(row_ref < it->ref) && !(it->ref < row_ref)) {
                                        member = true;
                                        // A prior query in this row loop may have overwritten
                                        // group_idx: reset explicitly on untracked entries.
                                        group_idx = (it->group_index != UINT32_MAX)
                                            ? it->group_index
                                            : ((!view.group_indices.empty()) ? view.group_indices[global_idx] : 0);
                                    }
                                }
                            } else if (csr_base != nullptr && c_end > c_begin) {
                                const segment_row_ref * rb = csr_base + c_begin;
                                member = std::binary_search(rb, rb + (c_end - c_begin), row_ref);
                            }
                            if (!member) {
                                head_base += query.n_q_heads;
                                continue;
                            }
                        } else if (!view.membership_mask.empty() && !view.membership_mask[global_idx]) {
                            head_base += query.n_q_heads;
                            continue;
                        }

                        if (query.causal_limit_pos >= 0 && storage_pos > query.causal_limit_pos) {
                            head_base += query.n_q_heads;
                            continue;
                        }

                        float scale = query.scale > 0.0f ? query.scale : 1.0f / std::sqrt(static_cast<float>(query.head_dim_k));

                        for (uint32_t qh = 0; qh < query.n_q_heads; ++qh) {
                            const float * q_ptr = get_q_ptr(query, group_idx, qh);
                            if (!q_ptr) continue;

                            float score = dot_product(q_ptr, final_k_ptr, batch_head_dim_k) * scale;
                            if (query.logit_softcap > 0.0f) {
                                score = query.logit_softcap * std::tanh(score / query.logit_softcap);
                            }

                            base_states[head_base + qh].update_single(score, cur_v_ptr, batch_head_dim_v);
                        }
                        head_base += query.n_q_heads;
                    }
                }
            }
        }

        // 3. Finalize each query in batch
        size_t fin_head_base = 0;
        for (size_t q = 0; q < n_queries; ++q) {
            const auto & query = queries[q];
            if (!query.sink_logits.empty()) {
                for (uint32_t qh = 0; qh < query.n_q_heads; ++qh) {
                    if (qh < query.sink_logits.size()) {
                        base_states[fin_head_base + qh].update_sink(query.sink_logits[qh], query.head_dim_v);
                    }
                }
            }

            batch_res.per_query[q].status = xkv_read_status::success;
            batch_res.per_query[q].output.resize(query.n_q_heads * query.head_dim_v);
            batch_res.per_query[q].b_cache_hit = any_b_cache_hit;
            for (uint32_t qh = 0; qh < query.n_q_heads; ++qh) {
                base_states[fin_head_base + qh].finalize(batch_res.per_query[q].output.data() + qh * query.head_dim_v, query.head_dim_v);
            }
            fin_head_base += query.n_q_heads;
        }

    } catch (const std::exception & e) {
        batch_res.status = xkv_read_status::codec_error;
        batch_res.per_query.clear();
        batch_res.error_message = std::string("xkv_read_attention_batch exception: ") + e.what();
        return batch_res;
    }

    // Stamp validation after
    if (store != nullptr) {
        xkv_snapshot_stamp stamp_after = store->current_stamp();
        if (stamp_after != expected_stamp) {
            batch_res.status = xkv_read_status::retry_stale_stamp;
            batch_res.per_query.clear();
            batch_res.error_message = "xkv_read_attention_batch: snapshot stamp changed during read";
            return batch_res;
        }
    }

    batch_res.status = xkv_read_status::success;
    batch_res.peak_workspace_bytes = peak_batch_ws;
    return batch_res;
}

// Convenience overloads accepting an explicit workspace
xkv_read_result xkv_read_attention(
    xkv_reader_workspace & ws,
    const xkv_query_input & query,
    const std::vector<xkv_hot_row> & hot_rows,
    const std::vector<xkv_segment_read_view> & segment_views,
    const phase_transform_fn & phase_tx,
    const xkv_snapshot_stamp & expected_stamp,
    llama_xkv_cache_store * store,
    const xkv_reader_config & config
) {
    xkv_reader_config cfg = config;
    cfg.workspace = &ws;
    return xkv_read_attention(query, hot_rows, segment_views, phase_tx, expected_stamp, store, cfg);
}

xkv_batch_read_result xkv_read_attention_batch(
    xkv_reader_workspace & ws,
    const std::vector<xkv_query_input> & queries,
    const std::vector<xkv_hot_row> & hot_rows,
    const std::vector<xkv_segment_read_view> & segment_views,
    const sr_batch_selection_result & batch_selection,
    const phase_transform_fn & phase_tx,
    const xkv_snapshot_stamp & expected_stamp,
    llama_xkv_cache_store * store,
    const xkv_reader_config & config
) {
    xkv_reader_config cfg = config;
    cfg.workspace = &ws;
    return xkv_read_attention_batch(queries, hot_rows, segment_views, batch_selection, phase_tx, expected_stamp, store, cfg);
}

// ============================================================================
// Dense reference attention oracle
// ============================================================================

std::vector<float> xkv_dense_attention_reference(
    const xkv_query_input & query,
    const std::vector<xkv_hot_row> & hot_rows,
    const std::vector<std::vector<float>> & cold_keys,
    const std::vector<std::vector<float>> & cold_values,
    const std::vector<uint32_t> & cold_group_indices,
    const std::vector<bool> & cold_mask
) {
    std::vector<float> result(query.n_q_heads * query.head_dim_v, 0.0f);

    float scale = query.scale;
    if (scale <= 0.0f) {
        scale = 1.0f / std::sqrt(static_cast<float>(query.head_dim_k));
    }

    for (uint32_t qh = 0; qh < query.n_q_heads; ++qh) {
        std::vector<float> scores;
        std::vector<const float *> values;

        // 1. Hot rows
        for (const auto & hr : hot_rows) {
            if (!hr.is_valid || !hr.k_ptr || !hr.v_ptr) continue;
            if (query.causal_limit_pos >= 0 && hr.storage_pos > query.causal_limit_pos) continue;
            if (!hr.is_visible_to_query(query.query_index)) continue;

            const uint32_t hr_group =
                hr.query_group_indices.empty() ? hr.group_index : hr.query_group_indices[0];
            const float * q_ptr = get_q_ptr(query, hr_group, qh);
            if (!q_ptr) continue;

            float s = dot_product(q_ptr, hr.k_ptr, query.head_dim_k) * scale;
            if (query.logit_softcap > 0.0f) {
                s = query.logit_softcap * std::tanh(s / query.logit_softcap);
            }
            scores.push_back(s);
            values.push_back(hr.v_ptr);
        }

        // 2. Cold rows
        for (size_t i = 0; i < cold_keys.size(); ++i) {
            if (!cold_mask.empty() && !cold_mask[i]) continue;

            uint32_t group_idx = (i < cold_group_indices.size()) ? cold_group_indices[i] : 0;
            const float * q_ptr = get_q_ptr(query, group_idx, qh);
            if (!q_ptr) continue;

            float s = dot_product(q_ptr, cold_keys[i].data(), query.head_dim_k) * scale;
            if (query.logit_softcap > 0.0f) {
                s = query.logit_softcap * std::tanh(s / query.logit_softcap);
            }
            scores.push_back(s);
            values.push_back(cold_values[i].data());
        }

        // 3. Sink
        bool has_sink = false;
        float sink_s = -INFINITY;
        if (qh < query.sink_logits.size() && std::isfinite(query.sink_logits[qh])) {
            has_sink = true;
            sink_s = query.sink_logits[qh];
        }

        // 4. Softmax
        float max_s = -INFINITY;
        for (float s : scores) {
            if (s > max_s) max_s = s;
        }
        if (has_sink && sink_s > max_s) {
            max_s = sink_s;
        }

        if (max_s <= -INFINITY) {
            // Empty / all-masked
            continue;
        }

        double sum_exp = 0.0;
        for (float s : scores) {
            sum_exp += std::exp(static_cast<double>(s - max_s));
        }
        if (has_sink) {
            sum_exp += std::exp(static_cast<double>(sink_s - max_s));
        }

        if (sum_exp <= 0.0) {
            continue;
        }

        float * out_head = result.data() + qh * query.head_dim_v;
        for (size_t i = 0; i < scores.size(); ++i) {
            float weight = static_cast<float>(std::exp(static_cast<double>(scores[i] - max_s)) / sum_exp);
            const float * v = values[i];
            for (uint32_t d = 0; d < query.head_dim_v; ++d) {
                out_head[d] += weight * v[d];
            }
        }
    }

    return result;
}

} // namespace llama_xkv
