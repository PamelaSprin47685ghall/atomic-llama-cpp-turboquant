#include "llama-xkv-landmark.h"
#include "llama-xkv-factor.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

namespace llama_xkv {

// ------------------------------------------------------------------------------------------------
// landmark_table implementation
// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------
// Internal checked arithmetic + shared exact-scoring helpers (no heap in bounded path).
// ------------------------------------------------------------------------------------------------
namespace landmark_detail {
inline bool ck_add(size_t a, size_t b, size_t & out) {
    if (SIZE_MAX - a < b) return false;
    out = a + b;
    return true;
}
inline bool ck_mul(size_t a, size_t b, size_t & out) {
    if (a != 0 && b > SIZE_MAX / a) return false;
    out = a * b;
    return true;
}
inline bool ck_mul_u32(uint32_t a, uint32_t b, size_t & out) {
    return ck_mul(static_cast<size_t>(a), static_cast<size_t>(b), out);
}
// Exact strict weak ordering for top-k: strictly greater score first, then strictly
// smaller stable identity. No epsilon comparator anywhere: near-equal scores never cycle.
// Non-finite scores fail closed upstream (validated before sort).
struct scored_candidate {
    uint32_t frag_index = 0;
    float score = 0.0f;
    float error_margin = 0.0f;
};
inline bool candidate_before(const scored_candidate & a, const scored_candidate & b) {
    if (a.score > b.score) return true;
    if (b.score > a.score) return false;
    return a.frag_index < b.frag_index;
}
// Insertion sort over caller scratch (no heap, deterministic, stable under candidate_before).
inline void sort_candidates(scored_candidate * cands, size_t n) {
    for (size_t i = 1; i < n; ++i) {
        scored_candidate cur = cands[i];
        size_t j = i;
        while (j > 0 && candidate_before(cur, cands[j - 1])) {
            cands[j] = cands[j - 1];
            --j;
        }
        cands[j] = cur;
    }
}
// In-place sort + unique for segment_row_ref arrays (no heap).
inline void sort_row_refs(segment_row_ref * refs, size_t n) {
    for (size_t i = 1; i < n; ++i) {
        segment_row_ref cur = refs[i];
        size_t j = i;
        while (j > 0 && cur < refs[j - 1]) {
            refs[j] = refs[j - 1];
            --j;
        }
        refs[j] = cur;
    }
}
inline size_t unique_row_refs(segment_row_ref * refs, size_t n) {
    if (n == 0) return 0;
    size_t w = 1;
    for (size_t r = 1; r < n; ++r) {
        if (refs[r] != refs[w - 1]) refs[w++] = refs[r];
    }
    return w;
}
inline bool row_ref_present_sorted(const segment_row_ref * refs, size_t n, const segment_row_ref & q) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (refs[mid] < q) lo = mid + 1;
        else hi = mid;
    }
    return lo < n && refs[lo] == q;
}
// Derive the landmark codec seed/descriptor inputs ONLY from immutable source, profile,
// config and slice identity. Segment positions/ids MUST NOT influence the seed: the same
// logical content re-packed at another position yields the identical code stream.
inline uint64_t derive_landmark_seed(const xkv_segment & seg, const xkv_factor_group_payload & group,
    uint32_t feature_offset, uint32_t feature_dim, ggml_type landmark_type, uint64_t phase_fp) {
    uint64_t h = 0xcbf29ce484222325ULL;
    auto mix = [&h](uint64_t v) { h ^= v; h *= 0x100000001b3ULL; };
    mix(seg.source_fingerprint);
    mix(seg.profile_fingerprint);
    mix(seg.layer_group_map_fingerprint);
    mix(group.descriptor_fingerprint);
    mix(group.config_fingerprint);
    mix(static_cast<uint64_t>(feature_offset));
    mix(static_cast<uint64_t>(feature_dim));
    mix(static_cast<uint64_t>(landmark_type));
    mix(phase_fp);
    return h;
}
// Validate [feature_offset, feature_offset + feature_dim) against the EXACT owning-layer
// slice inside the group (not just the total B width). False + err on any mismatch.
inline bool validate_feature_slice(const xkv_factor_group_payload & group, uint32_t owning_layer,
    uint32_t feature_offset, uint32_t feature_dim, std::string * err) {
    int32_t li = group.find_owning_layer_index(owning_layer);
    if (li < 0) {
        if (err) *err = "encode_fragment_landmark: owning_layer not in factor group";
        return false;
    }
    size_t s = static_cast<size_t>(li);
    if (s >= group.layer_feature_offsets_k.size() || s >= group.layer_feature_dims_k.size()) {
        if (err) *err = "encode_fragment_landmark: group layer slice metadata missing";
        return false;
    }
    uint64_t lay_off = group.layer_feature_offsets_k[s];
    uint64_t lay_dim = group.layer_feature_dims_k[s];
    uint64_t end = 0;
    if (!ck_add(feature_offset, feature_dim, end)) {
        if (err) *err = "encode_fragment_landmark: feature slice overflow";
        return false;
    }
    if (feature_dim == 0 || feature_offset < lay_off || end > lay_off + lay_dim) {
        if (err) *err = "encode_fragment_landmark: feature slice outside owning-layer slice";
        return false;
    }
    return true;
}
} // namespace landmark_detail

landmark_table::landmark_table(size_t max_bytes)
    : max_bytes_(max_bytes) {
}

std::shared_ptr<const legal_fragment> landmark_table::find(const landmark_fragment_key & key) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = table_.find(key);
    if (it != table_.end()) {
        // Move key to front of deterministic LRU list
        lru_order_.splice(lru_order_.begin(), lru_order_, it->second.lru_it);
        return it->second.frag;
    }
    return nullptr;
}

bool landmark_table::insert(std::shared_ptr<const legal_fragment> frag) {
    if (!frag) return false;
    std::lock_guard<std::mutex> lock(mutex_);
    size_t frag_bytes = frag->allocated_bytes();
    if (frag_bytes > max_bytes_) {
        // Reject oversize insert to strictly respect memory budget
        return false;
    }

    auto it = table_.find(frag->key);
    // Atomic preflight: prove replacement + evictable LRU bytes cover the insert BEFORE
    // mutating any accounting. Pinned/leased entries are never evictable, so a pinned
    // table that cannot free enough space refuses the insert with zero state change.
    size_t old_bytes = 0;
    if (it != table_.end()) {
        old_bytes = it->second.frag->allocated_bytes();
    }
    size_t base = (current_bytes_ >= old_bytes) ? (current_bytes_ - old_bytes) : 0;
    size_t need = 0;
    if (!landmark_detail::ck_add(base, frag_bytes, need)) return false; // overflow: fail closed
    if (need > max_bytes_) {
        size_t required_free = need - max_bytes_;
        size_t evictable = 0;
        for (auto rit = lru_order_.rbegin(); rit != lru_order_.rend(); ++rit) {
            if (*rit == frag->key) continue; // replaced entry already excluded via base
            auto tit = table_.find(*rit);
            if (tit == table_.end()) continue;
            if (tit->second.frag.use_count() > 1) continue; // externally leased: pinned
            size_t eb = tit->second.frag->allocated_bytes();
            if (!landmark_detail::ck_add(evictable, eb, evictable)) return false;
            if (evictable >= required_free) break;
        }
        if (evictable < required_free) return false; // no mutation performed
    }
    // Commit: replace existing entry, insert the new one, then evict the LRU tail.
    // Preflight above guarantees eviction can restore the hard bound.
    if (it != table_.end()) {
        current_bytes_ = base;
        lru_order_.erase(it->second.lru_it);
        table_.erase(it);
    }
    lru_order_.push_front(frag->key);
    table_[frag->key] = {frag, lru_order_.begin()};
    current_bytes_ = need;
    evict_if_needed();
    return current_bytes_ <= max_bytes_;
}

void landmark_table::evict_if_needed() {
    // Deterministic LRU eviction from the back of lru_order_
    // Entries with external leases (use_count > 1) are preserved; only unpinned/unleased entries can be evicted.
    auto it_lru = lru_order_.end();
    while (current_bytes_ > max_bytes_ && it_lru != lru_order_.begin()) {
        --it_lru;
        auto tbl_it = table_.find(*it_lru);
        if (tbl_it != table_.end()) {
            // Check if externally leased
            if (tbl_it->second.frag.use_count() > 1) {
                continue; // leased by caller, do not evict
            }
            size_t entry_bytes = tbl_it->second.frag->allocated_bytes();
            current_bytes_ = (current_bytes_ >= entry_bytes) ? (current_bytes_ - entry_bytes) : 0;
            table_.erase(tbl_it);
            it_lru = lru_order_.erase(it_lru);
        }
    }
}

void landmark_table::invalidate_segment(uint64_t segment_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = table_.begin(); it != table_.end(); ) {
        if (it->first.segment_id == segment_id) {
            size_t entry_bytes = it->second.frag->allocated_bytes();
            current_bytes_ = (current_bytes_ >= entry_bytes) ? (current_bytes_ - entry_bytes) : 0;
            lru_order_.erase(it->second.lru_it);
            it = table_.erase(it);
        } else {
            ++it;
        }
    }
}

void landmark_table::invalidate_epoch(uint64_t segment_id, uint64_t live_epoch) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = table_.begin(); it != table_.end(); ) {
        if (it->first.segment_id == segment_id && it->first.live_epoch != live_epoch) {
            size_t entry_bytes = it->second.frag->allocated_bytes();
            current_bytes_ = (current_bytes_ >= entry_bytes) ? (current_bytes_ - entry_bytes) : 0;
            lru_order_.erase(it->second.lru_it);
            it = table_.erase(it);
        } else {
            ++it;
        }
    }
}

void landmark_table::invalidate_content_epoch(uint64_t segment_id, uint64_t content_epoch) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = table_.begin(); it != table_.end(); ) {
        if (it->first.segment_id == segment_id && it->first.content_epoch != content_epoch) {
            size_t entry_bytes = it->second.frag->allocated_bytes();
            current_bytes_ = (current_bytes_ >= entry_bytes) ? (current_bytes_ - entry_bytes) : 0;
            lru_order_.erase(it->second.lru_it);
            it = table_.erase(it);
        } else {
            ++it;
        }
    }
}

void landmark_table::invalidate_codec_epoch(uint64_t segment_id, uint64_t codec_epoch) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = table_.begin(); it != table_.end(); ) {
        if (it->first.segment_id == segment_id && it->first.codec_epoch != codec_epoch) {
            size_t entry_bytes = it->second.frag->allocated_bytes();
            current_bytes_ = (current_bytes_ >= entry_bytes) ? (current_bytes_ - entry_bytes) : 0;
            lru_order_.erase(it->second.lru_it);
            it = table_.erase(it);
        } else {
            ++it;
        }
    }
}

void landmark_table::invalidate_stamp(const xkv_snapshot_stamp & current_stamp) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = table_.begin(); it != table_.end(); ) {
        if (it->first.live_epoch != current_stamp.live_epoch ||
            it->first.content_epoch != current_stamp.content_epoch ||
            it->first.codec_epoch != current_stamp.codec_epoch ||
            it->first.binding_epoch != current_stamp.binding_epoch ||
            it->first.view_topology_epoch != current_stamp.view.topology_epoch ||
            it->first.view_publish_epoch != current_stamp.view.publish_epoch ||
            it->first.view_layout_epoch != current_stamp.view.layout_epoch) {
            size_t entry_bytes = it->second.frag->allocated_bytes();
            current_bytes_ = (current_bytes_ >= entry_bytes) ? (current_bytes_ - entry_bytes) : 0;
            lru_order_.erase(it->second.lru_it);
            it = table_.erase(it);
        } else {
            ++it;
        }
    }
}
void landmark_table::invalidate_binding_epoch(uint64_t segment_id, uint64_t binding_epoch) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = table_.begin(); it != table_.end(); ) {
        if (it->first.segment_id == segment_id && it->first.binding_epoch != binding_epoch) {
            size_t entry_bytes = it->second.frag->allocated_bytes();
            current_bytes_ = (current_bytes_ >= entry_bytes) ? (current_bytes_ - entry_bytes) : 0;
            lru_order_.erase(it->second.lru_it);
            it = table_.erase(it);
        } else {
            ++it;
        }
    }
}
void landmark_table::invalidate_view(uint64_t segment_id, const xkv_snapshot_stamp & current_stamp) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = table_.begin(); it != table_.end(); ) {
        if (it->first.segment_id == segment_id &&
            (it->first.view_topology_epoch != current_stamp.view.topology_epoch ||
             it->first.view_publish_epoch != current_stamp.view.publish_epoch ||
             it->first.view_layout_epoch != current_stamp.view.layout_epoch)) {
            size_t entry_bytes = it->second.frag->allocated_bytes();
            current_bytes_ = (current_bytes_ >= entry_bytes) ? (current_bytes_ - entry_bytes) : 0;
            lru_order_.erase(it->second.lru_it);
            it = table_.erase(it);
        } else {
            ++it;
        }
    }
}
void landmark_table::invalidate_phase_source(uint64_t segment_id, uint64_t phase_tx_fingerprint,
    uint64_t source_fingerprint) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = table_.begin(); it != table_.end(); ) {
        if (it->first.segment_id == segment_id &&
            (it->first.phase_tx_fingerprint != phase_tx_fingerprint ||
             it->first.source_fingerprint != source_fingerprint)) {
            size_t entry_bytes = it->second.frag->allocated_bytes();
            current_bytes_ = (current_bytes_ >= entry_bytes) ? (current_bytes_ - entry_bytes) : 0;
            lru_order_.erase(it->second.lru_it);
            it = table_.erase(it);
        } else {
            ++it;
        }
    }
}
bool landmark_table::can_cache(const legal_fragment & frag, const xkv_snapshot_stamp & current_stamp,
    uint64_t phase_tx_fingerprint, std::string * err) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto & k = frag.key;
    if (k.live_epoch != current_stamp.live_epoch || k.content_epoch != current_stamp.content_epoch ||
        k.codec_epoch != current_stamp.codec_epoch || k.binding_epoch != current_stamp.binding_epoch ||
        k.view_topology_epoch != current_stamp.view.topology_epoch ||
        k.view_publish_epoch != current_stamp.view.publish_epoch ||
        k.view_layout_epoch != current_stamp.view.layout_epoch) {
        if (err) *err = "can_cache: fragment stamp epochs stale";
        return false;
    }
    if (k.phase_tx_fingerprint != phase_tx_fingerprint) {
        if (err) *err = "can_cache: fragment phase fingerprint stale";
        return false;
    }
    if (frag.allocated_bytes() > max_bytes_) {
        if (err) *err = "can_cache: fragment oversize for table budget";
        return false;
    }
    return true;
}  

void landmark_table::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    table_.clear();
    lru_order_.clear();
    current_bytes_ = 0;
}

size_t landmark_table::total_allocated_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return current_bytes_;
}

size_t landmark_table::fragment_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return table_.size();
}

// ------------------------------------------------------------------------------------------------
// Fragment construction
// ------------------------------------------------------------------------------------------------

bool build_legal_fragment_plans(const xkv_segment & segment, const xkv_snapshot_stamp & stamp,
    uint64_t segment_version, uint64_t default_storage_generation, uint32_t owning_layer,
    uint32_t kv_head, const row_meta * rows, size_t n_rows, uint32_t chunk_size,
    uint64_t phase_tx_fingerprint, int64_t causal_cutoff, landmark_fragment_plan * out_plans,
    size_t plans_cap, size_t * out_n_plans, uint32_t * out_row_indices, int64_t * out_positions,
    uint64_t * out_payload_ids, uint64_t * out_generations, size_t rows_cap, size_t * out_n_rows,
    std::string * err) {
    auto fail = [&](const char * msg) {
        if (err) *msg ? *err = msg : *err = "build_legal_fragment_plans: invalid input";
        return false;
    };
    if (out_n_plans) *out_n_plans = 0;
    if (out_n_rows) *out_n_rows = 0;
    if (chunk_size == 0) return fail("build_legal_fragment_plans: chunk_size is 0");
    if (n_rows == 0) return true; // empty input plans zero fragments (reference parity)
    if (!rows || !out_plans || !out_row_indices || !out_positions || !out_payload_ids ||
        !out_generations || !out_n_plans || !out_n_rows) {
        return fail("build_legal_fragment_plans: null buffer");
    }
    if (segment.segment_version != segment_version) {
        return fail("build_legal_fragment_plans: segment_version mismatch");
    }
    for (size_t i = 0; i < n_rows; ++i) {
        const row_meta & r = rows[i];
        if (r.segment_row >= segment.n_rows) {
            return fail("build_legal_fragment_plans: row index out of segment bounds");
        }
        if (i > 0 && r.storage_pos <= rows[i - 1].storage_pos) {
            return fail("build_legal_fragment_plans: rows must be strictly sorted/unique");
        }
        if (r.segment_row < segment.row_payload_ids.size() &&
            segment.row_payload_ids[r.segment_row] != r.payload_id) {
            return fail("build_legal_fragment_plans: payload_id mismatch");
        }
        if (r.segment_row < segment.live_rows.size() &&
            segment.live_rows[r.segment_row] != r.is_live) {
            return fail("build_legal_fragment_plans: live status mismatch");
        }
    }
    size_t n_plans = 0;
    size_t n_out_rows = 0;
    size_t i = 0;
    while (i < n_rows) {
        const row_meta & r = rows[i];
        if (!r.is_live || !r.reader_visible ||
            (causal_cutoff >= 0 && r.storage_pos > causal_cutoff)) {
            ++i;
            continue;
        }
        if (n_plans >= plans_cap) return fail("build_legal_fragment_plans: plans capacity short");
        landmark_fragment_plan & plan = out_plans[n_plans];
        plan.key.segment_id = segment.segment_id;
        plan.key.segment_version = segment_version;
        plan.key.storage_generation =
            (r.storage_generation != 0) ? r.storage_generation : default_storage_generation;
        plan.key.owning_layer = owning_layer;
        plan.key.kv_head = kv_head;
        plan.key.live_epoch = stamp.live_epoch;
        plan.key.content_epoch = stamp.content_epoch;
        plan.key.codec_epoch = stamp.codec_epoch;
        plan.key.view_topology_epoch = stamp.view.topology_epoch;
        plan.key.view_publish_epoch = stamp.view.publish_epoch;
        plan.key.view_layout_epoch = stamp.view.layout_epoch;
        plan.key.phase_tx_fingerprint = phase_tx_fingerprint;
        plan.key.binding_epoch = stamp.binding_epoch;
        plan.key.source_fingerprint = segment.source_fingerprint;
        plan.key.landmark_codec_fp = 0; // set at encode time
        plan.key.run_id = r.run_id;
        plan.key.storage_pos0 = r.storage_pos;
        plan.key.phase_delta = r.storage_pos - r.virtual_pos;
        plan.key.ddvr_group = r.ddvr_group;
        plan.key.visibility = r.visibility;
        plan.key.causal_cutoff = causal_cutoff;
        plan.row_begin = r.segment_row;
        plan.row_count = 0;
        plan.rows_offset = static_cast<uint32_t>(n_out_rows);
        size_t j = i;
        int64_t prev_pos = 0;
        bool have_prev = false;
        while (j < n_rows && plan.row_count < chunk_size) {
            const row_meta & nr = rows[j];
            if (!nr.is_live || !nr.reader_visible ||
                (causal_cutoff >= 0 && nr.storage_pos > causal_cutoff)) {
                break;
            }
            if (plan.row_count > 0) {
                if (nr.ddvr_group != plan.key.ddvr_group) break;
                if (nr.visibility != plan.key.visibility || nr.run_id != plan.key.run_id) break;
                if (nr.storage_pos - nr.virtual_pos != plan.key.phase_delta) break;
                if (!have_prev || nr.storage_pos != prev_pos + 1) break;
            }
            if (n_out_rows >= rows_cap) return fail("build_legal_fragment_plans: rows capacity short");
            out_row_indices[n_out_rows] = nr.segment_row;
            out_positions[n_out_rows] = nr.storage_pos;
            out_payload_ids[n_out_rows] = nr.payload_id;
            out_generations[n_out_rows] =
                (nr.storage_generation != 0) ? nr.storage_generation : default_storage_generation;
            ++n_out_rows;
            ++plan.row_count;
            prev_pos = nr.storage_pos;
            have_prev = true;
            ++j;
        }
        uint64_t live_fp = 0xcbf29ce484222325ULL;
        for (uint32_t k = 0; k < plan.row_count; ++k) {
            live_fp ^= out_payload_ids[plan.rows_offset + k]; live_fp *= 0x100000001b3ULL;
            live_fp ^= out_generations[plan.rows_offset + k]; live_fp *= 0x100000001b3ULL;
            live_fp ^= static_cast<uint64_t>(out_row_indices[plan.rows_offset + k]);
            live_fp *= 0x100000001b3ULL;
        }
        plan.key.live_set_fingerprint = live_fp;
        ++n_plans;
        i = j;
    }
    *out_n_plans = n_plans;
    *out_n_rows = n_out_rows;
    return true;
}
std::vector<legal_fragment> build_legal_fragments(
    const xkv_segment & segment,
    const xkv_snapshot_stamp & stamp,
    uint64_t segment_version,
    uint64_t default_storage_generation,
    uint32_t owning_layer,
    uint32_t kv_head,
    const std::vector<row_meta> & rows,
    uint32_t chunk_size,
    uint64_t phase_tx_fingerprint,
    int64_t causal_cutoff
) {
    std::vector<legal_fragment> fragments;
    if (rows.empty() || chunk_size == 0) {
        return fragments;
    }

    // Exact segment version validation
    if (segment.segment_version != segment_version) {
        throw std::invalid_argument("build_legal_fragments: segment_version mismatch with segment.segment_version");
    }

    // Validate that input rows are strictly sorted by storage_pos, strictly unique, within segment bounds,
    // and match segment live status and stable payload ID.
    for (size_t i = 0; i < rows.size(); ++i) {
        const auto & r = rows[i];
        if (r.segment_row >= segment.n_rows) {
            throw std::out_of_range("build_legal_fragments: row index out of segment bounds");
        }
        if (i > 0 && r.storage_pos <= rows[i - 1].storage_pos) {
            throw std::invalid_argument("build_legal_fragments: input rows must be strictly sorted and unique by storage_pos");
        }
        // Verify segment payload ID match
        if (r.segment_row < segment.row_payload_ids.size()) {
            if (segment.row_payload_ids[r.segment_row] != r.payload_id) {
                throw std::invalid_argument("build_legal_fragments: row payload_id does not match segment row_payload_ids");
            }
        }
        // Verify segment live status match
        if (r.segment_row < segment.live_rows.size()) {
            if (segment.live_rows[r.segment_row] != r.is_live) {
                throw std::invalid_argument("build_legal_fragments: row is_live does not match segment live_rows status");
            }
        }
    }

    // Partition rows into homogeneous legal runs.
    size_t i = 0;
    while (i < rows.size()) {
        const auto & r = rows[i];
        if (!r.is_live || !r.reader_visible || (causal_cutoff >= 0 && r.storage_pos > causal_cutoff)) {
            ++i;
            continue;
        }

        uint64_t gen = (r.storage_generation != 0) ? r.storage_generation : default_storage_generation;

        legal_fragment frag;
        frag.key.segment_id = segment.segment_id;
        frag.key.segment_version = segment_version;
        frag.key.storage_generation = gen;
        frag.key.owning_layer = owning_layer;
        frag.key.kv_head = kv_head;
        frag.key.live_epoch = stamp.live_epoch;
        frag.key.content_epoch = stamp.content_epoch;
        frag.key.codec_epoch = stamp.codec_epoch;
        frag.key.view_topology_epoch = stamp.view.topology_epoch;
        frag.key.view_publish_epoch = stamp.view.publish_epoch;
        frag.key.view_layout_epoch = stamp.view.layout_epoch;
        frag.key.phase_tx_fingerprint = phase_tx_fingerprint;
        frag.key.run_id = r.run_id;
        frag.key.binding_epoch = stamp.binding_epoch;
        frag.key.source_fingerprint = segment.source_fingerprint;
        frag.key.landmark_codec_fp = 0; // set at encode time
        frag.key.storage_pos0 = r.storage_pos;
        frag.key.phase_delta = r.storage_pos - r.virtual_pos;
        frag.key.ddvr_group = r.ddvr_group;
        frag.parent_group_id = r.ddvr_group;
        frag.key.visibility = r.visibility;
        frag.key.causal_cutoff = causal_cutoff;

        frag.row_begin = r.segment_row;
        frag.row_indices.push_back(r.segment_row);
        frag.storage_positions.push_back(r.storage_pos);
        frag.payload_ids.push_back(r.payload_id);
        frag.generations.push_back(gen);

        size_t j = i + 1;
        while (j < rows.size() && frag.row_indices.size() < chunk_size) {
            const auto & next_r = rows[j];
            if (!next_r.is_live || !next_r.reader_visible || (causal_cutoff >= 0 && next_r.storage_pos > causal_cutoff)) {
                break;
            }
            if (next_r.visibility != frag.key.visibility || next_r.run_id != frag.key.run_id) {
                break;
            }
            if (next_r.ddvr_group != frag.key.ddvr_group) {
                break;
            }
            int64_t next_phase = next_r.storage_pos - next_r.virtual_pos;
            if (next_phase != frag.key.phase_delta) {
                break;
            }
            if (next_r.storage_pos != frag.storage_positions.back() + 1) {
                break;
            }
            uint64_t next_gen = (next_r.storage_generation != 0) ? next_r.storage_generation : default_storage_generation;
            frag.row_indices.push_back(next_r.segment_row);
            frag.storage_positions.push_back(next_r.storage_pos);
            frag.payload_ids.push_back(next_r.payload_id);
            frag.generations.push_back(next_gen);
            ++j;
        }

        frag.row_count = static_cast<uint32_t>(frag.row_indices.size());

        // Compute deterministic live-set fingerprint for this fragment
        uint64_t live_fp = 0xcbf29ce484222325ULL;
        for (size_t k = 0; k < frag.row_count; ++k) {
            live_fp ^= frag.payload_ids[k]; live_fp *= 0x100000001b3ULL;
            live_fp ^= frag.generations[k]; live_fp *= 0x100000001b3ULL;
            live_fp ^= static_cast<uint64_t>(frag.row_indices[k]); live_fp *= 0x100000001b3ULL;
        }
        frag.key.live_set_fingerprint = live_fp;

        fragments.push_back(std::move(frag));
        i = j;
    }

    return fragments;
}

// ------------------------------------------------------------------------------------------------
// Landmark Encoding
// ------------------------------------------------------------------------------------------------

void encode_fragment_landmark(
    legal_fragment & frag,
    const xkv_segment & segment,
    uint32_t feature_offset,
    uint32_t feature_dim,
    const phase_transform_fn & phase_tx,
    uint64_t phase_tx_fingerprint,
    ggml_type landmark_type
) {
    const auto * g = segment.find_group_for_layer(frag.key.owning_layer);
    if (!g) {
        throw std::invalid_argument("encode_fragment_landmark: no factor group found in segment bundle for owning_layer " + std::to_string(frag.key.owning_layer));
    }
    encode_fragment_landmark(frag, segment, *g, feature_offset, feature_dim, phase_tx, phase_tx_fingerprint, landmark_type);
}

void encode_fragment_landmark(
    legal_fragment & frag,
    const xkv_segment & segment,
    const xkv_factor_group_payload & group,
    uint32_t feature_offset,
    uint32_t feature_dim,
    const phase_transform_fn & phase_tx,
    uint64_t phase_tx_fingerprint,
    ggml_type landmark_type
) {
    if (frag.row_count == 0 || feature_dim == 0) {
        throw std::invalid_argument("encode_fragment_landmark: empty fragment or zero feature_dim");
    }
    // Attached intact fragments reference immutable base bytes; re-encoding would orphan
    // the reference. Detach explicitly (base_landmark.reset()) or rebuild a partial copy.
    if (frag.uses_base_landmark()) {
        throw std::invalid_argument("encode_fragment_landmark: attached base fragment");
    }
    if (!group.b_k) {
        throw std::invalid_argument("encode_fragment_landmark: group.b_k is null");
    }

    // Validate A and B ranks match
    if (group.a_k.desc.logical_shape.cols != group.b_k->desc.logical_shape.cols ||
        group.a_k.desc.padded_shape.cols != group.b_k->desc.padded_shape.cols) {
        throw std::invalid_argument("encode_fragment_landmark: rank mismatch between A_K and B_K");
    }

    // Explicit error on unsupported landmark codec (no silent Q8 fallback)
    if (landmark_type != GGML_TYPE_Q8_0 && landmark_type != GGML_TYPE_TURBO4_0 &&
        landmark_type != GGML_TYPE_F32 && landmark_type != GGML_TYPE_F16) {
        throw std::invalid_argument("encode_fragment_landmark: unsupported landmark_type " + std::string(ggml_type_name(landmark_type)));
    }

    // Validate feature_offset and feature_dim against B_K matrix shape
    uint64_t b_total_features = group.b_k->desc.logical_shape.rows;
    if (uint64_t(feature_offset) + feature_dim > b_total_features) {
        throw std::out_of_range("encode_fragment_landmark: feature slice exceeds B_K feature dimension");
    }
    // Validate the slice against the EXACT owning-layer/head region inside the group:
    // a slice inside the total B width but outside this layer's region is rejected.
    {
        std::string slice_err;
        if (!landmark_detail::validate_feature_slice(group, frag.key.owning_layer,
            feature_offset, feature_dim, &slice_err)) {
            throw std::out_of_range(slice_err);
        }
    }
    // Validate fragment row bindings: counts, segment bounds, sorted unique positions,
    // and per-row generation presence when the vectors are populated.
    if (frag.row_indices.size() < frag.row_count ||
        frag.storage_positions.size() < frag.row_count) {
        throw std::invalid_argument("encode_fragment_landmark: fragment row vectors short");
    }
    if (!frag.payload_ids.empty() && frag.payload_ids.size() < frag.row_count) {
        throw std::invalid_argument("encode_fragment_landmark: fragment payload_ids short");
    }
    if (!frag.generations.empty() && frag.generations.size() < frag.row_count) {
        throw std::invalid_argument("encode_fragment_landmark: fragment generations short");
    }
    for (uint32_t r = 0; r < frag.row_count; ++r) {
        if (frag.row_indices[r] >= segment.n_rows) {
            throw std::out_of_range("encode_fragment_landmark: fragment row out of segment bounds");
        }
        if (r > 0 && frag.storage_positions[r] <= frag.storage_positions[r - 1]) {
            throw std::invalid_argument("encode_fragment_landmark: positions not strictly sorted");
        }
    }

    // 1. Decode ONLY the exact B^T feature slice [feature_offset, feature_offset + feature_dim)
    std::vector<uint64_t> b_feature_indices(feature_dim);
    for (uint32_t d = 0; d < feature_dim; ++d) {
        b_feature_indices[d] = feature_offset + d;
    }
    uint64_t padded_rank_k = group.b_k->desc.padded_shape.cols;
    std::vector<float> decoded_b_slice(feature_dim * padded_rank_k);
    decode_rows(*group.b_k, b_feature_indices.data(), feature_dim, decoded_b_slice.data(), decoded_b_slice.size(), value_domain::canonical);

    // Decode A_K rows for this fragment
    std::vector<uint64_t> a_row_indices(frag.row_indices.begin(), frag.row_indices.end());
    std::vector<float> decoded_a(frag.row_count * padded_rank_k);
    decode_rows(group.a_k, a_row_indices.data(), frag.row_count, decoded_a.data(), decoded_a.size(), value_domain::canonical);

    // 2. Reconstruct pre-RoPE K for each row, apply storage-position phase transform, and compute mean landmark
    std::vector<float> landmark_vec(feature_dim, 0.0f);
    std::vector<float> row_k_pre(feature_dim, 0.0f);
    std::vector<float> row_k_post(feature_dim, 0.0f);

    uint64_t source_fp = 0xcbf29ce484222325ULL;
    auto mix_fp = [&source_fp](float val) {
        uint32_t u;
        std::memcpy(&u, &val, sizeof(u));
        source_fp ^= u;
        source_fp *= 0x100000001b3ULL;
    };

    uint64_t logical_rank_k = group.a_k.desc.logical_shape.cols;

    for (size_t r = 0; r < frag.row_count; ++r) {
        const float * a_row = decoded_a.data() + r * padded_rank_k;
        for (uint32_t d = 0; d < feature_dim; ++d) {
            const float * b_feat = decoded_b_slice.data() + d * padded_rank_k;
            float sum = 0.0f;
            for (uint64_t k = 0; k < logical_rank_k; ++k) {
                sum += a_row[k] * b_feat[k];
            }
            row_k_pre[d] = sum;
        }

        // Apply storage-position phase transform (post-RoPE)
        if (phase_tx) {
            phase_tx(row_k_pre.data(), frag.storage_positions[r], frag.key.kv_head, row_k_post.data());
        } else {
            row_k_post = row_k_pre;
        }

        for (uint32_t d = 0; d < feature_dim; ++d) {
            landmark_vec[d] += row_k_post[d];
            mix_fp(row_k_post[d]);
        }
    }

    float inv_n = 1.0f / float(frag.row_count);
    for (uint32_t d = 0; d < feature_dim; ++d) {
        landmark_vec[d] *= inv_n;
    }

    frag.source_fingerprint = source_fp;
    frag.key.phase_tx_fingerprint = phase_tx_fingerprint;

    // 3. Encode landmark vector as independent code stream
    // Turbo4 logical feature dims are padded to block size (128) by make_codec_desc / encode_matrix
    codec_desc lm_desc = make_codec_desc(
        factor_role::landmark,
        landmark_type,
        orientation::token_major,
        {1, feature_dim},
        (landmark_type == GGML_TYPE_TURBO4_0) ? 128 : 0,
        landmark_detail::derive_landmark_seed(segment, group, feature_offset, feature_dim,
            landmark_type, phase_tx_fingerprint)
    );

    frag.landmark_matrix = encode_matrix(lm_desc, landmark_vec.data(), feature_dim);
    frag.key.landmark_codec_fp = frag.landmark_matrix.desc.fingerprint();

    // 4. Compute conservative error bound
    std::vector<float> decoded_lm = decode_matrix(frag.landmark_matrix, value_domain::canonical);
    double diff_sq_sum = 0.0;
    for (uint32_t d = 0; d < feature_dim; ++d) {
        double diff = decoded_lm[d] - landmark_vec[d];
        diff_sq_sum += diff * diff;
    }
    frag.error_bound = static_cast<float>(std::sqrt(diff_sq_sum) * 1.10);
    if (frag.error_bound < 1e-6f && landmark_type != GGML_TYPE_F32) {
        frag.error_bound = 1e-4f;
    }
}
bool attach_base_landmark(legal_fragment & frag, std::shared_ptr<const encoded_matrix> base,
    uint32_t base_row, const xkv_snapshot_stamp & stamp, uint64_t phase_tx_fingerprint,
    uint64_t chunk_source_fingerprint,
    const uint64_t * chunk_payload_ids, const uint64_t * chunk_generations,
    const int64_t * chunk_positions, size_t chunk_n, std::string * err) {
    auto fail = [&](const char * m) -> bool { if (err) *err = m; return false; };
    if (!base) return fail("attach: null base summary");
    if (frag.row_count == 0) return fail("attach: empty fragment");
    if (chunk_n != frag.row_count) return fail("attach: chunk/fragment row mismatch");
    if (!chunk_payload_ids || !chunk_generations || !chunk_positions) {
        return fail("attach: null chunk mapping");
    }
    const auto & k = frag.key;
    if (k.live_epoch != stamp.live_epoch || k.content_epoch != stamp.content_epoch ||
        k.codec_epoch != stamp.codec_epoch || k.binding_epoch != stamp.binding_epoch ||
        k.view_topology_epoch != stamp.view.topology_epoch ||
        k.view_publish_epoch != stamp.view.publish_epoch ||
        k.view_layout_epoch != stamp.view.layout_epoch) {
        return fail("attach: fragment epochs stale");
    }
    if (k.phase_tx_fingerprint != phase_tx_fingerprint) return fail("attach: phase mismatch");
    if (chunk_source_fingerprint == 0) return fail("attach: missing source fingerprint");
    if (frag.payload_ids.size() < frag.row_count ||
        frag.generations.size() < frag.row_count ||
        frag.storage_positions.size() < static_cast<size_t>(frag.row_count)) {
        return fail("attach: fragment row vectors short");
    }
    for (uint32_t r = 0; r < frag.row_count; ++r) {
        if (frag.payload_ids[r] != chunk_payload_ids[r] ||
            frag.generations[r] != chunk_generations[r] ||
            frag.storage_positions[r] != chunk_positions[r]) {
            return fail("attach: rows not intact (partial cannot attach)");
        }
    }
    const codec_desc & bdesc = base->desc;
    const codec_desc & odesc = frag.landmark_matrix.desc;
    if (bdesc.role != factor_role::landmark) return fail("attach: base is not a landmark stream");
    if (bdesc.type != odesc.type) return fail("attach: codec type mismatch");
    if (bdesc.logical_shape.cols != odesc.logical_shape.cols) {
        return fail("attach: feature width mismatch");
    }
    if (base_row >= bdesc.padded_shape.rows) return fail("attach: base row out of bounds");
    // Bitwise value proof: the base row must reproduce the fragment's own landmark row.
    std::vector<float> own;
    try {
        own = decode_matrix(frag.landmark_matrix, value_domain::canonical);
    } catch (const std::exception &) {
        return fail("attach: own landmark undecodable");
    }
    size_t pad = static_cast<size_t>(bdesc.padded_shape.cols);
    std::vector<float> brow(pad, 0.0f);
    try {
        uint64_t br = base_row;
        decode_rows(*base, &br, 1, brow.data(), brow.size(), value_domain::canonical);
    } catch (const std::exception &) {
        return fail("attach: base row undecodable");
    }
    if (own.size() < pad || brow.size() != pad ||
        std::memcmp(own.data(), brow.data(), pad * sizeof(float)) != 0) {
        return fail("attach: base value mismatch");
    }
    frag.base_landmark = std::move(base);
    frag.base_landmark_row = base_row;
    frag.landmark_matrix.bytes.clear();
    frag.landmark_matrix.bytes.shrink_to_fit();
    frag.source_fingerprint = chunk_source_fingerprint;
    frag.key.source_fingerprint = chunk_source_fingerprint;
    frag.key.landmark_codec_fp = frag.base_landmark->desc.fingerprint();
    return true;
}
uint64_t compute_base_table_fingerprint(const landmark_base_table & base) {
    uint64_t h = 0xcbf29ce484222325ULL;
    auto mix = [&h](uint64_t v) { h ^= v; h *= 0x100000001b3ULL; };
    mix((uint64_t) base.n_chunks);
    mix((uint64_t) base.n_rows_total);
    if (base.chunk_row_offsets) {
        for (size_t i = 0; i <= base.n_chunks; ++i) {
            mix(base.chunk_row_offsets[i]);
        }
    }
    if (base.chunk_error_bounds) {
        for (size_t c = 0; c < base.n_chunks; ++c) {
            uint32_t u = 0;
            float b = base.chunk_error_bounds[c];
            std::memcpy(&u, &b, sizeof(u));
            mix(u);
        }
    }
    if (base.chunk_source_fingerprints) {
        for (size_t c = 0; c < base.n_chunks; ++c) mix(base.chunk_source_fingerprints[c]);
    }
    if (base.row_payload_ids && base.row_generations && base.row_positions) {
        for (size_t r = 0; r < base.n_rows_total; ++r) {
            mix(base.row_payload_ids[r]);
            mix(base.row_generations[r]);
            mix((uint64_t) base.row_positions[r]);
        }
    }
    mix(base.stamp.live_epoch);
    mix(base.stamp.content_epoch);
    mix(base.stamp.codec_epoch);
    mix(base.stamp.binding_epoch);
    mix(base.stamp.view.topology_epoch);
    mix(base.stamp.view.publish_epoch);
    mix(base.stamp.view.layout_epoch);
    mix(base.phase_tx_fingerprint);
    if (base.landmark) mix(base.landmark->desc.fingerprint());
    return h;
}
bool bind_base_landmarks(legal_fragment * frags, size_t n_frags,
    const landmark_base_table & base, size_t * out_n_bound, std::string * err) {
    auto fail = [&](const char * m) -> bool { if (err) *err = m; return false; };
    if (out_n_bound) *out_n_bound = 0;
    if (!out_n_bound) return fail("bind: null output");
    if (n_frags > 0 && !frags) return fail("bind: null fragments");
    if (!base.landmark) return fail("bind: null base summary");
    if (!base.row_payload_ids || !base.row_generations || !base.row_positions ||
        !base.chunk_error_bounds || !base.chunk_source_fingerprints || !base.chunk_row_offsets) {
        return fail("bind: null base mapping");
}
    if (base.n_chunks == 0) return fail("bind: empty base");
    const codec_desc & bd = base.landmark->desc;
    std::string verr;
    if (!bd.validate(&verr)) return fail("bind: invalid base descriptor");
    if (bd.role != factor_role::landmark) return fail("bind: base is not a landmark stream");
    // Strictly advance: every chunk is nonempty, so chunks can never outnumber rows.
    if (base.n_chunks > base.n_rows_total) return fail("bind: more chunks than rows");
    // Offsets are uint32_t: row counts beyond 32 bits cannot be represented exactly.
    if (base.n_rows_total > (size_t) UINT32_MAX) return fail("bind: row count too large");
    if (bd.logical_shape.rows != (uint64_t) base.n_chunks ||
        bd.padded_shape.rows != (uint64_t) base.n_chunks) {
        return fail("bind: base rows must equal n_chunks");
    }
    if (base.chunk_row_offsets[0] != 0) return fail("bind: base offsets corrupt");
    for (size_t c = 0; c < base.n_chunks; ++c) {
        uint32_t lo = base.chunk_row_offsets[c];
        uint32_t hi = base.chunk_row_offsets[c + 1];
        if (hi <= lo) return fail("bind: empty chunk");
        if ((size_t) hi > base.n_rows_total) return fail("bind: base offsets corrupt");
    }
    if (base.chunk_row_offsets[base.n_chunks] != (uint32_t) base.n_rows_total) {
        return fail("bind: final offset mismatch");
    }
    for (size_t c = 0; c < base.n_chunks; ++c) {
        float b = base.chunk_error_bounds[c];
        if (!std::isfinite(b) || b < 0.0f) return fail("bind: invalid chunk bound");
        if (base.chunk_source_fingerprints[c] == 0) return fail("bind: missing chunk source fingerprint");
    }
    // Closure fingerprint: nonzero and exact, so no persisted field can be tampered
    // independently of the rest.
    if (base.bounds_fingerprint == 0) return fail("bind: missing closure fingerprint");
    if (compute_base_table_fingerprint(base) != base.bounds_fingerprint) {
        return fail("bind: closure fingerprint mismatch");
    }
    size_t n_bound = 0;
    for (size_t fi = 0; fi < n_frags; ++fi) {
        legal_fragment & frag = frags[fi];
        if (frag.row_count == 0) continue; // nothing to bind; not an error
        if (frag.uses_base_landmark()) continue; // already bound; keep
        if (!frag.landmark_matrix.bytes.empty()) continue; // owns bytes; seal path uses attach
        const auto & k = frag.key;
        if (k.live_epoch != base.stamp.live_epoch ||
            k.content_epoch != base.stamp.content_epoch ||
            k.codec_epoch != base.stamp.codec_epoch ||
            k.binding_epoch != base.stamp.binding_epoch ||
            k.view_topology_epoch != base.stamp.view.topology_epoch ||
            k.view_publish_epoch != base.stamp.view.publish_epoch ||
            k.view_layout_epoch != base.stamp.view.layout_epoch) {
            continue; // stale fragment: leave for scratch rebuild, not an error
}
        if (k.phase_tx_fingerprint != base.phase_tx_fingerprint) continue; // stale phase
        if (frag.payload_ids.size() < frag.row_count ||
            frag.storage_positions.size() < (size_t) frag.row_count) {
            continue; // incomplete identity: cannot prove intactness
}
        // Linear chunk scan for the exact intact row set (chunks are few; bounded).
        for (size_t c = 0; c < base.n_chunks; ++c) {
            uint32_t lo = base.chunk_row_offsets[c];
            uint32_t hi = base.chunk_row_offsets[c + 1];
            if ((uint64_t) (hi - lo) != frag.row_count) continue;
            bool intact = true;
            for (uint32_t r = 0; r < frag.row_count; ++r) {
                uint64_t gen = (r < frag.generations.size()) ? frag.generations[r]
                                                             : k.storage_generation;
                if (frag.payload_ids[r] != base.row_payload_ids[lo + r] ||
                    gen != base.row_generations[lo + r] ||
                    frag.storage_positions[r] != base.row_positions[lo + r]) {
                    intact = false;
                    break;
}
}
            if (!intact) continue;
            frag.base_landmark = base.landmark;
            frag.base_landmark_row = static_cast<uint32_t>(c);
            frag.error_bound = base.chunk_error_bounds[c]; // persisted seal bound
            frag.source_fingerprint = base.chunk_source_fingerprints[c];
            frag.key.source_fingerprint = base.chunk_source_fingerprints[c];
            frag.key.landmark_codec_fp = bd.fingerprint();
            ++n_bound;
            break;
}
}
    *out_n_bound = n_bound;
    return true;
}
// ------------------------------------------------------------------------------------------------
// SR Selection: single query
// ------------------------------------------------------------------------------------------------

sr_selection_result select_sr_query(
    const sr_query & query,
    const std::vector<legal_fragment> & legal_frags,
    const sr_selection_config & config,
    const xkv_segment * segment,
    uint32_t feature_offset,
    uint32_t feature_dim,
    const phase_transform_fn & phase_tx,
    uint64_t phase_tx_fingerprint
) {
    sr_selection_result result;
    if (legal_frags.empty() && config.hot_rows.empty() && config.recent_rows.empty() && config.outlier_rows.empty()) {
        return result;
    }

    uint32_t h_dim = (query.head_dim > 0) ? query.head_dim : feature_dim;
    if (h_dim == 0) {
        throw std::invalid_argument("select_sr_query: head_dim cannot be 0");
    }
    // The rebuild/refine path re-encodes with feature_dim but scores h_dim wide:
    // mixing widths would misread memory, so fail closed when both are set and differ.
    if (segment != nullptr && query.head_dim != 0 && feature_dim != 0 &&
        query.head_dim != feature_dim) {
        throw std::invalid_argument("select_sr_query: head_dim/feature_dim mismatch");
    }

    size_t n_q_heads = query.q_head_indices.empty() ? 1 : query.q_head_indices.size();
    if (query.q_vec.size() < n_q_heads * h_dim) {
        throw std::invalid_argument("select_sr_query: query vector size mismatch");
    }

    // Validate finiteness of query vector and scale
    for (float qv : query.q_vec) {
        if (!std::isfinite(qv)) {
            throw std::invalid_argument("select_sr_query: non-finite value in query vector");
        }
    }
    float scale = (query.scale > 0.0f) ? query.scale : (1.0f / std::sqrt(float(h_dim)));
    if (!std::isfinite(scale) || scale <= 0.0f) {
        throw std::invalid_argument("select_sr_query: non-finite or non-positive scale");
    }

    // 1. Enforce single-query causal cutoff: filter or rebuild partial causal fragments
    std::vector<legal_fragment> active_frags;
    active_frags.reserve(legal_frags.size());

    for (const auto & frag : legal_frags) {
        // Enforce query visibility: if query_visibility is specified, query must be explicitly eligible
        if (!frag.query_visibility.empty()) {
            if (query.query_index >= frag.query_visibility.size() || !frag.query_visibility[query.query_index]) {
                continue;
            }
        }
        if (query.causal_limit_pos >= 0) {
            if (frag.key.storage_pos0 > query.causal_limit_pos) {
                continue; // completely future
            }
            if (!frag.storage_positions.empty() && frag.storage_positions.back() > query.causal_limit_pos) {
                // Partial causal fragment: MUST rebuild landmark from ONLY legal rows
                if (!segment) {
                    throw std::invalid_argument("select_sr_query: partial causal chunk requires non-null segment to rebuild landmark");
                }
                legal_fragment sub_frag = frag;
                sub_frag.key.causal_cutoff = query.causal_limit_pos;
                sub_frag.row_indices.clear();
                sub_frag.storage_positions.clear();
                sub_frag.payload_ids.clear();
                sub_frag.generations.clear();
                sub_frag.base_landmark.reset(); // partial rows are never the intact chunk
                sub_frag.base_landmark_row = 0;
                for (size_t r = 0; r < frag.storage_positions.size(); ++r) {
                    if (frag.storage_positions[r] <= query.causal_limit_pos) {
                        sub_frag.row_indices.push_back(frag.row_indices[r]);
                        sub_frag.storage_positions.push_back(frag.storage_positions[r]);
                        sub_frag.payload_ids.push_back(frag.payload_ids[r]);
                        if (r < frag.generations.size()) sub_frag.generations.push_back(frag.generations[r]);
                    }
                }
                sub_frag.row_count = static_cast<uint32_t>(sub_frag.row_indices.size());
                if (sub_frag.row_count > 0) {
                    if (frag.key.phase_tx_fingerprint != phase_tx_fingerprint ||
                        frag.key.source_fingerprint != segment->source_fingerprint) {
                        throw std::invalid_argument("select_sr_query: stale phase/source for partial rebuild");
                    }
                    encode_fragment_landmark(
                        sub_frag,
                        *segment,
                        feature_offset,
                        feature_dim,
                        phase_tx,
                        phase_tx_fingerprint,
                        config.landmark_type
                    );
                    active_frags.push_back(std::move(sub_frag));
                }
                continue;
            }
        }
        active_frags.push_back(frag);
    }

    // Decode landmarks and validate decoded dimensions
    std::vector<std::vector<float>> decoded_landmarks(active_frags.size());
    for (size_t fi = 0; fi < active_frags.size(); ++fi) {
        if (active_frags[fi].row_count > 0) {
            // Intact chunks decode the shared stored-base row; derived frags own theirs.
            const legal_fragment & af = active_frags[fi];
            std::vector<float> lm;
            if (af.uses_base_landmark()) {
                const encoded_matrix * b = af.base_landmark.get();
                if (!b) throw std::invalid_argument("select_sr_query: null base landmark");
                uint64_t br = af.base_landmark_row;
                if (br >= b->desc.padded_shape.rows) {
                    throw std::out_of_range("select_sr_query: base row out of bounds");
                }
                lm.assign(static_cast<size_t>(b->desc.padded_shape.cols), 0.0f);
                decode_rows(*b, &br, 1, lm.data(), lm.size(), value_domain::canonical);
            } else {
                lm = decode_matrix(af.landmark_matrix, value_domain::canonical);
            }
            if (lm.size() < h_dim) {
                throw std::invalid_argument("select_sr_query: decoded landmark dimension smaller than head_dim");
            }
            for (uint32_t d = 0; d < h_dim; ++d) {
                if (!std::isfinite(lm[d])) {
                    throw std::invalid_argument("select_sr_query: non-finite decoded landmark");
                }
            }
            decoded_landmarks[fi] = std::move(lm);
        }
    }

    // Compute raw dot products for each Q head across all candidate fragments
    std::vector<std::vector<float>> raw_head_scores(n_q_heads, std::vector<float>(active_frags.size(), -std::numeric_limits<float>::infinity()));
    std::vector<float> q_head_norms(n_q_heads, 0.0f);

    for (size_t qh = 0; qh < n_q_heads; ++qh) {
        const float * q_ptr = query.q_vec.data() + qh * h_dim;
        double norm_sq = 0.0;
        for (uint32_t d = 0; d < h_dim; ++d) {
            norm_sq += q_ptr[d] * q_ptr[d];
        }
        q_head_norms[qh] = static_cast<float>(std::sqrt(norm_sq));

        for (size_t fi = 0; fi < active_frags.size(); ++fi) {
            if (active_frags[fi].row_count == 0) continue;
            const auto & lm = decoded_landmarks[fi];
            float dot = 0.0f;
            for (uint32_t d = 0; d < h_dim; ++d) {
                dot += q_ptr[d] * lm[d];
            }
            raw_head_scores[qh][fi] = dot * scale;
            if (!std::isfinite(raw_head_scores[qh][fi])) {
                throw std::invalid_argument("select_sr_query: non-finite raw landmark score");
            }
        }
    }

    // Per-head GQA normalized max: per-Q-head z-score normalization across all candidates before max pooling
    auto compute_normalized_scores = [&](const std::vector<std::vector<float>> & in_raw) {
        std::vector<float> std_devs(n_q_heads, 1.0f);
        std::vector<std::vector<float>> out_norm = in_raw;
        if (active_frags.size() > 1) {
            for (size_t qh = 0; qh < n_q_heads; ++qh) {
                double sum = 0.0;
                size_t valid_cnt = 0;
                for (size_t fi = 0; fi < active_frags.size(); ++fi) {
                    if (active_frags[fi].row_count > 0) {
                        sum += in_raw[qh][fi];
                        ++valid_cnt;
                    }
                }
                if (valid_cnt > 1) {
                    double mean = sum / double(valid_cnt);
                    double var_sum = 0.0;
                    for (size_t fi = 0; fi < active_frags.size(); ++fi) {
                        if (active_frags[fi].row_count > 0) {
                            double diff = in_raw[qh][fi] - mean;
                            var_sum += diff * diff;
                        }
                    }
                    double std_dev = std::sqrt(var_sum / double(valid_cnt));
                    if (std_dev < 1e-10) std_dev = 1e-10;
                    std_devs[qh] = static_cast<float>(std_dev);
                    for (size_t fi = 0; fi < active_frags.size(); ++fi) {
                        if (active_frags[fi].row_count > 0) {
                            out_norm[qh][fi] = static_cast<float>((in_raw[qh][fi] - mean) / std_dev);
                        }
                    }
                }
            }
        }
        return std::make_pair(out_norm, std_devs);
    };

    auto norm_pair = compute_normalized_scores(raw_head_scores);
    std::vector<std::vector<float>> norm_head_scores = norm_pair.first;
    std::vector<float> head_std_devs = norm_pair.second;

    struct candidate_frag {
        size_t frag_index = 0;
        float score = -std::numeric_limits<float>::infinity();
        float error_margin = 0.0f;
    };
    std::vector<candidate_frag> candidates;
    candidates.reserve(active_frags.size());

    for (size_t fi = 0; fi < active_frags.size(); ++fi) {
        if (active_frags[fi].row_count == 0) continue;
        float max_s = -std::numeric_limits<float>::infinity();
        float max_m = 0.0f;
        for (size_t qh = 0; qh < n_q_heads; ++qh) {
            float s = norm_head_scores[qh][fi];
            if (s > max_s) {
                max_s = s;
                // Conservative error margin in z-score normalized space:
                // raw_error <= error_bound * q_norm * scale, so normalized_error <= raw_error / std_dev
                max_m = (active_frags[fi].error_bound * q_head_norms[qh] * scale) / head_std_devs[qh];
            }
        }
        candidates.push_back({fi, max_s, max_m});
    }

    // Deterministic tie-break sorting
    std::sort(candidates.begin(), candidates.end(), [](const candidate_frag & a, const candidate_frag & b) {
        if (a.score != b.score) {
            return a.score > b.score;
        }
        return a.frag_index < b.frag_index;
    });

    uint32_t budget = config.sr_budget;
    if (budget == 0 || budget > candidates.size()) {
        budget = static_cast<uint32_t>(candidates.size());
    }

    // Boundary refinement under normalized ranking
    std::unordered_set<size_t> selected_frag_indices;
    for (size_t i = 0; i < budget; ++i) {
        selected_frag_indices.insert(candidates[i].frag_index);
    }

    if (config.refine_mode == LLAMA_XKV_LANDMARK_REFINE_BOUNDARY && segment != nullptr && budget < candidates.size()) {
        // Identify ONLY candidate fragments whose uncertainty interval [score - error_margin, score + error_margin]
        // overlaps with the top-k boundary cutoff interval [cutoff_score - cutoff_margin, cutoff_score + cutoff_margin]
        float cutoff_score = candidates[budget - 1].score;
        float cutoff_margin = candidates[budget - 1].error_margin;

        std::vector<size_t> boundary_candidate_indices;
        // Overflow-safe accumulation: saturate instead of wrapping so the cap
        // comparison can never be bypassed by row-count overflow.
        size_t boundary_rows = 0;
        for (size_t i = 0; i < candidates.size(); ++i) {
            float lo = candidates[i].score - candidates[i].error_margin;
            float hi = candidates[i].score + candidates[i].error_margin;
            if (hi >= cutoff_score - cutoff_margin && lo <= cutoff_score + cutoff_margin) {
                boundary_candidate_indices.push_back(i);
                size_t add = active_frags[candidates[i].frag_index].row_count;
                if (SIZE_MAX - boundary_rows < add) boundary_rows = SIZE_MAX;
                else boundary_rows += add;
            }
        }

        if (boundary_rows > config.refine_max_rows) {
            // Insufficient refinement budget: set refine_cap_hit and preserve deterministic approximate ordering
            result.refine_cap_hit = true;
        } else if (!boundary_candidate_indices.empty()) {
            // Refine candidates: rebuild exact landmarks, recompute raw head scores, z-score normalize, and re-pool
            std::vector<std::vector<float>> refined_raw_scores = raw_head_scores;
            uint32_t refined_row_count = 0;
            bool refined_overflow = false;

            uint32_t fault_n = 0; // §15.4 injection counter (rebuild events this selection)
            for (size_t c_idx : boundary_candidate_indices) {
                const auto & frag = active_frags[candidates[c_idx].frag_index];
                legal_fragment exact_frag = frag;
                exact_frag.base_landmark.reset(); // F32-exact rebuild owns its bytes
                exact_frag.base_landmark_row = 0;
                // Injected failure lands before any encode/mutation: caller result, table,
                // and cache are untouched (throw precedes emission and any commit).
                if (config.fault.stage == landmark_fault_stage::refine_rescore &&
                    fault_n++ == config.fault.occurrence) {
                    throw std::runtime_error("injected refine_rescore fault");
                }
                if (frag.key.phase_tx_fingerprint != phase_tx_fingerprint ||
                    frag.key.source_fingerprint != segment->source_fingerprint) {
                    throw std::invalid_argument("select_sr_query: stale phase/source for refine");
                }
                encode_fragment_landmark(
                    exact_frag,
                    *segment,
                    feature_offset,
                    feature_dim,
                    phase_tx,
                    phase_tx_fingerprint,
                    GGML_TYPE_F32
                );
                std::vector<float> exact_lm = decode_matrix(exact_frag.landmark_matrix, value_domain::canonical);

                for (size_t qh = 0; qh < n_q_heads; ++qh) {
                    const float * q_ptr = query.q_vec.data() + qh * h_dim;
                    float dot = 0.0f;
                    for (uint32_t d = 0; d < h_dim; ++d) {
                        dot += q_ptr[d] * exact_lm[d];
                    }
                    refined_raw_scores[qh][candidates[c_idx].frag_index] = dot * scale;
                }
                if (UINT32_MAX - refined_row_count < frag.row_count) refined_overflow = true;
                else refined_row_count += frag.row_count;
            }
            result.rows_refined = refined_overflow ? UINT32_MAX : refined_row_count;


            // Recompute z-score normalized scores consistently across the full candidate set
            auto refined_pair = compute_normalized_scores(refined_raw_scores);
            std::vector<std::vector<float>> refined_norm_scores = refined_pair.first;

            for (auto & cand : candidates) {
                float max_s = -std::numeric_limits<float>::infinity();
                for (size_t qh = 0; qh < n_q_heads; ++qh) {
                    float s = refined_norm_scores[qh][cand.frag_index];
                    if (s > max_s) max_s = s;
                }
                cand.score = max_s;
            }

            std::sort(candidates.begin(), candidates.end(), [](const candidate_frag & a, const candidate_frag & b) {
                if (a.score != b.score) {
                    return a.score > b.score;
                }
                return a.frag_index < b.frag_index;
            });

            selected_frag_indices.clear();
            for (size_t i = 0; i < budget; ++i) {
                selected_frag_indices.insert(candidates[i].frag_index);
            }
        }
    }

    // Collect all rows as 4-field segment_row_refs
    std::unordered_set<segment_row_ref, segment_row_ref_hash> row_set;
    // Zero-legal-fragment hard-keep path: no active rows exist to prove keeps against,
    // so explicit keeps union directly after exact segment-backed legality. With a causal
    // cutoff (or no segment) compliance is unprovable -> fail closed, never silently drop.
    if (candidates.empty() && (!config.hot_rows.empty() || !config.recent_rows.empty() ||
        !config.outlier_rows.empty())) {
        if (query.causal_limit_pos >= 0) {
            throw std::invalid_argument("select_sr_query: keep legality unprovable under causal cutoff");
        }
        if (!segment) {
            throw std::invalid_argument("select_sr_query: no segment to establish keep legality");
        }
        const std::vector<segment_row_ref> * prot[3] = {&config.hot_rows, &config.recent_rows,
            &config.outlier_rows};
        for (int pi = 0; pi < 3; ++pi) {
            for (size_t k = 0; k < prot[pi]->size(); ++k) {
                const segment_row_ref & ref = (*prot[pi])[k];
                if (ref.segment_id != segment->segment_id ||
                    ref.segment_version != segment->segment_version) {
                    throw std::invalid_argument("select_sr_query: keep owner mismatch");
                }
                if (ref.row >= segment->n_rows) {
                    throw std::out_of_range("select_sr_query: keep row out of range");
                }
                if (ref.row < segment->live_rows.size() && !segment->live_rows[ref.row]) {
                    throw std::invalid_argument("select_sr_query: keep row not live");
                }
                row_set.insert(ref);
            }
        }
        result.selected_rows.assign(row_set.begin(), row_set.end());
        std::sort(result.selected_rows.begin(), result.selected_rows.end());
        return result;
    }
    for (size_t fi : selected_frag_indices) {
        const auto & frag = active_frags[fi];
        for (size_t r = 0; r < frag.row_indices.size(); ++r) {
            uint64_t gen = (r < frag.generations.size()) ? frag.generations[r] : frag.key.storage_generation;
            row_set.insert({frag.key.segment_id, frag.key.segment_version, gen, frag.row_indices[r]});
        }
    }

    for (size_t i = 0; i < budget; ++i) {
        result.scores.push_back(candidates[i].score);
    }

    // Union + dedup with mandatory protection sets (hot, recent, outlier)
    // Mandatory sets cannot inject future rows beyond causal cutoff
    auto add_protected = [&](const std::vector<segment_row_ref> & protected_refs) {
        for (const auto & ref : protected_refs) {
            // Positive verification: row reference must be positively found within active_frags
            // (which have already been filtered and verified as legal for this reader and within causal cutoff).
            // Absent means reject/skip; never default legal.
            bool positively_legal = false;
            for (const auto & f : active_frags) {
                if (f.key.segment_id == ref.segment_id && f.key.segment_version == ref.segment_version) {
                    for (size_t r = 0; r < f.row_indices.size(); ++r) {
                        uint64_t gen = (r < f.generations.size()) ? f.generations[r] : f.key.storage_generation;
                        if (f.row_indices[r] == ref.row && gen == ref.storage_generation) {
                            if (query.causal_limit_pos < 0 || f.storage_positions[r] <= query.causal_limit_pos) {
                                positively_legal = true;
                                break;
                            }
                        }
                    }
                }
                if (positively_legal) break;
            }
            if (positively_legal) {
                row_set.insert(ref);
            }
        }
    };

    add_protected(config.hot_rows);
    add_protected(config.recent_rows);
    add_protected(config.outlier_rows);

    result.selected_rows.assign(row_set.begin(), row_set.end());
    std::sort(result.selected_rows.begin(), result.selected_rows.end());

    return result;
}

// ------------------------------------------------------------------------------------------------
// SR Selection: batch across queries
// ------------------------------------------------------------------------------------------------

sr_batch_selection_result select_sr_batch(
    const std::vector<sr_query> & queries,
    const std::vector<legal_fragment> & legal_frags,
    const sr_selection_config & config,
    const xkv_segment * segment,
    uint32_t feature_offset,
    uint32_t feature_dim,
    const phase_transform_fn & phase_tx,
    uint64_t phase_tx_fingerprint
) {
    sr_batch_selection_result batch_res;
    batch_res.per_query.resize(queries.size());
    batch_res.csr_ptrs.resize(queries.size() + 1, 0);

    if (queries.empty()) {
        return batch_res;
    }

    std::unordered_set<segment_row_ref, segment_row_ref_hash> global_row_union;

    for (size_t qi = 0; qi < queries.size(); ++qi) {
        const auto & q = queries[qi];

        sr_selection_result q_res = select_sr_query(
            q,
            legal_frags,
            config,
            segment,
            feature_offset,
            feature_dim,
            phase_tx,
            phase_tx_fingerprint
        );

        if (UINT32_MAX - batch_res.total_refined_rows < q_res.rows_refined) {
            batch_res.total_refined_rows = UINT32_MAX; // saturate, never wrap
        } else {
            batch_res.total_refined_rows += q_res.rows_refined;
        }
        if (q_res.refine_cap_hit && batch_res.total_refine_cap_hits < UINT32_MAX) {
            batch_res.total_refine_cap_hits += 1;
        }

        for (const auto & r : q_res.selected_rows) {
            global_row_union.insert(r);
        }

        batch_res.per_query[qi] = std::move(q_res);
    }

    // Build unique sorted gather_rows of segment_row_refs
    batch_res.gather_rows.assign(global_row_union.begin(), global_row_union.end());
    std::sort(batch_res.gather_rows.begin(), batch_res.gather_rows.end());

    // Map segment_row_ref -> index in gather_rows
    std::unordered_map<segment_row_ref, uint32_t, segment_row_ref_hash> gather_map;
    for (uint32_t i = 0; i < batch_res.gather_rows.size(); ++i) {
        gather_map[batch_res.gather_rows[i]] = i;
    }

    // Build CSR representation: csr_ptrs and csr_indices
    uint32_t total_indices = 0;
    for (size_t qi = 0; qi < queries.size(); ++qi) {
        batch_res.csr_ptrs[qi] = total_indices;
        for (const auto & r : batch_res.per_query[qi].selected_rows) {
            auto it = gather_map.find(r);
            if (it != gather_map.end()) {
                batch_res.csr_indices.push_back(it->second);
                // Reference path is group-agnostic: invalid sentinel. Production group
                // tracking lives in select_sr_batch_bounded (fragment views carry it).
                batch_res.csr_group_indices.push_back(UINT32_MAX);
                ++total_indices;
            }
        }
    }
    batch_res.csr_ptrs[queries.size()] = total_indices;

    return batch_res;
}
bool landmark_encode_scratch_bytes(uint32_t feature_dim, uint32_t padded_rank, uint32_t row_count,
    size_t & out_bytes, std::string * err) {
    using namespace landmark_detail;
    out_bytes = 0;
    if (feature_dim == 0 || padded_rank == 0 || row_count == 0) {
        if (err) *err = "landmark_encode_scratch_bytes: invalid dimensions";
        return false;
    }
    size_t b_slice = 0, a_rows = 0, three_d = 0, tmp = 0;
    if (!ck_mul_u32(feature_dim, padded_rank, b_slice) ||
        !ck_mul_u32(row_count, padded_rank, a_rows) ||
        !ck_mul(sizeof(float), static_cast<size_t>(feature_dim), tmp) ||
        !ck_mul(tmp, 4, three_d)) {
        if (err) *err = "landmark_encode_scratch_bytes: overflow";
        return false;
    }
    size_t floats = 0;
    if (!ck_add(b_slice, a_rows, floats) || !ck_add(floats, three_d, floats)) {
        if (err) *err = "landmark_encode_scratch_bytes: overflow";
        return false;
    }
    if (!ck_mul(floats, sizeof(float), out_bytes)) {
        if (err) *err = "landmark_encode_scratch_bytes: overflow";
        return false;
    }
    return true;
}
namespace landmark_detail {
// Exact single-core scratch need (no slack hidden): sections + 160 bytes alignment slack.
inline bool core_scratch_bytes(uint32_t n_frags, uint32_t n_heads, uint32_t head_dim,
    uint32_t max_rows_per_frag, uint32_t padded_rank, uint32_t lm_pad, size_t & out);
} // namespace landmark_detail
bool landmark_select_scratch_bytes(uint32_t n_frags, uint32_t n_q_heads, uint32_t head_dim,
    uint32_t max_rows_per_fragment, uint32_t padded_rank, size_t & out_bytes,
    std::string * err) {
    out_bytes = 0;
    if (n_q_heads == 0 || head_dim == 0) {
        if (err) *err = "landmark_select_scratch_bytes: invalid heads/dim";
        return false;
    }
    size_t pad = 0;
    if (!landmark_detail::ck_add(static_cast<size_t>(head_dim), size_t(127), pad)) {
        if (err) *err = "landmark_select_scratch_bytes: overflow";
        return false;
    }
    if (pad > UINT32_MAX) {
        if (err) *err = "landmark_select_scratch_bytes: dim too large";
        return false;
    }
    if (!landmark_detail::core_scratch_bytes(n_frags, n_q_heads, head_dim, max_rows_per_fragment,
        padded_rank, static_cast<uint32_t>(pad), out_bytes)) {
        if (err) *err = "landmark_select_scratch_bytes: overflow";
        return false;
    }
    return true;
}
bool landmark_workspace_required_bytes(const landmark_workspace_config & cfg, size_t & out_bytes,
    std::string * err) {
    out_bytes = 0;
    if (cfg.max_fragments == 0 || cfg.max_q_heads_per_query == 0 || cfg.max_head_dim == 0 ||
        cfg.max_rows_per_fragment == 0) {
        if (err) *err = "landmark_workspace_required_bytes: core maxima must be nonzero";
        return false;
    }
    size_t core = 0;
    if (!landmark_select_scratch_bytes(cfg.max_fragments, cfg.max_q_heads_per_query,
        cfg.max_head_dim, cfg.max_rows_per_fragment, cfg.max_padded_rank, core, err)) {
        return false;
    }
    // Batch staging: worst-case single-query row output + per-query counts.
    size_t stage_rows = 0, stage_bytes = 0, counts_bytes = 0;
    if (!landmark_detail::ck_add(static_cast<size_t>(cfg.max_rows_total),
        static_cast<size_t>(cfg.max_protected_rows), stage_rows) ||
        !landmark_detail::ck_mul(stage_rows, sizeof(segment_row_ref), stage_bytes) ||
        !landmark_detail::ck_mul(static_cast<size_t>(cfg.max_queries), sizeof(uint32_t),
            counts_bytes)) {
        if (err) *err = "landmark_workspace_required_bytes: overflow";
        return false;
    }
    if (!landmark_detail::ck_add(core, stage_bytes, core) ||
        !landmark_detail::ck_add(core, counts_bytes, core)) {
        if (err) *err = "landmark_workspace_required_bytes: overflow";
        return false;
    }
    // Batch staging attribution (frag src-index + effective key per staged row) and
    // per-query score temps sized for max_fragments.
    size_t st_idx = 0, st_key = 0, score_tmp = 0;
    if (!landmark_detail::ck_mul(stage_rows, sizeof(uint32_t), st_idx) ||
        !landmark_detail::ck_mul(stage_rows, sizeof(landmark_fragment_key), st_key) ||
        !landmark_detail::ck_mul(static_cast<size_t>(cfg.max_fragments), sizeof(float),
            score_tmp)) {
        if (err) *err = "landmark_workspace_required_bytes: overflow";
        return false;
    }
    // Pass-2 staging groups (one effective group u32 per staged row) for CSR merge-back.
    size_t st_groups = 0;
    if (!landmark_detail::ck_mul(stage_rows, sizeof(uint32_t), st_groups)) {
        if (err) *err = "landmark_workspace_required_bytes: overflow";
        return false;
    }
    if (!landmark_detail::ck_add(core, st_idx, core) ||
        !landmark_detail::ck_add(core, st_key, core) ||
        !landmark_detail::ck_add(core, score_tmp, core) ||
        !landmark_detail::ck_add(core, st_groups, core) ||
        !landmark_detail::ck_add(core, size_t(64), core)) {
        if (err) *err = "landmark_workspace_required_bytes: overflow";
        return false;
    }
    out_bytes = core;
    return true;
}
bool landmark_workspace::init(const landmark_workspace_config & cfg, std::string * err) {
    size_t need = 0;
    if (!landmark_workspace_required_bytes(cfg, need, err)) return false;
    size_t raw = 0;
    if (!landmark_detail::ck_add(need, size_t(63), raw)) {
        if (err) *err = "landmark_workspace::init: overflow";
        return false;
    }
    std::vector<uint8_t> buf;
    buf.resize(raw, 0);
    uintptr_t base = reinterpret_cast<uintptr_t>(buf.data());
    size_t off = (64 - (base % 64)) % 64;
    buffer_.swap(buf);
    cfg_ = cfg;
    align_off_ = off;
    return true;
}
void landmark_workspace::clear() {
    buffer_.clear();
    align_off_ = 0;
    cfg_ = landmark_workspace_config();
}
namespace landmark_detail {
struct scratch_carve {
    uint8_t * base = nullptr;
    size_t total = 0;
    size_t off = 0;
    bool fail = false;
    void * take(size_t bytes, size_t align) {
        if (fail) return nullptr;
        size_t aligned = off;
        size_t rem = aligned % align;
        if (rem != 0) {
            size_t add = align - rem;
            if (!ck_add(aligned, add, aligned)) { fail = true; return nullptr; }
        }
        size_t next = 0;
        if (!ck_add(aligned, bytes, next) || next > total) { fail = true; return nullptr; }
        off = next;
        return base + aligned;
    }
};
struct bounded_active {
    uint32_t src_index = 0;
    uint32_t row_count = 0;
    const uint32_t * row_indices = nullptr;
    const int64_t * storage_positions = nullptr;
    const uint64_t * generations = nullptr;
    const landmark_fragment_key * key = nullptr;
    float error_bound = 0.0f;
    uint64_t seg_id = 0;
    uint64_t seg_ver = 0;
    int64_t eff_cutoff = -2;
    uint32_t group_index = UINT32_MAX; // effective group carried verbatim per row
};
inline bool rebuild_exact_mean(const xkv_segment & segment, const landmark_fragment_view & fv,
    const uint32_t * rows, const int64_t * positions, uint32_t n_rows, uint32_t feature_offset,
    uint32_t feature_dim, const phase_transform_fn & phase_tx, float * tmp_b, float * tmp_a,
    float * tmp_pre, float * tmp_post, float * out_mean, uint8_t * dectmp, size_t dectmp_bytes,
    std::string * err);
inline const xkv_segment * resolve_owner_segment(const xkv_segment * single,
    const xkv_segment * const * table, size_t n_table, uint64_t seg_id, uint64_t seg_ver);
inline void sort_rows_attributed(segment_row_ref * rows, uint32_t * fidx,
    landmark_fragment_key * fkey, size_t n, uint32_t * gidx = nullptr);
inline size_t unique_rows_attributed(segment_row_ref * rows, uint32_t * fidx,
    landmark_fragment_key * fkey, size_t n, uint32_t * gidx = nullptr);
// Effective landmark source for a view: shared stored-base row for attached intact
// chunks, own matrix otherwise. Never copies bytes.
inline bool view_lm_source(const landmark_fragment_view & fv, const encoded_matrix *& m,
    uint64_t & row, std::string * err) {
    if (fv.base_landmark) {
        if (fv.base_landmark_row >= fv.base_landmark->desc.padded_shape.rows) {
            if (err) *err = "bounded select: base row out of bounds";
            return false;
        }
        m = fv.base_landmark;
        row = fv.base_landmark_row;
        return true;
    }
    if (!fv.landmark) {
        if (err) *err = "bounded select: null fragment landmark";
        return false;
    }
    m = fv.landmark;
    row = 0;
    return true;
}
// Zero-heap single-row decode into caller memory. The codec's decode_rows keeps a
// per-row heap scratch for F16/Q8_0 alignment; this entry instead memcpys into a
// 64-aligned caller scratch region and calls the same to_float / turbo dequantize,
// yielding bit-identical floats with no allocation. Fails closed on any mismatch.
inline bool decode_row_bytes(const codec_desc & desc, const uint8_t * bytes, size_t len,
    uint64_t row, float * dst, size_t dst_cap, uint8_t * tmp, size_t tmp_bytes,
    value_domain domain, std::string * err) {
    auto fail = [&](const char * m) -> bool { if (err) *err = m; return false; };
    std::string verr;
    if (!desc.validate(&verr)) return fail("bounded decode: invalid descriptor");
    if (!bytes && len > 0) return fail("bounded decode: null bytes");
    uint64_t pad_cols = desc.padded_shape.cols;
    uint64_t total_rows = desc.padded_shape.rows;
    if (pad_cols == 0 || total_rows == 0) return fail("bounded decode: empty shape");
    if (row >= total_rows) return fail("bounded decode: row out of bounds");
    if (dst_cap < static_cast<size_t>(pad_cols) || !dst) return fail("bounded decode: dst short");
    if (row > 0 && desc.row_stride_bytes > 0 && row > UINT64_MAX / desc.row_stride_bytes) {
        return fail("bounded decode: row offset overflow");
    }
    uint64_t off = row * desc.row_stride_bytes;
    size_t row_bytes = ggml_row_size(desc.type, static_cast<int64_t>(pad_cols));
    if (off + row_bytes > len) return fail("bounded decode: row out of buffer");
    if (!bytes && row_bytes > 0) return fail("bounded decode: null bytes");
    const uint8_t * src = bytes + off;
    switch (desc.type) {
        case GGML_TYPE_F32: {
            std::memcpy(dst, src, static_cast<size_t>(pad_cols) * sizeof(float));
            return true;
        }
        case GGML_TYPE_F16:
        case GGML_TYPE_Q8_0: {
            if (domain != value_domain::canonical) {
                return fail("bounded decode: non-canonical domain unsupported");
            }
            const struct ggml_type_traits * traits = ggml_get_type_traits(desc.type);
            if (!traits || !traits->to_float) return fail("bounded decode: missing to_float");
            if (!tmp || tmp_bytes < row_bytes) return fail("bounded decode: tmp short");
            std::memcpy(tmp, src, row_bytes);
            traits->to_float(tmp, dst, static_cast<int64_t>(pad_cols));
            return true;
        }
        case GGML_TYPE_TURBO2_0:
        case GGML_TYPE_TURBO3_0:
        case GGML_TYPE_TURBO4_0: {
            enum ggml_turbo_decode_domain dom = (domain == value_domain::canonical)
                ? GGML_TURBO_DECODE_CANONICAL : GGML_TURBO_DECODE_ROTATED;
            if (!ggml_dequantize_turbo_row(desc.type, src, dst,
                static_cast<int64_t>(pad_cols), 128, dom)) {
                return fail("bounded decode: turbo dequantize failed");
            }
            return true;
        }
        default:
            return fail("bounded decode: unsupported type");
    }
}
// Thin encoded_matrix wrapper preserving the historical entry point and its exact
// byte-size-match contract; bit-identical output to decode_row_bytes.
inline bool decode_row_bounded(const encoded_matrix & em, uint64_t row, float * dst,
    size_t dst_cap, uint8_t * tmp, size_t tmp_bytes, value_domain domain,
    std::string * err) {
    uint64_t expected = 0;
    try {
        expected = encoded_matrix_bytes(em.desc);
    } catch (...) {
        if (err) *err = "bounded decode: byte size overflow";
        return false;
    }
    if (em.bytes.size() != static_cast<size_t>(expected)) {
        if (err) *err = "bounded decode: byte size mismatch";
        return false;
    }
    return decode_row_bytes(em.desc, em.bytes.data(), em.bytes.size(), row, dst, dst_cap,
        tmp, tmp_bytes, domain, err);
}
// Exact single-core scratch need (no slack hidden): sections + 160 bytes alignment slack.
inline bool core_scratch_bytes(uint32_t n_frags, uint32_t n_heads, uint32_t head_dim,
    uint32_t max_rows_per_frag, uint32_t padded_rank, uint32_t lm_pad, size_t & out) {
    out = 0;
    size_t hn = 0, b = 0;
    if (!ck_mul_u32(n_heads, n_frags, hn)) return false;
    size_t two_hn = 0, two_h = 0;
    if (!ck_mul(hn, 2, two_hn) || !ck_mul(static_cast<size_t>(n_heads), 2, two_h)) return false;
    size_t n_score_floats = 0;
    if (!ck_add(two_hn, two_hn, n_score_floats)) return false;
    if (!ck_add(n_score_floats, two_h, n_score_floats)) return false;
    size_t score_bytes = 0;
    if (!ck_mul(n_score_floats, sizeof(float), score_bytes)) return false;
    size_t cand_bytes = 0, act_bytes = 0, lm_bytes = 0, prows_bytes = 0, rebuild_floats = 0;
    if (!ck_mul(static_cast<size_t>(n_frags), sizeof(scored_candidate), cand_bytes)) return false;
    if (!ck_mul(static_cast<size_t>(n_frags), sizeof(bounded_active), act_bytes)) return false;
    if (!ck_mul(static_cast<size_t>(lm_pad), sizeof(float), lm_bytes)) return false;
    size_t stage = 0, s4 = 0, s8 = 0;
    if (!ck_mul(static_cast<size_t>(n_frags), static_cast<size_t>(max_rows_per_frag), stage)) {
        return false;
    }
    if (!ck_mul(stage, size_t(4), s4) || !ck_add(s4, size_t(7), s4)) return false;
    s4 &= ~size_t(7);
    if (!ck_mul(stage, size_t(8), s8)) return false;
    prows_bytes = 0;
    if (!ck_add(s4, s8, prows_bytes) || !ck_add(prows_bytes, s8, prows_bytes) ||
        !ck_add(prows_bytes, s8, prows_bytes)) {
        return false;
    }
    size_t dp = 0;
    if (!ck_mul_u32(head_dim, padded_rank, dp)) return false;
    size_t three_d = 0;
    if (!ck_mul(static_cast<size_t>(head_dim), size_t(3), three_d)) return false;
    if (!ck_add(dp, static_cast<size_t>(padded_rank), rebuild_floats) ||
        !ck_add(rebuild_floats, three_d, rebuild_floats)) return false;
    size_t rebuild_bytes = 0;
    if (!ck_mul(rebuild_floats, sizeof(float), rebuild_bytes)) return false;
    // 64-aligned decode tmp for F16/Q8 rows: 2 * (lm_pad + padded_rank) bytes.
    size_t tmp_sum = 0, tmp_bytes = 0;
    if (!ck_add(static_cast<size_t>(lm_pad), static_cast<size_t>(padded_rank), tmp_sum) ||
        !ck_mul(tmp_sum, size_t(2), tmp_bytes)) return false;
    if (!ck_add(score_bytes, cand_bytes, b) || !ck_add(b, act_bytes, b) ||
        !ck_add(b, static_cast<size_t>(n_frags), b) || // flags
        !ck_add(b, lm_bytes, b) || !ck_add(b, prows_bytes, b) ||
        !ck_add(b, rebuild_bytes, b) || !ck_add(b, tmp_bytes, b) ||
        !ck_add(b, size_t(160), b)) return false;
    out = b;
    return true;
}
// Zero-heap single-query core shared by the single and batch bounded entry points.
// out_rows/out_scores are caller memory (production arrays or batch staging); scratch is
// the fixed workspace carve. Returns false + err on ANY shortfall, never partial.
inline bool select_query_core(const sr_query & query, const landmark_fragment_view * frags,
    size_t n_frags, const sr_selection_config & config, const xkv_segment * segment,
    uint32_t feature_offset, uint32_t feature_dim, const phase_transform_fn & phase_tx,
    uint64_t phase_fp, uint8_t * scratch, size_t scratch_bytes, segment_row_ref * out_rows,
    size_t rows_cap, size_t * out_n_rows, float * out_scores, size_t scores_cap,
    size_t * out_n_scores, uint32_t * out_rows_refined, bool * out_cap_hit,
    std::string * err, const xkv_segment * const * segments, size_t n_segments,
    sr_bounded_attribution * attrib, uint32_t * out_row_groups, size_t row_groups_cap) {
    if (out_n_rows) *out_n_rows = 0;
    if (out_n_scores) *out_n_scores = 0;
    if (out_rows_refined) *out_rows_refined = 0;
    if (out_cap_hit) *out_cap_hit = false;
    if (attrib) {
        if (attrib->out_n_segs) *attrib->out_n_segs = 0;
        if (attrib->out_n_seg_idx) *attrib->out_n_seg_idx = 0;
    }
    auto fail = [&](const char * m) -> bool { if (err) *err = m; return false; };
    if (!out_rows || !out_scores || !out_n_rows || !out_n_scores || !out_rows_refined ||
        !out_cap_hit) {
        return fail("bounded select: null output");
    }
    if (n_frags == 0 && config.hot_rows.empty() && config.recent_rows.empty() &&
        config.outlier_rows.empty()) {
        return true;
    }
    if (n_frags > UINT32_MAX) return fail("bounded select: too many fragments");
    uint32_t h_dim = query.head_dim ? query.head_dim : feature_dim;
    if (h_dim == 0) return fail("select_sr_query: head_dim cannot be 0");
    size_t n_heads = query.q_head_indices.empty() ? 1 : query.q_head_indices.size();
    if (n_heads > UINT32_MAX) return fail("bounded select: too many q heads");
    size_t need_q = 0;
    if (!ck_mul(n_heads, static_cast<size_t>(h_dim), need_q) || query.q_vec.size() < need_q) {
        return fail("select_sr_query: query vector size mismatch");
    }
    for (size_t vi = 0; vi < need_q; ++vi) {
        if (!std::isfinite(query.q_vec[vi])) return fail("select_sr_query: non-finite query");
    }
    float scale = query.scale > 0.0f ? query.scale : (1.0f / std::sqrt(static_cast<float>(h_dim)));
    if (!std::isfinite(scale) || scale <= 0.0f) {
        return fail("select_sr_query: non-finite or non-positive scale");
    }
    if (segment && feature_dim != 0 && query.head_dim != 0 && query.head_dim != feature_dim) {
        return fail("bounded select: head_dim/feature_dim mismatch for rebuild path");
    }
    if ((segment || n_segments > 0) && feature_dim != 0 && query.head_dim != 0 &&
        query.head_dim != feature_dim) {
        return fail("bounded select: head_dim/feature_dim mismatch for rebuild path");
    }
    uint32_t N = static_cast<uint32_t>(n_frags);
    uint32_t H = static_cast<uint32_t>(n_heads);
    uint32_t max_pad = 0, rmax = 0;
    for (uint32_t i = 0; i < N; ++i) {
        if (!frags[i].key) return fail("bounded select: null fragment");
        const encoded_matrix * lmsrc = nullptr;
        uint64_t lrow = 0;
        if (!view_lm_source(frags[i], lmsrc, lrow, err)) return false;
        if (frags[i].row_count > rmax) rmax = frags[i].row_count;
        uint64_t pc = lmsrc->desc.padded_shape.cols;
        if (pc == 0 || pc > UINT32_MAX) return fail("bounded select: invalid landmark width");
        if (static_cast<uint32_t>(pc) > max_pad) max_pad = static_cast<uint32_t>(pc);
    }
    uint32_t P = 0;
    if (segment || n_segments > 0) {
        for (uint32_t i = 0; i < N; ++i) {
            const xkv_segment * owner = resolve_owner_segment(segment, segments, n_segments,
                frags[i].key->segment_id, frags[i].key->segment_version);
            if (!owner) continue; // unresolvable here: hard fail only if rebuild is needed
            const auto * g = owner->find_group_for_layer(frags[i].key->owning_layer);
            if (!g || !g->b_k) continue;
            uint64_t pr = g->b_k->desc.padded_shape.cols;
            if (pr > 0 && pr <= UINT32_MAX && static_cast<uint32_t>(pr) > P) {
                P = static_cast<uint32_t>(pr);
            }
        }
    }
    size_t need = 0;
    if (!core_scratch_bytes(N, H, h_dim, rmax, P, max_pad, need)) {
        return fail("bounded select: scratch size overflow");
    }
    if (!scratch || scratch_bytes < need) return fail("bounded select: scratch capacity short");
    scratch_carve cv;
    cv.base = scratch;
    cv.total = scratch_bytes;
    float * raw = static_cast<float *>(cv.take(0, 16));
    size_t hn = static_cast<size_t>(H) * N;
    raw = static_cast<float *>(cv.take(hn * sizeof(float), 16));
    float * norm = static_cast<float *>(cv.take(hn * sizeof(float), 16));
    float * qnorms = static_cast<float *>(cv.take(static_cast<size_t>(H) * sizeof(float), 16));
    float * stddevs = static_cast<float *>(cv.take(static_cast<size_t>(H) * sizeof(float), 16));
    scored_candidate * cands = static_cast<scored_candidate *>(cv.take(
        static_cast<size_t>(N) * sizeof(scored_candidate), 16));
    bounded_active * acts = static_cast<bounded_active *>(cv.take(
        static_cast<size_t>(N) * sizeof(bounded_active), 16));
    uint8_t * flags = static_cast<uint8_t *>(cv.take(N, 1));
    float * lm_row = static_cast<float *>(cv.take(static_cast<size_t>(max_pad) * sizeof(float), 16));
    size_t stage_rows = 0, rows_b = 0, stage8 = 0;
    if (!ck_mul(static_cast<size_t>(N), static_cast<size_t>(rmax), stage_rows) ||
        !ck_mul(stage_rows, size_t(4), rows_b) || !ck_add(rows_b, size_t(7), rows_b) ||
        !ck_mul(stage_rows, size_t(8), stage8)) {
        return fail("bounded select: staging overflow");
    }
    rows_b &= ~size_t(7);
    uint32_t * st_rows = static_cast<uint32_t *>(cv.take(rows_b, 8));
    int64_t * st_pos = static_cast<int64_t *>(cv.take(stage8, 8));
    uint64_t * st_pid = static_cast<uint64_t *>(cv.take(stage8, 8));
    uint64_t * st_gen = static_cast<uint64_t *>(cv.take(stage8, 8));
    size_t rebuild_floats = 0;
    {
        size_t dp = static_cast<size_t>(h_dim) * P;
        rebuild_floats = dp + P + static_cast<size_t>(h_dim) * 3;
    }
    float * rebuild = static_cast<float *>(cv.take(rebuild_floats * sizeof(float), 16));
    size_t tmp_sum = 0, tmp_bytes = 0;
    if (!ck_add(static_cast<size_t>(max_pad), static_cast<size_t>(P), tmp_sum) ||
        !ck_mul(tmp_sum, size_t(2), tmp_bytes)) {
        return fail("bounded select: tmp size overflow");
    }
    uint8_t * dectmp = static_cast<uint8_t *>(cv.take(tmp_bytes, 64));
    if (cv.fail || !raw || !norm || !qnorms || !stddevs || !cands || !acts || !flags ||
        (tmp_bytes && !dectmp) ||
        (max_pad && !lm_row) || (rebuild_floats && !rebuild) ||
        (stage_rows && (!st_rows || !st_pos || !st_pid || !st_gen))) {
        return fail("bounded select: scratch carve failed");
    }
    for (uint32_t i = 0; i < N; ++i) flags[i] = 0;
    // Active set over legal rows only; partial-causal fragments compact into staging.
    // Each active immediately scores all heads while its landmark is resident in lm_row
    // (identical float op order to the reference: qh-outer dot over d, then normalize).
    size_t stage_used = 0;
    uint32_t n_act = 0;
    for (uint32_t i = 0; i < N; ++i) {
        const landmark_fragment_view & fv = frags[i];
        if (fv.row_count == 0) continue;
        if (!fv.row_indices || !fv.storage_positions || !fv.payload_ids || !fv.landmark) {
            return fail("bounded select: null fragment rows");
        }
        const uint32_t * use_rows = fv.row_indices;
        const int64_t * use_pos = fv.storage_positions;
        const uint64_t * use_gen = fv.generations;
        uint32_t use_n = fv.row_count;
        if (query.causal_limit_pos >= 0) {
            if (fv.key->storage_pos0 > query.causal_limit_pos) continue;
            if (fv.storage_positions[fv.row_count - 1] > query.causal_limit_pos) {
                if (!segment && n_segments == 0) {
                    return fail("bounded select: partial causal chunk needs segment");
                }
                if (stage_used + fv.row_count > stage_rows) {
                    return fail("bounded select: staging short");
                }
                size_t base = stage_used;
                uint32_t k = 0;
                for (uint32_t r = 0; r < fv.row_count; ++r) {
                    if (fv.storage_positions[r] <= query.causal_limit_pos) {
                        st_rows[base + k] = fv.row_indices[r];
                        st_pos[base + k] = fv.storage_positions[r];
                        st_pid[base + k] = fv.payload_ids[r];
                        st_gen[base + k] = fv.generations ? fv.generations[r]
                                                         : fv.key->storage_generation;
                        ++k;
                    }
                }
                if (k == 0) continue;
                const xkv_segment * owner = resolve_owner_segment(segment, segments,
                    n_segments, fv.key->segment_id, fv.key->segment_version);
                if (!owner) {
                    return fail("bounded select: unresolvable owning segment");
                }
                if (fv.key->phase_tx_fingerprint != phase_fp ||
                    fv.key->source_fingerprint != owner->source_fingerprint) {
                    return fail("bounded select: stale phase/source for rebuild");
                }
                stage_used = base + k;
                float * tmp_b = rebuild;
                float * tmp_a = rebuild + static_cast<size_t>(h_dim) * P;
                float * tmp_pre = tmp_a + P;
                float * tmp_post = tmp_pre + h_dim;
                float * tmp_mean = tmp_post + h_dim;
                if (!rebuild_exact_mean(*owner, fv, st_rows + base, st_pos + base, k,
                    feature_offset, feature_dim, phase_tx, tmp_b, tmp_a, tmp_pre,
                    tmp_post, tmp_mean, dectmp, tmp_bytes, err)) {
                    return false;
                }
                for (uint32_t d = 0; d < h_dim; ++d) lm_row[d] = tmp_mean[d];
                use_rows = st_rows + base;
                use_pos = st_pos + base;
                use_gen = st_gen + base;
                use_n = k;
            }
        }
        if (use_rows == fv.row_indices) {
            const encoded_matrix * lmsrc = nullptr;
            uint64_t lrow = 0;
            if (!view_lm_source(fv, lmsrc, lrow, err)) return false;
            uint64_t lm_w = lmsrc->desc.padded_shape.cols;
            if (lm_w < h_dim) return fail("select_sr_query: decoded landmark too small");
            if (!decode_row_bounded(*lmsrc, lrow, lm_row, static_cast<size_t>(max_pad),
                dectmp, tmp_bytes, value_domain::canonical, err)) {
                return false;
            }
            for (uint32_t d = 0; d < h_dim; ++d) {
                if (!std::isfinite(lm_row[d])) {
                    return fail("select_sr_query: non-finite decoded landmark");
                }
            }
        }
        bounded_active & a = acts[n_act];
        a.src_index = i;
        a.row_count = use_n;
        a.row_indices = use_rows;
        a.storage_positions = use_pos;
        a.generations = use_gen;
        a.key = fv.key;
        a.error_bound = fv.error_bound;
        a.seg_id = fv.key->segment_id;
        a.seg_ver = fv.key->segment_version;
        a.eff_cutoff = (use_rows == fv.row_indices) ? int64_t(-2) : query.causal_limit_pos;
        a.group_index = fv.group_index;
        for (uint32_t h = 0; h < H; ++h) {
            const float * qptr = query.q_vec.data() + static_cast<size_t>(h) * h_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < h_dim; ++d) dot += qptr[d] * lm_row[d];
            float rs = dot * scale;
            if (!std::isfinite(rs)) return fail("select_sr_query: non-finite raw score");
            raw[static_cast<size_t>(h) * N + n_act] = rs;
        }
        flags[i] = 1;
        ++n_act;
    }
    // q norms (reference order: per-head, computed alongside raw; values identical).
    for (uint32_t h = 0; h < H; ++h) {
        const float * qptr = query.q_vec.data() + static_cast<size_t>(h) * h_dim;
        double nsq = 0.0;
        for (uint32_t d = 0; d < h_dim; ++d) nsq += static_cast<double>(qptr[d]) * qptr[d];
        qnorms[h] = static_cast<float>(std::sqrt(nsq));
    }
    // Per-head z-score normalization across all actives, then max/union across GQA.
    for (uint32_t h = 0; h < H; ++h) {
        stddevs[h] = 1.0f;
        for (uint32_t a = 0; a < n_act; ++a) norm[static_cast<size_t>(h) * N + a] = raw[static_cast<size_t>(h) * N + a];
    }
    if (n_act > 1) {
        for (uint32_t h = 0; h < H; ++h) {
            double sum = 0.0;
            for (uint32_t a = 0; a < n_act; ++a) sum += raw[static_cast<size_t>(h) * N + a];
            double mean = sum / static_cast<double>(n_act);
            double vsum = 0.0;
            for (uint32_t a = 0; a < n_act; ++a) {
                double diff = raw[static_cast<size_t>(h) * N + a] - mean;
                vsum += diff * diff;
            }
            double sd = std::sqrt(vsum / static_cast<double>(n_act));
            if (sd < 1e-10) sd = 1e-10;
            stddevs[h] = static_cast<float>(sd);
            for (uint32_t a = 0; a < n_act; ++a) {
                norm[static_cast<size_t>(h) * N + a] =
                    static_cast<float>((raw[static_cast<size_t>(h) * N + a] - mean) / sd);
            }
        }
    }
    uint32_t n_cands = n_act;
    for (uint32_t a = 0; a < n_act; ++a) {
        float max_s = -std::numeric_limits<float>::infinity();
        float max_m = 0.0f;
        for (uint32_t h = 0; h < H; ++h) {
            float s = norm[static_cast<size_t>(h) * N + a];
            if (s > max_s) {
                max_s = s;
                max_m = (acts[a].error_bound * qnorms[h] * scale) / stddevs[h];
            }
        }
        cands[a].frag_index = acts[a].src_index;
        cands[a].score = max_s;
        cands[a].error_margin = max_m;
    }
    sort_candidates(cands, n_cands);
    uint32_t budget = config.sr_budget;
    if (budget == 0 || budget > n_cands) budget = n_cands;
    if (out_scores && scores_cap < budget) return fail("bounded select: scores capacity short");
    // Boundary refinement (F32-exact, mirrors reference op order bit-exactly).
    uint32_t rows_refined = 0;
    bool cap_hit = false;
    bool refined_overflow = false;
    if (config.refine_mode == LLAMA_XKV_LANDMARK_REFINE_BOUNDARY &&
        (segment || n_segments > 0) && budget < n_cands) {
        float cutoff_score = cands[budget - 1].score;
        float cutoff_margin = cands[budget - 1].error_margin;
        size_t boundary_rows = 0;
        uint32_t n_boundary = 0;
        for (uint32_t c = 0; c < n_cands; ++c) {
            // locate active slot for this candidate (stable src identity)
            uint32_t slot = UINT32_MAX;
            for (uint32_t a = 0; a < n_act; ++a) {
                if (acts[a].src_index == cands[c].frag_index) { slot = a; break; }
            }
            if (slot == UINT32_MAX) return fail("bounded select: candidate identity lost");
            float lo = cands[c].score - cands[c].error_margin;
            float hi = cands[c].score + cands[c].error_margin;
            if (hi >= cutoff_score - cutoff_margin && lo <= cutoff_score + cutoff_margin) {
                size_t add = acts[slot].row_count;
                if (SIZE_MAX - boundary_rows < add) boundary_rows = SIZE_MAX;
                else boundary_rows += add;
                ++n_boundary;
            }
        }
        if (boundary_rows > config.refine_max_rows) {
            cap_hit = true;
        } else if (n_boundary > 0) {
            uint32_t fault_n = 0; // §15.4 injection counter (rebuild events this selection)
            for (uint32_t c = 0; c < n_cands; ++c) {
                uint32_t slot = UINT32_MAX;
                for (uint32_t a = 0; a < n_act; ++a) {
                    if (acts[a].src_index == cands[c].frag_index) { slot = a; break; }
                }
                float lo = cands[c].score - cands[c].error_margin;
                float hi = cands[c].score + cands[c].error_margin;
                if (!(hi >= cutoff_score - cutoff_margin && lo <= cutoff_score + cutoff_margin)) {
                    continue;
                }
                // Injected failure lands before any caller output/count mutation: rows,
                // scores, and counts below are only committed at emission; scratch and
                // table/cache are untouched (selection never mutates them).
                if (config.fault.stage == landmark_fault_stage::refine_rescore &&
                    fault_n++ == config.fault.occurrence) {
                    return fail("injected refine_rescore fault");
                }
                const bounded_active & ba = acts[slot];
                float * tmp_b = rebuild;
                float * tmp_a = rebuild + static_cast<size_t>(h_dim) * P;
                float * tmp_pre = tmp_a + P;
                float * tmp_post = tmp_pre + h_dim;
                float * tmp_mean = tmp_post + h_dim;
                // exact-mean inputs live in staging for partials, in src for full frags
                const uint32_t * rr = ba.row_indices;
                const int64_t * rp = ba.storage_positions;
                landmark_fragment_view fv = frags[ba.src_index];
                const xkv_segment * owner = resolve_owner_segment(segment, segments,
                    n_segments, fv.key->segment_id, fv.key->segment_version);
                if (!owner) {
                    return fail("bounded select: unresolvable owning segment");
                }
                if (fv.key->phase_tx_fingerprint != phase_fp ||
                    fv.key->source_fingerprint != owner->source_fingerprint) {
                    return fail("bounded select: stale phase/source for refine");
                }
                if (!rebuild_exact_mean(*owner, fv, rr, rp, ba.row_count, feature_offset,
                    feature_dim, phase_tx, tmp_b, tmp_a, tmp_pre, tmp_post, tmp_mean,
                    dectmp, tmp_bytes, err)) {
                    return false;
                }
                for (uint32_t h = 0; h < H; ++h) {
                    const float * qptr = query.q_vec.data() + static_cast<size_t>(h) * h_dim;
                    float dot = 0.0f;
                    for (uint32_t d = 0; d < h_dim; ++d) dot += qptr[d] * tmp_mean[d];
                    float rs = dot * scale;
                    if (!std::isfinite(rs)) return fail("bounded select: non-finite refined score");
                    raw[static_cast<size_t>(h) * N + slot] = rs;
                }
                if (UINT32_MAX - rows_refined < ba.row_count) refined_overflow = true;
                else rows_refined += ba.row_count;
            }
            if (refined_overflow) rows_refined = UINT32_MAX;
            if (n_act > 1) {
                for (uint32_t h = 0; h < H; ++h) {
                    double sum = 0.0;
                    for (uint32_t a = 0; a < n_act; ++a) sum += raw[static_cast<size_t>(h) * N + a];
                    double mean = sum / static_cast<double>(n_act);
                    double vsum = 0.0;
                    for (uint32_t a = 0; a < n_act; ++a) {
                        double diff = raw[static_cast<size_t>(h) * N + a] - mean;
                        vsum += diff * diff;
                    }
                    double sd = std::sqrt(vsum / static_cast<double>(n_act));
                    if (sd < 1e-10) sd = 1e-10;
                    stddevs[h] = static_cast<float>(sd);
                    for (uint32_t a = 0; a < n_act; ++a) {
                        norm[static_cast<size_t>(h) * N + a] =
                            static_cast<float>((raw[static_cast<size_t>(h) * N + a] - mean) / sd);
                    }
                }
            } else {
                for (uint32_t h = 0; h < H; ++h) {
                    for (uint32_t a = 0; a < n_act; ++a) {
                        norm[static_cast<size_t>(h) * N + a] = raw[static_cast<size_t>(h) * N + a];
                    }
                }
            }
            // NOTE: cands[] is in pre-refine rank order here; norm[] is in active order.
            // Map each candidate back to its active slot via stable src identity
            // (mirrors refined_norm_scores[qh][cand.frag_index] in the reference).
            for (uint32_t c = 0; c < n_cands; ++c) {
                uint32_t slot = UINT32_MAX;
                for (uint32_t a = 0; a < n_act; ++a) {
                    if (acts[a].src_index == cands[c].frag_index) { slot = a; break; }
                }
                if (slot == UINT32_MAX) return fail("bounded select: re-pool identity lost");
                float max_s = -std::numeric_limits<float>::infinity();
                for (uint32_t h = 0; h < H; ++h) {
                    float s = norm[static_cast<size_t>(h) * N + slot];
                    if (s > max_s) max_s = s;
                }
                cands[c].score = max_s;
            }
            sort_candidates(cands, n_cands);
        }
    }
    size_t n_rows = 0;
    size_t n_scores_out = 0;
    // Attribution arrays ride parallel to out_rows (pre-sort emission order).
    uint32_t * at_idx = (attrib && attrib->row_frag_index) ? attrib->row_frag_index : nullptr;
    landmark_fragment_key * at_key = (attrib && attrib->row_frag_key) ? attrib->row_frag_key : nullptr;
    size_t at_cap = attrib ? attrib->rows_cap : 0;
    const std::vector<segment_row_ref> * prot[3] = {&config.hot_rows, &config.recent_rows,
        &config.outlier_rows};
    const bool has_keeps =
        !config.hot_rows.empty() || !config.recent_rows.empty() || !config.outlier_rows.empty();
    if (n_cands == 0 && has_keeps) {
        // Zero-legal-fragment hard-keep path: no active rows exist to prove keeps against,
        // so explicit keeps union directly after exact segment-backed legality. Causal
        // compliance is unprovable without fragment positions -> fail closed, as is any
        // missing owner, out-of-range/dead row, or short capacity. Validate and size the
        // complete deduplicated union before writing so every refusal is output-atomic.
        if (query.causal_limit_pos >= 0) {
            return fail("bounded select: keep legality unprovable under causal cutoff");
        }
        if (!segment && n_segments == 0) {
            return fail("bounded select: no segment to establish keep legality");
        }
        auto seen_before = [&](int pi, size_t k, const segment_row_ref & ref) {
            for (int pj = 0; pj <= pi; ++pj) {
                const size_t end = pj == pi ? k : prot[pj]->size();
                for (size_t j = 0; j < end; ++j) {
                    if ((*prot[pj])[j] == ref) return true;
                }
            }
            return false;
        };
        size_t unique_keeps = 0;
        for (int pi = 0; pi < 3; ++pi) {
            for (size_t k = 0; k < prot[pi]->size(); ++k) {
                const segment_row_ref & ref = (*prot[pi])[k];
                const xkv_segment * owner = resolve_owner_segment(segment, segments, n_segments,
                    ref.segment_id, ref.segment_version);
                if (!owner) return fail("bounded select: keep owner unresolvable");
                if (ref.row >= owner->n_rows) return fail("bounded select: keep row out of range");
                if (ref.row < owner->live_rows.size() && !owner->live_rows[ref.row]) {
                    return fail("bounded select: keep row not live");
                }
                if (!seen_before(pi, k, ref)) {
                    if (unique_keeps == SIZE_MAX) return fail("bounded select: keep count overflow");
                    ++unique_keeps;
                }
            }
        }
        if (unique_keeps > rows_cap || ((at_idx || at_key) && unique_keeps > at_cap) ||
            (out_row_groups && unique_keeps > row_groups_cap)) {
            return fail("bounded select: rows capacity short");
        }
        for (int pi = 0; pi < 3; ++pi) {
            for (size_t k = 0; k < prot[pi]->size(); ++k) {
                const segment_row_ref & ref = (*prot[pi])[k];
                if (seen_before(pi, k, ref)) continue;
                out_rows[n_rows] = ref;
                if (at_idx) at_idx[n_rows] = UINT32_MAX; // no source fragment
                if (at_key) at_key[n_rows] = landmark_fragment_key();
                if (out_row_groups) out_row_groups[n_rows] = UINT32_MAX; // no source group
                ++n_rows;
            }
        }
    } else {
        // Emit rows: top-budget candidates, then positively-legal protected refs.
        // Selected membership as src-index flags for the protected legality scan.
        for (uint32_t i = 0; i < N; ++i) flags[i] = 0;
        for (uint32_t c = 0; c < budget; ++c) {
            if (cands[c].frag_index < N) flags[cands[c].frag_index] = 1;
        }
        auto emit_row = [&](const segment_row_ref & ref, const bounded_active & ba) -> bool {
            if (n_rows >= rows_cap) return false;
            if ((at_idx || at_key) && n_rows >= at_cap) return false;
            out_rows[n_rows] = ref;
            if (at_idx) at_idx[n_rows] = ba.src_index;
            if (at_key) {
                at_key[n_rows] = *ba.key;
                if (ba.eff_cutoff != int64_t(-2)) at_key[n_rows].causal_cutoff = ba.eff_cutoff;
            }
            if (out_row_groups) {
                if (n_rows >= row_groups_cap) return false;
                out_row_groups[n_rows] = ba.group_index;
            }
            ++n_rows;
            return true;
        };
        for (uint32_t c = 0; c < budget; ++c) {
            uint32_t slot = UINT32_MAX;
            for (uint32_t a = 0; a < n_act; ++a) {
                if (acts[a].src_index == cands[c].frag_index) { slot = a; break; }
            }
            if (slot == UINT32_MAX) return fail("bounded select: selection identity lost");
            const bounded_active & ba = acts[slot];
            for (uint32_t r = 0; r < ba.row_count; ++r) {
                uint64_t gen = ba.generations ? ba.generations[r] : ba.key->storage_generation;
                segment_row_ref ref;
                ref.segment_id = ba.seg_id;
                ref.segment_version = ba.seg_ver;
                ref.storage_generation = gen;
                ref.row = ba.row_indices[r];
                if (!emit_row(ref, ba)) return fail("bounded select: rows capacity short");
            }
        }
        for (uint32_t c = 0; c < budget; ++c) {
            if (c >= scores_cap) return fail("bounded select: scores capacity short");
            out_scores[c] = cands[c].score;
        }
        for (int pi = 0; pi < 3; ++pi) {
            for (size_t k = 0; k < prot[pi]->size(); ++k) {
                const segment_row_ref & ref = (*prot[pi])[k];
                bool legal = false;
                uint32_t legal_slot = UINT32_MAX;
                for (uint32_t a = 0; a < n_act && !legal; ++a) {
                    const bounded_active & ba = acts[a];
                    if (ba.seg_id != ref.segment_id || ba.seg_ver != ref.segment_version) continue;
                    for (uint32_t r = 0; r < ba.row_count; ++r) {
                        uint64_t gen = ba.generations ? ba.generations[r] : ba.key->storage_generation;
                        if (ba.row_indices[r] == ref.row && gen == ref.storage_generation) {
                            if (query.causal_limit_pos < 0 ||
                                ba.storage_positions[r] <= query.causal_limit_pos) {
                                legal = true;
                                legal_slot = a;
                                break;
                            }
                        }
                    }
                }
                if (legal) {
                    if (legal_slot == UINT32_MAX) return fail("bounded select: attrib identity lost");
                    if (!emit_row(ref, acts[legal_slot])) {
                        return fail("bounded select: rows capacity short");
                    }
                }
            }
        }
        n_scores_out = budget;
    }
    if (at_idx || at_key || out_row_groups) {
        sort_rows_attributed(out_rows, at_idx, at_key, n_rows, out_row_groups);
        n_rows = unique_rows_attributed(out_rows, at_idx, at_key, n_rows, out_row_groups);
    } else {
        sort_row_refs(out_rows, n_rows);
        n_rows = unique_row_refs(out_rows, n_rows);
    }
    // Per-segment CSR over the final row order (distinct id/version, first appearance).
    if (attrib && attrib->seg_ids && attrib->seg_versions && attrib->out_n_segs &&
        attrib->seg_ptrs && attrib->seg_indices && attrib->out_n_seg_idx) {
        size_t n_segs = 0;
        for (size_t r = 0; r < n_rows; ++r) {
            bool known = false;
            for (size_t s = 0; s < n_segs; ++s) {
                if (attrib->seg_ids[s] == out_rows[r].segment_id &&
                    attrib->seg_versions[s] == out_rows[r].segment_version) {
                    known = true;
                    break;
                }
            }
            if (!known) {
                if (n_segs >= attrib->segs_cap) return fail("bounded select: segs capacity short");
                attrib->seg_ids[n_segs] = out_rows[r].segment_id;
                attrib->seg_versions[n_segs] = out_rows[r].segment_version;
                ++n_segs;
            }
        }
        // seg_ptrs caller contract: capacity segs_cap + 1 (final sentinel included).
        size_t idx = 0;
        for (size_t s = 0; s < n_segs; ++s) {
            if (idx > UINT32_MAX) return fail("bounded select: seg index overflow");
            attrib->seg_ptrs[s] = static_cast<uint32_t>(idx);
            for (size_t r = 0; r < n_rows; ++r) {
                if (out_rows[r].segment_id == attrib->seg_ids[s] &&
                    out_rows[r].segment_version == attrib->seg_versions[s]) {
                    if (idx >= attrib->seg_idx_cap) {
                        return fail("bounded select: seg indices capacity short");
                    }
                    if (r > UINT32_MAX) return fail("bounded select: seg index overflow");
                    attrib->seg_indices[idx++] = static_cast<uint32_t>(r);
                }
            }
        }
        if (idx > UINT32_MAX) return fail("bounded select: seg index overflow");
        attrib->seg_ptrs[n_segs] = static_cast<uint32_t>(idx);
        *attrib->out_n_segs = n_segs;
        *attrib->out_n_seg_idx = idx;
    }
    *out_n_rows = n_rows;
    *out_n_scores = n_scores_out;
    *out_rows_refined = rows_refined;
    *out_cap_hit = cap_hit;
    return true;
}
}
// ------------------------------------------------------------------------------------------------
// Production-bounded workspace + zero-heap selection.
//
// Partial-causal scoring note: the reference path re-quantizes the legal-row subset mean
// with config.landmark_type. The bounded path scores the EXACT FP32 subset mean computed in
// fixed scratch (same GEMM/phase/mean op order) under the fragment's stored conservative
// error margin. Both source ONLY legal rows; future rows never contribute. With F32 base
// landmarks (or no partial fragment) both paths agree bit-exactly; otherwise the bounded
// estimate is higher fidelity and legality-identical. Refinement (F32-exact in both paths)
// agrees bit-exactly.
// ------------------------------------------------------------------------------------------------
namespace landmark_detail {
// Multi-segment ownership resolution: exact (segment_id, segment_version) match.
// Table (when non-empty) is authoritative and misses fail closed; otherwise the legacy
// single segment must match exactly. Never silently reuses a foreign segment's factors.
inline const xkv_segment * resolve_owner_segment(const xkv_segment * single,
    const xkv_segment * const * table, size_t n_table, uint64_t seg_id, uint64_t seg_ver) {
    if (table && n_table > 0) {
        for (size_t i = 0; i < n_table; ++i) {
            const xkv_segment * s = table[i];
            if (s && s->segment_id == seg_id) {
                return (s->segment_version == seg_ver) ? s : nullptr;
            }
        }
        return nullptr;
    }
    if (single && single->segment_id == seg_id && single->segment_version == seg_ver) {
        return single;
    }
    return nullptr;
}
// Lockstep insertion sort of (row, frag-index, frag-key) triples by row.
inline void sort_rows_attributed(segment_row_ref * rows, uint32_t * fidx,
    landmark_fragment_key * fkey, size_t n, uint32_t * gidx) {
    for (size_t i = 1; i < n; ++i) {
        segment_row_ref cr = rows[i];
        uint32_t ci = fidx ? fidx[i] : 0;
        uint32_t gi = gidx ? gidx[i] : UINT32_MAX;
        landmark_fragment_key ck;
        bool have_k = (fkey != nullptr);
        if (have_k) ck = fkey[i];
        size_t j = i;
        while (j > 0 && cr < rows[j - 1]) {
            rows[j] = rows[j - 1];
            if (fidx) fidx[j] = fidx[j - 1];
            if (gidx) gidx[j] = gidx[j - 1];
            if (have_k) fkey[j] = fkey[j - 1];
            --j;
        }
        rows[j] = cr;
        if (fidx) fidx[j] = ci;
        if (gidx) gidx[j] = gi;
        if (have_k) fkey[j] = ck;
    }
}
// Lockstep unique over sorted triples; returns surviving count.
inline size_t unique_rows_attributed(segment_row_ref * rows, uint32_t * fidx,
    landmark_fragment_key * fkey, size_t n, uint32_t * gidx) {
    if (n == 0) return 0;
    size_t w = 1;
    for (size_t r = 1; r < n; ++r) {
        if (rows[r] != rows[w - 1]) {
            rows[w] = rows[r];
            if (fidx) fidx[w] = fidx[r];
            if (gidx) gidx[w] = gidx[r];
            if (fkey) fkey[w] = fkey[r];
            ++w;
        }
    }
    return w;
}
// Exact FP32 subset-mean rebuild over legal rows into caller floats (no heap).
// Returns false + err on any validation/lookup failure (fail closed).
inline bool rebuild_exact_mean(const xkv_segment & segment, const landmark_fragment_view & fv,
    const uint32_t * rows, const int64_t * positions, uint32_t n_rows, uint32_t feature_offset,
    uint32_t feature_dim, const phase_transform_fn & phase_tx, float * tmp_b, float * tmp_a,
    float * tmp_pre, float * tmp_post, float * out_mean, uint8_t * dectmp, size_t dectmp_bytes,
    std::string * err) {
    const xkv_factor_group_payload * group = segment.find_group_for_layer(fv.key->owning_layer);
    if (!group) {
        if (err) *err = "bounded rebuild: no factor group for owning_layer";
        return false;
    }
    if (!group->b_k) {
        if (err) *err = "bounded rebuild: group.b_k is null";
        return false;
    }
    if (!validate_feature_slice(*group, fv.key->owning_layer, feature_offset, feature_dim, err)) {
        return false;
    }
    if (group->a_k.desc.logical_shape.cols != group->b_k->desc.logical_shape.cols) {
        if (err) *err = "bounded rebuild: rank mismatch between A_K and B_K";
        return false;
    }
    uint64_t padded_rank = group->b_k->desc.padded_shape.cols;
    uint64_t logical_rank = group->a_k.desc.logical_shape.cols;
    if (padded_rank == 0 || padded_rank > UINT32_MAX) {
        if (err) *err = "bounded rebuild: invalid padded rank";
        return false;
    }
    for (uint32_t r = 0; r < n_rows; ++r) {
        if (rows[r] >= segment.n_rows) {
            if (err) *err = "bounded rebuild: row out of segment bounds";
            return false;
        }
    }
    // Decode B slice rows [feature_offset, +feature_dim) one feature at a time via
    // caller scratch (decode_rows writes only into caller memory).
    for (uint32_t d = 0; d < feature_dim; ++d) {
        uint64_t idx = static_cast<uint64_t>(feature_offset) + d;
        if (!decode_row_bounded(*group->b_k, idx, tmp_b + d * padded_rank,
            static_cast<size_t>(padded_rank), dectmp, dectmp_bytes,
            value_domain::canonical, err)) {
            return false;
        }
    }
    for (uint32_t r = 0; r < n_rows; ++r) {
        uint64_t idx = rows[r];
        if (!decode_row_bounded(group->a_k, idx, tmp_a, static_cast<size_t>(padded_rank),
            dectmp, dectmp_bytes, value_domain::canonical, err)) {
            return false;
        }
        for (uint32_t d = 0; d < feature_dim; ++d) {
            const float * b_feat = tmp_b + d * padded_rank;
            float sum = 0.0f;
            for (uint64_t k = 0; k < logical_rank; ++k) sum += tmp_a[k] * b_feat[k];
            tmp_pre[d] = sum;
        }
        if (phase_tx) {
            phase_tx(tmp_pre, positions[r], fv.key->kv_head, tmp_post);
        } else {
            for (uint32_t d = 0; d < feature_dim; ++d) tmp_post[d] = tmp_pre[d];
        }
        if (r == 0) {
            for (uint32_t d = 0; d < feature_dim; ++d) out_mean[d] = tmp_post[d];
        } else {
            for (uint32_t d = 0; d < feature_dim; ++d) out_mean[d] += tmp_post[d];
        }
    }
    float inv_n = 1.0f / static_cast<float>(n_rows);
    for (uint32_t d = 0; d < feature_dim; ++d) out_mean[d] *= inv_n;
    for (uint32_t d = 0; d < feature_dim; ++d) {
        if (!std::isfinite(out_mean[d])) {
            if (err) *err = "bounded rebuild: non-finite rebuilt landmark";
            return false;
        }
    }
    return true;
}
} // namespace landmark_detail
bool select_sr_query_bounded(const sr_query & query, const landmark_fragment_view * frags,
    size_t n_frags, const sr_selection_config & config, const xkv_segment * segment,
    uint32_t feature_offset, uint32_t feature_dim, const phase_transform_fn & phase_tx,
    uint64_t phase_tx_fingerprint, uint8_t * scratch, size_t scratch_bytes,
    segment_row_ref * out_rows, size_t rows_cap, size_t * out_n_rows, float * out_scores,
    size_t scores_cap, size_t * out_n_scores, uint32_t * out_rows_refined, bool * out_cap_hit,
    std::string * err, const xkv_segment * const * segments, size_t n_segments,
    sr_bounded_attribution * attrib, uint32_t * out_row_groups, size_t row_groups_cap) {
    if (n_frags > 0 && !frags) {
        if (err) *err = "bounded select: null fragments";
        return false;
    }
    return landmark_detail::select_query_core(query, frags, n_frags, config, segment,
        feature_offset, feature_dim, phase_tx, phase_tx_fingerprint, scratch, scratch_bytes,
        out_rows, rows_cap, out_n_rows, out_scores, scores_cap, out_n_scores,
        out_rows_refined, out_cap_hit, err, segments, n_segments, attrib, out_row_groups,
        row_groups_cap);
}
bool select_sr_batch_bounded(const sr_query * queries, size_t n_queries,
    const landmark_fragment_view * frags, size_t n_frags, const sr_selection_config & config,
    const xkv_segment * segment, uint32_t feature_offset, uint32_t feature_dim,
    const phase_transform_fn & phase_tx, uint64_t phase_tx_fingerprint, uint8_t * scratch,
    size_t scratch_bytes, segment_row_ref * out_gather, size_t gather_cap, size_t * out_n_gather,
    uint32_t * out_csr_ptrs, size_t csr_ptrs_cap, uint32_t * out_csr_indices, size_t csr_cap,
    size_t * out_n_csr, uint32_t * out_total_refined, uint32_t * out_total_cap_hits,
    std::string * err, const xkv_segment * const * segments, size_t n_segments,
    sr_bounded_attribution * attrib, uint32_t * out_csr_group_indices, size_t csr_group_cap) {
    using namespace landmark_detail;
    if (out_n_gather) *out_n_gather = 0;
    if (out_n_csr) *out_n_csr = 0;
    if (out_total_refined) *out_total_refined = 0;
    if (out_total_cap_hits) *out_total_cap_hits = 0;
    auto fail = [&](const char * m) -> bool { if (err) *err = m; return false; };
    if (!out_n_gather || !out_csr_ptrs || !out_csr_indices || !out_n_csr || !out_total_refined ||
        !out_total_cap_hits) {
        return fail("bounded batch: null output");
    }
    if (csr_ptrs_cap < n_queries + 1) return fail("bounded batch: csr_ptrs capacity short");
    if (n_queries == 0) {
        out_csr_ptrs[0] = 0;
        return true;
    }
    if (!queries || !scratch || !out_gather || !out_csr_indices) {
        return fail("bounded batch: null buffer");
    }
    if (n_frags > 0 && !frags) return fail("bounded batch: null fragments");
    if (n_frags > UINT32_MAX) return fail("bounded batch: too many fragments");
    uint32_t N = static_cast<uint32_t>(n_frags);
    uint32_t hmax = 0, dmax = 0, rmax = 0, max_pad = 0;
    for (size_t q = 0; q < n_queries; ++q) {
        uint32_t hd = queries[q].head_dim ? queries[q].head_dim : feature_dim;
        if (hd > dmax) dmax = hd;
        size_t nh = queries[q].q_head_indices.empty() ? 1 : queries[q].q_head_indices.size();
        if (nh > hmax) hmax = static_cast<uint32_t>(nh > UINT32_MAX ? UINT32_MAX : nh);
    }
    if (hmax == 0 || dmax == 0) return fail("bounded batch: invalid query geometry");
    size_t rows_sum = 0;
    for (uint32_t i = 0; i < N; ++i) {
        if (!frags[i].key) return fail("bounded batch: null fragment");
        const encoded_matrix * lmsrc = nullptr;
        uint64_t lrow = 0;
        if (!view_lm_source(frags[i], lmsrc, lrow, err)) return false;
        if (frags[i].row_count > rmax) rmax = frags[i].row_count;
        uint64_t pc = lmsrc->desc.padded_shape.cols;
        if (pc == 0 || pc > UINT32_MAX) return fail("bounded batch: invalid landmark width");
        if (static_cast<uint32_t>(pc) > max_pad) max_pad = static_cast<uint32_t>(pc);
        if (SIZE_MAX - rows_sum < frags[i].row_count) return fail("bounded batch: rows overflow");
        rows_sum += frags[i].row_count;
    }
    uint32_t P = 0;
    if (segment || n_segments > 0) {
        for (uint32_t i = 0; i < N; ++i) {
            const xkv_segment * owner = resolve_owner_segment(segment, segments, n_segments,
                frags[i].key->segment_id, frags[i].key->segment_version);
            if (!owner) continue;
            const auto * g = owner->find_group_for_layer(frags[i].key->owning_layer);
            if (!g || !g->b_k) continue;
            uint64_t pr = g->b_k->desc.padded_shape.cols;
            if (pr > 0 && pr <= UINT32_MAX && static_cast<uint32_t>(pr) > P) {
                P = static_cast<uint32_t>(pr);
            }
        }
    }
    size_t prot_sum = config.hot_rows.size() + config.recent_rows.size() + config.outlier_rows.size();
    size_t stage_refs = 0;
    if (!ck_add(rows_sum, prot_sum, stage_refs)) return fail("bounded batch: staging overflow");
    size_t core_need = 0;
    if (!core_scratch_bytes(N, hmax, dmax, rmax, P, max_pad, core_need)) {
        return fail("bounded batch: scratch size overflow");
    }
    size_t stage_bytes = 0, score_bytes = 0, counts_bytes = 0, total = 0;
    if (!ck_mul(stage_refs, sizeof(segment_row_ref), stage_bytes) ||
        !ck_mul(static_cast<size_t>(N), sizeof(float), score_bytes) ||
        !ck_mul(n_queries, sizeof(uint32_t), counts_bytes) ||
        !ck_add(core_need, stage_bytes, total) || !ck_add(total, score_bytes, total) ||
        !ck_add(total, counts_bytes, total)) {
        return fail("bounded batch: scratch size overflow");
    }
    size_t stg_bytes = 0;
    if (!ck_mul(stage_refs, sizeof(uint32_t), stg_bytes) ||
        !ck_add(total, stg_bytes, total) || !ck_add(total, size_t(64), total)) {
        return fail("bounded batch: scratch size overflow");
    }
    // stg_bytes (checked) backs the pass-2 staging-groups carve below.
    if (scratch_bytes < total) return fail("bounded batch: scratch capacity short");
    scratch_carve cv;
    cv.base = scratch;
    cv.total = scratch_bytes;
    uint8_t * core_mem = static_cast<uint8_t *>(cv.take(core_need, 16));
    segment_row_ref * staging = static_cast<segment_row_ref *>(cv.take(stage_bytes, 8));
    float * score_tmp = static_cast<float *>(cv.take(score_bytes ? score_bytes : 1, 16));
    uint32_t * q_counts = static_cast<uint32_t *>(cv.take(counts_bytes ? counts_bytes : 1, 8));
    // Staging attribution rides in scratch; caller gather-parallel attribution (when
    // requested) is filled by copying staging results into the gather tail per query.
    size_t st_idx_bytes = 0, st_key_bytes = 0;
    if (!ck_mul(stage_refs, sizeof(uint32_t), st_idx_bytes) ||
        !ck_mul(stage_refs, sizeof(landmark_fragment_key), st_key_bytes)) {
        return fail("bounded batch: scratch size overflow");
    }
    size_t total2 = total;
    if (!ck_add(total2, st_idx_bytes, total2) || !ck_add(total2, st_key_bytes, total2)) {
        return fail("bounded batch: scratch size overflow");
    }
    if (scratch_bytes < total2) return fail("bounded batch: scratch capacity short");
    cv.total = scratch_bytes;
    uint32_t * st_fidx = static_cast<uint32_t *>(cv.take(st_idx_bytes, 4));
    landmark_fragment_key * st_fkey = static_cast<landmark_fragment_key *>(cv.take(st_key_bytes, 8));
    uint32_t * st_groups = static_cast<uint32_t *>(cv.take(stg_bytes, 4));
    if (cv.fail || !core_mem || !score_tmp || !q_counts || (stage_refs && !staging) ||
        (stage_refs && (!st_fidx || !st_fkey || !st_groups))) {
        return fail("bounded batch: scratch carve failed");
    }
    sr_bounded_attribution st_attrib;
    st_attrib.row_frag_index = st_fidx;
    st_attrib.row_frag_key = st_fkey;
    st_attrib.rows_cap = stage_refs;
    uint32_t * g_idx = (attrib && attrib->row_frag_index) ? attrib->row_frag_index : nullptr;
    landmark_fragment_key * g_key = (attrib && attrib->row_frag_key) ? attrib->row_frag_key : nullptr;
    size_t g_at_cap = attrib ? attrib->rows_cap : 0;
    if (attrib && (g_idx || g_key) && g_at_cap < gather_cap) {
        return fail("bounded batch: attribution rows capacity short");
    }
    size_t gather_used = 0;
    uint32_t total_ref = 0, total_cap = 0;
    for (size_t q = 0; q < n_queries; ++q) {
        size_t n_rows = 0, n_scores = 0;
        uint32_t refined = 0;
        bool cap_hit = false;
        size_t remain = (gather_cap >= gather_used) ? (gather_cap - gather_used) : 0;
        if (!select_query_core(queries[q], frags, n_frags, config, segment, feature_offset,
            feature_dim, phase_tx, phase_tx_fingerprint, core_mem, core_need, staging,
            stage_refs, &n_rows, score_tmp, N, &n_scores, &refined, &cap_hit, err,
            segments, n_segments, &st_attrib, nullptr, 0)) {
            return false;
        }
        if (gather_used + n_rows < gather_used) return fail("bounded batch: gather overflow");
        if (n_rows > remain) return fail("bounded batch: gather capacity short");
        for (size_t k = 0; k < n_rows; ++k) {
            out_gather[gather_used + k] = staging[k];
            if (g_idx) g_idx[gather_used + k] = st_fidx[k];
            if (g_key) g_key[gather_used + k] = st_fkey[k];
        }
        q_counts[q] = (n_rows > UINT32_MAX) ? UINT32_MAX : static_cast<uint32_t>(n_rows);
        gather_used += n_rows;
        if (UINT32_MAX - total_ref < refined) total_ref = UINT32_MAX;
        else total_ref += refined;
        if (cap_hit && total_cap < UINT32_MAX) total_cap += 1;
    }
    if (g_idx || g_key) {
        sort_rows_attributed(out_gather, g_idx, g_key, gather_used, nullptr);
        gather_used = unique_rows_attributed(out_gather, g_idx, g_key, gather_used, nullptr);
    } else {
        sort_row_refs(out_gather, gather_used);
        gather_used = unique_row_refs(out_gather, gather_used);
    }
    size_t n_gather = gather_used;
    if (n_gather > UINT32_MAX) return fail("bounded batch: gather too large");
    // Per-segment CSR over the frozen gather union (caller attribution arrays).
    if (attrib && attrib->seg_ids && attrib->seg_versions && attrib->out_n_segs &&
        attrib->seg_ptrs && attrib->seg_indices && attrib->out_n_seg_idx) {
        size_t n_segs = 0;
        for (size_t r = 0; r < n_gather; ++r) {
            bool known = false;
            for (size_t s = 0; s < n_segs; ++s) {
                if (attrib->seg_ids[s] == out_gather[r].segment_id &&
                    attrib->seg_versions[s] == out_gather[r].segment_version) {
                    known = true;
                    break;
                }
            }
            if (!known) {
                if (n_segs >= attrib->segs_cap) {
                    return fail("bounded batch: segs capacity short");
                }
                attrib->seg_ids[n_segs] = out_gather[r].segment_id;
                attrib->seg_versions[n_segs] = out_gather[r].segment_version;
                ++n_segs;
            }
        }
        size_t idx = 0;
        for (size_t s = 0; s < n_segs; ++s) {
            if (idx > UINT32_MAX) return fail("bounded batch: seg index overflow");
            attrib->seg_ptrs[s] = static_cast<uint32_t>(idx);
            for (size_t r = 0; r < n_gather; ++r) {
                if (out_gather[r].segment_id == attrib->seg_ids[s] &&
                    out_gather[r].segment_version == attrib->seg_versions[s]) {
                    if (idx >= attrib->seg_idx_cap) {
                        return fail("bounded batch: seg indices capacity short");
                    }
                    if (r > UINT32_MAX) return fail("bounded batch: seg index overflow");
                    attrib->seg_indices[idx++] = static_cast<uint32_t>(r);
                }
            }
        }
        if (idx > UINT32_MAX) return fail("bounded batch: seg index overflow");
        attrib->seg_ptrs[n_segs] = static_cast<uint32_t>(idx);
        *attrib->out_n_segs = n_segs;
        *attrib->out_n_seg_idx = idx;
    }
    size_t csr_used = 0;
    for (size_t q = 0; q < n_queries; ++q) {
        if (csr_used > UINT32_MAX) return fail("bounded batch: csr overflow");
        out_csr_ptrs[q] = static_cast<uint32_t>(csr_used);
        size_t n_rows = 0, n_scores = 0;
        uint32_t refined = 0;
        bool cap_hit = false;
        if (!select_query_core(queries[q], frags, n_frags, config, segment, feature_offset,
            feature_dim, phase_tx, phase_tx_fingerprint, core_mem, core_need, staging,
            stage_refs, &n_rows, score_tmp, N, &n_scores, &refined, &cap_hit, err,
            segments, n_segments, nullptr, st_groups, stage_refs)) {
            return false;
        }
        for (size_t k = 0; k < n_rows; ++k) {
            size_t lo = 0, hi = n_gather;
            while (lo < hi) {
                size_t mid = lo + (hi - lo) / 2;
                if (out_gather[mid] < staging[k]) lo = mid + 1;
                else hi = mid;
            }
            if (lo >= n_gather || out_gather[lo] != staging[k]) {
                return fail("bounded batch: csr lookup failed");
            }
            if (csr_used >= csr_cap) return fail("bounded batch: csr capacity short");
            if (lo > UINT32_MAX) return fail("bounded batch: csr index overflow");
            out_csr_indices[csr_used++] = static_cast<uint32_t>(lo);
            // Effective group rides parallel: same entry order, no score/update duplication
            // (groups are metadata; scoring/union logic is untouched).
            if (out_csr_group_indices) {
                if (csr_used - 1 >= csr_group_cap) {
                    return fail("bounded batch: csr group capacity short");
                }
                out_csr_group_indices[csr_used - 1] = st_groups[k];
            }
        }
    }
    if (csr_used > UINT32_MAX) return fail("bounded batch: csr overflow");
    out_csr_ptrs[n_queries] = static_cast<uint32_t>(csr_used);
    *out_n_gather = n_gather;
    *out_n_csr = csr_used;
    *out_total_refined = total_ref;
    *out_total_cap_hits = total_cap;
    return true;
}
bool landmark_encode_f32_row(const float * mean, uint32_t dim, ggml_type type, uint64_t seed,
    uint8_t * out, size_t out_cap, size_t * out_n, codec_desc * out_desc,
    float * out_error_bound, std::string * err) {
    auto fail = [&](const char * m) -> bool { if (err) *err = m; return false; };
    if (!out_n || !out_desc || !out_error_bound) return fail("f32 row: null output");
    if (type != GGML_TYPE_F32) return fail("f32 row: type must be F32");
    if (dim == 0) return fail("f32 row: zero dim");
    codec_desc d;
    try {
        d = make_codec_desc(factor_role::landmark, GGML_TYPE_F32, orientation::token_major,
            {1, dim}, 0, seed);
    } catch (const std::exception &) {
        return fail("f32 row: descriptor rejected");
    }
    uint64_t need = 0;
    try {
        need = encoded_matrix_bytes(d);
    } catch (const std::exception &) {
        return fail("f32 row: byte size overflow");
    }
    *out_n = static_cast<size_t>(need);
    *out_desc = d;
    *out_error_bound = 0.0f;
    if (!out) return true; // size query: mean untouched
    if (!mean) return fail("f32 row: null mean");
    size_t want = 0;
    if (!landmark_detail::ck_mul(static_cast<size_t>(dim), sizeof(float), want) ||
        want != static_cast<size_t>(need)) {
        return fail("f32 row: size mismatch");
    }
    if (out_cap < static_cast<size_t>(need)) return fail("f32 row: output short");
    std::memcpy(out, mean, static_cast<size_t>(need));
    return true;
}
bool landmark_encode_quant_row(const float * mean, uint32_t dim, ggml_type type, uint64_t seed,
    uint8_t * out, size_t out_cap, size_t * out_n, codec_desc * out_desc,
    float * out_error_bound, std::string * err) {
    auto fail = [&](const char * m) -> bool { if (err) *err = m; return false; };
    if (!out_n || !out_desc || !out_error_bound) return fail("quant row: null output");
    if (type == GGML_TYPE_F32) {
        return landmark_encode_f32_row(mean, dim, type, seed, out, out_cap, out_n, out_desc,
            out_error_bound, err);
    }
    if (type != GGML_TYPE_Q8_0 && type != GGML_TYPE_TURBO4_0) {
        return fail("quant row: unsupported type (F32/Q8_0/Turbo4 only)");
    }
    if (dim == 0) return fail("quant row: zero dim");
    uint32_t group = (type == GGML_TYPE_TURBO4_0) ? 128 : 0;
    codec_desc d;
    try {
        d = make_codec_desc(factor_role::landmark, type, orientation::token_major,
            {1, dim}, group, seed);
    } catch (const std::exception &) {
        return fail("quant row: descriptor rejected");
    }
    uint64_t need = 0;
    try {
        need = encoded_matrix_bytes(d);
    } catch (const std::exception &) {
        return fail("quant row: byte size overflow");
    }
    *out_n = static_cast<size_t>(need);
    *out_desc = d;
    // Provisional bound; the builder re-measures authoritatively post-encode.
    // Direct callers get the measured conservative bound below (0.0 only on size query).
    *out_error_bound = 0.0f;
    if (!out) return true;
    if (!mean) return fail("quant row: null mean");
    if (out_cap < static_cast<size_t>(need)) return fail("quant row: output short");
    uint64_t pad = d.padded_shape.cols;
    if (type == GGML_TYPE_Q8_0) {
        if (ggml_blck_size(GGML_TYPE_Q8_0) != 32) return fail("quant row: Q8 block layout changed");
        if (pad % 32 != 0) return fail("quant row: Q8 pad misaligned");
        const struct ggml_type_traits * traits = ggml_get_type_traits(GGML_TYPE_Q8_0);
        if (!traits || !traits->from_float_ref) return fail("quant row: missing from_float_ref");
        if (!traits->to_float) return fail("quant row: missing to_float");
        const size_t blk_bytes = ggml_row_size(GGML_TYPE_Q8_0, 32);
        alignas(64) float in_blk[32];
        alignas(64) uint8_t q_blk[64];
        alignas(64) float dec_blk[32];
        if (blk_bytes == 0 || blk_bytes > sizeof(q_blk)) {
            return fail("quant row: Q8 block size");
        }
        uint64_t nb = pad / 32;
        size_t total = 0;
        if (!landmark_detail::ck_mul((size_t) nb, blk_bytes, total) ||
            total != static_cast<size_t>(need)) {
            return fail("quant row: Q8 size drift");
        }
        double ss = 0.0; // deterministic conservative L2 bound over logical dim
        for (uint64_t b = 0; b < nb; ++b) {
            uint64_t base = b * 32;
            for (uint32_t j = 0; j < 32; ++j) {
                uint64_t idx = base + j;
                in_blk[j] = (idx < dim) ? mean[idx] : 0.0f;
            }
            traits->from_float_ref(in_blk, q_blk, 32);
            std::memcpy(out + b * blk_bytes, q_blk, blk_bytes);
            traits->to_float(q_blk, dec_blk, 32);
            for (uint32_t j = 0; j < 32; ++j) {
                uint64_t idx = base + j;
                if (idx >= dim) break; // pad tail excluded: reference error convention
                double diff = (double) dec_blk[j] - (double) in_blk[j];
                ss += diff * diff;
            }
        }
        float eb = (float) (std::sqrt(ss) * 1.10);
        if (eb < 1e-6f) eb = 1e-4f;
        *out_error_bound = eb;
        return true;
    }
    if (pad % 128 != 0) return fail("quant row: Turbo pad misaligned");
    const size_t grp_bytes = ggml_row_size(GGML_TYPE_TURBO4_0, 128);
    if (grp_bytes == 0) return fail("quant row: Turbo group size");
    uint64_t ng = pad / 128;
    size_t total = 0;
    if (!landmark_detail::ck_mul((size_t) ng, grp_bytes, total) ||
        total != static_cast<size_t>(need)) {
        return fail("quant row: Turbo size drift");
    }
    alignas(64) float in_grp[128];
    for (uint64_t g = 0; g < ng; ++g) {
        uint64_t base = g * 128;
        for (uint32_t j = 0; j < 128; ++j) {
            uint64_t idx = base + j;
            in_grp[j] = (idx < dim) ? mean[idx] : 0.0f;
        }
        if (!ggml_quantize_turbo_row(GGML_TYPE_TURBO4_0, in_grp, out + g * grp_bytes, 128,
            128)) {
            return fail("quant row: turbo quantize failed");
        }
    }
    // Deterministic conservative L2 bound: canonical-decode each just-written group
    // from the output bytes (not the staged input) over the logical dim only.
    {
        alignas(64) float dec_grp[128];
        double ss = 0.0;
        for (uint64_t gg = 0; gg < ng; ++gg) {
            if (!ggml_dequantize_turbo_row(GGML_TYPE_TURBO4_0, out + gg * grp_bytes, dec_grp,
                128, 128, GGML_TURBO_DECODE_CANONICAL)) {
                return fail("quant row: turbo bound decode failed");
            }
            uint64_t base = gg * 128;
            for (uint32_t j = 0; j < 128; ++j) {
                uint64_t idx = base + j;
                if (idx >= dim) break;
                double diff = (double) dec_grp[j] - (double) mean[idx];
                ss += diff * diff;
            }
        }
        float eb = (float) (std::sqrt(ss) * 1.10);
        if (eb < 1e-6f) eb = 1e-4f;
        *out_error_bound = eb;
    }
    return true;
}
namespace landmark_detail {
struct build_geom {
    uint32_t P = 0;
    uint64_t rank = 0;
    uint32_t D = 0;
    uint32_t maxDim = 0;
    size_t n_chunks = 0;
    size_t tmp_row = 0;
    uint64_t seed = 0;
};
inline bool parse_build_spec(const landmark_build_spec & spec, build_geom & g,
    std::string * err) {
    auto fail = [&](const char * m) -> bool { if (err) *err = m; return false; };
    g = build_geom();
    if (!spec.a_k || !spec.b_k) return fail("build: null factor streams");
    if (spec.n_rows == 0) return fail("build: no rows");
    if (!spec.surviving_rows || !spec.storage_positions) return fail("build: null rows");
    if (!spec.layers || spec.n_layers == 0) return fail("build: no layers");
    if (spec.chunk_tokens == 0) return fail("build: chunk_tokens is 0");
    std::string verr;
    if (!spec.a_k->desc.validate(&verr) || !spec.b_k->desc.validate(&verr)) {
        return fail("build: invalid factor descriptor");
    }
    uint64_t a_pad = spec.a_k->desc.padded_shape.cols;
    uint64_t b_pad = spec.b_k->desc.padded_shape.cols;
    if (a_pad == 0 || a_pad != b_pad || a_pad > UINT32_MAX) {
        return fail("build: rank mismatch between A_K and B_K");
    }
    if (spec.a_k->desc.logical_shape.cols != spec.b_k->desc.logical_shape.cols) {
        return fail("build: rank mismatch between A_K and B_K");
    }
    g.P = static_cast<uint32_t>(a_pad);
    g.rank = spec.a_k->desc.logical_shape.cols;
    uint64_t b_rows = spec.b_k->desc.logical_shape.rows;
    uint64_t run = 0;
    for (size_t li = 0; li < spec.n_layers; ++li) {
        const landmark_build_layer & lay = spec.layers[li];
        if (lay.feature_dim == 0) return fail("build: empty layer slice");
        if (lay.feature_offset != run) return fail("build: slices must be contiguous from 0");
        if (!ck_add(run, lay.feature_dim, run) || run > b_rows) {
            return fail("build: layer slice exceeds B_K width");
        }
        if (lay.feature_dim > g.maxDim) g.maxDim = lay.feature_dim;
    }
    if (run == 0 || run > UINT32_MAX) return fail("build: invalid total width");
    g.D = static_cast<uint32_t>(run);
    size_t tail = 0;
    if (!ck_add(spec.n_rows, (size_t) spec.chunk_tokens - 1, tail)) {
        return fail("build: chunk count overflow");
    }
    g.n_chunks = tail / spec.chunk_tokens;
    size_t ra = ggml_row_size(spec.a_k->desc.type, static_cast<int64_t>(a_pad));
    size_t rb = ggml_row_size(spec.b_k->desc.type, static_cast<int64_t>(b_pad));
    if (ra == 0 || rb == 0) return fail("build: row size");
    g.tmp_row = (ra > rb) ? ra : rb;
    uint64_t h = 0xcbf29ce484222325ULL;
    auto mix = [&h](uint64_t v) { h ^= v; h *= 0x100000001b3ULL; };
    mix(spec.seed_extra);
    mix(spec.phase_tx_fingerprint);
    mix(static_cast<uint64_t>(spec.landmark_type));
    mix(spec.chunk_tokens);
    mix(run);
    for (size_t li = 0; li < spec.n_layers; ++li) {
        mix(spec.layers[li].owning_layer);
        mix(spec.layers[li].feature_offset);
        mix(spec.layers[li].feature_dim);
        mix(spec.layers[li].kv_head);
    }
    g.seed = h;
    return true;
}
inline bool build_scratch_floats(const build_geom & g, size_t & out_floats) {
    out_floats = 0;
    size_t tile = 0, acc4 = 0;
    if (!ck_mul_u32(g.maxDim, g.P, tile)) return false;
    if (!ck_mul(static_cast<size_t>(g.D), size_t(4), acc4)) return false;
    if (!ck_add(tile, (size_t) g.P, out_floats)) return false;
    if (!ck_add(out_floats, acc4, out_floats)) return false;
    return true;
}
// Full scratch need including the pad-wide decode-destination row (for the
// authoritative post-encode error-bound measurement) and decode tmp covering the
// worst factor row and the worst landmark row.
inline bool build_scratch_need(const build_geom & g, ggml_type lm_type, uint32_t pad,
    size_t & out_floats, size_t & out_tmp) {
    out_floats = 0;
    out_tmp = 0;
    if (!build_scratch_floats(g, out_floats)) return false;
    if (!ck_add(out_floats, (size_t) pad, out_floats)) return false;
    size_t lm_tmp = 0;
    if (lm_type == GGML_TYPE_Q8_0) {
        lm_tmp = ggml_row_size(GGML_TYPE_Q8_0, (int64_t) pad);
        if (lm_tmp == 0) return false;
    }
    out_tmp = (lm_tmp > g.tmp_row) ? lm_tmp : g.tmp_row;
    return true;
}
} // namespace landmark_detail
bool landmark_build_workspace_bytes(const landmark_build_spec & spec,
    landmark_chunk_encoder_fn encoder, size_t & out_workspace, size_t & out_arena,
    size_t & out_n_chunks, std::string * err) {
    using namespace landmark_detail;
    out_workspace = 0;
    out_arena = 0;
    out_n_chunks = 0;
    auto fail = [&](const char * m) -> bool { if (err) *err = m; return false; };
    if (!encoder) return fail("build: null encoder");
    build_geom g;
    if (!parse_build_spec(spec, g, err)) return false;
    size_t per_chunk = 0;
    codec_desc size_cd;
    {
        codec_desc cd;
        float eb = 0.0f;
        if (!encoder(nullptr, g.D, spec.landmark_type, g.seed, nullptr, 0, &per_chunk, &cd,
            &eb, err)) {
            return false;
        }
        size_cd = cd;
    }
    if (size_cd.type != spec.landmark_type) return fail("build: encoder type drift");
    if (size_cd.padded_shape.cols == 0 || size_cd.padded_shape.cols > UINT32_MAX) {
        return fail("build: encoder pad width");
    }
    uint32_t pad = static_cast<uint32_t>(size_cd.padded_shape.cols);
    size_t floats = 0, tmp = 0;
    if (!build_scratch_need(g, spec.landmark_type, pad, floats, tmp)) {
        return fail("build: scratch size overflow");
    }
    size_t fbytes = 0;
    if (!ck_mul(floats, sizeof(float), fbytes)) return fail("build: scratch size overflow");
    size_t ws = 0;
    if (!ck_add(fbytes, tmp, ws) || !ck_add(ws, size_t(160), ws)) {
        return fail("build: scratch size overflow");
    }
    size_t arena = 0;
    if (!ck_mul(g.n_chunks, per_chunk, arena)) return fail("build: arena size overflow");
    out_workspace = ws;
    out_arena = arena;
    out_n_chunks = g.n_chunks;
    return true;
}
bool build_landmarks_bounded(const landmark_build_spec & spec, const phase_transform_fn & phase_tx,
    landmark_chunk_encoder_fn encoder, uint8_t * scratch, size_t scratch_bytes, uint8_t * arena,
    size_t arena_bytes, landmark_build_output * out_chunks, size_t chunks_cap,
    size_t * out_n_chunks, std::string * err) {
    using namespace landmark_detail;
    if (out_n_chunks) *out_n_chunks = 0;
    auto fail = [&](const char * m) -> bool { if (err) *err = m; return false; };
    if (!encoder) return fail("build: null encoder");
    if (!scratch || !arena || !out_chunks || !out_n_chunks) return fail("build: null buffer");
    build_geom g;
    if (!parse_build_spec(spec, g, err)) return false;
    if (chunks_cap < g.n_chunks) return fail("build: chunks capacity short");
    size_t per_chunk = 0;
    codec_desc size_cd;
    {
        codec_desc cd;
        float eb = 0.0f;
        if (!encoder(nullptr, g.D, spec.landmark_type, g.seed, nullptr, 0, &per_chunk, &cd,
            &eb, err)) {
            return false;
        }
        size_cd = cd;
    }
    if (size_cd.type != spec.landmark_type) return fail("build: encoder type drift");
    if (size_cd.padded_shape.cols == 0 || size_cd.padded_shape.cols > UINT32_MAX) {
        return fail("build: encoder pad width");
    }
    uint32_t pad = static_cast<uint32_t>(size_cd.padded_shape.cols);
    size_t floats = 0, tmp = 0;
    if (!build_scratch_need(g, spec.landmark_type, pad, floats, tmp)) {
        return fail("build: scratch size overflow");
    }
    size_t fbytes = 0;
    if (!ck_mul(floats, sizeof(float), fbytes)) return fail("build: scratch size overflow");
    size_t need = 0;
    if (!ck_add(fbytes, tmp, need) || !ck_add(need, size_t(160), need)) {
        return fail("build: scratch size overflow");
    }
    if (scratch_bytes < need) return fail("build: scratch capacity short");
    size_t arena_need = 0;
    if (!ck_mul(g.n_chunks, per_chunk, arena_need)) return fail("build: arena size overflow");
    if (arena_bytes < arena_need) return fail("build: arena capacity short");
    scratch_carve cv;
    cv.base = scratch;
    cv.total = scratch_bytes;
    float * tile = static_cast<float *>(cv.take((size_t) g.maxDim * g.P * sizeof(float), 16));
    float * arow = static_cast<float *>(cv.take((size_t) g.P * sizeof(float), 16));
    float * recon = static_cast<float *>(cv.take((size_t) g.D * sizeof(float), 16));
    float * phased = static_cast<float *>(cv.take((size_t) g.D * sizeof(float), 16));
    float * acc = static_cast<float *>(cv.take((size_t) g.D * sizeof(float), 16));
    float * mean = static_cast<float *>(cv.take((size_t) g.D * sizeof(float), 16));
    float * dec = static_cast<float *>(cv.take((size_t) pad * sizeof(float), 16));
    uint8_t * dectmp = static_cast<uint8_t *>(cv.take(tmp, 64));
    if (cv.fail || !tile || !arow || !recon || !phased || !acc || !mean || !dec ||
        (tmp && !dectmp)) {
        return fail("build: scratch carve failed");
    }
    uint64_t a_rows = spec.a_k->desc.padded_shape.rows;
    size_t arena_used = 0;
    uint32_t loaded = UINT32_MAX;
    for (size_t c = 0; c < g.n_chunks; ++c) {
        size_t r0 = c * (size_t) spec.chunk_tokens;
        size_t r1 = r0 + (size_t) spec.chunk_tokens;
        if (r1 > spec.n_rows) r1 = spec.n_rows;
        for (uint32_t d = 0; d < g.D; ++d) acc[d] = 0.0f;
        uint64_t src_fp = 0xcbf29ce484222325ULL;
        auto mix_fp = [&src_fp](float v) {
            uint32_t u = 0;
            std::memcpy(&u, &v, sizeof(u));
            src_fp ^= u;
            src_fp *= 0x100000001b3ULL;
        };
        for (size_t ri = r0; ri < r1; ++ri) {
            uint32_t ar = spec.surviving_rows[ri];
            if ((uint64_t) ar >= a_rows) return fail("build: surviving row out of range");
            if (!decode_row_bounded(*spec.a_k, ar, arow, g.P, dectmp, tmp,
                value_domain::canonical, err)) {
                return false;
            }
            int64_t pos = spec.storage_positions[ri];
            for (size_t li = 0; li < spec.n_layers; ++li) {
                const landmark_build_layer & lay = spec.layers[li];
                if (loaded != (uint32_t) li) {
                    for (uint32_t d = 0; d < lay.feature_dim; ++d) {
                        if (!decode_row_bounded(*spec.b_k,
                            (uint64_t) lay.feature_offset + d, tile + (size_t) d * g.P,
                            g.P, dectmp, tmp, value_domain::canonical, err)) {
                            return false;
                        }
                    }
                    loaded = (uint32_t) li;
                }
                for (uint32_t d = 0; d < lay.feature_dim; ++d) {
                    const float * bf = tile + (size_t) d * g.P;
                    float sum = 0.0f;
                    for (uint64_t k = 0; k < g.rank; ++k) sum += arow[k] * bf[k];
                    recon[lay.feature_offset + d] = sum;
                }
                if (phase_tx) {
                    phase_tx(recon + lay.feature_offset, pos, lay.kv_head,
                        phased + lay.feature_offset);
                } else {
                    for (uint32_t d = 0; d < lay.feature_dim; ++d) {
                        phased[lay.feature_offset + d] = recon[lay.feature_offset + d];
                    }
                }
                for (uint32_t d = 0; d < lay.feature_dim; ++d) {
                    float v = phased[lay.feature_offset + d];
                    if (!std::isfinite(v)) return fail("build: non-finite phased row");
                    acc[lay.feature_offset + d] += v;
                    mix_fp(v);
                }
            }
        }
        float inv = 1.0f / (float) (r1 - r0);
        for (uint32_t d = 0; d < g.D; ++d) mean[d] = acc[d] * inv;
        if (arena_used + per_chunk < arena_used) return fail("build: arena offset overflow");
        if (arena_used + per_chunk > arena_bytes) return fail("build: arena capacity short");
        size_t wn = 0;
        codec_desc cd;
        float eb = 0.0f;
        if (!encoder(mean, g.D, spec.landmark_type, g.seed, arena + arena_used, per_chunk,
            &wn, &cd, &eb, err)) {
            return false;
        }
        // Authoritative conservative bound: decode the just-written bytes (zero-heap)
        // and compare against the mean, mirroring the reference error convention.
        if (wn != per_chunk) return fail("build: encoder size drift");
        if (cd.type != spec.landmark_type) return fail("build: encoder type drift");
        if (cd.padded_shape.cols != pad) return fail("build: encoder pad drift");
        if (!decode_row_bytes(cd, arena + arena_used, wn, 0, dec, (size_t) pad, dectmp,
            tmp, value_domain::canonical, err)) {
            return false;
        }
        double ss = 0.0;
        for (uint32_t d = 0; d < g.D; ++d) {
            double diff = (double) dec[d] - (double) mean[d];
            ss += diff * diff;
        }
        eb = (float) (std::sqrt(ss) * 1.10);
        if (eb < 1e-6f && spec.landmark_type != GGML_TYPE_F32) eb = 1e-4f;
        out_chunks[c].desc = cd;
        out_chunks[c].byte_offset = arena_used;
        out_chunks[c].byte_size = wn;
        out_chunks[c].error_bound = eb;
        out_chunks[c].source_fingerprint = src_fp;
        arena_used += wn;
    }
    *out_n_chunks = g.n_chunks;
    return true;
}
} // namespace llama_xkv
