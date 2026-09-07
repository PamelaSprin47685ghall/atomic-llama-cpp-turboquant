#include "llama-xkv-cache.h"
#include "llama-kv-cells.h"
#include "llama-xkv-factor.h"
#include "llama-xkv-backend.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace llama_xkv {

// Checked multiplication of two uint64_t values; returns false on overflow
static inline bool checked_mul_u64(uint64_t a, uint64_t b, uint64_t & out) {
    if (a == 0 || b == 0) {
        out = 0;
        return true;
    }
    if (a > std::numeric_limits<uint64_t>::max() / b) {
        return false;
    }
    out = a * b;
    return true;
}

// Checked addition of two uint64_t values; returns false on overflow
static inline bool checked_add_u64(uint64_t a, uint64_t b, uint64_t & out) {
    if (a > std::numeric_limits<uint64_t>::max() - b) {
        return false;
    }
    out = a + b;
    return true;
}

// Checked addition of two size_t values; returns false on overflow
static inline bool checked_add_size(size_t a, size_t b, size_t & out) {
    if (a > std::numeric_limits<size_t>::max() - b) {
        return false;
    }
    out = a + b;
    return true;
}

// COW pack helpers: atomic bundle cloning must never decode or memcpy
// DEVICE_OWNED bytes on host. Any landmark stream presence (any profile)
// requires a semantic rebuild; otherwise stale chunk summaries would be
// silently truncated. Host row-byte sizes are validated before any memcpy
// so corrupt sizes fail closed instead of reading out of bounds.
inline bool segment_needs_landmark_rebuild(const xkv_segment & seg) {
    if (seg.profile == LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS_LANDMARKS) {
        return true;
    }
    for (const auto & g : seg.groups) {
        if (!g.landmark_chunks.empty() || g.landmark_table_fingerprint != 0 ||
            g.landmark.desc.logical_shape.rows > 0 || !g.landmark.bytes.empty() ||
            g.bytes_landmark > 0) {
            return true;
        }
    }
    return false;
}

// Helper to extract unpadded row-major float matrix from decoded row-padded vector
static matrix unpad_matrix_from_decoded(const std::vector<float> & data, uint64_t rows, uint64_t log_cols, uint64_t pad_cols) {
    matrix m(rows, log_cols);
    for (uint64_t r = 0; r < rows; ++r) {
        std::memcpy(m.row_ptr(r), data.data() + r * pad_cols, log_cols * sizeof(float));
    }
    return m;
}

const char * xkv_state_to_str(xkv_state state) {
    switch (state) {
        case xkv_state::hot_writing:    return "HOT_WRITING";
        case xkv_state::hot_committed:  return "HOT_COMMITTED";
        case xkv_state::seal_candidate: return "SEAL_CANDIDATE";
        case xkv_state::flat_tq:        return "FLAT_TQ";
        case xkv_state::factored:       return "FACTORED";
    }
    return "UNKNOWN";
}

const char * xkv_skip_reason_to_str(xkv_skip_reason reason) {
    switch (reason) {
        case xkv_skip_reason::none:                     return "none";
        case xkv_skip_reason::not_committed:            return "not_committed";
        case xkv_skip_reason::unsupported_config:       return "unsupported_config";
        case xkv_skip_reason::preflight_oom:            return "preflight_oom";
        case xkv_skip_reason::factorization_failed:     return "factorization_failed";
        case xkv_skip_reason::codec_error:              return "codec_error";
        case xkv_skip_reason::error_threshold_exceeded: return "error_threshold_exceeded";
        case xkv_skip_reason::no_saving:                return "no_saving";
        case xkv_skip_reason::aborted:                  return "aborted";
        case xkv_skip_reason::landmark_required:        return "landmark_required";
        case xkv_skip_reason::store_capacity_exceeded:  return "store_capacity_exceeded";
    }
    return "unknown";
}

void xkv_factor_group_payload::refresh_descriptor_fingerprint() {
    uint64_t h = 0xcbf29ce484222325ULL;
    auto mix = [&h](uint64_t v) {
        h ^= v;
        h *= 0x100000001b3ULL;
    };
    mix(group_index);
    for (uint32_t l : owning_layers) mix(l);
    for (uint32_t off : layer_feature_offsets_k) mix(off);
    for (uint32_t dim : layer_feature_dims_k) mix(dim);
    for (uint32_t off : layer_feature_offsets_v) mix(off);
    for (uint32_t dim : layer_feature_dims_v) mix(dim);
    mix(total_dim_k);
    mix(total_dim_v);
    mix(baseline_original_row_bytes);
    mix(baseline_original_bytes);
    mix(a_k.desc.fingerprint());
    mix(b_k ? b_k->desc.fingerprint() : 0);
    mix(a_v.desc.fingerprint());
    mix(b_v ? b_v->desc.fingerprint() : 0);
    mix(landmark.desc.fingerprint());
    mix(landmark_table_fingerprint);
    for (const auto & ch : landmark_chunks) {
        mix(ch.row_begin);
        mix(ch.row_count);
        uint32_t bits = 0;
        std::memcpy(&bits, &ch.error_bound, sizeof(bits));
        mix(bits);
        mix(ch.source_fingerprint);
    }
    descriptor_fingerprint = h;
}

uint64_t compute_landmark_table_fingerprint(const std::vector<xkv_landmark_chunk> & chunks) {
    uint64_t h = 0xcbf29ce484222325ULL;
    auto mix = [&h](uint64_t v) {
        h ^= v;
        h *= 0x100000001b3ULL;
    };
    mix(chunks.size());
    for (const auto & ch : chunks) {
        mix(ch.row_begin);
        mix(ch.row_count);
        uint32_t bits = 0;
        std::memcpy(&bits, &ch.error_bound, sizeof(bits));
        mix(bits);
        mix(ch.source_fingerprint);
    }
    return h;
}

void xkv_factor_group_payload::update_byte_counters() {
    bytes_a_k = a_k.desc.logical_shape.rows > 0 ? encoded_matrix_bytes(a_k.desc) : a_k.bytes.size();
    bytes_b_k = (b_k && b_k->desc.logical_shape.rows > 0) ? encoded_matrix_bytes(b_k->desc) : (b_k ? b_k->bytes.size() : 0);
    bytes_a_v = a_v.desc.logical_shape.rows > 0 ? encoded_matrix_bytes(a_v.desc) : a_v.bytes.size();
    bytes_b_v = (b_v && b_v->desc.logical_shape.rows > 0) ? encoded_matrix_bytes(b_v->desc) : (b_v ? b_v->bytes.size() : 0);
    bytes_landmark = landmark.desc.logical_shape.rows > 0 ? encoded_matrix_bytes(landmark.desc) : landmark.bytes.size();

    // Size-based (never capacity): vector growth slack is nondeterministic
    // across implementations, while the preflight estimator counts sizes.
    // Capacity and estimate must agree exactly or the publish capacity
    // check trips on phantom bytes. Stream bytes are sizes on both sides.
    // Capacity-based: vectors are resident allocations (§2.2/11.1 requires
    // actual reserved memory, including slack). The preflight estimator
    // covers the worst case separately; never erase slack from accounting.
    bytes_metadata = owning_layers.capacity() * sizeof(uint32_t) +
                     layer_feature_offsets_k.capacity() * sizeof(uint32_t) +
                     layer_feature_dims_k.capacity() * sizeof(uint32_t) +
                     layer_feature_offsets_v.capacity() * sizeof(uint32_t) +
                     layer_feature_dims_v.capacity() * sizeof(uint32_t) +
                     sizeof(baseline_original_row_bytes) + sizeof(baseline_original_bytes) +
                     landmark_chunks.capacity() * sizeof(xkv_landmark_chunk) +
                     sizeof(landmark_table_fingerprint);

    // Logical sizes (not host vector sizes): for device bundles the host
    // vectors are released and these fields carry the allocated device bytes.
    // For host bundles logical == actual (validated), so behavior is unchanged.
    total_allocated_bytes = bytes_a_k + bytes_b_k +
                            bytes_a_v + bytes_b_v +
                            bytes_landmark + bytes_metadata;
}

void xkv_segment::update_byte_counters() {
    // Capacity-based like the payload counter above: resident allocation
    // (§2.2/11.1), including vector growth slack and word rounding.
    bytes_metadata_logical = row_payload_ids.capacity() * sizeof(uint64_t) +
                             (live_rows.capacity() + 7) / 8 +
                             sizeof(baseline_original_bytes);

    total_allocated_bytes = bytes_metadata_logical;
    for (auto & g : groups) {
        g.update_byte_counters();
        total_allocated_bytes += g.total_allocated_bytes;
    }

    uint64_t h = 0xcbf29ce484222325ULL;
    auto mix = [&h](uint64_t v) {
        h ^= v;
        h *= 0x100000001b3ULL;
    };
    mix(static_cast<uint64_t>(profile));
    mix(static_cast<uint64_t>(source));
    mix(static_cast<uint64_t>(residency));
    mix(layer_group_map_fingerprint);
    mix(baseline_original_bytes);
    mix(groups.size());
    for (const auto & g : groups) {
        mix(g.descriptor_fingerprint);
    }
    descriptor_fingerprint = h;
}

uint64_t compute_layer_group_map_fingerprint(const std::vector<xkv_factor_group_input> & group_inputs) {
    uint64_t h = 0xcbf29ce484222325ULL;
    auto mix = [&h](uint64_t v) {
        h ^= v;
        h *= 0x100000001b3ULL;
    };
    mix(group_inputs.size());
    for (const auto & g : group_inputs) {
        mix(g.group_index);
        mix(g.rank_k);
        mix(g.rank_v);
        for (uint32_t l : g.owning_layers) mix(l);
        for (uint32_t off : g.layer_feature_offsets_k) mix(off);
        for (uint32_t dim : g.layer_feature_dims_k) mix(dim);
        for (uint32_t off : g.layer_feature_offsets_v) mix(off);
        for (uint32_t dim : g.layer_feature_dims_v) mix(dim);
        mix(g.total_dim_k);
        mix(g.total_dim_v);
    }
    return h;
}

uint64_t compute_layer_group_map_fingerprint(const std::vector<xkv_factor_group_payload> & groups) {
    uint64_t h = 0xcbf29ce484222325ULL;
    auto mix = [&h](uint64_t v) {
        h ^= v;
        h *= 0x100000001b3ULL;
    };
    mix(groups.size());
    for (const auto & g : groups) {
        mix(g.group_index);
        mix(g.rank_k);
        mix(g.rank_v);
        for (uint32_t l : g.owning_layers) mix(l);
        for (uint32_t off : g.layer_feature_offsets_k) mix(off);
        for (uint32_t dim : g.layer_feature_dims_k) mix(dim);
        for (uint32_t off : g.layer_feature_offsets_v) mix(off);
        for (uint32_t dim : g.layer_feature_dims_v) mix(dim);
        mix(g.total_dim_k);
        mix(g.total_dim_v);
    }
    return h;
}

//
// xkv_arena_lease & xkv_workspace_arena
//

namespace xkv_arena_detail {
// Forward declaration: defined with the acquire path below; used by release.
inline void state_free_insert(xkv_arena_state & st, size_t offset, size_t size, int carry);
} // namespace xkv_arena_detail

xkv_arena_lease::xkv_arena_lease(std::shared_ptr<xkv_arena_state> state, size_t offset, size_t size, int32_t spare)
    : state(std::move(state)), leased_offset(offset), leased_size(size), spare_node(spare) {}

xkv_arena_lease::~xkv_arena_lease() {
    release();
}

xkv_arena_lease::xkv_arena_lease(xkv_arena_lease && o) noexcept
    : state(std::move(o.state)), leased_offset(o.leased_offset), leased_size(o.leased_size),
      spare_node(o.spare_node) {
    o.spare_node = -1;
    o.leased_offset = 0;
    o.leased_size = 0;
}

xkv_arena_lease & xkv_arena_lease::operator=(xkv_arena_lease && o) noexcept {
    if (this != &o) {
        release();
        state = std::move(o.state);
        leased_offset = o.leased_offset;
        leased_size = o.leased_size;
        spare_node = o.spare_node;
        o.spare_node = -1;
        o.leased_offset = 0;
        o.leased_size = 0;
    }
    return *this;
}

void xkv_arena_lease::release() {
    // Shared state keeps the backing buffer alive even if the arena is gone.
    if (state && leased_size > 0) {
        std::lock_guard<std::mutex> lock(state->mtx);
        if (state->live >= leased_size) {
            state->live -= leased_size;
        } else {
            state->live = 0;
        }
        if (state->live_count > 0) {
            state->live_count--;
        }
        if (state->base != nullptr && leased_offset + leased_size <= state->capacity) {
            xkv_arena_detail::state_free_insert(*state, leased_offset, leased_size, spare_node);
        }
        state.reset();
        leased_offset = 0;
        leased_size = 0;
        spare_node = -1;
    }
}

void * xkv_arena_lease::data() const {
    if (!valid()) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(state->mtx);
    if (state->base == nullptr || leased_offset >= state->capacity) {
        return nullptr;
    }
    return static_cast<uint8_t *>(state->base) + leased_offset;
}

namespace xkv_arena_detail {
// File-local free-list primitives on shared state. Caller holds state->mtx.
// No heap, no failure except explicit -1 when the preallocated pool is empty.
inline int state_alloc_node(xkv_arena_state & st) {
    if (st.free_nodes < 0) {
        return -1;
    }
    int idx = st.free_nodes;
    st.free_nodes = st.blocks[idx].next;
    st.blocks[idx].used = true;
    st.blocks[idx].next = -1;
    return idx;
}

inline void state_recycle_node(xkv_arena_state & st, int idx) {
    st.blocks[idx].used = false;
    st.blocks[idx].next = st.free_nodes;
    st.free_nodes = idx;
}

inline void state_pool_init(xkv_arena_state & st) {
    for (size_t i = 0; i < xkv_arena_state::kMaxBlocks; ++i) {
        st.blocks[i].used = false;
        st.blocks[i].next = (i + 1 < xkv_arena_state::kMaxBlocks) ? (int32_t)(i + 1) : -1;
    }
    st.free_nodes = 0;
    st.free_head = -1;
}
} // namespace xkv_arena_detail

xkv_workspace_arena::xkv_workspace_arena(size_t capacity_bytes) {
    xkv_arena_detail::state_pool_init(*state);
    if (capacity_bytes == 0) {
        return;
    }
    // Single init-time allocation of the whole backing buffer, 64B aligned.
    // Checked size; aligned new throws explicitly on resource exhaustion.
    size_t bytes = 0;
    if (!xkv_arena_detail::checked_round_up(capacity_bytes, kAlignment, bytes)) {
        throw std::invalid_argument("xkv_workspace_arena: capacity overflow");
    }
    state->base = ::operator new(bytes, std::align_val_t(kAlignment));
    state->capacity = bytes;
    int root = xkv_arena_detail::state_alloc_node(*state);
    state->blocks[root].offset = 0;
    state->blocks[root].size = bytes;
    state->blocks[root].next = -1;
    state->blocks[root].used = true;
    state->free_head = root;
}

xkv_workspace_arena::~xkv_workspace_arena() = default;

namespace xkv_arena_detail {
// Sorted (by offset) insert of the range into node `carry` (owned by the
// releasing lease; consumed or recycled by merges) with neighbor coalescing.
// The list stays sorted so every release merges all adjacent ranges in place.
// Caller holds state->mtx. No heap, no failure.
inline void state_free_insert(xkv_arena_state & st, size_t offset, size_t size, int carry) {
    st.blocks[carry].offset = offset;
    st.blocks[carry].size = size;

    int prev = -1;
    int cur = st.free_head;
    while (cur >= 0 && st.blocks[cur].offset < offset) {
        prev = cur;
        cur = st.blocks[cur].next;
    }
    st.blocks[carry].next = cur;
    if (prev < 0) {
        st.free_head = carry;
    } else {
        st.blocks[prev].next = carry;
    }
    // Merge with successor.
    if (cur >= 0 && offset + size == st.blocks[cur].offset) {
        st.blocks[carry].size += st.blocks[cur].size;
        st.blocks[carry].next = st.blocks[cur].next;
        st.blocks[cur].used = false;
        st.blocks[cur].next = st.free_nodes;
        st.free_nodes = cur;
    }
    // Merge with predecessor.
    if (prev >= 0 && st.blocks[prev].offset + st.blocks[prev].size == st.blocks[carry].offset) {
        st.blocks[prev].size += st.blocks[carry].size;
        st.blocks[prev].next = st.blocks[carry].next;
        st.blocks[carry].used = false;
        st.blocks[carry].next = st.free_nodes;
        st.free_nodes = carry;
    }
}
} // namespace xkv_arena_detail

bool xkv_workspace_arena::preflight(size_t needed_bytes) const {
    std::lock_guard<std::mutex> lock(state->mtx);
    if (state->base == nullptr || needed_bytes == 0) return false;
    size_t need = 0;
    if (!xkv_arena_detail::checked_round_up(needed_bytes, kAlignment, need)) {
        return false;
    }
    // True fit check: some free range must actually hold the rounded block.
    // Acquire additionally pops one spare node for the lease to carry, so a
    // fit alone is insufficient without spare metadata available.
    if (state->free_nodes < 0) {
        return false;
    }
    for (int cur = state->free_head; cur >= 0; cur = state->blocks[cur].next) {
        if (state->blocks[cur].size >= need) {
            return true;
        }
    }
    return false;
}

xkv_arena_lease xkv_workspace_arena::acquire(size_t size_bytes) {
    std::lock_guard<std::mutex> lock(state->mtx);
    if (state->base == nullptr || size_bytes == 0) {
        return xkv_arena_lease();
    }
    size_t need = 0;
    if (!xkv_arena_detail::checked_round_up(size_bytes, kAlignment, need)) {
        return xkv_arena_lease();
    }
    // Reserve one metadata node for this lease's eventual release BEFORE any
    // mutation: release can then never fail. Refusal leaves state untouched.
    // (One pop: this node travels with the lease. Splits reuse the carved
    // range node itself, so no second node is ever needed here.)
    int spare = xkv_arena_detail::state_alloc_node(*state);
    if (spare < 0) {
        return xkv_arena_lease();
    }
    // Best fit to limit fragmentation over long churn.
    int best = -1;
    for (int cur = state->free_head; cur >= 0; cur = state->blocks[cur].next) {
        if (state->blocks[cur].size >= need &&
            (best < 0 || state->blocks[cur].size < state->blocks[best].size)) {
            best = cur;
        }
    }
    if (best < 0) {
        // No fit: hand the popped spare back; free list otherwise untouched.
        xkv_arena_detail::state_recycle_node(*state, spare);
        return xkv_arena_lease();
    }
    // Unlink the chosen range.
    int prev = -1;
    for (int cur = state->free_head; cur != best; cur = state->blocks[cur].next) {
        prev = cur;
    }
    if (prev < 0) {
        state->free_head = state->blocks[best].next;
    } else {
        state->blocks[prev].next = state->blocks[best].next;
    }
    const size_t found_off = state->blocks[best].offset;
    const size_t found_size = state->blocks[best].size;
    // The spare popped above travels with the lease; the carved range node is
    // recycled (exact fit) or reused for the remainder (split).
    if (found_size > need) {
        // Split: remainder reuses the carved node, reinserted sorted.
        state->blocks[best].offset = found_off + need;
        state->blocks[best].size = found_size - need;
        int p = -1;
        int c = state->free_head;
        while (c >= 0 && state->blocks[c].offset < state->blocks[best].offset) {
            p = c;
            c = state->blocks[c].next;
        }
        state->blocks[best].next = c;
        if (p < 0) {
            state->free_head = best;
        } else {
            state->blocks[p].next = best;
        }
        // Coalesce with the new successor (predecessor cannot abut: the
        // carved head occupied [found_off, found_off + need)).
        if (c >= 0 && state->blocks[best].offset + state->blocks[best].size == state->blocks[c].offset) {
            state->blocks[best].size += state->blocks[c].size;
            state->blocks[best].next = state->blocks[c].next;
            xkv_arena_detail::state_recycle_node(*state, c);
        }
    } else {
        xkv_arena_detail::state_recycle_node(*state, best);
    }
    // Checked live accounting BEFORE publishing the lease; overflow refuses
    // with the free list already updated... see below for atomicity handling.
    size_t total = 0;
    if (!checked_add_size(state->live, need, total)) {
        // Practically unreachable (need <= capacity); unwind the carve.
        xkv_arena_detail::state_free_insert(*state, found_off, need, spare);
        return xkv_arena_lease();
    }
    state->live = total;
    state->live_count++;
    if (state->live > state->peak) {
        state->peak = state->live;
    }
    return xkv_arena_lease(state, found_off, need, spare);
}

size_t xkv_workspace_arena::get_live_bytes() const {
    std::lock_guard<std::mutex> lock(state->mtx);
    return state->live;
}

size_t xkv_workspace_arena::get_reserved_bytes() const {
    std::lock_guard<std::mutex> lock(state->mtx);
    return state->capacity;
}

size_t xkv_workspace_arena::get_peak_bytes() const {
    std::lock_guard<std::mutex> lock(state->mtx);
    return state->peak;
}

size_t xkv_workspace_arena::get_capacity_bytes() const {
    std::lock_guard<std::mutex> lock(state->mtx);
    return state->capacity;
}

bool xkv_workspace_arena::set_capacity_bytes(size_t bytes) {
    std::lock_guard<std::mutex> lock(state->mtx);
    // Init-time realloc only: outstanding leases hold spans into the old base.
    // Allocate the new buffer FIRST so failure leaves the old one intact.
    if (state->live != 0 || state->live_count != 0) {
        return false;
    }
    if (bytes == 0) {
        ::operator delete(state->base, std::align_val_t(kAlignment));
        state->base = nullptr;
        state->capacity = 0;
        state->peak = 0;
        state->free_head = -1;
        return true;
    }
    size_t rounded = 0;
    if (!xkv_arena_detail::checked_round_up(bytes, kAlignment, rounded)) {
        return false;
    }
    void * fresh = ::operator new(rounded, std::align_val_t(kAlignment));
    ::operator delete(state->base, std::align_val_t(kAlignment));
    state->base = fresh;
    state->capacity = rounded;
    state->peak = 0;
    // Rebuild the metadata pool from scratch: prior churn may have left the
    // free list fragmented across nodes that no longer describe this buffer.
    xkv_arena_detail::state_pool_init(*state);
    int root = xkv_arena_detail::state_alloc_node(*state);
    state->blocks[root].offset = 0;
    state->blocks[root].size = rounded;
    state->blocks[root].next = -1;
    state->blocks[root].used = true;
    state->free_head = root;
    return true;
}

void xkv_workspace_arena::reset_peak() {
    std::lock_guard<std::mutex> lock(state->mtx);
    state->peak = state->live;
}

const void * xkv_workspace_arena::base_data() const {
    std::lock_guard<std::mutex> lock(state->mtx);
    return state->base;
}

//
// xkv_reader_pin
//

xkv_reader_pin::xkv_reader_pin(std::shared_ptr<const xkv_segment> seg)
    : segment(std::move(seg)) {
    if (segment) {
        segment->pin_count.fetch_add(1, std::memory_order_relaxed);
    }
}

xkv_reader_pin::~xkv_reader_pin() {
    release();
}

xkv_reader_pin::xkv_reader_pin(xkv_reader_pin && o) noexcept
    : segment(std::move(o.segment)) {
    o.segment = nullptr;
}

xkv_reader_pin & xkv_reader_pin::operator=(xkv_reader_pin && o) noexcept {
    if (this != &o) {
        release();
        segment = std::move(o.segment);
        o.segment = nullptr;
    }
    return *this;
}

void xkv_reader_pin::release() {
    if (segment) {
        segment->pin_count.fetch_sub(1, std::memory_order_release);
        segment = nullptr;
    }
}

//
// llama_xkv_cache_store
//

// Checked conversion of the MiB budget to bytes; throws explicitly instead of
// wrapping. Defined here for the member initializer below.
static size_t checked_arena_bytes(const llama_cparams & cparams) {
    uint64_t bytes = 0;
    // Ownership: xkv_workspace_mib is the TOTAL; the runtime global B cache
    // consumes the xkv_decode_cache_mib subbudget when allocated separately.
    // DENSE/SR partition deterministically (total - decode); SHADOW keeps the
    // cache at zero and takes the full transient (metrics report this profile
    // via the resulting arena capacity).
    const uint64_t total_mib = (uint64_t) cparams.xkv_workspace_mib;
    uint64_t cache_mib = 0;
    if (cparams.xkv_mode != LLAMA_XKV_MODE_SHADOW) {
        cache_mib = (uint64_t) cparams.xkv_decode_cache_mib;
    }
    if (cache_mib > total_mib) {
        cache_mib = total_mib; // Clamp: arena floors at zero, never wraps.
    }
    const uint64_t arena_mib = total_mib - cache_mib;
    if (!checked_mul_u64(arena_mib, 1024ULL * 1024ULL, bytes) ||
        bytes > (uint64_t) std::numeric_limits<size_t>::max()) {
        throw std::invalid_argument("llama_xkv_cache_store: workspace byte conversion overflow");
    }
    return (size_t) bytes;
}

llama_xkv_cache_store::llama_xkv_cache_store(const llama_cparams & cparams)
    : cparams(cparams),
      workspace_arena(checked_arena_bytes(cparams)) {
    // Persistent ID generator starts at 1; restore reseeds from the image.
    alloc_id_gen = std::make_unique<xkv_allocation_id_generator>();
    stamp.live_epoch    = 1;
    stamp.content_epoch = 1;
    stamp.codec_epoch   = 1;
    stamp.binding_epoch = 1;
    // Pre-reserve the tiny skip-reason map so abort-path counter bumps never
    // allocate (and throw) after payload states were already mutated.
    skipped_counts.reserve(16);
}

llama_xkv_cache_store::~llama_xkv_cache_store() = default;

xkv_allocation_id_generator & llama_xkv_cache_store::allocation_id_generator() {
    return *alloc_id_gen;
}

bool llama_xkv_cache_store::find_location(uint64_t payload_id, xkv_location & out_loc) const {
    if (payload_id == 0) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mtx);
    auto it = payload_locations.find(payload_id);
    if (it != payload_locations.end()) {
        out_loc = it->second;
        return true;
    }
    return false;
}

bool llama_xkv_cache_store::register_hot_payload(
    uint64_t payload_id,
    uint32_t physical_row,
    uint64_t generation,
    xkv_state state,
    std::string * err
) {
    if (payload_id == 0) {
        if (err) *err = "register_hot_payload: payload_id cannot be 0";
        return false;
    }
    xkv_hot_payload_binding b;
    b.payload_id = payload_id;
    b.hot_slot_row = physical_row;
    b.storage_generation = generation;
    b.state = state;
    try {
        return register_hot_payloads({b}, err, nullptr);
    } catch (const std::exception & e) {
        if (err) *err = std::string("register_hot_payload failed: ") + e.what();
        return false;
    } catch (...) {
        if (err) *err = "register_hot_payload failed with an unknown exception";
        return false;
    }
}

bool llama_xkv_cache_store::register_hot_payloads(
    const std::vector<xkv_hot_payload_binding> & bindings,
    std::string * err,
    std::function<void()> test_failure_hook
) {
    // Whole-body catch-all: preflight construction (sets, node handles, hook,
    // validation, reserves) may allocate and must report false+unchanged;
    // the commit tail below performs no allocation and cannot throw.
    try {
    // Epoch reservation discipline: refused while held (registration never
    // carries a token).
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (check_reservation_locked(0)) {
            if (err) *err = "register_hot_payloads: epoch reservation held by another transaction";
            return false;
        }
    }
    if (bindings.empty()) return true;

    // Validate nonzero and unique PIDs + nonzero generations
    std::unordered_set<uint64_t> seen;
    seen.reserve(bindings.size());
    for (const auto & b : bindings) {
        if (b.payload_id == 0) {
            if (err) *err = "register_hot_payloads: payload_id cannot be 0";
            return false;
        }
        if (!seen.insert(b.payload_id).second) {
            if (err) *err = "register_hot_payloads: duplicate payload_id in batch: " + std::to_string(b.payload_id);
            return false;
        }
        if (b.storage_generation == 0) {
            if (err) *err = "register_hot_payloads: storage_generation cannot be 0";
            return false;
        }
        if (b.state != xkv_state::hot_writing && b.state != xkv_state::hot_committed) {
            if (err) *err = "register_hot_payloads: invalid initial hot state";
            return false;
        }
    }

    // Preallocate map node handles for any new insertions
    std::unordered_map<uint64_t, xkv_location> prealloc_map;
    for (const auto & b : bindings) {
        xkv_location loc;
        loc.kind = xkv_location_kind::hot;
        loc.segment_id = 0;
        loc.segment_version = 0;
        loc.row = b.hot_slot_row;
        loc.storage_generation = b.storage_generation;
        loc.state = b.state;
        prealloc_map[b.payload_id] = loc;
    }

    if (test_failure_hook) {
        // Injected/backend hooks must never escape: catch everything, report,
        // leave state untouched.
        try {
            test_failure_hook();
        } catch (const std::exception & e) {
            if (err) *err = std::string("register_hot_payloads failure hook threw: ") + e.what();
            return false;
        } catch (...) {
            if (err) *err = "register_hot_payloads failure hook threw an unknown exception";
            return false;
        }
    }

    std::lock_guard<std::mutex> lock(mtx);

    // Check for active seal transaction lock conflict on any existing payload
    for (const auto & b : bindings) {
        auto it = payload_locations.find(b.payload_id);
        if (it != payload_locations.end()) {
            if (it->second.state == xkv_state::seal_candidate && it->second.seal_tx_nonce != 0) {
                if (err) *err = "register_hot_payloads: payload " + std::to_string(b.payload_id) + " is locked in active seal transaction";
                return false;
            }
        }
    }

    bool bump_live = false;
    bool bump_binding = false;
    bool bump_content = false;

    for (const auto & b : bindings) {
        auto it = payload_locations.find(b.payload_id);
        if (it != payload_locations.end()) {
            // Idempotent exact repeats no-op
            if (it->second.kind == xkv_location_kind::hot &&
                it->second.row == b.hot_slot_row &&
                it->second.storage_generation == b.storage_generation &&
                it->second.state == b.state) {
                continue;
            }
            // Reject rebinding an existing FACTORED payload to hot directly without prior removal
            if (it->second.kind == xkv_location_kind::factored) {
                if (err) *err = "register_hot_payloads: cannot rebind factored payload " + std::to_string(b.payload_id) + " to hot directly";
                return false;
            }

            // Reject state downgrades: committed -> writing
            if (it->second.state == xkv_state::hot_committed && b.state == xkv_state::hot_writing) {
                if (err) *err = "register_hot_payloads: cannot downgrade state from hot_committed to hot_writing";
                return false;
            }

            if (it->second.storage_generation != b.storage_generation) {
                bump_binding = true;
                bump_content = true;
            }
            if (it->second.row != b.hot_slot_row) {
                bump_binding = true;
            }
            if (it->second.state == xkv_state::hot_writing && b.state == xkv_state::hot_committed) {
                bump_content = true;
            }
        } else {
            bump_live = true;
            bump_binding = true;
            if (b.state == xkv_state::hot_committed) {
                bump_content = true;
            }
        }
    }

    // Preflight can_bump BEFORE mutating any location or map
    uint8_t bump_mask = 0;
    if (bump_live) bump_mask |= bump_flag_live;
    if (bump_binding) bump_mask |= bump_flag_binding;
    if (bump_content) bump_mask |= bump_flag_content;

    if (!can_bump_locked(bump_mask, err)) {
        return false;
    }

    // Checked size + count before reserving
    size_t target_size = 0;
    if (!checked_add_size(payload_locations.size(), bindings.size(), target_size)) {
        if (err) *err = "register_hot_payloads: map size overflow";
        return false;
    }
    payload_locations.reserve(target_size);

    for (const auto & b : bindings) {
        auto it = payload_locations.find(b.payload_id);
        if (it != payload_locations.end()) {
            if (it->second.kind == xkv_location_kind::hot &&
                it->second.row == b.hot_slot_row &&
                it->second.storage_generation == b.storage_generation &&
                it->second.state == b.state) {
                continue;
            }

            it->second.kind = xkv_location_kind::hot;
            it->second.segment_id = 0;
            it->second.segment_version = 0;
            it->second.row = b.hot_slot_row;
            it->second.storage_generation = b.storage_generation;
            it->second.state = b.state;
        } else {
            auto node = prealloc_map.extract(b.payload_id);
            payload_locations.insert(std::move(node));
        }
    }

    bump_prechecked(bump_mask);

    return true;
    } catch (const std::exception & e) {
        if (err) *err = std::string("register_hot_payloads failed: ") + e.what();
        return false;
    } catch (...) {
        if (err) *err = "register_hot_payloads failed with an unknown exception";
        return false;
    }
}

bool llama_xkv_cache_store::remove_payload(
    uint64_t payload_id,
    std::string * err,
    xkv_landmark_rebuild_fn landmark_rebuild,
    xkv_removal_precommit_fn removal_precommit,
    uint64_t reservation_token
) {
    xkv_location loc;
    if (!find_location(payload_id, loc)) {
        if (err) *err = "remove_payload: payload " + std::to_string(payload_id) + " not found";
        return false;
    }
    try {
        return remove_payloads({payload_id}, {loc.storage_generation}, nullptr, err, nullptr,
                                 landmark_rebuild, removal_precommit, reservation_token);
    } catch (const std::exception & e) {
        if (err) *err = std::string("remove_payload failed: ") + e.what();
        return false;
    } catch (...) {
        if (err) *err = "remove_payload failed with an unknown exception";
        return false;
    }
}

bool llama_xkv_cache_store::remove_payloads(
    const std::vector<uint64_t> & payload_ids,
    const std::vector<uint64_t> & expected_generations,
    xkv_payload_removal_result * result,
    std::string * err,
    std::function<void()> test_failure_hook,
    xkv_landmark_rebuild_fn landmark_rebuild,
    xkv_removal_precommit_fn removal_precommit,
    uint64_t reservation_token
) {
    // Epoch reservation discipline: while held, only the live unspent token
    // with matching base epochs proceeds (the coordinator's single batch
    // remove). Anything else refuses with zero mutation.
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (check_reservation_locked(reservation_token)) {
            if (err) *err = reservation_token == 0
                ? "remove_payloads: epoch reservation held by another transaction"
                : "remove_payloads: invalid or stale reservation token";
            return false;
        }
    }
    if (payload_ids.empty()) {
        if (result) {
            std::lock_guard<std::mutex> lock(mtx);
            result->success = true;
            result->stamp = stamp;
        }
        return true;
    }

    if (expected_generations.size() != payload_ids.size()) {
        if (err) *err = "remove_payloads: payload_ids and expected_generations size mismatch";
        return false;
    }

    const size_t n = payload_ids.size();
    std::unordered_set<uint64_t> seen;
    seen.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        if (payload_ids[i] == 0) {
            if (err) *err = "remove_payloads: payload_id cannot be 0";
            return false;
        }
        if (!seen.insert(payload_ids[i]).second) {
            if (err) *err = "remove_payloads: duplicate payload_id in removal batch: " + std::to_string(payload_ids[i]);
            return false;
        }
        if (expected_generations[i] == 0) {
            if (err) *err = "remove_payloads: expected_generation cannot be 0";
            return false;
        }
    }

    struct seg_cloning_plan {
        uint64_t seg_id = 0;
        std::shared_ptr<const xkv_segment> old_seg;
        std::shared_ptr<const xkv_segment> new_seg;
        std::vector<std::pair<uint64_t, uint32_t>> surviving_payload_rows;
    };

    std::vector<seg_cloning_plan> plans;
    std::vector<uint32_t> out_hot_rows;
    std::vector<xkv_location_kind> out_kinds(n);
    xkv_snapshot_stamp snapshot_stamp = {};

    // Pre-validation under lock: verify all exist, match generation, and have no active seal tx
    {
        std::lock_guard<std::mutex> lock(mtx);
        snapshot_stamp = stamp;

        for (size_t i = 0; i < n; ++i) {
            uint64_t pid = payload_ids[i];
            auto it = payload_locations.find(pid);
            if (it == payload_locations.end()) {
                if (err) *err = "remove_payloads: payload " + std::to_string(pid) + " not found";
                return false;
            }
            if (it->second.storage_generation != expected_generations[i]) {
                if (err) *err = "remove_payloads: generation mismatch for payload " + std::to_string(pid);
                return false;
            }
            if (it->second.state == xkv_state::seal_candidate && it->second.seal_tx_nonce != 0) {
                if (err) *err = "remove_payloads: payload " + std::to_string(pid) + " is locked in active seal transaction";
                return false;
            }
            out_kinds[i] = it->second.kind;
            if (it->second.kind == xkv_location_kind::hot) {
                out_hot_rows.push_back(it->second.row);
            }
        }

        std::unordered_set<uint64_t> affected_seg_ids;
        for (uint64_t pid : payload_ids) {
            auto it = payload_locations.find(pid);
            if (it != payload_locations.end() && it->second.kind == xkv_location_kind::factored && it->second.segment_id > 0) {
                affected_seg_ids.insert(it->second.segment_id);
            }
        }

        for (uint64_t seg_id : affected_seg_ids) {
            auto seg_it = published_segments.find(seg_id);
            if (seg_it == published_segments.end()) {
                if (err) *err = "remove_payloads: published segment " + std::to_string(seg_id) + " not found";
                return false;
            }
            seg_cloning_plan plan;
            plan.seg_id = seg_id;
            plan.old_seg = seg_it->second;
            plans.push_back(plan);
        }
    }

    // Off-lock: construct replacement segment bundles for affected factored segments
    try {
        for (auto & plan : plans) {
            auto old_seg = plan.old_seg;
            std::vector<uint32_t> surviving_rows;
            for (uint32_t r = 0; r < old_seg->n_rows; ++r) {
                if (old_seg->live_rows[r] && seen.find(old_seg->row_payload_ids[r]) == seen.end()) {
                    surviving_rows.push_back(r);
                }
            }

            if (!surviving_rows.empty()) {
                const uint32_t new_n_rows = (uint32_t) surviving_rows.size();

                auto new_seg = std::make_shared<xkv_segment>();
                new_seg->segment_id = old_seg->segment_id;
                new_seg->segment_version = old_seg->segment_version + 1;
                new_seg->profile = old_seg->profile;
                new_seg->source = old_seg->source;
                new_seg->residency = old_seg->residency;
                new_seg->layer_group_map_fingerprint = old_seg->layer_group_map_fingerprint;
                new_seg->profile_fingerprint = old_seg->profile_fingerprint;
                new_seg->source_fingerprint = old_seg->source_fingerprint;
                // Fail before any host decode/memcpy: DEVICE_OWNED rows live in the
                // backend bundle (host bytes empty) and pack byte-for-byte only via
                // backend-native transactions owned outside the store.
                if (old_seg->residency == GGML_XKV_RES_DEVICE_OWNED) {
                    throw std::runtime_error(
                        "segment " + std::to_string(old_seg->segment_id) +
                        " is DEVICE_OWNED: host removal/pack forbidden; requires backend-native byte-preserving pack (no host decode/memcpy)");
                }
                new_seg->backend_bundle.reset(); // COW resets; host stays bundle-free.
                const bool is_landmarks = segment_needs_landmark_rebuild(*old_seg);

                if (is_landmarks && !landmark_rebuild) {
                    // Refusal path: chunk summaries are position-sensitive, so the
                    // store NEVER copies the first N chunks after arbitrary deletion,
                    // and duplicating the whole layout just to change the live mask
                    // breaks the byte budget. Without a semantic rebuild callback the
                    // transaction refuses with zero mutation; the old version stays
                    // valid and published.
                    throw std::runtime_error(
                        "LANDMARKS segment " + std::to_string(old_seg->segment_id) +
                        " requires a semantic landmark rebuild callback for removal/pack");
                } else {
                    new_seg->groups.resize(old_seg->groups.size());

                    for (size_t g_idx = 0; g_idx < old_seg->groups.size(); ++g_idx) {
                        const auto & old_g = old_seg->groups[g_idx];
                        auto & new_g = new_seg->groups[g_idx];

                        new_g.group_index = old_g.group_index;
                        new_g.owning_layers = old_g.owning_layers;
                        new_g.rank_k = old_g.rank_k;
                        new_g.rank_v = old_g.rank_v;
                        new_g.layer_feature_offsets_k = old_g.layer_feature_offsets_k;
                        new_g.layer_feature_dims_k = old_g.layer_feature_dims_k;
                        new_g.layer_feature_offsets_v = old_g.layer_feature_offsets_v;
                        new_g.layer_feature_dims_v = old_g.layer_feature_dims_v;
                        new_g.total_dim_k = old_g.total_dim_k;
                        new_g.total_dim_v = old_g.total_dim_v;
                        new_g.config_fingerprint = old_g.config_fingerprint;
                        new_g.baseline_original_row_bytes = old_g.baseline_original_row_bytes;
                        new_g.baseline_original_bytes = old_g.baseline_original_bytes;
                        new_g.b_k = old_g.b_k;
                        new_g.b_v = old_g.b_v;
                        new_g.landmark_chunks = old_g.landmark_chunks;
                        new_g.landmark_table_fingerprint = old_g.landmark_table_fingerprint;

                        const size_t stride_a_k = old_g.a_k.desc.row_stride_bytes;
                        const size_t stride_a_v = old_g.a_v.desc.row_stride_bytes;

                        // Exact host byte preconditions: fail closed before any memcpy
                        // so corrupt sizes read nothing out of bounds. Device bundles
                        // refuse above; reaching here with empty host bytes is corrupt.
                        if (stride_a_k == 0 || stride_a_v == 0) {
                            throw std::runtime_error("segment " + std::to_string(old_seg->segment_id) +
                                " group " + std::to_string(old_g.group_index) + " has zero row stride; host removal/pack forbidden");
                        }
                        if (old_g.a_k.bytes.size() != (size_t) old_seg->n_rows * stride_a_k ||
                            old_g.a_v.bytes.size() != (size_t) old_seg->n_rows * stride_a_v) {
                            throw std::runtime_error("segment " + std::to_string(old_seg->segment_id) +
                                " group " + std::to_string(old_g.group_index) + " host byte size mismatch; host removal/pack forbidden");
                        }

                        new_g.a_k.desc = old_g.a_k.desc;
                        new_g.a_k.desc.logical_shape.rows = new_n_rows;
                        new_g.a_k.desc.padded_shape.rows = new_n_rows;
                        new_g.a_k.bytes.resize(new_n_rows * stride_a_k);

                        new_g.a_v.desc = old_g.a_v.desc;
                        new_g.a_v.desc.logical_shape.rows = new_n_rows;
                        new_g.a_v.desc.padded_shape.rows = new_n_rows;
                        new_g.a_v.bytes.resize(new_n_rows * stride_a_v);

                        for (uint32_t dst_r = 0; dst_r < new_n_rows; ++dst_r) {
                            uint32_t src_r = surviving_rows[dst_r];
                            if (src_r >= old_seg->n_rows) {
                                throw std::runtime_error("segment " + std::to_string(old_seg->segment_id) +
                                    " surviving row out of range; host removal/pack forbidden");
                            }
                            std::memcpy(new_g.a_k.bytes.data() + dst_r * stride_a_k,
                                        old_g.a_k.bytes.data() + src_r * stride_a_k,
                                        stride_a_k);
                            std::memcpy(new_g.a_v.bytes.data() + dst_r * stride_a_v,
                                        old_g.a_v.bytes.data() + src_r * stride_a_v,
                                        stride_a_v);
                        }
                        // Non-LANDMARKS groups carry no landmark stream; LANDMARKS
                        // groups are filled atomically by the rebuild callback below.
                        new_g.landmark = encoded_matrix();
                        // Live covered bytes shrink with the row count: recompute the
                        // group total from the immutable per-row baseline (checked).
                        uint64_t packed_total = 0;
                        if (!checked_mul_u64((uint64_t) new_g.baseline_original_row_bytes,
                                             (uint64_t) new_n_rows, packed_total)) {
                            throw std::runtime_error("baseline byte total overflow for group " +
                                std::to_string(new_g.group_index));
                        }
                        new_g.baseline_original_bytes = packed_total;
                    }

                    new_seg->row_payload_ids.reserve(new_n_rows);
                    new_seg->live_rows.assign(new_n_rows, true);
                    new_seg->n_rows = new_n_rows;
                    new_seg->n_live_rows = new_n_rows;

                    for (uint32_t dst_r = 0; dst_r < new_n_rows; ++dst_r) {
                        uint32_t src_r = surviving_rows[dst_r];
                        uint64_t pid = old_seg->row_payload_ids[src_r];
                        new_seg->row_payload_ids.push_back(pid);
                        plan.surviving_payload_rows.push_back({pid, dst_r});
                    }

                    if (is_landmarks) {
                        std::string rebuild_err;
                        if (!landmark_rebuild(*old_seg, surviving_rows, *new_seg, &rebuild_err)) {
                            throw std::runtime_error(
                                "landmark rebuild refused pack for segment " +
                                std::to_string(old_seg->segment_id) + ": " + rebuild_err);
                        }
                    }
                    for (auto & new_g : new_seg->groups) {
                        new_g.refresh_descriptor_fingerprint();
                        new_g.update_byte_counters();
                    }
                    new_seg->update_byte_counters();
                    {
                        std::string cerr;
                        if (!validate_candidate(new_seg, &cerr)) {
                            throw std::runtime_error(
                                "rebuilt LANDMARKS bundle failed validation: " + cerr);
                        }
                    }
                    plan.new_seg = new_seg;
                }
            }
        }
    } catch (const std::exception & e) {
        if (err) *err = std::string("remove_payloads bundle cloning failed: ") + e.what();
        return false;
    } catch (...) {
        if (err) *err = "remove_payloads bundle cloning failed with an unknown exception";
        return false;
    }

    if (test_failure_hook) {
        try {
            test_failure_hook();
        } catch (const std::exception & e) {
            if (err) *err = std::string("remove_payloads failure hook threw: ") + e.what();
            return false;
        } catch (...) {
            if (err) *err = "remove_payloads failure hook threw an unknown exception";
            return false;
        }
    }

    // Phase 2: Commit under lock atomically
    std::vector<uint64_t> new_versions;
    std::vector<uint64_t> affected_list;

    // Prebuild the removal gate context and result copies off-lock: copies may
    // allocate, and after the gate returns true nothing may throw or allocate.
    xkv_removal_precommit_ctx pre_ctx;
    std::vector<uint64_t> removed_copy;
    try {
        pre_ctx.payload_ids = payload_ids;
        pre_ctx.affected_segment_ids.reserve(plans.size());
        pre_ctx.new_segment_versions.reserve(plans.size());
        for (const auto & plan : plans) {
            pre_ctx.affected_segment_ids.push_back(plan.seg_id);
            pre_ctx.new_segment_versions.push_back(plan.new_seg ? plan.new_seg->segment_version : 0);
        }
        if (result) {
            removed_copy = payload_ids;
        }
    } catch (const std::exception & e) {
        if (err) *err = std::string("remove gate context allocation failed: ") + e.what();
        return false;
    } catch (...) {
        if (err) *err = "remove gate context allocation failed with an unknown exception";
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(mtx);

        if (stamp != snapshot_stamp) {
            if (err) *err = "remove_payloads: concurrent transaction conflict (stamp mismatch)";
            return false;
        }

        for (const auto & plan : plans) {
            auto seg_it = published_segments.find(plan.seg_id);
            if (seg_it == published_segments.end() || seg_it->second != plan.old_seg) {
                if (err) *err = "remove_payloads: concurrent segment mutation conflict";
                return false;
            }
        }

        // Preflight can_bump BEFORE committing removals
        uint8_t rem_mask = bump_flag_live | bump_flag_binding | bump_flag_content;
        if (!can_bump_locked(rem_mask, err)) {
            return false;
        }

        // Token-carried path: revalidate liveness under the commit lock.
        if (active_res_token != 0) {
            if (reservation_token != active_res_token || reserve_spent ||
                !(stamp.live_epoch == reserve_base.live_epoch &&
                  stamp.content_epoch == reserve_base.content_epoch &&
                  stamp.codec_epoch == reserve_base.codec_epoch &&
                  stamp.binding_epoch == reserve_base.binding_epoch)) {
                if (err) *err = "remove_payloads: reservation lost before commit";
                return false;
            }
        }

        // Caught allocation preflight: reserves run before the gate; anything
        // that throws aborts with zero mutation.
        try {
            new_versions.reserve(plans.size());
            affected_list.reserve(plans.size());
            retired_segments.reserve(retired_segments.size() + plans.size());
        } catch (const std::exception & e) {
            if (err) *err = std::string("remove commit preflight allocation failed: ") + e.what();
            return false;
        }

        // Hard store cap: COW peak (live + retired + replacement versions,
        // shared B once) must fit before any mutation. Zero mutation on refusal.
        // (Walk allocates: inside the same atomic region, still pre-mutation.
        // A throw here escapes to the whole-body catch-all as clean false.)
        try {
            std::vector<std::shared_ptr<const xkv_segment>> cap_extras;
            cap_extras.reserve(plans.size());
            for (const auto & plan : plans) {
                if (plan.new_seg) cap_extras.push_back(plan.new_seg);
            }
            if (!ensure_accounting_scratch_locked(cap_extras, err)) return false;
            if (!store_capacity_fits_locked(cap_extras, err)) {
                return false;
            }
        } catch (const std::exception & e) {
            if (err) *err = std::string("remove capacity check failed: ") + e.what();
            return false;
        } catch (...) {
            if (err) *err = "remove capacity check failed with an unknown exception";
            return false;
        }

        // Removal precommit gate: external (pool) validation/release happens
        // here, atomically with the store commit. False/throw refuses with zero
        // mutation; after true the commit below performs no allocation and
        // cannot fail. The callback MUST NOT re-enter the store (lock held).
        if (removal_precommit) {
            std::string pre_err;
            bool pre_ok = false;
            try {
                pre_ok = removal_precommit(pre_ctx, &pre_err);
            } catch (const std::exception & e) {
                pre_err = e.what();
                pre_ok = false;
            } catch (...) {
                pre_err = "unknown non-standard exception";
                pre_ok = false;
            }
            if (!pre_ok) {
                if (err) *err = "removal precommit refused commit: " + pre_err;
                return false;
            }
        }

        // Single batch per token: mark spent atomically with the mutation.
        if (active_res_token != 0) {
            reserve_spent = true;
        }

        for (const auto & plan : plans) {
            auto seg_it = published_segments.find(plan.seg_id);
            if (plan.new_seg) {
                seg_it->second = plan.new_seg;
                for (const auto & pr : plan.surviving_payload_rows) {
                    auto loc_it = payload_locations.find(pr.first);
                    if (loc_it != payload_locations.end()) {
                        loc_it->second.segment_version = plan.new_seg->segment_version;
                        loc_it->second.row = pr.second;
                    }
                }
                new_versions.push_back(plan.new_seg->segment_version);
            } else {
                published_segments.erase(seg_it);
                new_versions.push_back(0);
            }
            retired_segments.push_back(plan.old_seg);
            affected_list.push_back(plan.seg_id);
        }

        for (uint64_t pid : payload_ids) {
            payload_locations.erase(pid);
        }

        bump_prechecked(rem_mask);

        reclaim_retired_segments_locked();

        if (result) {
            result->success = true;
            // No-throw tail: everything moved in was prebuilt/prereserved.
            result->removed_payload_ids = std::move(removed_copy);
            result->released_hot_rows = std::move(out_hot_rows);
            result->removed_kinds = std::move(out_kinds);
            result->affected_segment_ids = std::move(affected_list);
            result->new_segment_versions = std::move(new_versions);
            result->stamp = stamp;
        }

    }

    return true;
}
bool llama_xkv_cache_store::commit_hot_payload(uint64_t payload_id) {
    std::lock_guard<std::mutex> lock(mtx);
    if (check_reservation_locked(0)) {
        return false;
    }
    auto it = payload_locations.find(payload_id);
    if (it == payload_locations.end()) {
        return false;
    }
    if (it->second.state == xkv_state::hot_writing) {
        if (!can_bump_locked(bump_flag_content, nullptr)) {
            return false;
        }
        it->second.state = xkv_state::hot_committed;
        bump_prechecked(bump_flag_content);
        return true;
    }
    return (it->second.state == xkv_state::hot_committed);
}

bool llama_xkv_cache_store::commit_hot_payloads(const std::vector<uint64_t> & payload_ids) {
    std::lock_guard<std::mutex> lock(mtx);
    if (check_reservation_locked(0)) {
        return false;
    }
    // Pre-validate all exist and are in valid state; reject with zero partial mutation if any missing
    bool any_writing = false;
    for (uint64_t pid : payload_ids) {
        auto it = payload_locations.find(pid);
        if (it == payload_locations.end()) {
            return false;
        }
        if (it->second.state != xkv_state::hot_writing && it->second.state != xkv_state::hot_committed) {
            return false;
        }
        if (it->second.state == xkv_state::hot_writing) {
            any_writing = true;
        }
    }
    if (any_writing && !can_bump_locked(bump_flag_content, nullptr)) {
        return false;
    }
    for (uint64_t pid : payload_ids) {
        auto it = payload_locations.find(pid);
        if (it->second.state == xkv_state::hot_writing) {
            it->second.state = xkv_state::hot_committed;
        }
    }
    if (any_writing) {
        bump_prechecked(bump_flag_content);
    }
    return true;
}

bool llama_xkv_cache_store::mark_seal_candidates(const std::vector<uint64_t> & payload_ids,
                                                  const std::vector<uint64_t> & generations,
                                                  uint64_t * out_nonce,
                                                  std::string * err) {
    // Whole-body catch-all: map/node allocation outside try would otherwise
    // escape this bool API. Preflight construction happens before any lock;
    // the commit tail itself performs no allocation.
    try {
    if (payload_ids.size() != generations.size()) {
        if (err) *err = "mark_seal_candidates: payload_ids/generations size mismatch";
        return false;
    }
    std::lock_guard<std::mutex> lock(mtx);
    if (check_reservation_locked(0)) {
        if (err) *err = "mark_seal_candidates: epoch reservation held by another transaction";
        return false;
    }
    for (size_t i = 0; i < payload_ids.size(); ++i) {
        auto it = payload_locations.find(payload_ids[i]);
        if (it == payload_locations.end()) {
            if (err) *err = "mark_seal_candidates: payload " + std::to_string(payload_ids[i]) + " not found";
            return false;
        }
        if (it->second.state != xkv_state::hot_committed) {
            if (err) *err = "mark_seal_candidates: payload " + std::to_string(payload_ids[i]) + " not hot_committed";
            return false;
        }
        if (it->second.storage_generation != generations[i]) {
            if (err) *err = "mark_seal_candidates: stale generation for payload " + std::to_string(payload_ids[i]);
            return false;
        }
    }
    // Every seal (including this manual marking path) records a nonzero
    // monotonic transaction nonce so overlapping seals/rollbacks stay isolated.
    // Fail closed at the boundary: tokens never wrap or reuse (ABA would let
    // a stale abort authorize a foreign reservation's rollback).
    if (next_seal_tx_nonce == 0 || next_seal_tx_nonce == UINT64_MAX) {
        if (err) *err = "mark_seal_candidates: transaction nonce space exhausted";
        return false;
    }
    uint64_t nonce = next_seal_tx_nonce++;
    for (size_t i = 0; i < payload_ids.size(); ++i) {
        // Revalidated above; operator[] cannot insert here.
        payload_locations[payload_ids[i]].state = xkv_state::seal_candidate;
        payload_locations[payload_ids[i]].seal_tx_nonce = nonce;
    }
    if (out_nonce) *out_nonce = nonce;
    return true;
    } catch (const std::exception & e) {
        if (err) *err = std::string("mark_seal_candidates failed: ") + e.what();
        return false;
    } catch (...) {
        if (err) *err = "mark_seal_candidates failed with an unknown exception";
        return false;
    }
}

bool llama_xkv_cache_store::abort_seal_candidates(const std::vector<uint64_t> & payload_ids,
                                                   uint64_t nonce,
                                                   xkv_skip_reason reason) {
    try {
    std::lock_guard<std::mutex> lock(mtx);
    if (check_reservation_locked(0)) {
        return false;
    }
    for (uint64_t pid : payload_ids) {
        auto it = payload_locations.find(pid);
        // Nonce-scoped revert only: a stale abort must never cancel a later
        // overlapping seal that happens to cover the same payload ids.
        if (it != payload_locations.end() && it->second.state == xkv_state::seal_candidate &&
            it->second.seal_tx_nonce == nonce) {
            it->second.state = xkv_state::hot_committed;
            it->second.seal_tx_nonce = 0;
        }
    }
    if (reason != xkv_skip_reason::none) {
        skipped_counts[static_cast<uint8_t>(reason)]++;
    }
    return true;
    } catch (const std::exception & e) {
        (void) e;
        return false;
    } catch (...) {
        return false;
    }
}

bool llama_xkv_cache_store::find_payload_state(uint64_t payload_id, xkv_state & out_state) const {
    std::lock_guard<std::mutex> lock(mtx);
    auto it = payload_locations.find(payload_id);
    if (it != payload_locations.end()) {
        out_state = it->second.state;
        return true;
    }
    return false;
}

size_t llama_xkv_cache_store::get_sealed_count() const {
    std::lock_guard<std::mutex> lock(mtx);
    return sealed_count;
}

size_t llama_xkv_cache_store::get_skipped_count(xkv_skip_reason reason) const {
    std::lock_guard<std::mutex> lock(mtx);
    auto it = skipped_counts.find(static_cast<uint8_t>(reason));
    return (it != skipped_counts.end()) ? it->second : 0;
}

xkv_snapshot_stamp llama_xkv_cache_store::current_stamp() const {
    std::lock_guard<std::mutex> lock(mtx);
    return stamp;
}

uint64_t llama_xkv_cache_store::live_epoch() const {
    std::lock_guard<std::mutex> lock(mtx);
    return stamp.live_epoch;
}

uint64_t llama_xkv_cache_store::content_epoch() const {
    std::lock_guard<std::mutex> lock(mtx);
    return stamp.content_epoch;
}

uint64_t llama_xkv_cache_store::codec_epoch() const {
    std::lock_guard<std::mutex> lock(mtx);
    return stamp.codec_epoch;
}

uint64_t llama_xkv_cache_store::binding_epoch() const {
    std::lock_guard<std::mutex> lock(mtx);
    return stamp.binding_epoch;
}

bool llama_xkv_cache_store::can_bump(uint8_t mask, std::string * err) const {
    std::lock_guard<std::mutex> lock(mtx);
    return can_bump_locked(mask, err);
}

bool llama_xkv_cache_store::can_bump_locked(uint8_t mask, std::string * err) const {
    if ((mask & bump_flag_live) && stamp.live_epoch == UINT64_MAX) {
        if (err) *err = "xkv live_epoch overflow at UINT64_MAX";
        return false;
    }
    if ((mask & bump_flag_content) && stamp.content_epoch == UINT64_MAX) {
        if (err) *err = "xkv content_epoch overflow at UINT64_MAX";
        return false;
    }
    if ((mask & bump_flag_codec) && stamp.codec_epoch == UINT64_MAX) {
        if (err) *err = "xkv codec_epoch overflow at UINT64_MAX";
        return false;
    }
    if ((mask & bump_flag_binding) && stamp.binding_epoch == UINT64_MAX) {
        if (err) *err = "xkv binding_epoch overflow at UINT64_MAX";
        return false;
    }
    return true;
}

void llama_xkv_cache_store::bump_prechecked(uint8_t mask) noexcept {
    if (mask & bump_flag_live) stamp.live_epoch++;
    if (mask & bump_flag_content) stamp.content_epoch++;
    if (mask & bump_flag_codec) stamp.codec_epoch++;
    if (mask & bump_flag_binding) stamp.binding_epoch++;
}

bool llama_xkv_cache_store::bump_live_epoch(std::string * err) {
    std::lock_guard<std::mutex> lock(mtx);
    if (check_reservation_locked(0)) {
        if (err) *err = "bump_live_epoch: epoch reservation held by another transaction";
        return false;
    }
    if (!can_bump_locked(bump_flag_live, err)) return false;
    bump_prechecked(bump_flag_live);
    return true;
}

bool llama_xkv_cache_store::bump_content_epoch(std::string * err) {
    std::lock_guard<std::mutex> lock(mtx);
    if (check_reservation_locked(0)) {
        if (err) *err = "bump_content_epoch: epoch reservation held by another transaction";
        return false;
    }
    if (!can_bump_locked(bump_flag_content, err)) return false;
    bump_prechecked(bump_flag_content);
    return true;
}

bool llama_xkv_cache_store::bump_codec_epoch(std::string * err) {
    std::lock_guard<std::mutex> lock(mtx);
    if (check_reservation_locked(0)) {
        if (err) *err = "bump_codec_epoch: epoch reservation held by another transaction";
        return false;
    }
    if (!can_bump_locked(bump_flag_codec, err)) return false;
    bump_prechecked(bump_flag_codec);
    return true;
}

bool llama_xkv_cache_store::bump_binding_epoch(std::string * err) {
    std::lock_guard<std::mutex> lock(mtx);
    if (check_reservation_locked(0)) {
        if (err) *err = "bump_binding_epoch: epoch reservation held by another transaction";
        return false;
    }
    if (!can_bump_locked(bump_flag_binding, err)) return false;
    bump_prechecked(bump_flag_binding);
    return true;
}

void llama_xkv_cache_store::update_view_stamp(const llama_rerot_view_stamp & view) {
    std::lock_guard<std::mutex> lock(mtx);
    stamp.view = view;
}

bool llama_xkv_cache_store::validate_reservation_token(uint64_t token) const {
    std::lock_guard<std::mutex> lock(mtx);
    return token != 0 && token == active_res_token;
}

bool llama_xkv_cache_store::preflight_content_advance(xkv_stamp_reservation * out_res, std::string * err) {
    if (out_res == nullptr) {
        if (err) *err = "preflight_content_advance: null reservation output";
        return false;
    }
    std::lock_guard<std::mutex> lock(mtx);
    if (active_res_token != 0) {
        if (err) *err = "preflight_content_advance: reservation already held";
        return false;
    }
    if (!can_bump_locked(bump_flag_content, err)) {
        return false;
    }
    // Fail closed at the boundary: tokens never wrap or reuse (ABA would let
    // a stale token authorize a foreign reservation's epochs).
    if (next_res_token == 0 || next_res_token == UINT64_MAX) {
        if (err) *err = "preflight_content_advance: token space exhausted";
        return false;
    }
    uint64_t tok = next_res_token++;
    active_res_token = tok;
    reserve_base = stamp;
    reserve_spent = false;
    out_res->token = tok;
    out_res->base_stamp = stamp;
    out_res->reserved_stamp = stamp;
    out_res->reserved_stamp.content_epoch++;
    out_res->valid = true;
    return true;
}

xkv_snapshot_stamp llama_xkv_cache_store::apply_content_advance(
    const xkv_stamp_reservation & res, const xkv_removal_feedback & feedback) noexcept {
    std::lock_guard<std::mutex> lock(mtx);
    if (!res.valid || res.token == 0 || res.token != active_res_token) {
        return stamp;
    }
    if (feedback.has_store_stamp) {
        // Token-carried removal already advanced epochs. Adopt only on exact
        // match (view/codec included) so feedback can never mask a mismatch;
        // otherwise clear without adopting or bumping (fail-safe).
        if (reserve_spent && feedback.store_stamp == stamp) {
            stamp = feedback.store_stamp;
        }
        active_res_token = 0;
        reserve_spent = false;
        return stamp;
    }
    // No removal reported: reserved content advance once — unless a spent
    // removal already advanced epochs (defensive: never double-bump).
    // Valid by exclusion (preflighted at reserve; mutators refused since).
    if (!reserve_spent && stamp.content_epoch != UINT64_MAX) {
        stamp.content_epoch++;
    }
    active_res_token = 0;
    reserve_spent = false;
    return stamp;
}

void llama_xkv_cache_store::abort_content_advance(const xkv_stamp_reservation & res) noexcept {
    std::lock_guard<std::mutex> lock(mtx);
    // Legal only when the staged mutation did not commit; otherwise a no-op.
    if (res.valid && res.token == active_res_token && !reserve_spent) {
        active_res_token = 0;
    }
}

bool llama_xkv_cache_store::install_stable_view(const llama_rerot_view_stamp & stable_view,
                                                 xkv_snapshot_stamp * out_new_stamp,
                                                 std::string * err) {
    std::lock_guard<std::mutex> lock(mtx);
    if (check_reservation_locked(0)) {
        if (err) *err = "install_stable_view: epoch reservation held";
        return false;
    }
    stamp.view = stable_view;
    if (out_new_stamp != nullptr) {
        *out_new_stamp = stamp;
    }
    return true;
}

bool llama_xkv_cache_store::advance_context_shift(const llama_rerot_view_stamp & shifted_view,
                                                   xkv_snapshot_stamp * out_new_stamp,
                                                   std::string * err) {
    std::lock_guard<std::mutex> lock(mtx);
    if (check_reservation_locked(0)) {
        if (err) *err = "advance_context_shift: epoch reservation held";
        return false;
    }
    if (!can_bump_locked(bump_flag_binding, err)) {
        return false;
    }
    stamp.binding_epoch++;
    stamp.view = shifted_view;
    if (out_new_stamp != nullptr) {
        *out_new_stamp = stamp;
    }
    return true;
}

std::shared_ptr<xkv_segment> llama_xkv_cache_store::create_candidate_segment(
    llama_xkv_storage_profile profile,
    llama_xkv_source source,
    const std::vector<xkv_factor_group_payload> & groups
) {
    std::lock_guard<std::mutex> lock(mtx);
    auto seg = std::make_shared<xkv_segment>();
    seg->segment_id = next_segment_id++;
    seg->segment_version = 1;
    seg->profile = profile;
    seg->source = source;
    seg->groups = groups;
    std::vector<std::shared_ptr<const xkv_segment>> extras = { seg };
    ensure_accounting_scratch_locked(extras);
    return seg;
}

bool llama_xkv_cache_store::validate_candidate(
    const std::shared_ptr<const xkv_segment> & candidate,
    std::string * err
) const {
    if (!candidate) {
        if (err) *err = "candidate segment bundle is null";
        return false;
    }

    if (candidate->groups.empty()) {
        if (err) *err = "candidate segment bundle has no factor groups";
        return false;
    }

    const uint64_t bundle_rows = candidate->groups[0].a_k.desc.logical_shape.rows;
    if (bundle_rows == 0 || bundle_rows > std::numeric_limits<uint32_t>::max()) {
        if (err) *err = "candidate bundle rows invalid or exceed UINT32_MAX";
        return false;
    }

    std::unordered_set<uint32_t> seen_groups;
    std::unordered_set<uint32_t> seen_owning_layers;

    // Bundle-scope device cover: residency is segment-level and authoritative
    // (never inferred from profile or byte presence). Host bundles carry no
    // backend bundle; device bundles require one committed and verified.
    // Every device-covered stream across ALL groups is collected in the loop
    // below, then exact global cover is enforced once after the loop:
    // per-group enforcement would misreject other groups' handles as
    // unaccounted on multi-group device bundles.
    const auto & backend_bundle = candidate->backend_bundle;
    const bool is_device_bundle = (candidate->residency == GGML_XKV_RES_DEVICE_OWNED);
    if (!is_device_bundle && backend_bundle) {
        if (err) *err = "candidate host residency must not carry a backend bundle";
        return false;
    }
    if (is_device_bundle && (!backend_bundle || !backend_bundle->is_success() || !backend_bundle->is_committed())) {
        if (err) *err = "candidate device residency requires a committed verified backend bundle";
        return false;
    }
    struct need_cover { codec_desc desc; const char * name; uint32_t group_index; };
    std::vector<need_cover> device_needed;

    for (size_t g_idx = 0; g_idx < candidate->groups.size(); ++g_idx) {
        const auto & g = candidate->groups[g_idx];

        if (!seen_groups.insert(g.group_index).second) {
            if (err) *err = "duplicate group_index " + std::to_string(g.group_index) + " in bundle";
            return false;
        }

        if (g.owning_layers.empty()) {
            if (err) *err = "group " + std::to_string(g.group_index) + " has empty owning_layers";
            return false;
        }

        for (uint32_t layer : g.owning_layers) {
            if (!seen_owning_layers.insert(layer).second) {
                if (err) *err = "owning layer " + std::to_string(layer) + " appears in multiple factor groups";
                return false;
            }
        }

        // Validate layer feature offsets and dims
        if (g.layer_feature_offsets_k.size() != g.owning_layers.size() ||
            g.layer_feature_dims_k.size() != g.owning_layers.size() ||
            g.layer_feature_offsets_v.size() != g.owning_layers.size() ||
            g.layer_feature_dims_v.size() != g.owning_layers.size()) {
            if (err) *err = "group " + std::to_string(g.group_index) + " feature offset/dim vector sizes mismatch owning_layers count";
            return false;
        }

        // Validate contiguous, non-overlapping offsets starting at 0 and summing to total_dim_{k,v}
        uint64_t running_k = 0;
        for (size_t l = 0; l < g.owning_layers.size(); ++l) {
            if (g.layer_feature_dims_k[l] == 0) {
                if (err) *err = "group " + std::to_string(g.group_index) + " layer_feature_dims_k contains zero dimension";
                return false;
            }
            if (g.layer_feature_offsets_k[l] != running_k) {
                if (err) *err = "group " + std::to_string(g.group_index) + " layer_feature_offsets_k not contiguous or not starting at 0";
                return false;
            }
            running_k += g.layer_feature_dims_k[l];
        }
        if (running_k != g.total_dim_k || g.total_dim_k == 0) {
            if (err) *err = "group " + std::to_string(g.group_index) + " sum of layer_feature_dims_k != total_dim_k";
            return false;
        }

        uint64_t running_v = 0;
        for (size_t l = 0; l < g.owning_layers.size(); ++l) {
            if (g.layer_feature_dims_v[l] == 0) {
                if (err) *err = "group " + std::to_string(g.group_index) + " layer_feature_dims_v contains zero dimension";
                return false;
            }
            if (g.layer_feature_offsets_v[l] != running_v) {
                if (err) *err = "group " + std::to_string(g.group_index) + " layer_feature_offsets_v not contiguous or not starting at 0";
                return false;
            }
            running_v += g.layer_feature_dims_v[l];
        }
        if (running_v != g.total_dim_v || g.total_dim_v == 0) {
            if (err) *err = "group " + std::to_string(g.group_index) + " sum of layer_feature_dims_v != total_dim_v";
            return false;
        }

        // Validate stream descriptors
        if (!g.a_k.desc.validate(err)) return false;
        if (!g.b_k || !g.b_k->desc.validate(err)) {
            if (err && !g.b_k) *err = "group " + std::to_string(g.group_index) + " b_k is null";
            return false;
        }
        if (!g.a_v.desc.validate(err)) return false;
        if (!g.b_v || !g.b_v->desc.validate(err)) {
            if (err && !g.b_v) *err = "group " + std::to_string(g.group_index) + " b_v is null";
            return false;
        }

        // Validate B logical rows == total_dim and B cols == A rank
        if (g.b_k->desc.logical_shape.rows != g.total_dim_k || g.b_v->desc.logical_shape.rows != g.total_dim_v) {
            if (err) *err = "group " + std::to_string(g.group_index) + " B logical rows != total_dim_{k,v}";
            return false;
        }

        if (g.a_k.desc.logical_shape.cols != g.b_k->desc.logical_shape.cols ||
            g.a_v.desc.logical_shape.cols != g.b_v->desc.logical_shape.cols) {
            if (err) *err = "group " + std::to_string(g.group_index) + " rank dimension mismatch between A and B";
            return false;
        }

        // Validate factor pair compatibility
        if (!validate_factor_pair_compatibility(g.a_k.desc, g.b_k->desc, err)) {
            return false;
        }
        if (!validate_factor_pair_compatibility(g.a_v.desc, g.b_v->desc, err)) {
            return false;
        }

        // In production profile, all streams must be Turbo
        if (candidate->profile == LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS ||
            candidate->profile == LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS_LANDMARKS) {
            if (!can_direct_rotated_gemm(g.a_k.desc, g.b_k->desc) &&
                (g.a_k.desc.stored_domain != value_domain::turbo_rotated || g.b_k->desc.stored_domain != value_domain::turbo_rotated)) {
                if (err) *err = "production TQ profiles require Turbo pairs for A_K and B_K";
                return false;
            }
            if (!can_direct_rotated_gemm(g.a_v.desc, g.b_v->desc) &&
                (g.a_v.desc.stored_domain != value_domain::turbo_rotated || g.b_v->desc.stored_domain != value_domain::turbo_rotated)) {
                if (err) *err = "production TQ profiles require Turbo pairs for A_V and B_V";
                return false;
            }
        }

        if (g.a_k.desc.role != factor_role::a_k || g.b_k->desc.role != factor_role::b_k ||
            g.a_v.desc.role != factor_role::a_v || g.b_v->desc.role != factor_role::b_v) {
            if (err) *err = "group " + std::to_string(g.group_index) + " factor stream role mismatch";
            return false;
        }

        if (g.a_k.desc.orient != orientation::token_major || g.b_k->desc.orient != orientation::feature_major_transposed ||
            g.a_v.desc.orient != orientation::token_major || g.b_v->desc.orient != orientation::feature_major_transposed) {
            if (err) *err = "group " + std::to_string(g.group_index) + " orientation mismatch (A must be token_major, B feature_major_transposed)";
            return false;
        }

        if (g.a_k.desc.logical_shape.rows != bundle_rows || g.a_v.desc.logical_shape.rows != bundle_rows) {
            if (err) *err = "group " + std::to_string(g.group_index) + " row count mismatch with bundle row count";
            return false;
        }

        try {
            // Residency rule (bundle-scope state hoisted above the loop):
            // host streams demand exact host bytes; device streams demand
            // empty host bytes and record a cover requirement below. No mixing.
            auto check_stream = [&](const std::vector<uint8_t> & bytes, const codec_desc & desc,
                                    const char * name) -> bool {
                const size_t expect = encoded_matrix_bytes(desc);
                if (!is_device_bundle) {
                    if (bytes.size() != expect) {
                        if (err) *err = "group " + std::to_string(g.group_index) + " " +
                                         name + " byte size mismatch";
                        return false;
                    }
                    return true;
                }
                if (!bytes.empty()) {
                    if (err) *err = "group " + std::to_string(g.group_index) + " " +
                                     name + " device residency forbids host bytes";
                    return false;
                }
                device_needed.push_back({desc, name, g.group_index});
                return true;
            };
            if (!check_stream(g.a_k.bytes, g.a_k.desc, "a_k")) return false;
            if (!check_stream(g.b_k->bytes, g.b_k->desc, "b_k")) return false;
            if (!check_stream(g.a_v.bytes, g.a_v.desc, "a_v")) return false;
            if (!check_stream(g.b_v->bytes, g.b_v->desc, "b_v")) return false;
            if (candidate->profile == LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS_LANDMARKS) {
                if (g.landmark.desc.logical_shape.rows == 0) {
                    if (err) *err = "group " + std::to_string(g.group_index) + " missing required landmark stream";
                    return false;
                }
                if (!g.landmark.desc.validate(err)) return false;
                if (g.landmark.desc.role != factor_role::landmark) {
                    if (err) *err = "group " + std::to_string(g.group_index) + " landmark role mismatch";
                    return false;
                }
                if (g.landmark.desc.orient != orientation::token_major) {
                    if (err) *err = "group " + std::to_string(g.group_index) + " landmark must be token_major orientation";
                    return false;
                }
                if (g.landmark.desc.logical_shape.cols != g.total_dim_k) {
                    if (err) *err = "group " + std::to_string(g.group_index) + " landmark logical cols (" +
                                     std::to_string(g.landmark.desc.logical_shape.cols) + ") != total_dim_k (" +
                                     std::to_string(g.total_dim_k) + ")";
                    return false;
                }
                if (g.landmark.desc.type != GGML_TYPE_Q8_0 && g.landmark.desc.type != GGML_TYPE_TURBO4_0) {
                    if (err) *err = "group " + std::to_string(g.group_index) + " production TQ_FACTORS_LANDMARKS requires Q8_0 or Turbo4 landmark";
                    return false;
                }
                if (!check_stream(g.landmark.bytes, g.landmark.desc, "landmark")) return false;
            } else if (candidate->profile == LLAMA_XKV_STORAGE_PROFILE_REFERENCE) {
                if (g.landmark.desc.logical_shape.rows > 0) {
                    if (!g.landmark.desc.validate(err)) return false;
                    if (g.landmark.desc.role != factor_role::landmark) {
                        if (err) *err = "group " + std::to_string(g.group_index) + " landmark role mismatch";
                        return false;
                    }
                    if (g.landmark.desc.orient != orientation::token_major) {
                        if (err) *err = "group " + std::to_string(g.group_index) + " landmark must be token_major orientation";
                        return false;
                    }
                    if (g.landmark.desc.logical_shape.cols != g.total_dim_k) {
                        if (err) *err = "group " + std::to_string(g.group_index) + " landmark logical cols (" +
                                         std::to_string(g.landmark.desc.logical_shape.cols) + ") != total_dim_k (" +
                                         std::to_string(g.total_dim_k) + ")";
                        return false;
                    }
                    if (g.landmark.desc.type != GGML_TYPE_F32 && g.landmark.desc.type != GGML_TYPE_F16 &&
                        g.landmark.desc.type != GGML_TYPE_Q8_0 && g.landmark.desc.type != GGML_TYPE_TURBO4_0) {
                        if (err) *err = "group " + std::to_string(g.group_index) + " unsupported landmark type for reference profile";
                        return false;
                    }
                    if (!check_stream(g.landmark.bytes, g.landmark.desc, "landmark")) return false;
                } else if (!g.landmark.bytes.empty()) {
                    if (err) *err = "group " + std::to_string(g.group_index) + " landmark bytes without landmark descriptor";
                    return false;
                }
            }
            // Landmark error-bounds table: required exactly when a landmark
            // stream is present (any profile), canonical 8-stride layout.
            // Bounds travel atomically with landmarks through seal/pack/import.
            // Presence is descriptor-based so device bundles (released host
            // bytes) validate identically to host bundles.
            const bool has_lm_stream = g.landmark.desc.logical_shape.rows > 0;
            if (has_lm_stream) {
                if (g.landmark_chunks.empty() || g.landmark_chunks.size() != g.landmark.desc.logical_shape.rows) {
                    if (err) *err = "group " + std::to_string(g.group_index) + " landmark chunk count mismatch with landmark stream rows";
                    return false;
                }
                uint32_t expected_offset = 0;
                for (size_t ci = 0; ci < g.landmark_chunks.size(); ++ci) {
                    const auto & ch = g.landmark_chunks[ci];
                    if (ch.row_count < 1) {
                        if (err) *err = "group " + std::to_string(g.group_index) + " landmark chunk has zero row_count";
                        return false;
                    }
                    if (ch.row_begin != expected_offset) {
                        if (err) *err = "group " + std::to_string(g.group_index) + " landmark chunks not contiguous or not starting at 0";
                        return false;
                    }
                    if (!std::isfinite(ch.error_bound) || ch.error_bound < 0.0f) {
                        if (err) *err = "group " + std::to_string(g.group_index) + " landmark chunk error_bound not finite/nonnegative";
                        return false;
                    }
                    if (ch.source_fingerprint == 0) {
                        if (err) *err = "group " + std::to_string(g.group_index) + " landmark chunk source fingerprint missing";
                        return false;
                    }
                    expected_offset += ch.row_count;
                }
                if (expected_offset != bundle_rows) {
                    if (err) *err = "group " + std::to_string(g.group_index) + " sum of landmark chunk row_counts (" +
                                     std::to_string(expected_offset) + ") != bundle rows (" + std::to_string(bundle_rows) + ")";
                    return false;
                }
                if (g.landmark_table_fingerprint != compute_landmark_table_fingerprint(g.landmark_chunks)) {
                    if (err) *err = "group " + std::to_string(g.group_index) + " landmark bounds fingerprint mismatch";
                    return false;
                }
            } else if (!g.landmark_chunks.empty() || g.landmark_table_fingerprint != 0) {
                if (err) *err = "group " + std::to_string(g.group_index) + " landmark bounds without landmark stream";
                return false;
            }
        } catch (const std::exception & e) {
            if (err) *err = std::string("group descriptor byte error: ") + e.what();
            return false;
        }
    }

    // Exact global device cover, enforced once across the whole candidate.
    // Every device-recorded stream consumes exactly one handle allocation
    // with matching descriptor fingerprint and exact byte size; every bundle
    // allocation must be device-owned, single-backend, and accounted.
    if (!device_needed.empty()) {
        try {
            std::vector<char> used(backend_bundle->handles.size(), 0);
            for (const auto & nd : device_needed) {
                const size_t expect = encoded_matrix_bytes(nd.desc);
                bool hit = false;
                for (size_t i = 0; i < backend_bundle->handles.size(); ++i) {
                    const auto & h = backend_bundle->handles[i];
                    if (used[i] || !h) continue;
                    if (h->get_descriptor_fingerprint() == nd.desc.fingerprint() &&
                        h->get_actual_bytes() == expect) {
                        used[i] = 1;
                        hit = true;
                        break;
                    }
                }
                if (!hit) {
                    if (err) *err = "group " + std::to_string(nd.group_index) + " " +
                                     nd.name + " not covered by backend bundle";
                    return false;
                }
            }
            std::string backend_name;
            bool have_name = false;
            for (size_t i = 0; i < backend_bundle->handles.size(); ++i) {
                const auto & h = backend_bundle->handles[i];
                if (!h) {
                    if (err) *err = "candidate null backend allocation in bundle";
                    return false;
                }
                if (h->get_residency() != GGML_XKV_RES_DEVICE_OWNED) {
                    if (err) *err = "candidate backend allocation not device-owned";
                    return false;
                }
                const std::string & nm = h->get_owning_backend_name();
                if (!have_name) {
                    backend_name = nm;
                    have_name = true;
                } else if (nm != backend_name) {
                    if (err) *err = "candidate backend bundle spans multiple backends";
                    return false;
                }
                if (!used[i]) {
                    if (err) *err = "candidate unaccounted backend allocation in bundle";
                    return false;
                }
            }
        } catch (const std::exception & e) {
            if (err) *err = std::string("device cover verification failed: ") + e.what();
            return false;
        } catch (...) {
            if (err) *err = "device cover verification failed with an unknown exception";
            return false;
        }
    }

    return true;
}

bool estimate_segment_bundle_persistent_bytes(
    const std::vector<xkv_factor_group_input> & group_inputs,
    const xkv_bundle_sealing_params & params,
    size_t * out_bytes,
    std::string * err,
    size_t * out_encoded_stream_bytes
) {
    if (out_bytes == nullptr) {
        if (err) *err = "estimate_segment_bundle_persistent_bytes: null out_bytes";
        return false;
    }
    *out_bytes = 0;
    if (out_encoded_stream_bytes) *out_encoded_stream_bytes = 0;
    if (group_inputs.empty()) {
        return true;
    }
    if (params.chunk_tokens == 0) {
        if (err) *err = "estimate_segment_bundle_persistent_bytes: chunk_tokens must be greater than zero";
        return false;
    }
    const uint64_t n_rows = group_inputs[0].k_rows;
    if (n_rows == 0 || n_rows > (uint64_t) std::numeric_limits<uint32_t>::max()) {
        if (err) *err = "estimate_segment_bundle_persistent_bytes: invalid row count";
        return false;
    }

    auto is_turbo = [](ggml_type t) {
        return t == GGML_TYPE_TURBO2_0 || t == GGML_TYPE_TURBO3_0 || t == GGML_TYPE_TURBO4_0;
    };

    // Segment metadata, mirroring xkv_segment::update_byte_counters term by
    // term (size-based, no object overhead): row payload IDs + live mask +
    // baseline. Must stay identical to the counter or the publish capacity
    // check trips on phantom bytes.
    // Conservative live-mask word bound ((n+63)/64*8), safe for any stdlib word size <= 64 bits.
    const uint64_t live_mask_bytes = ((n_rows + 63) / 64) * 8;
    uint64_t seg_meta = 0;
    if (!checked_add_u64(n_rows * sizeof(uint64_t), live_mask_bytes, seg_meta) ||
        !checked_add_u64(seg_meta, sizeof(uint64_t), seg_meta)) {
        if (err) *err = "estimate_segment_bundle_persistent_bytes: segment metadata overflow";
        return false;
    }

    uint64_t total = seg_meta;
    uint64_t stream_total = 0;
    for (const auto & g_in : group_inputs) {
        if (g_in.k_rows != n_rows || g_in.v_rows != n_rows || g_in.k_cols == 0 || g_in.v_cols == 0) {
            if (err) *err = "estimate_segment_bundle_persistent_bytes: matrix dimension mismatch";
            return false;
        }
        const uint32_t grp_k = is_turbo(params.factor_a_k) ? 128 : 0;
        const uint32_t grp_v = is_turbo(params.factor_a_v) ? 128 : 0;
        const uint64_t fseed = params.seed + (uint64_t) g_in.group_index * 1000ULL;
        uint64_t enc = 0;
        uint64_t stream_bytes = 0;
        try {
            codec_desc d_ak = make_codec_desc(factor_role::a_k, params.factor_a_k,
                orientation::token_major, matrix_shape{n_rows, g_in.rank_k}, grp_k, fseed);
            codec_desc d_bk = make_codec_desc(factor_role::b_k, params.factor_b_k,
                orientation::feature_major_transposed, matrix_shape{g_in.k_cols, g_in.rank_k}, grp_k, fseed);
            codec_desc d_av = make_codec_desc(factor_role::a_v, params.factor_a_v,
                orientation::token_major, matrix_shape{n_rows, g_in.rank_v}, grp_v, fseed + 1);
            codec_desc d_bv = make_codec_desc(factor_role::b_v, params.factor_b_v,
                orientation::feature_major_transposed, matrix_shape{g_in.v_cols, g_in.rank_v}, grp_v, fseed + 1);
            uint64_t parts[4] = {encoded_matrix_bytes(d_ak), encoded_matrix_bytes(d_bk),
                                 encoded_matrix_bytes(d_av), encoded_matrix_bytes(d_bv)};
            for (uint64_t p : parts) {
                if (!checked_add_u64(enc, p, enc) ||
                    !checked_add_u64(stream_bytes, p, stream_bytes)) {
                    if (err) *err = "estimate_segment_bundle_persistent_bytes: stream size overflow";
                    return false;
                }
            }
            // Landmarks are required when profile != TQ_FACTORS (including TQ_FACTORS_LANDMARKS and REFERENCE when configured)
            const bool want_lm = (params.profile == LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS_LANDMARKS) ||
                                 (params.profile == LLAMA_XKV_STORAGE_PROFILE_REFERENCE && params.landmark_factory != nullptr);
            if (want_lm) {
                const uint32_t chunk_toks = params.chunk_tokens;
                const uint64_t chunks = (n_rows + chunk_toks - 1) / chunk_toks;
                codec_desc d_lm = make_codec_desc(factor_role::landmark, params.landmark_type,
                    orientation::token_major, matrix_shape{chunks, g_in.total_dim_k}, 0, 777);
                uint64_t lm = encoded_matrix_bytes(d_lm);
                if (!checked_add_u64(enc, lm, enc) ||
                    !checked_add_u64(stream_bytes, lm, stream_bytes)) {
                    if (err) *err = "estimate_segment_bundle_persistent_bytes: landmark size overflow";
                    return false;
                }
                // Add explicit chunk metadata table bytes (row_begin, row_count, error_bound)
                uint64_t chunk_table_bytes = chunks * sizeof(xkv_landmark_chunk);
                if (!checked_add_u64(enc, chunk_table_bytes, enc)) {
                    if (err) *err = "estimate_segment_bundle_persistent_bytes: landmark size overflow";
                    return false;
                }
            }
            // Group metadata, mirroring xkv_factor_group_payload::update_byte_counters
            // term by term (size-based, no object overhead): owning/offsets/dims
            // vectors + baselines + table fingerprint. Must stay identical to
            // the counter or the publish capacity check trips on phantom bytes.
            uint64_t meta = 0;
            uint64_t mparts[6] = {
                (uint64_t) g_in.owning_layers.size() * sizeof(uint32_t),
                (uint64_t) g_in.layer_feature_offsets_k.size() * sizeof(uint32_t),
                (uint64_t) g_in.layer_feature_dims_k.size() * sizeof(uint32_t),
                (uint64_t) g_in.layer_feature_offsets_v.size() * sizeof(uint32_t),
                (uint64_t) g_in.layer_feature_dims_v.size() * sizeof(uint32_t),
                sizeof(uint64_t) * 2 + sizeof(uint64_t) // baseline_original_row_bytes, baseline_original_bytes, landmark_table_fingerprint
            };
            for (uint64_t p : mparts) {
                if (!checked_add_u64(meta, p, meta)) {
                    if (err) *err = "estimate_segment_bundle_persistent_bytes: metadata overflow";
                    return false;
                }
            }
            if (!checked_add_u64(enc, meta, enc) ||
                !checked_add_u64(total, enc, total) ||
                !checked_add_u64(stream_total, stream_bytes, stream_total)) {
                if (err) *err = "estimate_segment_bundle_persistent_bytes: group size overflow";
                return false;
            }
        } catch (const std::exception & e) {
            if (err) *err = std::string("estimate_segment_bundle_persistent_bytes: ") + e.what();
            return false;
        } catch (...) {
            if (err) *err = "estimate_segment_bundle_persistent_bytes: unknown exception";
            return false;
        }
    }

    if (total > (uint64_t) std::numeric_limits<size_t>::max()) {
        if (err) *err = "estimate_segment_bundle_persistent_bytes: total size exceeds size_t limit";
        return false;
    }
    *out_bytes = (size_t) total;
    if (out_encoded_stream_bytes) *out_encoded_stream_bytes = (size_t) stream_total;
    return true;
}

bool llama_xkv_cache_store::publish_candidate(
    std::shared_ptr<xkv_segment> candidate,
    const std::vector<uint64_t> & payload_ids,
    const std::vector<uint64_t> & generations,
    std::string * err,
    xkv_landmark_rebuild_fn landmark_rebuild,
    xkv_seal_precommit_fn precommit,
    xkv_capacity_reservation * capacity_reservation
) {
    if (!validate_candidate(candidate, err)) {
        return false;
    }

    const uint64_t n_rows_64 = candidate->groups[0].a_k.desc.logical_shape.rows;
    if (n_rows_64 > std::numeric_limits<uint32_t>::max() || n_rows_64 == 0) {
        if (err) *err = "invalid row count for publish";
        return false;
    }
    const uint32_t n_rows = (uint32_t) n_rows_64;

    if (payload_ids.size() != n_rows || generations.size() != n_rows) {
        if (err) *err = "payload_ids or generations size must match bundle row count";
        return false;
    }

    std::unordered_set<uint64_t> seen_ids;
    seen_ids.reserve(n_rows);
    for (uint32_t i = 0; i < n_rows; ++i) {
        if (payload_ids[i] == 0) {
            if (err) *err = "payload_id cannot be 0";
            return false;
        }
        if (!seen_ids.insert(payload_ids[i]).second) {
            if (err) *err = "duplicate payload_id in candidate";
            return false;
        }
        if (generations[i] == 0) {
            if (err) *err = "generation cannot be 0";
            return false;
        }
    }

    // Two-phase publish:
    struct bundle_replacement_plan {
        uint64_t seg_id = 0;
        std::shared_ptr<const xkv_segment> old_seg;
        std::shared_ptr<const xkv_segment> new_seg;
        std::vector<std::pair<uint64_t, uint32_t>> surviving_payload_rows;
    };
    std::vector<bundle_replacement_plan> replacement_plans;

    {
        std::lock_guard<std::mutex> lock(mtx);

        for (uint32_t row = 0; row < n_rows; ++row) {
            const uint64_t pid = payload_ids[row];
            auto it = payload_locations.find(pid);
            if (it == payload_locations.end()) {
                if (err) *err = "payload " + std::to_string(pid) + " was concurrently evicted before publish";
                return false;
            }
            if (it->second.state != xkv_state::seal_candidate) {
                if (err) *err = "payload " + std::to_string(pid) + " not in seal_candidate state";
                return false;
            }
            if (it->second.storage_generation != generations[row]) {
                if (err) *err = "payload " + std::to_string(pid) + " generation changed concurrently";
                return false;
            }
        }

        std::unordered_set<uint64_t> affected_seg_ids;
        for (uint32_t row = 0; row < n_rows; ++row) {
            const uint64_t pid = payload_ids[row];
            auto it = payload_locations.find(pid);
            if (it != payload_locations.end() && it->second.kind == xkv_location_kind::factored && it->second.segment_id > 0) {
                affected_seg_ids.insert(it->second.segment_id);
            }
        }

        for (uint64_t aff_id : affected_seg_ids) {
            auto seg_it = published_segments.find(aff_id);
            if (seg_it == published_segments.end()) continue;
            bundle_replacement_plan plan;
            plan.seg_id = aff_id;
            plan.old_seg = seg_it->second;
            replacement_plans.push_back(plan);
        }
    }

    try {
        for (auto & plan : replacement_plans) {
            auto old_seg = plan.old_seg;
            std::vector<uint32_t> surviving_rows;
            for (uint32_t r = 0; r < old_seg->n_rows; ++r) {
                if (old_seg->live_rows[r] && seen_ids.find(old_seg->row_payload_ids[r]) == seen_ids.end()) {
                    surviving_rows.push_back(r);
                }
            }

            if (!surviving_rows.empty()) {
                const uint32_t new_n_rows = (uint32_t) surviving_rows.size();

                auto new_seg = std::make_shared<xkv_segment>();
                new_seg->segment_id = old_seg->segment_id;
                new_seg->segment_version = old_seg->segment_version + 1;
                new_seg->profile = old_seg->profile;
                new_seg->source = old_seg->source;
                new_seg->residency = old_seg->residency;
                new_seg->layer_group_map_fingerprint = old_seg->layer_group_map_fingerprint;
                new_seg->profile_fingerprint = old_seg->profile_fingerprint;
                new_seg->source_fingerprint = old_seg->source_fingerprint;
                // Fail before any host decode/memcpy: DEVICE_OWNED displacement packs
                // only via backend-native transactions owned outside the store.
                if (old_seg->residency == GGML_XKV_RES_DEVICE_OWNED) {
                    throw std::runtime_error(
                        "segment " + std::to_string(old_seg->segment_id) +
                        " is DEVICE_OWNED: host publish displacement forbidden; requires backend-native byte-preserving pack (no host decode/memcpy)");
                }
                new_seg->backend_bundle.reset(); // COW resets; host stays bundle-free.
                const bool is_landmarks = segment_needs_landmark_rebuild(*old_seg);

                if (is_landmarks && !landmark_rebuild) {
                    // Refusal path: never copy first-N landmark chunks after arbitrary
                    // displacement, and never duplicate the layout (byte budget).
                    // Without a semantic rebuild callback the displacement refuses
                    // with zero mutation; the old version stays valid and published.
                    throw std::runtime_error(
                        "LANDMARKS segment " + std::to_string(old_seg->segment_id) +
                        " requires a semantic landmark rebuild callback for publish displacement");
                } else {
                    new_seg->groups.resize(old_seg->groups.size());

                    for (size_t g_idx = 0; g_idx < old_seg->groups.size(); ++g_idx) {
                        const auto & old_g = old_seg->groups[g_idx];
                        auto & new_g = new_seg->groups[g_idx];

                        new_g.group_index = old_g.group_index;
                        new_g.owning_layers = old_g.owning_layers;
                        new_g.rank_k = old_g.rank_k;
                        new_g.rank_v = old_g.rank_v;
                        new_g.layer_feature_offsets_k = old_g.layer_feature_offsets_k;
                        new_g.layer_feature_dims_k = old_g.layer_feature_dims_k;
                        new_g.layer_feature_offsets_v = old_g.layer_feature_offsets_v;
                        new_g.layer_feature_dims_v = old_g.layer_feature_dims_v;
                        new_g.total_dim_k = old_g.total_dim_k;
                        new_g.total_dim_v = old_g.total_dim_v;
                        new_g.config_fingerprint = old_g.config_fingerprint;
                        // Immutable per-row baseline travels verbatim; the total is
                        // recomputed below for the survivor row count (checked).
                        new_g.baseline_original_row_bytes = old_g.baseline_original_row_bytes;
                        new_g.b_k = old_g.b_k; // Share immutable B handle across versions
                        new_g.b_v = old_g.b_v;

                        const size_t stride_a_k = old_g.a_k.desc.row_stride_bytes;
                        const size_t stride_a_v = old_g.a_v.desc.row_stride_bytes;

                        if (stride_a_k == 0 || stride_a_v == 0) {
                            throw std::runtime_error("segment " + std::to_string(old_seg->segment_id) +
                                " group " + std::to_string(old_g.group_index) + " has zero row stride; host publish displacement forbidden");
                        }
                        if (old_g.a_k.bytes.size() != (size_t) old_seg->n_rows * stride_a_k ||
                            old_g.a_v.bytes.size() != (size_t) old_seg->n_rows * stride_a_v) {
                            throw std::runtime_error("segment " + std::to_string(old_seg->segment_id) +
                                " group " + std::to_string(old_g.group_index) + " host byte size mismatch; host publish displacement forbidden");
                        }

                        new_g.a_k.desc = old_g.a_k.desc;
                        new_g.a_k.desc.logical_shape.rows = new_n_rows;
                        new_g.a_k.desc.padded_shape.rows = new_n_rows;
                        new_g.a_k.bytes.resize(new_n_rows * stride_a_k);

                        new_g.a_v.desc = old_g.a_v.desc;
                        new_g.a_v.desc.logical_shape.rows = new_n_rows;
                        new_g.a_v.desc.padded_shape.rows = new_n_rows;
                        new_g.a_v.bytes.resize(new_n_rows * stride_a_v);

                        for (uint32_t dst_r = 0; dst_r < new_n_rows; ++dst_r) {
                            uint32_t src_r = surviving_rows[dst_r];
                            if (src_r >= old_seg->n_rows) {
                                throw std::runtime_error("segment " + std::to_string(old_seg->segment_id) +
                                    " surviving row out of range; host publish displacement forbidden");
                            }
                            std::memcpy(new_g.a_k.bytes.data() + dst_r * stride_a_k,
                                        old_g.a_k.bytes.data() + src_r * stride_a_k,
                                        stride_a_k);
                            std::memcpy(new_g.a_v.bytes.data() + dst_r * stride_a_v,
                                        old_g.a_v.bytes.data() + src_r * stride_a_v,
                                        stride_a_v);
                        }
                        new_g.landmark = encoded_matrix();
                        uint64_t packed_total = 0;
                        if (!checked_mul_u64((uint64_t) new_g.baseline_original_row_bytes,
                                             (uint64_t) new_n_rows, packed_total)) {
                            throw std::runtime_error("baseline byte total overflow for group " +
                                std::to_string(new_g.group_index));
                        }
                        new_g.baseline_original_bytes = packed_total;
                    }

                    new_seg->row_payload_ids.reserve(new_n_rows);
                    new_seg->live_rows.assign(new_n_rows, true);
                    new_seg->n_rows = new_n_rows;
                    new_seg->n_live_rows = new_n_rows;

                    for (uint32_t dst_r = 0; dst_r < new_n_rows; ++dst_r) {
                        uint32_t src_r = surviving_rows[dst_r];
                        uint64_t spid = old_seg->row_payload_ids[src_r];
                        new_seg->row_payload_ids.push_back(spid);
                        plan.surviving_payload_rows.push_back({spid, dst_r});
                    }

                    if (is_landmarks) {
                        std::string rebuild_err;
                        if (!landmark_rebuild(*old_seg, surviving_rows, *new_seg, &rebuild_err)) {
                            throw std::runtime_error(
                                "landmark rebuild refused publish displacement for segment " +
                                std::to_string(old_seg->segment_id) + ": " + rebuild_err);
                        }
                    }
                    for (auto & new_g : new_seg->groups) {
                        new_g.refresh_descriptor_fingerprint();
                        new_g.update_byte_counters();
                    }
                    new_seg->update_byte_counters();
                    {
                        std::string cerr;
                        if (!validate_candidate(new_seg, &cerr)) {
                            throw std::runtime_error(
                                "rebuilt LANDMARKS bundle failed validation: " + cerr);
                        }
                    }
                    plan.new_seg = new_seg;
                }
            }
        }

        // Freeze candidate bundle
        candidate->n_rows = n_rows;
        candidate->n_live_rows = n_rows;
        candidate->live_rows.assign(n_rows, true);
        candidate->row_payload_ids = payload_ids;
        for (auto & g : candidate->groups) {
            g.refresh_descriptor_fingerprint();
            g.update_byte_counters();
        }
        candidate->update_byte_counters();
    } catch (const std::exception & e) {
        if (err) *err = std::string("candidate bundle cloning failed: ") + e.what();
        return false;
    } catch (...) {
        if (err) *err = "candidate bundle cloning failed with an unknown exception";
        return false;
    }

    // Phase 2: Commit under lock
    std::shared_ptr<const xkv_segment> const_candidate = candidate;

    // Epoch reservation discipline: refused while held (publication never
    // carries a token; the coordinator seals outside reservations).
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (check_reservation_locked(0)) {
            if (err) *err = "publish_candidate: epoch reservation held by another transaction";
            return false;
        }
    }

    // Preallocate node handle for published_segments before entering lock to avoid allocation failure under lock
    std::unordered_map<uint64_t, std::shared_ptr<const xkv_segment>> prealloc_map;
    prealloc_map.emplace(candidate->segment_id, const_candidate);
    auto prealloc_node = prealloc_map.extract(candidate->segment_id);
    // Precommit context payload/generation copies built off-lock; only the
    // small physical-row snapshot is taken under lock below.
    xkv_seal_precommit_ctx pre_ctx;
    if (precommit) {
        pre_ctx.segment_id = candidate->segment_id;
        pre_ctx.segment_version = candidate->segment_version;
        pre_ctx.payload_ids = payload_ids;
        pre_ctx.generations = generations;
        pre_ctx.candidate = candidate;
    }

    {
        std::lock_guard<std::mutex> lock(mtx);

        uint64_t commit_nonce = 0;
        for (uint32_t row = 0; row < n_rows; ++row) {
            const uint64_t pid = payload_ids[row];
            auto it = payload_locations.find(pid);
            if (it == payload_locations.end() || it->second.state != xkv_state::seal_candidate || it->second.storage_generation != generations[row]) {
                if (err) *err = "payload state or generation changed concurrently before commit";
                return false;
            }
            // Nonce unanimity: every row of the batch must carry the SAME
            // nonzero seal transaction nonce (one atomic mark). Mixed nonces
            // mean overlapping seals were stitched together: refuse.
            if (it->second.seal_tx_nonce == 0) {
                if (err) *err = "payload " + std::to_string(pid) + " has no seal transaction nonce";
                return false;
            }
            if (row == 0) {
                commit_nonce = it->second.seal_tx_nonce;
            } else if (it->second.seal_tx_nonce != commit_nonce) {
                if (err) *err = "payload seal transaction nonces disagree across batch (overlapping seals)";
                return false;
            }
        }

        for (const auto & plan : replacement_plans) {
            auto seg_it = published_segments.find(plan.seg_id);
            if (seg_it == published_segments.end() || seg_it->second != plan.old_seg) {
                if (err) *err = "concurrent mutation conflict during bundle publish";
                return false;
            }
        }

        // Preflight can_bump BEFORE committing published candidate
        uint8_t pub_mask = bump_flag_content | bump_flag_binding;
        if (!can_bump_locked(pub_mask, err)) {
            return false;
        }

        // sealed_count must not wrap either: publication bumps it once below.
        if (sealed_count == std::numeric_limits<size_t>::max()) {
            if (err) *err = "xkv sealed_count overflow";
            return false;
        }

        // Caught allocation preflight: every reserving operation (retired
        // capacity, bucket rehash, precommit row snapshot) runs here, before the
        // gate. Anything that throws aborts with zero mutation.
        try {
            retired_segments.reserve(retired_segments.size() + replacement_plans.size());
            published_segments.rehash(published_segments.size() + 1);

            // Track host/device peaks at publish commit:
            size_t commit_dev = 0;
            size_t commit_host = 0;
            if (candidate->residency == GGML_XKV_RES_DEVICE_OWNED) commit_dev += candidate->total_allocated_bytes;
            else commit_host += candidate->total_allocated_bytes;
            for (const auto & [_, seg] : published_segments) {
            if (seg->residency == GGML_XKV_RES_DEVICE_OWNED) commit_dev += seg->total_allocated_bytes;
            else commit_host += seg->total_allocated_bytes;
            }
            for (const auto & seg : retired_segments) {
            if (seg->residency == GGML_XKV_RES_DEVICE_OWNED) commit_dev += seg->total_allocated_bytes;
            else commit_host += seg->total_allocated_bytes;
            }
            if (commit_host > host_peak_bytes_) host_peak_bytes_ = commit_host;
            if (commit_dev > device_peak_bytes_) device_peak_bytes_ = commit_dev;
            if (precommit) {
                pre_ctx.physical_rows.reserve(n_rows);
                for (uint32_t row = 0; row < n_rows; ++row) {
                    auto it = payload_locations.find(payload_ids[row]);
                    pre_ctx.physical_rows.push_back(it != payload_locations.end() ? it->second.row : 0);
                }
            }
            // Hard store cap: exact deduplicated peak (live published +
            // retired + this candidate + replacement versions, shared B once)
            // must fit before any mutation. Zero mutation on refusal.
            {
                std::vector<std::shared_ptr<const xkv_segment>> cap_extras;
                cap_extras.reserve(replacement_plans.size() + 1);
                cap_extras.push_back(const_candidate);
                for (const auto & plan : replacement_plans) {
                    if (plan.new_seg) cap_extras.push_back(plan.new_seg);
                }
                if (!ensure_accounting_scratch_locked(cap_extras, err)) return false;

                size_t exclude_bytes = 0;
                // Reconcile capacity reservation if provided: actual <= reserved.
                // Null pointer means no reservation provided (optional path);
                // NON-NULL pointer MUST be valid, live, and match this store!
                if (capacity_reservation != nullptr) {
                    if (!capacity_reservation->valid()) {
                        if (err) *err = "publish store capacity check failed: stale, released, or invalid capacity reservation token";
                        return false;
                    }
                    if (capacity_reservation->store_ != this) {
                        if (err) *err = "publish store capacity check failed: foreign reservation token from another store";
                        return false;
                    }
                    auto it_v = pending_reservations_.find(capacity_reservation->token());
                    if (it_v == pending_reservations_.end()) {
                        if (err) *err = "publish store capacity check failed: stale or unknown reservation token";
                        return false;
                    }
                    if (it_v->second != capacity_reservation->reserved_bytes()) {
                        if (err) *err = "publish store capacity check failed: reservation bytes mismatch";
                        return false;
                    }

                    // Exact deduplicated incremental candidate/replacements delta
                    size_t base_alloc = 0;
                    std::vector<std::shared_ptr<const xkv_segment>> none;
                    if (!deduplicated_allocated_locked(none, base_alloc)) {
                        if (err) *err = "publish store capacity check failed: base accounting overflow";
                        return false;
                    }
                    size_t peak_alloc = 0;
                    if (!deduplicated_allocated_locked(cap_extras, peak_alloc)) {
                        if (err) *err = "publish store capacity check failed: peak accounting overflow";
                        return false;
                    }
                    size_t incremental_actual = (peak_alloc >= base_alloc) ? (peak_alloc - base_alloc) : 0;

                    size_t reserved_now = capacity_reservation->reserved_bytes();
                    if (incremental_actual > reserved_now) {
                        if (err) *err = "publish store capacity check failed: actual incremental bytes (" +
                                         std::to_string(incremental_actual) +
                                         ") exceed reserved bytes (" +
                                         std::to_string(capacity_reservation->reserved_bytes()) + ")";
                        return false;
                    }
                    exclude_bytes = reserved_now;
                }

                if (!store_capacity_fits_locked(cap_extras, err, exclude_bytes)) {
                    if (err && err->find("store capacity") == std::string::npos) {
                        *err = std::string("publish store capacity check failed: ") + *err;
                }
                    return false;
                }
            }
        } catch (const std::exception & e) {
            if (err) *err = std::string("publish preflight allocation failed: ") + e.what();
            return false;
        }

        // Precommit gate: every allocation above is complete. A false return or
        // throw refuses with zero store mutation; after true the commit below
        // performs no allocation and cannot fail (node insert without rehash,
        // in-capacity pushes, existing-value assignments, counter bumps).
        // The callback MUST NOT re-enter the store (lock held).
        if (precommit) {
            std::string pre_err;
            bool pre_ok = false;
            try {
                pre_ok = precommit(pre_ctx, &pre_err);
            } catch (const std::exception & e) {
                pre_err = e.what();
                pre_ok = false;
            } catch (...) {
                pre_err = "unknown non-standard exception";
                pre_ok = false;
            }
            if (!pre_ok) {
                if (err) *err = "seal precommit refused publish: " + pre_err;
                return false;
            }
        }

        // Insert new candidate via preallocated node handle (guaranteed no allocation)
        published_segments.insert(std::move(prealloc_node));

        for (const auto & plan : replacement_plans) {
            auto seg_it = published_segments.find(plan.seg_id);
            if (plan.new_seg) {
                seg_it->second = plan.new_seg;
                for (const auto & pr : plan.surviving_payload_rows) {
                    auto it = payload_locations.find(pr.first);
                    if (it != payload_locations.end()) {
                        it->second.segment_version = plan.new_seg->segment_version;
                        it->second.row = pr.second;
                    }
                }
            } else {
                published_segments.erase(seg_it);
            }
            retired_segments.push_back(plan.old_seg);
        }

        for (uint32_t row = 0; row < n_rows; ++row) {
            const uint64_t pid = payload_ids[row];
            auto it = payload_locations.find(pid);
            assert(it != payload_locations.end());
            it->second.kind = xkv_location_kind::factored;
            it->second.segment_id = candidate->segment_id;
            it->second.segment_version = candidate->segment_version;
            it->second.row = row;
            it->second.state = xkv_state::factored;
            it->second.seal_tx_nonce = 0;
        }

        // Commit and consume capacity reservation upon successful publish
        if (capacity_reservation != nullptr && capacity_reservation->valid()) {
            auto it_r = pending_reservations_.find(capacity_reservation->token());
            if (it_r != pending_reservations_.end()) {
                if (pending_reserved_store_bytes_ >= it_r->second) {
                    pending_reserved_store_bytes_ -= it_r->second;
                } else {
                    pending_reserved_store_bytes_ = 0;
                }
                pending_reservations_.erase(it_r);
            }
            capacity_reservation->store_ = nullptr;
            capacity_reservation->token_ = 0;
            capacity_reservation->reserved_bytes_ = 0;
        }

        sealed_count++;
        bump_prechecked(pub_mask);
        reclaim_retired_segments_locked();
    }

    return true;
}

void llama_xkv_cache_store::abort_candidate(std::shared_ptr<xkv_segment> & candidate) {
    if (candidate) {
        for (auto & g : candidate->groups) {
            g.a_k.bytes.clear();
            g.b_k.reset();
            g.a_v.bytes.clear();
            g.b_v.reset();
            g.landmark.bytes.clear();
        }
        candidate->groups.clear();
        candidate.reset();
    }
}

xkv_sealing_result llama_xkv_cache_store::seal_segment_bundle(
    const std::vector<xkv_factor_group_input> & group_inputs,
    const std::vector<uint64_t> & payload_ids,
    const std::vector<uint64_t> & generations,
    const xkv_bundle_sealing_params & params,
    xkv_seal_precommit_fn precommit,
    xkv_capacity_reservation * capacity_reservation
) {
    xkv_sealing_result result;
    result.success = false;

    xkv_capacity_reservation seal_cap_res;
    if (capacity_reservation != nullptr && params.capacity_reservation != nullptr &&
        capacity_reservation != params.capacity_reservation) {
        result.skip_reason = xkv_skip_reason::unsupported_config;
        result.message = "seal_segment_bundle: conflicting capacity_reservation argument and params.capacity_reservation";
        return result;
    }
    xkv_capacity_reservation * effective_res = capacity_reservation;
    if (effective_res == nullptr && params.capacity_reservation != nullptr) {
        effective_res = params.capacity_reservation;
    }
    const bool caller_provided_res = (effective_res != nullptr);
    if (caller_provided_res) {
        if (!effective_res->valid()) {
            result.skip_reason = xkv_skip_reason::store_capacity_exceeded;
            result.message = "seal_segment_bundle: caller capacity reservation token is invalid or released";
            return result;
        }
        if (effective_res->store_ != this) {
            result.skip_reason = xkv_skip_reason::store_capacity_exceeded;
            result.message = "seal_segment_bundle: foreign reservation token from another store";
            return result;
        }
    } else {
        if (!params.evaluate_only) {
            effective_res = &seal_cap_res;
        } else {
            effective_res = nullptr;
        }
    }

    // Epoch reservation discipline: sealing never carries a token.
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (check_reservation_locked(0)) {
            result.skip_reason = xkv_skip_reason::aborted;
            result.message = "seal_segment_bundle: epoch reservation held by another transaction";
            return result;
        }
    }

    // Whole-body catch-all: validation containers, canonical matrices and
    // backend calls may allocate/throw; any escape rolls back exactly this
    // transaction's nonce (if marked) and reports aborted with zero partial
    // publication. The commit tail itself performs no allocation.
    // Transaction state hoisted ABOVE try: the catch handlers below must see
    // it (names declared inside try are invisible to handlers). Rollback
    // touches only this nonce and runs exactly once per failure path.
    uint64_t this_tx_nonce = 0;
    auto rollback_transaction = [&](xkv_skip_reason reason, const std::string & msg) {
        seal_cap_res.release();
        std::lock_guard<std::mutex> lock(mtx);
        if (this_tx_nonce == 0) {
            return;
        }
        for (size_t i = 0; i < payload_ids.size(); ++i) {
            auto it = payload_locations.find(payload_ids[i]);
            // Rollback ONLY matching tx nonce! Never cancel or reset another concurrent seal!
            if (it != payload_locations.end() && it->second.seal_tx_nonce == this_tx_nonce) {
                it->second.state = xkv_state::hot_committed;
                it->second.seal_tx_nonce = 0;
            }
        }
        result.skip_reason = reason;
        result.message = msg;
        if (reason != xkv_skip_reason::none) {
            skipped_counts[static_cast<uint8_t>(reason)]++;
        }
    };

    try {

    if (payload_ids.empty() || payload_ids.size() > std::numeric_limits<uint32_t>::max() ||
        group_inputs.empty() || generations.size() != payload_ids.size()) {
        result.skip_reason = xkv_skip_reason::unsupported_config;
        result.message = "empty group inputs or payload/generation count mismatch";
        return result;
    }
    const uint32_t n_rows = static_cast<uint32_t>(payload_ids.size());

    // Validate nonzero and unique payload IDs
    std::unordered_set<uint64_t> unique_pids;
    unique_pids.reserve(n_rows);
    for (uint32_t i = 0; i < n_rows; ++i) {
        if (payload_ids[i] == 0) {
            result.skip_reason = xkv_skip_reason::unsupported_config;
            result.message = "payload_id cannot be 0";
            return result;
        }
        if (!unique_pids.insert(payload_ids[i]).second) {
            result.skip_reason = xkv_skip_reason::unsupported_config;
            result.message = "duplicate payload_id in seal request";
            return result;
        }
        if (generations[i] == 0) {
            result.skip_reason = xkv_skip_reason::unsupported_config;
            result.message = "generation cannot be 0";
            return result;
        }
    }

    // Expected group count validation: reject omitted, out-of-order, or extra groups
    if (params.expected_group_count > 0 && group_inputs.size() != params.expected_group_count) {
        result.skip_reason = xkv_skip_reason::unsupported_config;
        result.message = "group_inputs count (" + std::to_string(group_inputs.size()) +
                         ") does not match expected_group_count (" + std::to_string(params.expected_group_count) + ")";
        return result;
    }

    // Strict group ordering check: group_index must be strictly increasing 0, 1, 2, ...
    for (size_t g_i = 0; g_i < group_inputs.size(); ++g_i) {
        if (group_inputs[g_i].group_index != g_i) {
            result.skip_reason = xkv_skip_reason::unsupported_config;
            result.message = "group_inputs out of order or discontinuous group_index (expected " +
                             std::to_string(g_i) + ", got " + std::to_string(group_inputs[g_i].group_index) + ")";
            return result;
        }
    }

    // Expected group map fingerprint validation
    uint64_t actual_map_fp = compute_layer_group_map_fingerprint(group_inputs);
    if (params.expected_group_map_fingerprint != 0 && actual_map_fp != params.expected_group_map_fingerprint) {
        result.skip_reason = xkv_skip_reason::unsupported_config;
        result.message = "group_inputs layer_group_map_fingerprint mismatch with expected_group_map_fingerprint";
        return result;
    }

    // Profile-level precondition: landmark profile strictly requires a landmark factory callback.
    // Check after structural group validation so malformed group batches reject with unsupported_config first.
    if (params.profile == LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS_LANDMARKS && !params.landmark_factory) {
        result.skip_reason = xkv_skip_reason::landmark_required;
        result.message = "landmark profile requires group landmark factory callback";
        skipped_counts[static_cast<uint8_t>(result.skip_reason)]++;
        return result;
    }

    // Verify all group inputs have matching row counts and valid non-null floats
    std::unordered_set<uint32_t> seen_groups;
    std::unordered_set<uint32_t> seen_layers;
    // Advisory upfront bound: max per-group phase peak (groups factorize
    // sequentially, never simultaneously). Exact leases come from real
    // estimator output per group below.
    uint64_t max_group_scratch = 0;

    for (const auto & g_in : group_inputs) {
        if (!seen_groups.insert(g_in.group_index).second) {
            result.skip_reason = xkv_skip_reason::unsupported_config;
            result.message = "duplicate group_index in group_inputs";
            return result;
        }
        if (g_in.owning_layers.empty()) {
            result.skip_reason = xkv_skip_reason::unsupported_config;
            result.message = "group has empty owning_layers";
            return result;
        }
        for (uint32_t l : g_in.owning_layers) {
            if (!seen_layers.insert(l).second) {
                result.skip_reason = xkv_skip_reason::unsupported_config;
                result.message = "layer appears in multiple group inputs";
                return result;
            }
        }
        if (g_in.k_rows != n_rows || g_in.v_rows != n_rows || g_in.k_cols == 0 || g_in.v_cols == 0 ||
            (!params.canonical_source && (!g_in.canonical_k_data || !g_in.canonical_v_data))) {
            result.skip_reason = xkv_skip_reason::unsupported_config;
            result.message = "group input matrix dimension or pointer invalid";
            return result;
        }
        if (g_in.row_positions.size() != n_rows) {
            result.skip_reason = xkv_skip_reason::unsupported_config;
            result.message = "group input row_positions must match payload row count";
            return result;
        }
        // Measured baseline identity (§2.2/8.1): per-layer hot-row bytes must
        // parallel owning_layers exactly; zero/missing entries never fall
        // back to theory (heterogeneous layers need exact accounting).
        if (g_in.hot_bytes_per_row_k.size() != g_in.owning_layers.size() ||
            g_in.hot_bytes_per_row_v.size() != g_in.owning_layers.size()) {
            result.skip_reason = xkv_skip_reason::unsupported_config;
            result.message = "group input hot row bytes must parallel owning layers";
            return result;
        }
        for (size_t l = 0; l < g_in.owning_layers.size(); ++l) {
            if (g_in.hot_bytes_per_row_k[l] == 0 || g_in.hot_bytes_per_row_v[l] == 0) {
                result.skip_reason = xkv_skip_reason::unsupported_config;
                result.message = "group input hot row bytes must be nonzero measured values";
                return result;
            }
        }

        uint64_t k_elems = 0, v_elems = 0;
        if (!checked_mul_u64(g_in.k_rows, g_in.k_cols, k_elems) || !checked_mul_u64(g_in.v_rows, g_in.v_cols, v_elems)) {
            result.skip_reason = xkv_skip_reason::unsupported_config;
            result.message = "group matrix dimensions overflow";
            return result;
        }

        if (!params.canonical_source) {
            for (uint64_t i = 0; i < k_elems; ++i) {
                if (!std::isfinite(g_in.canonical_k_data[i])) {
                    result.skip_reason = xkv_skip_reason::unsupported_config;
                    result.message = "non-finite float in canonical K data";
                    return result;
                }
            }
            for (uint64_t i = 0; i < v_elems; ++i) {
                if (!std::isfinite(g_in.canonical_v_data[i])) {
                    result.skip_reason = xkv_skip_reason::unsupported_config;
                    result.message = "non-finite float in canonical V data";
                    return result;
                }
            }
        }

        // Scratch calculation for this group
        // No fallback to params.rank_k/rank_v: every group carries its explicit
        // rank (tail groups scale theirs down). Zero or oversized ranks reject
        // atomically before any state transition or factorization.
        if (g_in.rank_k == 0 || g_in.rank_v == 0) {
            result.skip_reason = xkv_skip_reason::unsupported_config;
            result.message = "group " + std::to_string(g_in.group_index) +
                             " requires explicit nonzero rank_k/rank_v";
            return result;
        }
        if (g_in.rank_k > g_in.k_cols || g_in.rank_v > g_in.v_cols ||
            g_in.rank_k > g_in.k_rows || g_in.rank_v > g_in.v_rows) {
            result.skip_reason = xkv_skip_reason::unsupported_config;
            result.message = "group " + std::to_string(g_in.group_index) +
                             " rank exceeds matrix dimensions";
            return result;
        }
        // Explicit feature map required: offsets/dims must cover owning_layers
        // exactly (contiguous from 0, dims nonzero, sums equal total_dim).
        // No silent fallback to whole-matrix flat bytes.
        if (g_in.layer_feature_offsets_k.size() != g_in.owning_layers.size() ||
            g_in.layer_feature_dims_k.size() != g_in.owning_layers.size() ||
            g_in.layer_feature_offsets_v.size() != g_in.owning_layers.size() ||
            g_in.layer_feature_dims_v.size() != g_in.owning_layers.size() ||
            g_in.total_dim_k == 0 || g_in.total_dim_v == 0) {
            result.skip_reason = xkv_skip_reason::unsupported_config;
            result.message = "group " + std::to_string(g_in.group_index) +
                             " requires explicit feature offsets/dims/total_dim";
            return result;
        }
        {
            uint64_t run_k = 0;
            for (size_t l = 0; l < g_in.owning_layers.size(); ++l) {
                if (g_in.layer_feature_dims_k[l] == 0 ||
                    g_in.layer_feature_offsets_k[l] != run_k) {
                    result.skip_reason = xkv_skip_reason::unsupported_config;
                    result.message = "group " + std::to_string(g_in.group_index) +
                                     " layer_feature K map not contiguous from 0";
                    return result;
                }
                run_k += g_in.layer_feature_dims_k[l];
            }
            if (run_k != g_in.total_dim_k) {
                result.skip_reason = xkv_skip_reason::unsupported_config;
                result.message = "group " + std::to_string(g_in.group_index) +
                                 " K dims sum != total_dim_k";
                return result;
            }
            uint64_t run_v = 0;
            for (size_t l = 0; l < g_in.owning_layers.size(); ++l) {
                if (g_in.layer_feature_dims_v[l] == 0 ||
                    g_in.layer_feature_offsets_v[l] != run_v) {
                    result.skip_reason = xkv_skip_reason::unsupported_config;
                    result.message = "group " + std::to_string(g_in.group_index) +
                                     " layer_feature V map not contiguous from 0";
                    return result;
                }
                run_v += g_in.layer_feature_dims_v[l];
            }
            if (run_v != g_in.total_dim_v) {
                result.skip_reason = xkv_skip_reason::unsupported_config;
                result.message = "group " + std::to_string(g_in.group_index) +
                                 " V dims sum != total_dim_v";
                return result;
            }
        }
        const uint32_t eff_rk = g_in.rank_k;
        const uint32_t eff_rv = g_in.rank_v;

        factor_config temp_fcfg;
        temp_fcfg.rank_k = eff_rk;
        temp_fcfg.rank_v = eff_rv;
        temp_fcfg.balance = params.balance;
        temp_fcfg.factor_a_k = params.factor_a_k;
        temp_fcfg.factor_b_k = params.factor_b_k;
        temp_fcfg.factor_a_v = params.factor_a_v;
        temp_fcfg.factor_b_v = params.factor_b_v;

        uint64_t workspace_k = 0;
        uint64_t workspace_v = 0;
        uint64_t output_k = 0;
        std::string estimate_error;
        if (!estimate_factorize_matrix_workspace_bytes(
                g_in.k_rows, g_in.k_cols, eff_rk,
                temp_fcfg.oversampling, temp_fcfg.power_iterations,
                &workspace_k, &estimate_error) ||
            !estimate_factorize_matrix_workspace_bytes(
                g_in.v_rows, g_in.v_cols, eff_rv,
                temp_fcfg.oversampling, temp_fcfg.power_iterations,
                &workspace_v, &estimate_error) ||
            !estimate_factor_output_bytes(
                g_in.k_rows, g_in.k_cols, eff_rk,
                &output_k, &estimate_error)) {
            result.skip_reason = xkv_skip_reason::preflight_oom;
            result.message = "factor workspace estimate failed: " + estimate_error;
            return result;
        }

        uint64_t k_output_plus_v = 0;
        if (output_k > std::numeric_limits<uint64_t>::max() - workspace_v) {
            result.skip_reason = xkv_skip_reason::preflight_oom;
            result.message = "factor workspace estimate overflow";
            return result;
        }
        k_output_plus_v = output_k + workspace_v;
        const uint64_t group_scratch = std::max(workspace_k, k_output_plus_v);

        if (group_scratch > max_group_scratch) {
            max_group_scratch = group_scratch;
        }
    }

    // Exact pre-encode store-capacity preflight: expected candidate bytes from
    // descriptor-level sizes (same make_codec_desc recipe + encoded_matrix_bytes
    // the shadow uses, expected landmark chunks, exact group/segment metadata
    // by size) plus live + retired-pinned current bytes must fit the hard cap
    // BEFORE any factorization/encoding allocates. Refusal touches zero state
    // (pre-mark, no rollback needed). The publish commit rechecks exact
    // post-encode actuals authoritatively.
    {
        auto is_turbo = [](ggml_type t) {
            return t == GGML_TYPE_TURBO2_0 || t == GGML_TYPE_TURBO3_0 || t == GGML_TYPE_TURBO4_0;
        };
        uint64_t expected_new = 0;
        bool cap_ok = true;
        std::string cap_err;
        for (const auto & g_in : group_inputs) {
            const uint32_t grp_k = is_turbo(params.factor_a_k) ? 128 : 0;
            const uint32_t grp_v = is_turbo(params.factor_a_v) ? 128 : 0;
            const uint64_t fseed = params.seed + (uint64_t) g_in.group_index * 1000ULL;
            uint64_t enc = 0;
            try {
                codec_desc d_ak = make_codec_desc(factor_role::a_k, params.factor_a_k,
                    orientation::token_major, matrix_shape{n_rows, g_in.rank_k}, grp_k, fseed);
                codec_desc d_bk = make_codec_desc(factor_role::b_k, params.factor_b_k,
                    orientation::feature_major_transposed, matrix_shape{g_in.k_cols, g_in.rank_k}, grp_k, fseed);
                codec_desc d_av = make_codec_desc(factor_role::a_v, params.factor_a_v,
                    orientation::token_major, matrix_shape{n_rows, g_in.rank_v}, grp_v, fseed + 1);
                codec_desc d_bv = make_codec_desc(factor_role::b_v, params.factor_b_v,
                    orientation::feature_major_transposed, matrix_shape{g_in.v_cols, g_in.rank_v}, grp_v, fseed + 1);
                uint64_t parts[4] = {encoded_matrix_bytes(d_ak), encoded_matrix_bytes(d_bk),
                                     encoded_matrix_bytes(d_av), encoded_matrix_bytes(d_bv)};
                for (uint64_t p : parts) {
                    if (!checked_add_u64(enc, p, enc)) { cap_ok = false; cap_err = "stream size overflow"; break; }
                }
                if (cap_ok && (params.profile == LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS_LANDMARKS ||
                               (params.profile == LLAMA_XKV_STORAGE_PROFILE_REFERENCE && params.landmark_factory != nullptr)) &&
                    params.landmark_factory != nullptr) {
                    const uint32_t chunk_toks = params.chunk_tokens > 0 ? params.chunk_tokens : 8;
                    const uint64_t chunks = ((uint64_t) n_rows + chunk_toks - 1) / chunk_toks;
                    codec_desc d_lm = make_codec_desc(factor_role::landmark, params.landmark_type,
                        orientation::token_major, matrix_shape{chunks, g_in.total_dim_k}, 0, 777);
                    uint64_t lm = encoded_matrix_bytes(d_lm);
                    if (!checked_add_u64(enc, lm, enc)) { cap_ok = false; cap_err = "landmark size overflow"; }
                }
                // Exact group metadata by size (vectors + baseline fields).
                uint64_t meta = 0;
                uint64_t mparts[6] = {(uint64_t) g_in.owning_layers.size() * sizeof(uint32_t),
                    (uint64_t) g_in.layer_feature_offsets_k.size() * sizeof(uint32_t),
                    (uint64_t) g_in.layer_feature_dims_k.size() * sizeof(uint32_t),
                    (uint64_t) g_in.layer_feature_offsets_v.size() * sizeof(uint32_t),
                    (uint64_t) g_in.layer_feature_dims_v.size() * sizeof(uint32_t), 16ULL};
                for (uint64_t p : mparts) {
                    if (!checked_add_u64(meta, p, meta)) { cap_ok = false; cap_err = "metadata size overflow"; break; }
                }
                if (cap_ok && !checked_add_u64(enc, meta, enc)) { cap_ok = false; cap_err = "group size overflow"; }
            } catch (const std::exception & e) {
                cap_ok = false; cap_err = e.what();
            } catch (...) {
                cap_ok = false; cap_err = "unknown exception";
            }
            if (!cap_ok) break;
            if (!checked_add_u64(expected_new, enc, expected_new)) {
                cap_ok = false; cap_err = "candidate size overflow"; break;
            }
        }
        if (cap_ok) {
            const uint32_t chunk_toks = params.chunk_tokens > 0 ? params.chunk_tokens : 8;
            const uint64_t chunks = ((uint64_t) n_rows + chunk_toks - 1) / chunk_toks;
            uint64_t seg_meta = (uint64_t) n_rows * sizeof(uint64_t) + chunks + 8ULL;
            if (!checked_add_u64(expected_new, seg_meta, expected_new)) {
                cap_ok = false; cap_err = "segment metadata overflow";
            }
        }
        if (cap_ok && expected_new > 0) {
            if (params.evaluate_only) {
                // Evaluate-only mode performs no publish; no reservation needed or consumed.
            } else
            // If caller already holds a valid capacity reservation (e.g. Runtime), validate and consume it;
            // otherwise acquire an internal seal capacity reservation.
            if (effective_res != &seal_cap_res) {
                if (effective_res->store_ != this) {
                    cap_ok = false; cap_err = "foreign reservation token from another store";
                } else {
                    std::lock_guard<std::mutex> lock(mtx);
                    auto it_v = pending_reservations_.find(effective_res->token());
                    if (it_v == pending_reservations_.end() || it_v->second != effective_res->reserved_bytes()) {
                        cap_ok = false; cap_err = "stale or invalid capacity reservation token";
                    } else if ((size_t) expected_new > effective_res->reserved_bytes()) {
                        cap_ok = false;
                        cap_err = "caller reservation bytes (" + std::to_string(effective_res->reserved_bytes()) +
                                  ") insufficient for expected candidate bytes (" + std::to_string(expected_new) + ")";
                    }
                }
            } else {
                size_t deficit = 0;
                seal_cap_res = reserve_capacity((size_t) expected_new, &cap_err, &deficit);
                if (!seal_cap_res.valid()) {
                    cap_ok = false;
                }
            }
        }
        if (!cap_ok) {
            result.skip_reason = xkv_skip_reason::store_capacity_exceeded;
            result.message = "seal store capacity preflight refused: " + cap_err;
            return result;
        }
    }

    // 1. Transactional state snapshot: ONLY hot_committed may enter seal!
    uint64_t snapshot_binding_epoch = 0;
    {
        std::lock_guard<std::mutex> lock(mtx);
        // Pre-validate that ALL payloads exist and are strictly in hot_committed state
        for (uint32_t i = 0; i < n_rows; ++i) {
            uint64_t pid = payload_ids[i];
            auto it = payload_locations.find(pid);
            if (it == payload_locations.end() || it->second.state != xkv_state::hot_committed) {
                result.skip_reason = xkv_skip_reason::not_committed;
                result.message = "payloads not in hot_committed state";
                skipped_counts[static_cast<uint8_t>(result.skip_reason)]++;
                return result;
            }
            if (it->second.storage_generation != generations[i]) {
                result.skip_reason = xkv_skip_reason::unsupported_config;
                result.message = "storage_generation mismatch before factorization for payload " + std::to_string(pid);
                skipped_counts[static_cast<uint8_t>(result.skip_reason)]++;
                return result;
            }
        }

        snapshot_binding_epoch = stamp.binding_epoch;
        this_tx_nonce = next_seal_tx_nonce++;
        if (this_tx_nonce == 0) this_tx_nonce = next_seal_tx_nonce++;

        // Mark all under this fresh transaction nonce
        for (uint32_t i = 0; i < n_rows; ++i) {
            uint64_t pid = payload_ids[i];
            auto it = payload_locations.find(pid);
            it->second.state = xkv_state::seal_candidate;
            it->second.seal_tx_nonce = this_tx_nonce;
        }
    }

    // 2. Advisory upfront scratch preflight on the max per-group phase peak.
    // Exact real leases are acquired per group below; a concurrent seal may
    // still force a clean per-group rollback (states restored, nothing published).
    if (max_group_scratch > std::numeric_limits<size_t>::max() ||
        !workspace_arena.preflight((size_t) max_group_scratch)) {
        rollback_transaction(xkv_skip_reason::preflight_oom, "insufficient workspace arena capacity for max group scratch");
        return result;
    }

    // 3. Process every factor group
    std::vector<xkv_factor_group_payload> group_payloads;
    group_payloads.reserve(group_inputs.size());

    size_t total_flat_source_bytes = 0;
    size_t total_candidate_alloc = 0;
    double max_rel_err_k = 0.0;
    double max_rel_err_v = 0.0;

    for (const auto & g_in : group_inputs) {
        // Explicit per-group ranks validated above; no global fallback.
        const uint32_t grp_rk = g_in.rank_k;
        const uint32_t grp_rv = g_in.rank_v;

        factor_config fcfg;
        fcfg.rank_k = grp_rk;
        fcfg.rank_v = grp_rv;
        fcfg.balance = params.balance;
        fcfg.seed = params.seed + g_in.group_index * 1000;
        fcfg.factor_a_k = params.factor_a_k;
        fcfg.factor_b_k = params.factor_b_k;
        fcfg.factor_a_v = params.factor_a_v;
        fcfg.factor_b_v = params.factor_b_v;

        const float * k_ptr = g_in.canonical_k_data;
        const float * v_ptr = g_in.canonical_v_data;
        if (params.canonical_source) {
            std::string src_err;
            if (!params.canonical_source(g_in.group_index, &k_ptr, &v_ptr, &src_err) || !k_ptr || !v_ptr) {
                rollback_transaction(xkv_skip_reason::unsupported_config,
                    "lazy canonical_source failed for group " + std::to_string(g_in.group_index) + ": " + src_err);
                return result;
            }
            uint64_t k_elems = g_in.k_rows * g_in.k_cols;
            uint64_t v_elems = g_in.v_rows * g_in.v_cols;
            for (uint64_t i = 0; i < k_elems; ++i) {
                if (!std::isfinite(k_ptr[i])) {
                    rollback_transaction(xkv_skip_reason::unsupported_config, "non-finite float in lazy canonical K data");
                    return result;
                }
            }
            for (uint64_t i = 0; i < v_elems; ++i) {
                if (!std::isfinite(v_ptr[i])) {
                    rollback_transaction(xkv_skip_reason::unsupported_config, "non-finite float in lazy canonical V data");
                    return result;
                }
            }
        }

        // Zero-copy external view over borrowed canonical floats: NO heap copy!
        matrix mat_k(g_in.k_rows, g_in.k_cols, k_ptr);
        matrix mat_v(g_in.v_rows, g_in.v_cols, v_ptr);

        // Exact factor scratch from the real estimator; one real arena lease per
        // group, released at each iteration end. No nested or accounting-only lease:
        // the span below IS the scratch the bounded calls carve temporaries from.
        // (Canonical mat_k/mat_v staging stays heap until runtime moves staging
        // into its held canonical lease; combined peak is covered because the
        // runtime holds that lease across this call and this scratch fits the
        // remainder or fails cleanly here.)
        uint64_t need_factor = 0;
        {
            std::string estimate_error;
            if (!estimate_factorize_kv_workspace_bytes(mat_k, mat_v, fcfg, &need_factor, &estimate_error)) {
                rollback_transaction(xkv_skip_reason::preflight_oom,
                    "factor workspace estimate failed for group " + std::to_string(g_in.group_index) +
                    ": " + estimate_error);
                return result;
            }
        }
        if (need_factor > std::numeric_limits<size_t>::max()) {
            rollback_transaction(xkv_skip_reason::preflight_oom,
                "factor workspace estimate overflow for group " + std::to_string(g_in.group_index));
            return result;
        }
        xkv_arena_lease group_lease = workspace_arena.acquire((size_t) need_factor);
        if (!group_lease) {
            rollback_transaction(xkv_skip_reason::preflight_oom,
                "failed to acquire factor scratch lease for group " + std::to_string(g_in.group_index));
            return result;
        }

        factor_result f_res = factorize_kv_bounded(mat_k, mat_v, fcfg, factor_workspace_span(group_lease));
        if (!f_res.success) {
            rollback_transaction(xkv_skip_reason::factorization_failed, "factorization failed for group " + std::to_string(g_in.group_index) + ": " + f_res.error_message);
            return result;
        }

        // Shadow reuses the same span sequentially (factor temporaries are dead).
        // Only grows the lease when the shadow phase genuinely needs more.
        {
            uint64_t need_shadow = 0;
            std::string estimate_error;
            if (!estimate_quantized_shadow_workspace_bytes(mat_k, mat_v, f_res.k, f_res.v, fcfg,
                                                            &need_shadow, &estimate_error)) {
                rollback_transaction(xkv_skip_reason::codec_error,
                    "shadow workspace estimate failed for group " + std::to_string(g_in.group_index) +
                    ": " + estimate_error);
                return result;
            }
            if (need_shadow > group_lease.size()) {
                if (need_shadow > std::numeric_limits<size_t>::max()) {
                    rollback_transaction(xkv_skip_reason::preflight_oom,
                        "shadow workspace estimate overflow for group " + std::to_string(g_in.group_index));
                    return result;
                }
                group_lease.release();
                group_lease = workspace_arena.acquire((size_t) need_shadow);
                if (!group_lease) {
                    rollback_transaction(xkv_skip_reason::preflight_oom,
                        "failed to acquire shadow scratch lease for group " + std::to_string(g_in.group_index));
                    return result;
                }
            }
        }

        factor_quantized_shadow shadow = evaluate_quantized_shadow_bounded(
            mat_k, mat_v, f_res.k, f_res.v, fcfg, factor_workspace_span(group_lease));
        if (!shadow.success) {
            rollback_transaction(xkv_skip_reason::codec_error, "codec evaluation failed for group " + std::to_string(g_in.group_index) + ": " + shadow.error_message);
            return result;
        }

        double rel_err_k = shadow.errors_k.ab_product.relative_error;
        double rel_err_v = shadow.errors_v.ab_product.relative_error;
        if (rel_err_k > max_rel_err_k) max_rel_err_k = rel_err_k;
        if (rel_err_v > max_rel_err_v) max_rel_err_v = rel_err_v;

        if (rel_err_k > params.max_relative_error || rel_err_v > params.max_relative_error) {
            // Permanent observability: name the measured errors, gate, shapes,
            // ranks, and codecs so a refusal diagnoses itself (same strict gate).
            std::string detail = "error threshold exceeded for group " + std::to_string(g_in.group_index) +
                " rel_err_k=" + std::to_string(rel_err_k) +
                " rel_err_v=" + std::to_string(rel_err_v) +
                " max_allowed=" + std::to_string(params.max_relative_error) +
                " k_rows=" + std::to_string(g_in.k_rows) +
                " k_cols=" + std::to_string(g_in.k_cols) +
                " v_rows=" + std::to_string(g_in.v_rows) +
                " v_cols=" + std::to_string(g_in.v_cols) +
                " rank_k=" + std::to_string(g_in.rank_k) +
                " rank_v=" + std::to_string(g_in.rank_v) +
                " codecs=" + ggml_type_name(params.factor_a_k) + std::string("/") +
                ggml_type_name(params.factor_b_k) + "/" + ggml_type_name(params.factor_a_v) + "/" +
                ggml_type_name(params.factor_b_v);
            rollback_transaction(xkv_skip_reason::error_threshold_exceeded, detail);
            return result;
        }

        encoded_matrix landmark_em;
        std::vector<xkv_landmark_chunk> landmark_chunks;
        if (params.profile == LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS_LANDMARKS ||
            (params.profile == LLAMA_XKV_STORAGE_PROFILE_REFERENCE && params.landmark_factory != nullptr)) {
            if (!params.landmark_factory) {
                rollback_transaction(xkv_skip_reason::landmark_required, "landmark profile requires group landmark factory callback");
                return result;
            }

            // Backend landmark construction (decodes + factory) must never escape:
            // any throw rolls the transaction back with zero mutation.
            try {
                // Bounded native path: final encoded streams + validated per-row
                // positions + caller arena span. No dense mirrors anywhere.
                std::string lm_err;
                factor_workspace_span lm_span(group_lease);
                if (!params.landmark_factory(g_in.group_index,
                                             shadow.stream_a_k, shadow.stream_b_k,
                                             g_in.row_positions.data(), n_rows,
                                             lm_span, landmark_em, landmark_chunks, &lm_err)) {
                    rollback_transaction(xkv_skip_reason::codec_error, "landmark factory failed for group " + std::to_string(g_in.group_index) + ": " + lm_err);
                    return result;
                }
                if (!landmark_em.desc.validate(&lm_err)) {
                    rollback_transaction(xkv_skip_reason::codec_error, "invalid landmark descriptor for group " + std::to_string(g_in.group_index) + ": " + lm_err);
                    return result;
                }
            } catch (const std::exception & e) {
                rollback_transaction(xkv_skip_reason::codec_error,
                    "landmark construction threw for group " + std::to_string(g_in.group_index) + ": " + e.what());
                return result;
            } catch (...) {
                rollback_transaction(xkv_skip_reason::codec_error,
                    "landmark construction threw an unknown exception for group " + std::to_string(g_in.group_index));
                return result;
            }
        }

        // Flat baseline from measured per-layer hot-row bytes (§2.2/8.1):
        // exact ggml_row_size over each owning layer's live K/V tensor
        // type/ne, populated by the runtime. Never the flat_type theory:
        // heterogeneous layers need exact accounting, not a fabricated
        // baseline. Parallelism/nonzero proven in validation above;
        // arithmetic stays checked here.
        size_t group_flat_bytes = 0;
        for (size_t l = 0; l < g_in.owning_layers.size(); ++l) {
            size_t row_bytes = 0;
            if (!checked_add_size((size_t) g_in.hot_bytes_per_row_k[l],
                    (size_t) g_in.hot_bytes_per_row_v[l], row_bytes)) {
                rollback_transaction(xkv_skip_reason::unsupported_config, "group hot row bytes overflow");
                return result;
            }
            uint64_t layer_flat = 0;
            if (!checked_mul_u64((uint64_t) row_bytes, (uint64_t) n_rows, layer_flat) ||
                layer_flat > (uint64_t) std::numeric_limits<size_t>::max()) {
                rollback_transaction(xkv_skip_reason::unsupported_config, "group flat bytes overflow");
                return result;
            }
            if (!checked_add_size(group_flat_bytes, (size_t) layer_flat, group_flat_bytes)) {
                rollback_transaction(xkv_skip_reason::unsupported_config, "group flat bytes overflow");
                return result;
            }
        }
        if (!checked_add_size(total_flat_source_bytes, group_flat_bytes, total_flat_source_bytes)) {
            rollback_transaction(xkv_skip_reason::unsupported_config, "flat source bytes overflow");
            return result;
        }
        xkv_factor_group_payload g_payload;
        if (n_rows > 0) {
            uint64_t group_row_baseline = 0;
            for (size_t l = 0; l < g_in.owning_layers.size(); ++l) {
                size_t row_bytes = 0;
                if (!checked_add_size((size_t) g_in.hot_bytes_per_row_k[l],
                        (size_t) g_in.hot_bytes_per_row_v[l], row_bytes) ||
                    !checked_add_u64(group_row_baseline, (uint64_t) row_bytes, group_row_baseline)) {
                    rollback_transaction(xkv_skip_reason::unsupported_config, "group baseline row bytes overflow");
                    return result;
                }
            }
            g_payload.baseline_original_row_bytes = group_row_baseline;
            g_payload.baseline_original_bytes = (uint64_t) group_flat_bytes;
        }

        g_payload.group_index = g_in.group_index;
        g_payload.owning_layers = g_in.owning_layers;
        g_payload.rank_k = grp_rk;
        g_payload.rank_v = grp_rv;
        g_payload.layer_feature_offsets_k = g_in.layer_feature_offsets_k;
        g_payload.layer_feature_dims_k = g_in.layer_feature_dims_k;
        g_payload.layer_feature_offsets_v = g_in.layer_feature_offsets_v;
        g_payload.layer_feature_dims_v = g_in.layer_feature_dims_v;
        g_payload.total_dim_k = g_in.total_dim_k;
        g_payload.total_dim_v = g_in.total_dim_v;

        g_payload.a_k = std::move(shadow.stream_a_k);
        g_payload.set_b_k(std::move(shadow.stream_b_k));
        g_payload.a_v = std::move(shadow.stream_a_v);
        g_payload.set_b_v(std::move(shadow.stream_b_v));
        g_payload.landmark = std::move(landmark_em);
        // Landmark bounds exist only with a landmark stream: TQ_FACTORS (and any
        // profile whose factory did not run) must publish zero bounds/fingerprint,
        // otherwise validate_candidate's strict bounds-without-stream rejection
        // would refuse every landmark-less bundle (FNV of empty chunks is nonzero).
        if (g_payload.landmark.desc.logical_shape.rows > 0) {
            g_payload.landmark_chunks = std::move(landmark_chunks);
            g_payload.landmark_table_fingerprint = compute_landmark_table_fingerprint(g_payload.landmark_chunks);
        } else {
            g_payload.landmark_chunks.clear();
            g_payload.landmark_table_fingerprint = 0;
        }

        g_payload.refresh_descriptor_fingerprint();
        g_payload.update_byte_counters();

        total_candidate_alloc += g_payload.total_allocated_bytes;
        group_payloads.push_back(std::move(g_payload));
    }

    const uint32_t chunk_toks = params.chunk_tokens > 0 ? params.chunk_tokens : 8;
    const size_t expected_chunks = (n_rows + chunk_toks - 1) / chunk_toks;
    const size_t bundle_metadata_bytes = n_rows * sizeof(uint64_t) + expected_chunks;
    total_candidate_alloc += bundle_metadata_bytes;

    // Assemble candidate bundle
    auto candidate = create_candidate_segment(params.profile, params.source, group_payloads);
    candidate->layer_group_map_fingerprint = actual_map_fp;
    candidate->residency = params.expected_residency;
    candidate->baseline_original_bytes = (uint64_t) total_flat_source_bytes;
    candidate->n_rows = n_rows;
    candidate->n_live_rows = n_rows;
    candidate->row_payload_ids = payload_ids;
    candidate->live_rows.assign(n_rows, true);
    candidate->update_byte_counters();

    // Authoritative exact min-saving gate using candidate_incremental_bytes:
    // factors in full segment metadata and exact B deduplication against live/retired store state
    size_t actual_factored_bytes = 0;
    std::string inc_err;
    if (!candidate_incremental_bytes(candidate, &actual_factored_bytes, &inc_err)) {
        rollback_transaction(xkv_skip_reason::unsupported_config, "candidate_incremental_bytes failed: " + inc_err);
        return result;
    }
    if (actual_factored_bytes == 0) {
        actual_factored_bytes = candidate->total_allocated_bytes;
    }

    result.flat_source_bytes = total_flat_source_bytes;
    result.factored_bytes = actual_factored_bytes;
    result.relative_error_k = max_rel_err_k;
    result.relative_error_v = max_rel_err_v;

    const double effective_min_saving = (params.min_saving_ratio > 0.0) ? params.min_saving_ratio : cparams.xkv_min_saving;
    // Permanent exact accounting for no-saving refusals: distinguishes
    // legitimate small-segment economics from an accounting defect.
    // Built lazily (failure path only) to keep the seal hot path heap-free.
    auto saving_detail = [&]() -> std::string {
        std::string d = "flat=" + std::to_string(total_flat_source_bytes) +
            "B factored_increment=" + std::to_string(actual_factored_bytes) + "B saved=" +
            std::to_string(total_flat_source_bytes > actual_factored_bytes
                ? total_flat_source_bytes - actual_factored_bytes : 0) +
            "B required_bytes=" + std::to_string(params.min_saving_bytes) +
            " required_ratio=" + std::to_string(effective_min_saving) +
            " profile=" + std::string(llama_xkv_storage_profile_name(params.profile)) +
            " rows=" + std::to_string(n_rows) +
            " ngroups=" + std::to_string(group_payloads.size());
        for (size_t gi = 0; gi < group_payloads.size(); ++gi) {
            d += " g" + std::to_string(gi) +
                ":k" + std::to_string(group_payloads[gi].rank_k) +
                "v" + std::to_string(group_payloads[gi].rank_v);
        }
        return d;
    };
    if (total_flat_source_bytes == 0 ||
        actual_factored_bytes >= total_flat_source_bytes ||
        (total_flat_source_bytes - actual_factored_bytes) < params.min_saving_bytes) {
        rollback_transaction(xkv_skip_reason::no_saving, "no net memory savings achieved for bundle compared to flat source (" + saving_detail() + ")");
        return result;
    }

    const double saving_ratio = (double)(total_flat_source_bytes - actual_factored_bytes) / (double)total_flat_source_bytes;
    if (saving_ratio < effective_min_saving) {
        rollback_transaction(xkv_skip_reason::no_saving, "bundle saving ratio below required minimum fraction (" + saving_detail() + " actual_ratio=" + std::to_string(saving_ratio) + ")");
        return result;
    }

    result.saved_bytes = total_flat_source_bytes - actual_factored_bytes;
    result.compression_ratio = (double) total_flat_source_bytes / (double) actual_factored_bytes;

    // If caller or internal seal holds a capacity reservation, ensure its reserved bytes
    // cover the exact actual_factored_bytes BEFORE calling publish_candidate!
    if (effective_res != nullptr && effective_res->valid()) {
        if (effective_res == &seal_cap_res && actual_factored_bytes > seal_cap_res.reserved_bytes()) {
            std::lock_guard<std::mutex> lock(mtx);
            size_t deficit = actual_factored_bytes - seal_cap_res.reserved_bytes();
            size_t new_pending = 0;
            if (checked_add_size(pending_reserved_store_bytes_, deficit, new_pending)) {
                pending_reserved_store_bytes_ = new_pending;
                pending_reservations_[seal_cap_res.token()] = actual_factored_bytes;
                seal_cap_res.reserved_bytes_ = actual_factored_bytes;
            }
        } else if (effective_res != &seal_cap_res && actual_factored_bytes > effective_res->reserved_bytes()) {
            rollback_transaction(xkv_skip_reason::store_capacity_exceeded,
                "caller reservation bytes (" + std::to_string(effective_res->reserved_bytes()) +
                ") insufficient for actual candidate bytes (" + std::to_string(actual_factored_bytes) + ")");
            return result;
        }
    }

    // Hard store cap: exact candidate peak (live + retired + candidate,
    // shared B once) must fit before publish. Refusal rolls back with the
    // dedicated reason; the walk allocates, hence inside the seal try.
    // (Lock released before rollback: rollback takes the mutex itself.)
    bool cap_fits = false;
    std::string cap_err;
    {
        std::lock_guard<std::mutex> lock(mtx);
        std::vector<std::shared_ptr<const xkv_segment>> cap_extras;
        cap_extras.push_back(candidate);
        if (!ensure_accounting_scratch_locked(cap_extras, &cap_err)) {
            cap_fits = false;
        } else {
        const size_t exclude_bytes = (effective_res != nullptr && effective_res->valid()) ? effective_res->reserved_bytes() : 0;
        cap_fits = store_capacity_fits_locked(cap_extras, &cap_err, exclude_bytes);
        }
    }
    if (!cap_fits) {
        rollback_transaction(xkv_skip_reason::store_capacity_exceeded,
            "store capacity exceeded for candidate bundle: " + cap_err);
        return result;
    }

    std::vector<uint32_t> current_physical_rows(n_rows);
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (stamp.binding_epoch != snapshot_binding_epoch) {
            // Refreshed positions
        }
        for (uint32_t i = 0; i < n_rows; ++i) {
            auto it = payload_locations.find(payload_ids[i]);
            if (it == payload_locations.end() || it->second.state != xkv_state::seal_candidate) {
                rollback_transaction(xkv_skip_reason::not_committed, "payload evicted or modified before publish");
                return result;
            }
            current_physical_rows[i] = it->second.row;
        }
    }

    std::string err;
    // Pre-build the release plan BEFORE publish: the vector copies may
    // allocate, and after a gated publish nothing may throw or allocate.
    xkv_hot_release_plan prebuilt_plan;
    prebuilt_plan.released_payload_ids = payload_ids;
    prebuilt_plan.released_physical_rows = current_physical_rows;
    prebuilt_plan.released_generations = generations;
    // SHADOW evaluate-only: gates passed; restore candidates without publishing
    // and report the would-be stats with an empty release plan. No segment is
    // published, no release plan emitted, no sealed_count bump. The precommit
    // gate is moot (nothing commits) and is skipped.
    if (params.evaluate_only) {
        if (effective_res == &seal_cap_res) {
            seal_cap_res.release();
        }
        {
            std::lock_guard<std::mutex> lock(mtx);
            for (uint32_t i = 0; i < n_rows; ++i) {
                auto it = payload_locations.find(payload_ids[i]);
                if (it != payload_locations.end() && it->second.seal_tx_nonce == this_tx_nonce) {
                    it->second.state = xkv_state::hot_committed;
                    it->second.seal_tx_nonce = 0;
                }
            }
        }
        abort_candidate(candidate);
        result.success = true;
        result.skip_reason = xkv_skip_reason::none;
        result.message = "evaluate-only: gates passed, nothing published";
        result.stamp = current_stamp();
        return result;
    }

    if (!publish_candidate(candidate, payload_ids, generations, &err, nullptr, precommit, effective_res)) {
        // A precommit refusal arrives here as a publish failure: roll back only
        // this transaction's nonce, exactly like any other publish failure.
        rollback_transaction(xkv_skip_reason::aborted, "atomic publish failed: " + err);
        return result;
    }

    result.success = true;
    result.skip_reason = xkv_skip_reason::none;
    result.segment_id = candidate->segment_id;
    result.segment_version = candidate->segment_version;
    result.stamp = current_stamp();

    // No-throw commit tail: move the prebuilt plan in (move-assign is noexcept).
    result.release_plan = std::move(prebuilt_plan);
    result.release_plan.expected_segment_id = candidate->segment_id;
    result.release_plan.expected_segment_version = candidate->segment_version;
    result.release_plan.dense_bytes_freed = total_flat_source_bytes;
    result.release_plan.stamp = result.stamp;
    // A gated publish means the precommit callback owned the pool release at
    // commit time: mark the plan consumed so the runtime cannot double-release.
    result.release_plan.pool_released_at_precommit = (precommit != nullptr);

    return result;
    } catch (const std::exception & e) {
        // Roll back only if this transaction actually marked candidates; a
        // pre-mark throw leaves zero state (nonce still zero matches nothing).
        if (this_tx_nonce != 0) {
            rollback_transaction(xkv_skip_reason::aborted,
                std::string("seal threw: ") + e.what());
        } else {
            result.skip_reason = xkv_skip_reason::aborted;
            result.message = std::string("seal threw before marking: ") + e.what();
        }
        return result;
    } catch (...) {
        if (this_tx_nonce != 0) {
            rollback_transaction(xkv_skip_reason::aborted, "seal threw an unknown exception");
        } else {
            result.skip_reason = xkv_skip_reason::aborted;
            result.message = "seal threw an unknown exception before marking";
        }
        return result;
    }
}

xkv_sealing_result llama_xkv_cache_store::evaluate_segment_bundle(
    const std::vector<xkv_factor_group_input> & group_inputs,
    const std::vector<uint64_t> & payload_ids,
    const std::vector<uint64_t> & generations,
    const xkv_bundle_sealing_params & params
) {
    xkv_bundle_sealing_params eval_params = params;
    eval_params.evaluate_only = true;
    return seal_segment_bundle(group_inputs, payload_ids, generations, eval_params, nullptr);
}

bool llama_xkv_cache_store::import_snapshot_segments(const xkv_snapshot_import_bundle & bundle,
                                                      std::string * err,
                                                      xkv_capacity_reservation * capacity_reservation) {
    // Whole-body catch-all: validation containers and map reserves may
    // allocate; any escape reports false with old state untouched. The commit
    // tail itself performs no allocation.
    try {
        // Reservation gate first (read-only). Merge import coexists with live
        // state: existing identical segments/locations are reused, only missing
        // ones are added, conflicts refuse.
        {
            std::lock_guard<std::mutex> lock(mtx);
            if (check_reservation_locked(0)) {
                if (err) *err = "import_snapshot_segments: epoch reservation held by another transaction";
                return false;
            }
        }
        // Off-side validation: complete segment/location closure.
        std::unordered_set<uint64_t> seen_segs;
        std::unordered_set<uint64_t> seen_pids;
        // Exact (segment, version, row) index per payload: a location with a
        // wrong segment_version must never pass on id/row alone. Segment ids
        // are full 64-bit and must not be truncated into 32 bits.
        struct import_row_ref {
            uint64_t seg_id = 0;
            uint64_t seg_version = 0;
            uint32_t row = 0;
        };
        std::unordered_map<uint64_t, import_row_ref> row_pid;
        uint64_t max_seg_id = 0;
        for (const auto & seg : bundle.segments) {
            if (!seg) {
                if (err) *err = "import_snapshot_segments: null segment in bundle";
                return false;
            }
            if (seg->segment_id == 0 || seg->segment_version == 0) {
                if (err) *err = "import_snapshot_segments: segment id/version must be nonzero";
                return false;
            }
            if (!seen_segs.insert(seg->segment_id).second) {
                if (err) *err = "import_snapshot_segments: duplicate segment id";
                return false;
            }
            if (seg->segment_id > max_seg_id) max_seg_id = seg->segment_id;
            std::string verr;
            if (!validate_candidate(seg, &verr)) {
                if (err) *err = "import_snapshot_segments: segment invalid: " + verr;
                return false;
            }
            if (seg->n_rows == 0 || seg->row_payload_ids.size() != seg->n_rows ||
                seg->live_rows.size() != seg->n_rows) {
                if (err) *err = "import_snapshot_segments: row map size mismatch";
                return false;
            }
            for (uint32_t r = 0; r < seg->n_rows; ++r) {
                uint64_t pid = seg->row_payload_ids[r];
                // Live-only coverage: dead rows carry no locations (tombstone
                // generations must never be fabricated); they import as dead.
                if (!seg->live_rows[r]) {
                    continue;
                }
                if (pid == 0 || !seen_pids.insert(pid).second) {
                    if (err) *err = "import_snapshot_segments: zero/duplicate row payload";
                    return false;
                }
                row_pid[pid] = {seg->segment_id, seg->segment_version, r};
            }
        }
        // Location closure: exactly the segment rows, factored, matching, live.
        {
            std::unordered_set<uint64_t> loc_pids;
            for (const auto & pl : bundle.locations) {
                uint64_t pid = pl.first;
                const xkv_location & loc = pl.second;
                if (pid == 0 || !loc_pids.insert(pid).second) {
                    if (err) *err = "import_snapshot_segments: zero/duplicate location pid";
                    return false;
                }
                if (loc.kind != xkv_location_kind::factored || loc.segment_id == 0 ||
                    loc.segment_version == 0 || loc.storage_generation == 0 ||
                    loc.seal_tx_nonce != 0 ||
                    (loc.state != xkv_state::factored && loc.state != xkv_state::flat_tq)) {
                    if (err) *err = "import_snapshot_segments: location not cleanly factored";
                    return false;
                }
                auto rp = row_pid.find(pid);
                if (rp == row_pid.end() ||
                    rp->second.seg_id != loc.segment_id ||
                    rp->second.seg_version != loc.segment_version ||
                    rp->second.row != loc.row) {
                    if (err) *err = "import_snapshot_segments: location/row mismatch";
                    return false;
                }
                auto seg_it = seen_segs.find(loc.segment_id);
                if (seg_it == seen_segs.end()) {
                    if (err) *err = "import_snapshot_segments: location references unknown segment";
                    return false;
                }
            }
            if (loc_pids.size() != seen_pids.size()) {
                if (err) *err = "import_snapshot_segments: location set does not cover segment rows";
                return false;
            }
        }
        // NOTE: live-row coverage is enforced per staged segment in the merge
        // classification below (reused segments keep their existing mappings).
        // Existing hot pids must not collide with imported factored pids.
        // Merge classification under lock:
        // Segments: if matching (id, version) exists, must be bit-identical
        // (segments_exact_equal); reused with no allocation. Otherwise must
        // not collide with published segment_id (unless retired version).
        // Locations: if pid exists in store, must match existing factored
        // location exactly; otherwise added. Existing hot pids must not collide.
        std::vector<std::shared_ptr<const xkv_segment>> staged_segments;
        std::vector<std::pair<uint64_t, xkv_location>> staged_locations;
        {
            std::lock_guard<std::mutex> lock(mtx);
            for (const auto & seg : bundle.segments) {
                auto pub_it = published_segments.find(seg->segment_id);
                if (pub_it != published_segments.end()) {
                    if (pub_it->second->segment_version == seg->segment_version) {
                        if (!xkv_store_detail::segments_exact_equal(*pub_it->second, *seg)) {
                            if (err) *err = "import_snapshot_segments: conflict with existing published segment " +
                                             std::to_string(seg->segment_id);
                            return false;
                        }
                        // Exact identical segment exists: reuse it!
                        continue;
                    } else {
                        if (err) *err = "import_snapshot_segments: segment id " + std::to_string(seg->segment_id) +
                                         " collides with different version in published store";
                        return false;
                    }
                }
                // Check retired segments for same (id, version)
                bool found_retired = false;
                for (const auto & rseg : retired_segments) {
                    if (rseg->segment_id == seg->segment_id && rseg->segment_version == seg->segment_version) {
                        if (!xkv_store_detail::segments_exact_equal(*rseg, *seg)) {
                            if (err) *err = "import_snapshot_segments: conflict with retired segment " +
                                             std::to_string(seg->segment_id);
                            return false;
                        }
                        found_retired = true;
                        break;
                    }
                }
                if (!found_retired) {
                    staged_segments.push_back(seg);
                }
            }

            for (const auto & pl : bundle.locations) {
                auto loc_it = payload_locations.find(pl.first);
                if (loc_it != payload_locations.end()) {
                    if (loc_it->second.kind != xkv_location_kind::factored ||
                        loc_it->second.segment_id != pl.second.segment_id ||
                        loc_it->second.segment_version != pl.second.segment_version ||
                        loc_it->second.row != pl.second.row ||
                        loc_it->second.storage_generation != pl.second.storage_generation) {
                        if (err) *err = "import_snapshot_segments: pid collision with live payload " + std::to_string(pl.first);
                        return false;
                    }
                    // Exact identical factored location exists: reuse!
                } else {
                    staged_locations.push_back(pl);
                }
            }
        }
        // High-water validation:
        if (bundle.next_seal_tx_nonce == 0 || bundle.next_alloc_id == 0) {
            if (err) *err = "import_snapshot_segments: high-waters must be nonzero";
            return false;
        }
        // Hard cap check over newly staged segments only:
        if (!staged_segments.empty()) {
            std::lock_guard<std::mutex> lock(mtx);
            size_t exclude_bytes = 0;
            if (capacity_reservation != nullptr) {
                if (!capacity_reservation->valid()) {
                    if (err) *err = "import_snapshot_segments: stale, released, or invalid reservation token";
                    return false;
                }
                if (capacity_reservation->store_ != this) {
                    if (err) *err = "import_snapshot_segments: foreign reservation token from another store";
                    return false;
                }
                auto it_v = pending_reservations_.find(capacity_reservation->token());
                if (it_v == pending_reservations_.end() || it_v->second != capacity_reservation->reserved_bytes()) {
                    if (err) *err = "import_snapshot_segments: stale or unknown reservation token";
                    return false;
                }
                exclude_bytes = capacity_reservation->reserved_bytes();
            }
            if (!ensure_accounting_scratch_locked(staged_segments, err)) return false;
            if (!store_capacity_fits_locked(staged_segments, err, exclude_bytes)) {
                return false;
            }
        }
        // Single no-throw commit of staged segments and locations:
        {
            std::lock_guard<std::mutex> lock(mtx);
            if (check_reservation_locked(0)) {
                if (err) *err = "import_snapshot_segments: epoch reservation held by another transaction";
                return false;
            }
            published_segments.reserve(published_segments.size() + staged_segments.size());
            payload_locations.reserve(payload_locations.size() + staged_locations.size());
            for (const auto & seg : staged_segments) {
                published_segments[seg->segment_id] = seg;
            }
            for (const auto & pl : staged_locations) {
                payload_locations[pl.first] = pl.second;
            }
            // Track host/device peak on import commit:
            size_t imp_dev = 0, imp_host = 0;
            for (const auto & [_, seg] : published_segments) {
                if (seg->residency == GGML_XKV_RES_DEVICE_OWNED) imp_dev += seg->total_allocated_bytes;
                else imp_host += seg->total_allocated_bytes;
            }
            for (const auto & seg : retired_segments) {
                if (seg->residency == GGML_XKV_RES_DEVICE_OWNED) imp_dev += seg->total_allocated_bytes;
                else imp_host += seg->total_allocated_bytes;
            }
            if (imp_host > host_peak_bytes_) host_peak_bytes_ = imp_host;
            if (imp_dev > device_peak_bytes_) device_peak_bytes_ = imp_dev;

            // Adopt max of epochs/high-waters
            stamp.live_epoch = std::max(stamp.live_epoch, bundle.stamp.live_epoch);
            stamp.content_epoch = std::max(stamp.content_epoch, bundle.stamp.content_epoch);
            stamp.codec_epoch = std::max(stamp.codec_epoch, bundle.stamp.codec_epoch);
            stamp.binding_epoch = std::max(stamp.binding_epoch, bundle.stamp.binding_epoch);
            sealed_count = std::max(sealed_count, (size_t)bundle.sealed_count);
            if (bundle.next_segment_id > next_segment_id) next_segment_id = bundle.next_segment_id;
            if (bundle.next_seal_tx_nonce > next_seal_tx_nonce) next_seal_tx_nonce = bundle.next_seal_tx_nonce;
            if (bundle.next_alloc_id > alloc_id_gen->current_id()) {
                alloc_id_gen->reset(bundle.next_alloc_id);
            }
            // Consume capacity reservation upon successful import commit
            if (capacity_reservation != nullptr && capacity_reservation->valid()) {
                auto it_r = pending_reservations_.find(capacity_reservation->token());
                if (it_r != pending_reservations_.end()) {
                    if (pending_reserved_store_bytes_ >= it_r->second) {
                        pending_reserved_store_bytes_ -= it_r->second;
                    } else {
                        pending_reserved_store_bytes_ = 0;
                    }
                    pending_reservations_.erase(it_r);
                }
                capacity_reservation->store_ = nullptr;
                capacity_reservation->token_ = 0;
                capacity_reservation->reserved_bytes_ = 0;
            }
            return true;
        }
    } catch (const std::exception & e) {
        if (err) *err = std::string("import_snapshot_segments failed: ") + e.what();
        return false;
    } catch (...) {
        if (err) *err = "import_snapshot_segments failed with an unknown exception";
        return false;
    }
}

xkv_reader_pin llama_xkv_cache_store::pin_segment(uint64_t segment_id) {
    std::lock_guard<std::mutex> lock(mtx);
    auto it = published_segments.find(segment_id);
    if (it != published_segments.end()) {
        return xkv_reader_pin(it->second);
    }
    return xkv_reader_pin();
}

xkv_reader_pin llama_xkv_cache_store::pin_segment_version(uint64_t segment_id, uint64_t segment_version) {
    std::lock_guard<std::mutex> lock(mtx);
    auto it = published_segments.find(segment_id);
    if (it != published_segments.end() && it->second->segment_version == segment_version) {
        return xkv_reader_pin(it->second);
    }
    for (const auto & s : retired_segments) {
        if (s->segment_id == segment_id && s->segment_version == segment_version) {
            return xkv_reader_pin(s);
        }
    }
    return xkv_reader_pin();
}

std::shared_ptr<const xkv_segment> llama_xkv_cache_store::get_segment(uint64_t segment_id) const {
    std::lock_guard<std::mutex> lock(mtx);
    auto it = published_segments.find(segment_id);
    if (it != published_segments.end()) {
        return it->second;
    }
    return nullptr;
}

std::shared_ptr<const xkv_segment> llama_xkv_cache_store::get_segment_version(uint64_t segment_id, uint64_t segment_version) const {
    std::lock_guard<std::mutex> lock(mtx);
    auto it = published_segments.find(segment_id);
    if (it != published_segments.end() && it->second->segment_version == segment_version) {
        return it->second;
    }
    for (const auto & s : retired_segments) {
        if (s->segment_id == segment_id && s->segment_version == segment_version) {
            return s;
        }
    }
    return nullptr;
}

void llama_xkv_cache_store::reclaim_retired_segments_locked() {
    auto it = retired_segments.begin();
    while (it != retired_segments.end()) {
        const auto & seg = *it;
        if (seg->pin_count.load(std::memory_order_relaxed) == 0 && seg.use_count() == 1) {
            it = retired_segments.erase(it);
        } else {
            ++it;
        }
    }
}

bool llama_xkv_cache_store::ensure_accounting_scratch_locked(
    const std::vector<std::shared_ptr<const xkv_segment>> & candidate_extras, std::string * err) const {
    // Derive exact upper bound of distinct B matrices and backend handles by summing
    // actual group streams and backend bundle allocations across all published,
    // retired, and candidate segments (checked adds throughout).
    size_t sum_b = 0;
    size_t sum_backend = 0;
    auto count_seg = [&](const std::shared_ptr<const xkv_segment> & seg) {
        if (!seg) return;
        for (const auto & g : seg->groups) {
            if (g.b_k) {
                if (sum_b < std::numeric_limits<size_t>::max()) ++sum_b;
            }
            if (g.b_v) {
                if (sum_b < std::numeric_limits<size_t>::max()) ++sum_b;
            }
        }
        if (seg->backend_bundle) {
            for (const auto & h : seg->backend_bundle->handles) {
                if (h && sum_backend < std::numeric_limits<size_t>::max()) ++sum_backend;
            }
        }
    };
    for (const auto & kv : published_segments) count_seg(kv.second);
    for (const auto & s : retired_segments) count_seg(s);
    for (const auto & s : candidate_extras) count_seg(s);

    const size_t new_cap_b_k = std::max(scratch_unique_b_k_.capacity(), sum_b);
    const size_t new_cap_b_v = std::max(scratch_unique_b_v_.capacity(), sum_b);
    const size_t new_cap_dev = std::max(scratch_backend_alloc_ids_.capacity(), sum_backend);
    const size_t new_scratch_bytes = new_cap_b_k * sizeof(const encoded_matrix *) +
                                     new_cap_b_v * sizeof(const encoded_matrix *) +
                                     new_cap_dev * sizeof(uint64_t) +
                                     new_cap_dev * sizeof(uint8_t);
    const size_t store_cap = store_capacity_bytes();
    if (store_cap != 0) {
        size_t total_needed = 0;
        if (!checked_add_size(new_scratch_bytes, pending_reserved_store_bytes_, total_needed)) {
            if (err) *err = "accounting scratch budget overflow";
            return false;
        }
        if (total_needed > store_cap) {
            if (err) *err = "accounting scratch exceeds store capacity budget";
            return false;
        }
    }
    if (scratch_unique_b_k_.capacity() < sum_b) {
        scratch_unique_b_k_.reserve(sum_b);
    }
    if (scratch_unique_b_v_.capacity() < sum_b) {
        scratch_unique_b_v_.reserve(sum_b);
    }
    if (scratch_backend_alloc_ids_.capacity() < sum_backend) {
        scratch_backend_alloc_ids_.reserve(sum_backend);
    }
    if (scratch_backend_alloc_accounted_.capacity() < sum_backend) {
        scratch_backend_alloc_accounted_.reserve(sum_backend);
    }
    return true;
}

void llama_xkv_cache_store::reclaim_retired_segments() {
    std::lock_guard<std::mutex> lock(mtx);
    reclaim_retired_segments_locked();
}

xkv_hot_bindings_snapshot llama_xkv_cache_store::list_hot_payload_bindings(
    const std::vector<xkv_state> & state_filter
) const {
    std::lock_guard<std::mutex> lock(mtx);
    xkv_hot_bindings_snapshot snap;
    snap.stamp = stamp;

    std::unordered_set<uint8_t> filter_set;
    for (auto s : state_filter) {
        filter_set.insert(static_cast<uint8_t>(s));
    }

    for (const auto & kv : payload_locations) {
        if (kv.second.kind == xkv_location_kind::hot) {
            if (filter_set.empty() || filter_set.find(static_cast<uint8_t>(kv.second.state)) != filter_set.end()) {
                xkv_hot_payload_binding b;
                b.payload_id = kv.first;
                b.hot_slot_row = kv.second.row;
                b.storage_generation = kv.second.storage_generation;
                b.state = kv.second.state;
                snap.bindings.push_back(b);
            }
        }
    }

    std::sort(snap.bindings.begin(), snap.bindings.end());
    return snap;
}

xkv_payload_segments_snapshot llama_xkv_cache_store::query_payload_segments(
    const std::vector<uint64_t> & payload_ids
) {
    std::lock_guard<std::mutex> lock(mtx);
    xkv_payload_segments_snapshot snap;
    snap.stamp = stamp;

    std::vector<uint64_t> sorted_ids = payload_ids;
    std::sort(sorted_ids.begin(), sorted_ids.end());
    sorted_ids.erase(std::unique(sorted_ids.begin(), sorted_ids.end()), sorted_ids.end());

    snap.views.reserve(sorted_ids.size());
    for (uint64_t pid : sorted_ids) {
        auto it = payload_locations.find(pid);
        if (it != payload_locations.end()) {
            xkv_payload_segment_view v;
            v.payload_id = pid;
            v.location = it->second;
            if (it->second.kind == xkv_location_kind::factored && it->second.segment_id > 0) {
                auto seg_it = published_segments.find(it->second.segment_id);
                if (seg_it != published_segments.end() && seg_it->second->segment_version == it->second.segment_version) {
                    v.pin = xkv_reader_pin(seg_it->second);
                } else {
                    for (const auto & rseg : retired_segments) {
                        if (rseg->segment_id == it->second.segment_id && rseg->segment_version == it->second.segment_version) {
                            v.pin = xkv_reader_pin(rseg);
                            break;
                        }
                    }
                }
            }
            snap.views.push_back(std::move(v));
        }
    }

    return snap;
}

bool llama_xkv_cache_store::validate_hot_release_plan(
    const xkv_hot_release_plan & plan,
    std::string * err
) const {
    std::lock_guard<std::mutex> lock(mtx);

    const size_t n = plan.released_payload_ids.size();
    if (n == 0) {
        if (err) *err = "release plan is empty";
        return false;
    }
    if (plan.released_physical_rows.size() != n) {
        if (err) *err = "release plan physical rows size mismatch";
        return false;
    }
    if (plan.released_generations.size() != n) {
        if (err) *err = "release plan generations size mismatch";
        return false;
    }
    if (plan.expected_segment_id == 0 || plan.expected_segment_version == 0) {
        if (err) *err = "release plan missing expected segment_id or segment_version";
        return false;
    }

    for (size_t i = 0; i < n; ++i) {
        uint64_t pid = plan.released_payload_ids[i];
        auto it = payload_locations.find(pid);
        if (it == payload_locations.end()) {
            if (err) *err = "payload " + std::to_string(pid) + " not found in store";
            return false;
        }
        if (it->second.kind != xkv_location_kind::factored) {
            if (err) *err = "payload " + std::to_string(pid) + " is not currently factored";
            return false;
        }
        if (it->second.segment_id != plan.expected_segment_id) {
            if (err) *err = "payload " + std::to_string(pid) + " segment_id mismatch";
            return false;
        }
        if (it->second.segment_version != plan.expected_segment_version) {
            if (err) *err = "payload " + std::to_string(pid) + " segment_version mismatch";
            return false;
        }
        if (it->second.storage_generation != plan.released_generations[i]) {
            if (err) *err = "payload " + std::to_string(pid) + " generation mismatch with release plan";
            return false;
        }
    }

    return true;
}

bool llama_xkv_cache_store::pack_segment(
    uint64_t segment_id,
    std::string * err,
    xkv_landmark_rebuild_fn landmark_rebuild
) {
    xkv_batch_mutation mutation;
    mutation.segments_to_pack.push_back(segment_id);
    mutation.landmark_rebuild = std::move(landmark_rebuild);
    return execute_mutation_transaction(mutation, nullptr, err);
}

namespace xkv_store_detail {
// Exact segment equality for merge-reuse: id, version, profile, source,
// fingerprints, baselines, row maps, live mask, every group descriptor +
// code bytes (B via shared-pointer fast path then bytes), and backend
// handle identity (both null, or equal allocation-id multisets).
bool streams_equal(const encoded_matrix & a, const encoded_matrix & b) {
        if (a.desc.fingerprint() != b.desc.fingerprint()) return false;
        return a.bytes == b.bytes;
}
bool shared_stream_equal(const std::shared_ptr<const encoded_matrix> & a,
                                  const std::shared_ptr<const encoded_matrix> & b) {
        if (a == b) return true;
        if (!a || !b) return false;
        return streams_equal(*a, *b);
}
bool backend_handles_equal(const std::shared_ptr<const xkv_backend_batch_result> & a,
                                    const std::shared_ptr<const xkv_backend_batch_result> & b) {
        if (a == b) return true;
        if (!a || !b) return false;
        std::unordered_map<uint64_t, size_t> counts;
        for (const auto & h : a->handles) {
            if (!h) return false;
            counts[h->get_allocation_id()]++;
        }
        for (const auto & h : b->handles) {
            if (!h) return false;
            auto it = counts.find(h->get_allocation_id());
            if (it == counts.end() || it->second == 0) return false;
            if (--(it->second) == 0) counts.erase(it);
        }
        return counts.empty();
}
bool segments_exact_equal(const xkv_segment & a, const xkv_segment & b) {
        if (a.segment_id != b.segment_id || a.segment_version != b.segment_version ||
            a.n_rows != b.n_rows || a.n_live_rows != b.n_live_rows ||
            a.profile != b.profile || a.source != b.source ||
            a.profile_fingerprint != b.profile_fingerprint ||
            a.source_fingerprint != b.source_fingerprint ||
            a.layer_group_map_fingerprint != b.layer_group_map_fingerprint ||
            a.descriptor_fingerprint != b.descriptor_fingerprint ||
            a.baseline_original_bytes != b.baseline_original_bytes ||
            a.row_payload_ids != b.row_payload_ids || a.live_rows != b.live_rows ||
            a.groups.size() != b.groups.size() ||
            !backend_handles_equal(a.backend_bundle, b.backend_bundle)) {
            return false;
        }
        for (size_t i = 0; i < a.groups.size(); ++i) {
            const auto & ga = a.groups[i];
            const auto & gb = b.groups[i];
            if (ga.group_index != gb.group_index || ga.owning_layers != gb.owning_layers ||
                ga.rank_k != gb.rank_k || ga.rank_v != gb.rank_v ||
                ga.layer_feature_offsets_k != gb.layer_feature_offsets_k ||
                ga.layer_feature_dims_k != gb.layer_feature_dims_k ||
                ga.layer_feature_offsets_v != gb.layer_feature_offsets_v ||
                ga.layer_feature_dims_v != gb.layer_feature_dims_v ||
                ga.total_dim_k != gb.total_dim_k || ga.total_dim_v != gb.total_dim_v ||
                ga.descriptor_fingerprint != gb.descriptor_fingerprint ||
                ga.config_fingerprint != gb.config_fingerprint ||
                ga.baseline_original_row_bytes != gb.baseline_original_row_bytes ||
                ga.baseline_original_bytes != gb.baseline_original_bytes ||
                !streams_equal(ga.a_k, gb.a_k) || !shared_stream_equal(ga.b_k, gb.b_k) ||
                !streams_equal(ga.a_v, gb.a_v) || !shared_stream_equal(ga.b_v, gb.b_v) ||
                !streams_equal(ga.landmark, gb.landmark)) {
                return false;
            }
        }
        return true;
}
} // namespace xkv_store_detail

bool llama_xkv_cache_store::execute_mutation_transaction(
    const xkv_batch_mutation & mutation,
    xkv_mutation_result * result,
    std::string * err
) {
    // Epoch reservation discipline: batch mutation never carries a token.
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (check_reservation_locked(0)) {
            if (err) *err = "execute_mutation_transaction: epoch reservation held by another transaction";
            if (result) {
                result->success = false;
                result->error = err ? *err : "";
            }
            return false;
        }
    }

    struct segment_cloning_plan {
        uint64_t seg_id = 0;
        std::shared_ptr<const xkv_segment> old_seg;
        std::shared_ptr<const xkv_segment> new_seg;
        std::vector<std::pair<uint64_t, uint32_t>> surviving_payload_rows;
    };

    std::vector<segment_cloning_plan> plans;
    xkv_snapshot_stamp snapshot_stamp = {};

    {
        std::lock_guard<std::mutex> lock(mtx);
        snapshot_stamp = stamp;

        if (mutation.payload_removals.empty() && mutation.segments_to_pack.empty()) {
            if (result) {
                result->success = true;
                result->stamp = stamp;
            }
            return true;
        }

        std::unordered_set<uint64_t> affected_segments(mutation.segments_to_pack.begin(), mutation.segments_to_pack.end());
        for (uint64_t pid : mutation.payload_removals) {
            auto it = payload_locations.find(pid);
            if (it != payload_locations.end() && it->second.kind == xkv_location_kind::factored && it->second.segment_id > 0) {
                affected_segments.insert(it->second.segment_id);
            }
        }

        for (uint64_t seg_id : affected_segments) {
            auto it = published_segments.find(seg_id);
            if (it == published_segments.end()) {
                if (err) *err = "preflight failed: published segment " + std::to_string(seg_id) + " not found";
                if (result) {
                    result->success = false;
                    result->error = err ? *err : "";
                }
                return false;
            }
            segment_cloning_plan plan;
            plan.seg_id = seg_id;
            plan.old_seg = it->second;
            plans.push_back(plan);
        }
    }

    std::unordered_set<uint64_t> removal_set(mutation.payload_removals.begin(), mutation.payload_removals.end());

    try {
        for (auto & plan : plans) {
            auto old_seg = plan.old_seg;

            std::vector<uint32_t> surviving_rows;
            surviving_rows.reserve(old_seg->n_rows);
            for (uint32_t r = 0; r < old_seg->n_rows; ++r) {
                if (old_seg->live_rows[r] && removal_set.find(old_seg->row_payload_ids[r]) == removal_set.end()) {
                    surviving_rows.push_back(r);
                }
            }

            if (!surviving_rows.empty()) {
                const uint32_t new_n_rows = (uint32_t) surviving_rows.size();

                auto new_seg = std::make_shared<xkv_segment>();
                new_seg->segment_id = old_seg->segment_id;
                new_seg->segment_version = old_seg->segment_version + 1;
                new_seg->profile = old_seg->profile;
                new_seg->source = old_seg->source;
                new_seg->residency = old_seg->residency;
                new_seg->layer_group_map_fingerprint = old_seg->layer_group_map_fingerprint;
                new_seg->profile_fingerprint = old_seg->profile_fingerprint;
                new_seg->source_fingerprint = old_seg->source_fingerprint;
                // Fail before any host decode/memcpy: DEVICE_OWNED mutation packs
                // only via backend-native transactions owned outside the store.
                if (old_seg->residency == GGML_XKV_RES_DEVICE_OWNED) {
                    throw std::runtime_error(
                        "segment " + std::to_string(old_seg->segment_id) +
                        " is DEVICE_OWNED: host mutation forbidden; requires backend-native byte-preserving pack (no host decode/memcpy)");
                        }
                new_seg->backend_bundle.reset(); // COW resets; host stays bundle-free.
                const bool is_landmarks = segment_needs_landmark_rebuild(*old_seg);

                if (is_landmarks && !mutation.landmark_rebuild) {
                    // Refusal path: never copy first-N landmark chunks after arbitrary
                    // deletion, and never duplicate the layout (byte budget). Without
                    // a semantic rebuild callback the mutation refuses with zero
                    // mutation; the old version stays valid and published.
                    throw std::runtime_error(
                        "LANDMARKS segment " + std::to_string(old_seg->segment_id) +
                        " requires a semantic landmark rebuild callback for mutation");
                } else {
                    new_seg->groups.resize(old_seg->groups.size());

                    for (size_t g_idx = 0; g_idx < old_seg->groups.size(); ++g_idx) {
                        const auto & old_g = old_seg->groups[g_idx];
                        auto & new_g = new_seg->groups[g_idx];

                        new_g.group_index = old_g.group_index;
                        new_g.owning_layers = old_g.owning_layers;
                        new_g.rank_k = old_g.rank_k;
                        new_g.rank_v = old_g.rank_v;
                        new_g.layer_feature_offsets_k = old_g.layer_feature_offsets_k;
                        new_g.layer_feature_dims_k = old_g.layer_feature_dims_k;
                        new_g.layer_feature_offsets_v = old_g.layer_feature_offsets_v;
                        new_g.layer_feature_dims_v = old_g.layer_feature_dims_v;
                        new_g.total_dim_k = old_g.total_dim_k;
                        new_g.total_dim_v = old_g.total_dim_v;
                        new_g.config_fingerprint = old_g.config_fingerprint;
                        // Immutable per-row baseline travels verbatim; the total is
                        // recomputed below for the survivor row count (checked).
                        new_g.baseline_original_row_bytes = old_g.baseline_original_row_bytes;
                        new_g.b_k = old_g.b_k; // Share immutable B handle across versions
                        new_g.b_v = old_g.b_v;

                        const size_t stride_a_k = old_g.a_k.desc.row_stride_bytes;
                        const size_t stride_a_v = old_g.a_v.desc.row_stride_bytes;

                        if (stride_a_k == 0 || stride_a_v == 0) {
                            throw std::runtime_error("segment " + std::to_string(old_seg->segment_id) +
                                " group " + std::to_string(old_g.group_index) + " has zero row stride; host mutation forbidden");
                        }
                        if (old_g.a_k.bytes.size() != (size_t) old_seg->n_rows * stride_a_k ||
                            old_g.a_v.bytes.size() != (size_t) old_seg->n_rows * stride_a_v) {
                            throw std::runtime_error("segment " + std::to_string(old_seg->segment_id) +
                                " group " + std::to_string(old_g.group_index) + " host byte size mismatch; host mutation forbidden");
                        }

                        new_g.a_k.desc = old_g.a_k.desc;
                        new_g.a_k.desc.logical_shape.rows = new_n_rows;
                        new_g.a_k.desc.padded_shape.rows = new_n_rows;
                        new_g.a_k.bytes.resize(new_n_rows * stride_a_k);

                        new_g.a_v.desc = old_g.a_v.desc;
                        new_g.a_v.desc.logical_shape.rows = new_n_rows;
                        new_g.a_v.desc.padded_shape.rows = new_n_rows;
                        new_g.a_v.bytes.resize(new_n_rows * stride_a_v);

                        for (uint32_t dst_r = 0; dst_r < new_n_rows; ++dst_r) {
                            uint32_t src_r = surviving_rows[dst_r];
                            if (src_r >= old_seg->n_rows) {
                                throw std::runtime_error("segment " + std::to_string(old_seg->segment_id) +
                                    " surviving row out of range; host mutation forbidden");
                    }
                            std::memcpy(new_g.a_k.bytes.data() + dst_r * stride_a_k,
                                        old_g.a_k.bytes.data() + src_r * stride_a_k,
                                        stride_a_k);
                            std::memcpy(new_g.a_v.bytes.data() + dst_r * stride_a_v,
                                        old_g.a_v.bytes.data() + src_r * stride_a_v,
                                        stride_a_v);
                        }
                        new_g.landmark = encoded_matrix();
                        uint64_t packed_total = 0;
                        if (!checked_mul_u64((uint64_t) new_g.baseline_original_row_bytes,
                                             (uint64_t) new_n_rows, packed_total)) {
                            throw std::runtime_error("baseline byte total overflow for group " +
                                std::to_string(new_g.group_index));
                        }
                        new_g.baseline_original_bytes = packed_total;
                    }

                    new_seg->row_payload_ids.reserve(new_n_rows);
                    new_seg->live_rows.assign(new_n_rows, true);
                    new_seg->n_rows = new_n_rows;
                    new_seg->n_live_rows = new_n_rows;

                    for (uint32_t dst_r = 0; dst_r < new_n_rows; ++dst_r) {
                        uint32_t src_r = surviving_rows[dst_r];
                        uint64_t pid = old_seg->row_payload_ids[src_r];
                        new_seg->row_payload_ids.push_back(pid);
                        plan.surviving_payload_rows.push_back({pid, dst_r});
                    }

                    if (is_landmarks) {
                        std::string rebuild_err;
                        if (!mutation.landmark_rebuild(*old_seg, surviving_rows, *new_seg, &rebuild_err)) {
                            throw std::runtime_error(
                                "landmark rebuild refused pack for segment " +
                                std::to_string(old_seg->segment_id) + ": " + rebuild_err);
                        }
                    }
                    for (auto & new_g : new_seg->groups) {
                        new_g.refresh_descriptor_fingerprint();
                        new_g.update_byte_counters();
                    }
                    new_seg->update_byte_counters();
                    {
                        std::string cerr;
                        if (!validate_candidate(new_seg, &cerr)) {
                            throw std::runtime_error(
                                "rebuilt LANDMARKS bundle failed validation: " + cerr);
                        }
                    }
                    plan.new_seg = new_seg;
                }
            }
        }
        // Single precommit failure-injection phase shared by single- and
        // multi-segment transactions: fires once after all clones are prepared
        // and before the atomic commit lock is taken.
        if (mutation.test_failure_hook) {
            mutation.test_failure_hook();
        }
    } catch (const std::exception & e) {
        if (err) *err = std::string("transaction preparation failed: ") + e.what();
        if (result) {
            result->success = false;
            result->error = err ? *err : "";
        }
        return false;
    } catch (...) {
        if (err) *err = "transaction preparation failed with an unknown exception";
        if (result) {
            result->success = false;
            result->error = err ? *err : "";
        }
        return false;
    }

    // Phase 2: Re-lock and commit atomically
    std::vector<uint64_t> new_versions;
    std::vector<uint64_t> affected_list;

    // Prebuild the removal gate context off-lock: copies may allocate, and
    // after the gate returns true nothing may throw or allocate.
    xkv_removal_precommit_ctx pre_ctx;
    try {
        pre_ctx.payload_ids = mutation.payload_removals;
        pre_ctx.affected_segment_ids.reserve(plans.size());
        pre_ctx.new_segment_versions.reserve(plans.size());
        for (const auto & plan : plans) {
            pre_ctx.affected_segment_ids.push_back(plan.seg_id);
            pre_ctx.new_segment_versions.push_back(plan.new_seg ? plan.new_seg->segment_version : 0);
        }
    } catch (const std::exception & e) {
        if (err) *err = std::string("mutation gate context allocation failed: ") + e.what();
        if (result) {
            result->success = false;
            result->error = err ? *err : "";
        }
        return false;
    } catch (...) {
        if (err) *err = "mutation gate context allocation failed with an unknown exception";
        if (result) {
            result->success = false;
            result->error = err ? *err : "";
        }
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(mtx);

        if (stamp != snapshot_stamp) {
            if (err) *err = "concurrent transaction stamp conflict";
            if (result) {
                result->success = false;
                result->error = err ? *err : "";
            }
            return false;
        }

        for (const auto & plan : plans) {
            auto seg_it = published_segments.find(plan.seg_id);
            if (seg_it == published_segments.end() || seg_it->second != plan.old_seg) {
                if (err) *err = "concurrent transaction pointer conflict";
                if (result) {
                    result->success = false;
                    result->error = err ? *err : "";
                }
                return false;
            }
        }

        // Caught allocation preflight: reserves run before the gate; anything
        // that throws aborts with zero mutation.
        try {
            new_versions.reserve(plans.size());
            affected_list.reserve(plans.size());
            retired_segments.reserve(retired_segments.size() + plans.size());
        } catch (const std::exception & e) {
            if (err) *err = std::string("mutation commit preflight allocation failed: ") + e.what();
            if (result) {
                result->success = false;
                result->error = err ? *err : "";
            }
            return false;
        }

        // Hard store cap: COW peak must fit before any mutation.
        try {
            std::vector<std::shared_ptr<const xkv_segment>> cap_extras;
            cap_extras.reserve(plans.size());
            for (const auto & plan : plans) {
                if (plan.new_seg) cap_extras.push_back(plan.new_seg);
            }
            if (!ensure_accounting_scratch_locked(cap_extras, err)) {
                if (result) { result->success = false; result->error = err ? *err : ""; }
                return false;
            }
            if (!store_capacity_fits_locked(cap_extras, err)) {
                if (result) {
                    result->success = false;
                    result->error = err ? *err : "";
                }
                return false;
            }
        } catch (const std::exception & e) {
            if (err) *err = std::string("mutation capacity check failed: ") + e.what();
            if (result) {
                result->success = false;
                result->error = err ? *err : "";
            }
            return false;
        } catch (...) {
            if (err) *err = "mutation capacity check failed with an unknown exception";
            if (result) {
                result->success = false;
                result->error = err ? *err : "";
            }
            return false;
        }

        // Preflight all required epoch increments BEFORE changing maps/bytes;
        // overflow fails atomically with zero partial state (never skip-a-bump).
        uint8_t mut_mask = bump_flag_binding;
        if (!mutation.payload_removals.empty()) {
            mut_mask |= bump_flag_live;
        }
        if (!can_bump_locked(mut_mask, err)) {
            if (result) {
                result->success = false;
                result->error = err ? *err : "";
            }
            return false;
        }

        // Removal precommit gate (also fires for pure packs so the owner
        // observes every version bump): false/throw refuses with zero mutation;
        // after true the commit below performs no allocation and cannot fail.
        // The callback MUST NOT re-enter the store (lock held).
        if (mutation.removal_precommit) {
            std::string pre_err;
            bool pre_ok = false;
            try {
                pre_ok = mutation.removal_precommit(pre_ctx, &pre_err);
            } catch (const std::exception & e) {
                pre_err = e.what();
                pre_ok = false;
            } catch (...) {
                pre_err = "unknown non-standard exception";
                pre_ok = false;
            }
            if (!pre_ok) {
                if (err) *err = "removal precommit refused commit: " + pre_err;
                if (result) {
                    result->success = false;
                    result->error = err ? *err : "";
                }
                return false;
            }
        }

        for (const auto & plan : plans) {
            auto seg_it = published_segments.find(plan.seg_id);
            if (plan.new_seg) {
                seg_it->second = plan.new_seg;
                for (const auto & pr : plan.surviving_payload_rows) {
                    auto loc_it = payload_locations.find(pr.first);
                    if (loc_it != payload_locations.end()) {
                        loc_it->second.segment_version = plan.new_seg->segment_version;
                        loc_it->second.row = pr.second;
                    }
                }
                new_versions.push_back(plan.new_seg->segment_version);
            } else {
                published_segments.erase(seg_it);
                new_versions.push_back(0);
            }
            retired_segments.push_back(plan.old_seg);
            affected_list.push_back(plan.seg_id);
        }

        for (uint64_t pid : mutation.payload_removals) {
            payload_locations.erase(pid);
        }

        bump_prechecked(mut_mask);

        reclaim_retired_segments_locked();

        if (result) {
            result->success = true;
            // No-throw tail: vectors were prereserved and filled in capacity.
            result->affected_segments = std::move(affected_list);
            result->new_segment_versions = std::move(new_versions);
            result->stamp = stamp;
        }
    }

    return true;
}

void llama_xkv_cache_store::register_layer_alias(uint32_t model_layer, uint32_t owning_layer) {
    std::lock_guard<std::mutex> lock(mtx);
    layer_aliases[model_layer] = owning_layer;
}

uint32_t llama_xkv_cache_store::resolve_owning_layer(uint32_t model_layer) const {
    std::lock_guard<std::mutex> lock(mtx);
    auto it = layer_aliases.find(model_layer);
    return it != layer_aliases.end() ? it->second : model_layer;
}

xkv_accounting llama_xkv_cache_store::get_accounting() const {
    std::lock_guard<std::mutex> lock(mtx);
    xkv_accounting acc;
    acc.total_payloads = payload_locations.size();
    acc.active_segments = published_segments.size();
    acc.codec_shared_bytes = llama_xkv_codec_shared_table_bytes();

    std::unordered_set<const encoded_matrix *> unique_b_k;
    std::unordered_set<const encoded_matrix *> unique_b_v;
    std::unordered_set<const encoded_matrix *> unique_live_b;
    std::unordered_set<uint64_t> accounted_backend_alloc_ids;
    std::unordered_set<uint64_t> accounted_live_backend_alloc_ids;

    // Compute padding bytes: row-stride padded bytes minus logical bytes
    auto add_padding = [&](const codec_desc & d, size_t row_count) {
        if (d.logical_shape.rows > 0 && d.padded_shape.cols > d.logical_shape.cols) {
            const size_t logical_row_bytes = (size_t) d.logical_shape.cols * ggml_type_size(d.type) / (size_t) std::max<int64_t>(1, ggml_blck_size(d.type));
            if (d.row_stride_bytes > logical_row_bytes) {
                acc.factor_padding_bytes += row_count * (d.row_stride_bytes - logical_row_bytes);
            }
        }
    };

    for (const auto & [_, seg] : published_segments) {
        if (seg->pin_count.load(std::memory_order_relaxed) > 0) {
            acc.pinned_segments++;
        }

        acc.index_bytes += seg->bytes_metadata_logical;
        acc.factor_metadata_bytes += seg->bytes_metadata_logical;
        acc.live_payload_bytes += seg->bytes_metadata_logical;
        acc.allocated_bytes += seg->bytes_metadata_logical;

        const bool is_device = (seg->residency == GGML_XKV_RES_DEVICE_OWNED && seg->backend_bundle != nullptr);

        for (const auto & g : seg->groups) {
            if (!is_device && g.b_k) {
                unique_b_k.insert(g.b_k.get());
                unique_live_b.insert(g.b_k.get());
            }
            if (!is_device && g.b_v) {
                unique_b_v.insert(g.b_v.get());
                unique_live_b.insert(g.b_v.get());
            }

            const size_t live_a_k = (size_t) seg->n_live_rows * g.a_k.desc.row_stride_bytes;
            const size_t live_a_v = (size_t) seg->n_live_rows * g.a_v.desc.row_stride_bytes;
            acc.live_payload_bytes += live_a_k + live_a_v + (is_device ? g.bytes_landmark : g.landmark.bytes.size()) + g.bytes_metadata;
            acc.factor_live_bytes  += (is_device ? (live_a_k + live_a_v) : (live_a_k + live_a_v));
            acc.factor_metadata_bytes += g.bytes_metadata;
            acc.landmark_payload_bytes += (is_device ? g.bytes_landmark : g.landmark.bytes.size());
            acc.landmark_metadata_bytes += g.landmark_chunks.size() * sizeof(xkv_landmark_chunk);

            // Baseline-equivalent original bytes for LIVE rows actually factored:
            acc.baseline_factored_bytes += (size_t) seg->n_live_rows * (size_t) g.baseline_original_row_bytes;

            add_padding(g.a_k.desc, (size_t) seg->n_live_rows);
            add_padding(g.a_v.desc, (size_t) seg->n_live_rows);
            if (g.b_k) add_padding(g.b_k->desc, (size_t) g.b_k->desc.logical_shape.rows);
            if (g.b_v) add_padding(g.b_v->desc, (size_t) g.b_v->desc.logical_shape.rows);

            // FP16-equivalent factor bytes for LIVE rows: (n_live_rows*rk + n_live_rows*rv)*sizeof(ggml_fp16_t)
            acc.factor_fp16_equivalent_bytes += ((size_t) seg->n_live_rows * g.rank_k +
                                                  (size_t) seg->n_live_rows * g.rank_v) * sizeof(ggml_fp16_t);

            if (!is_device) {
                acc.allocated_bytes += g.a_k.bytes.size() + g.a_v.bytes.size() +
                                       g.landmark.bytes.size() + g.bytes_metadata;
            } else {
                // Device segment: metadata only on host side; handle allocations added below
                acc.allocated_bytes += g.bytes_metadata;
            }
        }

        // For DEVICE_OWNED segment, account all device handles deduplicated by allocation ID
        if (is_device) {
            for (const auto & h : seg->backend_bundle->handles) {
                if (!h) continue;
                uint64_t aid = h->get_allocation_id();
                if (accounted_backend_alloc_ids.insert(aid).second) {
                    acc.allocated_bytes += h->get_actual_bytes();
                    const auto & d = h->get_desc();
                    if (d.role == factor_role::b_k || d.role == factor_role::b_v) {
                        acc.shared_b_bytes += h->get_actual_bytes();
                        acc.factor_fp16_equivalent_bytes += ((size_t) d.logical_shape.rows * d.logical_shape.cols) * sizeof(ggml_fp16_t);
                        add_padding(d, (size_t) d.logical_shape.rows);
                    }
                }
                if (accounted_live_backend_alloc_ids.insert(aid).second) {
                    const auto & d = h->get_desc();
                    if (d.role == factor_role::b_k || d.role == factor_role::b_v) {
                        acc.live_payload_bytes += h->get_actual_bytes();
                        acc.factor_live_bytes  += h->get_actual_bytes();
                    }
                }
            }
        }
    }

    for (const auto * b : unique_live_b) {
        acc.live_payload_bytes += b->bytes.size();
        acc.factor_live_bytes += b->bytes.size();
    }

    for (const auto & seg : retired_segments) {
        if (seg->pin_count.load(std::memory_order_relaxed) > 0) {
            acc.pinned_segments++;
            acc.snapshot_pinned_bytes += seg->total_allocated_bytes;
        }

        acc.allocated_bytes += seg->bytes_metadata_logical;

        const bool is_device = (seg->residency == GGML_XKV_RES_DEVICE_OWNED && seg->backend_bundle != nullptr);

        for (const auto & g : seg->groups) {
            if (!is_device && g.b_k) unique_b_k.insert(g.b_k.get());
            if (!is_device && g.b_v) unique_b_v.insert(g.b_v.get());

            if (!is_device) {
                acc.allocated_bytes += g.a_k.bytes.size() + g.a_v.bytes.size() +
                                       g.landmark.bytes.size() + g.bytes_metadata;
            } else {
                acc.allocated_bytes += g.bytes_metadata;
            }
        }

        if (is_device) {
            for (const auto & h : seg->backend_bundle->handles) {
                if (!h) continue;
                uint64_t aid = h->get_allocation_id();
                if (accounted_backend_alloc_ids.insert(aid).second) {
                    acc.allocated_bytes += h->get_actual_bytes();
                    const auto & d = h->get_desc();
                    if (d.role == factor_role::b_k || d.role == factor_role::b_v) {
                        acc.shared_b_bytes += h->get_actual_bytes();
                        acc.factor_fp16_equivalent_bytes += ((size_t) d.logical_shape.rows * d.logical_shape.cols) * sizeof(ggml_fp16_t);
                        add_padding(d, (size_t) d.logical_shape.rows);
                    }
                }
            }
        }
    }

    for (const auto * b : unique_b_k) {
        acc.shared_b_bytes += b->bytes.size();
        acc.allocated_bytes += b->bytes.size();
        acc.factor_fp16_equivalent_bytes += ((size_t) b->desc.logical_shape.rows * b->desc.logical_shape.cols) * sizeof(ggml_fp16_t);
    }
    for (const auto * b : unique_b_v) {
        acc.shared_b_bytes += b->bytes.size();
        acc.allocated_bytes += b->bytes.size();
        acc.factor_fp16_equivalent_bytes += ((size_t) b->desc.logical_shape.rows * b->desc.logical_shape.cols) * sizeof(ggml_fp16_t);
    }
    // Unique B matrices = host unique B + device unique B (tracked by allocation ID)
    std::unordered_set<uint64_t> unique_dev_b_ids;
    for (const auto & [_, s] : published_segments) {
        if (s->residency == GGML_XKV_RES_DEVICE_OWNED && s->backend_bundle) {
            for (const auto & h : s->backend_bundle->handles) {
                if (h && (h->get_desc().role == factor_role::b_k || h->get_desc().role == factor_role::b_v)) {
                    unique_dev_b_ids.insert(h->get_allocation_id());
                }
            }
        }
    }
    for (const auto & s : retired_segments) {
        if (s->residency == GGML_XKV_RES_DEVICE_OWNED && s->backend_bundle) {
            for (const auto & h : s->backend_bundle->handles) {
                if (h && (h->get_desc().role == factor_role::b_k || h->get_desc().role == factor_role::b_v)) {
                    unique_dev_b_ids.insert(h->get_allocation_id());
                }
            }
        }
    }
    acc.unique_b_matrices = unique_b_k.size() + unique_b_v.size() + unique_dev_b_ids.size();
    acc.aliased_payloads = layer_aliases.size();

    acc.factored_bytes = acc.allocated_bytes;

    acc.arena_live_bytes = workspace_arena.get_live_bytes();
    acc.arena_reserved_bytes = workspace_arena.get_reserved_bytes();
    acc.arena_peak_bytes = workspace_arena.get_peak_bytes();
    acc.arena_capacity_bytes = workspace_arena.get_capacity_bytes();

    // Include store-owned scratch capacity bytes in reserved memory accounting
    const size_t scratch_meta_bytes = scratch_unique_b_k_.capacity() * sizeof(const encoded_matrix *) +
                                      scratch_unique_b_v_.capacity() * sizeof(const encoded_matrix *) +
                                      scratch_backend_alloc_ids_.capacity() * sizeof(uint64_t) +
                                      scratch_backend_alloc_accounted_.capacity() * sizeof(uint8_t);
    acc.dedup_scratch_bytes = scratch_meta_bytes;
    acc.reserved_bytes = acc.allocated_bytes + acc.arena_reserved_bytes + scratch_meta_bytes;
    acc.workspace_budget_bytes = acc.arena_capacity_bytes;

    // Split allocated by residency and track peak high-water including published + retired
    size_t cur_device = 0;
    size_t cur_host = 0;
    for (const auto & [_, seg] : published_segments) {
        if (seg->residency == GGML_XKV_RES_DEVICE_OWNED) cur_device += seg->total_allocated_bytes;
        else cur_host += seg->total_allocated_bytes;
    }
    for (const auto & seg : retired_segments) {
        if (seg->residency == GGML_XKV_RES_DEVICE_OWNED) cur_device += seg->total_allocated_bytes;
        else cur_host += seg->total_allocated_bytes;
    }
    if (cur_host > host_peak_bytes_) host_peak_bytes_ = cur_host;
    if (cur_device > device_peak_bytes_) device_peak_bytes_ = cur_device;
    acc.host_peak_bytes = host_peak_bytes_;
    acc.device_peak_bytes = device_peak_bytes_;

    return acc;
}

size_t llama_xkv_cache_store::store_capacity_bytes() const {
    uint64_t bytes = 0;
    if (cparams.xkv_store_mib == 0) {
        return 0; // Unlimited.
    }
    if (!checked_mul_u64((uint64_t) cparams.xkv_store_mib, 1024ULL * 1024ULL, bytes) ||
        bytes > (uint64_t) std::numeric_limits<size_t>::max()) {
        return std::numeric_limits<size_t>::max(); // Saturated: effectively unlimited.
    }
    return (size_t) bytes;
}

bool llama_xkv_cache_store::deduplicated_allocated_locked(
    const std::vector<std::shared_ptr<const xkv_segment>> & extras, size_t & out_bytes) const {
    // Mirrors get_accounting's allocated walk exactly (published + retired +
    // unique B_K / B_V once each + unique backend allocation handles), plus pending extras for the COW peak.
    // Uses preallocated scratch vectors with linear dedup to guarantee ZERO heap allocations under mtx.
    // O(N log N) collect + sort + unique on preallocated member vectors (no hash table, no O(N^2) scan).
    // Strict cardinality check: in read-only paths (or if scratch was not pre-warmed to exact size),
    // fail closed before any push_back if capacity is insufficient. This prevents hidden allocations under mtx!
    size_t count_b = 0;
    size_t count_backend = 0;
    auto verify_cap = [&](const std::shared_ptr<const xkv_segment> & seg) {
        if (!seg) return;
        for (const auto & g : seg->groups) {
            if (g.b_k) { if (count_b < std::numeric_limits<size_t>::max()) ++count_b; }
            if (g.b_v) { if (count_b < std::numeric_limits<size_t>::max()) ++count_b; }
        }
        if (seg->backend_bundle) {
            for (const auto & h : seg->backend_bundle->handles) {
                if (h && count_backend < std::numeric_limits<size_t>::max()) ++count_backend;
            }
        }
    };
    for (const auto & kv : published_segments) verify_cap(kv.second);
    for (const auto & s : retired_segments) verify_cap(s);
    for (const auto & s : extras) verify_cap(s);

    if (scratch_unique_b_k_.capacity() < count_b ||
        scratch_unique_b_v_.capacity() < count_b ||
        scratch_backend_alloc_ids_.capacity() < count_backend ||
        scratch_backend_alloc_accounted_.capacity() < count_backend) {
        return false; // Fail closed without heap allocation!
    }

    scratch_unique_b_k_.clear();
    scratch_unique_b_v_.clear();
    scratch_backend_alloc_ids_.clear();
    scratch_backend_alloc_accounted_.clear();
    size_t bytes = 0;
    auto acc_seg = [&](const std::shared_ptr<const xkv_segment> & seg) -> bool {
        size_t t = 0;
        if (!checked_add_size(bytes, seg->bytes_metadata_logical, t)) return false;
        bytes = t;
        const bool is_device = (seg->residency == GGML_XKV_RES_DEVICE_OWNED && seg->backend_bundle != nullptr);
        for (const auto & g : seg->groups) {
            if (!is_device) {
                if (g.b_k) scratch_unique_b_k_.push_back(g.b_k.get());
                if (g.b_v) scratch_unique_b_v_.push_back(g.b_v.get());
                size_t parts[4] = {g.a_k.bytes.size(), g.a_v.bytes.size(),
                                   g.landmark.bytes.size(), g.bytes_metadata};
                for (size_t p : parts) {
                    if (!checked_add_size(bytes, p, t)) return false;
                    bytes = t;
                }
            } else {
                if (!checked_add_size(bytes, g.bytes_metadata, t)) return false;
                bytes = t;
            }
        }
        if (is_device) {
            for (const auto & h : seg->backend_bundle->handles) {
                if (!h) continue;
                scratch_backend_alloc_ids_.push_back(h->get_allocation_id());
                // Device handle byte dedup will be billed below after sort+unique
            }
        }
        return true;
    };
    for (const auto & kv : published_segments) {
        if (!acc_seg(kv.second)) return false;
    }
    for (const auto & seg : retired_segments) {
        if (!acc_seg(seg)) return false;
    }
    for (const auto & seg : extras) {
        if (seg && !acc_seg(seg)) return false;
    }

    // O(N log N) sort + unique on scratch vectors without allocating memory
    std::sort(scratch_unique_b_k_.begin(), scratch_unique_b_k_.end(), std::less<const encoded_matrix*>());
    scratch_unique_b_k_.erase(std::unique(scratch_unique_b_k_.begin(), scratch_unique_b_k_.end()), scratch_unique_b_k_.end());
    for (const auto * b : scratch_unique_b_k_) {
        if (b == nullptr) continue;
        size_t t = 0;
        if (!checked_add_size(bytes, b->bytes.size(), t)) return false;
        bytes = t;
    }

    std::sort(scratch_unique_b_v_.begin(), scratch_unique_b_v_.end(), std::less<const encoded_matrix*>());
    scratch_unique_b_v_.erase(std::unique(scratch_unique_b_v_.begin(), scratch_unique_b_v_.end()), scratch_unique_b_v_.end());
    for (const auto * b : scratch_unique_b_v_) {
        if (b == nullptr) continue;
        size_t t = 0;
        if (!checked_add_size(bytes, b->bytes.size(), t)) return false;
        bytes = t;
    }

    if (!scratch_backend_alloc_ids_.empty()) {
        std::sort(scratch_backend_alloc_ids_.begin(), scratch_backend_alloc_ids_.end());
        scratch_backend_alloc_ids_.erase(std::unique(scratch_backend_alloc_ids_.begin(), scratch_backend_alloc_ids_.end()), scratch_backend_alloc_ids_.end());
        scratch_backend_alloc_accounted_.assign(scratch_backend_alloc_ids_.size(), 0);
        // Lookup actual bytes once per unique allocation id
        auto bill_dev_h = [&](const std::shared_ptr<const xkv_segment> & seg) -> bool {
            if (!seg || seg->residency != GGML_XKV_RES_DEVICE_OWNED || !seg->backend_bundle) return true;
            for (const auto & h : seg->backend_bundle->handles) {
                if (!h) continue;
                uint64_t aid = h->get_allocation_id();
                auto it = std::lower_bound(scratch_backend_alloc_ids_.begin(), scratch_backend_alloc_ids_.end(), aid);
                if (it != scratch_backend_alloc_ids_.end() && *it == aid) {
                    size_t idx = static_cast<size_t>(std::distance(scratch_backend_alloc_ids_.begin(), it));
                    if (!scratch_backend_alloc_accounted_[idx]) {
                        scratch_backend_alloc_accounted_[idx] = 1; // Mark accounted without mutating sorted IDs
                    size_t t = 0;
                    if (!checked_add_size(bytes, h->get_actual_bytes(), t)) return false;
                    bytes = t;
                    }
                }
            }
            return true;
        };
        for (const auto & kv : published_segments) if (!bill_dev_h(kv.second)) return false;
        for (const auto & seg : retired_segments) if (!bill_dev_h(seg)) return false;
        for (const auto & seg : extras) if (!bill_dev_h(seg)) return false;
    }
    out_bytes = bytes;
    return true;
}

bool llama_xkv_cache_store::store_capacity_fits_locked(
    const std::vector<std::shared_ptr<const xkv_segment>> & extras, std::string * err,
    size_t exclude_reserved_bytes) const {
    const size_t cap = store_capacity_bytes();
    const size_t scratch_bytes = scratch_unique_b_k_.capacity() * sizeof(const encoded_matrix *) + scratch_unique_b_v_.capacity() * sizeof(const encoded_matrix *) + scratch_backend_alloc_ids_.capacity() * sizeof(uint64_t) + scratch_backend_alloc_accounted_.capacity() * sizeof(uint8_t);
    if (cap == 0) {
        return true; // Unlimited.
    }
    size_t total = 0;
    if (!deduplicated_allocated_locked(extras, total)) {
        if (err) *err = "store capacity accounting overflow";
        return false;
    }
    size_t pending_effective = pending_reserved_store_bytes_;
    if (exclude_reserved_bytes > 0) {
        pending_effective = (pending_effective >= exclude_reserved_bytes) ? (pending_effective - exclude_reserved_bytes) : 0;
    }
    if (!checked_add_size(total, pending_effective, total)) {
        if (err) *err = "store capacity accounting overflow with pending reservations";
        return false;
    }
    if (!checked_add_size(total, scratch_bytes, total)) {
        if (err) *err = "store capacity accounting overflow with scratch";
        return false;
    }
    if (total > cap) {
        if (err) *err = "store capacity exceeded";
        return false;
    }
    return true;
}

xkv_capacity_reservation::~xkv_capacity_reservation() {
    release();
}

xkv_capacity_reservation::xkv_capacity_reservation(xkv_capacity_reservation && o) noexcept
    : store_(o.store_), token_(o.token_), reserved_bytes_(o.reserved_bytes_),
      expected_removal_payload_ids_(std::move(o.expected_removal_payload_ids_)),
      expected_removal_generations_(std::move(o.expected_removal_generations_)) {
    o.store_ = nullptr;
    o.token_ = 0;
    o.reserved_bytes_ = 0;
    o.expected_removal_payload_ids_.clear();
    o.expected_removal_generations_.clear();
}

xkv_capacity_reservation & xkv_capacity_reservation::operator=(xkv_capacity_reservation && o) noexcept {
    if (this != &o) {
        release();
        store_ = o.store_;
        token_ = o.token_;
        reserved_bytes_ = o.reserved_bytes_;
        expected_removal_payload_ids_ = std::move(o.expected_removal_payload_ids_);
        expected_removal_generations_ = std::move(o.expected_removal_generations_);
        o.store_ = nullptr;
        o.token_ = 0;
        o.reserved_bytes_ = 0;
        o.expected_removal_payload_ids_.clear();
        o.expected_removal_generations_.clear();
    }
    return *this;
}

void xkv_capacity_reservation::release() noexcept {
    if (store_ != nullptr && token_ != 0) {
        store_->release_capacity_reservation_locked(token_);
        store_ = nullptr;
        token_ = 0;
        reserved_bytes_ = 0;
        expected_removal_payload_ids_.clear();
        expected_removal_generations_.clear();
    }
}

xkv_backend_store_reservation xkv_capacity_reservation::backend_reservation() const noexcept {
    xkv_backend_store_reservation r;
    r.reserved_bytes = (uint64_t) reserved_bytes_;
    if (store_ != nullptr) {
        r.cap_bytes = (uint64_t) store_->store_capacity_bytes();
    }
    return r;
}

void llama_xkv_cache_store::release_capacity_reservation_locked(uint64_t token) noexcept {
    std::lock_guard<std::mutex> lock(mtx);
    auto it = pending_reservations_.find(token);
    if (it != pending_reservations_.end()) {
        if (pending_reserved_store_bytes_ >= it->second) {
            pending_reserved_store_bytes_ -= it->second;
        } else {
            pending_reserved_store_bytes_ = 0;
        }
        pending_reservations_.erase(it);
    }
}

bool llama_xkv_cache_store::preflight_store_capacity(
    size_t expected_bytes, size_t * out_deficit, std::string * err) const {
    std::lock_guard<std::mutex> lock(mtx);
    const size_t cap = store_capacity_bytes();
    if (cap == 0) {
        if (out_deficit) *out_deficit = 0;
    return true;
    }

    size_t base_alloc = 0;
    if (!published_segments.empty() || !retired_segments.empty()) {
        std::vector<std::shared_ptr<const xkv_segment>> none;
        if (!deduplicated_allocated_locked(none, base_alloc)) {
            if (err) *err = "store capacity preflight accounting overflow";
            return false;
        }
    }

    size_t total = 0;
    if (!checked_add_size(base_alloc, pending_reserved_store_bytes_, total) ||
        !checked_add_size(total, expected_bytes, total)) {
        if (err) *err = "store capacity preflight addition overflow";
        if (out_deficit) *out_deficit = std::numeric_limits<size_t>::max();
        return false;
    }

    if (total > cap) {
        if (out_deficit) *out_deficit = total - cap;
        if (err) *err = "store capacity exceeded: required " + std::to_string(total) + " > cap " + std::to_string(cap);
        return false;
    }

    if (out_deficit) *out_deficit = 0;
    return true;
}

xkv_capacity_reservation llama_xkv_cache_store::reserve_capacity(
    size_t expected_bytes, std::string * err, size_t * out_deficit,
    const std::vector<uint64_t> & expected_removal_pids,
    const std::vector<uint64_t> & expected_removal_gens) {
    std::lock_guard<std::mutex> lock(mtx);
    const size_t cap = store_capacity_bytes();

    // Validate expected removals under lock if provided
    if (!expected_removal_pids.empty()) {
        if (!expected_removal_gens.empty() && expected_removal_gens.size() != expected_removal_pids.size()) {
            if (err) *err = "reserve_capacity: expected removal pids and generations size mismatch";
            return xkv_capacity_reservation();
        }
        for (size_t i = 0; i < expected_removal_pids.size(); ++i) {
            uint64_t pid = expected_removal_pids[i];
            auto it = payload_locations.find(pid);
            if (it == payload_locations.end()) {
                if (err) *err = "reserve_capacity: expected removal payload not found: " + std::to_string(pid);
            return xkv_capacity_reservation();
    }
            if (!expected_removal_gens.empty() && it->second.storage_generation != expected_removal_gens[i]) {
                if (err) *err = "reserve_capacity: expected removal generation mismatch for payload: " + std::to_string(pid);
            return xkv_capacity_reservation();
    }
        }
    }

    if (cap != 0) {
        size_t base_alloc = 0;
        std::vector<std::shared_ptr<const xkv_segment>> none;
        if (!deduplicated_allocated_locked(none, base_alloc)) {
            if (err) *err = "reserve_capacity: accounting overflow";
            return xkv_capacity_reservation();
        }

    size_t total = 0;
        if (!checked_add_size(base_alloc, pending_reserved_store_bytes_, total) ||
            !checked_add_size(total, expected_bytes, total)) {
            if (err) *err = "reserve_capacity: addition overflow";
            if (out_deficit) *out_deficit = std::numeric_limits<size_t>::max();
            return xkv_capacity_reservation();
        }

    if (total > cap) {
            if (out_deficit) *out_deficit = total - cap;
            if (err) *err = "reserve_capacity: store capacity exceeded";
            return xkv_capacity_reservation();
        }
    }

    // Check token space exhaustion: fail closed before wrap
    if (next_cap_res_token_ == 0 || next_cap_res_token_ == UINT64_MAX) {
        if (err) *err = "reserve_capacity: token counter exhausted";
        return xkv_capacity_reservation();
    }

    size_t new_pending = 0;
    if (!checked_add_size(pending_reserved_store_bytes_, expected_bytes, new_pending)) {
        if (err) *err = "reserve_capacity: pending bytes overflow";
        return xkv_capacity_reservation();
    }

    uint64_t token = next_cap_res_token_;
    // Preinsert into unordered_map before updating ledger so any bad_alloc leaves state clean
    try {
        pending_reservations_.emplace(token, expected_bytes);
    } catch (const std::bad_alloc &) {
        // Under allocator fault injection (g_fail_alloc), avoid dynamic string construction
        // which throws recursively and aborts. Return invalid reservation cleanly.
        return xkv_capacity_reservation();
    } catch (const std::exception & e) {
        if (err) *err = std::string("reserve_capacity: map insertion failed: ") + e.what();
        return xkv_capacity_reservation();
    } catch (...) {
        return xkv_capacity_reservation();
    }

    next_cap_res_token_++;
    pending_reserved_store_bytes_ = new_pending;
    if (out_deficit) *out_deficit = 0;

    return xkv_capacity_reservation(this, token, expected_bytes, expected_removal_pids, expected_removal_gens);
}

xkv_device_staging_reservation::~xkv_device_staging_reservation() {
    release();
}

xkv_device_staging_reservation::xkv_device_staging_reservation(xkv_device_staging_reservation && o) noexcept
    : store_(o.store_), reserved_bytes_(o.reserved_bytes_) {
    o.store_ = nullptr;
    o.reserved_bytes_ = 0;
}

xkv_device_staging_reservation & xkv_device_staging_reservation::operator=(xkv_device_staging_reservation && o) noexcept {
    if (this != &o) {
        release();
        store_ = o.store_;
        reserved_bytes_ = o.reserved_bytes_;
        o.store_ = nullptr;
        o.reserved_bytes_ = 0;
    }
    return *this;
}

void xkv_device_staging_reservation::release() noexcept {
    if (store_ != nullptr && reserved_bytes_ > 0) {
        store_->release_device_staging_reservation_locked(reserved_bytes_);
        store_ = nullptr;
        reserved_bytes_ = 0;
    }
}

void llama_xkv_cache_store::release_device_staging_reservation_locked(size_t bytes) noexcept {
    std::lock_guard<std::mutex> lock(mtx);
    if (device_staging_reserved_bytes_ >= bytes) {
        device_staging_reserved_bytes_ -= bytes;
    } else {
        device_staging_reserved_bytes_ = 0;
    }
}

xkv_device_staging_reservation llama_xkv_cache_store::reserve_device_staging(
    size_t bytes, std::string * err, size_t * out_deficit) {
    std::lock_guard<std::mutex> lock(mtx);
    const size_t arena_cap = workspace_arena.get_capacity_bytes();
    const size_t arena_live = workspace_arena.get_live_bytes();

    size_t total_transient = 0;
    if (!checked_add_size(arena_live, device_staging_reserved_bytes_, total_transient) ||
        !checked_add_size(total_transient, bytes, total_transient)) {
        if (err) *err = "reserve_device_staging: addition overflow";
        return xkv_device_staging_reservation();
    }
    if (arena_cap > 0 && total_transient > arena_cap) {
        if (out_deficit) *out_deficit = total_transient - arena_cap;
        if (err) *err = "reserve_device_staging: total transient bytes (" + std::to_string(total_transient) +
                         ") exceed workspace arena limit (" + std::to_string(arena_cap) + ")";
        return xkv_device_staging_reservation();
    }

    size_t new_staging = 0;
    if (!checked_add_size(device_staging_reserved_bytes_, bytes, new_staging)) {
        if (err) *err = "reserve_device_staging: staging reserved addition overflow";
        return xkv_device_staging_reservation();
    }
    device_staging_reserved_bytes_ = new_staging;
    if (device_staging_reserved_bytes_ > device_staging_peak_bytes_) {
        device_staging_peak_bytes_ = device_staging_reserved_bytes_;
    }
    if (out_deficit) *out_deficit = 0;
    return xkv_device_staging_reservation(this, bytes);
}

size_t llama_xkv_cache_store::get_pending_reserved_store_bytes() const {
    std::lock_guard<std::mutex> lock(mtx);
    return pending_reserved_store_bytes_;
}

size_t llama_xkv_cache_store::get_device_staging_reserved_bytes() const {
    std::lock_guard<std::mutex> lock(mtx);
    return device_staging_reserved_bytes_;
}

size_t llama_xkv_cache_store::get_device_staging_peak_bytes() const {
    std::lock_guard<std::mutex> lock(mtx);
    return device_staging_peak_bytes_;
}

bool llama_xkv_cache_store::check_store_capacity_for(
    const std::vector<std::shared_ptr<const xkv_segment>> & new_segments, std::string * err) const {
    try {
        std::lock_guard<std::mutex> lock(mtx);
        return store_capacity_fits_locked(new_segments, err);
    } catch (const std::exception & e) {
        if (err) *err = std::string("store capacity check failed: ") + e.what();
        return false;
    } catch (...) {
        if (err) *err = "store capacity check failed with an unknown exception";
        return false;
    }
}

bool llama_xkv_cache_store::prepare_accounting_scratch(
    const std::shared_ptr<const xkv_segment> & candidate, std::string * err) {
    if (!candidate) {
        if (err) *err = "prepare_accounting_scratch: null candidate segment";
        return false;
    }
    std::lock_guard<std::mutex> lock(mtx);
    std::vector<std::shared_ptr<const xkv_segment>> extras = { candidate };
    return ensure_accounting_scratch_locked(extras, err);
}

bool llama_xkv_cache_store::candidate_incremental_bytes(
    const std::shared_ptr<const xkv_segment> & candidate,
    size_t * out_bytes,
    std::string * err) const {
    if (out_bytes == nullptr) {
        if (err) *err = "candidate_incremental_bytes: null out_bytes pointer";
        return false;
    }
    *out_bytes = 0;
    if (!candidate) {
        if (err) *err = "candidate_incremental_bytes: null candidate segment";
        return false;
    }
    try {
        std::lock_guard<std::mutex> lock(mtx);
        size_t base_bytes = 0;
        std::vector<std::shared_ptr<const xkv_segment>> empty_extras;
        if (!deduplicated_allocated_locked(empty_extras, base_bytes)) {
            if (err) *err = "candidate_incremental_bytes: base accounting overflow";
            return false;
        }
        size_t total_with_cand = 0;
        std::vector<std::shared_ptr<const xkv_segment>> cand_extras = { candidate };
        if (!deduplicated_allocated_locked(cand_extras, total_with_cand)) {
            if (err) *err = "candidate_incremental_bytes: candidate accounting overflow";
            return false;
        }
        if (total_with_cand >= base_bytes) {
            *out_bytes = total_with_cand - base_bytes;
        } else {
            *out_bytes = 0;
        }
        return true;
    } catch (const std::exception & e) {
        if (err) *err = std::string("candidate_incremental_bytes failed: ") + e.what();
        return false;
    } catch (...) {
        if (err) *err = "candidate_incremental_bytes failed with unknown exception";
        return false;
    }
}

bool llama_xkv_cache_store::can_clear(std::string * err) const {
    std::lock_guard<std::mutex> lock(mtx);
    const uint8_t clr_mask = bump_flag_live | bump_flag_content | bump_flag_binding;
    if (!can_bump_locked(clr_mask, err)) {
        return false;
    }
    size_t retired_target = 0;
    if (!checked_add_size(retired_segments.size(), published_segments.size(), retired_target)) {
        if (err) *err = "xkv clear retired capacity overflow";
        return false;
    }
    return true;
}

bool llama_xkv_cache_store::clear(std::string * err) {
    std::lock_guard<std::mutex> lock(mtx);
    if (check_reservation_locked(0)) {
        if (err) *err = "clear: epoch reservation held by another transaction";
        return false;
    }
    // Preflight epochs AND retired capacity BEFORE changing maps/bytes;
    // failure reports with the store untouched (never clear-then-skip-a-bump).
    const uint8_t clr_mask = bump_flag_live | bump_flag_content | bump_flag_binding;
    if (!can_bump_locked(clr_mask, err)) {
        return false;
    }
    size_t retired_target = 0;
    if (!checked_add_size(retired_segments.size(), published_segments.size(), retired_target)) {
        if (err) *err = "xkv clear retired capacity overflow";
        return false;
    }
    retired_segments.reserve(retired_target);
    payload_locations.clear();

    for (auto & kv : published_segments) {
        retired_segments.push_back(std::move(kv.second));
    }
    published_segments.clear();

    bump_prechecked(clr_mask);

    reclaim_retired_segments_locked();
    return true;
}

} // namespace llama_xkv
