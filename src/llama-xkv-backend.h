#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-xkv.h"
#include "ggml-xkv-factor.h"
#include "llama-xkv-codec.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace llama_xkv {

// Exact deduplicated accounting breakdown separating logical, padded, actual, device, and host bytes.
// actual_bytes sums each UNIQUE backing buffer once (dedup by buffer identity):
// streams sharing one batch buffer do not multiply-count it. overflow is set
// (with the offending field saturated) if any checked sum overflows.
struct xkv_backend_accounting {
    size_t count = 0;          // Number of unique allocations (dedup by allocation ID)
    size_t logical_bytes = 0;  // Nominal unpadded bytes (ceil(elements * nominal_bits / 8))
    size_t padded_bytes = 0;   // Exact stream bytes (padded_cols * row_stride)
    size_t actual_bytes = 0;   // Unique backing-buffer bytes (dedup by buffer identity)
    size_t device_bytes = 0;   // Per-stream reserved bytes resident on device (residency == DEVICE_OWNED)
    size_t host_bytes = 0;     // Per-stream reserved bytes resident on host (residency == REFERENCE_HOST)
    bool overflow = false;     // True if any checked sum overflowed (field saturated to SIZE_MAX)

    bool operator==(const xkv_backend_accounting & o) const {
        return count == o.count &&
               logical_bytes == o.logical_bytes &&
               padded_bytes == o.padded_bytes &&
               actual_bytes == o.actual_bytes &&
               device_bytes == o.device_bytes &&
               host_bytes == o.host_bytes &&
               overflow == o.overflow;
    }
    bool operator!=(const xkv_backend_accounting & o) const { return !(*this == o); }
};

// Checked monotonic allocation ID generator with overflow refusal.
// Thread-safe: all methods lock internally.
// ID semantics are explicit: IDs handed out are NEVER reused, even if the
// batch that consumed them later fails. A failed build burns the IDs it
// consumed; the generator never rolls back (rollback would be unsafe under
// concurrency and would risk ID reuse across live handles).
class xkv_allocation_id_generator {
public:
    explicit xkv_allocation_id_generator(uint64_t start_id = 1, uint64_t max_id = UINT64_MAX);

    // Non-copyable/non-movable (internal mutex); share via pointer.
    xkv_allocation_id_generator(const xkv_allocation_id_generator &) = delete;
    xkv_allocation_id_generator & operator=(const xkv_allocation_id_generator &) = delete;

    // Allocates next ID monotonically. Refuses and returns false on overflow.
    bool allocate_id(uint64_t & id_out, std::string * err = nullptr);

    uint64_t current_id() const;
    uint64_t max_id() const;

    void reset(uint64_t start_id = 1, uint64_t max_id = UINT64_MAX);
    void set_max_id(uint64_t max_id);

private:
    mutable std::mutex mutex_;
    uint64_t next_id_ = 1;
    uint64_t max_id_ = UINT64_MAX;
};

// Returns nominal bits per element for supported XKV codecs (2 for Turbo2, 3 for Turbo3, 4 for Turbo4, 8 for Q8_0, 16 for F16, 32 for F32).
size_t xkv_nominal_bits_per_element(enum ggml_type type);

// Compute nominal unpadded logical bytes for given dimensions and type:
// ceil(rows * cols * nominal_bits / 8). Returns 0 for unsupported types,
// zero shapes, or unrepresentable (overflowing) sizes. Ceil (not floor) so
// sub-byte nominal widths (Turbo2/3) never under-report a partial byte.
size_t xkv_compute_logical_bytes(enum ggml_type type, uint64_t rows, uint64_t cols);

// FNV-1a 64 checksum helper for data buffers.
uint64_t xkv_backend_checksum(const uint8_t * data, size_t size);

// Seeded FNV-1a 64 (envelope chaining: h1 = seeded(desc_bytes, OFFSET),
// h2 = seeded(bytes, h1)). xkv_backend_checksum(d, n) == seeded(d, n, OFFSET).
uint64_t xkv_backend_checksum_seeded(const uint8_t * data, size_t size, uint64_t seed);

// FNV-1a 64 offset basis (seed for unchained checksums).
constexpr uint64_t XKV_BACKEND_FNV_OFFSET = 0xcbf29ce484222325ULL;

// Forward declarations
class xkv_backend_batch_builder;
struct xkv_backend_pack_request;
struct xkv_backend_pack_result;
struct xkv_backend_device_stream;
struct xkv_backend_adopt_result;

// Shared backing store for one batch: a single ggml context plus a single
// backend buffer holding every new tensor of the batch. Sharing one buffer
// avoids severe per-stream alignment overhead for many tiny streams. Freed
// (buffer first, then context) when the last referencing allocation dies, so
// graph pins (shared_ptr) keep the whole store alive past store retirement.
struct xkv_backend_batch_storage {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    size_t buffer_bytes = 0;
    // ggml_init(no_alloc=true) may borrow caller-provided metadata memory.
    // A transferred context must retain that allocation until ggml_free().
    std::vector<uint8_t> context_memory;
    // Shared owning backend executor (keeps backend alive with store/bundle lifetime).
    std::shared_ptr<struct ggml_backend> backend_executor;

    ~xkv_backend_batch_storage();

    xkv_backend_batch_storage() = default;
    xkv_backend_batch_storage(const xkv_backend_batch_storage &) = delete;
    xkv_backend_batch_storage & operator=(const xkv_backend_batch_storage &) = delete;
};

// Immutable backend-resident XKV code-stream allocation.
// Views one tensor inside a shared batch store with stable shared_ptr
// lifetime. Thread-safe for concurrent read access once marked immutable.
class xkv_backend_allocation {
public:
    ~xkv_backend_allocation();

    // Non-copyable (identity is by allocation ID; share via shared_ptr)
    xkv_backend_allocation(const xkv_backend_allocation &) = delete;
    xkv_backend_allocation & operator=(const xkv_backend_allocation &) = delete;

    // Movable: stable transfer of the store reference, tensor view and
    // allocation ID; the source is left empty (ID 0, null tensor).
    xkv_backend_allocation(xkv_backend_allocation && o) noexcept;
    xkv_backend_allocation & operator=(xkv_backend_allocation && o) noexcept;

    // Identity and residency
    uint64_t get_allocation_id() const { return allocation_id_; }
    ggml_xkv_residency get_residency() const { return residency_; }

    // Descriptor and fingerprint
    const codec_desc & get_desc() const { return desc_; }
    uint64_t get_descriptor_fingerprint() const { return descriptor_fingerprint_; }
    uint64_t get_checksum() const { return checksum_; }

    // Byte counters. actual_bytes is this stream's reserved slice (per-tensor
    // backend allocation size including alignment); the full shared backing
    // buffer size is available via get_buffer_bytes().
    size_t get_logical_bytes() const { return logical_bytes_; }
    size_t get_padded_bytes() const { return padded_bytes_; }
    size_t get_actual_bytes() const { return slice_bytes_; }
    size_t get_buffer_bytes() const;
    size_t get_device_bytes() const { return (residency_ == GGML_XKV_RES_DEVICE_OWNED) ? slice_bytes_ : 0; }
    size_t get_host_bytes() const { return (residency_ == GGML_XKV_RES_REFERENCE_HOST) ? slice_bytes_ : 0; }

    // Owning-backend record (values only, never dangling pointers): the buft
    // the tensor was allocated from plus the building backend's name/device.
    ggml_backend_buffer_type_t get_owning_buft() const { return owning_buft_; }
    const std::string & get_owning_backend_name() const { return owning_backend_name_; }
    int get_owning_dev_type() const { return owning_dev_type_; }
    bool owning_backend_was_null() const { return owning_backend_null_; }

    // Device-packed streams are never host-checksummed (no D2H allowed):
    // checksum_bound=false carries a pack_provenance audit tag instead.
    // Uploaded streams are always checksum-bound.
    bool is_checksum_bound() const { return checksum_bound_; }
    uint64_t get_pack_provenance() const { return pack_provenance_; }

    // GGML handles (valid while any shared_ptr to this allocation lives)
    ggml_context * get_ctx() const;
    ggml_tensor * get_tensor() const { return tensor_; }
    ggml_backend_buffer_t get_buffer() const;

    // Immutability
    bool is_immutable() const { return immutable_; }
    void mark_immutable() { immutable_ = true; }

    // Single-stream readback (downloads exact stream bytes). Validates that
    // the caller backend matches the owning backend record, then checksums
    // and descriptor exact bytes. out_bytes is assigned ONLY on success.
    bool readback(ggml_backend_t backend, std::vector<uint8_t> & out_bytes, std::string * err = nullptr) const;

private:
    friend class xkv_backend_batch_builder;

    xkv_backend_allocation() = default;

    uint64_t allocation_id_ = 0;
    ggml_xkv_residency residency_ = GGML_XKV_RES_REFERENCE_HOST;
    codec_desc desc_;
    uint64_t descriptor_fingerprint_ = 0;
    uint64_t checksum_ = 0;

    size_t logical_bytes_ = 0;
    size_t padded_bytes_ = 0;
    size_t slice_bytes_ = 0;

    std::shared_ptr<xkv_backend_batch_storage> storage_;
    ggml_tensor * tensor_ = nullptr;

    ggml_backend_buffer_type_t owning_buft_ = nullptr;
    std::string owning_backend_name_;
    int owning_dev_type_ = -1; // ggml_backend_dev_type value, or -1 when unknown
    bool owning_backend_null_ = true;
    bool checksum_bound_ = true;   // false for device-packed streams (no D2H)
    uint64_t pack_provenance_ = 0; // FNV audit tag for packed streams

    bool immutable_ = false;
};

// Store allocation reservation (hard xkv_store_mib cap plumbing). Issued by
// the store side from the xkv_store_mib budget and passed per batch/pack.
// The backend preflights estimated exact+alignment bytes BEFORE allocating
// and refuses — destroying off-side state with no attach or host release —
// when the estimate OR the actual buffer exceeds reserved/cap. Null = no
// enforcement (reference profile / tests). Read-only during build/pack;
// the store debits at attach time (bridge); concurrent batches must hold
// distinct reservations.
struct xkv_backend_store_reservation {
    uint64_t reserved_bytes = 0; // pre-authorized bytes for this batch/pack
    uint64_t cap_bytes = 0;      // hard store cap from store_capacity_bytes(), 0 = no cap
};

// Batch configuration and failure injection
struct xkv_backend_batch_config {
    ggml_xkv_residency residency = GGML_XKV_RES_DEVICE_OWNED;
    bool enforce_residency = false; // When false, adapts residency to buft (host buft -> host res)
    bool verify_readback = false;   // Read back and compare uploaded bytes
    bool verify_checksum = false;   // Verify FNV-1a checksum of uploaded bytes

    // True persistent ownership: caller supplies a real shared_ptr<ggml_backend>
    // managing the backend lifetime (typically with ggml_backend_free deleter).
    // If provided, batch and adopt results copy this owner and verify that
    // raw backend == backend_owner.get(). Never manufactured from a raw pointer!
    std::shared_ptr<struct ggml_backend> backend_owner = nullptr;

    // Caller-owned scratch for verification downloads. Required when
    // verify_readback/verify_checksum is set: must hold the combined host
    // peak (sum of new-stream exact bytes); build preflights the peak up
    // front and fails closed when the workspace is missing or short.
    // Verification never heap-allocates stream buffers internally.
    uint8_t * verify_workspace = nullptr;
    size_t verify_workspace_bytes = 0;

    // Injected faults for testing (0-indexed stream index, -1 disables)
    int inject_alloc_fail_at = -1;
    int inject_upload_fail_at = -1;
    int inject_readback_fail_at = -1;
    int inject_checksum_fail_at = -1;
    // Fires right AFTER the stream's upload set_async is queued (before the
    // batch synchronize), so fence-on-failure paths are covered at that index.
    int inject_post_submit_fail_at = -1;
    // Pack-only (§15.4): flattened copy ordinal across the pack batch in queue
    // order; checked pre-submission per copy (fence drains earlier copies).
    int inject_copy_fail_at = -1;
};

// Metrics for batch execution
struct xkv_backend_batch_stats {
    size_t sync_count = 0;        // Calls to ggml_backend_synchronize: at most 1 per
                                  // build (uploads and verification downloads queue
                                  // before a single synchronize) and at most 1 per
                                  // readback batch; 0 when no backend moved bytes.
    size_t uploaded_streams = 0;  // Newly uploaded stream count
    size_t shared_streams = 0;    // Reused existing stream count
    size_t uploaded_bytes = 0;    // Total exact bytes uploaded
};

// Batch build result with explicit host release commit contract.
// The result NEVER dereferences caller memory: shared host vectors are held
// by weak_ptr (expired sources are skipped, never touched — no UAF even if
// the result outlives the source), raw caller pointers are never captured
// (the caller clears those manually after success), and vector releases use
// clear() only (shrink_to_fit can throw). Caller callbacks are all attempted
// even if one throws; commit reports the first error and marks committed
// only when every release succeeded. vector::clear() itself is noexcept, so
// vector releases cannot throw or partially release.
class xkv_backend_batch_result {
public:
    xkv_backend_batch_result() = default;

    std::vector<std::shared_ptr<xkv_backend_allocation>> handles;
    xkv_backend_batch_stats stats;
    std::shared_ptr<struct ggml_backend> executor;
    ggml_backend_t get_executor() const { return executor.get(); }

    // Stream indices (into the builder's stream order) whose source host
    // bytes the caller may clear after a successful build. Only new streams
    // of a device-owned batch are listed; host-residency streams retain
    // their bytes and shared handles need no release. The result never
    // dereferences caller memory: borrowed raw vectors/encoded_matrix bytes
    // are NEVER auto-released (the caller clears them manually); shared
    // vectors clear via weak_ptr and release_fn runs at commit.
    std::vector<size_t> releasable_stream_indices;

    // Explicit caller commit: releases registered source host vectors only
    // after batch success. Only device-owned batches register releases; host
    // reference batches retain their bytes truthfully (commit is a no-op).
    // Idempotent on success; refuses on a failed batch.
    bool commit_host_release(std::string * err = nullptr);

    bool is_committed() const { return committed_; }
    bool is_success() const { return success_; }

private:
    friend class xkv_backend_batch_builder;
    friend struct xkv_backend_import_result;
    friend struct xkv_backend_adopt_result;
    bool success_ = false;
    bool committed_ = false;
    std::vector<std::weak_ptr<std::vector<uint8_t>>> host_weak_refs_;
    std::vector<std::function<void()>> release_fns_;
};

// Batch stream input descriptor. Raw data pointers and raw vector pointers
// are borrowed for the duration of build() only and are never retained.
struct xkv_backend_stream_input {
    codec_desc desc;
    const uint8_t * data = nullptr;
    size_t size = 0;
    std::vector<uint8_t> * source_host_vector = nullptr; // borrowed, never retained
    std::shared_ptr<std::vector<uint8_t>> shared_host_vector = nullptr; // shared ownership
    std::function<void()> release_fn = nullptr;
    std::shared_ptr<xkv_backend_allocation> existing_handle = nullptr;
};

// Batch builder: validates descriptors, creates off-side allocations in one
// shared store, uploads with async+one sync, optionally verifies
// readback/checksum, and atomically returns handles.
//
// Failure atomicity: build() and readback assign their output parameters
// ONLY on success. On ANY failure the outputs are left bit-identical to
// entry, off-side allocations are destroyed, and caller inputs are
// unchanged. All allocations (including std::bad_alloc from vector growth)
// are caught and reported as failure, never propagated.
class xkv_backend_batch_builder {
public:
    explicit xkv_backend_batch_builder(xkv_allocation_id_generator * id_gen = nullptr);

    // Non-copyable/non-movable (owns a thread-safe generator); one builder per batch.
    xkv_backend_batch_builder(const xkv_backend_batch_builder &) = delete;
    xkv_backend_batch_builder & operator=(const xkv_backend_batch_builder &) = delete;

    // Register a new stream with raw byte buffer (borrowed during build only).
    // release_fn runs at commit_host_release (device-owned success only);
    // it should be noexcept — a throwing callback fails commit.
    void add_stream(const codec_desc & desc, const uint8_t * data, size_t size,
                    std::function<void()> release_fn = nullptr);

    // Register a new stream with a borrowed source host vector. The vector is
    // read during build() and NEVER retained or auto-released: after a
    // device-owned success the caller clears it manually after commit.
    void add_stream(const codec_desc & desc, std::vector<uint8_t> * source_host_vector);

    // Register a new stream with shared host-vector ownership. The result
    // holds only a weak_ptr; commit clears the vector iff the source is
    // still alive (UAF-safe). Null is ignored.
    void add_stream(const codec_desc & desc,
                    const std::shared_ptr<std::vector<uint8_t>> & source_host_vector);

    // Register an encoded_matrix (desc + bytes borrowed during build only).
    // Null is ignored. The matrix bytes are never retained or auto-released.
    void add_stream(encoded_matrix * em);

    // Register an existing shared handle (e.g. shared B handle across segment
    // versions). The handle must be immutable and match the batch's effective
    // residency and buffer type, else the whole build fails. Null is ignored.
    void add_shared_handle(std::shared_ptr<xkv_backend_allocation> handle);

    size_t stream_count() const { return streams_.size(); }
    void clear() { streams_.clear(); }

    // Execute atomic batch allocation & upload. result_out untouched on failure.
    bool build(
        ggml_backend_t backend,
        ggml_backend_buffer_type_t buft,
        const xkv_backend_batch_config & config,
        xkv_backend_batch_result & result_out,
        std::string * err = nullptr,
        const xkv_backend_store_reservation * reservation = nullptr);

private:
    bool build_impl(
        ggml_backend_t backend,
        ggml_backend_buffer_type_t buft,
        const xkv_backend_batch_config & config,
        xkv_backend_batch_result & out_local,
        std::string * err,
        const xkv_backend_store_reservation * reservation);
    // Cross-tensor pack worker (free function xkv_backend_pack_batch below
    // delegates here). Member so it shares allocation-assembly access.
    static bool pack_impl(
        ggml_backend_t backend,
        ggml_backend_buffer_type_t buft,
        const std::vector<xkv_backend_pack_request> & requests,
        const xkv_backend_batch_config & config,
        xkv_allocation_id_generator & id_gen,
        xkv_backend_pack_result & out_local,
        std::string * err,
        const xkv_backend_store_reservation * reservation);
    // Narrow friendship: the public pack wrapper below is the only
    // non-member allowed to call pack_impl (no broad exposure).
    friend bool xkv_backend_pack_batch(
        ggml_backend_t backend,
        ggml_backend_buffer_type_t buft,
        const std::vector<xkv_backend_pack_request> & requests,
        const xkv_backend_batch_config & config,
        xkv_allocation_id_generator & id_gen,
        xkv_backend_pack_result & out,
        std::string * err,
        const xkv_backend_store_reservation * reservation);

    xkv_allocation_id_generator * id_gen_ = nullptr;
    std::vector<xkv_backend_stream_input> streams_;
    // Device-tensor adopt worker (free function below delegates here).
    static bool adopt_impl(
        ggml_backend_t backend,
        ggml_backend_buffer_type_t buft,
        const std::vector<xkv_backend_device_stream> & streams,
        const xkv_backend_batch_config & config,
        xkv_allocation_id_generator & id_gen,
        xkv_backend_adopt_result & out_local,
        std::string * err,
        const xkv_backend_store_reservation * reservation);
    // Narrow friendship: the public adopt wrapper below is the only
    // non-member allowed to call adopt_impl (no broad exposure).
    friend bool xkv_backend_adopt_device_tensors(
        ggml_backend_t backend,
        ggml_backend_buffer_type_t buft,
        const std::vector<xkv_backend_device_stream> & streams,
        const xkv_backend_batch_config & config,
        xkv_allocation_id_generator & id_gen,
        xkv_backend_adopt_result & out,
        std::string * err,
        const xkv_backend_store_reservation * reservation);
    // Ownership-transfer worker. Unlike adopt_impl, this wraps tensors in an
    // already-populated uniquely owned batch buffer and performs no copy/sync.
    static bool take_impl(
        ggml_backend_t backend,
        ggml_backend_buffer_type_t buft,
        std::shared_ptr<xkv_backend_batch_storage> storage,
        const std::vector<xkv_backend_device_stream> & streams,
        const xkv_backend_batch_config & config,
        xkv_allocation_id_generator & id_gen,
        xkv_backend_adopt_result & out_local,
        std::string * err,
        const xkv_backend_store_reservation * reservation);
    friend bool xkv_backend_take_device_tensors(
        ggml_backend_t backend,
        ggml_backend_buffer_type_t buft,
        std::shared_ptr<xkv_backend_batch_storage> storage,
        const std::vector<xkv_backend_device_stream> & streams,
        const xkv_backend_batch_config & config,
        xkv_allocation_id_generator & id_gen,
        xkv_backend_adopt_result & out,
        std::string * err,
        const xkv_backend_store_reservation * reservation);
};

// Readback batch result
struct xkv_backend_readback_result {
    std::vector<std::vector<uint8_t>> stream_bytes;
    xkv_backend_batch_stats stats;
};

// Downloads exact bytes for all allocations using async get + one sync.
// Validates owning backend, checksums and descriptors. out untouched on failure.
bool xkv_backend_readback_batch(
    ggml_backend_t backend,
    const std::vector<std::shared_ptr<const xkv_backend_allocation>> & allocations,
    const xkv_backend_batch_config & config,
    xkv_backend_readback_result & out,
    std::string * err = nullptr);

bool xkv_backend_readback_batch(
    ggml_backend_t backend,
    const std::vector<std::shared_ptr<xkv_backend_allocation>> & allocations,
    const xkv_backend_batch_config & config,
    xkv_backend_readback_result & out,
    std::string * err = nullptr);

// Accounting: aggregates bytes across allocations, deduplicating shared
// allocations by allocation ID and shared backing buffers by buffer identity.
// Never throws: allocation failures during accounting set overflow=true.
xkv_backend_accounting xkv_backend_calculate_accounting(
    const std::vector<std::shared_ptr<const xkv_backend_allocation>> & allocations);

xkv_backend_accounting xkv_backend_calculate_accounting(
    const std::vector<std::shared_ptr<xkv_backend_allocation>> & allocations);

// Whole-bundle verification for store adopt-or-refuse (no mutation, no D2H).
// Every handle must be non-null, immutable, uniform expected_residency and
// owning buffer type, with valid descriptors whose fingerprints and exact
// bytes match their tensors. Checksum-bound handles pass on upload-time
// provenance; device-packed handles (unbound + nonzero provenance tag) pass
// only when allow_unbound_packed. Empty set fails. Failures report via err;
// allocation faults are caught and reported, never propagated.
bool xkv_backend_bundle_verify(
    const std::vector<std::shared_ptr<const xkv_backend_allocation>> & handles,
    ggml_xkv_residency expected_residency,
    bool allow_unbound_packed,
    std::string * err = nullptr);

bool xkv_backend_bundle_verify(
    const std::vector<std::shared_ptr<xkv_backend_allocation>> & handles,
    ggml_xkv_residency expected_residency,
    bool allow_unbound_packed,
    std::string * err = nullptr);

// Capability query: inspects the ACTUAL backend / buft / device (including a
// live allocation probe — not just type claims) without global compile flags.
// Per-graph op dispatch support is additionally validated at execution time
// via ggml_backend_supports_op; this query covers residency placeability.

// Peak accounting across generations: live plus retired-but-pinned (COW).
// combined dedups shared allocations/buffers across both lists, so
// combined.actual_bytes is the resident COW peak footprint. Never throws.
struct xkv_backend_peak_accounting {
    xkv_backend_accounting live;
    xkv_backend_accounting pinned_retired;
    xkv_backend_accounting combined;
};

xkv_backend_peak_accounting xkv_backend_calculate_peak_accounting(
    const std::vector<std::shared_ptr<const xkv_backend_allocation>> & live,
    const std::vector<std::shared_ptr<const xkv_backend_allocation>> & pinned_retired);

xkv_backend_peak_accounting xkv_backend_calculate_peak_accounting(
    const std::vector<std::shared_ptr<xkv_backend_allocation>> & live,
    const std::vector<std::shared_ptr<xkv_backend_allocation>> & pinned_retired);

// Cross-tensor device-native pack: byte-preserving gather of surviving rows
// from immutable token-major A streams into newly allocated immutable COW
// destinations. The pinned source is NEVER mutated in place (in-place
// memmove regions cannot build a COW destination while the old version stays
// readable). One ordered submit (per-row view copies) plus one synchronize
// per pack batch; no D2H and no requantization. Landmarks are NOT row-packed
// here: landmark outputs attach only as exact phase-aware rebuilds through
// the batch build path (no truncation API exists by design — never first-N).
struct xkv_backend_pack_request {
    std::shared_ptr<xkv_backend_allocation> src; // immutable token-major A stream
    std::vector<uint32_t> surviving_rows;       // strictly ascending, in-range, nonempty
    codec_desc dst_desc; // same type/orient/padded-cols/stride; rows == surviving.size()
};

struct xkv_backend_pack_result {
    std::vector<std::shared_ptr<xkv_backend_allocation>> handles; // order-matched to requests
    xkv_backend_batch_stats stats; // sync_count <= 1 (0 when backend is null)
};

// Failure-atomic: out untouched on ANY failure (off-side dst store destroyed,
// src handles/buffers never mutated, no partial handles). id_gen IDs burn on
// failure per generator semantics. reservation (null = no enforcement)
// preflights exact+alignment bytes and refuses on estimate/actual excess.
// config injections honored: alloc per request index; copy pre-submission
// and post-submit per flattened copy ordinal; readback/checksum unused.
bool xkv_backend_pack_batch(
    ggml_backend_t backend,
    ggml_backend_buffer_type_t buft,
    const std::vector<xkv_backend_pack_request> & requests,
    const xkv_backend_batch_config & config,
    xkv_allocation_id_generator & id_gen,
    xkv_backend_pack_result & out,
    std::string * err = nullptr,
    const xkv_backend_store_reservation * reservation = nullptr);
bool xkv_backend_supports_residency(
    ggml_backend_t backend,
    ggml_backend_buffer_type_t buft,
    enum ggml_type type,
    ggml_xkv_residency residency,
    std::string * err = nullptr);
// Device-tensor adopt: DAG-computed tensors (no host bytes exist) become
// immutable allocations via full-tensor device copies into a new store.
// The caller guarantees src tensors are contiguous, backend-resident, and
// allocated from buft on backend's device; adopt validates type/shape/exact
// bytes structurally and fails closed otherwise. Adopted streams are always
// checksum-unbound with the caller provenance tag (no D2H allowed); verify_*
// flags are refused (nothing host-side to compare against). One ordered
// submit plus one synchronize per batch; null backend uses inline sync
// copies. Failure-atomic with fence drain, IDs burn, reservation enforced.
struct xkv_backend_device_stream {
    codec_desc desc;
    ggml_tensor * tensor = nullptr; // backend-resident src, borrowed during call
    uint64_t provenance = 0;        // nonzero audit tag (seal/kernel identity)
};

struct xkv_backend_adopt_result {
    std::vector<std::shared_ptr<xkv_backend_allocation>> handles; // order-matched
    xkv_backend_batch_stats stats; // sync_count <= 1 (0 when backend is null)
    std::shared_ptr<struct ggml_backend> executor;
    ggml_backend_t get_executor() const { return executor.get(); }

    // Convert into a committed xkv_backend_batch_result suitable for candidate->backend_bundle.
    // If adopted handles are DEVICE_OWNED, requires non-null owner with owner.get() == backend;
    // returns nullptr otherwise. NEVER manufactures ownership from a raw pointer!
    std::shared_ptr<xkv_backend_batch_result> make_batch_result(ggml_backend_t backend, std::shared_ptr<struct ggml_backend> owner = nullptr) const;
};

bool xkv_backend_adopt_device_tensors(
    ggml_backend_t backend,
    ggml_backend_buffer_type_t buft,
    const std::vector<xkv_backend_device_stream> & streams,
    const xkv_backend_batch_config & config,
    xkv_allocation_id_generator & id_gen,
    xkv_backend_adopt_result & out,
    std::string * err = nullptr,
    const xkv_backend_store_reservation * reservation = nullptr);

// Zero-copy counterpart to xkv_backend_adopt_device_tensors. `storage` must
// be uniquely owned, already allocated from `buft`, and contain every tensor
// in `streams`. On success ownership moves into immutable allocation handles;
// there are no backend copies or synchronizations. Validation and reservation
// checks precede handle publication; `out` is untouched on every failure.
bool xkv_backend_take_device_tensors(
    ggml_backend_t backend,
    ggml_backend_buffer_type_t buft,
    std::shared_ptr<xkv_backend_batch_storage> storage,
    const std::vector<xkv_backend_device_stream> & streams,
    const xkv_backend_batch_config & config,
    xkv_allocation_id_generator & id_gen,
    xkv_backend_adopt_result & out,
    std::string * err = nullptr,
    const xkv_backend_store_reservation * reservation = nullptr);

// State-restore import: persisted code streams are host bytes, but a Vulkan
// graph needs device handles. The store import preallocates/uploads every
// UNIQUE stream off-side under the reservation cap, verifies descriptor +
// fingerprint + checksum per stream, then the store atomically publishes
// segments+handles (import itself never touches the live store, so failure
// leaves the old store intact). Streams sharing a nonzero dedup_key (one
// state allocation referenced by many groups, e.g. shared B) upload ONCE
// and share one device handle, preserving B identity on device. Zero key =
// unique stream. Checksum mismatch, cap excess, or any fault fails the
// whole import with outputs untouched.
struct xkv_backend_import_stream {
    codec_desc desc;
    const uint8_t * data = nullptr; // persisted host bytes, borrowed during import
    size_t size = 0;
    uint64_t expected_checksum = 0; // FNV over bytes; verified pre-upload, fail closed
    // Optional exact serialized descriptor bytes (state image desc_bytes).
    // When present, expected_checksum is the chained envelope
    // FNV(desc_bytes || bytes) and is verified as such; otherwise it is
    // FNV(bytes). Either way the allocation itself stores FNV(bytes) so
    // later tensor readbacks compare against downloaded stream bytes.
    const uint8_t * desc_bytes = nullptr;
    size_t desc_size = 0;
    uint64_t dedup_key = 0;         // nonzero: shared state allocation, one upload
};

struct xkv_backend_import_result {
    std::vector<std::shared_ptr<xkv_backend_allocation>> handles; // order-matched to streams
    xkv_backend_batch_stats stats;

    // Convert into a committed xkv_backend_batch_result suitable for candidate->backend_bundle.
    // If imported handles are DEVICE_OWNED, requires non-null owner with owner.get() == backend;
    // returns nullptr otherwise. NEVER manufactures ownership from a raw pointer!
    std::shared_ptr<xkv_backend_batch_result> make_batch_result(ggml_backend_t backend, std::shared_ptr<struct ggml_backend> owner = nullptr) const;
};

bool xkv_backend_import_batch(
    ggml_backend_t backend,
    ggml_backend_buffer_type_t buft,
    const std::vector<xkv_backend_import_stream> & streams,
    const xkv_backend_batch_config & config,
    xkv_allocation_id_generator & id_gen,
    xkv_backend_import_result & out,
    std::string * err = nullptr,
    const xkv_backend_store_reservation * reservation = nullptr);

// State-restore wrapper alias matching exact Main instruction:
inline bool xkv_backend_import_code_streams(
    ggml_backend_t backend,
    ggml_backend_buffer_type_t buft,
    const std::vector<xkv_backend_import_stream> & streams,
    const xkv_backend_batch_config & config,
    xkv_allocation_id_generator & id_gen,
    xkv_backend_import_result & out,
    std::string * err = nullptr,
    const xkv_backend_store_reservation * reservation = nullptr) {
    return xkv_backend_import_batch(backend, buft, streams, config, id_gen, out, err, reservation);
}

// Inspects actual backend/buft to determine if it is a production Vulkan GPU backend.
bool xkv_backend_is_production_vulkan(
    ggml_backend_t backend,
    ggml_backend_buffer_type_t buft,
    std::string * err = nullptr);

// Helper to create a true shared_ptr<ggml_backend> with ggml_backend_free deleter
std::shared_ptr<struct ggml_backend> xkv_backend_make_owner(ggml_backend_t backend);

// Safe hot backend bulk readback keyed by physical hot slot.
// Reads exactly the requested physical slots from the cache's backend-resident
// hot K/V tensor into caller destination buffers (one device get per slot,
// queued before a single synchronize). Bounds-checked against hot_capacity,
// element_size_bytes must match the tensor row stride, slots must be unique,
// and dst_capacity_bytes must hold n_slots * element_size_bytes.
// Null backend uses inline sync copy. Failure leaves out untouched.
struct xkv_backend_hot_readback_request {
    const struct ggml_tensor * hot_tensor = nullptr; // backend-resident hot K/V
    uint32_t hot_capacity = 0;                       // physical row bound
    size_t row_stride_bytes = 0;                     // exact row bytes
    const uint32_t * physical_slots = nullptr;       // [n_slots] physical rows
    uint32_t n_slots = 0;
};

// Production caller-owned output span overload: writes directly to dst_bytes,
// carrying only execution stats in out_stats. Zero heap allocation.
bool xkv_backend_hot_readback_span(
    ggml_backend_t backend,
    const xkv_backend_hot_readback_request & req,
    uint8_t * dst_bytes,
    size_t dst_capacity_bytes,
    xkv_backend_batch_stats & out_stats,
    std::string * err = nullptr);

// Convenience vector wrapper: test-only/reference. Production callers MUST
// use xkv_backend_hot_readback_span with caller/arena-carved buffers.
struct xkv_backend_hot_readback_result {
    std::vector<uint8_t> bytes; // [n_slots * row_stride_bytes]
    xkv_backend_batch_stats stats; // sync_count <= 1
};

// Selected-hot canonicalize helper: dequantizes + inverse attention WHT + inverse RoPE
// for selected physical rows and positions into canonical pre-RoPE head floats.
// Uses ggml_xkv_canonicalize on the backend if available, or direct CPU oracle.
// Exactly one device graph / D2H sync per call.
struct xkv_backend_hot_canonicalize_request {
    const struct ggml_tensor * hot_kv = nullptr; // backend-resident hot tensor [padded_head_dim*n_heads, cap]
    uint32_t hot_capacity = 0;
    const uint32_t * physical_slots = nullptr;   // [n_rows]
    const int32_t  * storage_positions = nullptr;// [n_rows]
    uint32_t n_rows = 0;
    uint32_t kv_head = 0;
    uint32_t n_kv_heads = 0;
    uint32_t head_dim = 0;
    uint32_t padded_head_dim = 0; // 128-aligned for Turbo4
    uint32_t rotary_dim = 0;
    uint32_t rope_mode = 0;       // GGML_XKV_ROPE_HALF (0) or INTERLEAVED (1)
    const float * rope_tables = nullptr; // F32 [2*Fc] (omega + sqrt(freq_scale_sq))
    uint32_t rope_nelements = 0;         // 2*Fc
    const float * hadamard = nullptr;    // [hadamard_dim] WHT signs (optional)
    uint32_t hadamard_dim = 0;
    bool is_k = true;
};

// Caller-carved scratch for hot canonicalize: zero dynamic heap.
struct xkv_backend_hot_canonicalize_scratch {
    uint8_t * meta_buf = nullptr;          // metadata host scratch for rows/pos/status
    size_t    meta_buf_bytes = 0;
    float   * d2h_staging = nullptr;       // temporary staging buffer [n_rows * total_feat]
    size_t    d2h_staging_floats = 0;
};

bool xkv_backend_hot_canonicalize_carved(
    ggml_backend_t backend,
    const xkv_backend_hot_canonicalize_request & req,
    float * dst_canonical,
    size_t dst_capacity_elements,
    const xkv_backend_hot_canonicalize_scratch * scratch,
    xkv_backend_batch_stats & out_stats,
    std::string * err = nullptr);

bool xkv_backend_hot_canonicalize_batch(
    ggml_backend_t backend,
    const xkv_backend_hot_canonicalize_request & req,
    float * dst_canonical,
    size_t dst_capacity_elements,
    xkv_backend_batch_stats & out_stats,
    std::string * err = nullptr);

bool xkv_backend_hot_readback_batch(
    ggml_backend_t backend,
    const xkv_backend_hot_readback_request & req,
    xkv_backend_hot_readback_result & out,
    std::string * err = nullptr);

// Native selected-K fetch for TriAttention over DEVICE_OWNED handles.
// Reconstructs pre-RoPE half-layout K for exactly the requested segment
// rows in one device graph + one sync (selected rows only; never full-factor
// or full-history D2H; K mirror prohibited). Dispatches via ggml_xkv_reconstruct
// on owning backend (with pos=0 for pre-RoPE output) or direct decode dot.
// Handles must be non-null and belong to backend. Fails closed with false+err.
struct xkv_backend_tri_fetch_handles {
    std::shared_ptr<xkv_backend_allocation> a_k;
    std::shared_ptr<xkv_backend_allocation> b_k;
    // Optional paired V handles (reconstruct requires paired K/V per group).
    // If null, a dummy 1-row zero allocation is generated internally.
    std::shared_ptr<xkv_backend_allocation> a_v;
    std::shared_ptr<xkv_backend_allocation> b_v;
};


// Production caller-scratch interface for selected-K fetch: zero dynamic heap.
// Caller provides pre-carved metadata/temporary scratch buffers.
struct xkv_backend_tri_fetch_scratch {
    uint8_t * meta_buf = nullptr;       // temporary scratch for graph metadata
    size_t    meta_buf_bytes = 0;
    float   * d2h_staging = nullptr;    // temporary staging buffer [n_rows * (dim_k + dim_v)]
    size_t    d2h_staging_floats = 0;
};

bool xkv_backend_tri_fetch_selected_k_carved(
    ggml_backend_t backend,
    const xkv_backend_tri_fetch_handles & handles,
    const uint32_t * seg_rows,
    uint32_t n_rows,
    uint32_t head_dim,
    uint32_t kv_head,
    uint32_t n_kv_heads,
    uint32_t feature_offset_k,
    uint32_t feature_dim_k,
    float * dst_pre_rope,
    size_t dst_capacity_elements,
    const xkv_backend_tri_fetch_scratch * scratch,
    xkv_backend_batch_stats & out_stats,
    std::string * err = nullptr);

bool xkv_backend_tri_fetch_selected_k(
    ggml_backend_t backend,
    const xkv_backend_tri_fetch_handles & handles,
    const uint32_t * seg_rows,
    uint32_t n_rows,
    uint32_t head_dim,
    uint32_t kv_head,
    uint32_t n_kv_heads,
    uint32_t feature_offset_k,
    uint32_t feature_dim_k,
    float * dst_pre_rope,
    size_t dst_capacity_elements,
    std::string * err = nullptr);

} // namespace llama_xkv
