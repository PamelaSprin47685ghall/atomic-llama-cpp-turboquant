#pragma once

// XKV code-stream state image (fail-closed, versioned, endian-stable).
//
// Contract (XKV-SR PR11 / admission-state):
// - The state module consumes/produces an IMMUTABLE plain-data XKV state image
//   containing store config/fingerprints/epochs, unique bundles/groups/shared
//   allocations, payload locators/generation/live state, and the EXACT encoded
//   stream descriptor bytes. Version/magic are XKV-specific ("XKVS").
// - Code streams serialize byte-for-byte; the module NEVER decodes/re-encodes
//   streams, and NEVER persists decoded factors, reader fragments, tile cache,
//   candidate FP matrices, dense mirrors, or workspace data.
// - All sizes/checksums/fingerprints are checked BEFORE allocation/publication.
// - Accounting counters are RECOMPUTED from the image and must agree; serialized
//   accounting is never trusted for budget decisions. Budgets come from
//   caller-supplied limits derived from the factor-store/state budget plus hot
//   bytes (never from workspace/decode-cache sizes, never from the image). The
//   resolved persistent factor-store budget (xkv_store_mib, MiB) travels in
//   config.store_mib; actual factored stream bytes must fit inside both it and
//   limits.max_store_bytes. Workspace/decode-cache MiBs remain separate
//   transient budgets and never cover persistent state.
// - Restore validates into an off-side image first; any failure leaves the
//   destination image/store unchanged. No legacy OFF bytes change.
//
// Payload table discipline: segment row IDs/live bits retain dead rows, but the
// payload locator table covers LIVE rows only. The live store does not export
// tombstone generations, so no tombstone locator/generation records are
// fabricated: a dead row whose id appears in the payload table is rejected, and
// a live==0 payload record is rejected outright.
//
// Provenance: model / Tri-calibration / source identity travel as full 32-byte
// hashes where available (all-zero entry = unavailable and skipped on match).
// The legacy uint64 fingerprints are retained for rope/profile/factorizer/codec/
// backend domains that have no natural digest.
//
// Numeric behavior config (segment/chunk/SR/ranks/balance/refine/max
// rows/source/profile/seed/min gates) plus allocator high-waters travel
// explicitly in the envelope: an opaque fingerprint alone cannot reconstruct
// behavior, so restore never depends on process-global defaults.
//
// Device residency: capture reads host stream bytes verbatim. A stream with
// empty host bytes is device-resident and is REJECTED unless the caller
// supplies an explicit synchronized readback entry covering that exact
// (segment, version, group, role); the readback bytes win and must size-match
// the descriptor. Empty host bytes are never treated as valid data.
//
// Live-store capture/import seam (coordinated with XkvStoreSafety):
// - This module uses ONLY existing public store methods
//   (list_hot_payload_bindings, query_payload_segments, pin_segment(_version),
//   get_segment(_version), get_accounting, get_cparams, current_stamp).
// - The store currently exposes NO public enumeration of published segment ids
//   or factored payload ids, and no batch publish-from-image API. Therefore:
//   capture_image() takes caller-pinned segment snapshots plus caller-supplied
//   live payload records (resolved through the public query APIs), and
//   materialize_segments() rebuilds off-side immutable segments with
//   shared-B pointer identity. Atomically publishing materialized segments
//   into a live store is the remaining integration seam (needs a narrow store
//   batch-publish export, deferred to a later coordinated wave).

#include "llama-xkv-cache.h"
#include "llama-io.h"

#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace llama_xkv {

// ---------------------------------------------------------------------------
// Envelope identity
// ---------------------------------------------------------------------------

inline constexpr uint32_t XKV_STATE_MAGIC   = 0x53564B58u; // "XKVS" little-endian
inline constexpr uint32_t XKV_STATE_VERSION = 5; // v1-v4 layouts retired; refused as incompatible

inline constexpr uint32_t XKV_STATE_NO_STREAM = 0xFFFFFFFFu;

// ---------------------------------------------------------------------------
// Plain-data image
// ---------------------------------------------------------------------------

// Caller-supplied provenance fingerprints. Stored verbatim in the image and
// compared verbatim on decode/validate: ANY mismatch refuses (fail-closed).
struct xkv_state_fingerprints {
    uint64_t source           = 0; // xkv_source / decoded-hot vs prerope-capture
    uint64_t model            = 0; // model arch + weights identity (short fingerprint)
    uint64_t rope             = 0; // RoPE config / phase transform
    uint64_t tri_calibration  = 0; // Tri calibration selection (short fingerprint)
    uint64_t profile          = 0; // requested storage profile
    uint64_t factorizer       = 0; // factorizer config (rank/balance/seed policy)
    uint64_t codec            = 0; // global codec capability revision
    uint64_t backend          = 0; // backend capability set

    bool operator==(const xkv_state_fingerprints & o) const {
        return source == o.source && model == o.model && rope == o.rope &&
               tri_calibration == o.tri_calibration && profile == o.profile &&
               factorizer == o.factorizer && codec == o.codec && backend == o.backend;
    }
    bool operator!=(const xkv_state_fingerprints & o) const { return !(*this == o); }
};

// Full 32-byte provenance digests where available. An all-zero entry means
// "unavailable" and is skipped during expected-match (never forced to match).
struct xkv_state_provenance {
    uint8_t model_sha256[32]           = {};
    uint8_t tri_calibration_sha256[32] = {};
    uint8_t source_sha256[32]          = {};

    bool has_model() const;
    bool has_tri_calibration() const;
    bool has_source() const;

    bool operator==(const xkv_state_provenance & o) const;
    bool operator!=(const xkv_state_provenance & o) const { return !(*this == o); }
};

// Explicit effective numeric config. Every field that changes numeric behavior
// or admission travels here; restore never consults process-global defaults.
struct xkv_state_config {
    uint32_t profile                   = 0; // llama_xkv_storage_profile (requested)
    uint32_t source                    = 0; // llama_xkv_source
    uint32_t mode                      = 0; // llama_xkv_mode (stored: SHADOW/DENSE/SR)
    uint32_t group_size                = 0;
    uint32_t rank_k                    = 0;
    uint32_t rank_v                    = 0;
    uint32_t segment_tokens            = 0;
    uint32_t chunk_tokens              = 0;
    uint32_t sr_budget                 = 0; // 0 = SR selection off
    uint32_t factor_a_k                = 0; // ggml_type: exact effective codec
    uint32_t factor_b_k                = 0; // ggml_type: exact effective codec
    uint32_t factor_a_v                = 0; // ggml_type: exact effective codec
    uint32_t factor_b_v                = 0; // ggml_type: exact effective codec
    uint32_t factor_balance            = 0; // llama_xkv_factor_balance
    uint32_t landmark_type             = 0; // ggml_type value; GGML_TYPE_COUNT = unspecified
    uint32_t landmark_refine           = 0; // llama_xkv_landmark_refine
    uint32_t landmark_refine_max_rows  = 0;
    uint32_t workspace_mib             = 0; // transient workspace budget (must match image field)
    uint32_t decode_cache_mib          = 0; // transient decode-cache budget (must match image field)
    uint32_t store_mib                 = 0; // resolved persistent factor-store budget in MiB (must be > 0)
    uint64_t seed                      = 0;
    uint64_t min_saving_ppm            = 0; // fixed-point millionths of the saving fraction
    uint64_t min_coverage_ppm          = 0; // fixed-point millionths of the coverage fraction
    uint32_t factorizer                = 0; // llama_xkv_factorizer
    uint32_t min_saving_bytes          = 0;
    // Allocator high-waters (live allocator state, caller-supplied at capture).
    // All are nonzero and strictly exceed every id/nonce carried in the image.
    uint64_t next_segment_id   = 0;
    uint64_t next_alloc_id     = 0;
    uint64_t next_seal_tx_nonce = 0;

    bool operator==(const xkv_state_config & o) const;
    bool operator!=(const xkv_state_config & o) const { return !(*this == o); }
};

// Deduplicated allocation counters snapshot (mirrors xkv_accounting, plain data).
// Every exactly-recomputable field is re-derived from the image on validate and
// must agree; the rest must satisfy ordering invariants. Serialized accounting
// is NEVER used for budget decisions.
struct xkv_state_counters {
    uint64_t live_payload_bytes     = 0; // recomputed: streams of segments with live rows (dedup)
    uint64_t allocated_bytes        = 0; // attested: must be >= factored_bytes + hot_bytes
    uint64_t reserved_bytes         = 0; // attested: must be >= allocated_bytes
    uint64_t workspace_budget_bytes = 0; // must equal workspace_mib * 1MiB exactly
    uint64_t hot_bytes              = 0; // caller-attested (hot tensor sizes not in image)
    uint64_t factored_bytes         = 0; // recomputed: all referenced stream bytes (dedup)
    uint64_t active_segments        = 0; // recomputed: == segments.size()
    uint64_t total_payloads         = 0; // recomputed: == payloads.size()
    uint64_t unique_b_matrices      = 0; // recomputed: # B-referenced allocations
    uint64_t shared_b_bytes         = 0; // recomputed: bytes of B-referenced allocations

    bool operator==(const xkv_state_counters & o) const {
        return live_payload_bytes == o.live_payload_bytes && allocated_bytes == o.allocated_bytes &&
               reserved_bytes == o.reserved_bytes && workspace_budget_bytes == o.workspace_budget_bytes &&
               hot_bytes == o.hot_bytes && factored_bytes == o.factored_bytes &&
               active_segments == o.active_segments && total_payloads == o.total_payloads &&
               unique_b_matrices == o.unique_b_matrices && shared_b_bytes == o.shared_b_bytes;
    }
    bool operator!=(const xkv_state_counters & o) const { return !(*this == o); }
};

// One unique code-stream allocation. A streams are referenced exactly once;
// immutable shared B allocations may be referenced by many groups (identity is
// the allocation index; bytes stored exactly once).
struct xkv_state_allocation {
    uint32_t               alloc_id = 0; // unique within the image, nonzero
    codec_desc             desc;
    std::vector<uint8_t>   desc_bytes;   // exact serialize_desc(desc) output (100 bytes)
    std::vector<uint8_t>   bytes;        // exact encoded stream bytes
    uint64_t               checksum = 0; // FNV-1a64 over desc_bytes || bytes
};

// One landmark chunk: row interval plus the conservative error-bound epsilon
// (|q^T (L_hat - L)| <= ||q||_2 * epsilon) governing candidate refinement.
// Stored as exact bit patterns for endian-stable byte identity.
struct xkv_state_landmark_chunk {
    uint32_t row_begin = 0;
    uint32_t row_count = 0; // nonzero
    uint32_t error_bound_bits = 0; // float epsilon bits; finite and >= 0.0
    uint64_t source_fingerprint = 0; // exact final-factor/phase chunk provenance

    float error_bound() const {
        float v = 0.0f;
        std::memcpy(&v, &error_bound_bits, sizeof(v));
        return v;
    }
    static xkv_state_landmark_chunk make(uint32_t begin, uint32_t count, float eps, uint64_t source_fp) {
        xkv_state_landmark_chunk c;
        c.row_begin = begin;
        c.row_count = count;
        std::memcpy(&c.error_bound_bits, &eps, sizeof(eps));
        c.source_fingerprint = source_fp;
        return c;
    }
    bool operator==(const xkv_state_landmark_chunk & o) const {
        return row_begin == o.row_begin && row_count == o.row_count &&
               error_bound_bits == o.error_bound_bits &&
               source_fingerprint == o.source_fingerprint;
    }
    bool operator!=(const xkv_state_landmark_chunk & o) const {
        return !(*this == o);
    }
};

// Matches compute_base_table_fingerprint (src/llama-xkv-landmark.cpp) exactly:
// mixes in order: n_chunks, n_rows_total, offsets[0..n_chunks], error bound
// bits (float->u32 memcpy, mixed as u64), row payloads, generations,
// positions (int64 bit-cast), live/content/codec/binding epochs,
// topology/publish/layout epochs, phase fp, landmark desc fingerprint.
// Verified by recompute-equality on validate. Layered staleness: this binds
// table<->image-stamp; the landmark binder enforces image-vs-live freshness.
// Empty tables hash deterministically to 0.
uint64_t xkv_state_landmark_chunks_fingerprint(const std::vector<xkv_state_landmark_chunk> & chunks,
                                               uint64_t n_rows_total,
                                               const uint64_t * row_payload_ids,
                                               const uint64_t * row_generations,
                                               const int64_t * row_positions,
                                               uint64_t landmark_desc_fp,
                                               const xkv_snapshot_stamp & stamp, uint64_t rope_fp);

struct xkv_state_group {
    uint32_t              group_index = 0;
    std::vector<uint32_t> owning_layers; // strictly ascending, unique, nonempty


    uint32_t rank_k = 0;
    uint32_t rank_v = 0;

    std::vector<uint32_t> layer_feature_offsets_k;
    std::vector<uint32_t> layer_feature_dims_k;
    std::vector<uint32_t> layer_feature_offsets_v;
    std::vector<uint32_t> layer_feature_dims_v;

    uint32_t total_dim_k = 0;
    uint32_t total_dim_v = 0;

    // Indices into image.allocations.
    uint32_t stream_ak = XKV_STATE_NO_STREAM;
    uint32_t stream_bk = XKV_STATE_NO_STREAM;
    uint32_t stream_av = XKV_STATE_NO_STREAM;
    uint32_t stream_bv = XKV_STATE_NO_STREAM;
    bool     has_landmark = false;
    uint32_t stream_landmark = XKV_STATE_NO_STREAM;

    // Per-landmark-chunk conservative error bounds + closure metadata.
    // Empty = absent (chunk fields not yet populated live); nonempty requires
    // has_landmark and contiguously covers [0, landmark_rows) exactly once.
    // error_bound_bits is the exact float epsilon bit pattern (endian-stable).
    // Refinement safety depends on this table, not just the code stream.
    std::vector<xkv_state_landmark_chunk> landmark_chunks;
    uint64_t landmark_chunks_fingerprint = 0; // FNV over table; recomputed on validate

    uint64_t descriptor_fingerprint = 0;
    uint64_t config_fingerprint     = 0;
};

struct xkv_state_segment {
    uint64_t segment_id      = 0; // nonzero, strictly below config.next_segment_id
    uint64_t segment_version = 0; // nonzero
    uint32_t profile = 0;         // llama_xkv_storage_profile value (validated)
    uint32_t source  = 0;         // llama_xkv_source value (validated)

    uint64_t profile_fingerprint           = 0;
    uint64_t source_fingerprint            = 0;
    uint64_t layer_group_map_fingerprint   = 0;
    uint64_t descriptor_fingerprint        = 0;

    uint32_t n_rows      = 0; // nonzero
    uint32_t n_live_rows = 0; // == popcount(live_bits)

    std::vector<uint64_t> row_payload_ids; // size == n_rows; dead-row ids retain history
    std::vector<uint8_t>  live_bits;       // size == n_rows, values 0/1 only

    std::vector<xkv_state_group> groups;   // nonempty, unique group_index
};

// Payload locator record for a LIVE row. Tombstone (live==0) records are
// rejected: dead rows are retained by row_payload_ids/live_bits only, and a
// dead-row id MUST NOT appear in this table.
// Only factored locators reference a segment bundle; hot and flat_quantized
// locators carry no segment reference and are live.
struct xkv_state_payload {
    uint64_t     payload_id = 0; // nonzero
    uint64_t     generation = 0;
    uint8_t      live = 0;       // must be 1
    xkv_location locator;        // exact locator; storage_generation == generation
};

struct xkv_state_image {
    xkv_state_fingerprints fingerprints;
    xkv_state_provenance   provenance;
    xkv_state_config       config;
    xkv_snapshot_stamp     stamp;

    // Transient budgets in force at capture (from cparams / admission
    // snapshot). The persistent factor-store budget lives in config.store_mib.
    uint32_t workspace_mib    = 0;
    uint32_t decode_cache_mib = 0;

    xkv_state_counters counters;

    // Deterministic order: allocations by alloc_id, segments by (id, version),
    // payloads by payload_id.
    std::vector<xkv_state_allocation> allocations;
    std::vector<xkv_state_segment>    segments;
    std::vector<xkv_state_payload>    payloads;
};

// Explicit synchronized readback for device-resident streams. Each entry
// overrides the host bytes of exactly one (segment, version, group, role)
// stream; the bytes must size-match the stream descriptor. Streams with empty
// host bytes and no covering entry are rejected.
struct xkv_state_readback_entry {
    uint64_t             segment_id      = 0;
    uint64_t             segment_version = 0;
    uint32_t             group_index     = 0;
    uint32_t             role            = 0; // factor_role value
    std::vector<uint8_t> bytes;
};

struct xkv_state_readback {
    std::vector<xkv_state_readback_entry> entries;
};

// ---------------------------------------------------------------------------
// Decode/validate budgets (caller-supplied exact caps; no production defaults)
// ---------------------------------------------------------------------------

struct xkv_state_limits {
    uint64_t max_envelope_bytes     = 0;
    uint64_t max_stream_bytes       = 0;
    uint64_t max_store_bytes        = 0; // persistent factor-store byte budget (actual factored_bytes <= this)
    uint32_t max_segments           = 0;
    uint32_t max_groups_per_segment = 0;
    uint32_t max_payloads           = 0;
    uint32_t max_allocations        = 0;
    uint32_t max_owning_layers      = 0;
    uint32_t max_rows_per_segment   = 0;

    // Byte caps derive from the factor-store/state byte budget plus resident
    // hot bytes plus a fixed framing slack:
    //   slack = 1MiB + 128*max_payloads + 256*max_segments + 64*max_allocations
    // All arithmetic is checked and saturates (never wraps). max_store_bytes
    // carries the factor-store byte budget verbatim for actual-bytes validation.
    // Count caps are caller-supplied exact values.
    static xkv_state_limits from_store_budgets(uint64_t factor_store_bytes_budget, uint64_t hot_bytes,
                                               uint32_t max_segments, uint32_t max_groups_per_segment,
                                               uint32_t max_payloads, uint32_t max_allocations,
                                               uint32_t max_owning_layers, uint32_t max_rows_per_segment);
};

// ---------------------------------------------------------------------------
// Codec: encode / decode / validate (all fail-closed, destination untouched
// on failure, including allocation failure)
// ---------------------------------------------------------------------------

// Deterministic little-endian encode under `limits`. The exact encoded size is
// precomputed with checked arithmetic and reserved once before serializing;
// any invalid field, budget overrun, or allocation failure returns false with
// `out` unchanged.
bool encode_image(const xkv_state_image & image, std::vector<uint8_t> & out,
                  const xkv_state_limits & limits, std::string * err = nullptr);

// Exact precomputed encoded size of `image` (checked arithmetic, no output).
bool encoded_size(const xkv_state_image & image, uint64_t & size_out, std::string * err = nullptr);

// Decode + fully validate against expected fingerprints/provenance and limits.
// On ANY failure returns false and leaves `out` bit-identical to entry
// (decodes into a temp, swaps only on success). Rejects: bad magic, legacy
// (v0) or old (v1-v4) versions, unknown versions, truncation, trailing bytes,
// corrupt or unknown descriptors, checksum mismatch, fingerprint/provenance
// mismatch, duplicate payload/segment/allocation ids, bad group closure,
// orphan/dangling stream refs, locator non-closure, live-bit inconsistency,
// accounting disagreement, config/enum/high-water violations, budget overflow.
bool decode_image(
    const uint8_t * data,
    size_t size,
    xkv_state_image & out,
    const xkv_state_fingerprints & expected,
    const xkv_state_provenance & expected_prov,
    const xkv_state_limits & limits,
    std::string * err = nullptr);

// Off-side structural validation of an in-memory image (same checks as the
// decode tail, minus envelope framing). Destination is const: never mutated.
bool validate_image(
    const xkv_state_image & image,
    const xkv_state_fingerprints & expected,
    const xkv_state_provenance & expected_prov,
    const xkv_state_limits & limits,
    std::string * err = nullptr);

// ---------------------------------------------------------------------------
// Off-side capture / materialize (public-snapshot only, no store mutation)
// ---------------------------------------------------------------------------

// Build an image from caller-pinned immutable segment snapshots, hot binding
// records, live factored payload records, accounting counters, budgets,
// explicit config, fingerprints/provenance and stamp, under `limits`. The
// caller resolves segment/payload ids through the existing public store query
// APIs (list_hot_payload_bindings, query_payload_segments,
// pin_segment(_version)). Shared B identity is preserved by shared_ptr
// identity: groups whose b_k (resp. b_v) point at the same encoded_matrix
// share one allocation. Device-resident streams (empty host bytes) require a
// covering `readback` entry. Fully validated before return; `out` untouched
// on failure.
bool capture_image(
    const std::vector<std::shared_ptr<const xkv_segment>> & segments,
    const std::vector<xkv_hot_payload_binding> & hot_bindings,
    const std::vector<xkv_state_payload> & factored_payloads,
    const xkv_accounting & accounting,
    uint32_t workspace_mib,
    uint32_t decode_cache_mib,
    const xkv_state_config & config,
    const xkv_state_fingerprints & fingerprints,
    const xkv_state_provenance & provenance,
    const xkv_snapshot_stamp & stamp,
    const xkv_state_limits & limits,
    xkv_state_image & out,
    const xkv_state_readback * readback = nullptr,
    std::string * err = nullptr);

// Materialize off-side immutable segments from a validated image. Streams are
// copied verbatim (NEVER re-encoded); groups referencing the same B
// allocation share one std::shared_ptr<const encoded_matrix> (pointer
// identity preserved). Pure function: no store interaction. `out` untouched
// on failure.
bool materialize_segments(
    const xkv_state_image & image,
    std::vector<std::shared_ptr<const xkv_segment>> & out_segments,
    const xkv_state_limits & limits,
    std::string * err = nullptr);

// Re-encode comparison: true iff re-encoding `image` under `limits`
// reproduces `bytes` exactly (determinism / byte-for-byte roundtrip probe).
bool image_matches_bytes(const xkv_state_image & image, const uint8_t * data, size_t size,
                         const xkv_state_limits & limits);

// FNV-1a 64 used for per-stream and whole-envelope checksums.
uint64_t xkv_state_checksum(const uint8_t * data, size_t size);

// Self-contained SHA-256 (no external dependency) for artifact/content digests.
// out must hold 32 bytes. Used for model source-artifact digests and Tri
// calibration content digests carried in xkv_state_provenance.
void xkv_sha256(const uint8_t * data, size_t size, uint8_t out[32]);

// Streaming SHA-256 (same algorithm): init/update/final over chunks, so
// multi-gigabyte source files hash without buffering. Finalizes EXACTLY like
// the one-shot form (xkv_sha256(data) == init/update(data)/final).
struct xkv_sha256_ctx {
    uint32_t h[8];
    uint8_t  block[64];
    size_t   block_used = 0;
    uint64_t total_len  = 0;
};
void xkv_sha256_init(xkv_sha256_ctx & ctx);
void xkv_sha256_update(xkv_sha256_ctx & ctx, const uint8_t * data, size_t size);
void xkv_sha256_final(xkv_sha256_ctx & ctx, uint8_t out[32]);

// Exact source-artifact digest over canonical split file order: mixes domain
// tag + split count, then per file (in order) the canonical index, file size,
// and FULL streamed content. Host paths are never hashed: byte-identical
// models at different paths digest equally; reordered split contents digest
// differently. Same-size files differing by one byte hash differently. False
// (out untouched) when any path is missing/unreadable; never metadata-only.
bool xkv_source_files_sha256(const std::vector<std::string> & paths, uint8_t out[32],
                             std::string * err = nullptr);

// Exact effective-config compatibility between a decoded image and the live
// config. Every behavior/admission field must match exactly (mode, profile,
// source, group/rank/segment/chunk/SR sizes, all four factor codec types,
// balance, landmark type/refine/max-rows, workspace/decode/store budgets,
// seed, min-saving/coverage PPMs, factorizer, min-saving bytes). Allocator
// high-waters are excluded (closure-validated separately, live advances).
// Returns false + err on the first mismatch (names the offending field).
bool check_config_compatible(const xkv_state_config & image_cfg, const xkv_state_config & live_cfg,
                             std::string * err = nullptr);

// ---------------------------------------------------------------------------
// State-stream bridge (llama state save/restore trailer section)
// ---------------------------------------------------------------------------

// Section framing appended AFTER legacy dense bytes on full-state saves only
// (seq_id == -1) and on per-seq saves (RAM demote/restore, prompt cache,
// slot save/restore all funnel through the same entry points). XKV OFF writes
// zero extra bytes. The envelope section persists factored code streams
// byte-for-byte plus descriptors/layout/fingerprints/config/bindings/epochs;
// a second cellmap section persists the ordered factored cell bindings so
// restore can rejoin cells to payloads. Neither section persists decode
// cache, tile cache, reader fragments, or dense mirrors. Per-seq envelopes
// carry the transitive unique segment subset referenced by that seq (shared
// handles persisted once); restore dedups already-present segments and
// rebinds factored cells only (hot/flat stay legacy dense). Absent sections
// on restore mean legacy-only bytes (pre-trailer files); factored rows in
// such files restore as dense and MUST be re-validated by the caller path.
inline constexpr uint32_t XKV_STATE_SECTION_MAGIC = 0x53534B58u; // "XKSS"
inline constexpr uint32_t XKV_STATE_CELLMAP_MAGIC = 0x4D435658u; // "XKCM"
// Cell-binding section cap: entries bounded by caller (save sizes it exactly).
inline constexpr uint64_t XKV_STATE_CELLMAP_MAX_ENTRIES = 1 << 24;

// Writes: section magic (u32 LE) + envelope length (u64 LE) + envelope bytes.
// Returns false (out: nothing written... note: partial trailer write on io
// failure is the caller's to discard, like any other state_write failure).
bool write_state_section(llama_io_write_i & io, const std::vector<uint8_t> & envelope,
                         std::string * err = nullptr);

// Reads the trailer section. `absent` = legacy stream with no section (magic
// read throws at end-of-stream): restore proceeds with legacy bytes only.
// `ok` = section present and fully read into out_envelope (still needs
// decode_image + import). `corrupt` = magic mismatch, bad length, truncation,
// or over-budget length (no large allocation before the length check).
// `out_envelope` is untouched unless the result is `ok`.
enum class xkv_section_read_result : uint8_t { absent = 0, ok = 1, corrupt = 2 };
xkv_section_read_result read_state_section(llama_io_read_i & io, uint64_t max_section_bytes,
                                            std::vector<uint8_t> & out_envelope,
                                            std::string * err = nullptr);
// Factored cell binding in legacy cell order (save order == restore order per
// path: whole-cache preserves physical indices; per-seq installs sequentially
// into the granted slots, so the k-th entry rebinds the k-th restored cell).
// Factored rows only; hot/flat cells stay legacy dense and are never rebound.
struct xkv_cell_binding {
    uint32_t stream_idx = 0;
    uint32_t ordinal    = 0; // position among saved cells of the stream, in legacy save order
    uint64_t payload_id = 0;
    uint64_t generation = 0;
};

// Deterministic LE cellmap bytes: u64 count + entries (u32 stream, u32 cell,
// u64 pid, u64 gen) with checked arithmetic. Empty bindings encode to a
// header-only (8-byte) body, never to zero bytes.
bool encode_cellmap(const std::vector<xkv_cell_binding> & bindings, std::vector<uint8_t> & out,
                     std::string * err = nullptr);
bool decode_cellmap(const uint8_t * data, size_t size, std::vector<xkv_cell_binding> & out,
                     uint64_t max_entries, std::string * err = nullptr);

// Second trailer section (XKCM magic + framed body). Presence rule: written
// iff the envelope section is written; required on restore iff the envelope
// section is present (absent cellmap with present envelope = corrupt).
bool write_cellmap_section(llama_io_write_i & io, const std::vector<xkv_cell_binding> & bindings,
                           std::string * err = nullptr);
xkv_section_read_result read_cellmap_section(llama_io_read_i & io, uint64_t max_entries,
                                              std::vector<xkv_cell_binding> & out_bindings,
                                              std::string * err = nullptr);

// Pure bridge config builder from resolved cparams. High-waters are left zero
// and derived as max+1 at capture (capture_image closes them); explicit
// nonzero high-waters from a live allocator are preserved and validated.
xkv_state_config make_bridge_config(const llama_cparams & cp);

// Config-domain fingerprints derived deterministically from the config
// (source/profile/factorizer/codec/backend); model/rope/tri domains are filled
// by the save-side owner (cache tail) from model/hparams/scorer and ORed in.
xkv_state_fingerprints make_config_fingerprints(const xkv_state_config & cfg);

// Atomic import bundle mirroring the approved store contract
// (StoreSafety: struct xkv_snapshot_import_bundle + import_snapshot_segments).
// Field-for-field compatible by construction (plus the audited high-water/cap
// block below); the store-side adapter maps this bundle onto the store type
// when that API lands. Fencing (maintenance guard + reader drain) is the
// COMMIT CALLER's duty per TransactionCore contract; the plan itself is
// off-side and immutable.
struct xkv_state_import_bundle {
    std::vector<std::shared_ptr<const xkv_segment>> segments;
    std::vector<std::pair<uint64_t, xkv_location>>  locations; // pid -> locator, pid ascending
    xkv_snapshot_stamp stamp;
    uint64_t           sealed_count = 0; // live factored payload count
    // Validated allocator high-waters copied verbatim from the image config.
    // The store install must max() live high-waters with these (never derive
    // down to max+1: burned values above the image max must survive, else
    // IDs/nonces risk reuse). Zero here is impossible (validated pre-build).
    uint64_t next_segment_id    = 0;
    uint64_t next_alloc_id      = 0;
    uint64_t next_seal_tx_nonce = 0;
    // Byte cap the image was validated under (limits.max_store_bytes) plus the
    // full config/fingerprints in force at plan build. The install enforces
    // dedup bytes against min(bundle cap, live cap) and records provenance.
    uint64_t               store_cap_bytes = 0;
    xkv_state_config       config;
    xkv_state_fingerprints fingerprints;
};
using xkv_store_import_fn = std::function<bool(const xkv_state_import_bundle &, std::string *)>;

struct xkv_state_import_plan {
    xkv_state_import_bundle bundle;
};

// Off-side: validate (own fingerprints/provenance) + materialize + assemble.
// `out` untouched on failure. Shared-B pointer identity preserved.
bool build_import_plan(const xkv_state_image & image, const xkv_state_limits & limits,
                       xkv_state_import_plan & out, std::string * err = nullptr);

// Single commit call through the store import function. Propagates refusal;
// the plan is const and reusable for retry against another store.
bool commit_import_plan(const xkv_state_import_plan & plan, const xkv_store_import_fn & import_fn,
                        std::string * err = nullptr);

} // namespace llama_xkv
