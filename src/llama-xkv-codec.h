#pragma once

#include "ggml.h"

#include <cstdint>
#include <string>
#include <vector>

namespace llama_xkv {

// Factor and landmark roles
enum class factor_role : uint8_t {
    a_k = 0,
    b_k = 1,
    a_v = 2,
    b_v = 3,
    landmark = 4,
};

// Storage orientation
enum class orientation : uint8_t {
    token_major = 0,
    feature_major_transposed = 1,
};

// Value domain for stored and decoded representations
enum class value_domain : uint8_t {
    canonical = 0,
    turbo_rotated = 1,
};

struct matrix_shape {
    uint64_t rows = 0;
    uint64_t cols = 0;

    bool operator==(const matrix_shape & o) const {
        return rows == o.rows && cols == o.cols;
    }
    bool operator!=(const matrix_shape & o) const {
        return !(*this == o);
    }
};

// Descriptor for an encoded factor or landmark matrix
struct codec_desc {
    uint32_t     version = 2;
    factor_role  role = factor_role::a_k;
    ggml_type    type = GGML_TYPE_F32;
    orientation  orient = orientation::token_major;
    matrix_shape logical_shape = {0, 0};
    matrix_shape padded_shape = {0, 0};
    uint64_t     row_stride_bytes = 0;
    uint32_t     block_size = 0;
    uint32_t     group_size = 0;
    uint64_t     seed = 42;
    uint64_t     table_fingerprint = 0;
    value_domain stored_domain = value_domain::canonical;
    value_domain decoded_domain = value_domain::canonical;

    // Compiled layout/format revision fingerprint. Distinguishes decode-incompatible formats
    // (e.g. 4-bit PolarQuant vs legacy 3-bit+QJL, centroid tables, WHT signs, struct sizes).
    // For non-Turbo types, this is 0.
    uint64_t     format_revision = 0;

    bool validate(std::string * err = nullptr) const;
    uint64_t fingerprint() const;

    bool operator==(const codec_desc & o) const;
    bool operator!=(const codec_desc & o) const { return !(*this == o); }
};

// Encoded matrix container holding descriptor and contiguous bytes
struct encoded_matrix {
    codec_desc desc;
    std::vector<uint8_t> bytes;
};

// Calculate exact required bytes for an encoded matrix descriptor. Throws std::overflow_error or std::invalid_argument on error.
uint64_t encoded_matrix_bytes(const codec_desc & desc);

// Validate that a factor pair (A and B) have matching rotation, domain, rank, and format revision
// before performing GEMM / reconstruction. Returns true on success, or false with err message.
// Allows identical canonical non-Turbo pairs or explicitly mixed A Turbo / B F16-Q8 by decoding both
// canonical before GEMM. Direct rotated-domain GEMM optimization is valid only when rotation fingerprints match.
bool can_direct_rotated_gemm(
    const codec_desc & desc_a,
    const codec_desc & desc_b
);

bool validate_factor_pair_compatibility(
    const codec_desc & desc_a,
    const codec_desc & desc_b,
    std::string * err = nullptr
);

void assert_factor_pair_compatible(const codec_desc & desc_a, const codec_desc & desc_b);

// Construct and validate a codec descriptor for given parameters.
// For TurboQuant types (Turbo2, Turbo3, Turbo4), group_size MUST be 128 (group_size 64 is rejected in v1).
codec_desc make_codec_desc(
    factor_role  role,
    ggml_type    type,
    orientation  orient,
    matrix_shape logical_shape,
    uint32_t     group_size = 0,
    uint64_t     seed = 42
);

// Encode a row-major float matrix into an encoded_matrix.
// Zero-pads each row up to padded_shape.cols.
// Turbo2/3/4 encoding avoids shared mutable global contamination.
encoded_matrix encode_matrix(
    const codec_desc & desc,
    const float * src,
    size_t src_elements
);

// Decode selected rows from an encoded matrix into row-major canonical floats.
// Output contains n_rows * desc.padded_shape.cols floats.
// Tail padding elements beyond logical_shape.cols decode as zero.
// Canonical decoding applies inverse Turbo WHT for Turbo2/3/4 code streams.
void decode_rows(
    const encoded_matrix & em,
    const uint64_t * row_indices,
    size_t n_rows,
    float * dst,
    size_t dst_capacity_elements,
    value_domain target_domain = value_domain::canonical
);

// Zero-heap decode_rows overload using caller-supplied scratch (for strict
// bounded readers). tmp must provide at least
// ggml_row_size(desc.type, desc.padded_shape.cols) bytes when desc.type is
// F16/Q8_0; a short buffer throws std::invalid_argument (exact one-byte-short
// rejection). tmp is unused for F32/Turbo types (may be nullptr/0). On success
// the output is bit-identical to decode_rows and no heap allocation occurs.
void decode_rows(
    const encoded_matrix & em,
    const uint64_t * row_indices,
    size_t n_rows,
    float * dst,
    size_t dst_capacity_elements,
    uint8_t * tmp,
    size_t tmp_bytes,
    value_domain target_domain = value_domain::canonical
);
// Checked size query for the scratch overload above: sets out_bytes to the
// minimum caller-scratch size for desc (ggml_row_size for F16/Q8_0, 0 for
// types that never touch tmp). Returns false with *err set when desc is
// invalid; never throws. Strict readers call this during planning and fail
// closed with workspace_exhausted before decoding when out_bytes exceeds the
// reserved decode_tmp region, instead of catching mid-tile throws.
// tmp alignment: pass 64-byte aligned scratch (reader workspace buffers
// satisfy this). The old per-row vector path was alignas(64); the scalar
// to_float fallbacks do not require it, but SIMD backend paths may.
bool decode_rows_scratch_bytes(
    const codec_desc & desc,
    size_t & out_bytes,
    std::string * err = nullptr
);

// Convenience wrapper to decode all rows
std::vector<float> decode_matrix(
    const encoded_matrix & em,
    value_domain target_domain = value_domain::canonical
);

// Forward-WHT transformation helper on row-major float matrix (for testing rotated domain equivalence)
// Transforms each 128-element group along columns using the same WHT as TurboQuant encoding.
void apply_forward_wht_matrix(
    float * data,
    uint64_t rows,
    uint64_t padded_cols
);

// Exact versioned serialization of codec_desc (fixed 100-byte format, fixed-endian, magic header)
std::vector<uint8_t> serialize_desc(const codec_desc & desc);
bool deserialize_desc(const uint8_t * data, size_t size, codec_desc & out_desc, std::string * err = nullptr);

// Measured bytes of the process-global codec tables shared by every encoded
// stream (TurboQuant WHT rotation sign tables). Reported as
// xkv_codec_shared_bytes; never estimated per stream, never double-counted.
size_t llama_xkv_codec_shared_table_bytes();

} // namespace llama_xkv
