#include "llama-xkv-codec.h"

#include "ggml.h"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <sstream>
#include <algorithm>

namespace llama_xkv {

static const float TURBO_WHT_S1[128] = {
    -1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,
    -1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,1,
    -1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1,
    -1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1
};

static const float TURBO_WHT_S2[128] = {
    1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,
    1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,
    1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,
    1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1
};

static uint64_t compute_turbo_table_fingerprint(uint64_t seed, uint32_t group_size, ggml_type type) {
    uint64_t hash = 14695981039346656037ULL;
    auto add_byte = [&](uint8_t b) {
        hash ^= b;
        hash *= 1099511628211ULL;
    };
    auto add_u64 = [&](uint64_t v) {
        for (int i = 0; i < 8; ++i) {
            add_byte((uint8_t)(v >> (i * 8)));
        }
    };
    add_u64(seed);
    add_u64(group_size);
    add_u64((uint64_t)type);

    for (int i = 0; i < 128; ++i) {
        int8_t s1 = (int8_t)TURBO_WHT_S1[i];
        int8_t s2 = (int8_t)TURBO_WHT_S2[i];
        add_byte((uint8_t)s1);
        add_byte((uint8_t)s2);
    }
    return hash;
}

size_t llama_xkv_codec_shared_table_bytes() {
    return sizeof(TURBO_WHT_S1) + sizeof(TURBO_WHT_S2);
}

uint64_t encoded_matrix_bytes(const codec_desc & desc) {
    std::string err;
    if (!desc.validate(&err)) {
        throw std::invalid_argument("invalid codec descriptor: " + err);
    }

    if (desc.padded_shape.rows == 0 || desc.padded_shape.cols == 0) {
        return 0;
    }

    uint64_t rows = desc.padded_shape.rows;
    uint64_t stride = desc.row_stride_bytes;
    if (stride != 0 && rows > UINT64_MAX / stride) {
        throw std::overflow_error("encoded_matrix_bytes: integer overflow calculating total bytes");
    }

    return rows * stride;
}

bool codec_desc::validate(std::string * err) const {
    if (version != 2) {
        if (err) *err = "unsupported codec_desc version " + std::to_string(version);
        return false;
    }

    if (role != factor_role::a_k && role != factor_role::b_k &&
        role != factor_role::a_v && role != factor_role::b_v &&
        role != factor_role::landmark) {
        if (err) *err = "unknown factor_role";
        return false;
    }

    if (orient != orientation::token_major && orient != orientation::feature_major_transposed) {
        if (err) *err = "unknown orientation";
        return false;
    }

    if (type != GGML_TYPE_F32 && type != GGML_TYPE_F16 && type != GGML_TYPE_Q8_0 &&
        type != GGML_TYPE_TURBO2_0 && type != GGML_TYPE_TURBO3_0 && type != GGML_TYPE_TURBO4_0) {
        // ggml_type_name returns nullptr for out-of-range/corrupt types; never
        // construct std::string from null (would throw std::logic_error and
        // abort fail-closed validation). Report the raw integer instead.
        const char * type_name = (type >= 0 && type < GGML_TYPE_COUNT) ? ggml_type_name(type) : nullptr;
        if (err) {
            *err = "unsupported ggml_type for xkv codec: ";
            *err += (type_name != nullptr ? type_name : ("type#" + std::to_string((int) type)));
        }
        return false;
    }

    int64_t blk = ggml_blck_size(type);
    if (blk <= 0) {
        if (err) *err = "invalid ggml block size for type";
        return false;
    }

    if (block_size != (uint32_t)blk) {
        if (err) *err = "block_size mismatch with ggml type block size";
        return false;
    }

    if (logical_shape.cols > padded_shape.cols) {
        if (err) *err = "logical cols cannot exceed padded cols";
        return false;
    }
    if (logical_shape.rows > padded_shape.rows) {
        if (err) *err = "logical rows cannot exceed padded rows";
        return false;
    }

    bool is_turbo = (type == GGML_TYPE_TURBO2_0 || type == GGML_TYPE_TURBO3_0 || type == GGML_TYPE_TURBO4_0);

    // Domain validation
    if (is_turbo) {
        if (stored_domain != value_domain::turbo_rotated) {
            if (err) *err = "TurboQuant types must have stored_domain = value_domain::turbo_rotated";
            return false;
        }
        if (table_fingerprint == 0) {
            if (err) *err = "TurboQuant types must have non-zero table_fingerprint";
            return false;
        }
        if (format_revision == 0) {
            if (err) *err = "TurboQuant types must have non-zero format_revision";
            return false;
        }
        uint64_t compiled_rev = ggml_turbo_layout_fingerprint(type);
        if (compiled_rev == 0) {
            if (err) *err = "unsupported TurboQuant type in compiled backend";
            return false;
        }
        if (format_revision != compiled_rev) {
            if (err) *err = "format_revision mismatch with compiled backend layout";
            return false;
        }
    } else {
        if (stored_domain != value_domain::canonical) {
            if (err) *err = "Non-Turbo types must have stored_domain = value_domain::canonical";
            return false;
        }
        if (decoded_domain != value_domain::canonical) {
            if (err) *err = "Non-Turbo types must have decoded_domain = value_domain::canonical";
            return false;
        }
        if (format_revision != 0) {
            if (err) *err = "Non-Turbo types must have format_revision = 0";
            return false;
        }
    }

    // Zero-shape behavior: if shape is 0, allow 0 stride
    if (padded_shape.rows == 0 || padded_shape.cols == 0) {
        return true;
    }

    if (padded_shape.cols % block_size != 0) {
        if (err) *err = "padded cols must be a multiple of block size";
        return false;
    }

    // Check TurboQuant group sizes: v1 strictly requires group_size == 128 (reject 64)
    if (is_turbo) {
        if (group_size != 128) {
            if (err) *err = "TurboQuant group_size must be 128 in XKV v1 (group_size 64 rejected)";
            return false;
        }
        if (padded_shape.cols % 128 != 0) {
            if (err) *err = "padded cols must be divisible by 128 for Turbo types";
            return false;
        }
    }

    size_t expected_row_bytes = ggml_row_size(type, (int64_t)padded_shape.cols);
    if (row_stride_bytes < expected_row_bytes) {
        if (err) *err = "row_stride_bytes is smaller than ggml_row_size";
        return false;
    }

    return true;
}

uint64_t codec_desc::fingerprint() const {
    uint64_t hash = 14695981039346656037ULL;
    auto add_u64 = [&](uint64_t v) {
        for (int i = 0; i < 8; ++i) {
            hash ^= (uint8_t)(v >> (i * 8));
            hash *= 1099511628211ULL;
        }
    };
    add_u64(version);
    add_u64((uint64_t)role);
    add_u64((uint64_t)type);
    add_u64((uint64_t)orient);
    add_u64(logical_shape.rows);
    add_u64(logical_shape.cols);
    add_u64(padded_shape.rows);
    add_u64(padded_shape.cols);
    add_u64(row_stride_bytes);
    add_u64(block_size);
    add_u64(group_size);
    add_u64(seed);
    add_u64(table_fingerprint);
    add_u64((uint64_t)stored_domain);
    add_u64((uint64_t)decoded_domain);
    add_u64(format_revision);
    return hash;
}

bool codec_desc::operator==(const codec_desc & o) const {
    return version == o.version &&
           role == o.role &&
           type == o.type &&
           orient == o.orient &&
           logical_shape == o.logical_shape &&
           padded_shape == o.padded_shape &&
           row_stride_bytes == o.row_stride_bytes &&
           block_size == o.block_size &&
           group_size == o.group_size &&
           seed == o.seed &&
           table_fingerprint == o.table_fingerprint &&
           stored_domain == o.stored_domain &&
           decoded_domain == o.decoded_domain &&
           format_revision == o.format_revision;
}

codec_desc make_codec_desc(
    factor_role  role,
    ggml_type    type,
    orientation  orient,
    matrix_shape logical_shape,
    uint32_t     group_size,
    uint64_t     seed
) {
    codec_desc desc;
    desc.version = 2;
    desc.role = role;
    desc.type = type;
    desc.orient = orient;
    desc.logical_shape = logical_shape;
    desc.seed = seed;

    // Fail closed BEFORE any ggml type-table contact: out-of-range types
    // (e.g. GGML_TYPE_COUNT from corrupt configs) abort in ggml asserts.
    if ((int) type < 0 || (int) type >= (int) GGML_TYPE_COUNT) {
        throw std::invalid_argument("make_codec_desc: ggml_type out of range");
    }

    int64_t blk = ggml_blck_size(type);
    if (blk <= 0) {
        throw std::invalid_argument("unknown or invalid ggml_type block size");
    }
    desc.block_size = (uint32_t)blk;

    bool is_turbo = (type == GGML_TYPE_TURBO2_0 || type == GGML_TYPE_TURBO3_0 || type == GGML_TYPE_TURBO4_0);

    if (is_turbo) {
        if (group_size == 0) {
            group_size = 128;
        }
        if (group_size != 128) {
            throw std::invalid_argument("TurboQuant types require group_size 128 in XKV v1 (64 rejected)");
        }
        desc.group_size = 128;
        desc.stored_domain = value_domain::turbo_rotated;
        desc.decoded_domain = value_domain::canonical;
        uint64_t rev = ggml_turbo_layout_fingerprint(type);
        if (rev == 0) {
            throw std::invalid_argument("unsupported TurboQuant type or incompatible compiled layout for XKV: " + std::string(ggml_type_name(type)));
        }
        desc.format_revision = rev;
        desc.table_fingerprint = compute_turbo_table_fingerprint(seed, 128, type);
    } else {
        desc.group_size = 0;
        desc.stored_domain = value_domain::canonical;
        desc.decoded_domain = value_domain::canonical;
        desc.format_revision = 0;
        desc.table_fingerprint = 0;
    }

    if (logical_shape.rows == 0 || logical_shape.cols == 0) {
        desc.padded_shape = logical_shape;
        desc.row_stride_bytes = 0;
    } else {
        uint64_t align_cols = desc.block_size;
        if (desc.group_size > 0 && desc.group_size > align_cols) {
            align_cols = desc.group_size;
        }

        uint64_t padded_cols = logical_shape.cols;
        if (padded_cols % align_cols != 0) {
            padded_cols += (align_cols - (padded_cols % align_cols));
        }
        desc.padded_shape.rows = logical_shape.rows;
        desc.padded_shape.cols = padded_cols;
        desc.row_stride_bytes = ggml_row_size(type, (int64_t)desc.padded_shape.cols);
    }

    std::string err;
    if (!desc.validate(&err)) {
        throw std::invalid_argument("make_codec_desc validation failed: " + err);
    }

    return desc;
}

encoded_matrix encode_matrix(
    const codec_desc & desc,
    const float * src,
    size_t src_elements
) {
    std::string err;
    if (!desc.validate(&err)) {
        throw std::invalid_argument("encode_matrix invalid descriptor: " + err);
    }

    if (desc.logical_shape.rows != 0 && desc.logical_shape.cols > UINT64_MAX / desc.logical_shape.rows) {
        throw std::overflow_error("encode_matrix: integer overflow computing logical shape elements");
    }
    uint64_t total_logical_elements = desc.logical_shape.rows * desc.logical_shape.cols;
    if (total_logical_elements > 0 && src_elements < (size_t)total_logical_elements) {
        throw std::invalid_argument("encode_matrix: src_elements is smaller than logical shape elements");
    }

    encoded_matrix result;
    result.desc = desc;

    uint64_t total_bytes = encoded_matrix_bytes(desc);
    result.bytes.resize((size_t)total_bytes, 0);

    if (total_bytes == 0 || total_logical_elements == 0) {
        return result;
    }

    uint64_t rows = desc.logical_shape.rows;
    uint64_t log_cols = desc.logical_shape.cols;
    uint64_t pad_cols = desc.padded_shape.cols;

    std::vector<float> row_buf(pad_cols, 0.0f);
    const auto * traits = ggml_get_type_traits(desc.type);

    for (uint64_t r = 0; r < rows; ++r) {
        const float * row_src = src + r * log_cols;
        std::memcpy(row_buf.data(), row_src, (size_t)(log_cols * sizeof(float)));
        if (pad_cols > log_cols) {
            std::memset(row_buf.data() + log_cols, 0, (size_t)((pad_cols - log_cols) * sizeof(float)));
        }

        uint8_t * row_dst = result.bytes.data() + r * desc.row_stride_bytes;

        switch (desc.type) {
            case GGML_TYPE_F32: {
                std::memcpy(row_dst, row_buf.data(), (size_t)(pad_cols * sizeof(float)));
                break;
            }
            case GGML_TYPE_F16:
            case GGML_TYPE_Q8_0: {
                if (traits && traits->from_float_ref) {
                    size_t row_bytes = ggml_row_size(desc.type, (int64_t)pad_cols);
                    alignas(64) std::vector<uint8_t> aligned_scratch(row_bytes);
                    traits->from_float_ref(row_buf.data(), aligned_scratch.data(), (int64_t)pad_cols);
                    std::memcpy(row_dst, aligned_scratch.data(), row_bytes);
                } else {
                    throw std::runtime_error("missing from_float_ref for type in encode_matrix");
                }
                break;
            }
            case GGML_TYPE_TURBO2_0:
            case GGML_TYPE_TURBO3_0:
            case GGML_TYPE_TURBO4_0: {
                if (!ggml_quantize_turbo_row(desc.type, row_buf.data(), row_dst, (int64_t)pad_cols, 128)) {
                    throw std::runtime_error("ggml_quantize_turbo_row failed in encode_matrix");
                }
                break;
            }
            default:
                throw std::invalid_argument("unsupported type in encode_matrix");
        }
    }

    return result;
}

void decode_rows(
    const encoded_matrix & em,
    const uint64_t * row_indices,
    size_t n_rows,
    float * dst,
    size_t dst_capacity_elements,
    uint8_t * tmp,
    size_t tmp_bytes,
    value_domain target_domain
) {
    const codec_desc & desc = em.desc;
    std::string err;
    if (!desc.validate(&err)) {
        throw std::invalid_argument("decode_rows invalid descriptor: " + err);
    }

    bool is_turbo = (desc.type == GGML_TYPE_TURBO2_0 || desc.type == GGML_TYPE_TURBO3_0 || desc.type == GGML_TYPE_TURBO4_0);
    if (!is_turbo && target_domain != value_domain::canonical) {
        throw std::invalid_argument("decode_rows: non-Turbo types only support target_domain = value_domain::canonical");
    }

    uint64_t expected_bytes = encoded_matrix_bytes(desc);
    if (em.bytes.size() != (size_t)expected_bytes) {
        throw std::invalid_argument("decode_rows: em.bytes.size() does not match expected descriptor byte size");
    }

    if (n_rows == 0 || desc.padded_shape.rows == 0 || desc.padded_shape.cols == 0) {
        return;
    }

    uint64_t pad_cols = desc.padded_shape.cols;
    if (pad_cols != 0 && (uint64_t)n_rows > UINT64_MAX / pad_cols) {
        throw std::overflow_error("decode_rows: integer overflow calculating required decoded elements");
    }
    uint64_t req_elements = (uint64_t)n_rows * pad_cols;
    if (dst_capacity_elements < (size_t)req_elements) {
        throw std::invalid_argument("decode_rows: dst buffer capacity is smaller than required decoded elements");
    }

    // Caller scratch covers exactly one row for the F16/Q8_0 alignment copy.
    // F32/Turbo never touch tmp. Checked after the empty-decode early return
    // so zero-row decodes stay callable with nullptr/0 scratch.
    const bool needs_tmp = (desc.type == GGML_TYPE_F16 || desc.type == GGML_TYPE_Q8_0);
    const size_t row_bytes = ggml_row_size(desc.type, (int64_t)pad_cols);
    if (needs_tmp && tmp_bytes < row_bytes) {
        throw std::invalid_argument("decode_rows: caller scratch smaller than one row");
    }

    uint64_t total_rows = desc.padded_shape.rows;
    const auto * traits = ggml_get_type_traits(desc.type);

    for (size_t i = 0; i < n_rows; ++i) {
        uint64_t r = row_indices[i];
        if (r >= total_rows) {
            throw std::out_of_range("decode_rows: row index out of bounds: " + std::to_string(r));
        }

        if (r != 0 && desc.row_stride_bytes > UINT64_MAX / r) {
            throw std::overflow_error("decode_rows: integer overflow calculating row byte offset");
        }
        uint64_t row_byte_offset = r * desc.row_stride_bytes;
        size_t expected_row_bytes = ggml_row_size(desc.type, (int64_t)pad_cols);
        if (row_byte_offset + expected_row_bytes > em.bytes.size()) {
            throw std::out_of_range("decode_rows: row bytes out of buffer bounds");
        }

        const uint8_t * row_src = em.bytes.data() + row_byte_offset;
        float * row_dst = dst + i * pad_cols;

        switch (desc.type) {
            case GGML_TYPE_F32: {
                std::memcpy(row_dst, row_src, (size_t)(pad_cols * sizeof(float)));
                break;
            }
            case GGML_TYPE_F16:
            case GGML_TYPE_Q8_0: {
                if (traits && traits->to_float) {
                    // No heap here: single caller-scratch copy feeds to_float.
                    // Callers should pass 64-byte aligned scratch for SIMD paths.
                    std::memcpy(tmp, row_src, row_bytes);
                    traits->to_float(tmp, row_dst, (int64_t)pad_cols);
                } else {
                    throw std::runtime_error("missing to_float for type in decode_rows");
                }
                break;
            }
            case GGML_TYPE_TURBO2_0:
            case GGML_TYPE_TURBO3_0:
            case GGML_TYPE_TURBO4_0: {
                enum ggml_turbo_decode_domain dom = (target_domain == value_domain::canonical) ? GGML_TURBO_DECODE_CANONICAL : GGML_TURBO_DECODE_ROTATED;
                if (!ggml_dequantize_turbo_row(desc.type, row_src, row_dst, (int64_t)pad_cols, 128, dom)) {
                    throw std::runtime_error("ggml_dequantize_turbo_row failed in decode_rows");
                }
                break;
            }
            default:
                throw std::invalid_argument("unsupported type in decode_rows");
        }
    }
}

bool decode_rows_scratch_bytes(
    const codec_desc & desc,
    size_t & out_bytes,
    std::string * err
) {
    out_bytes = 0;
    std::string verr;
    if (!desc.validate(&verr)) {
        if (err) *err = verr;
        return false;
    }
    if (desc.type == GGML_TYPE_F16 || desc.type == GGML_TYPE_Q8_0) {
        if (desc.padded_shape.cols == 0) return true; // nothing to decode
        out_bytes = ggml_row_size(desc.type, (int64_t)desc.padded_shape.cols);
    }
    return true;
}

void decode_rows(
    const encoded_matrix & em,
    const uint64_t * row_indices,
    size_t n_rows,
    float * dst,
    size_t dst_capacity_elements,
    value_domain target_domain
) {
    // Reference convenience: one reusable scratch allocation shared by all
    // rows, then the shared zero-heap core above. Numerics are bit-identical
    // to the scratch overload (same memcpy + to_float per row).
    const codec_desc & desc = em.desc;
    std::vector<uint8_t> scratch;
    uint8_t * tmp = nullptr;
    size_t tmp_bytes = 0;
    if ((desc.type == GGML_TYPE_F16 || desc.type == GGML_TYPE_Q8_0) &&
        n_rows != 0 && desc.padded_shape.rows != 0 && desc.padded_shape.cols != 0) {
        // Allocate only for descriptors that validate; invalid descriptors
        // fall through with null scratch so the core throws the original
        // validation error text unchanged.
        std::string verr;
        if (desc.validate(&verr)) {
            scratch.resize(ggml_row_size(desc.type, (int64_t)desc.padded_shape.cols));
            tmp = scratch.data();
            tmp_bytes = scratch.size();
        }
    }
    decode_rows(em, row_indices, n_rows, dst, dst_capacity_elements, tmp, tmp_bytes, target_domain);
}

std::vector<float> decode_matrix(
    const encoded_matrix & em,
    value_domain target_domain
) {
    uint64_t rows = em.desc.logical_shape.rows;
    uint64_t pad_cols = em.desc.padded_shape.cols;
    if (rows == 0 || pad_cols == 0) {
        return {};
    }
    if (rows > UINT64_MAX / pad_cols) {
        throw std::overflow_error("decode_matrix: integer overflow computing decoded element count");
    }
    std::vector<float> result((size_t)(rows * pad_cols), 0.0f);
    std::vector<uint64_t> row_indices((size_t)rows);
    for (uint64_t i = 0; i < rows; ++i) {
        row_indices[(size_t)i] = i;
    }
    decode_rows(em, row_indices.data(), (size_t)rows, result.data(), result.size(), target_domain);
    return result;
}

// Explicit versioned serialization: fixed 100 bytes, fixed little-endian layout, zero padding/rejection of trailing/truncated bytes
// Magic: "XKV1" (0x31564B58)
static const uint32_t SERIALIZE_MAGIC = 0x31564B58;
static const size_t SERIALIZE_SIZE = 100;

static void write_u32_le(std::vector<uint8_t> & buf, uint32_t v) {
    buf.push_back((uint8_t)(v & 0xFF));
    buf.push_back((uint8_t)((v >> 8) & 0xFF));
    buf.push_back((uint8_t)((v >> 16) & 0xFF));
    buf.push_back((uint8_t)((v >> 24) & 0xFF));
}

static void write_u64_le(std::vector<uint8_t> & buf, uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        buf.push_back((uint8_t)((v >> (i * 8)) & 0xFF));
    }
}

static uint32_t read_u32_le(const uint8_t * p) {
    return (uint32_t)p[0] |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static uint64_t read_u64_le(const uint8_t * p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) {
        v |= ((uint64_t)p[i] << (i * 8));
    }
    return v;
}

std::vector<uint8_t> serialize_desc(const codec_desc & desc) {
    std::vector<uint8_t> buf;
    buf.reserve(SERIALIZE_SIZE);

    write_u32_le(buf, SERIALIZE_MAGIC);
    write_u32_le(buf, desc.version);
    write_u32_le(buf, (uint32_t)desc.role);
    write_u32_le(buf, (uint32_t)desc.type);
    write_u32_le(buf, (uint32_t)desc.orient);

    write_u64_le(buf, desc.logical_shape.rows);
    write_u64_le(buf, desc.logical_shape.cols);
    write_u64_le(buf, desc.padded_shape.rows);
    write_u64_le(buf, desc.padded_shape.cols);
    write_u64_le(buf, desc.row_stride_bytes);

    write_u32_le(buf, desc.block_size);
    write_u32_le(buf, desc.group_size);

    write_u64_le(buf, desc.seed);
    write_u64_le(buf, desc.table_fingerprint);

    write_u32_le(buf, (uint32_t)desc.stored_domain);
    write_u32_le(buf, (uint32_t)desc.decoded_domain);
    write_u64_le(buf, desc.format_revision);

    return buf;
}

bool deserialize_desc(const uint8_t * data, size_t size, codec_desc & out_desc, std::string * err) {
    if (size < SERIALIZE_SIZE) {
        if (err) *err = "deserialize_desc: input truncated (size " + std::to_string(size) + " < " + std::to_string(SERIALIZE_SIZE) + ")";
        return false;
    }
    if (size > SERIALIZE_SIZE) {
        if (err) *err = "deserialize_desc: trailing bytes rejected (size " + std::to_string(size) + " > " + std::to_string(SERIALIZE_SIZE) + ")";
        return false;
    }

    uint32_t magic = read_u32_le(data);
    if (magic != SERIALIZE_MAGIC) {
        if (err) *err = "deserialize_desc: invalid magic header";
        return false;
    }

    codec_desc d;
    d.version = read_u32_le(data + 4);
    if (d.version == 1) {
        if (err) *err = "deserialize_desc: v1 descriptor rejected (missing required format_revision / 100-byte layout)";
        return false;
    }
    // Canonical-form gate: role/orient/domains are uint8_t enums serialized as
    // u32, so upper bytes must be zero. Without this, high-byte corruption
    // (e.g. desc_bytes[10] ^= 0xFF) truncates silently on narrowing cast and
    // validates as the original value. Reject fail-closed before casting.
    const uint32_t raw_role = read_u32_le(data + 8);
    if (raw_role > (uint32_t) factor_role::landmark) {
        if (err) *err = "deserialize_desc: non-canonical corrupt factor_role";
        return false;
    }
    d.role = (factor_role) raw_role;
    d.type = (ggml_type)read_u32_le(data + 12);
    const uint32_t raw_orient = read_u32_le(data + 16);
    if (raw_orient > (uint32_t) orientation::feature_major_transposed) {
        if (err) *err = "deserialize_desc: non-canonical corrupt orientation";
        return false;
    }
    d.orient = (orientation) raw_orient;

    d.logical_shape.rows = read_u64_le(data + 20);
    d.logical_shape.cols = read_u64_le(data + 28);
    d.padded_shape.rows = read_u64_le(data + 36);
    d.padded_shape.cols = read_u64_le(data + 44);
    d.row_stride_bytes = read_u64_le(data + 52);

    d.block_size = read_u32_le(data + 60);
    d.group_size = read_u32_le(data + 64);

    d.seed = read_u64_le(data + 68);
    d.table_fingerprint = read_u64_le(data + 76);

    const uint32_t raw_stored = read_u32_le(data + 84);
    if (raw_stored > (uint32_t) value_domain::turbo_rotated) {
        if (err) *err = "deserialize_desc: non-canonical corrupt stored_domain";
        return false;
    }
    d.stored_domain = (value_domain) raw_stored;
    const uint32_t raw_decoded = read_u32_le(data + 88);
    if (raw_decoded > (uint32_t) value_domain::turbo_rotated) {
        if (err) *err = "deserialize_desc: non-canonical corrupt decoded_domain";
        return false;
    }
    d.decoded_domain = (value_domain) raw_decoded;
    d.format_revision = read_u64_le(data + 92);

    if (!d.validate(err)) {
        return false;
    }

    out_desc = d;
    return true;
}

bool validate_factor_pair_compatibility(
    const codec_desc & desc_a,
    const codec_desc & desc_b,
    std::string * err
) {
    std::string err_a, err_b;
    if (!desc_a.validate(&err_a)) {
        if (err) *err = "desc_a invalid: " + err_a;
        return false;
    }
    if (!desc_b.validate(&err_b)) {
        if (err) *err = "desc_b invalid: " + err_b;
        return false;
    }

    // Rank compatibility: A is (n x r), B^T is (m x r) so both inner cols must match
    if (desc_a.logical_shape.cols != desc_b.logical_shape.cols) {
        if (err) *err = "factor pair logical rank mismatch: A has " +
                        std::to_string(desc_a.logical_shape.cols) + ", B has " +
                        std::to_string(desc_b.logical_shape.cols);
        return false;
    }
    if (desc_a.padded_shape.cols != desc_b.padded_shape.cols) {
        if (err) *err = "factor pair padded rank mismatch: A has " +
                        std::to_string(desc_a.padded_shape.cols) + ", B has " +
                        std::to_string(desc_b.padded_shape.cols);
        return false;
    }

    bool a_is_turbo = (desc_a.type == GGML_TYPE_TURBO2_0 || desc_a.type == GGML_TYPE_TURBO3_0 || desc_a.type == GGML_TYPE_TURBO4_0);
    bool b_is_turbo = (desc_b.type == GGML_TYPE_TURBO2_0 || desc_b.type == GGML_TYPE_TURBO3_0 || desc_b.type == GGML_TYPE_TURBO4_0);

    if (a_is_turbo && b_is_turbo) {
        // Both are Turbo: must share identical table_fingerprint, seed, and format_revision for pair rotation identity
        if (desc_a.table_fingerprint != desc_b.table_fingerprint) {
            if (err) *err = "factor pair table_fingerprint mismatch: A has " +
                            std::to_string(desc_a.table_fingerprint) + ", B has " +
                            std::to_string(desc_b.table_fingerprint);
            return false;
        }
        if (desc_a.seed != desc_b.seed) {
            if (err) *err = "factor pair seed mismatch: A has " +
                            std::to_string(desc_a.seed) + ", B has " +
                            std::to_string(desc_b.seed);
            return false;
        }
        if (desc_a.format_revision != desc_b.format_revision) {
            if (err) *err = "factor pair format_revision mismatch: A has " +
                            std::to_string(desc_a.format_revision) + ", B has " +
                            std::to_string(desc_b.format_revision);
            return false;
        }
    } else if (!a_is_turbo && !b_is_turbo) {
        // Both are non-Turbo (canonical domain, e.g. FP32, FP16, Q8_0): fully compatible via canonical decode
    } else {
        // Explicitly mixed pair (e.g. A Turbo, B F16/Q8 or vice versa as in R1 ablation):
        // Allowed by decoding both to canonical domain before GEMM.
        // Direct rotated domain GEMM is NOT allowed.
    }

    return true;
}

bool can_direct_rotated_gemm(
    const codec_desc & desc_a,
    const codec_desc & desc_b
) {
    if (!validate_factor_pair_compatibility(desc_a, desc_b)) {
        return false;
    }
    bool a_is_turbo = (desc_a.type == GGML_TYPE_TURBO2_0 || desc_a.type == GGML_TYPE_TURBO3_0 || desc_a.type == GGML_TYPE_TURBO4_0);
    bool b_is_turbo = (desc_b.type == GGML_TYPE_TURBO2_0 || desc_b.type == GGML_TYPE_TURBO3_0 || desc_b.type == GGML_TYPE_TURBO4_0);

    if (a_is_turbo && b_is_turbo &&
        desc_a.table_fingerprint == desc_b.table_fingerprint &&
        desc_a.seed == desc_b.seed &&
        desc_a.format_revision == desc_b.format_revision) {
        return true;
    }
    return false;
}

void assert_factor_pair_compatible(const codec_desc & desc_a, const codec_desc & desc_b) {
    std::string err;
    if (!validate_factor_pair_compatibility(desc_a, desc_b, &err)) {
        throw std::invalid_argument("factor pair incompatible: " + err);
    }
}

void apply_forward_wht_matrix(float * data, uint64_t rows, uint64_t padded_cols) {
    if (!data || rows == 0 || padded_cols == 0) return;
    if (padded_cols % 128 != 0) {
        throw std::invalid_argument("apply_forward_wht_matrix: padded_cols must be divisible by 128");
    }
    uint64_t n_groups = padded_cols / 128;
    for (uint64_t r = 0; r < rows; ++r) {
        for (uint64_t g = 0; g < n_groups; ++g) {
            ggml_turbo_wht_row(data + r * padded_cols + g * 128, 128);
        }
    }
}

} // namespace llama_xkv
