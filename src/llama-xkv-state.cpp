#include "llama-xkv-state.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <new>
#include <set>
#include <utility>

namespace llama_xkv {

namespace {

// ---------------------------------------------------------------------------
// Checksums
// ---------------------------------------------------------------------------

uint64_t fnv1a64(const uint8_t * data, size_t size, uint64_t seed) {
    uint64_t h = seed;
    for (size_t i = 0; i < size; ++i) {
        h ^= data[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

constexpr uint64_t FNV_OFFSET = 0xcbf29ce484222325ULL;
constexpr uint64_t PPM_DENOM  = 1000000ULL;

void set_err(std::string * err, const std::string & msg) {
    if (err != nullptr) {
        *err = msg;
    }
}

// Checked add / mul: false on overflow.
bool add_ok(uint64_t a, uint64_t b, uint64_t & out) {
    if (b > std::numeric_limits<uint64_t>::max() - a) {
        return false;
    }
    out = a + b;
    return true;
}

bool mul_ok(uint64_t a, uint64_t b, uint64_t & out) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
        return false;
    }
    out = a * b;
    return true;
}

// ---------------------------------------------------------------------------
// Size accumulator (checked; mirrors the encode walk exactly)
// ---------------------------------------------------------------------------

struct sizer {
    uint64_t size = 0;
    bool     ok   = true;

    void add(uint64_t n) {
        uint64_t next = 0;
        if (!add_ok(size, n, next)) {
            ok = false;
            return;
        }
        size = next;
    }

    void add_n(uint64_t n, uint64_t elem) {
        uint64_t prod = 0;
        if (!mul_ok(n, elem, prod)) {
            ok = false;
            return;
        }
        add(prod);
    }

    void put_raw(size_t n) { add((uint64_t) n); }
    void put_u8() { add(1); }
    void put_u32() { add(4); }
    void put_u64() { add(8); }
    void put_u32_vec(const std::vector<uint32_t> & v) {
        add(8);
        add_n((uint64_t) v.size(), 4);
    }
};

// ---------------------------------------------------------------------------
// Writer (little-endian, deterministic)
// ---------------------------------------------------------------------------

struct writer {
    std::vector<uint8_t> buf;

    void put_u8(uint8_t v) { buf.push_back(v); }

    void put_u32(uint32_t v) {
        buf.push_back((uint8_t) (v & 0xFF));
        buf.push_back((uint8_t) ((v >> 8) & 0xFF));
        buf.push_back((uint8_t) ((v >> 16) & 0xFF));
        buf.push_back((uint8_t) ((v >> 24) & 0xFF));
    }

    void put_u64(uint64_t v) {
        for (int i = 0; i < 8; ++i) {
            buf.push_back((uint8_t) ((v >> (i * 8)) & 0xFF));
        }
    }

    void put_raw(const uint8_t * data, size_t size) {
        if (size == 0) {
            return;
        }
        buf.insert(buf.end(), data, data + size);
    }

    void put_bytes(const std::vector<uint8_t> & v) {
        put_u64((uint64_t) v.size());
        put_raw(v.data(), v.size());
    }

    void put_u32_vec(const std::vector<uint32_t> & v) {
        put_u64((uint64_t) v.size());
        for (uint32_t x : v) {
            put_u32(x);
        }
    }

    void put_hash(const uint8_t h[32]) { put_raw(h, 32); }
};

// ---------------------------------------------------------------------------
// Reader (bounds-checked, no allocation before length validation)
// ---------------------------------------------------------------------------

struct reader {
    const uint8_t * base = nullptr;
    size_t          size = 0;
    size_t          pos  = 0;

    bool need(size_t k, std::string * err, const char * what) {
        if (k > size || pos > size - k) {
            set_err(err, std::string("decode: truncated ") + what);
            return false;
        }
        return true;
    }

    bool get_u8(uint8_t & v, std::string * err, const char * what) {
        if (!need(1, err, what)) {
            return false;
        }
        v = base[pos++];
        return true;
    }

    bool get_u32(uint32_t & v, std::string * err, const char * what) {
        if (!need(4, err, what)) {
            return false;
        }
        v = (uint32_t) base[pos] | ((uint32_t) base[pos + 1] << 8) |
            ((uint32_t) base[pos + 2] << 16) | ((uint32_t) base[pos + 3] << 24);
        pos += 4;
        return true;
    }

    bool get_u64(uint64_t & v, std::string * err, const char * what) {
        if (!need(8, err, what)) {
            return false;
        }
        v = 0;
        for (int i = 0; i < 8; ++i) {
            v |= ((uint64_t) base[pos + i] << (i * 8));
        }
        pos += 8;
        return true;
    }

    bool get_hash(uint8_t h[32], std::string * err, const char * what) {
        if (!need(32, err, what)) {
            return false;
        }
        std::memcpy(h, base + pos, 32);
        pos += 32;
        return true;
    }

    size_t remaining() const { return pos <= size ? size - pos : 0; }
};

bool get_sized_bytes(reader & r, uint64_t len, uint64_t limit, std::vector<uint8_t> & out,
                     std::string * err, const char * what) {
    if (len > limit) {
        set_err(err, std::string("decode: ") + what + " length exceeds limit");
        return false;
    }
    if (len > r.remaining()) {
        set_err(err, std::string("decode: truncated ") + what + " (declared length exceeds buffer)");
        return false;
    }
    out.assign(r.base + r.pos, r.base + r.pos + (size_t) len);
    r.pos += (size_t) len;
    return true;
}

bool get_u32_vec(reader & r, uint64_t limit, std::vector<uint32_t> & out,
                 std::string * err, const char * what) {
    uint64_t n = 0;
    if (!r.get_u64(n, err, what)) {
        return false;
    }
    if (n > limit) {
        set_err(err, std::string("decode: ") + what + " count exceeds limit");
        return false;
    }
    if (n > r.remaining() / 4) {
        set_err(err, std::string("decode: truncated ") + what);
        return false;
    }
    out.resize((size_t) n);
    for (uint64_t i = 0; i < n; ++i) {
        uint32_t v = 0;
        if (!r.get_u32(v, err, what)) {
            return false;
        }
        out[(size_t) i] = v;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Closure / provenance helpers
// ---------------------------------------------------------------------------

// Slices [off, off+dim) must contiguously cover [0, total) exactly once.
bool check_slice_closure(const std::vector<uint32_t> & offs, const std::vector<uint32_t> & dims,
                         uint32_t total, std::string * err, const char * what) {
    if (offs.size() != dims.size()) {
        set_err(err, std::string("validate: ") + what + " offsets/dims size mismatch");
        return false;
    }
    if (offs.empty()) {
        set_err(err, std::string("validate: ") + what + " empty (owning layers empty)");
        return false;
    }
    if (total == 0) {
        set_err(err, std::string("validate: ") + what + " total dim is zero");
        return false;
    }
    std::vector<std::pair<uint32_t, uint32_t>> spans;
    spans.reserve(offs.size());
    for (size_t i = 0; i < offs.size(); ++i) {
        uint64_t end = (uint64_t) offs[i] + (uint64_t) dims[i];
        if (dims[i] == 0 || offs[i] >= total || end > total) {
            set_err(err, std::string("validate: ") + what + " slice out of range or empty");
            return false;
        }
        spans.emplace_back(offs[i], dims[i]);
    }
    std::sort(spans.begin(), spans.end());
    uint32_t cursor = 0;
    for (const auto & s : spans) {
        if (s.first != cursor) {
            set_err(err, std::string("validate: ") + what + " slices overlap or leave a gap");
            return false;
        }
        cursor += s.second;
    }
    if (cursor != total) {
        set_err(err, std::string("validate: ") + what + " slices do not cover total dim");
        return false;
    }
    return true;
}

bool strictly_ascending_u32(const std::vector<uint32_t> & v) {
    for (size_t i = 1; i < v.size(); ++i) {
        if (v[i] <= v[i - 1]) {
            return false;
        }
    }
    return true;
}

bool hash_is_zero(const uint8_t h[32]) {
    for (int i = 0; i < 32; ++i) {
        if (h[i] != 0) {
            return false;
        }
    }
    return true;
}

// Expected-match for one digest: an all-zero expected entry is unavailable and
// skipped; otherwise the image entry must agree byte-for-byte.
bool check_hash(const uint8_t img[32], const uint8_t exp[32], std::string * err, const char * what) {
    if (hash_is_zero(exp)) {
        return true;
    }
    if (std::memcmp(img, exp, 32) != 0) {
        set_err(err, std::string("validate: provenance mismatch (") + what + ")");
        return false;
    }
    return true;
}

uint64_t stream_checksum(const std::vector<uint8_t> & desc_bytes, const std::vector<uint8_t> & bytes) {
    uint64_t h = fnv1a64(desc_bytes.data(), desc_bytes.size(), FNV_OFFSET);
    h = fnv1a64(bytes.data(), bytes.size(), h);
    return h;
}

uint64_t stream_expected_bytes(const codec_desc & desc, bool & ok) {
    ok = false;
    uint64_t n = 0;
    try {
        n = encoded_matrix_bytes(desc);
    } catch (...) {
        return 0;
    }
    ok = true;
    return n;
}

bool valid_profile(uint32_t p) {
    return p == (uint32_t) LLAMA_XKV_STORAGE_PROFILE_REFERENCE ||
           p == (uint32_t) LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS ||
           p == (uint32_t) LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS_LANDMARKS;
}

bool valid_source(uint32_t s) {
    return s == (uint32_t) LLAMA_XKV_SOURCE_DECODED_HOT ||
           s == (uint32_t) LLAMA_XKV_SOURCE_PREROPE_CAPTURE;
}

} // namespace

// ---------------------------------------------------------------------------
// Provenance / config equality
// ---------------------------------------------------------------------------

bool xkv_state_provenance::has_model() const { return !hash_is_zero(model_sha256); }
bool xkv_state_provenance::has_tri_calibration() const { return !hash_is_zero(tri_calibration_sha256); }
bool xkv_state_provenance::has_source() const { return !hash_is_zero(source_sha256); }

bool xkv_state_provenance::operator==(const xkv_state_provenance & o) const {
    return std::memcmp(model_sha256, o.model_sha256, 32) == 0 &&
           std::memcmp(tri_calibration_sha256, o.tri_calibration_sha256, 32) == 0 &&
           std::memcmp(source_sha256, o.source_sha256, 32) == 0;
}

bool xkv_state_config::operator==(const xkv_state_config & o) const {
    return profile == o.profile && source == o.source && mode == o.mode &&
           group_size == o.group_size &&
           rank_k == o.rank_k && rank_v == o.rank_v && segment_tokens == o.segment_tokens &&
           chunk_tokens == o.chunk_tokens && sr_budget == o.sr_budget &&
           factor_a_k == o.factor_a_k && factor_b_k == o.factor_b_k &&
           factor_a_v == o.factor_a_v && factor_b_v == o.factor_b_v &&
           factor_balance == o.factor_balance && landmark_type == o.landmark_type &&
           landmark_refine == o.landmark_refine && landmark_refine_max_rows == o.landmark_refine_max_rows &&
           workspace_mib == o.workspace_mib && decode_cache_mib == o.decode_cache_mib &&
           store_mib == o.store_mib &&
           seed == o.seed && min_saving_ppm == o.min_saving_ppm &&
           min_coverage_ppm == o.min_coverage_ppm && factorizer == o.factorizer &&
           min_saving_bytes == o.min_saving_bytes && next_segment_id == o.next_segment_id &&
           next_alloc_id == o.next_alloc_id && next_seal_tx_nonce == o.next_seal_tx_nonce;
}

// ---------------------------------------------------------------------------
// Public checksums / limits
// ---------------------------------------------------------------------------

uint64_t xkv_state_checksum(const uint8_t * data, size_t size) {
    if (data == nullptr || size == 0) {
        return FNV_OFFSET;
    }
    return fnv1a64(data, size, FNV_OFFSET);
}

// SHA-256 compression core shared by one-shot and streaming forms.
namespace {
struct sha256_block_state {
    uint32_t h[8];
};
void sha256_compress_block(uint32_t h[8], const uint8_t block[64]) {
    static const uint32_t K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    auto rotr = [](uint32_t x, int n) { return (x >> n) | (x << (32 - n)); };
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
        w[i] = ((uint32_t) block[i * 4] << 24) | ((uint32_t) block[i * 4 + 1] << 16) |
               ((uint32_t) block[i * 4 + 2] << 8) | (uint32_t) block[i * 4 + 3];
    }
    for (int i = 16; i < 64; ++i) {
        const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; ++i) {
        const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        const uint32_t ch = (e & f) ^ ((~e) & g);
        const uint32_t t1 = hh + S1 + ch + K[i] + w[i];
        const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        const uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t t2 = S0 + mj;
        hh = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }
    h[0] += a;
    h[1] += b;
    h[2] += c;
    h[3] += d;
    h[4] += e;
    h[5] += f;
    h[6] += g;
    h[7] += hh;
}
} // namespace

void xkv_sha256_init(xkv_sha256_ctx & ctx) {
    ctx.h[0] = 0x6a09e667;
    ctx.h[1] = 0xbb67ae85;
    ctx.h[2] = 0x3c6ef372;
    ctx.h[3] = 0xa54ff53a;
    ctx.h[4] = 0x510e527f;
    ctx.h[5] = 0x9b05688c;
    ctx.h[6] = 0x1f83d9ab;
    ctx.h[7] = 0x5be0cd19;
    ctx.block_used = 0;
    ctx.total_len  = 0;
    std::memset(ctx.block, 0, sizeof(ctx.block));
}

void xkv_sha256_update(xkv_sha256_ctx & ctx, const uint8_t * data, size_t size) {
    if (data == nullptr || size == 0) {
        return;
    }
    ctx.total_len += (uint64_t) size;
    while (size > 0) {
        const size_t room = 64 - ctx.block_used;
        const size_t take = size < room ? size : room;
        std::memcpy(ctx.block + ctx.block_used, data, take);
        ctx.block_used += take;
        data += take;
        size -= take;
        if (ctx.block_used == 64) {
            sha256_compress_block(ctx.h, ctx.block);
            ctx.block_used = 0;
        }
    }
}

void xkv_sha256_final(xkv_sha256_ctx & ctx, uint8_t out[32]) {
    const uint64_t total_bits = ctx.total_len * 8;
    // Buffered bytes + 0x80 + zero pad to 56 mod 64, then the 64-bit length.
    // Worst case spans two blocks; tmp is sized for it (63 + 1 + 55 + 8 < 192).
    uint8_t tmp[192];
    size_t tlen = 0;
    std::memcpy(tmp, ctx.block, ctx.block_used);
    tlen = ctx.block_used;
    tmp[tlen++] = 0x80;
    while (tlen % 64 != 56) {
        tmp[tlen++] = 0;
    }
    for (int i = 7; i >= 0; --i) {
        tmp[tlen++] = (uint8_t) ((total_bits >> (i * 8)) & 0xFF);
    }
    for (size_t off = 0; off < tlen; off += 64) {
        uint8_t blk[64];
        std::memcpy(blk, tmp + off, 64);
        sha256_compress_block(ctx.h, blk);
    }
    for (int i = 0; i < 8; ++i) {
        out[i * 4]     = (uint8_t) ((ctx.h[i] >> 24) & 0xFF);
        out[i * 4 + 1] = (uint8_t) ((ctx.h[i] >> 16) & 0xFF);
        out[i * 4 + 2] = (uint8_t) ((ctx.h[i] >> 8) & 0xFF);
        out[i * 4 + 3] = (uint8_t) (ctx.h[i] & 0xFF);
    }
    // Scrub run state (no reuse after final).
    ctx.block_used = 0;
    ctx.total_len  = 0;
}

// Minimal self-contained SHA-256 (FIPS 180-4), one-shot form over streaming core.
void xkv_sha256(const uint8_t * data, size_t size, uint8_t out[32]) {
    if (out == nullptr) {
        return;
    }
    if (data == nullptr || size == 0) {
        xkv_sha256_ctx ctx;
        xkv_sha256_init(ctx);
        xkv_sha256_final(ctx, out);
        return;
    }
    xkv_sha256_ctx ctx;
    xkv_sha256_init(ctx);
    xkv_sha256_update(ctx, data, size);
    xkv_sha256_final(ctx, out);
}

bool xkv_source_files_sha256(const std::vector<std::string> & paths, uint8_t out[32],
                             std::string * err) {
    if (paths.empty()) {
        set_err(err, "source digest: no source paths retained");
        return false;
    }
    xkv_sha256_ctx ctx;
    xkv_sha256_init(ctx);
    auto mix_u64 = [&](uint64_t v) {
        uint8_t b[8];
        for (int i = 0; i < 8; ++i) {
            b[i] = (uint8_t) ((v >> (i * 8)) & 0xFF);
        }
        xkv_sha256_update(ctx, b, 8);
    };
    auto mix_bytes = [&](const uint8_t * d, size_t n) { xkv_sha256_update(ctx, d, n); };
    const char * dom = "XKV-SOURCE-V1";
    mix_bytes((const uint8_t *) dom, 13);
    mix_u64((uint64_t) paths.size());
    std::vector<uint8_t> chunk;
    try {
        chunk.resize(1 << 20);
    } catch (const std::bad_alloc &) {
        set_err(err, "source digest: scratch allocation failure");
        return false;
    }
    for (size_t i = 0; i < paths.size(); ++i) {
        // Path-independent content identity: canonical split index + file size
        // + full content only. Host paths (absolute or relative) are NEVER
        // hashed, so a byte-identical model deployed at another path restores
        // cleanly. Basename is deliberately omitted: split order is carried by
        // the canonical index, split count by the header above, and byte
        // identity by size + content. Display names remain covered by the
        // model_fp metadata screen, never by this digest.
        const std::string & path = paths[i];
        mix_u64((uint64_t) i); // split order domain-separates position
        FILE * f = std::fopen(path.c_str(), "rb");
        if (f == nullptr) {
            set_err(err, "source digest: cannot open " + path);
            return false;
        }
        if (std::fseek(f, 0, SEEK_END) != 0) {
            std::fclose(f);
            set_err(err, "source digest: cannot size " + path);
            return false;
        }
        const long sz = std::ftell(f);
        if (sz < 0) {
            std::fclose(f);
            set_err(err, "source digest: cannot size " + path);
            return false;
        }
        if (std::fseek(f, 0, SEEK_SET) != 0) {
            std::fclose(f);
            set_err(err, "source digest: cannot rewind " + path);
            return false;
        }
        mix_u64((uint64_t) sz); // file size domain-separated before content
        size_t remaining = (size_t) sz;
        while (remaining > 0) {
            const size_t want = remaining < chunk.size() ? remaining : chunk.size();
            const size_t got = std::fread(chunk.data(), 1, want, f);
            if (got == 0) {
                std::fclose(f);
                set_err(err, "source digest: short read on " + path);
                return false;
            }
            mix_bytes(chunk.data(), got);
            remaining -= got;
        }
        std::fclose(f);
    }
    xkv_sha256_final(ctx, out);
    return true;
}

uint64_t xkv_state_landmark_chunks_fingerprint(const std::vector<xkv_state_landmark_chunk> & chunks,
                                               uint64_t n_rows_total,
                                               const uint64_t * row_payload_ids,
                                               const uint64_t * row_generations,
                                               const int64_t * row_positions,
                                               uint64_t landmark_desc_fp,
                                               const xkv_snapshot_stamp & stamp, uint64_t rope_fp) {
    if (chunks.empty()) {
        return 0; // empty table has defined zero fingerprint (matches binder nonzero-guard)
    }
    uint64_t h = FNV_OFFSET;
    auto mix = [&h](uint64_t v) {
        for (int i = 0; i < 8; ++i) {
            h ^= (uint8_t) ((v >> (i * 8)) & 0xFF);
            h *= 0x100000001b3ULL;
        }
    };
    mix((uint64_t) chunks.size());
    mix(n_rows_total);
    // Offsets[0..n_chunks]: [0] == chunks[0].row_begin, [c+1] == row_begin + row_count.
    mix((uint64_t) chunks[0].row_begin);
    for (const auto & c : chunks) {
        mix((uint64_t) c.row_begin + c.row_count);
    }
    // Error bound bits and exact final-factor/phase provenance.
    for (const auto & c : chunks) {
        mix((uint64_t) c.error_bound_bits);
        mix(c.source_fingerprint);
    }
    // Row payload/generation/position arrays (u64 each, int64 bit-cast).
    if (row_payload_ids) {
        for (size_t r = 0; r < n_rows_total; ++r) {
            mix(row_payload_ids[r]);
            mix(row_generations ? row_generations[r] : 0ULL);
            uint64_t pu = 0;
            const int64_t p = row_positions ? row_positions[r] : 0LL;
            std::memcpy(&pu, &p, sizeof(pu));
            mix(pu);
        }
    }
    mix(stamp.live_epoch);
    mix(stamp.content_epoch);
    mix(stamp.codec_epoch);
    mix(stamp.binding_epoch);
    mix(stamp.view.topology_epoch);
    mix(stamp.view.publish_epoch);
    mix(stamp.view.layout_epoch);
    mix(rope_fp);
    if (landmark_desc_fp != 0) {
    mix(landmark_desc_fp);
    }
    return h;
}

xkv_state_limits xkv_state_limits::from_store_budgets(uint64_t factor_store_bytes_budget, uint64_t hot_bytes,
                                                      uint32_t max_segments, uint32_t max_groups_per_segment,
                                                      uint32_t max_payloads, uint32_t max_allocations,
                                                      uint32_t max_owning_layers,
                                                      uint32_t max_rows_per_segment) {
    xkv_state_limits l;
    // Framing slack: 1MiB base + per-record framing upper bounds.
    __uint128_t slack = (__uint128_t) 1048576 + (__uint128_t) max_payloads * 128 +
                        (__uint128_t) max_segments * 256 + (__uint128_t) max_allocations * 64;
    __uint128_t total = (__uint128_t) factor_store_bytes_budget + (__uint128_t) hot_bytes + slack;
    l.max_envelope_bytes = total > std::numeric_limits<uint64_t>::max()
        ? std::numeric_limits<uint64_t>::max()
        : (uint64_t) total;
    l.max_store_bytes        = factor_store_bytes_budget;
    l.max_stream_bytes       = l.max_envelope_bytes;
    l.max_segments           = max_segments;
    l.max_groups_per_segment = max_groups_per_segment;
    l.max_payloads           = max_payloads;
    l.max_allocations        = max_allocations;
    l.max_owning_layers      = max_owning_layers;
    l.max_rows_per_segment   = max_rows_per_segment;
    return l;
}

// ---------------------------------------------------------------------------
// Config validation
// ---------------------------------------------------------------------------

namespace {

bool validate_config(const xkv_state_config & c, std::string * err) {
    if (!valid_profile(c.profile)) {
        set_err(err, "validate: unknown requested storage profile");
        return false;
    }
    if (!valid_source(c.source)) {
        set_err(err, "validate: unknown source");
        return false;
    }
    if (c.group_size == 0 || c.rank_k == 0 || c.rank_v == 0 || c.segment_tokens == 0 ||
        c.chunk_tokens == 0 || c.store_mib == 0) {
        set_err(err, "validate: config group/rank/segment/chunk/store_mib contains zero");
        return false;
    }
    if (c.mode > 3) {
        set_err(err, "validate: unknown xkv mode");
        return false;
    }
    const uint32_t codecs[4] = {c.factor_a_k, c.factor_b_k, c.factor_a_v, c.factor_b_v};
    for (int i = 0; i < 4; ++i) {
        if (codecs[i] >= (uint32_t) GGML_TYPE_COUNT) {
            set_err(err, "validate: unknown factor codec type");
            return false;
        }
    }
    if (c.factor_balance != (uint32_t) LLAMA_XKV_FACTOR_BALANCE_UPSTREAM &&
        c.factor_balance != (uint32_t) LLAMA_XKV_FACTOR_BALANCE_SQRT &&
        c.factor_balance != (uint32_t) LLAMA_XKV_FACTOR_BALANCE_DIAGONAL) {
        set_err(err, "validate: unknown factor balance");
        return false;
    }
    if (c.landmark_refine != (uint32_t) LLAMA_XKV_LANDMARK_REFINE_NONE &&
        c.landmark_refine != (uint32_t) LLAMA_XKV_LANDMARK_REFINE_BOUNDARY) {
        set_err(err, "validate: unknown landmark refine mode");
        return false;
    }
    if (c.landmark_refine == (uint32_t) LLAMA_XKV_LANDMARK_REFINE_BOUNDARY &&
        c.landmark_refine_max_rows == 0) {
        set_err(err, "validate: boundary refine selected with zero max rows");
        return false;
    }
    if (c.landmark_type != (uint32_t) GGML_TYPE_COUNT &&
        c.landmark_type >= (uint32_t) GGML_TYPE_COUNT) {
        set_err(err, "validate: unknown landmark type");
        return false;
    }
    if (c.factorizer != (uint32_t) LLAMA_XKV_FACTORIZER_CPU_REFERENCE &&
        c.factorizer != (uint32_t) LLAMA_XKV_FACTORIZER_VULKAN &&
        c.factorizer != (uint32_t) LLAMA_XKV_FACTORIZER_VULKAN_HYBRID &&
        c.factorizer != (uint32_t) LLAMA_XKV_FACTORIZER_CUDA) {
        set_err(err, "validate: unknown factorizer");
        return false;
    }
    if (c.min_saving_ppm > PPM_DENOM || c.min_coverage_ppm > PPM_DENOM) {
        set_err(err, "validate: min saving/coverage fraction exceeds 1.0");
        return false;
    }
    if (c.next_segment_id == 0 || c.next_alloc_id == 0 || c.next_seal_tx_nonce == 0) {
        set_err(err, "validate: allocator high-water is zero");
        return false;
    }
    return true;
}

} // namespace

// ---------------------------------------------------------------------------
// validate_image
// ---------------------------------------------------------------------------

bool validate_image(
    const xkv_state_image & image,
    const xkv_state_fingerprints & expected,
    const xkv_state_provenance & expected_prov,
    const xkv_state_limits & limits,
    std::string * err) {

    if (image.fingerprints != expected) {
        set_err(err, "validate: fingerprint mismatch (source/model/RoPE/Tri/profile/factorizer/codec/backend)");
        return false;
    }
    if (!check_hash(image.provenance.model_sha256, expected_prov.model_sha256, err, "model")) {
        return false;
    }
    if (!check_hash(image.provenance.tri_calibration_sha256, expected_prov.tri_calibration_sha256, err,
                     "tri-calibration")) {
        return false;
    }
    if (!check_hash(image.provenance.source_sha256, expected_prov.source_sha256, err, "source")) {
        return false;
    }
    if (!validate_config(image.config, err)) {
        return false;
    }
    if (image.config.workspace_mib != image.workspace_mib ||
        image.config.decode_cache_mib != image.decode_cache_mib) {
        set_err(err, "validate: config budgets disagree with image budget fields");
        return false;
    }
    // Factored state requires non-skippable model provenance: all-zero model
    // digests (either side) refuse instead of silently skipping identity.
    if (!image.segments.empty()) {
        if (hash_is_zero(image.provenance.model_sha256) || hash_is_zero(expected_prov.model_sha256)) {
            set_err(err, "validate: factored restore requires model provenance digests (all-zero refused)");
            return false;
        }
    }

    if (image.allocations.size() > limits.max_allocations) {
        set_err(err, "validate: allocation count exceeds limit");
        return false;
    }
    if (image.segments.size() > limits.max_segments) {
        set_err(err, "validate: segment count exceeds limit");
        return false;
    }
    if (image.payloads.size() > limits.max_payloads) {
        set_err(err, "validate: payload count exceeds limit");
        return false;
    }

    // Allocations: unique ascending ids, exact descriptor bytes, checksums,
    // byte counts, per-stream budget.
    uint64_t stream_bytes_total = 0;
    uint64_t max_alloc_id       = 0;
    for (size_t i = 0; i < image.allocations.size(); ++i) {
        const auto & a = image.allocations[i];
        if (a.alloc_id == 0) {
            set_err(err, "validate: allocation id is zero");
            return false;
        }
        if (i > 0 && a.alloc_id <= image.allocations[i - 1].alloc_id) {
            set_err(err, "validate: allocation ids not strictly ascending (nondeterministic or duplicate)");
            return false;
        }
        max_alloc_id = a.alloc_id;
        if (a.bytes.size() > limits.max_stream_bytes) {
            set_err(err, "validate: stream bytes exceed per-stream limit");
            return false;
        }
        if (!add_ok(stream_bytes_total, a.bytes.size(), stream_bytes_total) ||
            stream_bytes_total > limits.max_envelope_bytes) {
            set_err(err, "validate: stream byte total exceeds envelope budget");
            return false;
        }
        codec_desc desc;
        std::string derr;
        if (a.desc_bytes.size() != 100) {
            set_err(err, "validate: descriptor bytes not exactly 100");
            return false;
        }
        if (!deserialize_desc(a.desc_bytes.data(), a.desc_bytes.size(), desc, &derr)) {
            set_err(err, "validate: bad descriptor: " + derr);
            return false;
        }
        if (!(desc == a.desc)) {
            set_err(err, "validate: descriptor bytes do not match decoded descriptor (non-canonical)");
            return false;
        }
        bool ok = false;
        const uint64_t want = stream_expected_bytes(desc, ok);
        if (!ok || want != a.bytes.size()) {
            set_err(err, "validate: stream byte size does not match descriptor layout");
            return false;
        }
        if (a.checksum != stream_checksum(a.desc_bytes, a.bytes)) {
            set_err(err, "validate: per-stream checksum mismatch");
            return false;
        }
    }

    // Segments: unique ascending (id, version), enums, live consistency, group closure.
    std::map<std::pair<uint64_t, uint64_t>, size_t> seg_index;
    uint64_t max_seg_id = 0;
    for (size_t si = 0; si < image.segments.size(); ++si) {
        const auto & s = image.segments[si];
        if (s.segment_id == 0 || s.segment_version == 0) {
            set_err(err, "validate: segment id/version is zero");
            return false;
        }
        const auto key = std::make_pair(s.segment_id, s.segment_version);
        if (si > 0) {
            const auto & p = image.segments[si - 1];
            if (key <= std::make_pair(p.segment_id, p.segment_version)) {
                set_err(err, "validate: segments not strictly ascending by (id, version)");
                return false;
            }
        }
        seg_index[key] = si;
        if (s.segment_id > max_seg_id) {
            max_seg_id = s.segment_id;
        }
        if (!valid_profile(s.profile)) {
            set_err(err, "validate: unknown segment storage profile");
            return false;
        }
        if (!valid_source(s.source)) {
            set_err(err, "validate: unknown segment source");
            return false;
        }
        if (s.n_rows == 0 || s.n_rows > limits.max_rows_per_segment) {
            set_err(err, "validate: segment row count zero or exceeds limit");
            return false;
        }
        if (s.groups.empty() || s.groups.size() > limits.max_groups_per_segment) {
            set_err(err, "validate: segment group count zero or exceeds limit");
            return false;
        }
        if (s.row_payload_ids.size() != s.n_rows || s.live_bits.size() != s.n_rows) {
            set_err(err, "validate: row payload/live arrays do not match n_rows");
            return false;
        }
        uint32_t live_count = 0;
        for (uint8_t b : s.live_bits) {
            if (b > 1) {
                set_err(err, "validate: live bit not 0/1");
                return false;
            }
            live_count += b;
        }
        if (live_count != s.n_live_rows) {
            set_err(err, "validate: n_live_rows does not match live-bit popcount");
            return false;
        }
        const bool need_landmarks =
            s.profile == (uint32_t) LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS_LANDMARKS;
        for (size_t gi = 0; gi < s.groups.size(); ++gi) {
            const auto & g = s.groups[gi];
            if (gi > 0 && g.group_index <= s.groups[gi - 1].group_index) {
                set_err(err, "validate: group indices not strictly ascending (duplicate or unordered)");
                return false;
            }
            if (g.owning_layers.empty() || g.owning_layers.size() > limits.max_owning_layers) {
                set_err(err, "validate: owning layers empty or exceeds limit");
                return false;
            }
            if (!strictly_ascending_u32(g.owning_layers)) {
                set_err(err, "validate: owning layers not strictly ascending (duplicate or unordered)");
                return false;
            }
            if (g.rank_k == 0 || g.rank_v == 0 || g.total_dim_k == 0 || g.total_dim_v == 0) {
                set_err(err, "validate: group rank/total dims contain zero");
                return false;
            }
            if (g.layer_feature_offsets_k.size() != g.owning_layers.size() ||
                g.layer_feature_dims_k.size() != g.owning_layers.size() ||
                g.layer_feature_offsets_v.size() != g.owning_layers.size() ||
                g.layer_feature_dims_v.size() != g.owning_layers.size()) {
                set_err(err, "validate: group offset/dim maps do not match owning layers");
                return false;
            }
            if (!check_slice_closure(g.layer_feature_offsets_k, g.layer_feature_dims_k,
                                     g.total_dim_k, err, "group K slices")) {
                return false;
            }
            if (!check_slice_closure(g.layer_feature_offsets_v, g.layer_feature_dims_v,
                                     g.total_dim_v, err, "group V slices")) {
                return false;
            }
            const uint32_t refs[4] = {g.stream_ak, g.stream_bk, g.stream_av, g.stream_bv};
            for (uint32_t r : refs) {
                if (r == XKV_STATE_NO_STREAM || r >= image.allocations.size()) {
                    set_err(err, "validate: group factor stream ref dangling or absent");
                    return false;
                }
            }
            if (g.has_landmark) {
                if (g.stream_landmark == XKV_STATE_NO_STREAM ||
                    g.stream_landmark >= image.allocations.size()) {
                    set_err(err, "validate: group landmark ref dangling");
                    return false;
                }
            } else if (g.stream_landmark != XKV_STATE_NO_STREAM) {
                set_err(err, "validate: landmark ref present without has_landmark");
                return false;
            }
            if (need_landmarks && !g.has_landmark) {
                set_err(err, "validate: landmarks profile requires a landmark stream per group");
                return false;
            }
            const codec_desc & dak = image.allocations[g.stream_ak].desc;
            const codec_desc & dbk = image.allocations[g.stream_bk].desc;
            const codec_desc & dav = image.allocations[g.stream_av].desc;
            const codec_desc & dbv = image.allocations[g.stream_bv].desc;
            if (dak.role != factor_role::a_k || dbk.role != factor_role::b_k ||
                dav.role != factor_role::a_v || dbv.role != factor_role::b_v) {
                set_err(err, "validate: stream role does not match group slot");
                return false;
            }
            if (dak.orient != orientation::token_major || dav.orient != orientation::token_major ||
                dbk.orient != orientation::feature_major_transposed ||
                dbv.orient != orientation::feature_major_transposed) {
                set_err(err, "validate: stream orientation does not match A/B layout contract");
                return false;
            }
            if (dak.logical_shape.rows != s.n_rows || dak.logical_shape.cols != g.rank_k ||
                dav.logical_shape.rows != s.n_rows || dav.logical_shape.cols != g.rank_v ||
                dbk.logical_shape.rows != g.total_dim_k || dbk.logical_shape.cols != g.rank_k ||
                dbv.logical_shape.rows != g.total_dim_v || dbv.logical_shape.cols != g.rank_v) {
                set_err(err, "validate: stream logical shape inconsistent with segment rows / group rank / dims");
                return false;
            }
            std::string perr;
            if (!validate_factor_pair_compatibility(dak, dbk, &perr)) {
                set_err(err, "validate: K factor pair incompatible: " + perr);
                return false;
            }
            if (!validate_factor_pair_compatibility(dav, dbv, &perr)) {
                set_err(err, "validate: V factor pair incompatible: " + perr);
                return false;
            }
            if (g.has_landmark) {
                const codec_desc & dlm = image.allocations[g.stream_landmark].desc;
                if (dlm.role != factor_role::landmark) {
                    set_err(err, "validate: landmark stream role mismatch");
                    return false;
                }
                if (dlm.logical_shape.rows == 0) {
                    set_err(err, "validate: landmark stream has zero rows");
                    return false;
                }
                if (image.config.landmark_type != (uint32_t) GGML_TYPE_COUNT &&
                    dlm.type != (ggml_type) image.config.landmark_type) {
                    set_err(err, "validate: landmark stream type disagrees with configured landmark type");
                    return false;
                }
            }
            // Landmark chunk table: closure over landmark rows, epsilon
            // finiteness/nonnegative, fingerprint binding to image context.
            // A table without a landmark stream is meaningless: rejected.
            if (!g.has_landmark && !g.landmark_chunks.empty()) {
                set_err(err, "validate: landmark chunks without a landmark stream");
                return false;
            }
            uint64_t lm_rows = 0;
            uint64_t lm_desc_fp = 0;
            if (g.has_landmark) {
                const codec_desc & dlm = image.allocations[g.stream_landmark].desc;
                lm_rows    = dlm.logical_shape.rows;
                lm_desc_fp = dlm.fingerprint();
                // Binder hardened gates: landmark stream rows == n_chunks.
                if (!g.landmark_chunks.empty()) {
                    if (dlm.logical_shape.rows != (uint64_t) g.landmark_chunks.size() ||
                        dlm.padded_shape.rows != (uint64_t) g.landmark_chunks.size()) {
                        set_err(err, "validate: landmark stream logical/padded rows must equal n_chunks");
                        return false;
                    }
                    if (g.landmark_chunks.size() > s.n_rows) {
                        set_err(err, "validate: more chunks than segment rows");
                        return false;
                    }
                }
                uint32_t cursor = 0;
                for (size_t ci = 0; ci < g.landmark_chunks.size(); ++ci) {
                    const auto & c = g.landmark_chunks[ci];
                    if (c.row_count == 0) {
                        set_err(err, "validate: landmark chunk with zero rows");
                        return false;
                    }
                    float eps = 0.0f;
                    std::memcpy(&eps, &c.error_bound_bits, sizeof(eps));
                    if (!std::isfinite(eps) || eps < 0.0f) {
                        set_err(err, "validate: landmark epsilon not finite/nonnegative");
                        return false;
                    }
                    if (c.source_fingerprint == 0) {
                        set_err(err, "validate: landmark source fingerprint missing");
                        return false;
                    }
                    if (c.row_begin != cursor) {
                        set_err(err, "validate: landmark chunks overlap or leave a gap");
                        return false;
                    }
                    if ((uint64_t) c.row_begin + c.row_count > s.n_rows) {
                        set_err(err, "validate: landmark chunk out of segment row range");
                        return false;
                    }
                    cursor += c.row_count;
                }
                if (!g.landmark_chunks.empty() && cursor != s.n_rows) {
                    set_err(err, "validate: landmark chunks final offset does not match segment row count");
                    return false;
                }
            }
            // Fingerprint check: non-empty chunks table requires nonzero and
            // exact match; empty chunks table requires zero fingerprint.
            std::vector<int64_t> flat_pos(s.n_rows, 0); // positions default 0 when not explicit
            const uint64_t chunks_fp = xkv_state_landmark_chunks_fingerprint(
                g.landmark_chunks, s.n_rows, s.row_payload_ids.data(),
                /*row_generations=*/nullptr, flat_pos.data(), lm_desc_fp, image.stamp,
                image.fingerprints.rope);
            if (!g.landmark_chunks.empty() && g.landmark_chunks_fingerprint == 0) {
                set_err(err, "validate: missing landmark chunks closure fingerprint");
                return false;
            }
            if (chunks_fp != g.landmark_chunks_fingerprint) {
                set_err(err, "validate: landmark chunks fingerprint mismatch");
                return false;
            }
        }
    }

    // Stream reference counts: A/landmark exactly once, B at least once,
    // no orphan allocations. Also flags per allocation for accounting.
    std::vector<uint64_t> refcount(image.allocations.size(), 0);
    std::vector<uint8_t>  ref_kind(image.allocations.size(), 0); // bit0=A/lm, bit1=B
    for (const auto & s : image.segments) {
        for (const auto & g : s.groups) {
            const uint32_t a_refs[2] = {g.stream_ak, g.stream_av};
            for (uint32_t r : a_refs) {
                refcount[r]++;
                ref_kind[r] |= 1;
            }
            const uint32_t b_refs[2] = {g.stream_bk, g.stream_bv};
            for (uint32_t r : b_refs) {
                refcount[r]++;
                ref_kind[r] |= 2;
            }
            if (g.has_landmark) {
                refcount[g.stream_landmark]++;
                ref_kind[g.stream_landmark] |= 1;
            }
        }
    }
    for (size_t i = 0; i < image.allocations.size(); ++i) {
        if (refcount[i] == 0) {
            set_err(err, "validate: orphan allocation (unreferenced stream bytes)");
            return false;
        }
        if ((ref_kind[i] & 1) && refcount[i] != 1 && (ref_kind[i] & 2) == 0) {
            // Pure A/landmark allocation referenced more than once: duplication.
            set_err(err, "validate: A/landmark stream shared by multiple groups (must be unique)");
            return false;
        }
        if ((ref_kind[i] & 1) && (ref_kind[i] & 2)) {
            set_err(err, "validate: allocation referenced as both unique (A/landmark) and shared (B)");
            return false;
        }
    }

    // Payloads: unique ascending ids, live-only, locator closure.
    std::map<uint64_t, size_t> payload_index;
    uint64_t max_nonce = 0;
    for (size_t i = 0; i < image.payloads.size(); ++i) {
        const auto & p = image.payloads[i];
        if (p.payload_id == 0) {
            set_err(err, "validate: payload id is zero");
            return false;
        }
        if (i > 0 && p.payload_id <= image.payloads[i - 1].payload_id) {
            set_err(err, "validate: payload ids not strictly ascending (duplicate or unordered)");
            return false;
        }
        if (p.live != 1) {
            set_err(err, "validate: payload table covers live rows only (tombstone records rejected)");
            return false;
        }
        if (p.generation == 0) {
            set_err(err, "validate: payload generation is zero");
            return false;
        }
        payload_index[p.payload_id] = i;
        if (p.locator.seal_tx_nonce > max_nonce) {
            max_nonce = p.locator.seal_tx_nonce;
        }
        if (p.locator.storage_generation != p.generation) {
            set_err(err, "validate: payload locator generation does not match payload generation");
            return false;
        }
        const auto kind = p.locator.kind;
        if (kind != xkv_location_kind::hot && kind != xkv_location_kind::flat_quantized &&
            kind != xkv_location_kind::factored) {
            set_err(err, "validate: payload locator has unknown storage kind");
            return false;
        }
        const auto st = p.locator.state;
        if (st != xkv_state::hot_writing && st != xkv_state::hot_committed &&
            st != xkv_state::seal_candidate && st != xkv_state::flat_tq && st != xkv_state::factored) {
            set_err(err, "validate: payload locator has unknown lifecycle state");
            return false;
        }
        if (kind == xkv_location_kind::hot) {
            if (p.locator.segment_id != 0 || p.locator.segment_version != 0) {
                set_err(err, "validate: hot locator must not reference a segment bundle");
                return false;
            }
            if (p.locator.row >= limits.max_rows_per_segment) {
                set_err(err, "validate: hot slot row exceeds limit");
                return false;
            }
        } else if (kind == xkv_location_kind::flat_quantized) {
            // Flat payloads live in dense per-layer storage, not factor bundles.
            if (p.locator.segment_id != 0 || p.locator.segment_version != 0) {
                set_err(err, "validate: flat locator must not reference a factor bundle");
                return false;
            }
            if (p.locator.row >= limits.max_rows_per_segment) {
                set_err(err, "validate: flat row exceeds limit");
                return false;
            }
        } else {
            const auto key = std::make_pair(p.locator.segment_id, p.locator.segment_version);
            const auto it  = seg_index.find(key);
            if (it == seg_index.end()) {
                set_err(err, "validate: payload locator references unknown segment bundle");
                return false;
            }
            const auto & s = image.segments[it->second];
            if (p.locator.row >= s.n_rows) {
                set_err(err, "validate: payload locator row out of bundle range");
                return false;
            }
            if (!s.live_bits[p.locator.row]) {
                set_err(err, "validate: live payload claims a dead bundle row");
                return false;
            }
            if (s.row_payload_ids[p.locator.row] != p.payload_id) {
                set_err(err, "validate: bundle row does not map back to payload id");
                return false;
            }
        }
    }

    // Every live bundle row is covered by exactly one live payload; dead-row
    // ids retain history but MUST NOT appear in the payload table.
    std::set<std::pair<std::pair<uint64_t, uint64_t>, uint32_t>> covered;
    for (const auto & p : image.payloads) {
        if (p.locator.kind == xkv_location_kind::hot) {
            continue;
        }
        if (p.locator.kind == xkv_location_kind::flat_quantized) {
            continue;
        }
        const auto rowkey =
            std::make_pair(std::make_pair(p.locator.segment_id, p.locator.segment_version), p.locator.row);
        if (!covered.insert(rowkey).second) {
            set_err(err, "validate: two payloads claim the same bundle row");
            return false;
        }
    }
    for (const auto & s : image.segments) {
        std::set<uint64_t> row_ids;
        for (uint32_t r = 0; r < s.n_rows; ++r) {
            if (!row_ids.insert(s.row_payload_ids[r]).second) {
                set_err(err, "validate: two bundle rows share one payload id");
                return false;
            }
            const auto it = payload_index.find(s.row_payload_ids[r]);
            if (s.live_bits[r]) {
                if (it == payload_index.end()) {
                    set_err(err, "validate: live bundle row maps to unknown payload id");
                    return false;
                }
                const auto rowkey = std::make_pair(std::make_pair(s.segment_id, s.segment_version), r);
                if (covered.find(rowkey) == covered.end()) {
                    set_err(err, "validate: live bundle row has no covering live payload");
                    return false;
                }
            } else if (it != payload_index.end()) {
                set_err(err, "validate: dead bundle row id appears in the live payload table");
                return false;
            }
        }
    }

    // Allocator high-water closure.
    if (!image.segments.empty() && image.config.next_segment_id <= max_seg_id) {
        set_err(err, "validate: segment id reaches the allocator high-water");
        return false;
    }
    if (!image.allocations.empty() && image.config.next_alloc_id <= max_alloc_id) {
        set_err(err, "validate: allocation id reaches the allocator high-water");
        return false;
    }
    if (!image.payloads.empty() && image.config.next_seal_tx_nonce <= max_nonce) {
        set_err(err, "validate: seal nonce reaches the allocator high-water");
        return false;
    }

    // Accounting: recompute exactly from the image; serialized counters must
    // agree (budgets are enforced from recomputed values + caller limits only).
    uint64_t sum_all = 0, sum_b = 0, sum_live = 0, n_b = 0;
    std::vector<uint8_t> alloc_live(image.allocations.size(), 0);
    for (const auto & s : image.segments) {
        if (s.n_live_rows == 0) {
            continue;
        }
        for (const auto & g : s.groups) {
            alloc_live[g.stream_ak] = 1;
            alloc_live[g.stream_bk] = 1;
            alloc_live[g.stream_av] = 1;
            alloc_live[g.stream_bv] = 1;
            if (g.has_landmark) {
                alloc_live[g.stream_landmark] = 1;
            }
        }
    }
    for (size_t i = 0; i < image.allocations.size(); ++i) {
        if (!add_ok(sum_all, image.allocations[i].bytes.size(), sum_all)) {
            set_err(err, "validate: accounting recompute overflow");
            return false;
        }
        if (ref_kind[i] & 2) {
            ++n_b;
            if (!add_ok(sum_b, image.allocations[i].bytes.size(), sum_b)) {
                set_err(err, "validate: accounting recompute overflow");
                return false;
            }
        }
        if (alloc_live[i]) {
            if (!add_ok(sum_live, image.allocations[i].bytes.size(), sum_live)) {
                set_err(err, "validate: accounting recompute overflow");
                return false;
            }
        }
    }
    const auto & c = image.counters;
    // Persistent store budget first, from recomputed bytes only: actual
    // factored stream bytes must fit the config store_mib budget AND the
    // caller-supplied limits cap. Serialized counters play no role here.
    uint64_t store_bytes_budget = 0;
    if (!mul_ok((uint64_t) image.config.store_mib, 1048576ULL, store_bytes_budget)) {
        set_err(err, "validate: store_mib budget overflow");
        return false;
    }
    if (sum_all > store_bytes_budget) {
        set_err(err, "validate: actual factored bytes exceed config store_mib budget");
        return false;
    }
    if (sum_all > limits.max_store_bytes) {
        set_err(err, "validate: actual factored bytes exceed limits.max_store_bytes");
        return false;
    }
    if (c.factored_bytes != sum_all) {
        set_err(err, "validate: factored_bytes disagrees with recomputed stream bytes");
        return false;
    }
    if (c.unique_b_matrices != n_b || c.shared_b_bytes != sum_b) {
        set_err(err, "validate: shared-B counters disagree with recomputed shared allocations");
        return false;
    }
    if (c.live_payload_bytes != sum_live) {
        set_err(err, "validate: live_payload_bytes disagrees with recomputed live streams");
        return false;
    }
    if (c.active_segments != image.segments.size()) {
        set_err(err, "validate: active_segments disagrees with segment count");
        return false;
    }
    if (c.total_payloads != image.payloads.size()) {
        set_err(err, "validate: total_payloads disagrees with payload count");
        return false;
    }
    uint64_t mib_bytes = 0;
    if (!mul_ok((uint64_t) image.workspace_mib, 1048576ULL, mib_bytes) ||
        c.workspace_budget_bytes != mib_bytes) {
        set_err(err, "validate: workspace budget disagrees with workspace_mib");
        return false;
    }
    uint64_t floor_alloc = 0;
    if (!add_ok(c.factored_bytes, c.hot_bytes, floor_alloc) || c.allocated_bytes < floor_alloc) {
        set_err(err, "validate: allocated_bytes below recomputed factored + hot floor");
        return false;
    }
    if (c.reserved_bytes < c.allocated_bytes) {
        set_err(err, "validate: reserved_bytes below allocated_bytes");
        return false;
    }

    return true;
}

// ---------------------------------------------------------------------------
// encoded_size + encode_image
// ---------------------------------------------------------------------------

namespace {

void size_config(sizer & z) {
    // 22 u32 fields: profile, source, mode, group_size, rank_k, rank_v,
    // segment_tokens, chunk_tokens, sr_budget, factor_a_k/b_k/a_v/b_v,
    // factor_balance, landmark_type, landmark_refine, landmark_refine_max_rows,
    // workspace_mib, decode_cache_mib, store_mib, factorizer, min_saving_bytes.
    for (int i = 0; i < 22; ++i) {
        z.put_u32();
    }
    z.put_u64(); // seed
    z.put_u64(); // min_saving_ppm
    z.put_u64(); // min_coverage_ppm
    z.put_u64(); // next_segment_id
    z.put_u64(); // next_alloc_id
    z.put_u64(); // next_seal_tx_nonce
}

void write_config(writer & w, const xkv_state_config & c) {
    w.put_u32(c.profile);
    w.put_u32(c.source);
    w.put_u32(c.mode);
    w.put_u32(c.group_size);
    w.put_u32(c.rank_k);
    w.put_u32(c.rank_v);
    w.put_u32(c.segment_tokens);
    w.put_u32(c.chunk_tokens);
    w.put_u32(c.sr_budget);
    w.put_u32(c.factor_a_k);
    w.put_u32(c.factor_b_k);
    w.put_u32(c.factor_a_v);
    w.put_u32(c.factor_b_v);
    w.put_u32(c.factor_balance);
    w.put_u32(c.landmark_type);
    w.put_u32(c.landmark_refine);
    w.put_u32(c.landmark_refine_max_rows);
    w.put_u32(c.workspace_mib);
    w.put_u32(c.decode_cache_mib);
    w.put_u32(c.store_mib);
    w.put_u64(c.seed);
    w.put_u64(c.min_saving_ppm);
    w.put_u64(c.min_coverage_ppm);
    w.put_u32(c.factorizer);
    w.put_u32(c.min_saving_bytes);
    w.put_u64(c.next_segment_id);
    w.put_u64(c.next_alloc_id);
    w.put_u64(c.next_seal_tx_nonce);
}

bool read_config(reader & r, xkv_state_config & c, std::string * err) {
    return r.get_u32(c.profile, err, "config") && r.get_u32(c.source, err, "config") &&
           r.get_u32(c.mode, err, "config") &&
           r.get_u32(c.group_size, err, "config") && r.get_u32(c.rank_k, err, "config") &&
           r.get_u32(c.rank_v, err, "config") && r.get_u32(c.segment_tokens, err, "config") &&
           r.get_u32(c.chunk_tokens, err, "config") && r.get_u32(c.sr_budget, err, "config") &&
           r.get_u32(c.factor_a_k, err, "config") && r.get_u32(c.factor_b_k, err, "config") &&
           r.get_u32(c.factor_a_v, err, "config") && r.get_u32(c.factor_b_v, err, "config") &&
           r.get_u32(c.factor_balance, err, "config") && r.get_u32(c.landmark_type, err, "config") &&
           r.get_u32(c.landmark_refine, err, "config") &&
           r.get_u32(c.landmark_refine_max_rows, err, "config") &&
           r.get_u32(c.workspace_mib, err, "config") && r.get_u32(c.decode_cache_mib, err, "config") &&
           r.get_u32(c.store_mib, err, "config") && r.get_u64(c.seed, err, "config") &&
           r.get_u64(c.min_saving_ppm, err, "config") && r.get_u64(c.min_coverage_ppm, err, "config") &&
           r.get_u32(c.factorizer, err, "config") && r.get_u32(c.min_saving_bytes, err, "config") &&
           r.get_u64(c.next_segment_id, err, "config") && r.get_u64(c.next_alloc_id, err, "config") &&
           r.get_u64(c.next_seal_tx_nonce, err, "config");
}

} // namespace

bool encoded_size(const xkv_state_image & image, uint64_t & size_out, std::string * err) {
    sizer z;
    z.put_u32(); // magic
    z.put_u32(); // version
    z.add_n(8, 8); // fingerprints
    z.add(96); // provenance
    size_config(z);
    z.add_n(7, 8); // stamp
    z.put_u32(); // workspace_mib
    z.put_u32(); // decode_cache_mib
    z.put_u32(); // reserved framing pad
    z.add_n(10, 8); // counters
    z.add(8); // allocation count
    for (const auto & a : image.allocations) {
        z.put_u32();
        z.add(8);
        z.put_raw(a.desc_bytes.size());
        z.add(8);
        z.put_raw(a.bytes.size());
        z.put_u64();
        if (!z.ok) {
            break;
        }
    }
    if (!z.ok) {
        set_err(err, "encode: size precompute overflow (allocations)");
        return false;
    }
    z.add(8); // segment count
    for (const auto & s : image.segments) {
        z.put_u64();
        z.put_u64();
        z.put_u32();
        z.put_u32();
        z.add_n(4, 8); // fingerprints
        z.put_u32();
        z.put_u32();
        z.add(8);
        z.add_n((uint64_t) s.row_payload_ids.size(), 8);
        z.add(8);
        z.put_raw(s.live_bits.size());
        z.add(8); // group count
        for (const auto & g : s.groups) {
            z.put_u32();
            z.put_u32_vec(g.owning_layers);
            z.put_u32();
            z.put_u32();
            z.put_u32_vec(g.layer_feature_offsets_k);
            z.put_u32_vec(g.layer_feature_dims_k);
            z.put_u32_vec(g.layer_feature_offsets_v);
            z.put_u32_vec(g.layer_feature_dims_v);
            z.put_u32();
            z.put_u32();
            z.add_n(4, 4); // stream refs
            z.put_u8();
            z.put_u32();
            z.add(8); // landmark chunk count
            z.add_n((uint64_t) g.landmark_chunks.size(), 20);
            z.put_u64(); // landmark chunks fingerprint
            z.put_u64(); // descriptor fingerprint
            z.put_u64(); // config fingerprint
            if (!z.ok) {
                break;
            }
        }
        if (!z.ok) {
            break;
        }
    }
    if (!z.ok) {
        set_err(err, "encode: size precompute overflow (segments)");
        return false;
    }
    z.add(8); // payload count
    z.add_n((uint64_t) image.payloads.size(), 55);
    z.put_u64(); // envelope checksum
    if (!z.ok) {
        set_err(err, "encode: size precompute overflow (payloads)");
        return false;
    }
    size_out = z.size;
    return true;
}

bool encode_image(const xkv_state_image & image, std::vector<uint8_t> & out,
                  const xkv_state_limits & limits, std::string * err) {
    try {
        if (!validate_image(image, image.fingerprints, image.provenance, limits, err)) {
            return false;
        }
        uint64_t pre = 0;
        if (!encoded_size(image, pre, err)) {
            return false;
        }
        if (pre > limits.max_envelope_bytes) {
            set_err(err, "encode: precomputed size exceeds envelope budget");
            return false;
        }
        writer w;
        w.buf.reserve((size_t) pre);

        w.put_u32(XKV_STATE_MAGIC);
        w.put_u32(XKV_STATE_VERSION);

        w.put_u64(image.fingerprints.source);
        w.put_u64(image.fingerprints.model);
        w.put_u64(image.fingerprints.rope);
        w.put_u64(image.fingerprints.tri_calibration);
        w.put_u64(image.fingerprints.profile);
        w.put_u64(image.fingerprints.factorizer);
        w.put_u64(image.fingerprints.codec);
        w.put_u64(image.fingerprints.backend);

        w.put_hash(image.provenance.model_sha256);
        w.put_hash(image.provenance.tri_calibration_sha256);
        w.put_hash(image.provenance.source_sha256);

        write_config(w, image.config);

        w.put_u64(image.stamp.view.topology_epoch);
        w.put_u64(image.stamp.view.publish_epoch);
        w.put_u64(image.stamp.view.layout_epoch);
        w.put_u64(image.stamp.live_epoch);
        w.put_u64(image.stamp.content_epoch);
        w.put_u64(image.stamp.codec_epoch);
        w.put_u64(image.stamp.binding_epoch);

        w.put_u32(image.workspace_mib);
        w.put_u32(image.decode_cache_mib);
        w.put_u32(0); // reserved framing pad for 12-byte alignment of the walk

        w.put_u64(image.counters.live_payload_bytes);
        w.put_u64(image.counters.allocated_bytes);
        w.put_u64(image.counters.reserved_bytes);
        w.put_u64(image.counters.workspace_budget_bytes);
        w.put_u64(image.counters.hot_bytes);
        w.put_u64(image.counters.factored_bytes);
        w.put_u64(image.counters.active_segments);
        w.put_u64(image.counters.total_payloads);
        w.put_u64(image.counters.unique_b_matrices);
        w.put_u64(image.counters.shared_b_bytes);

        w.put_u64((uint64_t) image.allocations.size());
        for (const auto & a : image.allocations) {
            w.put_u32(a.alloc_id);
            w.put_bytes(a.desc_bytes);
            w.put_bytes(a.bytes);
            w.put_u64(a.checksum);
        }

        w.put_u64((uint64_t) image.segments.size());
        for (const auto & s : image.segments) {
            w.put_u64(s.segment_id);
            w.put_u64(s.segment_version);
            w.put_u32(s.profile);
            w.put_u32(s.source);
            w.put_u64(s.profile_fingerprint);
            w.put_u64(s.source_fingerprint);
            w.put_u64(s.layer_group_map_fingerprint);
            w.put_u64(s.descriptor_fingerprint);
            w.put_u32(s.n_rows);
            w.put_u32(s.n_live_rows);
            w.put_u64((uint64_t) s.row_payload_ids.size());
            for (uint64_t id : s.row_payload_ids) {
                w.put_u64(id);
            }
            w.put_bytes(s.live_bits);
            w.put_u64((uint64_t) s.groups.size());
            for (const auto & g : s.groups) {
                w.put_u32(g.group_index);
                w.put_u32_vec(g.owning_layers);
                w.put_u32(g.rank_k);
                w.put_u32(g.rank_v);
                w.put_u32_vec(g.layer_feature_offsets_k);
                w.put_u32_vec(g.layer_feature_dims_k);
                w.put_u32_vec(g.layer_feature_offsets_v);
                w.put_u32_vec(g.layer_feature_dims_v);
                w.put_u32(g.total_dim_k);
                w.put_u32(g.total_dim_v);
                w.put_u32(g.stream_ak);
                w.put_u32(g.stream_bk);
                w.put_u32(g.stream_av);
                w.put_u32(g.stream_bv);
                w.put_u8(g.has_landmark ? 1 : 0);
                w.put_u32(g.stream_landmark);
                w.put_u64((uint64_t) g.landmark_chunks.size());
                for (const auto & c : g.landmark_chunks) {
                w.put_u32(c.row_begin);
                w.put_u32(c.row_count);
                w.put_u32(c.error_bound_bits);
                w.put_u64(c.source_fingerprint);
                }
                w.put_u64(g.landmark_chunks_fingerprint);
                w.put_u64(g.descriptor_fingerprint);
                w.put_u64(g.config_fingerprint);
            }
        }

        w.put_u64((uint64_t) image.payloads.size());
        for (const auto & p : image.payloads) {
            w.put_u64(p.payload_id);
            w.put_u64(p.generation);
            w.put_u8(p.live);
            w.put_u8((uint8_t) p.locator.kind);
            w.put_u64(p.locator.segment_id);
            w.put_u64(p.locator.segment_version);
            w.put_u32(p.locator.row);
            w.put_u64(p.locator.storage_generation);
            w.put_u8((uint8_t) p.locator.state);
            w.put_u64(p.locator.seal_tx_nonce);
        }

        // Whole-envelope checksum over all preceding bytes.
        w.put_u64(xkv_state_checksum(w.buf.data(), w.buf.size()));

        if (w.buf.size() != (size_t) pre) {
            set_err(err, "encode: size drift between precompute and serialization");
            return false;
        }
        out.assign(w.buf.begin(), w.buf.end());
        return true;
    } catch (const std::bad_alloc &) {
        set_err(err, "encode: allocation failure (output unchanged)");
        return false;
    }
}

// ---------------------------------------------------------------------------
// decode_image
// ---------------------------------------------------------------------------

bool decode_image(
    const uint8_t * data,
    size_t size,
    xkv_state_image & out,
    const xkv_state_fingerprints & expected,
    const xkv_state_provenance & expected_prov,
    const xkv_state_limits & limits,
    std::string * err) {

    xkv_state_image tmp; // off-side: `out` untouched until full success

    if (data == nullptr || size == 0) {
        set_err(err, "decode: empty input");
        return false;
    }

    try {
        reader r{data, size, 0};

        uint32_t magic = 0;
        uint32_t version = 0;
        if (!r.get_u32(magic, err, "magic") || !r.get_u32(version, err, "version")) {
            return false;
        }
        if (magic != XKV_STATE_MAGIC) {
            set_err(err, "decode: bad magic (not an XKV state envelope)");
            return false;
        }
        if (version == 0) {
            set_err(err, "decode: unsupported legacy XKV state version 0 (refusing legacy bytes)");
            return false;
        }
        if (version == 1) {
            set_err(err, "decode: unsupported old XKV state version 1 (layout retired; refusing)");
            return false;
        }
        if (version == 2) {
            set_err(err, "decode: unsupported old XKV state version 2 (layout retired; refusing)");
            return false;
        }
        if (version == 3) {
            set_err(err, "decode: unsupported old XKV state version 3 (layout retired: config lacks exact codecs/budgets; refusing)");
            return false;
        }
        if (version == 4) {
            set_err(err, "decode: unsupported old XKV state version 4 (layout retired: v4 landmark chunks without source fingerprint; refusing)");
            return false;
        }
        if (version != XKV_STATE_VERSION) {
            set_err(err, "decode: unknown XKV state version (refusing forward-incompatible bytes)");
            return false;
        }

        uint64_t f[8];
        for (int i = 0; i < 8; ++i) {
            if (!r.get_u64(f[i], err, "fingerprints")) {
                return false;
            }
        }
        tmp.fingerprints = {f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]};

        if (!r.get_hash(tmp.provenance.model_sha256, err, "provenance") ||
            !r.get_hash(tmp.provenance.tri_calibration_sha256, err, "provenance") ||
            !r.get_hash(tmp.provenance.source_sha256, err, "provenance")) {
            return false;
        }
        if (!read_config(r, tmp.config, err)) {
            return false;
        }

        uint64_t topo = 0, pub = 0, lay = 0, live = 0, content = 0, codec = 0, bind = 0;
        if (!r.get_u64(topo, err, "stamp") || !r.get_u64(pub, err, "stamp") ||
            !r.get_u64(lay, err, "stamp") || !r.get_u64(live, err, "stamp") ||
            !r.get_u64(content, err, "stamp") || !r.get_u64(codec, err, "stamp") ||
            !r.get_u64(bind, err, "stamp")) {
            return false;
        }
        tmp.stamp.view.topology_epoch = topo;
        tmp.stamp.view.publish_epoch   = pub;
        tmp.stamp.view.layout_epoch    = lay;
        tmp.stamp.live_epoch           = live;
        tmp.stamp.content_epoch        = content;
        tmp.stamp.codec_epoch          = codec;
        tmp.stamp.binding_epoch        = bind;

        uint32_t pad = 0;
        if (!r.get_u32(tmp.workspace_mib, err, "budgets") ||
            !r.get_u32(tmp.decode_cache_mib, err, "budgets") || !r.get_u32(pad, err, "framing pad")) {
            return false;
        }
        if (pad != 0) {
            set_err(err, "decode: nonzero framing pad");
            return false;
        }

        uint64_t c[10];
        for (int i = 0; i < 10; ++i) {
            if (!r.get_u64(c[i], err, "counters")) {
                return false;
            }
        }
        tmp.counters = {c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9]};

        uint64_t n_alloc = 0;
        if (!r.get_u64(n_alloc, err, "allocation count")) {
            return false;
        }
        if (n_alloc > limits.max_allocations) {
            set_err(err, "decode: allocation count exceeds limit");
            return false;
        }
        tmp.allocations.resize((size_t) n_alloc);
        for (uint64_t i = 0; i < n_alloc; ++i) {
            auto & a = tmp.allocations[(size_t) i];
            if (!r.get_u32(a.alloc_id, err, "allocation id")) {
                return false;
            }
            uint64_t dl = 0;
            if (!r.get_u64(dl, err, "descriptor length")) {
                return false;
            }
            if (dl != 100) {
                set_err(err, "decode: descriptor length is not exactly 100");
                return false;
            }
            if (!get_sized_bytes(r, dl, 100, a.desc_bytes, err, "descriptor bytes")) {
                return false;
            }
            uint64_t bl = 0;
            if (!r.get_u64(bl, err, "stream length")) {
                return false;
            }
            if (!get_sized_bytes(r, bl, limits.max_stream_bytes, a.bytes, err, "stream bytes")) {
                return false;
            }
            if (!r.get_u64(a.checksum, err, "stream checksum")) {
                return false;
            }
            // Hydrate the descriptor now so validate_image sees consistent structs.
            std::string derr;
            if (!deserialize_desc(a.desc_bytes.data(), a.desc_bytes.size(), a.desc, &derr)) {
                set_err(err, "decode: bad descriptor: " + derr);
                return false;
            }
        }

        uint64_t n_seg = 0;
        if (!r.get_u64(n_seg, err, "segment count")) {
            return false;
        }
        if (n_seg > limits.max_segments) {
            set_err(err, "decode: segment count exceeds limit");
            return false;
        }
        tmp.segments.resize((size_t) n_seg);
        for (uint64_t i = 0; i < n_seg; ++i) {
            auto & s = tmp.segments[(size_t) i];
            if (!r.get_u64(s.segment_id, err, "segment id") ||
                !r.get_u64(s.segment_version, err, "segment version") ||
                !r.get_u32(s.profile, err, "segment profile") || !r.get_u32(s.source, err, "segment source") ||
                !r.get_u64(s.profile_fingerprint, err, "segment fingerprints") ||
                !r.get_u64(s.source_fingerprint, err, "segment fingerprints") ||
                !r.get_u64(s.layer_group_map_fingerprint, err, "segment fingerprints") ||
                !r.get_u64(s.descriptor_fingerprint, err, "segment fingerprints") ||
                !r.get_u32(s.n_rows, err, "segment rows") || !r.get_u32(s.n_live_rows, err, "segment rows")) {
                return false;
            }
            uint64_t n_rows_decl = 0;
            if (!r.get_u64(n_rows_decl, err, "row payload count")) {
                return false;
            }
            if (n_rows_decl > limits.max_rows_per_segment || n_rows_decl > r.remaining() / 8) {
                set_err(err, "decode: row payload count exceeds limit (or truncated)");
                return false;
            }
            s.row_payload_ids.resize((size_t) n_rows_decl);
            for (uint64_t k = 0; k < n_rows_decl; ++k) {
                if (!r.get_u64(s.row_payload_ids[(size_t) k], err, "row payload ids")) {
                    return false;
                }
            }
            uint64_t lb_len = 0;
            if (!r.get_u64(lb_len, err, "live bits length")) {
                return false;
            }
            if (!get_sized_bytes(r, lb_len, (uint64_t) limits.max_rows_per_segment + 1, s.live_bits, err,
                                 "live bits")) {
                return false;
            }
            uint64_t n_groups = 0;
            if (!r.get_u64(n_groups, err, "group count")) {
                return false;
            }
            if (n_groups > limits.max_groups_per_segment) {
                set_err(err, "decode: group count exceeds limit");
                return false;
            }
            s.groups.resize((size_t) n_groups);
            for (uint64_t gi = 0; gi < n_groups; ++gi) {
                auto & g = s.groups[(size_t) gi];
                uint8_t has_lm = 0;
                uint64_t n_chunks = 0;
                if (!r.get_u32(g.group_index, err, "group index") ||
                    !get_u32_vec(r, limits.max_owning_layers, g.owning_layers, err, "owning layers") ||
                    !r.get_u32(g.rank_k, err, "group ranks") || !r.get_u32(g.rank_v, err, "group ranks") ||
                    !get_u32_vec(r, limits.max_owning_layers, g.layer_feature_offsets_k, err, "K offsets") ||
                    !get_u32_vec(r, limits.max_owning_layers, g.layer_feature_dims_k, err, "K dims") ||
                    !get_u32_vec(r, limits.max_owning_layers, g.layer_feature_offsets_v, err, "V offsets") ||
                    !get_u32_vec(r, limits.max_owning_layers, g.layer_feature_dims_v, err, "V dims") ||
                    !r.get_u32(g.total_dim_k, err, "group totals") ||
                    !r.get_u32(g.total_dim_v, err, "group totals") ||
                    !r.get_u32(g.stream_ak, err, "group streams") ||
                    !r.get_u32(g.stream_bk, err, "group streams") ||
                    !r.get_u32(g.stream_av, err, "group streams") ||
                    !r.get_u32(g.stream_bv, err, "group streams") || !r.get_u8(has_lm, err, "landmark flag") ||
                    !r.get_u32(g.stream_landmark, err, "landmark stream") ||
                    !r.get_u64(n_chunks, err, "landmark chunk count")) {
                return false;
            }
            if (n_chunks > limits.max_rows_per_segment + 1) {
                set_err(err, "decode: landmark chunk count exceeds limit");
                return false;
            }
            if (n_chunks > r.remaining() / 20) {
                set_err(err, "decode: truncated landmark chunks");
                return false;
            }
            g.landmark_chunks.resize((size_t) n_chunks);
            for (uint64_t ci = 0; ci < n_chunks; ++ci) {
                auto & c = g.landmark_chunks[(size_t) ci];
                if (!r.get_u32(c.row_begin, err, "landmark chunks") ||
                    !r.get_u32(c.row_count, err, "landmark chunks") ||
                    !r.get_u32(c.error_bound_bits, err, "landmark chunks") ||
                    !r.get_u64(c.source_fingerprint, err, "landmark chunks")) {
                    return false;
                }
            }
            if (!r.get_u64(g.landmark_chunks_fingerprint, err, "group fingerprints") ||
                !r.get_u64(g.descriptor_fingerprint, err, "group fingerprints") ||
                !r.get_u64(g.config_fingerprint, err, "group fingerprints")) {
                return false;
            }
                if (has_lm > 1) {
                    set_err(err, "decode: landmark flag not 0/1");
                    return false;
                }
                g.has_landmark = has_lm == 1;
            }
        }

        uint64_t n_pay = 0;
        if (!r.get_u64(n_pay, err, "payload count")) {
            return false;
        }
        if (n_pay > limits.max_payloads) {
            set_err(err, "decode: payload count exceeds limit");
            return false;
        }
        tmp.payloads.resize((size_t) n_pay);
        for (uint64_t i = 0; i < n_pay; ++i) {
            auto & p = tmp.payloads[(size_t) i];
            uint8_t live = 0, kind = 0, state = 0;
            if (!r.get_u64(p.payload_id, err, "payload id") ||
                !r.get_u64(p.generation, err, "payload generation") || !r.get_u8(live, err, "payload live") ||
                !r.get_u8(kind, err, "locator kind") || !r.get_u64(p.locator.segment_id, err, "locator segment") ||
                !r.get_u64(p.locator.segment_version, err, "locator segment") ||
                !r.get_u32(p.locator.row, err, "locator row") ||
                !r.get_u64(p.locator.storage_generation, err, "locator generation") ||
                !r.get_u8(state, err, "locator state") ||
                !r.get_u64(p.locator.seal_tx_nonce, err, "locator nonce")) {
                return false;
            }
            p.live          = live;
            p.locator.kind  = (xkv_location_kind) kind;
            p.locator.state = (xkv_state) state;
        }

        // Whole-envelope checksum must be the last 8 bytes; anything else trailing
        // (or a missing checksum) is rejected.
        if (r.remaining() != 8) {
            set_err(err, r.remaining() < 8 ? "decode: truncated (missing envelope checksum)"
                                           : "decode: trailing bytes after payload section");
            return false;
        }
        uint64_t stored_sum = 0;
        if (!r.get_u64(stored_sum, err, "envelope checksum")) {
            return false;
        }
        const uint64_t computed_sum = xkv_state_checksum(data, size - 8);
        if (stored_sum != computed_sum) {
            set_err(err, "decode: whole-envelope checksum mismatch");
            return false;
        }

        if (!validate_image(tmp, expected, expected_prov, limits, err)) {
            return false;
        }

        out = std::move(tmp);
        return true;
    } catch (const std::bad_alloc &) {
        set_err(err, "decode: allocation failure (output unchanged)");
        return false;
    }
}

// ---------------------------------------------------------------------------
// capture_image (public snapshots only)
// ---------------------------------------------------------------------------

namespace {

struct readback_key {
    uint64_t seg;
    uint64_t ver;
    uint32_t group;
    uint32_t role;

    bool operator<(const readback_key & o) const {
        if (seg != o.seg) {
            return seg < o.seg;
        }
        if (ver != o.ver) {
            return ver < o.ver;
        }
        if (group != o.group) {
            return group < o.group;
        }
        return role < o.role;
    }
};

} // namespace

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
    const xkv_state_readback * readback,
    std::string * err) {

    xkv_state_image tmp; // off-side: `out` untouched until full success

    try {
        tmp.fingerprints     = fingerprints;
        tmp.provenance       = provenance;
        tmp.config           = config;
        tmp.stamp            = stamp;
        tmp.workspace_mib    = workspace_mib;
        tmp.decode_cache_mib = decode_cache_mib;

        // Counters: exactly-recomputable fields are derived below from the
        // captured streams; only attested floors come from the caller. The
        // final validate_image recomputes and cross-checks every field.
        uint64_t live_sum = 0, sum_all = 0, sum_b = 0, n_b = 0;

        std::map<readback_key, const std::vector<uint8_t> *> rb_map;
        if (readback != nullptr) {
            for (const auto & e : readback->entries) {
                rb_map[{e.segment_id, e.segment_version, e.group_index, e.role}] = &e.bytes;
            }
        }

        // Deterministic segment order.
        std::vector<std::shared_ptr<const xkv_segment>> segs = segments;
        std::sort(segs.begin(), segs.end(), [](const auto & a, const auto & b) {
            if (a == nullptr || b == nullptr) {
                return a != nullptr;
            }
            if (a->segment_id != b->segment_id) {
                return a->segment_id < b->segment_id;
            }
            return a->segment_version < b->segment_version;
        });

        // Shared-B dedup by live pointer identity. A/landmark streams are unique.
        std::map<const encoded_matrix *, uint32_t> b_alloc;
        // Per-allocation live-segment reference for live-byte accounting.
        std::vector<uint8_t> alloc_live;
        uint32_t next_alloc = 1;

        auto note_live = [&](uint32_t ref, bool seg_live) {
            if (seg_live && ref < alloc_live.size()) {
                alloc_live[ref] = 1;
            }
        };

        // Emits one stream; returns false on id exhaustion, size mismatch, or
        // device-resident bytes without a covering readback entry.
        auto emit_stream = [&](uint64_t seg_id, uint64_t seg_ver, uint32_t group_index, factor_role role,
                               const encoded_matrix & em, bool shared, bool seg_live, std::string * e,
                               uint32_t & ref) -> bool {
            if (shared) {
                const auto it = b_alloc.find(&em);
                if (it != b_alloc.end()) {
                    ref = it->second;
                    note_live(ref, seg_live);
                    return true;
                }
            }
            if (next_alloc == std::numeric_limits<uint32_t>::max()) {
                set_err(e, "capture: allocation id space exhausted");
                return false;
            }
            bool size_ok = false;
            const uint64_t want = stream_expected_bytes(em.desc, size_ok);
            if (!size_ok) {
                set_err(e, "capture: stream descriptor invalid (cannot size bytes)");
                return false;
            }
            const std::vector<uint8_t> * src = &em.bytes;
            const auto rb_it =
                rb_map.find({seg_id, seg_ver, group_index, (uint32_t) role});
            if (rb_it != rb_map.end()) {
                // Explicit synchronized readback wins over host bytes.
                src = rb_it->second;
            } else if (em.bytes.empty()) {
                set_err(e, "capture: device-resident stream has no host bytes and no readback entry "
                           "(explicit synchronized readback required)");
                return false;
            }
            if (src->size() != (size_t) want) {
                set_err(e, "capture: stream bytes do not match descriptor layout");
                return false;
            }
            xkv_state_allocation a;
            a.alloc_id   = next_alloc;
            a.desc       = em.desc;
            a.desc_bytes = serialize_desc(em.desc);
            a.bytes      = *src;
            a.checksum   = stream_checksum(a.desc_bytes, a.bytes);
            tmp.allocations.push_back(std::move(a));
            ref = (uint32_t) tmp.allocations.size() - 1; // index into allocations table
            alloc_live.push_back(seg_live ? 1 : 0);
            if (!add_ok(sum_all, want, sum_all)) {
                set_err(e, "capture: byte total overflow");
                return false;
            }
            if (shared) {
                b_alloc[&em] = ref;
                ++n_b;
                if (!add_ok(sum_b, want, sum_b)) {
                    set_err(e, "capture: byte total overflow");
                    return false;
                }
            }
            if (seg_live) {
                if (!add_ok(live_sum, want, live_sum)) {
                    set_err(e, "capture: byte total overflow");
                    return false;
                }
            }
            ++next_alloc;
            return true;
        };

        for (const auto & seg : segs) {
            if (seg == nullptr) {
                set_err(err, "capture: null segment snapshot");
                return false;
            }
            const bool seg_live = seg->n_live_rows > 0;
            xkv_state_segment s;
            s.segment_id                  = seg->segment_id;
            s.segment_version             = seg->segment_version;
            s.profile                     = (uint32_t) seg->profile;
            s.source                      = (uint32_t) seg->source;
            s.profile_fingerprint         = seg->profile_fingerprint;
            s.source_fingerprint          = seg->source_fingerprint;
            s.layer_group_map_fingerprint = seg->layer_group_map_fingerprint;
            s.descriptor_fingerprint      = seg->descriptor_fingerprint;
            s.n_rows                      = seg->n_rows;
            s.n_live_rows                 = seg->n_live_rows;
            s.row_payload_ids             = seg->row_payload_ids;
            s.live_bits.reserve(seg->live_rows.size());
            for (bool b : seg->live_rows) {
                s.live_bits.push_back(b ? 1 : 0);
            }
            // Deterministic group order.
            std::vector<const xkv_factor_group_payload *> groups;
            for (const auto & g : seg->groups) {
                groups.push_back(&g);
            }
            std::sort(groups.begin(), groups.end(),
                      [](const auto * a, const auto * b) { return a->group_index < b->group_index; });
            for (const auto * g : groups) {
                if (g->b_k == nullptr || g->b_v == nullptr) {
                    set_err(err, "capture: group shares no immutable B handle (null B)");
                    return false;
                }
                xkv_state_group gi;
                gi.group_index             = g->group_index;
                gi.owning_layers           = g->owning_layers;
                gi.rank_k                  = g->rank_k;
                gi.rank_v                  = g->rank_v;
                gi.layer_feature_offsets_k = g->layer_feature_offsets_k;
                gi.layer_feature_dims_k    = g->layer_feature_dims_k;
                gi.layer_feature_offsets_v = g->layer_feature_offsets_v;
                gi.layer_feature_dims_v    = g->layer_feature_dims_v;
                gi.total_dim_k             = g->total_dim_k;
                gi.total_dim_v             = g->total_dim_v;
                const uint64_t sid = seg->segment_id;
                const uint64_t sv  = seg->segment_version;
                if (!emit_stream(sid, sv, g->group_index, factor_role::a_k, g->a_k, false, seg_live, err,
                                 gi.stream_ak) ||
                    !emit_stream(sid, sv, g->group_index, factor_role::b_k, *g->b_k, true, seg_live, err,
                                 gi.stream_bk) ||
                    !emit_stream(sid, sv, g->group_index, factor_role::a_v, g->a_v, false, seg_live, err,
                                 gi.stream_av) ||
                    !emit_stream(sid, sv, g->group_index, factor_role::b_v, *g->b_v, true, seg_live, err,
                                 gi.stream_bv)) {
                    return false;
                }
                gi.has_landmark = !g->landmark.bytes.empty() || g->landmark.desc.logical_shape.rows != 0;
                if (gi.has_landmark) {
                    // A landmark with empty host bytes and no readback entry is
                    // rejected inside emit_stream (device-resident rule).
                    if (!emit_stream(sid, sv, g->group_index, factor_role::landmark, g->landmark, false,
                                     seg_live, err, gi.stream_landmark)) {
                        return false;
                    }
                }
                gi.descriptor_fingerprint = g->descriptor_fingerprint;
                gi.config_fingerprint     = g->config_fingerprint;
            // Copy explicit per-chunk landmark metadata from the live group.
            // No assumed 8-stride or fixed row count: configurable chunk_tokens
            // and irregular/partial pack fragments copy exact begins, counts,
            // and conservative error bounds.
            if (gi.has_landmark && !g->landmark_chunks.empty()) {
                gi.landmark_chunks.reserve(g->landmark_chunks.size());
                for (const auto & ch : g->landmark_chunks) {
                    gi.landmark_chunks.push_back(
                        xkv_state_landmark_chunk::make(
                            ch.row_begin, ch.row_count, ch.error_bound, ch.source_fingerprint));
                }
            }
                const uint64_t lm_fp = gi.has_landmark ? g->landmark.desc.fingerprint() : 0;
                gi.landmark_chunks_fingerprint = xkv_state_landmark_chunks_fingerprint(
                gi.landmark_chunks, s.n_rows, s.row_payload_ids.data(), nullptr, nullptr,
                lm_fp, tmp.stamp, tmp.fingerprints.rope);
                s.groups.push_back(std::move(gi));
            }
            tmp.segments.push_back(std::move(s));
        }

        // Payloads: hot bindings (live) + caller-supplied live factored records.
        // No tombstone records are fabricated: dead rows live in row/live bits only.
        tmp.payloads.reserve(hot_bindings.size() + factored_payloads.size());
        for (const auto & h : hot_bindings) {
            xkv_state_payload p;
            p.payload_id              = h.payload_id;
            p.generation              = h.storage_generation;
            p.live                    = 1;
            p.locator.kind            = xkv_location_kind::hot;
            p.locator.segment_id      = 0;
            p.locator.segment_version = 0;
            p.locator.row             = h.hot_slot_row;
            p.locator.storage_generation = h.storage_generation;
            p.locator.state           = h.state;
            p.locator.seal_tx_nonce   = 0;
            tmp.payloads.push_back(std::move(p));
        }
        for (const auto & p : factored_payloads) {
            if (p.live != 1) {
                set_err(err, "capture: factored payload records must be live (no tombstone fabrication)");
                return false;
            }
            tmp.payloads.push_back(p);
        }
        std::sort(tmp.payloads.begin(), tmp.payloads.end(),
                  [](const auto & a, const auto & b) { return a.payload_id < b.payload_id; });

        // High-water closure: zero entries mean "no live allocator values" and
        // are derived as max+1 (empty table -> 1, the first valid id/nonce).
        // Explicit nonzero values are preserved for the closure check.
        if (tmp.config.next_segment_id == 0) {
            uint64_t m = 0;
            for (const auto & s : tmp.segments) {
                if (s.segment_id > m) {
                    m = s.segment_id;
                }
            }
            if (m == std::numeric_limits<uint64_t>::max()) {
                set_err(err, "capture: segment id space exhausted");
                return false;
            }
            tmp.config.next_segment_id = m + 1;
        }
        if (tmp.config.next_alloc_id == 0) {
            uint64_t m = 0;
            for (const auto & a : tmp.allocations) {
                if (a.alloc_id > m) {
                    m = a.alloc_id;
                }
            }
            if (m == std::numeric_limits<uint32_t>::max()) {
                set_err(err, "capture: allocation id space exhausted");
                return false;
            }
            tmp.config.next_alloc_id = m + 1;
        }
        if (tmp.config.next_seal_tx_nonce == 0) {
            uint64_t m = 0;
            for (const auto & p : tmp.payloads) {
                if (p.locator.seal_tx_nonce > m) {
                    m = p.locator.seal_tx_nonce;
                }
            }
            if (m == std::numeric_limits<uint64_t>::max()) {
                set_err(err, "capture: seal nonce space exhausted");
                return false;
            }
            tmp.config.next_seal_tx_nonce = m + 1;
        }

        // Counters from recomputed stream accounting; attested floors from caller.
        uint64_t mib_bytes = 0;
        if (!mul_ok((uint64_t) workspace_mib, 1048576ULL, mib_bytes)) {
            set_err(err, "capture: workspace budget overflow");
            return false;
        }
        tmp.counters.live_payload_bytes     = live_sum;
        tmp.counters.allocated_bytes        = accounting.allocated_bytes;
        tmp.counters.reserved_bytes         = accounting.reserved_bytes;
        tmp.counters.workspace_budget_bytes = mib_bytes;
        tmp.counters.hot_bytes              = accounting.hot_bytes;
        tmp.counters.factored_bytes         = sum_all;
        tmp.counters.active_segments        = (uint64_t) tmp.segments.size();
        tmp.counters.total_payloads         = (uint64_t) tmp.payloads.size();
        tmp.counters.unique_b_matrices      = n_b;
        tmp.counters.shared_b_bytes         = sum_b;

        if (!validate_image(tmp, fingerprints, provenance, limits, err)) {
            return false;
        }

        out = std::move(tmp);
        return true;
    } catch (const std::bad_alloc &) {
        set_err(err, "capture: allocation failure (output unchanged)");
        return false;
    }
}

// ---------------------------------------------------------------------------
// materialize_segments (verbatim copy, shared-B identity preserved)
// ---------------------------------------------------------------------------

bool materialize_segments(
    const xkv_state_image & image,
    std::vector<std::shared_ptr<const xkv_segment>> & out_segments,
    const xkv_state_limits & limits,
    std::string * err) {

    std::vector<std::shared_ptr<const xkv_segment>> tmp;

    try {
        if (!validate_image(image, image.fingerprints, image.provenance, limits, err)) {
            return false;
        }

        std::map<uint32_t, std::shared_ptr<const encoded_matrix>> b_cache;
        auto get_stream = [&](uint32_t ref, bool shared, encoded_matrix & em_out,
                              std::shared_ptr<const encoded_matrix> & shared_out) -> bool {
            if (ref >= image.allocations.size()) {
                set_err(err, "materialize: stream ref out of range");
                return false;
            }
            const auto & a = image.allocations[ref];
            if (shared) {
                const auto it = b_cache.find(ref);
                if (it != b_cache.end()) {
                    shared_out = it->second;
                    return true;
                }
                // Verbatim copy, shared by every group referencing this allocation.
                auto ptr   = std::make_shared<encoded_matrix>();
                ptr->desc  = a.desc;
                ptr->bytes = a.bytes;
                b_cache[ref] = ptr;
                shared_out   = ptr;
                return true;
            }
            em_out.desc  = a.desc;
            em_out.bytes = a.bytes;
            return true;
        };

        for (const auto & s : image.segments) {
            auto seg                         = std::make_shared<xkv_segment>();
            seg->segment_id                  = s.segment_id;
            seg->segment_version             = s.segment_version;
            seg->profile                     = (llama_xkv_storage_profile) s.profile;
            seg->source                      = (llama_xkv_source) s.source;
            seg->profile_fingerprint         = s.profile_fingerprint;
            seg->source_fingerprint          = s.source_fingerprint;
            seg->layer_group_map_fingerprint = s.layer_group_map_fingerprint;
            seg->descriptor_fingerprint      = s.descriptor_fingerprint;
            seg->n_rows                      = s.n_rows;
            seg->n_live_rows                 = s.n_live_rows;
            seg->row_payload_ids             = s.row_payload_ids;
            seg->live_rows.resize(s.live_bits.size());
            for (size_t i = 0; i < s.live_bits.size(); ++i) {
                seg->live_rows[i] = s.live_bits[i] != 0;
            }
            for (const auto & g : s.groups) {
                xkv_factor_group_payload gp;
                gp.group_index             = g.group_index;
                gp.owning_layers           = g.owning_layers;
                gp.rank_k                  = g.rank_k;
                gp.rank_v                  = g.rank_v;
                gp.layer_feature_offsets_k = g.layer_feature_offsets_k;
                gp.layer_feature_dims_k    = g.layer_feature_dims_k;
                gp.layer_feature_offsets_v = g.layer_feature_offsets_v;
                gp.layer_feature_dims_v    = g.layer_feature_dims_v;
                gp.total_dim_k             = g.total_dim_k;
                gp.total_dim_v             = g.total_dim_v;
                std::shared_ptr<const encoded_matrix> bk, bv;
                if (!get_stream(g.stream_ak, false, gp.a_k, bk) ||
                    !get_stream(g.stream_bk, true, gp.a_k, bk) ||
                    !get_stream(g.stream_av, false, gp.a_v, bv) ||
                    !get_stream(g.stream_bv, true, gp.a_v, bv)) {
                    return false;
                }
                gp.b_k = bk;
                gp.b_v = bv;
                if (g.has_landmark) {
                    std::shared_ptr<const encoded_matrix> dummy;
                    if (!get_stream(g.stream_landmark, false, gp.landmark, dummy)) {
                        return false;
                }
                    }
            // Materialize live group's landmark_chunks from image chunk tables.
                if (g.has_landmark && !g.landmark_chunks.empty()) {
                gp.landmark_chunks.reserve(g.landmark_chunks.size());
                for (const auto & c : g.landmark_chunks) {
                    xkv_landmark_chunk lch;
                    lch.row_begin   = c.row_begin;
                    lch.row_count   = c.row_count;
                    lch.error_bound = c.error_bound();
                    lch.source_fingerprint = c.source_fingerprint;
                    gp.landmark_chunks.push_back(lch);
                }
                gp.landmark_table_fingerprint =
                    compute_landmark_table_fingerprint(gp.landmark_chunks);
                }
                gp.descriptor_fingerprint = g.descriptor_fingerprint;
                gp.config_fingerprint     = g.config_fingerprint;
                gp.update_byte_counters();
                seg->groups.push_back(std::move(gp));
            }
            seg->update_byte_counters();
            tmp.push_back(std::move(seg));
        }

        out_segments = std::move(tmp);
        return true;
    } catch (const std::bad_alloc &) {
        set_err(err, "materialize: allocation failure (output unchanged)");
        return false;
    }
}

// ---------------------------------------------------------------------------
// image_matches_bytes
// ---------------------------------------------------------------------------

bool image_matches_bytes(const xkv_state_image & image, const uint8_t * data, size_t size,
                         const xkv_state_limits & limits) {
    if (data == nullptr) {
        return size == 0;
    }
    std::vector<uint8_t> re;
    if (!encode_image(image, re, limits, nullptr)) {
        return false;
    }
    if (re.size() != size) {
        return false;
    }
    return std::memcmp(re.data(), data, size) == 0;
}

// ---------------------------------------------------------------------------
// State-stream bridge
// ---------------------------------------------------------------------------

static bool write_framed_section(llama_io_write_i & io, uint32_t magic, const uint8_t * data, size_t size,
                           const char * what, std::string * err) {
    if (data == nullptr || size == 0) {
        set_err(err, std::string("bridge: refusing to write an empty ") + what + " section");
        return false;
    }
    try {
        uint8_t hdr[12];
        hdr[0] = (uint8_t) (magic & 0xFF);
        hdr[1] = (uint8_t) ((magic >> 8) & 0xFF);
        hdr[2] = (uint8_t) ((magic >> 16) & 0xFF);
        hdr[3] = (uint8_t) ((magic >> 24) & 0xFF);
        const uint64_t len = (uint64_t) size;
        for (int i = 0; i < 8; ++i) {
            hdr[4 + i] = (uint8_t) ((len >> (i * 8)) & 0xFF);
    }
        io.write(hdr, sizeof(hdr));
        io.write(data, size);
    } catch (const std::exception & e) {
        set_err(err, std::string("bridge: ") + what + " section write failed: " + e.what());
        return false;
    }
    return true;
}

static xkv_section_read_result read_framed_section(llama_io_read_i & io, uint32_t magic,
                                             uint64_t max_section_bytes, std::vector<uint8_t> & out,
                                             const char * what, std::string * err) {
    std::vector<uint8_t> tmp; // off-side: `out` untouched unless ok
    uint8_t mbuf[4];
    try {
        io.read(mbuf, sizeof(mbuf));
    } catch (const std::exception &) {
        return xkv_section_read_result::absent; // legacy stream end: no section
    }
    const uint32_t got = (uint32_t) mbuf[0] | ((uint32_t) mbuf[1] << 8) |
                         ((uint32_t) mbuf[2] << 16) | ((uint32_t) mbuf[3] << 24);
    if (got != magic) {
        set_err(err, std::string("bridge: ") + what + " section magic mismatch (corrupt trailer)");
        return xkv_section_read_result::corrupt;
    }
    uint8_t lb[8];
    try {
        io.read(lb, sizeof(lb));
    } catch (const std::exception & e) {
        set_err(err, std::string("bridge: truncated ") + what + " section length: " + e.what());
        return xkv_section_read_result::corrupt;
    }
    uint64_t len = 0;
    for (int i = 0; i < 8; ++i) {
        len |= ((uint64_t) lb[i] << (i * 8));
    }
    if (len == 0 || len > max_section_bytes) {
        set_err(err, std::string("bridge: ") + what + " section length zero/over budget (no alloc)");
        return xkv_section_read_result::corrupt;
    }
    try {
        tmp.resize((size_t) len);
    } catch (const std::bad_alloc &) {
        set_err(err, std::string("bridge: ") + what + " section alloc failure (output unchanged)");
        return xkv_section_read_result::corrupt;
    }
    try {
        io.read(tmp.data(), (size_t) len);
    } catch (const std::exception & e) {
        set_err(err, std::string("bridge: truncated ") + what + " section body: " + e.what());
        return xkv_section_read_result::corrupt;
    }
    out = std::move(tmp);
    return xkv_section_read_result::ok;
}

bool write_state_section(llama_io_write_i & io, const std::vector<uint8_t> & envelope,
                         std::string * err) {
    return write_framed_section(io, XKV_STATE_SECTION_MAGIC, envelope.data(), envelope.size(),
                                 "XKV", err);
}

xkv_section_read_result read_state_section(llama_io_read_i & io, uint64_t max_section_bytes,
                                            std::vector<uint8_t> & out_envelope, std::string * err) {
    return read_framed_section(io, XKV_STATE_SECTION_MAGIC, max_section_bytes, out_envelope, "XKV",
                                 err);
}

bool encode_cellmap(const std::vector<xkv_cell_binding> & bindings, std::vector<uint8_t> & out,
                         std::string * err) {
    sizer z;
    z.add(8);
    z.add_n((uint64_t) bindings.size(), 24);
    if (!z.ok) {
        set_err(err, "bridge: cellmap size precompute overflow");
        return false;
    }
    std::vector<uint8_t> tmp;
    try {
        tmp.reserve((size_t) z.size);
    } catch (const std::bad_alloc &) {
        set_err(err, "bridge: cellmap allocation failure (output unchanged)");
        return false;
    }
    writer w;
    w.buf.reserve((size_t) z.size);
    w.put_u64((uint64_t) bindings.size());
    for (const auto & b : bindings) {
        if (b.payload_id == 0 || b.generation == 0) {
            set_err(err, "bridge: cellmap entry with zero payload id/generation");
        return false;
    }
        w.put_u32(b.stream_idx);
        w.put_u32(b.ordinal);
        w.put_u64(b.payload_id);
        w.put_u64(b.generation);
    }
    if (w.buf.size() != (size_t) z.size) {
        set_err(err, "bridge: cellmap size drift");
        return false;
    }
    out.assign(w.buf.begin(), w.buf.end());
    return true;
}

bool decode_cellmap(const uint8_t * data, size_t size, std::vector<xkv_cell_binding> & out,
                     uint64_t max_entries, std::string * err) {
    std::vector<xkv_cell_binding> tmp; // off-side
    if (data == nullptr || size < 8) {
        set_err(err, "bridge: truncated cellmap header");
        return false;
    }
    uint64_t n = 0;
    for (int i = 0; i < 8; ++i) {
        n |= ((uint64_t) data[i] << (i * 8));
    }
    if (n > max_entries) {
        set_err(err, "bridge: cellmap count exceeds limit (no allocation attempted)");
        return false;
    }
    uint64_t want = 0;
    if (!mul_ok(n, 24, want) || !add_ok(want, 8, want) || want != size) {
        set_err(err, "bridge: cellmap size mismatch (truncated/trailing)");
        return false;
    }
    try {
        tmp.reserve((size_t) n);
    } catch (const std::bad_alloc &) {
        set_err(err, "bridge: cellmap allocation failure (output unchanged)");
        return false;
    }
    size_t pos = 8;
    for (uint64_t i = 0; i < n; ++i) {
        xkv_cell_binding b;
        b.stream_idx = (uint32_t) data[pos] | ((uint32_t) data[pos + 1] << 8) |
                       ((uint32_t) data[pos + 2] << 16) | ((uint32_t) data[pos + 3] << 24);
        b.ordinal = (uint32_t) data[pos + 4] | ((uint32_t) data[pos + 5] << 8) |
                    ((uint32_t) data[pos + 6] << 16) | ((uint32_t) data[pos + 7] << 24);
        b.payload_id = 0;
        b.generation = 0;
        for (int k = 0; k < 8; ++k) {
            b.payload_id |= ((uint64_t) data[pos + 8 + k] << (k * 8));
            b.generation |= ((uint64_t) data[pos + 16 + k] << (k * 8));
    }
        if (b.payload_id == 0 || b.generation == 0) {
            set_err(err, "bridge: cellmap entry with zero payload id/generation");
        return false;
}
        tmp.push_back(b);
        pos += 24;
    }
    out = std::move(tmp);
    return true;
}

bool write_cellmap_section(llama_io_write_i & io, const std::vector<xkv_cell_binding> & bindings,
                         std::string * err) {
    std::vector<uint8_t> body;
    if (!encode_cellmap(bindings, body, err)) {
        return false;
    }
    return write_framed_section(io, XKV_STATE_CELLMAP_MAGIC, body.data(), body.size(), "cellmap",
                                 err);
}

xkv_section_read_result read_cellmap_section(llama_io_read_i & io, uint64_t max_entries,
                                              std::vector<xkv_cell_binding> & out_bindings,
                         std::string * err) {
    std::vector<xkv_cell_binding> tmp; // off-side
    std::vector<uint8_t> body;
    if (max_entries > XKV_STATE_CELLMAP_MAX_ENTRIES) {
        set_err(err, "bridge: cellmap entry cap exceeds hard maximum");
        return xkv_section_read_result::corrupt;
    }
    // Framing cap mirrors the codec layout (8-byte count + 24 bytes/entry).
    // max_entries <= 2^24 here, so 8 + 24*max_entries cannot overflow u64.
    const uint64_t frame_cap = 8u + 24u * max_entries;
    const auto r =
        read_framed_section(io, XKV_STATE_CELLMAP_MAGIC, frame_cap, body, "cellmap", err);
    if (r != xkv_section_read_result::ok) {
        return r; // absent or corrupt propagate; `out_bindings` untouched
    }
    // Overflow-safe body cap recompute for decode (framing already bounded).
    if (!decode_cellmap(body.data(), body.size(), tmp, max_entries, err)) {
        return xkv_section_read_result::corrupt;
    }
    out_bindings = std::move(tmp);
    return xkv_section_read_result::ok;
}

xkv_state_config make_bridge_config(const llama_cparams & cp) {
    xkv_state_config c;
    c.profile                  = (uint32_t) cp.xkv_storage_profile;
    c.source                   = (uint32_t) cp.xkv_source;
    c.group_size               = cp.xkv_group_size;
    c.rank_k                   = cp.xkv_rank_k;
    c.rank_v                   = cp.xkv_rank_v;
    c.segment_tokens           = cp.xkv_segment_tokens;
    c.chunk_tokens             = cp.xkv_chunk_tokens;
    c.sr_budget                = cp.xkv_sr_budget;
    c.mode                     = (uint32_t) cp.xkv_mode;
    c.factor_a_k               = (uint32_t) cp.xkv_factor_a_k;
    c.factor_b_k               = (uint32_t) cp.xkv_factor_b_k;
    c.factor_a_v               = (uint32_t) cp.xkv_factor_a_v;
    c.factor_b_v               = (uint32_t) cp.xkv_factor_b_v;
    c.factor_balance           = (uint32_t) cp.xkv_factor_balance;
    c.landmark_type            = (uint32_t) cp.xkv_landmark_type;
    c.landmark_refine          = (uint32_t) cp.xkv_landmark_refine;
    c.landmark_refine_max_rows = cp.xkv_landmark_refine_max_rows;
    c.workspace_mib            = cp.xkv_workspace_mib;
    c.decode_cache_mib         = cp.xkv_decode_cache_mib;
    c.store_mib                = cp.xkv_store_mib;
    c.seed                     = cp.xkv_seed;
    const double saving        = cp.xkv_min_saving < 0.0 ? 0.0 : cp.xkv_min_saving;
    const double coverage      = cp.xkv_min_factor_coverage < 0.0 ? 0.0 : cp.xkv_min_factor_coverage;
    c.min_saving_ppm           = saving >= 1.0 ? PPM_DENOM : (uint64_t) (saving * 1000000.0 + 0.5);
    c.min_coverage_ppm         = coverage >= 1.0 ? PPM_DENOM : (uint64_t) (coverage * 1000000.0 + 0.5);
    c.factorizer               = (uint32_t) cp.xkv_factorizer;
    c.min_saving_bytes         = 1;
    // High-waters stay zero here: capture_image derives max+1 closure when the
    // caller has no live allocator values; explicit nonzero values are kept.
    return c;
}

xkv_state_fingerprints make_config_fingerprints(const xkv_state_config & cfg) {
    xkv_state_fingerprints fp;
    auto h32 = [](uint32_t v) {
        uint64_t h = FNV_OFFSET;
        for (int i = 0; i < 4; ++i) {
            h ^= (uint8_t) ((v >> (i * 8)) & 0xFF);
            h *= 0x100000001b3ULL;
    }
        return h;
    };
    auto mix = [&](uint64_t & acc, uint64_t v) {
        acc ^= v + 0x9e3779b97f4a7c15ULL + (acc << 6) + (acc >> 2);
    };
    uint64_t s = FNV_OFFSET;
    mix(s, h32(cfg.source));
    fp.source = s;
    uint64_t p = FNV_OFFSET;
    mix(p, h32(cfg.profile));
    mix(p, h32(cfg.group_size));
    mix(p, h32(cfg.segment_tokens));
    mix(p, h32(cfg.chunk_tokens));
    fp.profile = p;
    uint64_t fz = FNV_OFFSET;
    mix(fz, h32(cfg.factorizer));
    mix(fz, h32(cfg.factor_balance));
    mix(fz, h32(cfg.rank_k));
    mix(fz, h32(cfg.rank_v));
    fp.factorizer = fz;
    // codec: exact effective factor/landmark types plus their compiled layout
    // revisions. A Turbo2-saved bundle under a Turbo4-requested config (same
    // profile) mismatches here AND in check_config_compatible below.
    uint64_t cc = FNV_OFFSET;
    const uint32_t ctypes[5] = {cfg.factor_a_k, cfg.factor_b_k, cfg.factor_a_v, cfg.factor_b_v,
                                cfg.landmark_type};
    for (int i = 0; i < 5; ++i) {
        mix(cc, h32(ctypes[i]));
        mix(cc, ggml_turbo_layout_fingerprint((ggml_type) ctypes[i]));
    }
    fp.codec = cc;
    // backend: compiled codec-support capability set of this binary (which
    // Turbo layouts decode here). Per-stream format_revision enforcement at
    // validate time remains the hard gate; this names the capability set.
    uint64_t be = FNV_OFFSET;
    const ggml_type cap_types[3] = {GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO4_0};
    for (int i = 0; i < 3; ++i) {
        mix(be, ggml_turbo_layout_fingerprint(cap_types[i]));
    }
    fp.backend = be;
    // model/rope/tri are filled by the save-side owner from model/hparams/scorer.
    return fp;
}

// Exact effective-config compatibility: behavior/admission fields must match.
// High-waters excluded (closure-validated separately; live advances).
bool check_config_compatible(const xkv_state_config & image_cfg, const xkv_state_config & live_cfg,
                             std::string * err) {
    struct field {
        const char * name;
        uint32_t image_v;
        uint32_t live_v;
    };
    const field u32_fields[] = {
        {"mode", image_cfg.mode, live_cfg.mode},
        {"profile", image_cfg.profile, live_cfg.profile},
        {"source", image_cfg.source, live_cfg.source},
        {"group_size", image_cfg.group_size, live_cfg.group_size},
        {"rank_k", image_cfg.rank_k, live_cfg.rank_k},
        {"rank_v", image_cfg.rank_v, live_cfg.rank_v},
        {"segment_tokens", image_cfg.segment_tokens, live_cfg.segment_tokens},
        {"chunk_tokens", image_cfg.chunk_tokens, live_cfg.chunk_tokens},
        {"sr_budget", image_cfg.sr_budget, live_cfg.sr_budget},
        {"factor_a_k", image_cfg.factor_a_k, live_cfg.factor_a_k},
        {"factor_b_k", image_cfg.factor_b_k, live_cfg.factor_b_k},
        {"factor_a_v", image_cfg.factor_a_v, live_cfg.factor_a_v},
        {"factor_b_v", image_cfg.factor_b_v, live_cfg.factor_b_v},
        {"factor_balance", image_cfg.factor_balance, live_cfg.factor_balance},
        {"landmark_type", image_cfg.landmark_type, live_cfg.landmark_type},
        {"landmark_refine", image_cfg.landmark_refine, live_cfg.landmark_refine},
        {"landmark_refine_max_rows", image_cfg.landmark_refine_max_rows, live_cfg.landmark_refine_max_rows},
        {"workspace_mib", image_cfg.workspace_mib, live_cfg.workspace_mib},
        {"decode_cache_mib", image_cfg.decode_cache_mib, live_cfg.decode_cache_mib},
        {"store_mib", image_cfg.store_mib, live_cfg.store_mib},
        {"factorizer", image_cfg.factorizer, live_cfg.factorizer},
        {"min_saving_bytes", image_cfg.min_saving_bytes, live_cfg.min_saving_bytes},
    };
    for (const auto & f : u32_fields) {
        if (f.image_v != f.live_v) {
            set_err(err, std::string("config mismatch: ") + f.name);
            return false;
        }
    }
    if (image_cfg.seed != live_cfg.seed || image_cfg.min_saving_ppm != live_cfg.min_saving_ppm ||
        image_cfg.min_coverage_ppm != live_cfg.min_coverage_ppm) {
        set_err(err, "config mismatch: seed/min-saving/min-coverage gate");
        return false;
    }
    return true;
}

bool build_import_plan(const xkv_state_image & image, const xkv_state_limits & limits,
                       xkv_state_import_plan & out, std::string * err) {
    xkv_state_import_plan tmp;
    try {
        if (!validate_image(image, image.fingerprints, image.provenance, limits, err)) {
        return false;
    }
        std::vector<std::shared_ptr<const xkv_segment>> segs;
        if (!materialize_segments(image, segs, limits, err)) {
        return false;
}
        tmp.bundle.segments = std::move(segs);
        for (const auto & p : image.payloads) {
            tmp.bundle.locations.emplace_back(p.payload_id, p.locator);
        }
        tmp.bundle.stamp = image.stamp;
        uint64_t sealed = 0;
        for (const auto & p : image.payloads) {
            if (p.locator.kind == xkv_location_kind::factored) {
                ++sealed;
            }
        }
        tmp.bundle.sealed_count = sealed;
        // Audited high-water/cap block: verbatim copies from the validated image.
        // The store install max()es live high-waters with these (never derives
        // down), enforces dedup bytes against the cap, and records provenance.
        tmp.bundle.next_segment_id    = image.config.next_segment_id;
        tmp.bundle.next_alloc_id      = image.config.next_alloc_id;
        tmp.bundle.next_seal_tx_nonce = image.config.next_seal_tx_nonce;
        tmp.bundle.store_cap_bytes    = limits.max_store_bytes;
        tmp.bundle.config             = image.config;
        tmp.bundle.fingerprints       = image.fingerprints;
    } catch (const std::bad_alloc &) {
        set_err(err, "bridge: import plan allocation failure (output unchanged)");
        return false;
    }
    out = std::move(tmp);
    return true;
}

bool commit_import_plan(const xkv_state_import_plan & plan, const xkv_store_import_fn & import_fn,
                        std::string * err) {
    if (!import_fn) {
        set_err(err, "bridge: no store import function bound (refusing commit)");
        return false;
    }
    try {
        return import_fn(plan.bundle, err);
    } catch (const std::exception & e) {
        set_err(err, std::string("bridge: import function threw: ") + e.what());
        return false;
    }
}

} // namespace llama_xkv
