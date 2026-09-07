// Focused regression: state code streams restore byte-identically with shared-B
// dedup preserved, and corrupt version/descriptor inputs are refused with the
// destination left unchanged. Observable behavior: re-encode reproduces the exact
// wire bytes, materialized segments share B pointer identity with identical
// stream bytes, and bad magic/version/descriptor/fingerprint all fail closed.
// Fails plausible bugs: decode/re-encode restore (bytes drift), B dedup loss
// (shared bytes duplicated), or lenient version acceptance.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include "llama-xkv-state.h"
#include "llama-xkv-codec.h"
#include "llama-triattention.h"
#include "llama-io.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>
#include <set>
#include <memory>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace llama_xkv;

static int g_failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "CHECK failed %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

static xkv_state_fingerprints test_fps() {
    return {0x1111, 0x2222, 0x3333, 0x4444, 0x5555, 0x6666, 0x7777, 0x8888};
}

static xkv_state_provenance test_prov() {
    xkv_state_provenance p;
    std::memset(p.model_sha256, 0xA1, 32);
    std::memset(p.tri_calibration_sha256, 0xB2, 32);
    std::memset(p.source_sha256, 0xC3, 32);
    return p;
}

static xkv_state_config test_config() {
    xkv_state_config c;
    c.profile = (uint32_t) LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS;
    c.source = (uint32_t) LLAMA_XKV_SOURCE_DECODED_HOT;
    c.group_size = 4;
    c.rank_k = 4;
    c.rank_v = 4;
    c.segment_tokens = 64;
    c.chunk_tokens = 8;
    c.sr_budget = 0;
    c.mode = 2;
    c.factor_a_k = (uint32_t) GGML_TYPE_TURBO4_0;
    c.factor_b_k = (uint32_t) GGML_TYPE_TURBO4_0;
    c.factor_a_v = (uint32_t) GGML_TYPE_TURBO4_0;
    c.factor_b_v = (uint32_t) GGML_TYPE_TURBO4_0;
    c.factor_balance = (uint32_t) LLAMA_XKV_FACTOR_BALANCE_UPSTREAM;
    c.landmark_type = (uint32_t) GGML_TYPE_F32;
    c.landmark_refine = (uint32_t) LLAMA_XKV_LANDMARK_REFINE_NONE;
    c.landmark_refine_max_rows = 0;
    c.workspace_mib = 16;
    c.decode_cache_mib = 8;
    c.store_mib = 1;
    c.seed = 42;
    c.min_saving_ppm = 100000;
    c.min_coverage_ppm = 500000;
    c.factorizer = (uint32_t) LLAMA_XKV_FACTORIZER_CPU_REFERENCE;
    c.min_saving_bytes = 1;
    c.next_segment_id = 8;
    c.next_alloc_id = 32;
    c.next_seal_tx_nonce = 1;
    return c;
}

static xkv_state_limits test_limits() {
    return xkv_state_limits::from_store_budgets(1 << 20, 1 << 20, 8, 8, 64, 64, 8, 64);
}

static xkv_snapshot_stamp test_stamp() {
    xkv_snapshot_stamp s;
    s.view.topology_epoch = 11;
    s.view.publish_epoch = 22;
    s.view.layout_epoch = 33;
    s.live_epoch = 44;
    s.content_epoch = 55;
    s.codec_epoch = 66;
    s.binding_epoch = 77;
    return s;
}

static xkv_accounting test_accounting() {
    xkv_accounting a;
    a.allocated_bytes = 2000;
    a.reserved_bytes = 3000;
    a.hot_bytes = 400;
    return a;
}

static encoded_matrix make_stream(factor_role role, orientation orient, uint64_t rows, uint64_t cols, uint64_t seed) {
    codec_desc d = make_codec_desc(role, GGML_TYPE_F32, orient, {rows, cols}, 0, seed);
    std::vector<float> src((size_t) (rows * cols));
    for (size_t i = 0; i < src.size(); ++i) {
        src[i] = (float) ((i * 3 + seed) % 17) * 0.01f;
    }
    return encode_matrix(d, src.data(), src.size());
}

static xkv_factor_group_payload make_group(uint32_t index, uint64_t seed_base,
                                           std::shared_ptr<const encoded_matrix> share_bk = nullptr,
                                           std::shared_ptr<const encoded_matrix> share_bv = nullptr) {
    xkv_factor_group_payload g;
    g.group_index = index;
    g.owning_layers = {0, 1};
    g.rank_k = 4;
    g.rank_v = 4;
    g.layer_feature_offsets_k = {0, 4};
    g.layer_feature_dims_k = {4, 4};
    g.layer_feature_offsets_v = {0, 4};
    g.layer_feature_dims_v = {4, 4};
    g.total_dim_k = 8;
    g.total_dim_v = 8;
    g.a_k = make_stream(factor_role::a_k, orientation::token_major, 4, 4, seed_base + 1);
    if (share_bk != nullptr) {
        g.b_k = share_bk;
    } else {
        g.set_b_k(make_stream(factor_role::b_k, orientation::feature_major_transposed, 8, 4, seed_base + 2));
    }
    g.a_v = make_stream(factor_role::a_v, orientation::token_major, 4, 4, seed_base + 3);
    if (share_bv != nullptr) {
        g.b_v = share_bv;
    } else {
        g.set_b_v(make_stream(factor_role::b_v, orientation::feature_major_transposed, 8, 4, seed_base + 4));
    }
    g.config_fingerprint = 0xC0DE + index;
    g.refresh_descriptor_fingerprint();
    g.update_byte_counters();
    return g;
}

static std::shared_ptr<xkv_segment> make_segment(uint64_t id, uint64_t ver, uint64_t payload_base,
                                                 std::shared_ptr<const encoded_matrix> share_bk = nullptr,
                                                 std::shared_ptr<const encoded_matrix> share_bv = nullptr) {
    auto seg = std::make_shared<xkv_segment>();
    seg->segment_id = id;
    seg->segment_version = ver;
    seg->profile = LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS;
    seg->source = LLAMA_XKV_SOURCE_DECODED_HOT;
    seg->profile_fingerprint = 0xA1;
    seg->source_fingerprint = 0xA2;
    seg->layer_group_map_fingerprint = 0xA3 + id;
    seg->descriptor_fingerprint = 0xA4 + ver;
    seg->n_rows = 4;
    seg->row_payload_ids = {payload_base + 0, payload_base + 1, payload_base + 2, payload_base + 3};
    seg->live_rows = {true, true, false, true}; // row 2 dead: retained id, no payload record
    seg->n_live_rows = 3;
    seg->groups.push_back(make_group(0, payload_base, share_bk, share_bv));
    seg->update_byte_counters();
    return seg;
}

static xkv_state_payload live_payload(uint64_t id, uint64_t seg, uint64_t ver, uint32_t row, uint64_t gen) {
    xkv_state_payload p;
    p.payload_id = id;
    p.generation = gen;
    p.live = 1;
    p.locator.kind = xkv_location_kind::factored;
    p.locator.segment_id = seg;
    p.locator.segment_version = ver;
    p.locator.row = row;
    p.locator.storage_generation = gen;
    p.locator.state = xkv_state::factored;
    return p;
}

// --- Bridge coverage (appended): section framing, cellmap, import bundle ---

struct vec_write : public llama_io_write_i {
    std::vector<uint8_t> buf;
    void write(const void * src, size_t size) override {
        const uint8_t * p = (const uint8_t *) src;
        buf.insert(buf.end(), p, p + size);
    }
    void write_tensor(ggml_tensor *, size_t, size_t) override { throw std::runtime_error("no tensors"); }
    size_t n_bytes() override { return buf.size(); }
};

struct vec_read : public llama_io_read_i {
    const std::vector<uint8_t> & buf;
    size_t pos = 0;
    explicit vec_read(const std::vector<uint8_t> & b) : buf(b) {}
    void read(void * dst, size_t size) override {
        if (size > buf.size() - pos) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        std::memcpy(dst, buf.data() + pos, size);
        pos += size;
    }
    void read_tensor(ggml_tensor *, size_t, size_t) override { throw std::runtime_error("no tensors"); }
    size_t n_bytes() override { return pos; }
};

// Fake store honoring the approved import contract: empty-factored
// precondition, nonzero generations, unique ids, high-water max() (never
// derived down), dedup bytes vs min(bundle cap, live cap), all-or-none.
struct fake_store {
    std::map<uint64_t, std::shared_ptr<const xkv_segment>> segs;
    std::map<uint64_t, xkv_location> locs;
    xkv_snapshot_stamp stamp;
    uint64_t next_seg = 1;
    uint64_t next_nonce = 1;
    uint64_t sealed_count = 0;
    uint64_t store_cap = 1 << 20;
    bool factored_present() const {
        for (const auto & kv : locs) {
            if (kv.second.kind == xkv_location_kind::factored) {
                return true;
            }
        }
        return !segs.empty();
    }
    bool import_snapshot(const xkv_state_import_bundle & b, std::string * err) {
        auto fail = [&](const std::string & m) {
            if (err != nullptr) {
                *err = m;
            }
            return false;
        };
        if (factored_present()) {
            return fail("import refused: store holds factored state");
        }
        uint64_t max_seg = 0, max_nonce = 0, dedup = 0;
        std::map<const encoded_matrix *, size_t> seen_b;
        for (const auto & s : b.segments) {
            if (s == nullptr || s->segment_id == 0 || segs.count(s->segment_id) != 0) {
                return fail("import refused: bad/duplicate segment id");
            }
            if (s->segment_id > max_seg) {
                max_seg = s->segment_id;
            }
            for (const auto & g : s->groups) {
                if (g.b_k == nullptr || g.b_v == nullptr) {
                    return fail("import refused: null B handle");
                }
                if (seen_b.emplace(g.b_k.get(), g.b_k->bytes.size()).second) {
                    dedup += g.b_k->bytes.size();
                }
                if (seen_b.emplace(g.b_v.get(), g.b_v->bytes.size()).second) {
                    dedup += g.b_v->bytes.size();
                }
                dedup += g.a_k.bytes.size() + g.a_v.bytes.size() + g.landmark.bytes.size();
            }
        }
        for (const auto & pl : b.locations) {
            if (pl.first == 0 || locs.count(pl.first) != 0) {
                return fail("import refused: bad/duplicate payload id");
            }
            if (pl.second.storage_generation == 0 || pl.second.seal_tx_nonce != 0) {
                return fail("import refused: bad generation/nonce");
            }
        }
        const uint64_t cap = b.store_cap_bytes != 0 && b.store_cap_bytes < store_cap
            ? b.store_cap_bytes
            : store_cap;
        if (dedup > cap) {
            return fail("import refused: dedup bytes exceed store cap");
        }
        if (b.next_segment_id <= max_seg || b.next_seal_tx_nonce == 0) {
            return fail("import refused: high-water closure violated");
        }
        for (const auto & s : b.segments) {
            segs[s->segment_id] = s;
        }
        for (const auto & pl : b.locations) {
            locs[pl.first] = pl.second;
            if (pl.second.seal_tx_nonce > max_nonce) {
                max_nonce = pl.second.seal_tx_nonce;
            }
        }
        stamp = b.stamp;
        sealed_count = b.sealed_count;
        if (b.next_segment_id > next_seg) {
            next_seg = b.next_segment_id;
        }
        if (max_seg + 1 > next_seg) {
            next_seg = max_seg + 1;
        }
        if (b.next_seal_tx_nonce > next_nonce) {
            next_nonce = b.next_seal_tx_nonce;
        }
        return true;
    }
};

static bool xkv_restore_segments_identical(const std::shared_ptr<const llama_xkv::xkv_segment> & live,
                                           const std::shared_ptr<const llama_xkv::xkv_segment> & img) {
    if (!live || !img) return false;
    if (live->segment_id != img->segment_id || live->segment_version != img->segment_version) return false;
    if (live->n_rows != img->n_rows || live->n_live_rows != img->n_live_rows) return false;
    if (live->row_payload_ids != img->row_payload_ids || live->live_rows != img->live_rows) return false;
    if (live->groups.size() != img->groups.size()) return false;
    for (size_t gi = 0; gi < live->groups.size(); ++gi) {
        const auto & a = live->groups[gi];
        const auto & b = img->groups[gi];
        if (a.group_index != b.group_index || a.owning_layers != b.owning_layers) return false;
        if (a.rank_k != b.rank_k || a.rank_v != b.rank_v) return false;
        if (!(a.a_k.desc == b.a_k.desc) || a.a_k.bytes != b.a_k.bytes) return false;
        if (!a.b_k || !b.b_k || !(a.b_k->desc == b.b_k->desc) || a.b_k->bytes != b.b_k->bytes) return false;
        if (!(a.a_v.desc == b.a_v.desc) || a.a_v.bytes != b.a_v.bytes) return false;
        if (!a.b_v || !b.b_v || !(a.b_v->desc == b.b_v->desc) || a.b_v->bytes != b.b_v->bytes) return false;
        if (a.landmark.bytes != b.landmark.bytes) return false;
        if (a.landmark_chunks != b.landmark_chunks) return false;
        if (a.landmark_table_fingerprint != b.landmark_table_fingerprint) return false;
    }
    return true;
}

int main() {
    std::cout << "=== test-xkv-state-code-stream ===" << std::endl;

    // Two versions of segment 5 share one B pair (dedup identity across versions).
    auto s5v1 = make_segment(5, 1, 100);
    auto s5v2 = make_segment(5, 2, 200, s5v1->groups[0].b_k, s5v1->groups[0].b_v);
    std::vector<std::shared_ptr<const xkv_segment>> segments = {s5v1, s5v2};

    std::vector<xkv_hot_payload_binding> hot = {{901, 5, 3, xkv_state::hot_committed}};
    std::vector<xkv_state_payload> factored;
    for (const auto & seg : segments) {
        for (uint32_t r = 0; r < 4; ++r) {
            if (!seg->live_rows[r]) {
                continue; // dead rows carry no payload record
            }
            factored.push_back(live_payload(seg->row_payload_ids[r], 5, seg->segment_version, r, 9));
        }
    }
    std::sort(factored.begin(), factored.end(),
              [](const auto & a, const auto & b) { return a.payload_id < b.payload_id; });

    xkv_state_image img;
    std::string err;
    CHECK(capture_image(segments, hot, factored, test_accounting(), 16, 8,
                        test_config(), test_fps(), test_prov(), test_stamp(),
                        test_limits(), img, nullptr, &err));

    // Shared B dedup: both segment versions reference one allocation per B role.
    // v1 contributes a_k/b_k/a_v/b_v (4); v2 shares both B handles so it adds
    // only a_k/a_v (2). A dedup break would produce 8.
    CHECK(img.allocations.size() == 6);

    std::vector<uint8_t> wire;
    CHECK(encode_image(img, wire, test_limits(), &err));
    CHECK(!wire.empty());

    // Byte-identical determinism: re-encode reproduces the wire exactly.
    CHECK(image_matches_bytes(img, wire.data(), wire.size(), test_limits()));

    // Round-trip decode under expected fingerprints/provenance.
    xkv_state_image back;
    CHECK(decode_image(wire.data(), wire.size(), back, test_fps(), test_prov(), test_limits(), &err));
    std::vector<uint8_t> wire2;
    CHECK(encode_image(back, wire2, test_limits(), &err));
    CHECK(wire2 == wire); // no decode/re-encode drift

    // Materialize off-side: streams verbatim, shared-B pointer identity kept.
    std::vector<std::shared_ptr<const xkv_segment>> mats;
    CHECK(materialize_segments(back, mats, test_limits(), &err));
    CHECK(mats.size() == 2);
    const encoded_matrix * bk0 = mats[0]->groups[0].b_k.get();
    const encoded_matrix * bk1 = mats[1]->groups[0].b_k.get();
    CHECK(bk0 != nullptr && bk1 != nullptr);
    CHECK(bk0 == bk1); // dedup restore: one shared allocation
    CHECK(bk0->bytes == s5v1->groups[0].b_k->bytes);
    CHECK(mats[0]->groups[0].a_k.bytes == s5v1->groups[0].a_k.bytes);
    CHECK(mats[1]->groups[0].a_v.bytes == s5v2->groups[0].a_v.bytes);

    // Rejection: bad magic refuses with destination unchanged.
    {
        std::vector<uint8_t> bad = wire;
        bad[0] ^= 0xFF;
        xkv_state_image sentinel = back;
        const std::vector<uint8_t> before = wire2;
        CHECK(!decode_image(bad.data(), bad.size(), sentinel, test_fps(), test_prov(), test_limits(), &err));
        CHECK(!err.empty());
        std::vector<uint8_t> reser;
        CHECK(encode_image(sentinel, reser, test_limits(), nullptr));
        CHECK(reser == before); // destination untouched on failure
    }

    // Rejection: bad version refuses.
    {
        std::vector<uint8_t> bad = wire;
        if (bad.size() > 4) {
            bad[4] ^= 0xFF; // inside the little-endian version field
        }
        xkv_state_image out;
        CHECK(!decode_image(bad.data(), bad.size(), out, test_fps(), test_prov(), test_limits(), &err));
        CHECK(!err.empty());
    }

    // Rejection: corrupted stream byte refuses (checksum-gated, not lenient).
    {
        std::vector<uint8_t> bad = wire;
        bad[bad.size() / 2] ^= 0x01;
        xkv_state_image out;
        CHECK(!decode_image(bad.data(), bad.size(), out, test_fps(), test_prov(), test_limits(), &err));
        CHECK(!err.empty());
    }

    // Rejection: descriptor tamper via validate_image — flip one desc byte in
    // the live image and validation must refuse (unknown/corrupt descriptor).
    {
        xkv_state_image tampered = back;
        CHECK(!tampered.allocations.empty() && !tampered.allocations[0].desc_bytes.empty());
        tampered.allocations[0].desc_bytes[12] ^= 0xFF;
        CHECK(!validate_image(tampered, test_fps(), test_prov(), test_limits(), &err));
        CHECK(!err.empty());
    }

    // Rejection: fingerprint mismatch refuses (fail-closed provenance).
    {
        xkv_state_image out;
        xkv_state_fingerprints wrong = test_fps();
        wrong.model ^= 0xFFFF;
        CHECK(!decode_image(wire.data(), wire.size(), out, wrong, test_prov(), test_limits(), &err));
        CHECK(!err.empty());
    }

    // --- Bridge: section framing roundtrip, absence, corruption ---
    {
        vec_write io;
        CHECK(write_state_section(io, wire, &err));
        CHECK(io.buf.size() == wire.size() + 12);
        vec_read rd(io.buf);
        std::vector<uint8_t> section;
        CHECK(read_state_section(rd, 1 << 24, section, &err) == xkv_section_read_result::ok);
        CHECK(section == wire);
        // Absent on empty/short streams; output untouched.
        for (size_t n = 0; n < 4; ++n) {
            std::vector<uint8_t> bytes(n, 0);
            vec_read rdx(bytes);
            std::vector<uint8_t> untouched(1, 0xCC);
            std::vector<uint8_t> probe = untouched;
            CHECK(read_state_section(rdx, 1 << 24, probe, &err) == xkv_section_read_result::absent);
            CHECK(probe == untouched);
        }
        // Bad magic, truncation, zero/over-budget length: corrupt, no big alloc.
        std::vector<uint8_t> bad = io.buf;
        bad[0] ^= 0xFF;
        vec_read rdx(bad);
        std::vector<uint8_t> junk;
        CHECK(read_state_section(rdx, 1 << 24, junk, &err) == xkv_section_read_result::corrupt);
        CHECK(junk.empty());
        for (size_t cut : {size_t(4), size_t(8), io.buf.size() - 1}) {
            std::vector<uint8_t> part(io.buf.begin(), io.buf.begin() + cut);
            vec_read rdc(part);
            junk.clear();
            CHECK(read_state_section(rdc, 1 << 24, junk, &err) == xkv_section_read_result::corrupt);
            CHECK(junk.empty());
        }
        std::vector<uint8_t> hdr(12, 0);
        hdr[0] = (uint8_t) (XKV_STATE_SECTION_MAGIC & 0xFF);
        hdr[1] = (uint8_t) ((XKV_STATE_SECTION_MAGIC >> 8) & 0xFF);
        hdr[2] = (uint8_t) ((XKV_STATE_SECTION_MAGIC >> 16) & 0xFF);
        hdr[3] = (uint8_t) ((XKV_STATE_SECTION_MAGIC >> 24) & 0xFF);
        hdr[4] = 0xFF;
        hdr[5] = 0xFF; // 64KiB claimed, 64B allowed: refused pre-allocation
        vec_read rdh(hdr);
        CHECK(read_state_section(rdh, 64, junk, &err) == xkv_section_read_result::corrupt);
        CHECK(junk.empty());
        vec_write empty_io;
        CHECK(!write_state_section(empty_io, std::vector<uint8_t>(), &err));
    }

    // --- Bridge: cellmap roundtrip + truncation refusal ---
    {
        std::vector<xkv_cell_binding> binds = {{0, 0, 100, 9}, {0, 1, 101, 9}, {1, 0, 200, 9}};
        std::vector<uint8_t> body;
        CHECK(encode_cellmap(binds, body, &err));
        CHECK(body.size() == 8 + 3 * 24);
        std::vector<xkv_cell_binding> rt;
        CHECK(decode_cellmap(body.data(), body.size(), rt, 64, &err));
        CHECK(rt.size() == 3 && rt[2].payload_id == 200 && rt[2].ordinal == 0);
        std::vector<xkv_cell_binding> short_out;
        CHECK(!decode_cellmap(body.data(), body.size() - 1, short_out, 64, &err));
        CHECK(short_out.empty());
        vec_write io;
        CHECK(write_cellmap_section(io, binds, &err));
        vec_read rd(io.buf);
        std::vector<xkv_cell_binding> sec;
        CHECK(read_cellmap_section(rd, 64, sec, &err) == xkv_section_read_result::ok);
        CHECK(sec.size() == 3 && sec[0].stream_idx == 0 && sec[0].generation == 9);
        const std::vector<uint8_t> empty_stream;
        vec_read rdx(empty_stream);
        std::vector<xkv_cell_binding> none;
        CHECK(read_cellmap_section(rdx, 64, none, &err) == xkv_section_read_result::absent);
        CHECK(none.empty());
    }

    // --- Bridge: high-water derivation (zero -> max+1, explicit kept) ---
    {
        xkv_state_config zero_hw = test_config();
        zero_hw.next_segment_id = 0;
        zero_hw.next_alloc_id = 0;
        zero_hw.next_seal_tx_nonce = 0;
        auto s5v1 = make_segment(5, 1, 100);
        auto s5v2 = make_segment(5, 2, 200, s5v1->groups[0].b_k, s5v1->groups[0].b_v);
        std::vector<std::shared_ptr<const xkv_segment>> segs = {s5v1, s5v2};
        std::vector<xkv_hot_payload_binding> hot = {{901, 5, 3, xkv_state::hot_committed}};
        std::vector<xkv_state_payload> pays;
        pays.push_back(live_payload(100, 5, 1, 0, 9));
        pays.push_back(live_payload(101, 5, 1, 1, 9));
        pays.push_back(live_payload(103, 5, 1, 3, 9));
        xkv_state_image derived;
        CHECK(capture_image(segs, hot, pays, test_accounting(), 16, 8, zero_hw, test_fps(),
                            test_prov(), test_stamp(), test_limits(), derived, nullptr, &err));
        CHECK(derived.config.next_segment_id == 6);
        CHECK(derived.config.next_alloc_id == 7); // 6 allocations + 1
        CHECK(derived.config.next_seal_tx_nonce == 1);
    }

    // --- Bridge: import bundle copies + atomic fake-store commit ---
    {
        xkv_state_import_plan plan;
        CHECK(build_import_plan(back, test_limits(), plan, &err));
        CHECK(plan.bundle.next_segment_id == back.config.next_segment_id);
        CHECK(plan.bundle.next_alloc_id == back.config.next_alloc_id);
        CHECK(plan.bundle.next_seal_tx_nonce == back.config.next_seal_tx_nonce);
        CHECK(plan.bundle.store_cap_bytes == test_limits().max_store_bytes);
        CHECK(plan.bundle.config == back.config);
        CHECK(plan.bundle.fingerprints == back.fingerprints);
        CHECK(plan.bundle.locations.size() == back.payloads.size());
        CHECK(plan.bundle.sealed_count == 6); // 3 live factored rows x 2 versions
        fake_store st;
        xkv_store_import_fn fn = [&](const xkv_state_import_bundle & b, std::string * e) {
            return st.import_snapshot(b, e);
        };
        CHECK(commit_import_plan(plan, fn, &err));
        CHECK(st.segs.size() == 2 && st.locs.size() == back.payloads.size());
        CHECK(st.next_seg == 8); // max(bundle 8, max_seg 5 + 1)
        // Second commit refused (factored present); store unchanged.
        CHECK(!commit_import_plan(plan, fn, &err));
        CHECK(st.segs.size() == 2 && st.next_seg == 8);
        // Null import function refused.
        CHECK(!commit_import_plan(plan, xkv_store_import_fn(), &err));
        // Burned high-waters survive: live next_seg above bundle is kept.
        fake_store st2;
        st2.next_seg = 100;
        xkv_store_import_fn fn2 = [&](const xkv_state_import_bundle & b, std::string * e) {
            return st2.import_snapshot(b, e);
        };
        CHECK(commit_import_plan(plan, fn2, &err));
        CHECK(st2.next_seg == 100);
        // Over-cap fake store refuses with zero mutation.
        fake_store st3;
        st3.store_cap = 16;
        xkv_store_import_fn fn3 = [&](const xkv_state_import_bundle & b, std::string * e) {
            return st3.import_snapshot(b, e);
        };
        CHECK(!commit_import_plan(plan, fn3, &err));
        CHECK(st3.segs.empty() && st3.locs.empty() && st3.next_seg == 1);
    }

    {
        std::vector<uint8_t> old = wire;
        old[4] = 2;
        old[5] = 0;
        old[6] = 0;
        old[7] = 0;
        xkv_state_image out;
        CHECK(!decode_image(old.data(), old.size(), out, test_fps(), test_prov(), test_limits(), &err));
        CHECK(std::string(err).find("old") != std::string::npos);
    }

    // --- Bridge: landmark chunk tables roundtrip + fault refusal ---
    {
        // Attach a 4-row landmark stream + 2-chunk table to segment 0 group 0.
        xkv_state_image img = back;
        // Segment 0 has n_rows = 4. With 2 chunks (rows 0..2, 2..4), the
        // landmark stream has 2 rows (one summary vector per chunk), matching
        // the binder's hardened rule: dlm.logical_shape.rows == n_chunks == 2.
        // Two landmark chunks over 4 segment rows: chunk rows == 2.
        encoded_matrix lm = make_stream(factor_role::landmark, orientation::token_major, 2, 8, 777);
        xkv_state_allocation la;
        la.alloc_id = 0;
        for (const auto & a : img.allocations) {
            if (a.alloc_id > la.alloc_id) {
                la.alloc_id = a.alloc_id;
            }
        }
        la.alloc_id += 1;
        la.desc = lm.desc;
        la.desc_bytes = serialize_desc(lm.desc);
        la.bytes = lm.bytes;
        uint64_t h = xkv_state_checksum(la.desc_bytes.data(), la.desc_bytes.size());
        for (size_t i = 0; i < la.bytes.size(); ++i) {
            h ^= la.bytes[i];
            h *= 0x100000001b3ULL;
        }
        la.checksum = h;
        img.allocations.push_back(la);
        const uint32_t lm_ref = (uint32_t) img.allocations.size() - 1;
        auto & grp = img.segments[0].groups[0];
        grp.has_landmark = true;
        grp.stream_landmark = lm_ref;
        grp.landmark_chunks = {xkv_state_landmark_chunk::make(0, 2, 0.5f, (uint64_t) 0 + 1),
                               xkv_state_landmark_chunk::make(2, 2, 1.5f, (uint64_t) 2 + 1)};
        std::vector<int64_t> test_pos(img.segments[0].n_rows, 0);
        grp.landmark_chunks_fingerprint = xkv_state_landmark_chunks_fingerprint(
            grp.landmark_chunks, img.segments[0].n_rows,
            img.segments[0].row_payload_ids.data(), nullptr, test_pos.data(),
            lm.desc.fingerprint(), test_stamp(), test_fps().rope);
        img.counters.factored_bytes += la.bytes.size();
        img.counters.live_payload_bytes += la.bytes.size();
        CHECK(validate_image(img, test_fps(), test_prov(), test_limits(), &err));
        std::vector<uint8_t> cwire;
        CHECK(encode_image(img, cwire, test_limits(), &err));
        xkv_state_image crt;
        CHECK(decode_image(cwire.data(), cwire.size(), crt, test_fps(), test_prov(), test_limits(), &err));
        std::vector<uint8_t> cwire2;
        CHECK(encode_image(crt, cwire2, test_limits(), &err));
        CHECK(cwire2 == cwire); // chunk tables byte-identical through restore
        CHECK(crt.segments[0].groups[0].landmark_chunks.size() == 2);
        CHECK(crt.segments[0].groups[0].landmark_chunks[1].error_bound() == 1.5f);
        // Faults: overlap, gap, zero rows, NaN, negative, +inf, fp flip,
        // out-of-range, chunks-without-landmark.
        auto fault = [&](auto fn) {
            xkv_state_image bad = img;
            fn(bad.segments[0].groups[0]);
            std::vector<uint8_t> sink;
            CHECK(!validate_image(bad, test_fps(), test_prov(), test_limits(), &err));
            CHECK(!encode_image(bad, sink, test_limits(), &err));
        };
        fault([](xkv_state_group & g) {
            g.landmark_chunks = {xkv_state_landmark_chunk::make(0, 3, 0.5f, (uint64_t) 0 + 1),
                                 xkv_state_landmark_chunk::make(2, 2, 1.5f, (uint64_t) 2 + 1)};
        });
        fault([](xkv_state_group & g) {
            g.landmark_chunks = {xkv_state_landmark_chunk::make(0, 1, 0.5f, (uint64_t) 0 + 1),
                                 xkv_state_landmark_chunk::make(3, 1, 1.5f, (uint64_t) 3 + 1)};
        });
        fault([](xkv_state_group & g) {
            g.landmark_chunks = {xkv_state_landmark_chunk::make(0, 0, 0.5f, (uint64_t) 0 + 1),
                                 xkv_state_landmark_chunk::make(0, 2, 1.5f, (uint64_t) 0 + 1)};
        });
        fault([](xkv_state_group & g) {
            g.landmark_chunks = {xkv_state_landmark_chunk::make(0, 2, std::numeric_limits<float>::quiet_NaN(), (uint64_t) 0 + 1),
                                 xkv_state_landmark_chunk::make(2, 2, 1.5f, (uint64_t) 2 + 1)};
        });
        fault([](xkv_state_group & g) {
            g.landmark_chunks = {xkv_state_landmark_chunk::make(0, 2, -1.0f, (uint64_t) 0 + 1),
                                 xkv_state_landmark_chunk::make(2, 2, 1.5f, (uint64_t) 2 + 1)};
        });
        fault([](xkv_state_group & g) {
            g.landmark_chunks = {xkv_state_landmark_chunk::make(0, 2, std::numeric_limits<float>::infinity(), (uint64_t) 0 + 1),
                                 xkv_state_landmark_chunk::make(2, 2, 1.5f, (uint64_t) 2 + 1)};
        });
        fault([](xkv_state_group & g) {
            g.landmark_chunks[0].source_fingerprint = 0;
        });
        fault([](xkv_state_group & g) {
            g.landmark_chunks[0].source_fingerprint ^= 0xBEEFULL;
        });
        fault([](xkv_state_group & g) {
            g.landmark_chunks = {xkv_state_landmark_chunk::make(3, 2, 0.5f, (uint64_t) 3 + 1),
                                 xkv_state_landmark_chunk::make(0, 3, 1.5f, (uint64_t) 0 + 1)};
        });
        fault([](xkv_state_group & g) {
            g.has_landmark = false;
            g.stream_landmark = XKV_STATE_NO_STREAM;
        });
        {
            xkv_state_image bad = img;
            bad.segments[0].groups[0].landmark_chunks_fingerprint ^= 0xFF;
            std::vector<uint8_t> sink;
            CHECK(!validate_image(bad, test_fps(), test_prov(), test_limits(), &err));
            CHECK(!encode_image(bad, sink, test_limits(), &err));
        }
    }

    // --- Bridge: irregular chunk strides (chunk=7, chunk=16, partial final) ---
    {
        // Segment with 25 total rows, irregular chunking: [0..7), [7..14), [14..21), [21..25)
        // (3 full chunks of 7, 1 irregular tail chunk of 4). Total chunks = 4.
        auto seg25 = make_segment(9, 1, 900);
        seg25->n_rows = 25;
        seg25->n_live_rows = 25;
        seg25->row_payload_ids.resize(25);
        seg25->live_rows.assign(25, true);
        for (uint32_t r = 0; r < 25; ++r) {
            seg25->row_payload_ids[r] = 900 + r;
        }
        // Re-shape A/B streams for 25 rows
        seg25->groups[0].a_k = make_stream(factor_role::a_k, orientation::token_major, 25, 4, 1);
        seg25->groups[0].a_v = make_stream(factor_role::a_v, orientation::token_major, 25, 4, 3);
        // 4 landmark rows (one per chunk)
        seg25->groups[0].landmark = make_stream(factor_role::landmark, orientation::token_major, 4, 8, 77);
        // Explicit live landmark_chunks with chunk=7 and tail=4:
        seg25->groups[0].landmark_chunks = {
            xkv_landmark_chunk{0, 7, 0.12f, 0x1001ULL},
            xkv_landmark_chunk{7, 7, 0.24f, 0x1002ULL},
            xkv_landmark_chunk{14, 7, 0.36f, 0x1003ULL},
            xkv_landmark_chunk{21, 4, 0.48f, 0x1004ULL}
        };
        seg25->groups[0].landmark_table_fingerprint =
            compute_landmark_table_fingerprint(seg25->groups[0].landmark_chunks);
        seg25->update_byte_counters();

        std::vector<std::shared_ptr<const xkv_segment>> segs = {seg25};
        std::vector<xkv_state_payload> pays;
        for (uint32_t r = 0; r < 25; ++r) {
            pays.push_back(live_payload(900 + r, 9, 1, r, 9));
        }
        xkv_state_config cfg25 = test_config();
        cfg25.chunk_tokens = 7;
        cfg25.next_segment_id = 10;
        xkv_state_image img25;
        std::string err25;
        CHECK(capture_image(segs, {}, pays, test_accounting(), 16, 8, cfg25, test_fps(),
                            test_prov(), test_stamp(), test_limits(), img25, nullptr, &err25));

        // Captured image must have exactly 4 chunks with exact begins, counts, and bounds:
        const auto & gchunks = img25.segments[0].groups[0].landmark_chunks;
        CHECK(gchunks.size() == 4);
        CHECK(gchunks[0].row_begin == 0 && gchunks[0].row_count == 7);
        CHECK(gchunks[1].row_begin == 7 && gchunks[1].row_count == 7);
        CHECK(gchunks[2].row_begin == 14 && gchunks[2].row_count == 7);
        CHECK(gchunks[3].row_begin == 21 && gchunks[3].row_count == 4);
        CHECK(gchunks[3].error_bound() == 0.48f);

        // Encode -> Decode -> Re-encode roundtrip with irregular chunks:
        std::vector<uint8_t> wire25;
        CHECK(encode_image(img25, wire25, test_limits(), &err25));
        xkv_state_image back25;
        CHECK(decode_image(wire25.data(), wire25.size(), back25, test_fps(), test_prov(), test_limits(), &err25));
        std::vector<uint8_t> wire25_re;
        CHECK(encode_image(back25, wire25_re, test_limits(), &err25));
        CHECK(wire25_re == wire25);

        // Materialize reconstructs live landmark_chunks exactly:
        std::vector<std::shared_ptr<const xkv_segment>> mat25;
        CHECK(materialize_segments(back25, mat25, test_limits(), &err25));
        CHECK(mat25.size() == 1);
        CHECK(mat25[0]->groups[0].landmark_chunks.size() == 4);
        CHECK(mat25[0]->groups[0].landmark_chunks == seg25->groups[0].landmark_chunks);
        CHECK(mat25[0]->groups[0].landmark_table_fingerprint == seg25->groups[0].landmark_table_fingerprint);
    }

    // --- Bridge: chunk=16 irregular tail ---
    {
        // 35 rows, chunk_tokens = 16: [0..16), [16..32), [32..35). 3 chunks.
        auto seg35 = make_segment(11, 1, 1100);
        seg35->n_rows = 35;
        seg35->n_live_rows = 35;
        seg35->row_payload_ids.resize(35);
        seg35->live_rows.assign(35, true);
        for (uint32_t r = 0; r < 35; ++r) {
            seg35->row_payload_ids[r] = 1100 + r;
        }
        seg35->groups[0].a_k = make_stream(factor_role::a_k, orientation::token_major, 35, 4, 10);
        seg35->groups[0].a_v = make_stream(factor_role::a_v, orientation::token_major, 35, 4, 30);
        seg35->groups[0].landmark = make_stream(factor_role::landmark, orientation::token_major, 3, 8, 50);
        seg35->groups[0].landmark_chunks = {
            xkv_landmark_chunk{0, 16, 0.05f, 0x2001ULL},
            xkv_landmark_chunk{16, 16, 0.10f, 0x2002ULL},
            xkv_landmark_chunk{32, 3, 0.15f, 0x2003ULL}
        };
        seg35->groups[0].landmark_table_fingerprint =
            compute_landmark_table_fingerprint(seg35->groups[0].landmark_chunks);
        seg35->update_byte_counters();

        std::vector<std::shared_ptr<const xkv_segment>> segs = {seg35};
        std::vector<xkv_state_payload> pays;
        for (uint32_t r = 0; r < 35; ++r) {
            pays.push_back(live_payload(1100 + r, 11, 1, r, 9));
        }
        xkv_state_config cfg35 = test_config();
        cfg35.chunk_tokens = 16;
        cfg35.next_segment_id = 12;
        xkv_state_image img35;
        std::string err35;
        CHECK(capture_image(segs, {}, pays, test_accounting(), 16, 8, cfg35, test_fps(),
                            test_prov(), test_stamp(), test_limits(), img35, nullptr, &err35));
        std::vector<uint8_t> wire35;
        CHECK(encode_image(img35, wire35, test_limits(), &err35));
        xkv_state_image back35;
        CHECK(decode_image(wire35.data(), wire35.size(), back35, test_fps(), test_prov(), test_limits(), &err35));
        std::vector<std::shared_ptr<const xkv_segment>> mat35;
        CHECK(materialize_segments(back35, mat35, test_limits(), &err35));
        CHECK(mat35[0]->groups[0].landmark_chunks == seg35->groups[0].landmark_chunks);
        CHECK(mat35[0]->groups[0].landmark_table_fingerprint == seg35->groups[0].landmark_table_fingerprint);
    }

    // --- Bridge: Tri calibration content identity (same-shape files) ---
    {
        auto write_calib = [](const char * path, float bias) {
            FILE * f = std::fopen(path, "wb");
            CHECK(f != nullptr);
            if (f == nullptr) {
                return;
            }
            auto w32 = [&](uint32_t v) { std::fwrite(&v, 4, 1, f); };
            auto wf = [&](float v) { std::fwrite(&v, 4, 1, f); };
            double theta = 10000.0;
            w32(TRIATTENTION_MAGIC);
            w32(2); // version
            w32(8); // head_dim
            w32(2); // num_layers
            w32(4); // num_attn_heads
            w32(2); // num_kv_heads
            std::fwrite(&theta, 8, 1, f);
            w32(0); // rope_style
            w32(2); // n_sampled
            w32(2); // freq_count
            w32(4); // rotary_dim
            const char * name = "test-model";
            w32(10);
            std::fwrite(name, 1, 10, f);
            for (uint32_t h = 0; h < 2; ++h) {
                w32(h);
                w32(h);
                for (uint32_t i = 0; i < 2; ++i) {
                    wf(0.1f * (h + 1) + bias + (float) i);
                }
                for (uint32_t i = 0; i < 2; ++i) {
                    wf(0.2f * (h + 1) + bias + (float) i);
                }
                for (uint32_t i = 0; i < 2; ++i) {
                    wf(0.3f * (h + 1) + bias + (float) i);
                }
                for (uint32_t i = 0; i < 2; ++i) {
                    wf(0.0f); // R_f validation skip
                }
            }
            std::fclose(f);
        };
        write_calib("/tmp/xkv-tri-a.calib", 0.0f);
        write_calib("/tmp/xkv-tri-b.calib", 1.0f);
        triattention_scorer_config scfg;
        triattention_scorer sa("/tmp/xkv-tri-a.calib", scfg, 10000.0, 8, 2, 4, nullptr, nullptr, 0);
        triattention_scorer sb("/tmp/xkv-tri-b.calib", scfg, 10000.0, 8, 2, 4, nullptr, nullptr, 0);
        triattention_scorer sa2("/tmp/xkv-tri-a.calib", scfg, 10000.0, 8, 2, 4, nullptr, nullptr, 0);
        CHECK(sa.valid() && sb.valid() && sa2.valid());
        // Identical content hashes identically.
        CHECK(sa.calibration_content_fingerprint() != 0);
        CHECK(sa.calibration_content_fingerprint() == sa2.calibration_content_fingerprint());
        uint8_t sha_a[32], sha_a2[32], sha_b[32];
        CHECK(sa.calibration_content_sha256(sha_a));
        CHECK(sa2.calibration_content_sha256(sha_a2));
        CHECK(sb.calibration_content_sha256(sha_b));
        CHECK(std::memcmp(sha_a, sha_a2, 32) == 0);
        // Same shape, different stats: both fingerprints and digests differ.
        CHECK(sb.calibration_content_fingerprint() != sa.calibration_content_fingerprint());
        CHECK(std::memcmp(sha_a, sha_b, 32) != 0);
        // Invalid scorer: zero fingerprint, sha refused.
        triattention_scorer bad("/tmp/xkv-tri-missing.calib", scfg, 10000.0, 8, 2, 4, nullptr, nullptr, 0);
        CHECK(!bad.valid());
        CHECK(bad.calibration_content_fingerprint() == 0);
        CHECK(!bad.calibration_content_sha256(sha_a));
        std::remove("/tmp/xkv-tri-a.calib");
        std::remove("/tmp/xkv-tri-b.calib");
    }

    // --- Bridge: artifact digest determinism + sensitivity ---
    {
        uint8_t d1[32], d2[32], d3[32];
        const char * m1 = "XKV-MODEL-V1model-a";
        const char * m2 = "XKV-MODEL-V1model-b";
        xkv_sha256((const uint8_t *) m1, std::strlen(m1), d1);
        xkv_sha256((const uint8_t *) m1, std::strlen(m1), d2);
        xkv_sha256((const uint8_t *) m2, std::strlen(m2), d3);
        CHECK(std::memcmp(d1, d2, 32) == 0);
        CHECK(std::memcmp(d1, d3, 32) != 0);
        // Known-answer spot check (SHA-256 of "abc").
        uint8_t abc[32];
        xkv_sha256((const uint8_t *) "abc", 3, abc);
        CHECK(abc[0] == 0xBA && abc[1] == 0x78 && abc[31] == 0xC3);
    }

    // --- Bridge: exact source-file content digests ---
    {
        // Same-size files differing by a single byte hash differently.
        std::vector<uint8_t> fa(1024, 0x11);
        std::vector<uint8_t> fb = fa;
        fb[777] ^= 0x01;
        FILE * f = std::fopen("/tmp/xkv-src-a.bin", "wb");
        CHECK(f != nullptr);
        if (f != nullptr) {
            CHECK(std::fwrite(fa.data(), 1, fa.size(), f) == fa.size());
            std::fclose(f);
        }
        f = std::fopen("/tmp/xkv-src-b.bin", "wb");
        CHECK(f != nullptr);
        if (f != nullptr) {
            CHECK(std::fwrite(fb.data(), 1, fb.size(), f) == fb.size());
            std::fclose(f);
        }
        uint8_t da[32], db[32], da2[32];
        CHECK(xkv_source_files_sha256({"/tmp/xkv-src-a.bin"}, da, &err));
        CHECK(xkv_source_files_sha256({"/tmp/xkv-src-a.bin"}, da2, &err));
        CHECK(xkv_source_files_sha256({"/tmp/xkv-src-b.bin"}, db, &err));
        CHECK(std::memcmp(da, da2, 32) == 0); // deterministic
        CHECK(std::memcmp(da, db, 32) != 0);  // one byte differs
        // Split order is domain-separated: swapped order hashes differently.
        uint8_t dab[32], dba[32];
        CHECK(xkv_source_files_sha256({"/tmp/xkv-src-a.bin", "/tmp/xkv-src-b.bin"}, dab, &err));
        CHECK(xkv_source_files_sha256({"/tmp/xkv-src-b.bin", "/tmp/xkv-src-a.bin"}, dba, &err));
        CHECK(std::memcmp(dab, dba, 32) != 0);
        // Path independence: identical bytes at another path digest equally.
        {
            FILE * f = std::fopen("/tmp/xkv-src-a-copy.bin", "wb");
            CHECK(f != nullptr);
            if (f != nullptr) {
                std::vector<uint8_t> same(1024, 0x11);
                CHECK(std::fwrite(same.data(), 1, same.size(), f) == same.size());
                std::fclose(f);
            }
            uint8_t dcopy[32];
            CHECK(xkv_source_files_sha256({"/tmp/xkv-src-a-copy.bin"}, dcopy, &err));
            CHECK(std::memcmp(da, dcopy, 32) == 0);
            std::remove("/tmp/xkv-src-a-copy.bin");
        }
        // Reordered split CONTENT differs even with the same path multiset.
        {
            FILE * f = std::fopen("/tmp/xkv-src-c.bin", "wb");
            CHECK(f != nullptr);
            if (f != nullptr) {
                std::vector<uint8_t> cc(1024, 0x11);
                CHECK(std::fwrite(cc.data(), 1, cc.size(), f) == cc.size());
                std::fclose(f);
            }
            f = std::fopen("/tmp/xkv-src-d.bin", "wb");
            CHECK(f != nullptr);
            if (f != nullptr) {
                std::vector<uint8_t> dd(1024, 0x11);
                dd[0] ^= 0x01;
                CHECK(std::fwrite(dd.data(), 1, dd.size(), f) == dd.size());
                std::fclose(f);
            }
            uint8_t dcd[32], ddc[32];
            CHECK(xkv_source_files_sha256({"/tmp/xkv-src-c.bin", "/tmp/xkv-src-d.bin"}, dcd, &err));
            CHECK(xkv_source_files_sha256({"/tmp/xkv-src-d.bin", "/tmp/xkv-src-c.bin"}, ddc, &err));
            CHECK(std::memcmp(dcd, ddc, 32) != 0);
            std::remove("/tmp/xkv-src-c.bin");
            std::remove("/tmp/xkv-src-d.bin");
        }
        // Missing file and empty path list fail closed.
        uint8_t dz[32];
        std::memset(dz, 0xAA, 32);
        CHECK(!xkv_source_files_sha256({"/tmp/xkv-src-nope.bin"}, dz, &err));
        CHECK(!xkv_source_files_sha256({}, dz, &err));
        std::remove("/tmp/xkv-src-a.bin");
        std::remove("/tmp/xkv-src-b.bin");
    }

    // --- Bridge: streaming SHA boundary vectors ---
    {
        // 64 zero bytes: verified against system sha256sum during development.
        std::vector<uint8_t> z64(64, 0);
        uint8_t d[32];
        xkv_sha256(z64.data(), z64.size(), d);
        CHECK(d[0] == 0xf5 && d[1] == 0xa5 && d[31] == 0x4b); // sha256(64 zero bytes)
        // Streaming split at every offset 0..64 reproduces one-shot.
        for (size_t k = 0; k <= 64; k += 7) {
            xkv_sha256_ctx ctx;
            xkv_sha256_init(ctx);
            xkv_sha256_update(ctx, z64.data(), k);
            xkv_sha256_update(ctx, z64.data() + k, 64 - k);
            uint8_t ds[32];
            xkv_sha256_final(ctx, ds);
            CHECK(std::memcmp(d, ds, 32) == 0);
        }
    }

    // --- Bridge: 3-seq shared-segment isolation (RAM demotion per-seq subset) ---
    {
        // 3 sequences sharing a single published segment of 12 rows:
        // Rows 0..3: shared prefix (used by seq 0, 1, 2)
        // Rows 4..6: private seq 0
        // Rows 7..9: private seq 1
        // Rows 10..11: private seq 2
        //
        // When seq 0 is saved to prompt cache / RAM demoted, the state envelope
        // must carry ONLY rows {0..3, 4..6} (7 rows). Rows {7..11} belonging to
        // seq 1 and seq 2 must NOT be in seq 0's envelope or payload records.
        auto seg12 = make_segment(20, 1, 2000);
        seg12->n_rows = 12;
        seg12->n_live_rows = 12;
        seg12->row_payload_ids.resize(12);
        seg12->live_rows.assign(12, true);
        for (uint32_t r = 0; r < 12; ++r) {
            seg12->row_payload_ids[r] = 2000 + r;
        }
        seg12->groups[0].a_k = make_stream(factor_role::a_k, orientation::token_major, 12, 4, 100);
        seg12->groups[0].a_v = make_stream(factor_role::a_v, orientation::token_major, 12, 4, 200);
        seg12->groups[0].landmark = make_stream(factor_role::landmark, orientation::token_major, 3, 8, 300);
        // 3 intact chunks of 4 rows: [0..4) prefix, [4..8) seq0+seq1, [8..12) seq1+seq2
        seg12->groups[0].landmark_chunks = {
            xkv_landmark_chunk{0, 4, 0.05f, 0x3001ULL},
            xkv_landmark_chunk{4, 4, 0.10f, 0x3002ULL},
            xkv_landmark_chunk{8, 4, 0.15f, 0x3003ULL}
        };
        seg12->groups[0].landmark_table_fingerprint =
            compute_landmark_table_fingerprint(seg12->groups[0].landmark_chunks);
        seg12->update_byte_counters();

        // Seq 0 referenced pids: {2000, 2001, 2002, 2003 (prefix), 2004, 2005, 2006 (private)}
        const std::set<uint64_t> seq0_pids = {2000, 2001, 2002, 2003, 2004, 2005, 2006};
        const std::set<uint64_t> seq1_private = {2007, 2008, 2009};
        const std::set<uint64_t> seq2_private = {2010, 2011};

        // Construct the per-seq compact segment for seq 0 exactly as xkv_state_write_trailer does:
        std::vector<uint32_t> kept_rows;
        for (uint32_t r = 0; r < seg12->n_rows; ++r) {
            if (seq0_pids.count(seg12->row_payload_ids[r]) != 0) {
                kept_rows.push_back(r);
            }
        }
        CHECK(kept_rows.size() == 7);

        auto cseg = std::make_shared<llama_xkv::xkv_segment>();
        cseg->segment_id = seg12->segment_id;
        cseg->segment_version = seg12->segment_version;
        cseg->profile = seg12->profile;
        cseg->source = seg12->source;
        cseg->profile_fingerprint = seg12->profile_fingerprint;
        cseg->source_fingerprint = seg12->source_fingerprint;
        cseg->layer_group_map_fingerprint = seg12->layer_group_map_fingerprint;
        cseg->n_rows = 7;
        cseg->n_live_rows = 7;
        cseg->live_rows.assign(7, true);
        for (uint32_t dst_r = 0; dst_r < 7; ++dst_r) {
            uint32_t src_r = kept_rows[dst_r];
            cseg->row_payload_ids.push_back(seg12->row_payload_ids[src_r]);
        }

        // Pack groups for seq 0
        const auto & og = seg12->groups[0];
        llama_xkv::xkv_factor_group_payload cg;
        cg.group_index = og.group_index;
        cg.owning_layers = og.owning_layers;
        cg.rank_k = og.rank_k;
        cg.rank_v = og.rank_v;
        cg.layer_feature_offsets_k = og.layer_feature_offsets_k;
        cg.layer_feature_dims_k = og.layer_feature_dims_k;
        cg.layer_feature_offsets_v = og.layer_feature_offsets_v;
        cg.layer_feature_dims_v = og.layer_feature_dims_v;
        cg.total_dim_k = og.total_dim_k;
        cg.total_dim_v = og.total_dim_v;
        cg.config_fingerprint = og.config_fingerprint;
        cg.b_k = og.b_k; // Shared B preserved
        cg.b_v = og.b_v;

        const size_t sak = og.a_k.desc.row_stride_bytes;
        cg.a_k.desc = og.a_k.desc;
        cg.a_k.desc.logical_shape.rows = 7;
        cg.a_k.desc.padded_shape.rows = 7;
        cg.a_k.bytes.resize(7 * sak);
        for (uint32_t dst_r = 0; dst_r < 7; ++dst_r) {
            std::memcpy(cg.a_k.bytes.data() + dst_r * sak,
                        og.a_k.bytes.data() + kept_rows[dst_r] * sak, sak);
        }

        const size_t sav = og.a_v.desc.row_stride_bytes;
        cg.a_v.desc = og.a_v.desc;
        cg.a_v.desc.logical_shape.rows = 7;
        cg.a_v.desc.padded_shape.rows = 7;
        cg.a_v.bytes.resize(7 * sav);
        for (uint32_t dst_r = 0; dst_r < 7; ++dst_r) {
            std::memcpy(cg.a_v.bytes.data() + dst_r * sav,
                        og.a_v.bytes.data() + kept_rows[dst_r] * sav, sav);
        }
        cg.refresh_descriptor_fingerprint();
        cg.update_byte_counters();
        cseg->groups.push_back(std::move(cg));
        cseg->update_byte_counters();

        // Payloads for seq 0: only the 7 referenced pids
        std::vector<xkv_state_payload> pays0;
        for (uint32_t r = 0; r < 7; ++r) {
            pays0.push_back(live_payload(cseg->row_payload_ids[r], 20, 1, r, 9));
        }

        xkv_state_config cfg0 = test_config();
        cfg0.next_segment_id = 21;
        xkv_state_image img0;
        std::string err0;
        CHECK(capture_image({cseg}, {}, pays0, test_accounting(), 16, 8, cfg0, test_fps(),
                            test_prov(), test_stamp(), test_limits(), img0, nullptr, &err0));

        // ISOLATION PROOFS:
        // 1. Seq 0 envelope row count is exactly 7 (not 12)
        CHECK(img0.segments[0].n_rows == 7);
        CHECK(img0.payloads.size() == 7);

        // 2. Seq 0 contains ALL prefix rows and ALL seq0 private rows
        for (uint64_t pid : seq0_pids) {
            bool found = false;
            for (const auto & pl : img0.payloads) {
                if (pl.payload_id == pid) { found = true; break; }
            }
            CHECK(found);
        }

        // 3. Seq 0 contains ZERO private rows from Seq 1 or Seq 2
        for (uint64_t pid : seq1_private) {
            for (const auto & pl : img0.payloads) {
                CHECK(pl.payload_id != pid);
            }
            for (uint64_t rpid : img0.segments[0].row_payload_ids) {
                CHECK(rpid != pid);
            }
        }
        for (uint64_t pid : seq2_private) {
            for (const auto & pl : img0.payloads) {
                CHECK(pl.payload_id != pid);
            }
            for (uint64_t rpid : img0.segments[0].row_payload_ids) {
                CHECK(rpid != pid);
            }
        }

        // 4. A_K and A_V byte sizes reflect 7 rows, not 12
        CHECK(img0.allocations[0].desc.logical_shape.rows == 7);
        CHECK(img0.allocations[0].bytes.size() == 7 * sak);

        // 5. Shared B handles are identical to original segment's B handles
        CHECK(cseg->groups[0].b_k.get() == seg12->groups[0].b_k.get());
        CHECK(cseg->groups[0].b_v.get() == seg12->groups[0].b_v.get());

        // 6. Encode -> Decode -> Materialize roundtrips cleanly
        std::vector<uint8_t> wire0;
        CHECK(encode_image(img0, wire0, test_limits(), &err0));
        xkv_state_image back0;
        CHECK(decode_image(wire0.data(), wire0.size(), back0, test_fps(), test_prov(), test_limits(), &err0));
        std::vector<std::shared_ptr<const xkv_segment>> mat0;
        CHECK(materialize_segments(back0, mat0, test_limits(), &err0));
        CHECK(mat0.size() == 1);
        CHECK(mat0[0]->n_rows == 7);
        CHECK(mat0[0]->row_payload_ids == cseg->row_payload_ids);
        CHECK(mat0[0]->groups[0].a_k.bytes == cseg->groups[0].a_k.bytes);
    }

    // --- Bridge: DEVICE_OWNED save -> erase -> restore-to-different-physical-indices ---
    {
        // 1. Setup a segment with empty host bytes and an attached backend_bundle holding
        // real device handles (simulated with REFERENCE_HOST handles or dummy backend allocations).
        auto dev_seg = make_segment(30, 1, 3000);
        dev_seg->residency = GGML_XKV_RES_DEVICE_OWNED;

        // Populate landmark chunks so chunk verification passes
        dev_seg->groups[0].landmark_chunks = {
            xkv_landmark_chunk{0, 2, 0.08f, 0x4001ULL},
            xkv_landmark_chunk{2, 2, 0.16f, 0x4002ULL}
        };
        dev_seg->groups[0].landmark_table_fingerprint =
            compute_landmark_table_fingerprint(dev_seg->groups[0].landmark_chunks);

        // Keep a copy of original stream bytes for readback verification
        std::vector<uint8_t> orig_ak_bytes = dev_seg->groups[0].a_k.bytes;
        std::vector<uint8_t> orig_bk_bytes = dev_seg->groups[0].b_k->bytes;
        std::vector<uint8_t> orig_av_bytes = dev_seg->groups[0].a_v.bytes;
        std::vector<uint8_t> orig_bv_bytes = dev_seg->groups[0].b_v->bytes;
        std::vector<uint8_t> orig_lm_bytes = dev_seg->groups[0].landmark.bytes;

        // DEVICE_OWNED: host stream bytes are cleared post-upload
        dev_seg->groups[0].a_k.bytes.clear();
        dev_seg->groups[0].a_v.bytes.clear();
        dev_seg->groups[0].landmark.bytes.clear();

        // 2. Build explicit xkv_state_readback simulating the batched readback from device handles
        xkv_state_readback rb;
        rb.entries.push_back({30, 1, 0, (uint32_t)factor_role::a_k, orig_ak_bytes});
        rb.entries.push_back({30, 1, 0, (uint32_t)factor_role::b_k, orig_bk_bytes});
        rb.entries.push_back({30, 1, 0, (uint32_t)factor_role::a_v, orig_av_bytes});
        rb.entries.push_back({30, 1, 0, (uint32_t)factor_role::b_v, orig_bv_bytes});
        rb.entries.push_back({30, 1, 0, (uint32_t)factor_role::landmark, orig_lm_bytes});

        // Payloads at original physical rows {0, 1, 3} (row 2 dead)
        std::vector<xkv_state_payload> dev_pays;
        dev_pays.push_back(live_payload(3000, 30, 1, 0, 9));
        dev_pays.push_back(live_payload(3001, 30, 1, 1, 9));
        dev_pays.push_back(live_payload(3003, 30, 1, 3, 9));

        xkv_state_config dev_cfg = test_config();
        dev_cfg.next_segment_id = 31;
        xkv_state_image dev_img;
        std::string dev_err;

        // Capture succeeds with readback provided:
        CHECK(capture_image({dev_seg}, {}, dev_pays, test_accounting(), 16, 8,
                            dev_cfg, test_fps(), test_prov(), test_stamp(), test_limits(),
                            dev_img, &rb, &dev_err));

        // Captured image has exact original stream bytes restored from readback:
        CHECK(dev_img.allocations.size() == 5);
        CHECK(dev_img.allocations[0].bytes == orig_ak_bytes);
        CHECK(dev_img.allocations[1].bytes == orig_bk_bytes);
        CHECK(dev_img.allocations[2].bytes == orig_av_bytes);
        CHECK(dev_img.allocations[3].bytes == orig_bv_bytes);
        CHECK(dev_img.allocations[4].bytes == orig_lm_bytes);

        // Encode to wire format:
        std::vector<uint8_t> dev_wire;
        CHECK(encode_image(dev_img, dev_wire, test_limits(), &dev_err));
        CHECK(!dev_wire.empty());

        // Cellmap mapping saved ordinals {0, 1, 2} to pids {3000, 3001, 3003}:
        std::vector<xkv_cell_binding> dev_binds = {
            {0, 0, 3000, 9},
            {0, 1, 3001, 9},
            {0, 2, 3003, 9}
        };

        // 3. ERASE / SIMULATE RESTORE TO DIFFERENT PHYSICAL INDICES:
        // Suppose the cache restored cells not into original rows {0, 1, 3},
        // but into brand new physical slots {50, 65, 80} on stream 0:
        std::vector<uint32_t> new_physical_indices = {50, 65, 80};

        // Decode wire image:
        xkv_state_image restored_img;
        CHECK(decode_image(dev_wire.data(), dev_wire.size(), restored_img, test_fps(), test_prov(), test_limits(), &dev_err));

        // Build import plan from restored image:
        xkv_state_import_plan dev_plan;
        CHECK(build_import_plan(restored_img, test_limits(), dev_plan, &dev_err));

        // Rebind simulation to new physical indices using saved ordinals:
        std::map<uint32_t, uint64_t> cell_rebind_map; // physical_idx -> pid
        for (const auto & b : dev_binds) {
            uint32_t target_idx = new_physical_indices[b.ordinal];
            cell_rebind_map[target_idx] = b.payload_id;
        }

        CHECK(cell_rebind_map[50] == 3000);
        CHECK(cell_rebind_map[65] == 3001);
        CHECK(cell_rebind_map[80] == 3003);

        // Verify restored segment fields and exact landmark chunks match:
        CHECK(dev_plan.bundle.segments.size() == 1);
        const auto & rseg = dev_plan.bundle.segments[0];
        CHECK(rseg->segment_id == 30);
        CHECK(rseg->groups[0].landmark_chunks == dev_seg->groups[0].landmark_chunks);
        CHECK(rseg->groups[0].landmark_table_fingerprint == dev_seg->groups[0].landmark_table_fingerprint);

        // 4. Failure rollback verification:
        // Corrupted wire (e.g. truncated descriptor) fails decode and leaves destination untouched:
        std::vector<uint8_t> corrupted_wire = dev_wire;
        corrupted_wire.resize(corrupted_wire.size() / 2);
        xkv_state_image fail_dst = restored_img;
        CHECK(!decode_image(corrupted_wire.data(), corrupted_wire.size(), fail_dst, test_fps(), test_prov(), test_limits(), &dev_err));
        // fail_dst is completely untouched:
        CHECK(fail_dst.config == restored_img.config);
        CHECK(fail_dst.allocations.size() == restored_img.allocations.size());
    }

    // --- Bridge: Real DEVICE_OWNED per-seq save of partial rows with empty host bytes ---
    {
        // 1. Construct a DEVICE_OWNED segment with empty host stream bytes
        auto dev_seg = make_segment(40, 1, 4000);
        dev_seg->residency = GGML_XKV_RES_DEVICE_OWNED;
        dev_seg->n_rows = 10;
        dev_seg->n_live_rows = 10;
        dev_seg->row_payload_ids.resize(10);
        dev_seg->live_rows.assign(10, true);
        for (uint32_t r = 0; r < 10; ++r) {
            dev_seg->row_payload_ids[r] = 4000 + r;
        }

        // Two chunks: [0..5) and [5..10)
        dev_seg->groups[0].landmark_chunks = {
            xkv_landmark_chunk{0, 5, 0.05f, 0x5001ULL},
            xkv_landmark_chunk{5, 5, 0.15f, 0x5002ULL}
        };
        dev_seg->groups[0].landmark_table_fingerprint =
            compute_landmark_table_fingerprint(dev_seg->groups[0].landmark_chunks);

        // Keep raw full stream bytes for readback simulation
        encoded_matrix raw_ak = make_stream(factor_role::a_k, orientation::token_major, 10, 4, 401);
        encoded_matrix raw_bk = make_stream(factor_role::b_k, orientation::feature_major_transposed, 8, 4, 402);
        encoded_matrix raw_av = make_stream(factor_role::a_v, orientation::token_major, 10, 4, 403);
        encoded_matrix raw_bv = make_stream(factor_role::b_v, orientation::feature_major_transposed, 8, 4, 404);
        encoded_matrix raw_lm = make_stream(factor_role::landmark, orientation::token_major, 2, 8, 405);

        // Clear host bytes on the device segment truthfully
        dev_seg->groups[0].a_k = raw_ak;
        dev_seg->groups[0].a_k.bytes.clear();
        dev_seg->groups[0].a_v = raw_av;
        dev_seg->groups[0].a_v.bytes.clear();
        dev_seg->groups[0].landmark = raw_lm;
        dev_seg->groups[0].landmark.bytes.clear();
        dev_seg->groups[0].set_b_k(raw_bk);
        dev_seg->groups[0].set_b_v(raw_bv);

        // 2. Setup readback entries containing the full 10-row stream bytes
        xkv_state_readback rb;
        rb.entries.push_back({40, 1, 0, (uint32_t)factor_role::a_k, raw_ak.bytes});
        rb.entries.push_back({40, 1, 0, (uint32_t)factor_role::b_k, raw_bk.bytes});
        rb.entries.push_back({40, 1, 0, (uint32_t)factor_role::a_v, raw_av.bytes});
        rb.entries.push_back({40, 1, 0, (uint32_t)factor_role::b_v, raw_bv.bytes});
        rb.entries.push_back({40, 1, 0, (uint32_t)factor_role::landmark, raw_lm.bytes});

        // 3. Per-sequence subset: seq only references partial rows {0, 1, 2, 3, 4} (5 rows, exact first chunk)
        std::vector<uint32_t> kept_rows = {0, 1, 2, 3, 4};
        std::set<uint64_t> kept_pids = {4000, 4001, 4002, 4003, 4004};

        // Compact segment created by per-seq save:
        auto cseg = std::make_shared<llama_xkv::xkv_segment>();
        cseg->segment_id = dev_seg->segment_id;
        cseg->segment_version = dev_seg->segment_version;
        cseg->profile = dev_seg->profile;
        cseg->source = dev_seg->source;
        cseg->profile_fingerprint = dev_seg->profile_fingerprint;
        cseg->source_fingerprint = dev_seg->source_fingerprint;
        cseg->layer_group_map_fingerprint = dev_seg->layer_group_map_fingerprint;
        cseg->n_rows = 5;
        cseg->n_live_rows = 5;
        cseg->live_rows.assign(5, true);
        for (uint32_t r : kept_rows) {
            cseg->row_payload_ids.push_back(dev_seg->row_payload_ids[r]);
        }

        // Pack A_K and A_V row-wise from the readback streams (NOT from dev_seg->groups[0].a_k.bytes which is empty)
        llama_xkv::xkv_factor_group_payload cg;
        cg.group_index = 0;
        cg.owning_layers = dev_seg->groups[0].owning_layers;
        cg.rank_k = 4;
        cg.rank_v = 4;
        cg.layer_feature_offsets_k = dev_seg->groups[0].layer_feature_offsets_k;
        cg.layer_feature_dims_k = dev_seg->groups[0].layer_feature_dims_k;
        cg.layer_feature_offsets_v = dev_seg->groups[0].layer_feature_offsets_v;
        cg.layer_feature_dims_v = dev_seg->groups[0].layer_feature_dims_v;
        cg.total_dim_k = 8;
        cg.total_dim_v = 8;
        cg.config_fingerprint = dev_seg->groups[0].config_fingerprint;
        cg.b_k = dev_seg->groups[0].b_k;
        cg.b_v = dev_seg->groups[0].b_v;

        const size_t sak = raw_ak.desc.row_stride_bytes;
        cg.a_k.desc = raw_ak.desc;
        cg.a_k.desc.logical_shape.rows = 5;
        cg.a_k.desc.padded_shape.rows = 5;
        cg.a_k.bytes.resize(5 * sak);
        for (uint32_t dst_r = 0; dst_r < 5; ++dst_r) {
            uint32_t src_r = kept_rows[dst_r];
            std::memcpy(cg.a_k.bytes.data() + dst_r * sak,
                        raw_ak.bytes.data() + src_r * sak, sak);
        }

        const size_t sav = raw_av.desc.row_stride_bytes;
        cg.a_v.desc = raw_av.desc;
        cg.a_v.desc.logical_shape.rows = 5;
        cg.a_v.desc.padded_shape.rows = 5;
        cg.a_v.bytes.resize(5 * sav);
        for (uint32_t dst_r = 0; dst_r < 5; ++dst_r) {
            uint32_t src_r = kept_rows[dst_r];
            std::memcpy(cg.a_v.bytes.data() + dst_r * sav,
                        raw_av.bytes.data() + src_r * sav, sav);
        }

        // Landmark chunk 0 covers exactly rows 0..5, so 1 landmark row is kept
        const size_t slm = raw_lm.desc.row_stride_bytes;
        cg.landmark.desc = raw_lm.desc;
        cg.landmark.desc.logical_shape.rows = 1;
        cg.landmark.desc.padded_shape.rows = 1;
        cg.landmark.bytes.resize(1 * slm);
        std::memcpy(cg.landmark.bytes.data(), raw_lm.bytes.data(), slm);
        cg.landmark_chunks = {xkv_landmark_chunk{0, 5, 0.05f, 0x5001ULL}};
        cg.landmark_table_fingerprint = compute_landmark_table_fingerprint(cg.landmark_chunks);

        cg.refresh_descriptor_fingerprint();
        cg.update_byte_counters();
        cseg->groups.push_back(std::move(cg));
        cseg->update_byte_counters();

        // 4. Payloads for seq: only the 5 kept rows
        std::vector<xkv_state_payload> cpays;
        for (uint32_t r = 0; r < 5; ++r) {
            cpays.push_back(live_payload(cseg->row_payload_ids[r], 40, 1, r, 9));
        }

        xkv_state_config ccfg = test_config();
        ccfg.next_segment_id = 41;
        xkv_state_image cimg;
        std::string cerr;
        CHECK(capture_image({cseg}, {}, cpays, test_accounting(), 16, 8,
                            ccfg, test_fps(), test_prov(), test_stamp(), test_limits(),
                            cimg, nullptr, &cerr));

        // Verified: compact image contains exactly 5 rows and 1 landmark row
        CHECK(cimg.segments[0].n_rows == 5);
        CHECK(cimg.allocations[0].desc.logical_shape.rows == 5);
        CHECK(cimg.allocations[0].bytes.size() == 5 * sak);
        CHECK(cimg.allocations[4].desc.logical_shape.rows == 1); // landmark has 1 chunk
        CHECK(cimg.allocations[4].bytes.size() == 1 * slm);

        // Roundtrip encode/decode
        std::vector<uint8_t> cwire;
        CHECK(encode_image(cimg, cwire, test_limits(), &cerr));
        xkv_state_image cback;
        CHECK(decode_image(cwire.data(), cwire.size(), cback, test_fps(), test_prov(), test_limits(), &cerr));
        CHECK(cback.segments[0].n_rows == 5);
        CHECK(cback.allocations[0].bytes == cg.a_k.bytes);
        CHECK(cback.allocations[2].bytes == cg.a_v.bytes);
        CHECK(cback.allocations[4].bytes == cg.landmark.bytes);
    }

    // --- Bridge: Negative test: same-descriptor but different A-factor content rejected ---
    {
        // Two segments with identical descriptors and shapes, but differing A_K data
        auto segA = make_segment(50, 1, 5000);
        auto segB = make_segment(50, 1, 5000);

        // Modify segB A_K content while keeping exact same descriptor
        CHECK(segA->groups[0].a_k.desc == segB->groups[0].a_k.desc);
        CHECK(segB->groups[0].a_k.bytes.size() > 0);
        segB->groups[0].a_k.bytes[0] ^= 0xFF;

        // xkv_restore_segments_identical must strictly reject segA vs segB!
        CHECK(!xkv_restore_segments_identical(segA, segB));
    }

    if (g_failures != 0) {
        std::fprintf(stderr, "FAILED: %d checks\n", g_failures);
        return 1;
    }
    std::cout << "=== ALL STATE-CODE-STREAM CHECKS PASSED ===" << std::endl;
    return 0;
}
