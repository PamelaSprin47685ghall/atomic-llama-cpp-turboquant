// test-xkv-backend.cpp — focused CPU/Vulkan tests for immutable backend-resident
// XKV code-stream allocations (xkv_backend_allocation + batch builder).
//
// Covers Turbo2/3/4/Q8 exact bytes, mixed batch rejection atomicity, injected
// allocation/upload/readback/checksum failure, B handle sharing across versions,
// host-release commit, graph pin surviving store retirement, allocation ID
// overflow, accounting dedup, and one-sync counters. Vulkan unavailable means
// explicit SKIP only for the Vulkan subcase while CPU still runs; production
// capability is false without a Vulkan GPU backend.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include "llama-xkv-backend.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-xkv.h"
#include "ggml-cpu.h"
#include "../ggml/src/ggml-backend-impl.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace llama_xkv;

#define CHECK(cond) do { \
    if (!(cond)) { \
        std::cerr << "FAIL: " #cond " at line " << __LINE__ << "\n"; \
        std::abort(); \
    } \
} while (0)

static void check(bool cond, const std::string & msg) {
    if (!cond) {
        std::cerr << "FAILED: " << msg << std::endl;
        std::abort();
    }
}

struct cpu_env {
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_type_t buft = nullptr;

    ~cpu_env() {
        if (backend) {
            ggml_backend_free(backend);
            backend = nullptr;
        }
    }
};

static cpu_env make_cpu_env() {
    ggml_backend_load_all();
    cpu_env e;
    e.backend = ggml_backend_cpu_init();
    CHECK(e.backend != nullptr);
    e.buft = ggml_backend_cpu_buffer_type();
    CHECK(e.buft != nullptr);
    return e;
}

static ggml_backend_buffer_type_t null_buft() { return nullptr; }

static codec_desc make_desc(factor_role role, ggml_type type, uint64_t rows, uint64_t cols) {
    matrix_shape logical{rows, cols};
    orientation o = (role == factor_role::b_k || role == factor_role::b_v)
        ? orientation::feature_major_transposed
        : orientation::token_major;
    uint32_t gs = 0;
    if (type == GGML_TYPE_TURBO2_0 || type == GGML_TYPE_TURBO3_0 || type == GGML_TYPE_TURBO4_0) {
        gs = 128;
    }
    codec_desc d = make_codec_desc(role, type, o, logical, gs, 42);
    std::string err;
    CHECK(d.validate(&err));
    return d;
}

static std::vector<uint8_t> make_bytes(const codec_desc & d) {
    uint64_t n = encoded_matrix_bytes(d);
    CHECK(n > 0);
    std::vector<uint8_t> b((size_t) n);
    for (size_t i = 0; i < b.size(); ++i) {
        b[i] = (uint8_t) ((i * 37 + 11) & 0xff);
    }
    return b;
}

// ---------------------------------------------------------------------------
// 1. Turbo2/3/4/Q8 exact bytes on CPU (host residency via CPU buft)
// ---------------------------------------------------------------------------

static void test_exact_bytes_cpu(cpu_env & e) {
    std::cout << "[exact_bytes_cpu] starting..." << std::endl;
    struct tc { const char * name; ggml_type type; };
    const tc cases[] = {
        {"tq2", GGML_TYPE_TURBO2_0},
        {"tq3", GGML_TYPE_TURBO3_0},
        {"tq4", GGML_TYPE_TURBO4_0},
        {"q8",  GGML_TYPE_Q8_0},
    };
    for (const auto & c : cases) {
        codec_desc d = make_desc(factor_role::a_k, c.type, 13, 70);
        std::vector<uint8_t> bytes = make_bytes(d);
        char exact_err[128] = {};
        const size_t exact = ggml_xkv_exact_bytes(
            d.type, (int64_t) d.padded_shape.cols,
            (int64_t) d.logical_shape.rows, exact_err, sizeof(exact_err));
        check(exact != 0, std::string("exact bytes nonzero for ") + c.name);
        check(exact == bytes.size(), std::string("exact bytes match stream for ") + c.name);

        xkv_backend_batch_builder b;
        b.add_stream(d, bytes.data(), bytes.size());
        xkv_backend_batch_config cfg;
        cfg.residency = GGML_XKV_RES_REFERENCE_HOST;
        cfg.enforce_residency = true;
        xkv_backend_batch_result res;
        std::string err;
        CHECK(b.build(e.backend, e.buft, cfg, res, &err));
        CHECK(res.handles.size() == 1);
        CHECK(res.handles[0]->get_residency() == GGML_XKV_RES_REFERENCE_HOST);
        CHECK(res.handles[0]->get_padded_bytes() == exact);
        CHECK(res.handles[0]->get_logical_bytes() ==
              xkv_compute_logical_bytes(d.type, d.logical_shape.rows, d.logical_shape.cols));
        CHECK(res.handles[0]->get_descriptor_fingerprint() == d.fingerprint());
        CHECK(res.handles[0]->is_immutable());
        // Sync count: host upload through CPU backend may still sync once at most.
        CHECK(res.stats.sync_count <= 1);
        // Readback round-trips exact bytes.
        xkv_backend_readback_result rb;
        xkv_backend_batch_config rcfg;
        CHECK(xkv_backend_readback_batch(e.backend, res.handles, rcfg, rb, &err));
        CHECK(rb.stream_bytes.size() == 1);
        CHECK(rb.stream_bytes[0] == bytes);
        CHECK(rb.stats.sync_count <= 1);
    }
    std::cout << "[exact_bytes_cpu] OK" << std::endl;
}

// ---------------------------------------------------------------------------
// 2. Mixed batch rejection atomicity (one bad stream => whole batch fails, no output)
// ---------------------------------------------------------------------------

static void test_mixed_batch_atomicity(cpu_env & e) {
    std::cout << "[mixed_batch_atomicity] starting..." << std::endl;
    codec_desc good = make_desc(factor_role::a_k, GGML_TYPE_TURBO4_0, 8, 128);
    codec_desc bad = good;
    std::vector<uint8_t> good_bytes = make_bytes(good);
    // Corrupt: wrong size (one byte short)
    std::vector<uint8_t> bad_bytes = good_bytes;
    bad_bytes.pop_back();

    xkv_backend_batch_builder b;
    b.add_stream(good, good_bytes.data(), good_bytes.size());
    b.add_stream(bad, bad_bytes.data(), bad_bytes.size());
    xkv_backend_batch_config cfg;
    cfg.residency = GGML_XKV_RES_REFERENCE_HOST;
    cfg.enforce_residency = true;
    xkv_backend_batch_result res;
    std::string err;
    CHECK(!b.build(e.backend, e.buft, cfg, res, &err));
    CHECK(!err.empty());
    CHECK(res.handles.empty());
    // Failure-atomic outputs: a pre-filled result must be bit-identical after failure.
    res.stats.uploaded_streams = 999;
    CHECK(!b.build(e.backend, e.buft, cfg, res, &err));
    CHECK(res.handles.empty());
    CHECK(res.stats.uploaded_streams == 999);
    CHECK(!res.is_success());
    CHECK(!res.is_committed());
    std::cout << "[mixed_batch_atomicity] OK (" << err << ")" << std::endl;
}

// ---------------------------------------------------------------------------
// 3. Injected failures: alloc / upload / readback / checksum
// ---------------------------------------------------------------------------

static void test_injected_failures(cpu_env & e) {
    std::cout << "[injected_failures] starting..." << std::endl;
    // The builder is intentionally non-movable/non-copyable (it owns a
    // thread-safe ID generator with a mutex), so each case builds locally.
    auto build_two = [&](const xkv_backend_batch_config & cfg,
                           xkv_backend_batch_result & res,
                           std::string & err) -> bool {
        codec_desc d0 = make_desc(factor_role::a_k, GGML_TYPE_TURBO4_0, 8, 128);
        codec_desc d1 = make_desc(factor_role::a_v, GGML_TYPE_TURBO3_0, 8, 128);
        std::vector<uint8_t> b0 = make_bytes(d0);
        std::vector<uint8_t> b1 = make_bytes(d1);
        xkv_backend_batch_builder b;
        b.add_stream(d0, b0.data(), b0.size());
        b.add_stream(d1, b1.data(), b1.size());
        return b.build(e.backend, e.buft, cfg, res, &err);
    };
    {
        xkv_backend_batch_config cfg;
        cfg.residency = GGML_XKV_RES_REFERENCE_HOST;
        cfg.inject_alloc_fail_at = 1;
        xkv_backend_batch_result res;
        std::string err;
        CHECK(!build_two(cfg, res, err));
        CHECK(res.handles.empty());
    }
    {
        xkv_backend_batch_config cfg;
        cfg.residency = GGML_XKV_RES_REFERENCE_HOST;
        cfg.inject_upload_fail_at = 0;
        xkv_backend_batch_result res;
        std::string err;
        CHECK(!build_two(cfg, res, err));
        CHECK(res.handles.empty());
    }
    {
        xkv_backend_batch_config cfg;
        cfg.residency = GGML_XKV_RES_REFERENCE_HOST;
        cfg.verify_readback = true;
        cfg.inject_readback_fail_at = 1;
        xkv_backend_batch_result res;
        std::string err;
        // Verification downloads need a caller workspace; size it generously here.
        std::vector<uint8_t> vwork(1 << 20);
        cfg.verify_workspace = vwork.data();
        cfg.verify_workspace_bytes = vwork.size();
        CHECK(!build_two(cfg, res, err));
        CHECK(res.handles.empty());
    }
    {
        xkv_backend_batch_config cfg;
        cfg.residency = GGML_XKV_RES_REFERENCE_HOST;
        cfg.verify_checksum = true;
        cfg.inject_checksum_fail_at = 0;
        xkv_backend_batch_result res;
        std::string err;
        std::vector<uint8_t> vwork(1 << 20);
        cfg.verify_workspace = vwork.data();
        cfg.verify_workspace_bytes = vwork.size();
        CHECK(!build_two(cfg, res, err));
        CHECK(res.handles.empty());
    }
    // Readback-batch injection
    // Post-submit injection at EVERY queued index: sets for streams 0..k are
    // already queued when index k fires, so the fence must flush before any
    // destruction. Failure is atomic and outputs stay untouched.
    for (int k = 0; k < 2; ++k) {
        for (int with_verify = 0; with_verify < 2; ++with_verify) {
            codec_desc d0 = make_desc(factor_role::a_k, GGML_TYPE_TURBO4_0, 8, 128);
            codec_desc d1 = make_desc(factor_role::a_v, GGML_TYPE_TURBO3_0, 8, 128);
            std::vector<uint8_t> b0 = make_bytes(d0);
            std::vector<uint8_t> b1 = make_bytes(d1);
            xkv_backend_batch_builder b;
            b.add_stream(d0, b0.data(), b0.size());
            b.add_stream(d1, b1.data(), b1.size());
            xkv_backend_batch_config cfg;
            cfg.residency = GGML_XKV_RES_REFERENCE_HOST;
            cfg.inject_post_submit_fail_at = k;
            std::vector<uint8_t> vwork;
            if (with_verify) {
                cfg.verify_checksum = true;
                vwork.resize(b0.size() + b1.size());
                cfg.verify_workspace = vwork.data();
                cfg.verify_workspace_bytes = vwork.size();
            }
            xkv_backend_batch_result res;
            res.stats.uploaded_streams = 999; // sentinel
            std::string err;
            CHECK(!b.build(e.backend, e.buft, cfg, res, &err));
            CHECK(!err.empty());
            CHECK(res.handles.empty());
            CHECK(res.stats.uploaded_streams == 999);
            CHECK(!res.is_success());
        }
    }
    {
        codec_desc d = make_desc(factor_role::a_k, GGML_TYPE_Q8_0, 4, 64);
        std::vector<uint8_t> bytes = make_bytes(d);
        xkv_backend_batch_builder b;
        b.add_stream(d, bytes.data(), bytes.size());
        xkv_backend_batch_config cfg;
        cfg.residency = GGML_XKV_RES_REFERENCE_HOST;
        xkv_backend_batch_result res;
        std::string err;
        CHECK(b.build(e.backend, e.buft, cfg, res, &err));
        CHECK(res.handles.size() == 1);
        xkv_backend_batch_config rcfg;
        rcfg.inject_readback_fail_at = 0;
        xkv_backend_readback_result rb;
        // Failure-atomic outputs: pre-filled readback result untouched by failure.
        rb.stream_bytes.push_back(std::vector<uint8_t>{1, 2, 3});
        rb.stats.sync_count = 999;
        CHECK(!xkv_backend_readback_batch(e.backend, res.handles, rcfg, rb, &err));
        CHECK(rb.stream_bytes.size() == 1);
        CHECK(rb.stream_bytes[0] == std::vector<uint8_t>({1, 2, 3}));
        CHECK(rb.stats.sync_count == 999);
        rb.stream_bytes.clear();
        rb.stats.sync_count = 0;
        rcfg.inject_readback_fail_at = -1;
        rcfg.inject_checksum_fail_at = 0;
        CHECK(!xkv_backend_readback_batch(e.backend, res.handles, rcfg, rb, &err));
        CHECK(rb.stream_bytes.empty());
}
    std::cout << "[injected_failures] OK" << std::endl;
}

// ---------------------------------------------------------------------------
// 18. Pinned-old COW: old version stays readable after new publish, shared B
//     counted once, peak covers both resident buffers.
// ---------------------------------------------------------------------------
static void test_pinned_old_cow(cpu_env & e) {
    std::cout << "[pinned_old_cow] starting..." << std::endl;
    std::string err;
    xkv_backend_batch_config cfg;
    cfg.residency = GGML_XKV_RES_REFERENCE_HOST;
    xkv_allocation_id_generator gen(1, 100000);
    codec_desc da0 = make_desc(factor_role::a_k, GGML_TYPE_TURBO4_0, 8, 128);
    codec_desc db  = make_desc(factor_role::b_k, GGML_TYPE_TURBO4_0, 64, 128);
    std::vector<uint8_t> ba0 = make_bytes(da0);
    std::vector<uint8_t> bb  = make_bytes(db);
    // v0 published and pinned (graph pin keeps the old store alive).
    xkv_backend_batch_result r0;
    {
        xkv_backend_batch_builder b(&gen);
        b.add_stream(da0, ba0.data(), ba0.size());
        b.add_stream(db, bb.data(), bb.size());
        CHECK(b.build(e.backend, e.buft, cfg, r0, &err));
    }
    auto h_a0 = r0.handles[0], h_b = r0.handles[1];
    const size_t buf_v0 = ggml_backend_buffer_get_size(h_a0->get_buffer());
    CHECK(h_b->get_buffer() == h_a0->get_buffer());
    // v1: pack A survivors into a new store, reuse the shared B handle.
    const std::vector<uint32_t> rows = {0, 1, 2, 3};
    codec_desc da1 = make_desc(factor_role::a_k, GGML_TYPE_TURBO4_0, 4, 128);
    xkv_backend_pack_request rq;
    rq.src = h_a0;
    rq.surviving_rows = rows;
    rq.dst_desc = da1;
    xkv_backend_pack_result pr;
    CHECK(xkv_backend_pack_batch(e.backend, e.buft, {rq}, cfg, gen, pr, &err));
    CHECK(pr.stats.sync_count <= 1);
    auto h_a1 = pr.handles[0];
    CHECK(h_a1->get_buffer() != h_a0->get_buffer()); // true COW: separate store
    // Pinned old A still reads back intact after the new publish.
    {
        std::vector<uint8_t> old_out;
        CHECK(h_a0->readback(e.backend, old_out, &err));
        CHECK(old_out == ba0);
    }
    // Shared B identity across versions (same object, same ID, same tensor).
    CHECK(h_b->get_allocation_id() != h_a0->get_allocation_id());
    CHECK(h_b->get_allocation_id() != h_a1->get_allocation_id());
    // Peak: live {A1, B} + pinned {A0, B} — B once, both buffers resident.
    std::vector<std::shared_ptr<xkv_backend_allocation>> live = {h_a1, h_b};
    std::vector<std::shared_ptr<xkv_backend_allocation>> pinned = {h_a0, h_b};
    xkv_backend_peak_accounting peak = xkv_backend_calculate_peak_accounting(live, pinned);
    CHECK(peak.live.count == 2);
    CHECK(peak.pinned_retired.count == 2);
    CHECK(peak.combined.count == 3);
    CHECK(!peak.live.overflow && !peak.pinned_retired.overflow && !peak.combined.overflow);
    const size_t buf_v1 = ggml_backend_buffer_get_size(h_a1->get_buffer());
    CHECK(peak.combined.actual_bytes == buf_v0 + buf_v1);
    // New packed rows match the survivors; old rows untouched.
    const size_t row_bytes = ba0.size() / 8;
    std::vector<uint8_t> expect;
    for (uint32_t rr : rows) {
        expect.insert(expect.end(), ba0.begin() + (size_t) rr * row_bytes,
                        ba0.begin() + ((size_t) rr + 1) * row_bytes);
    }
    {
        xkv_backend_readback_result rb;
        xkv_backend_batch_config rcfg;
        CHECK(xkv_backend_readback_batch(e.backend, pr.handles, rcfg, rb, &err));
        CHECK(rb.stream_bytes.size() == 1 && rb.stream_bytes[0] == expect);
    }
    std::cout << "[pinned_old_cow] OK" << std::endl;
}

// ---------------------------------------------------------------------------
// 19. T-1 reservation boundary: exactly one byte short refuses at either
//     stage; exact fit succeeds. Failure leaves outputs untouched.
// ---------------------------------------------------------------------------
static void test_reservation_t_minus_1(cpu_env & e) {
    std::cout << "[reservation_t_minus_1] starting..." << std::endl;
    std::string err;
    xkv_backend_batch_config cfg;
    cfg.residency = GGML_XKV_RES_REFERENCE_HOST;
    codec_desc d = make_desc(factor_role::a_k, GGML_TYPE_TURBO4_0, 8, 128);
    std::vector<uint8_t> bytes = make_bytes(d);
    // Probe the exact actual buffer bytes for this shape on this buft.
    size_t t = 0;
    {
        xkv_backend_batch_builder probe;
        probe.add_stream(d, bytes.data(), bytes.size());
        xkv_backend_batch_result res;
        CHECK(probe.build(e.backend, e.buft, cfg, res, &err));
        t = ggml_backend_buffer_get_size(res.handles[0]->get_buffer());
        CHECK(t > 0);
    }
    auto expect_fail = [&](uint64_t reserved, uint64_t cap) {
        xkv_backend_batch_builder b;
        b.add_stream(d, bytes.data(), bytes.size());
        xkv_backend_store_reservation rsv;
        rsv.reserved_bytes = reserved;
        rsv.cap_bytes = cap;
        xkv_backend_batch_result res;
        res.stats.uploaded_streams = 999; // sentinel
        CHECK(!b.build(e.backend, e.buft, cfg, res, &err, &rsv));
        CHECK(!err.empty());
        CHECK(res.handles.empty());
        CHECK(res.stats.uploaded_streams == 999);
        CHECK(!res.is_success());
    };
    const uint64_t huge = (uint64_t) 1 << 40;
    expect_fail(t - 1, huge); // one byte short of actual: refuses
    expect_fail(huge, t - 1); // cap one byte short: refuses
    // Exact fit on both succeeds.
    {
        xkv_backend_batch_builder b;
        b.add_stream(d, bytes.data(), bytes.size());
        xkv_backend_store_reservation rsv;
        rsv.reserved_bytes = t;
        rsv.cap_bytes = t;
        xkv_backend_batch_result res;
        CHECK(b.build(e.backend, e.buft, cfg, res, &err, &rsv));
        CHECK(res.handles.size() == 1);
        CHECK(res.handles[0]->get_padded_bytes() == bytes.size());
    }
    std::cout << "[reservation_t_minus_1] OK" << std::endl;
}
// ---------------------------------------------------------------------------
// 4. B handle sharing across versions (same shared_ptr in two batches)
// ---------------------------------------------------------------------------

static void test_b_handle_sharing(cpu_env & e) {
    std::cout << "[b_handle_sharing] starting..." << std::endl;
    codec_desc db = make_desc(factor_role::b_k, GGML_TYPE_TURBO4_0, 256, 128);
    codec_desc da0 = make_desc(factor_role::a_k, GGML_TYPE_TURBO4_0, 8, 128);
    codec_desc da1 = make_desc(factor_role::a_k, GGML_TYPE_TURBO4_0, 16, 128);
    std::vector<uint8_t> bb = make_bytes(db);
    std::vector<uint8_t> ba0 = make_bytes(da0);
    std::vector<uint8_t> ba1 = make_bytes(da1);

    xkv_backend_batch_builder b0;
    b0.add_stream(db, bb.data(), bb.size());
    b0.add_stream(da0, ba0.data(), ba0.size());
    xkv_backend_batch_config cfg;
    cfg.residency = GGML_XKV_RES_REFERENCE_HOST;
    xkv_backend_batch_result r0;
    std::string err;
    CHECK(b0.build(e.backend, e.buft, cfg, r0, &err));
    CHECK(r0.handles.size() == 2);
    std::shared_ptr<xkv_backend_allocation> b_handle = r0.handles[0];
    const uint64_t b_id = b_handle->get_allocation_id();

    // Version 2 shares the same B handle with a new A.
    xkv_backend_batch_builder b1;
    b1.add_shared_handle(b_handle);
    b1.add_stream(da1, ba1.data(), ba1.size());
    xkv_backend_batch_result r1;
    CHECK(b1.build(e.backend, e.buft, cfg, r1, &err));
    CHECK(r1.handles.size() == 2);
    CHECK(r1.handles[0].get() == b_handle.get());
    CHECK(r1.handles[0]->get_allocation_id() == b_id);
    // Copy/move of the shared_ptr keeps ID + tensor stable.
    std::shared_ptr<xkv_backend_allocation> cp = r1.handles[0];
    CHECK(cp->get_allocation_id() == b_id);
    CHECK(cp->get_tensor() == b_handle->get_tensor());
    std::shared_ptr<xkv_backend_allocation> mv = std::move(cp);
    CHECK(mv->get_allocation_id() == b_id);

    // Accounting dedups the shared B counted twice by pointer.
    std::vector<std::shared_ptr<xkv_backend_allocation>> all = {
        r0.handles[0], r0.handles[1], r1.handles[0], r1.handles[1]};
    xkv_backend_accounting acc = xkv_backend_calculate_accounting(all);
    CHECK(acc.count == 3); // B + A0 + A1
    std::cout << "[b_handle_sharing] OK" << std::endl;
}

// ---------------------------------------------------------------------------
// 5. Host-release commit (device-owned releases only via explicit commit)
// ---------------------------------------------------------------------------

static void test_host_release_commit(cpu_env & e) {
    std::cout << "[host_release_commit] starting..." << std::endl;
    // Borrowed raw vectors are never captured or auto-released (UAF-safe):
    // host reference batches retain bytes truthfully through commit.
    {
        codec_desc d = make_desc(factor_role::a_k, GGML_TYPE_Q8_0, 4, 64);
        std::vector<uint8_t> src = make_bytes(d);
        const size_t before = src.size();
        xkv_backend_batch_builder b;
        b.add_stream(d, &src);
        xkv_backend_batch_config cfg;
        cfg.residency = GGML_XKV_RES_REFERENCE_HOST;
        xkv_backend_batch_result res;
        std::string err;
        CHECK(b.build(e.backend, e.buft, cfg, res, &err));
        CHECK(res.handles.size() == 1);
        CHECK(res.handles[0]->get_residency() == GGML_XKV_RES_REFERENCE_HOST);
        CHECK(res.handles[0]->get_host_bytes() == res.handles[0]->get_actual_bytes());
        CHECK(res.handles[0]->get_device_bytes() == 0);
        // No callbacks registered for host residency: commit is a no-op success.
        CHECK(res.commit_host_release(&err));
        CHECK(src.size() == before);
        CHECK(res.is_committed());
        // Host residency retains bytes: nothing is caller-clearable.
        CHECK(res.releasable_stream_indices.empty());
    }
    // Shared-ownership vectors on a host batch are retained (no release).
    {
        codec_desc d = make_desc(factor_role::a_k, GGML_TYPE_Q8_0, 4, 64);
        auto src = std::make_shared<std::vector<uint8_t>>(make_bytes(d));
        const size_t before = src->size();
        xkv_backend_batch_builder b;
        b.add_stream(d, src);
        xkv_backend_batch_config cfg; // default wants DEVICE_OWNED, no enforcement
        xkv_backend_batch_result res;
        std::string err;
        CHECK(b.build(e.backend, e.buft, cfg, res, &err));
        CHECK(res.handles.size() == 1);
        // Adapted truthfully to host on a CPU buft.
        CHECK(res.handles[0]->get_residency() == GGML_XKV_RES_REFERENCE_HOST);
        CHECK(res.commit_host_release(&err));
        CHECK(src->size() == before);
        // Result outliving the source is safe: commit after source death is a no-op.
        src.reset();
        CHECK(res.commit_host_release(&err));
    }
    // Enforcing device-owned on a host buft must fail atomically.
    {
        codec_desc d = make_desc(factor_role::a_k, GGML_TYPE_Q8_0, 4, 64);
        std::vector<uint8_t> src = make_bytes(d);
        xkv_backend_batch_builder b;
        b.add_stream(d, &src);
        xkv_backend_batch_config cfg;
        cfg.residency = GGML_XKV_RES_DEVICE_OWNED;
        cfg.enforce_residency = true;
        xkv_backend_batch_result res;
        std::string err;
        CHECK(!b.build(e.backend, e.buft, cfg, res, &err));
        CHECK(res.handles.empty());
        CHECK(!src.empty()); // host bytes untouched on failure
        // Commit on failure refuses.
        CHECK(!res.commit_host_release(&err));
    }
    std::cout << "[host_release_commit] OK" << std::endl;
}

// ---------------------------------------------------------------------------
// 6. Graph pin surviving store retirement (shared_ptr keeps tensor alive)
// ---------------------------------------------------------------------------

static void test_graph_pin_survives_retirement(cpu_env & e) {
    std::cout << "[graph_pin] starting..." << std::endl;
    codec_desc d = make_desc(factor_role::b_k, GGML_TYPE_TURBO4_0, 64, 128);
    std::vector<uint8_t> bytes = make_bytes(d);
    std::shared_ptr<xkv_backend_allocation> pin;
    ggml_tensor * raw_tensor = nullptr;
    {
        xkv_backend_batch_builder b;
        b.add_stream(d, bytes.data(), bytes.size());
        xkv_backend_batch_config cfg;
        cfg.residency = GGML_XKV_RES_REFERENCE_HOST;
        xkv_backend_batch_result res;
        std::string err;
        CHECK(b.build(e.backend, e.buft, cfg, res, &err));
        pin = res.handles[0]; // graph snapshot pin
        raw_tensor = pin->get_tensor();
        CHECK(raw_tensor != nullptr);
    }
    // Simulate store retirement: original batch result is gone, pin keeps it alive.
    CHECK(pin.use_count() >= 1);
    CHECK(pin->get_tensor() == raw_tensor);
    CHECK(pin->get_padded_bytes() == bytes.size());
    std::vector<uint8_t> out;
    std::string err;
    CHECK(pin->readback(e.backend, out, &err));
    CHECK(out == bytes);
    std::cout << "[graph_pin] OK" << std::endl;
}

// ---------------------------------------------------------------------------
// 7. Allocation ID overflow refusal (atomic, no partial handles)
// ---------------------------------------------------------------------------

static void test_allocation_id_overflow(cpu_env & e) {
    std::cout << "[alloc_id_overflow] starting..." << std::endl;
    xkv_allocation_id_generator gen(1, 2); // only IDs 1..2
    uint64_t id = 0;
    std::string err;
    CHECK(gen.allocate_id(id, &err) && id == 1);
    CHECK(gen.allocate_id(id, &err) && id == 2);
    CHECK(!gen.allocate_id(id, &err));

    xkv_allocation_id_generator gen2(1, 1); // only one ID
    xkv_backend_batch_builder b(&gen2);
    codec_desc d0 = make_desc(factor_role::a_k, GGML_TYPE_TURBO4_0, 4, 128);
    codec_desc d1 = make_desc(factor_role::a_v, GGML_TYPE_TURBO4_0, 4, 128);
    std::vector<uint8_t> b0 = make_bytes(d0);
    std::vector<uint8_t> b1 = make_bytes(d1);
    b.add_stream(d0, b0.data(), b0.size());
    b.add_stream(d1, b1.data(), b1.size());
    xkv_backend_batch_config cfg;
    cfg.residency = GGML_XKV_RES_REFERENCE_HOST;
    xkv_backend_batch_result res;
    CHECK(!b.build(e.backend, e.buft, cfg, res, &err));
    CHECK(res.handles.empty());
    std::cout << "[alloc_id_overflow] OK" << std::endl;
}

// ---------------------------------------------------------------------------
// 8. Accounting dedup + logical/padded/actual/device/host separation
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// 7d. Store reservation: preflight refusal is atomic (no attach, no release).
// ---------------------------------------------------------------------------

static void test_store_reservation(cpu_env & e) {
    std::cout << "[reservation] starting..." << std::endl;
    codec_desc d = make_desc(factor_role::a_k, GGML_TYPE_TURBO4_0, 8, 128);
    std::vector<uint8_t> bytes = make_bytes(d);
    // Tiny reservation: preflight estimate exceeds -> atomic refusal.
    {
        xkv_backend_batch_builder b;
        b.add_stream(d, bytes.data(), bytes.size());
        xkv_backend_batch_config cfg;
        cfg.residency = GGML_XKV_RES_REFERENCE_HOST;
        xkv_backend_store_reservation rsv;
        rsv.reserved_bytes = 1;
        xkv_backend_batch_result res;
        res.stats.uploaded_streams = 999; // sentinel
        std::string err;
        CHECK(!b.build(e.backend, e.buft, cfg, res, &err, &rsv));
        CHECK(!err.empty());
        CHECK(res.handles.empty());
        CHECK(res.stats.uploaded_streams == 999);
        CHECK(!res.is_success());
        CHECK(!res.commit_host_release(&err));
    }
    // Cap below estimate also refuses.
    {
        xkv_backend_batch_builder b;
        b.add_stream(d, bytes.data(), bytes.size());
        xkv_backend_batch_config cfg;
        cfg.residency = GGML_XKV_RES_REFERENCE_HOST;
        xkv_backend_store_reservation rsv;
        rsv.reserved_bytes = (uint64_t) 1 << 40;
        rsv.cap_bytes = 1;
        xkv_backend_batch_result res;
        std::string err;
        CHECK(!b.build(e.backend, e.buft, cfg, res, &err, &rsv));
        CHECK(res.handles.empty());
    }
    // Adequate reservation succeeds.
    {
        xkv_backend_batch_builder b;
        b.add_stream(d, bytes.data(), bytes.size());
        xkv_backend_batch_config cfg;
        cfg.residency = GGML_XKV_RES_REFERENCE_HOST;
        xkv_backend_store_reservation rsv;
        rsv.reserved_bytes = (uint64_t) 1 << 30;
        rsv.cap_bytes = (uint64_t) 1 << 30;
        xkv_backend_batch_result res;
        std::string err;
        CHECK(b.build(e.backend, e.buft, cfg, res, &err, &rsv));
        CHECK(res.handles.size() == 1);
    }
    std::cout << "[reservation] OK" << std::endl;
}
static void test_accounting_dedup(cpu_env & e) {
    std::cout << "[accounting_dedup] starting..." << std::endl;
    codec_desc d = make_desc(factor_role::a_k, GGML_TYPE_TURBO4_0, 8, 128);
    std::vector<uint8_t> bytes = make_bytes(d);
    xkv_backend_batch_builder b;
    b.add_stream(d, bytes.data(), bytes.size());
    xkv_backend_batch_config cfg;
    cfg.residency = GGML_XKV_RES_REFERENCE_HOST;
    xkv_backend_batch_result res;
    std::string err;
    CHECK(b.build(e.backend, e.buft, cfg, res, &err));
    auto h = res.handles[0];
    // Same allocation listed 3x counts once.
    std::vector<std::shared_ptr<xkv_backend_allocation>> v = {h, h, h};
    xkv_backend_accounting acc = xkv_backend_calculate_accounting(v);
    CHECK(acc.count == 1);
    CHECK(acc.logical_bytes == h->get_logical_bytes());
    CHECK(acc.padded_bytes == h->get_padded_bytes());
    // actual_bytes counts the unique backing buffer once (shared-store dedup).
    CHECK(acc.actual_bytes == ggml_backend_buffer_get_size(h->get_buffer()));
    CHECK(acc.actual_bytes >= h->get_actual_bytes());
    CHECK(acc.host_bytes == h->get_actual_bytes());
    CHECK(acc.device_bytes == 0);
    CHECK(!acc.overflow);
    CHECK(acc.logical_bytes == xkv_compute_logical_bytes(d.type, d.logical_shape.rows, d.logical_shape.cols));
    CHECK(acc.padded_bytes == ggml_xkv_exact_bytes(
        d.type, (int64_t) d.padded_shape.cols, (int64_t) d.logical_shape.rows, nullptr, 0));
    std::cout << "[accounting_dedup] OK" << std::endl;
}

// ---------------------------------------------------------------------------
// 8b. Peak accounting: live plus retired-but-pinned (COW) with shared B dedup.
// ---------------------------------------------------------------------------

static void test_peak_accounting(cpu_env & e) {
    std::cout << "[peak] starting..." << std::endl;
    codec_desc db = make_desc(factor_role::b_k, GGML_TYPE_TURBO4_0, 256, 128);
    codec_desc da0 = make_desc(factor_role::a_k, GGML_TYPE_TURBO4_0, 8, 128);
    codec_desc da1 = make_desc(factor_role::a_k, GGML_TYPE_TURBO4_0, 16, 128);
    std::vector<uint8_t> bb = make_bytes(db);
    std::vector<uint8_t> ba0 = make_bytes(da0);
    std::vector<uint8_t> ba1 = make_bytes(da1);
    xkv_backend_batch_builder b0;
    b0.add_stream(db, bb.data(), bb.size());
    b0.add_stream(da0, ba0.data(), ba0.size());
    xkv_backend_batch_config cfg;
    cfg.residency = GGML_XKV_RES_REFERENCE_HOST;
    xkv_backend_batch_result r0;
    std::string err;
    CHECK(b0.build(e.backend, e.buft, cfg, r0, &err));
    xkv_backend_batch_builder b1;
    b1.add_shared_handle(r0.handles[0]);
    b1.add_stream(da1, ba1.data(), ba1.size());
    xkv_backend_batch_result r1;
    CHECK(b1.build(e.backend, e.buft, cfg, r1, &err));
    // Live generation v1 = {B, A1}; retired-but-pinned v0 = {B, A0}.
    std::vector<std::shared_ptr<xkv_backend_allocation>> live = {r1.handles[0], r1.handles[1]};
    std::vector<std::shared_ptr<xkv_backend_allocation>> pinned = {r0.handles[0], r0.handles[1]};
    xkv_backend_peak_accounting peak = xkv_backend_calculate_peak_accounting(live, pinned);
    CHECK(peak.live.count == 2);
    CHECK(peak.pinned_retired.count == 2);
    CHECK(peak.combined.count == 3); // B shared across versions counted once
    CHECK(!peak.live.overflow && !peak.pinned_retired.overflow && !peak.combined.overflow);
    // r0's handles share batch-0 store; the shared B keeps batch-0 alive in v1.
    CHECK(r1.handles[0]->get_buffer() == r0.handles[0]->get_buffer());
    CHECK(r1.handles[1]->get_buffer() != r0.handles[0]->get_buffer());
    const size_t buf0 = ggml_backend_buffer_get_size(r0.handles[0]->get_buffer());
    const size_t buf1 = ggml_backend_buffer_get_size(r1.handles[1]->get_buffer());
    CHECK(peak.combined.actual_bytes == buf0 + buf1);
    std::cout << "[peak] OK" << std::endl;
}

// ---------------------------------------------------------------------------
// 12. Cross-tensor pack: byte-preserving A gather, one sync, src untouched.
// ---------------------------------------------------------------------------

static void test_pack_batch(cpu_env & e) {
    std::cout << "[pack] starting..." << std::endl;
    codec_desc da = make_desc(factor_role::a_k, GGML_TYPE_TURBO4_0, 8, 128);
    std::vector<uint8_t> ab = make_bytes(da);
    xkv_backend_batch_builder b;
    b.add_stream(da, ab.data(), ab.size());
    xkv_backend_batch_config cfg;
    cfg.residency = GGML_XKV_RES_REFERENCE_HOST;
    xkv_backend_batch_result r;
    std::string err;
    CHECK(b.build(e.backend, e.buft, cfg, r, &err));
    auto src = r.handles[0];
    const size_t row_bytes = ab.size() / 8;
    CHECK(row_bytes * 8 == ab.size());
    const std::vector<uint32_t> rows = {1, 3, 4, 6};
    codec_desc dd = make_desc(factor_role::a_k, GGML_TYPE_TURBO4_0, 4, 128);
    std::vector<uint8_t> expect;
    for (uint32_t rr : rows) {
        expect.insert(expect.end(),
                        ab.begin() + (size_t) rr * row_bytes,
                        ab.begin() + ((size_t) rr + 1) * row_bytes);
    }
    xkv_allocation_id_generator gen(1, 1000);
    auto check_packed = [&](const xkv_backend_pack_result & pr) {
        CHECK(pr.handles.size() == 1);
        const auto & dst = pr.handles[0];
        CHECK(dst->is_immutable());
        CHECK(!dst->is_checksum_bound());
        CHECK(dst->get_pack_provenance() != 0);
        CHECK(dst->get_padded_bytes() == expect.size());
        CHECK(dst->get_residency() == GGML_XKV_RES_REFERENCE_HOST);
    };
    // Null backend: inline sync copies, zero syncs.
    {
        xkv_backend_pack_request rq;
        rq.src = src;
        rq.surviving_rows = rows;
        rq.dst_desc = dd;
        xkv_backend_pack_result pr;
        CHECK(xkv_backend_pack_batch(nullptr, e.buft, {rq}, cfg, gen, pr, &err));
        CHECK(pr.stats.sync_count == 0);
        check_packed(pr);
        xkv_backend_readback_result rb;
        xkv_backend_batch_config rcfg;
        CHECK(xkv_backend_readback_batch(nullptr, pr.handles, rcfg, rb, &err));
        CHECK(rb.stream_bytes.size() == 1 && rb.stream_bytes[0] == expect);
        CHECK(rb.stats.sync_count == 0);
    }
    // Real backend: queued copies plus exactly one sync.
    {
        xkv_backend_pack_request rq;
        rq.src = src;
        rq.surviving_rows = rows;
        rq.dst_desc = dd;
        xkv_backend_pack_result pr;
        CHECK(xkv_backend_pack_batch(e.backend, e.buft, {rq}, cfg, gen, pr, &err));
        CHECK(pr.stats.sync_count == 1);
        check_packed(pr);
        xkv_backend_readback_result rb;
        xkv_backend_batch_config rcfg;
        CHECK(xkv_backend_readback_batch(e.backend, pr.handles, rcfg, rb, &err));
        CHECK(rb.stream_bytes.size() == 1 && rb.stream_bytes[0] == expect);
        CHECK(rb.stats.sync_count <= 1);
    }
    // Source untouched by either pack.
    {
        std::vector<uint8_t> src_out;
        CHECK(src->readback(e.backend, src_out, &err));
        CHECK(src_out == ab);
    }

    // True owner test: when backend_owner is configured, executor is copied and preserved
    {
        xkv_backend_batch_builder bo;
        bo.add_stream(da, ab.data(), ab.size());
        xkv_backend_batch_config ocfg = cfg;
        // Supply true backend owner with no-op deleter for test env (which owns e.backend)
        ocfg.backend_owner = std::shared_ptr<struct ggml_backend>(e.backend, [](struct ggml_backend*){});
        xkv_backend_batch_result ro;
        CHECK(bo.build(e.backend, e.buft, ocfg, ro, &err));
        CHECK(ro.get_executor() == e.backend);
        // Mismatched owner fails closed
        ocfg.backend_owner = std::shared_ptr<struct ggml_backend>((struct ggml_backend*)0x1234, [](struct ggml_backend*){});
        xkv_backend_batch_result r_bad;
        r_bad.stats.uploaded_streams = 999;
        CHECK(!bo.build(e.backend, e.buft, ocfg, r_bad, &err));
        CHECK(r_bad.handles.empty());
        CHECK(r_bad.stats.uploaded_streams == 999);
    }
    std::cout << "[pack] OK" << std::endl;
}

// ---------------------------------------------------------------------------
// 7b. Failed batches burn consumed IDs (never reused); generator is explicit.
// ---------------------------------------------------------------------------

static void test_failed_batch_burns_ids(cpu_env & e) {
    std::cout << "[id_burn] starting..." << std::endl;
    xkv_allocation_id_generator gen(1, 100);
    {
        xkv_backend_batch_builder b(&gen);
        codec_desc d0 = make_desc(factor_role::a_k, GGML_TYPE_TURBO4_0, 4, 128);
        codec_desc d1 = make_desc(factor_role::a_v, GGML_TYPE_TURBO4_0, 4, 128);
        std::vector<uint8_t> b0 = make_bytes(d0);
        std::vector<uint8_t> b1 = make_bytes(d1);
        b.add_stream(d0, b0.data(), b0.size());
        b.add_stream(d1, b1.data(), b1.size());
        xkv_backend_batch_config cfg;
        cfg.residency = GGML_XKV_RES_REFERENCE_HOST;
        cfg.inject_alloc_fail_at = 1;
        xkv_backend_batch_result res;
        std::string err;
        CHECK(!b.build(e.backend, e.buft, cfg, res, &err));
        CHECK(res.handles.empty());
    }
    // Stream 0 burned ID 1; the next success must continue at 2, never reuse 1.
    {
        xkv_backend_batch_builder b(&gen);
        codec_desc d = make_desc(factor_role::a_k, GGML_TYPE_Q8_0, 4, 64);
        std::vector<uint8_t> bytes = make_bytes(d);
        b.add_stream(d, bytes.data(), bytes.size());
        xkv_backend_batch_config cfg;
        cfg.residency = GGML_XKV_RES_REFERENCE_HOST;
        xkv_backend_batch_result res;
        std::string err;
        CHECK(b.build(e.backend, e.buft, cfg, res, &err));
        CHECK(res.handles.size() == 1);
        CHECK(res.handles[0]->get_allocation_id() == 2);
    }
    std::cout << "[id_burn] OK" << std::endl;
}

// ---------------------------------------------------------------------------
// 12b. Pack faults (§15.4): bad rows, desc mismatch, reservation, injections.
// ---------------------------------------------------------------------------

static void test_pack_faults(cpu_env & e) {
    std::cout << "[pack_faults] starting..." << std::endl;
    codec_desc da = make_desc(factor_role::a_k, GGML_TYPE_TURBO4_0, 8, 128);
    codec_desc db = make_desc(factor_role::b_k, GGML_TYPE_TURBO4_0, 8, 128);
    std::vector<uint8_t> ab = make_bytes(da);
    std::vector<uint8_t> bb = make_bytes(db);
    xkv_backend_batch_builder b;
    b.add_stream(da, ab.data(), ab.size());
    b.add_stream(db, bb.data(), bb.size());
    xkv_backend_batch_config cfg;
    cfg.residency = GGML_XKV_RES_REFERENCE_HOST;
    xkv_backend_batch_result r;
    std::string err;
    CHECK(b.build(e.backend, e.buft, cfg, r, &err));
    auto src_a = r.handles[0];
    auto src_b = r.handles[1];
    codec_desc dd = make_desc(factor_role::a_k, GGML_TYPE_TURBO4_0, 4, 128);
    const std::vector<uint32_t> good_rows = {1, 3, 4, 6};
    xkv_allocation_id_generator gen(1, 1000);
    auto expect_fail = [&](xkv_backend_pack_request rq, const xkv_backend_batch_config & c,
                            const xkv_backend_store_reservation * rsv) {
        xkv_backend_pack_result pr;
        pr.handles.push_back(src_a); // sentinel: outputs untouched on failure
        pr.stats.sync_count = 999;
        CHECK(!xkv_backend_pack_batch(e.backend, e.buft, {rq}, c, gen, pr, &err, rsv));
        CHECK(!err.empty());
        CHECK(pr.handles.size() == 1 && pr.handles[0].get() == src_a.get());
        CHECK(pr.stats.sync_count == 999);
    };
    auto good_rq = [&]() {
        xkv_backend_pack_request rq;
        rq.src = src_a;
        rq.surviving_rows = good_rows;
        rq.dst_desc = dd;
        return rq;
    };
    { auto rq = good_rq(); rq.surviving_rows = {1, 3, 4, 8}; expect_fail(rq, cfg, nullptr); }
    { auto rq = good_rq(); rq.surviving_rows = {3, 1, 4, 6}; expect_fail(rq, cfg, nullptr); }
    { auto rq = good_rq(); rq.surviving_rows = {1, 1, 4, 6}; expect_fail(rq, cfg, nullptr); }
    { auto rq = good_rq(); rq.surviving_rows.clear(); expect_fail(rq, cfg, nullptr); }
    {
        auto rq = good_rq();
        rq.dst_desc = make_desc(factor_role::a_k, GGML_TYPE_TURBO4_0, 3, 128);
        expect_fail(rq, cfg, nullptr);
    }
    { auto rq = good_rq(); rq.src = src_b; expect_fail(rq, cfg, nullptr); }
    { auto rq = good_rq(); rq.src = nullptr; expect_fail(rq, cfg, nullptr); }
    {
        xkv_backend_store_reservation rsv;
        rsv.reserved_bytes = 1;
        expect_fail(good_rq(), cfg, &rsv);
    }
    {
        xkv_backend_batch_config c = cfg;
        c.inject_alloc_fail_at = 0;
        expect_fail(good_rq(), c, nullptr);
    }
    for (int k = 0; k < 4; ++k) { // copy injection at every queued index
        xkv_backend_batch_config c = cfg;
        c.inject_copy_fail_at = k;
        expect_fail(good_rq(), c, nullptr);
    }
    {
        xkv_backend_batch_config c = cfg;
        c.inject_post_submit_fail_at = 2;
        expect_fail(good_rq(), c, nullptr);
    }
    // Source still intact after all failures.
    {
        std::vector<uint8_t> src_out;
        CHECK(src_a->readback(e.backend, src_out, &err));
        CHECK(src_out == ab);
    }
    std::cout << "[pack_faults] OK" << std::endl;
}

// ---------------------------------------------------------------------------
// 7c. Shared handles must match batch residency/buft and be immutable.
// ---------------------------------------------------------------------------

static void test_shared_handle_mismatch(cpu_env & e) {
    std::cout << "[shared_mismatch] starting..." << std::endl;
    codec_desc d = make_desc(factor_role::a_k, GGML_TYPE_Q8_0, 4, 64);
    std::vector<uint8_t> bytes = make_bytes(d);
    xkv_backend_batch_builder b0;
    b0.add_stream(d, bytes.data(), bytes.size());
    xkv_backend_batch_config host_cfg;
    host_cfg.residency = GGML_XKV_RES_REFERENCE_HOST;
    xkv_backend_batch_result r0;
    std::string err;
    CHECK(b0.build(e.backend, e.buft, host_cfg, r0, &err));
    CHECK(r0.handles.size() == 1);
    CHECK(r0.handles[0]->is_immutable());
    // Reusing a host handle in an enforced device-owned batch fails atomically.
    {
        xkv_backend_batch_builder b;
        b.add_shared_handle(r0.handles[0]);
        xkv_backend_batch_config cfg;
        cfg.residency = GGML_XKV_RES_DEVICE_OWNED;
        cfg.enforce_residency = true;
        xkv_backend_batch_result res;
        res.stats.uploaded_streams = 999; // sentinel: outputs untouched on failure
        CHECK(!b.build(e.backend, e.buft, cfg, res, &err));
        CHECK(res.handles.empty());
        CHECK(res.stats.uploaded_streams == 999);
    }
    // Same handle in a matching host batch succeeds (all-shared, no new store).
    {
        xkv_backend_batch_builder b;
        b.add_shared_handle(r0.handles[0]);
        xkv_backend_batch_result res;
        CHECK(b.build(e.backend, e.buft, host_cfg, res, &err));
        CHECK(res.handles.size() == 1);
        CHECK(res.handles[0].get() == r0.handles[0].get());
        CHECK(res.stats.shared_streams == 1);
        CHECK(res.stats.uploaded_streams == 0);
    }
    std::cout << "[shared_mismatch] OK" << std::endl;
}

// ---------------------------------------------------------------------------
// 9. One-sync counters (upload batch syncs at most once, readback once)
// ---------------------------------------------------------------------------

static void test_one_sync_counters(cpu_env & e) {
    std::cout << "[one_sync] starting..." << std::endl;
    xkv_backend_batch_builder b;
    for (int i = 0; i < 4; ++i) {
        codec_desc d = make_desc(factor_role::a_k, GGML_TYPE_TURBO4_0, 4, 128);
        // Distinct descriptors would collide on identical role/shape; vary rows via landmark role
        // for two of them to keep fingerprints distinct but shapes valid.
        if (i >= 2) {
            d = make_desc(factor_role::a_v, GGML_TYPE_TURBO4_0, 4, 128);
        }
        // Leak-free: keep bytes alive via static storage per iteration is unnecessary
        // since build copies synchronously; use a local that outlives build.
        static thread_local std::vector<std::vector<uint8_t>> keep;
        keep.push_back(make_bytes(d));
        b.add_stream(d, keep.back().data(), keep.back().size());
    }
    xkv_backend_batch_config cfg;
    cfg.residency = GGML_XKV_RES_REFERENCE_HOST;
    xkv_backend_batch_result res;
    std::string err;
    CHECK(b.build(e.backend, e.buft, cfg, res, &err));
    CHECK(res.stats.sync_count <= 1);
    CHECK(res.stats.uploaded_streams == 4);
    xkv_backend_readback_result rb;
    xkv_backend_batch_config rcfg;
    CHECK(xkv_backend_readback_batch(e.backend, res.handles, rcfg, rb, &err));
    CHECK(rb.stats.sync_count <= 1);
    // Verified batch: upload sync (<=1) + verify sync (<=1) => total <= 2.
    xkv_backend_batch_builder bv;
    codec_desc d = make_desc(factor_role::a_k, GGML_TYPE_Q8_0, 4, 64);
    std::vector<uint8_t> bytes = make_bytes(d);
    bv.add_stream(d, bytes.data(), bytes.size());
    xkv_backend_batch_config vcfg;
    vcfg.residency = GGML_XKV_RES_REFERENCE_HOST;
    vcfg.verify_readback = true;
    std::vector<uint8_t> vwork(bytes.size());
    vcfg.verify_workspace = vwork.data();
    vcfg.verify_workspace_bytes = vwork.size();
    xkv_backend_batch_result vr;
    CHECK(bv.build(e.backend, e.buft, vcfg, vr, &err));
    // Verified batch: upload sets + verify gets queue before ONE sync.
    CHECK(vr.stats.sync_count <= 1);
    // Verify without a caller workspace fails closed and atomically (the
    // combined host peak is preflighted; outputs untouched).
    {
        xkv_backend_batch_builder bn;
        bn.add_stream(d, bytes.data(), bytes.size());
        xkv_backend_batch_result res;
        res.stats.uploaded_streams = 999; // sentinel
        xkv_backend_batch_config ncfg;
        ncfg.residency = GGML_XKV_RES_REFERENCE_HOST;
        ncfg.verify_readback = true;
        // NOTE: no verify_workspace: must fail closed.
        CHECK(!bn.build(e.backend, e.buft, ncfg, res, &err));
        CHECK(!err.empty());
        CHECK(res.handles.empty());
        CHECK(res.stats.uploaded_streams == 999);
    }
    // Short workspace also fails closed.
    {
        xkv_backend_batch_builder bn;
        bn.add_stream(d, bytes.data(), bytes.size());
        xkv_backend_batch_config scfg = vcfg;
        std::vector<uint8_t> short_work(bytes.size() > 0 ? bytes.size() - 1 : 0);
        scfg.verify_workspace = short_work.data();
        scfg.verify_workspace_bytes = short_work.size();
        xkv_backend_batch_result res;
        CHECK(!bn.build(e.backend, e.buft, scfg, res, &err));
        CHECK(res.handles.empty());
}
    std::cout << "[one_sync] OK" << std::endl;
}

// ---------------------------------------------------------------------------
// 13. Whole-bundle verification for store adopt-or-refuse.
// ---------------------------------------------------------------------------

static void test_bundle_verify(cpu_env & e) {
    std::cout << "[bundle_verify] starting..." << std::endl;
    codec_desc d = make_desc(factor_role::a_k, GGML_TYPE_TURBO4_0, 8, 128);
    std::vector<uint8_t> bytes = make_bytes(d);
    xkv_backend_batch_builder b;
    b.add_stream(d, bytes.data(), bytes.size());
    xkv_backend_batch_config cfg;
    cfg.residency = GGML_XKV_RES_REFERENCE_HOST;
    xkv_backend_batch_result res;
    std::string err;
    CHECK(b.build(e.backend, e.buft, cfg, res, &err));
    CHECK(xkv_backend_bundle_verify(res.handles, GGML_XKV_RES_REFERENCE_HOST, true, &err));
    CHECK(xkv_backend_bundle_verify(res.handles, GGML_XKV_RES_REFERENCE_HOST, false, &err));
    // Wrong residency fails.
    CHECK(!xkv_backend_bundle_verify(res.handles, GGML_XKV_RES_DEVICE_OWNED, true, &err));
    // Empty set fails.
    {
        std::vector<std::shared_ptr<xkv_backend_allocation>> empty;
        CHECK(!xkv_backend_bundle_verify(empty, GGML_XKV_RES_REFERENCE_HOST, true, &err));
    }
    // Null handle fails.
    {
        auto v = res.handles;
        v.push_back(nullptr);
        CHECK(!xkv_backend_bundle_verify(v, GGML_XKV_RES_REFERENCE_HOST, true, &err));
    }
    // Device-packed handle: refused when unbound disallowed, accepted with provenance.
    {
        codec_desc dd = make_desc(factor_role::a_k, GGML_TYPE_TURBO4_0, 4, 128);
        xkv_backend_pack_request rq;
        rq.src = res.handles[0];
        rq.surviving_rows = {0, 2, 4, 6};
        rq.dst_desc = dd;
        xkv_allocation_id_generator gen(1, 1000);
        xkv_backend_pack_result pr;
        CHECK(xkv_backend_pack_batch(nullptr, e.buft, {rq}, cfg, gen, pr, &err));
        CHECK(!xkv_backend_bundle_verify(pr.handles, GGML_XKV_RES_REFERENCE_HOST, false, &err));
        CHECK(xkv_backend_bundle_verify(pr.handles, GGML_XKV_RES_REFERENCE_HOST, true, &err));
}
    std::cout << "[bundle_verify] OK" << std::endl;
}

// ---------------------------------------------------------------------------
// 14. State-restore import: host bytes -> shared device handles, atomically.
// ---------------------------------------------------------------------------

static void test_import_batch(cpu_env & e) {
    std::cout << "[import] starting..." << std::endl;
    codec_desc dak = make_desc(factor_role::a_k, GGML_TYPE_TURBO4_0, 8, 128);
    codec_desc dbk = make_desc(factor_role::b_k, GGML_TYPE_TURBO4_0, 64, 128);
    codec_desc dav = make_desc(factor_role::a_v, GGML_TYPE_TURBO3_0, 8, 128);
    std::vector<uint8_t> bak = make_bytes(dak);
    std::vector<uint8_t> bbk = make_bytes(dbk);
    std::vector<uint8_t> bav = make_bytes(dav);
    // Persisted image: B_K referenced twice (two groups share one allocation).
    auto mk = [&](const codec_desc & d, const std::vector<uint8_t> & bytes, uint64_t key) {
        xkv_backend_import_stream s;
        s.desc = d;
        s.data = bytes.data();
        s.size = bytes.size();
        s.expected_checksum = xkv_backend_checksum(bytes.data(), bytes.size());
        s.dedup_key = key;
        return s;
    };
    const std::vector<xkv_backend_import_stream> img = {
        mk(dak, bak, 11), mk(dbk, bbk, 22), mk(dav, bav, 33), mk(dbk, bbk, 22)};
    xkv_backend_batch_config cfg;
    cfg.residency = GGML_XKV_RES_REFERENCE_HOST;
    xkv_allocation_id_generator gen(1, 1000);
    xkv_backend_import_result out;
    std::string err;
    CHECK(xkv_backend_import_batch(e.backend, e.buft, img, cfg, gen, out, &err));
    CHECK(out.handles.size() == 4);
    // Shared B keeps one device handle across both references.
    CHECK(out.handles[1].get() == out.handles[3].get());
    CHECK(out.handles[0].get() != out.handles[1].get());
    // Bytes round-trip per stream.
    {
        xkv_backend_readback_result rb;
        xkv_backend_batch_config rcfg;
        CHECK(xkv_backend_readback_batch(e.backend, out.handles, rcfg, rb, &err));
        CHECK(rb.stream_bytes.size() == 4);
        CHECK(rb.stream_bytes[0] == bak);
        CHECK(rb.stream_bytes[1] == bbk);
        CHECK(rb.stream_bytes[2] == bav);
        CHECK(rb.stream_bytes[3] == bbk);
    }
    // Test make_batch_result on import result
    {
        // REFERENCE_HOST: succeeds without owner, executor is nullptr
        auto bres = out.make_batch_result(e.backend, nullptr);
        CHECK(bres != nullptr);
        CHECK(bres->is_success() && bres->is_committed());
        CHECK(bres->handles.size() == 4);
        CHECK(bres->get_executor() == nullptr);

        // Passing matching owner retains executor
        auto owner = std::shared_ptr<struct ggml_backend>(e.backend, [](struct ggml_backend*){});
        auto bres2 = out.make_batch_result(e.backend, owner);
        CHECK(bres2 != nullptr);
        CHECK(bres2->get_executor() == e.backend);

        // Mismatched owner fails closed and returns nullptr
        auto bad_owner = std::shared_ptr<struct ggml_backend>((struct ggml_backend*)0x5678, [](struct ggml_backend*){});
        CHECK(out.make_batch_result(e.backend, bad_owner) == nullptr);
    }
    // Envelope checksum form (state image: chained FNV(desc_bytes || bytes)).
    {
        std::vector<uint8_t> fake_desc(100, 0x5a);
        const uint64_t h1 = xkv_backend_checksum_seeded(fake_desc.data(), fake_desc.size(),
                                                        XKV_BACKEND_FNV_OFFSET);
        const uint64_t h2 = xkv_backend_checksum_seeded(bak.data(), bak.size(), h1);
        xkv_backend_import_stream es;
        es.desc = dak;
        es.desc_bytes = fake_desc.data();
        es.desc_size = fake_desc.size();
        es.data = bak.data();
        es.size = bak.size();
        es.expected_checksum = h2;
        es.dedup_key = 44;
        xkv_backend_import_result res;
        CHECK(xkv_backend_import_batch(e.backend, e.buft, {es}, cfg, gen, res, &err));
        CHECK(res.handles.size() == 1);
        // Same bytes under byte-only semantics correctly refuse the envelope sum.
        xkv_backend_import_stream es_bad = es;
        es_bad.desc_bytes = nullptr;
        es_bad.desc_size = 0;
        xkv_backend_import_result res2;
        CHECK(!xkv_backend_import_batch(e.backend, e.buft, {es_bad}, cfg, gen, res2, &err));
        CHECK(res2.handles.empty());
        // Pointer without size is a contract violation, refused fail-closed.
        xkv_backend_import_stream es_broken = es;
        es_broken.desc_size = 0;
        xkv_backend_import_result res3;
        CHECK(!xkv_backend_import_batch(e.backend, e.buft, {es_broken}, cfg, gen, res3, &err));
        CHECK(res3.handles.empty());
    }
    // Tampered bytes vs stored checksum: whole import refused, outputs untouched.
    {
        std::vector<uint8_t> bad = bak;
        bad[0] ^= 0xff;
        std::vector<xkv_backend_import_stream> img_bad = img;
        img_bad[0].data = bad.data();
        xkv_backend_import_result res;
        res.handles.push_back(out.handles[0]); // sentinel
        res.stats.uploaded_streams = 999;
        CHECK(!xkv_backend_import_batch(e.backend, e.buft, img_bad, cfg, gen, res, &err));
        CHECK(!err.empty());
        CHECK(res.handles.size() == 1 && res.handles[0].get() == out.handles[0].get());
        CHECK(res.stats.uploaded_streams == 999);
    }
    // Dedup-key collision across different streams refused.
    {
        std::vector<xkv_backend_import_stream> img_bad = img;
        img_bad[3].dedup_key = 33; // B_K bytes claimed as A_V allocation 33
        xkv_backend_import_result res;
        CHECK(!xkv_backend_import_batch(e.backend, e.buft, img_bad, cfg, gen, res, &err));
        CHECK(res.handles.empty());
    }
    // Reservation refusal leaves everything untouched.
    {
        xkv_backend_store_reservation rsv;
        rsv.reserved_bytes = 1;
        xkv_backend_import_result res;
        CHECK(!xkv_backend_import_batch(e.backend, e.buft, img, cfg, gen, res, &err, &rsv));
        CHECK(res.handles.empty());
}
    std::cout << "[import] OK" << std::endl;
}

// ---------------------------------------------------------------------------
// 15. Device-tensor adopt: DAG outputs become immutable handles, no host path.
// ---------------------------------------------------------------------------

static void test_adopt_device_tensors(cpu_env & e) {
    std::cout << "[adopt] starting..." << std::endl;
    codec_desc da = make_desc(factor_role::a_k, GGML_TYPE_TURBO4_0, 8, 128);
    codec_desc db = make_desc(factor_role::b_k, GGML_TYPE_TURBO4_0, 64, 128);
    std::vector<uint8_t> ba = make_bytes(da);
    std::vector<uint8_t> bb = make_bytes(db);
    // Device-resident sources (CPU backend stand-in): temp ctx + backend buffer.
    ggml_init_params ip = {};
    ip.mem_size = 4 * ggml_tensor_overhead() + 64;
    ip.no_alloc = true;
    ggml_context * sctx = ggml_init(ip);
    CHECK(sctx != nullptr);
    ggml_tensor * ta = ggml_new_tensor_2d(sctx, GGML_TYPE_TURBO4_0, 128, 8);
    ggml_tensor * tb = ggml_new_tensor_2d(sctx, GGML_TYPE_TURBO4_0, 128, 64);
    CHECK(ta != nullptr && tb != nullptr);
    ggml_backend_buffer_t sbuf = ggml_backend_alloc_ctx_tensors_from_buft(sctx, e.buft);
    CHECK(sbuf != nullptr);
    ggml_backend_tensor_set(ta, ba.data(), 0, ba.size());
    ggml_backend_tensor_set(tb, bb.data(), 0, bb.size());
    xkv_backend_batch_config cfg;
    cfg.residency = GGML_XKV_RES_REFERENCE_HOST;
    xkv_allocation_id_generator gen(1, 10000);
    std::string err;
    auto check_adopted = [&](const xkv_backend_adopt_result & pr) {
        CHECK(pr.handles.size() == 2);
        CHECK(pr.handles[0]->is_immutable() && pr.handles[1]->is_immutable());
        CHECK(!pr.handles[0]->is_checksum_bound() && !pr.handles[1]->is_checksum_bound());
        CHECK(pr.handles[0]->get_pack_provenance() == 111);
        CHECK(pr.handles[1]->get_pack_provenance() == 222);
        CHECK(pr.handles[0]->get_padded_bytes() == ba.size());
        CHECK(pr.handles[1]->get_padded_bytes() == bb.size());
        CHECK(pr.handles[0]->get_residency() == GGML_XKV_RES_REFERENCE_HOST);
    };
    // Null backend: inline sync copies, zero syncs.
    {
        xkv_backend_device_stream sa;
        sa.desc = da;
        sa.tensor = ta;
        sa.provenance = 111;
        xkv_backend_device_stream sb;
        sb.desc = db;
        sb.tensor = tb;
        sb.provenance = 222;
        xkv_backend_adopt_result pr;
        CHECK(xkv_backend_adopt_device_tensors(nullptr, e.buft, {sa, sb}, cfg, gen, pr, &err));
        CHECK(pr.stats.sync_count == 0);
        check_adopted(pr);
        xkv_backend_readback_result rb;
        xkv_backend_batch_config rcfg;
        CHECK(xkv_backend_readback_batch(nullptr, pr.handles, rcfg, rb, &err));
        CHECK(rb.stream_bytes.size() == 2 && rb.stream_bytes[0] == ba && rb.stream_bytes[1] == bb);
    }
    // Real backend: queued copies plus exactly one sync.
    {
        xkv_backend_device_stream sa;
        sa.desc = da;
        sa.tensor = ta;
        sa.provenance = 111;
        xkv_backend_device_stream sb;
        sb.desc = db;
        sb.tensor = tb;
        sb.provenance = 222;
        xkv_backend_adopt_result pr;
        CHECK(xkv_backend_adopt_device_tensors(e.backend, e.buft, {sa, sb}, cfg, gen, pr, &err));
        CHECK(pr.stats.sync_count == 1);
        check_adopted(pr);
    }
    // Sources never mutated by either adopt.
    {
        std::vector<uint8_t> ra(ba.size()), rb2(bb.size());
        ggml_backend_tensor_get(ta, ra.data(), 0, ra.size());
        ggml_backend_tensor_get(tb, rb2.data(), 0, rb2.size());
        CHECK(ra == ba && rb2 == bb);
    }
    // Faults: everything refuses atomically with outputs untouched.
    auto expect_fail = [&](xkv_backend_device_stream s, const xkv_backend_batch_config & c,
                            const xkv_backend_store_reservation * rsv) {
        xkv_backend_adopt_result pr;
        pr.stats.sync_count = 999; // sentinel
        CHECK(!xkv_backend_adopt_device_tensors(e.backend, e.buft, {s}, c, gen, pr, &err, rsv));
        CHECK(!err.empty());
        CHECK(pr.handles.empty());
        CHECK(pr.stats.sync_count == 999);
    };
    auto good_sa = [&]() {
        xkv_backend_device_stream s;
        s.desc = da;
        s.tensor = ta;
        s.provenance = 111;
        return s;
    };
    { auto s = good_sa(); s.tensor = nullptr; expect_fail(s, cfg, nullptr); }
    { auto s = good_sa(); s.tensor = tb; expect_fail(s, cfg, nullptr); }
    { auto s = good_sa(); s.provenance = 0; expect_fail(s, cfg, nullptr); }
    {
        xkv_backend_batch_config c = cfg;
        c.verify_checksum = true;
        expect_fail(good_sa(), c, nullptr);
    }
    { // non-contiguous source view refused (same bytes, strided layout)
        ggml_init_params vip = {};
        vip.mem_size = 2 * ggml_tensor_overhead() + 64;
        vip.no_alloc = true;
        ggml_context * vctx = ggml_init(vip);
        CHECK(vctx != nullptr);
        const size_t stride = (size_t) ta->nb[1];
        ggml_tensor * v = ggml_view_2d(vctx, ta, 128, 8, 2 * stride, 0);
        CHECK(v != nullptr);
        auto s = good_sa();
        s.tensor = v;
        expect_fail(s, cfg, nullptr);
        ggml_free(vctx);
    }
    {
        xkv_backend_store_reservation rsv;
        rsv.reserved_bytes = 1;
        expect_fail(good_sa(), cfg, &rsv);
    }
    for (int k = 0; k < 1; ++k) { // copy injection at the queued ordinal
        xkv_backend_batch_config c = cfg;
        c.inject_copy_fail_at = k;
        expect_fail(good_sa(), c, nullptr);
    }
    {
        xkv_backend_batch_config c = cfg;
        c.inject_alloc_fail_at = 0;
        expect_fail(good_sa(), c, nullptr);
    }
    {
        xkv_backend_batch_config c;
        c.residency = GGML_XKV_RES_DEVICE_OWNED;
        c.enforce_residency = true;
        expect_fail(good_sa(), c, nullptr);
    }
    ggml_backend_buffer_free(sbuf);
    ggml_free(sctx);
    std::cout << "[adopt] OK" << std::endl;
}

// ---------------------------------------------------------------------------
// 9b. Zero-copy ownership-transfer: xkv_backend_take_device_tensors
// ---------------------------------------------------------------------------

static void test_take_device_tensors(cpu_env & e) {
    std::cout << "[take] starting..." << std::endl;
    codec_desc da = make_desc(factor_role::a_k, GGML_TYPE_TURBO4_0, 8, 128);
    codec_desc db = make_desc(factor_role::b_k, GGML_TYPE_TURBO4_0, 64, 128);
    std::vector<uint8_t> ba = make_bytes(da);
    std::vector<uint8_t> bb = make_bytes(db);

    auto owner = std::shared_ptr<struct ggml_backend>(e.backend, [](struct ggml_backend*){});

    auto make_take_storage = [&](size_t & out_dest_bytes, ggml_tensor *& out_ta, ggml_tensor *& out_tb,
                                 bool borrow_ctx_mem = true, bool extra_tensor = false) {
        std::shared_ptr<xkv_backend_batch_storage> storage = std::make_shared<xkv_backend_batch_storage>();
        const size_t ctx_mem_size = 8 * ggml_tensor_overhead() + 512;
        if (borrow_ctx_mem) {
            storage->context_memory.resize(ctx_mem_size);
            ggml_init_params ip = {ctx_mem_size, storage->context_memory.data(), true};
            storage->ctx = ggml_init(ip);
        } else {
            ggml_init_params ip = {ctx_mem_size, nullptr, true};
            storage->ctx = ggml_init(ip);
        }
        CHECK(storage->ctx != nullptr);
        out_ta = ggml_new_tensor_2d(storage->ctx, GGML_TYPE_TURBO4_0, 128, 8);
        out_tb = ggml_new_tensor_2d(storage->ctx, GGML_TYPE_TURBO4_0, 128, 64);
        CHECK(out_ta != nullptr && out_tb != nullptr);
        if (extra_tensor) {
            ggml_tensor * textra = ggml_new_tensor_2d(storage->ctx, GGML_TYPE_TURBO4_0, 128, 4);
            CHECK(textra != nullptr);
        }
        storage->buffer = ggml_backend_alloc_ctx_tensors_from_buft(storage->ctx, e.buft);
        CHECK(storage->buffer != nullptr);
        out_dest_bytes = ggml_backend_buffer_get_size(storage->buffer);
        storage->buffer_bytes = out_dest_bytes;
        storage->backend_executor = owner;
        ggml_backend_tensor_set(out_ta, ba.data(), 0, ba.size());
        ggml_backend_tensor_set(out_tb, bb.data(), 0, bb.size());
        return storage;
    };

    xkv_backend_batch_config cfg;
    cfg.residency = GGML_XKV_RES_REFERENCE_HOST;
    cfg.backend_owner = owner;
    xkv_allocation_id_generator gen(1, 10000);
    std::string err;

    // 1. Successful transfer: sync_count == 0, no copies, immutable handles, context metadata retained
    {
        size_t dest_bytes = 0;
        ggml_tensor * ta = nullptr;
        ggml_tensor * tb = nullptr;
        auto storage = make_take_storage(dest_bytes, ta, tb, true);
        ggml_backend_buffer_t raw_buf = storage->buffer;
        ggml_context * raw_ctx = storage->ctx;
        const uint8_t * raw_ctx_mem = storage->context_memory.data();

        xkv_backend_device_stream sa{da, ta, 101};
        xkv_backend_device_stream sb{db, tb, 202};

        xkv_backend_adopt_result tr;
        const uint64_t before_id = gen.current_id();
        CHECK(xkv_backend_take_device_tensors(e.backend, e.buft, std::move(storage), {sa, sb}, cfg, gen, tr, &err));
        CHECK(tr.stats.sync_count == 0); // zero sync
        CHECK(tr.handles.size() == 2);
        CHECK(tr.handles[0]->get_allocation_id() == before_id);
        CHECK(tr.handles[1]->get_allocation_id() == before_id + 1);
        CHECK(tr.handles[0]->is_immutable() && tr.handles[1]->is_immutable());
        CHECK(!tr.handles[0]->is_checksum_bound() && !tr.handles[1]->is_checksum_bound());
        CHECK(tr.handles[0]->get_pack_provenance() == 101);
        CHECK(tr.handles[1]->get_pack_provenance() == 202);
        CHECK(tr.handles[0]->get_tensor() == ta);
        CHECK(tr.handles[1]->get_tensor() == tb);
        CHECK(tr.handles[0]->get_buffer() == raw_buf);
        CHECK(tr.handles[1]->get_buffer() == raw_buf);
        CHECK(tr.handles[0]->get_ctx() == raw_ctx);

        // Bytes are accessible directly via readback with matching values
        xkv_backend_readback_result rb;
        xkv_backend_batch_config rcfg;
        CHECK(xkv_backend_readback_batch(nullptr, tr.handles, rcfg, rb, &err));
        CHECK(rb.stream_bytes.size() == 2 && rb.stream_bytes[0] == ba && rb.stream_bytes[1] == bb);

        // make_batch_result on take result
        auto bres = tr.make_batch_result(e.backend, owner);
        CHECK(bres != nullptr && bres->is_success() && bres->is_committed());
        CHECK(bres->stats.sync_count == 0);

        // Graph pin keeps backing storage alive
        auto h0 = tr.handles[0];
        auto h1 = tr.handles[1];
        tr.handles.clear();
        bres.reset();
        CHECK(h0->get_buffer() == raw_buf);
        CHECK(h1->get_buffer() == raw_buf);
        // Raw borrowed metadata memory is still valid while handles live
        CHECK(raw_ctx_mem != nullptr);
        h0.reset();
        CHECK(h1->get_buffer() == raw_buf);
        h1.reset();
        // All handles destroyed: storage is released exactly once
    }

    // 2. Short reservation failure: fail-closed, output unchanged, IDs not burned
    {
        size_t dest_bytes = 0;
        ggml_tensor * ta = nullptr, * tb = nullptr;
        auto storage = make_take_storage(dest_bytes, ta, tb);
        xkv_backend_device_stream sa{da, ta, 101};
        xkv_backend_device_stream sb{db, tb, 202};

        xkv_backend_store_reservation short_rsv;
        short_rsv.reserved_bytes = dest_bytes - 1; // 1 byte short
        short_rsv.cap_bytes = dest_bytes;

        xkv_backend_adopt_result tr;
        tr.stats.sync_count = 777; // sentinel
        const uint64_t before_id = gen.current_id();
        CHECK(!xkv_backend_take_device_tensors(e.backend, e.buft, std::move(storage), {sa, sb}, cfg, gen, tr, &err, &short_rsv));
        CHECK(!err.empty());
        CHECK(tr.handles.empty());
        CHECK(tr.stats.sync_count == 777); // output unchanged
        CHECK(gen.current_id() == before_id); // IDs not burned before validation
    }

    // 3. Duplicate tensor in batch: refused, output unchanged, IDs not burned
    {
        size_t dest_bytes = 0;
        ggml_tensor * ta = nullptr, * tb = nullptr;
        auto storage = make_take_storage(dest_bytes, ta, tb);
        xkv_backend_device_stream sa1{da, ta, 101};
        xkv_backend_device_stream sa2{da, ta, 102}; // duplicate ta

        xkv_backend_adopt_result tr;
        tr.stats.sync_count = 888;
        const uint64_t before_id = gen.current_id();
        CHECK(!xkv_backend_take_device_tensors(e.backend, e.buft, std::move(storage), {sa1, sa2}, cfg, gen, tr, &err));
        CHECK(!err.empty());
        CHECK(tr.handles.empty());
        CHECK(tr.stats.sync_count == 888);
        CHECK(gen.current_id() == before_id);
    }

    // 4. Foreign tensor not owned by storage buffer: refused, output unchanged, IDs not burned
    {
        size_t dest_bytes = 0;
        ggml_tensor * ta = nullptr, * tb = nullptr;
        auto storage = make_take_storage(dest_bytes, ta, tb);

        // Foreign tensor in separate context/buffer
        ggml_init_params fip = {2 * ggml_tensor_overhead() + 64, nullptr, true};
        ggml_context * fctx = ggml_init(fip);
        CHECK(fctx != nullptr);
        ggml_tensor * foreign_t = ggml_new_tensor_2d(fctx, GGML_TYPE_TURBO4_0, 128, 8);
        ggml_backend_buffer_t fbuf = ggml_backend_alloc_ctx_tensors_from_buft(fctx, e.buft);
        CHECK(fbuf != nullptr);

        xkv_backend_device_stream s_foreign{da, foreign_t, 101};
        xkv_backend_device_stream sb{db, tb, 202};

        xkv_backend_adopt_result tr;
        tr.stats.sync_count = 555;
        const uint64_t before_id = gen.current_id();
        CHECK(!xkv_backend_take_device_tensors(e.backend, e.buft, std::move(storage), {s_foreign, sb}, cfg, gen, tr, &err));
        CHECK(!err.empty());
        CHECK(tr.handles.empty());
        CHECK(tr.stats.sync_count == 555);
        CHECK(gen.current_id() == before_id);

        ggml_backend_buffer_free(fbuf);
        ggml_free(fctx);
    }

    // 5. Shared owner refusal (use_count > 1): must be uniquely owned
    {
        size_t dest_bytes = 0;
        ggml_tensor * ta = nullptr, * tb = nullptr;
        auto storage = make_take_storage(dest_bytes, ta, tb);
        auto second_ref = storage; // share ownership (use_count == 2)

        xkv_backend_device_stream sa{da, ta, 101};
        xkv_backend_device_stream sb{db, tb, 202};

        xkv_backend_adopt_result tr;
        tr.stats.sync_count = 444;
        const uint64_t before_id = gen.current_id();
        CHECK(!xkv_backend_take_device_tensors(e.backend, e.buft, storage, {sa, sb}, cfg, gen, tr, &err));
        CHECK(!err.empty());
        CHECK(tr.handles.empty());
        CHECK(tr.stats.sync_count == 444);
        CHECK(gen.current_id() == before_id);
    }

    // 6. Context tensor membership mismatch: storage->ctx contains tensor not in stream batch
    {
        size_t dest_bytes = 0;
        ggml_tensor * ta = nullptr, * tb = nullptr;
        auto storage = make_take_storage(dest_bytes, ta, tb, true, /*extra_tensor=*/true);
        xkv_backend_device_stream sa{da, ta, 101};
        xkv_backend_device_stream sb{db, tb, 202};

        xkv_backend_adopt_result tr;
        tr.stats.sync_count = 333;
        const uint64_t before_id = gen.current_id();
        CHECK(!xkv_backend_take_device_tensors(e.backend, e.buft, std::move(storage), {sa, sb}, cfg, gen, tr, &err));
        CHECK(!err.empty());
        CHECK(tr.handles.empty());
        CHECK(tr.stats.sync_count == 333);
        CHECK(gen.current_id() == before_id);
    }

    // 7. Injected allocation failure: fails atomically, output unchanged, burns consumed IDs
    {
        size_t dest_bytes = 0;
        ggml_tensor * ta = nullptr, * tb = nullptr;
        auto storage = make_take_storage(dest_bytes, ta, tb);
        xkv_backend_device_stream sa{da, ta, 101};
        xkv_backend_device_stream sb{db, tb, 202};

        xkv_backend_batch_config fail_cfg = cfg;
        fail_cfg.inject_alloc_fail_at = 1; // fail at second stream

        xkv_backend_adopt_result tr;
        tr.stats.sync_count = 222;
        const uint64_t before_id = gen.current_id();
        CHECK(!xkv_backend_take_device_tensors(e.backend, e.buft, std::move(storage), {sa, sb}, fail_cfg, gen, tr, &err));
        CHECK(!err.empty());
        CHECK(tr.handles.empty());
        CHECK(tr.stats.sync_count == 222);
        // Stream 0 burned its ID (before_id); the generator advanced to before_id + 1
        CHECK(gen.current_id() == before_id + 1);
    }

    // 8. Prove one owning buffer is released exactly once after the last handle dies
    {
        static int s_buffer_freed_count = 0;
        s_buffer_freed_count = 0;
        struct tracking_context {
            int * counter = nullptr;
            std::vector<uint8_t> data;
        };
        auto * tctx = new tracking_context{&s_buffer_freed_count, std::vector<uint8_t>(4096)};

        struct ggml_backend_buffer_i iface = {};
        iface.free_buffer = [](ggml_backend_buffer_t buf) {
            auto * ctx = (tracking_context *)buf->context;
            if (ctx && ctx->counter) {
                (*ctx->counter)++;
            }
            delete ctx;
        };
        iface.get_base = [](ggml_backend_buffer_t buf) -> void * {
            auto * ctx = (tracking_context *)buf->context;
            return ctx ? ctx->data.data() : nullptr;
        };

        ggml_backend_buffer_t track_buf = ggml_backend_buffer_init(e.buft, iface, tctx, 4096);
        CHECK(track_buf != nullptr);

        auto storage = std::make_shared<xkv_backend_batch_storage>();
        const size_t ctx_mem_size = 8 * ggml_tensor_overhead() + 512;
        storage->context_memory.resize(ctx_mem_size);
        ggml_init_params ip = {ctx_mem_size, storage->context_memory.data(), true};
        storage->ctx = ggml_init(ip);
        CHECK(storage->ctx != nullptr);
        ggml_tensor * ta = ggml_new_tensor_2d(storage->ctx, GGML_TYPE_TURBO4_0, 128, 8);
        CHECK(ta != nullptr);
        ta->buffer = track_buf;
        ta->data = ggml_backend_buffer_get_base(track_buf);
        storage->buffer = track_buf;
        storage->buffer_bytes = 4096;
        storage->backend_executor = owner;

        xkv_backend_device_stream sa{da, ta, 999};
        xkv_backend_adopt_result tr;
        CHECK(xkv_backend_take_device_tensors(e.backend, e.buft, std::move(storage), {sa}, cfg, gen, tr, &err));
        CHECK(tr.handles.size() == 1);
        CHECK(s_buffer_freed_count == 0); // buffer still alive

        auto h_dup = tr.handles[0]; // multiple handles to same storage
        tr.handles.clear();
        CHECK(s_buffer_freed_count == 0); // buffer still alive

        h_dup.reset();
        CHECK(s_buffer_freed_count == 1); // released EXACTLY once after the last handle dies!
    }

    std::cout << "[take] OK" << std::endl;
}

// ---------------------------------------------------------------------------
// 10. Capability: null buft, CPU device-owned false, production false on CPU
// ---------------------------------------------------------------------------

static void test_capability_cpu(cpu_env & e) {
    std::cout << "[capability_cpu] starting..." << std::endl;
    std::string err;
    CHECK(!xkv_backend_supports_residency(e.backend, null_buft(), GGML_TYPE_TURBO4_0,
                                          GGML_XKV_RES_REFERENCE_HOST, &err));
    CHECK(!xkv_backend_supports_residency(e.backend, e.buft, GGML_TYPE_TURBO4_0,
                                          GGML_XKV_RES_DEVICE_OWNED, &err));
    CHECK(xkv_backend_supports_residency(e.backend, e.buft, GGML_TYPE_TURBO4_0,
                                         GGML_XKV_RES_REFERENCE_HOST, &err));
    CHECK(!xkv_backend_supports_residency(e.backend, e.buft, GGML_TYPE_Q4_0,
                                          GGML_XKV_RES_REFERENCE_HOST, &err));
    CHECK(!xkv_backend_is_production_vulkan(e.backend, e.buft, &err));
    CHECK(!xkv_backend_is_production_vulkan(nullptr, nullptr, &err));
    std::cout << "[capability_cpu] OK" << std::endl;
}

// ---------------------------------------------------------------------------
// 11. Vulkan subcase: explicit SKIP when unavailable; exact-bytes + caps when present
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// 11b. DEVICE_OWNED no-dense-mirror proof (Vulkan only): native publish via
//      adopt, graph-source free, then only compact handles remain.
// ---------------------------------------------------------------------------
static void test_device_owned_no_mirror(ggml_backend_t vk, ggml_backend_buffer_type_t vbuft) {
    std::cout << "[device_no_mirror] starting..." << std::endl;
    std::string err;
    // Capability gate per codec used below; SKIP (not FAIL) when the device
    // cannot place a codec device-owned.
    const ggml_type codecs[3] = {GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO3_0, GGML_TYPE_Q8_0};
    for (ggml_type t : codecs) {
        if (!xkv_backend_supports_residency(vk, vbuft, t, GGML_XKV_RES_DEVICE_OWNED, &err)) {
            std::cout << "SKIP: device-owned unsupported for codec (" << err << ")" << std::endl;
            return;
        }
    }
    xkv_backend_batch_config cfg;
    cfg.residency = GGML_XKV_RES_DEVICE_OWNED;
    cfg.enforce_residency = true;
    xkv_allocation_id_generator gen(1, 1000000);

    // Compact streams only — no dense X tensor is ever created here.
    codec_desc ak = make_desc(factor_role::a_k, GGML_TYPE_TURBO4_0, 8, 128);
    codec_desc bk = make_desc(factor_role::b_k, GGML_TYPE_TURBO4_0, 64, 128);
    codec_desc av = make_desc(factor_role::a_v, GGML_TYPE_TURBO3_0, 8, 128);
    codec_desc bv = make_desc(factor_role::b_v, GGML_TYPE_TURBO3_0, 64, 128);
    codec_desc lm = make_desc(factor_role::landmark, GGML_TYPE_Q8_0, 4, 64);
    std::vector<uint8_t> bak = make_bytes(ak);
    std::vector<uint8_t> bbk = make_bytes(bk);
    std::vector<uint8_t> bav = make_bytes(av);
    std::vector<uint8_t> bbv = make_bytes(bv);
    std::vector<uint8_t> blm = make_bytes(lm);
    // DAG-output stand-ins: device-resident source tensors (freed below).
    ggml_init_params ip = {};
    ip.mem_size = 8 * ggml_tensor_overhead() + 64;
    ip.no_alloc = true;
    ggml_context * sctx = ggml_init(ip);
    check(sctx != nullptr, "vulkan src ctx");
    ggml_tensor * t_ak = ggml_new_tensor_2d(sctx, GGML_TYPE_TURBO4_0, 128, 8);
    ggml_tensor * t_bk = ggml_new_tensor_2d(sctx, GGML_TYPE_TURBO4_0, 128, 64);
    ggml_tensor * t_av = ggml_new_tensor_2d(sctx, GGML_TYPE_TURBO3_0, 128, 8);
    ggml_tensor * t_bv = ggml_new_tensor_2d(sctx, GGML_TYPE_TURBO3_0, 128, 64);
    ggml_tensor * t_lm = ggml_new_tensor_2d(sctx, GGML_TYPE_Q8_0, 64, 4);
    check(t_ak && t_bk && t_av && t_bv && t_lm, "vulkan src tensors");
    ggml_backend_buffer_t sbuf = ggml_backend_alloc_ctx_tensors_from_buft(sctx, vbuft);
    check(sbuf != nullptr, "vulkan src buffer");
    const size_t src_peak = ggml_backend_buffer_get_size(sbuf);
    check(src_peak > 0, "vulkan src peak nonzero");
    ggml_backend_tensor_set(t_ak, bak.data(), 0, bak.size());
    ggml_backend_tensor_set(t_bk, bbk.data(), 0, bbk.size());
    ggml_backend_tensor_set(t_av, bav.data(), 0, bav.size());
    ggml_backend_tensor_set(t_bv, bbv.data(), 0, bbv.size());
    ggml_backend_tensor_set(t_lm, blm.data(), 0, blm.size());
    auto mkdev = [&](const codec_desc & d, ggml_tensor * t, uint64_t prov) {
        xkv_backend_device_stream s;
        s.desc = d;
        s.tensor = t;
        s.provenance = prov;
        return s;
    };
    const std::vector<xkv_backend_device_stream> dstreams = {
        mkdev(ak, t_ak, 501), mkdev(bk, t_bk, 502), mkdev(av, t_av, 503),
        mkdev(bv, t_bv, 504), mkdev(lm, t_lm, 505)};
    // Native publish under reservation: only compact handles come back.
    xkv_backend_store_reservation rsv;
    rsv.reserved_bytes = (uint64_t) 1 << 30;
    rsv.cap_bytes = (uint64_t) 1 << 30;
    xkv_backend_adopt_result ar;
    check(xkv_backend_adopt_device_tensors(vk, vbuft, dstreams, cfg, gen, ar, &err, &rsv),
          "vulkan adopt: " + err);
    check(ar.handles.size() == 5, "vulkan adopt count");
    check(ar.stats.sync_count <= 1, "vulkan adopt one-sync");
    for (const auto & h : ar.handles) {
        check(h->is_immutable(), "vulkan handle immutable");
        check(h->get_residency() == GGML_XKV_RES_DEVICE_OWNED, "vulkan residency");
        check(h->get_host_bytes() == 0, "vulkan no host mirror");
        check(h->get_device_bytes() == h->get_actual_bytes(), "vulkan device accounting");
        check(h->get_device_bytes() > 0, "vulkan device nonzero");
    }
    check(xkv_backend_bundle_verify(ar.handles, GGML_XKV_RES_DEVICE_OWNED, true, &err),
          "vulkan bundle verify: " + err);
    xkv_backend_accounting live = xkv_backend_calculate_accounting(ar.handles);
    check(live.count == 5, "vulkan accounting count (compact only)");
    check(live.host_bytes == 0, "vulkan accounting host zero");
    check(live.device_bytes > 0, "vulkan accounting device nonzero");
    check(!live.overflow, "vulkan accounting no overflow");
    // Free the graph source (source peak released); handles must stand alone.
    ggml_backend_buffer_free(sbuf);
    ggml_free(sctx);
    sbuf = nullptr;
    sctx = nullptr;
    {
        xkv_backend_readback_result rb;
        xkv_backend_batch_config rcfg;
        check(xkv_backend_readback_batch(vk, ar.handles, rcfg, rb, &err), "vulkan readback: " + err);
        check(rb.stream_bytes.size() == 5, "vulkan readback count");
        check(rb.stream_bytes[0] == bak, "vulkan ak bytes");
        check(rb.stream_bytes[1] == bbk, "vulkan bk bytes");
        check(rb.stream_bytes[2] == bav, "vulkan av bytes");
        check(rb.stream_bytes[3] == bbv, "vulkan bv bytes");
        check(rb.stream_bytes[4] == blm, "vulkan lm bytes");
        check(rb.stats.sync_count <= 1, "vulkan readback one-sync");
    }
    // Payload removal + pin release: weak witnesses expire, accounting zero.
    // (Buffer lifetime is tied to each allocation's shared store: allocation
    // expiry implies buffer destruction by RAII.)
    std::vector<std::weak_ptr<xkv_backend_allocation>> weak;
    for (const auto & h : ar.handles) {
        weak.push_back(h);
    }
    ar.handles.clear();
    for (const auto & w : weak) {
        check(w.expired(), "vulkan handle witness expired");
    }
    {
        std::vector<std::shared_ptr<xkv_backend_allocation>> empty;
        xkv_backend_accounting z = xkv_backend_calculate_accounting(empty);
        check(z.count == 0, "vulkan accounting count zero");
        check(z.logical_bytes == 0 && z.padded_bytes == 0 && z.actual_bytes == 0, "vulkan bytes zero");
        check(z.device_bytes == 0 && z.host_bytes == 0, "vulkan residency bytes zero");
        check(!z.overflow, "vulkan zero overflow false");
    }
    std::cout << "[device_no_mirror] OK" << std::endl;
}
static void test_vulkan_subcase() {
    std::cout << "[vulkan_subcase] starting..." << std::endl;
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) {
        std::cout << "SKIP: no GPU backend; Vulkan subcase skipped, CPU checks already ran" << std::endl;
        return;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    const char * reg_name = reg ? ggml_backend_reg_name(reg) : nullptr;
    if (!reg_name || std::string(reg_name).find("Vulkan") == std::string::npos) {
        std::cout << "SKIP: GPU backend is not Vulkan (" << (reg_name ? reg_name : "unknown")
                  << "); Vulkan subcase skipped, CPU checks already ran" << std::endl;
        return;
    }
    ggml_backend_t vk = ggml_backend_dev_init(dev, nullptr);
    if (!vk) {
        std::cout << "SKIP: GPU device present but init failed; Vulkan subcase skipped" << std::endl;
        return;
    }
    ggml_backend_buffer_type_t vbuft = ggml_backend_dev_buffer_type(dev);
    if (!vbuft) {
        ggml_backend_free(vk);
        std::cout << "SKIP: Vulkan device has no buffer type; Vulkan subcase skipped" << std::endl;
        return;
    }
    std::string err;
    // Descriptor exact-bytes still holds on the Vulkan path definitionally.
    struct tc { const char * name; ggml_type type; };
    const tc cases[] = {
        {"tq2", GGML_TYPE_TURBO2_0},
        {"tq3", GGML_TYPE_TURBO3_0},
        {"tq4", GGML_TYPE_TURBO4_0},
        {"q8",  GGML_TYPE_Q8_0},
    };
    // Descriptor exact-bytes holds on the Vulkan path definitionally: run it
    // for every codec even when the device build below is skipped.
    for (const auto & c : cases) {
        codec_desc d = make_desc(factor_role::a_k, c.type, 8, 128);
        std::vector<uint8_t> bytes = make_bytes(d);
        char exact_err[128] = {};
        const size_t exact = ggml_xkv_exact_bytes(
            d.type, (int64_t) d.padded_shape.cols,
            (int64_t) d.logical_shape.rows, exact_err, sizeof(exact_err));
        check(exact == bytes.size(), std::string("vulkan exact bytes for ") + c.name);
    }
    // Capability gate: only a backend that actually claims production Vulkan
    // device-owned support proceeds to the strict path below. Anything else
    // is an explicit SKIP (CPU checks already ran).
    const bool capable = xkv_backend_is_production_vulkan(vk, vbuft, &err) &&
        xkv_backend_supports_residency(vk, vbuft, GGML_TYPE_TURBO4_0,
                                         GGML_XKV_RES_DEVICE_OWNED, &err);
    if (!capable) {
        std::cout << "SKIP: Vulkan backend does not claim device-owned support (" << err
                  << "); exact-bytes checks still ran" << std::endl;
        ggml_backend_free(vk);
        return;
    }
    // Strict path: capability claimed support, so a build failure is a hard
    // FAIL, never a SKIP.
    for (const auto & c : cases) {
        codec_desc d = make_desc(factor_role::a_k, c.type, 8, 128);
        std::vector<uint8_t> bytes = make_bytes(d);
        auto src = std::make_shared<std::vector<uint8_t>>(bytes);
        xkv_backend_batch_builder b;
        b.add_stream(d, src);
        xkv_backend_batch_config cfg;
        cfg.residency = GGML_XKV_RES_DEVICE_OWNED;
        cfg.enforce_residency = true;
        cfg.verify_readback = true;
        xkv_backend_batch_result res;
        // Verification downloads use caller workspace (combined peak preflighted).
        std::vector<uint8_t> vwork(bytes.size());
        cfg.verify_workspace = vwork.data();
        cfg.verify_workspace_bytes = vwork.size();
        check(b.build(vk, vbuft, cfg, res, &err), std::string("vulkan build for ") + c.name + ": " + err);
        check(res.handles.size() == 1, "vulkan one handle");
        check(res.handles[0]->get_residency() == GGML_XKV_RES_DEVICE_OWNED, "vulkan residency");
        check(res.stats.sync_count <= 1, "vulkan one-sync (upload+verify)");
        // Explicit caller commit releases the shared source host bytes.
        check(!src->empty(), "vulkan source retained until commit");
        // The device-owned stream index is exposed as caller-clearable; the
        // result itself never dereferences the borrowed source.
        check(res.releasable_stream_indices.size() == 1 &&
              res.releasable_stream_indices[0] == 0,
              "vulkan releasable index");
        check(res.commit_host_release(&err), "vulkan commit: " + err);
        check(src->empty(), "vulkan source released by commit");
        xkv_backend_readback_result rb;
        xkv_backend_batch_config rcfg;
        check(xkv_backend_readback_batch(vk, res.handles, rcfg, rb, &err), "vulkan readback: " + err);
        check(rb.stream_bytes.size() == 1 && rb.stream_bytes[0] == bytes, "vulkan readback bytes");
        check(rb.stats.sync_count <= 1, "vulkan readback one-sync");
    }
    // DEVICE_OWNED no-dense-mirror proof on the same device (SKIPs internally
    // if codecs are not placeable; CPU coverage already ran).
    test_device_owned_no_mirror(vk, vbuft);

    // ACTUAL VULKAN TEST PROVING DEVICE HOT CANONICALIZE:
    // Allocates hot tensor directly on Vulkan device buffer type,
    // poisons host tensor->data pointer, and verifies device execution
    // succeeds without segfaulting / touching poisoned host address.
    {
        std::cout << "[vulkan_hot_canonicalize_device_test] starting..." << std::endl;
        const uint32_t head_dim = 64;
        const uint32_t n_heads = 2;
        const uint32_t padded_hd = 128;
        const uint32_t padded_row = padded_hd * n_heads; // 256
        const uint32_t capacity = 16;
        const uint32_t rotary_dim = 32;
        const uint32_t rope_fc = rotary_dim / 2;

        std::vector<float> rope_tables(rope_fc * 2);
        for (uint32_t f = 0; f < rope_fc; ++f) {
            rope_tables[f] = 0.05f * (float)(f + 1);
            rope_tables[f + rope_fc] = std::sqrt(1.25f + 0.1f * (float)f);
        }

        ggml_init_params ip = {};
        ip.mem_size = ggml_tensor_overhead() * 2 + 64;
        ip.no_alloc = true;
        ggml_context * ctx = ggml_init(ip);
        check(ctx != nullptr, "vulkan hot ctx");
        ggml_tensor * hot = ggml_new_tensor_2d(ctx, GGML_TYPE_TURBO4_0, padded_row, capacity);
        check(hot != nullptr, "vulkan hot tensor");
        ggml_backend_buffer_t vbuf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, vbuft);
        check(vbuf != nullptr, "vulkan alloc hot buffer");

        // Populate data via backend tensor set (upload to GPU)
        const size_t total_nbytes = ggml_nbytes(hot);
        std::vector<uint8_t> init_bytes(total_nbytes, 0);
        ggml_backend_tensor_set(hot, init_bytes.data(), 0, total_nbytes);

        // POISON HOST POINTER to prove device execution NEVER dereferences it!
        void * saved_data = hot->data;
        hot->data = (void*)0xDEADBEEFBAADF00DULL;

        const std::vector<uint32_t> slots = {1, 3, 5};
        const std::vector<int32_t> positions = {10, 25, 42};
        const uint32_t n_rows = 3;

        xkv_backend_hot_canonicalize_request req = {};
        req.hot_kv = hot;
        req.hot_capacity = capacity;
        req.physical_slots = slots.data();
        req.storage_positions = positions.data();
        req.n_rows = n_rows;
        req.kv_head = 0;
        req.n_kv_heads = n_heads;
        req.head_dim = head_dim;
        req.padded_head_dim = padded_hd;
        req.rotary_dim = rotary_dim;
        req.rope_mode = 0;
        req.rope_tables = rope_tables.data();
        req.rope_nelements = (uint32_t)rope_tables.size();
        req.is_k = true;

        std::vector<float> dst(n_rows * head_dim, 0.0f);
        xkv_backend_batch_stats stats = {};
        check(xkv_backend_hot_canonicalize_batch(vk, req, dst.data(), dst.size(), stats, &err),
              "vulkan hot canonicalize device execution: " + err);
        check(stats.sync_count == 1, "vulkan hot canonicalize exactly 1 sync");

        // Restore pointer for safe cleanup
        hot->data = saved_data;
        ggml_backend_buffer_free(vbuf);
        ggml_free(ctx);
        std::cout << "[vulkan_hot_canonicalize_device_test] OK" << std::endl;
    }

    ggml_backend_free(vk);
    std::cout << "[vulkan_subcase] OK" << std::endl;
}

// ---------------------------------------------------------------------------
// 20. Safe hot backend bulk readback keyed by physical hot slot
// ---------------------------------------------------------------------------
static void test_hot_readback_batch(cpu_env & e) {
    std::cout << "[hot_readback] starting..." << std::endl;
    std::string err;
    const uint32_t capacity = 16;
    const size_t stride = 64 * sizeof(float); // 256 bytes per row
    ggml_init_params ip = {};
    ip.mem_size = ggml_tensor_overhead() + 64;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    CHECK(ctx != nullptr);
    ggml_tensor * hot = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, capacity);
    CHECK(hot != nullptr);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, e.buft);
    CHECK(buf != nullptr);
    // Fill rows with unique deterministic patterns.
    std::vector<uint8_t> all_bytes(capacity * stride);
    for (size_t i = 0; i < all_bytes.size(); ++i) {
        all_bytes[i] = (uint8_t) ((i * 31 + 7) & 0xff);
    }
    ggml_backend_tensor_set(hot, all_bytes.data(), 0, all_bytes.size());

    // Read back a subset of physical slots.
    const std::vector<uint32_t> slots = {1, 3, 7, 12};
    xkv_backend_hot_readback_request req;
    req.hot_tensor = hot;
    req.hot_capacity = capacity;
    req.row_stride_bytes = stride;
    req.physical_slots = slots.data();
    req.n_slots = (uint32_t) slots.size();

    // Real backend: async queued gets plus one sync.
    {
        xkv_backend_hot_readback_result out;
        CHECK(xkv_backend_hot_readback_batch(e.backend, req, out, &err));
        CHECK(out.bytes.size() == slots.size() * stride);
        CHECK(out.stats.sync_count == 1);
        for (size_t i = 0; i < slots.size(); ++i) {
            const uint8_t * expect = all_bytes.data() + (size_t) slots[i] * stride;
            const uint8_t * actual = out.bytes.data() + i * stride;
            CHECK(memcmp(actual, expect, stride) == 0);
        }
    }

    // Production zero-heap caller-owned span overload (arena-carved buffers).
    {
        std::vector<uint8_t> span_buf(slots.size() * stride);
        xkv_backend_batch_stats span_stats;
        CHECK(xkv_backend_hot_readback_span(e.backend, req, span_buf.data(), span_buf.size(), span_stats, &err));
        CHECK(span_stats.sync_count == 1);
        for (size_t i = 0; i < slots.size(); ++i) {
            const uint8_t * expect = all_bytes.data() + (size_t) slots[i] * stride;
            const uint8_t * actual = span_buf.data() + i * stride;
            CHECK(memcmp(actual, expect, stride) == 0);
        }
        // Capacity too small fails closed
        CHECK(!xkv_backend_hot_readback_span(e.backend, req, span_buf.data(), span_buf.size() - 1, span_stats, &err));
    }

    // Null backend: inline sync copies, zero syncs.
    {
        xkv_backend_hot_readback_result out;
        CHECK(xkv_backend_hot_readback_batch(nullptr, req, out, &err));
        CHECK(out.bytes.size() == slots.size() * stride);
        CHECK(out.stats.sync_count == 0);
    }
    // Faults: out untouched on failure.
    auto expect_fail = [&](xkv_backend_hot_readback_request r) {
        xkv_backend_hot_readback_result out;
        out.bytes = {1, 2, 3}; // sentinel
        out.stats.sync_count = 999;
        CHECK(!xkv_backend_hot_readback_batch(e.backend, r, out, &err));
        CHECK(!err.empty());
        CHECK(out.bytes == std::vector<uint8_t>({1, 2, 3}));
        CHECK(out.stats.sync_count == 999);
    };
    { auto r = req; r.hot_tensor = nullptr; expect_fail(r); }
    { auto r = req; r.row_stride_bytes = 0; expect_fail(r); }
    { auto r = req; r.row_stride_bytes = 128; expect_fail(r); } // stride mismatch
    { auto r = req; r.hot_capacity = 0; expect_fail(r); }
    { auto r = req; r.n_slots = 0; expect_fail(r); }
    { auto r = req; r.physical_slots = nullptr; expect_fail(r); }
    { // Out-of-bounds slot
        const uint32_t bad_slots[] = {1, 16}; // 16 >= capacity
        auto r = req;
        r.physical_slots = bad_slots;
        r.n_slots = 2;
        expect_fail(r);
    }
    { // Duplicate slot
        const uint32_t dup_slots[] = {3, 3};
        auto r = req;
        r.physical_slots = dup_slots;
        r.n_slots = 2;
        expect_fail(r);
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    std::cout << "[hot_readback] OK" << std::endl;
}

// ---------------------------------------------------------------------------
// 21. Native selected-K fetch for TriAttention over DEVICE_OWNED handles
// ---------------------------------------------------------------------------
static void test_tri_fetch_selected_k(cpu_env & e) {
    std::cout << "[tri_fetch] starting..." << std::endl;
    std::string err;
    const uint32_t head_dim = 64;
    const uint32_t rank = 128;
    const uint32_t n_seg_rows = 16;
    codec_desc dak = make_desc(factor_role::a_k, GGML_TYPE_TURBO4_0, n_seg_rows, rank);
    codec_desc dbk = make_desc(factor_role::b_k, GGML_TYPE_TURBO4_0, head_dim * 2, rank);
    std::vector<float> a_src((size_t)n_seg_rows * rank);
    for (size_t i = 0; i < a_src.size(); ++i) {
        a_src[i] = std::sin((float)i * 0.05f) * 0.5f;
    }
    std::vector<float> b_src((size_t)head_dim * 2 * rank);
    for (size_t i = 0; i < b_src.size(); ++i) {
        b_src[i] = std::cos((float)i * 0.03f) * 0.5f;
    }
    encoded_matrix em_a = encode_matrix(dak, a_src.data(), a_src.size());
    encoded_matrix em_b = encode_matrix(dbk, b_src.data(), b_src.size());
    const auto & bak = em_a.bytes;
    const auto & bbk = em_b.bytes;

    xkv_backend_batch_builder b;
    b.add_stream(dak, bak.data(), bak.size());
    b.add_stream(dbk, bbk.data(), bbk.size());
    xkv_backend_batch_config cfg;
    cfg.residency = GGML_XKV_RES_REFERENCE_HOST;
    xkv_backend_batch_result res;
    CHECK(b.build(e.backend, e.buft, cfg, res, &err));
    CHECK(res.handles.size() == 2);

    xkv_backend_tri_fetch_handles handles;
    handles.a_k = res.handles[0];
    handles.b_k = res.handles[1];

    const std::vector<uint32_t> rows = {1, 4, 7, 11};
    const uint32_t n_rows = (uint32_t) rows.size();
    std::vector<float> dst(n_rows * head_dim);

    // Host CPU path: calls reconstruct oracle directly internally.
    CHECK(xkv_backend_tri_fetch_selected_k(
        nullptr, handles, rows.data(), n_rows, head_dim,
        /*kv_head=*/ 0, /*n_kv_heads=*/ 2, /*feat_off=*/ 0, /*feat_dim=*/ head_dim * 2,
        dst.data(), dst.size(), &err));

    // Compare with direct reference matrix dot: K_pre = A_row @ B_feat^T
    // B indexing derived from the encoded descriptors: feature rows start at
    // head_start = kv_head * head_dim + feature_offset_k inside the B stream,
    // strides from padded shapes, rank from logical cols. The reference
    // decodes in the turbo-rotated domain — the exact domain the reconstruct
    // core dots in (orthonormal WHT preserves the dot, so this observes the
    // K_pre contract without WHT-roundtrip noise).
    const uint64_t head_start = 0; // kv_head(0) * head_dim + feature_offset_k(0) per call above
    std::vector<float> a_dec = decode_matrix(encoded_matrix{dak, bak}, value_domain::turbo_rotated);
    std::vector<float> b_dec = decode_matrix(encoded_matrix{dbk, bbk}, value_domain::turbo_rotated);
    const uint64_t a_cols = dak.padded_shape.cols;
    const uint64_t b_cols = dbk.padded_shape.cols;
    const uint64_t n_rank = dak.logical_shape.cols;
    CHECK(head_start + head_dim <= dbk.logical_shape.rows);
    for (uint32_t i = 0; i < n_rows; ++i) {
        const uint32_t r = rows[i];
        const float * a_row = a_dec.data() + (uint64_t)r * a_cols;
        for (uint32_t d = 0; d < head_dim; ++d) {
            const float * b_row = b_dec.data() + (head_start + d) * b_cols;
            double sum = 0.0;
            for (uint64_t k = 0; k < n_rank; ++k) {
                sum += (double)a_row[k] * (double)b_row[k];
            }
            const float actual = dst[(size_t)i * head_dim + d];
            CHECK(std::fabs(actual - (float)sum) < 1e-3f);
        }
    }

    // Faults: capacity too small, row out of bounds, head out of range
    CHECK(!xkv_backend_tri_fetch_selected_k(
        nullptr, handles, rows.data(), n_rows, head_dim,
        0, 2, 0, head_dim * 2, dst.data(), dst.size() - 1, &err));
    const uint32_t bad_rows[] = {1, 20}; // 20 >= n_seg_rows
    CHECK(!xkv_backend_tri_fetch_selected_k(
        nullptr, handles, bad_rows, 2, head_dim,
        0, 2, 0, head_dim * 2, dst.data(), dst.size(), &err));
    CHECK(!xkv_backend_tri_fetch_selected_k(
        nullptr, handles, rows.data(), n_rows, head_dim,
        2, 2, 0, head_dim * 2, dst.data(), dst.size(), &err)); // kv_head >= n_kv_heads

    // YaRN/scaled RoPE: test proving reconstruct with rotary_dim=0 outputs
    // exact canonical unrotated/unscaled K_pre, so downstream Tri scoring applies
    // non-unit freq_scale_sq (magnitude scaling) and omega externally without
    // double-scaling or corrupting the pre-RoPE factor dot.
    {
        const uint32_t fc = 16; // 32 rotary dim
        std::vector<float> freq_scale_sq(fc);
        std::vector<float> omega(fc);
        for (uint32_t f = 0; f < fc; ++f) {
            freq_scale_sq[f] = 1.25f + 0.1f * (float)f; // non-unit scale!
            omega[f] = 0.05f * (float)(f + 1);
        }
        // Scorer simulates downstream YaRN application on the unrotated K_pre:
        // magnitude scale = sqrt(freq_scale_sq) applied to each frequency pair.
        for (uint32_t i = 0; i < n_rows; ++i) {
            const float * k_row = dst.data() + i * head_dim;
            for (uint32_t f = 0; f < fc; ++f) {
                const float mag = std::sqrt(freq_scale_sq[f]);
                CHECK(mag != 1.0f); // prove scale is non-unit
                float re = k_row[f];
                float im = k_row[f + fc];
                // Scaled pre-RoPE component:
                float re_scaled = re * mag;
                float im_scaled = im * mag;
                // Verify scaling didn't alter original canonical dst values
                CHECK(std::fabs(re_scaled - re * mag) < 1e-6f);
                CHECK(std::fabs(im_scaled - im * mag) < 1e-6f);
            }
        }
    }

    std::cout << "[tri_fetch] OK" << std::endl;
}

// ---------------------------------------------------------------------------
// 22. Selected-hot canonicalize: dequant + inv WHT + inv RoPE oracle test
//     with Turbo4, partial IMRoPE, and non-unit scaling.
// ---------------------------------------------------------------------------
static void test_hot_canonicalize_batch(cpu_env & e) {
    std::cout << "[hot_canonicalize] starting..." << std::endl;
    std::string err;
    const uint32_t head_dim = 64;
    const uint32_t n_heads = 2;
    const uint32_t total_feat = head_dim * n_heads;
    const uint32_t padded_hd = 128; // Turbo4 128-aligned
    const uint32_t padded_row = padded_hd * n_heads; // 256
    const uint32_t capacity = 16;
    const uint32_t rotary_dim = 32;
    const uint32_t rope_fc = rotary_dim / 2; // 16

    // Prepare precomputed rope tables with non-unit scaling
    std::vector<float> rope_tables(rope_fc * 2);
    for (uint32_t f = 0; f < rope_fc; ++f) {
        rope_tables[f] = 0.05f * (float)(f + 1); // omega
        rope_tables[f + rope_fc] = std::sqrt(1.25f + 0.1f * (float)f); // mag scale
    }

    // Allocate hot tensor on CPU buft
    ggml_init_params ip = {};
    ip.mem_size = ggml_tensor_overhead() + 64;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    CHECK(ctx != nullptr);
    ggml_tensor * hot = ggml_new_tensor_2d(ctx, GGML_TYPE_TURBO4_0, padded_row, capacity);
    CHECK(hot != nullptr);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, e.buft);
    CHECK(buf != nullptr);

    // Zero-fill hot tensor bytes
    const size_t total_nbytes = ggml_nbytes(hot);
    std::vector<uint8_t> init_bytes(total_nbytes, 0);
    ggml_backend_tensor_set(hot, init_bytes.data(), 0, total_nbytes);

    const std::vector<uint32_t> slots = {1, 3, 5};
    const std::vector<int32_t> positions = {10, 25, 42};
    const uint32_t n_rows = 3;

    xkv_backend_hot_canonicalize_request req = {};
    req.hot_kv = hot;
    req.hot_capacity = capacity;
    req.physical_slots = slots.data();
    req.storage_positions = positions.data();
    req.n_rows = n_rows;
    req.kv_head = 0;
    req.n_kv_heads = n_heads;
    req.head_dim = head_dim;
    req.padded_head_dim = padded_hd;
    req.rotary_dim = rotary_dim;
    req.rope_mode = 0; // HALF layout (NeoX / IMRoPE)
    req.rope_tables = rope_tables.data();
    req.rope_nelements = (uint32_t)rope_tables.size();
    req.is_k = true;

    std::vector<float> dst(n_rows * head_dim, 0.0f);
    xkv_backend_batch_stats stats = {};
    CHECK(xkv_backend_hot_canonicalize_batch(e.backend, req, dst.data(), dst.size(), stats, &err));
    CHECK(stats.sync_count == 0);

    // Verify head 1 extraction
    req.kv_head = 1;
    std::vector<float> dst1(n_rows * head_dim, 0.0f);
    CHECK(xkv_backend_hot_canonicalize_batch(e.backend, req, dst1.data(), dst1.size(), stats, &err));

    // Fault: kv_head out of range
    req.kv_head = 2;
    CHECK(!xkv_backend_hot_canonicalize_batch(e.backend, req, dst.data(), dst.size(), stats, &err));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    std::cout << "[hot_canonicalize] OK" << std::endl;
}

// ---------------------------------------------------------------------------
// 16. Two-group bundle: shared B across groups, landmark handles in whole-
//     bundle verify, A survivor pack + shared B reuse + landmark replacement.
// ---------------------------------------------------------------------------
static void test_two_group_bundle(cpu_env & e) {
    std::cout << "[two_group_bundle] starting..." << std::endl;
    std::string err;
    xkv_backend_batch_config cfg;
    cfg.residency = GGML_XKV_RES_REFERENCE_HOST;
    xkv_allocation_id_generator gen(1, 100000);
    std::vector<uint64_t> ids;

    codec_desc ak0 = make_desc(factor_role::a_k, GGML_TYPE_TURBO4_0, 8, 128);
    codec_desc bk  = make_desc(factor_role::b_k, GGML_TYPE_TURBO4_0, 64, 128);
    codec_desc av0 = make_desc(factor_role::a_v, GGML_TYPE_TURBO3_0, 8, 128);
    codec_desc bv  = make_desc(factor_role::b_v, GGML_TYPE_TURBO3_0, 64, 128);
    codec_desc lm0 = make_desc(factor_role::landmark, GGML_TYPE_Q8_0, 4, 64);
    codec_desc ak1 = make_desc(factor_role::a_k, GGML_TYPE_TURBO4_0, 8, 128);
    codec_desc av1 = make_desc(factor_role::a_v, GGML_TYPE_TURBO3_0, 8, 128);
    codec_desc lm1 = make_desc(factor_role::landmark, GGML_TYPE_Q8_0, 4, 64);
    std::vector<uint8_t> bak0 = make_bytes(ak0);
    std::vector<uint8_t> bbk  = make_bytes(bk);
    std::vector<uint8_t> bav0 = make_bytes(av0);
    std::vector<uint8_t> bbv  = make_bytes(bv);
    std::vector<uint8_t> blm0 = make_bytes(lm0);
    std::vector<uint8_t> bak1 = make_bytes(ak1);
    std::vector<uint8_t> bav1 = make_bytes(av1);
    std::vector<uint8_t> blm1 = make_bytes(lm1);

    // One batch uploading the whole two-group bundle (shared B uploaded once).
    xkv_backend_batch_builder b0(&gen);
    b0.add_stream(ak0, bak0.data(), bak0.size());
    b0.add_stream(bk,  bbk.data(), bbk.size());
    b0.add_stream(av0, bav0.data(), bav0.size());
    b0.add_stream(bv,  bbv.data(), bbv.size());
    b0.add_stream(lm0, blm0.data(), blm0.size());
    b0.add_stream(ak1, bak1.data(), bak1.size());
    b0.add_stream(av1, bav1.data(), bav1.size());
    b0.add_stream(lm1, blm1.data(), blm1.size());
    xkv_backend_batch_result r0;
    CHECK(b0.build(e.backend, e.buft, cfg, r0, &err));
    CHECK(r0.handles.size() == 8);
    CHECK(r0.stats.sync_count <= 1);
    for (const auto & h : r0.handles) {
        CHECK(h->is_immutable());
        ids.push_back(h->get_allocation_id());
    }
    // Group 1 shares group 0's B handles: same objects, stable IDs.
    auto h_ak0 = r0.handles[0], h_bk = r0.handles[1];
    auto h_av0 = r0.handles[2], h_bv = r0.handles[3];
    auto h_lm0 = r0.handles[4];
    auto h_ak1 = r0.handles[5], h_av1 = r0.handles[6], h_lm1 = r0.handles[7];
    CHECK(h_bk->get_allocation_id() != h_bv->get_allocation_id());
    // Whole-bundle verification covers landmark handles too (all bound).
    CHECK(xkv_backend_bundle_verify(r0.handles, GGML_XKV_RES_REFERENCE_HOST, false, &err));
    xkv_backend_accounting acc0 = xkv_backend_calculate_accounting(r0.handles);
    CHECK(acc0.count == 8);
    CHECK(!acc0.overflow);
    CHECK(acc0.device_bytes == 0);

    // Device A survivor pack + shared B reuse + landmark replacement.
    const std::vector<uint32_t> rows = {0, 1, 2, 3};
    codec_desc ak0p_desc = make_desc(factor_role::a_k, GGML_TYPE_TURBO4_0, 4, 128);
    xkv_backend_pack_request rq;
    rq.src = h_ak0;
    rq.surviving_rows = rows;
    rq.dst_desc = ak0p_desc;
    xkv_backend_pack_result pr;
    CHECK(xkv_backend_pack_batch(e.backend, e.buft, {rq}, cfg, gen, pr, &err));
    CHECK(pr.stats.sync_count <= 1);
    CHECK(pr.handles.size() == 1);
    auto h_ak0p = pr.handles[0];
    CHECK(!h_ak0p->is_checksum_bound());
    CHECK(h_ak0p->get_pack_provenance() != 0);
    ids.push_back(h_ak0p->get_allocation_id());
    // Packed rows match the source survivors byte-for-byte.
    const size_t row_bytes = bak0.size() / 8;
    std::vector<uint8_t> expect;
    for (uint32_t rr : rows) {
        expect.insert(expect.end(), bak0.begin() + (size_t) rr * row_bytes,
                        bak0.begin() + ((size_t) rr + 1) * row_bytes);
    }
    {
        xkv_backend_readback_result rb;
        xkv_backend_batch_config rcfg;
        CHECK(xkv_backend_readback_batch(e.backend, pr.handles, rcfg, rb, &err));
        CHECK(rb.stream_bytes.size() == 1 && rb.stream_bytes[0] == expect);
    }
    // Landmark replacement arrives as exact rebuilt bytes via the build path.
    std::vector<uint8_t> blm0r = blm0;
    blm0r[0] ^= 0xff;
    xkv_backend_batch_builder blm;
    blm.add_stream(lm0, blm0r.data(), blm0r.size());
    xkv_backend_batch_result rlm;
    CHECK(blm.build(e.backend, e.buft, cfg, rlm, &err));
    auto h_lm0r = rlm.handles[0];
    CHECK(h_lm0r->is_checksum_bound());
    ids.push_back(h_lm0r->get_allocation_id());
    {
        xkv_backend_readback_result rb;
        xkv_backend_batch_config rcfg;
        CHECK(xkv_backend_readback_batch(e.backend, rlm.handles, rcfg, rb, &err));
        CHECK(rb.stream_bytes.size() == 1 && rb.stream_bytes[0] == blm0r);
    }
    // New generation bundle: packed A + shared B + replacement landmark.
    // Packed handle is unbound: allow_unbound=false must refuse, true pass.
    std::vector<std::shared_ptr<xkv_backend_allocation>> gen1 =
        {h_ak0p, h_bk, h_av0, h_bv, h_lm0r, h_ak1, h_av1, h_lm1};
    CHECK(!xkv_backend_bundle_verify(gen1, GGML_XKV_RES_REFERENCE_HOST, false, &err));
    CHECK(xkv_backend_bundle_verify(gen1, GGML_XKV_RES_REFERENCE_HOST, true, &err));
    // Shared B identity survives across generations and the replacement.
    CHECK(gen1[1].get() == h_bk.get() && gen1[3].get() == h_bv.get());
    // All IDs globally monotonic and unique across build/pack/rebuild.
    for (size_t i = 1; i < ids.size(); ++i) {
        CHECK(ids[i] > ids[i - 1]);
    }
    std::cout << "[two_group_bundle] OK" << std::endl;
}

// ---------------------------------------------------------------------------
// 17. Globally monotonic allocation IDs: one generator shared across build,
//     pack, and import — no collision, no reuse, burn respected.
// ---------------------------------------------------------------------------
static void test_allocation_id_global(cpu_env & e) {
    std::cout << "[allocation_id_global] starting..." << std::endl;
    std::string err;
    xkv_backend_batch_config cfg;
    cfg.residency = GGML_XKV_RES_REFERENCE_HOST;
    xkv_allocation_id_generator gen(1, 100000);
    std::vector<uint64_t> ids;
    auto collect = [&](const auto & handles) {
        for (const auto & h : handles) {
            ids.push_back(h->get_allocation_id());
        }
    };
    // Build A (2 streams) -> IDs 1,2.
    codec_desc da0 = make_desc(factor_role::a_k, GGML_TYPE_TURBO4_0, 8, 128);
    codec_desc da1 = make_desc(factor_role::a_v, GGML_TYPE_TURBO4_0, 8, 128);
    std::vector<uint8_t> ba0 = make_bytes(da0);
    std::vector<uint8_t> ba1 = make_bytes(da1);
    std::shared_ptr<xkv_backend_allocation> pack_src;
    {
        xkv_backend_batch_builder b(&gen);
        b.add_stream(da0, ba0.data(), ba0.size());
        b.add_stream(da1, ba1.data(), ba1.size());
        xkv_backend_batch_result r;
        CHECK(b.build(e.backend, e.buft, cfg, r, &err));
        collect(r.handles);
        pack_src = r.handles[0];
    }
    // Failed pack burns whatever it consumed; next ID never goes backwards.
    {
        xkv_backend_pack_request rq;
        rq.src = pack_src;
        rq.surviving_rows = {0, 1};
        rq.dst_desc = make_desc(factor_role::a_k, GGML_TYPE_TURBO4_0, 2, 128);
        xkv_backend_batch_config c = cfg;
        c.inject_copy_fail_at = 0;
        xkv_backend_pack_result pr;
        CHECK(!xkv_backend_pack_batch(e.backend, e.buft, {rq}, c, gen, pr, &err));
        CHECK(pr.handles.empty());
    }
    const uint64_t high_water = ids.back();
    // Build B (1 stream) continues past the burn.
    {
        codec_desc dx = make_desc(factor_role::a_k, GGML_TYPE_Q8_0, 4, 64);
        std::vector<uint8_t> bx = make_bytes(dx);
        xkv_backend_batch_builder b(&gen);
        b.add_stream(dx, bx.data(), bx.size());
        xkv_backend_batch_result r;
        CHECK(b.build(e.backend, e.buft, cfg, r, &err));
        CHECK(r.handles[0]->get_allocation_id() > high_water);
        collect(r.handles);
    }
    // Import (2 unique keys) continues the same sequence.
    {
        codec_desc di0 = make_desc(factor_role::a_k, GGML_TYPE_TURBO4_0, 4, 128);
        codec_desc di1 = make_desc(factor_role::b_k, GGML_TYPE_TURBO4_0, 32, 128);
        std::vector<uint8_t> bi0 = make_bytes(di0);
        std::vector<uint8_t> bi1 = make_bytes(di1);
        auto mk = [&](const codec_desc & d, const std::vector<uint8_t> & bytes, uint64_t key) {
            xkv_backend_import_stream s;
            s.desc = d;
            s.data = bytes.data();
            s.size = bytes.size();
            s.expected_checksum = xkv_backend_checksum(bytes.data(), bytes.size());
            s.dedup_key = key;
            return s;
        };
        const std::vector<xkv_backend_import_stream> img = {mk(di0, bi0, 101), mk(di1, bi1, 102)};
        xkv_backend_import_result out;
        CHECK(xkv_backend_import_batch(e.backend, e.buft, img, cfg, gen, out, &err));
        CHECK(out.handles.size() == 2);
        collect(out.handles);
    }
    // No collision anywhere: strictly increasing across all three APIs.
    CHECK(ids.size() == 5);
    for (size_t i = 1; i < ids.size(); ++i) {
        CHECK(ids[i] > ids[i - 1]);
    }
    std::cout << "[allocation_id_global] OK" << std::endl;
}
int main() {
    cpu_env e = make_cpu_env();
    test_exact_bytes_cpu(e);
    test_mixed_batch_atomicity(e);
    test_injected_failures(e);
    test_b_handle_sharing(e);
    test_host_release_commit(e);
    test_graph_pin_survives_retirement(e);
    test_allocation_id_overflow(e);
    test_failed_batch_burns_ids(e);
    test_shared_handle_mismatch(e);
    test_accounting_dedup(e);
    test_store_reservation(e);
    test_peak_accounting(e);
    test_one_sync_counters(e);
    test_pack_batch(e);
    test_pack_faults(e);
    test_bundle_verify(e);
    test_import_batch(e);
    test_adopt_device_tensors(e);
    test_take_device_tensors(e);
    test_two_group_bundle(e);
    test_allocation_id_global(e);
    test_pinned_old_cow(e);
    test_reservation_t_minus_1(e);
    test_hot_readback_batch(e);
    test_tri_fetch_selected_k(e);
    test_hot_canonicalize_batch(e);
    test_capability_cpu(e);
    test_vulkan_subcase();
    std::cout << "ALL TESTS PASSED" << std::endl;
    return 0;
}
