// test-tp5-rank-local-readback.cpp
//
// Regression test for all-rank asynchronous logits readback on meta backend (W1).
// Exercises the shared contract using standard ggml APIs:
//   - ggml_backend_tensor_get_async / ggml_backend_synchronize
//
// Default runs CPU-only fallback with 5 ranks (no GPU required).
// Explicit --vulkan requires 5 Vulkan devices (backend reg name "Vulkan") or exits 77 (SKIP).

#include "ggml-backend.h"
#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int g_failures = 0;

#define CHECK(cond, ...)                                         \
    do {                                                         \
        if (!(cond)) {                                           \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
            fprintf(stderr, __VA_ARGS__);                        \
            fprintf(stderr, "\n");                               \
            g_failures++;                                        \
        }                                                        \
    } while (0)

namespace {

constexpr size_t  POISON_GUARD_SIZE = 64;
constexpr uint8_t POISON_BYTE       = 0xDE;

struct guarded_buffer {
    std::vector<uint8_t> storage;
    size_t               payload_bytes = 0;

    void resize(size_t bytes) {
        payload_bytes = bytes;
        storage.assign(bytes + 2 * POISON_GUARD_SIZE, POISON_BYTE);
    }

    uint8_t * data() { return storage.data() + POISON_GUARD_SIZE; }

    const uint8_t * data() const { return storage.data() + POISON_GUARD_SIZE; }

    void verify_guards(const char * tag) const {
        for (size_t i = 0; i < POISON_GUARD_SIZE; ++i) {
            if (storage[i] != POISON_BYTE) {
                CHECK(false, "%s: underflow guard corrupted at -%zu (got 0x%02x, want 0x%02x)", tag,
                      POISON_GUARD_SIZE - i, storage[i], POISON_BYTE);
                break;
            }
        }
        for (size_t i = 0; i < POISON_GUARD_SIZE; ++i) {
            size_t idx = POISON_GUARD_SIZE + payload_bytes + i;
            if (storage[idx] != POISON_BYTE) {
                CHECK(false, "%s: overflow guard corrupted at +%zu (got 0x%02x, want 0x%02x)", tag, i, storage[idx],
                      POISON_BYTE);
                break;
            }
        }
    }
};

struct split_config {
    int     axis      = GGML_BACKEND_SPLIT_AXIS_0;
    int64_t chunks[8] = { 0 };
    size_t  n_chunks  = 0;
};

static struct ggml_backend_meta_split_state test_split_cb(const struct ggml_tensor * tensor, void * userdata) {
    const split_config *                 cfg = (const split_config *) userdata;
    struct ggml_backend_meta_split_state st;
    memset(&st, 0, sizeof(st));

    if (tensor->name[0] != '\0' && strcmp(tensor->name, "mirrored_logits") == 0) {
        st.axis = GGML_BACKEND_SPLIT_AXIS_MIRRORED;
    } else {
        st.axis = (ggml_backend_meta_split_axis) cfg->axis;
    }

    for (size_t j = 0; j < cfg->n_chunks; j++) {
        st.ne[j] = cfg->chunks[j];
    }
    st.nr[0]       = 1;
    st.n_segments  = 1;
    st.mapped_span = false;
    return st;
}

static void fill_reference_data(std::vector<float> & ref, int64_t ne_total, int64_t rows) {
    ref.resize((size_t) (ne_total * rows));
    for (size_t r = 0; r < (size_t) rows; ++r) {
        for (size_t c = 0; c < (size_t) ne_total; ++c) {
            ref[r * (size_t) ne_total + c] = 1000.0f * (float) (r + 1) + 0.125f * (float) (c + 1);
        }
    }
}

// Case 1: Unequal shards, empty rank case, nonzero row offsets, zero-byte reads
static void test_unequal_shards_and_offsets(const std::vector<ggml_backend_dev_t> & devs) {
    const std::vector<int64_t> chunks   = { 32, 48, 0, 16, 24 };
    int64_t                    ne_total = 0;
    for (int64_t c : chunks) {
        ne_total += c;
    }
    const int64_t rows = 8;

    split_config cfg;
    cfg.axis     = GGML_BACKEND_SPLIT_AXIS_0;
    cfg.n_chunks = chunks.size();
    for (size_t i = 0; i < chunks.size(); ++i) {
        cfg.chunks[i] = chunks[i];
    }

    ggml_backend_dev_t meta_dev =
        ggml_backend_meta_device(const_cast<ggml_backend_dev_t *>(devs.data()), devs.size(), test_split_cb, &cfg);
    CHECK(meta_dev != nullptr, "meta dev creation failed");
    ggml_backend_t backend = ggml_backend_dev_init(meta_dev, nullptr);
    CHECK(backend != nullptr, "meta backend init failed");

    struct ggml_init_params ip  = { 16 * 1024 * 1024, nullptr, true };
    struct ggml_context *   ctx = ggml_init(ip);
    struct ggml_tensor *    t   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne_total, rows);
    ggml_set_name(t, "logits_unequal");
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_dev_buffer_type(meta_dev));
    CHECK(buf != nullptr, "buffer alloc failed");

    std::vector<float> ref;
    fill_reference_data(ref, ne_total, rows);
    ggml_backend_tensor_set(t, ref.data(), 0, ref.size() * sizeof(float));

    // Full readback
    guarded_buffer dst_full;
    dst_full.resize(ref.size() * sizeof(float));
    ggml_backend_tensor_get_async(backend, t, dst_full.data(), 0, dst_full.payload_bytes);
    ggml_backend_synchronize(backend);
    dst_full.verify_guards("full read");
    CHECK(memcmp(dst_full.data(), ref.data(), dst_full.payload_bytes) == 0, "full readback mismatch");

    // Partial row range at row offset 2
    const size_t   row_bytes = (size_t) (ne_total * sizeof(float));
    guarded_buffer dst_offset;
    dst_offset.resize(row_bytes);
    ggml_backend_tensor_get_async(backend, t, dst_offset.data(), 2 * row_bytes, row_bytes);
    ggml_backend_synchronize(backend);
    dst_offset.verify_guards("row offset 2");
    CHECK(memcmp(dst_offset.data(), ref.data() + 2 * ne_total, row_bytes) == 0, "row offset 2 mismatch");

    // Zero-byte read
    guarded_buffer dst_zero;
    dst_zero.resize(0);
    ggml_backend_tensor_get_async(backend, t, dst_zero.data(), row_bytes, 0);
    ggml_backend_synchronize(backend);
    dst_zero.verify_guards("zero byte read");

    ggml_free(ctx);
    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
}

// Case 2: Slot growth and reuse (1 row -> 7 rows -> 1 row)
static void test_slot_growth_and_reuse(const std::vector<ggml_backend_dev_t> & devs) {
    const std::vector<int64_t> chunks   = { 20, 20, 20, 20, 20 };
    const int64_t              ne_total = 100;
    const int64_t              rows     = 8;

    split_config cfg;
    cfg.axis     = GGML_BACKEND_SPLIT_AXIS_0;
    cfg.n_chunks = chunks.size();
    for (size_t i = 0; i < chunks.size(); ++i) {
        cfg.chunks[i] = chunks[i];
    }

    ggml_backend_dev_t meta_dev =
        ggml_backend_meta_device(const_cast<ggml_backend_dev_t *>(devs.data()), devs.size(), test_split_cb, &cfg);
    ggml_backend_t          backend = ggml_backend_dev_init(meta_dev, nullptr);
    struct ggml_init_params ip      = { 16 * 1024 * 1024, nullptr, true };
    struct ggml_context *   ctx     = ggml_init(ip);
    struct ggml_tensor *    t       = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne_total, rows);
    ggml_set_name(t, "logits_growth");
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_dev_buffer_type(meta_dev));

    std::vector<float> ref;
    fill_reference_data(ref, ne_total, rows);
    ggml_backend_tensor_set(t, ref.data(), 0, ref.size() * sizeof(float));

    const size_t row_bytes = (size_t) (ne_total * sizeof(float));

    // 1 row read
    guarded_buffer b1;
    b1.resize(row_bytes);
    ggml_backend_tensor_get_async(backend, t, b1.data(), 0, row_bytes);
    ggml_backend_synchronize(backend);
    b1.verify_guards("growth: 1 row");
    CHECK(memcmp(b1.data(), ref.data(), row_bytes) == 0, "growth: 1 row mismatch");

    // 7 rows read (triggers slot reallocation / growth)
    guarded_buffer b7;
    b7.resize(7 * row_bytes);
    ggml_backend_tensor_get_async(backend, t, b7.data(), row_bytes, 7 * row_bytes);
    ggml_backend_synchronize(backend);
    b7.verify_guards("growth: 7 rows");
    CHECK(memcmp(b7.data(), ref.data() + ne_total, 7 * row_bytes) == 0, "growth: 7 rows mismatch");

    // 1 row read (reuses grown slots)
    guarded_buffer b1_after;
    b1_after.resize(row_bytes);
    ggml_backend_tensor_get_async(backend, t, b1_after.data(), 4 * row_bytes, row_bytes);
    ggml_backend_synchronize(backend);
    b1_after.verify_guards("growth: 1 row after");
    CHECK(memcmp(b1_after.data(), ref.data() + 4 * ne_total, row_bytes) == 0, "growth: 1 row after mismatch");

    ggml_free(ctx);
    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
}

// Case 3: Two outstanding async gets before synchronize into distinct destinations
// First read is SMALL (1 row), second read is BIGGER (3 rows) while predecessor is pending.
// Exercises slot growth and transaction retirement with predecessor still pending.
// First destination MUST survive and receive full correct bytes.
static void test_two_outstanding_gets_growth_pending(const std::vector<ggml_backend_dev_t> & devs) {
    const std::vector<int64_t> chunks   = { 16, 24, 16, 32, 12 };
    const int64_t              ne_total = 100;
    const int64_t              rows     = 6;

    split_config cfg;
    cfg.axis     = GGML_BACKEND_SPLIT_AXIS_0;
    cfg.n_chunks = chunks.size();
    for (size_t i = 0; i < chunks.size(); ++i) {
        cfg.chunks[i] = chunks[i];
    }

    ggml_backend_dev_t meta_dev =
        ggml_backend_meta_device(const_cast<ggml_backend_dev_t *>(devs.data()), devs.size(), test_split_cb, &cfg);
    ggml_backend_t          backend = ggml_backend_dev_init(meta_dev, nullptr);
    struct ggml_init_params ip      = { 16 * 1024 * 1024, nullptr, true };
    struct ggml_context *   ctx     = ggml_init(ip);
    struct ggml_tensor *    t       = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne_total, rows);
    ggml_set_name(t, "logits_outstanding");
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_dev_buffer_type(meta_dev));

    std::vector<float> ref;
    fill_reference_data(ref, ne_total, rows);
    ggml_backend_tensor_set(t, ref.data(), 0, ref.size() * sizeof(float));

    const size_t row_bytes = (size_t) (ne_total * sizeof(float));

    guarded_buffer dst1_small;
    dst1_small.resize(1 * row_bytes);  // 1 row

    guarded_buffer dst2_big;
    dst2_big.resize(3 * row_bytes);  // 3 rows (bigger: exercises growth while dst1 is pending)

    // Enqueue first small get (row 0)
    ggml_backend_tensor_get_async(backend, t, dst1_small.data(), 0, 1 * row_bytes);

    // Enqueue second bigger get (rows 1..3) WITHOUT synchronizing
    ggml_backend_tensor_get_async(backend, t, dst2_big.data(), 1 * row_bytes, 3 * row_bytes);

    // Now synchronize the backend
    ggml_backend_synchronize(backend);

    dst1_small.verify_guards("outstanding get: dst1_small");
    dst2_big.verify_guards("outstanding get: dst2_big");

    CHECK(memcmp(dst1_small.data(), ref.data(), 1 * row_bytes) == 0,
          "outstanding get: dst1_small data corrupted, lost, or overwritten by second read");
    CHECK(memcmp(dst2_big.data(), ref.data() + ne_total, 3 * row_bytes) == 0,
          "outstanding get: dst2_big data mismatch");

    ggml_free(ctx);
    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
}

// Case 4: Pending gather followed immediately by fallback on SAME backend before synchronize.
// Uses one Meta device / backend context with both an axis0 tensor and a mirrored tensor.
// Enqueues gather on axis0 tensor, then enqueues get on mirrored tensor without intermediate sync.
// Asserts both outputs match oracle and guards intact.
static void test_pending_gather_followed_by_mirrored_fallback(const std::vector<ggml_backend_dev_t> & devs) {
    const std::vector<int64_t> chunks   = { 20, 20, 20, 20, 20 };
    const int64_t              ne_total = 100;
    const int64_t              rows     = 4;

    split_config cfg;
    cfg.axis     = GGML_BACKEND_SPLIT_AXIS_0;
    cfg.n_chunks = chunks.size();
    for (size_t i = 0; i < chunks.size(); ++i) {
        cfg.chunks[i] = chunks[i];
    }

    ggml_backend_dev_t meta_dev =
        ggml_backend_meta_device(const_cast<ggml_backend_dev_t *>(devs.data()), devs.size(), test_split_cb, &cfg);
    ggml_backend_t          backend = ggml_backend_dev_init(meta_dev, nullptr);
    struct ggml_init_params ip      = { 32 * 1024 * 1024, nullptr, true };
    struct ggml_context *   ctx     = ggml_init(ip);

    // Tensor 1: axis 0 (gather)
    struct ggml_tensor * t_gather = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne_total, rows);
    ggml_set_name(t_gather, "axis0_logits");

    // Tensor 2: mirrored (fallback), identified by name in test_split_cb
    struct ggml_tensor * t_mirror = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne_total, rows);
    ggml_set_name(t_mirror, "mirrored_logits");

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_dev_buffer_type(meta_dev));
    CHECK(buf != nullptr, "buffer alloc for dual tensors failed");

    std::vector<float> ref_gather;
    fill_reference_data(ref_gather, ne_total, rows);
    ggml_backend_tensor_set(t_gather, ref_gather.data(), 0, ref_gather.size() * sizeof(float));

    std::vector<float> ref_mirror(ne_total * rows);
    for (size_t i = 0; i < ref_mirror.size(); ++i) {
        ref_mirror[i] = -7.5f * (float) (i + 1);
    }
    ggml_backend_tensor_set(t_mirror, ref_mirror.data(), 0, ref_mirror.size() * sizeof(float));

    const size_t total_bytes = ref_gather.size() * sizeof(float);

    guarded_buffer dst_gather;
    dst_gather.resize(total_bytes);

    guarded_buffer dst_mirror;
    dst_mirror.resize(total_bytes);

    // 1) Enqueue gather on axis0 tensor
    ggml_backend_tensor_get_async(backend, t_gather, dst_gather.data(), 0, total_bytes);

    // 2) Immediately enqueue fallback get on mirrored tensor on the SAME backend WITHOUT synchronizing
    ggml_backend_tensor_get_async(backend, t_mirror, dst_mirror.data(), 0, total_bytes);

    // 3) Synchronize backend
    ggml_backend_synchronize(backend);

    // 4) Assert both outputs
    dst_gather.verify_guards("pending gather -> fallback: dst_gather");
    dst_mirror.verify_guards("pending gather -> fallback: dst_mirror");

    CHECK(memcmp(dst_gather.data(), ref_gather.data(), total_bytes) == 0,
          "gather output corrupted when followed by mirrored fallback before sync");
    CHECK(memcmp(dst_mirror.data(), ref_mirror.data(), total_bytes) == 0,
          "mirrored fallback output corrupted when queued after pending gather");

    ggml_free(ctx);
    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
}

// Case 5: Teardown retirement with source ctx/buffer still alive.
// Enqueues an async get into destination, then frees the backend directly while
// source ctx and buffer remain valid. Backend destruction must retire/flush the
// pending read so dst bytes and guards are fully verified after retirement.
static void test_teardown_retirement(const std::vector<ggml_backend_dev_t> & devs) {
    const std::vector<int64_t> chunks   = { 20, 20, 20, 20, 20 };
    const int64_t              ne_total = 100;
    const int64_t              rows     = 4;

    split_config cfg;
    cfg.axis     = GGML_BACKEND_SPLIT_AXIS_0;
    cfg.n_chunks = chunks.size();
    for (size_t i = 0; i < chunks.size(); ++i) {
        cfg.chunks[i] = chunks[i];
    }

    ggml_backend_dev_t meta_dev =
        ggml_backend_meta_device(const_cast<ggml_backend_dev_t *>(devs.data()), devs.size(), test_split_cb, &cfg);
    ggml_backend_t          backend = ggml_backend_dev_init(meta_dev, nullptr);
    struct ggml_init_params ip      = { 16 * 1024 * 1024, nullptr, true };
    struct ggml_context *   ctx     = ggml_init(ip);
    struct ggml_tensor *    t       = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne_total, rows);
    ggml_set_name(t, "logits_teardown");
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_dev_buffer_type(meta_dev));

    std::vector<float> ref;
    fill_reference_data(ref, ne_total, rows);
    ggml_backend_tensor_set(t, ref.data(), 0, ref.size() * sizeof(float));

    const size_t   total_bytes = ref.size() * sizeof(float);
    guarded_buffer dst;
    dst.resize(total_bytes);

    // Enqueue async get into dst
    ggml_backend_tensor_get_async(backend, t, dst.data(), 0, total_bytes);

    // Free backend FIRST while source ctx and buffer remain fully valid
    ggml_backend_free(backend);

    // Assert that dst received full payload bytes and guards remain intact
    dst.verify_guards("teardown retirement");
    CHECK(memcmp(dst.data(), ref.data(), total_bytes) == 0,
          "teardown retirement did not complete/flush readback before backend freed");

    // Clean up remaining resources
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

}  // namespace

int main(int argc, char ** argv) {
    bool use_vulkan = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--vulkan") == 0) {
            use_vulkan = true;
        }
    }

    std::vector<ggml_backend_dev_t> devs;
    if (use_vulkan) {
        ggml_backend_load_all();
        size_t count = ggml_backend_dev_count();
        for (size_t i = 0; i < count; ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            auto *             reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
            if (reg) {
                const char * name = ggml_backend_reg_name(reg);
                if (name && std::string(name).find("Vulkan") != std::string::npos) {
                    devs.push_back(dev);
                    if (devs.size() == 5) {
                        break;
                    }
                }
            }
        }
        if (devs.size() < 5) {
            fprintf(stderr, "SKIP: --vulkan requires 5 Vulkan devices, found %zu. Exiting with 77.\n", devs.size());
            return 77;
        }
        printf("Running with 5 real Vulkan GPU devices\n");
    } else {
        ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (!cpu) {
            fprintf(stderr, "FAIL: no CPU device available\n");
            return 1;
        }
        devs.assign(5, cpu);
        printf("Running in CPU fallback mode (5 simulated ranks)\n");
    }

    test_unequal_shards_and_offsets(devs);
    test_slot_growth_and_reuse(devs);
    test_two_outstanding_gets_growth_pending(devs);
    test_pending_gather_followed_by_mirrored_fallback(devs);
    test_teardown_retirement(devs);

    if (g_failures == 0) {
        printf("test-tp5-rank-local-readback: all passed\n");
        return 0;
    }
    fprintf(stderr, "test-tp5-rank-local-readback: %d failures\n", g_failures);
    return 1;
}
