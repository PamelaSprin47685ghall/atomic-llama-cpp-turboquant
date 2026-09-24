// W1 regression: TP5 rank-local logits readback.
//
// Exercises the new meta-layer per-rank readback entry against the existing
// single-destination gather on a CPU-only meta device (no GPU required):
//   - per-rank chunk query (ggml_backend_meta_tensor_rank_chunks)
//   - bit-exact equivalence of the per-rank async read and the single-dst read
//   - unequal vocab shards
//   - partial row ranges (decode-time offsets)
//   - split forms the per-rank entry must decline (mirrored) and their exact
//     single-destination fallback semantics
//
// Machine-level proof (five Vulkan ranks hitting pinned slots, no per-rank
// submit+fence) stays with the TP5 integration runs; see the W1 report.

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <vector>

static int g_failures = 0;

#define CHECK(cond, ...) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
        fprintf(stderr, __VA_ARGS__); \
        fprintf(stderr, "\n"); \
        g_failures++; \
    } \
} while (0)

namespace {

struct test_split_ud {
    int     axis;      // ggml_backend_meta_split_axis
    int64_t chunks[8]; // per-rank elements on the split axis (segment 0)
    size_t  n_chunks;
};

struct ggml_backend_meta_split_state test_split_cb(const struct ggml_tensor * tensor, void * userdata) {
    (void) tensor;
    const test_split_ud * ud = (const test_split_ud *) userdata;
    struct ggml_backend_meta_split_state st;
    memset(&st, 0, sizeof(st));
    st.axis = (ggml_backend_meta_split_axis) ud->axis;
    for (size_t j = 0; j < ud->n_chunks; j++) {
        st.ne[j] = ud->chunks[j];
    }
    st.nr[0]       = 1;
    st.n_segments  = 1;
    return st;
}

// verify the global strided view against the per-rank slot buffers
static void verify_slots(const char * tag, const std::vector<float> & ref, int64_t ne_total, int64_t rows,
                          const std::vector<int64_t> & chunks, const std::vector<std::vector<float>> & slots) {
    int64_t col0 = 0;
    for (size_t j = 0; j < chunks.size(); j++) {
        for (int64_t r = 0; r < rows; r++) {
            for (int64_t c = 0; c < chunks[j]; c++) {
                const float expect = ref[r * ne_total + col0 + c];
                const float got    = slots[j][r * chunks[j] + c];
                CHECK(expect == got, "%s: slot %zu row %ld col %ld: got %f want %f",
                      tag, j, (long) r, (long) c, got, expect);
            }
        }
        col0 += chunks[j];
    }
}

struct rig {
    ggml_backend_dev_t    meta_dev = nullptr;
    ggml_backend_t       backend  = nullptr;
    struct ggml_context * ctx      = nullptr;
    ggml_backend_buffer_t buf     = nullptr;
    struct ggml_tensor * t        = nullptr;

    void init(const std::vector<int64_t> & chunks, int axis, int64_t ne_total, int64_t rows, test_split_ud & ud) {
        ud.axis     = axis;
        ud.n_chunks = chunks.size();
        for (size_t j = 0; j < chunks.size(); j++) {
            ud.chunks[j] = chunks[j];
        }
        ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        CHECK(cpu != nullptr, "no CPU device");
        ggml_backend_dev_t devs[8];
        for (size_t j = 0; j < chunks.size(); j++) {
            devs[j] = cpu;
        }
        meta_dev = ggml_backend_meta_device(devs, chunks.size(), test_split_cb, &ud);
        CHECK(meta_dev != nullptr, "meta device creation failed");
        CHECK(ggml_backend_meta_dev_n_simple_devs(meta_dev) == chunks.size(),
              "expected %zu simple devs, got %zu", chunks.size(),
              meta_dev ? ggml_backend_meta_dev_n_simple_devs(meta_dev) : 0);
        backend = ggml_backend_dev_init(meta_dev, "test-w1");
        CHECK(backend != nullptr, "meta backend init failed");

        struct ggml_init_params ip = { /*.mem_size=*/ 4 * 1024 * 1024, /*.mem_buffer=*/ nullptr, /*.no_alloc=*/ true };
        ctx = ggml_init(ip);
        CHECK(ctx != nullptr, "ggml_init failed");
        t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne_total, rows);
        ggml_set_name(t, "logits_like");
        buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_dev_buffer_type(meta_dev));
        CHECK(buf != nullptr, "meta buffer allocation failed");
    }

    void destroy() {
        if (ctx) {
            ggml_free(ctx);
        }
        if (buf) {
            ggml_backend_buffer_free(buf);
        }
        if (backend) {
            ggml_backend_free(backend);
        }
        ctx = nullptr; buf = nullptr; backend = nullptr; t = nullptr;
    }
};

static void case_unequal_shards() {
    const std::vector<int64_t> chunks = {6, 4}; // unequal vocab shards
    const int64_t ne_total = 10;
    const int64_t rows     = 3;
    test_split_ud ud;
    rig r;
    r.init(chunks, GGML_BACKEND_SPLIT_AXIS_0, ne_total, rows, ud);
    if (r.backend == nullptr || r.t == nullptr) {
        r.destroy();
        return;
    }

    std::vector<float> ref((size_t) (ne_total * rows));
    for (size_t i = 0; i < ref.size(); i++) {
        ref[i] = 0.5f * (float) (i + 1);
    }
    ggml_backend_tensor_set(r.t, ref.data(), 0, ref.size() * sizeof(float));

    // 1) the existing single-destination gather is the oracle
    std::vector<float> got(ref.size(), 0.0f);
    ggml_backend_tensor_get(r.t, got.data(), 0, ref.size() * sizeof(float));
    CHECK(memcmp(got.data(), ref.data(), ref.size() * sizeof(float)) == 0,
          "single-destination readback differs from the source");

    // 2) per-rank chunk query must report the unequal shard planes
    const size_t n_ranks = chunks.size();
    std::vector<size_t> chunk_bytes(n_ranks, 0);
    size_t n_chunks = 0;
    CHECK(ggml_backend_meta_tensor_rank_chunks(r.backend, r.t, chunk_bytes.data(), &n_chunks),
          "rank_chunks declined a simple axis split");
    CHECK(n_chunks == n_ranks, "rank_chunks reported %zu chunks, want %zu", n_chunks, n_ranks);
    for (size_t j = 0; j < n_ranks; j++) {
        CHECK(chunk_bytes[j] == (size_t) chunks[j] * sizeof(float),
              "rank %zu chunk: got %zu bytes, want %zu",
              j, chunk_bytes[j], (size_t) chunks[j] * sizeof(float));
    }

    // 3) per-rank async read: each rank lands in its own slot, rank-local stride
    std::vector<std::vector<float>> slots(n_ranks);
    std::vector<void *> dsts(n_ranks, nullptr);
    for (size_t j = 0; j < n_ranks; j++) {
        slots[j].assign((size_t) (chunks[j] * rows), -1.0f);
        dsts[j] = slots[j].data();
    }
    ggml_backend_meta_tensor_get_2d_async_per_rank(r.backend, r.t, dsts.data(), got.data(), 0,
                                                      ref.size() * sizeof(float));
    ggml_backend_synchronize(r.backend);
    verify_slots("full", ref, ne_total, rows, chunks, slots);

    // 4) partial row range (decode-style: slot base at row 0, source at row 1)
    const size_t plane = (size_t) (ne_total * sizeof(float));
    for (size_t j = 0; j < n_ranks; j++) {
        std::fill(slots[j].begin(), slots[j].end(), -1.0f);
    }
    ggml_backend_meta_tensor_get_2d_async_per_rank(r.backend, r.t, dsts.data(), got.data(), plane, plane);
    ggml_backend_synchronize(r.backend);
    {
        int64_t col0 = 0;
        for (size_t j = 0; j < n_ranks; j++) {
            for (int64_t c = 0; c < chunks[j]; c++) {
                const float expect = ref[1 * ne_total + col0 + c];
                const float got    = slots[j][c];
                CHECK(expect == got, "partial: slot %zu col %ld: got %f want %f",
                      j, (long) c, got, expect);
            }
            col0 += chunks[j];
        }
    }

    // 5) dsts == nullptr keeps the exact single-destination semantics
    std::fill(got.begin(), got.end(), 0.0f);
    ggml_backend_meta_tensor_get_2d_async_per_rank(r.backend, r.t, nullptr, got.data(), 0,
                                                      ref.size() * sizeof(float));
    ggml_backend_synchronize(r.backend);
    CHECK(memcmp(got.data(), ref.data(), ref.size() * sizeof(float)) == 0,
          "per-rank entry with dsts=nullptr broke single-destination semantics");

    r.destroy();
}

static void case_declined_split_form() {
    // mirrored split: the per-rank entry must decline and keep the exact
    // single-destination behavior (writes into `data`, rank 0 owns all).
    const std::vector<int64_t> chunks = {6, 4};
    const int64_t ne_total = 10;
    const int64_t rows     = 2;
    test_split_ud ud;
    rig r;
    r.init(chunks, GGML_BACKEND_SPLIT_AXIS_MIRRORED, ne_total, rows, ud);
    if (r.backend == nullptr || r.t == nullptr) {
        r.destroy();
        return;
    }

    std::vector<float> ref((size_t) (ne_total * rows));
    for (size_t i = 0; i < ref.size(); i++) {
        ref[i] = -0.25f * (float) (i + 1);
    }
    ggml_backend_tensor_set(r.t, ref.data(), 0, ref.size() * sizeof(float));

    const size_t n_ranks = chunks.size();
    std::vector<size_t> chunk_bytes(n_ranks, 0);
    size_t n_chunks = 0;
    CHECK(!ggml_backend_meta_tensor_rank_chunks(r.backend, r.t, chunk_bytes.data(), &n_chunks),
          "rank_chunks accepted a mirrored split (caller would take the wrong path)");

    std::vector<std::vector<float>> slots(n_ranks);
    std::vector<void *> dsts(n_ranks, nullptr);
    for (size_t j = 0; j < n_ranks; j++) {
        slots[j].assign((size_t) (chunks[j] * rows), -7.0f);
        dsts[j] = slots[j].data();
    }
    std::vector<float> got(ref.size(), 0.0f);
    ggml_backend_meta_tensor_get_2d_async_per_rank(r.backend, r.t, dsts.data(), got.data(), 0,
                                                      ref.size() * sizeof(float));
    ggml_backend_synchronize(r.backend);
    CHECK(memcmp(got.data(), ref.data(), ref.size() * sizeof(float)) == 0,
          "mirrored split fell back incorrectly");
    for (size_t j = 0; j < n_ranks; j++) {
        for (float v : slots[j]) {
            CHECK(v == -7.0f, "mirrored split wrote a per-rank slot");
        }
    }

    r.destroy();
}

} // namespace

int main() {
    case_unequal_shards();
    case_declined_split_form();
    if (g_failures == 0) {
        printf("test-tp5-rank-local-readback: all passed\n");
        return 0;
    }
    fprintf(stderr, "test-tp5-rank-local-readback: %d failures\n", g_failures);
    return 1;
}
