#include "llama-kv-cache.h"
#include "llama-kv-transform.h"
#include "llama-model.h"
#include "llama-cparams.h"
#include "ggml-backend-impl.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <vector>

#define CHECK(x) do { if (!(x)) { throw std::runtime_error(std::string("FAIL line ") + std::to_string(__LINE__) + ": " #x); } } while (0)

static const char * calibration = "test-triattention-kv.triattention";

static void write_calibration() {
    std::ofstream file(calibration, std::ios::binary);
    CHECK(file.good());
    auto u32 = [&](uint32_t x) { file.write((const char *) &x, sizeof(x)); };
    u32(TRIATTENTION_MAGIC); u32(2); u32(128); u32(1); u32(1); u32(1);
    const double theta = 10000;
    file.write((const char *) &theta, sizeof(theta));
    u32(0); u32(1); u32(64); u32(128); u32(5);
    file.write("test", 5);
    u32(0); u32(0);
    for (int a = 0; a < 4; ++a) {
        const float value = a == 1 ? 0.f : 1.f;
        for (int i = 0; i < 64; ++i) { file.write((const char *) &value, sizeof(value)); }
    }
    CHECK(file.good());
}

static llama_cparams params() {
    llama_cparams p{};
    p.rope_freq_base = 10000.f;
    p.rope_freq_scale = 1.f;
    p.yarn_attn_factor = 1.f;
    p.yarn_beta_fast = 32.f;
    p.yarn_beta_slow = 1.f;
    p.n_ctx_orig_yarn = 4096;
    return p;
}

struct fixture {
    std::unique_ptr<llama_model> model;
    std::unique_ptr<llama_kv_cache> cache;
    bool unified;

    fixture(bool unified = true, ggml_type ktype = GGML_TYPE_F16, bool transposed = false) : unified(unified) {
        model.reset(llama_model_create(LLM_ARCH_QWEN2, llama_model_default_params()));
        auto & hp = model->hparams;
        hp.n_layer_all = 1;
        hp.n_embd = 128;
        hp.n_ctx_train = 4096;
        hp.n_head_arr.fill(1);
        hp.n_head_kv_arr.fill(1);
        hp.n_embd_head_k_full = hp.n_embd_head_v_full = 128;
        hp.n_rot_full = 128;
        hp.rope_type = LLAMA_ROPE_TYPE_NORM;
        hp.rope_freq_base_train = 10000.f;
        model->layers.resize(1);
        cache = std::make_unique<llama_kv_cache>(*model, hp, ktype, GGML_TYPE_F16,
            transposed, false, unified, 256, 2, 1, 0, LLAMA_SWA_TYPE_NONE,
            nullptr, nullptr, nullptr, nullptr);
    }

    void seed(llama_seq_id seq, uint32_t count = 192) {
        auto & cells = const_cast<llama_kv_cells &>(cache->get_cells(seq));
        auto * k = cache->get_k_storage(0);
        auto * v = cache->get_v_storage(0);
        const uint32_t stream = unified ? 0 : (uint32_t) seq;
        std::vector<float> row(128);
        std::vector<uint8_t> encoded(ggml_row_size(k->type, 128));
        std::vector<ggml_fp16_t> vrow(128);
        for (uint32_t i = 0; i < count; ++i) {
            cells.pos_set(i, i);
            cells.seq_add(i, seq);
            for (uint32_t d = 0; d < 128; ++d) {
                row[d] = std::sin((i + 1) * (d + 2) * .0071f);
                vrow[d] = ggml_fp32_to_fp16((float) (i + d) / 512);
            }
            ggml_get_type_traits(k->type)->from_float_ref(row.data(), encoded.data(), 128);
            ggml_backend_tensor_set(k, encoded.data(), stream * k->nb[2] + i * k->nb[1], encoded.size());
            // Tests below use ordinary V layout; transposed pack is covered
            // separately by byte-pattern snapshots.
            ggml_backend_tensor_set(v, vrow.data(), stream * v->nb[2] + i * v->nb[1], vrow.size() * sizeof(vrow[0]));
        }
    }
};

static std::vector<uint8_t> bytes(ggml_tensor * tensor) {
    std::vector<uint8_t> data(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, data.data(), 0, data.size());
    return data;
}

static llama_memory_kv_reclaim_request request(llama_seq_id seq) {
    llama_memory_kv_reclaim_request r{};
    r.required_free = 1;
    r.drain_to_floor = true;
    llama_memory_kv_reclaim_seq_hint h{};
    h.seq_id = seq; h.logical_tokens = 192; h.tail_guard = 16; h.eligible = true;
    r.seq_hints.push_back(h);
    return r;
}

static void test_reclaim_streams() {
    for (bool unified : {true, false}) {
        for (bool explicit_hint : {true, false}) {
            fixture f(unified);
            f.seed(1);
            f.cache->init_triattention(calibration, .25, 16, params());
            auto * k = f.cache->get_k_storage(0);
            auto * v = f.cache->get_v_storage(0);
            auto old_k = bytes(k), old_v = bytes(v);
            auto r = request(1);
            if (!explicit_hint) { r.seq_hints.clear(); }
            auto result = f.cache->reclaim_kv(r);
            CHECK(result.supported && result.changed && result.physical_freed == 144);
            CHECK(f.cache->get_kv_seq_used(1) == 48);
            CHECK(f.cache->get_kv_seq_used(0) == 0);
            const auto & cells = f.cache->get_cells(1);
            auto new_k = bytes(k), new_v = bytes(v);
            const uint32_t stream = unified ? 0 : 1;
            for (uint32_t i = 0; i < 48; ++i) {
                const uint32_t pos = (uint32_t) cells.pos_get(i);
                auto check_tensor = [&](ggml_tensor * t, const std::vector<uint8_t> & old_data, const std::vector<uint8_t> & new_data) {
                    const size_t stride = t->nb[1], base = stream * t->nb[2];
                    CHECK(std::equal(old_data.begin() + base + pos * stride,
                                     old_data.begin() + base + (pos + 1) * stride,
                                     new_data.begin() + base + i * stride));
                };
                check_tensor(k, old_k, new_k);
                check_tensor(v, old_v, new_v);
            }
        }
    }
}

static void test_atomic_failure_and_shared_prefix() {
    fixture f;
    f.seed(0);
    f.cache->init_triattention(calibration, .25, 16, params());
    auto r = request(0);
    auto invalid = r.seq_hints.front();
    invalid.seq_id = LLAMA_MAX_SEQ + 1;
    r.seq_hints.push_back(invalid);
    const auto old_k = bytes(f.cache->get_k_storage(0));
    bool rejected = false;
    try { f.cache->reclaim_kv(r); } catch (const std::runtime_error &) { rejected = true; }
    CHECK(rejected);
    CHECK(f.cache->get_kv_seq_used(0) == 192);
    CHECK(bytes(f.cache->get_k_storage(0)) == old_k);

    // Inject an unsupported backend without changing its tensor storage.
    auto buft = ggml_backend_cpu_buffer_type();
    auto is_host = buft->iface.is_host;
    buft->iface.is_host = [](ggml_backend_buffer_type_t) { return false; };
    llama_memory_kv_reclaim_result result;
    try { result = f.cache->reclaim_kv(request(0)); }
    catch (...) { buft->iface.is_host = is_host; throw; }
    buft->iface.is_host = is_host;
    CHECK(!result.supported && !result.changed && result.references_removed == 0);
    CHECK(f.cache->get_kv_seq_used(0) == 192);
    CHECK(bytes(f.cache->get_k_storage(0)) == old_k);

    f.cache->seq_cp(0, 1, -1, -1);
    result = f.cache->reclaim_kv(request(0));
    CHECK(result.changed && result.references_removed == 144 && result.physical_freed == 0);
    CHECK(f.cache->get_kv_seq_used(0) == 48 && f.cache->get_kv_seq_used(1) == 192);
}

static void test_turbo_shift() {
    for (auto type : {GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO4_0}) {
        fixture f(true, type);
        f.seed(0, 64);
        auto * k = f.cache->get_k_storage(0);
        const auto original = bytes(k);
        f.cache->seq_rm(0, 0, 32);
        f.cache->seq_add(0, 32, 64, -32);
        CHECK(f.cache->get_can_shift() && f.cache->get_has_shift());
        CHECK(f.cache->init_update(nullptr, false)->get_status() == LLAMA_MEMORY_STATUS_SUCCESS);
        f.cache->shift_turbo_keys(params());
        auto actual = bytes(k), expected = original;
        for (uint32_t i = 32; i < 64; ++i) {
            std::vector<float> row(128);
            llama_kv_decode_key(type, original.data() + i * k->nb[1], row.data(), 128);
            for (uint32_t d = 0; d < 64; ++d) {
                // Independently spell out the relative phase, rather than
                // reusing the production shift helper.
                const float w = std::pow(10000.f, -(float) d / 64.f);
                const float c = (float) std::cos((double) w * -32);
                const float s = (float) std::sin((double) w * -32);
                const float a = row[2 * d], b = row[2 * d + 1];
                row[2 * d] = a * c - b * s;
                row[2 * d + 1] = a * s + b * c;
            }
            ggml_get_type_traits(type)->from_float_ref(row.data(), expected.data() + i * k->nb[1], 128);
        }
        CHECK(actual != original);
        CHECK(actual == expected);
    }
}

static void test_transposed_v_pack() {
    fixture f(true, GGML_TYPE_F16, true);
    f.seed(0);
    auto * v = f.cache->get_v_storage(0);
    const auto before = bytes(v);
    f.cache->init_triattention(calibration, .01, 15, params());
    auto r = request(0);
    r.seq_hints[0].tail_guard = 15;
    const auto result = f.cache->reclaim_kv(r);
    CHECK(result.changed && result.physical_freed == 177);
    const auto after = bytes(v);
    for (uint32_t d = 0; d < 128; ++d) {
        for (uint32_t i = 0; i < 15; ++i) {
            const size_t from = (d * 256 + 177 + i) * sizeof(ggml_fp16_t);
            const size_t to = (d * 256 + i) * sizeof(ggml_fp16_t);
            CHECK(before[from] == after[to] && before[from + 1] == after[to + 1]);
        }
    }
}

int main() {
    try {
        ggml_backend_load_all();
        write_calibration();
        test_reclaim_streams();
        test_atomic_failure_and_shared_prefix();
        test_turbo_shift();
        test_transposed_v_pack();
        std::remove(calibration);
        std::fprintf(stderr, "PASS: TriAttention cache co-operation\n");
        return 0;
    } catch (const std::exception & error) {
        std::remove(calibration);
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
