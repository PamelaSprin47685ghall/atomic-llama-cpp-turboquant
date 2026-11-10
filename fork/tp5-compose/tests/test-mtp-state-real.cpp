// Real llama_state_seq_get_data_ext/set_data_ext round trip on a CPU context.
// Host flags restore the exact host state bytes; ON_DEVICE flags capture only
// metadata while the tensor snapshot lives in ctx->mem_storage[seq]. A device
// restore after a mutation must restore the exact host state. The ON_DEVICE
// metadata is versioned and bound to its context: a missing snapshot, stale
// generation, foreign context, short buffer, or truncated payload must return
// 0 without aborting or copying a partial state.
// Set TP5_TEST_MODEL to a small GGUF; without it the test reports SKIP.

#include <llama.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include "../../../ggml/src/ggml-backend-impl.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int fails = 0;
static void check(bool ok, const char * name) { std::printf("%s %s\n", ok ? "PASS" : "FAIL", name); if (!ok) ++fails; }

static ggml_backend_meta_split_state meta_split_state(const ggml_tensor * t, void *) {
    ggml_backend_meta_split_state s = {};
    s.n_segments = 1;
    s.nr[0] = 1;
    if (t->ne[1] > 1) {
        s.axis = GGML_BACKEND_SPLIT_AXIS_1;
        const int64_t n = t->ne[1];
        s.ne[0] = n/2;
        s.ne[1] = n - n/2;
    } else {
        s.axis = GGML_BACKEND_SPLIT_AXIS_0;
        const int64_t n = t->ne[0];
        if (std::strstr(t->name, "bad") != nullptr) {
            s.ne[0] = n/4;
            s.ne[1] = n - n/4;
        } else {
            s.ne[0] = n/2;
            s.ne[1] = n - n/2;
        }
    }
    return s;
}

struct meta_state_ctx {
    bool bad_state = false;
};

static ggml_backend_meta_split_state meta_state_split(const ggml_tensor * t, void * ud) {
    const auto * st = (const meta_state_ctx *) ud;
    ggml_backend_meta_split_state s = {};
    s.n_segments = 1;
    s.nr[0] = 1;
    const bool state_tensor = st != nullptr && st->bad_state;
    if (t->ne[1] > 1) {
        s.axis = GGML_BACKEND_SPLIT_AXIS_1;
        const int64_t n = t->ne[1];
        s.ne[0] = state_tensor ? n/4 : n/2;
        s.ne[1] = n - s.ne[0];
    } else {
        s.axis = GGML_BACKEND_SPLIT_AXIS_0;
        const int64_t n = t->ne[0];
        s.ne[0] = state_tensor ? n/4 : n/2;
        s.ne[1] = n - s.ne[0];
    }
    return s;
}

static bool run_meta_buffer_copy_test() {
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu == nullptr) { check(false, "meta: CPU device"); return false; }
    ggml_backend_dev_t devs[2] = { cpu, cpu };
    ggml_backend_dev_t meta_dev = ggml_backend_meta_device(devs, 2, meta_split_state, nullptr);
    if (meta_dev == nullptr) { check(false, "meta: device"); return false; }
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(meta_dev);
    if (buft == nullptr) { check(false, "meta: buffer type"); return false; }

    ggml_init_params params = {
        /*.mem_size   =*/ 4*1024*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * gctx = ggml_init(params);
    if (gctx == nullptr) { check(false, "meta: context"); return false; }

    const int64_t n = 64;
    ggml_tensor * src = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, n);
    ggml_tensor * dst = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, n);
    ggml_tensor * bad = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, n);
    ggml_tensor * bad_type = ggml_new_tensor_1d(gctx, GGML_TYPE_F16, n);
    ggml_tensor * src2 = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, 8, 8);
    ggml_tensor * dst2 = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, 8, 8);
    ggml_set_name(src, "src");
    ggml_set_name(dst, "dst");
    ggml_set_name(bad, "bad");
    ggml_set_name(bad_type, "bad_type");
    ggml_set_name(src2, "src2");
    ggml_set_name(dst2, "dst2");

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(gctx, buft);
    if (buf == nullptr) { check(false, "meta: alloc"); ggml_free(gctx); return false; }

    std::vector<float> orig(n), zero(n, 0.0f), neg(n, -1.0f), got(n, 0.0f);
    for (int64_t i = 0; i < n; ++i) {
        orig[i] = float(i + 1);
    }
    ggml_backend_tensor_set(src, orig.data(), 0, n*sizeof(float));
    ggml_backend_tensor_set(dst, zero.data(), 0, n*sizeof(float));
    ggml_backend_tensor_set(bad, neg.data(), 0, n*sizeof(float));

    // Capture and restore across two real CPU child buffers.
    const bool cap = ggml_backend_buffer_copy_tensor(src, dst);
    check(cap, "meta: capture multi-child");
    ggml_backend_tensor_get(dst, got.data(), 0, n*sizeof(float));
    check(std::memcmp(got.data(), orig.data(), n*sizeof(float)) == 0,
          "meta: capture covers both children");

    std::vector<float> mutated = orig;
    for (float & v : mutated) {
        v = -100.0f;
    }
    ggml_backend_tensor_set(src, mutated.data(), 0, n*sizeof(float));
    const bool res = ggml_backend_buffer_copy_tensor(dst, src);
    check(res, "meta: restore multi-child");
    ggml_backend_tensor_get(src, got.data(), 0, n*sizeof(float));
    check(std::memcmp(got.data(), orig.data(), n*sizeof(float)) == 0,
          "meta: restore covers both children");

    // Non-zero view: the view crosses both children.
    ggml_tensor * src_v = ggml_view_1d(gctx, src, n/2, (n/4)*sizeof(float));
    ggml_tensor * dst_v = ggml_view_1d(gctx, dst, n/2, (n/4)*sizeof(float));
    check(ggml_backend_view_init(src_v) == GGML_STATUS_SUCCESS, "meta: src view init");
    check(ggml_backend_view_init(dst_v) == GGML_STATUS_SUCCESS, "meta: dst view init");
    ggml_backend_tensor_set(dst_v, zero.data(), 0, (n/2)*sizeof(float));
    const bool vcap = ggml_backend_buffer_copy_tensor(src_v, dst_v);
    check(vcap, "meta: non-zero view capture");
    ggml_backend_tensor_get(dst, got.data(), 0, n*sizeof(float));
    check(std::memcmp(got.data() + n/4, orig.data() + n/4, (n/2)*sizeof(float)) == 0,
          "meta: non-zero view copies the right range");

    // 2D tensor split on axis 1: a different real child layout.
    std::vector<float> orig2(64), expect2(64), got2(64, 0.0f);
    for (int64_t i = 0; i < 64; ++i) {
        orig2[i] = float(1000 + i);
    }
    ggml_backend_tensor_set(src2, orig2.data(), 0, 64*sizeof(float));
    ggml_backend_tensor_set(dst2, got2.data(), 0, 64*sizeof(float));
    const bool cap2 = ggml_backend_buffer_copy_tensor(src2, dst2);
    check(cap2, "meta: axis-1 2D capture");
    ggml_backend_tensor_get(dst2, got2.data(), 0, 64*sizeof(float));
    check(std::memcmp(got2.data(), orig2.data(), 64*sizeof(float)) == 0,
          "meta: axis-1 2D covers both children");
    expect2 = orig2;
    for (float & v : orig2) {
        v = -7.0f;
    }
    ggml_backend_tensor_set(src2, orig2.data(), 0, 64*sizeof(float));
    const bool res2 = ggml_backend_buffer_copy_tensor(dst2, src2);
    check(res2, "meta: axis-1 2D restore");
    ggml_backend_tensor_get(src2, got2.data(), 0, 64*sizeof(float));
    check(std::memcmp(got2.data(), expect2.data(), 64*sizeof(float)) == 0,
          "meta: axis-1 2D restore covers both children");

    // A different split is a real slow-path copy, not an unsupported case.
    const bool bad_cap = ggml_backend_buffer_copy_tensor(src, bad);
    check(!bad_cap, "meta: different split fast preflight false");
    ggml_backend_tensor_copy(src, bad);
    ggml_backend_tensor_get(bad, got.data(), 0, n*sizeof(float));
    check(std::memcmp(got.data(), orig.data(), n*sizeof(float)) == 0,
          "meta: different split slow copy succeeds");

    // A whole-tensor type mismatch is impossible, not a slow copy.
    check(ggml_backend_meta_buffer_copy_kind(src, bad_type) == GGML_BACKEND_META_COPY_IMPOSSIBLE,
          "meta: type mismatch is impossible");
    const bool imp_cap = ggml_backend_buffer_copy_tensor(src, bad_type);
    check(!imp_cap, "meta: impossible fast copy rejected");

    ggml_backend_buffer_free(buf);
    ggml_free(gctx);
    return fails == 0;
}

static int decode_one_seq(llama_context * ctx, llama_token tok, llama_pos pos, llama_seq_id seq) {
    llama_batch b = llama_batch_init(1, 0, 1);
    if (!b.token || !b.pos || !b.n_seq_id || !b.seq_id || !b.logits) { llama_batch_free(b); return -1; }
    b.n_tokens = 1; b.token[0] = tok; b.pos[0] = pos; b.n_seq_id[0] = 1; b.seq_id[0][0] = seq; b.logits[0] = 1;
    const int rc = llama_decode(ctx, b);
    llama_batch_free(b);
    return rc;
}

static int decode_one(llama_context * ctx, llama_token tok, llama_pos pos) {
    return decode_one_seq(ctx, tok, pos, 0);
}

static std::vector<float> decode_logits(llama_context * ctx, llama_token tok, llama_pos pos, llama_seq_id seq, int n_vocab) {
    if (decode_one_seq(ctx, tok, pos, seq) != 0) {
        return {};
    }
    const float * logits = llama_get_logits_ith(ctx, 0);
    if (logits == nullptr) {
        return {};
    }
    return std::vector<float>(logits, logits + n_vocab);
}

// Capture exactly the current sequence state into a fresh buffer. A zero size
// or a short write yields an empty vector so the caller's check fails loudly.
static std::vector<uint8_t> capture_seq(llama_context * ctx, llama_seq_id seq, llama_state_seq_flags flags) {
    const size_t size = llama_state_seq_get_size_ext(ctx, seq, flags);
    std::vector<uint8_t> buf(size);
    if (size == 0) {
        return buf;
    }
    const size_t written = llama_state_seq_get_data_ext(ctx, buf.data(), buf.size(), seq, flags);
    if (written != size) {
        return {};
    }
    return buf;
}

static bool same_bytes(const std::vector<uint8_t> & a, const std::vector<uint8_t> & b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size()) == 0);
}

// Host blobs carry the existing magic and seq_id before the payload. Empty
// seq-agnostic payloads can be compared across sequence ids; non-empty PLE
// history carries the source seq id, so semantic content uses generation/file
// tests instead.
static bool same_payload(const std::vector<uint8_t> & a, const std::vector<uint8_t> & b) {
    constexpr size_t host_header = 4 + 4;
    return a.size() == b.size() && a.size() >= host_header &&
           std::memcmp(a.data() + host_header, b.data() + host_header, a.size() - host_header) == 0;
}

static bool run_state_meta_test(const char * path) {
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu == nullptr) { check(false, "meta state: CPU device"); return false; }
    ggml_backend_dev_t devs[2] = { cpu, cpu };
    meta_state_ctx ms;
    ggml_backend_dev_t meta_dev = ggml_backend_meta_device(devs, 2, meta_state_split, &ms);
    if (meta_dev == nullptr) { check(false, "meta state: device"); return false; }

    ggml_backend_dev_t model_devs[2] = { meta_dev, nullptr };
    auto mp = llama_model_default_params();
    mp.devices = model_devs;
    mp.n_gpu_layers = 2;
    mp.split_mode = LLAMA_SPLIT_MODE_NONE;
    llama_model * model = llama_model_load_from_file(path, mp);
    if (model == nullptr) { check(false, "meta state: load model"); return false; }

    auto cp = llama_context_default_params();
    cp.n_ctx = 64;
    cp.n_batch = 8;
    cp.n_ubatch = 8;
    cp.n_seq_max = 2;
    cp.n_threads = 1;
    cp.n_threads_batch = 1;
    cp.offload_kqv = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (ctx == nullptr) {
        check(false, "meta state: context");
        llama_model_free(model);
        return false;
    }

    const auto h1 = capture_seq(ctx, 0, 0);
    check(!h1.empty(), "meta state: host capture");
    const auto d1 = capture_seq(ctx, 0, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
    uint32_t magic = 0;
    if (d1.size() >= sizeof(magic)) {
        std::memcpy(&magic, d1.data(), sizeof(magic));
    }
    check(magic == 0x4f4e4445, "meta state: device capture is device format");
    const auto h1b = capture_seq(ctx, 0, 0);
    check(same_bytes(h1, h1b), "meta state: capture leaves the old state");

    const size_t set1 = llama_state_seq_set_data_ext(
        ctx, d1.data(), d1.size(), 0, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
    check(set1 == d1.size(), "meta state: restore");
    const auto h2 = capture_seq(ctx, 0, 0);
    check(same_bytes(h1, h2), "meta state: restore exact");

    // A second context has the Meta split callback disagree from creation
    // time. Its device snapshot must not be copied into the normal target;
    // the state API takes the host format and leaves the normal state intact.
    meta_state_ctx ms_bad;
    ms_bad.bad_state = true;
    ggml_backend_dev_t meta_dev_bad = ggml_backend_meta_device(devs, 2, meta_state_split, &ms_bad);
    check(meta_dev_bad != nullptr, "meta state: bad device");
    ggml_backend_dev_t model_devs_bad[2] = { meta_dev_bad, nullptr };
    auto mp_bad = llama_model_default_params();
    mp_bad.devices = model_devs_bad;
    mp_bad.n_gpu_layers = 2;
    mp_bad.split_mode = LLAMA_SPLIT_MODE_NONE;
    llama_model * model_bad = llama_model_load_from_file(path, mp_bad);
    check(model_bad != nullptr, "meta state: bad model");
    llama_context * ctx_bad = model_bad != nullptr ? llama_init_from_model(model_bad, cp) : nullptr;
    check(ctx_bad != nullptr, "meta state: bad context");
    const auto h_before = capture_seq(ctx, 0, 0);
    if (ctx_bad != nullptr) {
        const auto d_bad = capture_seq(ctx_bad, 0, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
        uint32_t magic_bad = 0;
        if (d_bad.size() >= sizeof(magic_bad)) {
            std::memcpy(&magic_bad, d_bad.data(), sizeof(magic_bad));
        }
        check(magic_bad == 0x4f4e4445, "meta state: bad context device capture");
        const size_t set_bad_self = llama_state_seq_set_data_ext(
            ctx_bad, d_bad.data(), d_bad.size(), 0, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
        check(set_bad_self == d_bad.size(), "meta state: bad context device restore");
        const size_t set_bad = llama_state_seq_set_data_ext(
            ctx, d_bad.data(), d_bad.size(), 0, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
        check(set_bad == 0, "meta state: cross-context device blob rejected");
        const auto h_bad = capture_seq(ctx_bad, 0, 0);
        const size_t set_bad_host = llama_state_seq_set_data_ext(
            ctx, h_bad.data(), h_bad.size(), 0, 0);
        check(set_bad_host == h_bad.size(), "meta state: host format restores across contexts");
    }
    const auto h_after_bad = capture_seq(ctx, 0, 0);
    check(same_bytes(h_before, h_after_bad), "meta state: mismatched split leaves the old state");
    check(!h_after_bad.empty(), "meta state: context usable after mismatched split");
    if (ctx_bad != nullptr) {
        llama_free(ctx_bad);
    }
    if (model_bad != nullptr) {
        llama_model_free(model_bad);
    }

    llama_free(ctx);
    llama_model_free(model);
    return true;
}

int main(int argc, char ** argv) {
    llama_backend_init();
    if (!run_meta_buffer_copy_test()) {
        llama_backend_free();
        return 1;
    }
    const char * path = argc > 1 ? argv[1] : std::getenv("TP5_TEST_MODEL");
    if (path == nullptr || path[0] == '\0') {
        std::printf("SKIP no model\n");
        llama_backend_free();
        return 77;
    }
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(path, mp);
    if (model == nullptr) { std::printf("FAIL load model\n"); return 1; }
    auto cp = llama_context_default_params();
    cp.n_ctx = 64; cp.n_batch = 8; cp.n_ubatch = 8; cp.n_seq_max = 2; cp.n_threads = 1; cp.n_threads_batch = 1;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (ctx == nullptr) { std::printf("FAIL context\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    std::printf("n_vocab=%d\n", n_vocab);
    llama_token tok = llama_vocab_bos(vocab);
    if (tok < 0 || tok >= n_vocab) tok = n_vocab > 1 ? 1 : 0;
    int rc = decode_one(ctx, tok, 0);
    std::printf("decode0 rc=%d\n", rc);
    if (rc != 0) { std::printf("FAIL decode0\n"); return 1; }

    const auto dev = LLAMA_STATE_SEQ_FLAGS_ON_DEVICE;

    const std::vector<uint8_t> h1 = capture_seq(ctx, 0, 0);
    std::printf("host size=%zu\n", h1.size());
    check(!h1.empty(), "host capture returns the requested size");
    uint32_t seq_magic = 0;
    if (h1.size() >= sizeof(seq_magic)) {
        std::memcpy(&seq_magic, h1.data(), sizeof(seq_magic));
    }
    // The host sequence blob keeps its existing magic.
    if (seq_magic != 0xaf143cd8) {
        std::printf("SKIP incompatible sequence blob format\n");
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 77;
    }
    const std::vector<uint8_t> probe_dev = capture_seq(ctx, 0, dev);
    uint32_t dev_magic = 0;
    uint32_t dev_version = 0;
    if (probe_dev.size() >= 12) {
        std::memcpy(&dev_magic, probe_dev.data(), sizeof(dev_magic));
        std::memcpy(&dev_version, probe_dev.data() + 8, sizeof(dev_version));
    }
    if (dev_magic != 0x4f4e4445 || dev_version != 1) {
        std::printf("SKIP incompatible device sequence format\n");
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 77;
    }

    rc = decode_one(ctx, tok, 1);
    std::printf("decode1 rc=%d\n", rc);
    check(rc == 0, "decode1");
    const size_t hset = llama_state_seq_set_data_ext(ctx, h1.data(), h1.size(), 0, 0);
    std::printf("host set=%zu\n", hset);
    check(hset == h1.size(), "host set returns the captured size");
    const std::vector<uint8_t> h2 = capture_seq(ctx, 0, 0);
    std::printf("host get2=%zu equal=%d\n", h2.size(), same_bytes(h1, h2));
    check(same_bytes(h1, h2), "host round trip restores the exact bytes");

    const std::vector<uint8_t> hb = capture_seq(ctx, 0, 0);
    check(!hb.empty(), "host capture B");
    const size_t hcross_seq = llama_state_seq_set_data_ext(ctx, hb.data(), hb.size(), 1, 0);
    check(hcross_seq == hb.size(), "host state restores across sequences");
    const std::vector<uint8_t> h_seq1 = capture_seq(ctx, 1, 0);
    // PLE history serializes the source seq id, so a non-empty cross-seq host
    // payload is not byte-identical even when the state is equivalent. The
    // existing file/generation tests cover the semantic content; here we only
    // prove the old host blob remains accepted across sequences.
    check(!h_seq1.empty(), "cross-seq host restore produces a state");
    const std::vector<float> logits_seq1 = decode_logits(ctx, tok, 1, 1, n_vocab);
    check(!logits_seq1.empty(), "cross-seq host state decodes on seq 1");
    const size_t hreset = llama_state_seq_set_data_ext(ctx, hb.data(), hb.size(), 0, 0);
    check(hreset == hb.size(), "host state resets seq 0 for reference");
    const std::vector<float> logits_seq0 = decode_logits(ctx, tok, 1, 0, n_vocab);
    check(!logits_seq0.empty(), "reference state decodes on seq 0");
    bool same_logits = logits_seq1.size() == logits_seq0.size();
    if (same_logits) {
        for (size_t i = 0; i < logits_seq1.size(); ++i) {
            if (std::fabs(logits_seq1[i] - logits_seq0[i]) > 1e-5f) {
                same_logits = false;
                break;
            }
        }
    }
    check(same_logits, "cross-seq host state produces the same logits");
    const size_t hreset2 = llama_state_seq_set_data_ext(ctx, hb.data(), hb.size(), 0, 0);
    check(hreset2 == hb.size(), "host state resets seq 0 for device capture");
    const std::vector<uint8_t> d1 = capture_seq(ctx, 0, dev);
    std::printf("dev size=%zu\n", d1.size());
    check(!d1.empty(), "device get returns the requested metadata size");

    // The device snapshot was captured at position 0, so the next token after
    // the restore is position 1, not the position of the pre-restore decode.
    rc = decode_one(ctx, tok, 1);
    std::printf("decode2 rc=%d\n", rc);
    check(rc == 0, "decode2");
    const size_t dset = llama_state_seq_set_data_ext(ctx, d1.data(), d1.size(), 0, dev);
    std::printf("dev set=%zu\n", dset);
    check(dset == d1.size(), "device set returns the captured metadata size");
    const std::vector<uint8_t> h2b = capture_seq(ctx, 0, 0);
    std::printf("host after device restore=%zu equal_B=%d\n", h2b.size(), same_bytes(hb, h2b));
    check(same_bytes(hb, h2b), "device restore restores the exact host state B");

    // A short metadata buffer must return 0 and must not abort. Five bytes let
    // the first header field fit and leave one byte, so the header write fails
    // before any device copy; the previous snapshot must remain restorable.
    std::vector<uint8_t> short_buf(5);
    const size_t short_get = llama_state_seq_get_data_ext(ctx, short_buf.data(), short_buf.size(), 0, dev);
    std::printf("dev short get=%zu (expect 0)\n", short_get);
    check(short_get == 0, "device get with a short buffer returns 0");
    check(llama_state_seq_set_data_ext(ctx, d1.data(), d1.size(), 0, dev) == d1.size(),
          "device restore still works after a failed capture");

    // A second device capture for the same seq replaces mem_storage[seq].
    // Restoring the older metadata must be detected and return 0 instead of
    // silently copying the newer tensors.
    rc = decode_one(ctx, tok, 1);
    std::printf("decode3 rc=%d\n", rc);
    check(rc == 0, "decode3");
    const std::vector<uint8_t> d3 = capture_seq(ctx, 0, dev);
    std::printf("dev second get=%zu\n", d3.size());
    check(!d3.empty(), "device second capture");
    const size_t stale = llama_state_seq_set_data_ext(ctx, d1.data(), d1.size(), 0, dev);
    std::printf("dev stale restore=%zu (expect 0)\n", stale);
    check(stale == 0, "device restore of overwritten metadata returns 0");

    // The source seq comes from the header and dest_seq_id is the restore
    // target. A current snapshot must still restore across sequences.
    const size_t cross_seq = llama_state_seq_set_data_ext(ctx, d3.data(), d3.size(), 1, dev);
    std::printf("dev cross-seq restore=%zu\n", cross_seq);
    check(cross_seq == d3.size(), "device restore to another seq still works");

    // Change the source seq_id to one without a device snapshot. The rest of
    // the header stays current, so the lookup, not the generation check, is
    // the reason for 0.
    std::vector<uint8_t> missing = d3;
    const int32_t sid = 1;
    std::memcpy(missing.data() + 4, &sid, sizeof(sid));
    const size_t mret = llama_state_seq_set_data_ext(ctx, missing.data(), missing.size(), 0, dev);
    std::printf("dev missing set=%zu (expect 0)\n", mret);
    check(mret == 0, "device restore without a snapshot returns 0");

    // Truncated metadata with a current generation: the header parses, the
    // state read fails, and the reader destructor must not abort or copy a
    // partial state into the destination sequence.
    std::vector<uint8_t> truncated = d3;
    truncated.pop_back();
    const size_t tret = llama_state_seq_set_data_ext(ctx, truncated.data(), truncated.size(), 1, dev);
    std::printf("dev truncated set=%zu (expect 0)\n", tret);
    check(tret == 0, "truncated device metadata returns 0");

    // Host metadata with ON_DEVICE must be rejected by the device magic.
    const size_t hdev = llama_state_seq_set_data_ext(ctx, hb.data(), hb.size(), 1, dev);
    std::printf("host-as-device set=%zu (expect 0)\n", hdev);
    check(hdev == 0, "host metadata with ON_DEVICE returns 0");

    // Device metadata with host flags must be rejected by the host parser.
    const size_t dhost = llama_state_seq_set_data_ext(ctx, d3.data(), d3.size(), 1, 0);
    std::printf("device-as-host set=%zu (expect 0)\n", dhost);
    check(dhost == 0, "device metadata with host flags returns 0");

    // Device blobs have their own versioned magic. A blob with the host
    // sequence magic passed as ON_DEVICE is rejected before the memory state
    // is touched; the host format itself remains readable.
    const uint32_t host_magic = 0xaf143cd8;
    std::vector<uint8_t> old_device = d3;
    std::memcpy(old_device.data(), &host_magic, sizeof(host_magic));
    check(llama_state_seq_set_data_ext(ctx, old_device.data(), old_device.size(), 1, dev) == 0,
          "host-magic device blob returns 0");

    // Generation 0 is the get_size placeholder, never a committed snapshot.
    // Device layout: magic(4) seq_id(4) version(4) context_id(8) generation(8).
    constexpr size_t generation_offset = 4 + 4 + 4 + 8;
    std::vector<uint8_t> zero_generation = d3;
    std::memset(zero_generation.data() + generation_offset, 0, sizeof(uint64_t));
    const size_t zret = llama_state_seq_set_data_ext(ctx, zero_generation.data(), zero_generation.size(), 1, dev);
    std::printf("zero-generation set=%zu (expect 0)\n", zret);
    check(zret == 0, "zero generation returns 0");

    // A second context has a different identity; its metadata must not restore
    // into the first context even when both contexts have a snapshot for seq 0.
    llama_context * ctx2 = llama_init_from_model(model, cp);
    if (ctx2 == nullptr) {
        check(false, "second context");
    } else {
        check(decode_one(ctx2, tok, 0) == 0, "second context decode");
        const std::vector<uint8_t> d_ctx2 = capture_seq(ctx2, 0, dev);
        check(!d_ctx2.empty(), "second context device capture");
        const size_t cret = llama_state_seq_set_data_ext(ctx, d_ctx2.data(), d_ctx2.size(), 1, dev);
        std::printf("foreign-context set=%zu (expect 0)\n", cret);
        check(cret == 0, "device metadata from another context returns 0");
        const std::vector<uint8_t> h_ctx2 = capture_seq(ctx2, 0, 0);
        check(!h_ctx2.empty(), "second context host capture");
        const size_t hcross = llama_state_seq_set_data_ext(ctx, h_ctx2.data(), h_ctx2.size(), 0, 0);
        std::printf("cross-context host set=%zu\n", hcross);
        check(hcross == h_ctx2.size(), "host state restores across contexts");
        llama_free(ctx2);
    }

    // Invalid destination sequences must return 0, not reach a memory-module
    // assert. The same boundary applies to get_size/get_data.
    const size_t neg_set = llama_state_seq_set_data_ext(ctx, d3.data(), d3.size(), -2, dev);
    check(neg_set == 0, "sequence id below -1 returns 0");
    const size_t hi_set = llama_state_seq_set_data_ext(
        ctx, d3.data(), d3.size(), (llama_seq_id) cp.n_seq_max, dev);
    check(hi_set == 0, "out-of-range destination seq returns 0");
    check(llama_state_seq_get_size_ext(ctx, -2, dev) == 0, "below -1 get_size seq returns 0");
    std::vector<uint8_t> invalid_buf(16);
    check(llama_state_seq_get_data_ext(ctx, invalid_buf.data(), invalid_buf.size(), -2, dev) == 0,
          "below -1 get_data seq returns 0");

    // seq_id -1 addresses all sequences, matching the internal sequence state
    // paths. It must remain readable and writable for host and device blobs.
    const size_t all_host_size = llama_state_seq_get_size_ext(ctx, -1, 0);
    check(all_host_size > 0, "whole-sequence host size");
    std::vector<uint8_t> all_host(all_host_size);
    const size_t all_host_get = llama_state_seq_get_data_ext(
        ctx, all_host.data(), all_host.size(), -1, 0);
    check(all_host_get == all_host_size, "whole-sequence host get");
    const size_t all_host_set = llama_state_seq_set_data_ext(
        ctx, all_host.data(), all_host.size(), -1, 0);
    check(all_host_set == all_host_size, "whole-sequence host set");
    const size_t all_dev_size = llama_state_seq_get_size_ext(ctx, -1, dev);
    check(all_dev_size > 0, "whole-sequence device size");
    std::vector<uint8_t> all_dev(all_dev_size);
    const size_t all_dev_get = llama_state_seq_get_data_ext(
        ctx, all_dev.data(), all_dev.size(), -1, dev);
    check(all_dev_get == all_dev_size, "whole-sequence device get");
    const size_t all_dev_set = llama_state_seq_set_data_ext(
        ctx, all_dev.data(), all_dev.size(), -1, dev);
    check(all_dev_set == all_dev_size, "whole-sequence device set");

    // ON_DEVICE metadata with PARTIAL_ONLY must be rejected by the host sequence
    // magic before the memory module can misread it as a cell count.
    const std::vector<uint8_t> before_partial = capture_seq(ctx, 0, 0);
    check(!before_partial.empty(), "seq 0 state before PARTIAL_ONLY misuse");
    const size_t partial = llama_state_seq_set_data_ext(
        ctx, d3.data(), d3.size(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    check(partial == 0, "device metadata with PARTIAL_ONLY returns 0");
    const std::vector<uint8_t> after_partial = capture_seq(ctx, 0, 0);
    check(same_bytes(before_partial, after_partial),
          "PARTIAL_ONLY misuse leaves the target unchanged");

    // A corrupt host payload with an oversized cell count must fail before the
    // recurrent cache reserves a ubatch for it.
    if (before_partial.size() < 12) {
        check(false, "oversized cell count source available");
    } else {
        std::vector<uint8_t> huge_count = before_partial;
        const uint32_t huge_cells = 0xffffffffu;
        std::memcpy(huge_count.data() + 8, &huge_cells, sizeof(huge_cells));
        const size_t huge_ret = llama_state_seq_set_data_ext(ctx, huge_count.data(), huge_count.size(), 1, 0);
        std::printf("oversized cell count restore=%zu (expect 0)\n", huge_ret);
        check(huge_ret == 0, "oversized cell count returns 0");
    }

    // A host restore that fails in the recurrent section after the attention
    // section was already applied follows the memory clear-on-failure policy.
    const std::vector<uint8_t> full0 = capture_seq(ctx, 0, 0);
    check(!full0.empty(), "seq 0 host state before truncated restore");
    std::vector<uint8_t> trunc_host = full0;
    trunc_host.pop_back();
    const size_t hret = llama_state_seq_set_data_ext(ctx, trunc_host.data(), trunc_host.size(), 1, 0);
    std::printf("host truncated restore=%zu (expect 0)\n", hret);
    check(hret == 0, "truncated host metadata returns 0");

    // An empty source restored across sequences must clear the destination,
    // including its attention cells.
    check(llama_memory_seq_rm(llama_get_memory(ctx), 1, -1, -1), "clear seq 1 before empty source test");
    const std::vector<uint8_t> empty1_dev = capture_seq(ctx, 1, dev);
    check(!empty1_dev.empty(), "empty seq 1 device state");
    const std::vector<uint8_t> empty1_host = capture_seq(ctx, 1, 0);
    const std::vector<uint8_t> seq0_before_empty = capture_seq(ctx, 0, 0);
    check(!seq0_before_empty.empty(), "seq 0 state before empty restore");
    const size_t eret = llama_state_seq_set_data_ext(ctx, empty1_dev.data(), empty1_dev.size(), 0, dev);
    std::printf("empty cross-seq restore=%zu\n", eret);
    check(eret == empty1_dev.size(), "empty device state restores successfully");
    const std::vector<uint8_t> seq0_after_empty = capture_seq(ctx, 0, 0);
    check(same_payload(empty1_host, seq0_after_empty),
          "empty cross-seq restore clears the destination attention");

    // Quantized KV tensors must round-trip through the device view code. The
    // model may reject Q8_0 KV; skip only when the context does not use it.
    llama_context_params cp_q8 = cp;
    cp_q8.type_k = GGML_TYPE_Q8_0;
    cp_q8.type_v = GGML_TYPE_Q8_0;
    llama_context * ctx_q8 = llama_init_from_model(model, cp_q8);
    if (ctx_q8 == nullptr) {
        std::printf("SKIP Q8_0 KV context\n");
    } else {
        const enum ggml_type k_type = llama_get_kv_cache_type_k(ctx_q8);
        const enum ggml_type v_type = llama_get_kv_cache_type_v(ctx_q8);
        std::printf("Q8_0 KV context types: k=%d v=%d\n", (int) k_type, (int) v_type);
        if (k_type != GGML_TYPE_Q8_0 || v_type != GGML_TYPE_Q8_0) {
            std::printf("SKIP Q8_0 KV not used by this model\n");
        } else {
            check(decode_one(ctx_q8, tok, 0) == 0, "Q8_0 decode0");
            const std::vector<uint8_t> qh1 = capture_seq(ctx_q8, 0, 0);
            const std::vector<uint8_t> qd1 = capture_seq(ctx_q8, 0, dev);
            check(!qh1.empty() && !qd1.empty(), "Q8_0 host and device capture");
            check(decode_one(ctx_q8, tok, 1) == 0, "Q8_0 decode1");
            const size_t qset = llama_state_seq_set_data_ext(ctx_q8, qd1.data(), qd1.size(), 0, dev);
            check(qset == qd1.size(), "Q8_0 device restore");
            const std::vector<uint8_t> qh2 = capture_seq(ctx_q8, 0, 0);
            check(same_bytes(qh1, qh2), "Q8_0 device restore restores exact host state");
        }
        llama_free(ctx_q8);
    }

    llama_free(ctx);
    llama_model_free(model);
    check(run_state_meta_test(path), "meta state: state API");
    llama_backend_free();
    std::printf("%s failures=%d\n", fails ? "FAILED" : "OK", fails);
    return fails ? 1 : 0;
}
