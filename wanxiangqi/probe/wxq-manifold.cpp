// Raw activation sampler: intrinsic dimension needs samples, not moments.
//
// Everything measured so far about the expert input has been second order --
// covariance, its eigenbasis, log-det, Gauss-Newton = A (x) G. All of it is
// blind to the one thing that decides whether an arbitrary function family can
// halve the parameters: a thin curved manifold in high ambient dimension has
// near-full-rank covariance. A 1-D spiral in R^3 has a full-rank 3x3 covariance.
// So r_99(A) = 1877 of 2048 says nothing about how many degrees of freedom the
// activations actually have, and the whole "behaviour dimension = 0.944 of the
// parameter count" result rests on reading it as if it did.
//
// The long flat tail of the spectrum is one of three things:
//   * manifold curvature  -> tail coords are deterministic functions of the
//                            intrinsic coords: predictable, cheap for a
//                            nonlinear family, invisible to a linear one
//   * genuine noise       -> irrelevant, discardable
//   * independent signal  -> fatal, needs its own parameters
// Only the third kills parameter reduction, and only the third was assumed.
//
// What this dumps, per selected layer:
//   x    expert input          [n_samples, n_embd]   (src1 of the gate mul_mat_id)
//   h    post-SwiGLU hidden    [n_samples, n_ff]     (src1 of the down mul_mat_id, slot 0)
//   ids  selected experts      [n_samples, n_exp_used]
// so intrinsic dimension can be estimated on the input, on the hidden
// representation, and conditioned per expert.
//
// Traversal: each of the 42 requests is walked root->leaf as its own sequence
// with its own context reset. No branching, so no state snapshots. Sampling with
// a wide stride and spread across all requests, because nearest-neighbour
// dimension estimators are wrecked by temporally adjacent samples -- adjacent
// token activations are nearly identical and would drag any ID estimate down.

#include "llama.h"
#include "common.h"
#include "arg.h"
#include "log.h"
#include "ggml.h"
#include "ggml-backend.h"

#include "trie.h"

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using wxq::trie;
using wxq::load_trie;

// ---------------------------------------------------------------------------
// collector
// ---------------------------------------------------------------------------

struct layer_samples {
    uint32_t n_embd = 0, n_ff = 0, n_exp_used = 0;
    std::vector<float>   x;    // [n, n_embd]
    std::vector<float>   h;    // [n, n_ff]
    std::vector<int32_t> ids;  // [n, n_exp_used]
    uint64_t seen_x = 0, seen_h = 0;
};

struct collector {
    std::vector<int> layers;          // which blocks to sample
    uint32_t stride = 16;             // token stride
    uint64_t max_per_layer = 6000;
    std::map<int, layer_samples> out;
    std::vector<uint8_t> scratch;

    bool wanted(int il) const { return std::find(layers.begin(), layers.end(), il) != layers.end(); }

    enum class kind { none, gate, down };

    static kind kind_of(const ggml_tensor * t) {
        if (t->op != GGML_OP_MUL_MAT_ID) return kind::none;
        const char * n = t->src[0]->name;
        if (std::strstr(n, "ffn_gate_exps")) return kind::gate;
        if (std::strstr(n, "ffn_down_exps")) return kind::down;
        return kind::none;
    }

    static int layer_of(const char * name) {
        int il = -1;
        return std::sscanf(name, "blk.%d.", &il) == 1 ? il : -1;
    }

    bool want(const ggml_tensor * t) const {
        const kind k = kind_of(t);
        if (k == kind::none) return false;
        const int il = layer_of(t->src[0]->name);
        if (il < 0 || !wanted(il)) return false;
        auto it = out.find(il);
        if (it == out.end()) return true;
        return (k == kind::gate ? it->second.seen_x : it->second.seen_h) / stride < max_per_layer;
    }

    // read a [ne0, ne1, ne2] f32/f16 tensor row-block into floats
    bool fetch(const ggml_tensor * t, std::vector<float> & dst) {
        const size_t nb = ggml_nbytes(t);
        scratch.resize(nb);
        if (ggml_backend_buffer_is_host(t->buffer)) std::memcpy(scratch.data(), t->data, nb);
        else                                       ggml_backend_tensor_get(t, scratch.data(), 0, nb);
        const int64_t n = ggml_nelements(t);
        dst.resize((size_t) n);
        if (t->type == GGML_TYPE_F32) {
            std::memcpy(dst.data(), scratch.data(), (size_t) n * 4);
        } else if (t->type == GGML_TYPE_F16) {
            ggml_fp16_to_fp32_row((const ggml_fp16_t *) scratch.data(), dst.data(), n);
        } else {
            return false;
        }
        return true;
    }

    void observe(ggml_tensor * t) {
        const kind k = kind_of(t);
        if (k == kind::none) return;
        const int il = layer_of(t->src[0]->name);
        if (il < 0 || !wanted(il)) return;

        const ggml_tensor * src1 = t->src[1];
        const ggml_tensor * ids  = t->src[2];
        layer_samples & S = out[il];

        std::vector<float> v;
        if (!fetch(src1, v)) return;

        const int64_t d      = src1->ne[0];
        const int64_t slots  = src1->ne[1];
        const int64_t ntok   = src1->ne[2];

        if (k == kind::gate) {
            S.n_embd = (uint32_t) d;
            std::vector<float> vi;
            const bool have_ids = ids && fetch_ids(ids, vi);
            const int64_t nu = ids ? ids->ne[0] : 0;
            if (have_ids) S.n_exp_used = (uint32_t) nu;
            for (int64_t j = 0; j < ntok; ++j) {
                const uint64_t gi = S.seen_x++;
                if (gi % stride) continue;
                if (S.x.size() / std::max<size_t>(1, (size_t) d) >= max_per_layer) continue;
                const float * row = v.data() + (size_t) j*slots*d;   // slot 0
                S.x.insert(S.x.end(), row, row + d);
                if (have_ids) {
                    const int32_t * ir = (const int32_t *) (idbuf.data()) + (size_t) j*nu;
                    S.ids.insert(S.ids.end(), ir, ir + nu);
                }
            }
        } else {
            S.n_ff = (uint32_t) d;
            for (int64_t j = 0; j < ntok; ++j) {
                const uint64_t gi = S.seen_h++;
                if (gi % stride) continue;
                if (S.h.size() / std::max<size_t>(1, (size_t) d) >= max_per_layer) continue;
                const float * row = v.data() + (size_t) j*slots*d;   // expert slot 0
                S.h.insert(S.h.end(), row, row + d);
            }
        }
    }

    std::vector<uint8_t> idbuf;
    bool fetch_ids(const ggml_tensor * t, std::vector<float> &) {
        if (t->type != GGML_TYPE_I32) return false;
        const size_t nb = ggml_nbytes(t);
        idbuf.resize(nb);
        if (ggml_backend_buffer_is_host(t->buffer)) std::memcpy(idbuf.data(), t->data, nb);
        else                                       ggml_backend_tensor_get(t, idbuf.data(), 0, nb);
        return true;
    }
};

static bool cb_eval(ggml_tensor * t, bool ask, void * ud) {
    collector * c = (collector *) ud;
    if (ask) return c->want(t);
    c->observe(t);
    return true;
}

static void write_dump(const std::string & path, collector & c) {
    std::ofstream o(path, std::ios::binary | std::ios::trunc);
    o.write("WXQMAN01", 8);
    const uint32_t n_layers = (uint32_t) c.out.size();
    o.write((const char *) &n_layers, 4);
    o.write((const char *) &c.stride, 4);
    for (auto & [il, S] : c.out) {
        const uint32_t layer = (uint32_t) il;
        const uint32_t nx = S.n_embd ? (uint32_t) (S.x.size() / S.n_embd) : 0;
        const uint32_t nh = S.n_ff   ? (uint32_t) (S.h.size() / S.n_ff)   : 0;
        o.write((const char *) &layer, 4);
        o.write((const char *) &S.n_embd, 4);
        o.write((const char *) &S.n_ff, 4);
        o.write((const char *) &S.n_exp_used, 4);
        o.write((const char *) &nx, 4);
        o.write((const char *) &nh, 4);
        o.write((const char *) S.x.data(),   (std::streamsize) S.x.size()*4);
        o.write((const char *) S.h.data(),   (std::streamsize) S.h.size()*4);
        o.write((const char *) S.ids.data(), (std::streamsize) S.ids.size()*4);
        LOG_INF("%s: blk.%-2u  x %5u x %4u   h %5u x %4u   ids %u\n",
                __func__, layer, nx, S.n_embd, nh, S.n_ff, S.n_exp_used);
    }
    o.close();
}

int main(int argc, char ** argv) {
    common_params params;
    params.n_batch  = 2048;
    params.n_ubatch = 512;
    params.n_ctx    = 16384;

    std::string trie_dir, dump = "/tmp/opencode/manifold.bin";
    std::vector<int> layers = {0, 2, 8, 18, 28, 34, 38, 39};
    uint32_t stride = 16, cap = 2048;
    uint64_t max_per_layer = 6000;

    std::vector<char *> rest{argv[0]};
    for (int i = 1; i < argc; ++i) {
        const char * a = argv[i];
        auto val = [&]() -> const char * { if (i + 1 >= argc) throw std::runtime_error("missing value"); return argv[++i]; };
        if      (!std::strcmp(a, "--trie"))         trie_dir = val();
        else if (!std::strcmp(a, "--man-dump"))     dump = val();
        else if (!std::strcmp(a, "--man-stride"))   stride = (uint32_t) std::atoi(val());
        else if (!std::strcmp(a, "--man-cap"))      cap = (uint32_t) std::atoi(val());
        else if (!std::strcmp(a, "--man-max"))      max_per_layer = std::strtoull(val(), nullptr, 10);
        else if (!std::strcmp(a, "--man-layers")) {
            layers.clear();
            for (char * p = std::strtok((char *) val(), ","); p; p = std::strtok(nullptr, ",")) layers.push_back(std::atoi(p));
        } else rest.push_back(argv[i]);
    }

    common_init();
    if (!common_params_parse((int) rest.size(), rest.data(), params, LLAMA_EXAMPLE_COMMON)) return 1;
    if (trie_dir.empty()) { LOG_ERR("usage: %s -m MODEL --trie DIR [--man-layers a,b,c] [--man-stride N] [--man-cap N]\n", argv[0]); return 1; }

    collector coll;
    coll.layers = layers;
    coll.stride = stride ? stride : 1;
    coll.max_per_layer = max_per_layer;

    params.cb_eval           = cb_eval;
    params.cb_eval_user_data = &coll;

    llama_backend_init();
    llama_numa_init(params.numa);
    common_init_result_ptr ir = common_init_from_params(params);
    if (!ir || !ir->model() || !ir->context()) { LOG_ERR("failed to load model\n"); return 1; }
    llama_context * ctx = ir->context();
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(ir->model()));

    trie t = load_trie(trie_dir, n_vocab);
    LOG_INF("%s: %" PRIu64 " nodes, %zu requests; cap %u tok/request, stride %u\n",
            __func__, t.n_nodes(), t.leaves.size(), cap, coll.stride);

    llama_batch batch = llama_batch_init((int32_t) params.n_batch, 0, 1);
    const int64_t t0 = ggml_time_us();
    uint64_t decoded = 0;

    for (size_t r = 0; r < t.leaves.size(); ++r) {
        const std::vector<llama_token> toks = t.path_to(t.leaves[r], cap);
        llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
        llama_pos pos = 0;
        for (size_t off = 0; off < toks.size(); off += (size_t) params.n_batch) {
            const size_t n = std::min((size_t) params.n_batch, toks.size() - off);
            common_batch_clear(batch);
            for (size_t i = 0; i < n; ++i) common_batch_add(batch, toks[off + i], pos + (llama_pos) i, { 0 }, i + 1 == n);
            if (llama_decode(ctx, batch) != 0) { LOG_ERR("decode failed on request %zu\n", r); break; }
            pos += (llama_pos) n;
            decoded += n;
        }
        LOG_INF("%s: request %2zu/%zu  %5zu tok  (total %" PRIu64 ", %.0f tok/s)\n",
                __func__, r + 1, t.leaves.size(), toks.size(), decoded,
                (double) decoded / std::max(1e-6, (double) (ggml_time_us() - t0)/1e6));
    }

    write_dump(dump, coll);
    LOG_INF("%s: decoded %" PRIu64 " tokens -> %s\n", __func__, decoded, dump.c_str());

    llama_batch_free(batch);
    llama_backend_free();
    return 0;
}
