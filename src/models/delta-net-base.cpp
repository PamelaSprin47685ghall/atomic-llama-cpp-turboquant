#include "models.h"

#include "llama-impl.h"
#include "llama-memory-recurrent.h"
#include "../../ggml/src/ggml-impl.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>

// utility to get one slice from the third dimension
// input dim:  [x, y, c, b]
// output dim: [x, y, 1, b]
static ggml_tensor * get_slice_2d(ggml_context * ctx0, ggml_tensor * t, int64_t c) {
    return ggml_view_4d(ctx0, t, t->ne[0], t->ne[1], 1, t->ne[3],
        t->nb[1], t->nb[2], t->nb[3], t->nb[2] * c);
}

llm_build_delta_net_base::llm_build_delta_net_base(const llm_graph_params & params) : llm_graph_context(params) {}

static bool rerot_shared_rbb_enabled() {
    const char * mode_env = std::getenv("LLAMA_REROT_RBB_ABLATION");
    if (!mode_env) {
        return false;
    }
    const std::string mode(mode_env);
    return mode == "shared-rbb" || mode == "raw-redundant";
}

static ggml_tensor * rerot_rbb_rows(
        const llm_graph_input_rs::rbb_group_input & group,
        bool shared_rbb) {
    return shared_rbb ? group.public_rows : group.default_shared_rows;
}

std::pair<ggml_tensor *, ggml_tensor *> llm_build_delta_net_base::build_delta_net_chunking(
        ggml_tensor * q,
        ggml_tensor * k,
        ggml_tensor * v,
        ggml_tensor * g,
        ggml_tensor * b,
        ggml_tensor * s,
        int           il) {
    const int64_t S_k      = q->ne[0];
    const int64_t H_k      = q->ne[1];
    const int64_t n_tokens = q->ne[2];
    const int64_t n_seqs   = q->ne[3];

    const int64_t S_v = v->ne[0];
    const int64_t H_v = v->ne[1];
    const bool kda = (g->ne[0] == S_k && g->ne[1] == H_k);

    GGML_ASSERT(S_k == S_v);
    GGML_ASSERT(H_v % H_k == 0);

    GGML_ASSERT(q->ne[0] == S_k && q->ne[1] == H_k && q->ne[2] == n_tokens && q->ne[3] == n_seqs);
    GGML_ASSERT(k->ne[0] == S_k && k->ne[1] == H_k && k->ne[2] == n_tokens && k->ne[3] == n_seqs);
    GGML_ASSERT(v->ne[0] == S_v && v->ne[1] == H_v && v->ne[2] == n_tokens && v->ne[3] == n_seqs);

    GGML_ASSERT(g->ne[0] == 1   || g->ne[0] == S_v);
    GGML_ASSERT(                   g->ne[1] == H_v && g->ne[2] == n_tokens && g->ne[3] == n_seqs);
    GGML_ASSERT(b->ne[0] == 1   && b->ne[1] == H_v && b->ne[2] == n_tokens && b->ne[3] == n_seqs);
    GGML_ASSERT(s->ne[0] == S_v && s->ne[1] == S_v && s->ne[2] == H_v      && s->ne[3] == n_seqs);

    const float scale = 1.0f / sqrtf(S_k);

    q = ggml_scale(ctx0, q, scale);

    cb(q, "q_in", il);
    cb(k, "k_in", il);
    cb(v, "v_in", il);
    cb(b, "b_in", il);
    cb(g, "g_in", il);

    q = ggml_permute(ctx0, q, 0, 2, 1, 3); // [S_k, n_tokens, H_k, n_seqs]
    k = ggml_permute(ctx0, k, 0, 2, 1, 3); // [S_k, n_tokens, H_k, n_seqs]
    v = ggml_permute(ctx0, v, 0, 2, 1, 3); // [S_v, n_tokens, H_v, n_seqs]
    g = ggml_permute(ctx0, g, 0, 2, 1, 3); // [g_0, n_tokens, H_v, n_seqs]
    b = ggml_permute(ctx0, b, 0, 2, 1, 3); // [  1, n_tokens, H_v, n_seqs]

    const int CS = kda ? 16 : 64; // chunk size

    const int pad = (CS - n_tokens % CS) % CS;
    const int n_chunks = (n_tokens + pad) / CS;

    q = ggml_pad(ctx0, q, 0, pad, 0, 0);
    k = ggml_pad(ctx0, k, 0, pad, 0, 0);
    v = ggml_pad(ctx0, v, 0, pad, 0, 0);
    g = ggml_pad(ctx0, g, 0, pad, 0, 0);
    b = ggml_pad(ctx0, b, 0, pad, 0, 0);

    ggml_tensor * v_b = ggml_mul(ctx0, v, b);
    ggml_tensor * k_b = ggml_mul(ctx0, k, b);

    cb(v_b, "v_b", il);
    cb(k_b, "k_b", il);

    q   = ggml_reshape_4d(ctx0, q,   S_k, CS, n_chunks, H_k * n_seqs);
    k   = ggml_reshape_4d(ctx0, k,   S_k, CS, n_chunks, H_k * n_seqs);
    k_b = ggml_reshape_4d(ctx0, k_b, S_k, CS, n_chunks, H_v * n_seqs);
    v   = ggml_reshape_4d(ctx0, v,   S_v, CS, n_chunks, H_v * n_seqs);
    v_b = ggml_reshape_4d(ctx0, v_b, S_v, CS, n_chunks, H_v * n_seqs);

    g = ggml_reshape_4d(ctx0, g, g->ne[0], CS, n_chunks, H_v * n_seqs);
    b = ggml_reshape_4d(ctx0, b, 1,        CS, n_chunks, H_v * n_seqs);

    // [CS, g_0, n_chunks, H_v * n_seqs]
    // TODO: extend ggml_cumsum with axis parameter to avoid transpose
    ggml_tensor * g_cs = ggml_cumsum(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, g)));
    cb(g_cs, "g_cs", il);

    ggml_tensor * kb = nullptr;
    ggml_tensor * kq = nullptr;
    if (kda) {
        const int64_t CHB = n_chunks * H_k * n_seqs;

        ggml_tensor * g_cs_i = ggml_reshape_4d(ctx0, g_cs, CS, 1, S_k, CHB);  // [chunk_size, 1, S_k, CHB]
        ggml_tensor * g_cs_j = ggml_reshape_4d(ctx0, g_cs, 1, CS, S_k, CHB);  // [1, chunk_size, S_k, CHB]

        g_cs_j = ggml_repeat_4d(ctx0, g_cs_j, CS, CS, S_k, CHB);  // [1, chunk_size, S_k, CHB] -> [chunk_size, chunk_size, S_k, CHB]

        // decay_mask [chunk_size,chunk_size,S_k,CHB]
        ggml_tensor * decay_mask;
        decay_mask = ggml_sub(ctx0, g_cs_j, g_cs_i);
        decay_mask = ggml_tri(ctx0, decay_mask, GGML_TRI_TYPE_LOWER_DIAG);
        decay_mask = ggml_exp(ctx0, decay_mask);
        cb(decay_mask, "decay_mask", il);

        // decay_mask [S_k,BT_j,BT_i,CHB] *Note* second and third chunk_sizes are switched
        decay_mask = ggml_cont_4d(ctx0, ggml_permute(ctx0, decay_mask, 2, 1, 0, 3), S_k, CS, CS, CHB);

        ggml_tensor * k_b_i = ggml_reshape_4d(ctx0, k_b, S_k, CS,  1, CHB);
        ggml_tensor * k_j   = ggml_reshape_4d(ctx0, k,   S_k,  1, CS, CHB);
        ggml_tensor * q_i   = ggml_reshape_4d(ctx0, q,   S_k, CS,  1, CHB);

        ggml_tensor * decay_k_b_i = ggml_mul(ctx0, decay_mask, k_b_i);
        ggml_tensor * decay_q_i   = ggml_mul(ctx0, decay_mask, q_i);

        // decay_k_b_i [S,BT,BT,CHB] @ k_j [S,1,BT,CHB] = Akk [BT,1,BT,CHB]
        kb = ggml_mul_mat(ctx0, decay_k_b_i, k_j);
        kq = ggml_mul_mat(ctx0, decay_q_i,   k_j);

        kb = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_reshape_4d(ctx0, kb, CS, CS, n_chunks, H_v * n_seqs)));
        kq = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_reshape_4d(ctx0, kq, CS, CS, n_chunks, H_v * n_seqs)));
    } else {
        ggml_tensor * g_cs_i = g_cs;
        ggml_tensor * g_cs_j = ggml_reshape_4d(ctx0, g_cs, 1, CS, n_chunks, H_v * n_seqs);

        g_cs_j = ggml_repeat_4d(ctx0, g_cs_j, CS, CS, n_chunks, H_v * n_seqs);

        // [CS, CS, n_chunks, H_v * n_seqs]
        ggml_tensor * decay_mask;
        decay_mask = ggml_sub(ctx0, g_cs_j, g_cs_i);
        decay_mask = ggml_tri(ctx0, decay_mask, GGML_TRI_TYPE_LOWER_DIAG);
        decay_mask = ggml_exp(ctx0, decay_mask);
        cb(decay_mask, "decay_mask", il);

        // [CS, CS, n_chunks, H_k * n_seqs]
        kb = ggml_mul_mat(ctx0, k,  k_b);
        kb = ggml_mul    (ctx0, kb, decay_mask);

        // [CS, CS, n_chunks, H_k * n_seqs]
        kq = ggml_mul_mat(ctx0, k, q);
        kq = ggml_mul(ctx0, kq, decay_mask);
    }

    kq = ggml_tri(ctx0, kq, GGML_TRI_TYPE_LOWER_DIAG);
    cb(kq, "kq", il);

    // [CS, CS, n_chunks, H_k * n_seqs]
    ggml_tensor * attn;
    attn = ggml_tri(ctx0, kb, GGML_TRI_TYPE_LOWER);
    cb(attn, "attn", il);

    ggml_tensor * identity;
    identity = ggml_view_1d(ctx0, attn, CS, 0);
    identity = ggml_fill   (ctx0, identity, 1.0f);
    identity = ggml_diag   (ctx0, identity);

    ggml_tensor * lhs = ggml_add(ctx0, attn, identity);
    cb(lhs, "dnet_add_ch_lhs", il);

    attn = ggml_neg(ctx0, attn);
    cb(attn, "attn_pre_solve", il);

    ggml_tensor * lin_solve = ggml_solve_tri(ctx0, lhs, attn, true, true, false);
    attn = ggml_add(ctx0, lin_solve, identity);
    cb(attn, "dnet_add_ch_attn_solved", il); // [CS, CS, n_chunks, H_k * n_seqs]

    // [S_v, CS, n_chunks, H_v * n_seqs]
    v = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, v_b)), attn);

    // [CS, 1, n_chunks, H_v * n_seqs] KDA: [CS, S_k, n_chunks, H_v * n_seqs]
    ggml_tensor * g_exp = ggml_exp(ctx0, g_cs);

    k_b = ggml_cont(ctx0, ggml_transpose(ctx0, k_b));

    // [CS, S_k, n_chunks, H_k * n_seqs]
    ggml_tensor * kbg = ggml_mul(ctx0, k_b, g_exp);
    cb(kbg, "k_beta_g_exp", il);

    // [S_k, CS, n_chunks, H_k * n_seqs]
    ggml_tensor * k_cd = ggml_mul_mat(ctx0, kbg, attn);
    cb(k_cd, "k_cumdecay", il);

    // [1, CS, n_chunks, H_k * n_seqs] KDA: [S_k, CS, n_chunks, H_k * n_seqs]
    ggml_tensor * g_exp_t = ggml_cont(ctx0, ggml_transpose(ctx0, g_exp));
    ggml_tensor * q_g_exp = ggml_mul(ctx0, q, g_exp_t);

    // vectorized calculation of key_gdiff
    // improved from the chunked version:
    //   g_last = torch.clamp(g_cum[:, :, -1], max=50.0).exp().unsqueeze(-1).unsqueeze(-1)
    //   g_diff = torch.clamp(g_cum[:, :, -1:] - g_cum, max=50.0).exp()
    //   key_gdiff = key * g_diff.unsqueeze(-1)
    //   kgdmulvnew = (key_gdiff).transpose(-1, -2) @ v_new
    //   last_recurrent_state = last_recurrent_state * g_last + kgdmulvnew

    // get last element in g_cumsum along CS dimension (ne0)
    // example: [[x, y, z, ..., last], ...] -> [[last], ...]
    // [1, 1, n_chunks, H_v * n_seqs] KDA: [1, S_k, n_chunks, H_v * n_seqs]
    ggml_tensor * g_last = ggml_view_4d(ctx0, g_cs, 1, g_cs->ne[1], g_cs->ne[2], g_cs->ne[3],
            g_cs->nb[1],
            g_cs->nb[2],
            g_cs->nb[3],
            ggml_row_size(g_cs->type, g_cs->ne[0] - 1));
    cb(g_last, "g_last", il);

    // TODO: remove this cont when CUDA supports non-cont unary ops
    g_last = ggml_cont(ctx0, g_last);

    // [1, 1, n_chunks, H_v * n_seqs] KDA: [S_k, 1, n_chunks, H_v * n_seqs]
    ggml_tensor * g_last_exp_t = ggml_transpose(ctx0, ggml_exp(ctx0, g_last));
    cb(g_last_exp_t, "g_last_exp_t", il);

    // [CS, 1, n_chunks, H_v * n_seqs] KDA: [CS, S_k, n_chunks, H_v * n_seqs]
    ggml_tensor * g_diff = ggml_neg(ctx0, ggml_sub(ctx0, g_cs, g_last));
    cb(g_diff, "g_diff", il);

    ggml_tensor * g_diff_exp_t = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_exp(ctx0, g_diff)));

    // [S_k, CS, n_chunks, H_v * n_seqs]
    ggml_tensor * kg = ggml_mul(ctx0, k, g_diff_exp_t);
    cb(kg, "key_gdiff", il);

    // [CS, S_k, n_chunks, H_v * n_seqs]
    ggml_tensor * kg_t = ggml_cont(ctx0, ggml_transpose(ctx0, kg));
    cb(kg_t, "key_gdiff_t", il);

    s = ggml_reshape_4d(ctx0, s, S_v, S_v, 1, H_v * n_seqs);
    cb(s, "dnet_add_ch_state", il);

    // [CS, S_v, n_chunks, H_v * n_seqs]
    ggml_tensor * v_t = ggml_cont(ctx0, ggml_transpose(ctx0, v));

    for (int64_t chunk = 0; chunk < n_chunks; chunk++) {
        ggml_tensor * ch_k_cd    = get_slice_2d(ctx0, k_cd,    chunk); // [S_k,  CS, 1, H_k * n_seqs]
        ggml_tensor * ch_v_t     = get_slice_2d(ctx0, v_t,     chunk); // [ CS, S_v, 1, H_v * n_seqs]
        ggml_tensor * ch_kq      = get_slice_2d(ctx0, kq,      chunk); // [ CS,  CS, 1, H_k * n_seqs]
        ggml_tensor * ch_q_g_exp = get_slice_2d(ctx0, q_g_exp, chunk); // [S_k,  CS, 1, H_k * n_seqs]
        ggml_tensor * ch_kg_t    = get_slice_2d(ctx0, kg_t,    chunk); // [ CS, S_k, 1, H_v * n_seqs]

        // [CS, S_v, 1, H_v * n_seqs]
        ggml_tensor * v_t_p = ggml_mul_mat(ctx0, ch_k_cd, s);
        cb(v_t_p, "v_prime", il);

        // [CS, S_v, 1, H_v * n_seqs]
        ggml_tensor * v_t_new = ggml_sub(ctx0, ch_v_t, v_t_p);
        cb(v_t_new, "v_t_new", il);

        // [S_v, CS, 1, H_v * n_seqs]
        ggml_tensor * v_attn = ggml_mul_mat(ctx0, v_t_new, ch_kq);
        cb(v_attn, "v_attn", il);

        // [S_v, CS, 1, H_v * n_seqs]
        ggml_tensor * attn_inter = ggml_mul_mat(ctx0, s, ch_q_g_exp);
        cb(attn_inter, "attn_inter", il);

        // [S_v, CS, 1, H_v * n_seqs]
        ggml_tensor * o_ch = ggml_add(ctx0, attn_inter, v_attn);
        cb(o_ch, "dnet_add_ch_attn_out", il);

        v = ggml_set_inplace(ctx0, v, o_ch, v->nb[1], v->nb[2], v->nb[3], chunk * v->nb[2]);

        // kgdmulvnew = (key_gdiff).transpose(-1, -2) @ v_new
        // TODO: head broadcast might not work here - probably will need a transpose
        ggml_tensor * kgv = ggml_mul_mat(ctx0, ch_kg_t, v_t_new); // [S_k, S_v, 1, H_k * n_seqs]

        // last_recurrent_state = last_recurrent_state * g_last + kgdmulvnew
        ggml_tensor * ch_g_last_exp_t = get_slice_2d(ctx0, g_last_exp_t, chunk);

        s = ggml_mul(ctx0, s, ch_g_last_exp_t);
        s = ggml_add(ctx0, s, kgv);
        cb(s, "dnet_add_ch_state", il);
    }

    // truncate padded tokens
    ggml_tensor * o = ggml_view_4d(ctx0, v,
            S_v, n_tokens, H_v, n_seqs,
            ggml_row_size(v->type, S_v),
            ggml_row_size(v->type, S_v * CS * n_chunks),
            ggml_row_size(v->type, S_v * CS * n_chunks * H_v), 0);
    o = ggml_permute  (ctx0, o, 0, 2, 1, 3); // [S_v, H_v, n_tokens, n_seqs]
    s = ggml_reshape_4d(ctx0, s, S_v, S_v, H_v, n_seqs);
    cb(s, "output_state", il);

    return {o, s};
}

std::pair<ggml_tensor *, ggml_tensor *> llm_build_delta_net_base::build_delta_net_autoregressive(
        ggml_tensor * q,
        ggml_tensor * k,
        ggml_tensor * v,
        ggml_tensor * g,
        ggml_tensor * b, // beta
        ggml_tensor * s, // state
        int           il) {
    const int64_t S_k      = q->ne[0];
    const int64_t H_k      = q->ne[1];
    const int64_t n_tokens = q->ne[2];
    const int64_t n_seqs   = q->ne[3];

    const int64_t S_v = v->ne[0];
    const int64_t H_v = v->ne[1];

    GGML_ASSERT(n_tokens == 1);

    GGML_ASSERT(S_k == S_v);
    GGML_ASSERT(H_v % H_k == 0);

    GGML_ASSERT(q->ne[0] == S_k && q->ne[1] == H_k && q->ne[2] == n_tokens && q->ne[3] == n_seqs);
    GGML_ASSERT(k->ne[0] == S_k && k->ne[1] == H_k && k->ne[2] == n_tokens && k->ne[3] == n_seqs);
    GGML_ASSERT(v->ne[0] == S_v && v->ne[1] == H_v && v->ne[2] == n_tokens && v->ne[3] == n_seqs);

    GGML_ASSERT(g->ne[0] == 1   || g->ne[0] == S_v);
    GGML_ASSERT(                   g->ne[1] == H_v && g->ne[2] == n_tokens && g->ne[3] == n_seqs);
    GGML_ASSERT(b->ne[0] == 1   && b->ne[1] == H_v && b->ne[2] == n_tokens && b->ne[3] == n_seqs);
    GGML_ASSERT(s->ne[0] == S_v && s->ne[1] == S_v && s->ne[2] == H_v      && s->ne[3] == n_seqs);

    const float scale = 1.0f / sqrtf(S_k);

    q = ggml_scale(ctx0, q, scale);

    q = ggml_permute(ctx0, q, 0, 2, 1, 3); // [S_k, n_tokens, H_k, n_seqs]
    k = ggml_permute(ctx0, k, 0, 2, 1, 3); // [S_k, n_tokens, H_k, n_seqs]
    v = ggml_permute(ctx0, v, 0, 2, 1, 3); // [S_v, n_tokens, H_v, n_seqs]

    cb(q, "q_in", il);
    cb(k, "k_in", il);
    cb(v, "v_in", il);
    cb(b, "b_in", il);
    cb(g, "g_in", il);

    // GDA: [1,  1,  H_v, n_seqs]
    // KDA: [1, S_k, H_v, n_seqs]
    g = ggml_reshape_4d(ctx0, g, 1, g->ne[0], H_v, n_seqs);
    b = ggml_reshape_4d(ctx0, b, 1,        1, H_v, n_seqs);

    // [S_v, S_v, H_v, n_seqs]
    g = ggml_exp(ctx0, g);
    s = ggml_mul(ctx0, s, g);

    // [1, S_v, H_v, n_seqs]
    ggml_tensor * sk;
    sk = ggml_mul     (ctx0, s, k);
    sk = ggml_sum_rows(ctx0, sk);

    // [S_v, 1, H_v, n_seqs]
    ggml_tensor * d;
    d = ggml_sub(ctx0, v, ggml_transpose(ctx0, sk));
    d = ggml_mul(ctx0, d, b);

    // [1, S_v, H_v, n_seqs]
    ggml_tensor * d_t;
    d_t = ggml_transpose(ctx0, d);

    // [S_v, S_v, H_v, n_seqs]
    ggml_tensor * kd;
    k  = ggml_repeat(ctx0, k, s);
    kd = ggml_mul   (ctx0, k, d_t);

    s = ggml_add(ctx0, s, kd);

    cb(s, "dnet_add_ar_state", il);

    ggml_tensor * s_q = ggml_mul     (ctx0, s, q);
    ggml_tensor * o   = ggml_sum_rows(ctx0, s_q);

    o = ggml_permute  (ctx0, o, 2, 0, 1, 3); // [S_v, H_v, n_tokens, n_seqs]

    return {o, s};
}

std::pair<ggml_tensor *, ggml_tensor *> llm_build_delta_net_base::build_delta_net_fused(
        ggml_tensor * q,
        ggml_tensor * k,
        ggml_tensor * v,
        ggml_tensor * g,
        ggml_tensor * b,
        ggml_tensor * s,
        int           il) {
    const int64_t S_k      = q->ne[0];
    const int64_t H_k      = q->ne[1];
    const int64_t n_tokens = q->ne[2];
    const int64_t n_seqs   = q->ne[3];

    const int64_t S_v = v->ne[0];
    const int64_t H_v = v->ne[1];

    GGML_ASSERT(S_k == S_v);
    GGML_ASSERT(H_v % H_k == 0);

    GGML_ASSERT(q->ne[0] == S_k && q->ne[1] == H_k && q->ne[2] == n_tokens && q->ne[3] == n_seqs);
    GGML_ASSERT(k->ne[0] == S_k && k->ne[1] == H_k && k->ne[2] == n_tokens && k->ne[3] == n_seqs);
    GGML_ASSERT(v->ne[0] == S_v && v->ne[1] == H_v && v->ne[2] == n_tokens && v->ne[3] == n_seqs);

    GGML_ASSERT(g->ne[0] == 1   || g->ne[0] == S_v);
    GGML_ASSERT(                   g->ne[1] == H_v && g->ne[2] == n_tokens && g->ne[3] == n_seqs);
    GGML_ASSERT(b->ne[0] == 1   && b->ne[1] == H_v && b->ne[2] == n_tokens && b->ne[3] == n_seqs);
    GGML_ASSERT(s->ne[0] == S_v && s->ne[1] == S_v && s->ne[2] == H_v      && s->ne[3] == n_seqs);

    // K=1: output carries the final state only. state s is 4D [S_v, S_v, H_v, n_seqs].
    ggml_tensor * result = ggml_gated_delta_net(ctx0, q, k, v, g, b, s, /*K=*/1);
    if (n_tokens == 1) {
        res->add_fused_node({LLM_FUSED_OP_GDN_AR, result, il});
    } else {
        res->add_fused_node({LLM_FUSED_OP_GDN_CH, result, il});
    }

    ggml_tensor * output = ggml_view_4d(ctx0, result,
            S_v, H_v, n_tokens, n_seqs,
            ggml_row_size(result->type, S_v),
            ggml_row_size(result->type, S_v * H_v),
            ggml_row_size(result->type, S_v * H_v * n_tokens), 0);

    ggml_tensor * new_state = ggml_view_4d(ctx0, result,
            S_v, S_v, H_v, n_seqs,
            ggml_row_size(result->type, S_v),
            ggml_row_size(result->type, S_v * S_v),
            ggml_row_size(result->type, S_v * S_v * H_v),
            ggml_row_size(result->type, S_v * H_v * n_tokens * n_seqs));

    return {output, new_state};
}

llm_build_delta_net_base::delta_net_rbb_result
llm_build_delta_net_base::build_delta_net_rbb(
        ggml_tensor * q,
        ggml_tensor * k,
        ggml_tensor * v,
        ggml_tensor * g,
        ggml_tensor * b,
        ggml_tensor * brain_state,
        ggml_tensor * native_state,
        int           il) {
    const int64_t S_v = v->ne[0];
    const int64_t H_v = v->ne[1];
    const int64_t n_tokens = v->ne[2];
    const int64_t n_seqs = v->ne[3];

    GGML_ASSERT(n_tokens == 1);
    GGML_ASSERT(n_seqs >= 1);
    GGML_ASSERT(g->ne[0] == 1);
    brain_state =
        ggml_reshape_4d(ctx0, brain_state, S_v, S_v, H_v, 1);
    native_state =
        ggml_reshape_4d(ctx0, native_state, S_v, S_v, H_v, n_seqs);

    ggml_tensor * result =
        ggml_gated_delta_net_rbb(
            ctx0, q, k, v, g, b, brain_state, native_state);
    const char * rbb_mode = std::getenv("LLAMA_REROT_RBB_ABLATION");
    const bool raw_redundant = rbb_mode && std::string(rbb_mode) == "raw-redundant";
    ggml_set_op_params_i32(result, 2, raw_redundant ? 0 : 1);
    cb(result, "rerot_block_result", il);
    res->add_fused_node({LLM_FUSED_OP_GDN_AR, result, il});

    ggml_tensor * output = ggml_view_4d(
        ctx0, result,
        S_v, H_v, n_tokens, n_seqs,
        ggml_row_size(result->type, S_v),
        ggml_row_size(result->type, S_v * H_v),
        ggml_row_size(result->type, S_v * H_v * n_tokens),
        0);

    const size_t output_bytes =
        ggml_row_size(result->type, S_v * H_v * n_tokens * n_seqs);
    const size_t state_bytes =
        ggml_row_size(result->type, S_v * S_v * H_v);

    ggml_tensor * merged_state = ggml_view_4d(
        ctx0, result,
        S_v, S_v, H_v, 1,
        ggml_row_size(result->type, S_v),
        ggml_row_size(result->type, S_v * S_v),
        state_bytes,
        output_bytes);
    ggml_tensor * hand_deltas = ggml_view_4d(
        ctx0, result,
        S_v, S_v, H_v, n_seqs,
        ggml_row_size(result->type, S_v),
        ggml_row_size(result->type, S_v * S_v),
        state_bytes,
        output_bytes + state_bytes);
    return {output, merged_state, hand_deltas};
}

std::pair<ggml_tensor *, ggml_tensor *> llm_build_delta_net_base::build_delta_net(
        ggml_tensor * q,
        ggml_tensor * k,
        ggml_tensor * v,
        ggml_tensor * g,
        ggml_tensor * b,
        ggml_tensor * s,
        int           il) {
    const int64_t n_seq_tokens = q->ne[2];

    if (n_seq_tokens == 1) {
        if (cparams.fused_gdn_ar) {
            return build_delta_net_fused(q, k, v, g, b, s, il);
        }
        return build_delta_net_autoregressive(q, k, v, g, b, s, il);
    }

    if (cparams.fused_gdn_ch) {
        return build_delta_net_fused(q, k, v, g, b, s, il);
    }

    return build_delta_net_chunking(q, k, v, g, b, s, il);
}

ggml_tensor * llm_build_delta_net_base::build_conv_state(
        llm_graph_input_rs * inp,
        ggml_tensor *        conv_states_all,
        ggml_tensor *        qkv_mixed,
        int64_t              conv_kernel_size,
        int64_t              conv_channels,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const auto kv_head  = mctx_cur->get_head();
    const auto mem_size = mctx_cur->get_size();

    const int64_t n_seqs = ubatch.n_seqs;

    ggml_tensor * conv_states = build_rs(inp, conv_states_all, hparams.n_embd_r(), n_seqs);
    cb(conv_states, "conv_states", il);

    conv_states = ggml_reshape_3d(ctx0, conv_states, conv_kernel_size - 1, conv_channels, n_seqs);
    cb(conv_states, "conv_states_reshaped", il);

    qkv_mixed = ggml_transpose(ctx0, qkv_mixed);
    cb(qkv_mixed, "qkv_mixed_transposed", il);

    ggml_tensor * conv_input = ggml_concat(ctx0, conv_states, qkv_mixed, 0);
    cb(conv_input, "conv_input", il);

    const int64_t row_count = (conv_kernel_size - 1) * conv_channels;

    const size_t row_size  = ggml_row_size(conv_states_all->type, row_count);

    if (cparams.n_rs_seq == 0) {
        const int64_t s_idx  = conv_input->ne[0] - conv_states->ne[0];
        const int64_t s_slot = 0;

        ggml_tensor * conv_state_last =
            ggml_view_3d(ctx0, conv_input,
                    conv_kernel_size - 1, conv_channels, n_seqs,
                    conv_input->nb[1], conv_input->nb[2],
                    ggml_row_size(conv_input->type, s_idx));
        cb(conv_state_last, "conv_state_last", il);

        ggml_tensor * conv_state_update =
            ggml_view_2d(ctx0, conv_states_all,
                    row_count, n_seqs, conv_states_all->nb[1],
                    (s_slot * mem_size + kv_head) * row_size);
        cb(conv_state_update, "conv_state_update", il);

        ggml_build_forward_expand(gf, ggml_cpy(ctx0, conv_state_last, conv_state_update));
    } else {
        // [TAG_RECURRENT_ROLLBACK_SPLITS]
        // this logic assumes that the last (n_rs_seq + 1) tokens of a sequence in a batch are inside
        //   the same ubatch, which `split_equal()` guarantees via its n_keep_tail argument

        const int64_t K = (int64_t) cparams.n_rs_seq + 1;

        for (int64_t t = 1; t <= K; ++t) {
            const int64_t s_idx  = std::max<int64_t>(0, conv_input->ne[0] - conv_states->ne[0] - K + t);
            const int64_t s_slot = K - t;

            ggml_tensor * conv_state_last =
                ggml_view_3d(ctx0, conv_input,
                        conv_kernel_size - 1, conv_channels, n_seqs,
                        conv_input->nb[1], conv_input->nb[2],
                        ggml_row_size(conv_input->type, s_idx));

            ggml_tensor * conv_state_update =
                ggml_view_2d(ctx0,
                        conv_states_all, row_count, n_seqs,
                        conv_states_all->nb[1],
                        (s_slot * mem_size + kv_head) * row_size);

            ggml_build_forward_expand(gf, ggml_cpy(ctx0, conv_state_last, conv_state_update));
        }
    }

    return conv_input;
}

void llm_build_delta_net_base::build_rbb_parallel_commit(
        llm_graph_input_rs * inp,
        ggml_tensor *        ssm_states_all,
        ggml_tensor *        candidate_states,
        int64_t              n_snapshots,
        int                  il) {
    const auto * mctx_cur = inp->mctx;
    GGML_ASSERT(mctx_cur->is_s_shared(il));
    GGML_ASSERT(inp->brain_copy != nullptr);
    GGML_ASSERT(n_snapshots > 0);

    const int64_t D = hparams.n_embd_s();
    const int64_t n_seqs = ubatch.n_seqs;
    const uint32_t brain_size = mctx_cur->get_brain_size();
    const size_t row_size =
        ggml_row_size(ssm_states_all->type, hparams.n_embd_s());

    ggml_tensor * candidates =
        ggml_reshape_3d(ctx0, candidate_states, D, n_seqs, n_snapshots);

    const bool shared_rbb = rerot_shared_rbb_enabled();

    for (const auto & group : inp->rbb_groups) {
        GGML_ASSERT(group.brain_row >= 0);
        GGML_ASSERT((uint32_t) group.brain_row < brain_size);
        GGML_ASSERT(group.public_rows != nullptr);
        GGML_ASSERT(group.public_rows->ne[0] > 0);

        ggml_tensor * rows = rerot_rbb_rows(group, shared_rbb);
        if (!rows || rows->ne[0] == 0) {
            continue;
        }

        for (int64_t snapshot = 0; snapshot < n_snapshots; ++snapshot) {
            ggml_tensor * candidate_snapshot = ggml_view_2d(
                ctx0, candidates, D, n_seqs, candidates->nb[1],
                (size_t) snapshot * candidates->nb[2]);
            ggml_tensor * selected =
                ggml_get_rows(ctx0, candidate_snapshot, rows);
            ggml_tensor * reduced = selected;
            if (rows->ne[0] > 1) {
                ggml_tensor * transposed =
                    ggml_cont(ctx0, ggml_transpose(ctx0, selected));
                reduced = ggml_sum_rows(ctx0, transposed);
            }
            reduced = ggml_reshape_1d(ctx0, reduced, D);

            const uint32_t public_brains = brain_size / 2;
            if (snapshot > 0 &&
                (uint32_t) group.brain_row >= public_brains) {
                continue;
            }
            const size_t destination_row = snapshot == 0
                ? (uint32_t) group.brain_row
                : (size_t) brain_size +
                    (size_t) (snapshot - 1) * public_brains +
                    (uint32_t) group.brain_row;
            ggml_tensor * destination = ggml_view_1d(
                ctx0, ssm_states_all, D, destination_row * row_size);

            if (rows->ne[0] > 1) {
                // The candidate-state path uses the documented symmetric
                // mean; scaling deltas by 1/sqrt(N) amplifies concurrent writes.
                reduced = ggml_scale(
                    ctx0, reduced, 1.0f / (float) rows->ne[0]);
            }
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, reduced, destination));
        }
    }
}

bool llm_build_delta_net_base::uses_parallel_delta(
        const llm_graph_input_rs * inp, int il) const {
    if (!inp->mctx->is_s_shared(il) ||
        cparams.n_rs_seq != 0 ||
        ubatch.n_seq_tokens != 1 ||
        ubatch.n_seqs < 1 ||
        inp->rbb_groups.size() != 1) {
        return false;
    }
    ggml_tensor * rows = rerot_rbb_rows(inp->rbb_groups[0], rerot_shared_rbb_enabled());
    return rows != nullptr &&
        rows->ne[0] == ubatch.n_seqs;
}

ggml_tensor * llm_build_delta_net_base::build_recurrent_attn(
        llm_graph_input_rs * inp,
        ggml_tensor *        ssm_states_all,
        ggml_tensor *        state_base,
        ggml_tensor *        hand_echo_all,
        ggml_tensor *        q,
        ggml_tensor *        k,
        ggml_tensor *        v,
        ggml_tensor *        g,
        ggml_tensor *        b,
        ggml_tensor *        s,
        int                  il) {
    const auto * mctx_cur   = inp->mctx;
    const auto   kv_head    = mctx_cur->get_head();
    const uint32_t mem_size = mctx_cur->get_size();

    const int64_t S_v          = v->ne[0];
    const int64_t H_v          = v->ne[1];
    const int64_t n_seqs       = v->ne[3];
    const int64_t n_seq_tokens = q->ne[2];

    const bool keep = cparams.n_rs_seq > 0;
    // StateCarryFix instrumented run: per-ubatch-per-layer carry geometry.
    // NOTE: s_copy values are deliberately NOT read here — s_copy() resets
    // rollback selectors (llama-memory-recurrent.cpp), so values are logged
    // at set_input time instead (see [statecarry] set_input line).
    const int64_t K_log = keep ? (int64_t) cparams.n_rs_seq + 1 : -1;
    const int64_t n_written_log = keep ? std::min<int64_t>(n_seq_tokens, K_log) : -1;
    LLAMA_LOG_ERROR("[statecarry] build_recurrent_attn il=%d n_seq_tokens=%lld n_seqs=%lld keep=%d K=%lld n_written=%lld kv_head=%u n_rs=%u rs_z=%d s_copy_ne=%lld\n",
        il, (long long) n_seq_tokens, (long long) n_seqs, (int) keep,
        (long long) K_log, (long long) n_written_log, (unsigned) kv_head,
        (unsigned) mctx_cur->get_n_rs(), (int) mctx_cur->get_rs_z(),
        (long long) (inp && inp->s_copy ? inp->s_copy->ne[0] : -1));

    // Exact trained recurrence is the default for every child row, regardless
    // of what the scheduler co-batches beside it. DDVR is already the
    // cross-lane memory channel; blindly re-encoding DDVR-conditioned hidden
    // states into one global recurrent brain double-counts peer evidence, and
    // the innovation component is not identifiable from q/k/v alone.
    // Shared RBB therefore remains an explicit research mode until such an
    // innovation operator is mathematically defined. The row membership is
    // carried by graph inputs; it must never be inferred from "all rows happen
    // to be children in this ubatch".
    const bool shared_rbb = rerot_shared_rbb_enabled();

    if (uses_parallel_delta(inp, il)) {
        GGML_ASSERT(g->ne[0] == 1);
        GGML_ASSERT(state_base != nullptr);
        GGML_ASSERT(hand_echo_all != nullptr);

        // PUBLIC rows advance the shared brain through one order-free block
        // transition. Private hand overlays evolve independently; each reader
        // uses the new shared brain plus its own overlay. The overlay must not
        // subtract the shared update and cancel peer influence.
        const auto rbb =
            build_delta_net_rbb(q, k, v, g, b, state_base, s, il);
        cb(rbb.output, "attn_output", il);
        cb(rbb.merged_state, "new_state", il);

        const auto & group = inp->rbb_groups[0];
        ggml_tensor * rows = rerot_rbb_rows(group, shared_rbb);
        GGML_ASSERT(rows != nullptr && rows->ne[0] == n_seqs);
        const size_t row_size =
            ggml_row_size(ssm_states_all->type, hparams.n_embd_s());
        ggml_tensor * destination = ggml_view_1d(
            ctx0,
            ssm_states_all,
            hparams.n_embd_s(),
            (size_t) group.brain_row * row_size);
        ggml_build_forward_expand(
            gf, ggml_cpy(ctx0, rbb.merged_state, destination));

        ggml_tensor * hand_destination = ggml_view_2d(
            ctx0,
            hand_echo_all,
            hparams.n_embd_s(),
            n_seqs,
            hand_echo_all->nb[1],
            (size_t) kv_head * hand_echo_all->nb[1]);
        ggml_tensor * hand_delta = ggml_reshape_2d(
            ctx0,
            rbb.hand_deltas,
            hparams.n_embd_s(),
            n_seqs);
        ggml_build_forward_expand(
            gf, ggml_cpy(ctx0, hand_delta, hand_destination));
        return rbb.output;
    }

    if (!keep && mctx_cur->is_s_shared(il) && n_seq_tokens == 1 && !inp->rbb_groups.empty()) {
        // Mixed visibility and multi-person batches are resolved per row. In
        // the default contract child rows keep native lane-local recurrence;
        // only non-child rows may advance a shared brain. Explicit shared-RBB
        // research modes opt child rows back into the block operator. Thus the
        // same logical child transition is invariant to scheduler composition.
        const int64_t D = hparams.n_embd_s();
        GGML_ASSERT(state_base != nullptr && hand_echo_all != nullptr);
        ggml_tensor * base_rows = ggml_cont(ctx0, ggml_reshape_2d(ctx0, state_base, D, n_seqs));
        const auto native = build_delta_net(q, k, v, g, b, s, il);
        ggml_tensor * outputs = ggml_cont(ctx0, ggml_reshape_2d(ctx0, native.first, S_v * H_v, n_seqs));
        // A non-committing PRIVATE/PENDING row retains its entire local
        // transition relative to the unchanged input brain, not just A_i H_i.
        ggml_tensor * hands = ggml_sub(ctx0,
            ggml_reshape_2d(ctx0, native.second, D, n_seqs), base_rows);
        ggml_build_forward_expand(gf, outputs);
        ggml_build_forward_expand(gf, hands);

        auto select = [&](ggml_tensor * tensor, ggml_tensor * rows) {
            GGML_ASSERT(tensor->ne[3] == n_seqs);
            ggml_tensor * flat = ggml_reshape_2d(ctx0, ggml_cont(ctx0, tensor),
                tensor->ne[0] * tensor->ne[1] * tensor->ne[2], n_seqs);
            return ggml_reshape_4d(ctx0, ggml_get_rows(ctx0, flat, rows),
                tensor->ne[0], tensor->ne[1], tensor->ne[2], rows->ne[0]);
        };
        for (const auto & group : inp->rbb_groups) {
            ggml_tensor * rows = rerot_rbb_rows(group, shared_rbb);
            if (!rows || rows->ne[0] == 0) {
                continue;
            }
            ggml_tensor * selected_base = ggml_get_rows(ctx0, base_rows, rows);
            const auto rbb = build_delta_net_rbb(
                select(q, rows), select(k, rows), select(v, rows),
                select(g, rows), select(b, rows),
                ggml_view_1d(ctx0, selected_base, D, 0), select(s, rows), il);
            const size_t row_size = ggml_row_size(ssm_states_all->type, D);
            ggml_tensor * destination = ggml_view_1d(ctx0, ssm_states_all, D,
                (size_t) group.brain_row * row_size);
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, rbb.merged_state, destination));
            hands = ggml_set_rows(ctx0, hands,
                ggml_reshape_2d(ctx0, rbb.hand_deltas, D, rows->ne[0]), rows);
            ggml_build_forward_expand(gf, outputs);
            ggml_build_forward_expand(gf, hands);
        }
        ggml_tensor * hand_destination = ggml_view_2d(ctx0, hand_echo_all, D, n_seqs,
            hand_echo_all->nb[1], (size_t) kv_head * hand_echo_all->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, hands, hand_destination));
        return ggml_reshape_4d(ctx0, outputs, S_v, H_v, 1, n_seqs);
    }

    if (!keep) {
        const int64_t D = hparams.n_embd_s();

        auto attn_out = build_delta_net(q, k, v, g, b, s, il);
        ggml_tensor * output    = attn_out.first;
        ggml_tensor * new_state = attn_out.second;
        cb(output, "attn_output", il);
        cb(new_state, "new_state", il);

        if (mctx_cur->is_s_shared(il)) {
            GGML_ASSERT(state_base != nullptr);
            GGML_ASSERT(hand_echo_all != nullptr);
            ggml_tensor * baseline_state = ggml_reshape_4d(
                ctx0, state_base, S_v, S_v, H_v, n_seqs);
            ggml_tensor * hand_reference =
                ggml_reshape_2d(ctx0, baseline_state, D, n_seqs);

            bool has_shared_writers = false;
            for (const auto & group : inp->rbb_groups) {
                ggml_tensor * rows = rerot_rbb_rows(group, shared_rbb);
                has_shared_writers = has_shared_writers || (rows && rows->ne[0] > 0);
            }
            if (has_shared_writers) {
                const auto baseline =
                    build_delta_net(q, k, v, g, b, baseline_state, il);
                cb(baseline.second, "brain_candidate_states", il);
                ggml_tensor * baseline_rows =
                    ggml_reshape_2d(ctx0, baseline.second, D, n_seqs);
                // Only rows that actually participate in a shared commit use
                // T(B) as the hand reference. Default child rows keep B itself
                // as the reference, so H' = T(B+H)-B even in a mixed ubatch.
                for (const auto & group : inp->rbb_groups) {
                    ggml_tensor * rows = rerot_rbb_rows(group, shared_rbb);
                    if (!rows || rows->ne[0] == 0) {
                        continue;
                    }
                    hand_reference = ggml_set_rows(
                        ctx0,
                        hand_reference,
                        ggml_get_rows(ctx0, baseline_rows, rows),
                        rows);
                }
                build_rbb_parallel_commit(
                    inp, ssm_states_all, baseline.second,
                    /*n_snapshots=*/1, il);
            }

            ggml_tensor * hand_destination = ggml_view_2d(
                ctx0,
                hand_echo_all,
                D,
                n_seqs,
                hand_echo_all->nb[1],
                (size_t) kv_head * hand_echo_all->nb[1]);
            ggml_tensor * hand_delta = ggml_reshape_2d(
                ctx0,
                ggml_sub(
                    ctx0,
                    ggml_reshape_2d(ctx0, new_state, D, n_seqs),
                    hand_reference),
                D,
                n_seqs);
            ggml_build_forward_expand(
                gf, ggml_cpy(ctx0, hand_delta, hand_destination));
        } else {
            ggml_tensor * sc_dst_nk = ggml_view_2d(ctx0, ssm_states_all, hparams.n_embd_s(), n_seqs, ssm_states_all->nb[1],
                kv_head * hparams.n_embd_s() * ggml_element_size(ssm_states_all));
            // StateCarryFix: verify no-keep commit (K=1 full-state write):
            // new_state element count vs dst, dst pointer vs s_l base +
            // kv_head offset, and cpy node presence in graph (not fused/dropped).
            {
                const size_t sc_elt = ggml_element_size(ssm_states_all);
                const uint8_t * sc_base = ssm_states_all->data ? (const uint8_t *) ssm_states_all->data : nullptr;
                const uint8_t * sc_expect = sc_base ? sc_base + (size_t) kv_head * (size_t) hparams.n_embd_s() * sc_elt : nullptr;
                ggml_tensor * sc_cpy = ggml_cpy(ctx0, new_state, sc_dst_nk);
                ggml_build_forward_expand(gf, sc_cpy);
                bool sc_in = false;
                for (int sc_i = 0, sc_nn = ggml_graph_n_nodes(gf); sc_i < sc_nn; ++sc_i) {
                    if (ggml_graph_node(gf, sc_i) == sc_cpy) { sc_in = true; break; }
                }
                LLAMA_LOG_ERROR("[statecarry] nokeep-commit il=%d new_ne=[%lld,%lld,%lld,%lld] embd_s=%lld n_seqs=%lld kv_head=%u nelem_match=%d dst_match=%d in_graph=%d\n",
                    il,
                    (long long) new_state->ne[0], (long long) new_state->ne[1],
                    (long long) new_state->ne[2], (long long) new_state->ne[3],
                    (long long) hparams.n_embd_s(), (long long) n_seqs, (unsigned) kv_head,
                    (int) (ggml_nelements(new_state) == ggml_nelements(sc_dst_nk)),
                    (int) (sc_base && sc_dst_nk->data && sc_dst_nk->data == sc_expect), (int) sc_in);
            }
        }

        return output;
    }

    const int64_t D = S_v * S_v * H_v;
    const int64_t K = cparams.n_rs_seq + 1;

    // state s is 4D [S_v, S_v, H_v, n_seqs]; K snapshot slots are written into the output.
    ggml_tensor * gdn_out = ggml_gated_delta_net(ctx0, q, k, v, g, b, s, K);
    if (n_seq_tokens > 1) {
        res->add_fused_node({LLM_FUSED_OP_GDN_CH, gdn_out, il});
    } else {
        res->add_fused_node({LLM_FUSED_OP_GDN_AR, gdn_out, il});
    }

    const int64_t attn_score_elems    = S_v * H_v * n_seq_tokens * n_seqs;
    const int64_t state_size_per_snap = S_v * S_v * H_v * n_seqs;

    ggml_tensor * output = ggml_view_4d(ctx0, gdn_out,
        S_v, H_v, n_seq_tokens, n_seqs,
        ggml_row_size(gdn_out->type, S_v),
        ggml_row_size(gdn_out->type, S_v * H_v),
        ggml_row_size(gdn_out->type, S_v * H_v * n_seq_tokens),
        0);
    cb(output, "attn_output", il);

    const size_t row_size = hparams.n_embd_s() * ggml_element_size(ssm_states_all);

    // op writes the last min(n_seq_tokens, K) snapshots; trailing slots are left unwritten
    const int64_t n_written = std::min<int64_t>(n_seq_tokens, K);

    // write the produced snapshots into the recurrent cache (snapshot slot i -> rollback group i)
    ggml_tensor * src = ggml_view_3d(ctx0, gdn_out,
        D, n_seqs, n_written,
        ggml_row_size(gdn_out->type, D),
        ggml_row_size(gdn_out->type, state_size_per_snap),
        ggml_row_size(gdn_out->type, attn_score_elems));

    if (mctx_cur->is_s_shared(il)) {
        GGML_ASSERT(state_base != nullptr && hand_echo_all != nullptr);
        // Rollback changes how many states are retained, not the recurrence.
        // The GDN snapshots are complete lane-local states. Keep each hand
        // snapshot relative to that snapshot's committed brain (or the
        // unchanged input brain for non-committing child/private rows).
        // Previously only the shared brain was written here: child rows do
        // not commit that brain, so enabling rollback silently lost every
        // recurrent transition after its immediate output.
        ggml_tensor * base_rows = ggml_reshape_2d(ctx0, state_base, D, n_seqs);
        for (int64_t snapshot = 0; snapshot < n_written; ++snapshot) {
            ggml_tensor * candidate = ggml_view_2d(ctx0, src, D, n_seqs,
                src->nb[1], (size_t) snapshot * src->nb[2]);
            ggml_tensor * reference = base_rows;
            for (const auto & group : inp->rbb_groups) {
                ggml_tensor * rows = rerot_rbb_rows(group, shared_rbb);
                if (!rows || rows->ne[0] == 0) {
                    continue;
                }
                ggml_tensor * selected = ggml_get_rows(ctx0, candidate, rows);
                ggml_tensor * committed = selected;
                if (rows->ne[0] > 1) {
                    committed = ggml_sum_rows(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, selected)));
                    committed = ggml_scale(ctx0, committed, 1.0f / (float) rows->ne[0]);
                }
                committed = ggml_reshape_2d(ctx0, committed, D, 1);
                reference = ggml_set_rows(ctx0, reference,
                    ggml_repeat(ctx0, committed, selected), rows);
            }
            ggml_tensor * hand = ggml_sub(ctx0, candidate, reference);
            ggml_tensor * destination = ggml_view_2d(ctx0, hand_echo_all, D, n_seqs,
                hand_echo_all->nb[1],
                ((size_t) snapshot * mem_size + kv_head) * hand_echo_all->nb[1]);
            // Materialize all hands before updating any shared brain buffer.
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, hand, destination));
        }
        build_rbb_parallel_commit(
            inp, ssm_states_all, src, n_written, il);
    } else {
        ggml_tensor * dst = ggml_view_3d(ctx0, ssm_states_all,
            D, n_seqs, n_written,
            ssm_states_all->nb[1],
            (size_t) mem_size * row_size,
            (size_t) kv_head * row_size);

        // StateCarryFix: verify cpy dst == s_l base + kv_head*D*elt and that
        // the cpy node is present in the graph (not fused/dropped).
        {
            const size_t sc_elt = ggml_element_size(ssm_states_all);
            const uint8_t * sc_base = ssm_states_all->data ? (const uint8_t *) ssm_states_all->data : nullptr;
            const uint8_t * sc_expect = sc_base ? sc_base + (size_t) kv_head * (size_t) D * sc_elt : nullptr;
            ggml_tensor * sc_cpy = ggml_cpy(ctx0, src, dst);
            ggml_build_forward_expand(gf, sc_cpy);
            bool sc_in = false;
            for (int sc_i = 0, sc_nn = ggml_graph_n_nodes(gf); sc_i < sc_nn; ++sc_i) {
                if (ggml_graph_node(gf, sc_i) == sc_cpy) { sc_in = true; break; }
            }
            LLAMA_LOG_ERROR("[statecarry] keep-commit il=%d D=%lld embd_s=%lld n_written=%lld kv_head=%u dst_match=%d in_graph=%d src_off=%lld expect_src_off=%lld\n",
                il, (long long) D, (long long) hparams.n_embd_s(), (long long) n_written, (unsigned) kv_head,
                (int) (sc_base && dst->data && dst->data == sc_expect), (int) sc_in,
                (long long) (src->data && gdn_out->data ? (const uint8_t *) src->data - (const uint8_t *) gdn_out->data : -1),
                (long long) ((size_t) attn_score_elems * ggml_element_size(gdn_out)));
        }
    }

    return output;
}
