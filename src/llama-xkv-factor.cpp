#include "llama-xkv-factor.h"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <unordered_set>
#include <unordered_map>
#include <sstream>

namespace llama_xkv {

// Matrix implementation
matrix::matrix(uint64_t r, uint64_t c, float init_val)
    : rows(r), cols(c) {
    if (r != 0 && c > UINT64_MAX / r) {
        throw std::overflow_error("matrix: integer overflow computing rows * cols");
    }
    data.assign(r * c, init_val);
}

matrix::matrix(uint64_t r, uint64_t c, std::vector<float> d)
    : rows(r), cols(c), data(std::move(d)) {
    if (r != 0 && c > UINT64_MAX / r) {
        throw std::overflow_error("matrix: integer overflow computing rows * cols");
    }
    if (rows * cols != data.size()) {
        throw std::invalid_argument("matrix: rows * cols != data.size()");
    }
}

matrix::matrix(uint64_t r, uint64_t c, float * ext_ptr, float init_val)
    : rows(r), cols(c), external_data(ext_ptr) {
    if (r != 0 && c > UINT64_MAX / r) {
        throw std::overflow_error("matrix: integer overflow computing rows * cols");
    }
    if (ext_ptr && r * c > 0) {
        std::fill(ext_ptr, ext_ptr + (r * c), init_val);
    }
}

float & matrix::at(uint64_t r, uint64_t c) {
    if (const_external_data) {
        throw std::logic_error("matrix::at: cannot mutate read-only external matrix view");
    }
    float * base = external_data ? external_data : data.data();
    return base[r * cols + c];
}

const float & matrix::at(uint64_t r, uint64_t c) const {
    const float * base = const_external_data ? const_external_data : (external_data ? external_data : data.data());
    return base[r * cols + c];
}

float * matrix::row_ptr(uint64_t r) {
    if (const_external_data) {
        throw std::logic_error("matrix::row_ptr: cannot mutate read-only external matrix view");
    }
    float * base = external_data ? external_data : data.data();
    return base + r * cols;
}

const float * matrix::row_ptr(uint64_t r) const {
    const float * base = const_external_data ? const_external_data : (external_data ? external_data : data.data());
    return base + r * cols;
}

uint64_t matrix::elements() const {
    if (rows != 0 && cols > UINT64_MAX / rows) {
        throw std::overflow_error("matrix::elements: integer overflow");
    }
    return rows * cols;
}

bool matrix::empty() const {
    if (rows == 0 || cols == 0) return true;
    if (const_external_data || external_data) return false;
    return data.empty();
}

bool matrix::has_non_finite() const {
    uint64_t n = elements();
    const float * base = const_external_data ? const_external_data : (external_data ? external_data : data.data());
    for (uint64_t i = 0; i < n; ++i) {
        float v = base[i];
        if (!std::isfinite(v)) return true;
    }
    return false;
}

bool matrix::operator==(const matrix & o) const {
    if (rows != o.rows || cols != o.cols) return false;
    uint64_t n = elements();
    const float * b1 = const_external_data ? const_external_data : (external_data ? external_data : data.data());
    const float * b2 = o.const_external_data ? o.const_external_data : (o.external_data ? o.external_data : o.data.data());
    for (uint64_t i = 0; i < n; ++i) {
        if (b1[i] != b2[i]) return false;
    }
    return true;
}

matrix matrix_transpose(const matrix & m) {
    if (m.empty()) {
        return matrix(m.cols, m.rows);
    }
    matrix res(m.cols, m.rows);
    for (uint64_t r = 0; r < m.rows; ++r) {
        for (uint64_t c = 0; c < m.cols; ++c) {
            res.at(c, r) = m.at(r, c);
        }
    }
    return res;
}

matrix matrix_matmul(const matrix & a, const matrix & b) {
    if (a.cols != b.rows) {
        throw std::invalid_argument("matrix_matmul: dimension mismatch: a.cols != b.rows");
    }
    if (a.rows == 0 || a.cols == 0 || b.cols == 0) {
        return matrix(a.rows, b.cols);
    }
    matrix res(a.rows, b.cols, 0.0f);
    for (uint64_t i = 0; i < a.rows; ++i) {
        for (uint64_t k = 0; k < a.cols; ++k) {
            float av = a.at(i, k);
            if (av == 0.0f) continue;
            for (uint64_t j = 0; j < b.cols; ++j) {
                res.at(i, j) += av * b.at(k, j);
            }
        }
    }
    return res;
}

matrix matrix_reconstruct(const matrix & a, const matrix & b_transposed) {
    // a is (n x r)
    // b_transposed is (m x r) where each row is a feature across rank
    if (a.cols != b_transposed.cols) {
        throw std::invalid_argument("matrix_reconstruct: rank dimension mismatch (a.cols != b_transposed.cols)");
    }
    uint64_t n = a.rows;
    uint64_t m = b_transposed.rows;
    uint64_t r = a.cols;
    if (n == 0 || m == 0 || r == 0) {
        return matrix(n, m, 0.0f);
    }
    matrix res(n, m, 0.0f);
    for (uint64_t i = 0; i < n; ++i) {
        const float * a_row = a.row_ptr(i);
        float * out_row = res.row_ptr(i);
        for (uint64_t j = 0; j < m; ++j) {
            const float * b_feat = b_transposed.row_ptr(j);
            double sum = 0.0;
            for (uint64_t k = 0; k < r; ++k) {
                sum += (double)a_row[k] * (double)b_feat[k];
            }
            out_row[j] = (float)sum;
        }
    }
    return res;
}

layer_group_map build_layer_group_map(
    const std::vector<uint32_t> & owning_layers,
    uint32_t group_size,
    uint32_t dim_k,
    uint32_t dim_v
) {
    if (group_size == 0) {
        throw std::invalid_argument("build_layer_group_map: group_size must be >= 1");
    }
    if (dim_k == 0 || dim_v == 0) {
        throw std::invalid_argument("build_layer_group_map: dimensions must be non-zero");
    }

    layer_group_map map;
    map.group_size = group_size;

    // Deduplicate owning layers while preserving first-seen order
    std::unordered_set<uint32_t> seen;
    for (uint32_t layer : owning_layers) {
        if (seen.insert(layer).second) {
            map.unique_owning_layers.push_back(layer);
        }
    }

    if (map.unique_owning_layers.empty()) {
        return map;
    }

    uint32_t max_layer_id = *std::max_element(map.unique_owning_layers.begin(), map.unique_owning_layers.end());
    map.model_layer_to_group.assign(max_layer_id + 1, UINT32_MAX);
    map.model_layer_to_group_offset.assign(max_layer_id + 1, UINT32_MAX);

    size_t num_layers = map.unique_owning_layers.size();
    size_t num_groups = (num_layers + group_size - 1) / group_size;

    map.groups.resize(num_groups);
    for (size_t g = 0; g < num_groups; ++g) {
        layer_group & grp = map.groups[g];
        grp.group_index = (uint32_t)g;

        size_t start_idx = g * group_size;
        size_t end_idx = std::min(start_idx + group_size, num_layers);

        uint32_t off_k = 0;
        uint32_t off_v = 0;

        for (size_t i = start_idx; i < end_idx; ++i) {
            uint32_t layer = map.unique_owning_layers[i];
            grp.owning_layers.push_back(layer);

            grp.layer_feature_offsets_k.push_back(off_k);
            grp.layer_feature_dims_k.push_back(dim_k);
            if (UINT32_MAX - off_k < dim_k) {
                throw std::overflow_error("build_layer_group_map: uint32 overflow on total_dim_k");
            }
            off_k += dim_k;

            grp.layer_feature_offsets_v.push_back(off_v);
            grp.layer_feature_dims_v.push_back(dim_v);
            if (UINT32_MAX - off_v < dim_v) {
                throw std::overflow_error("build_layer_group_map: uint32 overflow on total_dim_v");
            }
            off_v += dim_v;

            map.model_layer_to_group[layer] = (uint32_t)g;
            map.model_layer_to_group_offset[layer] = (uint32_t)(i - start_idx);
        }

        grp.total_dim_k = off_k;
        grp.total_dim_v = off_v;
    }

    return map;
}

layer_group_map build_layer_group_map_ex(
    const std::vector<uint32_t> & model_to_owning,
    const std::vector<bool>     & is_attention_layer,
    uint32_t group_size,
    const std::vector<uint32_t> & dim_k_per_layer,
    const std::vector<uint32_t> & dim_v_per_layer
) {
    if (group_size == 0) {
        throw std::invalid_argument("build_layer_group_map_ex: group_size must be >= 1");
    }

    size_t num_model_layers = model_to_owning.size();
    if (is_attention_layer.size() != num_model_layers) {
        throw std::invalid_argument("build_layer_group_map_ex: is_attention_layer size mismatch with model_to_owning");
    }

    layer_group_map map;
    map.group_size = group_size;
    map.model_layer_to_group.assign(num_model_layers, UINT32_MAX);
    map.model_layer_to_group_offset.assign(num_model_layers, UINT32_MAX);

    // Filter attention owning layers, deduplicating aliases
    std::unordered_set<uint32_t> seen_owning;
    std::unordered_map<uint32_t, uint32_t> owning_to_group_idx;
    std::unordered_map<uint32_t, uint32_t> owning_to_offset_idx;

    for (size_t m = 0; m < num_model_layers; ++m) {
        if (!is_attention_layer[m]) continue;
        uint32_t owning = model_to_owning[m];
        if (seen_owning.insert(owning).second) {
            map.unique_owning_layers.push_back(owning);
        }
    }

    if (map.unique_owning_layers.empty()) {
        return map;
    }

    size_t num_layers = map.unique_owning_layers.size();
    size_t num_groups = (num_layers + group_size - 1) / group_size;

    map.groups.resize(num_groups);
    for (size_t g = 0; g < num_groups; ++g) {
        layer_group & grp = map.groups[g];
        grp.group_index = (uint32_t)g;

        size_t start_idx = g * group_size;
        size_t end_idx = std::min(start_idx + group_size, num_layers);

        uint32_t off_k = 0;
        uint32_t off_v = 0;

        for (size_t i = start_idx; i < end_idx; ++i) {
            uint32_t layer = map.unique_owning_layers[i];
            grp.owning_layers.push_back(layer);

            if (layer >= dim_k_per_layer.size() || layer >= dim_v_per_layer.size()) {
                throw std::invalid_argument("build_layer_group_map_ex: layer index out of range for dim_k/dim_v_per_layer");
            }
            uint32_t dk = dim_k_per_layer[layer];
            uint32_t dv = dim_v_per_layer[layer];
            if (dk == 0 || dv == 0) {
                throw std::invalid_argument("build_layer_group_map_ex: per-layer dimensions must be non-zero");
            }

            grp.layer_feature_offsets_k.push_back(off_k);
            grp.layer_feature_dims_k.push_back(dk);
            if (UINT32_MAX - off_k < dk) {
                throw std::overflow_error("build_layer_group_map_ex: uint32 overflow on total_dim_k");
            }
            off_k += dk;

            grp.layer_feature_offsets_v.push_back(off_v);
            grp.layer_feature_dims_v.push_back(dv);
            if (UINT32_MAX - off_v < dv) {
                throw std::overflow_error("build_layer_group_map_ex: uint32 overflow on total_dim_v");
            }
            off_v += dv;

            owning_to_group_idx[layer] = (uint32_t)g;
            owning_to_offset_idx[layer] = (uint32_t)(i - start_idx);
        }

        grp.total_dim_k = off_k;
        grp.total_dim_v = off_v;
    }

    // Now populate model layer to group mappings, including aliases
    for (size_t m = 0; m < num_model_layers; ++m) {
        if (!is_attention_layer[m]) continue;
        uint32_t owning = model_to_owning[m];
        auto it_g = owning_to_group_idx.find(owning);
        if (it_g != owning_to_group_idx.end()) {
            map.model_layer_to_group[m] = it_g->second;
            map.model_layer_to_group_offset[m] = owning_to_offset_idx[owning];
        }
    }

    return map;
}

const char * factor_balance_to_str(factor_balance balance) {
    switch (balance) {
        case LLAMA_XKV_FACTOR_BALANCE_UPSTREAM: return "upstream (alpha=1.0)";
        case LLAMA_XKV_FACTOR_BALANCE_SQRT:     return "sqrt (alpha=0.5)";
        case LLAMA_XKV_FACTOR_BALANCE_DIAGONAL: return "diagonal (row/col norm)";
        default: return "unknown";
    }
}

uint64_t factor_config::fingerprint() const {
    uint64_t hash = 14695981039346656037ULL;
    auto add_u64 = [&](uint64_t v) {
        for (int i = 0; i < 8; ++i) {
            hash ^= (uint8_t)(v >> (i * 8));
            hash *= 1099511628211ULL;
        }
    };
    add_u64(algorithm_version);
    add_u64(random_distribution);
    add_u64(rank_k);
    add_u64(rank_v);
    add_u64((uint64_t)balance);
    add_u64(seed);
    add_u64(group_size);
    add_u64(oversampling);
    add_u64(power_iterations);
    add_u64((uint64_t)factor_a_k);
    add_u64((uint64_t)factor_b_k);
    add_u64((uint64_t)factor_a_v);
    add_u64((uint64_t)factor_b_v);

    uint64_t tol_bits = 0;
    std::memcpy(&tol_bits, &svd_tolerance, sizeof(double));
    add_u64(tol_bits);
    add_u64(max_svd_sweeps);

    return hash;
}

namespace {

inline bool safe_add_u64(uint64_t a, uint64_t b, uint64_t & out) {
    if (UINT64_MAX - a < b) {
        out = UINT64_MAX;
        return false;
    }
    out = a + b;
    return true;
}

inline bool safe_mul_u64(uint64_t a, uint64_t b, uint64_t & out) {
    if (a != 0 && b > UINT64_MAX / a) {
        out = UINT64_MAX;
        return false;
    }
    out = a * b;
    return true;
}

struct workspace_bump_carver {
    uint8_t * base = nullptr;
    size_t capacity = 0;
    size_t offset = 0;

    workspace_bump_carver(void * ptr, size_t cap)
        : base(static_cast<uint8_t*>(ptr)), capacity(cap), offset(0) {}

    template <typename T>
    T * alloc(size_t count, size_t align = 64) {
        // Checked bump allocation: validate inputs, compute with overflow
        // checks, and preserve watermark (offset) on any refusal.
        if (count == 0) return nullptr;
        if (base == nullptr || capacity == 0 || offset > capacity) return nullptr;
        if (align == 0 || (align & (align - 1)) != 0) return nullptr;
        if (count > SIZE_MAX / sizeof(T)) return nullptr;
        size_t bytes = count * sizeof(T);
        uintptr_t base_addr = reinterpret_cast<uintptr_t>(base);
        // offset already validated <= capacity, so base+offset cannot wrap
        // past the allocation only if capacity itself wrapped (it cannot:
        // capacity is a size_t supplied by the caller/arena).
        uintptr_t cur_addr = base_addr + offset;
        size_t misalign = cur_addr % align;
        size_t pad = (misalign == 0) ? 0 : (align - misalign);
        if (pad > SIZE_MAX - offset) return nullptr;
        size_t aligned_off = offset + pad;
        if (aligned_off > capacity) return nullptr;
        if (bytes > capacity - aligned_off) return nullptr;
        T * res = reinterpret_cast<T*>(base + aligned_off);
        offset = aligned_off + bytes;
        return res;
    }

    void reset(size_t off = 0) {
        offset = off;
    }

    size_t watermark() const {
        return offset;
    }
};

} // anonymous namespace

bool estimate_factor_output_bytes(
    uint64_t rows,
    uint64_t cols,
    uint32_t rank,
    uint64_t * out_bytes,
    std::string * err
) {
    if (out_bytes) *out_bytes = UINT64_MAX;
    uint64_t min_nm = std::min(rows, cols);
    uint64_t r = std::min((uint64_t)rank, min_nm);
    uint64_t nr = 0, mr = 0, sum_elems = 0, total_bytes = 0;
    if (!safe_mul_u64(rows, r, nr) || !safe_mul_u64(cols, r, mr) ||
        !safe_add_u64(nr, mr, sum_elems) || !safe_add_u64(sum_elems, r, sum_elems) ||
        !safe_mul_u64(sum_elems, sizeof(float), total_bytes)) {
        if (err) *err = "estimate_factor_output_bytes: integer overflow";
        return false;
    }
    if (out_bytes) *out_bytes = total_bytes;
    return true;
}
bool estimate_factorize_matrix_workspace_bytes(
    uint64_t rows,
    uint64_t cols,
    uint32_t requested_rank,
    uint32_t oversampling,
    uint32_t power_iterations,
    uint64_t * out_bytes,
    std::string * err
) {
    if (out_bytes) *out_bytes = UINT64_MAX;
    if (rows == 0 || cols == 0) {
        if (out_bytes) *out_bytes = 0;
        return true;
    }
    uint64_t total_elements = 0;
    if (!safe_mul_u64(rows, cols, total_elements)) {
        if (err) *err = "workspace estimator: integer overflow on total elements";
        return false;
    }
    if (rows > UINT32_MAX || cols > UINT32_MAX) {
        if (err) *err = "workspace estimator: rows or cols exceeds uint32 representation";
        return false;
    }
    if (requested_rank == 0) {
        if (err) *err = "estimate_factorize_matrix_workspace_bytes: requested_rank cannot be 0";
        return false;
    }
    if (power_iterations == 0) {
        if (err) *err = "estimate_factorize_matrix_workspace_bytes: power_iterations cannot be 0";
        return false;
    }

    uint64_t min_nm = std::min(rows, cols);
    uint64_t r = (uint64_t)requested_rank;
    if (r > min_nm) r = min_nm;

    // If conservative small reference bound: direct Jacobi
    if (min_nm <= 64) {
        uint64_t n = std::max(rows, cols);
        uint64_t m = std::min(rows, cols);
        uint64_t nm = 0, mm = 0;
        if (!safe_mul_u64(n, m, nm) || !safe_mul_u64(m, m, mm)) {
            if (err) *err = "workspace estimator: integer overflow calculating dimension products";
            return false;
        }
        uint64_t peak = 0;
        uint64_t t1 = 0, t2 = 0;
        uint64_t j_tile = 0;
        uint64_t df = 0, c64 = 0;
        if (!safe_add_u64(nm, mm, t1) ||
            !safe_add_u64((uint64_t)sizeof(double), (uint64_t)sizeof(float), df) ||
            !safe_mul_u64(t1, df, t2) ||
            !safe_mul_u64((uint64_t)64, (uint64_t)sizeof(float), c64) ||
            !safe_mul_u64(m, c64, j_tile) ||
            !safe_add_u64(t2, j_tile, peak)) {
            if (err) *err = "workspace estimator: integer overflow on direct Jacobi peak calculation";
            return false;
        }
        if (out_bytes) *out_bytes = peak;
        return true;
    }

    if (r == min_nm && min_nm > 64) {
        if (err) {
            *err = "direct full-rank Jacobi above conservative reference bound (64) rejected to prevent accidental O(n^3)";
        }
        return false;
    }

    // Randomized SVD workspace with checked bounds:
    uint64_t l = 0;
    if (!safe_add_u64(r, (uint64_t)oversampling, l)) {
        l = min_nm;
    }
    if (l > min_nm) l = min_nm;

    uint64_t n = rows;
    uint64_t m = cols;

    uint64_t nl = 0, ml = 0, ll = 0, nr = 0, mr = 0;
    if (!safe_mul_u64(n, l, nl) || !safe_mul_u64(m, l, ml) || !safe_mul_u64(l, l, ll) ||
        !safe_mul_u64(n, r, nr) || !safe_mul_u64(m, r, mr)) {
        if (err) *err = "workspace estimator: integer overflow computing dimension products";
        return false;
    }

    // Peak live temporaries during rSVD:
    // Output A (nr*4), BT (mr*4), S (r*4)
    // Q (nl*8), Z (ml*8), Y_p (nl*8) during power iteration
    // C (ml*8), VT_c (ll*8), tile streaming (64*m*4)
    // All arithmetic below uses only safe_add_u64/safe_mul_u64; no raw
    // `*`/`+` on dimension-derived values (nr/mr/nl/ml/ll/m/l).
    uint64_t nr_mr = 0, nr_mr_r = 0, out_factors_bytes = 0;
    uint64_t nl2 = 0, ml2 = 0, ml2_ll = 0, scratch_elems = 0, scratch_double_bytes = 0;
    uint64_t f64 = 0, tile_bytes = 0, c128 = 0, l_overhead = 0, tile_sum = 0, peak = 0;
    if (!safe_add_u64(nr, mr, nr_mr) ||
        !safe_add_u64(nr_mr, r, nr_mr_r) ||
        !safe_mul_u64(nr_mr_r, (uint64_t)sizeof(float), out_factors_bytes) ||
        !safe_mul_u64(nl, (uint64_t)2, nl2) ||
        !safe_mul_u64(ml, (uint64_t)2, ml2) ||
        !safe_add_u64(ml2, ll, ml2_ll) ||
        !safe_add_u64(nl2, ml2_ll, scratch_elems) ||
        !safe_mul_u64(scratch_elems, (uint64_t)sizeof(double), scratch_double_bytes) ||
        !safe_add_u64(out_factors_bytes, scratch_double_bytes, peak) ||
        !safe_mul_u64((uint64_t)64, (uint64_t)sizeof(float), f64) ||
        !safe_mul_u64(f64, m, tile_bytes) ||
        !safe_mul_u64(l, (uint64_t)128, l_overhead) ||
        !safe_add_u64(tile_bytes, l_overhead, tile_sum) ||
        !safe_add_u64(peak, tile_sum, peak)) {
        if (err) *err = "workspace estimator: integer overflow computing rSVD peak memory";
        return false;
    }

    if (out_bytes) *out_bytes = peak;
    return true;
}

bool estimate_factorize_kv_workspace_bytes(
    const matrix & x_k,
    const matrix & x_v,
    const factor_config & config,
    uint64_t * out_bytes,
    std::string * err
) {
    if (out_bytes) *out_bytes = UINT64_MAX;
    uint64_t ws_k = 0, ws_v = 0;
    uint64_t out_k = 0, out_v = 0;
    if (!estimate_factorize_matrix_workspace_bytes(
            x_k.rows, x_k.cols, config.rank_k, config.oversampling, config.power_iterations, &ws_k, err)) {
        return false;
    }
    if (!estimate_factorize_matrix_workspace_bytes(
            x_v.rows, x_v.cols, config.rank_v, config.oversampling, config.power_iterations, &ws_v, err)) {
        return false;
    }
    if (!estimate_factor_output_bytes(x_k.rows, x_k.cols, config.rank_k, &out_k, err)) {
        return false;
    }
    if (!estimate_factor_output_bytes(x_v.rows, x_v.cols, config.rank_v, &out_v, err)) {
        return false;
    }

    // Exact simultaneous peak terms:
    // Term 1: Peak during K factorization = ws_k (which includes K scratch + K output)
    // Term 2: Peak during V factorization while K output remains live in memory = out_k + ws_v
    // Peak = max(Term 1, Term 2)
    uint64_t term1 = ws_k;
    uint64_t term2 = 0;
    if (!safe_add_u64(out_k, ws_v, term2)) {
        if (err) *err = "estimate_factorize_kv_workspace_bytes: overflow computing simultaneous peak";
        return false;
    }
    if (out_bytes) *out_bytes = std::max(term1, term2);
    return true;
}

bool estimate_quantized_shadow_workspace_bytes(
    const matrix & x_k,
    const matrix & x_v,
    const factor_pair & factor_k,
    const factor_pair & factor_v,
    const factor_config & config,
    uint64_t * out_bytes,
    std::string * err
) {
    if (out_bytes) *out_bytes = UINT64_MAX;

    // Live FP factor inputs
    uint64_t fp_factors = 0;
    uint64_t k_bytes = 0, v_bytes = 0;
    if (!safe_mul_u64(factor_k.a.elements() + factor_k.b_transposed.elements(), sizeof(float), k_bytes) ||
        !safe_mul_u64(factor_v.a.elements() + factor_v.b_transposed.elements(), sizeof(float), v_bytes) ||
        !safe_add_u64(k_bytes, v_bytes, fp_factors)) {
        if (err) *err = "estimate_quantized_shadow_workspace_bytes: overflow on FP factors";
        return false;
    }

    // Encoded matrix byte sizes for 4 streams
    auto is_turbo = [](ggml_type t) {
        return t == GGML_TYPE_TURBO2_0 || t == GGML_TYPE_TURBO3_0 || t == GGML_TYPE_TURBO4_0;
    };
    uint32_t grp_k = is_turbo(config.factor_a_k) ? 128 : 0;
    uint32_t grp_v = is_turbo(config.factor_a_v) ? 128 : 0;

    codec_desc d_a_k = make_codec_desc(factor_role::a_k, config.factor_a_k, orientation::token_major, matrix_shape{factor_k.a.rows, factor_k.a.cols}, grp_k, config.seed);
    codec_desc d_b_k = make_codec_desc(factor_role::b_k, config.factor_b_k, orientation::feature_major_transposed, matrix_shape{factor_k.b_transposed.rows, factor_k.b_transposed.cols}, grp_k, config.seed);
    codec_desc d_a_v = make_codec_desc(factor_role::a_v, config.factor_a_v, orientation::token_major, matrix_shape{factor_v.a.rows, factor_v.a.cols}, grp_v, config.seed + 1);
    codec_desc d_b_v = make_codec_desc(factor_role::b_v, config.factor_b_v, orientation::feature_major_transposed, matrix_shape{factor_v.b_transposed.rows, factor_v.b_transposed.cols}, grp_v, config.seed + 1);

    uint64_t enc_a_k = 0, enc_b_k = 0, enc_a_v = 0, enc_b_v = 0;
    try {
        enc_a_k = encoded_matrix_bytes(d_a_k);
        enc_b_k = encoded_matrix_bytes(d_b_k);
        enc_a_v = encoded_matrix_bytes(d_a_v);
        enc_b_v = encoded_matrix_bytes(d_b_v);
    } catch (const std::exception & ex) {
        if (err) *err = std::string("estimate_quantized_shadow_workspace_bytes: ") + ex.what();
        return false;
    }

    uint64_t encoded_total = 0;
    if (!safe_add_u64(enc_a_k, enc_b_k, encoded_total) ||
        !safe_add_u64(encoded_total, enc_a_v, encoded_total) ||
        !safe_add_u64(encoded_total, enc_b_v, encoded_total)) {
        if (err) *err = "estimate_quantized_shadow_workspace_bytes: overflow on encoded stream sizes";
        return false;
    }

    // Exact decode tile scratch accounting matching evaluate_quantized_shadow:
    // For K stream:
    //   tile_a_buf: 64 * pad_r_a_k * sizeof(float)
    //   feat_b_buf: 64 * pad_r_b_k * sizeof(float)
    //   indices: (64 + 64) * sizeof(uint64_t)
    //   tmp: max(req_tmp_a, req_tmp_b)
    // For V stream:
    //   tile_a_buf: 64 * pad_r_a_v * sizeof(float)
    //   feat_b_buf: 64 * pad_r_b_v * sizeof(float)
    //   indices: (64 + 64) * sizeof(uint64_t)
    //   tmp: max(req_tmp_a, req_tmp_b)
    // Scratch is reused sequentially between K and V, so take max(scratch_k, scratch_v)
    uint64_t pad_ra = std::max(d_a_k.padded_shape.cols, d_a_v.padded_shape.cols);
    uint64_t pad_rb = std::max(d_b_k.padded_shape.cols, d_b_v.padded_shape.cols);

    uint64_t scratch_tile_a = 0, scratch_feat_b = 0, scratch_indices = 0;
    size_t tmp_a_k = 0, tmp_b_k = 0, tmp_a_v = 0, tmp_b_v = 0;
    decode_rows_scratch_bytes(d_a_k, tmp_a_k);
    decode_rows_scratch_bytes(d_b_k, tmp_b_k);
    decode_rows_scratch_bytes(d_a_v, tmp_a_v);
    decode_rows_scratch_bytes(d_b_v, tmp_b_v);
    uint64_t max_decode_tmp = std::max({tmp_a_k, tmp_b_k, tmp_a_v, tmp_b_v});
    uint64_t decode_tmp_budget = 0;
    if (!safe_add_u64(max_decode_tmp, 1024, decode_tmp_budget)) { // alignment headroom
        if (err) *err = "estimate_quantized_shadow_workspace_bytes: overflow on decode tmp";
        return false;
    }

    uint64_t scratch_tile = 0;
    uint64_t f32row = 0, u64idx = 0;
    if (!safe_mul_u64((uint64_t)64, (uint64_t)sizeof(float), f32row) ||
        !safe_mul_u64((uint64_t)128, (uint64_t)sizeof(uint64_t), u64idx) ||
        !safe_mul_u64(f32row, pad_ra, scratch_tile_a) ||
        !safe_mul_u64(f32row, pad_rb, scratch_feat_b) ||
        !safe_mul_u64(u64idx, (uint64_t)1, scratch_indices) ||
        !safe_add_u64(scratch_tile_a, scratch_feat_b, scratch_tile) ||
        !safe_add_u64(scratch_tile, scratch_indices, scratch_tile) ||
        !safe_add_u64(scratch_tile, decode_tmp_budget, scratch_tile)) {
        if (err) *err = "estimate_quantized_shadow_workspace_bytes: overflow on scratch tile";
        return false;
    }

    uint64_t total = 0;
    if (!safe_add_u64(fp_factors, encoded_total, total) ||
        !safe_add_u64(total, scratch_tile, total)) {
        if (err) *err = "estimate_quantized_shadow_workspace_bytes: overflow on total";
        return false;
    }

    if (out_bytes) *out_bytes = total;
    return true;
}

factor_error_report compute_error_report(const matrix & original, const matrix & approx) {
    factor_error_report rep;
    if (original.empty()) {
        return rep;
    }
    if (original.rows != approx.rows || original.cols != approx.cols) {
        throw std::invalid_argument("compute_error_report: dimension mismatch");
    }

    double sum_sq_orig = 0.0;
    double sum_sq_err = 0.0;
    double max_abs_err = 0.0;

    uint64_t total = original.elements();
    const float * orig_ptr = original.data.data();
    const float * app_ptr = approx.data.data();

    for (uint64_t i = 0; i < total; ++i) {
        double o = (double)orig_ptr[i];
        double a = (double)app_ptr[i];
        double err = std::abs(o - a);
        sum_sq_orig += o * o;
        sum_sq_err += err * err;
        if (err > max_abs_err) {
            max_abs_err = err;
        }
    }

    rep.frobenius_norm_original = std::sqrt(sum_sq_orig);
    rep.frobenius_norm_error    = std::sqrt(sum_sq_err);
    rep.max_absolute_error      = max_abs_err;

    if (rep.frobenius_norm_original > 1e-12) {
        rep.relative_error = rep.frobenius_norm_error / rep.frobenius_norm_original;
    } else {
        rep.relative_error = rep.frobenius_norm_error;
    }

    return rep;
}

factor_error_report compute_factor_residual_report_tiled(
    const matrix & x,
    const matrix & a,
    const matrix & b_transposed,
    uint32_t tile_rows
) {
    factor_error_report rep;
    if (x.empty() || a.empty() || b_transposed.empty()) {
        return rep;
    }
    if (x.rows != a.rows || x.cols != b_transposed.rows || a.cols != b_transposed.cols) {
        throw std::invalid_argument("compute_factor_residual_report_tiled: dimension mismatch");
    }

    uint64_t n = x.rows;
    uint64_t m = x.cols;
    uint64_t r = a.cols;

    if (tile_rows == 0) tile_rows = 64;

    double sum_sq_orig = 0.0;
    double sum_sq_err = 0.0;
    double max_abs_err = 0.0;

    // Stream across row tiles: tile size tile_rows x m
    for (uint64_t r_start = 0; r_start < n; r_start += tile_rows) {
        uint64_t r_end = std::min(r_start + tile_rows, n);
        for (uint64_t i = r_start; i < r_end; ++i) {
            const float * a_row = a.row_ptr(i);
            const float * x_row = x.row_ptr(i);

            for (uint64_t j = 0; j < m; ++j) {
                const float * bt_feat = b_transposed.row_ptr(j);
                double recon = 0.0;
                for (uint64_t k = 0; k < r; ++k) {
                    recon += (double)a_row[k] * (double)bt_feat[k];
                }

                double o = (double)x_row[j];
                double err = std::abs(o - recon);
                sum_sq_orig += o * o;
                sum_sq_err += err * err;
                if (err > max_abs_err) {
                    max_abs_err = err;
                }
            }
        }
    }

    rep.frobenius_norm_original = std::sqrt(sum_sq_orig);
    rep.frobenius_norm_error    = std::sqrt(sum_sq_err);
    rep.max_absolute_error      = max_abs_err;

    if (rep.frobenius_norm_original > 1e-12) {
        rep.relative_error = rep.frobenius_norm_error / rep.frobenius_norm_original;
    } else {
        rep.relative_error = rep.frobenius_norm_error;
    }

    return rep;
}

namespace {

// Deterministic 64-bit Xorshift PRNG + Integer-derived Rademacher distribution (+-1)
struct deterministic_rng {
    uint64_t state = 123456789ULL;

    explicit deterministic_rng(uint64_t seed) {
        state = seed != 0 ? seed : 123456789ULL;
    }

    uint64_t next_u64() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return state;
    }

    // Exact cross-libm integer-derived Rademacher distribution (+-1.0)
    // No log, sin, cos; zero drift across compilers and standard C math libraries.
    double next_rademacher() {
        return (next_u64() & 1ULL) ? 1.0 : -1.0;
    }
};

// Double-stable Modified Gram-Schmidt with reorthogonalization (CGS2 / MGS2):
// Q: n x l (row-major: n rows, l cols; columns are length n)
void qr_orthonormalize(double * Q, uint64_t n, uint64_t l) {
    for (uint64_t j = 0; j < l; ++j) {
        // Two passes of Gram-Schmidt against previous columns j0 < j
        for (int pass = 0; pass < 2; ++pass) {
            for (uint64_t j0 = 0; j0 < j; ++j0) {
                double dot = 0.0;
                for (uint64_t i = 0; i < n; ++i) {
                    dot += Q[i * l + j0] * Q[i * l + j];
                }
                for (uint64_t i = 0; i < n; ++i) {
                    Q[i * l + j] -= dot * Q[i * l + j0];
                }
            }
        }

        // Normalize column j
        double norm = 0.0;
        for (uint64_t i = 0; i < n; ++i) {
            double v = Q[i * l + j];
            norm += v * v;
        }
        norm = std::sqrt(norm);
        double inv_norm = norm > 1e-12 ? 1.0 / norm : 0.0;
        for (uint64_t i = 0; i < n; ++i) {
            Q[i * l + j] *= inv_norm;
        }
    }
}

// One-sided Jacobi SVD on tall double-precision matrix:
// Input: U_d (n x m), V_d (m x m) initialized to identity
bool jacobi_svd_tall_double(
    double * U_d,
    double * V_d,
    uint64_t n,
    uint64_t m,
    std::vector<double> * S_d,
    double tol,
    uint32_t max_sweeps,
    std::string * err,
    double * col_sq_buf = nullptr,
    double * s_out_buf = nullptr
) {
    if (n == 0 || m == 0) return true;

    std::vector<double> col_sq_vec;
    double * col_sq = col_sq_buf;
    if (!col_sq) {
        col_sq_vec.assign(m, 0.0);
        col_sq = col_sq_vec.data();
    }

    for (uint64_t j = 0; j < m; ++j) {
        double s = 0.0;
        for (uint64_t i = 0; i < n; ++i) {
            double val = U_d[i * m + j];
            s += val * val;
        }
        col_sq[j] = s;
    }

    bool converged = false;
    double final_correlation = 0.0;

    for (uint32_t sweep = 0; sweep < max_sweeps; ++sweep) {
        double max_correlation = 0.0;
        uint32_t rotations = 0;

        for (uint64_t j1 = 0; j1 < m; ++j1) {
            for (uint64_t j2 = j1 + 1; j2 < m; ++j2) {
                double dot = 0.0;
                for (uint64_t i = 0; i < n; ++i) {
                    dot += U_d[i * m + j1] * U_d[i * m + j2];
                }

                double norm1_sq = col_sq[j1];
                double norm2_sq = col_sq[j2];

                double denom = std::sqrt(norm1_sq * norm2_sq);
                if (denom > 1e-15) {
                    double corr = std::abs(dot) / denom;
                    if (corr > max_correlation) {
                        max_correlation = corr;
                    }
                }

                if (std::abs(dot) <= tol * denom || denom < 1e-15) {
                    continue;
                }

                rotations++;

                double tau = (norm1_sq - norm2_sq) / (2.0 * dot);
                double t = tau >= 0.0 ? 1.0 / (tau + std::sqrt(1.0 + tau * tau))
                                      : -1.0 / (-tau + std::sqrt(1.0 + tau * tau));
                double c = 1.0 / std::sqrt(1.0 + t * t);
                double s = t * c;

                for (uint64_t i = 0; i < n; ++i) {
                    double u1 = U_d[i * m + j1];
                    double u2 = U_d[i * m + j2];
                    U_d[i * m + j1] = c * u1 + s * u2;
                    U_d[i * m + j2] = -s * u1 + c * u2;
                }

                for (uint64_t i = 0; i < m; ++i) {
                    double v1 = V_d[i * m + j1];
                    double v2 = V_d[i * m + j2];
                    V_d[i * m + j1] = c * v1 + s * v2;
                    V_d[i * m + j2] = -s * v1 + c * v2;
                }

                double new_norm1_sq = norm1_sq + t * dot;
                double new_norm2_sq = norm2_sq - t * dot;
                col_sq[j1] = std::max(0.0, new_norm1_sq);
                col_sq[j2] = std::max(0.0, new_norm2_sq);
            }
        }

        // Recompute column norms periodically to avoid numerical drift
        for (uint64_t j = 0; j < m; ++j) {
            double s = 0.0;
            for (uint64_t i = 0; i < n; ++i) {
                double val = U_d[i * m + j];
                s += val * val;
            }
            col_sq[j] = s;
        }

        final_correlation = max_correlation;
        if (max_correlation < tol || rotations == 0) {
            converged = true;
            break;
        }
    }

    if (!converged && m > 1) {
        if (err) {
            *err = "Jacobi SVD did not converge within max_sweeps=" + std::to_string(max_sweeps) +
                   " (final_correlation=" + std::to_string(final_correlation) + " > tol=" + std::to_string(tol) + ")";
        }
        return false;
    }

    double * s_ptr = s_out_buf;
    if (!s_ptr) {
        if (S_d) {
            S_d->assign(m, 0.0);
            s_ptr = S_d->data();
        }
    }
    for (uint64_t j = 0; j < m; ++j) {
        double norm = 0.0;
        for (uint64_t i = 0; i < n; ++i) {
            double val = U_d[i * m + j];
            norm += val * val;
        }
        s_ptr[j] = std::sqrt(norm);
    }

    return true;
}

} // anonymous namespace

bool factorize_matrix(
    const matrix & x,
    uint32_t requested_rank,
    factor_balance balance,
    uint64_t seed,
    factor_pair & out_pair,
    std::string * err,
    double tolerance,
    uint32_t max_sweeps,
    uint32_t oversampling,
    uint32_t power_iterations,
    uint64_t max_workspace_bytes,
    factor_workspace_span workspace,
    factor_fault_injection fault
) {
    if (requested_rank == 0) {
        if (err) *err = "factorize_matrix: requested_rank cannot be 0";
        return false;
    }
    if (power_iterations == 0) {
        if (err) *err = "factorize_matrix: power_iterations cannot be 0";
        return false;
    }
    if (max_sweeps == 0) {
        if (err) *err = "factorize_matrix: max_sweeps must be >= 1";
        return false;
    }
    if (!std::isfinite(tolerance) || tolerance <= 0.0) {
        if (err) *err = "factorize_matrix: tolerance must be positive and finite";
        return false;
    }
    if (x.empty()) {
        if (err) *err = "factorize_matrix: empty input matrix";
        return false;
    }
    if (x.has_non_finite()) {
        if (err) *err = "factorize_matrix: input contains NaN or Inf";
        return false;
    }

    uint64_t n = x.rows;
    uint64_t m = x.cols;
    uint64_t min_nm_64 = std::min(n, m);
    if (min_nm_64 > UINT32_MAX) {
        if (err) *err = "factorize_matrix: dimension exceeds uint32 representation";
        return false;
    }
    uint32_t min_nm = (uint32_t)min_nm_64;
    uint32_t r = std::min((uint64_t)requested_rank, (uint64_t)min_nm);

    // Workspace preflight check
    uint64_t required_workspace = 0;
    if (!estimate_factorize_matrix_workspace_bytes(
            n, m, requested_rank, oversampling, power_iterations, &required_workspace, err)) {
        return false;
    }
    if (required_workspace > max_workspace_bytes) {
        if (err) {
            *err = "factorize_matrix preflight: required workspace (" + std::to_string(required_workspace) +
                   " bytes) exceeds max_workspace_bytes (" + std::to_string(max_workspace_bytes) + ")";
        }
        return false;
    }
    if (workspace.valid() && workspace.size_bytes < required_workspace) {
        if (err) {
            *err = "factorize_matrix: workspace buffer (" + std::to_string(workspace.size_bytes) +
                   " bytes) smaller than required workspace (" + std::to_string(required_workspace) + " bytes)";
        }
        return false;
    }

    // Prepare bump carver over workspace or fallback buffer
    std::vector<uint8_t> fallback_ws;
    workspace_bump_carver carver(workspace.data, workspace.size_bytes);
    if (!workspace.valid()) {
        fallback_ws.resize(required_workspace);
        carver = workspace_bump_carver(fallback_ws.data(), fallback_ws.size());
    }

    // Prepare output buffers: reuse caller-provided buffers if pre-sized, otherwise allocate once off-side
    // CRITICAL: To preserve the atomic failure contract (outputs unmodified on failure),
    // stage factorization results in temporary matrices/buffers and commit only on full success!
    // Production bounded path: validate workspace pointer/size BEFORE any
    // out_pair mutation; checked element counts BEFORE carver.alloc; any null
    // carve with a valid workspace fails immediately (no heap fallback).
    // Heap fallback exists ONLY when workspace is explicitly absent
    // (reference mode).
    uint64_t nr_elems = 0, mr_elems = 0;
    if (!safe_mul_u64(n, (uint64_t)r, nr_elems) ||
        !safe_mul_u64(m, (uint64_t)r, mr_elems)) {
        if (err) *err = "factorize_matrix: integer overflow computing output counts";
        return false;
    }
    if (nr_elems > (uint64_t)(SIZE_MAX / sizeof(float)) ||
        mr_elems > (uint64_t)(SIZE_MAX / sizeof(float)) ||
        (uint64_t)r > (uint64_t)(SIZE_MAX / sizeof(float))) {
        if (err) *err = "factorize_matrix: output dimensions exceed addressable memory";
        return false;
    }
    if (workspace.valid() && workspace.data == nullptr) {
        if (err) *err = "factorize_matrix: workspace reports valid size but null pointer";
        return false;
    }
    float * temp_a_buf = carver.alloc<float>((size_t)nr_elems);
    float * temp_bt_buf = carver.alloc<float>((size_t)mr_elems);
    float * temp_sr_buf = carver.alloc<float>((size_t)r);
    if (workspace.valid() && (!temp_a_buf || !temp_bt_buf || !temp_sr_buf)) {
        if (err) *err = "factorize_matrix: workspace buffer too small for staged outputs (" +
                   std::to_string(workspace.size_bytes) + " bytes)";
        return false;
    }

    // Heap fallback ONLY for explicit reference mode (!workspace.valid()).
    matrix temp_A = temp_a_buf ? matrix(n, r, temp_a_buf, 0.0f) : matrix(n, r, 0.0f);
    matrix temp_BT = temp_bt_buf ? matrix(m, r, temp_bt_buf, 0.0f) : matrix(m, r, 0.0f);
    std::vector<float> temp_S_vec;
    float * S_r_ptr = temp_sr_buf;
    if (!S_r_ptr) {
        temp_S_vec.assign(r, 0.0f);
        S_r_ptr = temp_S_vec.data();
    }

    matrix & A = temp_A;
    matrix & BT = temp_BT;
    float * S_r = S_r_ptr;

    // Branch 1: Direct Jacobi SVD for small reference bound (exact reference)
    if (min_nm <= 64) {
        bool tall = (n >= m);
        uint64_t tn = tall ? n : m;
        uint64_t tm = tall ? m : n;

        double * U_d = carver.alloc<double>(tn * tm);
        double * V_d = carver.alloc<double>(tm * tm);
        double * col_sq_buf = carver.alloc<double>(tm);
        double * s_out_buf = carver.alloc<double>(tm);
        if (!U_d || !V_d || !col_sq_buf || !s_out_buf) {
            if (err) *err = "factorize_matrix: workspace memory allocation shortfall";
            return false;
        }

        if (tall) {
            for (uint64_t i = 0; i < tn; ++i) {
                const float * rptr = x.row_ptr(i);
                for (uint64_t j = 0; j < tm; ++j) U_d[i * tm + j] = (double)rptr[j];
            }
        } else {
            for (uint64_t r0 = 0; r0 < n; ++r0) {
                for (uint64_t c0 = 0; c0 < m; ++c0) {
                    U_d[c0 * n + r0] = (double)x.at(r0, c0);
                }
            }
        }
        for (uint64_t i = 0; i < tm; ++i) V_d[i * tm + i] = 1.0;

        if (!jacobi_svd_tall_double(U_d, V_d, tn, tm, nullptr, tolerance, max_sweeps, err, col_sq_buf, s_out_buf)) {
            return false;
        }

        if (fault == factor_fault_injection::fail_svd) {
            if (err) *err = "factorize_matrix: injected SVD failure (XKV-SR §15.4)";
            return false;
        }

        uint64_t * order = carver.alloc<uint64_t>(tm);
        if (!order) {
            if (err) *err = "factorize_matrix: workspace order allocation shortfall";
            return false;
        }
        for (uint64_t i = 0; i < tm; ++i) order[i] = i;
        std::sort(order, order + tm, [&](uint64_t a, uint64_t b) {
            return s_out_buf[a] > s_out_buf[b];
        });

        for (uint32_t k = 0; k < r; ++k) {
            uint64_t src = order[k];
            double s_val = s_out_buf[src];
            S_r[k] = (float)s_val;
            double inv_s = s_val > 1e-12 ? 1.0 / s_val : 0.0;

            if (tall) {
                for (uint64_t i = 0; i < n; ++i) {
                    A.at(i, k) = (float)(U_d[i * m + src] * inv_s);
                }
                for (uint64_t j = 0; j < m; ++j) {
                    BT.at(j, k) = (float)V_d[j * m + src];
                }
            } else {
                for (uint64_t i = 0; i < n; ++i) {
                    A.at(i, k) = (float)V_d[i * n + src];
                }
                for (uint64_t j = 0; j < m; ++j) {
                    BT.at(j, k) = (float)(U_d[j * n + src] * inv_s);
                }
            }
        }
    } else {
        if (r == min_nm) {
            if (err) {
                *err = "direct full-rank Jacobi above conservative reference bound (64) rejected to prevent accidental O(n^3)";
            }
            return false;
        }
        // Branch 2: Bounded Deterministic Randomized SVD (rSVD)
        // l = min(min_nm, r + oversampling)
        uint64_t l = std::min((uint64_t)min_nm, (uint64_t)r + (uint64_t)oversampling);

        double * Q = carver.alloc<double>(n * l);
        if (!Q) {
            if (err) *err = "factorize_matrix: workspace allocation shortfall for Q";
            return false;
        }

        // 1. Draw deterministic integer-derived Rademacher test matrix Omega (m x l) using seed
        deterministic_rng rng(seed);
        double * Omega = carver.alloc<double>(m * l);
        if (!Omega) {
            if (err) *err = "factorize_matrix: workspace allocation shortfall for Omega";
            return false;
        }
        for (uint64_t i = 0; i < m * l; ++i) {
            Omega[i] = rng.next_rademacher();
        }

        // 2. Y = X * Omega (n x l)
        for (uint64_t i = 0; i < n; ++i) {
            const float * x_row = x.row_ptr(i);
            for (uint64_t j = 0; j < l; ++j) {
                double sum = 0.0;
                for (uint64_t k = 0; k < m; ++k) {
                    sum += (double)x_row[k] * Omega[k * l + j];
                }
                Q[i * l + j] = sum;
            }
        }

        // 3. Initial double-stable QR on Q (n x l)
        qr_orthonormalize(Q, n, l);
        if (fault == factor_fault_injection::fail_qr) {
            if (err) *err = "factorize_matrix: injected QR failure (XKV-SR §15.4)";
            return false;
        }

        // Reclaim Omega memory for power iterations
        size_t post_q_offset = reinterpret_cast<uint8_t*>(Q + n * l) - carver.base;
        carver.reset(post_q_offset);

        // 4. Power iterations: (X * X^T)^q * Y with reorthogonalization
        double * Z = carver.alloc<double>(m * l);
        double * Y_p = carver.alloc<double>(n * l);
        if (!Z || !Y_p) {
            if (err) *err = "factorize_matrix: workspace allocation shortfall for Z/Y_p";
            return false;
        }


        for (uint32_t it = 0; it < power_iterations; ++it) {
            // Z = X^T * Q (m x l)
            for (uint64_t i = 0; i < m; ++i) {
                for (uint64_t j = 0; j < l; ++j) {
                    double sum = 0.0;
                    for (uint64_t k = 0; k < n; ++k) {
                        sum += (double)x.at(k, i) * Q[k * l + j];
                    }
                    Z[i * l + j] = sum;
                }
            }
            qr_orthonormalize(Z, m, l);

            // Y = X * Z (n x l)
            for (uint64_t i = 0; i < n; ++i) {
                const float * x_row = x.row_ptr(i);
                for (uint64_t j = 0; j < l; ++j) {
                    double sum = 0.0;
                    for (uint64_t k = 0; k < m; ++k) {
                        sum += (double)x_row[k] * Z[k * l + j];
                    }
                    Y_p[i * l + j] = sum;
                }
            }
            std::memcpy(Q, Y_p, n * l * sizeof(double));
            qr_orthonormalize(Q, n, l);
        }


        // Reclaim Z and Y_p; carve CT (m x l) and VT_c (l x l)
        carver.reset(post_q_offset);
        double * CT = carver.alloc<double>(m * l);
        if (!CT) {
            if (err) *err = "factorize_matrix: workspace allocation shortfall for CT";
            return false;
        }
        for (uint64_t i = 0; i < l; ++i) {
            for (uint64_t j = 0; j < m; ++j) {
                double sum = 0.0;
                for (uint64_t k = 0; k < n; ++k) {
                    sum += Q[k * l + i] * (double)x.at(k, j);
                }
                CT[j * l + i] = sum;
            }
        }

        // Release Q now that CT is computed!
        // Carve U_full accumulation during step 7 requires Q and VT_c; wait, Q is needed for U_full = Q * VT_c.
        // So Q stays until step 7 is complete.

        // Pre-allocate Jacobi col_sq in carver so jacobi_svd_tall_double doesn't allocate on heap:
        // But jacobi_svd_tall_double takes S_c vector; let's pass a double* for S_c and col_sq!

        double * VT_c = carver.alloc<double>(l * l);
        double * col_sq_buf = carver.alloc<double>(l);
        double * S_c_buf = carver.alloc<double>(l);
        if (!VT_c || !col_sq_buf || !S_c_buf) {
            if (err) *err = "factorize_matrix: workspace allocation shortfall for VT_c/S_c";
            return false;
        }
        std::memset(VT_c, 0, l * l * sizeof(double));
        for (uint64_t i = 0; i < l; ++i) VT_c[i * l + i] = 1.0;

        if (!jacobi_svd_tall_double(CT, VT_c, m, l, nullptr, tolerance, max_sweeps, err, col_sq_buf, S_c_buf)) {
            return false;
        }
        if (fault == factor_fault_injection::fail_svd) {
            if (err) *err = "factorize_matrix: injected SVD failure (XKV-SR §15.4)";
            return false;
        }

        // Sort singular values of small SVD descending
        uint64_t * order = carver.alloc<uint64_t>(l);
        if (!order) {
            if (err) *err = "factorize_matrix: workspace allocation shortfall for order";
            return false;
        }
        for (uint64_t i = 0; i < l; ++i) order[i] = i;
        std::sort(order, order + l, [&](uint64_t a, uint64_t b) {
            return S_c_buf[a] > S_c_buf[b];
        });

        // 7. Reconstruct U_full = Q (n x l) * VT_c (l x l)
        for (uint32_t k = 0; k < r; ++k) {
            uint64_t src = order[k];
            double s_val = S_c_buf[src];
            S_r[k] = (float)s_val;

            // Compute column k of U_full: sum_j Q[i, j] * VT_c[j, src]
            for (uint64_t i = 0; i < n; ++i) {
                double u_ik = 0.0;
                for (uint64_t j0 = 0; j0 < l; ++j0) {
                    u_ik += Q[i * l + j0] * VT_c[j0 * l + src];
                }
                A.at(i, k) = (float)u_ik;
            }

            // Column src of CT is UT[:, src] * s_val, so normalize by inv_s
            double inv_s = s_val > 1e-12 ? 1.0 / s_val : 0.0;
            for (uint64_t j = 0; j < m; ++j) {
                BT.at(j, k) = (float)(CT[j * l + src] * inv_s);
            }
        }
        carver.reset(post_q_offset);
    }

    // Apply balancing: Upstream (alpha=1.0), Sqrt (alpha=0.5), or Diagonal
    switch (balance) {
        case LLAMA_XKV_FACTOR_BALANCE_UPSTREAM: {
            for (uint64_t i = 0; i < n; ++i) {
                for (uint32_t k = 0; k < r; ++k) {
                    A.at(i, k) *= S_r[k];
                }
            }
            break;
        }
        case LLAMA_XKV_FACTOR_BALANCE_SQRT: {
            float * sqrt_S = carver.alloc<float>(r);
            if (!sqrt_S) sqrt_S = temp_sr_buf; // fallback if already allocated
            for (uint32_t k = 0; k < r; ++k) {
                sqrt_S[k] = std::sqrt(std::max(0.0f, S_r[k]));
            }
            for (uint64_t i = 0; i < n; ++i) {
                for (uint32_t k = 0; k < r; ++k) {
                    A.at(i, k) *= sqrt_S[k];
                }
            }
            for (uint64_t j = 0; j < m; ++j) {
                for (uint32_t k = 0; k < r; ++k) {
                    BT.at(j, k) *= sqrt_S[k];
                }
            }
            break;
        }
        case LLAMA_XKV_FACTOR_BALANCE_DIAGONAL: {
            float * sqrt_S = carver.alloc<float>(r);
            if (!sqrt_S) sqrt_S = temp_sr_buf; // fallback if already allocated
            for (uint32_t k = 0; k < r; ++k) {
                sqrt_S[k] = std::sqrt(std::max(0.0f, S_r[k]));
            }
            for (uint64_t i = 0; i < n; ++i) {
                for (uint32_t k = 0; k < r; ++k) {
                    A.at(i, k) *= sqrt_S[k];
                }
            }
            for (uint64_t j = 0; j < m; ++j) {
                for (uint32_t k = 0; k < r; ++k) {
                    BT.at(j, k) *= sqrt_S[k];
                }
            }

            for (uint32_t k = 0; k < r; ++k) {
                double norm_a = 0.0;
                for (uint64_t i = 0; i < n; ++i) {
                    norm_a += (double)A.at(i, k) * (double)A.at(i, k);
                }
                norm_a = std::sqrt(norm_a);

                double norm_b = 0.0;
                for (uint64_t j = 0; j < m; ++j) {
                    norm_b += (double)BT.at(j, k) * (double)BT.at(j, k);
                }
                norm_b = std::sqrt(norm_b);

                if (norm_a > 1e-12 && norm_b > 1e-12) {
                    double d = std::sqrt(norm_b / norm_a);
                    for (uint64_t i = 0; i < n; ++i) {
                        A.at(i, k) = (float)(A.at(i, k) * d);
                    }
                    float inv_d = (float)(1.0 / d);
                    for (uint64_t j = 0; j < m; ++j) {
                        BT.at(j, k) *= inv_d;
                    }
                }
            }
            break;
        }
        default:
            if (err) *err = std::string("unknown factor_balance mode");
            return false;
    }

    out_pair.rank = r;

    // Compute residual report via streaming tiles (no full recon matrix allocation!)
    factor_error_report res_report = compute_factor_residual_report_tiled(x, A, BT, 64);

    // ATOMIC COMMIT: Now that factorization and evaluation fully succeeded, commit to out_pair!
    if (out_pair.a.rows != n || out_pair.a.cols != r) {
        out_pair.a = matrix(n, r, 0.0f);
    }
    if (out_pair.b_transposed.rows != m || out_pair.b_transposed.cols != r) {
        out_pair.b_transposed = matrix(m, r, 0.0f);
    }
    if (out_pair.a.row_ptr(0) != A.row_ptr(0)) {
        std::memcpy(out_pair.a.row_ptr(0), A.row_ptr(0), n * r * sizeof(float));
    }
    if (out_pair.b_transposed.row_ptr(0) != BT.row_ptr(0)) {
        std::memcpy(out_pair.b_transposed.row_ptr(0), BT.row_ptr(0), m * r * sizeof(float));
    }

    out_pair.singular_values.resize(r);
    for (uint32_t k = 0; k < r; ++k) {
        out_pair.singular_values[k] = S_r[k];
    }
    out_pair.rank = r;
    out_pair.residual_report = std::move(res_report);

    return true;
}

bool factorize_matrix_bounded(
    const matrix & x,
    uint32_t requested_rank,
    factor_balance balance,
    uint64_t seed,
    factor_pair & out_pair,
    factor_workspace_span workspace,
    std::string * err,
    double tolerance,
    uint32_t max_sweeps,
    uint32_t oversampling,
    uint32_t power_iterations,
    uint64_t max_workspace_bytes,
    factor_fault_injection fault
) {
    return factorize_matrix(x, requested_rank, balance, seed, out_pair, err, tolerance, max_sweeps, oversampling, power_iterations, max_workspace_bytes, workspace, fault);
}

// Factor Pair Ownership & Lifetime Contract:
// factorize_kv produces factor_result containing factor_pair k and v.
// These factor_pairs own their A and authoritative B^T matrices (either via heap std::vector
// or via caller-managed persistent buffers). Factor matrices MUST persist across the entire
// shadow evaluation and subsequent landmark extraction.
// When factorize_kv is called with a caller workspace arena lease, the lease is used strictly
// for ephemeral rSVD temporaries (Q, Omega, Z, Y_p, CT, VT_c, etc.), leaving the persistent
// factor outputs safely retained so subsequent calls (e.g. evaluate_quantized_shadow) can reuse
// the exact same workspace arena without clobbering factor_k or factor_v.
factor_result factorize_kv(
    const matrix & x_k,
    const matrix & x_v,
    const factor_config & config,
    factor_workspace_span workspace
) {
    factor_result result;
    result.config_fingerprint = config.fingerprint();
    try {

    // Validate inputs
    if (x_k.empty() || x_v.empty()) {
        result.success = false;
        result.error_message = "factorize_kv: x_k or x_v is empty";
        return result;
    }

    if (x_k.rows != x_v.rows) {
        result.success = false;
        result.error_message = "factorize_kv: row count mismatch between x_k and x_v (" +
                               std::to_string(x_k.rows) + " vs " + std::to_string(x_v.rows) + ")";
        return result;
    }

    if (x_k.has_non_finite() || x_v.has_non_finite()) {
        result.success = false;
        result.error_message = "factorize_kv: inputs contain non-finite numbers";
        return result;
    }

    // Preflight workspace check for combined factorize_kv
    uint64_t required_ws = 0;
    if (!estimate_factorize_kv_workspace_bytes(x_k, x_v, config, &required_ws, &result.error_message)) {
        result.success = false;
        return result;
    }
    if (required_ws > config.max_workspace_bytes) {
        result.success = false;
        result.error_message = "factorize_kv preflight: required workspace (" + std::to_string(required_ws) +
                               " bytes) exceeds max_workspace_bytes (" + std::to_string(config.max_workspace_bytes) + ")";
        return result;
    }
    if (workspace.valid() && workspace.size_bytes < required_ws) {
        result.success = false;
        result.error_message = "factorize_kv: workspace buffer (" + std::to_string(workspace.size_bytes) +
                               " bytes) smaller than required workspace (" + std::to_string(required_ws) + " bytes)";
        return result;
    }

    // Create carved workspace slices if workspace is provided
    factor_workspace_span ws_k = workspace;
    factor_workspace_span ws_v = workspace;

    factor_pair temp_k, temp_v;
    std::string err_k, err_v;

    // Resolve per-stage fault injection based on config.fault and fault_occurrence
    factor_fault_injection fault_k = (config.fault_occurrence == 0) ? config.fault : factor_fault_injection::none;
    factor_fault_injection fault_v = (config.fault_occurrence == 1) ? config.fault : factor_fault_injection::none;

    // Factorize K
    if (!factorize_matrix(
            x_k, config.rank_k, config.balance, config.seed, temp_k, &err_k,
            config.svd_tolerance, config.max_svd_sweeps, config.oversampling,
            config.power_iterations, config.max_workspace_bytes, ws_k, fault_k)) {
        result.success = false;
        result.error_message = "factorize_kv K failed: " + err_k;
        return result;
    }

    // Factorize V independently
    if (!factorize_matrix(
            x_v, config.rank_v, config.balance, config.seed + 1, temp_v, &err_v,
            config.svd_tolerance, config.max_svd_sweeps, config.oversampling,
            config.power_iterations, config.max_workspace_bytes, ws_v, fault_v)) {
        result.success = false;
        result.error_message = "factorize_kv V failed: " + err_v;
        return result;
    }

    // Atomic success: populate output only after both completed successfully
    result.k = std::move(temp_k);
    result.v = std::move(temp_v);
    result.success = true;
    return result;
    } catch (const std::bad_alloc & ex) {
        result.success = false;
        result.error_message = std::string("factorize_kv std::bad_alloc: ") + ex.what();
        return result;
    } catch (const std::exception & ex) {
        result.success = false;
        result.error_message = std::string("factorize_kv exception: ") + ex.what();
        return result;
    }
}

factor_result factorize_kv_bounded(
    const matrix & x_k,
    const matrix & x_v,
    const factor_config & config,
    factor_workspace_span workspace
) {
    return factorize_kv(x_k, x_v, config, workspace);
}

factor_quantized_shadow evaluate_quantized_shadow(
    const matrix & x_k,
    const matrix & x_v,
    const factor_pair & factor_k,
    const factor_pair & factor_v,
    const factor_config & config,
    factor_workspace_span workspace
) {
    factor_quantized_shadow shadow;

    try {
        uint64_t req_ws = 0;
        std::string est_err;
        if (!estimate_quantized_shadow_workspace_bytes(x_k, x_v, factor_k, factor_v, config, &req_ws, &est_err)) {
            shadow.success = false;
            shadow.error_message = "evaluate_quantized_shadow estimator failed: " + est_err;
            return shadow;
        }
        if (workspace.valid() && workspace.size_bytes < req_ws) {
            shadow.success = false;
            shadow.error_message = "evaluate_quantized_shadow: workspace buffer (" +
                                   std::to_string(workspace.size_bytes) + " bytes) smaller than required workspace (" +
                                   std::to_string(req_ws) + " bytes)";
            return shadow;
        }

        auto is_turbo = [](ggml_type t) {
            return t == GGML_TYPE_TURBO2_0 || t == GGML_TYPE_TURBO3_0 || t == GGML_TYPE_TURBO4_0;
        };

        // All Turbo types in v1 require group_size=128; for non-Turbo group_size is 0
        uint32_t grp_a_k = is_turbo(config.factor_a_k) ? 128 : 0;
        uint32_t grp_b_k = is_turbo(config.factor_b_k) ? 128 : 0;
        uint32_t grp_a_v = is_turbo(config.factor_a_v) ? 128 : 0;
        uint32_t grp_b_v = is_turbo(config.factor_b_v) ? 128 : 0;

        // Encode Stream A_K: token-major (n x r_k)
        codec_desc desc_a_k = make_codec_desc(
            factor_role::a_k,
            config.factor_a_k,
            orientation::token_major,
            matrix_shape{factor_k.a.rows, factor_k.a.cols},
            grp_a_k,
            config.seed
        );
        shadow.stream_a_k = encode_matrix(desc_a_k, factor_k.a.row_ptr(0), factor_k.a.elements());

        // Encode Stream B_K: feature-major transposed (m_k x r_k)
        codec_desc desc_b_k = make_codec_desc(
            factor_role::b_k,
            config.factor_b_k,
            orientation::feature_major_transposed,
            matrix_shape{factor_k.b_transposed.rows, factor_k.b_transposed.cols},
            grp_b_k,
            config.seed // B_K must share the exact same rotation seed as A_K for pair compatibility
        );
        shadow.stream_b_k = encode_matrix(desc_b_k, factor_k.b_transposed.row_ptr(0), factor_k.b_transposed.elements());

        // Encode Stream A_V: token-major (n x r_v)
        codec_desc desc_a_v = make_codec_desc(
            factor_role::a_v,
            config.factor_a_v,
            orientation::token_major,
            matrix_shape{factor_v.a.rows, factor_v.a.cols},
            grp_a_v,
            config.seed + 1 // V pair uses seed + 1
        );
        shadow.stream_a_v = encode_matrix(desc_a_v, factor_v.a.row_ptr(0), factor_v.a.elements());

        // Encode Stream B_V: feature-major transposed (m_v x r_v)
        codec_desc desc_b_v = make_codec_desc(
            factor_role::b_v,
            config.factor_b_v,
            orientation::feature_major_transposed,
            matrix_shape{factor_v.b_transposed.rows, factor_v.b_transposed.cols},
            grp_b_v,
            config.seed + 1 // B_V must share the exact same rotation seed as A_V for pair compatibility
        );
        shadow.stream_b_v = encode_matrix(desc_b_v, factor_v.b_transposed.row_ptr(0), factor_v.b_transposed.elements());

        // STREAMING RESIDUAL EVALUATION WITHOUT MATERIALIZING FOUR FULL DECODED MATRICES!
        // We stream across row tiles (tile_rows = 64) of A and X, while streaming feature tiles of B^T.
        const uint32_t tile_rows = 64;

        // Prepare bump carver over workspace or fallback buffer for shadow streaming scratch
        std::vector<uint8_t> fallback_shadow_scratch;
        workspace_bump_carver shadow_carver(workspace.data, workspace.size_bytes);
        if (!workspace.valid()) {
            fallback_shadow_scratch.resize(req_ws + 4096);
            shadow_carver = workspace_bump_carver(fallback_shadow_scratch.data(), fallback_shadow_scratch.size());
        }

        // Helper to evaluate residual reports for one stream pair (e.g. K or V)
        auto stream_pair_residuals = [&](
            const matrix & X,
            const factor_pair & FP,
            const encoded_matrix & stream_a,
            const encoded_matrix & stream_b,
            workspace_bump_carver & carver,
            shadow_stream_errors & out_errors
        ) {
            uint64_t n = X.rows;
            uint64_t m = X.cols;
            uint64_t r = FP.rank;
            uint64_t pad_r_a = stream_a.desc.padded_shape.cols;
            uint64_t pad_r_b = stream_b.desc.padded_shape.cols;

            double a_sum_sq_orig = 0.0, a_sum_sq_err = 0.0, a_max_abs_err = 0.0;
            double b_sum_sq_orig = 0.0, b_sum_sq_err = 0.0, b_max_abs_err = 0.0;
            double ab_sum_sq_orig = 0.0, ab_sum_sq_err = 0.0, ab_max_abs_err = 0.0;

            size_t initial_mark = carver.watermark();

            // Scratch tile buffers carved from carver:
            float * tile_a_buf = carver.alloc<float>(tile_rows * pad_r_a);
            uint64_t * tile_row_indices = carver.alloc<uint64_t>(tile_rows);

            // Query decode_rows scratch requirement for stream_a and stream_b
            size_t req_tmp_a = 0, req_tmp_b = 0;
            decode_rows_scratch_bytes(stream_a.desc, req_tmp_a);
            decode_rows_scratch_bytes(stream_b.desc, req_tmp_b);
            size_t max_tmp_bytes = std::max(req_tmp_a, req_tmp_b);
            uint8_t * tmp_ptr = max_tmp_bytes > 0 ? carver.alloc<uint8_t>(max_tmp_bytes) : nullptr;

            const uint32_t feat_tile_size = 64;
            float * feat_b_buf = carver.alloc<float>(feat_tile_size * pad_r_b);
            uint64_t * feat_indices = carver.alloc<uint64_t>(feat_tile_size);

            if (!tile_a_buf || !tile_row_indices || (max_tmp_bytes > 0 && !tmp_ptr) || !feat_b_buf || !feat_indices) {
                throw std::runtime_error("evaluate_quantized_shadow: workspace scratch allocation shortfall");
            }

            // Process row tiles of X and A
            for (uint64_t r_start = 0; r_start < n; r_start += tile_rows) {
                uint64_t cur_tile_rows = std::min((uint64_t)tile_rows, n - r_start);
                for (uint64_t i = 0; i < cur_tile_rows; ++i) {
                    tile_row_indices[i] = r_start + i;
                }
                decode_rows(stream_a, tile_row_indices, cur_tile_rows, tile_a_buf, cur_tile_rows * pad_r_a, tmp_ptr, max_tmp_bytes, value_domain::canonical);

                // Process feature tiles of B
                for (uint64_t f_start = 0; f_start < m; f_start += feat_tile_size) {
                    uint64_t cur_feat_cols = std::min((uint64_t)feat_tile_size, m - f_start);
                    for (uint64_t j = 0; j < cur_feat_cols; ++j) {
                        feat_indices[j] = f_start + j;
                    }
                    decode_rows(stream_b, feat_indices, cur_feat_cols, feat_b_buf, cur_feat_cols * pad_r_b, tmp_ptr, max_tmp_bytes, value_domain::canonical);

                    for (uint64_t i = 0; i < cur_tile_rows; ++i) {
                        uint64_t row_idx = r_start + i;
                        const float * orig_x_row = X.row_ptr(row_idx);
                        const float * orig_a_row = FP.a.row_ptr(row_idx);
                        const float * dec_a_row  = tile_a_buf + i * pad_r_a;

                        for (uint64_t j = 0; j < cur_feat_cols; ++j) {
                            uint64_t col_idx = f_start + j;
                            const float * orig_b_feat = FP.b_transposed.row_ptr(col_idx);
                            const float * dec_b_feat  = feat_b_buf + j * pad_r_b;

                            double orig_val = (double)orig_x_row[col_idx];
                            double a_recon = 0.0;
                            double b_recon = 0.0;
                            double ab_recon = 0.0;

                            for (uint64_t k = 0; k < r; ++k) {
                                a_recon  += (double)dec_a_row[k]  * (double)orig_b_feat[k];
                                b_recon  += (double)orig_a_row[k] * (double)dec_b_feat[k];
                                ab_recon += (double)dec_a_row[k]  * (double)dec_b_feat[k];
                            }

                            double err_a  = std::abs(orig_val - a_recon);
                            double err_b  = std::abs(orig_val - b_recon);
                            double err_ab = std::abs(orig_val - ab_recon);

                            double o_sq = orig_val * orig_val;
                            a_sum_sq_orig += o_sq;
                            a_sum_sq_err  += err_a * err_a;
                            if (err_a > a_max_abs_err) a_max_abs_err = err_a;

                            b_sum_sq_orig += o_sq;
                            b_sum_sq_err  += err_b * err_b;
                            if (err_b > b_max_abs_err) b_max_abs_err = err_b;

                            ab_sum_sq_orig += o_sq;
                            ab_sum_sq_err  += err_ab * err_ab;
                            if (err_ab > ab_max_abs_err) ab_max_abs_err = err_ab;
                        }
                    }
                }
            }

            auto finalize_rep = [](factor_error_report & rep, double sq_orig, double sq_err, double max_err) {
                rep.frobenius_norm_original = std::sqrt(sq_orig);
                rep.frobenius_norm_error    = std::sqrt(sq_err);
                rep.max_absolute_error      = max_err;
                rep.relative_error = rep.frobenius_norm_original > 1e-12 ? (rep.frobenius_norm_error / rep.frobenius_norm_original) : rep.frobenius_norm_error;
            };

            finalize_rep(out_errors.a_only, a_sum_sq_orig, a_sum_sq_err, a_max_abs_err);
            finalize_rep(out_errors.b_only, b_sum_sq_orig, b_sum_sq_err, b_max_abs_err);
            finalize_rep(out_errors.ab_product, ab_sum_sq_orig, ab_sum_sq_err, ab_max_abs_err);

            carver.reset(initial_mark);
        };

        stream_pair_residuals(x_k, factor_k, shadow.stream_a_k, shadow.stream_b_k, shadow_carver, shadow.errors_k);
        stream_pair_residuals(x_v, factor_v, shadow.stream_a_v, shadow.stream_b_v, shadow_carver, shadow.errors_v);

        shadow.success = true;
    } catch (const std::exception & ex) {
        shadow.success = false;
        shadow.error_message = ex.what();
    }

    return shadow;
}

factor_quantized_shadow evaluate_quantized_shadow_bounded(
    const matrix & x_k,
    const matrix & x_v,
    const factor_pair & factor_k,
    const factor_pair & factor_v,
    const factor_config & config,
    factor_workspace_span workspace
) {
    return evaluate_quantized_shadow(x_k, x_v, factor_k, factor_v, config, workspace);
}

} // namespace llama_xkv
