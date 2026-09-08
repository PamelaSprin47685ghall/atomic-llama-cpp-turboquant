#include "llama-xkv-graph-ref.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

#include <map>
#include <mutex>
#include <set>

#include "ggml-backend.h"
#include "llama-impl.h"
#include "ggml-xkv-landmark-build.h"
// Device SR selection/merge/rows operators.
#include "ggml-vulkan-landmark.h"

namespace llama_xkv {

namespace {
// File-local helpers defined later in this TU (same anonymous namespace).
// Used by xkv_graph_op_handle::compute.
bool is_turbo_type(ggml_type t);
bool storage_row_geometry(const ggml_tensor * t, size_t & row_bytes, size_t & cell_stride, std::string * err);
bool fetch_hot_storage_rows(const ggml_tensor * t, const std::vector<int64_t> & cells, bool host_resident,
    std::vector<std::vector<uint8_t>> & out_rows, std::string * err);
bool dequant_storage_row(ggml_type type, const uint8_t * raw, int64_t n_elements, std::vector<float> & out, std::string * err);
bool dequant_storage_row_ptr(ggml_type type, const uint8_t * raw, int64_t n_elements, float * out, std::string * err);
bool apply_inverse_rotation(std::vector<float> & vec, const std::vector<float> & mat, uint32_t dim, std::string * err);
} // namespace

// External-linkage helpers defined later in this TU, used by compute.
bool preflight_gather_temps(const xkv_graph_snapshot & snap, size_t n_gather,
    int64_t nk_floats, int64_t nv_floats, size_t cell_stride_k, size_t cell_stride_v,
    size_t sink_floats, bool & over_budget, std::string * err);
bool run_sr_after_q(const xkv_graph_snapshot & snap, const std::vector<xkv_query_input> & queries,
    sr_batch_selection_result & out_sel, std::string * err);
bool xform_hot_head_span(float * io, uint32_t pad, const std::vector<float> & inv_rot, uint32_t rot_dim,
    const std::vector<float> & chan_mul, float * tmp, std::string * err);
bool fill_required_queries(const xkv_graph_snapshot & snap, const float * q_raw, size_t q_nelements, std::string * err);
bool fill_required_hot(const xkv_graph_snapshot & snap, std::string * err);
xkv_read_status run_sr_bounded_after_q(xkv_graph_snapshot & snap, const std::vector<xkv_query_input> & queries, std::string * err);

uint32_t xkv_graph_snapshot::get_query_group_count(uint32_t q) const {
    if (!query_ddvr_group_counts.empty()) {
        if (q < query_ddvr_group_counts.size()) {
            return query_ddvr_group_counts[q];
        }
        return 0;
    }
    return n_ddvr_groups;
}

bool xkv_graph_snapshot::get_required_q_elements(size_t & out_elements, std::string * err) const {
    out_elements = 0;
    if (head_dim_k == 0 || head_dim_v == 0 || n_q_heads == 0) {
        if (err) *err = "get_required_q_elements: zero dimensions or heads";
        return false;
    }
    if (n_queries == 0) {
        return true;
    }

    size_t q_head_elements = 0;
    if (!safe_mul(static_cast<size_t>(n_q_heads), static_cast<size_t>(head_dim_k), q_head_elements)) {
        if (err) *err = "get_required_q_elements: overflow computing q_head_elements";
        return false;
    }

    if (!query_ddvr_group_counts.empty()) {
        if (query_ddvr_group_counts.size() != n_queries) {
            if (err) *err = "get_required_q_elements: query_ddvr_group_counts size (" +
                std::to_string(query_ddvr_group_counts.size()) + ") != n_queries (" +
                std::to_string(n_queries) + ")";
            return false;
        }

        if (!query_ddvr_group_offsets.empty()) {
            if (query_ddvr_group_offsets.size() != n_queries) {
                if (err) *err = "get_required_q_elements: query_ddvr_group_offsets size (" +
                    std::to_string(query_ddvr_group_offsets.size()) + ") != n_queries (" +
                    std::to_string(n_queries) + ")";
                return false;
            }
            size_t max_end = 0;
            for (uint32_t q = 0; q < n_queries; ++q) {
                uint32_t g_count = query_ddvr_group_counts[q];
                size_t needed_groups = (g_count > 0) ? g_count : 1;
                size_t q_elems = 0;
                if (!safe_mul(needed_groups, q_head_elements, q_elems)) {
                    if (err) *err = "get_required_q_elements: overflow computing query elements";
                    return false;
                }
                size_t end_offset = 0;
                if (!safe_add(query_ddvr_group_offsets[q], q_elems, end_offset)) {
                    if (err) *err = "get_required_q_elements: overflow computing query end offset";
                    return false;
                }
                max_end = std::max(max_end, end_offset);
            }
            out_elements = max_end;
            return true;
        }

        size_t total_elements = 0;
        for (uint32_t q = 0; q < n_queries; ++q) {
            uint32_t g_count = query_ddvr_group_counts[q];
            size_t needed_groups = (g_count > 0) ? g_count : 1;
            size_t q_elems = 0;
            if (!safe_mul(needed_groups, q_head_elements, q_elems)) {
                if (err) *err = "get_required_q_elements: overflow computing query elements";
                return false;
            }
            if (!safe_add(total_elements, q_elems, total_elements)) {
                if (err) *err = "get_required_q_elements: overflow computing total elements";
                return false;
            }
        }
        out_elements = total_elements;
        return true;
    }

    if (!query_ddvr_group_offsets.empty()) {
        if (query_ddvr_group_offsets.size() != n_queries) {
            if (err) *err = "get_required_q_elements: query_ddvr_group_offsets size (" +
                std::to_string(query_ddvr_group_offsets.size()) + ") != n_queries (" +
                std::to_string(n_queries) + ")";
            return false;
        }
        size_t groups = (n_ddvr_groups > 0) ? n_ddvr_groups : 1;
        size_t q_elems = 0;
        if (!safe_mul(groups, q_head_elements, q_elems)) {
            if (err) *err = "get_required_q_elements: overflow computing query elements";
            return false;
        }
        size_t max_end = 0;
        for (uint32_t q = 0; q < n_queries; ++q) {
            size_t end_offset = 0;
            if (!safe_add(query_ddvr_group_offsets[q], q_elems, end_offset)) {
                if (err) *err = "get_required_q_elements: overflow computing query end offset";
                return false;
            }
            max_end = std::max(max_end, end_offset);
        }
        out_elements = max_end;
        return true;
    }

    size_t groups = (n_ddvr_groups > 0) ? n_ddvr_groups : 1;
    size_t q_stride = 0;
    if (!safe_mul(groups, q_head_elements, q_stride)) {
        if (err) *err = "get_required_q_elements: overflow computing q_stride";
        return false;
    }
    if (!safe_mul(static_cast<size_t>(n_queries), q_stride, out_elements)) {
        if (err) *err = "get_required_q_elements: overflow computing out_elements";
        return false;
    }
    return true;
}

bool xkv_graph_snapshot::validate_hot_rows(std::string * err) const {
    if (max_hot_rows > 0 && hot_data.size() > max_hot_rows) {
        if (err) *err = "hot row count (" + std::to_string(hot_data.size()) +
            ") exceeds max_hot_rows bound (" + std::to_string(max_hot_rows) + ")";
        return false;
    }

    size_t total_hot_bytes = 0;
    for (size_t i = 0; i < hot_data.size(); ++i) {
        const auto & hd = hot_data[i];
        // Storage-gather rows carry no owned copies; their bytes are validated
        // against live storage tensors inside compute. Only fallback rows are
        // checked here.
        if (hd.is_valid && !hd.use_storage_gather()) {
            if (hd.k_data.size() != head_dim_k) {
                if (err) *err = "hot row " + std::to_string(i) + " k_data size (" +
                    std::to_string(hd.k_data.size()) + ") != head_dim_k (" +
                    std::to_string(head_dim_k) + ")";
                return false;
            }
            if (hd.v_data.size() != head_dim_v) {
                if (err) *err = "hot row " + std::to_string(i) + " v_data size (" +
                    std::to_string(hd.v_data.size()) + ") != head_dim_v (" +
                    std::to_string(head_dim_v) + ")";
                return false;
            }
            for (float v : hd.k_data) {
                if (!std::isfinite(v)) {
                    if (err) *err = "hot row " + std::to_string(i) + " k_data contains non-finite value";
                    return false;
                }
            }
            for (float v : hd.v_data) {
                if (!std::isfinite(v)) {
                    if (err) *err = "hot row " + std::to_string(i) + " v_data contains non-finite value";
                    return false;
                }
            }
        }

        if (!hd.query_visibility.empty() && hd.query_visibility.size() != n_queries) {
            if (err) *err = "hot row " + std::to_string(i) + " query_visibility size (" +
                std::to_string(hd.query_visibility.size()) + ") != n_queries (" +
                std::to_string(n_queries) + ")";
            return false;
        }

        // Validate DDVR group visibility
        if (hd.is_valid) {
            if (!hd.query_group_indices.empty() && hd.query_group_indices.size() != n_queries) {
                if (err) *err = "hot row " + std::to_string(i) + " query_group_indices size (" +
                    std::to_string(hd.query_group_indices.size()) + ") != n_queries (" +
                    std::to_string(n_queries) + ")";
                return false;
            }
            for (uint32_t q = 0; q < n_queries; ++q) {
                if (hd.is_visible_to_query(q)) {
                    uint32_t q_groups = get_query_group_count(q);
                    uint32_t g = hd.query_group_indices.empty() ? hd.group_index : hd.query_group_indices[q];
                    if (g == UINT32_MAX) {
                        if (err) *err = "hot row " + std::to_string(i) + " visible query " +
                            std::to_string(q) + " has invalid group";
                        return false;
                    }
                    if (q_groups > 0 && g >= q_groups) {
                        if (err) *err = "hot row " + std::to_string(i) + " group_index (" +
                            std::to_string(g) + ") exceeds query " +
                            std::to_string(q) + " group count (" + std::to_string(q_groups) + ")";
                        return false;
                    }
                    if (q_groups == 0 && g != 0) {
                        if (err) *err = "hot row " + std::to_string(i) + " non-zero group_index (" +
                            std::to_string(g) + ") without DDVR groups on query " +
                            std::to_string(q);
                        return false;
                    }
                }
            }
        }

        size_t row_bytes = 0;
        size_t k_bytes = 0;
        size_t v_bytes = 0;
        if (!safe_mul(hd.k_data.size(), sizeof(float), k_bytes) ||
            !safe_mul(hd.v_data.size(), sizeof(float), v_bytes) ||
            !safe_add(k_bytes, v_bytes, row_bytes) ||
            !safe_add(total_hot_bytes, row_bytes, total_hot_bytes)) {
            if (err) *err = "overflow computing hot row bytes";
            return false;
        }
    }

    if (max_hot_bytes > 0 && total_hot_bytes > max_hot_bytes) {
        if (err) *err = "hot row total bytes (" + std::to_string(total_hot_bytes) +
            ") exceeds max_hot_bytes bound (" + std::to_string(max_hot_bytes) + ")";
        return false;
    }

    return true;
}

bool xkv_graph_snapshot::validate_hot_store_bindings(std::string * err) const {
    if (store == nullptr) {
        return true;
    }

    for (size_t i = 0; i < hot_data.size(); ++i) {
        const auto & hd = hot_data[i];
        if (hd.payload_id == 0 || !hd.is_valid) {
            continue;
        }

        xkv_location loc = {};
        if (!store->find_location(hd.payload_id, loc)) {
            if (err) *err = "hot row " + std::to_string(i) + " payload_id " +
                std::to_string(hd.payload_id) + " not found in store";
            return false;
        }

        if (loc.kind != xkv_location_kind::hot) {
            if (err) *err = "hot row " + std::to_string(i) + " payload_id " +
                std::to_string(hd.payload_id) + " location is not hot";
            return false;
        }

        if (loc.row != hd.row_index) {
            if (err) *err = "hot row " + std::to_string(i) + " payload_id " +
                std::to_string(hd.payload_id) + " row mismatch: store row " +
                std::to_string(loc.row) + " != snapshot row " + std::to_string(hd.row_index);
            return false;
        }

        if (hd.storage_generation != 0 && loc.storage_generation != hd.storage_generation) {
            if (err) *err = "hot row " + std::to_string(i) + " payload_id " +
                std::to_string(hd.payload_id) + " generation mismatch: store gen " +
                std::to_string(loc.storage_generation) + " != snapshot gen " +
                std::to_string(hd.storage_generation);
            return false;
        }

        if (loc.state != hd.expected_state) {
            if (err) *err = "hot row " + std::to_string(i) + " payload_id " +
                std::to_string(hd.payload_id) + " state mismatch: store state " +
                std::to_string(static_cast<uint32_t>(loc.state)) + " != expected state " +
                std::to_string(static_cast<uint32_t>(hd.expected_state));
            return false;
        }
    }

    return true;
}

std::vector<xkv_hot_row> xkv_graph_snapshot::build_hot_rows() const {
    std::vector<xkv_hot_row> rows;
    rows.reserve(hot_data.size());
    for (const auto & hd : hot_data) {
        xkv_hot_row r;
        r.row_index = hd.row_index;
        r.storage_pos = hd.storage_pos;
        r.group_index = hd.group_index;
        r.is_valid = hd.is_valid;
        r.query_visibility = hd.query_visibility;
        // Per-query group descriptors: prefer the builder-filled mapping when
        // present (verbatim copy; the reader validates sizes downstream),
        // else derive from visibility (visible queries use the row's group,
        // invisible carry UINT32_MAX). Empty visibility (visible to all)
        // leaves the array empty and the reader falls back to group_index.
        if (!hd.query_group_indices.empty()) {
            r.query_group_indices = hd.query_group_indices;
        } else if (!hd.query_visibility.empty()) {
            r.query_group_indices.assign(hd.query_visibility.size(), UINT32_MAX);
            for (size_t q = 0; q < hd.query_visibility.size(); ++q) {
                if (hd.query_visibility[q]) r.query_group_indices[q] = hd.group_index;
            }
        }
        r.k_ptr = hd.k_data.empty() ? nullptr : hd.k_data.data();
        r.v_ptr = hd.v_data.empty() ? nullptr : hd.v_data.data();
        rows.push_back(std::move(r));
    }
    return rows;
}

bool xkv_graph_snapshot::build_query_inputs(
    const float * q_tensor_data,
    size_t q_tensor_elements,
    std::vector<xkv_query_input> & out_queries,
    std::string * err
) const {
    out_queries.clear();
    if (!q_tensor_data) {
        if (err) *err = "build_query_inputs: null q_tensor_data";
        return false;
    }

    if (head_dim_k == 0 || head_dim_v == 0 || n_q_heads == 0) {
        if (err) *err = "build_query_inputs: zero dimensions or heads";
        return false;
    }

    size_t req_elements = 0;
    if (!get_required_q_elements(req_elements, err)) {
        return false;
    }
    if (q_tensor_elements < req_elements) {
        if (err) *err = "build_query_inputs: q_tensor_elements (" +
            std::to_string(q_tensor_elements) + ") < required elements (" +
            std::to_string(req_elements) + ")";
        return false;
    }

    size_t q_head_elements = static_cast<size_t>(n_q_heads) * head_dim_k;
    out_queries.resize(n_queries);

    size_t current_offset = 0;
    for (uint32_t q = 0; q < n_queries; ++q) {
        auto & qi = out_queries[q];
        qi.query_index = q;
        qi.n_q_heads = n_q_heads;
        qi.head_dim_k = head_dim_k;
        qi.head_dim_v = head_dim_v;
        qi.scale = scale;
        qi.logit_softcap = logit_softcap;

        if (q < query_causal_limits.size()) {
            qi.causal_limit_pos = query_causal_limits[q];
        } else {
            qi.causal_limit_pos = -1;
        }

        if (q < query_sink_logits.size()) {
            qi.sink_logits = query_sink_logits[q];
        }

        size_t q_offset = 0;
        if (!query_ddvr_group_offsets.empty()) {
            q_offset = query_ddvr_group_offsets[q];
        } else if (!query_ddvr_group_counts.empty()) {
            q_offset = current_offset;
        } else {
            size_t q_stride_elements = (n_ddvr_groups > 0 ? n_ddvr_groups : 1) * q_head_elements;
            q_offset = static_cast<size_t>(q) * q_stride_elements;
        }

        uint32_t q_groups = get_query_group_count(q);
        if (q_groups > 0) {
            qi.q_groups.resize(q_groups);
            for (uint32_t g = 0; g < q_groups; ++g) {
                size_t g_offset = q_offset + static_cast<size_t>(g) * q_head_elements;
                if (g_offset + q_head_elements > q_tensor_elements) {
                    if (err) *err = "build_query_inputs: query " + std::to_string(q) +
                        " group " + std::to_string(g) + " exceeds q_tensor_elements";
                    return false;
                }
                const float * g_ptr = q_tensor_data + g_offset;
                qi.q_groups[g].assign(g_ptr, g_ptr + q_head_elements);
            }
            if (query_ddvr_group_offsets.empty() && !query_ddvr_group_counts.empty()) {
                current_offset += static_cast<size_t>(q_groups) * q_head_elements;
            }
        } else {
            if (q_offset + q_head_elements > q_tensor_elements) {
                if (err) *err = "build_query_inputs: query " + std::to_string(q) +
                    " exceeds q_tensor_elements";
                return false;
            }
            const float * ptr = q_tensor_data + q_offset;
            qi.q_vec.assign(ptr, ptr + q_head_elements);
            if (query_ddvr_group_offsets.empty() && !query_ddvr_group_counts.empty()) {
                current_offset += q_head_elements;
            }
        }

        if (!qi.validate(err)) {
            return false;
        }
    }

    return true;
}

xkv_graph_op_handle::xkv_graph_op_handle(std::unique_ptr<xkv_graph_snapshot> snapshot)
    : snapshot_(std::move(snapshot)) {
}

xkv_graph_op_handle::~xkv_graph_op_handle() {
    magic = 0;
}

void xkv_graph_op_handle::ggml_custom_op_callback(struct ggml_tensor * dst, int ith, int nth, void * userdata) {
    (void) nth;
    LLAMA_LOG_ERROR("[xkv_graph_op_handle] ggml_custom_op_callback: ith=%d nth=%d userdata=%p\n", ith, nth, userdata);
    if (!userdata) {
        LLAMA_LOG_ERROR("[xkv_graph_op_handle] ggml_custom_op_callback: null userdata\n");
        if (dst && dst->data) {
            std::memset(dst->data, 0, ggml_nbytes(dst));
        }
        return;
    }

    auto * handle = static_cast<xkv_graph_op_handle *>(userdata);
    if (handle->magic != HANDLE_MAGIC) {
        LLAMA_LOG_ERROR("[xkv_graph_op_handle] ggml_custom_op_callback: bad magic 0x%llx\n",
            (unsigned long long) handle->magic);
        if (dst && dst->data) {
            std::memset(dst->data, 0, ggml_nbytes(dst));
        }
        return;
    }
    handle->compute(dst, ith, nth);
}

void xkv_graph_op_handle::compute(struct ggml_tensor * dst, int ith, int nth) noexcept {
    (void) nth;
    // Only thread task 0 performs execution to ensure deterministic reference behavior
    LLAMA_LOG_ERROR("[xkv_graph_op_handle] compute ith=%d nth=%d snapshot=%p\n", ith, nth, (void*)snapshot_.get());
    if (ith != 0) {
        return;
    }

    if (!snapshot_) {
        if (dst && dst->data) {
            std::memset(dst->data, 0, ggml_nbytes(dst));
        }
        return;
    }

    auto & snap = *snapshot_;
    snap.is_computed = true;
    snap.last_status = xkv_read_status::invalid_argument;
    snap.last_error = "compute entered without error update";

    // Fail-closed initialization: zero out the output destination tensor completely
    if (dst && dst->data) {
        std::memset(dst->data, 0, ggml_nbytes(dst));
    } else {
        snap.last_status = xkv_read_status::invalid_argument;
        snap.last_error = "null destination tensor or data buffer";
        return;
    }

    // Forced stale retry (failed snapshot refresh in set_input): report
    // stale before any reads so the outer loop rebuilds. Output stays zeroed.
    if (snap.force_retry_stale) {
        snap.last_status = xkv_read_status::retry_stale_stamp;
        snap.last_error = "forced stale retry after failed snapshot refresh";
        return;
    }

    // Checked output shape arithmetic
    size_t ne0_calc = 0;
    if (!safe_mul(static_cast<size_t>(snap.head_dim_v), static_cast<size_t>(snap.n_q_heads), ne0_calc)) {
        snap.last_status = xkv_read_status::invalid_argument;
        snap.last_error = "overflow computing expected destination ne0";
        return;
    }

    int64_t expected_ne0 = static_cast<int64_t>(ne0_calc);
    int64_t expected_ne1 = static_cast<int64_t>(snap.n_queries);

    if (dst->ne[0] != expected_ne0 || dst->ne[1] != expected_ne1 || dst->ne[2] != 1 || dst->ne[3] != 1 || dst->type != GGML_TYPE_F32) {
        snap.last_status = xkv_read_status::invalid_argument;
        snap.last_error = "invalid destination shape or type: expected [" +
            std::to_string(expected_ne0) + ", " + std::to_string(expected_ne1) + "], got [" +
            std::to_string(dst->ne[0]) + ", " + std::to_string(dst->ne[1]) + "]";
        LLAMA_LOG_ERROR("[xkv_graph_op_handle] %s\n", snap.last_error.c_str());
        return;
    }

    if (!ggml_is_contiguous(dst)) {
        snap.last_status = xkv_read_status::invalid_argument;
        snap.last_error = "destination tensor must be contiguous";
        LLAMA_LOG_ERROR("[xkv_graph_op_handle] %s\n", snap.last_error.c_str());
        return;
    }

    size_t total_dst_elements = 0;
    if (!safe_mul(ne0_calc, static_cast<size_t>(snap.n_queries), total_dst_elements)) {
        snap.last_status = xkv_read_status::invalid_argument;
        snap.last_error = "overflow computing total destination elements";
        return;
    }
    size_t expected_dst_bytes = 0;
    if (!safe_mul(total_dst_elements, sizeof(float), expected_dst_bytes) || ggml_nbytes(dst) < expected_dst_bytes) {
        snap.last_status = xkv_read_status::invalid_argument;
        snap.last_error = "destination tensor byte size truncated or overflow";
        return;
    }

    // Verify Q source tensor (src[0])
    const struct ggml_tensor * q_tensor = dst->src[0];
    if (!q_tensor || !q_tensor->data) {
        snap.last_status = xkv_read_status::invalid_argument;
        snap.last_error = "missing or null source q_tensor";
        LLAMA_LOG_ERROR("[xkv_graph_op_handle] %s\n", snap.last_error.c_str());
        return;
    }

    if (q_tensor->type != GGML_TYPE_F32) {
        snap.last_status = xkv_read_status::invalid_argument;
        snap.last_error = "source q_tensor must be F32";
        return;
    }

    if (!ggml_is_contiguous(q_tensor)) {
        snap.last_status = xkv_read_status::invalid_argument;
        snap.last_error = "source q_tensor must be contiguous";
        return;
    }

    size_t q_nelements = static_cast<size_t>(ggml_nelements(q_tensor));
    size_t expected_q_bytes = 0;
    if (!safe_mul(q_nelements, sizeof(float), expected_q_bytes) || ggml_nbytes(q_tensor) < expected_q_bytes) {
        snap.last_status = xkv_read_status::invalid_argument;
        snap.last_error = "source q_tensor byte size truncated or overflow";
        return;
    }

    // Validate explicit dependencies if configured
    for (size_t d = 0; d < snap.explicit_dependencies.size(); ++d) {
        const struct ggml_tensor * expected_dep = snap.explicit_dependencies[d];
        const struct ggml_tensor * actual_dep = dst->src[1 + d];
        if (!actual_dep || actual_dep != expected_dep) {
            snap.last_status = xkv_read_status::invalid_argument;
            snap.last_error = "explicit dependency mismatch or null at index " + std::to_string(1 + d);
            LLAMA_LOG_ERROR("[xkv_graph_op_handle] %s\n", snap.last_error.c_str());
            return;
        }
        if (!actual_dep->data) {
            snap.last_status = xkv_read_status::invalid_argument;
            snap.last_error = "explicit dependency data null at index " + std::to_string(1 + d);
            LLAMA_LOG_ERROR("[xkv_graph_op_handle] %s\n", snap.last_error.c_str());
            return;
        }
    }

    // Validate hot rows: bounds, dimensions, and visibility
    std::string hot_err;
    if (!snap.validate_hot_rows(&hot_err)) {
        snap.last_status = xkv_read_status::invalid_argument;
        snap.last_error = "hot rows validation failed: " + hot_err;
        LLAMA_LOG_ERROR("[xkv_graph_op_handle] %s\n", snap.last_error.c_str());
        return;
    }

    // ---- Bounded-hot wiring: sink parse + storage layout validation ----
    // Gather itself happens below, inside the try block, so any failure keeps
    // the zeroed output and a fail-closed status.
    bool needs_gather = false;
    for (const auto & hd : snap.hot_data) {
        if (hd.is_valid && hd.use_storage_gather()) { needs_gather = true; break; }
    }
    const ggml_tensor * k_storage = nullptr;
    const ggml_tensor * v_storage = nullptr;
    auto resolve_dep = [&](int idx, const char * name, const ggml_tensor *& out, std::string & err) -> bool {
        if (idx < 0 || idx >= GGML_MAX_SRC) {
            err = std::string("invalid ") + name + " dep index " + std::to_string(idx);
            return false;
        }
        out = dst->src[idx];
        if (!out) {
            err = std::string("missing ") + name + " dependency tensor at src " + std::to_string(idx);
            return false;
        }
        return true;
    };
    // ---- Snapshot-owned workspace gate (P0-WS) ----
    // Production bounded snapshots (workspace_required) must arrive with a
    // backing lease + layout + bound reader workspace. Budget>0 without
    // backing fails closed here (workspace_exceeded), never legacy heap.
    // Default snapshots keep the preflight-bounded heap path below.
    bool span_mode = false;
    if (snap.workspace_required) {
        if (!snap.workspace_ready()) {
            snap.last_status = xkv_read_status::workspace_exceeded;
            snap.last_error = "bounded snapshot without backing workspace (budget>0, null workspace)";
            LLAMA_LOG_ERROR("[xkv_graph_op_handle] %s\n", snap.last_error.c_str());
            return;
        }
        if (!snap.reader_config.workspace) {
            snap.last_status = xkv_read_status::workspace_exceeded;
            snap.last_error = "bounded snapshot without reader workspace";
            LLAMA_LOG_ERROR("[xkv_graph_op_handle] %s\n", snap.last_error.c_str());
            return;
        }
        // Region consistency: every slice must fit inside the declared total.
        // (The layout function guarantees this; hand-built layouts fail here.)
        const auto & lay = snap.workspace_layout;
        const size_t total = lay.total_bytes;
        const size_t regs[7][2] = {
            {lay.sink_off, lay.sink_bytes}, {lay.bulk_k_off, lay.bulk_k_bytes},
            {lay.bulk_v_off, lay.bulk_v_bytes}, {lay.deq_off, lay.deq_bytes},
            {lay.gather_off, lay.gather_bytes}, {lay.rot_off, lay.rot_bytes},
            {lay.reader_off, lay.reader_bytes},
        };
        for (const auto & r : regs) {
            if (r[0] > total || r[1] > total - r[0]) {
                snap.last_status = xkv_read_status::workspace_exceeded;
                snap.last_error = "bounded snapshot workspace region outside total";
                return;
            }
        }
        // Non-overlap: slices must be pairwise disjoint (zero-byte empty).
        for (size_t a = 0; a < 7; ++a) {
            for (size_t b = a + 1; b < 7; ++b) {
                if (regs[a][1] == 0 || regs[b][1] == 0) continue;
                size_t a_end = 0, b_end = 0;
                if (!safe_add(regs[a][0], regs[a][1], a_end) || !safe_add(regs[b][0], regs[b][1], b_end)) {
                    snap.last_status = xkv_read_status::invalid_argument;
                    snap.last_error = "bounded snapshot workspace region overflow";
                    return;
                }
                if (regs[a][0] < b_end && regs[b][0] < a_end) {
                    snap.last_status = xkv_read_status::workspace_exceeded;
                    snap.last_error = "bounded snapshot workspace regions overlap";
                    return;
                }
            }
        }
        span_mode = true;
    }
    if (snap.sink_dep >= 0 || needs_gather) {
        std::string dep_err;
        if (snap.sink_dep >= 0) {
            const ggml_tensor * sink_t = nullptr;
            if (!resolve_dep(snap.sink_dep, "sink", sink_t, dep_err)) {
                snap.last_status = xkv_read_status::invalid_argument;
                snap.last_error = dep_err;
                return;
            }
            if (sink_t->type != GGML_TYPE_F32) {
                snap.last_status = xkv_read_status::invalid_argument;
                snap.last_error = "sink dependency tensor must be F32";
                return;
            }
            uint32_t total_q = snap.n_q_heads_total != 0 ? snap.n_q_heads_total : snap.n_q_heads;
            if (total_q == 0 || snap.n_q_heads == 0) {
                snap.last_status = xkv_read_status::invalid_argument;
                snap.last_error = "sink parse with zero head counts";
                return;
            }
            if (sink_t->ne[0] != (int64_t) total_q || (sink_t->ne[1] != 1 && sink_t->ne[1] != (int64_t) snap.n_queries)) {
                snap.last_status = xkv_read_status::invalid_argument;
                snap.last_error = "sink dependency shape mismatch";
                return;
            }
            size_t n_sink = static_cast<size_t>(sink_t->ne[0]) * static_cast<size_t>(sink_t->ne[1]);
            size_t sink_bytes = 0;
            if (!safe_mul(n_sink, sizeof(float), sink_bytes) || ggml_nbytes(sink_t) < sink_bytes) {
                snap.last_status = xkv_read_status::invalid_argument;
                snap.last_error = "sink dependency byte size truncated";
                return;
            }
            // Span-backed when the snapshot owns a lease (zero heap); the
            // region bound was preflighted against the declared budget.
            float * sink_ptr = nullptr;
            std::vector<float> sink_heap;
            if (span_mode) {
                if (snap.workspace_layout.sink_bytes < sink_bytes) {
                    snap.last_status = xkv_read_status::workspace_exceeded;
                    snap.last_error = "sink region short for sink tensor";
                    return;
                }
                sink_ptr = reinterpret_cast<float *>(snap.workspace_base() + snap.workspace_layout.sink_off);
            } else {
                sink_heap.resize(n_sink);
                sink_ptr = sink_heap.data();
            }
            // Device pointers are never dereferenced: non-host residency
            // forces an explicit bounded backend copy.
            if (snap.hot_storage_host_resident && sink_t->data) {
                std::memcpy(sink_ptr, sink_t->data, sink_bytes);
            } else {
                ggml_backend_tensor_get(sink_t, sink_ptr, 0, sink_bytes);
            }
            // Parse exactly once per query/Q-head: each (query, local head)
            // takes the slice [q_group_begin, q_group_begin + n_q_heads).
            if (snap.q_group_begin + snap.n_q_heads > total_q) {
                snap.last_status = xkv_read_status::invalid_argument;
                snap.last_error = "sink parse: GQA group slice out of range";
                return;
            }
            // Required snapshots use build-sized storage (zero heap here);
            // default snapshots grow it (heap, preflight-bounded).
            if (snap.workspace_required) {
                if (snap.query_sink_logits.size() != snap.n_queries) {
                    snap.last_status = xkv_read_status::invalid_argument;
                    snap.last_error = "sink cache query count mismatch";
                    return;
                }
                for (uint32_t q = 0; q < snap.n_queries; ++q) {
                    if (snap.query_sink_logits[q].size() != snap.n_q_heads) {
                        snap.last_status = xkv_read_status::invalid_argument;
                        snap.last_error = "sink cache head count mismatch";
                        return;
                    }
                }
            } else {
                snap.query_sink_logits.assign(snap.n_queries, std::vector<float>(snap.n_q_heads));
            }
            bool per_query = sink_t->ne[1] != 1;
            for (uint32_t q = 0; q < snap.n_queries; ++q) {
                for (uint32_t h = 0; h < snap.n_q_heads; ++h) {
                    size_t src_idx = per_query
                        ? static_cast<size_t>(q) * total_q + snap.q_group_begin + h
                        : static_cast<size_t>(snap.q_group_begin) + h;
                    float v = sink_ptr[src_idx];
                    if (!std::isfinite(v)) {
                        snap.last_status = xkv_read_status::invalid_argument;
                        snap.last_error = "sink dependency contains non-finite value";
                        return;
                    }
                    snap.query_sink_logits[q][h] = v;
                }
            }
        }
        if (needs_gather) {
            if (snap.hot_layout.v_transposed) {
                snap.last_status = xkv_read_status::invalid_argument;
                snap.last_error = "transposed V layout is unsupported for hot gather";
                return;
            }
            if (!resolve_dep(snap.k_storage_dep, "k_storage", k_storage, dep_err) ||
                !resolve_dep(snap.v_storage_dep, "v_storage", v_storage, dep_err)) {
                snap.last_status = xkv_read_status::invalid_argument;
                snap.last_error = dep_err;
                return;
            }
            if (k_storage->type != snap.hot_layout.k_type || v_storage->type != snap.hot_layout.v_type) {
                snap.last_status = xkv_read_status::invalid_argument;
                snap.last_error = "storage tensor type does not match hot layout descriptor";
                LLAMA_LOG_ERROR("[xkv_graph_op_handle] %s\n", snap.last_error.c_str());
                return;
            }
            if (snap.hot_layout.head_dim_k != snap.head_dim_k || snap.hot_layout.head_dim_v != snap.head_dim_v) {
                snap.last_status = xkv_read_status::invalid_argument;
                snap.last_error = "hot layout head dims do not match snapshot dims";
                LLAMA_LOG_ERROR("[xkv_graph_op_handle] %s\n", snap.last_error.c_str());
                return;
            }
            if (is_turbo_type(k_storage->type) && snap.expected_turbo_fp_k != 0 &&
                ggml_turbo_layout_fingerprint(k_storage->type) != snap.expected_turbo_fp_k) {
                snap.last_status = xkv_read_status::invalid_argument;
                snap.last_error = "turbo K layout fingerprint mismatch";
                return;
            }
            if (is_turbo_type(v_storage->type) && snap.expected_turbo_fp_v != 0 &&
                ggml_turbo_layout_fingerprint(v_storage->type) != snap.expected_turbo_fp_v) {
                snap.last_status = xkv_read_status::invalid_argument;
                snap.last_error = "turbo V layout fingerprint mismatch";
                return;
            }
        }
    }

    // Validate hot payload store bindings (row + generation + state) before read
    std::string binding_err;
    if (!snap.validate_hot_store_bindings(&binding_err)) {
        snap.last_status = xkv_read_status::invalid_argument;
        snap.last_error = "hot store bindings pre-validation failed: " + binding_err;
        LLAMA_LOG_ERROR("[xkv_graph_op_handle] %s\n", snap.last_error.c_str());
        return;
    }

    // Check stamp before compute: must match expected_stamp fail-closed
    if (snap.store != nullptr) {
        xkv_snapshot_stamp current_stamp = snap.store->current_stamp();
        if (current_stamp != snap.expected_stamp) {
            snap.last_status = xkv_read_status::retry_stale_stamp;
            snap.last_error = "snapshot stamp mismatch before compute (stale stamp)";
            LLAMA_LOG_ERROR("[xkv_graph_op_handle] %s\n", snap.last_error.c_str());
            return;
        }
    }

    try {
        const float * q_raw = static_cast<const float *>(q_tensor->data);
        std::string err;
        // Required snapshots fill runtime-preallocated caches (zero heap in
        // callback); default snapshots use the heap builder. Both produce
        // identical query contents (covered by oracle tests on each path).
        std::vector<xkv_query_input> queries_heap;
        std::vector<xkv_query_input> * queries_ptr = nullptr;
        if (snap.workspace_required) {
            std::string verr;
            if (!snap.compute_caches_valid(&verr)) {
                snap.last_status = xkv_read_status::invalid_argument;
                snap.last_error = "stale descriptor caches: " + verr;
                return;
            }
            if (!fill_required_queries(snap, q_raw, q_nelements, &err)) {
                snap.last_status = xkv_read_status::invalid_argument;
                snap.last_error = "fill_required_queries failed: " + err;
                return;
            }
            queries_ptr = &snap.query_cache;
        } else {
            if (!snap.build_query_inputs(q_raw, q_nelements, queries_heap, &err)) {
                snap.last_status = xkv_read_status::invalid_argument;
                snap.last_error = "build_query_inputs failed: " + err;
                LLAMA_LOG_ERROR("[xkv_graph_op_handle] %s\n", snap.last_error.c_str());
                return;
            }
            queries_ptr = &queries_heap;
        }
        std::vector<xkv_query_input> & queries = *queries_ptr;

        // Check if there are any queries
        if (queries.empty()) {
            snap.last_status = xkv_read_status::success;
            snap.last_error.clear();
            return;
        }

        // Assemble hot rows pointing into owned hot_data
        // Hot rows: required snapshots fill preallocated caches (visibility
        // copied at build); default snapshots build owned views (heap).
        std::vector<xkv_hot_row> hot_rows_heap;
        std::vector<xkv_hot_row> * hot_rows_ptr = nullptr;
        if (snap.workspace_required) {
            if (!fill_required_hot(snap, &err)) {
                snap.last_status = xkv_read_status::invalid_argument;
                snap.last_error = "fill_required_hot failed: " + err;
                return;
            }
            hot_rows_ptr = &snap.hot_cache;
        } else {
            hot_rows_heap = snap.build_hot_rows();
            hot_rows_ptr = &hot_rows_heap;
        }
        std::vector<xkv_hot_row> & hot_rows = *hot_rows_ptr;

        // SR-after-Q (P0-2): SR mode runs bounded selection now that Q
        // exists, independently per query/head. Dense/off modes keep the
        // precomputed sr_selection (dense streams all legal rows via views).
        // No copy of the precomputed selection in the common path (mode 0).
        const sr_batch_selection_result * sel_ptr = &snap.sr_selection;
        sr_batch_selection_result computed_sel;
        if (snap.sr_mode == 1) {
            // Required snapshots use the bounded core (global budget with
            // owning-segment attribution, zero heap). Default snapshots keep
            // the heap merge path (exact when representable, else refused
            // inside run_sr_after_q).
            if (snap.workspace_required) {
                std::string berr;
                xkv_read_status bst = run_sr_bounded_after_q(snap, queries, &berr);
                if (bst != xkv_read_status::success) {
                    snap.last_status = bst;
                    snap.last_error = "bounded SR selection failed: " + berr;
                    return;
                }
                sel_ptr = &snap.sr_result_cache;
            } else {
                std::string sr_err;
                if (!run_sr_after_q(snap, queries, computed_sel, &sr_err)) {
                    snap.last_status = xkv_read_status::invalid_argument;
                    snap.last_error = "SR-after-Q selection failed: " + sr_err;
                    return;
                }
                sel_ptr = &computed_sel;
            }
        }

        // Gather exact physical hot slots from the scheduler-visible storage
        // tensors now, after writes have executed. Fallback rows keep owned data.
        std::vector<std::vector<float>> gather_k;
        std::vector<std::vector<float>> gather_v;
        if (needs_gather) {
            uint32_t pad_k = snap.hot_layout.eff_padded_k();
            uint32_t pad_v = snap.hot_layout.eff_padded_v();
            if (pad_k == 0) pad_k = snap.head_dim_k;
            if (pad_v == 0) pad_v = snap.head_dim_v;
            if ((uint32_t) k_storage->ne[0] != pad_k || (uint32_t) v_storage->ne[0] != pad_v) {
                snap.last_status = xkv_read_status::invalid_argument;
                snap.last_error = "storage row width does not match hot layout padded dims";
                return;
            }
            // Full-row element counts (checked; geometry re-validates inside fetch).
            int64_t nk_floats = 0, nv_floats = 0;
            if (k_storage->ne[0] == 0 || k_storage->ne[1] > INT64_MAX / k_storage->ne[0] ||
                v_storage->ne[0] == 0 || v_storage->ne[1] > INT64_MAX / v_storage->ne[0]) {
                snap.last_status = xkv_read_status::invalid_argument;
                snap.last_error = "storage row element count overflow";
                return;
            }
            nk_floats = k_storage->ne[0] * k_storage->ne[1];
            nv_floats = v_storage->ne[0] * v_storage->ne[1];
            // Collect gather positions, preflight bounded temps (P0-5), then
            // ONE bulk copy per K/V layer (P0-4: K bulk + V bulk = two syncs
            // on device; single-sync needs async on one backend handle).
            // Required snapshots use runtime-preallocated index caches
            // (sized + validated above; zero heap here). Default snapshots
            // build locals (heap, preflight-bounded).
            const std::vector<int64_t> * cells_ptr = nullptr;
            const std::vector<int64_t> * pos_ptr = nullptr;
            std::vector<int64_t> gather_pos_local;
            std::vector<int64_t> gather_cells_local;
            if (snap.workspace_required) {
                cells_ptr = &snap.gather_cells_cache;
                pos_ptr = &snap.gather_pos_cache;
            } else {
                gather_pos_local.assign(snap.hot_data.size(), -1);
                for (size_t i = 0; i < snap.hot_data.size(); ++i) {
                    const auto & hd0 = snap.hot_data[i];
                    if (!hd0.is_valid || !hd0.use_storage_gather()) continue;
                    gather_pos_local[i] = (int64_t) gather_cells_local.size();
                    gather_cells_local.push_back(hd0.cell);
                }
                cells_ptr = &gather_cells_local;
                pos_ptr = &gather_pos_local;
            }
            const std::vector<int64_t> & gather_cells = *cells_ptr;
            const std::vector<int64_t> & gather_pos = *pos_ptr;
            {
                bool over_budget = false;
                std::string perr;
                size_t sink_floats = 0;
                for (const auto & sq : snap.query_sink_logits) {
                    if (!safe_add(sink_floats, sq.size(), sink_floats)) break;
                }
                if (!preflight_gather_temps(snap, gather_cells.size(), nk_floats, nv_floats,
                        (size_t) k_storage->nb[2], (size_t) v_storage->nb[2], sink_floats, over_budget, &perr)) {
                    snap.last_status = over_budget ? xkv_read_status::workspace_exceeded : xkv_read_status::invalid_argument;
                    snap.last_error = perr;
                    return;
                }
            }
            std::vector<std::vector<uint8_t>> bulk_k, bulk_v;
            if (span_mode) {
                // Zero-heap span path: bulk rows, dequant floats, outputs and
                // rot tmp all live in the snapshot-owned lease slices.
                const xkv_callback_layout & lay = snap.workspace_layout;
                uint8_t * wbase = snap.workspace_base();
                std::string gerr;
                size_t row_k = 0, stride_k = 0, row_v = 0, stride_v = 0;
                if (!storage_row_geometry(k_storage, row_k, stride_k, &gerr) ||
                    !storage_row_geometry(v_storage, row_v, stride_v, &gerr)) {
                    snap.last_status = xkv_read_status::invalid_argument;
                    snap.last_error = "hot gather span geometry failed: " + gerr;
                    return;
                }
                size_t need_bk = 0, need_bv = 0, need_deq = 0, need_out = 0, deq_row = 0, out_row = 0;
                if (!safe_mul(gather_cells.size(), stride_k, need_bk) || need_bk > lay.bulk_k_bytes ||
                    !safe_mul(gather_cells.size(), stride_v, need_bv) || need_bv > lay.bulk_v_bytes) {
                    snap.last_status = xkv_read_status::workspace_exceeded;
                    snap.last_error = "hot gather span bulk region short";
                    return;
                }
                if (!safe_add((size_t) nk_floats, (size_t) nv_floats, deq_row) ||
                    !safe_mul(gather_cells.size(), deq_row, need_deq) ||
                    !safe_mul(need_deq, sizeof(float), need_deq) || need_deq > lay.deq_bytes) {
                    snap.last_status = xkv_read_status::workspace_exceeded;
                    snap.last_error = "hot gather span dequant region short";
                    return;
                }
                if (!safe_add((size_t) snap.head_dim_k, (size_t) snap.head_dim_v, out_row) ||
                    !safe_mul(gather_cells.size(), out_row, need_out) ||
                    !safe_mul(need_out, sizeof(float), need_out) || need_out > lay.gather_bytes) {
                    snap.last_status = xkv_read_status::workspace_exceeded;
                    snap.last_error = "hot gather span output region short";
                    return;
                }
                if (lay.rot_bytes < (size_t) std::max(pad_k, pad_v) * sizeof(float)) {
                    snap.last_status = xkv_read_status::workspace_exceeded;
                    snap.last_error = "hot gather span rot region short";
                    return;
                }
                uint8_t * bk = wbase + lay.bulk_k_off;
                uint8_t * bv = wbase + lay.bulk_v_off;
                float * deq_base = reinterpret_cast<float *>(wbase + lay.deq_off);
                float * out_base = reinterpret_cast<float *>(wbase + lay.gather_off);
                float * rot_tmp = reinterpret_cast<float *>(wbase + lay.rot_off);
                if (!gather_cells.empty()) {
                    if (snap.hot_storage_host_resident && k_storage->data && v_storage->data) {
                        for (size_t j = 0; j < gather_cells.size(); ++j) {
                            if (gather_cells[j] < 0 || gather_cells[j] >= k_storage->ne[2] ||
                                gather_cells[j] >= v_storage->ne[2]) {
                                snap.last_status = xkv_read_status::invalid_argument;
                                snap.last_error = "hot gather cell out of storage range";
                                return;
                            }
                            size_t rk = 0, rv = 0, ok = 0, ov = 0;
                            if (!safe_mul(j, stride_k, rk) ||
                                !safe_mul((size_t) gather_cells[j], stride_k, ok) ||
                                !safe_mul(j, stride_v, rv) ||
                                !safe_mul((size_t) gather_cells[j], stride_v, ov) ||
                                rk + row_k > lay.bulk_k_bytes ||
                                rv + row_v > lay.bulk_v_bytes) {
                                snap.last_status = xkv_read_status::invalid_argument;
                                snap.last_error = "hot gather span offset overflow";
                                return;
                            }
                            std::memcpy(bk + rk, static_cast<const uint8_t *>(k_storage->data) + ok, row_k);
                            std::memcpy(bv + rv, static_cast<const uint8_t *>(v_storage->data) + ov, row_v);
                        }
                    } else {
                        for (size_t j = 0; j < gather_cells.size(); ++j) {
                            if (gather_cells[j] < 0 || gather_cells[j] >= k_storage->ne[2] ||
                                gather_cells[j] >= v_storage->ne[2]) {
                                snap.last_status = xkv_read_status::invalid_argument;
                                snap.last_error = "hot gather cell out of storage range";
                                return;
                            }
                            size_t rk = 0, rv = 0, ok = 0, ov = 0;
                            if (!safe_mul(j, stride_k, rk) ||
                                !safe_mul((size_t) gather_cells[j], stride_k, ok) ||
                                !safe_mul(j, stride_v, rv) ||
                                !safe_mul((size_t) gather_cells[j], stride_v, ov) ||
                                rk + row_k > lay.bulk_k_bytes ||
                                rv + row_v > lay.bulk_v_bytes) {
                                snap.last_status = xkv_read_status::invalid_argument;
                                snap.last_error = "hot gather span offset overflow";
                                return;
                            }
                            ggml_backend_tensor_get(k_storage, bk + rk, ok, row_k);
                            ggml_backend_tensor_get(v_storage, bv + rv, ov, row_v);
                        }
                    }
                }
                for (size_t i = 0; i < snap.hot_data.size(); ++i) {
                    const auto & hd = snap.hot_data[i];
                    if (!hd.is_valid || !hd.use_storage_gather()) continue;
                    if (hd.stream != 0) {
                        snap.last_status = xkv_read_status::invalid_argument;
                        snap.last_error = "multi-stream hot gather is unsupported";
                        return;
                    }
                    if (hd.kv_head != snap.kv_head_index) {
                        snap.last_status = xkv_read_status::invalid_argument;
                        snap.last_error = "hot row KV head does not match snapshot KV head";
                        return;
                    }
                    if (hd.kv_head >= (uint32_t) k_storage->ne[1] || hd.kv_head >= (uint32_t) v_storage->ne[1]) {
                        snap.last_status = xkv_read_status::invalid_argument;
                        snap.last_error = "hot row KV head out of storage range";
                        return;
                    }
                    size_t bp = (size_t) gather_pos[i];
                    if (bp >= gather_cells.size()) {
                        snap.last_status = xkv_read_status::codec_error;
                        snap.last_error = "hot gather span position out of range";
                        return;
                    }
                    size_t rk = 0, rv = 0;
                    if (!safe_mul(bp, stride_k, rk) ||
                        !safe_mul(bp, stride_v, rv) ||
                        rk + row_k > lay.bulk_k_bytes ||
                        rv + row_v > lay.bulk_v_bytes) {
                        snap.last_status = xkv_read_status::invalid_argument;
                        snap.last_error = "hot gather span offset overflow";
                        return;
                    }
                    float * dek = deq_base + bp * deq_row;
                    float * dev = dek + nk_floats;
                    if (!dequant_storage_row_ptr(k_storage->type, bk + rk, nk_floats, dek, &gerr) ||
                        !dequant_storage_row_ptr(v_storage->type, bv + rv, nv_floats, dev, &gerr)) {
                        snap.last_status = xkv_read_status::codec_error;
                        snap.last_error = "hot gather span dequant failed: " + gerr;
                        return;
                    }
                    float * hk = dek + (size_t) hd.kv_head * pad_k;
                    float * hv = dev + (size_t) hd.kv_head * pad_v;
                    if (!xform_hot_head_span(hk, pad_k, snap.hot_k_inv_rot,
                            snap.hot_k_rot_dim ? snap.hot_k_rot_dim : pad_k, snap.hot_k_channel_mul, rot_tmp, &gerr) ||
                        !xform_hot_head_span(hv, pad_v, snap.hot_v_inv_rot,
                            snap.hot_v_rot_dim ? snap.hot_v_rot_dim : pad_v, snap.hot_v_channel_mul, rot_tmp, &gerr)) {
                        snap.last_status = xkv_read_status::invalid_argument;
                        snap.last_error = "hot gather span rotation failed: " + gerr;
                        return;
                    }
                    if (pad_k < snap.head_dim_k || pad_v < snap.head_dim_v) {
                        snap.last_status = xkv_read_status::invalid_argument;
                        snap.last_error = "hot gather padded width smaller than logical head dim";
                        return;
                    }
                    float * ok = out_base + bp * out_row;
                    float * ov = ok + snap.head_dim_k;
                    std::memcpy(ok, hk, (size_t) snap.head_dim_k * sizeof(float));
                    std::memcpy(ov, hv, (size_t) snap.head_dim_v * sizeof(float));
                    for (uint32_t d = 0; d < snap.head_dim_k; ++d) if (!std::isfinite(ok[d])) {
                        snap.last_status = xkv_read_status::codec_error;
                        std::string err_msg = "hot gather produced non-finite K (span_mode, row " + std::to_string(i) +
                            ", cell " + std::to_string(snap.hot_data[i].cell) + ", pid " + std::to_string(snap.hot_data[i].payload_id) +
                            ", pos " + std::to_string(snap.hot_data[i].storage_pos) + ", d " + std::to_string(d) +
                            ", val " + std::to_string(ok[d]) + ")";
                        LLAMA_LOG_ERROR("[xkv_graph_ref] %s\n", err_msg.c_str());
                        char hex[128] = {0};
                        for (size_t b = 0; b < std::min<size_t>(16, (size_t)snap.head_dim_k * sizeof(float)); ++b) {
                            snprintf(hex + b * 3, sizeof(hex) - b * 3, "%02x ", ((const uint8_t *)ok)[b]);
                        }
                        LLAMA_LOG_ERROR("[xkv_graph_ref] span ok hex: %s\n", hex);
                        snap.last_error = err_msg;
                        return;
                    }
                    for (uint32_t d = 0; d < snap.head_dim_v; ++d) if (!std::isfinite(ov[d])) {
                        snap.last_status = xkv_read_status::codec_error;
                        snap.last_error = "hot gather produced non-finite V";
                        return;
                    }
                    hot_rows[i].k_ptr = ok;
                    hot_rows[i].v_ptr = ov;
                }
            } else {
            gather_k.resize(snap.hot_data.size());
            gather_v.resize(snap.hot_data.size());
            {
                std::string gerr0;
                if (!fetch_hot_storage_rows(k_storage, gather_cells, snap.hot_storage_host_resident, bulk_k, &gerr0) ||
                    !fetch_hot_storage_rows(v_storage, gather_cells, snap.hot_storage_host_resident, bulk_v, &gerr0)) {
                    snap.last_status = xkv_read_status::invalid_argument;
                    snap.last_error = "hot gather bulk fetch failed: " + gerr0;
                    return;
                }
            }
            for (size_t i = 0; i < snap.hot_data.size(); ++i) {
                const auto & hd = snap.hot_data[i];
                if (!hd.is_valid || !hd.use_storage_gather()) continue;
                if (hd.stream != 0) {
                    snap.last_status = xkv_read_status::invalid_argument;
                    snap.last_error = "multi-stream hot gather is unsupported";
                    return;
                }
                if (hd.kv_head != snap.kv_head_index) {
                    snap.last_status = xkv_read_status::invalid_argument;
                    snap.last_error = "hot row KV head does not match snapshot KV head";
                    return;
                }
                if (hd.kv_head >= (uint32_t) k_storage->ne[1] || hd.kv_head >= (uint32_t) v_storage->ne[1]) {
                    snap.last_status = xkv_read_status::invalid_argument;
                    snap.last_error = "hot row KV head out of storage range";
                    return;
                }
                // Full-row bytes from the bulk copy (P0-3: bytes and decoded
                // elements now cover the same head-contiguous row).
                std::string gerr;
                size_t bp = (size_t) gather_pos[i];
                if (bp >= bulk_k.size() || bp >= bulk_v.size()) {
                    snap.last_status = xkv_read_status::codec_error;
                    snap.last_error = "hot gather bulk position out of range";
                    return;
                }
                const std::vector<uint8_t> & raw_k = bulk_k[bp];
                const std::vector<uint8_t> & raw_v = bulk_v[bp];
                std::vector<float> row_k, row_v;
                if (!dequant_storage_row(k_storage->type, raw_k.data(), nk_floats, row_k, &gerr)) {
                    snap.last_status = xkv_read_status::codec_error;
                    snap.last_error = "hot gather dequant K failed: " + gerr;
                    return;
                }
                bool k_nan = false;
                for (size_t di = 0; di < row_k.size(); ++di) {
                    if (!std::isfinite(row_k[di])) {
                        k_nan = true;
                        LLAMA_LOG_ERROR("[xkv_graph_ref] dequant raw K row %zu (bp=%zu, cell=%lld, pid=%llu, pos=%lld) has non-finite at elem %zu: %f\n",
                            i, bp, (long long)hd.cell, (unsigned long long)hd.payload_id, (long long)hd.storage_pos, di, row_k[di]);
                        break;
                    }
                }
                if (k_nan) {
                    // Log first 16 bytes of raw_k
                    char hex[128] = {0};
                    for (size_t b = 0; b < std::min<size_t>(16, raw_k.size()); ++b) {
                        snprintf(hex + b * 3, sizeof(hex) - b * 3, "%02x ", raw_k[b]);
                    }
                    LLAMA_LOG_ERROR("[xkv_graph_ref] raw_k hex bytes: %s (type=%d)\n", hex, (int)k_storage->type);
                }
                if (!dequant_storage_row(v_storage->type, raw_v.data(), nv_floats, row_v, &gerr)) {
                    snap.last_status = xkv_read_status::codec_error;
                    snap.last_error = "hot gather dequant failed: " + gerr;
                    return;
                }
                // Slice this KV head's padded block, invert the attention
                // rotation only (never inverse RoPE), apply channel scale,
                // then truncate to logical dims.
                std::vector<float> hk(row_k.begin() + (size_t) hd.kv_head * pad_k,
                                      row_k.begin() + (size_t)(hd.kv_head + 1) * pad_k);
                std::vector<float> hv(row_v.begin() + (size_t) hd.kv_head * pad_v,
                                      row_v.begin() + (size_t)(hd.kv_head + 1) * pad_v);
                if (!apply_inverse_rotation(hk, snap.hot_k_inv_rot, snap.hot_k_rot_dim ? snap.hot_k_rot_dim : pad_k, &gerr) ||
                    !apply_inverse_rotation(hv, snap.hot_v_inv_rot, snap.hot_v_rot_dim ? snap.hot_v_rot_dim : pad_v, &gerr)) {
                    snap.last_status = xkv_read_status::invalid_argument;
                    snap.last_error = "hot gather inverse rotation failed: " + gerr;
                    return;
                }
                // Identity when the multiplier vector is empty; size must
                // otherwise match the padded width exactly.
                if (!snap.hot_k_channel_mul.empty()) {
                    if (snap.hot_k_channel_mul.size() != pad_k) {
                        snap.last_status = xkv_read_status::invalid_argument;
                        snap.last_error = "hot K channel multiplier size mismatch";
                        return;
                    }
                    for (uint32_t d = 0; d < pad_k; ++d) hk[d] *= snap.hot_k_channel_mul[d];
                }
                if (!snap.hot_v_channel_mul.empty()) {
                    if (snap.hot_v_channel_mul.size() != pad_v) {
                        snap.last_status = xkv_read_status::invalid_argument;
                        snap.last_error = "hot V channel multiplier size mismatch";
                        return;
                    }
                    for (uint32_t d = 0; d < pad_v; ++d) hv[d] *= snap.hot_v_channel_mul[d];
                }
                if (pad_k < snap.head_dim_k || pad_v < snap.head_dim_v) {
                    snap.last_status = xkv_read_status::invalid_argument;
                    snap.last_error = "hot gather padded width smaller than logical head dim";
                    return;
                }
                hk.resize(snap.head_dim_k);
                hv.resize(snap.head_dim_v);
                for (float v : hk) if (!std::isfinite(v)) {
                    snap.last_status = xkv_read_status::codec_error;
                    std::string err_msg = "hot gather produced non-finite K (heap_mode, row " + std::to_string(i) +
                        ", cell " + std::to_string(snap.hot_data[i].cell) + ", pid " + std::to_string(snap.hot_data[i].payload_id) +
                        ", kv_head " + std::to_string(hd.kv_head) + " of " + std::to_string(k_storage->ne[1]) +
                        ", pad_k " + std::to_string(pad_k) + ", nk_floats " + std::to_string(nk_floats) +
                        ", raw_k_bytes " + std::to_string(raw_k.size()) +
                        ", pos " + std::to_string(snap.hot_data[i].storage_pos) +
                        ", val " + std::to_string(v) + ")";
                    LLAMA_LOG_ERROR("[xkv_graph_ref] %s\n", err_msg.c_str());
                    char hex[128] = {0};
                    for (size_t b = 0; b < std::min<size_t>(16, hk.size() * sizeof(float)); ++b) {
                        snprintf(hex + b * 3, sizeof(hex) - b * 3, "%02x ", ((const uint8_t *)hk.data())[b]);
                    }
                    LLAMA_LOG_ERROR("[xkv_graph_ref] heap hk hex: %s\n", hex);
                    snap.last_error = err_msg;
                    return;
                }
                for (float v : hv) if (!std::isfinite(v)) {
                    snap.last_status = xkv_read_status::codec_error;
                    snap.last_error = "hot gather produced non-finite V";
                    return;
                }
                gather_k[i] = std::move(hk);
                gather_v[i] = std::move(hv);
                hot_rows[i].k_ptr = gather_k[i].data();
                hot_rows[i].v_ptr = gather_v[i].data();
            }
            } // end heap path (span path sets hot_rows pointers inline above)
            // Unified completion check (both paths set hot_rows pointers inline).
            for (size_t i = 0; i < snap.hot_data.size() && i < hot_rows.size(); ++i) {
                if (snap.hot_data[i].is_valid && snap.hot_data[i].use_storage_gather() &&
                    (hot_rows[i].k_ptr == nullptr || hot_rows[i].v_ptr == nullptr)) {
                    snap.last_status = xkv_read_status::codec_error;
                    snap.last_error = "hot gather left a valid row undecoded";
                    return;
                }
            }
        }

        // Execute batch attention
        xkv_batch_read_result res = xkv_read_attention_batch(
            queries,
            hot_rows,
            snap.segment_views,
            *sel_ptr,
            snap.phase_tx,
            snap.expected_stamp,
            snap.store,
            snap.reader_config
        );

        snap.last_status = res.status;
        snap.last_error = res.error_message;
        snap.peak_workspace_bytes = res.peak_workspace_bytes;
        if (res.status != xkv_read_status::success) { LLAMA_LOG_ERROR("[xkv_graph_op_handle] dense attention failed: status=%d error=%s\n", (int)res.status, res.error_message.c_str()); }

        if (res.status != xkv_read_status::success) {
            // Already zeroed out; keep zeroed out
            return;
        }

        // Validate hot payload store bindings (row + generation + state) after read
        if (!snap.validate_hot_store_bindings(&binding_err)) {
            snap.last_status = xkv_read_status::invalid_argument;
            snap.last_error = "hot store bindings post-validation failed: " + binding_err;
            LLAMA_LOG_ERROR("[xkv_graph_op_handle] %s\n", snap.last_error.c_str());
            if (dst && dst->data) {
                std::memset(dst->data, 0, ggml_nbytes(dst));
            }
            return;
        }

        // Write output to destination tensor
        // dst layout: [head_dim_v * n_q_heads, n_queries] in column-major / row-major GGML:
        // dst->data[q * ne[0] + offset]
        float * dst_ptr = static_cast<float *>(dst->data);
        size_t query_out_elems = static_cast<size_t>(expected_ne0);

        for (uint32_t q = 0; q < snap.n_queries; ++q) {
            if (q < res.per_query.size() && !res.per_query[q].output.empty()) {
                const auto & q_out = res.per_query[q].output;
                size_t copy_elems = std::min(query_out_elems, q_out.size());
                int bad = 0;
                for (size_t di = 0; di < copy_elems; ++di) {
                    if (!std::isfinite(q_out[di])) ++bad;
                }
                if (bad > 0) {
                    LLAMA_LOG_ERROR("[xkv_compute] query %u (snap kv_head %u): %d/%zu output elems non-finite (res status=%d)\n",
                        q, snap.kv_head_index, bad, copy_elems, (int)res.per_query[q].status);
                }
                std::memcpy(dst_ptr + q * query_out_elems, q_out.data(), copy_elems * sizeof(float));
            }
        }

        snap.last_error.clear();
    } catch (const std::exception & e) {
        snap.last_status = xkv_read_status::codec_error;
        snap.last_error = std::string("compute exception: ") + e.what();
        LLAMA_LOG_ERROR("[xkv_graph_op_handle] %s\n", snap.last_error.c_str());
        if (dst && dst->data) {
            std::memset(dst->data, 0, ggml_nbytes(dst));
        }
    } catch (...) {
        snap.last_status = xkv_read_status::codec_error;
        snap.last_error = "unknown compute exception";
        LLAMA_LOG_ERROR("[xkv_graph_op_handle] %s\n", snap.last_error.c_str());
        if (dst && dst->data) {
            std::memset(dst->data, 0, ggml_nbytes(dst));
        }
    }
}

struct ggml_tensor * xkv_build_graph_attention_ref(
    struct ggml_context * ctx,
    struct ggml_tensor * q_tensor,
    std::unique_ptr<xkv_graph_snapshot> snapshot,
    std::shared_ptr<xkv_graph_op_handle> & out_handle,
    const std::vector<struct ggml_tensor *> & dependencies
) {
    if (!ctx || !q_tensor || !snapshot) {
        return nullptr;
    }

    // We need 1 slot for Q and N slots for explicit dependencies
    // Total args = 1 + dependencies.size()
    if (1 + dependencies.size() > GGML_MAX_SRC) {
        return nullptr;
    }

    // Checked arithmetic for output dimensions
    size_t ne0_calc = 0;
    if (!safe_mul(static_cast<size_t>(snapshot->head_dim_v), static_cast<size_t>(snapshot->n_q_heads), ne0_calc)) {
        return nullptr;
    }

    int64_t ne0 = static_cast<int64_t>(ne0_calc);
    int64_t ne1 = static_cast<int64_t>(snapshot->n_queries);
    int64_t ne2 = 1;
    int64_t ne3 = 1;

    // Record explicit dependencies in snapshot for compute validation
    snapshot->explicit_dependencies.clear();
    for (auto * dep : dependencies) {
        snapshot->explicit_dependencies.push_back(dep);
    }

    auto shared_handle = std::make_shared<xkv_graph_op_handle>(std::move(snapshot));
    void * userdata = shared_handle.get();

    struct ggml_tensor * args[GGML_MAX_SRC];
    args[0] = q_tensor;
    for (size_t i = 0; i < dependencies.size(); ++i) {
        args[1 + i] = dependencies[i];
    }
    int n_args = 1 + static_cast<int>(dependencies.size());

    struct ggml_tensor * dst = ggml_custom_4d(
        ctx,
        GGML_TYPE_F32,
        ne0,
        ne1,
        ne2,
        ne3,
        args,
        n_args,
        xkv_graph_op_handle::ggml_custom_op_callback,
        1, // single task deterministic reference execution
        userdata
    );

    out_handle = shared_handle;
    return dst;
}

// ==================== Bounded-hot graph runtime wiring ====================

namespace {

bool is_turbo_type(ggml_type t) {
    return t == GGML_TYPE_TURBO2_0 || t == GGML_TYPE_TURBO3_0 || t == GGML_TYPE_TURBO4_0;
}

// Fetch one full cell row (all heads, row width ne[0]) from a storage view
// tensor. Uses the direct data pointer on host tensors; otherwise performs an
// explicit bounded backend subrange copy. Never caches across computes.
// Full cell-row geometry for a [head, n_head, n_cell] storage view: one row
// spans all heads contiguously (nb[1] bytes) and cells stride by nb[2].
// Anything else (packed/transposed views) fails closed here, never silently
// mis-sliced. Row SPACE must span physical hot rows: every gathered cell is
// range-checked against ne[2] below (P0-10).
bool storage_row_geometry(const ggml_tensor * t, size_t & row_bytes, size_t & cell_stride, std::string * err) {
    if (!t) {
        if (err) *err = "storage_row_geometry: null storage tensor";
        return false;
    }
    if (t->ne[0] <= 0 || t->ne[1] <= 0 || t->ne[2] <= 0) {
        if (err) *err = "storage_row_geometry: non-positive storage dims";
        return false;
    }
    if (t->ne[0] != 0 && t->ne[1] > INT64_MAX / t->ne[0]) {
        if (err) *err = "storage_row_geometry: row element count overflow";
        return false;
    }
    // Full cell row spans all heads contiguously: logical row bytes cover
    // ne[0]*ne[1] elements, strided by nb[2]; nb[1] is one head's bytes.
    size_t expect_full = ggml_row_size(t->type, t->ne[0] * t->ne[1]);
    size_t expect_head = ggml_row_size(t->type, t->ne[0]);
    if (expect_full == 0 || expect_head == 0) {
        if (err) *err = "storage_row_geometry: zero row bytes";
        return false;
    }
    if (static_cast<size_t>(t->nb[1]) != expect_head) {
        if (err) *err = "storage_row_geometry: head stride does not match head bytes (view is not head-contiguous)";
        return false;
    }
    if (static_cast<size_t>(t->nb[2]) < expect_full) {
        if (err) *err = "storage_row_geometry: cell stride smaller than full-row bytes (overlapping rows)";
        return false;
    }
    row_bytes = expect_full;
    cell_stride = static_cast<size_t>(t->nb[2]);
    return true;
}

// Fetch full cell rows for the requested cells. Host path copies per row
// (no sync). Device path performs ONE bounded backend subrange copy over
// [min,max] and slices per cell: one bulk copy per call (K and V each take
// one sync, two total; never per-row sync).
bool fetch_hot_storage_rows(const ggml_tensor * t, const std::vector<int64_t> & cells, bool host_resident,
    std::vector<std::vector<uint8_t>> & out_rows, std::string * err) {
    out_rows.clear();
    out_rows.resize(cells.size());
    if (cells.empty()) return true;
    size_t row_bytes = 0, cell_stride = 0;
    if (!storage_row_geometry(t, row_bytes, cell_stride, err)) return false;
    int64_t min_c = cells[0], max_c = cells[0];
    for (int64_t c : cells) {
        if (c < 0 || c >= t->ne[2]) {
            if (err) *err = "fetch_hot_storage_rows: cell " + std::to_string(c) +
                " out of storage row space [0, " + std::to_string(t->ne[2]) + ")";
            return false;
        }
        min_c = std::min(min_c, c);
        max_c = std::max(max_c, c);
    }
    if (host_resident && t->data) {
        const uint8_t * base = static_cast<const uint8_t *>(t->data);
        for (size_t i = 0; i < cells.size(); ++i) {
            out_rows[i].resize(row_bytes);
            size_t off = 0;
            if (!safe_mul(static_cast<size_t>(cells[i]), cell_stride, off)) {
                if (err) *err = "fetch_hot_storage_rows: offset overflow";
                return false;
            }
            std::memcpy(out_rows[i].data(), base + off, row_bytes);
        }
        return true;
    }
    // One bulk device-to-host span copy, then slice. Never per-row sync.
    size_t span_rows = 0;
    if (!safe_add(static_cast<size_t>(max_c - min_c), 1, span_rows)) {
        if (err) *err = "fetch_hot_storage_rows: span overflow";
        return false;
    }
    size_t span_bytes = 0, base_off = 0;
    if (!safe_mul(static_cast<size_t>(min_c), cell_stride, base_off)) {
        if (err) *err = "fetch_hot_storage_rows: base offset overflow";
        return false;
    }
    // Span covers whole strides; last row needs row_bytes past its start.
    size_t tail = 0;
    if (!safe_mul(span_rows, cell_stride, tail)) {
        if (err) *err = "fetch_hot_storage_rows: span overflow";
        return false;
    }
    span_bytes = tail; // stride >= row_bytes, so the span covers every row fully
    std::vector<uint8_t> span(span_bytes);
    ggml_backend_tensor_get(t, span.data(), base_off, span_bytes);
    for (size_t i = 0; i < cells.size(); ++i) {
        out_rows[i].resize(row_bytes);
        size_t rel = 0;
        if (!safe_mul(static_cast<size_t>(cells[i] - min_c), cell_stride, rel)) {
            if (err) *err = "fetch_hot_storage_rows: slice overflow";
            return false;
        }
        std::memcpy(out_rows[i].data(), span.data() + rel, row_bytes);
    }
    return true;
}

// Dequantize a full storage row to canonical floats. Turbo types decode in the
// canonical domain (inverts the write-time WHT); other types use type traits.
bool dequant_storage_row_ptr(ggml_type type, const uint8_t * raw, int64_t n_elements, float * out, std::string * err) {
    if (n_elements <= 0 || !raw || !out) {
        if (err) *err = "dequant_storage_row: non-positive element count or null buffer";
        return false;
    }
    // Decoded hot F32 rows are already canonical floats: gather directly.
    // Only encoded factor codecs (Turbo/quantized) go through dequantize.
    if (type == GGML_TYPE_F32) {
        size_t nbytes = 0;
        if (!safe_mul((size_t) n_elements, sizeof(float), nbytes)) {
            if (err) *err = "dequant_storage_row: F32 byte size overflow";
            return false;
        }
        std::memcpy(out, raw, nbytes);
        return true;
    }
    if (is_turbo_type(type)) {
        if (n_elements % 128 != 0) {
            if (err) *err = "dequant_storage_row: turbo row elements not a multiple of 128";
            return false;
        }
        if (!ggml_dequantize_turbo_row(type, raw, out, n_elements, 128, GGML_TURBO_DECODE_CANONICAL)) {
            if (err) *err = "dequant_storage_row: turbo canonical decode failed";
            return false;
        }
        return true;
    }
    const ggml_type_traits * traits = ggml_get_type_traits(type);
    if (!traits || !traits->to_float) {
        if (err) *err = std::string("dequant_storage_row: unsupported codec ") + ggml_type_name(type);
        return false;
    }
    int64_t blck = ggml_blck_size(type);
    if (blck <= 0 || n_elements % blck != 0) {
        if (err) *err = "dequant_storage_row: row elements not a multiple of block size";
        return false;
    }
    traits->to_float(raw, out, n_elements);
    return true;
}

bool dequant_storage_row(ggml_type type, const uint8_t * raw, int64_t n_elements, std::vector<float> & out, std::string * err) {
    if (n_elements <= 0) {
        if (err) *err = "dequant_storage_row: non-positive element count";
        return false;
    }
    out.resize(static_cast<size_t>(n_elements));
    return dequant_storage_row_ptr(type, raw, n_elements, out.data(), err);
}

// y = M * x with row-major M (dim x dim), in place via scratch.
bool apply_inverse_rotation(std::vector<float> & vec, const std::vector<float> & mat, uint32_t rot_dim, std::string * err) {
    if (mat.empty()) return true;
    if (rot_dim == 0 || vec.size() == 0 || vec.size() < rot_dim || vec.size() % rot_dim != 0) {
        if (err) *err = "apply_inverse_rotation: dimension mismatch";
        return false;
    }
    size_t need = 0;
    if (!safe_mul(static_cast<size_t>(rot_dim), static_cast<size_t>(rot_dim), need) || mat.size() != need) {
        if (err) *err = "apply_inverse_rotation: matrix size mismatch";
        return false;
    }
    std::vector<float> tmp(rot_dim);
    for (size_t b = 0; b < vec.size(); b += rot_dim) {
        for (uint32_t i = 0; i < rot_dim; ++i) {
            double acc = 0.0;
            for (uint32_t j = 0; j < rot_dim; ++j) acc += static_cast<double>(mat[i * rot_dim + j]) * vec[b + j];
            tmp[i] = static_cast<float>(acc);
        }
        for (uint32_t i = 0; i < rot_dim; ++i) {
            vec[b + i] = tmp[i];
        }
    }
    return true;
}

} // namespace

namespace {
std::mutex & native_cap_mutex() {
    static std::mutex m;
    return m;
}
std::map<std::string, bool> & native_cap_map() {
    static std::map<std::string, bool> map;
    return map;
}
} // namespace

void xkv_register_native_reconstruct_capability(const char * backend_name, bool available) {
    if (!backend_name || !backend_name[0]) return;
    std::lock_guard<std::mutex> lock(native_cap_mutex());
    native_cap_map()[std::string(backend_name)] = available;
}

bool xkv_native_reconstruct_available_for(const char * backend_name) {
    if (!backend_name || !backend_name[0]) return false;
    std::lock_guard<std::mutex> lock(native_cap_mutex());
    auto it = native_cap_map().find(std::string(backend_name));
    return it != native_cap_map().end() && it->second;
}

// Bounded temporary planner for in-compute callback allocations (P0-5).
// All callback/query/sink/hot-gather temporaries are preflighted here with
// checked arithmetic against the caller-configured workspace budget. Heap is
// still used below, but it is hard-bounded by the declared budget; zero-heap
// execution additionally requires a ReaderBounded scratch API for the hot
// path (coordinated; reader internals already honor their workspace).
bool preflight_gather_temps(const xkv_graph_snapshot & snap, size_t n_gather,
    int64_t nk_floats, int64_t nv_floats, size_t cell_stride_k, size_t cell_stride_v,
    size_t sink_floats, bool & over_budget, std::string * err) {
    over_budget = false;
    size_t budget = snap.reader_config.workspace_budget_bytes;
    if (budget == 0) return true; // no bound declared (legacy heap path)
    size_t need = 0, tmp = 0;
    auto acc = [&](size_t v) -> bool {
        return safe_add(need, v, need);
    };
    if (!safe_mul(sink_floats, sizeof(float), tmp) || !acc(tmp)) goto overflow;
    if (!safe_mul(n_gather, cell_stride_k, tmp) || !acc(tmp)) goto overflow;
    if (!safe_mul(n_gather, cell_stride_v, tmp) || !acc(tmp)) goto overflow;
    if (nk_floats < 0 || nv_floats < 0) goto overflow;
    {
        size_t fl = 0;
        if (!safe_add(static_cast<size_t>(nk_floats), static_cast<size_t>(nv_floats), fl)) goto overflow;
        if (!safe_mul(n_gather, fl, tmp)) goto overflow;
        if (!safe_mul(tmp, sizeof(float), tmp) || !acc(tmp)) goto overflow;
    }
    {
        size_t od = 0;
        if (!safe_add(static_cast<size_t>(snap.head_dim_k), static_cast<size_t>(snap.head_dim_v), od)) goto overflow;
        if (!safe_mul(n_gather, od, tmp)) goto overflow;
        if (!safe_mul(tmp, sizeof(float), tmp) || !acc(tmp)) goto overflow;
    }
    {
        size_t pad = std::max(snap.hot_layout.eff_padded_k(), snap.hot_layout.eff_padded_v());
        if (!safe_mul(pad == 0 ? 1 : pad, sizeof(float), tmp) || !acc(tmp)) goto overflow;
    }
    if (need > budget) {
        over_budget = true;
        if (err) *err = "compute temps (" + std::to_string(need) +
            " bytes) exceed workspace budget (" + std::to_string(budget) + " bytes)";
        return false;
    }
    return true;
overflow:
    if (err) *err = "preflight_gather_temps: size overflow";
    return false;
}

// Combined lease layout (P0-WS): every region 64-aligned (reader external
// backing requires it); overflow fails. Bulk regions reserve n*stride (upper
// bound on any [min,max] span). Query/hot/gather descriptors are
// runtime-preallocated snapshot metadata sized from build-known dims.
bool xkv_compute_callback_layout(uint32_t n_gather, size_t cell_stride_k, size_t cell_stride_v,
    size_t nk_floats, size_t nv_floats, uint32_t head_dim_k, uint32_t head_dim_v, uint32_t pad_max,
    size_t sink_floats, size_t reader_bytes, size_t sr_scratch_bytes,
    xkv_callback_layout & out, std::string * err) {
    xkv_callback_layout lay;
    size_t off = 0;
    auto place = [&](size_t n, size_t & o, size_t & nb) -> bool {
        if (!safe_align(off, 64, off)) return false;
        o = off;
        nb = n;
        return safe_add(off, n, off);
    };
    size_t tmp = 0;
    if (!safe_mul(sink_floats, sizeof(float), tmp) || !place(tmp, lay.sink_off, lay.sink_bytes)) goto overflow;
    if (!safe_mul(n_gather, cell_stride_k, tmp) || !place(tmp, lay.bulk_k_off, lay.bulk_k_bytes)) goto overflow;
    if (!safe_mul(n_gather, cell_stride_v, tmp) || !place(tmp, lay.bulk_v_off, lay.bulk_v_bytes)) goto overflow;
    {
        size_t fl = 0;
        if (!safe_add(nk_floats, nv_floats, fl)) goto overflow;
        if (!safe_mul(n_gather, fl, tmp)) goto overflow;
        if (!safe_mul(tmp, sizeof(float), tmp) || !place(tmp, lay.deq_off, lay.deq_bytes)) goto overflow;
    }
    {
        size_t od = 0;
        if (!safe_add(static_cast<size_t>(head_dim_k), static_cast<size_t>(head_dim_v), od)) goto overflow;
        if (!safe_mul(n_gather, od, tmp)) goto overflow;
        if (!safe_mul(tmp, sizeof(float), tmp) || !place(tmp, lay.gather_off, lay.gather_bytes)) goto overflow;
    }
    if (!safe_mul(static_cast<size_t>(pad_max), sizeof(float), tmp) || !place(tmp, lay.rot_off, lay.rot_bytes)) goto overflow;
    if (!place(reader_bytes, lay.reader_off, lay.reader_bytes)) goto overflow;
    if (!place(sr_scratch_bytes, lay.sr_off, lay.sr_bytes)) goto overflow;
    lay.total_bytes = off;
    out = lay;
    return true;
overflow:
    if (err) *err = "xkv_compute_callback_layout: size overflow";
    return false;
}

// Size descriptor caches from build-known dims. Mirrors the sizing branches
// of build_query_inputs (counts vs n_ddvr_groups, groups vs vec) and
// build_hot_rows, plus the gather index structure. BUILD-time allocation.
bool xkv_graph_snapshot::preallocate_compute_state(std::string * err) {
    query_cache.clear();
    hot_cache.clear();
    gather_cells_cache.clear();
    gather_pos_cache.clear();
    if (head_dim_k == 0 || head_dim_v == 0 || n_q_heads == 0) {
        if (err) *err = "preallocate_compute_state: zero dims or heads";
        return false;
    }
    if (!query_ddvr_group_counts.empty() && query_ddvr_group_counts.size() != n_queries) {
        if (err) *err = "preallocate_compute_state: group counts size mismatch";
        return false;
    }
    if (!query_ddvr_group_offsets.empty() && query_ddvr_group_offsets.size() != n_queries) {
        if (err) *err = "preallocate_compute_state: group offsets size mismatch";
        return false;
    }
    size_t per_head = 0;
    if (!safe_mul(static_cast<size_t>(n_q_heads), static_cast<size_t>(head_dim_k), per_head)) {
        if (err) *err = "preallocate_compute_state: q head elements overflow";
        return false;
    }
    query_cache.resize(n_queries);
    // Sink presence must mirror the heap path exactly: the reader treats
    // non-empty sink_logits as an active sink term. Forcing [nq][nh] zeros
    // when the builder declared no sinks would add a spurious exp(0) term.
    const bool want_sink = !query_sink_logits.empty() || sink_dep >= 0;
    for (uint32_t q = 0; q < n_queries; ++q) {
        auto & qq = query_cache[q];
        qq.query_index = q;
        qq.n_q_heads = n_q_heads;
        qq.head_dim_k = head_dim_k;
        qq.head_dim_v = head_dim_v;
        uint32_t gcount = !query_ddvr_group_counts.empty() ? query_ddvr_group_counts[q] : n_ddvr_groups;
        if (gcount > 0) {
            qq.q_groups.resize(gcount);
            for (uint32_t g = 0; g < gcount; ++g) qq.q_groups[g].resize(per_head);
            qq.q_vec.clear();
        } else {
            qq.q_vec.resize(per_head);
            qq.q_groups.clear();
        }
        if (want_sink) {
        qq.sink_logits.resize(n_q_heads);
        } else {
            qq.sink_logits.clear();
        }
    }
    hot_cache.resize(hot_data.size());
    for (size_t i = 0; i < hot_data.size(); ++i) {
        const auto & hd = hot_data[i];
        auto & hr = hot_cache[i];
        hr.query_visibility = hd.query_visibility;
        // Pre-size per-query group descriptors from the builder mapping when
        // present (verbatim copy), else visibility-sized for compute fill.
        // Values refreshed in compute; sizing here is build-time only.
        if (!hd.query_group_indices.empty()) {
            hr.query_group_indices = hd.query_group_indices;
        } else {
            hr.query_group_indices.assign(hd.query_visibility.size(), UINT32_MAX);
        }
    }
    // Sink logits storage (values refreshed in compute from the sink dep).
    // Sized without disturbing builder-prefilled values
    // (compute refreshes from the sink dep when present; otherwise the
    // prefilled values flow exactly as on the heap path).
    if (want_sink) {
        if (query_sink_logits.size() != n_queries) query_sink_logits.resize(n_queries);
        for (uint32_t q = 0; q < n_queries; ++q) query_sink_logits[q].resize(n_q_heads);
    }
    gather_pos_cache.assign(hot_data.size(), -1);
    for (size_t i = 0; i < hot_data.size(); ++i) {
        const auto & hd = hot_data[i];
        if (!hd.is_valid || !hd.use_storage_gather()) continue;
        gather_pos_cache[i] = static_cast<int64_t>(gather_cells_cache.size());
        gather_cells_cache.push_back(hd.cell);
    }
    // Bounded SR selection resources (sized when SR mode will select).
    // Fragment views alias frag storage (no copy); selector queries mirror
    // the run-time group expansion with sized floats; segment table follows
    // frag first-appearance order with pins resolved from views.
    sr_frag_views.clear();
    sr_selector_queries.clear();
    sr_selector_parents.clear();
    sr_seg_table.clear();
    sr_expanded_gather.clear();
    sr_expanded_ptrs.clear();
    sr_expanded_indices.clear();
    sr_result_cache = sr_batch_selection_result();
    sr_padded_rank = 0;
    sr_max_frag_rows = 0;
    sr_scratch_bytes = 0;
    if (sr_mode == 1 && !sr_legal_frags.empty()) {
        if (sr_legal_frags.size() > UINT32_MAX) {
            if (err) *err = "preallocate_compute_state: too many frags";
            return false;
        }
        sr_frag_views.resize(sr_legal_frags.size());
        size_t total_rows = 0;
        for (size_t i = 0; i < sr_legal_frags.size(); ++i) {
            const auto & f = sr_legal_frags[i];
            sr_frag_views[i] = view_of_fragment(f);
            if (f.row_indices.size() > UINT32_MAX) {
                if (err) *err = "preallocate_compute_state: frag too large";
                return false;
            }
            if ((uint32_t) f.row_indices.size() > sr_max_frag_rows) {
                sr_max_frag_rows = (uint32_t) f.row_indices.size();
            }
            if (!safe_add(total_rows, f.row_indices.size(), total_rows)) {
                if (err) *err = "preallocate_compute_state: frag rows overflow";
                return false;
            }
            bool known = false;
            for (const auto * s : sr_seg_table) {
                if (s && s->segment_id == f.key.segment_id) { known = true; break; }
            }
            if (!known) {
                const xkv_segment * segptr = nullptr;
                for (const auto & v : segment_views) {
                    const xkv_segment * s = v.get_segment();
                    if (s && s->segment_id == f.key.segment_id) { segptr = s; break; }
                }
                if (!segptr) {
                    if (err) *err = "preallocate_compute_state: frags reference unpinned segment " +
                        std::to_string(f.key.segment_id);
                    return false;
                }
                sr_seg_table.push_back(segptr);
            }
        }
        // Expanded per-(query,group) selector queries (counts mirror the
        // run-time expansion exactly).
        for (uint32_t q = 0; q < n_queries; ++q) {
            uint32_t ng = !query_ddvr_group_counts.empty() ? query_ddvr_group_counts[q] : n_ddvr_groups;
            if (ng == 0) ng = 1;
            for (uint32_t g = 0; g < ng; ++g) {
                sr_query s;
                s.q_vec.resize(per_head);
                s.q_head_indices.resize(n_q_heads);
                for (uint32_t h = 0; h < n_q_heads; ++h) s.q_head_indices[h] = h;
                s.head_dim = head_dim_k;
                sr_selector_queries.push_back(std::move(s));
                sr_selector_parents.push_back(q);
            }
        }
        // Padded rank: max A width over pinned views (conservative scratch bound).
        for (const auto & v : segment_views) {
            std::string gerr2;
            const auto * grp = v.resolve_group(&gerr2);
            if (!grp || !grp->b_k) {
                if (err) *err = "preallocate_compute_state: unresolvable factor group: " + gerr2;
                return false;
            }
            uint64_t pr = grp->a_k.desc.padded_shape.cols;
            if (pr > sr_padded_rank) {
                if (pr > UINT32_MAX) {
                    if (err) *err = "preallocate_compute_state: padded rank overflow";
                    return false;
                }
                sr_padded_rank = (uint32_t) pr;
            }
        }
        // Output caps: gather unions and CSR index space bounded by total rows.
        if (total_rows > UINT32_MAX) {
            if (err) *err = "preallocate_compute_state: total rows overflow";
            return false;
        }
        sr_expanded_gather.resize(total_rows);
        sr_expanded_indices.resize(total_rows);
        sr_expanded_ptrs.assign(sr_selector_queries.size() + 1, 0);
        sr_result_cache.gather_rows.resize(total_rows);
        sr_result_cache.csr_ptrs.assign(n_queries + 1, 0);
        sr_result_cache.csr_indices.resize(total_rows);
        std::string serr;
        if (!landmark_select_scratch_bytes((uint32_t) sr_legal_frags.size(), n_q_heads, head_dim_k,
                sr_max_frag_rows, sr_padded_rank, sr_scratch_bytes, &serr)) {
            if (err) *err = "preallocate_compute_state: scratch sizing failed: " + serr;
            return false;
        }
    }
    std::string verr;
    if (!compute_caches_valid(&verr)) {
        if (err) *err = "preallocate_compute_state: self-check failed: " + verr;
        return false;
    }
    return true;
}

bool xkv_graph_snapshot::compute_caches_valid(std::string * err) const {
    if (query_cache.size() != n_queries) {
        if (err) *err = "compute_caches_valid: query count mismatch";
        return false;
    }
    size_t per_head = static_cast<size_t>(n_q_heads) * head_dim_k;
    const bool want_sink_cached = !query_sink_logits.empty() || sink_dep >= 0;
    for (uint32_t q = 0; q < n_queries; ++q) {
        const auto & qq = query_cache[q];
        if (qq.n_q_heads != n_q_heads || qq.head_dim_k != head_dim_k || qq.head_dim_v != head_dim_v) {
            if (err) *err = "compute_caches_valid: query dims mismatch";
            return false;
        }
        uint32_t gcount = !query_ddvr_group_counts.empty()
            ? (q < query_ddvr_group_counts.size() ? query_ddvr_group_counts[q] : UINT32_MAX)
            : n_ddvr_groups;
        if (gcount == UINT32_MAX) {
            if (err) *err = "compute_caches_valid: group counts size mismatch";
            return false;
        }
        if (gcount > 0) {
            if (qq.q_groups.size() != gcount) {
                if (err) *err = "compute_caches_valid: query groups mismatch";
                return false;
            }
            for (const auto & g : qq.q_groups) {
                if (g.size() != per_head) {
                    if (err) *err = "compute_caches_valid: query group floats mismatch";
                    return false;
                }
            }
        } else if (qq.q_vec.size() != per_head) {
            if (err) *err = "compute_caches_valid: query vec floats mismatch";
            return false;
        }
        if (qq.sink_logits.size() != (want_sink_cached ? n_q_heads : 0)) {
            if (err) *err = "compute_caches_valid: sink logits mismatch";
            return false;
        }
    }
    if (hot_cache.size() != hot_data.size()) {
        if (err) *err = "compute_caches_valid: hot count mismatch";
        return false;
    }
    if (want_sink_cached) {
        if (query_sink_logits.size() != n_queries) {
            if (err) *err = "compute_caches_valid: sink query count mismatch";
            return false;
        }
    }
    if (want_sink_cached) {
        for (uint32_t q = 0; q < n_queries; ++q) {
            if (query_sink_logits[q].size() != n_q_heads) {
                if (err) *err = "compute_caches_valid: sink head count mismatch";
                return false;
            }
        }
    }
    for (size_t i = 0; i < hot_data.size(); ++i) {
        if (hot_cache[i].query_visibility.size() != hot_data[i].query_visibility.size()) {
            if (err) *err = "compute_caches_valid: hot visibility mismatch";
            return false;
        }
        size_t want_groups = !hot_data[i].query_group_indices.empty()
            ? hot_data[i].query_group_indices.size()
            : hot_data[i].query_visibility.size();
        if (hot_cache[i].query_group_indices.size() != want_groups) {
            if (err) *err = "compute_caches_valid: hot group descriptor mismatch";
            return false;
        }
    }
    if (gather_pos_cache.size() != hot_data.size()) {
        if (err) *err = "compute_caches_valid: gather pos size mismatch";
        return false;
    }
    size_t n = 0;
    for (size_t i = 0; i < hot_data.size(); ++i) {
        const auto & hd = hot_data[i];
        bool want = hd.is_valid && hd.use_storage_gather();
        if (!want) {
            if (gather_pos_cache[i] != -1) {
                if (err) *err = "compute_caches_valid: gather pos must be -1";
                return false;
            }
            continue;
        }
        if (gather_pos_cache[i] != static_cast<int64_t>(n)) {
            if (err) *err = "compute_caches_valid: gather pos order mismatch";
            return false;
        }
        if (n >= gather_cells_cache.size() || gather_cells_cache[n] != hd.cell) {
            if (err) *err = "compute_caches_valid: gather cell mismatch";
            return false;
        }
        ++n;
    }
    if (gather_cells_cache.size() != n) {
        if (err) *err = "compute_caches_valid: gather cell count mismatch";
        return false;
    }
    // Bounded SR caches: required if and only if SR mode will select.
    bool want_sr = (sr_mode == 1 && !sr_legal_frags.empty());
    if (!want_sr) {
        if (!sr_frag_views.empty() || !sr_selector_queries.empty() || !sr_seg_table.empty() ||
            !sr_expanded_gather.empty() || !sr_expanded_ptrs.empty() || !sr_expanded_indices.empty() ||
            !sr_result_cache.gather_rows.empty() || !sr_result_cache.csr_ptrs.empty() ||
            !sr_result_cache.csr_indices.empty()) {
            if (err) *err = "compute_caches_valid: stale SR caches without SR mode";
            return false;
        }
        return true;
    }
    if (sr_frag_views.size() != sr_legal_frags.size()) {
        if (err) *err = "compute_caches_valid: frag view count mismatch";
        return false;
    }
    for (size_t i = 0; i < sr_legal_frags.size(); ++i) {
        // Pointer stability: views must alias current frag storage.
        if (sr_frag_views[i].key != &sr_legal_frags[i].key ||
            sr_frag_views[i].row_count != sr_legal_frags[i].row_indices.size()) {
            if (err) *err = "compute_caches_valid: frag view stale";
            return false;
        }
    }
    // Recompute expansion + totals independently and compare.
    size_t n_exp = 0;
    for (uint32_t q = 0; q < n_queries; ++q) {
        uint32_t ng = !query_ddvr_group_counts.empty() ? query_ddvr_group_counts[q] : n_ddvr_groups;
        if (ng == 0) ng = 1;
        n_exp += ng;
    }
    if (sr_selector_queries.size() != n_exp || sr_selector_parents.size() != n_exp) {
        if (err) *err = "compute_caches_valid: selector query count mismatch";
        return false;
    }
    size_t per = static_cast<size_t>(n_q_heads) * head_dim_k;
    for (const auto & s : sr_selector_queries) {
        if (s.q_vec.size() != per || s.q_head_indices.size() != n_q_heads || s.head_dim != head_dim_k) {
            if (err) *err = "compute_caches_valid: selector query size mismatch";
            return false;
        }
    }
    size_t total_rows = 0;
    for (const auto & f : sr_legal_frags) {
        if (!safe_add(total_rows, f.row_indices.size(), total_rows)) {
            if (err) *err = "compute_caches_valid: frag rows overflow";
            return false;
        }
    }
    if (sr_expanded_gather.size() != total_rows || sr_expanded_indices.size() != total_rows ||
        sr_expanded_ptrs.size() != n_exp + 1 || sr_result_cache.gather_rows.size() != total_rows ||
        sr_result_cache.csr_ptrs.size() != (size_t) n_queries + 1 || sr_result_cache.csr_indices.size() != total_rows) {
        if (err) *err = "compute_caches_valid: SR output caps mismatch";
        return false;
    }
    // Segment table must match frag first-appearance order with live pins.
    // Allocation-free: quadratic scans over tiny tables, no temporaries.
    size_t n_distinct = 0;
    for (size_t i = 0; i < sr_legal_frags.size(); ++i) {
        bool seen_before = false;
        for (size_t j = 0; j < i; ++j) {
            if (sr_legal_frags[j].key.segment_id == sr_legal_frags[i].key.segment_id) { seen_before = true; break; }
        }
        if (seen_before) continue;
        if (n_distinct >= sr_seg_table.size()) {
            if (err) *err = "compute_caches_valid: seg table size mismatch";
            return false;
        }
        const xkv_segment * s = sr_seg_table[n_distinct];
        if (!s || s->segment_id != sr_legal_frags[i].key.segment_id) {
            if (err) *err = "compute_caches_valid: seg table order mismatch";
            return false;
        }
        ++n_distinct;
    }
    if (n_distinct != sr_seg_table.size()) {
        if (err) *err = "compute_caches_valid: seg table size mismatch";
        return false;
    }
    return true;
}

// Fill preallocated query caches from the Q tensor (no allocation; sizes
// pre-validated). Mirrors build_query_inputs offset/branch structure.
bool fill_required_queries(const xkv_graph_snapshot & snap, const float * q_raw, size_t q_nelements, std::string * err) {
    size_t per_head = static_cast<size_t>(snap.n_q_heads) * snap.head_dim_k;
    size_t current_offset = 0;
    for (uint32_t q = 0; q < snap.n_queries; ++q) {
        auto & qq = const_cast<xkv_query_input &>(snap.query_cache[q]);
        qq.scale = snap.scale;
        qq.logit_softcap = snap.logit_softcap;
        qq.causal_limit_pos = (q < snap.query_causal_limits.size()) ? snap.query_causal_limits[q] : -1;
        if (q < snap.query_sink_logits.size()) {
            if (snap.query_sink_logits[q].size() != qq.sink_logits.size()) {
                if (err) *err = "fill_required_queries: sink size mismatch";
                return false;
            }
            std::memcpy(qq.sink_logits.data(), snap.query_sink_logits[q].data(), qq.sink_logits.size() * sizeof(float));
        }
        size_t q_offset = 0;
        if (!snap.query_ddvr_group_offsets.empty()) {
            q_offset = snap.query_ddvr_group_offsets[q];
        } else if (!snap.query_ddvr_group_counts.empty()) {
            q_offset = current_offset;
        } else {
            size_t stride = 0;
            if (!safe_mul((snap.n_ddvr_groups > 0 ? snap.n_ddvr_groups : 1), per_head, stride)) {
                if (err) *err = "fill_required_queries: stride overflow";
                return false;
            }
            if (!safe_mul(static_cast<size_t>(q), stride, q_offset)) {
                if (err) *err = "fill_required_queries: offset overflow";
                return false;
            }
        }
        uint32_t gcount = !snap.query_ddvr_group_counts.empty() ? snap.query_ddvr_group_counts[q] : snap.n_ddvr_groups;
        if (gcount > 0) {
            for (uint32_t g = 0; g < gcount; ++g) {
                size_t goff = 0, gend = 0;
                if (!safe_mul(static_cast<size_t>(g), per_head, goff) ||
                    !safe_add(q_offset, goff, goff) || !safe_add(goff, per_head, gend) || gend > q_nelements) {
                    if (err) *err = "fill_required_queries: group range invalid";
                    return false;
                }
                std::memcpy(qq.q_groups[g].data(), q_raw + goff, per_head * sizeof(float));
            }
            if (snap.query_ddvr_group_offsets.empty() && !snap.query_ddvr_group_counts.empty()) {
                size_t adv = 0;
                if (!safe_mul(static_cast<size_t>(gcount), per_head, adv) ||
                    !safe_add(current_offset, adv, current_offset)) {
                    if (err) *err = "fill_required_queries: offset overflow";
                    return false;
                }
            }
        } else {
            size_t gend = 0;
            if (!safe_add(q_offset, per_head, gend) || gend > q_nelements) {
                if (err) *err = "fill_required_queries: query range invalid";
                return false;
            }
            std::memcpy(qq.q_vec.data(), q_raw + q_offset, per_head * sizeof(float));
            if (snap.query_ddvr_group_offsets.empty() && !snap.query_ddvr_group_counts.empty()) {
                if (!safe_add(current_offset, per_head, current_offset)) {
                    if (err) *err = "fill_required_queries: offset overflow";
                    return false;
                }
            }
        }
        std::string verr;
        if (!qq.validate(&verr)) {
            if (err) *err = "fill_required_queries: invalid query: " + verr;
            return false;
        }
    }
    return true;
}

// Fill preallocated hot rows (scalars + fallback pointers; visibility was
// copied at build). Gather rows get nulls here; the gather stage sets them.
bool fill_required_hot(const xkv_graph_snapshot & snap, std::string * err) {
    if (snap.hot_cache.size() != snap.hot_data.size()) {
        if (err) *err = "fill_required_hot: hot count mismatch";
        return false;
    }
    for (size_t i = 0; i < snap.hot_data.size(); ++i) {
        const auto & hd = snap.hot_data[i];
        auto & hr = const_cast<xkv_hot_row &>(snap.hot_cache[i]);
        hr.row_index = hd.row_index;
        hr.storage_pos = hd.storage_pos;
        hr.group_index = hd.group_index;
        hr.is_valid = hd.is_valid;
        // Per-query groups mirror build_hot_rows: prefer the builder mapping
        // when present (memcpy into pre-sized storage, no alloc), else
        // derive from visibility (visible use the row's group).
        if (!hd.query_group_indices.empty()) {
            if (hr.query_group_indices.size() != hd.query_group_indices.size()) {
                if (err) *err = "fill_required_hot: group descriptor size mismatch";
                return false;
            }
            std::memcpy(hr.query_group_indices.data(), hd.query_group_indices.data(),
                hd.query_group_indices.size() * sizeof(uint32_t));
        } else {
            if (hr.query_group_indices.size() != hd.query_visibility.size()) {
                if (err) *err = "fill_required_hot: group descriptor size mismatch";
                return false;
            }
            for (size_t q = 0; q < hd.query_visibility.size(); ++q) {
                hr.query_group_indices[q] = hd.query_visibility[q] ? hd.group_index : UINT32_MAX;
            }
        }
        if (!hd.use_storage_gather()) {
            hr.k_ptr = hd.k_data.empty() ? nullptr : hd.k_data.data();
            hr.v_ptr = hd.v_data.empty() ? nullptr : hd.v_data.data();
        } else {
            hr.k_ptr = nullptr;
            hr.v_ptr = nullptr;
        }
    }
    return true;
}

// Head canonicalization on raw pointers (both heap and span paths): inverse
// attention rotation only (never inverse RoPE), then channel multipliers.
// Truncation to logical dims is the caller's copy. tmp must hold pad floats.
bool xform_hot_head_span(float * io, uint32_t pad, const std::vector<float> & inv_rot, uint32_t rot_dim,
    const std::vector<float> & chan_mul, float * tmp, std::string * err) {
    if (!io || pad == 0 || !tmp) {
        if (err) *err = "xform_hot_head_span: null buffer or zero width";
        return false;
    }
    if (!inv_rot.empty()) {
        if (rot_dim == 0 || pad < rot_dim || pad % rot_dim != 0) {
            if (err) *err = "xform_hot_head_span: rotation dimension mismatch";
            return false;
        }
        size_t need = 0;
        if (!safe_mul(static_cast<size_t>(rot_dim), static_cast<size_t>(rot_dim), need) || inv_rot.size() != need) {
            if (err) *err = "xform_hot_head_span: rotation matrix size mismatch";
            return false;
        }
        for (size_t b = 0; b < pad; b += rot_dim) {
            for (uint32_t i = 0; i < rot_dim; ++i) {
                double acc = 0.0;
                for (uint32_t j = 0; j < rot_dim; ++j) acc += static_cast<double>(inv_rot[i * rot_dim + j]) * io[b + j];
                tmp[i] = static_cast<float>(acc);
            }
            for (uint32_t i = 0; i < rot_dim; ++i) {
                io[b + i] = tmp[i];
            }
        }
    }
    if (!chan_mul.empty()) {
        if (chan_mul.size() != pad) {
            if (err) *err = "xform_hot_head_span: channel multiplier size mismatch";
            return false;
        }
        for (uint32_t d = 0; d < pad; ++d) io[d] *= chan_mul[d];
    }
    return true;
}

// SR-after-Q selection (P0-2): runs select_sr_batch after Q is available,
// expanding DDVR groups into isolated selector queries and merging each
// parent query's groups back into one CSR. Dense/off modes never enter here
// (they stream precomputed views); sr_mode 0 keeps snap.sr_selection.
bool run_sr_after_q(const xkv_graph_snapshot & snap, const std::vector<xkv_query_input> & queries,
    sr_batch_selection_result & out_sel, std::string * err) {
    out_sel = sr_batch_selection_result();
    if (snap.sr_legal_frags.empty()) return true; // no cold candidates; hot still flows
    std::vector<sr_query> sq;
    std::vector<uint32_t> sq_parent;
    std::vector<uint32_t> sq_group; // selector group index per expanded entry
    for (uint32_t q = 0; q < (uint32_t) queries.size(); ++q) {
        const auto & qq = queries[q];
        uint32_t ng = (uint32_t) qq.q_groups.size();
        if (ng == 0) ng = 1;
        for (uint32_t g = 0; g < ng; ++g) {
            sr_query s;
            s.query_index = (uint32_t) sq.size();
            s.query_virtual_pos = (int64_t) q;
            s.causal_limit_pos = qq.causal_limit_pos;
            if (!qq.q_groups.empty()) {
                s.q_vec = qq.q_groups[g];
            } else {
                s.q_vec = qq.q_vec;
            }
            s.q_head_indices.resize(qq.n_q_heads);
            for (uint32_t h = 0; h < qq.n_q_heads; ++h) s.q_head_indices[h] = h;
            s.head_dim = qq.head_dim_k;
            s.scale = qq.scale;
            sq.push_back(std::move(s));
            sq_parent.push_back(q);
            sq_group.push_back(g);
        }
    }
    if (sq.empty()) return true;
    uint32_t feat_dim = snap.sr_feature_dim != 0 ? snap.sr_feature_dim : snap.head_dim_k;
    // One GLOBAL selection across the union of all legal fragments from every
    // pinned segment: single top-k / normalized max-union with one budget and
    // one refine cap. Per-segment calls would multiply the budget and break
    // global ranking. The single segment* below is only consumed for partial
    // causal rebuilds and boundary refinement of its own fragments.
    std::map<uint64_t, const xkv_segment *> seg_by_id;
    for (const auto & v : snap.segment_views) {
        const xkv_segment * s = v.get_segment();
        if (s) seg_by_id[s->segment_id] = s;
    }
    for (const auto & f : snap.sr_legal_frags) {
        if (!seg_by_id.count(f.key.segment_id)) {
            if (err) *err = "run_sr_after_q: legal frags reference unpinned segment " +
                std::to_string(f.key.segment_id);
            return false;
        }
    }
    // Multi-segment ownership check: a single call can only attribute
    // partial-causal rebuilds and boundary refinement to one segment.
    // With several pinned segments, prove upfront that neither triggers
    // (no partial causal cut anywhere, refine NONE); otherwise fail closed
    // instead of silently splitting budget/caps per segment.
    const xkv_segment * segptr = (seg_by_id.size() == 1) ? seg_by_id.begin()->second : nullptr;
    if (seg_by_id.size() > 1) {
        bool need_owner = (snap.sr_config.refine_mode != LLAMA_XKV_LANDMARK_REFINE_NONE);
        if (!need_owner) {
            for (const auto & s : sq) {
                if (s.causal_limit_pos < 0) continue;
                for (const auto & f : snap.sr_legal_frags) {
                    if (f.storage_positions.empty()) continue;
                    int64_t mn = f.storage_positions.front(), mx = f.storage_positions.front();
                    for (int64_t p : f.storage_positions) { mn = std::min(mn, p); mx = std::max(mx, p); }
                    if (mn <= s.causal_limit_pos && mx > s.causal_limit_pos) { need_owner = true; break; }
                }
                if (need_owner) break;
            }
        }
        if (need_owner) {
            if (err) *err = "run_sr_after_q: multi-segment SR with causal-cut/refine needs XkvLandmarkBounded attributed global selection (pending); refusing split budget";
            return false;
        }
    }
    sr_batch_selection_result part;
    try {
        part = select_sr_batch(sq, snap.sr_legal_frags, snap.sr_config, segptr,
            snap.sr_feature_offset, feat_dim, snap.phase_tx, snap.sr_phase_fingerprint);
    } catch (const std::exception & e) {
        if (err) *err = std::string("run_sr_after_q: select failed: ") + e.what();
        return false;
    }
    if (part.csr_ptrs.size() != sq.size() + 1) {
        if (err) *err = "run_sr_after_q: selector CSR size mismatch";
        return false;
    }
    // Merge DDVR-group slices back into per-parent-query CSR membership.
    // Merge DDVR-group slices back into per-parent-query CSR membership with
    // per-(query,row) groups: each parent updates a row exactly once, with
    // its own Q group (first expanded slice wins deterministically); no
    // dedup overwrite, no double-count. Groups of one query are distinct Q
    // vectors (own selections); segments share one global budget above.
    std::vector<std::map<segment_row_ref, uint32_t>> parent_row_group(queries.size());
    for (size_t e = 0; e < sq.size(); ++e) {
        if (part.csr_ptrs[e] > part.csr_ptrs[e + 1] || part.csr_ptrs[e + 1] > part.csr_indices.size()) {
            if (err) *err = "run_sr_after_q: selector CSR range invalid";
            return false;
        }
        for (uint32_t c = part.csr_ptrs[e]; c < part.csr_ptrs[e + 1]; ++c) {
            uint32_t gi = part.csr_indices[c];
            if (gi >= part.gather_rows.size()) {
                if (err) *err = "run_sr_after_q: selector CSR index out of range";
                return false;
            }
            auto & rg = parent_row_group[sq_parent[e]];
            if (!rg.count(part.gather_rows[gi])) {
                rg[part.gather_rows[gi]] = sq_group[e];
            }
        }
    }
    std::map<segment_row_ref, uint32_t> union_index;
    std::vector<segment_row_ref> gather;
    for (size_t q = 0; q < parent_row_group.size(); ++q) {
        for (const auto & kv : parent_row_group[q]) {
            if (!union_index.count(kv.first)) {
                union_index[kv.first] = (uint32_t) gather.size();
                gather.push_back(kv.first);
            }
        }
    }
    out_sel.gather_rows = gather;
    out_sel.csr_ptrs.resize(queries.size() + 1);
    out_sel.csr_group_indices.clear();
    out_sel.csr_ptrs[0] = 0;
    for (size_t q = 0; q < queries.size(); ++q) {
        for (const auto & kv : parent_row_group[q]) {
            out_sel.csr_indices.push_back(union_index[kv.first]);
            out_sel.csr_group_indices.push_back(kv.second);
        }
        out_sel.csr_ptrs[q + 1] = (uint32_t) out_sel.csr_indices.size();
    }
    out_sel.total_refined_rows = part.total_refined_rows;
    out_sel.total_refine_cap_hits = part.total_refine_cap_hits;
    // per_query diagnostics are indexed by expanded (query,group) inputs and
    // cannot represent merged parents; the operative outputs are the merged
    // gather_rows/CSR above. Totals are preserved.
    return true;
}

// Bounded global SR selection (XkvLandmarkBounded): one call over concatenated
// frag views with the owning-segment table; global budget/refine in-core,
// partial rebuilds attributed per fragment. All outputs land in preallocated
// snapshot buffers; group slices merge per parent with in-place sort+unique.
// Zero heap when caches are valid (checked by the caller via
// compute_caches_valid). sr_*_exceeded maps to workspace_exceeded.
xkv_read_status run_sr_bounded_after_q(xkv_graph_snapshot & snap, const std::vector<xkv_query_input> & queries, std::string * err) {
    auto fail = [&](xkv_read_status st, const std::string & msg) {
        if (err) *err = msg;
        return st;
    };
    auto & res = snap.sr_result_cache;
    // No clears: preallocated buffers are reused in place below (index-fill +
    // shrink-down; shrinking never allocates). Totals reset (scalars).
    res.total_refined_rows = 0;
    res.total_refine_cap_hits = 0;
    if (snap.sr_legal_frags.empty()) {
        res.gather_rows.resize(0);
        res.csr_indices.resize(0);
        if (res.csr_ptrs.size() != (size_t) snap.n_queries + 1) {
            return fail(xkv_read_status::invalid_argument, "run_sr_bounded: csr ptrs size mismatch");
        }
        for (uint32_t q = 0; q <= (uint32_t) snap.n_queries; ++q) res.csr_ptrs[q] = 0;
        return xkv_read_status::success; // hot still flows
    }
    // Refresh selector queries from live Q (memcpy into sized floats).
    // Fill in place; sizes pre-validated.
    size_t e = 0;
    for (uint32_t q = 0; q < snap.n_queries; ++q) {
        if (q >= queries.size()) return fail(xkv_read_status::invalid_argument, "run_sr_bounded: query count mismatch");
        const auto & qq = queries[q];
        uint32_t ng = (uint32_t) qq.q_groups.size();
        if (ng == 0) ng = 1;
        for (uint32_t g = 0; g < ng; ++g) {
            if (e >= snap.sr_selector_queries.size()) {
                return fail(xkv_read_status::invalid_argument, "run_sr_bounded: selector expansion mismatch");
            }
            auto & sq = snap.sr_selector_queries[e];
            const float * src = nullptr;
            size_t n = 0;
            if (!qq.q_groups.empty()) {
                if (g >= qq.q_groups.size()) return fail(xkv_read_status::invalid_argument, "run_sr_bounded: group index mismatch");
                src = qq.q_groups[g].data();
                n = qq.q_groups[g].size();
            } else {
                src = qq.q_vec.data();
                n = qq.q_vec.size();
            }
            if (n != sq.q_vec.size()) return fail(xkv_read_status::invalid_argument, "run_sr_bounded: selector q size mismatch");
            if (n > 0) std::memcpy(sq.q_vec.data(), src, n * sizeof(float));
            sq.query_index = (uint32_t) e;
            sq.query_virtual_pos = (int64_t) q;
            sq.causal_limit_pos = qq.causal_limit_pos;
            sq.scale = qq.scale;
            ++e;
        }
    }
    if (e != snap.sr_selector_queries.size()) {
        return fail(xkv_read_status::invalid_argument, "run_sr_bounded: selector expansion count mismatch");
    }
    // Scratch from the lease sr slice (span, bounded).
    uint8_t * wbase = snap.workspace_base();
    if (!wbase) return fail(xkv_read_status::workspace_exceeded, "run_sr_bounded: null workspace base");
    if (snap.workspace_layout.sr_bytes < snap.sr_scratch_bytes) {
        return fail(xkv_read_status::workspace_exceeded, "run_sr_bounded: sr scratch region short");
    }
    uint8_t * scr = wbase + snap.workspace_layout.sr_off;
    uint32_t feat_dim = snap.sr_feature_dim != 0 ? snap.sr_feature_dim : snap.head_dim_k;
    const xkv_segment * single = (snap.sr_seg_table.size() == 1) ? snap.sr_seg_table[0] : nullptr;
    size_t out_n_gather = 0, out_n_csr = 0;
    uint32_t out_refined = 0, out_hits = 0;
    bool ok = false;
    std::string berr;
    try {
        ok = select_sr_batch_bounded(
            snap.sr_selector_queries.data(), snap.sr_selector_queries.size(),
            snap.sr_frag_views.data(), snap.sr_frag_views.size(),
            snap.sr_config, single,
            snap.sr_feature_offset, feat_dim,
            snap.phase_tx, snap.sr_phase_fingerprint,
            scr, snap.sr_scratch_bytes,
            snap.sr_expanded_gather.data(), snap.sr_expanded_gather.size(), &out_n_gather,
            snap.sr_expanded_ptrs.data(), snap.sr_expanded_ptrs.size(),
            snap.sr_expanded_indices.data(), snap.sr_expanded_indices.size(), &out_n_csr,
            &out_refined, &out_hits,
            &berr,
            snap.sr_seg_table.data(), snap.sr_seg_table.size(),
            nullptr);
    } catch (const std::exception & ex) {
        return fail(xkv_read_status::codec_error, std::string("run_sr_bounded: select threw: ") + ex.what());
    }
    if (!ok) return fail(xkv_read_status::invalid_argument, "run_sr_bounded: select failed: " + berr);
    if (out_n_gather > snap.sr_expanded_gather.size() || out_n_csr > snap.sr_expanded_indices.size()) {
        return fail(xkv_read_status::codec_error, "run_sr_bounded: output overrun");
    }
    // Merge expanded slices per parent into the result cache (in place).
    // Expanded ptrs must exactly cover the expanded query count.
    if (snap.sr_expanded_ptrs.size() != snap.sr_selector_queries.size() + 1) {
        return fail(xkv_read_status::codec_error, "run_sr_bounded: expanded ptrs size mismatch");
    }
    if (snap.sr_result_cache.gather_rows.size() < out_n_gather) {
        return fail(xkv_read_status::codec_error, "run_sr_bounded: result gather cap short");
    }
    for (size_t i = 0; i < out_n_gather; ++i) {
        snap.sr_result_cache.gather_rows[i] = snap.sr_expanded_gather[i];
    }
    snap.sr_result_cache.gather_rows.resize(out_n_gather); // shrink only
    size_t wpos = 0;
    // Merge expanded slices per parent with per-(query,row) groups (no heap):
    // each parent updates a row exactly once, with its own Q group (first
    // expanded slice wins deterministically); no dedup overwrite, no
    // double-count. Groups ride parallel to indices in csr_group_indices.
    // Expanded entries of one parent are contiguous (expansion order).
    size_t exp_base = 0;
    for (uint32_t q = 0; q < snap.n_queries; ++q) {
        snap.sr_result_cache.csr_ptrs[q] = (uint32_t) wpos;
        if (q >= queries.size()) {
            return fail(xkv_read_status::invalid_argument, "run_sr_bounded: parent query missing");
        }
        uint32_t ng = (uint32_t) queries[q].q_groups.size();
        if (ng == 0) ng = 1;
        for (uint32_t g = 0; g < ng; ++g) {
            size_t ex = 0;
            if (!safe_add(exp_base, (size_t) g, ex) || ex >= snap.sr_selector_queries.size() ||
                ex >= snap.sr_selector_parents.size() || snap.sr_selector_parents[ex] != q) {
                return fail(xkv_read_status::codec_error, "run_sr_bounded: expansion order mismatch");
            }
            if (snap.sr_expanded_ptrs[ex] > snap.sr_expanded_ptrs[ex + 1] ||
                snap.sr_expanded_ptrs[ex + 1] > out_n_csr) {
                return fail(xkv_read_status::codec_error, "run_sr_bounded: expanded CSR range invalid");
            }
            for (uint32_t c = snap.sr_expanded_ptrs[ex]; c < snap.sr_expanded_ptrs[ex + 1]; ++c) {
                uint32_t gi = snap.sr_expanded_indices[c];
                if (gi >= out_n_gather) {
                    return fail(xkv_read_status::codec_error, "run_sr_bounded: expanded index out of range");
                }
                // First-wins dedup over the accepted parent range (allocation-free).
                bool seen = false;
                for (size_t k = snap.sr_result_cache.csr_ptrs[q]; k < wpos; ++k) {
                    if (snap.sr_result_cache.csr_indices[k] == gi) { seen = true; break; }
                }
                if (seen) continue;
                if (wpos >= snap.sr_result_cache.csr_indices.size() ||
                    wpos >= snap.sr_result_cache.csr_group_indices.size()) {
                    return fail(xkv_read_status::workspace_exceeded, "run_sr_bounded: result CSR cap short");
                }
                snap.sr_result_cache.csr_indices[wpos] = gi;
                snap.sr_result_cache.csr_group_indices[wpos] = g;
                ++wpos;
            }
        }
        size_t adv = 0;
        if (!safe_add(exp_base, (size_t) ng, adv)) {
            return fail(xkv_read_status::invalid_argument, "run_sr_bounded: expansion overflow");
        }
        exp_base = adv;
    }
    if (exp_base != snap.sr_selector_queries.size()) {
        return fail(xkv_read_status::codec_error, "run_sr_bounded: expansion count mismatch");
    }
    snap.sr_result_cache.csr_ptrs[snap.n_queries] = (uint32_t) wpos;
    snap.sr_result_cache.csr_indices.resize(wpos); // shrink only
    snap.sr_result_cache.csr_group_indices.resize(wpos); // shrink only
    snap.sr_result_cache.total_refined_rows = out_refined;
    snap.sr_result_cache.total_refine_cap_hits = out_hits;
    return xkv_read_status::success;
}

bool xkv_validate_native_tensors(
    const struct ggml_tensor * a_k, const struct ggml_tensor * b_k,
    const struct ggml_tensor * a_v, const struct ggml_tensor * b_v,
    const struct ggml_tensor * refs, const struct ggml_tensor * positions,
    const struct ggml_tensor * group_meta, const struct ggml_tensor * rope_tables,
    const struct ggml_tensor * layer_meta,
    const struct ggml_tensor * dst,
    const ggml_xkv_reconstruct_params * params, std::string * err) {
    if (!params) {
        if (err) *err = "xkv_validate_native_tensors: null params";
        return false;
    }
    if (params->version != GGML_XKV_VERSION) {
        if (err) *err = "xkv_validate_native_tensors: params version mismatch";
        return false;
    }
    if (params->n_groups != 1) {
        if (err) *err = "xkv_validate_native_tensors: one op per exact segment-group (n_groups must be 1)";
        return false;
    }
    char buf[256] = {};
    if (!ggml_xkv_reconstruct_supports(a_k, b_k, a_v, b_v, refs, positions, group_meta,
            layer_meta, rope_tables, dst, params, buf, sizeof(buf))) {
        if (err) *err = std::string("xkv_validate_native_tensors: ") + buf;
        return false;
    }
    return true;
}

bool xkv_validate_build_caps(const xkv_graph_build_caps & caps, std::string * err) {
    if (!xkv_graph_covers_bounded_hot(caps)) {
        if (err) *err = "xkv_validate_build_caps: XKV mode does not cover bounded-hot attention";
        return false;
    }
    if (caps.use_alibi) {
        if (err) *err = "xkv_validate_build_caps: ALiBi bias is unsupported";
        return false;
    }
    if (caps.rope_type == GGML_ROPE_TYPE_VISION) {
        if (err) *err = "xkv_validate_build_caps: VISION/multimodal RoPE coordinates are unsupported";
        return false;
    }
    if (caps.v_transposed) {
        if (err) *err = "xkv_validate_build_caps: transposed V layout is unsupported";
        return false;
    }
    if (caps.has_kq_bias) {
        if (err) *err = "xkv_validate_build_caps: kq bias is unsupported";
        return false;
    }
    for (int side = 0; side < 2; ++side) {
        ggml_type t = (side == 0) ? caps.k_type : caps.v_type;
        if (t == GGML_TYPE_F32) continue; // standard float hot storage
        if (t < 0 || t >= GGML_TYPE_COUNT) {
            if (err) *err = "xkv_validate_build_caps: codec out of range";
            return false;
        }
        if (is_turbo_type(t)) {
            if (ggml_turbo_layout_fingerprint(t) == 0) {
                if (err) *err = std::string("xkv_validate_build_caps: unsupported turbo layout ") + ggml_type_name(t);
                return false;
            }
            continue;
        }
        const ggml_type_traits * traits = ggml_get_type_traits(t);
        if (!traits || !traits->to_float) {
            if (err) *err = std::string("xkv_validate_build_caps: unsupported codec ") + ggml_type_name(t);
            return false;
        }
    }
    return true;
}

bool xkv_select_exec_branch(const xkv_graph_build_caps & caps, xkv_exec_branch & out_branch, std::string * err) {
    if (!xkv_graph_covers_bounded_hot(caps)) {
        if (err) *err = "xkv_select_exec_branch: bounded-hot XKV not active";
        return false;
    }
    if (caps.storage_needs_native) {
        if (!caps.native_backend_registered) {
            if (err) *err = "xkv_select_exec_branch: device-owned XKV profile requires a registered native backend (absent)";
            return false;
        }
        out_branch = xkv_exec_branch::native_reconstruct;
        return true;
    }
    out_branch = xkv_exec_branch::cpu_reference;
    return true;
}

xkv_post_action xkv_graph_postcompute_action(xkv_read_status status) {
    switch (status) {
        case xkv_read_status::success: return xkv_post_action::ok;
        case xkv_read_status::retry_stale_stamp: return xkv_post_action::retry;
        default: return xkv_post_action::hard_error;
    }
}

bool xkv_graph_postcompute_ok(const xkv_graph_snapshot & snap, std::string * err) {
    if (!snap.is_computed) {
        if (err) *err = "xkv postcompute: op never computed";
        return false;
    }
    if (snap.last_status != xkv_read_status::success) {
        if (err) *err = "xkv postcompute: " + snap.last_error;
        return false;
    }
    return true;
}

// Caller-owned metadata footprint (capacities) for workspace reserve
// accounting. Lease-backed spans report layout.total_bytes (arena-counted);
// shared segment pins are not double-counted.
size_t xkv_snapshot_metadata_bytes(const xkv_graph_snapshot & snap) {
    size_t n = sizeof(xkv_graph_snapshot);
    for (const auto & hd : snap.hot_data) {
        n += hd.k_data.capacity() * sizeof(float) + hd.v_data.capacity() * sizeof(float);
        n += hd.query_visibility.size() / 8 + 1;
    }
    for (const auto & v : snap.segment_views) {
        n += sizeof(v);
        n += v.selected_rows.capacity() * sizeof(uint32_t);
        n += v.storage_positions.capacity() * sizeof(int64_t);
        n += v.row_generations.capacity() * sizeof(uint64_t);
        n += v.group_indices.capacity() * sizeof(uint32_t);
        n += v.membership_mask.size() / 8 + 1;
    }
    n += snap.sr_selection.gather_rows.capacity() * sizeof(segment_row_ref);
    n += snap.sr_selection.csr_ptrs.capacity() * sizeof(uint32_t);
    n += snap.sr_selection.csr_indices.capacity() * sizeof(uint32_t);
    n += snap.sr_selection.csr_group_indices.capacity() * sizeof(uint32_t);
    for (const auto & f : snap.sr_legal_frags) {
        n += sizeof(f);
        n += f.row_indices.capacity() * sizeof(uint32_t);
        n += f.storage_positions.capacity() * sizeof(int64_t);
        n += f.payload_ids.capacity() * sizeof(uint64_t);
        n += f.generations.capacity() * sizeof(uint64_t);
        n += f.landmark_matrix.bytes.capacity();
    }
    for (const auto & q : snap.query_cache) {
        n += sizeof(q);
        n += q.q_vec.capacity() * sizeof(float);
        n += q.sink_logits.capacity() * sizeof(float);
        for (const auto & g : q.q_groups) n += g.capacity() * sizeof(float);
    }
    for (const auto & r : snap.hot_cache) {
        n += sizeof(r) + r.query_visibility.size() / 8 + 1 + r.query_group_indices.capacity() * sizeof(uint32_t);
    }
    n += snap.gather_cells_cache.capacity() * sizeof(int64_t);
    n += snap.gather_pos_cache.capacity() * sizeof(int64_t);
    n += snap.sr_frag_views.capacity() * sizeof(landmark_fragment_view);
    for (const auto & s : snap.sr_selector_queries) {
        n += s.q_vec.capacity() * sizeof(float) + s.q_head_indices.capacity() * sizeof(uint32_t);
    }
    n += snap.sr_selector_parents.capacity() * sizeof(uint32_t);
    n += snap.sr_seg_table.capacity() * sizeof(void *);
    n += snap.sr_expanded_gather.capacity() * sizeof(segment_row_ref);
    n += snap.sr_expanded_ptrs.capacity() * sizeof(uint32_t);
    n += snap.sr_expanded_indices.capacity() * sizeof(uint32_t);
    n += snap.sr_result_cache.gather_rows.capacity() * sizeof(segment_row_ref);
    n += snap.sr_result_cache.csr_ptrs.capacity() * sizeof(uint32_t);
    n += snap.sr_result_cache.csr_indices.capacity() * sizeof(uint32_t);
    n += snap.sr_result_cache.csr_group_indices.capacity() * sizeof(uint32_t);
    n += snap.workspace_layout.total_bytes;
    n += snap.hot_k_inv_rot.capacity() * sizeof(float) + snap.hot_v_inv_rot.capacity() * sizeof(float);
    n += snap.hot_k_channel_mul.capacity() * sizeof(float) + snap.hot_v_channel_mul.capacity() * sizeof(float);
    for (const auto & s : snap.query_sink_logits) n += s.capacity() * sizeof(float);
    n += snap.query_causal_limits.capacity() * sizeof(int64_t);
    n += snap.query_ddvr_group_counts.capacity() * sizeof(uint32_t);
    n += snap.query_ddvr_group_offsets.capacity() * sizeof(size_t);
    n += snap.last_error.capacity();
    return n;
}

// ==================== Native tiled attention chain ====================
//
// One KV head: reconstruct each selected (segment, group) in
// workspace-bounded tiles, then fold tiles into one global FP32 online
// softmax via carry chaining (ggml_xkv_attention). Hot rows and the sink
// participate exactly once, on the first tile. Output matches the reference
// builder shape [Dv_logic*gqa, NQ]. Host-side enumeration mirrors the CPU
// reference reader exactly (visibility, causal, CSR membership, group
// fallback, skip rules); any divergence fails closed instead of silently
// differing. Graph build may allocate host scratch freely; per-decode
// uploads never happen (code streams are borrowed arenas).

namespace {

constexpr int64_t kXkvNativeTileRows = 1024; // == GGML_XKV_ATTN_DENSE_COLD_MAX

struct xkv_native_hot_ent {
    int64_t cell = -1;
    uint32_t slot = 0;
};

struct xkv_native_cold_sel {
    size_t view_idx = 0;
    uint32_t sel_idx = 0; // index into view.selected_rows
    uint32_t union_pos = 0;
    uint32_t slot = 0; // global q slot for this (query, group)
};

// Q-slot map mirroring build_query_inputs packing: per query, the global q
// slots of its groups in order. Fails closed on overflow, misalignment, or
// out-of-range slots.
bool build_native_qslots(const xkv_graph_snapshot & snap, int64_t q_slots_total,
                          std::vector<std::vector<uint32_t>> & out_qslots, std::string * err) {
    out_qslots.clear();
    const uint32_t nq = snap.n_queries;
    if (nq == 0) {
        if (err) *err = "native qslots: no queries";
        return false;
    }
    if (q_slots_total <= 0 || q_slots_total > (int64_t) INT32_MAX) {
        if (err) *err = "native qslots: q slots out of range";
        return false;
    }
    size_t qhe = 0;
    if (!safe_mul((size_t) snap.n_q_heads, (size_t) snap.head_dim_k, qhe) || qhe == 0) {
        if (err) *err = "native qslots: q head elements overflow/zero";
        return false;
    }
    const bool has_offsets = !snap.query_ddvr_group_offsets.empty();
    const bool has_counts = !snap.query_ddvr_group_counts.empty();
    if (has_offsets && snap.query_ddvr_group_offsets.size() != nq) {
        if (err) *err = "native qslots: group offsets size mismatch";
        return false;
    }
    if (has_counts && snap.query_ddvr_group_counts.size() != nq) {
        if (err) *err = "native qslots: group counts size mismatch";
        return false;
    }
    out_qslots.resize(nq);
    size_t cur = 0;
    for (uint32_t q = 0; q < nq; ++q) {
        uint32_t gc = has_counts ? snap.query_ddvr_group_counts[q] : snap.n_ddvr_groups;
        uint32_t nj = gc > 0 ? gc : 1;
        size_t base = 0;
        if (has_offsets) {
            size_t off = snap.query_ddvr_group_offsets[q];
            if (off % qhe != 0) {
                if (err) *err = "native qslots: group offset not slot-aligned";
                return false;
            }
            base = off / qhe;
        } else if (has_counts) {
            base = cur;
            if (!safe_add(cur, (size_t) nj, cur)) {
                if (err) *err = "native qslots: slot cursor overflow";
                return false;
            }
        } else {
            size_t stride = snap.n_ddvr_groups > 0 ? snap.n_ddvr_groups : 1;
            if (!safe_mul((size_t) q, stride, base)) {
                if (err) *err = "native qslots: slot offset overflow";
                return false;
            }
            nj = (uint32_t) stride;
        }
        for (uint32_t j = 0; j < nj; ++j) {
            size_t s = 0;
            if (!safe_add(base, (size_t) j, s) || s >= (size_t) q_slots_total) {
                if (err) *err = "native qslots: slot out of range";
                return false;
            }
            out_qslots[q].push_back((uint32_t) s);
        }
    }
    return true;
}

// Exact arena match for one (segment, group); fail closed on miss.
bool check_i32_range(int64_t v, const char * what, std::string * err) {
    if (v < 0 || v > (int64_t) INT32_MAX) {
        if (err) *err = std::string("native range: ") + what + " out of int32 range";
        return false;
    }
    return true;
}

// Host-filled graph tensor helper: creates a set_input tensor and records
// exact bytes for set_input application (refuse truncation).
ggml_tensor * new_native_filled(ggml_context * ctx, ggml_type type, int64_t ne0, int64_t ne1,
        const void * data, size_t nbytes, std::vector<xkv_native_fill_item> & fills, std::string * err) {
    if (ne0 <= 0 || ne1 <= 0) {
        if (err) *err = "native fill: non-positive dims";
        return nullptr;
    }
    ggml_tensor * t = ggml_new_tensor_2d(ctx, type, ne0, ne1);
    if (!t) {
        if (err) *err = "native fill: tensor creation failed";
        return nullptr;
    }
    if (nbytes != ggml_nbytes(t)) {
        if (err) *err = "native fill: byte size mismatch";
        return nullptr;
    }
    ggml_set_input(t);
    xkv_native_fill_item item;
    item.tensor = t;
    item.bytes.assign(static_cast<const uint8_t *>(data), static_cast<const uint8_t *>(data) + nbytes);
    fills.push_back(std::move(item));
    return t;
}

ggml_tensor * new_native_filled_1d(ggml_context * ctx, ggml_type type, int64_t ne0,
        const void * data, size_t nbytes, std::vector<xkv_native_fill_item> & fills, std::string * err) {
    if (ne0 <= 0) {
        if (err) *err = "native fill: non-positive dims";
        return nullptr;
    }
    ggml_tensor * t = ggml_new_tensor_1d(ctx, type, ne0);
    if (!t) {
        if (err) *err = "native fill: tensor creation failed";
        return nullptr;
    }
    if (nbytes != ggml_nbytes(t)) {
        if (err) *err = "native fill: byte size mismatch";
        return nullptr;
    }
    ggml_set_input(t);
    xkv_native_fill_item item;
    item.tensor = t;
    item.bytes.assign(static_cast<const uint8_t *>(data), static_cast<const uint8_t *>(data) + nbytes);
    fills.push_back(std::move(item));
    return t;
}

enum class xkv_native_score_set_kind : uint8_t {
    sealed_landmarks,
    rebuilt_partial,
};

struct xkv_native_score_set_plan {
    xkv_native_score_set_kind kind = xkv_native_score_set_kind::sealed_landmarks;
    uint32_t arena_index = 0;
    uint32_t fragment_begin = 0;
    uint32_t fragment_count = 0;
    uint32_t base_landmark_row = 0;
    uint32_t rebuild_request_index = UINT32_MAX;
};

struct xkv_native_sr_plan {
    std::vector<xkv_sr_device_arena> arenas;
    std::vector<const xkv_factor_group_payload *> groups;
    std::vector<uint32_t> owning_layers;
    std::vector<uint32_t> kv_heads;
    std::vector<xkv_sr_device_frag> fragments;
    std::vector<xkv_native_score_set_plan> score_sets;
};

bool build_native_sr_plan(const xkv_graph_snapshot & snap, xkv_native_sr_plan & out, std::string * err) {
    auto fail = [&](const std::string & message) {
        if (err) *err = "native SR plan: " + message;
        return false;
    };
    out = {};
    if (snap.sr_legal_frags.empty()) return fail("no legal fragments");
    if (snap.native_group_arenas.empty()) return fail("no native arenas");

    out.arenas.reserve(snap.native_group_arenas.size());
    std::vector<const xkv_factor_group_payload *> groups(snap.native_group_arenas.size(), nullptr);
    std::vector<uint32_t> owning_layers(snap.native_group_arenas.size(), UINT32_MAX);
    std::vector<uint32_t> kv_heads(snap.native_group_arenas.size(), UINT32_MAX);
    for (size_t a = 0; a < snap.native_group_arenas.size(); ++a) {
        const auto & native = snap.native_group_arenas[a];
        if (!native.a_k || !native.landmarks || native.a_k->ne[1] <= 0) {
            return fail("incomplete arena tensors");
        }
        const xkv_factor_group_payload * matched = nullptr;
        for (const auto & view : snap.segment_views) {
            const xkv_segment * seg = view.get_segment();
            if (!seg || seg->segment_id != native.segment_id) continue;
            const uint64_t version = view.segment_version_id != 0 ? view.segment_version_id : seg->segment_version;
            if (version != native.segment_version) continue;
            std::string group_err;
            const xkv_factor_group_payload * group = view.resolve_group(&group_err);
            if (!group) return fail("group resolve failed: " + group_err);
            if (group->group_index != native.group_index) continue;
            if (matched && matched != group) return fail("ambiguous arena group");
            if (owning_layers[a] != UINT32_MAX &&
                (owning_layers[a] != view.owning_layer || kv_heads[a] != view.kv_head)) {
                return fail("arena spans multiple layer/head views in one attention snapshot");
            }
            matched = group;
            owning_layers[a] = view.owning_layer;
            kv_heads[a] = view.kv_head;
        }
        if (!matched) return fail("arena group not represented by a pinned view");
        if (matched->landmark_chunks.empty() ||
            matched->landmark_chunks.size() != (size_t) native.landmarks->ne[1] ||
            matched->landmark.desc.logical_shape.rows != matched->landmark_chunks.size()) {
            return fail("sealed landmark chunk table mismatch");
        }
        xkv_sr_device_arena arena;
        arena.segment_id = native.segment_id;
        arena.segment_version = native.segment_version;
        arena.group_index = native.group_index;
        arena.landmark_codec_fp = matched->landmark.desc.fingerprint();
        arena.row_count = (uint64_t) native.a_k->ne[1];
        if (arena.landmark_codec_fp == 0) return fail("zero arena landmark codec fingerprint");
        out.arenas.push_back(arena);
        groups[a] = matched;
    }
    out.groups = groups;
    out.owning_layers = std::move(owning_layers);
    out.kv_heads = std::move(kv_heads);

    std::vector<uint32_t> legal_arena(snap.sr_legal_frags.size(), UINT32_MAX);
    for (size_t f = 0; f < snap.sr_legal_frags.size(); ++f) {
        const legal_fragment & frag = snap.sr_legal_frags[f];
        for (const auto & view : snap.segment_views) {
            const xkv_segment * seg = view.get_segment();
            if (!seg || seg->segment_id != frag.key.segment_id ||
                view.owning_layer != frag.key.owning_layer || view.kv_head != frag.key.kv_head) {
                continue;
            }
            const uint64_t version = view.segment_version_id != 0 ? view.segment_version_id : seg->segment_version;
            if (version != frag.key.segment_version) continue;
            std::string group_err;
            const xkv_factor_group_payload * group = view.resolve_group(&group_err);
            if (!group) return fail("fragment group resolve failed: " + group_err);
            for (size_t a = 0; a < out.arenas.size(); ++a) {
                if (out.arenas[a].segment_id == frag.key.segment_id &&
                    out.arenas[a].segment_version == frag.key.segment_version &&
                    out.arenas[a].group_index == group->group_index) {
                    if (legal_arena[f] != UINT32_MAX && legal_arena[f] != a) {
                        return fail("fragment maps to multiple arenas");
                    }
                    legal_arena[f] = (uint32_t) a;
                }
            }
        }
        if (legal_arena[f] == UINT32_MAX) return fail("legal fragment has no exact native arena");
    }

    auto make_fragment = [&](size_t f, uint32_t arena_index, xkv_sr_device_frag & dst) -> bool {
        const legal_fragment & frag = snap.sr_legal_frags[f];
        const xkv_sr_device_arena & arena = out.arenas[arena_index];
        dst.segment_id = frag.key.segment_id;
        dst.segment_version = frag.key.segment_version;
        dst.group_index = arena.group_index;
        dst.ddvr_group = frag.key.ddvr_group;
        dst.owning_layer = frag.key.owning_layer;
        dst.kv_head = frag.key.kv_head;
        dst.phase_tx_fingerprint = frag.key.phase_tx_fingerprint;
        dst.landmark_codec_fp = frag.key.landmark_codec_fp;
        dst.storage_generation = frag.key.storage_generation;
        dst.row_begin = frag.row_begin;
        dst.row_count = frag.row_count;
        dst.sparse = frag.row_indices.size() != frag.row_count;
        for (uint32_t i = 0; !dst.sparse && i < frag.row_count; ++i) {
            dst.sparse = frag.row_indices[i] != frag.row_begin + i;
        }
        if (dst.sparse) dst.sparse_rows.assign(frag.row_indices.begin(), frag.row_indices.end());
        dst.storage_positions = frag.storage_positions;
        dst.eligible = 1;
        if (dst.storage_positions.empty()) return fail("fragment positions missing");
        const int64_t max_pos = *std::max_element(dst.storage_positions.begin(), dst.storage_positions.end());
        if (max_pos < INT32_MIN || max_pos > INT32_MAX) return fail("fragment position out of int32 range");
        dst.frag_pos = (int32_t) max_pos;
        dst.query_visibility = frag.query_visibility;
        return true;
    };

    std::vector<uint8_t> represented(snap.sr_legal_frags.size(), 0);
    for (size_t a = 0; a < out.arenas.size(); ++a) {
        const auto & chunks = groups[a]->landmark_chunks;
        std::map<uint32_t, std::vector<std::vector<size_t>>> intact_by_group;
        for (size_t f = 0; f < snap.sr_legal_frags.size(); ++f) {
            if (legal_arena[f] != a) continue;
            const legal_fragment & frag = snap.sr_legal_frags[f];
            if (frag.is_derived_partial || frag.requires_native_rebuild) continue;
            if (frag.base_landmark_row >= chunks.size()) return fail("base landmark row out of range");
            const xkv_landmark_chunk & chunk = chunks[frag.base_landmark_row];
            if (frag.row_begin != chunk.row_begin || frag.row_count != chunk.row_count ||
                frag.source_fingerprint == 0 || frag.source_fingerprint != chunk.source_fingerprint) {
                return fail("intact fragment provenance/interval mismatch");
            }
            auto & per_chunk = intact_by_group[frag.key.ddvr_group];
            if (per_chunk.empty()) per_chunk.resize(chunks.size());
            per_chunk[frag.base_landmark_row].push_back(f);
        }
        for (const auto & group_entry : intact_by_group) {
            const auto & intact = group_entry.second;
            size_t row = 0;
            while (row < intact.size()) {
                while (row < intact.size() && intact[row].empty()) ++row;
                if (row == intact.size()) break;
                const size_t row_begin = row;
                const uint32_t frag_begin = (uint32_t) out.fragments.size();
                while (row < intact.size() && !intact[row].empty()) {
                    xkv_sr_device_frag merged;
                    if (!make_fragment(intact[row][0], (uint32_t) a, merged)) return false;
                    for (size_t j = 0; j < intact[row].size(); ++j) {
                        const size_t f = intact[row][j];
                        xkv_sr_device_frag candidate;
                        if (!make_fragment(f, (uint32_t) a, candidate)) return false;
                        if (candidate.segment_id != merged.segment_id ||
                            candidate.segment_version != merged.segment_version ||
                            candidate.group_index != merged.group_index ||
                            candidate.ddvr_group != merged.ddvr_group ||
                            candidate.owning_layer != merged.owning_layer ||
                            candidate.kv_head != merged.kv_head ||
                            candidate.phase_tx_fingerprint != merged.phase_tx_fingerprint ||
                            candidate.landmark_codec_fp != merged.landmark_codec_fp ||
                            candidate.storage_generation != merged.storage_generation ||
                            candidate.row_begin != merged.row_begin || candidate.row_count != merged.row_count ||
                            candidate.sparse != merged.sparse || candidate.sparse_rows != merged.sparse_rows ||
                            candidate.storage_positions != merged.storage_positions || candidate.frag_pos != merged.frag_pos) {
                            return fail("per-parent intact fragments disagree on physical identity");
                        }
                        if (merged.query_visibility.empty() || candidate.query_visibility.empty()) {
                            merged.query_visibility.clear();
                        } else {
                            if (merged.query_visibility.size() != candidate.query_visibility.size()) {
                                return fail("per-parent fragment visibility width mismatch");
                            }
                            for (size_t q = 0; q < merged.query_visibility.size(); ++q) {
                                merged.query_visibility[q] = merged.query_visibility[q] || candidate.query_visibility[q];
                            }
                        }
                        represented[f] = 1;
                    }
                    out.fragments.push_back(std::move(merged));
                    ++row;
                }
                xkv_native_score_set_plan set;
                set.kind = xkv_native_score_set_kind::sealed_landmarks;
                set.arena_index = (uint32_t) a;
                set.fragment_begin = frag_begin;
                set.fragment_count = (uint32_t) (row - row_begin);
                set.base_landmark_row = (uint32_t) row_begin;
                out.score_sets.push_back(set);
            }
        }
    }

    for (size_t r = 0; r < snap.native_landmark_rebuild_requests.size(); ++r) {
        const auto & req = snap.native_landmark_rebuild_requests[r];
        if (req.fragment_index >= snap.sr_legal_frags.size()) return fail("rebuild fragment index out of range");
        const size_t f = req.fragment_index;
        const legal_fragment & frag = snap.sr_legal_frags[f];
        if (!frag.is_derived_partial || !frag.requires_native_rebuild || represented[f]) {
            return fail("invalid or duplicate rebuild request");
        }
        const uint32_t arena_index = legal_arena[f];
        const auto & arena = out.arenas[arena_index];
        if (req.segment_id != arena.segment_id || req.segment_version != arena.segment_version ||
            req.group_index != arena.group_index || req.owning_layer != frag.key.owning_layer ||
            req.kv_head != frag.key.kv_head || req.row_indices != frag.row_indices ||
            req.storage_positions != frag.storage_positions || req.phase_tx_fingerprint != frag.key.phase_tx_fingerprint ||
            req.landmark_type != groups[arena_index]->landmark.desc.type) {
            return fail("rebuild request identity mismatch");
        }
        xkv_sr_device_frag fragment;
        if (!make_fragment(f, arena_index, fragment)) return false;
        const uint32_t fragment_begin = (uint32_t) out.fragments.size();
        out.fragments.push_back(std::move(fragment));
        represented[f] = 1;
        xkv_native_score_set_plan set;
        set.kind = xkv_native_score_set_kind::rebuilt_partial;
        set.arena_index = arena_index;
        set.fragment_begin = fragment_begin;
        set.fragment_count = 1;
        set.rebuild_request_index = (uint32_t) r;
        out.score_sets.push_back(set);
    }

    if (std::find(represented.begin(), represented.end(), 0) != represented.end()) {
        return fail("legal fragment is not represented by a sealed or rebuilt landmark");
    }
    if (out.fragments.empty() || out.score_sets.empty()) return fail("empty score plan");
    return true;
}

ggml_tensor * build_native_sr_selection(
        ggml_context * ctx,
        const xkv_graph_snapshot & snap,
        const xkv_native_sr_plan & plan,
        const xkv_sr_device_inputs & inputs,
        ggml_tensor * selector_q,
        ggml_tensor * rope_tables,
        uint32_t top_k,
        float scale,
        std::vector<xkv_native_status_item> & out_status_tensors,
        std::vector<xkv_native_fill_item> & out_fills,
        std::string * err) {
    auto fail = [&](const std::string & message) -> ggml_tensor * {
        if (err) *err = "native SR selection: " + message;
        return nullptr;
    };
    const uint32_t nq = inputs.n_queries;
    if (!ctx || !selector_q || !rope_tables || nq == 0 || plan.fragments.empty() ||
        plan.score_sets.empty() || plan.groups.size() != plan.arenas.size()) {
        return fail("invalid plan/tensors");
    }
    if (top_k == 0 || top_k > GGML_XKV_LANDMARK_MAX_TOP_K) return fail("top-k out of range");

    ggml_tensor * query_meta = new_native_filled(ctx, GGML_TYPE_I32, 6, nq,
        inputs.query_meta.data(), inputs.query_meta.size() * sizeof(int32_t), out_fills, err);
    if (!query_meta) return nullptr;

    std::vector<ggml_xkv_landmark_params> score_params(plan.score_sets.size());
    size_t max_score_scratch = 0;
    for (size_t s = 0; s < plan.score_sets.size(); ++s) {
        const auto & set = plan.score_sets[s];
        if (set.fragment_count == 0 || set.fragment_begin > inputs.n_frags ||
            set.fragment_count > inputs.n_frags - set.fragment_begin) {
            return fail("score-set fragment range invalid");
        }
        const auto & arena = snap.native_group_arenas[set.arena_index];
        if (!arena.landmarks) return fail("score-set landmark tensor missing");
        ggml_xkv_landmark_params p = {};
        p.version = GGML_XKV_LANDMARK_VERSION;
        p.n_queries = nq;
        p.n_frags = set.fragment_count;
        p.head_dim = snap.head_dim_k;
        p.padded_dim = snap.sr_feature_dim != 0 ? snap.sr_feature_dim : snap.head_dim_k;
        p.rotary_dim = snap.native_params.rotary_dim;
        p.rope_mode = snap.native_params.rope_mode;
        p.landmark_type = plan.groups[set.arena_index]->landmark.desc.type;
        p.top_k = std::min(top_k, set.fragment_count);
        p.max_top_k = top_k;
        p.refine_cap = 0;
        p.frag_size = inputs.max_frag_rows;
        p.scale = scale;
        p.n_q_heads = snap.n_q_heads;
        p.n_rows_total = inputs.n_rows_total;
        p._reserved = GGML_XKV_LANDMARK_FLAG_PERQ_LEGAL;
        size_t scratch_bytes = 0;
        char msg[256] = {};
        if (!ggml_xkv_landmark_workspace_bytes(&p, &scratch_bytes, msg, sizeof(msg))) {
            return fail(std::string("score workspace sizing failed: ") + msg);
        }
        max_score_scratch = std::max(max_score_scratch, scratch_bytes);
        score_params[s] = p;
    }
    if (max_score_scratch == 0 || max_score_scratch > (size_t) INT64_MAX - 3) {
        return fail("score scratch size invalid");
    }
    const int64_t score_words = (int64_t) ((max_score_scratch + sizeof(int32_t) - 1) / sizeof(int32_t));
    ggml_tensor * score_scratch = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, score_words);
    if (!score_scratch) return fail("score scratch tensor allocation failed");

    std::vector<ggml_xkv_landmark_build_params> rebuild_params(plan.score_sets.size());
    size_t max_rebuild_scratch = 0;
    for (size_t s = 0; s < plan.score_sets.size(); ++s) {
        const auto & set = plan.score_sets[s];
        if (set.kind != xkv_native_score_set_kind::rebuilt_partial) continue;
        if (set.rebuild_request_index >= snap.native_landmark_rebuild_requests.size()) {
            return fail("rebuild request index invalid");
        }
        const auto & req = snap.native_landmark_rebuild_requests[set.rebuild_request_index];
        const auto & arena = snap.native_group_arenas[set.arena_index];
        const auto * group = plan.groups[set.arena_index];
        if (!arena.a_k || !arena.b_k || !group || req.row_indices.empty() ||
            req.row_indices.size() != req.storage_positions.size() ||
            req.row_indices.size() > UINT32_MAX || req.feature_dim != snap.head_dim_k ||
            req.feature_offset > group->total_dim_k || req.feature_dim > group->total_dim_k - req.feature_offset ||
            arena.a_k->ne[0] <= 0 || arena.a_k->ne[0] > UINT32_MAX) {
            return fail("rebuild geometry invalid");
        }
        for (uint32_t row : req.row_indices) {
            if ((int64_t) row >= arena.a_k->ne[1]) return fail("rebuild row outside A arena");
        }
        const int64_t block = ggml_blck_size(req.landmark_type);
        if (block <= 0 || req.feature_dim % (uint32_t) block != 0 ||
            req.feature_offset % (uint32_t) block != 0) {
            return fail("rebuild landmark feature slice is not codec-block aligned");
        }
        ggml_xkv_landmark_build_params p = {};
        p.version = GGML_XKV_LANDMARK_BUILD_VERSION;
        p.n_rows = (uint32_t) req.row_indices.size();
        p.n_chunks = 1;
        p.chunk_tokens = p.n_rows;
        p.total_dim = req.feature_dim;
        p.padded_dim = req.feature_dim;
        p.rank = group->rank_k;
        p.pad_rank = (uint32_t) arena.a_k->ne[0];
        p.n_layers = 1;
        p.max_feature_dim = req.feature_dim;
        p.landmark_type = (uint32_t) req.landmark_type;
        p.a_type = (uint32_t) arena.a_k->type;
        p.b_type = (uint32_t) arena.b_k->type;
        p.seed = (uint32_t) group->landmark.desc.seed;
        p.phase_lo = (uint32_t) req.phase_tx_fingerprint;
        p.phase_hi = (uint32_t) (req.phase_tx_fingerprint >> 32);
        size_t scratch_bytes = 0;
        char msg[256] = {};
        if (!ggml_xkv_landmark_build_scratch_bytes(&p, &scratch_bytes, msg, sizeof(msg))) {
            return fail(std::string("rebuild workspace sizing failed: ") + msg);
        }
        max_rebuild_scratch = std::max(max_rebuild_scratch, scratch_bytes);
        rebuild_params[s] = p;
    }
    ggml_tensor * rebuild_scratch = nullptr;
    if (max_rebuild_scratch > 0) {
        if (max_rebuild_scratch > (size_t) INT64_MAX) return fail("rebuild scratch size overflow");
        rebuild_scratch = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, (int64_t) max_rebuild_scratch);
        if (!rebuild_scratch) return fail("rebuild scratch tensor allocation failed");
    }

    struct candidate_set {
        ggml_tensor * indices = nullptr;
        ggml_tensor * scores = nullptr;
        uint32_t base = 0;
    };
    std::vector<candidate_set> candidates;
    candidates.reserve(plan.score_sets.size());
    const uint32_t global_words_per_query = (inputs.n_frags + 31u) / 32u;
    for (size_t s = 0; s < plan.score_sets.size(); ++s) {
        const auto & set = plan.score_sets[s];
        const auto & arena = snap.native_group_arenas[set.arena_index];
        const auto * group = plan.groups[set.arena_index];
        const auto & p = score_params[s];
        ggml_tensor * landmarks = nullptr;
        if (set.kind == xkv_native_score_set_kind::sealed_landmarks) {
            if (arena.landmarks->type != group->landmark.desc.type ||
                set.base_landmark_row > (uint32_t) arena.landmarks->ne[1] ||
                set.fragment_count > (uint32_t) arena.landmarks->ne[1] - set.base_landmark_row) {
                return fail("sealed landmark score view out of range");
            }
            const uint32_t feature_dim = snap.sr_feature_dim != 0 ? snap.sr_feature_dim : snap.head_dim_k;
            const uint32_t feature_offset = snap.sr_feature_offset;
            const int64_t block = ggml_blck_size(arena.landmarks->type);
            const size_t type_size = ggml_type_size(arena.landmarks->type);
            if (block <= 0 || type_size == 0 || feature_dim != snap.head_dim_k ||
                feature_dim % (uint32_t) block != 0 || feature_offset % (uint32_t) block != 0 ||
                feature_offset > group->total_dim_k || feature_dim > group->total_dim_k - feature_offset) {
                return fail("sealed landmark feature slice is not codec-block aligned");
            }
            size_t row_offset = 0;
            size_t feature_offset_bytes = 0;
            size_t view_offset = 0;
            if (!safe_mul((size_t) set.base_landmark_row, (size_t) arena.landmarks->nb[1], row_offset) ||
                !safe_mul((size_t) (feature_offset / (uint32_t) block), type_size, feature_offset_bytes) ||
                !safe_add(row_offset, feature_offset_bytes, view_offset)) {
                return fail("sealed landmark view offset overflow");
            }
            landmarks = ggml_view_2d(ctx, arena.landmarks, feature_dim, set.fragment_count,
                arena.landmarks->nb[1], view_offset);
            if (!landmarks) return fail("sealed landmark view creation failed");
        } else {
            const auto & req = snap.native_landmark_rebuild_requests[set.rebuild_request_index];
            const auto & bp = rebuild_params[s];
            std::vector<int32_t> rows(req.row_indices.begin(), req.row_indices.end());
            std::vector<int64_t> positions(req.storage_positions.begin(), req.storage_positions.end());
            std::vector<int32_t> layer_meta = {
                (int32_t) req.feature_offset, (int32_t) req.feature_dim, (int32_t) req.feature_dim,
                (int32_t) snap.native_params.rotary_dim, (int32_t) snap.native_params.rope_mode, 0};
            ggml_tensor * rows_t = new_native_filled_1d(ctx, GGML_TYPE_I32, rows.size(),
                rows.data(), rows.size() * sizeof(int32_t), out_fills, err);
            ggml_tensor * positions_t = new_native_filled_1d(ctx, GGML_TYPE_I64, positions.size(),
                positions.data(), positions.size() * sizeof(int64_t), out_fills, err);
            ggml_tensor * layer_meta_t = new_native_filled(ctx, GGML_TYPE_I32, 6, 1,
                layer_meta.data(), layer_meta.size() * sizeof(int32_t), out_fills, err);
            ggml_tensor * error_bound = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
            ggml_tensor * source_fp = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1);
            ggml_tensor * build_status = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);
            if (!rows_t || !positions_t || !layer_meta_t || !error_bound || !source_fp || !build_status || !rebuild_scratch) {
                return fail("rebuild tensor allocation failed");
            }
            landmarks = ggml_xkv_landmark_build(ctx, arena.a_k, arena.b_k, rows_t, positions_t,
                layer_meta_t, rope_tables, rebuild_scratch, error_bound, source_fp, build_status, &bp);
            if (!landmarks) return fail("native landmark rebuild node failed");
            char msg[256] = {};
            if (!ggml_xkv_landmark_build_supports(arena.a_k, arena.b_k, rows_t, positions_t,
                    layer_meta_t, rope_tables, rebuild_scratch, error_bound, source_fp, build_status,
                    landmarks, &bp, msg, sizeof(msg))) {
                return fail(std::string("native landmark rebuild unsupported: ") + msg);
            }
            out_status_tensors.push_back({build_status, xkv_native_status_policy::code_only});
        }

        std::vector<int32_t> frag_meta((size_t) set.fragment_count * 8, 0);
        const uint32_t local_words_per_query = (set.fragment_count + 31u) / 32u;
        std::vector<int32_t> elig_bits((size_t) local_words_per_query * nq, 0);
        for (uint32_t lf = 0; lf < set.fragment_count; ++lf) {
            const uint32_t gf = set.fragment_begin + lf;
            for (uint32_t k = 0; k < 6; ++k) {
                frag_meta[(size_t) lf * 8 + k] = inputs.frag_meta[(size_t) gf * 6 + k];
            }
            frag_meta[(size_t) lf * 8 + 6] = inputs.frag_positions[gf];
            frag_meta[(size_t) lf * 8 + 7] = inputs.frag_gen[gf];
            for (uint32_t q = 0; q < nq; ++q) {
                const uint32_t global_word = gf / 32u;
                const uint32_t global_bit = gf % 32u;
                if ((inputs.elig_bits[(size_t) q * global_words_per_query + global_word] & (1u << global_bit)) != 0) {
                    elig_bits[(size_t) q * local_words_per_query + lf / 32u] |= (int32_t) (1u << (lf % 32u));
                }
            }
        }
        ggml_tensor * frag_meta_t = new_native_filled(ctx, GGML_TYPE_I32, 8, set.fragment_count,
            frag_meta.data(), frag_meta.size() * sizeof(int32_t), out_fills, err);
        ggml_tensor * elig_bits_t = new_native_filled_1d(ctx, GGML_TYPE_I32, elig_bits.size(),
            elig_bits.data(), elig_bits.size() * sizeof(int32_t), out_fills, err);
        ggml_tensor * csr_ptrs = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, nq + 1);
        ggml_tensor * scores = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, p.top_k, nq);
        ggml_tensor * status = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);
        if (!frag_meta_t || !elig_bits_t || !csr_ptrs || !scores || !status) {
            return fail("selector tensor allocation failed");
        }
        ggml_tensor * indices = ggml_xkv_landmark(ctx, selector_q, landmarks, elig_bits_t,
            frag_meta_t, query_meta, rope_tables, score_scratch, csr_ptrs, scores, status, &p);
        if (!indices) return fail("selector node failed");
        char msg[256] = {};
        if (!ggml_xkv_landmark_supports(selector_q, landmarks, elig_bits_t, frag_meta_t, query_meta,
                rope_tables, score_scratch, csr_ptrs, scores, status, indices, &p, msg, sizeof(msg))) {
            return fail(std::string("selector unsupported: ") + msg);
        }
        out_status_tensors.push_back({status, xkv_native_status_policy::code_only});

        if (p.top_k < top_k) {
            const uint32_t tail = top_k - p.top_k;
            std::vector<int32_t> sentinel_idx((size_t) tail * nq, -1);
            std::vector<float> sentinel_scores((size_t) tail * nq, -1e30f);
            ggml_tensor * tail_idx = new_native_filled(ctx, GGML_TYPE_I32, tail, nq,
                sentinel_idx.data(), sentinel_idx.size() * sizeof(int32_t), out_fills, err);
            ggml_tensor * tail_scores = new_native_filled(ctx, GGML_TYPE_F32, tail, nq,
                sentinel_scores.data(), sentinel_scores.size() * sizeof(float), out_fills, err);
            if (!tail_idx || !tail_scores) return nullptr;
            indices = ggml_concat(ctx, indices, tail_idx, 0);
            scores = ggml_concat(ctx, scores, tail_scores, 0);
            if (!indices || !scores) return fail("selector sentinel padding failed");
        }
        candidates.push_back({indices, scores, set.fragment_begin});
    }

    bool first_merge_level = true;
    while (!candidates.empty()) {
        std::vector<candidate_set> next;
        next.reserve((candidates.size() + GGML_XKV_LANDMARK_MAX_SEGMENTS - 1) /
            GGML_XKV_LANDMARK_MAX_SEGMENTS);
        for (size_t begin = 0; begin < candidates.size(); begin += GGML_XKV_LANDMARK_MAX_SEGMENTS) {
            const size_t end = std::min(candidates.size(), begin + GGML_XKV_LANDMARK_MAX_SEGMENTS);
            ggml_tensor * stacked_idx = nullptr;
            ggml_tensor * stacked_scores = nullptr;
            std::vector<int32_t> bases(end - begin, 0);
            for (size_t i = begin; i < end; ++i) {
                ggml_tensor * idx3 = ggml_reshape_3d(ctx, candidates[i].indices, top_k, nq, 1);
                ggml_tensor * score3 = ggml_reshape_3d(ctx, candidates[i].scores, top_k, nq, 1);
                if (!idx3 || !score3) return fail("merge reshape failed");
                stacked_idx = stacked_idx ? ggml_concat(ctx, stacked_idx, idx3, 2) : idx3;
                stacked_scores = stacked_scores ? ggml_concat(ctx, stacked_scores, score3, 2) : score3;
                bases[i - begin] = first_merge_level ? (int32_t) candidates[i].base : 0;
            }
            ggml_tensor * bases_t = new_native_filled_1d(ctx, GGML_TYPE_I32, bases.size(),
                bases.data(), bases.size() * sizeof(int32_t), out_fills, err);
            ggml_tensor * merged_scores = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, top_k, nq);
            ggml_tensor * merge_status = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);
            if (!bases_t || !merged_scores || !merge_status) return fail("merge tensor allocation failed");
            ggml_xkv_landmark_merge_params mp = {};
            mp.version = GGML_XKV_LANDMARK_VERSION;
            mp.n_queries = nq;
            mp.n_sets = (uint32_t) (end - begin);
            mp.top_k = top_k;
            mp.set_cap = top_k;
            mp.has_set_base = first_merge_level ? 1 : 0;
            ggml_tensor * merged_idx = ggml_xkv_landmark_merge(ctx, stacked_idx, stacked_scores,
                bases_t, merged_scores, merge_status, &mp);
            if (!merged_idx) return fail("merge node failed");
            char msg[256] = {};
            if (!ggml_xkv_landmark_merge_supports(stacked_idx, stacked_scores, bases_t,
                    merged_scores, merge_status, merged_idx, &mp, msg, sizeof(msg))) {
                return fail(std::string("merge unsupported: ") + msg);
            }
            out_status_tensors.push_back({merge_status, xkv_native_status_policy::code_only});
            next.push_back({merged_idx, merged_scores, 0});
        }
        if (next.size() == 1) return next[0].indices;
        candidates = std::move(next);
        first_merge_level = false;
    }
    return fail("merge produced no candidate list");
}

// Resolve a CSR reference to (view, selected index), mirroring the reader's
// matching rule exactly. Fails closed on dangling refs (builder bug).
bool resolve_native_csr_ref(const xkv_graph_snapshot & snap, const segment_row_ref & ref,
                             size_t & out_view, size_t & out_idx, std::string * err) {
    for (size_t v = 0; v < snap.segment_views.size(); ++v) {
        const auto & vw = snap.segment_views[v];
        const xkv_segment * seg = vw.get_segment();
        if (!seg || seg->segment_id != ref.segment_id) continue;
        uint64_t eff_ver = vw.segment_version_id != 0 ? vw.segment_version_id : seg->segment_version;
        if (ref.segment_version != eff_ver) continue;
        for (size_t i = 0; i < vw.selected_rows.size(); ++i) {
            if (vw.selected_rows[i] == ref.row && vw.get_row_generation(i) == ref.storage_generation) {
                out_view = v;
                out_idx = i;
                return true;
            }
        }
    }
    if (err) *err = "native resolve: CSR ref unresolvable (builder bug)";
    return false;
}

} // namespace

// Exact arena match for one (segment, version, group); fail closed on
// miss/ambiguity (never host fallback). Shared by the chain builder and the
// llm dispatch precheck so both resolve identically.
const xkv_graph_snapshot::xkv_native_group_arenas * xkv_find_native_arenas(
    const xkv_graph_snapshot & snap, uint64_t seg_id, uint64_t seg_version,
    uint32_t group_index, std::string * err) {
    const xkv_graph_snapshot::xkv_native_group_arenas * found = nullptr;
    for (const auto & a : snap.native_group_arenas) {
        if (a.segment_id == seg_id && a.segment_version == seg_version && a.group_index == group_index) {
            if (found) {
                if (err) *err = "native arenas: ambiguous bundle for segment/group";
                return nullptr;
            }
            found = &a;
        }
    }
    if (!found && err) {
        *err = "native arenas not wired for segment " + std::to_string(seg_id) +
            " group " + std::to_string(group_index);
    }
    return found;
}

static bool xkv_is_turbo(ggml_type t) {
    return t == GGML_TYPE_TURBO2_0 || t == GGML_TYPE_TURBO3_0 || t == GGML_TYPE_TURBO4_0;
}

// Vulkan narrowing mirror for one K/V codec pair side: matched Turbo pairs
// (same type) or canonical pairs; exactly-one-turbo rejected. CPU allows
// mixed pairs (it decodes). Explicit init-time failure on non-CPU backends.
bool xkv_native_codec_pairs_ok(ggml_type a_k, ggml_type b_k, ggml_type a_v, ggml_type b_v,
    bool backend_is_cpu, std::string * err) {
    if (backend_is_cpu) return true;
    auto side_ok = [&](ggml_type a, ggml_type b, const char * side) -> bool {
        bool ta = xkv_is_turbo(a);
        bool tb = xkv_is_turbo(b);
        if (ta != tb) {
            if (err) *err = std::string("native codecs: mixed Turbo/canonical pair rejected (") + side + ")";
            return false;
        }
        if (ta && a != b) {
            if (err) *err = std::string("native codecs: Turbo pair type mismatch (") + side + ")";
            return false;
        }
        if (!ggml_xkv_codec_supported(a) || !ggml_xkv_codec_supported(b)) {
            if (err) *err = std::string("native codecs: unsupported codec (") + side + ")";
            return false;
        }
        return true;
    };
    return side_ok(a_k, b_k, "K") && side_ok(a_v, b_v, "V");
}

// Per-query resolved selection: hot cells and cold (view, sel, slot) triples.
// Group slots index the q tensor; rows outside any valid group are skipped
// exactly like the reference reader's null-Q skip.
struct xkv_native_query_sel {
    std::vector<xkv_native_hot_ent> hot;
    struct cold_ref {
        size_t view_idx = 0;
        uint32_t sel_idx = 0;
        uint32_t slot = 0;
    };
    std::vector<cold_ref> cold;
};

ggml_tensor * xkv_build_attention_native(
    ggml_context * ctx,
    ggml_tensor * q_h,
    const xkv_graph_snapshot & snap,
    ggml_tensor * k_store,
    ggml_tensor * v_store,
    ggml_tensor * sinks_head,
    bool backend_is_cpu,
    std::vector<xkv_native_status_item> & out_status_tensors,
    std::vector<xkv_native_fill_item> & out_fills,
    std::string * err) {
    auto fail = [&](const std::string & m) -> ggml_tensor * {
        if (err) *err = "xkv_build_attention_native: " + m;
        return nullptr;
    };
    if (!ctx || !q_h) return fail("null ctx/q");
    if (!k_store || !v_store) return fail("null hot storage views");
    const uint32_t DkL = snap.head_dim_k;
    const uint32_t DvL = snap.head_dim_v;
    const uint32_t gqa = snap.n_q_heads;
    const uint32_t NQ = snap.n_queries;
    const uint32_t h = snap.kv_head_index;
    if (DkL == 0 || DvL == 0 || gqa == 0 || NQ == 0) return fail("zero dims/heads/queries");
    if (q_h->type != GGML_TYPE_F32) return fail("q must be F32");
    if (q_h->ne[0] != (int64_t) DkL || q_h->ne[1] != (int64_t) gqa) return fail("q head layout mismatch");
    const int64_t G = q_h->ne[2];
    if (G <= 0) return fail("q groups nonlinear");
    if (!snap.hot_k_inv_rot.empty() || !snap.hot_v_inv_rot.empty()) {
        return fail("custom attention rotation not invertible on native path");
    }
    if (!snap.query_sink_logits.empty()) return fail("snapshot sinks require tensor path");
    if (snap.native_params.version != GGML_XKV_VERSION) {
        return fail("native params not wired by builder");
    }
    // Master arenas switch: the builder sets it only when backend-resident
    // bundles exist. Per-segment misses below stay specific.
    if (!snap.native_arenas_present) {
        return fail("native arenas not wired by builder");
    }
    // ---- hot storage geometry (mirrors reference gather validation) ----
    if (snap.hot_layout.v_transposed) return fail("transposed V layout unsupported");
    if (v_store->nb[1] > v_store->nb[2]) return fail("transposed V view unsupported");
    if (k_store->type != snap.hot_layout.k_type || v_store->type != snap.hot_layout.v_type) {
        return fail("storage tensor type does not match hot layout descriptor");
    }
    if (snap.hot_layout.head_dim_k != DkL || snap.hot_layout.head_dim_v != DvL) {
        return fail("hot layout head dims do not match snapshot dims");
    }
    auto hot_codec_ok = [&](ggml_type t, int64_t dim, const char * side) -> bool {
        if (t != GGML_TYPE_F32 && t != GGML_TYPE_F16 && t != GGML_TYPE_Q8_0 &&
            t != GGML_TYPE_TURBO2_0 && t != GGML_TYPE_TURBO3_0 && t != GGML_TYPE_TURBO4_0) {
            if (err) *err = std::string("xkv_build_attention_native: unsupported hot codec ") + ggml_type_name(t) + " (" + side + ")";
            return false;
        }
        if (t == GGML_TYPE_Q8_0 && dim % 32 != 0) {
            if (err) *err = "xkv_build_attention_native: Q8 hot dim not a multiple of 32";
            return false;
        }
        if ((t == GGML_TYPE_TURBO2_0 || t == GGML_TYPE_TURBO3_0 || t == GGML_TYPE_TURBO4_0)) {
            if (dim % 128 != 0) {
                if (err) *err = "xkv_build_attention_native: Turbo hot dim not a multiple of 128";
                return false;
            }
            if (ggml_turbo_layout_fingerprint(t) == 0) {
                if (err) *err = "xkv_build_attention_native: unsupported turbo layout";
                return false;
            }
        }
        return true;
    };
    const int64_t dim_k = k_store->ne[0];
    const int64_t dim_v = v_store->ne[0];
    if (dim_k <= 0 || dim_v <= 0) return fail("hot storage row width non-positive");
    if (!hot_codec_ok(k_store->type, dim_k, "K") || !hot_codec_ok(v_store->type, dim_v, "V")) return nullptr;
    // Storage rows are head-width (Turbo-padded) or logical; never narrower.
    if (dim_k < (int64_t) DkL || dim_v < (int64_t) DvL) return fail("storage narrower than head dim");
    if (k_store->type != GGML_TYPE_TURBO2_0 && k_store->type != GGML_TYPE_TURBO3_0 &&
        k_store->type != GGML_TYPE_TURBO4_0 && dim_k != (int64_t) DkL) {
        return fail("non-Turbo hot K row width mismatch");
    }
    if (v_store->type != GGML_TYPE_TURBO2_0 && v_store->type != GGML_TYPE_TURBO3_0 &&
        v_store->type != GGML_TYPE_TURBO4_0 && dim_v != (int64_t) DvL) {
        return fail("non-Turbo hot V row width mismatch");
    }
    if (k_store->ne[2] != v_store->ne[2]) return fail("hot K/V row spaces differ");
    const int64_t hot_rows_bound = k_store->ne[2];
    if (h >= (uint32_t) k_store->ne[1] || h >= (uint32_t) v_store->ne[1]) return fail("kv head out of storage range");
    // ---- Q slots ----
    std::vector<std::vector<uint32_t>> qslots;
    if (!build_native_qslots(snap, G, qslots, err)) return nullptr;
    // ---- scale / softcap (mirror reference defaults) ----
    double scale_d = snap.scale > 0.0f ? (double) snap.scale : 1.0 / std::sqrt((double) DkL);
    if (!std::isfinite(scale_d) || scale_d <= 0.0) return fail("non-finite/non-positive scale");
    if (!std::isfinite((double) snap.logit_softcap)) return fail("non-finite softcap");
    const float scale = (float) scale_d;
    // ---- sinks (broadcast only; per-query-varying sinks unrepresentable) ----
    ggml_tensor * sinks_t = nullptr;
    if (sinks_head) {
        if (sinks_head->type != GGML_TYPE_F32 || sinks_head->ne[0] != (int64_t) gqa ||
            !ggml_is_contiguous(sinks_head)) {
            return fail("sink head must be contiguous F32[gqa]");
        }
        sinks_t = sinks_head;
    }
    // ---- hot entries per query (mirror reference skip rules exactly) ----
    // V0: every view validates structurally (non-empty selection included),
    // exactly like the reference batch preflight.
    for (size_t v = 0; v < snap.segment_views.size(); ++v) {
        std::string verr;
        if (!snap.segment_views[v].validate(&verr)) return fail("view invalid: " + verr);
    }
    const bool has_counts_v = !snap.query_ddvr_group_counts.empty();
    // V1: cold group preflight over ALL view rows x ALL queries.
    for (size_t v = 0; v < snap.segment_views.size(); ++v) {
        const auto & vw = snap.segment_views[v];
        if (!vw.group_indices.empty() && vw.group_indices.size() != vw.selected_rows.size()) {
            return fail("view group length mismatch");
        }
        if (vw.storage_positions.size() != vw.selected_rows.size()) {
            return fail("view positions length mismatch");
        }
        if (!vw.row_generations.empty() && vw.row_generations.size() != vw.selected_rows.size()) {
            return fail("view generations length mismatch");
        }
        for (size_t i = 0; i < vw.selected_rows.size(); ++i) {
            uint32_t gidx = vw.group_indices.empty() ? 0 : vw.group_indices[i];
            for (uint32_t q = 0; q < NQ; ++q) {
                uint32_t gc = has_counts_v ? snap.query_ddvr_group_counts[q] : snap.n_ddvr_groups;
                if (gc > 0) {
                    if (gidx >= gc) return fail("cold row group exceeds query groups");
                } else if (gidx != 0) {
                    return fail("non-zero cold group without query groups");
                }
            }
        }
    }
    // V2: hot visibility/group mapping upfront (visible queries must resolve).
    for (const auto & hd : snap.hot_data) {
        if (!hd.is_valid) continue;
        if (!hd.query_group_indices.empty() && hd.query_group_indices.size() != NQ) {
            return fail("hot query-group array size mismatch");
        }
        for (uint32_t q = 0; q < NQ; ++q) {
            if (!hd.is_visible_to_query(q)) continue;
            // Per-query mapped group when nonempty, else ordinary fallback to
            // group_index; visible query with UINT32_MAX must fail closed.
            uint32_t g = hd.query_group_indices.empty() ? hd.group_index : hd.query_group_indices[q];
            if (g == UINT32_MAX) return fail("hot visible query has invalid group");
            uint32_t gc = has_counts_v ? snap.query_ddvr_group_counts[q] : snap.n_ddvr_groups;
            if (gc > 0) {
                if (g >= gc) return fail("hot row group exceeds query groups");
            } else if (g != 0) {
                return fail("non-zero hot group without query groups");
            }
        }
    }
    for (const auto & hd : snap.hot_data) {
        if (!hd.is_valid) continue;
        if (hd.kv_head != h) return fail("hot row KV head does not match snapshot KV head");
        if (!hd.use_storage_gather()) return fail("owned fallback hot rows unsupported on native path");
    }
    // Hot group arrays must match query count when present (reader rule).
    for (const auto & hd : snap.hot_data) {
        if (!hd.is_valid) continue;
        // Visibility sizing mirrors the reference reader (non-empty must
        // cover every query); per-query groups come from build_hot_rows
        // semantics (visible queries use the row's single group).
        if (!hd.query_visibility.empty() && hd.query_visibility.size() != NQ) {
            return fail("hot visibility size mismatch");
        }
    }
    std::vector<std::vector<xkv_native_hot_ent>> hot_per_q(NQ);
    for (uint32_t q = 0; q < NQ; ++q) {
        int64_t limit = q < snap.query_causal_limits.size() ? snap.query_causal_limits[q] : -1;
        for (const auto & hd : snap.hot_data) {
            if (!hd.is_valid) continue;
            if (!hd.is_visible_to_query(q)) continue;
            if (limit >= 0 && hd.storage_pos > limit) continue;
            uint32_t j = hd.query_group_indices.empty() ? hd.group_index : hd.query_group_indices[q];
            if (j >= qslots[q].size()) continue; // mirror reference null-Q skip
            if (hd.cell < 0 || hd.cell >= hot_rows_bound) return fail("hot cell out of storage range");
            if (hd.cell > (int64_t) INT32_MAX) return fail("hot cell out of int32 range");
            hot_per_q[q].push_back({hd.cell, qslots[q][j]});
        }
    }
    // Stream validation: each query must see a stream-homogeneous set of hot
    // rows (multi-sequence batch query isolation). A single query seeing hot
    // rows from different streams is a builder invariant violation; different
    // queries in the same batch may draw from different streams (handled via
    // stream partitioning in the sub-DAG below).
    std::vector<uint32_t> query_stream(NQ, 0);
    for (uint32_t q = 0; q < NQ; ++q) {
        bool have_st = false;
        for (const auto & hd : snap.hot_data) {
            if (!hd.is_valid || !hd.is_visible_to_query(q)) continue;
            if (!have_st) { query_stream[q] = hd.stream; have_st = true; }
            else if (hd.stream != query_stream[q]) {
                return fail("single query sees mixed hot streams (unsupported)");
            }
        }
        if (query_stream[q] >= (uint32_t) k_store->ne[3] ||
            query_stream[q] >= (uint32_t) v_store->ne[3]) {
            return fail("hot stream out of storage range");
        }
    }
    // ---- Multi-stream partitioning: stream-homogeneous sub-DAGs ----
    // If queries belong to different streams, partition into stream-homogeneous
    // subsets, build sub-DAG per stream, and concatenate back in original order.
    std::map<uint32_t, std::vector<uint32_t>> stream_groups;
    for (uint32_t q = 0; q < NQ; ++q) {
        stream_groups[query_stream[q]].push_back(q);
    }
    if (stream_groups.size() > 1) {
        // Sub-DAG per stream, then concatenate slices in original query order
        std::vector<ggml_tensor *> q_out_slices(NQ, nullptr);
        for (const auto & kv : stream_groups) {
            const uint32_t st_id = kv.first;
            const std::vector<uint32_t> & q_indices = kv.second;
            const uint32_t sub_nq = (uint32_t) q_indices.size();

            // Create sub-snapshot filtered to this stream's queries (move-only type)
            xkv_graph_snapshot sub_snap;
            sub_snap.head_dim_k = snap.head_dim_k;
            sub_snap.head_dim_v = snap.head_dim_v;
            sub_snap.n_q_heads = snap.n_q_heads;
            sub_snap.kv_head_index = snap.kv_head_index;
            sub_snap.scale = snap.scale;
            sub_snap.logit_softcap = snap.logit_softcap;
            sub_snap.hot_layout = snap.hot_layout;
            sub_snap.native_params = snap.native_params;
            sub_snap.native_arenas_present = snap.native_arenas_present;
            sub_snap.native_group_arenas = snap.native_group_arenas;
            sub_snap.native_rope_tables = snap.native_rope_tables;
            sub_snap.native_rope_table_data = snap.native_rope_table_data;
            sub_snap.native_landmarks = snap.native_landmarks;
            // Borrow segment views by pointer/pin reference
            for (const auto & v : snap.segment_views) {
                xkv_segment_read_view sub_v;
                sub_v.pin = xkv_reader_pin(v.pin.handle());
                sub_v.segment_version_id = v.segment_version_id;
                sub_v.storage_generation = v.storage_generation;
                sub_v.factor_group_index = v.factor_group_index;
                sub_v.owning_layer = v.owning_layer;
                sub_v.kv_head = v.kv_head;
                sub_v.selected_rows = v.selected_rows;
                sub_v.storage_positions = v.storage_positions;
                sub_v.row_generations = v.row_generations;
                sub_v.group_indices = v.group_indices;
                sub_v.membership_mask = v.membership_mask;
                sub_snap.segment_views.push_back(std::move(sub_v));
            }
            sub_snap.sr_mode = snap.sr_mode;
            sub_snap.sr_config = snap.sr_config;
            sub_snap.sr_legal_frags = snap.sr_legal_frags;
            sub_snap.sr_max_frag_rows = snap.sr_max_frag_rows;
            sub_snap.expected_stamp = snap.expected_stamp;
            sub_snap.store = snap.store;
            sub_snap.n_queries = sub_nq;
            sub_snap.sr_feature_offset = snap.sr_feature_offset;
            sub_snap.sr_feature_dim = snap.sr_feature_dim;
            sub_snap.sr_phase_fingerprint = snap.sr_phase_fingerprint;
            sub_snap.native_landmark_rebuild_requests = snap.native_landmark_rebuild_requests;
            // Remap precomputed sr_selection to sub-queries if present
            if (!snap.sr_selection.csr_ptrs.empty()) {
                sub_snap.sr_selection.gather_rows = snap.sr_selection.gather_rows;
                sub_snap.sr_selection.csr_ptrs.assign(sub_nq + 1, 0);
                for (uint32_t sub_q = 0; sub_q < sub_nq; ++sub_q) {
                    uint32_t orig_q = q_indices[sub_q];
                    if (orig_q < snap.sr_selection.csr_ptrs.size() - 1) {
                        uint32_t c_begin = snap.sr_selection.csr_ptrs[orig_q];
                        uint32_t c_end = snap.sr_selection.csr_ptrs[orig_q + 1];
                        for (uint32_t c = c_begin; c < c_end && c < snap.sr_selection.csr_indices.size(); ++c) {
                            sub_snap.sr_selection.csr_indices.push_back(snap.sr_selection.csr_indices[c]);
                            if (c < snap.sr_selection.csr_group_indices.size()) {
                                sub_snap.sr_selection.csr_group_indices.push_back(snap.sr_selection.csr_group_indices[c]);
                            }
                        }
                    }
                    sub_snap.sr_selection.csr_ptrs[sub_q + 1] = (uint32_t) sub_snap.sr_selection.csr_indices.size();
                }
            }

            for (uint32_t sub_q = 0; sub_q < sub_nq; ++sub_q) {
                uint32_t orig_q = q_indices[sub_q];
                if (orig_q < snap.query_causal_limits.size()) {
                    sub_snap.query_causal_limits.push_back(snap.query_causal_limits[orig_q]);
                }
                if (orig_q < snap.query_sink_logits.size()) {
                    sub_snap.query_sink_logits.push_back(snap.query_sink_logits[orig_q]);
                }
                if (orig_q < snap.query_ddvr_group_counts.size()) {
                    sub_snap.query_ddvr_group_counts.push_back(snap.query_ddvr_group_counts[orig_q]);
                }
                // When sub_q_tensor is constructed by slicing each query's exact group slots
                // and concatenating them along dim 2, the sub-DAG query slots become contiguous
                // [0, sub_slots_total). Leaving sub_snap.query_ddvr_group_offsets empty ensures
                // build_native_qslots correctly assigns contiguous local slot indices starting at 0.
            }
            // Filter hot data visibility to sub-queries
            sub_snap.hot_data.clear();
            for (const auto & hd : snap.hot_data) {
                if (!hd.is_valid || hd.stream != st_id) continue;
                xkv_graph_snapshot::hot_row_data sub_hd = hd;
                sub_hd.query_visibility.clear();
                sub_hd.query_group_indices.clear();
                bool any_vis = false;
                for (uint32_t sub_q = 0; sub_q < sub_nq; ++sub_q) {
                    uint32_t orig_q = q_indices[sub_q];
                    bool vis = hd.is_visible_to_query(orig_q);
                    sub_hd.query_visibility.push_back(vis);
                    if (vis) any_vis = true;
                    if (!hd.query_group_indices.empty() && orig_q < hd.query_group_indices.size()) {
                        sub_hd.query_group_indices.push_back(hd.query_group_indices[orig_q]);
                    }
                }
                if (any_vis) {
                    sub_snap.hot_data.push_back(std::move(sub_hd));
                }
            }

            // Build sub-DAG Q view: for continuous or single query subsets
            // Slice all Q slots belonging to this stream's queries and concatenate along dim 2
            // so sub_q_tensor's slots match the remapped local Q slots [0, sub_slots_total).
            ggml_tensor * sub_q_tensor = nullptr;
            ggml_tensor * cur_cat = nullptr;
            for (uint32_t sub_q = 0; sub_q < sub_nq; ++sub_q) {
                uint32_t orig_q = q_indices[sub_q];
                for (uint32_t slot : qslots[orig_q]) {
                    const size_t off = (size_t) slot * (size_t) q_h->nb[2];
                    ggml_tensor * slice = ggml_view_3d(ctx, q_h, q_h->ne[0], q_h->ne[1], 1,
                        q_h->nb[1], q_h->nb[2], off);
                    cur_cat = (cur_cat == nullptr) ? slice : ggml_concat(ctx, cur_cat, slice, 2);
                }
            }
            if (!cur_cat) return fail("empty sub-DAG query tensor");
            sub_q_tensor = cur_cat;

            // sub_snap's local slots are packed consecutively per sub-query according to
            // sub_snap.query_ddvr_group_counts (or n_ddvr_groups), so query_ddvr_group_offsets
            // must be empty to use contiguous local slot offsets starting at 0.
            sub_q_tensor = ggml_cont(ctx, sub_q_tensor);

            ggml_tensor * sub_out = xkv_build_attention_native(ctx, sub_q_tensor, sub_snap,
                k_store, v_store, sinks_head, backend_is_cpu, out_status_tensors, out_fills, err);
            if (!sub_out) return nullptr;

            // Slice sub-DAG output per query: sub_out is [(int64_t) DvL * gqa, sub_nq]
            for (uint32_t sub_q = 0; sub_q < sub_nq; ++sub_q) {
                uint32_t orig_q = q_indices[sub_q];
                const size_t off = (size_t) sub_q * (size_t) sub_out->nb[1];
                ggml_tensor * q_slice = ggml_view_2d(ctx, sub_out, sub_out->ne[0], 1, sub_out->nb[1], off);
                q_out_slices[orig_q] = q_slice;
            }
        }
        // Concatenate all slices back in original query order [DvL * gqa, NQ]
        ggml_tensor * result_concat = nullptr;
        for (uint32_t q = 0; q < NQ; ++q) {
            if (!q_out_slices[q]) return fail("missing sub-DAG query output slice");
            result_concat = (result_concat == nullptr) ? q_out_slices[q] : ggml_concat(ctx, result_concat, q_out_slices[q], 1);
        }
        return ggml_cont(ctx, result_concat);
    }

    uint32_t stream = query_stream[0];
    ggml_tensor * native_rope_t = snap.native_rope_tables;
    if (snap.native_params.rotary_dim > 0) {
        const uint32_t fc = snap.native_params.rotary_dim / 2;
        if (snap.native_params.rotary_dim % 2 != 0 || fc == 0) return fail("invalid rotary dimension");
        const size_t want_floats = (size_t) fc * 2;
        if (native_rope_t) {
            if (native_rope_t->type != GGML_TYPE_F32 ||
                (size_t) ggml_nelements(native_rope_t) != want_floats || !native_rope_t->buffer) {
                return fail("borrowed rope table shape/type/allocation mismatch");
            }
        } else {
            if (snap.native_rope_table_data.size() != want_floats) {
                return fail("exact native rope table data missing");
            }
            for (uint32_t f = 0; f < fc; ++f) {
                if (!std::isfinite(snap.native_rope_table_data[f]) ||
                    !std::isfinite(snap.native_rope_table_data[fc + f]) ||
                    !(snap.native_rope_table_data[fc + f] > 0.0f)) {
                    return fail("native rope table contains invalid values");
                }
            }
            native_rope_t = new_native_filled_1d(ctx, GGML_TYPE_F32, want_floats,
                snap.native_rope_table_data.data(), want_floats * sizeof(float), out_fills, err);
            if (!native_rope_t) return fail("native rope table tensor allocation failed");
        }
    } else {
        if (native_rope_t || !snap.native_rope_table_data.empty()) {
            return fail("rope table supplied for non-rotary layout");
        }
        native_rope_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 0);
        if (!native_rope_t) return fail("empty rope tensor allocation failed");
    }
    // ---- SR-after-Q / precomputed CSR selection (mirror reader exactly) ----
    // sr_mode 0: precomputed sr_selection (dense mode / off).
    // sr_mode 1: execute landmark selection now that Q is available.
    // When q_h->data is available (CPU reference backend), execute host oracle selection.
    // When on a device backend without host q_h->data, dispatch device-side landmark selector v3.
    const sr_batch_selection_result * active_sel = &snap.sr_selection;
    sr_batch_selection_result live_sel;
    ggml_tensor * device_sel_idx = nullptr;
    struct device_rows_tile {
        uint32_t arena_index = 0;
        uint32_t capacity = 0;
        ggml_tensor * refs = nullptr;
        ggml_tensor * positions = nullptr;
        ggml_tensor * entries = nullptr;
        ggml_tensor * row_ptrs = nullptr;
        ggml_tensor * status = nullptr;
    };
    std::vector<device_rows_tile> device_rows_tiles;
    uint32_t device_expanded_queries = 0;
    std::vector<int32_t> device_query_map;
    xkv_native_sr_plan device_sr_plan;
    xkv_sr_device_inputs device_inputs;

    if (snap.sr_mode != 0) {
        if (q_h->data) {
            std::vector<xkv_query_input> q_inputs;
            std::string q_err;
            if (!snap.build_query_inputs(static_cast<const float *>(q_h->data),
                    (size_t) ggml_nelements(q_h), q_inputs, &q_err)) {
                return fail("SR query input build failed: " + q_err);
            }
            std::string sr_err;
            if (!run_sr_after_q(snap, q_inputs, live_sel, &sr_err)) {
                return fail("SR-after-Q selection failed: " + sr_err);
            }
            active_sel = &live_sel;
        } else {
            // Device-side landmark selection v3 across all legal spans/segments.
            std::string plan_err;
            if (!build_native_sr_plan(snap, device_sr_plan, &plan_err)) return fail(plan_err);

            size_t expanded = 0;
            for (uint32_t q = 0; q < NQ; ++q) {
                if (!safe_add(expanded, qslots[q].size(), expanded)) return fail("expanded query count overflow");
            }
            if (expanded == 0 || expanded > GGML_XKV_LANDMARK_MAX_QUERIES) {
                return fail("expanded query count out of selector range");
            }
            std::vector<xkv_sr_device_query> dev_queries;
            dev_queries.reserve(expanded);
            device_query_map.assign(expanded * 2, 0);
            for (uint32_t q = 0; q < NQ; ++q) {
                for (uint32_t j = 0; j < qslots[q].size(); ++j) {
                    xkv_sr_device_query query;
                    query.parent_query = q;
                    query.causal_limit_pos = q < snap.query_causal_limits.size() ? snap.query_causal_limits[q] : -1;
                    query.query_pos = 0;
                    query.n_q_heads = gqa;
                    query.kv_head = h;
                    query.vis_group = (int32_t) j;
                    const size_t e = dev_queries.size();
                    dev_queries.push_back(query);
                    device_query_map[e * 2 + 0] = (int32_t) q;
                    device_query_map[e * 2 + 1] = (int32_t) qslots[q][j];
                }
            }
            device_expanded_queries = (uint32_t) dev_queries.size();

            uint64_t exp_phase_fp = snap.sr_phase_fingerprint;
            if (exp_phase_fp == 0) exp_phase_fp = snap.sr_legal_frags[0].key.phase_tx_fingerprint;
            std::string emit_err;
            if (!xkv_sr_emit_device_inputs(device_sr_plan.fragments, dev_queries, device_sr_plan.arenas,
                    exp_phase_fp, device_inputs, &emit_err)) {
                return fail("xkv_sr_emit_device_inputs failed: " + emit_err);
            }

            uint32_t top_k = snap.sr_config.sr_budget > 0 ? snap.sr_config.sr_budget : 16;
            uint32_t refine_cap = snap.sr_config.refine_max_rows > 0 ? snap.sr_config.refine_max_rows : 64;

            ggml_tensor * selector_q = nullptr;
            for (uint32_t e = 0; e < device_expanded_queries; ++e) {
                const uint32_t slot = (uint32_t) device_query_map[(size_t) e * 2 + 1];
                const size_t off = (size_t) slot * (size_t) q_h->nb[2];
                ggml_tensor * slice = ggml_view_2d(ctx, q_h, DkL, gqa, q_h->nb[1], off);
                selector_q = selector_q ? ggml_concat(ctx, selector_q, slice, 1) : slice;
            }
            selector_q = ggml_cont(ctx, selector_q);
            if (!selector_q) return fail("selector query packing failed");
            device_sel_idx = build_native_sr_selection(ctx, snap, device_sr_plan, device_inputs,
                selector_q, native_rope_t, top_k, scale, out_status_tensors, out_fills, err);
            if (!device_sel_idx) return nullptr;

            ggml_tensor * query_map_t = new_native_filled(ctx, GGML_TYPE_I32, 2, device_expanded_queries,
                device_query_map.data(), device_query_map.size() * sizeof(int32_t), out_fills, err);
            if (!query_map_t) return fail("device query map tensor allocation failed");
            ggml_tensor * mapped_sel_idx = ggml_concat(ctx, query_map_t, device_sel_idx, 0);
            if (!mapped_sel_idx) return fail("mapped selection tensor creation failed");

            ggml_tensor * fmeta_rows_t = new_native_filled(ctx, GGML_TYPE_I32, 6, device_inputs.n_frags,
                device_inputs.frag_meta.data(), device_inputs.frag_meta.size() * sizeof(int32_t), out_fills, err);
            ggml_tensor * frow_off_t = new_native_filled_1d(ctx, GGML_TYPE_I32, device_inputs.frag_row_off.size(),
                device_inputs.frag_row_off.data(), device_inputs.frag_row_off.size() * sizeof(int32_t), out_fills, err);
            const int32_t zero_i32 = 0;
            ggml_tensor * frow_ids_t = device_inputs.frag_row_ids.empty() ?
                new_native_filled_1d(ctx, GGML_TYPE_I32, 1, &zero_i32, sizeof(zero_i32), out_fills, err) :
                new_native_filled_1d(ctx, GGML_TYPE_I32, device_inputs.frag_row_ids.size(),
                    device_inputs.frag_row_ids.data(), device_inputs.frag_row_ids.size() * sizeof(int32_t), out_fills, err);
            ggml_tensor * frag_kv_t = new_native_filled(ctx, GGML_TYPE_I32, 3, device_inputs.n_frags,
                device_inputs.frag_kv.data(), device_inputs.frag_kv.size() * sizeof(int32_t), out_fills, err);
            ggml_tensor * row_pos_t = new_native_filled_1d(ctx, GGML_TYPE_I32, device_inputs.row_pos.size(),
                device_inputs.row_pos.data(), device_inputs.row_pos.size() * sizeof(int32_t), out_fills, err);
            if (!fmeta_rows_t || !frow_off_t || !frow_ids_t || !frag_kv_t || !row_pos_t) {
                return fail("row metadata tensor allocation failed");
            }

            size_t max_rows = 0;
            if (!safe_mul((size_t) device_expanded_queries, (size_t) refine_cap, max_rows) || max_rows == 0) {
                return fail("row tile bound overflow/zero");
            }
            const uint32_t tile_capacity = (uint32_t) std::min(max_rows, (size_t) kXkvNativeTileRows);
            std::vector<uint32_t> arena_bases(device_sr_plan.arenas.size(), 0);
            uint64_t global_base = 0;
            for (size_t a = 0; a < device_sr_plan.arenas.size(); ++a) {
                if (global_base > UINT32_MAX || device_sr_plan.arenas[a].row_count > UINT32_MAX - global_base) {
                    return fail("arena row domain exceeds uint32");
                }
                arena_bases[a] = (uint32_t) global_base;
                global_base += device_sr_plan.arenas[a].row_count;
            }
            std::vector<uint8_t> arena_has_frag(device_sr_plan.arenas.size(), 0);
            for (const auto & set : device_sr_plan.score_sets) arena_has_frag[set.arena_index] = 1;
            for (uint32_t a = 0; a < device_sr_plan.arenas.size(); ++a) {
                if (!arena_has_frag[a]) continue;
                for (size_t output_begin = 0; output_begin < max_rows; output_begin += tile_capacity) {
                    if (output_begin > UINT32_MAX) return fail("row tile offset exceeds uint32");
                    ggml_xkv_landmark_rows_params row_p = {};
                    row_p.version = GGML_XKV_LANDMARK_VERSION;
                    row_p.n_queries = device_expanded_queries;
                    row_p.top_k = top_k;
                    row_p.n_frags = device_inputs.n_frags;
                    row_p.refine_cap = refine_cap;
                    row_p.fstride = 6;
                    row_p.flags = device_inputs.sparse ? GGML_XKV_LANDMARK_FLAG_SPARSE_ROWS : 0;
                    row_p.max_frag_rows = device_inputs.max_frag_rows;
                    row_p.n_rows_total = (uint32_t) device_inputs.row_pos.size();
                    row_p.n_parent_queries = NQ;
                    row_p.has_query_map = 1;
                    row_p.arena_filter = a;
                    row_p.global_row_base = arena_bases[a];
                    row_p.arena_row_count = (uint32_t) device_sr_plan.arenas[a].row_count;
                    row_p.output_row_begin = (uint32_t) output_begin;

                    ggml_tensor * row_ptrs = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, NQ + 1);
                    ggml_tensor * row_positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, tile_capacity);
                    ggml_tensor * row_entries = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 4, tile_capacity);
                    ggml_tensor * row_status = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);
                    if (!row_ptrs || !row_positions || !row_entries || !row_status) {
                        return fail("row output tensor allocation failed");
                    }
                    ggml_tensor * row_refs = ggml_xkv_landmark_rows(ctx, mapped_sel_idx, fmeta_rows_t,
                        frow_off_t, frow_ids_t, frag_kv_t, row_pos_t, row_ptrs, row_positions,
                        row_entries, row_status, &row_p);
                    if (!row_refs) return fail("ggml_xkv_landmark_rows node failed");
                    char row_msg[256] = {};
                    if (!ggml_xkv_landmark_rows_supports(mapped_sel_idx, fmeta_rows_t, frow_off_t,
                            frow_ids_t, frag_kv_t, row_pos_t, row_ptrs, row_refs, row_positions,
                            row_entries, row_status, &row_p, row_msg, sizeof(row_msg))) {
                        return fail(std::string("ggml_xkv_landmark_rows unsupported: ") + row_msg);
                    }
                    out_status_tensors.push_back({row_status, xkv_native_status_policy::rows_clamp_retry});
                    device_rows_tiles.push_back({a, tile_capacity, row_refs, row_positions,
                        row_entries, row_ptrs, row_status});
                }
            }
            if (device_rows_tiles.empty()) return fail("device ROWS produced no static tiles");

        }
    }
    const bool device_sr = !device_rows_tiles.empty();
    const bool has_csr = !device_sr && !active_sel->csr_ptrs.empty();
    if (has_csr) {
        const auto & sel = *active_sel;
        if (sel.csr_ptrs.size() != (size_t) NQ + 1 || sel.csr_ptrs[0] != 0) return fail("CSR ptrs malformed");
        for (uint32_t q = 0; q < NQ; ++q) {
            if (sel.csr_ptrs[q] > sel.csr_ptrs[q + 1] || sel.csr_ptrs[q + 1] > sel.csr_indices.size()) {
                return fail("CSR range invalid");
            }
        }
        if (!sel.csr_group_indices.empty() && sel.csr_group_indices.size() != sel.csr_indices.size()) {
            return fail("CSR groups size mismatch");
        }
        // V3: CSR group validation over ALL entries (even causally masked
        // ones), mirroring the reference batch preflight exactly.
        if (!sel.csr_group_indices.empty()) {
            for (uint32_t q = 0; q < NQ; ++q) {
                for (uint32_t c = sel.csr_ptrs[q]; c < sel.csr_ptrs[q + 1]; ++c) {
                    uint32_t g = sel.csr_group_indices[c];
                    if (g == UINT32_MAX) continue;
                    uint32_t gc = has_counts_v ? snap.query_ddvr_group_counts[q] : snap.n_ddvr_groups;
                    if (gc > 0) {
                        if (g >= gc) return fail("CSR group exceeds query groups");
                    } else if (g != 0) {
                        return fail("non-zero CSR group without query groups");
                    }
                }
            }
        }
    }
    // Per (query, view) selection lists for linear tiling emission.
    std::vector<std::vector<std::vector<xkv_native_cold_sel>>> cold_by_qv(
        NQ, std::vector<std::vector<xkv_native_cold_sel>>(snap.segment_views.size()));
    for (uint32_t q = 0; !device_sr && q < NQ; ++q) {
        int64_t limit = q < snap.query_causal_limits.size() ? snap.query_causal_limits[q] : -1;
        auto emit_for_view_row = [&](size_t v, size_t i, uint32_t j) -> bool {
            if (j >= qslots[q].size()) return true; // mirror reference null-Q skip
            cold_by_qv[q][v].push_back({v, (uint32_t) i, 0, qslots[q][j]});
            return true;
        };
        if (has_csr) {
            const auto & sel = *active_sel;
            bool has_groups = !sel.csr_group_indices.empty();
            for (uint32_t c = sel.csr_ptrs[q]; c < sel.csr_ptrs[q + 1]; ++c) {
                uint32_t gi = sel.csr_indices[c];
                if (gi >= sel.gather_rows.size()) return fail("CSR index out of range");
                size_t v = 0, idx = 0;
                if (!resolve_native_csr_ref(snap, sel.gather_rows[gi], v, idx, err)) return nullptr;
                const auto & vw = snap.segment_views[v];
                if (!vw.membership_mask.empty()) {
                    if (idx >= vw.membership_mask.size()) return fail("membership mask index out of range");
                    if (!vw.membership_mask[idx]) continue;
                }
                if (limit >= 0 && vw.storage_positions[idx] > limit) continue;
                uint32_t j = 0;
                if (has_groups) {
                    uint32_t g = sel.csr_group_indices[c];
                    if (g == UINT32_MAX) {
                        j = vw.group_indices.empty() ? 0 : vw.group_indices[idx];
                    } else {
                        j = g;
                    }
                } else {
                    j = vw.group_indices.empty() ? 0 : vw.group_indices[idx];
                }
                if (!emit_for_view_row(v, idx, j)) return nullptr;
            }
        } else {
            for (size_t v = 0; v < snap.segment_views.size(); ++v) {
                const auto & vw = snap.segment_views[v];
                if (!vw.membership_mask.empty() && vw.membership_mask.size() != vw.selected_rows.size()) {
                    return fail("membership mask size mismatch");
                }
                for (size_t i = 0; i < vw.selected_rows.size(); ++i) {
                    if (!vw.membership_mask.empty() && !vw.membership_mask[i]) continue;
                    if (limit >= 0 && vw.storage_positions[i] > limit) continue;
                    uint32_t j = vw.group_indices.empty() ? 0 : vw.group_indices[i];
                    if (!emit_for_view_row(v, i, j)) return nullptr;
                }
            }
        }
    }
    // ---- union per view (ordered, deterministic) ----
    struct view_union {
        std::vector<uint32_t> sel;
        std::vector<uint32_t> pos_of;
    };
    std::vector<view_union> unions(snap.segment_views.size());
    for (size_t v = 0; v < snap.segment_views.size(); ++v) {
        unions[v].pos_of.assign(snap.segment_views[v].selected_rows.size(), UINT32_MAX);
    }
    for (uint32_t q = 0; q < NQ; ++q) {
        for (size_t v = 0; v < snap.segment_views.size(); ++v) {
            for (const auto & e : cold_by_qv[q][v]) {
                if (e.sel_idx >= unions[v].pos_of.size()) return fail("union index out of range");
                if (unions[v].pos_of[e.sel_idx] == UINT32_MAX) {
                    unions[v].pos_of[e.sel_idx] = (uint32_t) unions[v].sel.size();
                    unions[v].sel.push_back(e.sel_idx);
                }
            }
        }
    }
    for (uint32_t q = 0; q < NQ; ++q) {
        for (size_t v = 0; v < snap.segment_views.size(); ++v) {
            auto & b = cold_by_qv[q][v];
            for (auto & e : b) {
                uint32_t p = unions[v].pos_of[e.sel_idx];
                if (p == UINT32_MAX) return fail("union position missing");
                e.union_pos = p;
            }
            std::sort(b.begin(), b.end(), [](const xkv_native_cold_sel & a, const xkv_native_cold_sel & b) {
                if (a.union_pos != b.union_pos) return a.union_pos < b.union_pos;
                return a.slot < b.slot;
            });
        }
    }
    // ---- Q pad for Turbo-padded hot storage (zeros contribute nothing) ----
    ggml_tensor * q_use = q_h;
    if (dim_k > (int64_t) DkL) {
        int64_t pad = dim_k - (int64_t) DkL;
        if (pad > (int64_t) INT32_MAX) return fail("q pad out of range");
        q_use = ggml_cont(ctx, ggml_pad(ctx, q_h, (int) pad, 0, 0, 0));
        if (!q_use) return fail("q pad failed");
    }
    // ---- reconstruct + tile chain ----
    struct tile_desc {
        size_t view_idx = 0;
        uint32_t u0 = 0;
        uint32_t u1 = 0;
        ggml_tensor * k_cold = nullptr;
        ggml_tensor * v_cold = nullptr;
        uint32_t n_cold = 0;
        ggml_tensor * device_entries = nullptr;
        ggml_tensor * device_offsets = nullptr;
    };
    std::vector<tile_desc> tiles;
    ggml_tensor * dummy_k = nullptr;
    ggml_tensor * dummy_v = nullptr;
    for (size_t v = 0; v < snap.segment_views.size(); ++v) {
        const auto & vw = snap.segment_views[v];
        if (unions[v].sel.empty()) continue;
        const xkv_segment * seg = vw.get_segment();
        if (!seg) return fail("view missing segment");
        std::string gerr;
        const xkv_factor_group_payload * grp = vw.resolve_group(&gerr);
        if (!grp) return fail("group resolve failed: " + gerr);
        int32_t li = grp->find_owning_layer_index(vw.owning_layer);
        if (li < 0 || (size_t) li >= grp->layer_feature_offsets_k.size() ||
            (size_t) li >= grp->layer_feature_dims_k.size() ||
            (size_t) li >= grp->layer_feature_offsets_v.size() ||
            (size_t) li >= grp->layer_feature_dims_v.size()) {
            return fail("layer missing from group map");
        }
        uint32_t lay_off_k = grp->layer_feature_offsets_k[(size_t) li];
        uint32_t lay_dim_k = grp->layer_feature_dims_k[(size_t) li];
        uint32_t lay_off_v = grp->layer_feature_offsets_v[(size_t) li];
        uint32_t lay_dim_v = grp->layer_feature_dims_v[(size_t) li];
        if (lay_dim_k == 0 || lay_dim_k % DkL != 0) return fail("layer K width not a multiple of head dim");
        uint32_t nh_layer = lay_dim_k / DkL;
        if (h >= nh_layer) return fail("head out of layer bounds");
        if (lay_dim_v != nh_layer * DvL) return fail("V layer width mismatch");
        uint64_t eff_ver = vw.segment_version_id != 0 ? vw.segment_version_id : seg->segment_version;
        // Semantic group_index: matches backend/store identity and allows
        // non-contiguous group IDs or tail groups (vector ordinal aliasing
        // is invalid).
        const uint32_t group_index = grp->group_index;
        const xkv_graph_snapshot::xkv_native_group_arenas * arenas =
            xkv_find_native_arenas(snap, seg->segment_id, eff_ver, group_index, err);
        if (!arenas) return nullptr;
        if (!arenas->a_k || !arenas->b_k || !arenas->a_v || !arenas->b_v) {
            return fail("arena bundle incomplete");
        }
        if (!arenas->a_k->buffer || !arenas->b_k->buffer || !arenas->a_v->buffer || !arenas->b_v->buffer) {
            return fail("arena tensors not allocated");
        }
        // Codec pairing rule (mirrors the Vulkan backend hook for explicit
        // init-time failure; CPU decodes mixed pairs).
        {
            std::string perr;
            if (!xkv_native_codec_pairs_ok(arenas->a_k->type, arenas->b_k->type,
                    arenas->a_v->type, arenas->b_v->type, backend_is_cpu, &perr)) {
                return fail(perr);
            }
        }
        if (grp->rank_k == 0 || grp->rank_v == 0) return fail("zero factor rank");
        size_t bk_end = 0, bv_end = 0;
        if (!safe_add((size_t) lay_off_k, (size_t) lay_dim_k, bk_end) ||
            !safe_add((size_t) lay_off_v, (size_t) lay_dim_v, bv_end)) {
            return fail("B slice overflow");
        }
        if ((int64_t) bk_end > arenas->b_k->ne[1] || (int64_t) bv_end > arenas->b_v->ne[1]) {
            return fail("B slice out of arena bounds");
        }
        if (!check_i32_range((int64_t) lay_off_k, "layer K offset", err) ||
            !check_i32_range((int64_t) lay_dim_k, "layer K width", err) ||
            !check_i32_range((int64_t) lay_off_v, "layer V offset", err) ||
            !check_i32_range((int64_t) lay_dim_v, "layer V width", err) ||
            !check_i32_range((int64_t) nh_layer, "layer heads", err) ||
            !check_i32_range((int64_t) grp->rank_k, "rank K", err) ||
            !check_i32_range((int64_t) grp->rank_v, "rank V", err)) return nullptr;
        ggml_tensor * rope_t = native_rope_t;
        if (snap.native_params.rope_mode != GGML_XKV_ROPE_HALF &&
            snap.native_params.rope_mode != GGML_XKV_ROPE_INTERLEAVED) {
            return fail("unknown rope mode");
        }
        ggml_xkv_reconstruct_params rp = snap.native_params;
        rp.n_groups = 1;
        rp.rank_k = grp->rank_k;
        rp.rank_v = grp->rank_v;
        rp.dim_k = DkL;
        rp.dim_v = DvL;
        const std::vector<uint32_t> & usel = unions[v].sel;
        for (size_t c0 = 0; c0 < usel.size(); c0 += (size_t) kXkvNativeTileRows) {
            size_t c1 = std::min(usel.size(), c0 + (size_t) kXkvNativeTileRows);
            uint32_t n_sel = (uint32_t) (c1 - c0);
            rp.n_sel = n_sel;
            std::vector<int32_t> refs;
            refs.reserve((size_t) n_sel * 4);
            std::vector<int32_t> positions;
            positions.reserve(n_sel);
            for (size_t k = c0; k < c1; ++k) {
                uint32_t sel = usel[k];
                if (sel >= vw.selected_rows.size() || sel >= vw.storage_positions.size()) {
                    return fail("union index out of view range");
                }
                uint32_t a_row = vw.selected_rows[sel];
                int64_t pos = vw.storage_positions[sel];
                if (!check_i32_range((int64_t) a_row, "A row", err) ||
                    !check_i32_range(pos, "storage position", err)) return nullptr;
                if ((int64_t) a_row >= arenas->a_k->ne[1] || (int64_t) a_row >= arenas->a_v->ne[1]) {
                    return fail("A row out of arena bounds");
                }
                refs.push_back((int32_t) a_row);
                refs.push_back(0);
                refs.push_back(0);
                refs.push_back((int32_t) h);
                positions.push_back((int32_t) pos);
            }
            std::vector<int32_t> group_meta = {
                1, (int32_t) nh_layer, (int32_t) lay_off_k, (int32_t) lay_dim_k,
                (int32_t) lay_off_v, (int32_t) lay_dim_v, (int32_t) grp->rank_k, (int32_t) grp->rank_v};
            std::vector<int32_t> layer_meta = {
                (int32_t) lay_off_k, (int32_t) DkL, (int32_t) lay_off_v, (int32_t) DvL, (int32_t) nh_layer};
            ggml_tensor * refs_t = new_native_filled(ctx, GGML_TYPE_I32, 4, n_sel,
                refs.data(), refs.size() * sizeof(int32_t), out_fills, err);
            ggml_tensor * pos_t = new_native_filled_1d(ctx, GGML_TYPE_I32, n_sel,
                positions.data(), positions.size() * sizeof(int32_t), out_fills, err);
            ggml_tensor * gm_t = new_native_filled(ctx, GGML_TYPE_I32, 8, 1,
                group_meta.data(), group_meta.size() * sizeof(int32_t), out_fills, err);
            ggml_tensor * lm_t = new_native_filled(ctx, GGML_TYPE_I32, 5, 1,
                layer_meta.data(), layer_meta.size() * sizeof(int32_t), out_fills, err);
            if (!refs_t || !pos_t || !gm_t || !lm_t) return nullptr;
            ggml_tensor * rec_node = ggml_xkv_reconstruct(ctx, arenas->a_k, arenas->b_k, arenas->a_v,
                arenas->b_v, refs_t, pos_t, gm_t, lm_t, rope_t, &rp);
            if (!rec_node) return fail("reconstruct node failed");
            // Post-build validation on the real node (no duplicate dst
            // allocation); abandoned on failure, never returned.
            char sup_msg[256] = {};
            if (!ggml_xkv_reconstruct_supports(arenas->a_k, arenas->b_k, arenas->a_v, arenas->b_v,
                    refs_t, pos_t, gm_t, lm_t, rope_t, rec_node, &rp, sup_msg, sizeof(sup_msg))) {
                if (err) *err = std::string("xkv_build_attention_native: reconstruct unsupported: ") + sup_msg;
                return nullptr;
            }
            ggml_tensor * k_cold = ggml_view_2d(ctx, rec_node, DkL, n_sel, rec_node->nb[1], 0);
            ggml_tensor * v_cold = ggml_view_2d(ctx, rec_node, DvL, n_sel, rec_node->nb[1], (size_t) DkL * sizeof(float));
            if (!k_cold || !v_cold) return fail("cold views failed");
            // Pad reconstructed cold K/V to Turbo hot widths before
            // attention when logical Dk/Dv are narrower than storage dim_k/
            // dim_v (attention kernel requires k_cold ne[0] == dim_k ==
            // k_hot ne[0], v_cold ne[0] == dim_v == v_hot ne[0]).
            if (dim_k > (int64_t) DkL) {
                int64_t pad_k = dim_k - (int64_t) DkL;
                if (pad_k > (int64_t) INT32_MAX) return fail("cold k pad out of range");
                k_cold = ggml_cont(ctx, ggml_pad(ctx, k_cold, (int) pad_k, 0, 0, 0));
                if (!k_cold) return fail("cold k pad failed");
            }
            if (dim_v > (int64_t) DvL) {
                int64_t pad_v = dim_v - (int64_t) DvL;
                if (pad_v > (int64_t) INT32_MAX) return fail("cold v pad out of range");
                v_cold = ggml_cont(ctx, ggml_pad(ctx, v_cold, (int) pad_v, 0, 0, 0));
                if (!v_cold) return fail("cold v pad failed");
            }
            tile_desc td;
            td.view_idx = v;
            td.u0 = (uint32_t) c0;
            td.u1 = (uint32_t) (c0 + (c1 - c0));
            td.k_cold = k_cold;
            td.v_cold = v_cold;
            td.n_cold = n_sel;
            tiles.push_back(td);
        }
    }
    if (device_sr) {
        for (const auto & dt : device_rows_tiles) {
            if (dt.arena_index >= snap.native_group_arenas.size() ||
                dt.arena_index >= device_sr_plan.groups.size() || dt.capacity == 0 ||
                dt.capacity > (uint32_t) kXkvNativeTileRows || !dt.refs || !dt.positions ||
                !dt.entries || !dt.row_ptrs) {
                return fail("device row tile descriptor invalid");
            }
            const auto & arena = snap.native_group_arenas[dt.arena_index];
            const xkv_factor_group_payload * group = device_sr_plan.groups[dt.arena_index];
            const uint32_t owning_layer = device_sr_plan.owning_layers[dt.arena_index];
            const uint32_t arena_head = device_sr_plan.kv_heads[dt.arena_index];
            if (!group || arena_head != h || !arena.a_k || !arena.b_k || !arena.a_v || !arena.b_v) {
                return fail("device reconstruct arena identity incomplete");
            }
            const int32_t li = group->find_owning_layer_index(owning_layer);
            if (li < 0 || (size_t) li >= group->layer_feature_offsets_k.size() ||
                (size_t) li >= group->layer_feature_dims_k.size() ||
                (size_t) li >= group->layer_feature_offsets_v.size() ||
                (size_t) li >= group->layer_feature_dims_v.size()) {
                return fail("device reconstruct layer missing from factor group");
            }
            const uint32_t lay_off_k = group->layer_feature_offsets_k[(size_t) li];
            const uint32_t lay_dim_k = group->layer_feature_dims_k[(size_t) li];
            const uint32_t lay_off_v = group->layer_feature_offsets_v[(size_t) li];
            const uint32_t lay_dim_v = group->layer_feature_dims_v[(size_t) li];
            if (lay_dim_k == 0 || lay_dim_k % DkL != 0) return fail("device reconstruct K layer width invalid");
            const uint32_t nh_layer = lay_dim_k / DkL;
            if (arena_head >= nh_layer || lay_dim_v != nh_layer * DvL) {
                return fail("device reconstruct head/V geometry mismatch");
            }
            size_t end_k = 0;
            size_t end_v = 0;
            if (!safe_add((size_t) lay_off_k, (size_t) lay_dim_k, end_k) ||
                !safe_add((size_t) lay_off_v, (size_t) lay_dim_v, end_v) ||
                end_k > (size_t) arena.b_k->ne[1] || end_v > (size_t) arena.b_v->ne[1]) {
                return fail("device reconstruct B slice out of range");
            }
            std::string pair_err;
            if (!xkv_native_codec_pairs_ok(arena.a_k->type, arena.b_k->type,
                    arena.a_v->type, arena.b_v->type, backend_is_cpu, &pair_err)) {
                return fail(pair_err);
            }
            if (group->rank_k == 0 || group->rank_v == 0 ||
                !check_i32_range(lay_off_k, "device layer K offset", err) ||
                !check_i32_range(lay_dim_k, "device layer K width", err) ||
                !check_i32_range(lay_off_v, "device layer V offset", err) ||
                !check_i32_range(lay_dim_v, "device layer V width", err) ||
                !check_i32_range(nh_layer, "device layer heads", err) ||
                !check_i32_range(group->rank_k, "device rank K", err) ||
                !check_i32_range(group->rank_v, "device rank V", err)) {
                return nullptr;
            }
            std::vector<int32_t> group_meta = {
                1, (int32_t) nh_layer, (int32_t) lay_off_k, (int32_t) lay_dim_k,
                (int32_t) lay_off_v, (int32_t) lay_dim_v, (int32_t) group->rank_k, (int32_t) group->rank_v};
            std::vector<int32_t> layer_meta = {
                (int32_t) lay_off_k, (int32_t) DkL, (int32_t) lay_off_v, (int32_t) DvL, (int32_t) nh_layer};
            ggml_tensor * group_meta_t = new_native_filled(ctx, GGML_TYPE_I32, 8, 1,
                group_meta.data(), group_meta.size() * sizeof(int32_t), out_fills, err);
            ggml_tensor * layer_meta_t = new_native_filled(ctx, GGML_TYPE_I32, 5, 1,
                layer_meta.data(), layer_meta.size() * sizeof(int32_t), out_fills, err);
            if (!group_meta_t || !layer_meta_t) return nullptr;

            ggml_xkv_reconstruct_params rp = snap.native_params;
            rp.n_groups = 1;
            rp.n_sel = dt.capacity;
            rp.rank_k = group->rank_k;
            rp.rank_v = group->rank_v;
            rp.dim_k = DkL;
            rp.dim_v = DvL;
            ggml_tensor * rec_node = ggml_xkv_reconstruct(ctx, arena.a_k, arena.b_k, arena.a_v,
                arena.b_v, dt.refs, dt.positions, group_meta_t, layer_meta_t, native_rope_t, &rp);
            if (!rec_node) return fail("device-driven reconstruct node failed");
            char support_msg[256] = {};
            if (!ggml_xkv_reconstruct_supports(arena.a_k, arena.b_k, arena.a_v, arena.b_v,
                    dt.refs, dt.positions, group_meta_t, layer_meta_t, native_rope_t, rec_node,
                    &rp, support_msg, sizeof(support_msg))) {
                return fail(std::string("device-driven reconstruct unsupported: ") + support_msg);
            }
            ggml_tensor * k_cold = ggml_view_2d(ctx, rec_node, DkL, dt.capacity, rec_node->nb[1], 0);
            ggml_tensor * v_cold = ggml_view_2d(ctx, rec_node, DvL, dt.capacity, rec_node->nb[1],
                (size_t) DkL * sizeof(float));
            if (!k_cold || !v_cold) return fail("device cold views failed");
            if (dim_k > (int64_t) DkL) {
                k_cold = ggml_cont(ctx, ggml_pad(ctx, k_cold, (int) (dim_k - DkL), 0, 0, 0));
                if (!k_cold) return fail("device cold K pad failed");
            }
            if (dim_v > (int64_t) DvL) {
                v_cold = ggml_cont(ctx, ggml_pad(ctx, v_cold, (int) (dim_v - DvL), 0, 0, 0));
                if (!v_cold) return fail("device cold V pad failed");
            }
            tile_desc tile;
            tile.view_idx = (size_t) -1;
            tile.k_cold = k_cold;
            tile.v_cold = v_cold;
            tile.n_cold = dt.capacity;
            tile.device_entries = dt.entries;
            tile.device_offsets = dt.row_ptrs;
            tiles.push_back(tile);
        }
    }
    if (device_sr || tiles.empty()) {
        std::vector<float> zeros_k((size_t) dim_k, 0.0f);
        std::vector<float> zeros_v((size_t) dim_v, 0.0f);
        dummy_k = new_native_filled(ctx, GGML_TYPE_F32, dim_k, 1,
            zeros_k.data(), zeros_k.size() * sizeof(float), out_fills, err);
        dummy_v = new_native_filled(ctx, GGML_TYPE_F32, dim_v, 1,
            zeros_v.data(), zeros_v.size() * sizeof(float), out_fills, err);
        if (!dummy_k || !dummy_v) return nullptr;
        tile_desc td;
        td.view_idx = (size_t) -1;
        td.u0 = 0;
        td.u1 = 0;
        td.k_cold = dummy_k;
        td.v_cold = dummy_v;
        td.n_cold = 1;
        if (device_sr) {
            tiles.insert(tiles.begin(), td);
        } else {
            tiles.push_back(td);
        }
    }
    // ---- attention tiles with carry chaining (hot + sink once, first tile) ----
    ggml_tensor * carry = nullptr;
    ggml_tensor * final_dst = nullptr;
    for (size_t ti = 0; ti < tiles.size(); ++ti) {
        const tile_desc & td = tiles[ti];
        const bool first = (ti == 0);
        const bool last = (ti + 1 == tiles.size());
        ggml_tensor * entries_t = td.device_entries;
        ggml_tensor * offsets_t = td.device_offsets;
        uint32_t n_entries = 0;
        if (entries_t || offsets_t) {
            if (!entries_t || !offsets_t || first || entries_t->type != GGML_TYPE_I32 ||
                offsets_t->type != GGML_TYPE_I32 || entries_t->ne[0] != 4 ||
                entries_t->ne[1] <= 0 || entries_t->ne[1] > UINT32_MAX ||
                offsets_t->ne[0] != (int64_t) NQ + 1) {
                return fail("device attention entry tensors invalid");
            }
            n_entries = (uint32_t) entries_t->ne[1];
        } else {
            struct tile_ent { int32_t source; int64_t row; uint32_t slot; };
            std::vector<std::vector<tile_ent>> per_q(NQ);
            if (first) {
                for (uint32_t q = 0; q < NQ; ++q) {
                    for (const auto & he : hot_per_q[q]) {
                        per_q[q].push_back({GGML_XKV_ATTN_SOURCE_HOT, he.cell, he.slot});
                    }
                }
            }
            if (td.view_idx != (size_t) -1) {
            for (uint32_t q = 0; q < NQ; ++q) {
                const auto & bucket = cold_by_qv[q][td.view_idx];
                auto it0 = std::lower_bound(bucket.begin(), bucket.end(), td.u0,
                    [](const xkv_native_cold_sel & a, uint32_t v) { return a.union_pos < v; });
                for (auto it = it0; it != bucket.end() && it->union_pos < td.u1; ++it) {
                    size_t rel = (size_t) it->union_pos - td.u0;
                    if (rel > (size_t) UINT32_MAX) return fail("cold index overflow");
                    per_q[q].push_back({GGML_XKV_ATTN_SOURCE_COLD, (int64_t) rel, it->slot});
                }
            }
            }
            size_t total_entries = 0;
            for (uint32_t q = 0; q < NQ; ++q) {
                if (!safe_add(total_entries, per_q[q].size(), total_entries)) return fail("entry count overflow");
            }
            if (total_entries > (size_t) INT32_MAX - 1) return fail("entry count out of int32 range");
            std::vector<int32_t> entries;
            entries.reserve((total_entries + 1) * 4);
            std::vector<int32_t> offsets;
            offsets.reserve(NQ + 1);
            uint32_t running = 0;
            for (uint32_t q = 0; q < NQ; ++q) {
                offsets.push_back((int32_t) running);
                for (const auto & e : per_q[q]) {
                    if (!check_i32_range(e.row,
                            e.source == GGML_XKV_ATTN_SOURCE_HOT ? "hot cell" : "cold index", err)) {
                        return nullptr;
                    }
                    entries.push_back(e.source);
                    entries.push_back((int32_t) e.row);
                    if (e.slot > (uint32_t) INT32_MAX) return fail("group slot out of int32 range");
                    entries.push_back((int32_t) e.slot);
                    entries.push_back(1);
                    ++running;
                }
            }
            offsets.push_back((int32_t) running);
            entries.insert(entries.end(), {0, 0, 0, 0});
            entries_t = new_native_filled(ctx, GGML_TYPE_I32, 4, (int64_t) running + 1,
                entries.data(), entries.size() * sizeof(int32_t), out_fills, err);
            offsets_t = new_native_filled_1d(ctx, GGML_TYPE_I32, (int64_t) NQ + 1,
                offsets.data(), offsets.size() * sizeof(int32_t), out_fills, err);
            if (!entries_t || !offsets_t) return nullptr;
            n_entries = running + 1;
        }
        ggml_xkv_attention_params ap = {};
        ap.version = GGML_XKV_ATTN_VERSION;
        ap.n_queries = NQ;
        ap.n_groups = (uint32_t) G;
        ap.gqa_ratio = gqa;
        if (dim_k > (int64_t) UINT32_MAX || dim_v > (int64_t) UINT32_MAX ||
            hot_rows_bound > (int64_t) UINT32_MAX) {
            return fail("attention dims out of uint32 range");
        }
        ap.dim_k = (uint32_t) dim_k;
        ap.dim_v = (uint32_t) dim_v;
        ap.hot_rows = (uint32_t) hot_rows_bound;
        ap.n_cold = td.n_cold;
        ap.n_entries = n_entries;
        ap.kv_head = h;
        ap.stream = stream;
        ap.scale = scale;
        ap.logit_softcap = snap.logit_softcap;
        ap.flags = (first ? GGML_XKV_ATTN_FLAG_FIRST_TILE : 0) | (last ? GGML_XKV_ATTN_FLAG_FINAL_TILE : 0);
        ggml_tensor * st = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        if (!st) return fail("status tensor failed");
        ggml_tensor * node = ggml_xkv_attention(ctx, q_use, k_store, v_store, td.k_cold, td.v_cold,
            entries_t, offsets_t, first ? sinks_t : nullptr, st, carry, &ap);
        if (!node) return fail("attention node failed");
        // Post-build validation on the real node (no duplicate dst
        // allocation); abandoned on failure, never returned.
        char asup_msg[256] = {};
        if (!ggml_xkv_attention_supports(q_use, k_store, v_store, td.k_cold, td.v_cold,
                entries_t, offsets_t, first ? sinks_t : nullptr, st, carry, node, &ap,
                asup_msg, sizeof(asup_msg))) {
            if (err) *err = std::string("xkv_build_attention_native: attention unsupported: ") + asup_msg;
            return nullptr;
        }
        out_status_tensors.push_back({st, xkv_native_status_policy::code_only});
        carry = node;
        final_dst = node;
    }
    // ---- output: slice logical V rows, match reference shape ----
    if (!final_dst) return fail("no attention tiles built");
    ggml_tensor * out_v = ggml_view_3d(ctx, final_dst, DvL, gqa, NQ,
        final_dst->nb[1], final_dst->nb[2], 0);
    if (!out_v) return fail("output slice failed");
    ggml_tensor * out_c = ggml_cont(ctx, out_v);
    if (!out_c) return fail("output cont failed");
    ggml_tensor * out = ggml_reshape_2d(ctx, out_c, (int64_t) DvL * gqa, NQ);
    if (!out) return fail("output reshape failed");
    return out;
}

// ---- Device SR metadata emission / rows-output partition ----
// Graph side of ggml_xkv_landmark_rows: emits exact oracle/builder input
// metadata from plain tables and partitions rows-op outputs into per-arena
// tile ref lists. One GLOBAL rows call over concatenated segments; global
// row ids are per-(segment,group) arena ranges assigned cumulative in arenas
// order (disjoint, so rows-op dedup collapses only true duplicates).
bool xkv_sr_emit_device_inputs(const std::vector<xkv_sr_device_frag> & frags,
    const std::vector<xkv_sr_device_query> & queries,
    const std::vector<xkv_sr_device_arena> & arenas,
    uint64_t expected_phase_tx_fp,
    xkv_sr_device_inputs & out, std::string * err) {
    auto fail = [&](const std::string & m) {
        if (err) *err = std::string("xkv_sr_emit_device_inputs: ") + m;
        return false;
    };
    out = xkv_sr_device_inputs();
    if (frags.empty()) return fail("no frags");
    if (frags.size() > (size_t) INT32_MAX) return fail("frag count overflow");
    if (queries.empty() || queries.size() > GGML_XKV_LANDMARK_MAX_QUERIES) {
        return fail("query count out of range");
    }
    if (arenas.empty()) return fail("no arenas");
    if (arenas.size() > (size_t) INT32_MAX) return fail("arena count overflow");
    const uint32_t nf = (uint32_t) frags.size();
    const uint32_t nq = (uint32_t) queries.size();
    // Arena global bases, cumulative in order.
    std::vector<uint64_t> base(arenas.size(), 0);
    uint64_t total = 0;
    for (size_t a = 0; a < arenas.size(); ++a) {
        base[a] = total;
        total += arenas[a].row_count;
        if (total > (uint64_t) INT32_MAX) return fail("global row id overflow");
    }
    // (segment, version, group) -> arena index; duplicates are ambiguous.
    std::map<std::tuple<uint64_t, uint64_t, uint32_t>, size_t> arena_of;
    for (size_t a = 0; a < arenas.size(); ++a) {
        auto key = std::make_tuple(arenas[a].segment_id, arenas[a].segment_version,
            arenas[a].group_index);
        if (arena_of.count(key)) return fail("duplicate arena for segment/group");
        arena_of[key] = a;
    }
    // Graph-assigned segment ordinals (kernel-ignored, debuggability only).
    std::map<std::pair<uint64_t, uint64_t>, int32_t> seg_ord;
    out.frag_meta.resize((size_t) nf * 6);
    out.frag_positions.resize(nf);
    out.frag_gen.resize(nf);
    out.frag_kv.resize((size_t) nf * 3);
    out.frag_row_off.assign(nf + 1, 0);
    out.query_meta.resize((size_t) nq * 6);
    out.row_pos.assign((size_t) total, -1);
    std::vector<char> pos_filled(total, 0);
    uint32_t max_rc = 0;
    bool any_sparse = false;
    const uint32_t words_per_q = (nf + 31u) / 32u;
    out.elig_bits.assign((size_t) words_per_q * nq, 0);
    for (uint32_t f = 0; f < nf; ++f) {
        const xkv_sr_device_frag & fg = frags[f];
        if (fg.row_count == 0) return fail("empty frag");
        if (fg.row_count > GGML_XKV_LANDMARK_MAX_FRAG_ROWS) return fail("frag row_count above maximum");
        if (fg.eligible != 0 && fg.eligible != 1) return fail("eligible not 0/1 (bitmasks rejected)");
        if (fg.phase_tx_fingerprint != expected_phase_tx_fp) {
            return fail("phase transform fingerprint mismatch (stale/wrong-domain landmark)");
        }
        if (fg.storage_generation > (uint64_t) INT32_MAX) return fail("storage generation overflow");
        if (fg.kv_head > (uint32_t) INT32_MAX) return fail("kv head overflow");
        if (fg.owning_layer > (uint32_t) INT32_MAX) return fail("owning layer overflow");
        if (fg.group_index > (uint32_t) INT32_MAX) return fail("group index overflow");
        if (fg.ddvr_group > (uint32_t) INT32_MAX) return fail("DDVR group overflow");
        if (fg.storage_positions.size() != fg.row_count) return fail("positions length mismatch");
        auto akey = std::make_tuple(fg.segment_id, fg.segment_version, fg.group_index);
        auto ait = arena_of.find(akey);
        if (ait == arena_of.end()) return fail("arena miss for frag segment/group");
        const size_t a = ait->second;
        if (fg.landmark_codec_fp != arenas[a].landmark_codec_fp) {
            return fail("landmark codec fingerprint mismatch for arena");
        }
        auto skey = std::make_pair(fg.segment_id, fg.segment_version);
        auto sit = seg_ord.find(skey);
        int32_t sord = 0;
        if (sit == seg_ord.end()) {
            if (seg_ord.size() >= (size_t) INT32_MAX) return fail("segment ordinal overflow");
            sord = (int32_t) seg_ord.size();
            seg_ord[skey] = sord;
        } else {
            sord = sit->second;
        }
        const int32_t gen32 = (int32_t) fg.storage_generation;
        const int32_t elig = fg.eligible;
        if (fg.row_count > max_rc) max_rc = fg.row_count;
        if (!fg.sparse) {
            uint64_t end = (uint64_t) fg.row_begin + fg.row_count;
            if (end > arenas[a].row_count) return fail("contiguous rows out of arena range");
            const uint64_t grb = base[a] + fg.row_begin;
            out.frag_meta[(size_t) f * 6 + 0] = (int32_t) grb;
            out.frag_meta[(size_t) f * 6 + 1] = (int32_t) fg.row_count;
            out.frag_meta[(size_t) f * 6 + 2] = sord;
            out.frag_meta[(size_t) f * 6 + 3] = elig;
            out.frag_meta[(size_t) f * 6 + 4] = -1; // contiguous
            out.frag_meta[(size_t) f * 6 + 5] = (int32_t) fg.ddvr_group;
            for (uint32_t k = 0; k < fg.row_count; ++k) {
                const uint64_t g = grb + k;
                if (!check_i32_range(fg.storage_positions[k], "storage position", err)) return false;
                const int32_t p = (int32_t) fg.storage_positions[k];
                if (pos_filled[(size_t) g] && out.row_pos[(size_t) g] != p) {
                    return fail("conflicting positions for one global row");
                }
                out.row_pos[(size_t) g] = p;
                pos_filled[(size_t) g] = 1;
            }
            out.frag_row_off[f + 1] = out.frag_row_off[f];
        } else {
            if (fg.row_begin != 0) return fail("sparse frag must have row_begin 0");
            if (fg.sparse_rows.size() != fg.row_count) return fail("sparse list length mismatch");
            const int32_t off = (int32_t) out.frag_row_ids.size();
            for (uint32_t k = 0; k < fg.row_count; ++k) {
                const int32_t id = fg.sparse_rows[k];
                if (id < -1) return fail("bad sparse hole marker");
                if (id < 0) {
                    out.frag_row_ids.push_back(-1);
                    continue;
                }
                if ((uint64_t) (uint32_t) id >= arenas[a].row_count) {
                    return fail("sparse row out of arena range");
                }
                const uint64_t g = base[a] + (uint32_t) id;
                if (!check_i32_range(fg.storage_positions[k], "storage position", err)) return false;
                const int32_t p = (int32_t) fg.storage_positions[k];
                if (pos_filled[(size_t) g] && out.row_pos[(size_t) g] != p) {
                    return fail("conflicting positions for one global row");
                }
                out.row_pos[(size_t) g] = p;
                pos_filled[(size_t) g] = 1;
                out.frag_row_ids.push_back((int32_t) g);
            }
            if (out.frag_row_ids.size() > (size_t) INT32_MAX) return fail("sparse pool overflow");
            out.frag_meta[(size_t) f * 6 + 0] = 0;
            out.frag_meta[(size_t) f * 6 + 1] = (int32_t) fg.row_count;
            out.frag_meta[(size_t) f * 6 + 2] = sord;
            out.frag_meta[(size_t) f * 6 + 3] = elig;
            out.frag_meta[(size_t) f * 6 + 4] = off;
            out.frag_meta[(size_t) f * 6 + 5] = (int32_t) fg.ddvr_group;
            out.frag_row_off[f + 1] = (int32_t) out.frag_row_ids.size();
            any_sparse = true;
        }
        out.frag_positions[f] = fg.frag_pos;
        out.frag_gen[f] = gen32;
        out.frag_kv[(size_t) f * 3 + 0] = (int32_t) fg.group_index;
        out.frag_kv[(size_t) f * 3 + 1] = (int32_t) a; // GLOBAL tile ordinal = arena index
        out.frag_kv[(size_t) f * 3 + 2] = (int32_t) fg.kv_head;

        // Populate per-query eligibility bitset (tested in v3 selector BEFORE scoring):
        const uint32_t f_word = f / 32u;
        const uint32_t f_bit  = f % 32u;
        for (uint32_t q = 0; q < nq; ++q) {
            const uint32_t parent = queries[q].parent_query == UINT32_MAX ? q : queries[q].parent_query;
            if (!fg.query_visibility.empty() && parent >= fg.query_visibility.size()) {
                return fail("query parent index out of visibility range");
            }
            bool vis = true;
            if (!fg.query_visibility.empty()) {
                vis = parent < fg.query_visibility.size() && fg.query_visibility[parent];
            }
            if (elig && vis) {
                out.elig_bits[(size_t) q * words_per_q + f_word] |= (1 << f_bit);
            }
        }
    }
    for (uint32_t q = 0; q < nq; ++q) {
        const xkv_sr_device_query & qq = queries[q];
        if (qq.causal_limit_pos < -1) return fail("causal limit below -1");
        if (qq.causal_limit_pos > INT32_MAX) return fail("causal limit above int32 range");
        if (qq.query_pos < INT32_MIN || qq.query_pos > INT32_MAX) return fail("query position outside int32 range");
        if (qq.vis_group < -1) return fail("visibility group below wildcard");
        if (qq.n_q_heads == 0 || qq.n_q_heads > GGML_XKV_LANDMARK_MAX_Q_HEADS) {
            return fail("n_q_heads out of range");
        }
        if (qq.kv_head > (uint32_t) INT32_MAX) return fail("query kv head overflow");
        out.query_meta[(size_t) q * 6 + 0] = (int32_t) qq.causal_limit_pos;
        out.query_meta[(size_t) q * 6 + 1] = (int32_t) qq.query_pos;
        out.query_meta[(size_t) q * 6 + 2] = (int32_t) qq.n_q_heads;
        out.query_meta[(size_t) q * 6 + 3] = (int32_t) qq.kv_head;
        out.query_meta[(size_t) q * 6 + 4] = qq.vis_group;
        out.query_meta[(size_t) q * 6 + 5] = 0;
    }
    out.n_frags = nf;
    out.n_queries = nq;
    out.n_rows_total = (uint32_t) total;
    out.max_frag_rows = max_rc;
    out.sparse = any_sparse;
    return true;
}

bool xkv_sr_partition_rows_outputs(const int32_t * row_ptrs, const int32_t * row_refs,
    const int32_t * out_row_pos, const int32_t * row_status, uint32_t n_queries,
    uint32_t refine_cap, const std::vector<xkv_sr_device_arena> & arenas,
    const std::vector<int32_t> & row_pos, xkv_sr_partition_result & out,
    std::string * err) {
    auto fail = [&](const std::string & m) {
        if (err) *err = std::string("xkv_sr_partition_rows_outputs: ") + m;
        return false;
    };
    out = xkv_sr_partition_result();
    if (!row_ptrs || !row_refs || !out_row_pos || !row_status) return fail("null pointer");
    if (n_queries == 0 || n_queries > GGML_XKV_LANDMARK_MAX_QUERIES) {
        return fail("query count out of range");
    }
    if (refine_cap == 0 || refine_cap > GGML_XKV_LANDMARK_MAX_REFINE_CAP) {
        return fail("refine_cap out of range");
    }
    if (arenas.empty()) return fail("no arenas");
    if (arenas.size() > (size_t) INT32_MAX) return fail("arena count overflow");
    if (row_status[0] != GGML_XKV_LANDMARK_STATUS_OK) {
        return fail("rows status != OK");
    }
    if (row_status[2] < 0 || (uint64_t) row_status[2] > n_queries) {
        return fail("bad clamp count");
    }
    out.clamped_queries = (uint32_t) row_status[2];
    out.stale = (row_status[2] > 0); // clamped: retry with larger cap, never silent
    std::vector<uint64_t> base(arenas.size(), 0);
    uint64_t total = 0;
    for (size_t a = 0; a < arenas.size(); ++a) {
        base[a] = total;
        total += arenas[a].row_count;
        if (total > (uint64_t) INT32_MAX) return fail("global row id overflow");
        if (arenas[a].group_index > (uint32_t) INT32_MAX) return fail("group unrepresentable");
    }
    if (row_ptrs[0] != 0) return fail("row_ptrs[0] != 0");
    std::vector<std::vector<int32_t>> t_refs(arenas.size());
    std::vector<std::vector<int32_t>> t_pos(arenas.size());
    for (uint32_t q = 0; q < n_queries; ++q) {
        const int32_t c0 = row_ptrs[q], c1 = row_ptrs[q + 1];
        if (c1 < c0) return fail("row_ptrs non-monotonic");
        const uint32_t cnt = (uint32_t) (c1 - c0);
        if (cnt > refine_cap) return fail("count exceeds refine_cap");
        int64_t prev = -1;
        for (uint32_t i = 0; i < refine_cap; ++i) {
            const size_t slot = ((size_t) q * refine_cap + i) * 4;
            const int32_t g = row_refs[slot + 0];
            const int32_t grp = row_refs[slot + 1];
            const int32_t sl = row_refs[slot + 2];
            const int32_t hd = row_refs[slot + 3];
            const int32_t p = out_row_pos[(size_t) q * refine_cap + i];
            if (i < cnt) {
                if (g < 0) return fail("negative row in counted range");
                if (sl < 0 || (uint64_t) sl >= arenas.size()) return fail("layer slot OOB");
                if (grp < 0 || (uint32_t) grp != arenas[(size_t) sl].group_index) {
                    return fail("group cross-check mismatch");
                }
                const uint64_t b = base[(size_t) sl];
                if ((uint64_t) g < b || (uint64_t) g >= b + arenas[(size_t) sl].row_count) {
                    return fail("row outside tile range");
                }
                if ((int64_t) g <= prev) return fail("rows not strictly ascending");
                prev = g;
                if (hd < 0) return fail("negative head");
                if ((size_t) g >= row_pos.size() || p != row_pos[(size_t) g]) {
                    return fail("position cross-check mismatch");
                }
                t_refs[(size_t) sl].push_back((int32_t) ((uint64_t) g - b));
                t_refs[(size_t) sl].push_back(grp);
                t_refs[(size_t) sl].push_back(sl);
                t_refs[(size_t) sl].push_back(hd);
                t_pos[(size_t) sl].push_back(p);
            } else {
                if (g != -1 || grp != -1 || sl != -1 || hd != -1 || p != -1) {
                    return fail("padding past count not -1");
                }
            }
        }
    }
    if (row_ptrs[n_queries] < 0) return fail("negative total");
    out.total_rows = (uint32_t) row_ptrs[n_queries];
    for (size_t a = 0; a < arenas.size(); ++a) {
        if (t_refs[a].empty()) continue;
        xkv_sr_device_tile_rows t;
        t.arena_index = (uint32_t) a;
        t.refs = std::move(t_refs[a]);
        t.positions = std::move(t_pos[a]);
        out.tiles.push_back(std::move(t));
    }
    return true;
}

bool xkv_snapshot_refresh_content(xkv_graph_snapshot & snap,
    const xkv_snapshot_refresh_data & fresh, std::string * err) {
    auto fail = [&](const std::string & msg) {
        if (err) *err = "xkv_snapshot_refresh_content: " + msg;
        return false;
    };

    // 1. Topology epoch validation: view.* epochs MUST match exactly.
    // Any change to topology_epoch, publish_epoch, or layout_epoch requires a graph rebuild.
    if (fresh.stamp.view.topology_epoch != snap.expected_stamp.view.topology_epoch ||
        fresh.stamp.view.publish_epoch != snap.expected_stamp.view.publish_epoch ||
        fresh.stamp.view.layout_epoch != snap.expected_stamp.view.layout_epoch) {
        return fail("topology epoch mismatch (rebuild required)");
    }

    // 2. Capacity & shape validation for hot rows.
    if (fresh.hot_rows.size() != snap.hot_data.size()) {
        return fail("hot row count changed (" + std::to_string(snap.hot_data.size()) +
                    " -> " + std::to_string(fresh.hot_rows.size()) + ", rebuild required)");
    }
    if (snap.capacities.hot_max > 0 && fresh.hot_rows.size() > snap.capacities.hot_max) {
        return fail("hot row count exceeds capacity bound (" +
                    std::to_string(fresh.hot_rows.size()) + " > " +
                    std::to_string(snap.capacities.hot_max) + ")");
    }

    for (size_t i = 0; i < fresh.hot_rows.size(); ++i) {
        const auto & r = fresh.hot_rows[i];
        if (!r.query_visibility.empty() && r.query_visibility.size() != snap.n_queries) {
            return fail("hot row query_visibility size mismatch at index " + std::to_string(i));
        }
        if (!r.query_group_indices.empty() && r.query_group_indices.size() != snap.n_queries) {
            return fail("hot row query_group_indices size mismatch at index " + std::to_string(i));
        }
    }

    // 3. Query causal limits validation.
    if (!fresh.query_causal_limits.empty() && fresh.query_causal_limits.size() != snap.n_queries) {
        return fail("query_causal_limits size mismatch (" +
                    std::to_string(fresh.query_causal_limits.size()) + " != " +
                    std::to_string(snap.n_queries) + ")");
    }

    // 4. Sink logits validation.
    if (!fresh.query_sink_logits.empty()) {
        if (fresh.query_sink_logits.size() != snap.n_queries) {
            return fail("query_sink_logits query count mismatch (" +
                        std::to_string(fresh.query_sink_logits.size()) + " != " +
                        std::to_string(snap.n_queries) + ")");
        }
        for (uint32_t q = 0; q < snap.n_queries; ++q) {
            if (fresh.query_sink_logits[q].size() != snap.n_q_heads) {
                return fail("query_sink_logits head count mismatch at query " + std::to_string(q));
            }
        }
    }

    // 5. DDVR group counts validation: must match existing effective topology.
    if (!fresh.query_ddvr_group_counts.empty()) {
        if (fresh.query_ddvr_group_counts.size() != snap.n_queries) {
            return fail("query_ddvr_group_counts size mismatch");
        }
        for (uint32_t q = 0; q < snap.n_queries; ++q) {
            uint32_t cur = snap.get_query_group_count(q);
            if (fresh.query_ddvr_group_counts[q] != cur) {
                return fail("query DDVR group count changed at query " + std::to_string(q) +
                            " (" + std::to_string(cur) + " -> " +
                            std::to_string(fresh.query_ddvr_group_counts[q]) + ", rebuild required)");
            }
        }
    }

    // 6. Apply content updates (same shape).
    snap.expected_stamp = fresh.stamp;
    snap.hot_data = fresh.hot_rows;
    if (!fresh.query_causal_limits.empty()) {
        snap.query_causal_limits = fresh.query_causal_limits;
    }
    if (!fresh.query_sink_logits.empty()) {
        snap.query_sink_logits = fresh.query_sink_logits;
    }
    if (!fresh.query_ddvr_group_counts.empty()) {
        snap.query_ddvr_group_counts = fresh.query_ddvr_group_counts;
    }

    // 7. Re-sync compute caches (gather_cells_cache, gather_pos_cache, hot_cache).
    std::string perr;
    if (!snap.preallocate_compute_state(&perr)) {
        return fail("re-syncing preallocated compute state failed: " + perr);
    }

    snap.force_retry_stale = false;
    return true;
}

} // namespace llama_xkv



