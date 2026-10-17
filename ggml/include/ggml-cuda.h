#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#ifdef GGML_USE_HIP
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif
#define GGML_CUDA_MAX_DEVICES       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_cuda_init(int device);

GGML_BACKEND_API bool ggml_backend_is_cuda(ggml_backend_t backend);

// Synchronize a CUDA backend and discard captured graphs whose device
// pointers must not survive a compute-arena repartition.
GGML_BACKEND_API bool ggml_backend_cuda_graph_reset(ggml_backend_t backend);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// Physical CUDA storage shared by phase-specific compute and KV slices.
typedef struct ggml_backend_cuda_phase_arena * ggml_backend_cuda_phase_arena_t;

GGML_BACKEND_API ggml_backend_cuda_phase_arena_t ggml_backend_cuda_phase_arena_new(
    int device, size_t size);
GGML_BACKEND_API void ggml_backend_cuda_phase_arena_free(
    ggml_backend_cuda_phase_arena_t arena);
GGML_BACKEND_API size_t ggml_backend_cuda_phase_arena_size(
    ggml_backend_cuda_phase_arena_t arena);
GGML_BACKEND_API bool ggml_backend_cuda_phase_arena_set_compute(
    ggml_backend_cuda_phase_arena_t arena, size_t offset, size_t size);
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_phase_arena_buffer_type(
    ggml_backend_cuda_phase_arena_t arena);

// conduct allreduce operation between devices
GGML_BACKEND_API bool ggml_backend_cuda_allreduce_tensor(ggml_backend_t * backends, struct ggml_tensor ** tensors, size_t n_backends);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

// Per-context resources for block-granular KV streaming. The buffer type owns
// authoritative pinned-host tensor storage; the runtime owns CUDA staging.
typedef struct ggml_backend_cuda_kv_stream_runtime * ggml_backend_cuda_kv_stream_runtime_t;

struct ggml_backend_cuda_kv_stream_params {
    int device;
    size_t stage_bytes;
    uint32_t stage_slots;
    size_t pool_bytes;
    size_t conversion_bytes;
    uint32_t resident_layer_count;
    uint32_t page_tokens;
    uint32_t decode_span_pages;
};

struct ggml_backend_cuda_kv_stream_type_capabilities {
    bool classified;
    bool storage;
    bool online_write;
    bool decode_f16;
    bool direct_attention;
    bool requires_initialization;
    bool requires_importance_matrix;
    bool auxiliary;
};

enum ggml_backend_cuda_kv_stream_attention_mode {
    GGML_BACKEND_CUDA_KV_STREAM_ATTENTION_UNSUPPORTED = 0,
    GGML_BACKEND_CUDA_KV_STREAM_ATTENTION_DIRECT      = 1,
    GGML_BACKEND_CUDA_KV_STREAM_ATTENTION_F16         = 2,
};

GGML_BACKEND_API struct ggml_backend_cuda_kv_stream_type_capabilities
ggml_backend_cuda_kv_stream_get_type_capabilities(enum ggml_type type);
GGML_BACKEND_API enum ggml_backend_cuda_kv_stream_attention_mode
ggml_backend_cuda_kv_stream_get_attention_mode(enum ggml_type type_k, enum ggml_type type_v);

GGML_BACKEND_API ggml_backend_cuda_kv_stream_runtime_t ggml_backend_cuda_kv_stream_runtime_new(
    struct ggml_backend_cuda_kv_stream_params params);
GGML_BACKEND_API ggml_backend_cuda_kv_stream_runtime_t
ggml_backend_cuda_kv_stream_runtime_new_in_phase_arena(
    ggml_backend_cuda_phase_arena_t arena,
    size_t maximum_pool_bytes,
    struct ggml_backend_cuda_kv_stream_params params);
GGML_BACKEND_API void ggml_backend_cuda_kv_stream_runtime_free(
    ggml_backend_cuda_kv_stream_runtime_t runtime);
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_kv_stream_buffer_type(
    ggml_backend_cuda_kv_stream_runtime_t runtime);
GGML_BACKEND_API bool ggml_backend_cuda_kv_stream_resize_pool(
    ggml_backend_cuda_kv_stream_runtime_t runtime,
    size_t pool_bytes,
    uint32_t active_pages_per_layer,
    uint32_t stage_slots);
GGML_BACKEND_API bool ggml_backend_cuda_kv_stream_reconfigure(
    ggml_backend_cuda_kv_stream_runtime_t runtime,
    uint32_t active_pages_per_layer,
    uint32_t stage_slots);
GGML_BACKEND_API bool ggml_backend_cuda_kv_stream_repartition(
    ggml_backend_cuda_kv_stream_runtime_t runtime,
    uint32_t stage_slots);
GGML_BACKEND_API bool ggml_backend_cuda_kv_stream_set_decode_layout(
    ggml_backend_cuda_kv_stream_runtime_t runtime,
    uint32_t active_pages_per_layer);
GGML_BACKEND_API bool ggml_backend_cuda_kv_stream_mark_dirty_rows(
    ggml_backend_cuda_kv_stream_runtime_t runtime,
    const int64_t * rows,
    size_t count);

GGML_BACKEND_API int  ggml_backend_cuda_get_device_count(void);
GGML_BACKEND_API void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_BACKEND_API void ggml_backend_cuda_unregister_host_buffer(void * buffer);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cuda_reg(void);

// ---- TriAttention GPU scoring ----
// Opaque handle for GPU-resident scoring state
typedef struct triattention_gpu_state triattention_gpu_state;

struct triattention_gpu_head_calib {
    const float * q_mean_real;    // [freq_count]
    const float * q_mean_imag;
    const float * q_mean_abs;
    const float * extra_weight;
};

struct triattention_gpu_config {
    uint32_t head_dim;
    uint32_t freq_count;
    uint32_t n_kv_heads;
    uint32_t n_sampled;
    uint32_t n_offsets;
    uint32_t rope_style; // 0=half/NeoX, 1=adjacent even/odd
    enum ggml_type k_type;
    bool     need_wht_inv;
    bool     disable_trig;
};

GGML_BACKEND_API triattention_gpu_state * triattention_gpu_init(
    const struct triattention_gpu_config * config,
    const struct triattention_gpu_head_calib * head_calibs,
    const float * omega,
    const float * freq_scale_sq,
    const float * offsets,
    void * stream);

GGML_BACKEND_API void triattention_gpu_score_head(
    triattention_gpu_state * state,
    const void   * k_data_dev,
    uint64_t       n_embd_k_gqa,
    size_t         row_bytes,
    uint32_t       kv_head_idx,
    uint32_t       head_calib_idx,
    const uint32_t * cell_indices_dev,
    const int32_t  * positions_dev,
    uint32_t       n_cells,
    int64_t        round_start,
    int            agg_mode,
    float        * scores_dev,
    void * stream);

GGML_BACKEND_API void triattention_gpu_scores_to_host(
    float * scores_host,
    const float * scores_dev,
    uint32_t n_cells,
    void * stream);

GGML_BACKEND_API void triattention_gpu_upload_cells(
    uint32_t     ** cell_indices_dev,
    int32_t      ** positions_dev,
    const uint32_t * cell_indices_host,
    const int32_t  * positions_host,
    uint32_t        n_cells,
    void * stream);

GGML_BACKEND_API float * triattention_gpu_alloc_scores(uint32_t n_cells, void * stream);
GGML_BACKEND_API void    triattention_gpu_free_dev(void * ptr);
GGML_BACKEND_API void    triattention_gpu_free(triattention_gpu_state * state);

// Number of times a graph-level fusion fired on this backend since it was created, or -1 for an
// unknown name. Names: "elem_chain", "fused_add", "fused_mul", "q8_cache_hits". Also reachable
// through
// ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_fusion_count").
GGML_BACKEND_API int64_t ggml_backend_cuda_fusion_count(ggml_backend_t backend, const char * name);
#ifdef  __cplusplus
}
#endif
