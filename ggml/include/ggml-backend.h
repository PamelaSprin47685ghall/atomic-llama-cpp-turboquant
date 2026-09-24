#pragma once

#include "ggml.h"
#include "ggml-alloc.h"

#ifdef GGML_BACKEND_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef GGML_BACKEND_BUILD
#            define GGML_BACKEND_API __declspec(dllexport) extern
#        else
#            define GGML_BACKEND_API __declspec(dllimport) extern
#        endif
#    else
#        define GGML_BACKEND_API __attribute__ ((visibility ("default"))) extern
#    endif
#else
#    define GGML_BACKEND_API extern
#endif

#ifdef  __cplusplus
extern "C" {
#endif

    typedef struct ggml_backend_buffer_type * ggml_backend_buffer_type_t;
    typedef struct ggml_backend_buffer * ggml_backend_buffer_t;
    typedef struct ggml_backend_event * ggml_backend_event_t;
    typedef struct ggml_backend * ggml_backend_t;
    typedef void * ggml_backend_graph_plan_t;
    typedef struct ggml_backend_reg * ggml_backend_reg_t;
    typedef struct ggml_backend_device * ggml_backend_dev_t;
    struct ggml_predefined_frame;


    //
    // Backend buffer type
    //

    GGML_API const char *          ggml_backend_buft_name          (ggml_backend_buffer_type_t buft);
    GGML_API ggml_backend_buffer_t ggml_backend_buft_alloc_buffer  (ggml_backend_buffer_type_t buft, size_t size);
    GGML_API size_t                ggml_backend_buft_get_alignment (ggml_backend_buffer_type_t buft);
    GGML_API size_t                ggml_backend_buft_get_max_size  (ggml_backend_buffer_type_t buft);
    GGML_API size_t                ggml_backend_buft_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor);
    GGML_API bool                  ggml_backend_buft_is_host       (ggml_backend_buffer_type_t buft);
    GGML_API ggml_backend_dev_t    ggml_backend_buft_get_device    (ggml_backend_buffer_type_t buft);

    //
    // Backend buffer
    //

    enum ggml_backend_buffer_usage {
        GGML_BACKEND_BUFFER_USAGE_ANY = 0,
        GGML_BACKEND_BUFFER_USAGE_WEIGHTS = 1,
        GGML_BACKEND_BUFFER_USAGE_COMPUTE = 2,
    };

    GGML_API const char *                   ggml_backend_buffer_name          (ggml_backend_buffer_t buffer);
    GGML_API void                           ggml_backend_buffer_free          (ggml_backend_buffer_t buffer);
    GGML_API void *                         ggml_backend_buffer_get_base      (ggml_backend_buffer_t buffer);
    GGML_API size_t                         ggml_backend_buffer_get_size      (ggml_backend_buffer_t buffer);
    GGML_API enum ggml_status               ggml_backend_buffer_init_tensor   (ggml_backend_buffer_t buffer, struct ggml_tensor * tensor);
    GGML_API size_t                         ggml_backend_buffer_get_alignment (ggml_backend_buffer_t buffer);
    GGML_API size_t                         ggml_backend_buffer_get_max_size  (ggml_backend_buffer_t buffer);
    GGML_API size_t                         ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor);
    GGML_API void                           ggml_backend_buffer_clear         (ggml_backend_buffer_t buffer, uint8_t value);
    GGML_API bool                           ggml_backend_buffer_is_host       (ggml_backend_buffer_t buffer);
    GGML_API void                           ggml_backend_buffer_set_usage     (ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage);
    GGML_API enum ggml_backend_buffer_usage ggml_backend_buffer_get_usage     (ggml_backend_buffer_t buffer);
    GGML_API ggml_backend_buffer_type_t     ggml_backend_buffer_get_type      (ggml_backend_buffer_t buffer);
    GGML_API void                           ggml_backend_buffer_reset         (ggml_backend_buffer_t buffer);

    // tensor copy between different backends
    GGML_API void ggml_backend_tensor_copy(const struct ggml_tensor * src, struct ggml_tensor * dst);

    // In-place data movement for backend-native compaction. Regions are
    // applied in array order with memmove semantics. Each region may describe
    // one contiguous copy or a strided series of equal-sized copies. The
    // offsets and strides are relative to tensor->data. Returns false without
    // modifying data when the backing buffer/backend cannot implement the
    // operation natively.
    struct ggml_backend_tensor_memmove_region {
        struct ggml_tensor * tensor;
        size_t src_offset;
        size_t dst_offset;
        size_t size;
        size_t n_copies;
        size_t src_stride;
        size_t dst_stride;
    };

    GGML_API bool ggml_backend_tensor_memmove_regions(
        const struct ggml_backend_tensor_memmove_region * regions,
        size_t n_regions);

    // Preflight the same operation without changing tensor data. Backends may
    // reserve/reuse internal scratch during this call so a later execution is
    // not surprised by an allocation failure after another buffer was moved.
    GGML_API bool ggml_backend_tensor_memmove_regions_supported(
        const struct ggml_backend_tensor_memmove_region * regions,
        size_t n_regions);

    //
    // Backend (stream)
    //

    GGML_API ggml_guid_t  ggml_backend_guid(ggml_backend_t backend);
    GGML_API const char * ggml_backend_name(ggml_backend_t backend);
    GGML_API void         ggml_backend_free(ggml_backend_t backend);

    GGML_API ggml_backend_buffer_type_t ggml_backend_get_default_buffer_type(ggml_backend_t backend);
    GGML_API ggml_backend_buffer_t      ggml_backend_alloc_buffer(ggml_backend_t backend, size_t size);
    GGML_API size_t                     ggml_backend_get_alignment(ggml_backend_t backend);
    GGML_API size_t                     ggml_backend_get_max_size(ggml_backend_t backend);

    GGML_API void ggml_backend_tensor_set_async   (ggml_backend_t backend,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
    GGML_API void ggml_backend_tensor_get_async   (ggml_backend_t backend, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size);
    GGML_API void ggml_backend_tensor_set_2d_async(ggml_backend_t backend,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
    GGML_API void ggml_backend_tensor_get_2d_async(ggml_backend_t backend, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
    GGML_API bool ggml_backend_tensor_set_snapshot_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size, bool dry_run);

    // "offset" refers to the offset in tensor->data for setting/getting data
    GGML_API void ggml_backend_tensor_set   (      struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
    GGML_API void ggml_backend_tensor_get   (const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size);
    GGML_API void ggml_backend_tensor_set_2d(      struct ggml_tensor * tensor, const void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
    GGML_API void ggml_backend_tensor_get_2d(const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
    GGML_API void ggml_backend_tensor_memset(      struct ggml_tensor * tensor,     uint8_t value, size_t offset, size_t size);

    GGML_API void ggml_backend_synchronize(ggml_backend_t backend);

    GGML_API ggml_backend_graph_plan_t ggml_backend_graph_plan_create(ggml_backend_t backend, struct ggml_cgraph * cgraph);
    GGML_API void                      ggml_backend_graph_plan_free  (ggml_backend_t backend, ggml_backend_graph_plan_t plan);

    GGML_API enum ggml_status ggml_backend_graph_plan_compute (ggml_backend_t backend, ggml_backend_graph_plan_t plan);
    GGML_API enum ggml_status ggml_backend_graph_compute      (ggml_backend_t backend, struct ggml_cgraph * cgraph);
    GGML_API enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, struct ggml_cgraph * cgraph);

    // NOTE: will be removed, use device version instead
    GGML_API bool ggml_backend_supports_op(ggml_backend_t backend, const struct ggml_tensor * op);
    GGML_API bool ggml_backend_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft);
    GGML_API bool ggml_backend_offload_op(ggml_backend_t backend, const struct ggml_tensor * op);

    // asynchronous copy
    // the copy is performed after all the currently queued operations in backend_src
    // backend_dst will wait for the copy to complete before performing other operations
    // automatic fallback to sync copy if async is not supported
    GGML_API void ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const struct ggml_tensor * src, struct ggml_tensor * dst);

    GGML_API ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t backend);

    //
    // Events
    //

    GGML_API ggml_backend_event_t ggml_backend_event_new(ggml_backend_dev_t device);
    GGML_API void                 ggml_backend_event_free(ggml_backend_event_t event);
    GGML_API void                 ggml_backend_event_record(ggml_backend_event_t event, ggml_backend_t backend);
    GGML_API void                 ggml_backend_event_synchronize(ggml_backend_event_t event);
    GGML_API void                 ggml_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event);

    //
    // Backend device
    //

    enum ggml_backend_dev_type {
        // CPU device using system memory
        GGML_BACKEND_DEVICE_TYPE_CPU,
        // GPU device using dedicated memory
        GGML_BACKEND_DEVICE_TYPE_GPU,
        // integrated GPU device using host memory
        GGML_BACKEND_DEVICE_TYPE_IGPU,
        // accelerator devices intended to be used together with the CPU backend (e.g. BLAS or AMX)
        GGML_BACKEND_DEVICE_TYPE_ACCEL,
        // "meta" device wrapping multiple other devices for tensor parallelism
        GGML_BACKEND_DEVICE_TYPE_META,
    };

    // functionality supported by the device
    struct ggml_backend_dev_caps {
        // asynchronous operations
        bool async;
        // pinned host buffer
        bool host_buffer;
        // creating buffers from host ptr
        bool buffer_from_host_ptr;
        // event synchronization
        bool events;
    };

    // all the device properties
    struct ggml_backend_dev_props {
        // device name
        const char * name;
        // device description
        const char * description;
        // device free memory in bytes
        size_t memory_free;
        // device total memory in bytes
        size_t memory_total;
        // device type
        enum ggml_backend_dev_type type;
        // device id
        //   for PCI devices, this should be the lower-case PCI bus id formatted as "domain:bus:device.function" (e.g. "0000:c1:00.0")
        //   if the id is unknown, this should be NULL
        const char * device_id;
        // device capabilities
        struct ggml_backend_dev_caps caps;
    };

    GGML_API const char *                  ggml_backend_dev_name(ggml_backend_dev_t device);
    GGML_API const char *                  ggml_backend_dev_description(ggml_backend_dev_t device);
    GGML_API void                          ggml_backend_dev_memory(ggml_backend_dev_t device, size_t * free, size_t * total);
    GGML_API enum ggml_backend_dev_type    ggml_backend_dev_type(ggml_backend_dev_t device);
    GGML_API void                          ggml_backend_dev_get_props(ggml_backend_dev_t device, struct ggml_backend_dev_props * props);
    GGML_API ggml_backend_reg_t            ggml_backend_dev_backend_reg(ggml_backend_dev_t device);
    GGML_API ggml_backend_t                ggml_backend_dev_init(ggml_backend_dev_t device, const char * params);
    GGML_API ggml_backend_buffer_type_t    ggml_backend_dev_buffer_type(ggml_backend_dev_t device);
    GGML_API ggml_backend_buffer_type_t    ggml_backend_dev_host_buffer_type(ggml_backend_dev_t device);
    GGML_API ggml_backend_buffer_t         ggml_backend_dev_buffer_from_host_ptr(ggml_backend_dev_t device, void * ptr, size_t size, size_t max_tensor_size);

    GGML_API bool                          ggml_backend_dev_supports_op(ggml_backend_dev_t device, const struct ggml_tensor * op);
    GGML_API bool                          ggml_backend_dev_supports_buft(ggml_backend_dev_t device, ggml_backend_buffer_type_t buft);
    GGML_API bool                          ggml_backend_dev_offload_op(ggml_backend_dev_t device, const struct ggml_tensor * op);

    //
    // Backend (reg)
    //

    GGML_API const char *       ggml_backend_reg_name(ggml_backend_reg_t reg);
    GGML_API size_t             ggml_backend_reg_dev_count(ggml_backend_reg_t reg);
    GGML_API ggml_backend_dev_t ggml_backend_reg_dev_get(ggml_backend_reg_t reg, size_t index);
    GGML_API void *             ggml_backend_reg_get_proc_address(ggml_backend_reg_t reg, const char * name);

    // Common functions that may be obtained using ggml_backend_reg_get_proc_address

    // Context management and operations for faster communication between backends, used for tensor parallelism (meta backend)
    typedef void * (*ggml_backend_comm_init_t)(ggml_backend_t * backends, size_t n_backends);
    typedef void   (*ggml_backend_comm_free_t)(void * comm_ctx);
    // Optional shutdown extension. false means that native work did not drain;
    // the caller MUST retain the communicator and its child backends.
    typedef bool   (*ggml_backend_comm_free_safe_t)(void * comm_ctx);
    typedef bool   (*ggml_backend_comm_allreduce_tensor_t)(void * comm_ctx, struct ggml_tensor ** tensors);

    // Split buffer type for tensor parallelism (old)
    typedef ggml_backend_buffer_type_t   (*ggml_backend_split_buffer_type_t)(int main_device, const float * tensor_split);
    // Set the number of threads for the backend
    typedef void                         (*ggml_backend_set_n_threads_t)(ggml_backend_t backend, int n_threads);
    // Get additional buffer types provided by the device (returns a NULL-terminated array)
    typedef ggml_backend_buffer_type_t * (*ggml_backend_dev_get_extra_bufts_t)(ggml_backend_dev_t device);
    // Set the abort callback for the backend
    typedef void                         (*ggml_backend_set_abort_callback_t)(ggml_backend_t backend, ggml_abort_callback abort_callback, void * abort_callback_data);
    // Get a list of feature flags supported by the backend (returns a NULL-terminated array)
    struct ggml_backend_feature {
        const char * name;
        const char * value;
    };
    typedef struct ggml_backend_feature * (*ggml_backend_get_features_t)(ggml_backend_reg_t reg);

    //
    // Backend registry
    //

    GGML_API void ggml_backend_register(ggml_backend_reg_t reg);

    GGML_API void ggml_backend_device_register(ggml_backend_dev_t device);

    // Backend (reg) enumeration
    GGML_API size_t             ggml_backend_reg_count(void);
    GGML_API ggml_backend_reg_t ggml_backend_reg_get(size_t index);
    GGML_API ggml_backend_reg_t ggml_backend_reg_by_name(const char * name);

    // Device enumeration
    GGML_API size_t             ggml_backend_dev_count(void);
    GGML_API ggml_backend_dev_t ggml_backend_dev_get(size_t index);
    GGML_API ggml_backend_dev_t ggml_backend_dev_by_name(const char * name);
    GGML_API ggml_backend_dev_t ggml_backend_dev_by_type(enum ggml_backend_dev_type type);

    // Direct backend (stream) initialization
    // = ggml_backend_dev_init(ggml_backend_dev_by_name(name), params)
    GGML_API ggml_backend_t ggml_backend_init_by_name(const char * name, const char * params);
    // = ggml_backend_dev_init(ggml_backend_dev_by_type(type), params)
    GGML_API ggml_backend_t ggml_backend_init_by_type(enum ggml_backend_dev_type type, const char * params);
    // = ggml_backend_dev_init(ggml_backend_dev_by_type(GPU) OR ggml_backend_dev_by_type(CPU), NULL)
    GGML_API ggml_backend_t ggml_backend_init_best(void);

    // Load a backend from a dynamic library and register it
    GGML_API ggml_backend_reg_t ggml_backend_load(const char * path);
    // Unload a backend if loaded dynamically and unregister it
    GGML_API void               ggml_backend_unload(ggml_backend_reg_t reg);
    // Load all known backends from dynamic libraries
    GGML_API void               ggml_backend_load_all(void);
    GGML_API void               ggml_backend_load_all_from_path(const char * dir_path);

    //
    // Backend scheduler
    //

    // The backend scheduler allows for multiple backend devices to be used together
    // Handles compute buffer allocation, assignment of tensors to backends, and copying of tensors between backends
    // The backends are selected based on:
    // - the backend that supports the operation
    // - the location of the pre-allocated tensors (e.g. the weights)
    /*
      Example usage:

        // operations that use tensors allocated in a buffer with USAGE_WEIGHTS will be assigned
        // preferably to run on the same backend as the buffer
        ggml_backend_buffer_set_usage(buf_weights, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        sched = ggml_backend_sched_new({backend_gpu, backend_gpu2, backend_cpu}, NULL, num_backends, GGML_DEFAULT_GRAPH_SIZE, false, true);

        // initialize buffers from a max size graph (optional)
        reserve_graph = build_graph(sched, max_batch_size);

        // manually assign nodes to a backend (optional, should not be needed in most cases)
        struct ggml_tensor * node = ggml_mul_mat(ctx, ...);
        ggml_backend_sched_set_tensor_backend(sched, node, backend_gpu);

        ggml_backend_sched_reserve(sched, reserve_graph);

        // compute
        graph = build_graph(sched); // the graph and its tensors are single-use in terms of allocation, multi-use in terms of computation
        for (int i = 0; i < 10; ++i) {
            ggml_backend_sched_graph_compute(sched, graph); // on the first iteration the graph is allocated automatically
        }

        // if there are graph inputs:
        graph = build_graph(sched); // get a new graph that is not allocated (the metadata for the old graph is freed once ggml_free is called)
        ggml_backend_sched_reset(sched); // clear the allocation of the previous graph
        ggml_backend_sched_alloc_graph(sched, graph); // explicitly allocate the new graph but do not execute it
        ggml_backend_tensor_set(input_tensor, ...); // copy data to the newly allocated graph tensors
        ggml_backend_sched_graph_compute(sched, graph); // execute the graph

        // as an alternative to the above it is also possible to assign the inputs to a dedicated context and
        // allocate them statically via ggml_backend_alloc_ctx_tensors
    }
    */

    typedef struct ggml_backend_sched * ggml_backend_sched_t;
    typedef struct ggml_backend_sched_compute_pool * ggml_backend_sched_compute_pool_t;

    // Evaluation callback for each node in the graph (set with ggml_backend_sched_set_eval_callback)
    // when ask == true, the scheduler wants to know if the user wants to observe this node
    // this allows the scheduler to batch nodes together in order to evaluate them in a single call
    //
    // when ask == false, the scheduler is passing the node tensor to the user for observation
    // if the user returns false, the scheduler will cancel the graph compute
    //
    typedef bool (*ggml_backend_sched_eval_callback)(struct ggml_tensor * t, bool ask, void * user_data);

    // Initialize a backend scheduler, backends with low index are given priority over backends with high index
    GGML_API ggml_backend_sched_compute_pool_t ggml_backend_sched_compute_pool_new(void);
    GGML_API ggml_backend_sched_compute_pool_t ggml_backend_sched_compute_pool_ref(ggml_backend_sched_compute_pool_t pool);
    // Keep one largest shared temporary arena for all serial execution phases.
    // Does not remove ownership synchronization or allow concurrent alias use.
    GGML_API void ggml_backend_sched_compute_pool_set_retain_capacity(ggml_backend_sched_compute_pool_t pool, bool retain);
    GGML_API bool ggml_backend_sched_compute_pool_get_retain_capacity(ggml_backend_sched_compute_pool_t pool);
    GGML_API void ggml_backend_sched_compute_pool_set_capacity_sealed(ggml_backend_sched_compute_pool_t pool, bool sealed);
    GGML_API bool ggml_backend_sched_compute_pool_get_capacity_sealed(ggml_backend_sched_compute_pool_t pool);
    GGML_API void                              ggml_backend_sched_compute_pool_free(ggml_backend_sched_compute_pool_t pool);

    GGML_API ggml_backend_sched_t ggml_backend_sched_new(ggml_backend_t * backends, ggml_backend_buffer_type_t * bufts, int n_backends, size_t graph_size, bool parallel, bool op_offload);
    GGML_API ggml_backend_sched_t ggml_backend_sched_new_shared(ggml_backend_t * backends, ggml_backend_buffer_type_t * bufts, int n_backends, size_t graph_size, bool parallel, bool op_offload, ggml_backend_sched_compute_pool_t compute_pool);
    GGML_API void                 ggml_backend_sched_free(ggml_backend_sched_t sched);
    GGML_API void                 ggml_backend_sched_compute_begin(ggml_backend_sched_t sched);
    GGML_API void                 ggml_backend_sched_compute_end(ggml_backend_sched_t sched);

    // Initialize backend buffers from a measure graph
    GGML_API void                 ggml_backend_sched_reserve_size(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph, size_t * sizes);
    GGML_API bool                 ggml_backend_sched_reserve(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph); // returns success

    GGML_API int                  ggml_backend_sched_get_n_backends(ggml_backend_sched_t sched);
    GGML_API ggml_backend_t       ggml_backend_sched_get_backend(ggml_backend_sched_t sched, int i);

    // Get the number of splits of the last graph
    GGML_API int                  ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched);
    GGML_API int                  ggml_backend_sched_get_n_copies(ggml_backend_sched_t sched);
    GGML_API bool                 ggml_backend_sched_is_allocated(ggml_backend_sched_t sched);

    GGML_API ggml_backend_buffer_type_t ggml_backend_sched_get_buffer_type(ggml_backend_sched_t sched, ggml_backend_t backend);
    GGML_API size_t                     ggml_backend_sched_get_buffer_size(ggml_backend_sched_t sched, ggml_backend_t backend);

    GGML_API void                 ggml_backend_sched_set_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node, ggml_backend_t backend);
    GGML_API ggml_backend_t       ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node);

    // Split graph without allocating it
    GGML_API void                 ggml_backend_sched_split_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph);

    // Allocate and compute graph on the backend scheduler
    GGML_API bool                 ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph); // returns success
    GGML_API enum ggml_status     ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, struct ggml_cgraph * graph);
    GGML_API enum ggml_status     ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, struct ggml_cgraph * graph);
    GGML_API void                 ggml_backend_sched_synchronize(ggml_backend_sched_t sched);

    // Reset all assignments and allocators - must be called before changing the node backends or allocating a new graph.
    // This in effect deallocates all tensors that were previously allocated and leaves them with dangling pointers.
    // The correct way to use this API is to discard the deallocated tensors and create new ones.
    GGML_API void                 ggml_backend_sched_reset(ggml_backend_sched_t sched);

    // Drop the compute buffers, keeping the scheduler: the next reserve sizes them for its own
    // graph. Call only after every reusable graph has been invalidated.
    GGML_API void                 ggml_backend_sched_release_buffers(ggml_backend_sched_t sched);


    // Set a callback to be called for each resulting node during graph compute
    GGML_API void                 ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, void * user_data);

    //
    // Meta backend
    //

#define GGML_BACKEND_META_MAX_DEVICES 16

    enum ggml_backend_meta_split_axis {
        // tensor split by tensor dimensions:
        GGML_BACKEND_SPLIT_AXIS_0 = 0,
        GGML_BACKEND_SPLIT_AXIS_1 = 1,
        GGML_BACKEND_SPLIT_AXIS_2 = 2,
        GGML_BACKEND_SPLIT_AXIS_3 = 3,

        GGML_BACKEND_SPLIT_AXIS_MIRRORED = 10, // all values on all backends
        GGML_BACKEND_SPLIT_AXIS_PARTIAL  = 11, // each backend has a partial sum

        // for internal bookkeeping only:
        GGML_BACKEND_SPLIT_AXIS_NONE    = 98,
        GGML_BACKEND_SPLIT_AXIS_UNKNOWN = 99,
    };

    // TP5 private local-node arithmetic head map (opt-in GGML_TP5_GDN_HEADMAP=1 /
    // GGML_TP5_QSA_HEADMAP=1). Only META-produced rank-local
    // GGML_OP_GATED_DELTA_NET / GGML_OP_FLASH_ATTN_EXT nodes carry this; global
    // graph nodes keep ordinary op_params. ABI (op_params int32 slots):
    //   slot  8: magic 0x54503548 ('TP5H')
    //   slot  9: packed 3-bit local head indices, entry h at bits (3*h .. 3*h+2)
    //            (LSB-first, up to 10 entries)
    //   slot 10: entry count (1..10)
    // For GATED_DELTA_NET the entries are per-local-V-head local QK head indices
    // (qk[i] < k->ne[1]); for FLASH_ATTN_EXT they are per-local-Q-head local KV
    // head indices (kv[h] < k->ne[2]). Slots 8..10 are unused by ordinary nodes
    // of both ops. A node without the magic keeps native uniform behavior.
    // Absent (no magic) and invalid (magic but bad count/index) are distinct:
    // absent = native uniform mapping; invalid = fail closed, never a silent
    // fallback to uniform.
#define GGML_TP5_HEADMAP_MAGIC 0x54503548
#define GGML_TP5_HEADMAP_MAX_ENTRIES 10

    // Pack count local head indices (each in [0,7]) into slot 9 and write
    // magic/count. count must be in [1, GGML_TP5_HEADMAP_MAX_ENTRIES];
    // indices must be < 8. Aborts on violation.
    GGML_API void ggml_tp5_headmap_set(struct ggml_tensor * op, const uint8_t * local_heads, int32_t count);

    // Returns the entry count (>= 1) and fills local_heads[0..count-1] when the
    // magic is present and the encoding is valid; returns 0 when the magic is
    // absent (native node); returns -1 when the magic is present but the
    // encoding is invalid (count out of range or an index >= 8). Consumers must
    // treat -1 as an execution error, never fall back to uniform mapping.
    GGML_API int32_t ggml_tp5_headmap_get(const struct ggml_tensor * op, uint8_t * local_heads);
    GGML_API const char * ggml_backend_meta_split_axis_name(enum ggml_backend_meta_split_axis split_axis);

    struct ggml_backend_meta_split_state {
        enum ggml_backend_meta_split_axis axis;

        // for tensors with axis >= 0 && axis < GGML_MAX_DIMS:
        //   - each device has a slice of the tensor along the split axis
        //   - most tensors have n_segments == 1 and a contiguous slice of the tensor data
        //   - some tensors have an inhomogenenous data layout along the split axis,
        //     those tensors are divided into segments which are each individually split across devices
        //   - ne has one entry per segment and device and that segment repeats nr times,
        //     in total when accounting for repetitions the segments add up to ggml_tensor::ne for that axis,
        //     the outer/inner loops are over segments/devices like [seg0_dev0_r0, seg0_dev1_r0, seg0_dev0_r1, seg0_dev1_r1, seg1_dev0_r0, seg1_dev1_r0],
        //   - for example, a transformer may have a fused QKV matrix rather than 3 matrices, those would be 3 separate segments
        //     that each need to be split individually across devices so that each device gets a slice of Q, K, and V,
        //     the Q matrix can be larger than the K and V matrices so this can either be expressed as 3 segments or as 2 segments
        //     where the segment for K/V repeats twice
        int64_t  ne[16*GGML_BACKEND_META_MAX_DEVICES];
        uint32_t nr[16];
        uint32_t n_segments;

        // For tensors with indexed replicas (indexed_replica = true, n_segments = 1, nr[0] = 1):
        // Explicit logical starting element along the split axis for each device.
        bool     indexed_replica;
        int64_t  replica_start[GGML_BACKEND_META_MAX_DEVICES];

        // TP5 balanced head mapping (opt-in): bounded indexed spans mapping
        // noncontiguous original-element ranges -> concatenated rank-local
        // storage (GGML_TP5_GDN_HEADMAP=1 / GGML_TP5_QSA_HEADMAP=1).
        //
        // When mapped_span = true (implies n_segments == 1 && nr[0] == 1 and
        // axis is 0 or 1): each device j owns span_count[j] spans; span s of
        // device j covers original logical elements
        //   [span_start[idx], span_start[idx] + span_len[idx])
        // along the split axis, where idx = span_count[0] + ... + span_count[j-1] + s
        // (flat prefix-sum layout, spans stored in device order). The device's
        // local storage is the concatenation of its spans in span order, so
        // ne[j] == sum of span_len over device j's spans.
        //
        // Physical transport (set/get/async copies) only happens on the
        // flattened storage root (axis 0/1). Derived tensors may reshape/view
        // the storage onto other axes (e.g. recurrent state [128,128,V,nseq]
        // reshaped from the flat state cache lands on axis 2): the mapping is
        // carried through with coordinates rescaled by the flattened inner
        // block (e.g. 128*128 = 16384 elements per head), preserving the
        // per-head identity map. Such derived split states keep mapped_span
        // true with the rescaled span geometry; they never abort and never
        // silently drop to a dense/uniform fallback.
        //
        // Overlapping logical ranges across devices are allowed only as
        // replicas of identical original data (e.g. shared Q/K head pairs):
        // set_tensor writes every owning device, get_tensor reads a
        // deterministic single owner (lowest device index covering the range).
        // No silent partial fallback: unsupported combinations abort.
#define GGML_BACKEND_META_MAX_SPANS_PER_DEVICE 16
#define GGML_BACKEND_META_MAX_SPANS (GGML_BACKEND_META_MAX_SPANS_PER_DEVICE * GGML_BACKEND_META_MAX_DEVICES)
        bool     mapped_span;
        // span_count[j] <= GGML_BACKEND_META_MAX_SPANS_PER_DEVICE; spans of
        // device j live at flat indices [offset(j), offset(j) + span_count[j])
        // with offset(j) = span_count[0] + ... + span_count[j-1].
        int32_t  span_count[GGML_BACKEND_META_MAX_DEVICES];
        int64_t  span_start[GGML_BACKEND_META_MAX_SPANS];
        int64_t  span_len[GGML_BACKEND_META_MAX_SPANS];
    };

    // function to assign split states for statically allocated tensors, compute tensor split states will be assigned to be compatible:
    typedef struct ggml_backend_meta_split_state(*ggml_backend_meta_get_split_state_t)(const struct ggml_tensor * tensor, void * userdata);

    // create a new meta device from "simple" devices, meta buffer type/buffer/backend is then derived from this:
    // TODO: this looks a bit strange - a backend API creates a device. I think we should try
    //       express this as a backend registry functionality instead
    GGML_API ggml_backend_dev_t ggml_backend_meta_device(
        ggml_backend_dev_t * devs, size_t n_devs, ggml_backend_meta_get_split_state_t get_split_state, void * get_split_state_ud);


    // TP5 local-node head map (opt-in GGML_TP5_QSA_HEADMAP=1 / GGML_TP5_GDN_HEADMAP=1):
    // optional per-rank local arithmetic head map stamped on the rank-local clones
    // the meta backend creates. The meta backend calls this hook when it clones a
    // GGML_OP_FLASH_ATTN_EXT or GGML_OP_GATED_DELTA_NET node; rank is the
    // simple-backend index. The hook stamps the private op_params head-map ABI
    // (ggml_tp5_headmap_set) on the rank-local node only; the global graph node
    // is never touched. Return value: whether a map was stamped. A NULL hook
    // (default) keeps native uniform behavior.
    //
    // The hook is per meta device (stored in its device context), set via
    // ggml_backend_meta_set_local_node_hook BEFORE creating buffers/backends
    // from the device; it reuses the device's get_split_state userdata, so a
    // second model's device never sees the first model's hook and a freed
    // model leaves no dangling global state. One hook serves both GDN and FA;
    // it dispatches on node->op.
    typedef bool (*ggml_backend_meta_local_node_hook_t)(struct ggml_tensor * local_node, size_t rank, void * userdata);

    GGML_API void ggml_backend_meta_set_local_node_hook(ggml_backend_dev_t meta_dev,
                                                        ggml_backend_meta_local_node_hook_t hook);
    GGML_API struct ggml_tensor * ggml_backend_meta_buffer_simple_tensor(const struct ggml_tensor * tensor, size_t index);

    // W1: rank-local output readback support (TP5 five-GPU logits readback).
    //
    // Number of simple (per-rank) devices wrapped by a meta device; 0 for any
    // non-meta device. Doubles as the "is meta" predicate at the llama layer.
    GGML_API size_t ggml_backend_meta_dev_n_simple_devs(ggml_backend_dev_t dev);

    // The index-th simple device of a meta device. Valid only when
    // ggml_backend_meta_dev_n_simple_devs(dev) > index.
    GGML_API ggml_backend_dev_t ggml_backend_meta_dev_get_simple_dev(ggml_backend_dev_t dev, size_t index);

    // Invoke the meta device's own split-state callback for `tensor`. This is
    // the only sanctioned channel for callers that need the per-rank split
    // plan (e.g. sizing rank-local output buffers): the split policy stays
    // owned by whoever created the meta device.
    GGML_API struct ggml_backend_meta_split_state ggml_backend_meta_dev_compute_split_state(ggml_backend_dev_t dev, const struct ggml_tensor * tensor);

    // Query the per-rank chunk sizes (bytes of one row-plane per rank) that
    // ggml_backend_tensor_get_async reads back for `tensor`. Fills
    // chunk_bytes[0..*n_chunks-1] and sets *n_chunks to the number of simple
    // backends. Returns false for non-meta backends, or when the tensor's
    // split is not the simple contiguous-chunks form (replica, mapped-span or
    // mirrored), in which case the caller must use the single-destination
    // path.
    GGML_API bool ggml_backend_meta_tensor_rank_chunks(ggml_backend_t backend, const struct ggml_tensor * tensor, size_t * chunk_bytes, size_t * n_chunks);

    // Per-rank destination variant of ggml_backend_tensor_get_2d_async for
    // meta backends. Matches the single-destination call except that each
    // rank j's slice is written at dsts[j] with rank-local row stride
    // (chunk_j), instead of being gathered into the global strided layout at
    // `data`. dsts[j] must be non-null for every rank with a non-empty chunk.
    // When dsts is nullptr, or the backend is not meta, or the tensor's split
    // is not the simple contiguous-chunks form, this is exactly
    // ggml_backend_tensor_get_async(backend, tensor, data, offset, size).
    GGML_API void ggml_backend_meta_tensor_get_2d_async_per_rank(ggml_backend_t backend, const struct ggml_tensor * tensor, void * const * dsts, void * data, size_t offset, size_t size);

    // Predefined-row execution contract for TP-style meta backends. The graph
    // owns capacity-shaped tensors; active_rows is useful work for this
    // invocation. This only supplies host-side execution metadata. Backends
    // must still lower every stateful/operator dispatch before capacity_rows >
    // active_rows can execute safely. Values are consumed synchronously while
    // building/submitting the graph and never replace native retirement.
    GGML_API bool ggml_backend_meta_set_predefined_rows(
        ggml_backend_t backend, uint32_t active_rows, uint32_t capacity_rows);
    // Full execution-frame variant used by the single maximum-capacity TP5
    // definition. capacity_rows is immutable for the definition; all useful
    // work (phase, tokens, outputs, context, draft/catch-up state) comes from
    // frame. The frame is copied synchronously into the meta/backend owner;
    // the caller may reuse its storage immediately after this call.
    GGML_API bool ggml_backend_meta_set_predefined_frame(
        ggml_backend_t backend, const struct ggml_predefined_frame * frame,
        uint32_t capacity_rows, uint32_t capacity_outputs);

    //
    // Utils
    //

    struct ggml_backend_graph_copy {
        ggml_backend_buffer_t buffer;
        struct ggml_context * ctx_allocated;
        struct ggml_context * ctx_unallocated;
        struct ggml_cgraph * graph;
    };

    // Copy a graph to a different backend
    GGML_API struct ggml_backend_graph_copy ggml_backend_graph_copy(ggml_backend_t backend, struct ggml_cgraph * graph);
    GGML_API void                           ggml_backend_graph_copy_free(struct ggml_backend_graph_copy copy);

    typedef bool (*ggml_backend_eval_callback)(int node_index, struct ggml_tensor * t1, struct ggml_tensor * t2, void * user_data);

    // Compare the output of two backends
    GGML_API bool ggml_backend_compare_graph_backend(ggml_backend_t backend1, ggml_backend_t backend2, struct ggml_cgraph * graph, ggml_backend_eval_callback callback, void * user_data, struct ggml_tensor const * const * test_nodes, size_t num_test_nodes);

    // Tensor initialization
    GGML_API enum ggml_status ggml_backend_tensor_alloc(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, void * addr);
    GGML_API enum ggml_status ggml_backend_view_init(struct ggml_tensor * tensor);

    // CPU buffer types are always available
    GGML_API ggml_backend_buffer_t      ggml_backend_cpu_buffer_from_ptr(void * ptr, size_t size);
    GGML_API ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type(void);

#ifdef  __cplusplus
}
#endif
