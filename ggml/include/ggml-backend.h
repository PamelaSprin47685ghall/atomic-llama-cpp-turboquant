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

    //
    // Backend (stream)
    //

    GGML_API ggml_guid_t  ggml_backend_guid(ggml_backend_t backend);
    GGML_API const char * ggml_backend_name(ggml_backend_t backend);
    GGML_API void         ggml_backend_free(ggml_backend_t backend);
    // Optional blocking owner protocol: close new submissions and wait for
    // the backend's begin/pending counters. It does not wait for receipts or
    // command buffers; free performs those checks. Backends without the
    // "ggml_backend_submission_quiesce" proc return true (no extra contract).
    GGML_API bool         ggml_backend_quiesce(ggml_backend_t backend);

    GGML_API ggml_backend_buffer_type_t ggml_backend_get_default_buffer_type(ggml_backend_t backend);
    GGML_API ggml_backend_buffer_t      ggml_backend_alloc_buffer(ggml_backend_t backend, size_t size);
    GGML_API size_t                     ggml_backend_get_alignment(ggml_backend_t backend);
    GGML_API size_t                     ggml_backend_get_max_size(ggml_backend_t backend);

    GGML_API void ggml_backend_tensor_set_async   (ggml_backend_t backend,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
    GGML_API void ggml_backend_tensor_get_async   (ggml_backend_t backend, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size);
    GGML_API void ggml_backend_tensor_set_2d_async(ggml_backend_t backend,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
    GGML_API void ggml_backend_tensor_get_2d_async(ggml_backend_t backend, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);

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
    // Returns true when the copy was accepted and its completion is ordered,
    // or when a rejected async copy was completed synchronously by this helper.
    // Returns false when an unknown submission is already outstanding on either
    // backend (no copy is attempted), or when the accepted copy cannot be
    // proven complete. The destination must not be replayed, freed or
    // overwritten, and the caller must keep the source, destination and both
    // backends alive until the process/context owner resolves it. Either
    // optional registry capability "ggml_backend_submission_error" (a recorded
    // native error) or "ggml_backend_submission_pending" (outstanding work)
    // makes the return false, including pending work with no recorded error
    // yet. The helper never clears that state: a retry keeps returning false
    // until the process/context owner resolves it.
    // Contract: submission_pending must report only accepted-but-unprovable
    // work. Ordinary ordered in-flight work must not set it; the scheduler
    // treats any true answer as a permanent fail-stop state.
    GGML_API bool ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const struct ggml_tensor * src, struct ggml_tensor * dst);

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

    // Evaluation callback for each node in the graph (set with ggml_backend_sched_set_eval_callback)
    // when ask == true, the scheduler wants to know if the user wants to observe this node
    // this allows the scheduler to batch nodes together in order to evaluate them in a single call
    //
    // when ask == false, the scheduler is passing the node tensor to the user for observation
    // if the user returns false, the scheduler will cancel the graph compute
    //
    typedef bool (*ggml_backend_sched_eval_callback)(struct ggml_tensor * t, bool ask, void * user_data);

    // Initialize a backend scheduler, backends with low index are given priority over backends with high index
    GGML_API ggml_backend_sched_t ggml_backend_sched_new(ggml_backend_t * backends, ggml_backend_buffer_type_t * bufts, int n_backends, size_t graph_size, bool parallel, bool op_offload);
    GGML_API void                 ggml_backend_sched_free(ggml_backend_sched_t sched);
    // Associate a scheduler with the context/backend that owns its backends.
    // A scheduler that free cannot release is kept in a process-level list
    // under this owner; drain_retained retries it while the owner still owns
    // the backends.
    GGML_API void                 ggml_backend_sched_set_owner(ggml_backend_sched_t sched, void * owner);
    // Retry every retained scheduler belonging to owner. The caller must still
    // own the backends. Returns the number of schedulers actually freed.
    GGML_API int                  ggml_backend_sched_drain_retained(void * owner);
    GGML_API size_t               ggml_backend_sched_retained_count(void * owner);
    // Query every backend's optional submission error/pending while the
    // scheduler's backends are still alive. Returns true when an asynchronous
    // submission is unproven. The scheduler keeps its copy slots and refuses
    // reset, reserve, alloc, release_buffers, compute, replan_graph,
    // update_split_copies, refresh_split_copy_views and set_moe_cache. The
    // owner must call this before destroying the backends.
    // ggml_backend_sched_free uses only the cached answer and never
    // dereferences the backends.
    GGML_API bool                 ggml_backend_sched_has_unproven(ggml_backend_sched_t sched);
    // Test/diagnostic only: force the cached unproven state.
    GGML_API void                 ggml_backend_sched_mark_unproven(ggml_backend_sched_t sched);
    // Test/diagnostic only: the active-slot dependency view attached to the
    // split copy of source, or NULL. The scheduler owns the pointer.
    GGML_API struct ggml_tensor * ggml_backend_sched_debug_copy_dependency(
            ggml_backend_sched_t sched, const struct ggml_tensor * source);

    // Versioned per-attempt completion receipt.  submit records the real
    // scheduler attempt and the backend attempts/resources it submitted;
    // ACCEPTED is not completion.  wait() proves the exact attempt:
    // COMPLETED_SUCCESS/COMPLETED_FAILURE are facts, UNKNOWN is not success.
    // A void synchronize is never used as a success proof here.
    enum ggml_backend_sched_receipt_state {
        GGML_BACKEND_SCHED_RECEIPT_ACCEPTED          = 0,
        GGML_BACKEND_SCHED_RECEIPT_COMPLETED_SUCCESS = 1,
        GGML_BACKEND_SCHED_RECEIPT_COMPLETED_FAILURE = 2,
        GGML_BACKEND_SCHED_RECEIPT_UNKNOWN           = 3,
    };
    #define GGML_BACKEND_SCHED_RECEIPT_API_VERSION 1
    struct ggml_backend_sched_receipt {
        uint32_t version;
        uint64_t attempt;
        int      state;
        int      error;
        bool     submitted; // true iff at least one split was handed to a backend
        bool     in_flight; // UNKNOWN due to a previous/unresolved submission
        void *   priv; // scheduler-owned attempt/resources; release it
    };
    // Returns true only when the graph was submitted and an attempt exists.
    // A false return with COMPLETED_FAILURE means no work was submitted.
    GGML_API bool ggml_backend_sched_graph_compute_receipt(
            ggml_backend_sched_t sched, struct ggml_cgraph * graph,
            struct ggml_backend_sched_receipt * receipt);
    // Blocks until the recorded attempt is proven or cannot be proven.
    // Returns the receipt state.  Callers must not treat ACCEPTED/UNKNOWN as
    // success.
    GGML_API int ggml_backend_sched_receipt_wait(
            ggml_backend_sched_t sched, struct ggml_backend_sched_receipt * receipt);
    GGML_API bool ggml_backend_sched_receipt_release(
            ggml_backend_sched_t sched, struct ggml_backend_sched_receipt * receipt);
    // Drain only backends that currently report a pending/error submission.
    // Used to retire a quarantined host undo when no per-attempt receipt exists.
    GGML_API bool ggml_backend_sched_drain_pending(
            ggml_backend_sched_t sched, int * error);

    // Initialize backend buffers from a measure graph
    GGML_API void                 ggml_backend_sched_reserve_size(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph, size_t * sizes);
    GGML_API bool                 ggml_backend_sched_reserve(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph); // returns success
    GGML_API bool                 ggml_backend_sched_reserve_min(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph, const size_t * min_sizes); // TP5 max-footprint share: reserve at a per-backend minimum size
    GGML_API void                 ggml_backend_sched_reserve_size_preserve(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph, size_t * sizes);

    GGML_API int                  ggml_backend_sched_get_n_backends(ggml_backend_sched_t sched);
    GGML_API ggml_backend_t       ggml_backend_sched_get_backend(ggml_backend_sched_t sched, int i);

    // Get the number of splits of the last graph
    GGML_API int                  ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched);
    GGML_API int                  ggml_backend_sched_get_n_copies(ggml_backend_sched_t sched);

    GGML_API ggml_backend_buffer_type_t ggml_backend_sched_get_buffer_type(ggml_backend_sched_t sched, ggml_backend_t backend);
    GGML_API size_t                     ggml_backend_sched_get_buffer_size(ggml_backend_sched_t sched, ggml_backend_t backend);

    GGML_API void                 ggml_backend_sched_set_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node, ggml_backend_t backend);
    GGML_API ggml_backend_t       ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node);

    // Split graph without allocating it
    GGML_API void                 ggml_backend_sched_split_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph);

    // Allocate and compute graph on the backend scheduler
    GGML_API bool                 ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph); // returns success
    // On an unproven submission this returns FAILED and does not synchronize.
    GGML_API enum ggml_status     ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, struct ggml_cgraph * graph);

    GGML_API enum ggml_status     ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, struct ggml_cgraph * graph);

    // The caller retains the graph and all host input sources, alive and unmodified,
    // until ggml_backend_sched_synchronize(). Eligible single-split host batches
    // with no evaluation callback may finish uploading after this function returns.
    // Failures still drain queued input transfers before returning or unwinding.
    GGML_API enum ggml_status     ggml_backend_sched_graph_compute_async_retained_inputs(ggml_backend_sched_t sched, struct ggml_cgraph * graph);
    GGML_API void                 ggml_backend_sched_synchronize(ggml_backend_sched_t sched);
    // Invalidate the backend-side executable cache of the live split graphs
    // (e.g. CUDA/HIP graph captures). The next compute re-checks the tensor
    // properties and rebuilds its capture. Used after an in-place metadata
    // patch of the live graph, which the backends cannot observe otherwise.
    GGML_API void                 ggml_backend_sched_invalidate_graph_cache(ggml_backend_sched_t sched);

    // True when the live split graph holds a cross-backend copy of this
    // tensor. Rebinding such a tensor requires tracked copy updates below.
    GGML_API bool                 ggml_backend_sched_tensor_has_split_copy(ggml_backend_sched_t sched, const struct ggml_tensor * tensor);

    // Enable before allocating a graph whose split copies may be rebound.
    // The setting survives reset and reserve. Toggling requires no allocated
    // graph; disabling releases tracking records. The caller must exclude
    // concurrent scheduler use, as for reset and allocation.
    GGML_API bool ggml_backend_sched_set_split_copy_tracking(ggml_backend_sched_t sched, bool enabled);

    struct ggml_backend_sched_tensor_layout {
        struct ggml_tensor * tensor;
        int64_t ne[GGML_MAX_DIMS];
        size_t  nb[GGML_MAX_DIMS];
    };

    // Validate all proposed source layouts against the captured allocation of
    // every scheduler copy. Tracking must have been enabled before allocation.
    // With apply=true, update copies only after all layouts pass. This does not
    // update source tensors. Synchronize and exclude concurrent scheduler use;
    // after updating source layouts and view pointers, refresh copy views.
    // Returns false while an async submission is unproven.
    GGML_API bool ggml_backend_sched_update_split_copies(
            ggml_backend_sched_t sched, const struct ggml_backend_sched_tensor_layout * layouts,
            size_t n_layouts, bool apply);
    // Refuses (and mutates nothing) while an async submission is unproven.
    GGML_API void ggml_backend_sched_refresh_split_copy_views(ggml_backend_sched_t sched);

    // Replan one live host-only split. Record the complete tensor list and
    // fixed-storage flags before the first allocation; proposed layouts refer
    // only to non-fixed entries. view_offsets has n_layouts entries.
    // Serialize scheduler access and do not change tensor metadata before
    // this call. On false, layouts, addresses, allocator and cache remain
    // unchanged. On success, graph-owned inputs must be set again before
    // compute. Non-input arena constants, split copies and an unproven
    // submission are refused.
    GGML_API bool ggml_backend_sched_replan_graph(ggml_backend_sched_t sched,
            struct ggml_tensor * const * tensors, const uint8_t * fixed, size_t count,
            const struct ggml_backend_sched_tensor_layout * layouts, const size_t * view_offsets, size_t n_layouts);
    // The requested per-tensor plan size excludes reusable arena capacity.
    GGML_API bool ggml_backend_sched_get_tensor_plan_size(ggml_backend_sched_t sched,
            const struct ggml_tensor * tensor, size_t * size);

    // True only for a copy in the active slot of the live split graph.
    GGML_API bool ggml_backend_sched_is_tensor_copy(
            ggml_backend_sched_t sched, const struct ggml_tensor * original, const struct ggml_tensor * candidate);

    // Reset all assignments and allocators - must be called before changing the node backends or allocating a new graph.
    // This in effect deallocates all tensors that were previously allocated and leaves them with dangling pointers.
    // The correct way to use this API is to discard the deallocated tensors and create new ones.
    GGML_API void                 ggml_backend_sched_reset(ggml_backend_sched_t sched);

    // Synchronize all scheduler backends, discard the current allocation plan,
    // and release the buffers owned by the graph allocator. The scheduler,
    // backend instances, evaluation callback, and MoE cache session survive.
    // All previously allocated graph tensors become invalid: discard their
    // metadata and build a fresh graph before the next allocation/computation.
    // Externally allocated tensors (for example model weights) are unaffected.
    // The caller must serialize access to this scheduler. Backend buffer types
    // that lease an external arena keep that arena's separate ownership rules.
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
    };

    // function to assign split states for statically allocated tensors, compute tensor split states will be assigned to be compatible:
    typedef struct ggml_backend_meta_split_state(*ggml_backend_meta_get_split_state_t)(const struct ggml_tensor * tensor, void * userdata);

    // create a new meta device from "simple" devices, meta buffer type/buffer/backend is then derived from this:
    // TODO: this looks a bit strange - a backend API creates a device. I think we should try
    //       express this as a backend registry functionality instead
    GGML_API ggml_backend_dev_t ggml_backend_meta_device(
        ggml_backend_dev_t * devs, size_t n_devs, ggml_backend_meta_get_split_state_t get_split_state, void * get_split_state_ud);

    // v4 staging owner credit. The provider issues a monotonic owner/generation
    // and a finite credit window; the combiner holds the token and passes the
    // same owner to the target and its MTP draft context. The token is opaque:
    // it names the issuing provider and an active owner entry. Pass it
    // unchanged to drain/release and to set_staging_owner. set_staging_owner
    // validates that the token is active and that the issuing provider is the
    // backend's child provider; it refuses to overwrite a different bound
    // owner. Acquire through the _for_reg/_for_device/_for_backend forms so the
    // provider comes from the real target, not a last-registered default.
    // Drain is independent of the context native drain; only a drained owner
    // may be released.
    // Legacy form: accepted only while exactly one provider is registered.
    GGML_API bool ggml_backend_meta_staging_owner_acquire(
        uint64_t run_id, uint32_t window_id, uint64_t * owner, uint64_t * generation);
    // Provider-bound acquisition: the token is issued by the staging provider
    // of the given registry/device/backend, never by a global last-registered
    // provider. The backend form uses the Meta backend's child provider.
    GGML_API bool ggml_backend_meta_staging_owner_acquire_for_reg(
        ggml_backend_reg_t reg, uint64_t run_id, uint32_t window_id,
        uint64_t * owner, uint64_t * generation);
    // Common child registry for a Meta device, or the device's own registry
    // for a non-Meta device. Null means mixed children/no common provider.
    GGML_API ggml_backend_reg_t ggml_backend_meta_staging_provider_reg_for_device(
        ggml_backend_dev_t dev);
    GGML_API bool ggml_backend_meta_staging_owner_acquire_for_device(
        ggml_backend_dev_t dev, uint64_t run_id, uint32_t window_id,
        uint64_t * owner, uint64_t * generation);
    GGML_API bool ggml_backend_meta_staging_owner_acquire_for_backend(
        ggml_backend_t backend, uint64_t run_id, uint32_t window_id,
        uint64_t * owner, uint64_t * generation);
    // Waits in bounded 100 ms steps for provider credit backpressure; returns
    // false only for an illegal/duplicate request or an unusable provider.
    GGML_API bool ggml_backend_meta_staging_owner_acquire_for_backend_wait(
        ggml_backend_t backend, uint64_t run_id, uint32_t window_id,
        uint64_t * owner, uint64_t * generation, uint32_t timeout_ms);
    // Atomically reserves one token per unique provider in the lane set under
    // a global acquisition order. Returns 0 OK, 1 temporary busy, 2 invalid.
    // timeout_ms=0 waits until a running context releases credit.
    GGML_API int ggml_backend_meta_staging_owner_acquire_set(
        ggml_backend_t * lanes, size_t n_lanes, uint64_t run_id, uint32_t window_id,
        uint64_t * owners, uint64_t * generations, size_t * out_count,
        uint32_t timeout_ms);
    GGML_API int  ggml_backend_meta_staging_owner_drain(uint64_t owner, uint64_t generation);
    // Per-context references for a token shared by target/draft. Release is
    // refused until every bound context unbinds.
    GGML_API ggml_backend_reg_t ggml_backend_meta_staging_provider_reg_for_backend(
        ggml_backend_t backend);
    GGML_API ggml_backend_reg_t ggml_backend_meta_staging_owner_provider_reg(
        uint64_t owner, uint64_t generation);
    GGML_API bool ggml_backend_meta_clear_staging_owner(ggml_backend_t backend);
    GGML_API bool ggml_backend_meta_staging_owner_context_bind(uint64_t owner, uint64_t generation);
    GGML_API bool ggml_backend_meta_staging_owner_context_unbind(uint64_t owner, uint64_t generation);
    GGML_BASE_API void ggml_backend_meta_staging_owner_retry_orphans(void);
    GGML_BASE_API size_t ggml_backend_meta_staging_owner_orphan_count(void);
    GGML_BASE_API void ggml_backend_retry_retained(void);
    GGML_API bool ggml_backend_meta_staging_owner_context_bind_many(
        const uint64_t * owners, const uint64_t * generations, size_t n);
    GGML_API bool ggml_backend_meta_staging_owner_context_unbind_many(
        const uint64_t * owners, const uint64_t * generations, size_t n);
    GGML_API bool ggml_backend_meta_staging_owner_release(uint64_t owner, uint64_t generation);
    GGML_API bool ggml_backend_meta_set_staging_owner(
        ggml_backend_t backend, uint64_t owner, uint64_t generation);
    // Validates the token's acquired run/window in addition to provider/child
    // identity. New callers should use this form.
    GGML_API bool ggml_backend_meta_set_staging_owner_for_run(
        ggml_backend_t backend, uint64_t owner, uint64_t generation,
        uint64_t run_id, uint32_t window_id);

    // DUALWIN-COMPOSE-EVENT (GGML_TP5_DW_COMPOSE_EVENT=1): output-copy
    // completion expressed as device events instead of a scheduler-wide
    // synchronize.
    //
    // ggml_backend_meta_output_record(meta_backend, tensor) records one private
    // event on every child stream that the Meta D2H copy of `tensor` uses
    // (exactly one stream for a MIRRORED tensor, every contributing child for a
    // split axis), ordered behind everything already enqueued there —
    // including the asynchronous D2H copies issued before this call. A
    // non-null event still held from an earlier record means that record was
    // never waited on (its epoch was abandoned before output_wait); it is
    // released first, without waiting, so the next wait cannot report a stale
    // completion. It returns the number of streams recorded in THIS call, or 0
    // when the backend is not Meta, the tensor is not Meta-buffered or is
    // PARTIAL/unknown, the gate is off, or a required stream lacks an event
    // interface. On 0 the caller must keep its full synchronize; no event is
    // left allocated.
    //
    // ggml_backend_meta_output_wait(meta_backend) host-completes those events
    // and releases them, returning the number proven complete. Only device event
    // synchronizes run: no scheduler backend, CPU backend or relay counter
    // participates.
    //
    // The pair owns backend-level state and is not thread-safe; one caller owns
    // the record/wait window.
    GGML_API size_t ggml_backend_meta_output_record(ggml_backend_t meta_backend, const struct ggml_tensor * tensor);
    GGML_API size_t ggml_backend_meta_output_wait  (ggml_backend_t meta_backend);

    // TP5 Meta submit hooks. Both are caller-thread-scoped observation points:
    // the ready hook runs after private conversion/prepare and before native
    // submit; the submitted hook runs only after a successful, sealed native
    // submit and after the provider lock is released. The caller owns userdata
    // until it clears the hook with (nullptr, nullptr). Returns false for a
    // partial hook/userdata pair or a nested ready hook.
    GGML_API bool ggml_backend_meta_set_submit_ready_hook(bool (*hook)(void *), void * userdata);
    GGML_API bool ggml_backend_meta_set_submitted_hook  (bool (*hook)(void *), void * userdata);

    // ggml_backend_meta_first_child_device(meta_backend) returns the device of
    // the Meta backend's FIRST simple child, or nullptr when there is none (not
    // a Meta backend, no initialized child, or child 0 gone). A Meta backend
    // enumerates its children in its Meta device's simple-device order, so
    // child 0's device is that device. The pointer is owned by the child's own
    // backend registry, so it stays valid for device-identity comparison while
    // that registry lives; the caller must not free it. This probe is null-safe
    // where the sibling accessors assert, and allocates, records and prints
    // nothing.
    GGML_API ggml_backend_dev_t ggml_backend_meta_first_child_device(ggml_backend_t meta_backend);

    // ggml_backend_meta_first_child_backend(meta_backend) returns the FIRST
    // simple child BACKEND of a Meta backend, or nullptr when there is none. A
    // device is only the template a backend is initialized from: the Meta
    // device's own interface deliberately has no event_new/event_record (event
    // use on it would change scheduler-wide synchronization), so a caller that
    // needs real stream events must be handed the child backend, whose iface
    // carries event_record/event_wait. The pointer is owned by the Meta backend
    // (valid until ggml_backend_meta_free) and must not be freed. Null-safe like
    // its device sibling; allocates, records and prints nothing.
    GGML_API ggml_backend_t ggml_backend_meta_first_child_backend(ggml_backend_t meta_backend);

    // ggml_backend_meta_first_event_child_backend(meta_backend) returns the
    // FIRST simple child backend whose device can create a real stream event,
    // or nullptr when no child can. "Can create" is decided HERE, inside the
    // Meta implementation, by asking the child device for an event through the
    // public API (ggml_backend_event_new) and releasing it immediately: a
    // caller must never inspect a device interface itself, because
    // struct ggml_backend_device is incomplete outside this library. The child
    // order is the Meta device's simple-device order, so child 0 wins whenever
    // it can create events. The returned backend is owned by the Meta backend
    // (valid until ggml_backend_meta_free) and must not be freed. Null-safe;
    // the probe creates and frees one event per candidate and prints nothing.
    GGML_API ggml_backend_t ggml_backend_meta_first_event_child_backend(ggml_backend_t meta_backend);

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
