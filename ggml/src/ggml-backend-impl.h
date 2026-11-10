#pragma once

// ggml-backend internal header

#include "ggml-backend.h"
#ifdef __cplusplus
#include <memory>
#endif

#ifdef  __cplusplus
// How the registry knows the module behind a backend reg. A static backend
// has no DSO handle, which is a legal lifetime; an unloaded dynamic backend
// has no handle because its code may be gone. Callers must not conflate the
// two.
enum class ggml_backend_module_state : unsigned char {
    unknown,            // never registered by this process (test/external reg)
    static_backend,     // registered without a DSO handle
    dynamic_backend,    // registered from a DSO, handle held
    unloading_backend,  // unload in progress, handle held
    unloaded_backend,   // a registered dynamic backend was removed
};

struct ggml_backend_module_ref {
    std::shared_ptr<void> module;
    ggml_backend_module_state state = ggml_backend_module_state::unknown;
};

GGML_BASE_API ggml_backend_module_ref ggml_backend_reg_module_hold(ggml_backend_reg_t reg);

// Attach the loader's module reference to provider entries that were created
// before the loader published the handle (a self-registering DSO). The
// failed-load variant fills every entry created by the transaction, so a
// quarantined provider keeps the DSO mapped.
GGML_BASE_API void ggml_backend_meta_staging_attach_module(
        ggml_backend_reg_t reg, std::shared_ptr<void> module);
GGML_BASE_API void ggml_backend_meta_staging_attach_load_module(
        ggml_backend_reg_t reg, uint32_t tag_floor, const void * module_identity,
        std::shared_ptr<void> module);
// Pin-aware free. Returns true only when the backend was actually freed;
// false means it was retained and its host pin was put back.
GGML_BASE_API bool ggml_backend_free_and_report(ggml_backend_t backend);

extern "C" {
#endif

    // Provider free callbacks that retain the backend object call this before
    // returning; ggml_backend_free keeps the host pin for the retained object.
    // C linkage so C providers can use the same contract.
    GGML_BASE_API void ggml_backend_mark_retained(ggml_backend_t backend);

    // Internal: free a Meta backend and report whether it was actually freed.
    // A false return means the owner was retained because completion is
    // unproved; the caller must keep the provider module mapped.
    GGML_BASE_API bool ggml_backend_meta_free_and_report(ggml_backend_t backend);

    // Loader transaction: snapshot the provider tag counter before
    // backend_init_fn and remove only the provider entries created after it
    // when init fails.
    GGML_BASE_API uint32_t ggml_backend_meta_staging_tag_floor(void);
    GGML_BASE_API void ggml_backend_meta_staging_rollback_load(
            ggml_backend_reg_t reg, uint32_t tag_floor, const void * module_identity);

    #define GGML_BACKEND_API_VERSION 2

    //
    // Backend buffer type
    //

    struct ggml_backend_buffer_type_i {
        const char *          (*get_name)      (ggml_backend_buffer_type_t buft);
        // allocate a buffer of this type
        ggml_backend_buffer_t (*alloc_buffer)  (ggml_backend_buffer_type_t buft, size_t size);
        // tensor alignment
        size_t                (*get_alignment) (ggml_backend_buffer_type_t buft);
        // (optional) max buffer size that can be allocated (defaults to SIZE_MAX)
        size_t                (*get_max_size)  (ggml_backend_buffer_type_t buft);
        // (optional) data size needed to allocate the tensor, including padding (defaults to ggml_nbytes)
        size_t                (*get_alloc_size)(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor);
        // (optional) check if tensor data is in host memory and uses standard ggml tensor layout (defaults to false)
        bool                  (*is_host)       (ggml_backend_buffer_type_t buft);
    };

    struct ggml_backend_buffer_type {
        struct ggml_backend_buffer_type_i  iface;
        ggml_backend_dev_t device;
        void * context;
    };

    //
    // Backend buffer
    //

    struct ggml_backend_buffer_i {
        // (optional) free the buffer
        void         (*free_buffer)  (ggml_backend_buffer_t buffer);
        // base address of the buffer
        void *       (*get_base)     (ggml_backend_buffer_t buffer);
        // (optional) initialize a tensor in the buffer (eg. add tensor extras)
        enum ggml_status (*init_tensor)(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor);
        // tensor data access
        void         (*memset_tensor)(ggml_backend_buffer_t buffer,       struct ggml_tensor * tensor,     uint8_t value, size_t offset, size_t size);
        void         (*set_tensor)   (ggml_backend_buffer_t buffer,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
        void         (*get_tensor)   (ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size);
        // (optional) 2d data copies
        void         (*set_tensor_2d)(ggml_backend_buffer_t buffer,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
        void         (*get_tensor_2d)(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);

        // (optional) tensor copy: dst is in the buffer, src may be in any buffer, including buffers from a different backend (return false if not supported)
        bool         (*cpy_tensor)   (ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst);
        // clear the entire buffer
        void         (*clear)        (ggml_backend_buffer_t buffer, uint8_t value);
        // (optional) reset any internal state due to tensor initialization, such as tensor extras
        void         (*reset)        (ggml_backend_buffer_t buffer);
    };

    struct ggml_backend_buffer {
        struct ggml_backend_buffer_i  iface;
        ggml_backend_buffer_type_t    buft;
        void * context;
        size_t size;
        enum ggml_backend_buffer_usage usage;
    };

    GGML_API ggml_backend_buffer_t ggml_backend_buffer_init(
                   ggml_backend_buffer_type_t buft,
            struct ggml_backend_buffer_i      iface,
                   void *                     context,
                   size_t                     size);

    // do not use directly, use ggml_backend_tensor_copy instead
    GGML_API bool ggml_backend_buffer_copy_tensor(const struct ggml_tensor * src, struct ggml_tensor * dst);

    // multi-buffer
    // buffer that contains a collection of buffers
    GGML_API ggml_backend_buffer_t ggml_backend_multi_buffer_alloc_buffer(ggml_backend_buffer_t * buffers, size_t n_buffers);
    GGML_API bool                  ggml_backend_buffer_is_multi_buffer(ggml_backend_buffer_t buffer);
    GGML_API void                  ggml_backend_multi_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage);

    //
    // Backend (meta)
    //

    GGML_API bool ggml_backend_is_meta       (ggml_backend_t backend);
    GGML_API bool ggml_backend_buffer_is_meta(ggml_backend_buffer_t buf);
    GGML_API bool ggml_backend_buft_is_meta  (ggml_backend_buffer_type_t buft);
    // Pure Meta-to-Meta preflight. IMPOSSIBLE leaves both tensors untouched;
    // EXACT uses the child fast copy; SLOW uses the host-staging tensor copy.
    // A copy true is a submission, not a device-completion proof.
    typedef enum ggml_backend_meta_copy_kind {
        GGML_BACKEND_META_COPY_IMPOSSIBLE = 0,
        GGML_BACKEND_META_COPY_EXACT      = 1,
        GGML_BACKEND_META_COPY_SLOW       = 2,
    } ggml_backend_meta_copy_kind;
    GGML_API ggml_backend_meta_copy_kind ggml_backend_meta_buffer_copy_kind(
        struct ggml_tensor * src, struct ggml_tensor * dst);
    GGML_API bool ggml_backend_meta_buffer_can_copy(struct ggml_tensor * src, struct ggml_tensor * dst);

    GGML_API size_t         ggml_backend_meta_n_backends    (ggml_backend_t meta_backend);
    GGML_API ggml_backend_t ggml_backend_meta_simple_backend(ggml_backend_t meta_backend, size_t index);

    // The first-child probes (ggml_backend_meta_first_child_device /
    // ..._first_child_backend / ..._first_event_child_backend) are declared in
    // the public ggml-backend.h header; no private declaration is needed here.

    // temporary workaround to statically allocate tensors from a context in a deduplicated way:
    GGML_API struct ggml_backend_buffer * ggml_backend_meta_alloc_ctx_tensors_from_buft(struct ggml_context * ctx, ggml_backend_buffer_type_t buft);

    // Optional drain receipt negotiated under the name
    // "ggml_backend_drain_v1". Only DRAINED is a native completion fact; a
    // missing or failed receipt never clears a sticky submission.
    enum ggml_backend_drain_result {
        GGML_BACKEND_DRAIN_DRAINED = 0,
        GGML_BACKEND_DRAIN_BUSY    = 1,
        GGML_BACKEND_DRAIN_FAILED  = 2,
        // Physically drained, but a submission recorded a data error. The
        // context is being destroyed, not replayed; the resources are safe to
        // release.
        GGML_BACKEND_DRAIN_DRAINED_WITH_ERROR = 3,
    };
    #define GGML_BACKEND_DRAIN_API_VERSION 1
    struct ggml_backend_drain_api {
        uint32_t abi_version;
        int (*drain)(ggml_backend_t backend);
    };

    // Optional gated synchronize capability negotiated under the name
    // "ggml_backend_synchronize_if_proven_v1". The backend must not drain an
    // unproven submission: it synchronizes under its own ownership gate and
    // returns false without synchronizing when the submission is unproven.
    #define GGML_BACKEND_SYNCHRONIZE_IF_PROVEN_API_VERSION 1
    struct ggml_backend_synchronize_if_proven_api {
        uint32_t abi_version;
        bool (*sync)(ggml_backend_t backend);
    };

    //
    // Backend (stream)
    //

    struct ggml_backend_i {
        const char * (*get_name)(ggml_backend_t backend);

        void (*free)(ggml_backend_t backend);

        // (optional) asynchronous tensor data access
        void (*set_tensor_async)   (ggml_backend_t backend,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
        void (*get_tensor_async)   (ggml_backend_t backend, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size);
        void (*set_tensor_2d_async)(ggml_backend_t backend,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
        void (*get_tensor_2d_async)(ggml_backend_t backend, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
        // (optional) asynchronous tensor copy. Return false only when no copy
        // was queued and the destination was not written, so the caller may
        // fall back to a blocking copy. A callback must not return false after
        // queueing work or writing the destination.
        bool (*cpy_tensor_async)(ggml_backend_t backend_src, ggml_backend_t backend_dst, const struct ggml_tensor * src, struct ggml_tensor * dst);

        // (optional) complete all pending operations (required if the backend supports async operations)
        void (*synchronize)(ggml_backend_t backend);

        // (optional) graph plans (not used currently)
        // compute graph with a plan
        ggml_backend_graph_plan_t (*graph_plan_create) (ggml_backend_t backend, const struct ggml_cgraph * cgraph);
        void                      (*graph_plan_free)   (ggml_backend_t backend, ggml_backend_graph_plan_t plan);
        // update the plan with a new graph - this should be faster than creating a new plan when the graph has the same topology
        void                      (*graph_plan_update) (ggml_backend_t backend, ggml_backend_graph_plan_t plan, const struct ggml_cgraph * cgraph);
        // compute the graph with the plan
        enum ggml_status          (*graph_plan_compute)(ggml_backend_t backend, ggml_backend_graph_plan_t plan);

        // compute graph (always async if supported by the backend)
        enum ggml_status          (*graph_compute)     (ggml_backend_t backend, struct ggml_cgraph * cgraph);

        // (optional) event synchronization
        // record an event on this stream
        void (*event_record)(ggml_backend_t backend, ggml_backend_event_t event);
        // wait for an event on on a different stream
        void (*event_wait)  (ggml_backend_t backend, ggml_backend_event_t event);

        // (optional) sort/optimize the nodes in the graph
        void                      (*graph_optimize)    (ggml_backend_t backend, struct ggml_cgraph * cgraph);

    };

    struct ggml_backend {
        ggml_guid_t guid;
        struct ggml_backend_i iface;
        ggml_backend_dev_t device;
        void * context;
    };

    struct ggml_backend_event {
        struct ggml_backend_device * device;
        void * context;
    };

    //
    // Backend device
    //

    // Note: if additional properties are needed, we should add a struct with all of them
    //       the current functions to obtain the properties can remain, since they are more convenient for often used properties
    struct ggml_backend_device_i {
        // device name: short identifier for this device, such as "CPU" or "CUDA0"
        const char * (*get_name)(ggml_backend_dev_t dev);

        // device description: short informative description of the device, could be the model name
        const char * (*get_description)(ggml_backend_dev_t dev);

        // device memory in bytes: 0 bytes to indicate no memory to report
        void         (*get_memory)(ggml_backend_dev_t dev, size_t * free, size_t * total);

        // device type
        enum ggml_backend_dev_type (*get_type)(ggml_backend_dev_t dev);

        // device properties
        void (*get_props)(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props);

        // backend (stream) initialization
        ggml_backend_t (*init_backend)(ggml_backend_dev_t dev, const char * params);

        // preferred buffer type
        ggml_backend_buffer_type_t (*get_buffer_type)(ggml_backend_dev_t dev);

        // (optional) host buffer type (in system memory, typically this is a pinned memory buffer for faster transfers between host and device)
        ggml_backend_buffer_type_t (*get_host_buffer_type)(ggml_backend_dev_t dev);

        // (optional) buffer from pointer: create a buffer from a host pointer (useful for memory mapped models and importing data from other libraries)
        ggml_backend_buffer_t (*buffer_from_host_ptr)(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size);

        // check if the backend can compute an operation
        bool (*supports_op)(ggml_backend_dev_t dev, const struct ggml_tensor * op);

        // check if the backend can use tensors allocated in a buffer type
        bool (*supports_buft)(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft);

        // (optional) check if the backend wants to run an operation, even if the weights are allocated in an incompatible buffer
        // these should be expensive operations that may benefit from running on this backend instead of the CPU backend
        bool (*offload_op)(ggml_backend_dev_t dev, const struct ggml_tensor * op);

        // (optional) event synchronization
        ggml_backend_event_t (*event_new)         (ggml_backend_dev_t dev);
        void                 (*event_free)        (ggml_backend_dev_t dev, ggml_backend_event_t event);
        void                 (*event_synchronize) (ggml_backend_dev_t dev, ggml_backend_event_t event);
    };

    struct ggml_backend_device {
        struct ggml_backend_device_i iface;
        ggml_backend_reg_t reg;
        void * context;
    };

    //
    // Backend (reg)
    //

    struct ggml_backend_reg_i {
        const char * (*get_name)(ggml_backend_reg_t reg);

        // enumerate available devices
        size_t             (*get_device_count)(ggml_backend_reg_t reg);
        ggml_backend_dev_t (*get_device)(ggml_backend_reg_t reg, size_t index);

        // (optional) get a pointer to a function in the backend
        // backends can add custom functions that are not part of the standard ggml-backend interface
        void * (*get_proc_address)(ggml_backend_reg_t reg, const char * name);
    };

    struct ggml_backend_reg {
        int api_version; // initialize to GGML_BACKEND_API_VERSION
        struct ggml_backend_reg_i iface;
        void * context;
    };

    // Optional per-attempt completion receipt, negotiated under the name
    // "ggml_backend_completion_v1".  A backend that accepts asynchronous graph
    // work publishes an attempt id in graph_compute.  wait() must report the
    // outcome of that exact attempt without aborting on a native error:
    // SUCCESS and FAILED are completed facts; UNKNOWN means the backend cannot
    // prove the attempt.  A missing receipt is never a success.
    enum ggml_backend_completion_result {
        GGML_BACKEND_COMPLETION_SUCCESS = 0,
        GGML_BACKEND_COMPLETION_FAILED  = 1,
        GGML_BACKEND_COMPLETION_UNKNOWN = 2,
    };
    #define GGML_BACKEND_COMPLETION_API_VERSION 3
    struct ggml_backend_completion_api {
        uint32_t abi_version;
        // Latest accepted graph attempt, or 0 when none is recorded.
        // Atomically create a completion ticket for exactly this submission.
        // Normal graph/copy calls do not call begin and create no record.
        uint64_t (*begin)(ggml_backend_t backend);
        // Seal the ticket after the backend submission call. No record is
        // created outside begin/end.
        void (*end)(ggml_backend_t backend);
        uint64_t (*attempt)(ggml_backend_t backend);
        // Wait for one attempt.  error receives the native/ggml error for
        // FAILED; it may be NULL.  The call must not terminate the process.
        int (*wait)(ggml_backend_t backend, uint64_t attempt, int * error);
        // Prove that all backend work has reached a completion boundary.
        // Used only to retire a quarantined host undo after an UNKNOWN
        // attempt; it is not a per-attempt success claim.
        int (*drain)(ggml_backend_t backend, int * error);
        // Release a submission token after wait/drain/terminal state.
        void (*release)(ggml_backend_t backend, uint64_t attempt);
    };

    // Add backend dynamic loading support to the backend

    // Initialize the backend
    typedef ggml_backend_reg_t (*ggml_backend_init_t)(void);
    // Optional: obtain a score for the backend based on the system configuration
    // Higher scores are preferred, 0 means the backend is not supported in the current system
    typedef int                (*ggml_backend_score_t)(void);

#ifdef GGML_BACKEND_DL
#    ifdef __cplusplus
#        define GGML_BACKEND_DL_IMPL(reg_fn)                             \
            extern "C" {                                                 \
            GGML_BACKEND_API ggml_backend_reg_t ggml_backend_init(void); \
            }                                                            \
            ggml_backend_reg_t ggml_backend_init(void) {                 \
                return reg_fn();                                         \
            }
#        define GGML_BACKEND_DL_SCORE_IMPL(score_fn)       \
            extern "C" {                                   \
            GGML_BACKEND_API int ggml_backend_score(void); \
            }                                              \
            int ggml_backend_score(void) {                 \
                return score_fn();                         \
            }
#    else
#        define GGML_BACKEND_DL_IMPL(reg_fn)                              \
            GGML_BACKEND_API ggml_backend_reg_t ggml_backend_init(void);  \
            ggml_backend_reg_t                  ggml_backend_init(void) { \
                return reg_fn();                                          \
            }
#        define GGML_BACKEND_DL_SCORE_IMPL(score_fn)        \
            GGML_BACKEND_API int ggml_backend_score(void);  \
            int                  ggml_backend_score(void) { \
                return score_fn();                          \
            }
#    endif
#else
#    define GGML_BACKEND_DL_IMPL(reg_fn)
#    define GGML_BACKEND_DL_SCORE_IMPL(score_fn)
#endif

#ifdef  __cplusplus
}
#endif
