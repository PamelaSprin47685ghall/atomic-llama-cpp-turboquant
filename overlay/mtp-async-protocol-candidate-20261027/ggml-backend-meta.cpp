#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend.h"
#include "tp5-meta-submit-ready.h"
#include "ggml-backend-impl.h"
#include "ggml-relay-impl.h"
#include "ggml-alloc.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

struct ggml_backend_meta_device;
struct ggml_backend_meta_buffer_type;
struct ggml_backend_meta_buffer;
struct ggml_backend_meta;

const char * ggml_backend_meta_split_axis_name(enum ggml_backend_meta_split_axis split_axis) {
    switch (split_axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
            return "0";
        case GGML_BACKEND_SPLIT_AXIS_1:
            return "1";
        case GGML_BACKEND_SPLIT_AXIS_2:
            return "2";
        case GGML_BACKEND_SPLIT_AXIS_3:
            return "3";
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            return "MIRRORED";
        case GGML_BACKEND_SPLIT_AXIS_PARTIAL:
            return "PARTIAL";
        case GGML_BACKEND_SPLIT_AXIS_NONE:
            return "NONE";
        case GGML_BACKEND_SPLIT_AXIS_UNKNOWN:
            return "UNKNOWN";
        default:
            GGML_ABORT("fatal error");
    }
}

//
// meta backend device
//

struct ggml_backend_meta_device_context {
    std::vector<ggml_backend_dev_t>     simple_devs;
    ggml_backend_meta_get_split_state_t get_split_state;
    void *                              get_split_state_ud;

    std::string name;
    std::string description;

    ggml_backend_meta_device_context(
            std::vector<ggml_backend_dev_t> simple_devs, ggml_backend_meta_get_split_state_t get_split_state, void * get_split_state_ud) :
            simple_devs(std::move(simple_devs)), get_split_state(get_split_state), get_split_state_ud(get_split_state_ud) {
        name        = std::string("Meta(");
        description = std::string("Meta(");
        for (size_t i = 0; i < simple_devs.size(); i++) {
            if (i > 0) {
                name        += ",";
                description += ",";
            }
            name        += ggml_backend_dev_name       (simple_devs[i]);
            description += ggml_backend_dev_description(simple_devs[i]);
        }
        name        += ")";
        description += ")";
    }

    bool operator<(const ggml_backend_meta_device_context & other) const {
        return std::tie(simple_devs, get_split_state, get_split_state_ud)
            < std::tie(other.simple_devs, other.get_split_state, other.get_split_state_ud);
    }
};

static bool ggml_backend_dev_is_meta(ggml_backend_dev_t dev);

static const char * ggml_backend_meta_device_get_name(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    return meta_dev_ctx->name.c_str();
}

static const char * ggml_backend_meta_device_get_description(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    return meta_dev_ctx->description.c_str();
}

static void ggml_backend_meta_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    *free  = 0;
    *total = 0;
    for (ggml_backend_dev_t dev : meta_dev_ctx->simple_devs) {
        size_t tmp_free, tmp_total;
        ggml_backend_dev_memory(dev, &tmp_free, &tmp_total);
        *free  += tmp_free;
        *total += tmp_total;
    }
}

static enum ggml_backend_dev_type ggml_backend_meta_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_META;

    GGML_UNUSED(dev);
}

static void ggml_backend_meta_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;

    // TODO replace placeholders
    props->name        = ggml_backend_meta_device_get_name(dev);
    props->description = ggml_backend_meta_device_get_description(dev);
    props->type        = ggml_backend_meta_device_get_type(dev);
    props->device_id   = 0;

    ggml_backend_meta_device_get_memory(dev, &props->memory_free, &props->memory_total);

    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ false, // Not implemented.
        /* .buffer_from_host_ptr  = */ false, // Not implemented.
        /* .events                = */ false, // Not implemented.
    };
    for (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_dev_props tmp_props;
        ggml_backend_dev_get_props(simple_dev, &tmp_props);
        props->caps.async                = props->caps.async                && tmp_props.caps.async;
        props->caps.host_buffer          = props->caps.host_buffer          && tmp_props.caps.host_buffer;
        props->caps.buffer_from_host_ptr = props->caps.buffer_from_host_ptr && tmp_props.caps.buffer_from_host_ptr;
        props->caps.events               = props->caps.events               && tmp_props.caps.events;
    }
}

static ggml_backend_t ggml_backend_meta_device_init_backend(ggml_backend_dev_t dev, const char * params);

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_buffer_type(ggml_backend_dev_t dev);

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_host_buffer_type(ggml_backend_dev_t dev);

static bool ggml_backend_meta_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    return std::all_of(meta_dev_ctx->simple_devs.begin(), meta_dev_ctx->simple_devs.end(),
        [op](ggml_backend_dev_t simple_dev) { return ggml_backend_dev_supports_op(simple_dev, op); });
}

static bool ggml_backend_meta_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ggml_backend_dev_t dev_buft = ggml_backend_buft_get_device(buft);
    if (!ggml_backend_dev_is_meta(dev_buft)) {
        return false;
    }
    const ggml_backend_meta_device_context * meta_dev_ctx      = (const ggml_backend_meta_device_context *) dev->context;
    const ggml_backend_meta_device_context * meta_buft_dev_ctx = (const ggml_backend_meta_device_context *) dev_buft->context;
    if (meta_dev_ctx->simple_devs.size() != meta_buft_dev_ctx->simple_devs.size()) {
        return false;
    }
    for (size_t i = 0; i < meta_dev_ctx->simple_devs.size(); i++) {
        if (meta_dev_ctx->simple_devs[i] != meta_buft_dev_ctx->simple_devs[i]) {
            return false;
        }
    }
    return true;
}

static const ggml_backend_device_i ggml_backend_meta_device_iface = {
    /* .get_name             = */ ggml_backend_meta_device_get_name,
    /* .get_description      = */ ggml_backend_meta_device_get_description,
    /* .get_memory           = */ ggml_backend_meta_device_get_memory,
    /* .get_type             = */ ggml_backend_meta_device_get_type,
    /* .get_props            = */ ggml_backend_meta_device_get_props,
    /* .init_backend         = */ ggml_backend_meta_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_meta_device_get_buffer_type,
    /* .get_host_buffer_type = */ ggml_backend_meta_device_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_backend_meta_device_supports_op,
    /* .supports_buft        = */ ggml_backend_meta_device_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

static bool ggml_backend_dev_is_meta(ggml_backend_dev_t dev) {
    return dev != nullptr && dev->iface.get_name == ggml_backend_meta_device_iface.get_name;
}

static size_t ggml_backend_meta_dev_n_devs(ggml_backend_dev_t meta_dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(meta_dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) meta_dev->context;
    return meta_dev_ctx->simple_devs.size();
}

static ggml_backend_dev_t ggml_backend_meta_dev_simple_dev(ggml_backend_dev_t meta_dev, size_t index) {
    GGML_ASSERT(ggml_backend_dev_is_meta(meta_dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) meta_dev->context;
    GGML_ASSERT(index < meta_dev_ctx->simple_devs.size());
    return meta_dev_ctx->simple_devs[index];
}

ggml_backend_dev_t ggml_backend_meta_device(
        ggml_backend_dev_t * devs, size_t n_devs, ggml_backend_meta_get_split_state_t get_split_state, void * get_split_state_ud) {
    GGML_ASSERT(n_devs <= GGML_BACKEND_META_MAX_DEVICES);
    // TODO: this is not thread-safe - needs to be fixed
    static std::vector<std::unique_ptr<ggml_backend_meta_device_context>>         ctxs;
    static std::map<ggml_backend_meta_device_context, struct ggml_backend_device> meta_devs;

    std::vector<ggml_backend_dev_t> simple_devs;
    simple_devs.reserve(n_devs);
    for (size_t i = 0; i < n_devs; i++) {
        simple_devs.push_back(devs[i]);
    }
    ggml_backend_meta_device_context ctx(simple_devs, get_split_state, get_split_state_ud);

    {
        auto it = meta_devs.find(ctx);
        if (it != meta_devs.end()) {
            return &it->second;
        }
    }
    ctxs.push_back(std::make_unique<ggml_backend_meta_device_context>(ctx));

    struct ggml_backend_device meta_dev = {
        /*iface  =*/ ggml_backend_meta_device_iface,
        /*reg    =*/ nullptr,
        /*ctx    =*/ ctxs.back().get(),
    };

    auto result = meta_devs.emplace(*ctxs.back(), meta_dev);
    return &result.first->second;
}

//
// meta backend buffer type
//

struct ggml_backend_meta_buffer_type_context {
    std::vector<ggml_backend_buffer_type_t> simple_bufts;

    std::string name;

    ggml_backend_meta_buffer_type_context(std::vector<ggml_backend_buffer_type_t> simple_bufts) : simple_bufts(std::move(simple_bufts)) {
        name = "Meta(";
        for (size_t i = 0; i < simple_bufts.size(); i++) {
            if (i > 0) {
                name += ",";
            }
            name += ggml_backend_buft_name(simple_bufts[i]);
        }
        name += ")";
    }

    bool operator<(const ggml_backend_meta_buffer_type_context & other) const {
        return simple_bufts < other.simple_bufts;
    }
};

static size_t ggml_backend_meta_buft_n_bufts(ggml_backend_buffer_type_t meta_buft) {
    GGML_ASSERT(ggml_backend_buft_is_meta(meta_buft));
    const ggml_backend_meta_buffer_type_context * meta_buft_ctx = (const ggml_backend_meta_buffer_type_context *) meta_buft->context;
    return meta_buft_ctx->simple_bufts.size();
}

static const char * ggml_backend_meta_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_backend_buft_is_meta(buft));
    const ggml_backend_meta_buffer_type_context * meta_buft_ctx = (const ggml_backend_meta_buffer_type_context *) buft->context;
    return meta_buft_ctx->name.c_str();
}

static ggml_backend_buffer_type_t ggml_backend_meta_buft_simple_buft(ggml_backend_buffer_type_t meta_buft, size_t index) {
    GGML_ASSERT(ggml_backend_buft_is_meta(meta_buft));
    const ggml_backend_meta_buffer_type_context * meta_buft_ctx = (const ggml_backend_meta_buffer_type_context *) meta_buft->context;
    GGML_ASSERT(index < meta_buft_ctx->simple_bufts.size());
    return meta_buft_ctx->simple_bufts[index];
}

static ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size);

static size_t ggml_backend_meta_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_alignment = 1;
    for (size_t i = 0; i < n_simple_bufts; i++) {
        const size_t alignment = ggml_backend_buft_get_alignment(ggml_backend_meta_buft_simple_buft(buft, i));
        max_alignment = std::max(max_alignment, alignment);
        GGML_ASSERT(max_alignment % alignment == 0);
    }
    return max_alignment;
}

static size_t ggml_backend_meta_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_size = SIZE_MAX;
    for (size_t i = 0; i < n_simple_bufts; i++) {
        max_size = std::min(max_size, ggml_backend_buft_get_max_size(ggml_backend_meta_buft_simple_buft(buft, i)));
    }
    return max_size;
}

static size_t ggml_backend_meta_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_alloc_size = 0;
    for (size_t i = 0; i < n_simple_bufts; i++) {
        const size_t alloc_size = ggml_backend_buft_get_alloc_size(ggml_backend_meta_buft_simple_buft(buft, i), tensor);
        max_alloc_size = std::max(max_alloc_size, alloc_size);
    }
    return max_alloc_size;
}

static bool ggml_backend_meta_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    for (size_t i = 0; i < n_simple_bufts; i++) {
        if (!ggml_backend_buft_is_host(ggml_backend_meta_buft_simple_buft(buft, i))) {
            return false;
        }
    }
    return true;
}

static const struct ggml_backend_buffer_type_i ggml_backend_meta_buffer_type_iface = {
    /* .get_name         = */ ggml_backend_meta_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_meta_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_meta_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_meta_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_meta_buffer_type_get_alloc_size,
    /* .is_host          = */ ggml_backend_meta_buffer_type_is_host,
};

bool ggml_backend_buft_is_meta(ggml_backend_buffer_type_t buft) {
    return buft != nullptr && buft->iface.get_name == ggml_backend_meta_buffer_type_iface.get_name;
}

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_buffer_type(ggml_backend_dev_t dev) {
    static std::map<ggml_backend_dev_t, struct ggml_backend_buffer_type> meta_bufts;
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    {
        auto it = meta_bufts.find(dev);
        if (it != meta_bufts.end()) {
            return &it->second;
        }
    }

    const size_t n_devs = ggml_backend_meta_dev_n_devs(dev);
    std::vector<ggml_backend_buffer_type_t> simple_bufts;
    simple_bufts.reserve(n_devs);
    for (size_t i = 0; i < n_devs; i++) {
        simple_bufts.push_back(ggml_backend_dev_buffer_type(ggml_backend_meta_dev_simple_dev(dev, i)));
    }
    ggml_backend_meta_buffer_type_context * buft_ctx = new ggml_backend_meta_buffer_type_context(simple_bufts);

    struct ggml_backend_buffer_type meta_buft = {
        /*iface  =*/ ggml_backend_meta_buffer_type_iface,
        /*device =*/ dev,
        /*ctx    =*/ buft_ctx,
    };
    auto result = meta_bufts.emplace(dev, meta_buft);
    return &result.first->second;
}

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;

    ggml_backend_buffer_type_t host_buft = nullptr;
    for (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_buffer_type_t simple_host_buft = ggml_backend_dev_host_buffer_type(simple_dev);
        if (simple_host_buft == nullptr) {
            return nullptr;
        }
        if (host_buft == nullptr) {
            host_buft = simple_host_buft;
        } else if (host_buft != simple_host_buft) {
            // if different simple devices have different host buffer types,
            // we cannot provide a single host buffer type for the meta device
            return nullptr;
        }
    }
    return host_buft;
}

//
// meta backend buffer
//

// Container to hold the tensor slices per simple ggml backend buffer.
struct ggml_backend_meta_simple_tensor_container {
    std::vector<ggml_context_ptr> ctxs;
    std::map<const ggml_tensor *, std::vector<ggml_tensor *>> simple_tensors;

    ggml_backend_meta_simple_tensor_container(const ggml_init_params & params, const int n_simple) {
        ctxs.reserve(n_simple);
        for (int i = 0; i < n_simple; i++) {
            ctxs.emplace_back(ggml_init(params));
        }
    }
    ggml_backend_meta_simple_tensor_container() {}
};

struct ggml_backend_meta_buffer_context {
    // FIXME
    // Most tensors can simply be stored statically in their own buffer.
    // Externally created views however also need a mapping to simple tensors but they use the buffer of the view source.
    // If external views are simply using that buffer they will slowly deplete its memory.
    // Current solution: rotating set of 2 "compute" containers to hold external views, works correctly for llama.cpp.
    // Long-term: tie the lifetime of external views to the meta backend executing the graph instead,
    //     currently not possible due to graph-external operations in the backend scheduler.
    ggml_backend_meta_simple_tensor_container stc_static;
    ggml_backend_meta_simple_tensor_container stc_compute[2];
    int stc_compute_index      = 0;
    int stc_compute_index_next = 0;
    std::vector<ggml_backend_buffer_ptr> bufs;

    // FIXME
    // The size of the split state cache is unbounded and can theoretically grow infinitely large.
    // However, it is also expensive to build and clearing it on every rebuild in ggml_backend_meta_graph_compute is too expensive.
    static constexpr size_t nbtc = GGML_TENSOR_SIZE - sizeof(ggml_tensor::padding);
    // Shared model buffers are read by independent context workers. Only
    // cache lookup/invalidation/commit holds this lock, never recursion or HIP.
    std::mutex split_state_mutex;
    std::map<std::pair<const ggml_tensor *, bool>, std::pair<ggml_backend_meta_split_state, char[nbtc]>> split_state_cache;

    int debug;

    ggml_backend_meta_buffer_context(
            ggml_backend_meta_simple_tensor_container & stc_static,
            ggml_backend_meta_simple_tensor_container & stc_compute_0,
            ggml_backend_meta_simple_tensor_container & stc_compute_1,
            const std::vector<ggml_backend_buffer_t> & bufs)
            : stc_static(std::move(stc_static)), stc_compute{std::move(stc_compute_0), std::move(stc_compute_1)} {
        this->bufs.reserve(bufs.size());
        for (ggml_backend_buffer_t buf : bufs) {
            this->bufs.emplace_back(buf);
        }
        const char * GGML_META_DEBUG = getenv("GGML_META_DEBUG");
        debug = GGML_META_DEBUG ? atoi(GGML_META_DEBUG) : 0;
    }

    ggml_backend_meta_simple_tensor_container & get_simple_tensor_container(const ggml_tensor * tensor) {
        if (stc_static.simple_tensors.find(tensor) != stc_static.simple_tensors.end()) {
            return stc_static;
        }
        return stc_compute[stc_compute_index];
    }
};

static void ggml_backend_meta_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    delete buf_ctx;
}

static size_t ggml_backend_meta_buffer_n_bufs(ggml_backend_buffer_t meta_buf) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(meta_buf));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) meta_buf->context;
    return buf_ctx->bufs.size();
}

static ggml_backend_buffer_t ggml_backend_meta_buffer_simple_buffer(ggml_backend_buffer_t meta_buf, size_t index) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(meta_buf));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) meta_buf->context;
    GGML_ASSERT(index < buf_ctx->bufs.size());
    return buf_ctx->bufs[index].get();
}

static struct ggml_tensor * ggml_backend_meta_buffer_simple_tensor(const struct ggml_tensor * tensor, size_t index) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(tensor->buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    GGML_ASSERT(index < buf_ctx->bufs.size());

    ggml_backend_meta_simple_tensor_container & stc = buf_ctx->get_simple_tensor_container(tensor);
    auto it = stc.simple_tensors.find(tensor);
    if (it == stc.simple_tensors.end()) {
        return nullptr;
    }
    return it->second[index];
}

static struct ggml_backend_meta_split_state ggml_backend_meta_get_split_state(const struct ggml_tensor * tensor, bool assume_sync);

static struct ggml_backend_meta_split_state ggml_backend_meta_get_split_state(
        ggml_backend_meta_simple_tensor_container & stc, const struct ggml_tensor * tensor, bool assume_sync) {
    // FIXME Currently this function preserves/erases the information in n_segments and nr in an inconsistent way.
    // Since the operations in question are developed specifically for llama.cpp this currently does not manifest as a bug there.
    // However, in a broader ggml context with arbitrary ggml graphs this can lead to unexpected results.
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;

    auto split_states_equal = [&](const ggml_backend_meta_split_state & a, const ggml_backend_meta_split_state & b) -> bool {
        if (a.axis != b.axis) {
            return false;
        }
        for (size_t j = 0; j < n_bufs; j++) {
            int64_t sum_a = 0;
            for (size_t s = 0; s < a.n_segments; s++) {
                sum_a += a.ne[s*n_bufs + j] * a.nr[s];
            }
            int64_t sum_b = 0;
            for (size_t s = 0; s < b.n_segments; s++) {
                sum_b += b.ne[s*n_bufs + j] * b.nr[s];
            }
            if (sum_a != sum_b) {
                return false;
            }
        }
        return true;
    };

    auto handle_generic = [&](const std::vector<ggml_backend_meta_split_state> & src_ss, bool scalar_only) -> ggml_backend_meta_split_state {
        ggml_backend_meta_split_state ret = {GGML_BACKEND_SPLIT_AXIS_NONE, {0}, {1}, 1};
        for (size_t i = 0; i < GGML_MAX_SRC; i++) {
            if (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                continue;
            }
            if (ret.axis == GGML_BACKEND_SPLIT_AXIS_NONE) {
                ret = src_ss[i];
            } else if (!split_states_equal(src_ss[i], ret)) {
                ret = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
                break;
            }
        }
        if (ret.axis == GGML_BACKEND_SPLIT_AXIS_NONE) {
            ret = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        if (scalar_only && ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
            ret = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        GGML_ASSERT(ret.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
        return ret;
    };

    // Some ops process data on a per-row bases:
    auto handle_per_row = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0);
        return src_ss[0];
    };

    // Some ops broadcast the src1 data across src0:
    auto handle_bin_bcast = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS &&
                tensor->src[1]->ne[src_ss[0].axis] == 1 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        if (src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && (src_ss[0].axis == src_ss[1].axis ||
           (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL)))) {
            return src_ss[0]; // GGML_OP_ADD_ID
        }
        GGML_ASSERT(tensor->src[2] == nullptr || src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        return handle_generic(src_ss, /*scalar_only =*/ false);
    };

    auto handle_concat = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        const ggml_backend_meta_split_axis concat_axis = ggml_backend_meta_split_axis(ggml_get_op_params_i32(tensor, 0));
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis >= 0 && src_ss[1].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(concat_axis != src_ss[1].axis);
            return src_ss[1];
        }
        if (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(concat_axis != src_ss[0].axis);
            return src_ss[0];
        }
        if (src_ss[0].axis == src_ss[1].axis && src_ss[0].axis != concat_axis) {
            return src_ss[0];
        }
        return handle_generic(src_ss, /*scalar_only =*/ true);
    };

    auto handle_mul_mat = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            ggml_backend_meta_split_state ret = src_ss[0];
            ret.axis = GGML_BACKEND_SPLIT_AXIS_0;
            ret.nr[0] = 1;
            ret.n_segments = 1;
            return ret;
        }
        if (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_1 && src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[1];
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[1].axis >= GGML_BACKEND_SPLIT_AXIS_2 && src_ss[1].axis < GGML_MAX_DIMS) {
            return src_ss[1];
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(split_states_equal(src_ss[0], src_ss[1]));
            return {assume_sync ? GGML_BACKEND_SPLIT_AXIS_MIRRORED : GGML_BACKEND_SPLIT_AXIS_PARTIAL, {0}, {1}, 1};
        }
        if (src_ss[0].axis == src_ss[1].axis && src_ss[0].axis >= GGML_BACKEND_SPLIT_AXIS_2 &&
                src_ss[0].axis < GGML_MAX_DIMS) {
            // Batched attention matmuls broadcast KV heads into Q-head
            // groups. Rank boundaries must preserve complete GQA groups,
            // not equal absolute head counts (e.g. Qwen3: 32 Q / 8 KV).
            const int axis = src_ss[0].axis;
            const int64_t heads_kv = tensor->src[0]->ne[axis];
            const int64_t heads_q = tensor->src[1]->ne[axis];
            GGML_ASSERT(heads_kv > 0 && heads_q % heads_kv == 0);
            const int64_t groups = heads_q / heads_kv;
            for (size_t rank = 0; rank < n_bufs; ++rank) {
                int64_t local_kv = 0;
                int64_t local_q = 0;
                for (size_t segment = 0; segment < src_ss[0].n_segments; ++segment) {
                    local_kv += src_ss[0].ne[segment*n_bufs + rank] * src_ss[0].nr[segment];
                }
                for (size_t segment = 0; segment < src_ss[1].n_segments; ++segment) {
                    local_q += src_ss[1].ne[segment*n_bufs + rank] * src_ss[1].nr[segment];
                }
                GGML_ASSERT(local_q == local_kv * groups);
            }
            return src_ss[1];
        }
        GGML_ABORT("unsupported mul_mat split states: node=%s src0=%s axis=%d src1=%s axis=%d",
            tensor->name, tensor->src[0]->name, (int) src_ss[0].axis, tensor->src[1]->name, (int) src_ss[1].axis);
        //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
    };

    auto handle_reshape = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        switch (src_ss[0].axis) {
            case GGML_BACKEND_SPLIT_AXIS_0:
            case GGML_BACKEND_SPLIT_AXIS_1:
            case GGML_BACKEND_SPLIT_AXIS_2:
            case GGML_BACKEND_SPLIT_AXIS_3: {
                GGML_ASSERT(src_ss[0].n_segments == 1);
                if (src_ss[0].axis == ggml_n_dims(tensor->src[0]) - 1 && src_ss[0].nr[0] == 1) {
                    return {ggml_backend_meta_split_axis(ggml_n_dims(tensor) - 1), {0}, {1}, 1};
                }
                int64_t base_ne_in = tensor->src[0]->ne[0];
                for (int dim = 1; dim <= src_ss[0].axis; dim++) {
                    base_ne_in *= tensor->src[0]->ne[dim];
                }
                base_ne_in /= src_ss[0].nr[0];
                int64_t base_ne_out = 1;
                for (int dim = 0; dim < GGML_MAX_DIMS; dim++) {
                    const int64_t base_ne_out_next = base_ne_out *= tensor->ne[dim];
                    if (base_ne_out_next % base_ne_in == 0) {
                        return {ggml_backend_meta_split_axis(dim), {0}, {uint32_t(base_ne_out_next/base_ne_in)}, 1};
                    }
                    if (base_ne_out_next > base_ne_in) {
                        GGML_ASSERT(src_ss[0].n_segments == 1);
                        GGML_ASSERT(src_ss[0].nr[0]      == 1);
                        return {ggml_backend_meta_split_axis(dim), {0}, {1}, 1};
                    }
                    base_ne_out = base_ne_out_next;
                }
                GGML_ABORT("shape mismatch for %s", ggml_op_name(tensor->op));
            }
            case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                return src_ss[0];
            }
            default: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    auto handle_cpy = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            return handle_reshape(src_ss);
        }
        return handle_generic(src_ss, /*scalar_only =*/ false);
    };

    auto handle_view = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (ggml_is_contiguous(tensor) && ggml_is_contiguous(tensor->src[0])) {
            return handle_reshape(src_ss);
        }
        const int axis = src_ss[0].axis;
        {
            bool all_strides_the_same = true;
            for (int dim = 0; dim < GGML_MAX_DIMS; dim++) {
                if (tensor->ne[dim] == 1 && tensor->src[0]->ne[dim] == 1) {
                    continue;
                }
                if (tensor->nb[dim] != tensor->src[0]->nb[dim]) {
                    all_strides_the_same = false;
                    break;
                }
            }
            if (all_strides_the_same) {
                return src_ss[0];
            }
        }
        if (!ggml_is_permuted(tensor) && !ggml_is_permuted(tensor->src[0]) && axis >= 0 && axis < GGML_MAX_DIMS-1) {
            for (int dim = 0; dim < GGML_MAX_DIMS-1; dim++) {
                if (tensor->nb[dim+1] == tensor->src[0]->nb[axis+1]) {
                    return {ggml_backend_meta_split_axis(dim), {0}, {1}, 1};
                }
            }
            GGML_ABORT("fatal error");
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED || src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
            return src_ss[0];
        }
        GGML_ABORT("view of permuted tensor not implemented");
        //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
    };

    auto handle_permute = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        switch (src_ss[0].axis) {
            case GGML_BACKEND_SPLIT_AXIS_0:
            case GGML_BACKEND_SPLIT_AXIS_1:
            case GGML_BACKEND_SPLIT_AXIS_2:
            case GGML_BACKEND_SPLIT_AXIS_3: {
                GGML_ASSERT(src_ss[0].n_segments == 1 || src_ss[0].nr[0] == 1);
                return {ggml_backend_meta_split_axis(tensor->op_params[src_ss[0].axis]), {0}, {src_ss[0].nr[0]}, 1};
            }
            case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                return src_ss[0];
            }
            default: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    auto handle_transpose = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        switch (src_ss[0].axis) {
            case GGML_BACKEND_SPLIT_AXIS_0:
            case GGML_BACKEND_SPLIT_AXIS_1: {
                GGML_ASSERT(src_ss[0].n_segments == 1 || src_ss[0].nr[0] == 1);
                return {ggml_backend_meta_split_axis(int(src_ss[0].axis) ^ 1), {0}, {src_ss[0].nr[0]}, 1};
            }
            case GGML_BACKEND_SPLIT_AXIS_2:
            case GGML_BACKEND_SPLIT_AXIS_3:
            case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                return src_ss[0];
            }
            default: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    auto handle_get_rows = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        return handle_generic(src_ss, /*scalar_only =*/ true);
    };

    auto handle_set_rows = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        GGML_ASSERT(split_states_equal(src_ss[0], src_ss[2]));
        return src_ss[0];
    };

    auto handle_rope = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        return src_ss[0];
    };

    auto handle_pad = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(tensor->op_params[2*src_ss[0].axis + 0] == 0);
            GGML_ASSERT(tensor->op_params[2*src_ss[0].axis + 1] == 0);
        }
        return src_ss[0];
    };

    auto handle_flash_attn_ext = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(tensor->src[3] == nullptr || src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            GGML_ASSERT(src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            GGML_ASSERT(tensor->src[4] == nullptr || src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
        }

        GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_2);
        const bool kv_split = src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_2 &&
                src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_2;
        const bool kv_mirrored = src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED;
        GGML_ASSERT(kv_split || kv_mirrored);
        GGML_ASSERT(tensor->src[4] == nullptr || src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_0);
        return {GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1};
    };

    auto handle_lightning_indexer = [&](
            const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        for (size_t i = 0; i < 4; i++) {
            GGML_ASSERT(src_ss[i].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        }
        return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
    };

    auto handle_flash_attn_ext_banded = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(                             src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_2);
        GGML_ASSERT(                             src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_2);
        GGML_ASSERT(                             src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_2);
        GGML_ASSERT(tensor->src[3] == nullptr || src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        // rel_logits is [E, H, Q, B], so its head shard is axis 1.
        GGML_ASSERT(                             src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_1);
        return {GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1};
    };

    auto handle_ssm_conv = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == src_ss[1].axis) {
            if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0) {
                return {GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1};
            }
            if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1) {
                return {GGML_BACKEND_SPLIT_AXIS_0, {0}, {1}, 1};
            }
        }
        return handle_generic(src_ss, /*scalar_only =*/ false);
    };

    auto handle_gated_delta_net = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_1);
        // state shape is [S_v, S_v, H_v, n_seqs] (s0 only); the heads dim is its own axis 2,
        // so a head-aligned split on the input cache lands on axis 2 here.
        GGML_ASSERT(src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_2 || src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_1 || src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_0);
        return {GGML_BACKEND_SPLIT_AXIS_0, {0}, {1}, 1};
    };

    auto calculate_split_state = [&]() -> ggml_backend_meta_split_state {
        if (ggml_nelements(tensor) == 0) {
            return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        if (ggml_backend_buffer_get_usage(tensor->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE && tensor->view_src == nullptr) {
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(tensor->buffer));
            const ggml_backend_meta_device_context * dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
            ggml_backend_meta_split_state ret = dev_ctx->get_split_state(tensor, dev_ctx->get_split_state_ud);
            if (ret.axis >= 0 && ret.axis <= GGML_MAX_DIMS) {
                const int64_t granularity = ret.axis == GGML_BACKEND_SPLIT_AXIS_0 ? ggml_blck_size(tensor->type) : 1;
                int64_t ne_sum = 0;
                for (size_t s = 0; s < ret.n_segments; s++) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        GGML_ASSERT(ret.ne[s*n_bufs + j] % granularity == 0);
                        ne_sum += ret.ne[s*n_bufs + j] * ret.nr[s];
                    }
                }
                GGML_ASSERT(ne_sum == tensor->ne[ret.axis]);
            }
            return ret;
        }

        std::vector<ggml_backend_meta_split_state> src_ss(GGML_MAX_SRC, {GGML_BACKEND_SPLIT_AXIS_NONE, {0}, {1}, 1});
        for (size_t i = 0; i < GGML_MAX_SRC; i++) {
            if (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                src_ss[i] = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
                continue;
            }
            src_ss[i] = ggml_backend_meta_get_split_state(stc, tensor->src[i], /*assume_sync =*/ true);
            if (src_ss[i].axis == GGML_BACKEND_SPLIT_AXIS_UNKNOWN) {
                // Isolated read-only diagnostic for genuine Qwen4Exp MTP
                // GDN ingredient replay on five-RX6800 Meta TP5. Never
                // replace the original fail-closed unknown-axis assertion.
                const ggml_tensor * bad = tensor->src[i];
                GGML_LOG_ERROR(
                    "TP5_MTP_GDN_META_UNKNOWN_SPLIT dst_name='%s' dst_op=%s dst_shape=[%lld,%lld,%lld,%lld] source_index=%zu src_name='%s' src_op=%s src_shape=[%lld,%lld,%lld,%lld] src_view_of='%s' src_nelements=%lld src_is_meta=%u src_has_buffer=%u\n",
                    ggml_get_name(tensor), ggml_op_name(tensor->op),
                    (long long)tensor->ne[0], (long long)tensor->ne[1], (long long)tensor->ne[2], (long long)tensor->ne[3],
                    i, ggml_get_name(bad), ggml_op_name(bad->op),
                    (long long)bad->ne[0], (long long)bad->ne[1], (long long)bad->ne[2], (long long)bad->ne[3],
                    bad->view_src ? ggml_get_name(bad->view_src) : "(none)",
                    (long long)ggml_nelements(bad), unsigned(bad->buffer && ggml_backend_buffer_is_meta(bad->buffer)),
                    unsigned(bad->buffer != nullptr));
            }
            GGML_ASSERT(src_ss[i].axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
        }

        ggml_backend_meta_split_state split_state;
        switch (tensor->op) {
            case GGML_OP_NONE: {
                split_state = {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
            } break;
            case GGML_OP_DUP: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_ADD:
            case GGML_OP_ADD_ID: {
                split_state = handle_bin_bcast(src_ss);
            } break;
            case GGML_OP_ADD1:
            case GGML_OP_ACC: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SUB:
            case GGML_OP_MUL:
            case GGML_OP_DIV: {
                split_state = handle_bin_bcast(src_ss);
            } break;
            case GGML_OP_SQR:
            case GGML_OP_SQRT:
            case GGML_OP_LOG:
            case GGML_OP_SIN:
            case GGML_OP_COS: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_SUM: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SUM_ROWS:
            case GGML_OP_CUMSUM:
            case GGML_OP_MEAN:
            case GGML_OP_ARGMAX:
            case GGML_OP_COUNT_EQUAL: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_REPEAT:
            case GGML_OP_REPEAT_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_CONCAT: {
                split_state = handle_concat(src_ss);
            } break;
            case GGML_OP_SILU_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_NORM:
            case GGML_OP_RMS_NORM:
            case GGML_OP_RMS_NORM_BACK:
            case GGML_OP_GROUP_NORM:
            case GGML_OP_L2_NORM: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_MUL_MAT:
            case GGML_OP_MUL_MAT_ID: {
                split_state = handle_mul_mat(src_ss);
            } break;
            case GGML_OP_OUT_PROD: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SCALE: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_SET: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_CPY: {
                split_state = handle_cpy(src_ss);
            } break;
            case GGML_OP_CONT:
            case GGML_OP_RESHAPE: {
                split_state = handle_reshape(src_ss);
            } break;
            case GGML_OP_VIEW: {
                split_state = handle_view(src_ss);
            } break;
            case GGML_OP_PERMUTE: {
                split_state = handle_permute(src_ss);
            } break;
            case GGML_OP_TRANSPOSE: {
                split_state = handle_transpose(src_ss);
            } break;
            case GGML_OP_GET_ROWS: {
                split_state = handle_get_rows(src_ss);
            } break;
            case GGML_OP_GET_ROWS_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SET_ROWS: {
                split_state = handle_set_rows(src_ss);
            } break;
            case GGML_OP_DIAG:
            case GGML_OP_DIAG_MASK_INF:
            case GGML_OP_DIAG_MASK_ZERO: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SOFT_MAX: {
                // A shared causal mask broadcasts over rank-local heads.
                // Never split the row's normalization/reduction dimension.
                GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0);
                split_state = tensor->src[1] == nullptr && tensor->src[2] == nullptr ? handle_per_row(src_ss) :
                    tensor->src[1] != nullptr && tensor->src[2] == nullptr ? handle_bin_bcast(src_ss) :
                    handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_SOFT_MAX_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_ROPE: {
                split_state = handle_rope(src_ss);
            } break;
            case GGML_OP_ROPE_BACK: {
                split_state = handle_rope(src_ss);
            } break;
            case GGML_OP_CLAMP: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_CONV_TRANSPOSE_1D:
            case GGML_OP_IM2COL:
            case GGML_OP_IM2COL_BACK:
            case GGML_OP_IM2COL_3D:
            case GGML_OP_CONV_2D:
            case GGML_OP_CONV_3D:
            case GGML_OP_CONV_2D_DW:
            case GGML_OP_CONV_TRANSPOSE_2D:
            case GGML_OP_POOL_1D:
            case GGML_OP_POOL_2D:
            case GGML_OP_POOL_2D_BACK:
            case GGML_OP_UPSCALE: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_PAD: {
                split_state = handle_pad(src_ss);
            } break;
            case GGML_OP_PAD_REFLECT_1D:
            case GGML_OP_ROLL:
            case GGML_OP_ARANGE:
            case GGML_OP_TIMESTEP_EMBEDDING: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_ARGSORT:
            case GGML_OP_TOP_K: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_LEAKY_RELU: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_TRI: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_FILL: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_FLASH_ATTN_EXT: {
                split_state = handle_flash_attn_ext(src_ss);
            } break;
            case GGML_OP_FLASH_ATTN_EXT_BANDED: {
                split_state = handle_flash_attn_ext_banded(src_ss);
            } break;
            case GGML_OP_FLASH_ATTN_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SSM_CONV: {
                split_state = handle_ssm_conv(src_ss);
            } break;
            case GGML_OP_SSM_SCAN:
            case GGML_OP_WIN_PART:
            case GGML_OP_WIN_UNPART:
            case GGML_OP_GET_REL_POS:
            case GGML_OP_ADD_REL_POS:
            case GGML_OP_RWKV_WKV6:
            case GGML_OP_GATED_LINEAR_ATTN:
            case GGML_OP_RWKV_WKV7:
            case GGML_OP_SOLVE_TRI: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_GATED_DELTA_NET: {
                split_state = handle_gated_delta_net(src_ss);
            } break;
            case GGML_OP_LIGHTNING_INDEXER: {
                split_state = handle_lightning_indexer(src_ss);
            } break;
            case GGML_OP_DSV4_HC_COMB:
            case GGML_OP_DSV4_HC_PRE:
            case GGML_OP_DSV4_HC_POST: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_UNARY: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_MAP_CUSTOM1:
            case GGML_OP_MAP_CUSTOM2:
            case GGML_OP_MAP_CUSTOM3:
            case GGML_OP_CUSTOM: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_CROSS_ENTROPY_LOSS:
            case GGML_OP_CROSS_ENTROPY_LOSS_BACK: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_OPT_STEP_ADAMW:
            case GGML_OP_OPT_STEP_SGD:
            case GGML_OP_GLU: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_TURBO_WHT: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            default: {
                GGML_ABORT("ggml op not implemented: %s", ggml_op_name(tensor->op));
                split_state = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            } break;
        }
        if (split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS) {
            bool first_src_split_by_axis = true;
            const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);

            for (size_t i = 0; i < GGML_MAX_SRC; i++) {
                if (tensor->src[i] == nullptr || src_ss[i].axis < 0 || src_ss[i].axis >= GGML_MAX_DIMS) {
                    continue;
                }
                if (first_src_split_by_axis) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        // Take over ratio from src:
                        for (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            split_state.ne[s*n_bufs + j] = 0;
                        }
                        for (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            split_state.ne[j] += src_ss[i].ne[s*n_bufs + j] * src_ss[i].nr[s];
                        }
                        split_state.ne[j] *= tensor->ne[split_state.axis];
                        if (split_state.ne[j] != 0 || tensor->src[i]->ne[src_ss[i].axis] != 0) {
                            const int64_t div = tensor->src[i]->ne[src_ss[i].axis] * split_state.nr[0];
                            GGML_ASSERT(split_state.ne[j] % div == 0);
                            split_state.ne[j] /= div;
                        }
                    }
                } else {
                    GGML_ASSERT(split_state.n_segments == 1);
                    for (size_t j = 0; j < n_bufs; j++) {
                        // Assert that ratio is consistent:
                        int64_t sum = 0;
                        for (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            sum += src_ss[i].ne[s*n_bufs + j] * src_ss[i].nr[s];
                        }
                        GGML_ASSERT(split_state.ne[j]*split_state.nr[0] * tensor->src[i]->ne[src_ss[i].axis]
                                                                 == sum * tensor->ne[split_state.axis]);
                    }
                }
                first_src_split_by_axis = false;
            }
            GGML_ASSERT(!first_src_split_by_axis);
        }
        return split_state;
    };

    const std::pair key = std::make_pair(tensor, assume_sync);
    ggml_backend_meta_split_state ret;
    bool cached = false;
    {
        std::lock_guard<std::mutex> lock(buf_ctx->split_state_mutex);
        auto it = buf_ctx->split_state_cache.find(key);
        if (it != buf_ctx->split_state_cache.end()) {
            if (memcmp(it->second.second, tensor, sizeof(it->second.second)) != 0) {
                buf_ctx->split_state_cache.clear();
            } else {
                ret = it->second.first;
                cached = true;
            }
        }
    }

    if (!cached) {
        // Recursion can cross buffers in either direction (for example a KV
        // write refers to its cache destination). Never nest buffer locks.
        ret = calculate_split_state();
        {
            std::lock_guard<std::mutex> lock(buf_ctx->split_state_mutex);
            auto & entry = buf_ctx->split_state_cache[key];
            entry.first = ret;
            memcpy(entry.second, tensor, sizeof(entry.second));
        }
        if (buf_ctx->debug > 0) {
            std::string srcs_info;
            for (size_t i = 0; i < GGML_MAX_SRC; i++) {
                if (tensor->src[i] == nullptr) {
                    continue;
                }
                if (!srcs_info.empty()) {
                    srcs_info += ", ";
                }
                const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor->src[0], true);
                GGML_ASSERT(split_state.n_segments == 1);
                const char * axis_name = ggml_backend_meta_split_axis_name(split_state.axis);
                std::string ne_info;
                for (size_t j = 0; j < n_bufs; j++) {
                    if (!ne_info.empty()) {
                        ne_info += ", ";
                    }
                    ne_info += std::to_string(split_state.ne[j]) + "x" + std::to_string(split_state.nr[0]);
                }
                srcs_info += std::string(tensor->src[i]->name) + "[" + ggml_op_name(tensor->src[i]->op) + ", " + axis_name + ", {" + ne_info + "}]";
            }
            std::string ne_info;
            for (size_t j = 0; j < n_bufs; j++) {
                if (!ne_info.empty()) {
                    ne_info += ", ";
                }
                ne_info += std::to_string(ret.ne[j]) + "x" + std::to_string(ret.nr[0]);
            }
            GGML_LOG_DEBUG("SPLIT_STATE: {%s} -> %s[%s, %s, {%s}]\n", srcs_info.c_str(), tensor->name, ggml_op_name(tensor->op),
                ggml_backend_meta_split_axis_name(ret.axis), ne_info.c_str());
        }
    }

    GGML_ASSERT(ret.axis != GGML_BACKEND_SPLIT_AXIS_NONE);
#ifndef NDEBUG
    if (ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
        int64_t ne_ret = 0;
        for (size_t s = 0; s < ret.n_segments; s++) {
            for (size_t j = 0; j < n_bufs; j++) {
                ne_ret += ret.ne[s*n_bufs + j] * ret.nr[s];
            }
        }
        assert(ne_ret == tensor->ne[int(ret.axis)]);
    }
#endif // NDEBUG
    return ret;
}

static struct ggml_backend_meta_split_state ggml_backend_meta_get_split_state(const struct ggml_tensor * tensor, bool assume_sync) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(tensor->buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    return ggml_backend_meta_get_split_state(buf_ctx->get_simple_tensor_container(tensor), tensor, assume_sync);
}

static void * ggml_backend_meta_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_UNUSED(buffer);
    return (void *) 0x1000000000000000; // FIXME
}

static enum ggml_status ggml_backend_meta_buffer_init_tensor_impl(ggml_backend_meta_simple_tensor_container & stc, ggml_tensor * tensor) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(tensor->buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    const size_t n_simple_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(stc, tensor, /*assume_sync =*/ true);
    GGML_ASSERT(ggml_nelements(tensor) == 0 || split_state.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
    GGML_ASSERT(split_state.n_segments <= 16);

    int split_dim = split_state.axis;
    int64_t ne[GGML_MAX_DIMS];
    size_t  nb[GGML_MAX_DIMS];
    for (size_t k = 0; k < GGML_MAX_DIMS; k++) {
        ne[k] = tensor->ne[k];
        nb[k] = tensor->nb[k];
    }

    std::vector<ggml_tensor *> simple_tensors;
    simple_tensors.reserve(n_simple_bufs);
    for (size_t j = 0; j < n_simple_bufs; j++) {
        ggml_context          * simple_ctx = stc.ctxs[j].get();
        ggml_backend_buffer_t   simple_buf = buf_ctx->bufs[j].get();

        if ((simple_buf != nullptr) && ggml_backend_buffer_is_multi_buffer(simple_buf)) {
            // see https://github.com/ggml-org/llama.cpp/issues/22197
            GGML_ABORT("multi buffers are not supported by the meta backend");
        }

        if (split_dim >= 0 && split_dim < GGML_MAX_DIMS) {
            // TODO: the following assert fails for llama-parallel even though the results are correct:
            // GGML_ASSERT(ggml_is_contiguously_allocated(tensor));
            ne[split_dim] = 0;
            for (size_t s = 0; s < split_state.n_segments; s++) {
                ne[split_dim] += split_state.ne[s*n_simple_bufs + j] * split_state.nr[s];
            }
            for (int i = 0; i < GGML_MAX_DIMS; i++) {
                if (tensor->nb[i] > tensor->nb[split_dim]) {
                    nb[i] = tensor->nb[i] * ne[split_dim]/tensor->ne[split_dim];
                }
            }
        }

        ggml_tensor * t_ij = ggml_new_tensor(simple_ctx, tensor->type, GGML_MAX_DIMS, ne);
        t_ij->op = tensor->op;
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            t_ij->nb[i] = nb[i];
        }
        t_ij->flags = tensor->flags;
        memcpy(t_ij->op_params, tensor->op_params, sizeof(tensor->op_params));
        ggml_set_name(t_ij, tensor->name);
        t_ij->buffer = simple_buf;
        t_ij->view_src = tensor->view_src;
        t_ij->view_offs = tensor->view_offs;
        if (t_ij->view_src != nullptr && ggml_backend_buffer_is_meta(t_ij->view_src->buffer)) {
            t_ij->view_src = ggml_backend_meta_buffer_simple_tensor(tensor->view_src, j);
            if (t_ij->view_offs > 0 && split_dim >= 0 && split_dim < GGML_MAX_DIMS) {
                GGML_ASSERT(tensor->ne[split_dim] != 0);
                const int split_dim_view_src = ggml_backend_meta_get_split_state(tensor->view_src, /*assume_sync =*/ true).axis;
                GGML_ASSERT(split_dim_view_src >= 0 && split_dim_view_src < GGML_MAX_DIMS);

                // The offset can be internal to the data split, in those cases the view offset should not be scaled.
                // If however, the offset is larger than the data split then it needs to be scaled proportionally.
                bool split_internal_offset = t_ij->view_offs <= tensor->view_src->nb[split_dim_view_src];
                for (int i = 0; i < GGML_MAX_DIMS; i++) {
                    const size_t dim_size = tensor->ne[i] * tensor->nb[i];
                    if (tensor->view_offs <= dim_size && dim_size < tensor->nb[split_dim]) {
                        split_internal_offset = true;
                        break;
                    }
                }
                if (!split_internal_offset) {
                    t_ij->view_offs = t_ij->view_offs * ne[split_dim]/tensor->ne[split_dim];
                }
            }
        }
        if (t_ij->view_src != nullptr) {
            t_ij->data = (char *) t_ij->view_src->data + t_ij->view_offs;
        } else if (simple_buf != nullptr) {
            t_ij->data = (char *) ggml_backend_buffer_get_base(simple_buf)
                + size_t(tensor->data) - size_t(ggml_backend_buffer_get_base(tensor->buffer));
        }
        t_ij->extra = tensor->extra;
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            t_ij->src[i] = tensor->src[i];
            if (tensor->src[i] == tensor) {
                t_ij->src[i] = t_ij;
            } else if (t_ij->src[i] != nullptr && ggml_backend_buffer_is_meta(t_ij->src[i]->buffer)) {
                t_ij->src[i] = ggml_backend_meta_buffer_simple_tensor(tensor->src[i], j);
            }
        }

        simple_tensors.push_back(t_ij);
    }

    // If one of the sources has a zero-sized slice, disable the computation:
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (tensor->src[i] == nullptr || !ggml_backend_buffer_is_meta(tensor->src[i]->buffer)) {
            continue;
        }

        const ggml_backend_meta_split_state split_state_src = ggml_backend_meta_get_split_state(tensor->src[i], /*assume_sync =*/ true);
        if (split_state_src.axis < 0 || split_state_src.axis >= GGML_MAX_DIMS) {
            continue;
        }
        for (size_t j = 0; j < n_simple_bufs; j++) {
            int64_t ne_sum = 0;
            for (size_t s = 0; s < split_state_src.n_segments; s++) {
                ne_sum += split_state_src.ne[s*n_simple_bufs + j] * split_state_src.nr[s];
            }
            if (ne_sum == 0) {
                simple_tensors[j]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
            }
        }
    }

    stc.simple_tensors[tensor] = simple_tensors;

    return GGML_STATUS_SUCCESS;
}

static enum ggml_status ggml_backend_meta_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    buf_ctx->stc_compute_index = buf_ctx->stc_compute_index_next;
    return ggml_backend_meta_buffer_init_tensor_impl(buf_ctx->get_simple_tensor_container(tensor), tensor);
}

static void ggml_backend_meta_buffer_memset_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(buffer);
    const ggml_backend_meta_split_state split_state =
            ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(ggml_is_contiguous(tensor) || split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

    if (split_state.n_segments != 1 || split_state.nr[0] != 1) {
        GGML_ASSERT(split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(split_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        std::vector<size_t> simple_offsets(n_bufs, 0);
        if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            const size_t row_stride = tensor->nb[1];
            GGML_ASSERT(offset % row_stride == 0);
            GGML_ASSERT(size   % row_stride == 0);
            const int64_t row_start = offset / row_stride;
            const int64_t row_count = size   / row_stride;
            GGML_ASSERT(row_start + row_count <= tensor->ne[1]);

            const int64_t blck_size = ggml_blck_size(tensor->type);
            for (size_t s = 0; s < split_state.n_segments; s++) {
                for (size_t r = 0; r < split_state.nr[s]; r++) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                        GGML_ASSERT(split_state.ne[s*n_bufs + j] % blck_size == 0);
                        const size_t nbytes = split_state.ne[s*n_bufs + j]/blck_size * tensor->nb[0];
                        for (int64_t row = 0; row < row_count; row++) {
                            ggml_backend_tensor_memset(simple_tensor, value,
                                    simple_offsets[j] + (row_start + row)*simple_tensor->nb[1], nbytes);
                        }
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            return;
        }

        GGML_ASSERT(split_state.axis == GGML_BACKEND_SPLIT_AXIS_1);

        const size_t row_stride = tensor->nb[2];
        GGML_ASSERT(offset % row_stride == 0);
        GGML_ASSERT(size   % row_stride == 0);
        const int64_t row_start = offset / row_stride;
        const int64_t row_count = size   / row_stride;
        GGML_ASSERT(row_start + row_count <= tensor->ne[2]);

        for (size_t s = 0; s < split_state.n_segments; s++) {
            for (size_t r = 0; r < split_state.nr[s]; r++) {
                for (size_t j = 0; j < n_bufs; j++) {
                    ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    const size_t nbytes = split_state.ne[s*n_bufs + j] * tensor->nb[1];
                    for (int64_t row = 0; row < row_count; row++) {
                        ggml_backend_tensor_memset(simple_tensor, value,
                                simple_offsets[j] + (row_start + row)*simple_tensor->nb[2], nbytes);
                    }
                    simple_offsets[j] += nbytes;
                }
            }
        }
        return;
    }

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset         / chunk_size_full;
            const int64_t i_stop  = (offset + size) / chunk_size_full;
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size = simple_tensor->nb[split_state.axis + 1];
                if (chunk_size == 0) {
                    continue;
                }
                for (int64_t i = i_start; i < i_stop; i++) {
                    ggml_backend_tensor_memset(simple_tensor, value, i*chunk_size, chunk_size);
                }
            }
        } break;
        case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
            GGML_ASSERT(value == 0);
            [[fallthrough]];
        }
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ggml_backend_tensor_memset(simple_tensor, value, offset, size);
            }
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static void ggml_backend_meta_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(buffer);
    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(ggml_is_contiguous(tensor) || split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

    if (split_state.n_segments != 1 || split_state.nr[0] != 1) {
        GGML_ASSERT(split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(split_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        size_t offset_data = 0;
        std::vector<size_t> simple_offsets(n_bufs, 0);
        if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            const size_t row_stride = tensor->nb[1];
            GGML_ASSERT(offset % row_stride == 0);
            GGML_ASSERT(size   % row_stride == 0);
            const int64_t row_start = offset / row_stride;
            const int64_t row_count = size   / row_stride;
            GGML_ASSERT(row_start + row_count <= tensor->ne[1]);

            const int64_t blck_size = ggml_blck_size(tensor->type);
            for (size_t s = 0; s < split_state.n_segments; s++) {
                for (size_t r = 0; r < split_state.nr[s]; r++) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                        GGML_ASSERT(split_state.ne[s*n_bufs + j] % blck_size == 0);
                        const size_t nbytes = split_state.ne[s*n_bufs + j]/blck_size * tensor->nb[0];
                        ggml_backend_tensor_set_2d(simple_tensor, (const char *) data + offset_data,
                            simple_offsets[j] + row_start * simple_tensor->nb[1], nbytes,
                            row_count, simple_tensor->nb[1], tensor->nb[1]);
                        offset_data       += nbytes;
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            GGML_ASSERT(offset_data*row_count == size);
            return;
        }
        GGML_ASSERT(split_state.axis == GGML_BACKEND_SPLIT_AXIS_1);

        const size_t row_stride = tensor->nb[2];
        GGML_ASSERT(offset % row_stride == 0);
        GGML_ASSERT(size   % row_stride == 0);
        const int64_t row_start = offset / row_stride;
        const int64_t row_count = size   / row_stride;
        GGML_ASSERT(row_start + row_count <= tensor->ne[2]);

        for (size_t s = 0; s < split_state.n_segments; s++) {
            for (size_t r = 0; r < split_state.nr[s]; r++) {
                for (size_t j = 0; j < n_bufs; j++) {
                    ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    const size_t nbytes = split_state.ne[s*n_bufs + j] * tensor->nb[1];
                    ggml_backend_tensor_set_2d(simple_tensor, (const char *) data + offset_data,
                        simple_offsets[j] + row_start * simple_tensor->nb[2], nbytes,
                        row_count, simple_tensor->nb[2], tensor->nb[2]);
                    offset_data       += nbytes;
                    simple_offsets[j] += nbytes;
                }
            }
        }
        GGML_ASSERT(offset_data*row_count == size);
        return;
    }

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        /chunk_size_full;
            const int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                if (chunk_size_j == 0) {
                    continue;
                }
                const size_t simple_offset = i_start * chunk_size_j;
                ggml_backend_tensor_set_2d(simple_tensor, (const char *) data + offset_j, simple_offset, chunk_size_j, i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ggml_backend_tensor_set(simple_tensor, data, offset, size);
            }
        } break;
        case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
            GGML_ASSERT(tensor->type == GGML_TYPE_F32);
            const int64_t ne = ggml_nelements(tensor);
            std::vector<float> tmp;
            tmp.reserve(ne);
            for (int64_t i = 0; i < ne; i++) {
                tmp.push_back(((const float *) data)[i] / n_bufs);
            }
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ggml_backend_tensor_set(simple_tensor, tmp.data(), offset, size);
            }
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static void ggml_backend_meta_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(buffer);
    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(ggml_is_contiguous(tensor) || split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

    if (split_state.n_segments != 1 || split_state.nr[0] != 1) {
        GGML_ASSERT(split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(split_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        size_t offset_data = 0;
        std::vector<size_t> simple_offsets(n_bufs, 0);
        if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            const size_t row_stride = tensor->nb[1];
            GGML_ASSERT(offset % row_stride == 0);
            GGML_ASSERT(size   % row_stride == 0);
            const int64_t row_start = offset / row_stride;
            const int64_t row_count = size   / row_stride;
            GGML_ASSERT(row_start + row_count <= tensor->ne[1]);

            const int64_t blck_size = ggml_blck_size(tensor->type);
            for (size_t s = 0; s < split_state.n_segments; s++) {
                for (size_t r = 0; r < split_state.nr[s]; r++) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                        GGML_ASSERT(split_state.ne[s*n_bufs + j] % blck_size == 0);
                        const size_t nbytes = split_state.ne[s*n_bufs + j]/blck_size * tensor->nb[0];
                        ggml_backend_tensor_get_2d(simple_tensor, (char *) data + offset_data,
                            simple_offsets[j] + row_start * simple_tensor->nb[1], nbytes,
                            row_count, simple_tensor->nb[1], tensor->nb[1]);
                        offset_data       += nbytes;
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            GGML_ASSERT(offset_data*row_count == size);
            return;
        }
        GGML_ASSERT(split_state.axis == GGML_BACKEND_SPLIT_AXIS_1);

        const size_t row_stride = tensor->nb[2];
        GGML_ASSERT(offset % row_stride == 0);
        GGML_ASSERT(size   % row_stride == 0);
        const int64_t row_start = offset / row_stride;
        const int64_t row_count = size   / row_stride;
        GGML_ASSERT(row_start + row_count <= tensor->ne[2]);

        for (size_t s = 0; s < split_state.n_segments; s++) {
            for (size_t r = 0; r < split_state.nr[s]; r++) {
                for (size_t j = 0; j < n_bufs; j++) {
                    const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    const size_t nbytes = split_state.ne[s*n_bufs + j] * tensor->nb[1];
                    ggml_backend_tensor_get_2d(simple_tensor, (char *) data + offset_data,
                        simple_offsets[j] + row_start * simple_tensor->nb[2], nbytes,
                        row_count, simple_tensor->nb[2], tensor->nb[2]);
                    offset_data       += nbytes;
                    simple_offsets[j] += nbytes;
                }
            }
        }
        GGML_ASSERT(offset_data*row_count == size);
        return;
    }

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        /chunk_size_full;
            const int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            for (size_t j = 0; j < n_bufs; j++){
                const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                if (chunk_size_j == 0) {
                    continue;
                }
                const size_t simple_offset = i_start * chunk_size_j;
                ggml_backend_tensor_get_2d(simple_tensor, (char *) data + offset_j, simple_offset, chunk_size_j, i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            // TODO other simple backend may be better
            const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, 0);
            ggml_backend_tensor_get(simple_tensor, data, offset, size);
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static void ggml_backend_meta_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    const size_t n_buffers = ggml_backend_meta_buffer_n_bufs(buffer);
    for (size_t i = 0; i < n_buffers; i++) {
        ggml_backend_buffer_clear(ggml_backend_meta_buffer_simple_buffer(buffer, i), value);
    }
}

static void ggml_backend_meta_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    for (size_t i = 0; i < buf_ctx->bufs.size(); i++) {
        ggml_backend_buffer_reset(ggml_backend_meta_buffer_simple_buffer(buffer, i));
    }
}

static const ggml_backend_buffer_i ggml_backend_meta_buffer_iface = {
    /* .free_buffer     = */ ggml_backend_meta_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_meta_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_meta_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_meta_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_meta_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_meta_buffer_get_tensor,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ ggml_backend_meta_buffer_clear,
    /* .reset           = */ ggml_backend_meta_buffer_reset,
};

bool ggml_backend_buffer_is_meta(ggml_backend_buffer_t buf) {
    return buf != nullptr && buf->iface.free_buffer == ggml_backend_meta_buffer_iface.free_buffer;
}

static ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);

    const ggml_init_params params = {
        /*.mem_size   =*/ 1024*1024*ggml_tensor_overhead(), // FIXME
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_backend_meta_simple_tensor_container stc_static;
    ggml_backend_meta_simple_tensor_container stc_compute_0(params, n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_1(params, n_simple_bufts);

    size_t max_size = 0;
    std::vector<ggml_backend_buffer_t> bufs;
    bufs.reserve(n_simple_bufts);
    for (size_t i = 0; i < n_simple_bufts; i++) {
        bufs.push_back(ggml_backend_buft_alloc_buffer(ggml_backend_meta_buft_simple_buft(buft, i), size));
        GGML_ASSERT(bufs.back() != nullptr);
        max_size = std::max(max_size, ggml_backend_buffer_get_size(bufs.back()));
    }
    ggml_backend_meta_buffer_context * buf_ctx = new ggml_backend_meta_buffer_context(stc_static, stc_compute_0, stc_compute_1, bufs);

    return ggml_backend_buffer_init(buft, ggml_backend_meta_buffer_iface, buf_ctx, max_size);
}

struct ggml_backend_buffer * ggml_backend_meta_alloc_ctx_tensors_from_buft(struct ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);

    constexpr size_t compute_headroom = 16; // Maximum number of views per statically allocated tensor that can be created between evals.
    const ggml_init_params params_static = {
        /*.mem_size   =*/ ggml_get_mem_size(ctx),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    const ggml_init_params params_compute = {
        /*.mem_size   =*/ compute_headroom*ggml_get_mem_size(ctx),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_backend_meta_simple_tensor_container stc_static   (params_static,  n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_0(params_compute, n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_1(params_compute, n_simple_bufts);

    std::vector<ggml_backend_buffer_t> bufs(n_simple_bufts, nullptr);
    ggml_backend_meta_buffer_context * meta_buf_ctx = new ggml_backend_meta_buffer_context(stc_static, stc_compute_0, stc_compute_1, bufs);

    ggml_backend_buffer_t meta_buf = ggml_backend_buffer_init(buft, ggml_backend_meta_buffer_iface, meta_buf_ctx, 0);
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        t->buffer = meta_buf;
        ggml_backend_meta_buffer_init_tensor_impl(meta_buf_ctx->stc_static, t);
        t->data = (void *) 0x2000000000000000; // FIXME
    }
    for (size_t i = 0; i < n_simple_bufts; i++) {
        ggml_context * ctx = meta_buf_ctx->stc_static.ctxs[i].get();
        ggml_backend_buffer_type_t simple_buft = ggml_backend_meta_buft_simple_buft(buft, i);

        // If a ggml_context only has zero-sized tensors, ggml_backend_alloc_ctx_tensors_from_buft returns NULL.
        // For those edge cases, allocate a dummy buffer instead.
        bool any_nonzero_slice = false;
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            if (ggml_nelements(t) != 0) {
                any_nonzero_slice = true;
                break;
            }
        }
        if (any_nonzero_slice) {
            meta_buf_ctx->bufs[i].reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx, simple_buft));
        } else {
            meta_buf_ctx->bufs[i].reset(ggml_backend_buft_alloc_buffer(simple_buft, 0));
            for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
                t->buffer = meta_buf_ctx->bufs[i].get();
            }
        }
        GGML_ASSERT(meta_buf_ctx->bufs[i]);
        meta_buf->size = std::max(meta_buf->size, ggml_backend_buffer_get_size(meta_buf_ctx->bufs[i].get()));
    }
    return meta_buf;
}

//
// meta backend
//

static ggml_guid_t ggml_backend_meta_guid() {
    static ggml_guid guid = {0xf1, 0x0e, 0x34, 0xcf, 0x9c, 0x6f, 0x43, 0xcb, 0x96, 0x92, 0xbe, 0x8e, 0xbb, 0x71, 0x3f, 0xda};
    return &guid;
}

struct ggml_backend_meta_context {
    struct cgraph_config {
        ggml_cgraph * cgraph_main = nullptr;
        int           offset      = 0; // Node offset vs. original graph

        std::vector<ggml_cgraph *> cgraphs_aux;
    };
    struct backend_config {
        ggml_backend_t backend;

        std::vector<cgraph_config>           cgraphs;
        std::vector<ggml_tensor *>           nodes;
        std::vector<ggml_backend_buffer_ptr> bufs;

        backend_config(ggml_backend_t backend, const size_t n_reduce_steps) : backend(backend) {
            bufs.resize(n_reduce_steps);
        }
    };
    std::string                 name;
    std::vector<backend_config> backend_configs;
    ggml_context_ptr            ctx;
    std::vector<ggml_cgraph *>  cgraphs_aux;
    std::vector<ggml_tensor *>  nodes_aux;
    size_t                      n_reduce_steps = 0;
    int                         max_nnodes    = 0;
    size_t                      max_tmp_size  = 0;
    size_t                      max_subgraphs = 0;
    size_t                      n_subgraphs   = 0;
    uint64_t                    uid           = 0;

    void *                               comm_ctx       = nullptr;
    ggml_backend_comm_allreduce_tensor_t comm_allreduce = nullptr;

    const ggml_relay_provider_i * relay_i = nullptr;
    const ggml_relay_wire_provider_i * relay_wire_i = nullptr;
    // Providers start in F32. Mirror only successful public format changes.
    ggml_relay_wire_type relay_wire_type = GGML_RELAY_WIRE_F32;
    void * relay = nullptr;
    bool relay_mode = false;
    bool cpu_staged_f32 = false;
    bool shared_expert_shard_started = false;
    bool shared_expert_shard_enabled = false;
    bool hc_router_owner_started = false;
    bool hc_router_owner_enabled = false;
    bool valid = false;
    // Physical HC weights remain shared and mirrored. These domains and child
    // flags belong only to this backend's converted compute graph.
    struct hc_owner_domain {
        ggml_backend_meta_split_state split;
        uint32_t rank;
    };
    bool hc_owner_configured = false;
    bool hc_owner_started = false;
    uint32_t hc_owner_offset = 0;
    std::unordered_map<const ggml_tensor *, hc_owner_domain> hc_owner_domains;
    std::vector<int> hc_owner_up_indices;
    std::vector<int> hc_owner_terminal_indices;
    // Authenticate only this context's previously qualified router/packet flags.
    // These are execution records, not shared physical split-state overrides.
    std::unordered_map<const ggml_tensor *, uint32_t> hc_router_owner_nodes;
    std::vector<ggml_tensor *> relay_terminals;
    std::vector<ggml_cgraph *> relay_graphs;
    std::vector<float> staged_partial;
    std::vector<float> staged_sum;

    ggml_backend_meta_context(ggml_backend_dev_t meta_dev, const char * params) {
        const size_t n_devs = ggml_backend_meta_dev_n_devs(meta_dev);
        relay_mode = params != nullptr && (std::strcmp(params, "tp5-relay") == 0 ||
                                          std::strcmp(params, "tp5-relay-native") == 0);
        cpu_staged_f32 = params != nullptr && std::strcmp(params, "quantized-f32") == 0;
        if (params != nullptr && params[0] != '\0' && !relay_mode && !cpu_staged_f32) {
            GGML_LOG_ERROR("%s: unsupported Meta parameters: %s\n", __func__, params);
            return;
        }
        if (relay_mode) {
            if (n_devs != GGML_RELAY_RANKS) {
                GGML_LOG_ERROR("%s: tp5-relay requires exactly five HIP devices\n", __func__);
                return;
            }
            ggml_backend_reg_t reg = nullptr;
            std::set<ggml_backend_dev_t> seen;
            for (size_t i = 0; i < n_devs; ++i) {
                ggml_backend_dev_t dev = ggml_backend_meta_dev_simple_dev(meta_dev, i);
                ggml_backend_reg_t child_reg = ggml_backend_dev_backend_reg(dev);
                if (child_reg == nullptr || std::strcmp(ggml_backend_reg_name(child_reg), "ROCm") != 0 ||
                    ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU || !seen.insert(dev).second ||
                    (reg != nullptr && reg != child_reg)) {
                    GGML_LOG_ERROR("%s: tp5-relay requires five distinct devices from one HIP registry\n", __func__);
                    return;
                }
                reg = child_reg;
            }
            auto get_interface = (ggml_backend_relay_get_interface_t)
                ggml_backend_reg_get_proc_address(reg, "ggml_backend_relay_get_interface");
            relay_i = get_interface != nullptr ? get_interface() : nullptr;
            if (relay_i == nullptr || relay_i->abi_version != GGML_RELAY_PROVIDER_ABI_VERSION ||
                !relay_i->create || !relay_i->destroy || !relay_i->prepare || !relay_i->submit ||
                !relay_i->poll || !relay_i->publish || !relay_i->snapshot || !relay_i->abort) {
                GGML_LOG_ERROR("%s: HIP tp5-relay capability is unavailable or incompatible\n", __func__);
                return;
            }
            auto get_wire_interface = (ggml_backend_relay_get_wire_interface_t)
                ggml_backend_reg_get_proc_address(reg, "ggml_backend_relay_get_wire_interface");
            relay_wire_i = get_wire_interface != nullptr ? get_wire_interface() : nullptr;
            if (relay_wire_i != nullptr &&
                    (relay_wire_i->abi_version != GGML_RELAY_WIRE_PROVIDER_ABI_VERSION ||
                     !relay_wire_i->set_wire_type || !relay_wire_i->poll_v2 || !relay_wire_i->publish_v2)) {
                relay_wire_i = nullptr; // Old F32 callers do not require the optional capability.
            }
        }
        n_reduce_steps = std::ceil(std::log2(n_devs));
        name = cpu_staged_f32 ? "Meta-quantized-f32-cpu-staged-ordered-f32(" : "Meta(";
        std::vector<ggml_backend_t> simple_backends;
        backend_configs.reserve(n_devs);
        simple_backends.reserve(n_devs);
        for (size_t i = 0; i < n_devs; i++) {
            ggml_backend_dev_t simple_dev = ggml_backend_meta_dev_simple_dev(meta_dev, i);
            if (cpu_staged_f32 && (ggml_backend_dev_type(simple_dev) != GGML_BACKEND_DEVICE_TYPE_GPU ||
                    std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(simple_dev)), "ROCm") != 0)) {
                GGML_LOG_ERROR("%s: quantized-f32 CPU-staged reference requires HIP GPU children\n", __func__);
                return;
            }
            if (i > 0) {
                name += ",";
            }
            name += ggml_backend_dev_name(simple_dev);
            simple_backends.push_back(ggml_backend_dev_init(simple_dev, params));
            if (simple_backends.back() == nullptr) {
                GGML_LOG_ERROR("%s: failed to initialize child backend\n", __func__);
                return;
            }
            backend_configs.emplace_back(simple_backends.back(), n_reduce_steps);
        }
        name += ")";

        if (relay_mode) {
            relay = relay_i->create(simple_backends.data(), simple_backends.size());
            valid = relay != nullptr;
            if (!valid) {
                GGML_LOG_ERROR("%s: HIP rejected the tp5-relay runtime profile\n", __func__);
            }
            return;
        }
        // This reference is selected before any graph submission. It must never
        // create a GPU communicator or be entered after a failed relay submit.
        if (cpu_staged_f32) {
            GGML_LOG_INFO("%s: quantized-f32 CPU-staged reference: %zu HIP ranks, original quantized weights, "
                          "F32 activations, rank-order CPU F32 SUM, synchronous host broadcast; no GPU allreduce\n",
                          __func__, n_devs);
            valid = true;
            return;
        }
        if (n_devs > 1) {
            ggml_backend_comm_init_t comm_init = (ggml_backend_comm_init_t) ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_init");
            if (comm_init != nullptr) {
                comm_ctx = comm_init(simple_backends.data(), simple_backends.size());
            }
        }
        if (comm_ctx != nullptr) {
            comm_allreduce = (ggml_backend_comm_allreduce_tensor_t)
                ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(
                    ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_allreduce_tensor");
            GGML_ASSERT(comm_allreduce != nullptr);
        }
        valid = true;
    }

    ~ggml_backend_meta_context() {
        if (comm_ctx != nullptr) {
            ggml_backend_comm_free_t comm_free = (ggml_backend_comm_free_t) ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_configs[0].backend)), "ggml_backend_comm_free");
            GGML_ASSERT(comm_free != nullptr);
            comm_free(comm_ctx);
        }
        for (auto & bc : backend_configs) {
            ggml_backend_free(bc.backend);
        }
    }
};

static const char * ggml_backend_meta_get_name(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    const ggml_backend_meta_context * backend_ctx = (const ggml_backend_meta_context *) backend->context;
    return backend_ctx->name.c_str();
}

static void ggml_backend_meta_free(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;
    if (backend_ctx->relay != nullptr) {
        if (!backend_ctx->relay_i->destroy(backend_ctx->relay)) {
            // A native reader may still own graph and child-backend storage.
            // Retain the entire Meta owner instead of freeing behind it.
            GGML_LOG_ERROR("%s: retaining tp5-relay owners: native completion is unproved\n", __func__);
            return;
        }
        backend_ctx->relay = nullptr;
    }
    delete backend_ctx;
    delete backend;
}

static void ggml_backend_meta_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(split_state.n_segments == 1);
    GGML_ASSERT(split_state.nr[0]      == 1);

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        /chunk_size_full;
            const int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            for (size_t j = 0; j < n_backends; j++){
                ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                if (chunk_size_j == 0) {
                    continue;
                }
                const size_t simple_offset = i_start * chunk_size_j;
                ggml_backend_tensor_set_2d_async(simple_backend, simple_tensor, (const char *) data + offset_j, simple_offset, chunk_size_j,
                    i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            for (size_t j = 0; j < n_backends; j++) {
                ggml_backend_tensor_set_async(
                    ggml_backend_meta_simple_backend(backend, j), ggml_backend_meta_buffer_simple_tensor(tensor, j), data, offset, size);
            }
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static bool ggml_backend_meta_cpy_tensor_async(
        ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    if (backend_src == nullptr || !ggml_backend_is_meta(backend_dst) ||
            !ggml_are_same_layout(src, dst) || !ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        return false;
    }

    const ggml_backend_buffer_t src_buf = src->view_src ? src->view_src->buffer : src->buffer;
    const ggml_backend_buffer_t dst_buf = dst->view_src ? dst->view_src->buffer : dst->buffer;
    // A CPU-child Meta buffer also reports host storage, but its public data
    // pointer is not a host allocation. Only ordinary host sources are valid.
    if (src_buf == nullptr || ggml_backend_buffer_is_meta(src_buf) || !ggml_backend_buffer_is_host(src_buf) ||
            !ggml_backend_buffer_is_meta(dst_buf) || dst->buffer != dst_buf ||
            ggml_backend_buft_get_device(ggml_backend_buffer_get_type(dst_buf)) != ggml_backend_get_device(backend_dst)) {
        return false;
    }

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(dst, /*assume_sync =*/ false);
    if (split_state.axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED || split_state.n_segments != 1 || split_state.nr[0] != 1) {
        return false;
    }

    // No rejection is allowed after this point: child uploads may already be
    // queued if a later child throws. The caller owns completion/source lifetime.
    ggml_backend_synchronize(backend_src);
    ggml_backend_meta_set_tensor_async(backend_dst, dst, src->data, 0, ggml_nbytes(src));
    return true;
}

static void ggml_backend_meta_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(split_state.n_segments == 1);
    GGML_ASSERT(split_state.nr[0]      == 1);

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        /chunk_size_full;
            const int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            for (size_t j = 0; j < n_backends; j++){
                ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
                const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                if (chunk_size_j == 0) {
                    continue;
                }
                const size_t simple_offset = i_start * chunk_size_j;
                ggml_backend_tensor_get_2d_async(simple_backend, simple_tensor, (char *) data + offset_j, simple_offset, chunk_size_j,
                    i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            // TODO other simple backend may be better
            ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, 0);
            const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, 0);
            ggml_backend_tensor_get_async(simple_backend, simple_tensor, data, offset, size);
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static void ggml_backend_meta_synchronize(ggml_backend_t backend) {
    // No Meta/provider lock is held here: the independent CPU event worker
    // must remain able to publish while child streams wait for their sums.
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    for (size_t i = 0; i < n_backends; i++) {
        ggml_backend_synchronize(ggml_backend_meta_simple_backend(backend, i));
    }
}

// Caller-scoped observation/coordination point; no provider lock is held here.
static thread_local bool (*tp5_meta_submit_ready_hook)(void *) = nullptr;
static thread_local void * tp5_meta_submit_ready_userdata = nullptr;
bool ggml_backend_meta_set_submit_ready_hook(bool (*hook)(void *), void * userdata) {
    if ((hook == nullptr) != (userdata == nullptr) || (hook && tp5_meta_submit_ready_hook)) return false;
    tp5_meta_submit_ready_hook = hook;
    tp5_meta_submit_ready_userdata = userdata;
    return true;
}

// Successful native-submit notification; called after the provider has released its lock.
static thread_local bool (*tp5_meta_submitted_hook)(void *) = nullptr;
static thread_local void * tp5_meta_submitted_userdata = nullptr;
bool ggml_backend_meta_set_submitted_hook(bool (*hook)(void *), void * userdata) {
    if ((hook == nullptr) != (userdata == nullptr) || (hook && tp5_meta_submitted_hook)) return false;
    tp5_meta_submitted_hook = hook;
    tp5_meta_submitted_userdata = userdata;
    return true;
}

static enum ggml_status ggml_backend_meta_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    const int64_t diag_start = ggml_time_us();
    GGML_ASSERT(cgraph->grads == nullptr);
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;
    // Graph reservation does not enter graph_compute. The first real compute
    // attempt freezes configuration even if qualification subsequently fails.
    backend_ctx->hc_owner_started = true;
    const char * hc_layout_value = std::getenv("GGML_TP5_HC_CONTEXT_OWNER");
    const bool hc_layout_enabled = hc_layout_value && std::strcmp(hc_layout_value, "1") == 0;
    const bool hc_layout_valid = hc_layout_value == nullptr ||
        std::strcmp(hc_layout_value, "0") == 0 || hc_layout_enabled;
    if (!hc_layout_valid || hc_layout_enabled != backend_ctx->hc_owner_configured) {
        GGML_LOG_ERROR("%s: HC context-owner layout and pre-compute offset configuration disagree\n", __func__);
        if (backend_ctx->relay_mode) {
            backend_ctx->relay_i->abort(backend_ctx->relay, GGML_STATUS_FAILED);
        }
        return GGML_STATUS_FAILED;
    }
    const char * shared_value = std::getenv("GGML_TP5_SHARED_EXPERT_SHARD");
    const bool shared_enabled = shared_value && std::strcmp(shared_value, "1") == 0;
    if ((shared_value && std::strcmp(shared_value, "0") != 0 && !shared_enabled) ||
            (shared_enabled && (!backend_ctx->relay_mode || n_backends != 5)) ||
            (backend_ctx->shared_expert_shard_started &&
             shared_enabled != backend_ctx->shared_expert_shard_enabled)) {
        GGML_LOG_ERROR("%s: shared-expert shard requires fixed 0/1 configuration and five-rank relay mode\n", __func__);
        if (backend_ctx->relay_mode) {
            backend_ctx->relay_i->abort(backend_ctx->relay, GGML_STATUS_FAILED);
        }
        return GGML_STATUS_FAILED;
    }
    backend_ctx->shared_expert_shard_started = true;
    backend_ctx->shared_expert_shard_enabled = shared_enabled;
    const char * router_value = std::getenv("GGML_TP5_HC_ROUTER_OWNER");
    const bool router_enabled = router_value && std::strcmp(router_value, "1") == 0;
    const auto disabled_option = [](const char * name) {
        const char * value = std::getenv(name);
        return value == nullptr || std::strcmp(value, "0") == 0;
    };
    if ((router_value && std::strcmp(router_value, "0") != 0 && !router_enabled) ||
            (backend_ctx->hc_router_owner_started &&
             router_enabled != backend_ctx->hc_router_owner_enabled) ||
            (router_enabled && (!backend_ctx->relay_mode || n_backends != 5 ||
             !backend_ctx->hc_owner_configured ||
             backend_ctx->relay_wire_type != GGML_RELAY_WIRE_F32 ||
             !disabled_option("GGML_TP5_HC_PACK_INJECT") ||
             !disabled_option("GGML_TP5_HC_OWNER_NORM")))) {
        GGML_LOG_ERROR("%s: HC router-owner requires fixed 0/1 mode, configured five-rank HC ownership, F32 wire, pack=0 and norm=0\n", __func__);
        if (backend_ctx->relay_mode) {
            backend_ctx->relay_i->abort(backend_ctx->relay, GGML_STATUS_FAILED);
        }
        return GGML_STATUS_FAILED;
    }
    backend_ctx->hc_router_owner_started = true;
    backend_ctx->hc_router_owner_enabled = router_enabled;
    auto execution_split_state = [&](const ggml_tensor * tensor, const bool assume_sync) {
        if (backend_ctx->hc_owner_configured) {
            const auto found = backend_ctx->hc_owner_domains.find(tensor);
            if (found != backend_ctx->hc_owner_domains.end()) {
                auto split = found->second.split;
                if (assume_sync && split.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                    split.axis = GGML_BACKEND_SPLIT_AXIS_MIRRORED;
                }
                return split;
            }
        }
        return ggml_backend_meta_get_split_state(tensor, assume_sync);
    };

    if (backend_ctx->cpu_staged_f32 && cgraph->n_nodes != 0 &&
        ggml_backend_buffer_is_meta(cgraph->nodes[cgraph->n_nodes - 1]->buffer) &&
        ggml_backend_meta_get_split_state(cgraph->nodes[cgraph->n_nodes - 1], false).axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
        // Meta's public final readback gathers shards or reads a mirrored
        // tensor; it cannot expose an unreduced rank partial as final output.
        GGML_LOG_ERROR("%s: CPU-staged reference requires a non-partial final output\n", __func__);
        return GGML_STATUS_FAILED;
    }

    if (backend_ctx->relay_mode) {
        ggml_relay_status previous = {};
        // Snapshot is nonblocking and can lose its try-lock to the event
        // worker. Retry only that CPU contention, before mutating any graph
        // owners; a successfully observed unfinished epoch is never waited on.
        const auto snapshot_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
        bool have_snapshot;
        do {
            have_snapshot = backend_ctx->relay_i->snapshot(backend_ctx->relay, &previous);
            if (have_snapshot) {
                break;
            }
            // ASYNC-PROTOCOL: busy-wait, no scheduler yield.
#if defined(__x86_64__) || defined(__i386__)
            __builtin_ia32_pause();
#endif
        } while (std::chrono::steady_clock::now() < snapshot_deadline);
        if (!have_snapshot || previous.failed ||
            (previous.epoch != 0 && !previous.finished)) {
            backend_ctx->relay_i->abort(backend_ctx->relay, GGML_STATUS_FAILED);
            return GGML_STATUS_FAILED;
        }
        if (cgraph->n_nodes == 0 || !ggml_backend_buffer_is_meta(cgraph->nodes[cgraph->n_nodes - 1]->buffer) ||
            ggml_backend_meta_get_split_state(
                cgraph->nodes[cgraph->n_nodes - 1], false).axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
            backend_ctx->relay_i->abort(backend_ctx->relay, GGML_STATUS_FAILED);
            return GGML_STATUS_FAILED;
        }
    }

    // If the previous cgraph had a defined UID it can be used to skip rebuilding the subgraphs per simple backend.
    const int64_t diag_prepared = ggml_time_us();
    const bool needs_rebuild = (cgraph->uid == 0) || (cgraph->uid != backend_ctx->uid);

    bool max_nnodes_raised = false;
    if (cgraph->n_nodes > backend_ctx->max_nnodes) {
        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            bcj.nodes.resize(cgraph->n_nodes);
            bcj.cgraphs.resize(cgraph->n_nodes);
        }
        backend_ctx->max_nnodes = cgraph->n_nodes;
        max_nnodes_raised = true;
        assert(needs_rebuild);
    }

    if (needs_rebuild) {
        // A changed UID can reuse the current compute arena without another
        // allocator pass. Keep the previous private plan only to authenticate
        // its already-disabled flags; shared physical state is still mirrored.
        const auto previous_hc_owner_domains = std::move(backend_ctx->hc_owner_domains);
        const auto previous_hc_router_owner_nodes = std::move(backend_ctx->hc_router_owner_nodes);
        backend_ctx->hc_router_owner_nodes.clear();
        backend_ctx->hc_owner_domains.clear();
        backend_ctx->hc_owner_up_indices.clear();
        backend_ctx->hc_owner_terminal_indices.clear();
        std::set<ggml_backend_buffer_t> used_buffers;
        auto use_compute_buffer = [&](const ggml_tensor * tensor) {
            if (!ggml_backend_buffer_is_meta(tensor->buffer)) {
                return;
            }
            auto * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
            // stc_static is fully populated before a model/cache is exposed.
            // Reading shared weights must not rotate/reset their unused view
            // arenas while another context is converting its own graph.
            if (buf_ctx->stc_static.simple_tensors.find(tensor) == buf_ctx->stc_static.simple_tensors.end()) {
                used_buffers.emplace(tensor->buffer);
            }
        };
        for (int i = 0; i < cgraph->n_leafs; i++) {
            use_compute_buffer(cgraph->leafs[i]);
        }
        for (int i = 0; i < cgraph->n_nodes; i++) {
            use_compute_buffer(cgraph->nodes[i]);
        }
        for (ggml_backend_buffer_t buf : used_buffers) {
            ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buf->context;
            buf_ctx->stc_compute_index_next = buf_ctx->stc_compute_index ^ 1;
            ggml_backend_meta_simple_tensor_container & stc = buf_ctx->stc_compute[buf_ctx->stc_compute_index_next];
            for (ggml_context_ptr & ctx : stc.ctxs) {
                ggml_reset(ctx.get());
            }
            stc.simple_tensors.clear();
        }
        size_t n_subgraphs  = 0;
        size_t max_tmp_size = 0;

        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];

            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];
                if (node->view_src != nullptr && node->view_src->op == GGML_OP_NONE && ggml_backend_buffer_is_host(node->view_src->buffer)) {
                    // FIXME s_copy_main is on the CPU and its view seems to be incorrectly added to the graph nodes.
                    // For regular usage this doesn't matter since it's a noop but trying to call ggml_backend_meta_buffer_simple_tensor results in a crash.
                    bcj.nodes[i] = node;
                    continue;
                }
                bcj.nodes[i] = ggml_backend_meta_buffer_simple_tensor(node, j);
                GGML_ASSERT(bcj.nodes[i]);
            }
        }

        {
            // Index full-graph aliases once per rebuild. Child stage graphs do
            // not retain every external alias, so this proof stays in Meta.
            bool hc_views_ready = false;
            std::unordered_map<const ggml_tensor *, size_t> hc_view_counts;
            auto hc_has_external_alias = [&](const int first, const int last) {
                if (!hc_views_ready) {
                    hc_views_ready = true;
                    auto count_views = [&](const ggml_tensor * tensor) {
                        for (const ggml_tensor * base = tensor->view_src; base; base = base->view_src) {
                            ++hc_view_counts[base];
                        }
                    };
                    for (int k = 0; k < cgraph->n_nodes; ++k) {
                        count_views(cgraph->nodes[k]);
                    }
                    for (int k = 0; k < cgraph->n_leafs; ++k) {
                        count_views(cgraph->leafs[k]);
                    }
                }
                for (int k = first; k < last; ++k) {
                    const ggml_tensor * omitted = cgraph->nodes[k];
                    size_t internal = 0;
                    for (int use = first; use <= last; ++use) {
                        for (const ggml_tensor * base = cgraph->nodes[use]->view_src; base; base = base->view_src) {
                            internal += base == omitted;
                        }
                    }
                    const auto found = hc_view_counts.find(omitted);
                    const size_t total = found == hc_view_counts.end() ? 0 : found->second;
                    if (total != internal) {
                        return true;
                    }
                }
                return false;
            };

            // A preview uses the proposed owner only while checking the
            // existing late-join matcher. No tensor flag changes during it.
            bool hc_owner_preview = false;
            auto child_computes = [&](const size_t rank, const int index) {
                if (hc_owner_preview) {
                    const auto found = backend_ctx->hc_owner_domains.find(cgraph->nodes[index]);
                    if (found != backend_ctx->hc_owner_domains.end()) {
                        return rank == found->second.rank;
                    }
                }
                return bool(backend_ctx->backend_configs[rank].nodes[index]->flags & GGML_TENSOR_FLAG_COMPUTE);
            };
            auto private_compute_node = [&](const int index) {
                const ggml_tensor * tensor = cgraph->nodes[index];
                if (!ggml_backend_buffer_is_meta(tensor->buffer)) {
                    return false;
                }
                const auto * buffer = (const ggml_backend_meta_buffer_context *) tensor->buffer->context;
                if (buffer->stc_static.simple_tensors.count(tensor) != 0) {
                    return false;
                }
                for (size_t rank = 0; rank < n_backends; ++rank) {
                    if (backend_ctx->backend_configs[rank].nodes[index] == tensor) {
                        return false;
                    }
                }
                return true;
            };
            // Child stage visited sets omit aliases in other stages and leaves.
            // When the optional down fusion would omit an externally aliased
            // value, require its context-private child to stay materialized.
            // This is metadata only: no new node, dependency, split or COMPUTE change.
            static const bool hc_down_fusion_enabled = [] {
                const char * value = std::getenv("GGML_CUDA_FUSE_HC_DOWN_SILU");
                return value && std::strcmp(value, "1") == 0;
            }();
            if (hc_down_fusion_enabled) {
                const auto vector_f32 = [](const ggml_tensor * t, const int64_t width) {
                    return t && t->type == GGML_TYPE_F32 && t->ne[0] == width &&
                            t->ne[1] == 1 && t->ne[2] == 1 && t->ne[3] == 1 &&
                            !t->view_src && ggml_is_contiguous(t);
                };
                for (int i = 0; i + 2 < cgraph->n_nodes; ++i) {
                    const auto * down = cgraph->nodes[i];
                    const auto * scaled = cgraph->nodes[i + 1];
                    const auto * lo = cgraph->nodes[i + 2];
                    const auto * weights = down->src[0];
                    const auto * xn = down->src[1];
                    if (down->op != GGML_OP_MUL_MAT || !vector_f32(down, 320) ||
                            scaled->op != GGML_OP_SCALE || scaled->src[0] != down || !vector_f32(scaled, 320) ||
                            lo->op != GGML_OP_UNARY || ggml_get_unary_op(lo) != GGML_UNARY_OP_SILU ||
                            lo->src[0] != scaled || !vector_f32(lo, 320) ||
                            !weights || (weights->type != GGML_TYPE_BF16 && weights->type != GGML_TYPE_Q8_0 &&
                                         weights->type != GGML_TYPE_Q4_0) ||
                            weights->op != GGML_OP_NONE ||
                            weights->view_src || weights->ne[0] != 10240 || weights->ne[1] != 320 ||
                            weights->ne[2] != 1 || weights->ne[3] != 1 || !ggml_is_contiguous(weights) ||
                            !xn || xn->type != GGML_TYPE_F32 || xn->ne[0] != 10240 || xn->ne[1] != 1 ||
                            xn->ne[2] != 1 || xn->ne[3] != 1 || !ggml_is_contiguous(xn)) {
                        continue;
                    }
                    float params[2];
                    memcpy(params, scaled->op_params, sizeof(params));
                    if (params[0] != 0.25f || params[1] != 0.0f) {
                        continue;
                    }
                    for (int omitted = i; omitted < i + 2; ++omitted) {
                        if (!hc_has_external_alias(omitted, omitted + 1)) {
                            continue;
                        }
                        if (!private_compute_node(omitted)) {
                            GGML_LOG_ERROR("%s: HC down fusion cannot materialize a shared/static aliased intermediate\n", __func__);
                            if (backend_ctx->relay_mode) {
                                backend_ctx->relay_i->abort(backend_ctx->relay, GGML_STATUS_FAILED);
                            }
                            return GGML_STATUS_FAILED;
                        }
                        for (size_t rank = 0; rank < n_backends; ++rank) {
                            backend_ctx->backend_configs[rank].nodes[omitted]->flags |= GGML_TENSOR_FLAG_OUTPUT;
                        }
                    }
                }
            }

            std::vector<int> hc_router_all_up_indices;
            auto build_hc_owner_plan = [&]() {
                if (!backend_ctx->hc_owner_configured) {
                    return true;
                }
                if (!backend_ctx->relay_mode || n_backends != 5) {
                    return false;
                }
                constexpr int pairs = 97;
                if (router_enabled) {
                    hc_router_all_up_indices.assign(pairs, -1);
                }
                std::unordered_map<std::string, int> names;
                std::vector<int> projections(2 * pairs, -1);
                for (int pair = 0; pair < pairs; ++pair) {
                    const std::string prefix = pair == 96 ? "output_hc_" :
                        "blk." + std::to_string(pair / 2) + (pair % 2 ? ".hc_ffn_" : ".hc_attn_");
                    names.emplace(prefix + "down.weight", 2 * pair);
                    names.emplace(prefix + "up.weight", 2 * pair + 1);
                }
                for (int index = 0; index < cgraph->n_nodes; ++index) {
                    const auto * tensor = cgraph->nodes[index];
                    if (!tensor->src[0]) {
                        continue;
                    }
                    const auto found = names.find(tensor->src[0]->name);
                    if (found == names.end()) {
                        continue;
                    }
                    if (tensor->op != GGML_OP_MUL_MAT || projections[found->second] != -1) {
                        return false;
                    }
                    projections[found->second] = index;
                }
                // Empty final-layer gathers stay in cgraph. They have no
                // execution domain and must not become artificial relay work.
                for (int pair = 0; pair < pairs; ++pair) {
                    const int up = projections[2 * pair + 1];
                    if (up < 3 || up + 12 >= cgraph->n_nodes || projections[2 * pair] != up - 3) {
                        return false;
                    }
                }
                auto rows_f32 = [](const ggml_tensor * tensor, const int64_t width, const int64_t rows) {
                    return tensor && tensor->type == GGML_TYPE_F32 && tensor->ne[0] == width &&
                        tensor->ne[1] == rows && tensor->ne[2] == 1 && tensor->ne[3] == 1 &&
                        ggml_is_contiguous(tensor);
                };
                auto vector_f32 = [&](const ggml_tensor * tensor, const int64_t width) {
                    return rows_f32(tensor, width, 1);
                };
                auto streams_f32 = [](const ggml_tensor * tensor, const int64_t rows) {
                    return tensor && tensor->type == GGML_TYPE_F32 && tensor->ne[0] == 2560 &&
                        tensor->ne[1] == 4 && tensor->ne[2] == rows && tensor->ne[3] == 1 &&
                        ggml_is_contiguous(tensor);
                };
                auto scaled_by = [](const ggml_tensor * tensor, const float value) {
                    if (!tensor || tensor->op != GGML_OP_SCALE) {
                        return false;
                    }
                    float params[2];
                    memcpy(params, tensor->op_params, sizeof(params));
                    return params[0] == value && params[1] == 0.0f;
                };
                const auto * last_ffn_down = cgraph->nodes[projections[190]];
                const auto * head_down = cgraph->nodes[projections[192]];
                const bool no_output = last_ffn_down->ne[1] == 0;
                const int active_pairs = no_output ? 95 : 97;
                if (no_output) {
                    // Prove the exact Qwen last-attention empty-output gather,
                    // not a missing pair or an unrelated empty input. Both
                    // downstream HC normalizations retain this residual.
                    auto norm_input = [&](const ggml_tensor * flat) -> const ggml_tensor * {
                        if (!rows_f32(flat, 10240, 0) || flat->op != GGML_OP_RESHAPE ||
                                flat->view_offs != 0) {
                            return nullptr;
                        }
                        const auto * gamma = flat->src[0];
                        if (!streams_f32(gamma, 0) || gamma->op != GGML_OP_MUL ||
                                !streams_f32(gamma->src[1], 1)) {
                            return nullptr;
                        }
                        const auto * norm = gamma->src[0];
                        if (!streams_f32(norm, 0) || norm->op != GGML_OP_RMS_NORM ||
                                !streams_f32(norm->src[0], 0)) {
                            return nullptr;
                        }
                        return norm->src[0];
                    };
                    const auto * attn_residual = norm_input(last_ffn_down->src[1]);
                    const auto * ffn_residual = norm_input(head_down->src[1]);
                    if (!attn_residual || !ffn_residual ||
                            attn_residual->op != GGML_OP_ADD || ffn_residual->op != GGML_OP_ADD ||
                            ffn_residual->src[0] != attn_residual) {
                        return false;
                    }
                    const auto * residual = attn_residual->src[0];
                    const auto * product = attn_residual->src[1];
                    if (!streams_f32(residual, 0) || residual->op != GGML_OP_RESHAPE ||
                            !streams_f32(product, 0) || product->op != GGML_OP_MUL) {
                        return false;
                    }
                    const auto * gathered_residual = residual->src[0];
                    if (!rows_f32(gathered_residual, 10240, 0) ||
                            gathered_residual->op != GGML_OP_GET_ROWS) {
                        return false;
                    }
                    const auto * output_ids = gathered_residual->src[1];
                    // Scheduler layout copies with n_copies == 1 do not retain INPUT.
                    // A shared zero-length I32 selector has no index payload; its
                    // scheduling flags do not affect the three empty gathers below.
                    if (!output_ids || output_ids->op != GGML_OP_NONE || output_ids->view_src ||
                            output_ids->type != GGML_TYPE_I32 ||
                            output_ids->ne[0] != 0 || output_ids->ne[1] != 1 ||
                            output_ids->ne[2] != 1 || output_ids->ne[3] != 1) {
                        return false;
                    }
                    auto empty_gather = [&](const ggml_tensor * tensor, const int64_t width) {
                        return rows_f32(tensor, width, 0) && tensor->op == GGML_OP_GET_ROWS &&
                            tensor->src[1] == output_ids && vector_f32(tensor->src[0], width);
                    };
                    const auto * repeated = product->src[0];
                    const auto * weights = product->src[1];
                    if (!repeated || repeated->op != GGML_OP_REPEAT ||
                            !repeated->src[0] || repeated->src[0]->op != GGML_OP_RESHAPE ||
                            !weights || weights->op != GGML_OP_RESHAPE) {
                        return false;
                    }
                    const auto * gathered_block = repeated->src[0]->src[0];
                    const auto * gathered_inject = weights->src[0];
                    if (scaled_by(gathered_inject, 2.0f)) {
                        const auto * sigmoid = gathered_inject->src[0];
                        if (!sigmoid || sigmoid->op != GGML_OP_UNARY ||
                                ggml_get_unary_op(sigmoid) != GGML_UNARY_OP_SIGMOID ||
                                !scaled_by(sigmoid->src[0], 0.25f)) {
                            return false;
                        }
                        gathered_inject = sigmoid->src[0]->src[0];
                    }
                    if (!empty_gather(gathered_residual, 10240) ||
                            !empty_gather(gathered_block, 2560) ||
                            !empty_gather(gathered_inject, 4)) {
                        return false;
                    }
                }
                // The loader commits one immutable type for all 194 HC matrices.
                // Recheck the complete graph and physical children; neither an env
                // flag nor one quantized tensor qualifies a partially converted model.
                const ggml_type hc_weight_type = cgraph->nodes[projections[0]]->src[0]->type;
                if (hc_weight_type != GGML_TYPE_BF16 && hc_weight_type != GGML_TYPE_Q8_0 &&
                        hc_weight_type != GGML_TYPE_Q4_0) {
                    return false;
                }
                auto full_weight = [&](const ggml_tensor * tensor, const int64_t k, const int64_t n) {
                    if (!tensor || tensor->type != hc_weight_type || tensor->op != GGML_OP_NONE ||
                        tensor->view_src || tensor->ne[0] != k || tensor->ne[1] != n ||
                        tensor->ne[2] != 1 || tensor->ne[3] != 1 || !ggml_is_contiguous(tensor) ||
                        !ggml_backend_buffer_is_meta(tensor->buffer) ||
                        ggml_backend_meta_get_split_state(tensor, false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                        return false;
                    }
                    const auto * buffer = (const ggml_backend_meta_buffer_context *) tensor->buffer->context;
                    const auto found = buffer->stc_static.simple_tensors.find(tensor);
                    if (found == buffer->stc_static.simple_tensors.end() || found->second.size() != n_backends) {
                        return false;
                    }
                    for (size_t rank = 0; rank < n_backends; ++rank) {
                        const auto * child = found->second[rank];
                        if (!child || child->type != hc_weight_type || child->ne[0] != k || child->ne[1] != n ||
                            child->ne[2] != 1 || child->ne[3] != 1 || !ggml_is_contiguous(child)) {
                            return false;
                        }
                    }
                    return true;
                };
                const char * keep_value = std::getenv("GGML_TP5_HC_KEEP_LO");
                const bool keep_lo = keep_value && std::atoi(keep_value) != 0;
                backend_ctx->hc_owner_up_indices.resize(active_pairs);
                backend_ctx->hc_owner_terminal_indices.assign(active_pairs, -1);
                for (int pair = 0; pair < pairs; ++pair) {
                    const int index = projections[2 * pair + 1];
                    const int64_t rows = pair < active_pairs ? 1 : 0;
                    if (router_enabled) {
                        hc_router_all_up_indices[pair] = index;
                    }
                    const auto * down = cgraph->nodes[index - 3];
                    const auto * scaled = cgraph->nodes[index - 2];
                    const auto * lo = cgraph->nodes[index - 1];
                    const auto * up = cgraph->nodes[index];
                    float scale[2];
                    memcpy(scale, scaled->op_params, sizeof(scale));
                    if (!full_weight(down->src[0], 10240, 320) || !full_weight(up->src[0], 320, 10240) ||
                        !rows_f32(down->src[1], 10240, rows) ||
                        down->src[1] != cgraph->nodes[index + 2]->src[0] ||
                        !rows_f32(down, 320, rows) || down->view_src ||
                        scaled->op != GGML_OP_SCALE || scaled->src[0] != down ||
                        scale[0] != 0.25f || scale[1] != 0.0f || !rows_f32(scaled, 320, rows) || scaled->view_src ||
                        lo->op != GGML_OP_UNARY || ggml_get_unary_op(lo) != GGML_UNARY_OP_SILU ||
                        lo->src[0] != scaled || !rows_f32(lo, 320, rows) || lo->view_src ||
                        up->src[1] != lo || !rows_f32(up, 10240, rows) || up->view_src ||
                        ((down->flags | scaled->flags) & GGML_TENSOR_FLAG_OUTPUT) ||
                        ((lo->flags & GGML_TENSOR_FLAG_OUTPUT) && !keep_lo)) {
                        return false;
                    }
                    for (int index_prefix = index - 3; index_prefix < index; ++index_prefix) {
                        if (ggml_node_get_use_count(cgraph, index_prefix) != 1) {
                            return false;
                        }
                    }
                    if (pair >= active_pairs) {
                        // Keep the real empty nodes and their flags untouched.
                        // Only these two fully present canonical tails may be
                        // inactive; all five child descriptors must stay empty.
                        const ggml_op empty_ops[] = {
                            GGML_OP_MUL_MAT, GGML_OP_SCALE, GGML_OP_UNARY,
                            GGML_OP_MUL_MAT, GGML_OP_UNARY, GGML_OP_MUL, GGML_OP_RESHAPE,
                            GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
                            GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_SCALE,
                        };
                        for (int k = index - 3; k <= index + 11; ++k) {
                            const auto * tensor = cgraph->nodes[k];
                            if (tensor->op != empty_ops[k - (index - 3)] ||
                                    tensor->type != GGML_TYPE_F32 || !ggml_is_empty(tensor) ||
                                    !ggml_backend_buffer_is_meta(tensor->buffer) ||
                                    ggml_backend_meta_get_split_state(tensor, false).axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN) {
                                return false;
                            }
                            for (size_t rank = 0; rank < n_backends; ++rank) {
                                const auto * child = backend_ctx->backend_configs[rank].nodes[k];
                                if (!child || !ggml_is_empty(child) || !ggml_are_same_shape(child, tensor)) {
                                    return false;
                                }
                            }
                        }
                        continue;
                    }
                    // This is a storage-layout check, not an execution-domain
                    // query. All four nodes must have been converted as full
                    // mirrored tensors before installing a private plan.
                    const uint32_t owner = (uint32_t(pair) + backend_ctx->hc_owner_offset) % 5;
                    for (int k = index - 3; k <= index; ++k) {
                        const auto * tensor = cgraph->nodes[k];
                        const int64_t width = k == index ? 10240 : 320;
                        if (!private_compute_node(k) ||
                            ggml_backend_meta_get_split_state(tensor, false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED ||
                            !(tensor->flags & GGML_TENSOR_FLAG_COMPUTE)) {
                            return false;
                        }
                        uint32_t actual_mask = 0;
                        for (size_t rank = 0; rank < n_backends; ++rank) {
                            if (!vector_f32(backend_ctx->backend_configs[rank].nodes[k], width)) {
                                return false;
                            }
                            actual_mask |= uint32_t(child_computes(rank, k)) << rank;
                        }
                        if (actual_mask != 31) {
                            const auto previous = previous_hc_owner_domains.find(tensor);
                            if (actual_mask != (uint32_t(1) << owner) ||
                                previous == previous_hc_owner_domains.end() || previous->second.rank != owner ||
                                previous->second.split.n_segments != 1 || previous->second.split.nr[0] != 1 ||
                                previous->second.split.axis !=
                                    (k == index ? GGML_BACKEND_SPLIT_AXIS_PARTIAL : GGML_BACKEND_SPLIT_AXIS_0) ||
                                (k != index && previous->second.split.ne[owner] != 320)) {
                                return false;
                            }
                        }
                        ggml_backend_meta_split_state split = {
                            k == index ? GGML_BACKEND_SPLIT_AXIS_PARTIAL : GGML_BACKEND_SPLIT_AXIS_0,
                            {0}, {1}, 1
                        };
                        if (k != index) {
                            split.ne[owner] = 320;
                        }
                        if (!backend_ctx->hc_owner_domains.emplace(tensor,
                                ggml_backend_meta_context::hc_owner_domain{ split, owner }).second) {
                            return false;
                        }
                    }
                    backend_ctx->hc_owner_up_indices[pair] = index;
                }
                return true;
            };
            if (!build_hc_owner_plan()) {
                GGML_LOG_ERROR("%s: HC context-owner requires 97 complete BF16 HC4/K320 pairs and an exact T1 or no-output profile\n", __func__);
                backend_ctx->relay_i->abort(backend_ctx->relay, GGML_STATUS_FAILED);
                return GGML_STATUS_FAILED;
            }
            // Qualify every real router and all five immutable weight children
            // before committing any new owner flags. No router is synthesized for T0.
            std::unordered_map<int, int> hc_router_up_layers;
            auto build_hc_router_owner_plan = [&]() {
                if (!router_enabled) {
                    return true;
                }
                if (hc_router_all_up_indices.size() != 97 ||
                        (backend_ctx->hc_owner_up_indices.size() != 95 &&
                         backend_ctx->hc_owner_up_indices.size() != 97)) {
                    return false;
                }
                std::unordered_map<std::string, int> names;
                std::vector<int> routers(48, -1);
                for (int layer = 0; layer < 48; ++layer) {
                    names.emplace("blk." + std::to_string(layer) + ".ffn_gate_inp.weight", layer);
                }
                for (int index = 0; index < cgraph->n_nodes; ++index) {
                    const auto * tensor = cgraph->nodes[index];
                    if (!tensor->src[0]) {
                        continue;
                    }
                    const auto found = names.find(tensor->src[0]->name);
                    if (found == names.end()) {
                        continue;
                    }
                    if (tensor->op != GGML_OP_MUL_MAT || routers[found->second] != -1) {
                        return false;
                    }
                    routers[found->second] = index;
                }
                const auto matrix_f32 = [](const ggml_tensor * tensor) {
                    return tensor && tensor->op == GGML_OP_NONE && !tensor->view_src &&
                        tensor->type == GGML_TYPE_F32 && tensor->ne[0] == 2560 && tensor->ne[1] == 512 &&
                        tensor->ne[2] == 1 && tensor->ne[3] == 1 && ggml_is_contiguous(tensor);
                };
                const auto rows_f32 = [](const ggml_tensor * tensor, const int64_t width, const int64_t rows) {
                    return tensor && tensor->type == GGML_TYPE_F32 &&
                        tensor->ne[0] == width && tensor->ne[1] == rows &&
                        tensor->ne[2] == 1 && tensor->ne[3] == 1 && ggml_is_contiguous(tensor);
                };
                for (int layer = 0; layer < 48; ++layer) {
                    const int pair = 2 * layer + 1;
                    const bool active = size_t(pair) < backend_ctx->hc_owner_up_indices.size();
                    const int up = hc_router_all_up_indices[pair];
                    const int index = routers[layer];
                    if (up < 0 || up + 11 >= cgraph->n_nodes || index < 0) {
                        return false;
                    }
                    const auto * router = cgraph->nodes[index];
                    const auto * weight = router->src[0];
                    if (!matrix_f32(weight) || !ggml_backend_buffer_is_meta(weight->buffer) ||
                            ggml_backend_meta_get_split_state(weight, false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                        return false;
                    }
                    const auto * buffer = (const ggml_backend_meta_buffer_context *) weight->buffer->context;
                    const auto children = buffer->stc_static.simple_tensors.find(weight);
                    if (children == buffer->stc_static.simple_tensors.end() || children->second.size() != n_backends) {
                        return false;
                    }
                    for (const auto * child : children->second) {
                        if (!matrix_f32(child)) {
                            return false;
                        }
                    }
                    if (!private_compute_node(index) || router->view_src ||
                            router->src[1] != cgraph->nodes[up + 11] ||
                            !rows_f32(router, 512, active ? 1 : 0) ||
                            !rows_f32(router->src[1], 2560, active ? 1 : 0)) {
                        return false;
                    }
                    for (size_t rank = 0; rank < n_backends; ++rank) {
                        if (!rows_f32(backend_ctx->backend_configs[rank].nodes[index], 512, active ? 1 : 0)) {
                            return false;
                        }
                    }
                    if (active) {
                        if (index != up + 12 ||
                                !hc_router_up_layers.emplace(up, layer).second) {
                            return false;
                        }
                    } else {
                        // The final no-output router remains the original empty
                        // MM -> softmax branch. It must not acquire an empty packet
                        // or any new consumer that could manufacture a relay join.
                        if (layer != 47 || index + 1 >= cgraph->n_nodes ||
                                ggml_backend_meta_get_split_state(router, false).axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN ||
                                ggml_node_get_use_count(cgraph, index) != 1) {
                            return false;
                        }
                        const auto * softmax = cgraph->nodes[index + 1];
                        if (softmax->op != GGML_OP_SOFT_MAX || softmax->src[0] != router ||
                                !rows_f32(softmax, 512, 0)) {
                            return false;
                        }
                    }
                }
                return hc_router_up_layers.size() ==
                    (backend_ctx->hc_owner_up_indices.size() == 95 ? 47u : 48u);
            };
            if (!build_hc_router_owner_plan()) {
                GGML_LOG_ERROR("%s: HC router-owner requires all 48 canonical mirrored F32 routers and exactly 47/48 active FFN HC branches\n", __func__);
                backend_ctx->relay_i->abort(backend_ctx->relay, GGML_STATUS_FAILED);
                return GGML_STATUS_FAILED;
            }
            hc_owner_preview = backend_ctx->hc_owner_configured;

            // A single contributing rank may finish an exclusive gated four-stream
            // mean before SUM. Multiple contributors must still join before sigmoid.
            auto get_i_single_owner_mean = [&](const int i, const bool apply_norm = true) -> int {
                // A following stage is required: relay joins every stage except the last.
                if (!backend_ctx->relay_mode || i + 12 >= cgraph->n_nodes ||
                    !ggml_can_fuse_subgraph(cgraph, i,
                        { GGML_OP_MUL_MAT, GGML_OP_UNARY, GGML_OP_MUL, GGML_OP_RESHAPE,
                          GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
                          GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_SCALE }, { i + 11 })) {
                    return i;
                }
                if (backend_ctx->hc_owner_configured) {
                    for (int k = i; k < i + 12; ++k) {
                        if (!private_compute_node(k)) {
                            return i;
                        }
                    }
                }
                const ggml_tensor * up      = cgraph->nodes[i];
                const ggml_tensor * gate    = cgraph->nodes[i + 1];
                const ggml_tensor * gated   = cgraph->nodes[i + 2];
                const ggml_tensor * streams = cgraph->nodes[i + 3];
                const ggml_tensor * mixed   = cgraph->nodes[i + 11];
                const int64_t width = up->ne[0];
                const int64_t dim = width / 4;
                const int64_t token_rows = up->ne[1];
                const char * rows_option = std::getenv("GGML_TP5_HC_MEAN_ROWS4");
                if (token_rows != 1 && (token_rows < 2 || token_rows > 4 ||
                        !rows_option || std::strcmp(rows_option,"1") != 0)) {
                    return i;
                }
                auto vector_f32 = [token_rows](const ggml_tensor * t, const int64_t n) {
                    return t != nullptr && t->type == GGML_TYPE_F32 &&
                        t->ne[0] == n && t->ne[1] == token_rows && t->ne[2] == 1 && t->ne[3] == 1 &&
                        ggml_is_contiguous(t);
                };
                if (width <= 0 || width % 4 != 0 || !vector_f32(up, width) || up->view_src ||
                    execution_split_state(up, false).axis != GGML_BACKEND_SPLIT_AXIS_PARTIAL ||
                    gate->src[0] != up || ggml_get_unary_op(gate) != GGML_UNARY_OP_SIGMOID ||
                    !vector_f32(gate, width) || gate->view_src ||
                    gated->src[1] != gate || !vector_f32(gated->src[0], width) ||
                    execution_split_state(gated->src[0], false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED ||
                    !vector_f32(gated, width) || gated->view_src ||
                    streams->src[0] != gated || streams->view_src != gated || streams->view_offs != 0 ||
                    streams->ne[0] != dim || streams->ne[1] != 4 || streams->ne[2] != token_rows || streams->ne[3] != 1 ||
                    !ggml_is_contiguous(streams)) {
                    return i;
                }
                for (int k = 1; k < 12; ++k) {
                    const ggml_tensor * t = cgraph->nodes[i + k];
                    if (t->type != GGML_TYPE_F32 ||
                        execution_split_state(t, false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                        return i;
                    }
                }
                for (int k = 0; k < 4; ++k) {
                    const ggml_tensor * view = cgraph->nodes[i + 4 + k];
                    // ggml views flatten view_src to the underlying gated tensor.
                    if (view->src[0] != streams || view->view_src != gated ||
                        view->view_offs != size_t(k * dim) * sizeof(float) ||
                        view->ne[0] != dim || view->ne[1] != token_rows || view->ne[2] != 1 || view->ne[3] != 1 ||
                        view->nb[0] != sizeof(float) || view->nb[1] != size_t(width) * sizeof(float) ||
                        view->nb[2] != view->nb[1] * size_t(token_rows) || view->nb[3] != view->nb[2]) {
                        return i;
                    }
                }
                for (int k = 0; k < 3; ++k) {
                    const ggml_tensor * add = cgraph->nodes[i + 8 + k];
                    if (add->src[0] != cgraph->nodes[i + (k == 0 ? 4 : 7 + k)] ||
                        add->src[1] != cgraph->nodes[i + 5 + k] ||
                        !vector_f32(add, dim) || add->view_src) {
                        return i;
                    }
                }
                float scale[2];
                memcpy(scale, mixed->op_params, sizeof(scale));
                if (mixed->src[0] != cgraph->nodes[i + 10] || !vector_f32(mixed, dim) ||
                    mixed->view_src || scale[0] != 0.25f || scale[1] != 0.0f) {
                    return i;
                }

                // Inspect the actual child flags after zero-sized split conversion.
                // The existing propagation below will disable the whole nonowner tail.
                size_t owner = n_backends;
                for (size_t rank = 0; rank < n_backends; ++rank) {
                    const auto & bc = backend_ctx->backend_configs[rank];
                    if (!vector_f32(bc.nodes[i], width) || !vector_f32(bc.nodes[i + 11], dim)) {
                        return i;
                    }
                    if (child_computes(rank, i)) {
                        if (owner != n_backends) {
                            return i;
                        }
                        owner = rank;
                    }
                }
                if (owner == n_backends) {
                    return i;
                }
                for (int k = 1; k < 12; ++k) {
                    if (!child_computes(owner, i + k)) {
                        return i;
                    }
                }

                // The original twelve-node join remains the fallback. Its
                // omitted values must have no external aliases or OUTPUT uses.
                if (hc_has_external_alias(i, i + 11)) {
                    return i;
                }
                int last = i + 11;
                auto scaled_by = [](const ggml_tensor * t, const float value) {
                    float params[2];
                    memcpy(params, t->op_params, sizeof(params));
                    return t->op == GGML_OP_SCALE && params[0] == value && params[1] == 0.0f;
                };
                const auto dense_matrix = [](const ggml_tensor * t, const int64_t k, const int64_t n) {
                    return t != nullptr &&
                        (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16 || t->type == GGML_TYPE_BF16) &&
                        t->ne[0] == k && t->ne[1] == n && t->ne[2] == 1 && t->ne[3] == 1 &&
                        ggml_is_contiguous(t);
                };

                if (router_enabled) {
                    const auto router_pair = hc_router_up_layers.find(i);
                    if (router_pair == hc_router_up_layers.end()) {
                        // Attention HC and the head retain their original terminal.
                        // The router mode explicitly excludes inject/norm expansion.
                        return last;
                    }
                    auto can_pack_router = [&]() {
                        // The two exported views and unchanged softmax must follow.
                        // They remain in the next stage and execute on all five ranks.
                        if (i + 16 >= cgraph->n_nodes || dim != 2560 || width != 10240) {
                            return false;
                        }
                        const auto * router = cgraph->nodes[i + 12];
                        const auto * packed = cgraph->nodes[i + 13];
                        const auto * mixed_view = cgraph->nodes[i + 14];
                        const auto * logits_view = cgraph->nodes[i + 15];
                        const auto * softmax = cgraph->nodes[i + 16];
                        if (router->op != GGML_OP_MUL_MAT || router->src[1] != mixed ||
                                !vector_f32(router, 512) || router->view_src ||
                                packed->op != GGML_OP_CONCAT || ggml_get_op_params_i32(packed, 0) != 0 ||
                                packed->src[0] != mixed || packed->src[1] != router ||
                                !vector_f32(packed, 3072) || packed->view_src ||
                                ggml_node_get_use_count(cgraph, i + 11) != 2 ||
                                ggml_node_get_use_count(cgraph, i + 12) != 1 ||
                                ggml_node_get_use_count(cgraph, i + 13) != 2) {
                            return false;
                        }
                        auto packet_view = [&](const ggml_tensor * tensor, const int64_t count, const size_t offset) {
                            return tensor->op == GGML_OP_VIEW && tensor->src[0] == packed &&
                                tensor->view_src == packed && tensor->view_offs == offset &&
                                vector_f32(tensor, count) && tensor->nb[0] == sizeof(float) &&
                                tensor->nb[1] == size_t(3072) * sizeof(float) &&
                                tensor->nb[2] == tensor->nb[1] && tensor->nb[3] == tensor->nb[1];
                        };
                        if (!packet_view(mixed_view, 2560, 0) ||
                                !packet_view(logits_view, 512, size_t(2560) * sizeof(float)) ||
                                softmax->op != GGML_OP_SOFT_MAX || softmax->src[0] != logits_view ||
                                !vector_f32(softmax, 512) ||
                                ggml_node_get_use_count(cgraph, i + 15) != 1) {
                            return false;
                        }
                        for (int k = 12; k <= 15; ++k) {
                            const auto * tensor = cgraph->nodes[i + k];
                            if (!private_compute_node(i + k) ||
                                    execution_split_state(tensor, false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED ||
                                    !(tensor->flags & GGML_TENSOR_FLAG_COMPUTE)) {
                                return false;
                            }
                            uint32_t actual_mask = 0;
                            for (size_t rank = 0; rank < n_backends; ++rank) {
                                const auto * child = backend_ctx->backend_configs[rank].nodes[i + k];
                                if (child->type != GGML_TYPE_F32 || !ggml_are_same_shape(child, tensor) ||
                                        !ggml_is_contiguous(child)) {
                                    return false;
                                }
                                actual_mask |= uint32_t(child_computes(rank, i + k)) << rank;
                            }
                            if (k >= 14) {
                                if (actual_mask != 31) {
                                    return false;
                                }
                            } else if (actual_mask != 31) {
                                const auto previous = previous_hc_router_owner_nodes.find(tensor);
                                if (actual_mask != (uint32_t(1) << owner) ||
                                        previous == previous_hc_router_owner_nodes.end() || previous->second != owner) {
                                    return false;
                                }
                            }
                        }
                        return ggml_can_fuse_subgraph(cgraph, i,
                            { GGML_OP_MUL_MAT, GGML_OP_UNARY, GGML_OP_MUL, GGML_OP_RESHAPE,
                              GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
                              GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_SCALE,
                              GGML_OP_MUL_MAT, GGML_OP_CONCAT }, { i + 13 }) &&
                            !hc_has_external_alias(i, i + 13);
                    };
                    // An enabled FFN router must qualify completely. Falling back
                    // to the old twelve-node join would silently repeat its MM.
                    return can_pack_router() ? i + 13 : i;
                }

                // The mirrored inject matrix stays mirrored in storage. Only
                // this exclusive branch runs on the already proven up owner.
                auto can_pack_inject = [&]() {
                    if (i + 17 >= cgraph->n_nodes) {
                        return false;
                    }
                    const ggml_tensor * inject = cgraph->nodes[i + 12];
                    const ggml_tensor * scaled = cgraph->nodes[i + 13];
                    const ggml_tensor * sigmoid = cgraph->nodes[i + 14];
                    const ggml_tensor * weights = cgraph->nodes[i + 15];
                    const ggml_tensor * packed = cgraph->nodes[i + 16];
                    if (inject->op != GGML_OP_MUL_MAT || inject->src[1] != gated->src[0] ||
                            !dense_matrix(inject->src[0], width, 4) ||
                            execution_split_state(inject->src[0], false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED ||
                            !vector_f32(inject, 4) || inject->view_src ||
                            !scaled_by(scaled, 0.25f) || scaled->src[0] != inject ||
                            !vector_f32(scaled, 4) || scaled->view_src ||
                            sigmoid->op != GGML_OP_UNARY || ggml_get_unary_op(sigmoid) != GGML_UNARY_OP_SIGMOID ||
                            sigmoid->src[0] != scaled || !vector_f32(sigmoid, 4) || sigmoid->view_src ||
                            !scaled_by(weights, 2.0f) || weights->src[0] != sigmoid ||
                            !vector_f32(weights, 4) || weights->view_src ||
                            packed->op != GGML_OP_CONCAT || ggml_get_op_params_i32(packed, 0) != 0 ||
                            packed->src[0] != mixed || packed->src[1] != weights ||
                            !vector_f32(packed, dim + 4) || packed->view_src) {
                        return false;
                    }
                    for (int k = 12; k <= 16; ++k) {
                        if (backend_ctx->hc_owner_configured && !private_compute_node(i + k)) {
                            return false;
                        }
                        if (execution_split_state(cgraph->nodes[i + k], false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED ||
                                !child_computes(owner, i + k)) {
                            return false;
                        }
                    }
                    for (size_t rank = 0; rank < n_backends; ++rank) {
                        if (!vector_f32(backend_ctx->backend_configs[rank].nodes[i + 16], dim + 4)) {
                            return false;
                        }
                    }
                    if (!ggml_can_fuse_subgraph(cgraph, i,
                            { GGML_OP_MUL_MAT, GGML_OP_UNARY, GGML_OP_MUL, GGML_OP_RESHAPE,
                              GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
                              GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_SCALE,
                              GGML_OP_MUL_MAT, GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE,
                              GGML_OP_CONCAT }, { i + 16 }) ||
                            hc_has_external_alias(i, i + 16)) {
                        return false;
                    }
                    return true;
                };
                if (can_pack_inject()) {
                    last = i + 16;
                }

                // Optional, narrow dead-prefix removal. Validate the complete
                // region before changing any child flag; no general DCE.
                auto can_skip_nonowner_norm = [&]() {
                    static const bool enabled = [] {
                        const char * value = std::getenv("GGML_TP5_HC_OWNER_NORM");
                        return value != nullptr && std::atoi(value) != 0;
                    }();
                    if (!enabled || i < 6) {
                        return false;
                    }
                    const int first = i - 6;
                    const ggml_tensor * norm = cgraph->nodes[first];
                    const ggml_tensor * gamma = cgraph->nodes[first + 1];
                    const ggml_tensor * xn = cgraph->nodes[first + 2];
                    const ggml_tensor * down = cgraph->nodes[first + 3];
                    const ggml_tensor * down_scaled = cgraph->nodes[first + 4];
                    const ggml_tensor * lo = cgraph->nodes[first + 5];
                    auto streams_f32 = [&](const ggml_tensor * t) {
                        return t != nullptr && t->type == GGML_TYPE_F32 &&
                            t->ne[0] == dim && t->ne[1] == 4 && t->ne[2] == 1 && t->ne[3] == 1 &&
                            ggml_is_contiguous(t);
                    };
                    // Only HC projections add immutable Q8/Q4 support. The separate
                    // inject/scatter qualification above remains dense-only.
                    const auto hc_matrix = [&](const ggml_tensor * t, const int64_t k, const int64_t n) {
                        return dense_matrix(t, k, n) ||
                            (t && (t->type == GGML_TYPE_Q8_0 || t->type == GGML_TYPE_Q4_0) &&
                              t->op == GGML_OP_NONE && !t->view_src &&
                             t->ne[0] == k && t->ne[1] == n && t->ne[2] == 1 && t->ne[3] == 1 &&
                             ggml_is_contiguous(t));
                    };
                    if (norm->op != GGML_OP_RMS_NORM || !streams_f32(norm) || norm->view_src ||
                            !streams_f32(norm->src[0]) ||
                            gamma->op != GGML_OP_MUL || gamma->src[0] != norm ||
                            !streams_f32(gamma) || !streams_f32(gamma->src[1]) || gamma->view_src ||
                            xn->op != GGML_OP_RESHAPE || xn->src[0] != gamma ||
                            xn->view_src != gamma || xn->view_offs != 0 || !vector_f32(xn, width) ||
                            gated->src[0] != xn ||
                            down->op != GGML_OP_MUL_MAT || down->src[1] != xn ||
                            !hc_matrix(down->src[0], width, 320) || !vector_f32(down, 320) || down->view_src ||
                            !scaled_by(down_scaled, 0.25f) || down_scaled->src[0] != down ||
                            !vector_f32(down_scaled, 320) || down_scaled->view_src ||
                            lo->op != GGML_OP_UNARY || ggml_get_unary_op(lo) != GGML_UNARY_OP_SILU ||
                            lo->src[0] != down_scaled || !vector_f32(lo, 320) || lo->view_src ||
                            up->src[1] != lo || !hc_matrix(up->src[0], 320, width) ||
                            ((down->src[0]->type == GGML_TYPE_Q8_0 || up->src[0]->type == GGML_TYPE_Q8_0 ||
                              down->src[0]->type == GGML_TYPE_Q4_0 || up->src[0]->type == GGML_TYPE_Q4_0) &&
                             down->src[0]->type != up->src[0]->type) ||
                            execution_split_state(norm->src[0], false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED ||
                            execution_split_state(gamma->src[1], false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                        return false;
                    }
                    for (int k = 0; k < 3; ++k) {
                        if (execution_split_state(cgraph->nodes[first + k], false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                            return false;
                        }
                    }
                    // Down and its two pointwise successors must already be
                    // disabled on precisely the same nonowner ranks as up.
                    for (size_t rank = 0; rank < n_backends; ++rank) {
                        for (int k = 3; k < 6; ++k) {
                            if (child_computes(rank, first + k) != (rank == owner)) {
                                return false;
                            }
                        }
                    }
                    for (int k = 0; k < 3; ++k) {
                        if (!child_computes(owner, first + k) ||
                            (backend_ctx->hc_owner_configured && !private_compute_node(first + k))) {
                            return false;
                        }
                    }
                    const ggml_op ops[] = {
                        GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_RESHAPE, GGML_OP_MUL_MAT,
                        GGML_OP_SCALE, GGML_OP_UNARY,
                        GGML_OP_MUL_MAT, GGML_OP_UNARY, GGML_OP_MUL, GGML_OP_RESHAPE,
                        GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
                        GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_SCALE,
                        GGML_OP_MUL_MAT, GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_CONCAT,
                    };
                    static const bool keep_lo_enabled = [] {
                        const char * value = std::getenv("GGML_TP5_HC_KEEP_LO");
                        return value != nullptr && std::atoi(value) != 0;
                    }();
                    const bool kept_lo = keep_lo_enabled &&
                            (lo->flags & GGML_TENSOR_FLAG_OUTPUT) &&
                            (down->src[0]->type == GGML_TYPE_BF16 || down->src[0]->type == GGML_TYPE_Q8_0 ||
                             down->src[0]->type == GGML_TYPE_Q4_0) &&
                            up->src[0]->type == down->src[0]->type;
                    // lo is still computed on its original sole owner and is
                    // now intentionally materialized. The output exception
                    // must not admit an external lo user or a view alias.
                    if (kept_lo && ggml_node_get_use_count(cgraph, first + 5) != 1) {
                        return false;
                    }
                    const int outputs[] = { last, first + 5 };
                    return ggml_can_fuse_subgraph(cgraph, first, last - first + 1,
                               ops, outputs, kept_lo ? 2 : 1) &&
                           !hc_has_external_alias(first, last);
                };
                if (apply_norm && can_skip_nonowner_norm()) {
                    for (size_t rank = 0; rank < n_backends; ++rank) {
                        if (rank == owner) {
                            continue;
                        }
                        for (int k = i - 6; k < i - 3; ++k) {
                            backend_ctx->backend_configs[rank].nodes[k]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
                        }
                    }
                }
                if (token_rows > 1 && apply_norm && std::getenv("TP5_DIAG_GRAPH")) {
                    static thread_local unsigned seen_rows = 0;
                    if (!(seen_rows & (1u << token_rows))) {
                        seen_rows |= (1u << token_rows);
                        GGML_LOG_INFO("TP5_HC_MEAN_ROWS_EXEC rows=%lld owner=%zu width=%lld terminal_width=%lld\n",
                            (long long)token_rows,owner,(long long)width,(long long)dim);
                    }
                }
                return last;
            };

            if (backend_ctx->hc_owner_configured) {
                for (size_t pair = 0; pair < backend_ctx->hc_owner_up_indices.size(); ++pair) {
                    const int index = backend_ctx->hc_owner_up_indices[pair];
                    const int last = get_i_single_owner_mean(index, false);
                    if (last == index || hc_has_external_alias(index - 3, last)) {
                        GGML_LOG_ERROR("%s: HC context-owner pair %zu has a nonexclusive late-join closure\n", __func__, pair);
                        backend_ctx->relay_i->abort(backend_ctx->relay, GGML_STATUS_FAILED);
                        return GGML_STATUS_FAILED;
                    }
                    backend_ctx->hc_owner_terminal_indices[pair] = last;
                }
                // All 47/48 router closures have passed the same dry run.
                // Save their identity now; nonowner flags are cleared only by
                // the existing stage walk after this whole-profile validation.
                if (router_enabled) {
                    for (const auto & entry : hc_router_up_layers) {
                        const int index = entry.first;
                        const uint32_t owner = (uint32_t(2 * entry.second + 1) + backend_ctx->hc_owner_offset) % 5;
                        if (backend_ctx->hc_owner_terminal_indices[2 * entry.second + 1] != index + 13) {
                            backend_ctx->relay_i->abort(backend_ctx->relay, GGML_STATUS_FAILED);
                            return GGML_STATUS_FAILED;
                        }
                        backend_ctx->hc_router_owner_nodes.emplace(cgraph->nodes[index + 12], owner);
                        backend_ctx->hc_router_owner_nodes.emplace(cgraph->nodes[index + 13], owner);
                    }
                }
                hc_owner_preview = false;
                // Only context-private converted descriptors are written. All
                // master tensors and shared static weight children stay intact.
                for (size_t pair = 0; pair < backend_ctx->hc_owner_up_indices.size(); ++pair) {
                    const int index = backend_ctx->hc_owner_up_indices[pair];
                    const uint32_t owner = (uint32_t(pair) + backend_ctx->hc_owner_offset) % 5;
                    for (size_t rank = 0; rank < n_backends; ++rank) {
                        if (rank != owner) {
                            for (int k = index - 3; k <= index; ++k) {
                                backend_ctx->backend_configs[rank].nodes[k]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
                            }
                        }
                    }
                }
            }

            // Each shared hidden coordinate belongs to exactly one rank. Keep
            // its gated partial beside the routed partial until their original
            // join: sum_r (M_r + g*S_r) = sum_r M_r + g*sum_r S_r.
            // No source flag, weight descriptor or relay ABI is changed here.
            std::unordered_map<int, int> shared_join_ends;
            size_t shared_expected_joins = 0;
            auto build_shared_expert_plan = [&]() {
                if (!shared_enabled) {
                    return true;
                }
                std::unordered_map<std::string, int> names;
                std::vector<int> down_indices(48, -1);
                for (int layer = 0; layer < 48; ++layer) {
                    names.emplace("blk." + std::to_string(layer) + ".ffn_down_shexp.weight", layer);
                }
                for (int i = 0; i < cgraph->n_nodes; ++i) {
                    const auto * node = cgraph->nodes[i];
                    if (node->op != GGML_OP_MUL_MAT || !node->src[0]) {
                        continue;
                    }
                    const auto found = names.find(ggml_get_name(node->src[0]));
                    if (found != names.end()) {
                        if (down_indices[found->second] >= 0) {
                            return false;
                        }
                        down_indices[found->second] = i;
                    }
                }
                const auto rows = [](const ggml_tensor * t, const ggml_type type,
                                     const int64_t n0, const int64_t n1, const int64_t n2 = 1) {
                    return t && t->type == type && t->ne[0] == n0 && t->ne[1] == n1 &&
                            t->ne[2] == n2 && t->ne[3] == 1;
                };
                const auto matrix = [&](const ggml_tensor * t, const int64_t k, const int64_t n,
                                        const int64_t experts = 1) {
                    return t && t->op == GGML_OP_NONE && !t->view_src && ggml_is_quantized(t->type) &&
                            rows(t, t->type, k, n, experts) && ggml_is_contiguous(t) &&
                            ggml_backend_buffer_is_meta(t->buffer) &&
                            ggml_backend_buffer_get_usage(t->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE;
                };
                const auto split128 = [&](const ggml_tensor * t, const ggml_backend_meta_split_axis axis) {
                    if (!t || !ggml_backend_buffer_is_meta(t->buffer)) {
                        return false;
                    }
                    const auto split = execution_split_state(t, false);
                    if (split.axis != axis || split.n_segments != 1 || split.nr[0] != 1) {
                        return false;
                    }
                    for (size_t rank = 0; rank < n_backends; ++rank) {
                        if (split.ne[rank] != 128) {
                            return false;
                        }
                    }
                    return true;
                };
                const ggml_op ops[] = {
                    GGML_OP_MUL_MAT_ID, GGML_OP_MUL, GGML_OP_PERMUTE, GGML_OP_CONT,
                    GGML_OP_SUM_ROWS, GGML_OP_RESHAPE,
                    GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU, GGML_OP_MUL_MAT,
                    GGML_OP_MUL_MAT, GGML_OP_UNARY, GGML_OP_MUL, GGML_OP_ADD,
                };
                for (int layer = 0; layer < 48; ++layer) {
                    const int first = down_indices[layer] - 9;
                    const int last = first + 13;
                    if (down_indices[layer] < 0 || first < 0 || last >= cgraph->n_nodes) {
                        return false;
                    }
                    const auto * routed = cgraph->nodes[first];
                    const auto * weighted = cgraph->nodes[first + 1];
                    const auto * perm = cgraph->nodes[first + 2];
                    const auto * cont = cgraph->nodes[first + 3];
                    const auto * sum = cgraph->nodes[first + 4];
                    const auto * moe = cgraph->nodes[first + 5];
                    const auto * gate = cgraph->nodes[first + 6];
                    const auto * up = cgraph->nodes[first + 7];
                    const auto * glu = cgraph->nodes[first + 8];
                    const auto * down = cgraph->nodes[first + 9];
                    const auto * scalar = cgraph->nodes[first + 10];
                    const auto * sigmoid = cgraph->nodes[first + 11];
                    const auto * gated = cgraph->nodes[first + 12];
                    const auto * out = cgraph->nodes[last];
                    const auto * input = gate->src[1];
                    const int64_t tokens = routed->ne[2];
                    const std::string prefix = "blk." + std::to_string(layer) + ".ffn_";
                    if ((tokens != 0 && tokens != 1) || (tokens == 0 && layer != 47) ||
                            !rows(input, GGML_TYPE_F32, 2560, tokens) ||
                            !ggml_is_contiguous(input) || !ggml_backend_buffer_is_meta(input->buffer)) {
                        return false;
                    }
                    for (int k = 0; k < 14; ++k) {
                        if (cgraph->nodes[first + k]->op != ops[k] || !private_compute_node(first + k)) {
                            return false;
                        }
                    }
                    if (!matrix(routed->src[0], 640, 2560, 512) ||
                            std::strcmp(ggml_get_name(routed->src[0]), (prefix + "down_exps.weight").c_str()) != 0 ||
                            !rows(routed->src[1], GGML_TYPE_F32, 640, 10, tokens) ||
                            !rows(routed->src[2], GGML_TYPE_I32, 10, tokens) ||
                            !rows(routed, GGML_TYPE_F32, 2560, 10, tokens) ||
                            routed->view_src || !ggml_is_contiguous(routed) ||
                            weighted->src[0] != routed ||
                            !rows(weighted->src[1], GGML_TYPE_F32, 1, 10, tokens) ||
                            !rows(weighted, GGML_TYPE_F32, 2560, 10, tokens) ||
                            weighted->view_src || !ggml_is_contiguous(weighted) ||
                            perm->src[0] != weighted || perm->view_src != weighted || perm->view_offs != 0 ||
                            !rows(perm, GGML_TYPE_F32, 10, 2560, tokens) ||
                            perm->nb[0] != weighted->nb[1] || perm->nb[1] != weighted->nb[0] ||
                            perm->nb[2] != weighted->nb[2] || perm->nb[3] != weighted->nb[3] ||
                            cont->src[0] != perm || cont->view_src ||
                            !rows(cont, GGML_TYPE_F32, 10, 2560, tokens) || !ggml_is_contiguous(cont) ||
                            sum->src[0] != cont || sum->view_src ||
                            !rows(sum, GGML_TYPE_F32, 1, 2560, tokens) || !ggml_is_contiguous(sum) ||
                            moe->src[0] != sum || moe->view_src != sum || moe->view_offs != 0 ||
                            !rows(moe, GGML_TYPE_F32, 2560, tokens) || !ggml_is_contiguous(moe)) {
                        return false;
                    }
                    if (!matrix(gate->src[0], 2560, 640) || !matrix(up->src[0], 2560, 640) ||
                            !matrix(down->src[0], 640, 2560) ||
                            std::strcmp(ggml_get_name(gate->src[0]), (prefix + "gate_shexp.weight").c_str()) != 0 ||
                            std::strcmp(ggml_get_name(up->src[0]), (prefix + "up_shexp.weight").c_str()) != 0 ||
                            std::strcmp(ggml_get_name(down->src[0]), (prefix + "down_shexp.weight").c_str()) != 0 ||
                            !rows(gate, GGML_TYPE_F32, 640, tokens) || gate->view_src ||
                            !rows(up, GGML_TYPE_F32, 640, tokens) || up->view_src || up->src[1] != input ||
                            glu->src[0] != gate || glu->src[1] != up ||
                            ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU ||
                            !rows(glu, GGML_TYPE_F32, 640, tokens) || glu->view_src ||
                            down->src[1] != glu || !rows(down, GGML_TYPE_F32, 2560, tokens) || down->view_src ||
                            !rows(scalar->src[0], GGML_TYPE_F32, 2560, 1) ||
                            scalar->src[0]->op != GGML_OP_NONE || scalar->src[0]->view_src ||
                            std::strcmp(ggml_get_name(scalar->src[0]), (prefix + "gate_inp_shexp.weight").c_str()) != 0 ||
                            scalar->src[1] != input || !rows(scalar, GGML_TYPE_F32, 1, tokens) || scalar->view_src ||
                            sigmoid->src[0] != scalar || ggml_get_unary_op(sigmoid) != GGML_UNARY_OP_SIGMOID ||
                            !rows(sigmoid, GGML_TYPE_F32, 1, tokens) || sigmoid->view_src ||
                            gated->src[0] != down || gated->src[1] != sigmoid ||
                            !rows(gated, GGML_TYPE_F32, 2560, tokens) || gated->view_src ||
                            out->src[0] != moe || out->src[1] != gated ||
                            !rows(out, GGML_TYPE_F32, 2560, tokens) || out->view_src) {
                        return false;
                    }
                    for (int k = 6; k < 14; ++k) {
                        if (!ggml_is_contiguous(cgraph->nodes[first + k])) {
                            return false;
                        }
                    }
                    if (!split128(routed->src[0], GGML_BACKEND_SPLIT_AXIS_0) ||
                            !split128(gate->src[0], GGML_BACKEND_SPLIT_AXIS_1) ||
                            !split128(up->src[0], GGML_BACKEND_SPLIT_AXIS_1) ||
                            !split128(down->src[0], GGML_BACKEND_SPLIT_AXIS_0) ||
                            !ggml_backend_buffer_is_meta(scalar->src[0]->buffer) ||
                            execution_split_state(scalar->src[0], false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED ||
                            !ggml_can_fuse_subgraph(cgraph, first, 14, ops, &last, 1) ||
                            hc_has_external_alias(first, last)) {
                        return false;
                    }
                    if (tokens == 0) {
                        // The complete final FFN remains in a no-logit graph
                        // with T=0. Do not create a join or touch its children.
                        for (int k = 0; k < 14; ++k) {
                            if (!ggml_is_empty(cgraph->nodes[first + k]) ||
                                    execution_split_state(cgraph->nodes[first + k], false).axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN) {
                                return false;
                            }
                            for (size_t rank = 0; rank < n_backends; ++rank) {
                                if (!ggml_is_empty(backend_ctx->backend_configs[rank].nodes[first + k])) {
                                    return false;
                                }
                            }
                        }
                        continue;
                    }
                    if (execution_split_state(input, false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED ||
                            !split128(routed->src[1], GGML_BACKEND_SPLIT_AXIS_0) ||
                            execution_split_state(routed->src[2], false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED ||
                            execution_split_state(weighted->src[1], false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED ||
                            execution_split_state(routed, false).axis != GGML_BACKEND_SPLIT_AXIS_PARTIAL ||
                            execution_split_state(down, false).axis != GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                        return false;
                    }
                    for (size_t rank = 0; rank < n_backends; ++rank) {
                        const auto & nodes = backend_ctx->backend_configs[rank].nodes;
                        for (int k = 0; k < 14; ++k) {
                            if (!child_computes(rank, first + k)) {
                                return false;
                            }
                        }
                        if (!rows(nodes[first + 6], GGML_TYPE_F32, 128, 1) ||
                                !rows(nodes[first + 7], GGML_TYPE_F32, 128, 1) ||
                                !rows(nodes[first + 8], GGML_TYPE_F32, 128, 1) ||
                                !rows(nodes[first + 9]->src[0], down->src[0]->type, 128, 2560) ||
                                !rows(nodes[last], GGML_TYPE_F32, 2560, 1)) {
                            return false;
                        }
                    }
                    if (!shared_join_ends.emplace(first, last).second) {
                        return false;
                    }
                    ++shared_expected_joins;
                }
                return shared_expected_joins == 47 || shared_expected_joins == 48;
            };
            if (!build_shared_expert_plan()) {
                GGML_LOG_ERROR("%s: shared-expert shard requires all 48 complete closed FFNs with five disjoint 128-wide slices\n", __func__);
                backend_ctx->relay_i->abort(backend_ctx->relay, GGML_STATUS_FAILED);
                return GGML_STATUS_FAILED;
            }

            // For MoE models it may make sense to delay the AllReduce in order to reduce I/O:
            auto get_i_delayed = [&](const int i) -> int {
                if (shared_enabled) {
                    const auto shared = shared_join_ends.find(i);
                    if (shared != shared_join_ends.end()) {
                        return shared->second;
                    }
                }
                const int single_owner_mean = get_i_single_owner_mean(i);
                if (single_owner_mean != i) {
                    return single_owner_mean;
                }
                int id = i; // i_delayed
                int idr = i; // i_delayed return, last safe return value

                ggml_tensor * node = cgraph->nodes[id];
                int32_t n_used = ggml_node_get_use_count(cgraph, id);
                bool exclusive_chain = n_used == 1;
                if (node->flags & GGML_TENSOR_FLAG_OUTPUT) {
                    return idr;
                }

                // Skip MIRRORED nodes that don't consume node
                auto skip_unrelated = [&]() {
                    while (id + 1 < cgraph->n_nodes) {
                        ggml_tensor * next = cgraph->nodes[id+1];
                        // Empty state-copy views have no device reads/writes.
                        // They must not block a linear expert join merely
                        // because no split domain exists for their zero axis.
                        if (ggml_is_empty(next)) {
                            id++;
                            continue;
                        }
                        if (execution_split_state(next, false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                            break;
                        }
                        bool safe = true;
                        for (int s = 0; s < GGML_MAX_SRC; s++) {
                            if (next->src[s] == nullptr || ggml_is_empty(next->src[s])) {
                                continue;
                            }
                            if (next->src[s] == node) {
                                safe = false;
                                break;
                            }
                            if (execution_split_state(next->src[s], false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                                safe = false;
                                break;
                            }
                        }
                        if (!safe) {
                            break;
                        }
                        id++;
                    }
                };

                skip_unrelated();
                if (id + 1 >= cgraph->n_nodes) {
                    return idr;
                }
                {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op == GGML_OP_ADD_ID && next->src[0] == node &&
                            execution_split_state(next->src[1], false).axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL &&
                            execution_split_state(next->src[2], false).axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                        node = next;
                        id++;
                        idr = id;
                        n_used = ggml_node_get_use_count(cgraph, id);
                        exclusive_chain = exclusive_chain && n_used == 1;
                    }
                }
                // Chain of MULs with MIRRORED src[1]
                while (true) {
                    skip_unrelated();
                    if (node->flags & GGML_TENSOR_FLAG_OUTPUT) {
                        return idr;
                    }
                    if (id + 1 >= cgraph->n_nodes) {
                        return idr;
                    }
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op == GGML_OP_MUL && next->src[0] == node &&
                            execution_split_state(next->src[1], false).axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                        node = next;
                        id++;
                        idr = id;
                        n_used = ggml_node_get_use_count(cgraph, id);
                        exclusive_chain = exclusive_chain && n_used == 1;
                    } else {
                        break;
                    }
                }

                // Current upstream reduces experts as PERMUTE -> CONT ->
                // SUM_ROWS -> RESHAPE, not the older VIEW/ADD chain below.
                // All operations here are linear and each intermediate must
                // have one consumer, so fold the small hidden result once.
                if (cgraph->nodes[i]->op == GGML_OP_MUL_MAT_ID && exclusive_chain &&
                    n_used == 1 && id + 4 < cgraph->n_nodes) {
                    const ggml_tensor * perm = cgraph->nodes[id + 1];
                    const ggml_tensor * cont = cgraph->nodes[id + 2];
                    const ggml_tensor * sum = cgraph->nodes[id + 3];
                    const ggml_tensor * reshaped = cgraph->nodes[id + 4];
                    if (perm->op == GGML_OP_PERMUTE && perm->src[0] == node &&
                        perm->ne[0] == node->ne[1] && perm->ne[1] == node->ne[0] &&
                        perm->ne[2] == node->ne[2] && perm->ne[3] == node->ne[3] &&
                        perm->nb[0] == node->nb[1] && perm->nb[1] == node->nb[0] &&
                        perm->nb[2] == node->nb[2] && perm->nb[3] == node->nb[3] &&
                        cont->op == GGML_OP_CONT && cont->src[0] == perm &&
                        sum->op == GGML_OP_SUM_ROWS && sum->src[0] == cont &&
                        reshaped->op == GGML_OP_RESHAPE && reshaped->src[0] == sum &&
                        !((perm->flags | cont->flags | sum->flags) & GGML_TENSOR_FLAG_OUTPUT) &&
                        ggml_node_get_use_count(cgraph, id + 1) == 1 &&
                        ggml_node_get_use_count(cgraph, id + 2) == 1 &&
                        ggml_node_get_use_count(cgraph, id + 3) == 1) {
                        // CUDA may fuse this complete linear expert tail. Its
                        // stage visited set omits other stages and leaf aliases,
                        // while use_counts only counts ordinary src edges.
                        // Close those aliases here in the complete source graph.
                        auto aliases_omitted = [&](const ggml_tensor * t) {
                            for (const ggml_tensor * base = t->view_src; base; base = base->view_src) {
                                for (int k = i; k <= id + 2; ++k) {
                                    if (base == cgraph->nodes[k]) {
                                        return true;
                                    }
                                }
                            }
                            return false;
                        };
                        for (int k = 0; k < cgraph->n_nodes; ++k) {
                            if ((k < i || k > id + 4) && aliases_omitted(cgraph->nodes[k])) {
                                return i;
                            }
                        }
                        for (int k = 0; k < cgraph->n_leafs; ++k) {
                            if (aliases_omitted(cgraph->leafs[k])) {
                                return i;
                            }
                        }
                        // SUM_ROWS and the final reshape share the materialized
                        // small result, so aliases of that result stay valid.
                        return id + 4;
                    }
                }

                if (n_used != node->ne[1] || id + 2*n_used-1 >= cgraph->n_nodes) {
                    return idr;
                }
                for (int32_t k = 0; k < n_used; k++) {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op != GGML_OP_VIEW || next->view_src != node || next->view_offs != k*node->nb[1] ||
                            next->ne[0] != node->ne[0] || next->ne[1] != node->ne[2] || next->nb[1] != node->nb[2] ||
                            ggml_node_get_use_count(cgraph, id+1) != 1) {
                        return idr;
                    }
                    id++;
                }
                {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op != GGML_OP_ADD || next->src[0] != cgraph->nodes[id - (n_used-1)] ||
                            next->src[1] != cgraph->nodes[id - (n_used-2)] || ggml_node_get_use_count(cgraph, id+1) != 1) {
                        return idr;
                    }
                    id++;
                }
                for (int32_t k = 0; k < n_used - 2; k++) {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op != GGML_OP_ADD || next->src[0] != cgraph->nodes[id] ||
                            next->src[1] != cgraph->nodes[id - (n_used-2)] || ggml_node_get_use_count(cgraph, id+1) != 1) {
                        return idr;
                    }
                    id++;
                }
                idr = id;
                return idr;
            };

            int i_start = 0;
            size_t hc_actual_joins = 0;
            size_t shared_actual_joins = 0;
            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];
                if (node->view_src != nullptr && node->view_src->op == GGML_OP_NONE && ggml_backend_buffer_is_host(node->view_src->buffer)) {
                    continue;
                }
                const ggml_backend_meta_split_state split_state = execution_split_state(node, /*assume_sync =*/ false);
                if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                    max_tmp_size = std::max(max_tmp_size, ggml_nbytes(node));
                }
                const bool new_subgraph = i + 1 == cgraph->n_nodes || split_state.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL;
                if (!new_subgraph) {
                    continue;
                }

                const int i_delayed = get_i_delayed(i);
                if (shared_enabled && shared_join_ends.count(i)) {
                    ++shared_actual_joins;
                }

                // If we can delay the AllReduce we need to consider the interaction with zero-sized tensor slices.
                // A backend with such a slice would normally have valid data after participating in the AllReduce with a node that has
                //     its compute flag disabled and thus gets its data zeroed out.
                // If the AllReduce is delayed then the nodes until that point also need to have their compute flag disabled.
                if (i_delayed > i) {
                    for (size_t j = 0; j < n_backends; j++) {
                        auto & bcj = backend_ctx->backend_configs[j];
                        if ((bcj.nodes[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                            for (int ii = i + 1; ii <= i_delayed; ii++) {
                                bcj.nodes[ii]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
                            }
                        }
                    }
                }

                if (backend_ctx->hc_owner_configured) {
                    const auto found = backend_ctx->hc_owner_domains.find(node);
                    if (found != backend_ctx->hc_owner_domains.end() &&
                            found->second.split.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                        const auto pair_it = std::find(backend_ctx->hc_owner_up_indices.begin(),
                                                      backend_ctx->hc_owner_up_indices.end(), i);
                        if (pair_it == backend_ctx->hc_owner_up_indices.end()) {
                            backend_ctx->relay_i->abort(backend_ctx->relay, GGML_STATUS_FAILED);
                            return GGML_STATUS_FAILED;
                        }
                        const size_t pair = size_t(pair_it - backend_ctx->hc_owner_up_indices.begin());
                        if (i_delayed != backend_ctx->hc_owner_terminal_indices[pair]) {
                            GGML_LOG_ERROR("%s: HC context-owner late-join changed after qualification\n", __func__);
                            backend_ctx->relay_i->abort(backend_ctx->relay, GGML_STATUS_FAILED);
                            return GGML_STATUS_FAILED;
                        }
                        for (size_t rank = 0; rank < n_backends; ++rank) {
                            if (child_computes(rank, i_delayed) != (rank == found->second.rank)) {
                                backend_ctx->relay_i->abort(backend_ctx->relay, GGML_STATUS_FAILED);
                                return GGML_STATUS_FAILED;
                            }
                        }
                        ++hc_actual_joins;
                    }
                }
                i = i_delayed;

                for (size_t j = 0; j < n_backends; j++) {
                    auto & bcj = backend_ctx->backend_configs[j];
                    bcj.cgraphs[n_subgraphs].offset = i_start;
                }
                n_subgraphs++;
                i_start = i + 1;
            }
            GGML_ASSERT(i_start == cgraph->n_nodes);
            if (shared_enabled) {
                const size_t expected_channels = shared_expected_joins == 47 ? 178 : 181;
                if (shared_actual_joins != shared_expected_joins || n_subgraphs != expected_channels + 1) {
                    GGML_LOG_ERROR("%s: shared-expert shard requires %zu combined FFN joins and %zu total channels, got %zu and %zu\n",
                                   __func__, shared_expected_joins, expected_channels, shared_actual_joins,
                                   n_subgraphs ? n_subgraphs - 1 : 0);
                    backend_ctx->relay_i->abort(backend_ctx->relay, GGML_STATUS_FAILED);
                    return GGML_STATUS_FAILED;
                }
            }
            if (backend_ctx->hc_owner_configured) {
                const size_t expected_hc = backend_ctx->hc_owner_up_indices.size();
                const size_t expected_channels = expected_hc == 95 ? 178 : 181;
                if (hc_actual_joins != expected_hc || n_subgraphs != expected_channels + 1) {
                    GGML_LOG_ERROR("%s: HC context-owner requires %zu HC joins and %zu total channels, got %zu and %zu\n",
                                   __func__, expected_hc, expected_channels, hc_actual_joins,
                                   n_subgraphs ? n_subgraphs - 1 : 0);
                    backend_ctx->relay_i->abort(backend_ctx->relay, GGML_STATUS_FAILED);
                    return GGML_STATUS_FAILED;
                }
            }
        }

        backend_ctx->uid         = cgraph->uid;
        backend_ctx->n_subgraphs = n_subgraphs;

        if (!backend_ctx->relay_mode && !backend_ctx->cpu_staged_f32 && max_tmp_size > backend_ctx->max_tmp_size) {
            for (size_t j = 0; j < n_backends; j++) {
                auto & bcj = backend_ctx->backend_configs[j];
                for (size_t i = 0; i < backend_ctx->n_reduce_steps; i++) {
                    bcj.bufs[i].reset(ggml_backend_alloc_buffer(bcj.backend, max_tmp_size));
                }
            }
            backend_ctx->max_tmp_size = max_tmp_size;
        }

        if (max_nnodes_raised || n_subgraphs > backend_ctx->max_subgraphs) {
            backend_ctx->max_subgraphs = std::max(backend_ctx->max_subgraphs, n_subgraphs);
            const bool native_or_cpu_staged = backend_ctx->relay_mode || backend_ctx->cpu_staged_f32;
            const size_t n_nodes_per_device = native_or_cpu_staged ? 0 : 3 * backend_ctx->n_reduce_steps; // tmp + ADD (+zeroing) graph per step and device
            const size_t n_cgraphs_per_device = native_or_cpu_staged ? 0 : 2 * backend_ctx->n_reduce_steps; // ADD ( + zeroing) graph per step and device
            const size_t mem_per_device_graphs_main = backend_ctx->max_subgraphs*ggml_graph_overhead_custom(backend_ctx->max_nnodes, cgraph->grads);
            const size_t mem_per_device_graphs_aux = n_cgraphs_per_device*backend_ctx->max_subgraphs*ggml_graph_overhead_custom(1, cgraph->grads);
            const size_t mem_per_device_nodes_aux = n_nodes_per_device*backend_ctx->max_subgraphs*ggml_tensor_overhead();
            const ggml_init_params params = {
                /*.mem_size   =*/ n_backends * (mem_per_device_graphs_main + mem_per_device_graphs_aux + mem_per_device_nodes_aux),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            backend_ctx->ctx.reset(ggml_init(params));
            for (size_t j = 0; j < n_backends; j++) {
                auto & bcj = backend_ctx->backend_configs[j];
                // Replacing ctx invalidates every previous graph pointer.
                // Rebuild both retained capacity dimensions: a later profile
                // can grow stages or nodes without raising either maximum.
                for (size_t i = 0; i < backend_ctx->max_subgraphs; i++) {
                    bcj.cgraphs[i].cgraph_main = ggml_new_graph_custom(backend_ctx->ctx.get(), backend_ctx->max_nnodes, /*grads =*/ false);
                }
            }
            backend_ctx->cgraphs_aux.resize(n_backends*n_cgraphs_per_device*backend_ctx->max_subgraphs);
            for (size_t k = 0; k < backend_ctx->cgraphs_aux.size(); k++) {
                backend_ctx->cgraphs_aux[k] = ggml_new_graph_custom(backend_ctx->ctx.get(), 1, cgraph->grads);
            }
            backend_ctx->nodes_aux.resize(n_backends*n_nodes_per_device*backend_ctx->max_subgraphs);
            for (size_t k = 0; k < backend_ctx->nodes_aux.size(); k++) {
                backend_ctx->nodes_aux[k] = ggml_new_tensor_1d(backend_ctx->ctx.get(), GGML_TYPE_F32, 1);
            }
        }

        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            for (size_t i_graph = 0; i_graph < n_subgraphs; i_graph++) {
                ggml_cgraph * cgraph_ij = bcj.cgraphs[i_graph].cgraph_main;
                const size_t i_node_start = bcj.cgraphs[i_graph].offset;
                const size_t i_node_stop = i_graph + 1 < n_subgraphs ? bcj.cgraphs[i_graph + 1].offset : cgraph->n_nodes;
                cgraph_ij->n_nodes = i_node_stop - i_node_start;
                ggml_hash_set_reset(&cgraph_ij->visited_hash_set);
                for (size_t i_node = i_node_start; i_node < i_node_stop; i_node++) {
                    ggml_tensor * node_ij = bcj.nodes[i_node];
                    cgraph_ij->nodes[i_node - i_node_start] = node_ij;
                    const size_t hash_pos_orig = ggml_hash_find(&cgraph->visited_hash_set, cgraph->nodes[i_node]);
                    const size_t hash_pos_ij = ggml_hash_insert(&cgraph_ij->visited_hash_set, node_ij);
                    cgraph_ij->use_counts[hash_pos_ij] = cgraph->use_counts[hash_pos_orig];
                }
                cgraph_ij->uid = ggml_graph_next_uid();
            }
        }
    }

    const int64_t diag_built = ggml_time_us();
    if (backend_ctx->relay_mode) {
        const auto * ri = backend_ctx->relay_i;
        void * relay = backend_ctx->relay;
        if (backend_ctx->n_subgraphs == 0) {
            ri->abort(relay, GGML_STATUS_FAILED);
            return GGML_STATUS_FAILED;
        }
        const size_t channels = backend_ctx->n_subgraphs - 1;
        backend_ctx->relay_terminals.resize(channels * n_backends);
        for (size_t channel = 0; channel < channels; ++channel) {
            for (size_t rank = 0; rank < n_backends; ++rank) {
                ggml_cgraph * stage = backend_ctx->backend_configs[rank].cgraphs[channel].cgraph_main;
                if (stage->n_nodes == 0) {
                    ri->abort(relay, GGML_STATUS_FAILED);
                    return GGML_STATUS_FAILED;
                }
                backend_ctx->relay_terminals[channel * n_backends + rank] = stage->nodes[stage->n_nodes - 1];
            }
        }
        // Every rank/stage tensor has been converted before preparing any
        // native communication storage. No lazy allocation after begin.
        if (needs_rebuild && !ri->prepare(relay, backend_ctx->relay_terminals.data(), channels)) {
            ri->abort(relay, GGML_STATUS_FAILED);
            return GGML_STATUS_FAILED;
        }
        backend_ctx->relay_graphs.resize(backend_ctx->n_subgraphs * n_backends);
        for (size_t stage = 0; stage < backend_ctx->n_subgraphs; ++stage) {
            for (size_t rank = 0; rank < n_backends; ++rank) {
                backend_ctx->relay_graphs[stage * n_backends + rank] =
                    backend_ctx->backend_configs[rank].cgraphs[stage].cgraph_main;
            }
        }
        const int64_t diag_submit = ggml_time_us();
        try {
            // Inputs have completed, every child graph is private and prepare has
            // released its provider lock. Nothing is active until submit begins.
            if (tp5_meta_submit_ready_hook &&
                    !tp5_meta_submit_ready_hook(tp5_meta_submit_ready_userdata)) {
                ri->abort(relay, GGML_STATUS_FAILED);
                return GGML_STATUS_FAILED;
            }
            if (!ri->submit(relay, backend_ctx->relay_graphs.data(), backend_ctx->n_subgraphs)) {
                ri->abort(relay, GGML_STATUS_FAILED);
                return GGML_STATUS_FAILED;
            }
            // All private child graphs are accepted and sealed. No shared
            // conversion follows; only this context's diagnostics and return.
            if (tp5_meta_submitted_hook &&
                    !tp5_meta_submitted_hook(tp5_meta_submitted_userdata)) {
                ri->abort(relay, GGML_STATUS_FAILED);
                return GGML_STATUS_FAILED;
            }
        } catch (...) {
            // The scheduler may drain retained uploads while unwinding. Release
            // any partially submitted relay waits before that stream fence.
            ri->abort(relay, GGML_STATUS_FAILED);
            throw;
        }
        if (std::getenv("TP5_DIAG_GRAPH")) {
            GGML_LOG_INFO("TP5_DIAG_META ctx=%p uid=%llu rebuild=%d nodes=%d channels=%zu pre_us=%lld plan_us=%lld prepare_us=%lld submit_us=%lld total_us=%lld\n",
                (void *) backend_ctx, (unsigned long long)cgraph->uid, int(needs_rebuild), cgraph->n_nodes, channels,
                (long long)(diag_prepared-diag_start),(long long)(diag_built-diag_prepared),
                (long long)(diag_submit-diag_built),(long long)(ggml_time_us()-diag_submit),(long long)(ggml_time_us()-diag_start));
        }
        return GGML_STATUS_SUCCESS;
    }

    if (backend_ctx->cpu_staged_f32) {
        static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559 &&
                      std::numeric_limits<float>::digits == 24, "CPU-staged SUM requires IEEE binary32");
        GGML_ASSERT(backend_ctx->comm_ctx == nullptr);
        auto drain = [&]() {
            for (auto & bc : backend_ctx->backend_configs) {
                ggml_backend_synchronize(bc.backend);
            }
        };
        // Transfers are synchronous and scratch belongs to this Meta context.
        // Pack views by their real strides, never padding or adjacent storage.
        auto transfer_f32 = [](ggml_tensor * tensor, float * data, bool upload) {
            auto transfer = [&](size_t offset, size_t width, size_t copies, size_t stride_tensor, float * packed) {
                if (upload) {
                    ggml_backend_tensor_set_2d(tensor, packed, offset, width, copies, stride_tensor, width);
                } else {
                    ggml_backend_tensor_get_2d(tensor, packed, offset, width, copies, stride_tensor, width);
                }
            };
            if (ggml_is_contiguous(tensor)) {
                if (upload) {
                    ggml_backend_tensor_set(tensor, data, 0, ggml_nbytes(tensor));
                } else {
                    ggml_backend_tensor_get(tensor, data, 0, ggml_nbytes(tensor));
                }
                return;
            }
            for (int64_t i3 = 0; i3 < tensor->ne[3]; ++i3) {
                for (int64_t i2 = 0; i2 < tensor->ne[2]; ++i2) {
                    const size_t plane = i3 * tensor->nb[3] + i2 * tensor->nb[2];
                    if (tensor->nb[0] == sizeof(float)) {
                        transfer(plane, tensor->ne[0] * sizeof(float), tensor->ne[1], tensor->nb[1], data);
                        data += tensor->ne[0] * tensor->ne[1];
                    } else {
                        for (int64_t i1 = 0; i1 < tensor->ne[1]; ++i1) {
                            transfer(plane + i1 * tensor->nb[1], sizeof(float), tensor->ne[0], tensor->nb[0], data);
                            data += tensor->ne[0];
                        }
                    }
                }
            }
        };
        for (size_t stage = 0; stage < backend_ctx->n_subgraphs; ++stage) {
            for (auto & bc : backend_ctx->backend_configs) {
                const ggml_status status = ggml_backend_graph_compute_async(bc.backend, bc.cgraphs[stage].cgraph_main);
                if (status != GGML_STATUS_SUCCESS) {
                    drain();
                    return status;
                }
            }
            drain();
            const bool reduce = n_backends > 1 && stage + 1 < backend_ctx->n_subgraphs;
            size_t elements = 0;
            ggml_tensor * first = nullptr;
            for (size_t rank = 0; rank < n_backends; ++rank) {
                ggml_cgraph * graph = backend_ctx->backend_configs[rank].cgraphs[stage].cgraph_main;
                if (graph->n_nodes == 0) {
                    GGML_LOG_ERROR("%s: empty CPU-staged rank graph\n", __func__);
                    return GGML_STATUS_FAILED;
                }
                ggml_tensor * node = graph->nodes[graph->n_nodes - 1];
                if (node->type != GGML_TYPE_F32) {
                    if (!reduce) {
                        continue;
                    }
                    GGML_LOG_ERROR("%s: CPU-staged SUM requires F32 rank partials\n", __func__);
                    return GGML_STATUS_FAILED;
                }
                if (reduce) {
                    if (rank == 0) {
                        first = node;
                        elements = ggml_nelements(node);
                        backend_ctx->staged_sum.resize(elements);
                    } else {
                        for (size_t dim = 0; dim < GGML_MAX_DIMS; ++dim) {
                            if (node->ne[dim] != first->ne[dim]) {
                                GGML_LOG_ERROR("%s: mismatched CPU-staged rank partial shapes\n", __func__);
                                return GGML_STATUS_FAILED;
                            }
                        }
                    }
                } else {
                    elements = ggml_nelements(node);
                }
                auto & partial = backend_ctx->staged_partial;
                partial.resize(elements);
                if (reduce && !(node->flags & GGML_TENSOR_FLAG_COMPUTE)) {
                    // A disabled zero-width shard contributes exact +0, not
                    // 0 * uninitialized data (which can preserve a NaN).
                    std::fill(partial.begin(), partial.end(), 0.0f);
                } else if (elements != 0) {
                    transfer_f32(node, partial.data(), false);
                }
                for (size_t element = 0; element < elements; ++element) {
                    if (!std::isfinite(partial[element])) {
                        GGML_LOG_ERROR("%s: nonfinite CPU-staged F32 value at stage %zu rank %zu element %zu\n",
                                       __func__, stage, rank, element);
                        return GGML_STATUS_FAILED;
                    }
                    if (reduce) {
                        // Store every rank boundary in binary32: exactly
                        // (((rank0 + rank1) + rank2) + rank3) + rank4 for TP5.
                        auto & sum = backend_ctx->staged_sum;
                        sum[element] = rank == 0 ? partial[element] : sum[element] + partial[element];
                        if (!std::isfinite(sum[element])) {
                            GGML_LOG_ERROR("%s: nonfinite CPU-staged F32 SUM at stage %zu rank %zu element %zu\n",
                                           __func__, stage, rank, element);
                            return GGML_STATUS_FAILED;
                        }
                    }
                }
            }
            if (reduce && elements != 0) {
                // Publish only after all inputs and ordered sums pass finite
                // checks. No GPU arithmetic participates in this SUM.
                for (auto & bc : backend_ctx->backend_configs) {
                    ggml_cgraph * graph = bc.cgraphs[stage].cgraph_main;
                    transfer_f32(graph->nodes[graph->n_nodes - 1], backend_ctx->staged_sum.data(), true);
                }
            }
        }
        // Existing readback gathers sharded outputs or selects mirrored rank 0.
        // Final child work and all host transfers are already complete.
        return GGML_STATUS_SUCCESS;
    }

    size_t iga = 0; // i graph aux
    size_t ina = 0; // i node aux

    auto get_node_aux = [&](ggml_tensor * t) -> ggml_tensor * {
        ggml_tensor * ret = backend_ctx->nodes_aux[ina++];
        memset(ret, 0, sizeof(ggml_tensor));
        ret->op   = GGML_OP_NONE;
        ret->type = t->type;
        for (size_t k = 0; k < GGML_MAX_DIMS; k++) {
            ret->ne[k] = t->ne[k];
            ret->nb[k] = t->nb[k];
        }
        return ret;
    };
    auto set_tmp_data = [&](ggml_tensor * tensor, const size_t j, const size_t i_buf) {
        auto & bcj = backend_ctx->backend_configs[j];
        ggml_backend_buffer_ptr & buf_ptr = bcj.bufs[i_buf];
        if (!buf_ptr || ggml_backend_buffer_get_size(buf_ptr.get()) < backend_ctx->max_tmp_size) {
            buf_ptr.reset(ggml_backend_alloc_buffer(bcj.backend, backend_ctx->max_tmp_size));
        }
        tensor->buffer = buf_ptr.get();
        tensor->data   = ggml_backend_buffer_get_base(buf_ptr.get());
    };
    // FIXME usage_counts
    auto get_cgraph_aux = [&]() -> ggml_cgraph * {
        ggml_cgraph * ret = backend_ctx->cgraphs_aux[iga++];
        return ret;
    };

    // Preferentially use backend-specific allreduce_tensor_async (e.g. NCCL for CUDA), use a generic fallback if unavailable:
    auto allreduce_fallback = [&](size_t i) -> ggml_status {
        std::vector<ggml_cgraph *> step_cgraphs(n_backends, nullptr);

        // Zero out nodes that were disabled due to having a zero-sized slice:
        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            ggml_tensor * node = bcj.cgraphs[i].cgraph_main->nodes[bcj.cgraphs[i].cgraph_main->n_nodes - 1];
            if (node->flags & GGML_TENSOR_FLAG_COMPUTE) {
                continue;
            }
            ggml_tensor * node_zero = get_node_aux(node);
            node_zero->op = GGML_OP_SCALE; // FIXME 0.0f * NaN == NaN
            node_zero->src[0] = node;
            ggml_set_op_params_f32(node_zero, 0, 0.0f);
            node_zero->data = node->data;
            node_zero->buffer = node->buffer;
            node_zero->flags |= GGML_TENSOR_FLAG_COMPUTE;

            step_cgraphs[j] = get_cgraph_aux();
            step_cgraphs[j]->nodes[0] = node_zero;
            step_cgraphs[j]->n_nodes = 1;
            const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, step_cgraphs[j]);
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
        }
        std::fill(step_cgraphs.begin(), step_cgraphs.end(), nullptr);

        auto push_data = [&](const size_t j_src, const size_t j_dst, const size_t i_buf) {
            assert(step_cgraphs[j_dst] == nullptr);
            auto & bcj_src = backend_ctx->backend_configs[j_src];
            auto & bcj_dst = backend_ctx->backend_configs[j_dst];

            ggml_tensor * node_src = bcj_src.cgraphs[i].cgraph_main->nodes[bcj_src.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_tensor * node_dst = bcj_dst.cgraphs[i].cgraph_main->nodes[bcj_dst.cgraphs[i].cgraph_main->n_nodes - 1];
            GGML_ASSERT(ggml_is_contiguous(node_src));
            GGML_ASSERT(ggml_is_contiguous(node_dst));

            ggml_tensor * node_tmp = get_node_aux(node_dst);
            set_tmp_data(node_tmp, j_dst, i_buf);

            ggml_backend_tensor_copy_async(bcj_src.backend, bcj_dst.backend, node_src, node_tmp);

            ggml_tensor * node_red = get_node_aux(node_dst);
            node_red->view_src = node_dst->view_src == nullptr ? node_dst : node_dst->view_src;
            node_red->view_offs = node_dst->view_offs;
            node_red->op = GGML_OP_ADD;
            node_red->src[0] = node_dst;
            node_red->src[1] = node_tmp;
            node_red->flags |= GGML_TENSOR_FLAG_COMPUTE;
            ggml_backend_view_init(node_red);

            ggml_cgraph * cgraph_aux = get_cgraph_aux();
            cgraph_aux->nodes[0] = node_red;
            cgraph_aux->n_nodes = 1;
            step_cgraphs[j_dst] = cgraph_aux;
        };

        size_t offset_j = n_backends/2;
        while ((offset_j & (offset_j - 1)) != 0) {
            offset_j--;
        }
        const size_t offset_j_max = offset_j;
        size_t i_buf = 0;

        // If n_backends is not a power of 2, fold in the excess prior to butterfly reduction:
        for (size_t j_src = 2*offset_j_max; j_src < n_backends; j_src++) {
            const size_t j_dst = j_src - 2*offset_j_max;
            push_data(j_src, j_dst, i_buf);
            const ggml_status status = ggml_backend_graph_compute_async(backend_ctx->backend_configs[j_dst].backend, step_cgraphs[j_dst]);
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
            i_buf = 1;
        }

        // Butterfly reduction:
        for (; offset_j >= 1; offset_j /= 2) {
            std::fill(step_cgraphs.begin(), step_cgraphs.end(), nullptr);

            for (size_t j = 0; j < 2*offset_j_max; j++) {
                const size_t j_other = j ^ offset_j;
                if (j_other >= n_backends) {
                    continue;
                }
                push_data(j, j_other, i_buf);
            }

            for (size_t j = 0; j < 2*offset_j_max; j++) {
                if (step_cgraphs[j] == nullptr) {
                    continue;
                }
                auto & bcj = backend_ctx->backend_configs[j];
                const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, step_cgraphs[j]);
                if (status != GGML_STATUS_SUCCESS) {
                    return status;
                }
            }
            i_buf++;
        }
        assert(i_buf == backend_ctx->n_reduce_steps);

        // If n_backends is not a power of 2, copy back the reduced tensors to the excess:
        for (size_t j = 2*offset_j_max; j < n_backends; j++) {
            auto & bcj_src = backend_ctx->backend_configs[j - 2*offset_j_max];
            auto & bcj_dst = backend_ctx->backend_configs[j];

            ggml_tensor * node_src = bcj_src.cgraphs[i].cgraph_main->nodes[bcj_src.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_tensor * node_dst = bcj_dst.cgraphs[i].cgraph_main->nodes[bcj_dst.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_backend_tensor_copy_async(bcj_src.backend, bcj_dst.backend, node_src, node_dst);
        }

        return GGML_STATUS_SUCCESS;
    };


    for (size_t i = 0; i < backend_ctx->n_subgraphs; i++) {
        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, bcj.cgraphs[i].cgraph_main);
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
        }

        if (n_backends > 1 && i < backend_ctx->n_subgraphs - 1) {
            bool backend_allreduce_success = false;
            if (backend_ctx->comm_ctx) {
                std::vector<ggml_tensor *> nodes;
                nodes.reserve(n_backends);
                for (size_t j = 0; j < n_backends; j++) {
                    auto & bcj = backend_ctx->backend_configs[j];
                    ggml_cgraph * cgraph_ij = bcj.cgraphs[i].cgraph_main;
                    nodes.push_back(cgraph_ij->nodes[cgraph_ij->n_nodes-1]);
                }
                backend_allreduce_success = backend_ctx->comm_allreduce(backend_ctx->comm_ctx, nodes.data());
            }

            if (!backend_allreduce_success) {
                const ggml_status status = allreduce_fallback(i);
                if (status != GGML_STATUS_SUCCESS) {
                    return status;
                }
            }
        }
    }
    return GGML_STATUS_SUCCESS;
}

static const ggml_backend_i ggml_backend_meta_i = {
    /* .get_name                = */ ggml_backend_meta_get_name,
    /* .free                    = */ ggml_backend_meta_free,
    /* .set_tensor_async        = */ ggml_backend_meta_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_meta_get_tensor_async,
    /* .set_tensor_2d_async     = */ nullptr,
    /* .get_tensor_2d_async     = */ nullptr,
    /* .cpy_tensor_async        = */ ggml_backend_meta_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_meta_synchronize,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ ggml_backend_meta_graph_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .graph_optimize          = */ nullptr,
};

bool ggml_backend_is_meta(ggml_backend_t backend) {
    return backend != nullptr && backend->iface.get_name == ggml_backend_meta_i.get_name;
}

static ggml_backend_t ggml_backend_meta_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    ggml_backend_meta_context * backend_ctx = new ggml_backend_meta_context(dev, params);
    if (!backend_ctx->valid) {
        delete backend_ctx;
        return nullptr;
    }

    ggml_backend_t backend = new struct ggml_backend;
    backend->guid    = ggml_backend_meta_guid();
    backend->iface   = ggml_backend_meta_i;
    backend->device  = dev;
    backend->context = backend_ctx;
    return backend;
}

size_t ggml_backend_meta_n_backends(ggml_backend_t meta_backend) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    const ggml_backend_meta_context * backend_ctx = (const ggml_backend_meta_context *) meta_backend->context;
    return backend_ctx->backend_configs.size();
}

ggml_backend_t ggml_backend_meta_simple_backend(ggml_backend_t meta_backend, size_t index) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    const ggml_backend_meta_context * backend_ctx = (const ggml_backend_meta_context *) meta_backend->context;
    return backend_ctx->backend_configs[index].backend;
}

static ggml_backend_meta_context * ggml_backend_relay_meta_context(ggml_backend_t backend) {
    if (!ggml_backend_is_meta(backend)) {
        return nullptr;
    }
    auto * ctx = (ggml_backend_meta_context *) backend->context;
    return ctx->relay_mode && ctx->relay != nullptr ? ctx : nullptr;
}

int ggml_backend_relay_poll(ggml_backend_t backend, ggml_relay_event * event) {
    auto * ctx = ggml_backend_relay_meta_context(backend);
    return ctx != nullptr && event != nullptr ? ctx->relay_i->poll(ctx->relay, event) : -1;
}

bool ggml_backend_relay_publish(ggml_backend_t backend, const ggml_relay_event * event) {
    auto * ctx = ggml_backend_relay_meta_context(backend);
    return ctx != nullptr && event != nullptr && ctx->relay_i->publish(ctx->relay, event);
}

bool ggml_backend_relay_set_hc_owner_offset(ggml_backend_t backend, uint32_t offset) {
    auto * ctx = ggml_backend_relay_meta_context(backend);
    const char * layout = std::getenv("GGML_TP5_HC_CONTEXT_OWNER");
    if (ctx == nullptr || offset >= 5 || ctx->backend_configs.size() != 5 ||
        ctx->hc_owner_started || ctx->hc_owner_configured || !layout || std::strcmp(layout, "1") != 0) {
        return false;
    }
    ctx->hc_owner_offset = offset;
    ctx->hc_owner_configured = true;
    return true;
}

bool ggml_backend_relay_set_wire_type(ggml_backend_t backend, ggml_relay_wire_type wire_type) {
    auto * ctx = ggml_backend_relay_meta_context(backend);
    if (ctx == nullptr || ctx->relay_wire_i == nullptr ||
            !ctx->relay_wire_i->set_wire_type(ctx->relay, wire_type)) {
        return false;
    }
    ctx->relay_wire_type = wire_type;
    return true;
}

int ggml_backend_relay_poll_v2(ggml_backend_t backend, ggml_relay_event_v2 * event, size_t event_size) {
    auto * ctx = ggml_backend_relay_meta_context(backend);
    return ctx != nullptr && ctx->relay_wire_i != nullptr && event != nullptr &&
           event_size >= sizeof(*event) ?
           ctx->relay_wire_i->poll_v2(ctx->relay, event, event_size) : -1;
}

bool ggml_backend_relay_publish_v2(
        ggml_backend_t backend, const ggml_relay_event_v2 * event, size_t event_size) {
    auto * ctx = ggml_backend_relay_meta_context(backend);
    return ctx != nullptr && ctx->relay_wire_i != nullptr && event != nullptr &&
           event_size >= sizeof(*event) &&
           ctx->relay_wire_i->publish_v2(ctx->relay, event, event_size);
}

bool ggml_backend_relay_snapshot(ggml_backend_t backend, ggml_relay_status * status) {
    auto * ctx = ggml_backend_relay_meta_context(backend);
    return ctx != nullptr && status != nullptr && ctx->relay_i->snapshot(ctx->relay, status);
}

void ggml_backend_relay_abort(ggml_backend_t backend, int32_t error_code) {
    auto * ctx = ggml_backend_relay_meta_context(backend);
    if (ctx != nullptr) {
        ctx->relay_i->abort(ctx->relay, error_code);
    }
}
