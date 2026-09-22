#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-alloc.h"
#include "ggml-cpp.h"
#include "ggml-tp5-profile.h"
#include "ggml-device-copy.h"
#include "ggml-predefined.h"

typedef bool (*ggml_backend_comm_prepare_graph_t)(void *               comm_ctx,
                                                  size_t               rank,
                                                  struct ggml_cgraph * cgraph,
                                                  bool                 reduce,
                                                  size_t               stage);

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
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
// TP5 private local-node head map (see ggml-backend.h for the ABI contract)
//

void ggml_tp5_headmap_set(struct ggml_tensor * op, const uint8_t * local_heads, int32_t count) {
    GGML_ASSERT(op != nullptr);
    GGML_ASSERT(local_heads != nullptr);
    GGML_ASSERT(count >= 1 && count <= GGML_TP5_HEADMAP_MAX_ENTRIES);
    uint32_t packed = 0;
    for (int32_t i = 0; i < count; ++i) {
        GGML_ASSERT(local_heads[i] < 8);
        packed |= uint32_t(local_heads[i]) << (3u * i);
    }
    ggml_set_op_params_i32(op, 8, GGML_TP5_HEADMAP_MAGIC);
    ggml_set_op_params_i32(op, 9, int32_t(packed));
    ggml_set_op_params_i32(op, 10, count);
}

int32_t ggml_tp5_headmap_get(const struct ggml_tensor * op, uint8_t * local_heads) {
    GGML_ASSERT(op != nullptr);
    if (ggml_get_op_params_i32(op, 8) != GGML_TP5_HEADMAP_MAGIC) {
        return 0;
    }
    const uint32_t packed = uint32_t(ggml_get_op_params_i32(op, 9));
    const int32_t  count  = ggml_get_op_params_i32(op, 10);
    if (count < 1 || count > GGML_TP5_HEADMAP_MAX_ENTRIES) {
        return -1;
    }
    for (int32_t i = 0; i < count; ++i) {
        const uint32_t h = (packed >> (3u * i)) & 0x7u;
        // A 3-bit field can only encode 0..7; consumers validate the decoded
        // index against the local head count of the specific op inputs.
        if (local_heads != nullptr) {
            local_heads[i] = uint8_t(h);
        }
    }
    return count;
}

//
// meta backend device
//

struct ggml_backend_meta_device_context {
    std::vector<ggml_backend_dev_t>     simple_devs;
    ggml_backend_meta_get_split_state_t get_split_state;
    void *                              get_split_state_ud;

    // TP5 local-node head map hook (per device, no process globals): stamps
    // the private head-map ABI on rank-local FLASH_ATTN_EXT / GATED_DELTA_NET
    // clones. NULL keeps native uniform behavior. Reuses get_split_state_ud
    // as userdata so a second model's device never sees the first's hook.
    ggml_backend_meta_local_node_hook_t  local_node_hook = nullptr;

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

// TP5 local-node head map hook (opt-in): per-device, stored in the meta
// device context (see ggml-backend.h). No process globals: a second model's
// meta device carries its own hook and a freed model leaves nothing behind.
void ggml_backend_meta_set_local_node_hook(ggml_backend_dev_t meta_dev,
                                            ggml_backend_meta_local_node_hook_t hook) {
    GGML_ASSERT(ggml_backend_dev_is_meta(meta_dev));
    ggml_backend_meta_device_context * dev_ctx = (ggml_backend_meta_device_context *) meta_dev->context;
    dev_ctx->local_node_hook = hook;
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

struct ggml_tensor * ggml_backend_meta_buffer_simple_tensor(const struct ggml_tensor * tensor, size_t index) {
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


//
// TP5 mapped spans (see ggml-backend.h): bounded indexed spans mapping
// noncontiguous original-element ranges -> concatenated rank-local storage.
//

// Validate a mapped_span split state against the tensor geometry. Returns
// false (caller must abort/fail closed) when the layout is inconsistent.
static size_t ggml_meta_mapped_plane_bytes(const ggml_tensor * tensor, int axis) {
    const int64_t block = axis == 0 ? ggml_blck_size(tensor->type) : 1;
    return tensor->nb[axis] * (tensor->ne[axis] / block);
}

static bool ggml_meta_mapped_spans_valid(const ggml_tensor * tensor,
                                         const ggml_backend_meta_split_state & ss,
                                         size_t n_devices) {
    if (!ss.mapped_span) {
        return true;
    }
    if (ss.n_segments != 1 || ss.nr[0] != 1 || ss.indexed_replica) {
        return false;
    }
    if (ss.axis < 0 || ss.axis >= GGML_MAX_DIMS || n_devices > GGML_BACKEND_META_MAX_DEVICES) {
        return false;
    }
    int64_t total_spans = 0;
    for (size_t j = 0; j < n_devices; ++j) {
        if (ss.span_count[j] < 0 || ss.span_count[j] > GGML_BACKEND_META_MAX_SPANS_PER_DEVICE) {
            return false;
        }
        total_spans += ss.span_count[j];
        int64_t local_len = 0;
        for (int32_t s = 0; s < ss.span_count[j]; ++s) {
            const int64_t idx = total_spans - ss.span_count[j] + s;
            if (idx >= GGML_BACKEND_META_MAX_SPANS) {
                return false;
            }
            if (ss.span_start[idx] < 0 || ss.span_len[idx] <= 0) {
                return false;
            }
            if (ss.span_start[idx] + ss.span_len[idx] > tensor->ne[ss.axis]) {
                return false;
            }
            local_len += ss.span_len[idx];
        }
        if (local_len != ss.ne[j]) {
            return false;
        }
    }
    if (total_spans > GGML_BACKEND_META_MAX_SPANS) {
        return false;
    }
    return true;
}

// Flat span index of device j's span s (prefix-sum layout).
static int64_t ggml_meta_mapped_span_index(const ggml_backend_meta_split_state & ss, size_t j, int32_t s) {
    int64_t idx = s;
    for (size_t r = 0; r < j; ++r) {
        idx += ss.span_count[r];
    }
    return idx;
}

// Map a logical element interval [p0, p1) along the split axis to device j's
// local interval. Returns false when the interval is not fully covered by a
// single span of device j (mapped storage is span-granular: partial-span
// subranges are only legal when they fall inside one span).
static bool ggml_meta_mapped_local_range(const ggml_backend_meta_split_state & ss, size_t j,
                                         int64_t p0, int64_t p1,
                                         int64_t * local_start, int64_t * len) {
    for (int32_t s = 0; s < ss.span_count[j]; ++s) {
        const int64_t idx = ggml_meta_mapped_span_index(ss, j, s);
        const int64_t start = ss.span_start[idx];
        const int64_t slen  = ss.span_len[idx];
        if (start <= p0 && p1 <= start + slen) {
            *local_start = (p0 - start) + (s == 0 ? 0 : 0);
            // local offset accumulates previous spans of this device
            int64_t base = 0;
            for (int32_t t = 0; t < s; ++t) {
                base += ss.span_len[ggml_meta_mapped_span_index(ss, j, t)];
            }
            *local_start = base + (p0 - start);
            *len = p1 - p0;
            return true;
        }
    }
    return false;
}

// Restrict a mapped axis to a logical interval, then express its coordinates
// in the destination axis units. Selected storage must remain contiguous on
// each rank: a view cannot gather disjoint local storage behind the caller.
static ggml_backend_meta_split_state ggml_meta_mapped_slice(
        const ggml_backend_meta_split_state & src, size_t n_devices,
        int64_t begin, int64_t end, int axis, int64_t numerator, int64_t denominator) {
    GGML_ASSERT(src.mapped_span && begin >= 0 && end >= begin && numerator > 0 && denominator > 0);
    ggml_backend_meta_split_state dst = {};
    dst.axis = ggml_backend_meta_split_axis(axis);
    dst.n_segments = dst.nr[0] = 1;
    dst.mapped_span = true;
    size_t source_index = 0, destination_index = 0;
    for (size_t rank = 0; rank < n_devices; ++rank) {
        int64_t local = 0, selected_end = -1;
        for (int32_t s = 0; s < src.span_count[rank]; ++s, ++source_index) {
            const int64_t start = src.span_start[source_index];
            const int64_t length = src.span_len[source_index];
            const int64_t lo = std::max(begin, start);
            const int64_t hi = std::min(end, start + length);
            if (lo < hi) {
                const int64_t local_start = local + lo - start;
                GGML_ASSERT(selected_end < 0 || selected_end == local_start);
                selected_end = local_start + hi - lo;
                const int64_t new_start = (lo - begin) * numerator;
                const int64_t new_length = (hi - lo) * numerator;
                GGML_ASSERT(new_start % denominator == 0 && new_length % denominator == 0);
                GGML_ASSERT(destination_index < GGML_BACKEND_META_MAX_SPANS);
                dst.span_start[destination_index] = new_start / denominator;
                dst.span_len[destination_index] = new_length / denominator;
                dst.ne[rank] += dst.span_len[destination_index++];
                ++dst.span_count[rank];
            }
            local += length;
        }
    }
    return dst;
}

// The mapped coordinate lives inside one source storage plane. Outer view
// strides retain the full source plane, even when the view selects only Q,
// K, or V from a non-proportionally sharded QKV projection.
static int ggml_meta_mapped_view_axis(const ggml_tensor * view, const ggml_tensor * source,
                                      const ggml_backend_meta_split_state & src) {
    const size_t block = src.axis == 0 ? ggml_blck_size(source->type) : 1;
    const size_t plane = source->nb[src.axis] * (source->ne[src.axis] / block);
    for (int axis = 0; axis < GGML_MAX_DIMS; ++axis) {
        if (axis == GGML_MAX_DIMS - 1 || view->nb[axis + 1] >= plane) {
            return axis;
        }
    }
    GGML_ABORT("cannot locate mapped view axis for '%s'", view->name);
}

// Deterministic canonical owner of logical interval [p0, p1): the lowest
// device index with a span fully covering it. Overlapping replicas read from
// exactly one owner so no two concurrent copies write overlapping host ranges.
static int ggml_meta_mapped_owner(const ggml_backend_meta_split_state & ss, size_t n_devices,
                                  int64_t p0, int64_t p1) {
    for (size_t j = 0; j < n_devices; ++j) {
        int64_t local_start = 0, len = 0;
        if (ggml_meta_mapped_local_range(ss, j, p0, p1, &local_start, &len)) {
            return int(j);
        }
    }
    return -1;
}

// Enumerate canonical logical intervals covering [0, ne_axis) from the mapped
// spans: collect all span endpoints, sort/dedup, and for each adjacent
// interval find the lowest device whose span fully covers it. This handles
// partial overlaps correctly (e.g. r0 Q [0,4) and r1 Q [2,8) yield intervals
// [0,2) [2,4) [4,8) with lowest-device owners) — unlike a plain sort-by-
// start scan, which drops uncovered tails and does not guarantee the lowest
// device owner. Returns the number of intervals; aborts on uncovered gaps.
static size_t ggml_meta_mapped_canonical_intervals(
        const ggml_tensor * tensor,
        const ggml_backend_meta_split_state & ss,
        size_t n_devices,
        int64_t (*out_start)[2],   // [start, len) per interval
        int * out_owner,
        int64_t * out_local,       // owner-local start of the interval
        size_t max_intervals) {
    const int64_t ne_axis = tensor->ne[ss.axis];
    int64_t endpoints[2 * GGML_BACKEND_META_MAX_SPANS + 2];
    size_t n_endpoints = 0;
    endpoints[n_endpoints++] = 0;
    endpoints[n_endpoints++] = ne_axis;
    int64_t flat = 0;
    for (size_t j = 0; j < n_devices; ++j) {
        for (int32_t k = 0; k < ss.span_count[j]; ++k, ++flat) {
            endpoints[n_endpoints++] = ss.span_start[flat];
            endpoints[n_endpoints++] = ss.span_start[flat] + ss.span_len[flat];
        }
    }
    std::sort(endpoints, endpoints + n_endpoints);
    size_t n_unique = 0;
    for (size_t i = 0; i < n_endpoints; i++) {
        if (n_unique == 0 || endpoints[i] != endpoints[n_unique - 1]) {
            endpoints[n_unique++] = endpoints[i];
        }
    }
    size_t n_out = 0;
    for (size_t i = 0; i + 1 < n_unique; i++) {
        const int64_t p0 = endpoints[i];
        const int64_t p1 = endpoints[i + 1];
        if (p0 >= p1) continue;
        const int owner = ggml_meta_mapped_owner(ss, n_devices, p0, p1);
        GGML_ASSERT(owner >= 0); // no uncovered gaps allowed on readback
        int64_t local_start = 0, len = 0;
        const bool ok = ggml_meta_mapped_local_range(ss, (size_t) owner, p0, p1, &local_start, &len);
        GGML_ASSERT(ok);
        GGML_ASSERT(n_out < max_intervals);
        out_start[n_out][0] = p0;
        out_start[n_out][1] = p1 - p0;
        out_owner[n_out] = owner;
        out_local[n_out] = local_start;
        n_out++;
    }
    return n_out;
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
        if (a.indexed_replica != b.indexed_replica) {
            return false;
        }
        if (a.mapped_span != b.mapped_span) {
            return false;
        }
        if (a.mapped_span) {
            // Mapped operands are only interchangeable when every device
            // owns the SAME ordered logical spans (coalescing-equivalent
            // layouts allowed: same coverage per device in the same span
            // order after merging adjacent runs). Same per-device counts
            // with different ordered global heads must NOT elementwise
            // combine (z/gate/state identity).
            for (size_t j = 0; j < n_bufs; j++) {
                if (a.span_count[j] != b.span_count[j]) {
                    return false;
                }
            }
            auto span_at = [](const ggml_backend_meta_split_state & ss, size_t j, int k) -> int64_t {
                int64_t idx = k;
                for (size_t r = 0; r < j; r++) {
                    idx += ss.span_count[r];
                }
                return idx;
            };
            for (size_t j = 0; j < n_bufs; j++) {
                for (int k = 0; k < a.span_count[j]; k++) {
                    const int64_t ia = span_at(a, j, k);
                    const int64_t ib = span_at(b, j, k);
                    if (a.span_start[ia] != b.span_start[ib] || a.span_len[ia] != b.span_len[ib]) {
                        return false;
                    }
                }
            }
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
            if (a.indexed_replica && a.replica_start[j] != b.replica_start[j]) {
                return false;
            }
        }
        return true;
    };

    auto handle_generic = [&](const std::vector<ggml_backend_meta_split_state> & src_ss, bool scalar_only) -> ggml_backend_meta_split_state {
        ggml_backend_meta_split_state ret = { GGML_BACKEND_SPLIT_AXIS_NONE, {0}, {1}, 1, false, {0} };
        for (size_t i = 0; i < GGML_MAX_SRC; i++) {
            if (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                continue;
            }
            if (ret.axis == GGML_BACKEND_SPLIT_AXIS_NONE) {
                ret = src_ss[i];
            } else if (!split_states_equal(src_ss[i], ret)) {
                ret = { GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1, false, {0} };
                break;
            }
        }
        // TP5 mapped spans: split_states_equal already enforces matching
        // ordered span identity between mapped srcs (scalar_only callers
        // reject any split src below, mapped or not).
        if (ret.axis == GGML_BACKEND_SPLIT_AXIS_NONE) {
            ret = { GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1, false, {0} };
        }
        if (scalar_only && ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
            ret = { GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1, false, {0} };
        }
        if (ret.axis == GGML_BACKEND_SPLIT_AXIS_UNKNOWN) {
            // Name the offender instead of aborting on a bare assert (TP5.md 19.2): a
            // tensor-parallel graph reached an op whose inputs cannot be reconciled.
            char msg[1024];
            int n = snprintf(msg, sizeof(msg), "no uniform split state for '%s' (%s):",
                             tensor->name, ggml_op_name(tensor->op));
            for (size_t i = 0; i < GGML_MAX_SRC && n > 0 && n < (int) sizeof(msg); i++) {
                if (tensor->src[i] == nullptr) continue;
                n += snprintf(msg + n, sizeof(msg) - n, " [%zu] %s(%s) ne=[%lld,%lld] axis=%d segs=%u"
                              " {ne0=%lld,ne1=%lld,ne2=%lld,ne3=%lld,ne4=%lld}", i,
                              tensor->src[i]->name, ggml_op_name(tensor->src[i]->op),
                              (long long) tensor->src[i]->ne[0], (long long) tensor->src[i]->ne[1],
                              (int) src_ss[i].axis, (unsigned) src_ss[i].n_segments,
                              (long long)src_ss[i].ne[0], (long long)src_ss[i].ne[1], (long long)src_ss[i].ne[2], (long long)src_ss[i].ne[3], (long long)src_ss[i].ne[4]);
            }
            // Allow elementwise operations between MIRRORED/aligned and unevenly sharded tensors
            // (e.g. Qwen4EXP TP5 where mirrored parameters or un-aligned broadcast meets sharded activations).
            for (size_t i = 0; i < GGML_MAX_SRC; i++) {
                if (tensor->src[i] && src_ss[i].axis >= 0 && src_ss[i].axis < GGML_MAX_DIMS) {
                    return src_ss[i];
                }
            }
            GGML_ABORT("%s", msg);
        }
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
        if (src_ss[1].axis >= 0 && src_ss[1].axis < GGML_MAX_DIMS && tensor->src[0]->ne[src_ss[1].axis] == 1 &&
            src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[1];
        }
        if (src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && (src_ss[0].axis == src_ss[1].axis ||
           (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL)))) {
            return src_ss[0]; // GGML_OP_ADD_ID
        }
        // Shared expert gated addition: PARTIAL moe_out + PARTIAL ffn_shexp_gated -> PARTIAL
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
            return src_ss[0];
        }
        // PARTIAL scaled by MIRRORED gate or scalar (e.g. ffn_shexp * shared_gate):
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        if (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL && src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[1];
        }
        // If both inputs are partitioned along the same axis but with uneven per-rank head counts
        // (e.g. Qwen4EXP TP5 GDN normalized output × gate), follow the destination/lhs partition.
        // TP5 mapped spans: two mapped operands with the same per-rank counts
        // but DIFFERENT ordered global heads must not elementwise combine;
        // require matching ordered span identity (split_states_equal covers
        // coalescing-equivalent layouts). A MIRRORED scalar broadcast onto a
        // mapped operand stays legal (identical value on every element).
        if (src_ss[0].axis == src_ss[1].axis && src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            if ((src_ss[0].mapped_span || src_ss[1].mapped_span)) {
                if (src_ss[0].mapped_span != src_ss[1].mapped_span ||
                    (src_ss[0].mapped_span && !split_states_equal(src_ss[0], src_ss[1]))) {
                    GGML_ABORT("mapped-span operands of '%s' (%s) have different ordered head identities; elementwise combine would corrupt z/gate/state identity",
                               tensor->name, ggml_op_name(tensor->op));
                }
            }
            return src_ss[0];
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
        if (src_ss[0].axis == src_ss[1].axis && src_ss[0].axis == concat_axis) {
            ggml_backend_meta_split_state ret = src_ss[0];
            for (size_t j = 0; j < sizeof(ret.ne) / sizeof(ret.ne[0]); j++) {
                ret.ne[j] = src_ss[0].ne[j] + src_ss[1].ne[j];
            }
            return ret;
        }
        return handle_generic(src_ss, /*scalar_only =*/ true);
    };

    auto handle_mul_mat = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return { GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1, false, {0} };
        }
        // Column-parallel Matmul / MoE: W is split on axis 1 (rows/intermediate), x is MIRRORED.
        // Output is sharded on axis 0 (intermediate).
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            ggml_backend_meta_split_state ret = src_ss[0];
            ret.axis = GGML_BACKEND_SPLIT_AXIS_0;
            return ret;
        }
        // Row-parallel Matmul / MoE Down: W is split on axis 0 (contracting dim), x is split on axis 0.
        // Each device computes a local partial product -> PARTIAL output.
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_0) {
            return { assume_sync ? GGML_BACKEND_SPLIT_AXIS_MIRRORED : GGML_BACKEND_SPLIT_AXIS_PARTIAL,
                     { 0 },
                     { 1 },
                     1,
                     false,
                     { 0 } };
        }
        // Row-parallel with mirrored activation (e.g. down projection where input was reduced or mirrored):
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return { assume_sync ? GGML_BACKEND_SPLIT_AXIS_MIRRORED : GGML_BACKEND_SPLIT_AXIS_PARTIAL,
                     { 0 },
                     { 1 },
                     1,
                     false,
                     { 0 } };
        }
        if (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_1 && src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[1];
        }
        // Replicated weight × sharded activation (HC FFN, etc.): each rank keeps a local partial.
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
            return src_ss[1];
        }
        if (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
            return src_ss[0];
        }
        // Batched / Multi-head Matmul where both operands are partitioned along the same batch/head axis (axis 2 or 3):
        // Each device independently computes its local heads without cross-head reduction!
        if ((src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_2 || src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_3) &&
            src_ss[0].axis == src_ss[1].axis) {
            GGML_ASSERT(split_states_equal(src_ss[0], src_ss[1]));
            return src_ss[0];
        }
        char buf[512];
        snprintf(buf, sizeof(buf), "handle_mul_mat unsupported split combination on '%s': src0 '%s' (axis=%d), src1 '%s' (axis=%d)",
                 tensor->name, tensor->src[0]->name, (int)src_ss[0].axis, tensor->src[1]->name, (int)src_ss[1].axis);
        GGML_ABORT("%s", buf);
        //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
    };

    auto handle_reshape = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].mapped_span) {
            const ggml_tensor * source = tensor->src[0];
            int64_t old_inner = 1;
            for (int d = 0; d < src_ss[0].axis; ++d) {
                old_inner *= source->ne[d];
            }
            const int64_t plane = old_inner * source->ne[src_ss[0].axis];
            int64_t new_inner = 1;
            for (int axis = 0; axis < GGML_MAX_DIMS; ++axis) {
                if (new_inner * tensor->ne[axis] == plane) {
                    return ggml_meta_mapped_slice(src_ss[0], n_bufs, 0, source->ne[src_ss[0].axis],
                                                  axis, old_inner, new_inner);
                }
                new_inner *= tensor->ne[axis];
            }
            GGML_ABORT("mapped reshape changes the storage plane of '%s'", tensor->name);
        }
        switch (src_ss[0].axis) {
            case GGML_BACKEND_SPLIT_AXIS_0:
            case GGML_BACKEND_SPLIT_AXIS_1:
            case GGML_BACKEND_SPLIT_AXIS_2:
            case GGML_BACKEND_SPLIT_AXIS_3: {
                GGML_ASSERT(src_ss[0].n_segments == 1);
                if (src_ss[0].axis == ggml_n_dims(tensor->src[0]) - 1 && src_ss[0].nr[0] == 1) {
                    ggml_backend_meta_split_state ret = {ggml_backend_meta_split_axis(ggml_n_dims(tensor) - 1), {0}, {1}, 1, false, {0}};
                    if (src_ss[0].indexed_replica) {
                        ret.indexed_replica = true;
                    }
                    return ret;
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
                        ggml_backend_meta_split_state ret = {ggml_backend_meta_split_axis(dim), {0}, {uint32_t(base_ne_out_next/base_ne_in)}, 1, false, {0}};
                        if (src_ss[0].indexed_replica) {
                            ret.indexed_replica = true;
                        }
                        return ret;
                    }
                    if (base_ne_out_next > base_ne_in) {
                        GGML_ASSERT(src_ss[0].n_segments == 1);
                        GGML_ASSERT(src_ss[0].nr[0]      == 1);
                        ggml_backend_meta_split_state ret = {ggml_backend_meta_split_axis(dim), {0}, {1}, 1, false, {0}};
                        if (src_ss[0].indexed_replica) {
                            ret.indexed_replica = true;
                        }
                        return ret;
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
        // GDN packs scores and state snapshots into one result. Its flat
        // split is proportional to V, but only these views have a head axis.
        if (tensor->view_src && tensor->view_src->op == GGML_OP_GATED_DELTA_NET) {
            const ggml_tensor * gdn = tensor->view_src;
            const auto v_ss = ggml_backend_meta_get_split_state(stc, gdn->src[2], true);
            if (v_ss.mapped_span) {
                const int64_t width = gdn->src[2]->ne[0];
                const int64_t heads = gdn->src[2]->ne[1];
                const int64_t tokens = gdn->src[2]->ne[2];
                const int64_t sequences = gdn->src[2]->ne[3];
                const size_t scores_bytes = sizeof(float) * width * heads * tokens * sequences;
                const size_t state_plane = sizeof(float) * width * width * heads;
                auto ret = v_ss;
                if (tensor->view_offs == 0 && tensor->ne[0] == width && tensor->ne[1] == heads &&
                    tensor->ne[2] == tokens && tensor->ne[3] == sequences) {
                    ret.axis = GGML_BACKEND_SPLIT_AXIS_1;
                } else if (tensor->view_offs >= scores_bytes &&
                           (tensor->view_offs - scores_bytes) % state_plane == 0 &&
                           tensor->ne[0] == width && tensor->ne[1] == width && tensor->ne[2] == heads &&
                           tensor->view_offs + ggml_nbytes(tensor) <= ggml_nbytes(gdn)) {
                    ret.axis = GGML_BACKEND_SPLIT_AXIS_2;
                } else {
                    GGML_ABORT("unsupported mapped GDN result view '%s'", tensor->name);
                }
                return ret;
            }
        }
        if (src_ss[0].mapped_span) {
            const ggml_tensor * source = tensor->view_src ? tensor->view_src : tensor->src[0];
            const auto root = ggml_backend_meta_get_split_state(stc, source, true);
            GGML_ASSERT(root.mapped_span);
            const size_t block = root.axis == 0 ? ggml_blck_size(source->type) : 1;
            const size_t stride = source->nb[root.axis];
            const size_t plane = stride * (source->ne[root.axis] / block);
            const int axis = ggml_meta_mapped_view_axis(tensor, source, root);
            const size_t offset = tensor->view_offs % plane;
            const size_t extent = tensor->nb[axis] * tensor->ne[axis];
            // If the view covers the full extent of the mapped axis (e.g. an offset
            // along a non-split dimension inside a conv matrix), the split axis
            // spans are identical to the source; inherit root mapping without slicing.
            if (tensor->ne[axis] == source->ne[root.axis] && extent == plane && offset < stride) {
                auto ret = root;
                ret.axis = ggml_backend_meta_split_axis(axis);
                return ret;
            }
            if (offset % stride != 0 || extent % stride != 0 || offset + extent > plane) {
                fprintf(stderr, "MAPPED_VIEW_ASSERT: tensor='%s' source='%s' axis=%d root_axis=%d offset=%zu stride=%zu extent=%zu plane=%zu ne_axis=%lld nb_axis=%zu\n",
                        tensor->name, source->name, axis, root.axis, offset, stride, extent, plane, (long long)tensor->ne[axis], tensor->nb[axis]);
                GGML_ASSERT(false);
            }
            return ggml_meta_mapped_slice(root, n_bufs, offset / stride * block,
                                          (offset + extent) / stride * block, axis,
                                          stride, tensor->nb[axis] * block);
        }
        if (ggml_is_contiguous(tensor) && ggml_is_contiguous(tensor->src[0])) {
            return handle_reshape(src_ss);
        }
        if (src_ss[0].indexed_replica) {
            // Storage view over an indexed_replica tensor (e.g. cache_k row view):
            // Preserves replica layout along the same split axis if geometry matches.
            if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && tensor->ne[0] == tensor->src[0]->ne[0]) {
                return src_ss[0];
            }
            if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1 && tensor->ne[1] == tensor->src[0]->ne[1]) {
                return src_ss[0];
            }
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
                    return { ggml_backend_meta_split_axis(dim), {0}, {1}, 1, false, {0} };
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
                ggml_backend_meta_split_state ret = {ggml_backend_meta_split_axis(tensor->op_params[src_ss[0].axis]), {0}, {src_ss[0].nr[0]}, 1, false, {0}};
                if (src_ss[0].indexed_replica) {
                    ret.indexed_replica = true;
                }
                return ret;
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
                ggml_backend_meta_split_state ret = {ggml_backend_meta_split_axis(int(src_ss[0].axis) ^ 1), {0}, {src_ss[0].nr[0]}, 1, false, {0}};
                if (src_ss[0].indexed_replica) {
                    ret.indexed_replica = true;
                }
                return ret;
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
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        return handle_generic(src_ss, /*scalar_only =*/ true);
    };

    auto handle_set_rows = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        const bool indices_mirrored = (src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        if (!indices_mirrored && !split_states_equal(src_ss[0], src_ss[2])) {
            char buf[512];
            snprintf(buf, sizeof(buf), "handle_set_rows mismatch on tensor '%s': src0 '%s' (axis=%d), src1 '%s' (axis=%d), src2 '%s' (axis=%d)",
                     tensor->name, tensor->src[0]->name, (int)src_ss[0].axis, tensor->src[1]->name, (int)src_ss[1].axis, tensor->src[2]->name, (int)src_ss[2].axis);
            GGML_ABORT("%s", buf);
        }
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

    auto handle_mirrored_attention = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) {
        for (size_t i = 0; i < GGML_MAX_SRC; ++i) {
            GGML_ASSERT(tensor->src[i] == nullptr || src_ss[i].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        }
        return src_ss[0];
    };

    auto handle_flash_attn_ext =
        [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return handle_mirrored_attention(src_ss);
        }
        GGML_ASSERT(                             src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_2);
        GGML_ASSERT(                             src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_2);
        GGML_ASSERT(                             src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_2);
        GGML_ASSERT(tensor->src[4] == nullptr || src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        GGML_ASSERT(tensor->src[4] == nullptr || src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_0);
        return { GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1, false, {0} };
    };

    auto handle_flash_attn_ext_banded =
        [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return handle_mirrored_attention(src_ss);
        }
        GGML_ASSERT(                             src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_2);
        GGML_ASSERT(                             src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_2);
        GGML_ASSERT(                             src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_2);
        GGML_ASSERT(tensor->src[3] == nullptr || src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        // rel_logits is [E, H, Q, B], so its head shard is axis 1.
        GGML_ASSERT(                             src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_1);
        return { GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1, false, {0} };
    };

    auto handle_flash_attn_ext_rerot =
        [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return handle_mirrored_attention(src_ss);
        }
        // q_groups/K/V all carry the attention-head dimension on axis 2.
        // The indexed entry list and query offsets are shared control data;
        // optional sinks follow Q-head ownership on their vector axis.
        GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_2);
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_2);
        GGML_ASSERT(src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_2);
        GGML_ASSERT(src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        GGML_ASSERT(src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        GGML_ASSERT(tensor->src[5] == nullptr ||
                    src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_0 ||
                    src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        return { GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1, false, {0} };
    };

    auto handle_ssm_conv = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == src_ss[1].axis) {
            if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0) {
                ggml_backend_meta_split_state ret = src_ss[0];
                ret.axis                          = GGML_BACKEND_SPLIT_AXIS_1;
                return ret;
            }
            if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1) {
                ggml_backend_meta_split_state ret = src_ss[0];
                ret.axis                          = GGML_BACKEND_SPLIT_AXIS_0;
                return ret;
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
        // TP5 mapped GDN (opt-in): the flat output packs [S*H*T*nseq scores |
        // S*S*H*nseq states]; its whole size is proportional to the V head
        // count, so the flat ret carries per-rank ne_j from the V ratio
        // WITHOUT mapped_span — there is no honest flat span encoding at
        // prefill T>1 (score head spans repeat T times and would explode
        // past the span budget). Views over this output derive their V-head
        // span identity from the op's mapped V source instead (handle_view).
        if (src_ss[2].mapped_span) {
            const size_t n = ggml_backend_meta_buffer_n_bufs(tensor->buffer);
            ggml_backend_meta_split_state ret = { GGML_BACKEND_SPLIT_AXIS_0, {0}, {1}, 1, false, {0} };
            for (size_t j = 0; j < n; ++j) {
                GGML_ASSERT(tensor->ne[0] % tensor->src[2]->ne[1] == 0);
                ret.ne[j] = (tensor->ne[0] / tensor->src[2]->ne[1]) * src_ss[2].ne[j];
            }
            return ret;
        }
        return { GGML_BACKEND_SPLIT_AXIS_0, {0}, {1}, 1, false, {0} };
    };

    auto calculate_split_state = [&]() -> ggml_backend_meta_split_state {
        if (ggml_nelements(tensor) == 0) {
            return { GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1, false, {0} };
        }
        if (ggml_backend_buffer_get_usage(tensor->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE && tensor->view_src == nullptr) {
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(tensor->buffer));
            const ggml_backend_meta_device_context * dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
            ggml_backend_meta_split_state ret = dev_ctx->get_split_state(tensor, dev_ctx->get_split_state_ud);
            if (ret.mapped_span) {
                // Bounded indexed spans: validate count/bounds/coverage before
                // anything downstream writes or allocates from it.
                GGML_ASSERT(ggml_meta_mapped_spans_valid(tensor, ret, n_bufs));
            }
            if (ret.axis >= 0 && ret.axis <= GGML_MAX_DIMS) {
                const int64_t granularity = ret.axis == GGML_BACKEND_SPLIT_AXIS_0 ? ggml_blck_size(tensor->type) : 1;
                int64_t ne_sum = 0;
                for (size_t s = 0; s < ret.n_segments; s++) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        GGML_ASSERT(ret.ne[s*n_bufs + j] % granularity == 0);
                        ne_sum += ret.ne[s*n_bufs + j] * ret.nr[s];
                    }
                }
                if (ret.mapped_span) {
                    // mapped spans allow cross-device overlap (replicas);
                    // ggml_meta_mapped_spans_valid already checked per-device
                    // coverage and bounds.
                } else if (ret.indexed_replica) {
                    GGML_ASSERT(ret.n_segments == 1 && ret.nr[0] == 1);
                    for (size_t j = 0; j < n_bufs; j++) {
                        if (ret.ne[j] > 0) {
                            GGML_ASSERT(ret.replica_start[j] >= 0);
                            GGML_ASSERT(ret.replica_start[j] % granularity == 0);
                            GGML_ASSERT(ret.replica_start[j] + ret.ne[j] <= tensor->ne[ret.axis]);
                        }
                    }
                } else {
                    GGML_ASSERT(ne_sum == tensor->ne[ret.axis]);
                }
            }
            return ret;
        }

        std::vector<ggml_backend_meta_split_state> src_ss(GGML_MAX_SRC, {GGML_BACKEND_SPLIT_AXIS_NONE, {0}, {1}, 1, false, {0}});
        for (size_t i = 0; i < GGML_MAX_SRC; i++) {
            if (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                src_ss[i] = { GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1, false, {0} };
                continue;
            }
            src_ss[i] = ggml_backend_meta_get_split_state(stc, tensor->src[i], /*assume_sync =*/ true);
            GGML_ASSERT(src_ss[i].axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
        }

        ggml_backend_meta_split_state split_state;
        switch (tensor->op) {
            case GGML_OP_NONE: {
                split_state = { GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1, false, {0} };
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
            case GGML_OP_REROT_SPAN_EXPAND:
            case GGML_OP_REROT_Q_PREP: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
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
            case GGML_OP_DIAG: {
                if (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
                    split_state = src_ss[0];
                } else {
                    split_state = handle_generic(src_ss, /*scalar_only =*/ false);
                }
            } break;
            case GGML_OP_DIAG_MASK_INF:
            case GGML_OP_DIAG_MASK_ZERO: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_SOFT_MAX:
            case GGML_OP_SOFT_MAX_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_ROPE: {
                split_state = handle_rope(src_ss);
            } break;
            case GGML_OP_ROPE_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
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
                // TRI operates on the 2D matrix plane (axes 0 and 1).
                // For multi-head / batch axes (axis >= 2), the operation is independent per head!
                if (src_ss[0].axis >= GGML_BACKEND_SPLIT_AXIS_2 && src_ss[0].axis < GGML_MAX_DIMS) {
                    split_state = src_ss[0];
                } else {
                    split_state = handle_generic(src_ss, /*scalar_only =*/ true);
                }
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
            case GGML_OP_FLASH_ATTN_EXT_REROT: {
                split_state = handle_flash_attn_ext_rerot(src_ss);
            } break;
            case GGML_OP_XKV_RECONSTRUCT: {
            case GGML_OP_XKV_ATTENTION:
            case GGML_OP_XKV_FACTORIZE:
            case GGML_OP_XKV_CANONICALIZE:
            case GGML_OP_XKV_LANDMARK:
            case GGML_OP_XKV_LANDMARK_BUILD:
            case GGML_OP_XKV_LANDMARK_ROWS:
            case GGML_OP_XKV_LANDMARK_MERGE:
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
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
            case GGML_OP_DSV4_HC_COMB:
            case GGML_OP_DSV4_HC_PRE:
            case GGML_OP_DSV4_HC_POST: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_FLASH_PREFILL_POOL:
            case GGML_OP_FLASH_PREFILL_SELECT:
            case GGML_OP_FLASH_PREFILL_ATTN: {
                // Pool/select/attn each run as an actual graph op with a backend
                // barrier; never split across devices (counts live in op_params
                // and explicit host capacities, invalid metadata is an execution
                // error, not a silent dense fallback).
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
            case GGML_OP_TURBO_WHT: {
                // Orthogonal Walsh-Hadamard transform preserves tensor shape & slice partition
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_OPT_STEP_ADAMW:
            case GGML_OP_OPT_STEP_SGD:
            case GGML_OP_GLU: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            default: {
                GGML_ABORT("ggml op not implemented: %s", ggml_op_name(tensor->op));
                split_state = { GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1, false, {0} };
            } break;
        }
        if (split_state.mapped_span) {
            GGML_ASSERT(ggml_meta_mapped_spans_valid(tensor, split_state, n_bufs));
            return split_state;
        }
        if (tensor->op == GGML_OP_GATED_DELTA_NET && src_ss[2].mapped_span) {
            return split_state;
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
                            if (split_state.ne[j] % div != 0) {
                                // Local rank-local slice view (e.g. Q0/Q1 in TP5 bridge rank GQA):
                                // The view operates on local heads, so global ratio cannot cleanly divide.
                                split_state.ne[j] = std::min(split_state.ne[j] / div, tensor->ne[split_state.axis]);
                                continue;
                            }
                            split_state.ne[j] /= div;
                        }
                    }
                    if (src_ss[i].indexed_replica) {
                        split_state.indexed_replica = true;
                        for (size_t j = 0; j < n_bufs; j++) {
                            const int64_t div = tensor->src[i]->ne[src_ss[i].axis] * split_state.nr[0];
                            if (div != 0) {
                                split_state.replica_start[j] = (src_ss[i].replica_start[j] * tensor->ne[split_state.axis]) / div;
                            } else {
                                split_state.replica_start[j] = 0;
                            }
                        }
                    }
                } else {
                    GGML_ASSERT(split_state.n_segments == 1);
                    for (size_t j = 0; j < n_bufs; j++) {
                        int64_t sum = 0;
                        for (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            sum += src_ss[i].ne[s*n_bufs + j] * src_ss[i].nr[s];
                        }
                        // Allow GQA Flash Attention where KV heads are replicated across devices
                        // (e.g. Qwen4EXP with 2 KV heads over 5 devices, where ranks hold KV replicas
                        // following the TP5 role table instead of strict equal-ratio partitioning).
                        const bool is_gqa_flash_attn_kv = (tensor->op == GGML_OP_FLASH_ATTN_EXT ||
                                                           tensor->op == GGML_OP_FLASH_ATTN_EXT_BANDED) && (i == 1 || i == 2);
                        // Allow uneven recurrent conv_input concat along tokens (e.g. Qwen4EXP TP5
                        // where conv state slices follow V head counts [10,10,10,10,8] while sequence length is 1).
                        const bool is_uneven_concat = (tensor->op == GGML_OP_CONCAT &&
                                                       ggml_get_op_params_i32(tensor, 0) != split_state.axis);
                        const bool is_ssm_conv = (tensor->op == GGML_OP_SSM_CONV);
                        const bool is_gdn = (tensor->op == GGML_OP_GATED_DELTA_NET);
                        const bool is_norm_gated_mul = (tensor->op == GGML_OP_MUL &&
                                                        tensor->src[0] && tensor->src[1] &&
                                                        tensor->src[0]->ne[0] == tensor->src[1]->ne[0] &&
                                                        tensor->src[0]->ne[1] == tensor->src[1]->ne[1]);
                        if (is_uneven_concat) {
                            continue;
                        }
                        if (is_ssm_conv) {
                            continue;
                        }
                        if (is_gdn) {
                            continue;
                        }
                        if (is_norm_gated_mul) {
                            continue;
                        }
                        if (is_gqa_flash_attn_kv) {
                            continue;
                        }
                        int64_t lhs = split_state.ne[j] * split_state.nr[0] * tensor->src[i]->ne[src_ss[i].axis];
                        int64_t rhs = sum * tensor->ne[split_state.axis];
                        // Under uneven tensor parallelism (e.g. TP5 [10,10,10,10,8]), element ratios naturally vary across ranks.
                        if (lhs != rhs && n_bufs != 5) {
                            char buf[512];
                            snprintf(buf, sizeof(buf), "ratio mismatch on tensor '%s' (%s) src[%zu] '%s': j=%zu lhs=%lld rhs=%lld (ne_j=%lld sum=%lld)",
                                     tensor->name, ggml_op_name(tensor->op), i, tensor->src[i]->name, j, (long long)lhs, (long long)rhs, (long long)split_state.ne[j], (long long)sum);
                            GGML_ABORT("%s", buf);
                        }
                    }
                }
                first_src_split_by_axis = false;
            }
            GGML_ASSERT(!first_src_split_by_axis);
        }
        return split_state;
    };

    const std::pair key = std::make_pair(tensor, assume_sync);
    auto it = buf_ctx->split_state_cache.find(key);
    if (it != buf_ctx->split_state_cache.end() && memcmp(it->second.second, (const char *) tensor, sizeof(it->second.second)) != 0) {
        buf_ctx->split_state_cache.clear();
        it = buf_ctx->split_state_cache.end();
    }

    if (it == buf_ctx->split_state_cache.end()) {
        buf_ctx->split_state_cache[key].first = calculate_split_state();
        memcpy(buf_ctx->split_state_cache[key].second, tensor, sizeof(buf_ctx->split_state_cache[key].second));
        if (buf_ctx->debug > 0) {
            std::string srcs_info;
            for (size_t i = 0; i < GGML_MAX_SRC; i++) {
                if (tensor->src[i] == nullptr) {
                    continue;
                }
                if (!srcs_info.empty()) {
                    srcs_info += ", ";
                }
                srcs_info += std::string(tensor->src[i]->name) + "[" + ggml_op_name(tensor->src[i]->op) +
                             ", ne=[" + std::to_string(tensor->src[i]->ne[0]) + "x" + std::to_string(tensor->src[i]->ne[1]) + "]]";
            }
            std::string ne_info;
            for (size_t j = 0; j < n_bufs; j++) {
                if (!ne_info.empty()) {
                    ne_info += ", ";
                }
                const ggml_backend_meta_split_state & ss = buf_ctx->split_state_cache[key].first;
                ne_info += std::to_string(ss.ne[j]) + "x" + std::to_string(ss.nr[0]);
            }
            GGML_LOG_DEBUG("SPLIT_STATE: {%s} -> %s[%s, %s, {%s}]\n", srcs_info.c_str(), tensor->name, ggml_op_name(tensor->op),
                ggml_backend_meta_split_axis_name(buf_ctx->split_state_cache[key].first.axis), ne_info.c_str());
        }
    }

    ggml_backend_meta_split_state ret = buf_ctx->split_state_cache[key].first;
    GGML_ASSERT(ret.axis != GGML_BACKEND_SPLIT_AXIS_NONE);
#ifndef NDEBUG
    if (ret.axis >= 0 && ret.axis < GGML_MAX_DIMS && !ret.indexed_replica && !ret.mapped_span) {
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
    ggml_backend_meta_local_node_hook_t dev_local_node_hook = nullptr;
    void * dev_local_node_hook_userdata = nullptr;
    if (tensor->op == GGML_OP_FLASH_ATTN_EXT || tensor->op == GGML_OP_FLASH_ATTN_EXT_REROT ||
        tensor->op == GGML_OP_GATED_DELTA_NET) {
        const auto dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(tensor->buffer));
        const auto * dev_ctx = static_cast<const ggml_backend_meta_device_context *>(dev->context);
        dev_local_node_hook = dev_ctx->local_node_hook;
        dev_local_node_hook_userdata = dev_ctx->get_split_state_ud;
    }

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
        // Multi-buffer support: the wrapper has no get_base and its context is not a per-rank
        // backend buffer, so it must never become the simple tensor's buffer. When the rank's
        // store is fragmented into several VkBuffers, ggml_backend_alloc_ctx_tensors_from_buft
        // already bound each simple tensor to its real sub-buffer via ggml_backend_tensor_alloc;
        // leaving the field untouched here preserves that mapping.
        if (simple_buf != nullptr && !ggml_backend_buffer_is_multi_buffer(simple_buf)) {
            t_ij->buffer = simple_buf;
        }
        t_ij->view_src = tensor->view_src;
        t_ij->view_offs = tensor->view_offs;
        if (t_ij->view_src != nullptr && ggml_backend_buffer_is_meta(t_ij->view_src->buffer)) {
            t_ij->view_src = ggml_backend_meta_buffer_simple_tensor(tensor->view_src, j);
            const auto source_split = ggml_backend_meta_get_split_state(tensor->view_src, true);
            if (source_split.mapped_span) {
                const ggml_tensor * source = tensor->view_src;
                const int axis = source_split.axis;
                const size_t block = axis == 0 ? ggml_blck_size(source->type) : 1;
                const size_t stride = source->nb[axis];
                const size_t plane = stride * (source->ne[axis] / block);
                const size_t local_plane = t_ij->view_src->nb[axis] * (t_ij->view_src->ne[axis] / block);
                const int view_split_dim = (split_dim >= 0 && split_dim < GGML_MAX_DIMS) ? split_dim :
                                           ggml_meta_mapped_view_axis(tensor, source, source_split);
                GGML_ASSERT(view_split_dim >= 0 && view_split_dim < GGML_MAX_DIMS);
                const size_t inner = tensor->view_offs % plane;
                const size_t extent = tensor->nb[view_split_dim] * tensor->ne[view_split_dim];
                if (tensor->ne[view_split_dim] == source->ne[axis] && extent == plane && inner < stride) {
                    t_ij->view_offs = (tensor->view_offs / plane) * local_plane + (tensor->view_offs % stride);
                    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
                        t_ij->nb[d] = tensor->nb[d];
                        if (tensor->nb[d] >= plane) {
                            GGML_ASSERT(tensor->nb[d] % plane == 0);
                            t_ij->nb[d] = (tensor->nb[d] / plane) * local_plane;
                        }
                    }
                } else {
                    GGML_ASSERT(inner % stride == 0 && extent % stride == 0 && inner + extent <= plane);
                    const int64_t begin = inner / stride * block;
                    const int64_t end = (inner + extent) / stride * block;
                    int64_t base = 0, local_start = -1;
                    for (int32_t s = 0; s < source_split.span_count[j]; ++s) {
                        const auto index = ggml_meta_mapped_span_index(source_split, j, s);
                        const int64_t lo = std::max(begin, source_split.span_start[index]);
                        const int64_t hi = std::min(end, source_split.span_start[index] + source_split.span_len[index]);
                        if (lo < hi) {
                            local_start = base + lo - source_split.span_start[index];
                            break;
                        }
                        base += source_split.span_len[index];
                    }
                    if (local_start < 0) {
                        t_ij->ne[view_split_dim] = 0;
                        t_ij->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
                        t_ij->view_offs = 0;
                    } else {
                        t_ij->view_offs = (tensor->view_offs / plane) * local_plane +
                                          (local_start / block) * stride;
                    }
                    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
                        t_ij->nb[d] = tensor->nb[d];
                        if (tensor->nb[d] >= plane) {
                            GGML_ASSERT(tensor->nb[d] % plane == 0);
                            t_ij->nb[d] = (tensor->nb[d] / plane) * local_plane;
                        }
                    }
                }
            } else if (t_ij->view_offs > 0 && split_dim >= 0 && split_dim < GGML_MAX_DIMS) {
                GGML_ASSERT(tensor->ne[split_dim] != 0);
                const int split_dim_view_src = source_split.axis;
                if (split_dim_view_src >= 0 && split_dim_view_src < GGML_MAX_DIMS) {
                    bool split_internal_offset = t_ij->view_offs <= tensor->view_src->nb[split_dim_view_src];
                    for (int i = 0; i < GGML_MAX_DIMS; i++) {
                        const size_t dim_size = tensor->ne[i] * tensor->nb[i];
                        if (tensor->view_offs <= dim_size && dim_size < tensor->nb[split_dim]) {
                            split_internal_offset = true;
                            break;
                        }
                    }
                    if (!split_internal_offset) {
                        t_ij->view_offs = (t_ij->view_offs * t_ij->view_src->ne[split_dim_view_src]) /
                                          tensor->view_src->ne[split_dim_view_src];
                    }
                }
            }
        }
        if (t_ij->view_src != nullptr) {
            t_ij->data = (char *) t_ij->view_src->data + t_ij->view_offs;
            // Mirror ggml_backend_view_init: a view must carry its source's buffer. With a
            // fragmented rank store the source's simple tensor is bound to the real sub-buffer
            // (never the multi-buffer wrapper), so the Vulkan backend resolves the right one.
            if (t_ij->buffer == nullptr) {
                t_ij->buffer = t_ij->view_src->buffer;
            }
        } else if (simple_buf != nullptr) {
            // Multi-buffer or single-buffer pointer mapping:
            // For multi_buffer get_base is null, but tallocr sets real data during alloc_ctx_tensors_from_buft
            if (!ggml_backend_buffer_is_multi_buffer(simple_buf)) {
                t_ij->data = (char *) ggml_backend_buffer_get_base(simple_buf)
                    + size_t(tensor->data) - size_t(ggml_backend_buffer_get_base(tensor->buffer));
            }
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

        // TP5 local-node head map (opt-in): per-device hook stamps the private
        // op_params head-map ABI (ggml_tp5_headmap_set) on the rank-local clone
        // only; the global graph node stays untouched. Runs AFTER the rank-local
        // src remap so the hook can inspect local_node->src[0]/src[1] shapes.
        // Dispatches on node->op (ordinary/indexed FA: Q->KV map; GDN: V->QK map).
        if (dev_local_node_hook != nullptr &&
                (tensor->op == GGML_OP_FLASH_ATTN_EXT || tensor->op == GGML_OP_FLASH_ATTN_EXT_REROT ||
                 tensor->op == GGML_OP_GATED_DELTA_NET)) {
            dev_local_node_hook(t_ij, j, dev_local_node_hook_userdata);
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

struct ggml_meta_canonical_span {
    size_t  rank;
    int64_t logical_start;
    int64_t length;
    int64_t local_start;
};


// Computes canonical lowest-rank spans covering [0, tensor->ne[split_state.axis]) for indexed_replica tensors.
// Validates that every declared replica range is within tensor bounds and aligned to blck_size,
// and validates that the full tensor range is covered without gaps before returning.
static size_t ggml_meta_get_canonical_spans(
        const ggml_tensor * tensor,
        const ggml_backend_meta_split_state & split_state,
        size_t n_devices,
        ggml_meta_canonical_span * out_spans,
        size_t max_spans) {
    GGML_ASSERT(split_state.indexed_replica);
    GGML_ASSERT(split_state.n_segments == 1 && split_state.nr[0] == 1);
    GGML_ASSERT(split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS);
    GGML_ASSERT(n_devices <= GGML_BACKEND_META_MAX_DEVICES);

    const int64_t ne_axis   = tensor->ne[split_state.axis];
    const int64_t blck_size = split_state.axis == GGML_BACKEND_SPLIT_AXIS_0 ? ggml_blck_size(tensor->type) : 1;

    // Collect and validate declared replica boundaries
    int64_t endpoints[2 * GGML_BACKEND_META_MAX_DEVICES + 2];
    size_t n_endpoints = 0;
    endpoints[n_endpoints++] = 0;
    endpoints[n_endpoints++] = ne_axis;

    for (size_t j = 0; j < n_devices; j++) {
        const int64_t len = split_state.ne[j];
        GGML_ASSERT(len >= 0);
        if (len == 0) continue;
        const int64_t start = split_state.replica_start[j];
        GGML_ASSERT(start >= 0);
        GGML_ASSERT(start % blck_size == 0);
        GGML_ASSERT(len % blck_size == 0);
        GGML_ASSERT(start <= ne_axis && len <= ne_axis - start);
        endpoints[n_endpoints++] = start;
        endpoints[n_endpoints++] = start + len;
    }

    // Sort and deduplicate endpoints
    std::sort(endpoints, endpoints + n_endpoints);
    size_t n_unique = 0;
    for (size_t i = 0; i < n_endpoints; i++) {
        if (n_unique == 0 || endpoints[i] != endpoints[n_unique - 1]) {
            endpoints[n_unique++] = endpoints[i];
        }
    }

    // For each adjacent interval [p0, p1), determine the lowest-rank owner
    size_t n_spans = 0;
    for (size_t i = 0; i + 1 < n_unique; i++) {
        const int64_t p0 = endpoints[i];
        const int64_t p1 = endpoints[i + 1];
        if (p0 >= p1) continue;

        int owner_rank = -1;
        for (size_t j = 0; j < n_devices; j++) {
            if (split_state.ne[j] == 0) continue;
            const int64_t r_start = split_state.replica_start[j];
            const int64_t r_end   = r_start + split_state.ne[j];
            if (r_start <= p0 && p1 <= r_end) {
                owner_rank = (int)j;
                break; // First matching rank in index order = lowest rank
            }
        }
        // Coverage check: each interval must have at least one owner
        GGML_ASSERT(owner_rank >= 0);

        const size_t rank = (size_t)owner_rank;
        const int64_t len = p1 - p0;
        const int64_t local_start = p0 - split_state.replica_start[rank];

        // Merge with previous span if same rank and contiguous
        if (n_spans > 0 && out_spans[n_spans - 1].rank == rank &&
            out_spans[n_spans - 1].logical_start + out_spans[n_spans - 1].length == p0 &&
            out_spans[n_spans - 1].local_start + out_spans[n_spans - 1].length == local_start) {
            out_spans[n_spans - 1].length += len;
        } else {
            GGML_ASSERT(n_spans < max_spans);
            out_spans[n_spans++] = { rank, p0, len, local_start };
        }
    }

    // Final verification: total length of spans must exactly equal ne_axis
    int64_t covered_len = 0;
    for (size_t s = 0; s < n_spans; s++) {
        covered_len += out_spans[s].length;
    }
    GGML_ASSERT(covered_len == ne_axis);

    return n_spans;
}

static void ggml_backend_meta_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(buffer);
    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(ggml_is_contiguous(tensor) || split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

    if (split_state.mapped_span) {
        // TP5 mapped spans: original logical ranges -> rank-local concatenation.
        // Writes fan out to every device owning the covered logical range
        // (replica semantics for shared Q/K head pairs).
        GGML_ASSERT(ggml_meta_mapped_spans_valid(tensor, split_state, n_bufs));
        GGML_ASSERT(ggml_is_contiguous(tensor));
        const size_t chunk_size_full = ggml_meta_mapped_plane_bytes(tensor, split_state.axis);
        GGML_ASSERT(offset % chunk_size_full == 0);
        GGML_ASSERT(size   % chunk_size_full == 0);
        const int64_t i_start =  offset        / chunk_size_full;
        const int64_t i_stop  = (offset + size)/ chunk_size_full;
        const int64_t blck_size = split_state.axis == GGML_BACKEND_SPLIT_AXIS_0 ? ggml_blck_size(tensor->type) : 1;
        const size_t element_bytes = tensor->nb[split_state.axis];
        for (size_t j = 0; j < n_bufs; ++j) {
            ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
            const size_t chunk_size_j = ggml_meta_mapped_plane_bytes(simple_tensor, split_state.axis);
            if (chunk_size_j == 0) continue;
            for (int32_t s = 0; s < split_state.span_count[j]; ++s) {
                const int64_t idx = ggml_meta_mapped_span_index(split_state, j, s);
                const int64_t start = split_state.span_start[idx];
                const int64_t slen  = split_state.span_len[idx];
                GGML_ASSERT(start % blck_size == 0 && slen % blck_size == 0);
                int64_t base = 0;
                for (int32_t t = 0; t < s; ++t) {
                    base += split_state.span_len[ggml_meta_mapped_span_index(split_state, j, t)];
                }
                const size_t src_off = (size_t)(start / blck_size) * element_bytes;
                const size_t nbytes   = (size_t)(slen / blck_size) * element_bytes;
                ggml_backend_tensor_set_2d(simple_tensor, (const char *) data + src_off,
                                           i_start * chunk_size_j + (size_t)(base / blck_size) * element_bytes,
                                           nbytes, i_stop - i_start,
                                           chunk_size_j, chunk_size_full);
            }
        }
        return;
    }

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

            if (split_state.indexed_replica) {
                const int64_t blck_size = split_state.axis == GGML_BACKEND_SPLIT_AXIS_0 ? ggml_blck_size(tensor->type) : 1;
                const size_t element_bytes = tensor->nb[split_state.axis];
                for (size_t j = 0; j < n_bufs; j++) {
                    ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                    if (chunk_size_j == 0) continue;
                    GGML_ASSERT(split_state.replica_start[j] % blck_size == 0);
                    const size_t src_offset_in_chunk = (size_t)(split_state.replica_start[j] / blck_size) * element_bytes;
                    GGML_ASSERT(src_offset_in_chunk + chunk_size_j <= chunk_size_full);
                    const size_t simple_offset = i_start * chunk_size_j;
                    ggml_backend_tensor_set_2d(simple_tensor, (const char *) data + src_offset_in_chunk,
                                               simple_offset, chunk_size_j, i_stop - i_start,
                                               chunk_size_j, chunk_size_full);
                }
                return;
            }

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
            if (offset_j != chunk_size_full) {
                char err_buf[256];
                snprintf(err_buf, sizeof(err_buf),
                    "ggml_backend_meta_buffer_set_tensor: tensor '%s' axis %d chunk offset mismatch: offset_j=%zu != chunk_size_full=%zu",
                    tensor->name, (int)split_state.axis, offset_j, chunk_size_full);
                GGML_ABORT("%s", err_buf);
            }
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

    if (split_state.mapped_span) {
        // Canonical inverse readback: enumerate boundary intervals of all
        // spans and read each logical interval from its deterministic lowest
        // device owner (handles partial overlaps; never two concurrent host
        // writes of the same bytes).
        GGML_ASSERT(ggml_meta_mapped_spans_valid(tensor, split_state, n_bufs));
        GGML_ASSERT(ggml_is_contiguous(tensor));
        const size_t chunk_size_full = ggml_meta_mapped_plane_bytes(tensor, split_state.axis);
        GGML_ASSERT(offset % chunk_size_full == 0);
        GGML_ASSERT(size   % chunk_size_full == 0);
        const int64_t i_start =  offset        / chunk_size_full;
        const int64_t i_stop  = (offset + size)/ chunk_size_full;
        const int64_t blck_size = split_state.axis == GGML_BACKEND_SPLIT_AXIS_0 ? ggml_blck_size(tensor->type) : 1;
        const size_t element_bytes = tensor->nb[split_state.axis];
        int64_t iv_span[2 * GGML_BACKEND_META_MAX_SPANS + 2][2];
        int iv_owner[2 * GGML_BACKEND_META_MAX_SPANS + 2];
        int64_t iv_local[2 * GGML_BACKEND_META_MAX_SPANS + 2];
        const size_t n_iv = ggml_meta_mapped_canonical_intervals(
                tensor, split_state, n_bufs, iv_span, iv_owner, iv_local,
                sizeof(iv_span) / sizeof(iv_span[0]));
        for (size_t s = 0; s < n_iv; ++s) {
            const size_t rank = (size_t) iv_owner[s];
            const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, rank);
            const size_t chunk_size_j = ggml_meta_mapped_plane_bytes(simple_tensor, split_state.axis);
            GGML_ASSERT(iv_span[s][0] % blck_size == 0 && iv_span[s][1] % blck_size == 0);
            const size_t dst_off = (size_t)(iv_span[s][0] / blck_size) * element_bytes;
            const size_t nbytes  = (size_t)(iv_span[s][1] / blck_size) * element_bytes;
ggml_backend_tensor_get_2d(simple_tensor, (char *) data + dst_off,
                                       i_start * chunk_size_j + (size_t)(iv_local[s] / blck_size) * element_bytes,
                                       nbytes, i_stop - i_start, chunk_size_j, chunk_size_full);
        }
        return;
    }

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

            if (split_state.indexed_replica) {
                const int64_t blck_size = split_state.axis == GGML_BACKEND_SPLIT_AXIS_0 ? ggml_blck_size(tensor->type) : 1;
                const size_t element_bytes = tensor->nb[split_state.axis];
                ggml_meta_canonical_span spans[2 * GGML_BACKEND_META_MAX_DEVICES + 2];
                const size_t n_spans = ggml_meta_get_canonical_spans(tensor, split_state, n_bufs, spans, sizeof(spans)/sizeof(spans[0]));
                for (size_t s = 0; s < n_spans; s++) {
                    const size_t rank = spans[s].rank;
                    const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, rank);
                    const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                    GGML_ASSERT(spans[s].logical_start % blck_size == 0);
                    GGML_ASSERT(spans[s].local_start % blck_size == 0);
                    GGML_ASSERT(spans[s].length % blck_size == 0);
                    const size_t dst_offset_in_chunk = (size_t)(spans[s].logical_start / blck_size) * element_bytes;
                    const size_t local_offset_in_chunk = (size_t)(spans[s].local_start / blck_size) * element_bytes;
                    const size_t copy_bytes = (size_t)(spans[s].length / blck_size) * element_bytes;
                    ggml_backend_tensor_get_2d(simple_tensor, (char *) data + dst_offset_in_chunk,
                        i_start * chunk_size_j + local_offset_in_chunk, copy_bytes,
                        i_stop - i_start, chunk_size_j, chunk_size_full);
                }
                return;
            }

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

static bool ggml_backend_meta_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    if (!ggml_backend_buffer_is_meta(src->buffer) || !ggml_backend_buffer_is_meta(dst->buffer)) {
        return false;
    }
    const size_t n_bufs_src = ggml_backend_meta_buffer_n_bufs(src->buffer);
    const size_t n_bufs_dst = ggml_backend_meta_buffer_n_bufs(dst->buffer);
    if (n_bufs_src != n_bufs_dst) {
        return false;
    }
    // Pre-flight check: ensure all shard layouts are compatible and copyable BEFORE copying any shard
    for (size_t j = 0; j < n_bufs_dst; ++j) {
        const struct ggml_tensor * simple_src = ggml_backend_meta_buffer_simple_tensor(src, j);
        struct ggml_tensor       * simple_dst = ggml_backend_meta_buffer_simple_tensor(dst, j);
        if (simple_src == nullptr || simple_dst == nullptr) {
            return false;
        }
        if (!ggml_are_same_layout(simple_src, simple_dst)) {
            return false;
        }
    }
    // Execute same-rank device copies
    for (size_t j = 0; j < n_bufs_dst; ++j) {
        const struct ggml_tensor * simple_src = ggml_backend_meta_buffer_simple_tensor(src, j);
        struct ggml_tensor       * simple_dst = ggml_backend_meta_buffer_simple_tensor(dst, j);
        if (!ggml_backend_buffer_copy_tensor(simple_src, simple_dst)) {
            ggml_backend_tensor_copy(simple_src, simple_dst);
        }
    }
    return true;
}

static const ggml_backend_buffer_i ggml_backend_meta_buffer_iface = {
    /* .free_buffer     = */ ggml_backend_meta_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_meta_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_meta_buffer_init_tensor,
    /* .memset_tensor   = */ nullptr, // TODO implement
    /* .set_tensor      = */ ggml_backend_meta_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_meta_buffer_get_tensor,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ ggml_backend_meta_buffer_cpy_tensor,
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
    std::vector<std::vector<std::vector<void *>>> chain_compute_cbs;
    std::vector<std::vector<ggml_tensor *>>       chain_tensors;
    // The definition belongs to this backend/session entry. A function-static
    // slot lets target and MTP evict (or accidentally reuse) one another. This
    // is a single owner-local definition, not a map of token-shape graphs.
    bool                        predefined_valid = false;
    size_t                      predefined_n_subgraphs = 0;
    uint64_t                    predefined_uid = 0;
    size_t                      n_reduce_steps;
    int                                           n_nodes       = 0;
    size_t                      max_tmp_size  = 0;
    size_t                      n_subgraphs   = 0;
    uint64_t                    uid           = 0;
    int                         debug         = 0;
    uint32_t                    predefined_active_rows   = 0;
    uint32_t                    predefined_capacity_rows = 0;
    uint32_t                    predefined_capacity_outputs = 0;
    ggml_predefined_frame       predefined_frame{};
    bool                        predefined_frame_valid = false;

    void *                               comm_ctx       = nullptr;
    ggml_backend_comm_allreduce_tensor_t comm_allreduce = nullptr;
    ggml_backend_comm_prepare_graph_t    comm_prepare   = nullptr;

    bool release_comm() {
        if (comm_ctx == nullptr) {
            return true;
        }
        if (backend_configs.empty()) {
            GGML_LOG_ERROR("%s: communicator has no child backends\n", __func__);
            return false;
        }
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(
            ggml_backend_get_device(backend_configs[0].backend));
        ggml_backend_comm_free_safe_t comm_free_safe = (ggml_backend_comm_free_safe_t)
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_free_safe");
        if (comm_free_safe != nullptr) {
            if (!comm_free_safe(comm_ctx)) {
                GGML_LOG_ERROR("%s: native communicator did not drain; retaining meta and child backends\n", __func__);
                return false;
            }
        } else {
            ggml_backend_comm_free_t comm_free = (ggml_backend_comm_free_t)
                ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_free");
            GGML_ASSERT(comm_free != nullptr);
            comm_free(comm_ctx);
        }
        comm_ctx = nullptr;
        return true;
    }

    void release_backends() {
        std::vector<ggml_backend_t> child_backends;
        child_backends.reserve(backend_configs.size());
        for (auto & bc : backend_configs) {
            // These buffers are allocated from bc.backend and must be freed
            // before its Vulkan context/device is destroyed.
            bc.bufs.clear();
            child_backends.push_back(bc.backend);
        }
        backend_configs.clear();
        for (ggml_backend_t child : child_backends) {
            ggml_backend_free(child);
        }
    }

    void release_graph_state() {
        predefined_valid = false;
        predefined_n_subgraphs = 0;
        predefined_uid = 0;
        chain_tensors.clear();
        chain_compute_cbs.clear();
        nodes_aux.clear();
        cgraphs_aux.clear();
        ctx.reset();
    }

    bool shutdown() {
        if (!release_comm()) {
            return false;
        }
        release_graph_state();
        release_backends();
        return true;
    }

    ggml_backend_meta_context(ggml_backend_dev_t meta_dev, const char * params) {
        const size_t n_devs = ggml_backend_meta_dev_n_devs(meta_dev);
        n_reduce_steps = std::ceil(std::log2(n_devs));
        name = "Meta(";
        std::vector<ggml_backend_t> simple_backends;
        backend_configs.reserve(n_devs);
        simple_backends.reserve(n_devs);
        for (size_t i = 0; i < n_devs; i++) {
            ggml_backend_dev_t simple_dev = ggml_backend_meta_dev_simple_dev(meta_dev, i);
            if (i > 0) {
                name += ",";
            }
            name += ggml_backend_dev_name(simple_dev);
            ggml_backend_t simple_backend = ggml_backend_dev_init(simple_dev, params);
            if (simple_backend == nullptr) {
                GGML_LOG_ERROR("%s: failed to initialize simple backend for device '%s'\n", __func__, ggml_backend_dev_name(simple_dev));
                release_backends();
                return;
            }
            simple_backends.push_back(simple_backend);
            backend_configs.emplace_back(simple_backends.back(), n_reduce_steps);
        }
        name += ")";

        const char * GGML_META_DEBUG = getenv("GGML_META_DEBUG");
        debug = GGML_META_DEBUG ? atoi(GGML_META_DEBUG) : 0;

        if (n_devs > 1) {
            ggml_backend_comm_init_t comm_init = (ggml_backend_comm_init_t) ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_init");
            if (comm_init != nullptr) {
                // A backend advertising comm_init promises a complete native communicator
                // (comm_init + comm_allreduce_tensor + comm_free). Verify the trio up front so a
                // broken advertisement fails here without creating a comm_ctx that could not be freed.
                comm_allreduce = (ggml_backend_comm_allreduce_tensor_t)
                    ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(
                        ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_allreduce_tensor");
                comm_prepare = (ggml_backend_comm_prepare_graph_t) ggml_backend_reg_get_proc_address(
                    ggml_backend_dev_backend_reg(ggml_backend_get_device(simple_backends[0])),
                    "ggml_backend_comm_prepare_graph");
                ggml_backend_comm_free_t comm_free = (ggml_backend_comm_free_t) ggml_backend_reg_get_proc_address(
                    ggml_backend_dev_backend_reg(ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_free");
                if (comm_allreduce == nullptr || comm_free == nullptr) {
                    GGML_LOG_ERROR("%s: backend advertised comm_init but lacks %s%s%s\n", __func__,
                        comm_allreduce == nullptr ? "comm_allreduce_tensor" : "",
                        (comm_allreduce == nullptr && comm_free == nullptr) ? " and " : "",
                        comm_free == nullptr ? "comm_free" : "");
                    release_backends();
                    return;
                }
                comm_ctx = comm_init(simple_backends.data(), simple_backends.size());
                if (comm_ctx == nullptr) {
                    // Native collective initialization rejected (e.g. peer mesh unavailable):
                    // fail backend init cleanly so graph_compute never silently falls back to host relay.
                    GGML_LOG_ERROR("%s: native communicator initialization failed\n", __func__);
                    release_backends();
                    return;
                }
            }
        }
    }

    ~ggml_backend_meta_context() {
        GGML_ASSERT(comm_ctx == nullptr);
        GGML_ASSERT(backend_configs.empty());
    }
};

bool ggml_backend_meta_set_predefined_rows(ggml_backend_t backend, uint32_t active_rows, uint32_t capacity_rows) {
    if (!backend || !ggml_backend_dev_is_meta(ggml_backend_get_device(backend)) ||
        active_rows == 0 || capacity_rows == 0 || active_rows > capacity_rows) {
        return false;
    }
    auto * ctx = static_cast<ggml_backend_meta_context *>(backend->context);
    typedef bool (*set_rows_t)(ggml_backend_t, uint32_t, uint32_t);
    std::vector<set_rows_t> setters(ctx->backend_configs.size(), nullptr);
    for (size_t i = 0; i < ctx->backend_configs.size(); ++i) {
        auto child = ctx->backend_configs[i].backend;
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(child));
        if (!reg) return false;
        setters[i] = reinterpret_cast<set_rows_t>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_predefined_rows"));
        if (!setters[i]) return false;
    }
    for (size_t i = 0; i < ctx->backend_configs.size(); ++i) {
        if (!setters[i](ctx->backend_configs[i].backend, active_rows, capacity_rows)) return false;
    }
    ctx->predefined_active_rows   = active_rows;
    ctx->predefined_capacity_rows = capacity_rows;
    ctx->predefined_capacity_outputs = capacity_rows;
    ctx->predefined_frame = {};
    ctx->predefined_frame.version = GGML_PREDEFINED_ABI_VERSION;
    ctx->predefined_frame.phase = GGML_PREDEFINED_DRAFT;
    ctx->predefined_frame.active_sequences = 1;
    ctx->predefined_frame.active_tokens = active_rows;
    ctx->predefined_frame.active_outputs = active_rows;
    ctx->predefined_frame.context_tokens = active_rows;
    ctx->predefined_frame.payload_elements = active_rows;
    ctx->predefined_frame_valid = true;
    return true;
}

bool ggml_backend_meta_set_predefined_frame(
        ggml_backend_t backend, const ggml_predefined_frame * frame,
        uint32_t capacity_rows, uint32_t capacity_outputs) {
    if (!backend || !frame || !ggml_backend_dev_is_meta(ggml_backend_get_device(backend)) ||
        frame->version != GGML_PREDEFINED_ABI_VERSION || frame->phase >= GGML_PREDEFINED_PHASE_COUNT ||
        frame->active_sequences == 0 || frame->active_tokens == 0 || capacity_rows == 0 ||
        capacity_outputs == 0 || frame->active_tokens > capacity_rows ||
        frame->active_outputs > capacity_outputs) {
        return false;
    }
    auto * ctx = static_cast<ggml_backend_meta_context *>(backend->context);
    typedef bool (*set_frame_t)(ggml_backend_t, const ggml_predefined_frame *, uint32_t, uint32_t);
    std::vector<set_frame_t> setters(ctx->backend_configs.size(), nullptr);
    for (size_t i = 0; i < ctx->backend_configs.size(); ++i) {
        auto child = ctx->backend_configs[i].backend;
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(child));
        if (!reg) return false;
        setters[i] = reinterpret_cast<set_frame_t>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_predefined_frame"));
        if (!setters[i]) return false;
    }
    // Preflight all child entry points before mutating any child state. The
    // setter itself only copies host metadata and cannot fail after validation.
    for (size_t i = 0; i < ctx->backend_configs.size(); ++i) {
        if (!setters[i](ctx->backend_configs[i].backend, frame, capacity_rows, capacity_outputs)) return false;
    }
    ctx->predefined_active_rows   = frame->active_tokens;
    ctx->predefined_capacity_rows = capacity_rows;
    ctx->predefined_capacity_outputs = capacity_outputs;
    ctx->predefined_frame         = *frame;
    ctx->predefined_frame_valid   = true;
    return true;
}

static const char * ggml_backend_meta_get_name(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    const ggml_backend_meta_context * backend_ctx = (const ggml_backend_meta_context *) backend->context;
    return backend_ctx->name.c_str();
}

static void ggml_backend_meta_free(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;
    if (!backend_ctx->shutdown()) {
        // A false return means native GPU work did not drain within its bounded
        // wait. Deliberately leak this meta backend rather than destroy buffers
        // or devices that may still be referenced by the driver.
        GGML_LOG_ERROR("%s: retaining unsafe meta backend after failed native teardown\n", __func__);
        return;
    }
    delete backend_ctx;
    delete backend;
}

bool ggml_backend_meta_device_copy_ranges(ggml_backend_t backend,
        const ggml_device_copy_range * ranges, size_t n_ranges, bool dry_run) {
    if (!backend || !ggml_backend_dev_is_meta(ggml_backend_get_device(backend)) ||
        !ggml_device_copy_ranges_valid(ranges, n_ranges)) {
        return false;
    }
    auto * ctx = static_cast<ggml_backend_meta_context *>(backend->context);
    const size_t ranks = ctx->backend_configs.size();
    if (ranks == 0 || ranks > GGML_BACKEND_META_MAX_DEVICES) {
        return false;
    }
    ggml_device_copy_range local[GGML_BACKEND_META_MAX_DEVICES][GGML_DEVICE_COPY_MAX_RANGES]{};
    for (size_t k = 0; k < n_ranges; ++k) {
        const auto & r = ranges[k];
        if (!ggml_backend_buffer_is_meta(r.src->buffer) || !ggml_backend_buffer_is_meta(r.dst->buffer) ||
            ggml_backend_meta_buffer_n_bufs(r.src->buffer) != ranks ||
            ggml_backend_meta_buffer_n_bufs(r.dst->buffer) != ranks ||
            ggml_backend_meta_get_split_state(r.src, false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED ||
            ggml_backend_meta_get_split_state(r.dst, false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return false;
        }
        for (size_t rank = 0; rank < ranks; ++rank) {
            local[rank][k] = {ggml_backend_meta_buffer_simple_tensor(r.src, rank),
                              ggml_backend_meta_buffer_simple_tensor(r.dst, rank),
                              r.src_offset, r.dst_offset, r.bytes};
        }
    }
    // Preflight every rank before recording the first. Model-specific wrappers
    // stay separate; copies are local to the matching physical device only.
    for (size_t rank = 0; rank < ranks; ++rank) {
        if (!ggml_backend_device_copy_ranges(ctx->backend_configs[rank].backend,
                                              local[rank], n_ranges, true)) {
            return false;
        }
    }
    if (!dry_run) {
        for (size_t rank = 0; rank < ranks; ++rank) {
            if (!ggml_backend_device_copy_ranges(ctx->backend_configs[rank].backend,
                                                  local[rank], n_ranges, false)) {
                GGML_LOG_ERROR("%s: device copy recording failed after preflight on rank %zu\n", __func__, rank);
                return false;
            }
        }
    }
    return true;
}

static void ggml_backend_meta_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(split_state.n_segments == 1);
    GGML_ASSERT(split_state.nr[0]      == 1);

    if (split_state.mapped_span) {
        GGML_ASSERT(ggml_meta_mapped_spans_valid(tensor, split_state, n_backends));
        const size_t chunk_size_full = ggml_meta_mapped_plane_bytes(tensor, split_state.axis);
        GGML_ASSERT(offset % chunk_size_full == 0);
        GGML_ASSERT(size   % chunk_size_full == 0);
        const int64_t i_start =  offset        / chunk_size_full;
        const int64_t i_stop  = (offset + size)/ chunk_size_full;
        const int64_t blck_size = split_state.axis == GGML_BACKEND_SPLIT_AXIS_0 ? ggml_blck_size(tensor->type) : 1;
        const size_t element_bytes = tensor->nb[split_state.axis];
        for (size_t j = 0; j < n_backends; ++j) {
            ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
            ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
            const size_t chunk_size_j = ggml_meta_mapped_plane_bytes(simple_tensor, split_state.axis);
            if (chunk_size_j == 0) continue;
            for (int32_t s = 0; s < split_state.span_count[j]; ++s) {
                const int64_t idx = ggml_meta_mapped_span_index(split_state, j, s);
                int64_t base = 0;
                for (int32_t t = 0; t < s; ++t) {
                    base += split_state.span_len[ggml_meta_mapped_span_index(split_state, j, t)];
                }
                const int64_t start = split_state.span_start[idx];
                const int64_t slen  = split_state.span_len[idx];
                const size_t src_off = (size_t)(start / blck_size) * element_bytes;
                const size_t nbytes   = (size_t)(slen / blck_size) * element_bytes;
                ggml_backend_tensor_set_2d_async(simple_backend, simple_tensor, (const char *) data + src_off,
                                                 i_start * chunk_size_j + (size_t)(base / blck_size) * element_bytes,
                                                 nbytes, i_stop - i_start, chunk_size_j, chunk_size_full);
            }
        }
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

            if (split_state.indexed_replica) {
                const int64_t blck_size = split_state.axis == GGML_BACKEND_SPLIT_AXIS_0 ? ggml_blck_size(tensor->type) : 1;
                const size_t element_bytes = tensor->nb[split_state.axis];
                for (size_t j = 0; j < n_backends; j++) {
                    ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
                    ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                    if (chunk_size_j == 0) continue;
                    GGML_ASSERT(split_state.replica_start[j] % blck_size == 0);
                    const size_t src_offset_in_chunk = (size_t)(split_state.replica_start[j] / blck_size) * element_bytes;
                    GGML_ASSERT(src_offset_in_chunk + chunk_size_j <= chunk_size_full);
                    const size_t simple_offset = i_start * chunk_size_j;
                    ggml_backend_tensor_set_2d_async(simple_backend, simple_tensor, (const char *) data + src_offset_in_chunk, simple_offset, chunk_size_j,
                        i_stop - i_start, chunk_size_j, chunk_size_full);
                }
                return;
            }

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

static bool ggml_backend_meta_set_tensor_snapshot_async(ggml_backend_t backend,
                                                        ggml_tensor *  tensor,
                                                        const void *   data,
                                                        size_t         offset,
                                                        size_t         size,
                                                        bool           dry_run) {
    if (!tensor || !tensor->buffer || !ggml_is_contiguous(tensor) || offset > ggml_nbytes(tensor) ||
        size > ggml_nbytes(tensor) - offset)
        return false;
    if (ggml_backend_meta_get_split_state(tensor, false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED)
        return false;
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    // A heterogeneous meta backend must not partially enqueue a mirrored write
    // and then fall back. Validate every destination before capturing any data.
    for (size_t j = 0; j < n_backends; ++j) {
        auto   simple = ggml_backend_meta_simple_backend(backend, j);
        auto * dst    = ggml_backend_meta_buffer_simple_tensor(tensor, j);
        if (!simple->iface.set_tensor_snapshot_async ||
            !simple->iface.set_tensor_snapshot_async(simple, dst, data, offset, size, true))
            return false;
    }
    if (dry_run)
        return true;
    for (size_t j = 0; j < n_backends; ++j) {
        auto       simple   = ggml_backend_meta_simple_backend(backend, j);
        auto *     dst      = ggml_backend_meta_buffer_simple_tensor(tensor, j);
        const bool captured = simple->iface.set_tensor_snapshot_async(simple, dst, data, offset, size, false);
        GGML_ASSERT(captured);
    }
    return true;
}

static void ggml_backend_meta_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(split_state.n_segments == 1);
    GGML_ASSERT(split_state.nr[0]      == 1);

    if (split_state.mapped_span) {
        // Canonical inverse readback (same deterministic lowest-device-owner
        // interval enumeration as the sync path; handles partial overlaps).
        GGML_ASSERT(ggml_meta_mapped_spans_valid(tensor, split_state, n_backends));
        const size_t chunk_size_full = ggml_meta_mapped_plane_bytes(tensor, split_state.axis);
        GGML_ASSERT(offset % chunk_size_full == 0);
        GGML_ASSERT(size   % chunk_size_full == 0);
        const int64_t i_start =  offset        / chunk_size_full;
        const int64_t i_stop  = (offset + size)/ chunk_size_full;
        const int64_t blck_size = split_state.axis == GGML_BACKEND_SPLIT_AXIS_0 ? ggml_blck_size(tensor->type) : 1;
        const size_t element_bytes = tensor->nb[split_state.axis];
        int64_t iv_span[2 * GGML_BACKEND_META_MAX_SPANS + 2][2];
        int iv_owner[2 * GGML_BACKEND_META_MAX_SPANS + 2];
        int64_t iv_local[2 * GGML_BACKEND_META_MAX_SPANS + 2];
        const size_t n_iv = ggml_meta_mapped_canonical_intervals(
                tensor, split_state, n_backends, iv_span, iv_owner, iv_local,
                sizeof(iv_span) / sizeof(iv_span[0]));
        for (size_t s = 0; s < n_iv; ++s) {
            const size_t rank = (size_t) iv_owner[s];
            ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, rank);
            const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, rank);
            const size_t chunk_size_j = ggml_meta_mapped_plane_bytes(simple_tensor, split_state.axis);
            GGML_ASSERT(iv_span[s][0] % blck_size == 0 && iv_span[s][1] % blck_size == 0);
            const size_t dst_off = (size_t)(iv_span[s][0] / blck_size) * element_bytes;
            const size_t nbytes  = (size_t)(iv_span[s][1] / blck_size) * element_bytes;
            ggml_backend_tensor_get_2d_async(simple_backend, simple_tensor, (char *) data + dst_off,
                                             i_start * chunk_size_j + (size_t)(iv_local[s] / blck_size) * element_bytes,
                                             nbytes, i_stop - i_start, chunk_size_j, chunk_size_full);
        }
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

            if (split_state.indexed_replica) {
                const int64_t blck_size = split_state.axis == GGML_BACKEND_SPLIT_AXIS_0 ? ggml_blck_size(tensor->type) : 1;
                const size_t element_bytes = tensor->nb[split_state.axis];
                ggml_meta_canonical_span spans[2 * GGML_BACKEND_META_MAX_DEVICES + 2];
                const size_t n_spans = ggml_meta_get_canonical_spans(tensor, split_state, n_backends, spans, sizeof(spans)/sizeof(spans[0]));
                for (size_t s = 0; s < n_spans; s++) {
                    const size_t rank = spans[s].rank;
                    ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, rank);
                    const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, rank);
                    const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                    GGML_ASSERT(spans[s].logical_start % blck_size == 0);
                    GGML_ASSERT(spans[s].local_start % blck_size == 0);
                    GGML_ASSERT(spans[s].length % blck_size == 0);
                    const size_t dst_offset_in_chunk = (size_t)(spans[s].logical_start / blck_size) * element_bytes;
                    const size_t local_offset_in_chunk = (size_t)(spans[s].local_start / blck_size) * element_bytes;
                    const size_t copy_bytes = (size_t)(spans[s].length / blck_size) * element_bytes;
                    ggml_backend_tensor_get_2d_async(simple_backend, simple_tensor, (char *) data + dst_offset_in_chunk,
                        i_start * chunk_size_j + local_offset_in_chunk, copy_bytes,
                        i_stop - i_start, chunk_size_j, chunk_size_full);
                }
                return;
            }

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
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    for (size_t i = 0; i < n_backends; i++) {
        ggml_backend_synchronize(ggml_backend_meta_simple_backend(backend, i));
    }
}

static enum ggml_status ggml_backend_meta_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    static int meta_compute_calls = 0;
    if (++meta_compute_calls <= 10 || meta_compute_calls % 100 == 0) {
        fprintf(stderr, "[META_GRAPH_COMPUTE] call=%d nodes=%d n_backends=%zu\n",
                meta_compute_calls, cgraph->n_nodes, ggml_backend_meta_n_backends(backend));
    }

    GGML_ASSERT(cgraph->grads == nullptr);
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;

    static std::atomic<uint64_t> s_tp5_exec_id{0};
    ggml_tp5_profile_begin(++s_tp5_exec_id, cgraph->n_nodes <= 4);
    struct ggml_tp5_profile_scope {
        ~ggml_tp5_profile_scope() { ggml_tp5_profile_end(); }
    } tp5_profile_scope;

    // If the previous cgraph had a defined UID it can be used to skip rebuilding the subgraphs per simple backend.
    const bool needs_rebuild =
        (cgraph->uid == 0) || (cgraph->uid != backend_ctx->uid) || (cgraph->n_nodes != backend_ctx->n_nodes);

    if (needs_rebuild) {
        // Old CB/tensor handles stop being executable BEFORE their graphs are
        // reset below. A same-sized replacement is still a different definition.
        backend_ctx->predefined_valid = false;
        backend_ctx->predefined_n_subgraphs = 0;
        backend_ctx->predefined_uid = 0;
        backend_ctx->chain_compute_cbs.clear();
        backend_ctx->chain_tensors.clear();
        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            bcj.nodes.resize(cgraph->n_nodes);
            bcj.cgraphs.clear();
        }
        std::set<ggml_backend_buffer_t> used_buffers;
        for (int i = 0; i < cgraph->n_leafs; i++) {
            if (ggml_backend_buffer_is_meta(cgraph->leafs[i]->buffer)) {
                used_buffers.emplace(cgraph->leafs[i]->buffer);
            }
        }
        for (int i = 0; i < cgraph->n_nodes; i++) {
            if (ggml_backend_buffer_is_meta(cgraph->nodes[i]->buffer)) {
                used_buffers.emplace(cgraph->nodes[i]->buffer);
            }
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
            // For MoE models it may make sense to delay the AllReduce in order to reduce I/O:
            auto get_i_delayed = [&](const int i) -> int {
                int id = i; // i_delayed
                int idr = i; // i_delayed return, last safe return value

                ggml_tensor * node = cgraph->nodes[id];
                int32_t n_used = ggml_node_get_use_count(cgraph, id);

                // Skip MIRRORED nodes that don't consume node
                auto skip_unrelated = [&]() {
                    while (id + 1 < cgraph->n_nodes) {
                        ggml_tensor * next = cgraph->nodes[id+1];
                        if (ggml_backend_meta_get_split_state(next, false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                            break;
                        }
                        bool safe = true;
                        for (int s = 0; s < GGML_MAX_SRC; s++) {
                            if (next->src[s] == nullptr) {
                                continue;
                            }
                            if (next->src[s] == node) {
                                safe = false;
                                break;
                            }
                            if (ggml_backend_meta_get_split_state(next->src[s], false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
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
                            ggml_backend_meta_get_split_state(next->src[1], false).axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL &&
                            ggml_backend_meta_get_split_state(next->src[2], false).axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                        node = next;
                        id++;
                        idr = id;
                        n_used = ggml_node_get_use_count(cgraph, id);
                    }
                }
                // Chain of MULs with MIRRORED src[1]
                while (true) {
                    skip_unrelated();
                    if (id + 1 >= cgraph->n_nodes) {
                        return idr;
                    }
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op == GGML_OP_MUL && next->src[0] == node &&
                            ggml_backend_meta_get_split_state(next->src[1], false).axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                        node = next;
                        id++;
                        idr = id;
                        n_used = ggml_node_get_use_count(cgraph, id);
                    } else {
                        break;
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

            // Conservative structurally justified linear region for PARTIAL deferral:
            // A PARTIAL node can defer its AllReduce if it is consumed solely within a local
            // linear subdag whose ultimate sink will reduce.
            // Invariants:
            // 1. Never defer past non-linear operations (activations, norms, rots).
            // 2. Never defer into an ADD with a MIRRORED operand (residual) without reducing first;
            //    otherwise each rank would duplicate the residual (sum_{r=1}^P (x_r + res) = sum(x_r) + P*res).
            // 3. Never defer past divergent uses or observable outputs (flags & (OUTPUT | LOSS)).
            // 4. PARTIAL routed + PARTIAL (shared * mirrored_gate) can combine once before reducing.
            // Reuse the graph's identity hash. Repeated linear searches here
            // otherwise rescan the entire graph for every ancestor of each sink.
            std::vector<int> node_indices(cgraph->visited_hash_set.size, -1);
            for (int idx = 0; idx < cgraph->n_nodes; ++idx) {
                const size_t pos = ggml_hash_find(&cgraph->visited_hash_set, cgraph->nodes[idx]);
                GGML_ASSERT(pos < node_indices.size());
                node_indices[pos] = idx;
            }
            auto get_node_idx = [&](const ggml_tensor * t) -> int {
                const size_t pos = ggml_hash_find(&cgraph->visited_hash_set, t);
                return pos < node_indices.size() ? node_indices[pos] : -1;
            };

            auto can_defer_linear_partial = [&](const int from) -> int {
                int id = from;
                if (ggml_node_get_use_count(cgraph, id) != 1) {
                    return from;
                }
                if (cgraph->nodes[id]->flags & (GGML_TENSOR_FLAG_OUTPUT | GGML_TENSOR_FLAG_LOSS)) {
                    return from;
                }

                // Graph-local speculative split state tracker initialized freshly for this 'from'
                std::map<const ggml_tensor *, ggml_backend_meta_split_state> local_ss;
                auto get_local_ss = [&](const ggml_tensor * t) -> ggml_backend_meta_split_state {
                    if (!t) return { GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1, false, {0} };
                    int idx = get_node_idx(t);
                    if (idx >= 0 && idx < from) {
                        // Already reduced in an earlier subgraph -> MIRRORED
                        return { GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1, false, {0} };
                    }
                    auto it = local_ss.find(t);
                    if (it != local_ss.end()) {
                        return it->second;
                    }
                    return ggml_backend_meta_get_split_state(t, false);
                };

                auto is_mirrored = [&](const ggml_tensor * t) -> bool {
                    return get_local_ss(t).axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED;
                };
                auto is_partial = [&](const ggml_tensor * t) -> bool {
                    return get_local_ss(t).axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL;
                };

                // Initialize from node
                local_ss[cgraph->nodes[id]] = { GGML_BACKEND_SPLIT_AXIS_PARTIAL, {0}, {1}, 1, false, {0} };

                // Topologically classify all nodes in the interval [from + 1, n_nodes - 1]
                std::vector<int> candidates;
                ggml_tensor * cur = cgraph->nodes[id];

                for (int k = id + 1; k < cgraph->n_nodes; ++k) {
                    ggml_tensor * next = cgraph->nodes[k];

                    // Classify next using linear rules:
                    if (next->op == GGML_OP_MUL_MAT || next->op == GGML_OP_MUL_MAT_ID) {
                        // Intrinsic matmul split rule
                        local_ss[next] = ggml_backend_meta_get_split_state(next, false);
                    } else if (next->op == GGML_OP_MUL) {
                        bool p0 = is_partial(next->src[0]);
                        bool p1 = is_partial(next->src[1]);
                        if ((p0 && is_mirrored(next->src[1])) || (p1 && is_mirrored(next->src[0]))) {
                            local_ss[next] = { GGML_BACKEND_SPLIT_AXIS_PARTIAL, {0}, {1}, 1, false, {0} };
                        } else if (is_mirrored(next->src[0]) && is_mirrored(next->src[1])) {
                            local_ss[next] = { GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1, false, {0} };
                        } else {
                            local_ss[next] = ggml_backend_meta_get_split_state(next, false);
                        }
                    } else if (next->op == GGML_OP_ADD || next->op == GGML_OP_ADD_ID) {
                        bool p0 = is_partial(next->src[0]);
                        bool p1 = is_partial(next->src[1]);
                        if (p0 && p1) {
                            local_ss[next] = { GGML_BACKEND_SPLIT_AXIS_PARTIAL, {0}, {1}, 1, false, {0} };
                        } else {
                            local_ss[next] = ggml_backend_meta_get_split_state(next, false);
                        }
                    } else if (next->op == GGML_OP_SCALE || next->op == GGML_OP_VIEW ||
                               next->op == GGML_OP_RESHAPE || next->op == GGML_OP_TRANSPOSE ||
                               next->op == GGML_OP_PERMUTE) {
                        if (is_partial(next->src[0])) {
                            local_ss[next] = { GGML_BACKEND_SPLIT_AXIS_PARTIAL, {0}, {1}, 1, false, {0} };
                        } else if (is_mirrored(next->src[0])) {
                            local_ss[next] = { GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1, false, {0} };
                        } else {
                            local_ss[next] = ggml_backend_meta_get_split_state(next, false);
                        }
                    } else {
                        local_ss[next] = ggml_backend_meta_get_split_state(next, false);
                    }

                    // Now check if next consumes cur
                    bool uses_cur = false;
                    for (int s = 0; s < GGML_MAX_SRC; ++s) {
                        if (next->src[s] == cur) {
                            uses_cur = true;
                            break;
                        }
                    }
                    if (!uses_cur) {
                        continue;
                    }

                    if (next->flags & (GGML_TENSOR_FLAG_OUTPUT | GGML_TENSOR_FLAG_LOSS)) {
                        break;
                    }

                    if (next->op == GGML_OP_ADD || next->op == GGML_OP_ADD_ID) {
                        if (is_partial(next->src[0]) && is_partial(next->src[1])) {
                            cur = next;
                            candidates.push_back(k);
                            if (ggml_node_get_use_count(cgraph, k) > 1) {
                                break;
                            }
                            continue;
                        }
                        break; // One is mirrored residual -> stop
                    } else if (next->op == GGML_OP_MUL) {
                        if (is_partial(next)) {
                            cur = next;
                            candidates.push_back(k);
                            if (ggml_node_get_use_count(cgraph, k) > 1) {
                                break;
                            }
                            continue;
                        }
                        break;
                    } else if (next->op == GGML_OP_SCALE || next->op == GGML_OP_VIEW ||
                               next->op == GGML_OP_RESHAPE || next->op == GGML_OP_TRANSPOSE ||
                               next->op == GGML_OP_PERMUTE) {
                        if (is_partial(next)) {
                            cur = next;
                            candidates.push_back(k);
                            if (ggml_node_get_use_count(cgraph, k) > 1) {
                                break;
                            }
                            continue;
                        }
                        break;
                    } else {
                        break;
                    }
                }

                // Validate interval closure for candidates
                for (auto it = candidates.rbegin(); it != candidates.rend(); ++it) {
                    const int candidate_sink = *it;
                    ggml_tensor * sink_node = cgraph->nodes[candidate_sink];
                    std::set<const ggml_tensor *> sink_ancestors;
                    auto collect_ancestors = [&](auto & self, const ggml_tensor * t) -> void {
                        if (!t || sink_ancestors.count(t)) return;
                        const int idx = get_node_idx(t);
                        // Earlier topological nodes have already crossed a
                        // reduction boundary. Their ancestors cannot belong
                        // to the interval being validated below.
                        if (idx >= 0 && idx < from)
                            return;
                        sink_ancestors.insert(t);
                        for (int s = 0; s < GGML_MAX_SRC; ++s) {
                            if (t->src[s]) self(self, t->src[s]);
                        }
                    };
                    collect_ancestors(collect_ancestors, sink_node);

                    bool interval_valid = true;
                    for (int mid = from; mid < candidate_sink; ++mid) {
                        ggml_tensor * mid_node = cgraph->nodes[mid];
                        if (is_partial(mid_node)) {
                            if (!sink_ancestors.count(mid_node)) {
                                interval_valid = false;
                                break;
                            }
                            for (int fwd = from; fwd < cgraph->n_nodes; ++fwd) {
                                ggml_tensor * consumer = cgraph->nodes[fwd];
                                for (int s = 0; s < GGML_MAX_SRC; ++s) {
                                    if (consumer->src[s] == mid_node) {
                                        if (fwd > candidate_sink) {
                                            interval_valid = false;
                                            break;
                                        }
                                    }
                                }
                                if (!interval_valid) break;
                            }
                        }
                        if (!interval_valid) break;
                    }

                    if (interval_valid) {
                        for (const ggml_tensor * anc : sink_ancestors) {
                            int anc_idx = get_node_idx(anc);
                            if (anc_idx >= from && anc_idx <= candidate_sink && is_partial(anc)) {
                                for (size_t j = 0; j < n_backends; ++j) {
                                    if ((backend_ctx->backend_configs[j].nodes[anc_idx]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                                        interval_valid = false;
                                        break;
                                    }
                                }
                            }
                            if (!interval_valid) break;
                        }
                    }
                    if (interval_valid) return candidate_sink;
                }
                return from;
            };

            int i_start = 0;
            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];
                if (node->view_src != nullptr && node->view_src->op == GGML_OP_NONE && ggml_backend_buffer_is_host(node->view_src->buffer)) {
                    continue;
                }
                const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(node, /*assume_sync =*/ false);
                if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                    max_tmp_size = std::max(max_tmp_size, ggml_nbytes(node));
                }
                const bool new_subgraph = i + 1 == cgraph->n_nodes || split_state.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL;
                if (!new_subgraph) {
                    continue;
                }

                int i_delayed = get_i_delayed(i);
                const int i_expert_delayed = i_delayed;

                // If get_i_delayed deferred across an expert view/add stack or stopped at i,
                // chain into the conservative structural linear region analyzer:
                // Allows combining the routed MoE output with the shared expert gated output
                // into a single PARTIAL sum before reducing (exact 1 AllReduce per FFN).
                const int i_linear = can_defer_linear_partial(i_delayed);
                if (i_linear > i_delayed) {
                    // Only update i_delayed if the linear deferral chain reaches a valid sink
                    i_delayed = i_linear;
                }

                // If get_i_delayed deferred across zero-sized expert slices, disable compute
                // ONLY for the expert segment (i+1 to i_expert_delayed), preserving valid shared work:
                if (i_expert_delayed > i) {
                    for (size_t j = 0; j < n_backends; j++) {
                        auto & bcj = backend_ctx->backend_configs[j];
                        if ((bcj.nodes[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                            for (int ii = i + 1; ii <= i_expert_delayed; ii++) {
                                bcj.nodes[ii]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
                            }
                        }
                    }
                }

                // If we can delay the AllReduce we need to consider the interaction with zero-sized tensor slices.
                // A backend with such a slice would normally have valid data after participating in the AllReduce with a node that has
                i = i_delayed;

                for (size_t j = 0; j < n_backends; j++) {
                    auto & bcj = backend_ctx->backend_configs[j];
                    bcj.cgraphs.push_back({ nullptr, i_start });
                }
                n_subgraphs++;
                i_start = i + 1;
            }
            GGML_ASSERT(i_start == cgraph->n_nodes);
        }

        backend_ctx->uid         = cgraph->uid;
        backend_ctx->n_nodes     = cgraph->n_nodes;
        backend_ctx->n_subgraphs = n_subgraphs;

        if (needs_rebuild && backend_ctx->debug > 0) {
            fprintf(stderr, "[meta-tp5] subgraph boundaries: %zu\n",
                    n_subgraphs);
        }

        if (max_tmp_size > backend_ctx->max_tmp_size) {
            for (size_t j = 0; j < n_backends; j++) {
                auto & bcj = backend_ctx->backend_configs[j];
                for (size_t i = 0; i < backend_ctx->n_reduce_steps; i++) {
                    bcj.bufs[i].reset(ggml_backend_alloc_buffer(bcj.backend, max_tmp_size));
                }
            }
            backend_ctx->max_tmp_size = max_tmp_size;
        }

        if (needs_rebuild) {
            const size_t n_nodes_per_device = 3 * backend_ctx->n_reduce_steps; // tmp + ADD (+zeroing) graph per step and device
            const size_t n_cgraphs_per_device = 2 * backend_ctx->n_reduce_steps; // ADD ( + zeroing) graph per step and device
            const auto & ranges        = backend_ctx->backend_configs.front().cgraphs;
            const auto   subgraph_size = [&](size_t i) {
                const int stop = i + 1 < n_subgraphs ? ranges[i + 1].offset : cgraph->n_nodes;
                return size_t(stop - ranges[i].offset);
            };
            // Each local graph owns only its reduction span, not a copy of the
            // entire model's node/hash capacity at every reduction boundary.
            size_t mem_per_device_graphs_main = 0;
            for (size_t i = 0; i < n_subgraphs; ++i) {
                mem_per_device_graphs_main += ggml_graph_overhead_custom(subgraph_size(i), false);
            }
            const size_t mem_per_device_graphs_aux =
                n_cgraphs_per_device * n_subgraphs * ggml_graph_overhead_custom(1, false);
            const size_t           mem_per_device_nodes_aux = n_nodes_per_device * n_subgraphs * ggml_tensor_overhead();
            const ggml_init_params params = {
                /*.mem_size   =*/ n_backends * (mem_per_device_graphs_main + mem_per_device_graphs_aux + mem_per_device_nodes_aux),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            backend_ctx->ctx.reset(ggml_init(params));
            for (size_t j = 0; j < n_backends; j++) {
                auto & bcj = backend_ctx->backend_configs[j];
                for (size_t i = 0; i < n_subgraphs; i++) {
                    bcj.cgraphs[i].cgraph_main =
                        ggml_new_graph_custom(backend_ctx->ctx.get(), subgraph_size(i), /*grads =*/false);
                }
            }
            backend_ctx->cgraphs_aux.resize(n_backends * n_cgraphs_per_device * n_subgraphs);
            for (size_t k = 0; k < backend_ctx->cgraphs_aux.size(); k++) {
                backend_ctx->cgraphs_aux[k] = ggml_new_graph_custom(backend_ctx->ctx.get(), 1, cgraph->grads);
            }
            backend_ctx->nodes_aux.resize(n_backends * n_nodes_per_device * n_subgraphs);
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
                // Stable UID across decode tokens when graph structure does not change (TP5.md 13.5)
                // Replay cache in simple backends looks up by cgraph_uid to hit recorded command buffers.
                uint64_t stable_subgraph_uid = (cgraph->uid != 0) ? ((cgraph->uid << 16) | (uint64_t) i_graph) : ggml_graph_next_uid();
                cgraph_ij->uid = stable_subgraph_uid;
            }
        }
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


    // FAST PATH: Single-submit epoch chain for TP5 decode (O(1) CPU submissions per token)
    typedef bool (*tp5_submit_epoch_chain_t)(void * comm_ctx,
                                            const std::vector<std::vector<std::vector<void *>>> & stage_compute_cbs,
                                            const std::vector<std::vector<ggml_tensor *>> & stage_tensors,
                                            const ggml_predefined_frame * frame,
                                            uint32_t capacity_rows, uint32_t capacity_outputs);
    typedef bool (*tp5_get_cached_cmd_bufs_t)(ggml_backend_t backend, ggml_cgraph * cgraph, std::vector<void *> & out_cbs);
    static tp5_submit_epoch_chain_t pfn_submit_chain = nullptr;
    static tp5_get_cached_cmd_bufs_t pfn_get_cbs = nullptr;
    static bool pfn_chain_checked = false;
    static bool printed_route_once = false;
    if (!printed_route_once) {
        printed_route_once = true;
        fprintf(stderr, "[tp5-route-check] comm_ctx=%p n_backends=%zu n_subgraphs=%zu pfn_submit_chain=%p pfn_get_cbs=%p\n",
                backend_ctx->comm_ctx, n_backends, backend_ctx->n_subgraphs, (void*)pfn_submit_chain, (void*)pfn_get_cbs);
    }
    if (!pfn_chain_checked && backend_ctx->comm_ctx && n_backends == 5) {
        pfn_chain_checked = true;
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_ctx->backend_configs[0].backend));
        pfn_submit_chain = (tp5_submit_epoch_chain_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_submit_epoch_chain");
        pfn_get_cbs = (tp5_get_cached_cmd_bufs_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_vk_get_cached_cmd_bufs");
    }

    if (pfn_submit_chain && pfn_get_cbs && backend_ctx->comm_ctx && n_backends == 5 && backend_ctx->n_subgraphs > 1 &&
        (backend_ctx->n_subgraphs - 1) <= 128) {
        // 纯 Predefine 路线（唯一真理）：一旦初次建立预定义图句柄后，直接 100% 绝对复用，跳过逐轮 validation 开销
        auto & stage_compute_cbs = backend_ctx->chain_compute_cbs;
        auto & stage_tensors     = backend_ctx->chain_tensors;

        if (backend_ctx->predefined_valid &&
            backend_ctx->predefined_n_subgraphs == backend_ctx->n_subgraphs &&
            backend_ctx->predefined_uid == backend_ctx->uid) {
            static const bool chain_timing_enabled = getenv("GGML_TP5_PROFILE") != nullptr;
            static int chain_hit_log = 0;
            bool chain_ok = false;
            if (chain_timing_enabled) {
                const auto t_chain_call_0 = std::chrono::high_resolution_clock::now();
                chain_ok = pfn_submit_chain(backend_ctx->comm_ctx, stage_compute_cbs, stage_tensors,
                                            backend_ctx->predefined_frame_valid ? &backend_ctx->predefined_frame : nullptr,
                                            backend_ctx->predefined_capacity_rows,
                                            backend_ctx->predefined_capacity_outputs);
                const auto t_chain_call_1 = std::chrono::high_resolution_clock::now();
                const double chain_call_ms = std::chrono::duration<double, std::milli>(t_chain_call_1 - t_chain_call_0).count();
                fprintf(stderr, "\n[META_CHAIN_EXEC_TIME] pfn_submit_chain took %6.2f ms (ok=%d, n_stages=%zu, predefine=true)\n\n",
                        chain_call_ms, (int)chain_ok, backend_ctx->n_subgraphs);
            } else {
                chain_ok = pfn_submit_chain(backend_ctx->comm_ctx, stage_compute_cbs, stage_tensors,
                                            backend_ctx->predefined_frame_valid ? &backend_ctx->predefined_frame : nullptr,
                                            backend_ctx->predefined_capacity_rows,
                                            backend_ctx->predefined_capacity_outputs);
            }
            if (chain_ok) {
                if (++chain_hit_log <= 5 || chain_hit_log % 100 == 0) {
                    fprintf(stderr, "[tp5-meta] SUBMIT_EPOCH_CHAIN (PREDEFINED TRUTH): hits=%d\n", chain_hit_log);
                }
                return GGML_STATUS_SUCCESS;
            }
            return GGML_STATUS_FAILED;
        }

        bool all_replay_cached = true;
        // Every compute graph, including the non-reducing model tail, belongs
        // to the same rank submission. Only the first N-1 graphs reduce.
        // Reuse vector capacity; every entry is still refreshed and validated
        // before submission, including after a graph rebuild or cache miss.
        stage_compute_cbs.resize(backend_ctx->n_subgraphs);
        stage_tensors.resize(backend_ctx->n_subgraphs - 1);

        for (size_t i = 0; i < backend_ctx->n_subgraphs; ++i) {
            stage_compute_cbs[i].resize(n_backends);
            if (i < stage_tensors.size())
                stage_tensors[i].resize(n_backends);
            for (size_t j = 0; j < n_backends; ++j) {
                auto & bcj = backend_ctx->backend_configs[j];
                ggml_cgraph * cgraph_ij = bcj.cgraphs[i].cgraph_main;
                if (i < stage_tensors.size())
                    stage_tensors[i][j] = cgraph_ij->nodes[cgraph_ij->n_nodes - 1];
                if (backend_ctx->comm_prepare &&
                    !backend_ctx->comm_prepare(backend_ctx->comm_ctx, j, cgraph_ij, i < stage_tensors.size(), i)) {
                    return GGML_STATUS_FAILED;
                }
                if (!pfn_get_cbs(bcj.backend, cgraph_ij, stage_compute_cbs[i][j])) {
                    static int miss_log = 0;
                    if (++miss_log <= 5) {
                        fprintf(stderr, "[meta-chain-miss] stage=%zu rank=%zu nodes=%d uid=%llu\n", i, j,
                                cgraph_ij->n_nodes, (unsigned long long) cgraph_ij->uid);
                    }
                    all_replay_cached = false;
                    break;
                }
            }
            if (!all_replay_cached) break;
        }
        if (all_replay_cached) {
            static bool segments_dumped = false;
            if (!segments_dumped) {
                if (const char * path = getenv("GGML_TP5_SEGMENT_DUMP")) {
                    FILE * file = fopen(path, "w");
                    GGML_ASSERT(file);
                    for (size_t stage = 0; stage < backend_ctx->n_subgraphs; ++stage) {
                        for (size_t rank = 0; rank < n_backends; ++rank) {
                            const ggml_cgraph * graph = backend_ctx->backend_configs[rank].cgraphs[stage].cgraph_main;
                            for (int node = 0; node < graph->n_nodes; ++node) {
                                const ggml_tensor * tensor = graph->nodes[node];
                                for (int slot = -1; slot < GGML_MAX_SRC; ++slot) {
                                    const ggml_tensor * value = slot < 0 ? tensor : tensor->src[slot];
                                    if (!value)
                                        continue;
                                    fprintf(file, "%zu\t%zu\t%d\t%d\t%p\t%s\t%s\t%s\t%u\t%p\t%zu\t%p\t%p", stage, rank,
                                            node, slot, (const void *) value, value->name, ggml_op_desc(value),
                                            ggml_type_name(value->type), (unsigned) value->flags,
                                            (const void *) value->view_src, value->view_offs, value->data,
                                            (const void *) value->buffer);
                                    for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
                                        fprintf(file, "\t%lld\t%zu", (long long) value->ne[dim], value->nb[dim]);
                                    }
                                    for (int param = 0; param < GGML_MAX_OP_PARAMS / 4; ++param) {
                                        fprintf(file, "\t%08x", (unsigned) value->op_params[param]);
                                    }
                                    fputc('\n', file);
                                }
                            }
                        }
                    }
                    GGML_ASSERT(fclose(file) == 0);
                    segments_dumped = true;
                }
            }
            // One queue call per rank; owner retirement may wait once per chain,
            // never between individual compute/reduction stages.
            static const bool chain_timing_enabled = getenv("GGML_TP5_PROFILE") != nullptr;
            static int        chain_hit_log = 0;
            bool              chain_ok;
            if (chain_timing_enabled) {
                const auto t_chain_call_0 = std::chrono::high_resolution_clock::now();
                chain_ok = pfn_submit_chain(backend_ctx->comm_ctx, stage_compute_cbs, stage_tensors,
                                            backend_ctx->predefined_frame_valid ? &backend_ctx->predefined_frame : nullptr,
                                            backend_ctx->predefined_capacity_rows,
                                            backend_ctx->predefined_capacity_outputs);
                const auto t_chain_call_1 = std::chrono::high_resolution_clock::now();
                const double chain_call_ms = std::chrono::duration<double, std::milli>(t_chain_call_1 - t_chain_call_0).count();
                fprintf(stderr, "\n[META_CHAIN_EXEC_TIME] pfn_submit_chain took %6.2f ms (ok=%d, n_stages=%zu)\n\n",
                        chain_call_ms, (int)chain_ok, backend_ctx->n_subgraphs);
            } else {
                chain_ok = pfn_submit_chain(backend_ctx->comm_ctx, stage_compute_cbs, stage_tensors,
                                            backend_ctx->predefined_frame_valid ? &backend_ctx->predefined_frame : nullptr,
                                            backend_ctx->predefined_capacity_rows,
                                            backend_ctx->predefined_capacity_outputs);
            }
            if (chain_ok) {
                backend_ctx->predefined_valid = true;
                backend_ctx->predefined_n_subgraphs = backend_ctx->n_subgraphs;
                backend_ctx->predefined_uid = backend_ctx->uid;
                if (++chain_hit_log <= 5 || chain_hit_log % 100 == 0) {
                    fprintf(stderr, "[tp5-meta] SUBMIT_EPOCH_CHAIN SUCCESS: predefined truth locked! (hits=%d)\n", chain_hit_log);
                }
                return GGML_STATUS_SUCCESS;
            } else {
                static int chain_fail_log = 0;
                if (++chain_fail_log <= 5) {
                    fprintf(stderr, "[tp5-meta] SUBMIT_EPOCH_CHAIN FAILED; refusing fallback after native submission\n");
                }
                return GGML_STATUS_FAILED;
            }
        }
    }

    for (size_t i = 0; i < backend_ctx->n_subgraphs; i++) {
        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            if (backend_ctx->comm_prepare &&
                !backend_ctx->comm_prepare(backend_ctx->comm_ctx, j, bcj.cgraphs[i].cgraph_main,
                                           i < backend_ctx->n_subgraphs - 1, i)) {
                return GGML_STATUS_FAILED;
            }
            const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, bcj.cgraphs[i].cgraph_main);
            if (status != GGML_STATUS_SUCCESS) {
                return GGML_STATUS_FAILED;
            }
        }

        if (n_backends > 1 && i < backend_ctx->n_subgraphs - 1) {
            bool backend_allreduce_success = false;
            if (backend_ctx->comm_ctx) {
                // Radical optimization (TP5.md 13.3 / 13.6): Asynchronously flush all backend
                // compute command buffers to the GPU queue with ZERO host fence wait!
                typedef void (*tp5_flush_async_t)(ggml_backend_t backend);
                static tp5_flush_async_t pfn_flush = nullptr;
                static bool pfn_checked = false;
                if (!pfn_checked) {
                    pfn_checked = true;
                    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_ctx->backend_configs[0].backend));
                    pfn_flush = (tp5_flush_async_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_vk_flush_async");
                }
                if (pfn_flush && backend_ctx->comm_allreduce) {
                    for (size_t j = 0; j < n_backends; j++) {
                        pfn_flush(backend_ctx->backend_configs[j].backend);
                    }
                }
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
                // If no native comm_ctx is configured (e.g. CPU meta test/execution), use generic CPU allreduce_fallback.
                // If comm_ctx is configured and failed, fail closed without falling back.
                if (!backend_ctx->comm_ctx) {
                    const ggml_status status = allreduce_fallback(i);
                    if (status != GGML_STATUS_SUCCESS) {
                        return status;
                    }
                } else {
                    ggml_tensor * last_node = cgraph->nodes[backend_ctx->backend_configs[0].cgraphs[i].offset + backend_ctx->backend_configs[0].cgraphs[i].cgraph_main->n_nodes - 1];
                    fprintf(stderr, "ggml-backend-meta: AllReduce failed on subgraph %zu (node '%s')!\n",
                            i, last_node ? last_node->name : "null");
                    return GGML_STATUS_FAILED;
                }
            }
        }
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_meta_graph_optimize(ggml_backend_t                       backend,
                                             ggml_cgraph *                        graph,
                                             ggml_backend_graph_optimize_params * params) {
    auto * ctx         = (ggml_backend_meta_context *) backend->context;
    using alloc_deps_t = void (*)(ggml_backend_t, ggml_cgraph *, ggml_backend_graph_optimize_params *);
    for (size_t i = 0; i < ctx->backend_configs.size(); ++i) {
        ggml_backend_t     child  = ctx->backend_configs[i].backend;
        ggml_backend_dev_t device = ggml_backend_get_device(child);
        if (!device)
            continue;
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
        if (!reg)
            continue;
        bool already_seen = false;
        for (size_t j = 0; j < i; ++j) {
            ggml_backend_dev_t previous = ggml_backend_get_device(ctx->backend_configs[j].backend);
            if (previous && ggml_backend_dev_backend_reg(previous) == reg) {
                already_seen = true;
                break;
            }
        }
        if (already_seen)
            continue;
        const auto advise = (alloc_deps_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_graph_alloc_deps");
        // This optional procedure only adds allocation dependencies. Calling
        // graph_optimize directly could reorder cross-rank reduction boundaries.
        if (advise)
            advise(child, graph, params);
    }
}

static const ggml_backend_i ggml_backend_meta_i = {
    /* .get_name                = */ ggml_backend_meta_get_name,
    /* .free                    = */ ggml_backend_meta_free,
    /* .set_tensor_async        = */ ggml_backend_meta_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_meta_get_tensor_async,
    /* .set_tensor_2d_async     = */ nullptr,
    /* .get_tensor_2d_async     = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ ggml_backend_meta_synchronize,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ ggml_backend_meta_graph_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .graph_optimize          = */ ggml_backend_meta_graph_optimize,
    /* .set_tensor_snapshot_async = */ ggml_backend_meta_set_tensor_snapshot_async,
};

bool ggml_backend_is_meta(ggml_backend_t backend) {
    return backend != nullptr && backend->iface.get_name == ggml_backend_meta_i.get_name;
}

static ggml_backend_t ggml_backend_meta_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    ggml_backend_meta_context * backend_ctx = new ggml_backend_meta_context(dev, params);
    if (backend_ctx->backend_configs.empty()) {
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
