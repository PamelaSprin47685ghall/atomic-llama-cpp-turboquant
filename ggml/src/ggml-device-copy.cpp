#include "ggml-device-copy.h"
#include "ggml-impl.h"
#include <exception>

// Meta owns the model-specific tensor wrappers. Keep their mapping there,
// rather than pretending two meta buffer types have the same split policy.
bool ggml_backend_meta_device_copy_ranges(
        ggml_backend_t, const ggml_device_copy_range *, size_t, bool);

bool ggml_device_copy_ranges_valid(const ggml_device_copy_range * ranges, size_t n) {
    if (n > GGML_DEVICE_COPY_MAX_RANGES || (n != 0 && ranges == nullptr)) {
        return false;
    }
    for (size_t i = 0; i < n; ++i) {
        const auto & r = ranges[i];
        if (!r.src || !r.dst || !r.src->buffer || !r.dst->buffer ||
            r.src->type != GGML_TYPE_F32 || r.dst->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(r.src) || !ggml_is_contiguous(r.dst) ||
            ((r.src_offset | r.dst_offset | r.bytes) % sizeof(float)) != 0) {
            return false;
        }
        const size_t src_size = ggml_nbytes(r.src), dst_size = ggml_nbytes(r.dst);
        if (r.src_offset > src_size || r.bytes > src_size - r.src_offset ||
            r.dst_offset > dst_size || r.bytes > dst_size - r.dst_offset) {
            return false;
        }
    }
    return true;
}

bool ggml_backend_device_copy_ranges(ggml_backend_t executor, const ggml_device_copy_range * ranges,
                                      size_t n, bool dry_run) {
    if (!executor || !ggml_device_copy_ranges_valid(ranges, n)) {
        return false;
    }
    if (n == 0) {
        return true;
    }
    auto * device = ggml_backend_get_device(executor);
    if (!device) {
        return false;
    }
    if (ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_META) {
        try {
            return ggml_backend_meta_device_copy_ranges(executor, ranges, n, dry_run);
        } catch (const std::exception & e) {
            GGML_LOG_ERROR("%s: meta device copy failed: %s\n", __func__, e.what());
            return false;
        }
    }
    auto * reg = ggml_backend_dev_backend_reg(device);
    if (!reg) {
        return false;
    }
    auto fn = reinterpret_cast<ggml_backend_device_copy_ranges_t>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_device_copy_ranges"));
    try {
        return fn && fn(executor, ranges, n, dry_run);
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("%s: native device copy failed: %s\n", __func__, e.what());
        return false;
    }
}
