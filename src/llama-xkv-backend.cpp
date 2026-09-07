// llama-xkv-backend.cpp — immutable backend-resident XKV code-stream allocations.
//
// Implements xkv_backend_allocation and batch builder with failure-atomic uploads.
// Owned by XkvBackendResidency. Avoid llama-xkv-cache/runtime/graph, ggml native op
// implementation, server or state.

#include "llama-xkv-backend.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>

namespace llama_xkv {

namespace {

// Checked arithmetic helpers (fail-closed, never wrap).
inline bool add_ok_u64(uint64_t a, uint64_t b, uint64_t & out) {
    if (b > std::numeric_limits<uint64_t>::max() - a) {
        return false;
    }
    out = a + b;
    return true;
}

inline bool mul_ok_u64(uint64_t a, uint64_t b, uint64_t & out) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
        return false;
    }
    out = a * b;
    return true;
}

inline bool add_ok_size(size_t a, size_t b, size_t & out) {
    if (b > std::numeric_limits<size_t>::max() - a) {
        return false;
    }
    out = a + b;
    return true;
}

// FNV-1a 64 detail
constexpr uint64_t FNV_OFFSET = 0xcbf29ce484222325ULL;

void set_err(std::string * err, const std::string & msg) {
    if (err != nullptr) {
        *err = msg;
    }
}

struct ggml_context_deleter {
    void operator()(ggml_context * ctx) const {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct ggml_buffer_deleter {
    void operator()(ggml_backend_buffer_t buf) const {
        if (buf != nullptr) {
            ggml_backend_buffer_free(buf);
        }
    }
};

} // namespace

// ---------------------------------------------------------------------------
// Allocation ID generator (thread-safe; failed batches burn consumed IDs)
// ---------------------------------------------------------------------------

xkv_allocation_id_generator::xkv_allocation_id_generator(uint64_t start_id, uint64_t max_id)
    : next_id_(start_id == 0 ? 1 : start_id), max_id_(max_id == 0 ? UINT64_MAX : max_id) {}

bool xkv_allocation_id_generator::allocate_id(uint64_t & id_out, std::string * err) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (next_id_ == 0) { // wrapped past UINT64_MAX
        set_err(err, "xkv_backend: allocation ID space exhausted (wrap)");
        return false;
    }
    if (next_id_ > max_id_) {
        set_err(err, "xkv_backend: allocation ID overflow (exceeded max_id)");
        return false;
    }
    id_out = next_id_;
    if (next_id_ == UINT64_MAX) {
        next_id_ = 0; // sentinel for exhausted; next call refuses
    } else {
        next_id_++;
    }
    return true;
}

uint64_t xkv_allocation_id_generator::current_id() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return next_id_;
}

uint64_t xkv_allocation_id_generator::max_id() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return max_id_;
}

void xkv_allocation_id_generator::reset(uint64_t start_id, uint64_t max_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    next_id_ = start_id == 0 ? 1 : start_id;
    max_id_ = max_id == 0 ? UINT64_MAX : max_id;
}

void xkv_allocation_id_generator::set_max_id(uint64_t max_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    max_id_ = max_id == 0 ? UINT64_MAX : max_id;
}

// ---------------------------------------------------------------------------
// Byte helpers
// ---------------------------------------------------------------------------

size_t xkv_nominal_bits_per_element(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:       return 32;
        case GGML_TYPE_F16:       return 16;
        case GGML_TYPE_Q8_0:      return 8;
        case GGML_TYPE_TURBO2_0:  return 2;
        case GGML_TYPE_TURBO3_0:  return 3;
        case GGML_TYPE_TURBO4_0:  return 4;
        default:                  return 0;
    }
}

size_t xkv_compute_logical_bytes(enum ggml_type type, uint64_t rows, uint64_t cols) {
    if (rows == 0 || cols == 0) {
        return 0;
    }
    const size_t bits = xkv_nominal_bits_per_element(type);
    if (bits == 0) {
        return 0;
    }
    // (2^64-1)^2 < 2^128, so the element product always fits; the bit product
    // is verified explicitly below.
    const __uint128_t elems = (__uint128_t) rows * (__uint128_t) cols;
    const __uint128_t total_bits = elems * (__uint128_t) bits;
    if (total_bits / (__uint128_t) bits != elems) {
        return 0; // unrepresentable shape
    }
    constexpr __uint128_t U128_MAX = ~(__uint128_t) 0;
    if (total_bits > U128_MAX - 7) {
        return 0;
    }
    const __uint128_t bytes = (total_bits + 7) / 8; // ceil to whole bytes
    if (bytes > (__uint128_t) std::numeric_limits<size_t>::max()) {
        return 0;
    }
    return (size_t) bytes;
}

uint64_t xkv_backend_checksum(const uint8_t * data, size_t size) {
    return xkv_backend_checksum_seeded(data, size, FNV_OFFSET);
}

uint64_t xkv_backend_checksum_seeded(const uint8_t * data, size_t size, uint64_t seed) {
    if (data == nullptr || size == 0) {
        return seed;
    }
    uint64_t h = seed;
    for (size_t i = 0; i < size; ++i) {
        h ^= data[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}
// ---------------------------------------------------------------------------
// Shared batch storage
// ---------------------------------------------------------------------------

xkv_backend_batch_storage::~xkv_backend_batch_storage() {
    if (buffer != nullptr) {
        ggml_backend_buffer_free(buffer);
        buffer = nullptr;
    }
    if (ctx != nullptr) {
        ggml_free(ctx);
        ctx = nullptr;
    }
}

// ---------------------------------------------------------------------------
// xkv_backend_allocation
// ---------------------------------------------------------------------------

xkv_backend_allocation::~xkv_backend_allocation() = default;

xkv_backend_allocation::xkv_backend_allocation(xkv_backend_allocation && o) noexcept
    : allocation_id_(o.allocation_id_),
      residency_(o.residency_),
      desc_(o.desc_),
      descriptor_fingerprint_(o.descriptor_fingerprint_),
      checksum_(o.checksum_),
      logical_bytes_(o.logical_bytes_),
      padded_bytes_(o.padded_bytes_),
      slice_bytes_(o.slice_bytes_),
      storage_(std::move(o.storage_)),
      tensor_(o.tensor_),
      owning_buft_(o.owning_buft_),
      owning_backend_name_(std::move(o.owning_backend_name_)),
      owning_dev_type_(o.owning_dev_type_),
      owning_backend_null_(o.owning_backend_null_),
      checksum_bound_(o.checksum_bound_),
      pack_provenance_(o.pack_provenance_),
      immutable_(o.immutable_) {
    o.allocation_id_ = 0;
    o.tensor_ = nullptr;
    o.owning_buft_ = nullptr;
    o.owning_dev_type_ = -1;
    o.owning_backend_null_ = true;
    o.immutable_ = false;
    o.logical_bytes_ = 0;
    o.padded_bytes_ = 0;
    o.slice_bytes_ = 0;
    o.descriptor_fingerprint_ = 0;
    o.checksum_ = 0;
    o.checksum_bound_ = true;
    o.pack_provenance_ = 0;
}

xkv_backend_allocation & xkv_backend_allocation::operator=(xkv_backend_allocation && o) noexcept {
    if (this != &o) {
        allocation_id_ = o.allocation_id_;
        residency_ = o.residency_;
        desc_ = o.desc_;
        descriptor_fingerprint_ = o.descriptor_fingerprint_;
        checksum_ = o.checksum_;
        logical_bytes_ = o.logical_bytes_;
        padded_bytes_ = o.padded_bytes_;
        slice_bytes_ = o.slice_bytes_;
        storage_ = std::move(o.storage_);
        tensor_ = o.tensor_;
        owning_buft_ = o.owning_buft_;
        owning_backend_name_ = std::move(o.owning_backend_name_);
        owning_dev_type_ = o.owning_dev_type_;
        owning_backend_null_ = o.owning_backend_null_;
        checksum_bound_ = o.checksum_bound_;
        pack_provenance_ = o.pack_provenance_;
        immutable_ = o.immutable_;
        o.allocation_id_ = 0;
        o.tensor_ = nullptr;
        o.owning_buft_ = nullptr;
        o.owning_dev_type_ = -1;
        o.owning_backend_null_ = true;
        o.immutable_ = false;
        o.logical_bytes_ = 0;
        o.padded_bytes_ = 0;
        o.slice_bytes_ = 0;
        o.descriptor_fingerprint_ = 0;
        o.checksum_ = 0;
        o.checksum_bound_ = true;
        o.pack_provenance_ = 0;
    }
    return *this;
}

size_t xkv_backend_allocation::get_buffer_bytes() const {
    return storage_ ? storage_->buffer_bytes : 0;
}

ggml_context * xkv_backend_allocation::get_ctx() const {
    return storage_ ? storage_->ctx : nullptr;
}

ggml_backend_buffer_t xkv_backend_allocation::get_buffer() const {
    return storage_ ? storage_->buffer : nullptr;
}

namespace detail {

// Validates that the caller backend matches the allocation's owning-backend
// record. Value comparison only (recorded name/device/buft) — no dangling
// raw backend pointers are ever stored or dereferenced.
// P0 safety fence: if async copies were queued (submitted==true) and the
// batch exits — failure return OR exception — without its counted
// synchronize, flush the backend queue BEFORE any buffer/host/store
// destruction below. Declare AFTER the protected objects so it destroys
// first; disarm (submitted=false) right after the counted synchronize.
// Honest semantics: ggml_backend_synchronize is void and cannot report
// failure, so this fence is an uncounted safety drain per the ggml
// same-backend in-order queue contract — not a counted batch sync.
struct submit_fence {
    ggml_backend_t backend = nullptr;
    const bool * submitted = nullptr;
    ~submit_fence() noexcept {
        if (backend != nullptr && submitted != nullptr && *submitted) {
            ggml_backend_synchronize(backend);
        }
    }
};
static bool validate_readback_backend(const xkv_backend_allocation & a, ggml_backend_t backend,
                                      size_t index, std::string * err) {
    auto fail = [&](const std::string & m) {
        if (err) {
            *err = "xkv_backend: readback stream " + std::to_string(index) + ": " + m;
        }
        return false;
    };
    if (!a.is_immutable()) {
        return fail("allocation not immutable/published");
    }
    if (a.get_tensor() == nullptr || a.get_buffer() == nullptr) {
        return fail("empty tensor/buffer");
    }
    if (a.get_residency() == GGML_XKV_RES_DEVICE_OWNED) {
        if (backend == nullptr) {
            return fail("device-owned readback requires the owning backend");
        }
        if (a.get_owning_buft() == nullptr || !ggml_backend_supports_buft(backend, a.get_owning_buft())) {
            return fail("backend does not support the owning buffer type");
        }
        if (a.get_owning_dev_type() >= 0) {
            ggml_backend_dev_t dev = ggml_backend_get_device(backend);
            if (dev != nullptr && (int) ggml_backend_dev_type(dev) != a.get_owning_dev_type()) {
                return fail("backend device mismatch with owning backend");
            }
        }
        if (!a.get_owning_backend_name().empty()) {
            const char * name = ggml_backend_name(backend);
            if (name != nullptr && a.get_owning_backend_name() != name) {
                return fail("backend mismatch with owning backend");
            }
        }
        return true;
    }
    // REFERENCE_HOST: a null backend uses the synchronous host path; a given
    // backend must still support the owning buffer type.
    if (backend != nullptr && a.get_owning_buft() != nullptr &&
        !ggml_backend_supports_buft(backend, a.get_owning_buft())) {
        return fail("backend does not support the owning buffer type");
    }
    return true;
}

static bool validate_new_stream_input(const xkv_backend_stream_input & in, size_t index,
                                      size_t & exact_out, std::string * err) {
    auto fail = [&](const std::string & m) {
        if (err) {
            *err = "xkv_backend: stream " + std::to_string(index) + ": " + m;
        }
        return false;
    };
    std::string derr;
    if (!in.desc.validate(&derr)) {
        return fail("invalid codec descriptor: " + derr);
    }
    if (!ggml_xkv_codec_supported(in.desc.type)) {
        return fail("unsupported codec type");
    }
    if (in.desc.logical_shape.rows == 0 || in.desc.logical_shape.cols == 0 ||
        in.desc.padded_shape.rows == 0 || in.desc.padded_shape.cols == 0) {
        return fail("zero-shape stream rejected");
    }
    uint64_t expected = 0;
    try {
        expected = encoded_matrix_bytes(in.desc);
    } catch (const std::exception & e) {
        return fail(std::string("encoded_matrix_bytes failed: ") + e.what());
    } catch (...) {
        return fail("encoded_matrix_bytes failed (unknown)");
    }
    if (expected == 0) {
        return fail("descriptor yields zero bytes");
    }
    if (in.data == nullptr && expected != 0) {
        return fail("null stream data");
    }
    if ((uint64_t) in.size != expected) {
        return fail("byte size mismatch: got " + std::to_string(in.size) +
                    " expected " + std::to_string(expected));
    }
    char exact_err[128] = {};
    const size_t exact = ggml_xkv_exact_bytes(
        in.desc.type,
        (int64_t) in.desc.padded_shape.cols,
        (int64_t) in.desc.logical_shape.rows,
        exact_err, sizeof(exact_err));
    if (exact == 0) {
        return fail(std::string("ggml_xkv_exact_bytes failed: ") + exact_err);
    }
    if ((uint64_t) exact != expected || (uint64_t) in.size != (uint64_t) exact) {
        return fail("ggml_xkv_exact_bytes does not match encoded_matrix_bytes");
    }
    exact_out = exact;
    return true;
}

// Existing shared handles must be immutable and match the batch's effective
// residency and buffer type; a foreign handle fails the whole batch.
static bool validate_shared_handle(const xkv_backend_stream_input & in, size_t index,
                                   ggml_xkv_residency effective_res,
                                   ggml_backend_buffer_type_t buft,
                                   std::string * err) {
    auto fail = [&](const std::string & m) {
        if (err) {
            *err = "xkv_backend: stream " + std::to_string(index) + ": shared handle " + m;
        }
        return false;
    };
    const auto & h = in.existing_handle;
    if (!h) {
        return fail("null");
    }
    if (!h->is_immutable()) {
        return fail("not immutable");
    }
    if (!(in.desc == h->get_desc())) {
        return fail("descriptor mismatch");
    }
    if (h->get_residency() != effective_res) {
        return fail("residency mismatch with batch");
    }
    if (h->get_owning_buft() != buft) {
        return fail("buffer-type mismatch with batch");
    }
    if (h->get_tensor() == nullptr || h->get_buffer() == nullptr) {
        return fail("empty tensor/buffer");
    }
    return true;
}

} // namespace detail

bool xkv_backend_allocation::readback(ggml_backend_t backend, std::vector<uint8_t> & out_bytes,
                                      std::string * err) const {
    // out_bytes assigned ONLY on success (failure-atomic).
    std::vector<uint8_t> tmp;
    try {
        if (!detail::validate_readback_backend(*this, backend, 0, err)) {
            return false;
        }
        if (padded_bytes_ == 0) {
            out_bytes.clear();
            return true;
        }
        tmp.resize(padded_bytes_);
        if (backend != nullptr) {
            ggml_backend_tensor_get_async(backend, tensor_, tmp.data(), 0, padded_bytes_);
            ggml_backend_synchronize(backend);
        } else {
            ggml_backend_tensor_get(tensor_, tmp.data(), 0, padded_bytes_);
        }
    } catch (const std::bad_alloc &) {
        set_err(err, "xkv_backend: readback out of memory");
        return false;
    } catch (const std::exception & e) {
        set_err(err, std::string("xkv_backend: readback failed: ") + e.what());
        return false;
    } catch (...) {
        set_err(err, "xkv_backend: readback failed (unknown)");
        return false;
    }
    const uint64_t got = xkv_backend_checksum(tmp.data(), tmp.size());
    // Device-packed streams carry provenance instead of a host checksum
    // (no D2H allowed); only bound streams compare.
    if (checksum_bound_ && got != checksum_) {
        set_err(err, "xkv_backend: readback checksum mismatch");
        return false;
    }
    char exact_err[128] = {};
    const size_t exact = ggml_xkv_exact_bytes(
        desc_.type,
        (int64_t) desc_.padded_shape.cols,
        (int64_t) desc_.logical_shape.rows,
        exact_err, sizeof(exact_err));
    if (exact == 0 || exact != padded_bytes_ || exact != tmp.size()) {
        set_err(err, std::string("xkv_backend: readback exact-bytes mismatch: ") + exact_err);
        return false;
    }
    out_bytes = std::move(tmp);
    return true;
}

// ---------------------------------------------------------------------------
// Batch result host-release commit (UAF-safe, exception-safe)
// ---------------------------------------------------------------------------

bool xkv_backend_batch_result::commit_host_release(std::string * err) {
    try {
        if (!success_) {
            set_err(err, "xkv_backend: cannot commit host release on failed batch");
            return false;
        }
        if (committed_) {
            return true; // idempotent
        }
        // Phase 1: pin every live source (no caller mutation yet). Expired
        // sources are skipped — never dereferenced, so no UAF when the result
        // outlives a source.
        std::vector<std::shared_ptr<std::vector<uint8_t>>> pinned;
        pinned.reserve(host_weak_refs_.size());
        for (auto & w : host_weak_refs_) {
            if (auto s = w.lock()) {
                pinned.push_back(std::move(s));
            }
        }
        // Phase 2: release. vector::clear() is noexcept: cannot throw, so it
        // cannot partially release. (No shrink_to_fit: it can throw.)
        for (auto & s : pinned) {
            s->clear();
        }
        // Phase 3: caller callbacks. Each is attempted even if an earlier one
        // throws; the first error is reported and committed stays false.
        std::string first_err;
        for (auto & fn : release_fns_) {
            if (!fn) {
                continue;
            }
            try {
                fn();
            } catch (const std::exception & e) {
                if (first_err.empty()) {
                    first_err = e.what();
                }
            } catch (...) {
                if (first_err.empty()) {
                    first_err = "unknown exception";
                }
            }
        }
        if (!first_err.empty()) {
            set_err(err, "xkv_backend: commit release callback failed: " + first_err);
            return false;
        }
        committed_ = true;
        return true;
    } catch (const std::bad_alloc &) {
        set_err(err, "xkv_backend: commit out of memory (nothing released)");
        return false;
    } catch (const std::exception & e) {
        set_err(err, std::string("xkv_backend: commit failed: ") + e.what());
        return false;
    } catch (...) {
        set_err(err, "xkv_backend: commit failed (unknown)");
        return false;
    }
}

// ---------------------------------------------------------------------------
// Batch builder
// ---------------------------------------------------------------------------

xkv_backend_batch_builder::xkv_backend_batch_builder(xkv_allocation_id_generator * id_gen)
    : id_gen_(id_gen != nullptr ? id_gen : &default_id_gen_) {}

void xkv_backend_batch_builder::add_stream(const codec_desc & desc, const uint8_t * data, size_t size,
                                           std::function<void()> release_fn) {
    xkv_backend_stream_input in;
    in.desc = desc;
    in.data = data;
    in.size = size;
    in.release_fn = std::move(release_fn);
    streams_.push_back(std::move(in));
}

void xkv_backend_batch_builder::add_stream(const codec_desc & desc, std::vector<uint8_t> * source_host_vector) {
    xkv_backend_stream_input in;
    in.desc = desc;
    if (source_host_vector != nullptr) {
        in.data = source_host_vector->data();
        in.size = source_host_vector->size();
        in.source_host_vector = source_host_vector;
    }
    streams_.push_back(std::move(in));
}

void xkv_backend_batch_builder::add_stream(const codec_desc & desc,
                                           const std::shared_ptr<std::vector<uint8_t>> & source_host_vector) {
    if (!source_host_vector) {
        return;
    }
    xkv_backend_stream_input in;
    in.desc = desc;
    in.data = source_host_vector->data();
    in.size = source_host_vector->size();
    in.shared_host_vector = source_host_vector;
    streams_.push_back(std::move(in));
}

void xkv_backend_batch_builder::add_stream(encoded_matrix * em) {
    if (em == nullptr) {
        return;
    }
    xkv_backend_stream_input in;
    in.desc = em->desc;
    in.data = em->bytes.data();
    in.size = em->bytes.size();
    in.source_host_vector = &em->bytes;
    streams_.push_back(std::move(in));
}

void xkv_backend_batch_builder::add_shared_handle(std::shared_ptr<xkv_backend_allocation> handle) {
    if (!handle) {
        return;
    }
    xkv_backend_stream_input in;
    in.desc = handle->get_desc();
    in.existing_handle = std::move(handle);
    streams_.push_back(std::move(in));
}

bool xkv_backend_batch_builder::build(
    ggml_backend_t backend,
    ggml_backend_buffer_type_t buft,
    const xkv_backend_batch_config & config,
    xkv_backend_batch_result & result_out,
    std::string * err,
    const xkv_backend_store_reservation * reservation) {
    // Failure-atomic: result_out is assigned ONLY on success; on ANY failure
    // it is left bit-identical to entry.
    try {
        xkv_backend_batch_result local;
        std::string local_err;
        if (!build_impl(backend, buft, config, local, &local_err, reservation)) {
            set_err(err, local_err);
            return false;
        }
        result_out = std::move(local); // move-assign of vectors/shared_ptrs is noexcept
        return true;
    } catch (const std::bad_alloc &) {
        set_err(err, "xkv_backend: build out of memory (outputs unchanged)");
        return false;
    } catch (const std::exception & e) {
        set_err(err, std::string("xkv_backend: build failed: ") + e.what() + " (outputs unchanged)");
        return false;
    } catch (...) {
        set_err(err, "xkv_backend: build failed (unknown; outputs unchanged)");
        return false;
    }
}

bool xkv_backend_batch_builder::build_impl(
    ggml_backend_t backend,
    ggml_backend_buffer_type_t buft,
    const xkv_backend_batch_config & config,
    xkv_backend_batch_result & out_local,
    std::string * err,
    const xkv_backend_store_reservation * reservation) {
    // NOTE: out_local is a function-local of build(); partial fills on the
    // failure paths below are discarded, so result_out stays untouched.
    if (buft == nullptr) {
        set_err(err, "xkv_backend: null buffer type");
        return false;
    }
    if (streams_.empty()) {
        set_err(err, "xkv_backend: empty batch");
        return false;
    }

    // Pre-flight owner identity validation BEFORE any allocation/upload/ID consumption:
    // if caller provided a backend_owner, its raw pointer MUST match backend exactly.
    if (config.backend_owner && config.backend_owner.get() != backend) {
        set_err(err, "xkv_backend: backend != config.backend_owner.get()");
        return false;
    }

    // Refresh shared-ownership views from the live vectors (shared ownership
    // keeps them alive, so re-reading here closes the add-to-build
    // reallocation window for the shared path; raw borrowed pointers stay
    // caller-responsibility per the header contract).
    for (auto & in : streams_) {
        if (in.shared_host_vector) {
            in.data = in.shared_host_vector->data();
            in.size = in.shared_host_vector->size();
        }
    }

    // ---- Phase 0: validate every new stream before allocating (zero mutation) ----
    std::vector<size_t> exact_bytes(streams_.size(), 0);
    for (size_t i = 0; i < streams_.size(); ++i) {
        if (streams_[i].existing_handle != nullptr) {
            continue;
        }
        if (!detail::validate_new_stream_input(streams_[i], i, exact_bytes[i], err)) {
            return false;
        }
    }

    // ---- Phase 1: resolve residency for the batch ----
    const bool buft_is_host = ggml_backend_buft_is_host(buft);
    ggml_xkv_residency effective_res = config.residency;
    if (!config.enforce_residency && buft_is_host && effective_res == GGML_XKV_RES_DEVICE_OWNED) {
        // Truthful adaptation: host buft cannot be device-owned.
        effective_res = GGML_XKV_RES_REFERENCE_HOST;
    }

    for (size_t i = 0; i < streams_.size(); ++i) {
        const auto & in = streams_[i];
        if (in.existing_handle != nullptr) {
            if (!detail::validate_shared_handle(in, i, effective_res, buft, err)) {
                return false;
            }
            continue;
        }
        if (!ggml_xkv_residency_supported(in.desc.type, effective_res)) {
            if (err) {
                *err = "xkv_backend: stream " + std::to_string(i) + ": residency unsupported for codec";
            }
            return false;
        }
        if (config.enforce_residency) {
            if (effective_res == GGML_XKV_RES_DEVICE_OWNED && buft_is_host) {
                if (err) {
                    *err = "xkv_backend: stream " + std::to_string(i) + ": device-owned residency on host buft";
                }
                return false;
            }
            if (effective_res == GGML_XKV_RES_REFERENCE_HOST && !buft_is_host) {
                if (err) {
                    *err = "xkv_backend: stream " + std::to_string(i) + ": host residency on device buft";
                }
                return false;
            }
        }
        if (effective_res == GGML_XKV_RES_DEVICE_OWNED && backend != nullptr) {
            if (!ggml_backend_supports_buft(backend, buft)) {
                if (err) {
                    *err = "xkv_backend: stream " + std::to_string(i) + ": backend does not support buft";
                }
                return false;
            }
        }
        if (effective_res == GGML_XKV_RES_DEVICE_OWNED && buft_is_host) {
            if (err) {
                *err = "xkv_backend: stream " + std::to_string(i) + ": device-owned on host buft refused";
            }
            return false;
        }
    }

    // ---- Phase 2: create one shared store (single ctx, single buffer) ----
    // Count new streams first; an all-shared batch needs no store at all.
    size_t n_new = 0;
    for (const auto & in : streams_) {
        if (in.existing_handle == nullptr) {
            n_new++;
        }
    }

    std::shared_ptr<xkv_backend_batch_storage> storage;
    // new_tensors[i] parallels streams_[i] (null for shared handles).
    std::vector<ggml_tensor *> new_tensors(streams_.size(), nullptr);

    if (n_new > 0) {
        uint64_t overhead = 0;
        {
            const uint64_t oh = (uint64_t) ggml_tensor_overhead();
            uint64_t tensors_mem = 0;
            if (!mul_ok_u64(n_new, oh, tensors_mem)) {
                set_err(err, "xkv_backend: context size overflow");
                return false;
            }
            if (!add_ok_u64(tensors_mem, 64, overhead)) {
                set_err(err, "xkv_backend: context size overflow");
                return false;
            }
            if (overhead > (uint64_t) std::numeric_limits<size_t>::max()) {
                set_err(err, "xkv_backend: context size unrepresentable");
                return false;
            }
        }
        std::unique_ptr<ggml_context, ggml_context_deleter> ctx_guard;
        {
            ggml_init_params ip = {};
            ip.mem_size = (size_t) overhead;
            ip.no_alloc = true;
            ggml_context * ctx = ggml_init(ip);
            if (ctx == nullptr) {
                set_err(err, "xkv_backend: ggml_init failed");
                return false;
            }
            ctx_guard.reset(ctx);
        }
        for (size_t i = 0; i < streams_.size(); ++i) {
            const auto & in = streams_[i];
            if (in.existing_handle != nullptr) {
                continue;
            }
            const int64_t ne0 = (int64_t) in.desc.padded_shape.cols;
            const int64_t ne1 = (int64_t) in.desc.logical_shape.rows;
            ggml_tensor * t = ggml_new_tensor_2d(ctx_guard.get(), in.desc.type, ne0, ne1);
            if (t == nullptr) {
                set_err(err, "xkv_backend: stream " + std::to_string(i) + ": tensor create failed");
                return false;
            }
            if ((size_t) ggml_nbytes(t) != exact_bytes[i]) {
                set_err(err, "xkv_backend: stream " + std::to_string(i) + ": ggml_nbytes != exact bytes");
                return false;
            }
            new_tensors[i] = t;
        }
        {
        // Preflight the exact backend buffer/alignment bytes BEFORE allocating:
        // sum per-tensor buft alloc sizes with checked arithmetic and enforce
        // the store reservation/cap. No IDs consumed yet, nothing published.
        if (reservation != nullptr) {
            uint64_t estimate = 0;
            for (size_t ei = 0; ei < streams_.size(); ++ei) {
                if (streams_[ei].existing_handle != nullptr) {
                    continue;
                }
                const size_t es = ggml_backend_buft_get_alloc_size(buft, new_tensors[ei]);
                if (es == 0) {
                    set_err(err, "xkv_backend: preflight slice query failed");
                    return false;
                }
                uint64_t next_est = 0;
                if (!add_ok_u64(estimate, (uint64_t) es, next_est)) {
                    set_err(err, "xkv_backend: preflight estimate overflow");
                    return false;
                }
                estimate = next_est;
            }
            if (estimate > reservation->reserved_bytes) {
                set_err(err, "xkv_backend: preflight estimate " + std::to_string(estimate) +
                             " exceeds reserved " + std::to_string(reservation->reserved_bytes));
                return false;
            }
            if (reservation->cap_bytes != 0 && estimate > reservation->cap_bytes) {
                set_err(err, "xkv_backend: preflight estimate " + std::to_string(estimate) +
                             " exceeds cap " + std::to_string(reservation->cap_bytes));
                return false;
            }
        }
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_guard.get(), buft);
            if (buf == nullptr) {
                set_err(err, "xkv_backend: shared backend buffer alloc failed");
                return false;
            }
            std::unique_ptr<ggml_backend_buffer, ggml_buffer_deleter> buf_guard(buf);
            storage = std::shared_ptr<xkv_backend_batch_storage>(new xkv_backend_batch_storage());
            storage->ctx = ctx_guard.release();
            storage->buffer = buf_guard.release();
            storage->buffer_bytes = ggml_backend_buffer_get_size(storage->buffer);
            // Enforce the reservation against the ACTUAL allocation: on excess,
            // the shared_ptr locals destroy the off-side store on return, with
            // no attach and no host release (nothing published, no callbacks).
            if (reservation != nullptr) {
                const uint64_t actual = (uint64_t) storage->buffer_bytes;
                if (actual > reservation->reserved_bytes) {
                    set_err(err, "xkv_backend: actual buffer " + std::to_string(actual) +
                                 " exceeds reserved " + std::to_string(reservation->reserved_bytes));
                    return false;
                }
                if (reservation->cap_bytes != 0 && actual > reservation->cap_bytes) {
                    set_err(err, "xkv_backend: actual buffer " + std::to_string(actual) +
                                 " exceeds cap " + std::to_string(reservation->cap_bytes));
                    return false;
                }
            }
        }
    }

    // ---- Phase 3: assemble off-side allocations (IDs burn on later failure) ----
    std::vector<std::shared_ptr<xkv_backend_allocation>> created;
    created.reserve(streams_.size());
    xkv_backend_batch_stats stats;

    struct pending_upload {
        std::shared_ptr<xkv_backend_allocation> alloc;
        const uint8_t * data = nullptr;
        size_t size = 0;
        size_t index = 0;
    };
    std::vector<pending_upload> pending;
    pending.reserve(n_new);

    // Owning-backend record shared by every new allocation of this batch.
    std::string owner_name;
    int owner_dev_type = -1;
    if (backend != nullptr) {
        const char * bname = ggml_backend_name(backend);
        if (bname != nullptr) {
            owner_name = bname;
        }
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        if (dev == nullptr) {
            dev = ggml_backend_buft_get_device(buft);
        }
        if (dev != nullptr) {
            owner_dev_type = (int) ggml_backend_dev_type(dev);
        }
    } else {
        ggml_backend_dev_t bdev = ggml_backend_buft_get_device(buft);
        if (bdev != nullptr) {
            owner_dev_type = (int) ggml_backend_dev_type(bdev);
        }
    }

    for (size_t i = 0; i < streams_.size(); ++i) {
        const auto & in = streams_[i];
        if (in.existing_handle != nullptr) {
            created.push_back(in.existing_handle);
            stats.shared_streams++;
            continue;
        }
        uint64_t alloc_id = 0;
        // Injected allocation failure. Streams before this index already
        // consumed their IDs, which burn (never reused) per the documented
        // generator semantics.
        if ((int) i == config.inject_alloc_fail_at) {
            set_err(err, "xkv_backend: injected allocation failure at stream " + std::to_string(i));
            return false;
        }
        if (!id_gen_->allocate_id(alloc_id, err)) {
            if (err && err->empty()) {
                *err = "xkv_backend: allocation ID overflow at stream " + std::to_string(i);
            } else if (err) {
                *err = "xkv_backend: stream " + std::to_string(i) + ": " + *err;
            }
            return false;
        }

        auto alloc = std::shared_ptr<xkv_backend_allocation>(new xkv_backend_allocation());
        alloc->allocation_id_ = alloc_id;
        alloc->residency_ = effective_res;
        alloc->desc_ = in.desc;
        alloc->descriptor_fingerprint_ = in.desc.fingerprint();
        alloc->checksum_ = xkv_backend_checksum(in.data, in.size);
        alloc->logical_bytes_ = xkv_compute_logical_bytes(
            in.desc.type, in.desc.logical_shape.rows, in.desc.logical_shape.cols);
        alloc->padded_bytes_ = exact_bytes[i];
        if (alloc->logical_bytes_ == 0) {
            set_err(err, "xkv_backend: stream " + std::to_string(i) + ": unsupported type for logical bytes");
            return false;
        }
        alloc->storage_ = storage;
        alloc->tensor_ = new_tensors[i];
        const size_t slice = ggml_backend_buft_get_alloc_size(buft, alloc->tensor_);
        if (slice == 0) {
            set_err(err, "xkv_backend: stream " + std::to_string(i) + ": slice size query failed");
            return false;
        }
        alloc->slice_bytes_ = slice;
        alloc->owning_buft_ = buft;
        alloc->owning_backend_name_ = owner_name;
        alloc->owning_dev_type_ = owner_dev_type;
        alloc->owning_backend_null_ = (backend == nullptr);

        pending.push_back(pending_upload{alloc, in.data, in.size, i});
        created.push_back(std::move(alloc));
        stats.uploaded_streams++;
        size_t next_uploaded = 0;
        if (!add_ok_size(stats.uploaded_bytes, exact_bytes[i], next_uploaded)) {
            set_err(err, "xkv_backend: uploaded-bytes counter overflow");
            return false;
        }
        stats.uploaded_bytes = next_uploaded;
    }

    // ---- Phase 4+5: queue all uploads, then verification downloads, then ONE sync ----
    // Same-backend queue order is in-order: every set_async precedes every
    // get_async issued on the same backend handle, so a single trailing
    // synchronize covers the whole batch (upload + verify). sync_count is
    // reported truthfully: 1 when a non-null backend moved bytes, else 0.
    // Verification downloads land in the caller workspace (preflighted
    // combined host peak below), never in unaccounted internal vectors.
    const bool verifying = (config.verify_readback || config.verify_checksum) && !pending.empty();
    // P0: track async submission; the fence (declared after all protected
    // objects) flushes the queue if we exit without the counted sync below.
    bool submitted = false;
    detail::submit_fence fence{backend, &submitted};
    size_t verify_total = 0;
    if (verifying) {
        for (const auto & p : pending) {
            size_t next = 0;
            if (!add_ok_size(verify_total, p.size, next)) {
                set_err(err, "xkv_backend: verify workspace size overflow");
                return false;
            }
            verify_total = next;
        }
        if (config.verify_workspace == nullptr ||
            config.verify_workspace_bytes < verify_total) {
            set_err(err, "xkv_backend: verify requires caller workspace of " +
                         std::to_string(verify_total) + " bytes, have " +
                         std::to_string(config.verify_workspace_bytes));
            return false;
        }
    }
    for (auto & p : pending) {
        if ((int) p.index == config.inject_upload_fail_at) {
            set_err(err, "xkv_backend: injected upload failure at stream " + std::to_string(p.index));
            return false;
        }
        if (backend != nullptr) {
            ggml_backend_tensor_set_async(backend, p.alloc->tensor_, p.data, 0, p.size);
            submitted = true;
        } else {
            ggml_backend_tensor_set(p.alloc->tensor_, p.data, 0, p.size);
        }
        if ((int) p.index == config.inject_post_submit_fail_at) {
            set_err(err, "xkv_backend: injected post-submit failure at stream " + std::to_string(p.index));
            return false; // fence flushes the queued sets before destruction
        }
    }
    struct verify_slice {
        uint8_t * dst = nullptr; // carved from the caller workspace
        size_t size = 0;
        size_t index = 0;
        const uint8_t * src = nullptr;
        uint64_t checksum = 0;
    };
    std::vector<verify_slice> vslices;
    if (verifying) {
        vslices.reserve(pending.size());
        size_t off = 0;
        for (auto & p : pending) {
            if ((int) p.index == config.inject_readback_fail_at) {
                set_err(err, "xkv_backend: injected readback failure at stream " + std::to_string(p.index));
                return false;
            }
            // Bounded by construction: off + p.size <= verify_total <= workspace bytes.
            uint8_t * dst = config.verify_workspace + off;
            vslices.push_back(verify_slice{dst, p.size, p.index, p.data, p.alloc->checksum_});
            off += p.size;
            if (backend != nullptr) {
                ggml_backend_tensor_get_async(backend, p.alloc->tensor_, dst, 0, p.size);
                submitted = true;
            } else {
                ggml_backend_tensor_get(p.alloc->tensor_, dst, 0, p.size);
            }
        }
    }
    if (backend != nullptr && !pending.empty()) {
        ggml_backend_synchronize(backend);
        stats.sync_count++;
        submitted = false; // queue flushed: disarm the fence, no double sync
    }
    if (verifying) {
        for (const auto & v : vslices) {
            if ((int) v.index == config.inject_checksum_fail_at) {
                set_err(err, "xkv_backend: injected checksum failure at stream " + std::to_string(v.index));
                return false;
            }
            const uint64_t got = xkv_backend_checksum(v.dst, v.size);
            if (got != v.checksum) {
                set_err(err, "xkv_backend: checksum mismatch at stream " + std::to_string(v.index));
                return false;
            }
            if (config.verify_readback && memcmp(v.dst, v.src, v.size) != 0) {
                set_err(err, "xkv_backend: readback mismatch at stream " + std::to_string(v.index));
                return false;
            }
        }
    }

    // ---- Phase 6: atomically publish into out_local (discarded on failure above) ----
    out_local.success_ = true;
    out_local.stats = stats;
    // If caller provided a real backend_owner, verify identity and copy the owner.
    // NEVER manufacture ownership with a no-op deleter from a raw pointer!
    if (config.backend_owner) {
        if (config.backend_owner.get() != backend) {
            set_err(err, "xkv_backend: backend != config.backend_owner.get()");
            return false;
        }
        out_local.executor = config.backend_owner;
        if (storage) {
            storage->backend_executor = config.backend_owner;
        }
    }
    out_local.handles = std::move(created);
    for (auto & h : out_local.handles) {
        h->mark_immutable();
    }
    // Register host releases: only device-owned batches release, only through
    // explicit caller commit, and never via raw caller pointers.
    if (effective_res == GGML_XKV_RES_DEVICE_OWNED) {
        for (size_t i = 0; i < streams_.size(); ++i) {
            const auto & in = streams_[i];
            if (in.existing_handle != nullptr) {
                continue;
            }
            if (in.shared_host_vector) {
                out_local.host_weak_refs_.push_back(in.shared_host_vector);
            } else if (in.release_fn) {
                out_local.release_fns_.push_back(in.release_fn);
            }
            // Borrowed raw vectors / encoded_matrix bytes: never captured;
            // the caller clears them manually after a successful commit.
            // Every new device-owned stream is caller-clearable after success;
            // raw borrowings are listed here too (never captured or auto-touched).
            out_local.releasable_stream_indices.push_back(i);
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Readback batch (failure-atomic: out untouched on failure)
// ---------------------------------------------------------------------------

static bool readback_batch_impl(
    ggml_backend_t backend,
    const std::vector<std::shared_ptr<const xkv_backend_allocation>> & allocations,
    const xkv_backend_batch_config & config,
    xkv_backend_readback_result & out_local,
    std::string * err) {
    if (allocations.empty()) {
        set_err(err, "xkv_backend: readback empty batch");
        return false;
    }
    std::vector<std::vector<uint8_t>> bufs(allocations.size());
    // P0: fence flushes queued gets if we exit (return or throw) before the
    // counted synchronize below. Declared after bufs so it destroys first:
    // host destinations are still alive when the fence drains the queue.
    bool submitted = false;
    detail::submit_fence fence{backend, &submitted};
    for (size_t i = 0; i < allocations.size(); ++i) {
        const auto & a = allocations[i];
        if (!a) {
            set_err(err, "xkv_backend: readback null allocation at " + std::to_string(i));
            return false;
        }
        if ((int) i == config.inject_readback_fail_at) {
            set_err(err, "xkv_backend: injected readback failure at stream " + std::to_string(i));
            return false;
        }
        if (!detail::validate_readback_backend(*a, backend, i, err)) {
            return false;
        }
        bufs[i].resize(a->get_padded_bytes());
        if (a->get_padded_bytes() == 0) {
            continue;
        }
        if (backend != nullptr) {
            ggml_backend_tensor_get_async(backend, a->get_tensor(), bufs[i].data(), 0, bufs[i].size());
            submitted = true;
        } else {
            ggml_backend_tensor_get(a->get_tensor(), bufs[i].data(), 0, bufs[i].size());
        }
    }
    xkv_backend_batch_stats stats;
    if (backend != nullptr) {
        ggml_backend_synchronize(backend);
        stats.sync_count++;
        submitted = false; // flushed: disarm the fence
    }
    for (size_t i = 0; i < allocations.size(); ++i) {
        const auto & a = allocations[i];
        if ((int) i == config.inject_checksum_fail_at) {
            set_err(err, "xkv_backend: injected checksum failure at stream " + std::to_string(i));
            return false;
        }
        const uint64_t got = xkv_backend_checksum(bufs[i].data(), bufs[i].size());
        if (a->is_checksum_bound() && got != a->get_checksum()) {
            set_err(err, "xkv_backend: readback checksum mismatch at stream " + std::to_string(i));
            return false;
        }
        char exact_err[128] = {};
        const size_t exact = ggml_xkv_exact_bytes(
            a->get_desc().type,
            (int64_t) a->get_desc().padded_shape.cols,
            (int64_t) a->get_desc().logical_shape.rows,
            exact_err, sizeof(exact_err));
        if (exact == 0 || exact != a->get_padded_bytes() || exact != bufs[i].size()) {
            set_err(err, std::string("xkv_backend: readback descriptor mismatch at ") +
                         std::to_string(i) + ": " + exact_err);
            return false;
        }
    }
    out_local.stream_bytes = std::move(bufs);
    out_local.stats = stats;
    return true;
}

static bool readback_batch_entry(
    ggml_backend_t backend,
    const std::vector<std::shared_ptr<const xkv_backend_allocation>> & allocations,
    const xkv_backend_batch_config & config,
    xkv_backend_readback_result & out,
    std::string * err) {
    try {
        xkv_backend_readback_result local;
        std::string local_err;
        if (!readback_batch_impl(backend, allocations, config, local, &local_err)) {
            set_err(err, local_err);
            return false;
        }
        out = std::move(local);
        return true;
    } catch (const std::bad_alloc &) {
        set_err(err, "xkv_backend: readback out of memory (outputs unchanged)");
        return false;
    } catch (const std::exception & e) {
        set_err(err, std::string("xkv_backend: readback failed: ") + e.what() + " (outputs unchanged)");
        return false;
    } catch (...) {
        set_err(err, "xkv_backend: readback failed (unknown; outputs unchanged)");
        return false;
    }
}

bool xkv_backend_readback_batch(
    ggml_backend_t backend,
    const std::vector<std::shared_ptr<const xkv_backend_allocation>> & allocations,
    const xkv_backend_batch_config & config,
    xkv_backend_readback_result & out,
    std::string * err) {
    return readback_batch_entry(backend, allocations, config, out, err);
}

bool xkv_backend_readback_batch(
    ggml_backend_t backend,
    const std::vector<std::shared_ptr<xkv_backend_allocation>> & allocations,
    const xkv_backend_batch_config & config,
    xkv_backend_readback_result & out,
    std::string * err) {
    try {
        std::vector<std::shared_ptr<const xkv_backend_allocation>> const_view;
        const_view.reserve(allocations.size());
        for (const auto & a : allocations) {
            const_view.push_back(a);
        }
        return readback_batch_entry(backend, const_view, config, out, err);
    } catch (const std::bad_alloc &) {
        set_err(err, "xkv_backend: readback out of memory (outputs unchanged)");
        return false;
    } catch (const std::exception & e) {
        set_err(err, std::string("xkv_backend: readback failed: ") + e.what() + " (outputs unchanged)");
        return false;
    } catch (...) {
        set_err(err, "xkv_backend: readback failed (unknown; outputs unchanged)");
        return false;
    }
}

// ---------------------------------------------------------------------------
// Accounting: dedup shared allocations by ID, shared buffers by identity.
// Never throws: bad_alloc sets overflow=true and returns partial sums.
// ---------------------------------------------------------------------------

static xkv_backend_accounting accounting_from_const(
    const std::vector<std::shared_ptr<const xkv_backend_allocation>> & allocations) noexcept {
    xkv_backend_accounting acc;
    try {
        std::vector<uint64_t> seen_ids;
        seen_ids.reserve(allocations.size());
        std::vector<ggml_backend_buffer_t> seen_bufs;
        seen_bufs.reserve(allocations.size());
        auto checked_add = [&](size_t & field, size_t v) {
            size_t next = 0;
            if (!add_ok_size(field, v, next)) {
                field = std::numeric_limits<size_t>::max();
                acc.overflow = true;
                return;
            }
            field = next;
        };
        for (const auto & a : allocations) {
            if (!a) {
                continue;
            }
            const uint64_t id = a->get_allocation_id();
            bool dup = false;
            for (uint64_t s : seen_ids) {
                if (s == id) {
                    dup = true;
                    break;
                }
            }
            if (dup) {
                continue;
            }
            seen_ids.push_back(id);
            if (acc.count < std::numeric_limits<size_t>::max()) {
                acc.count++;
            } else {
                acc.overflow = true;
            }
            checked_add(acc.logical_bytes, a->get_logical_bytes());
            checked_add(acc.padded_bytes, a->get_padded_bytes());
            checked_add(acc.device_bytes, a->get_device_bytes());
            checked_add(acc.host_bytes, a->get_host_bytes());
            ggml_backend_buffer_t buf = a->get_buffer();
            if (buf == nullptr) {
                checked_add(acc.actual_bytes, a->get_actual_bytes());
                continue;
            }
            bool buf_dup = false;
            for (auto b : seen_bufs) {
                if (b == buf) {
                    buf_dup = true;
                    break;
                }
            }
            if (buf_dup) {
                continue;
            }
            seen_bufs.push_back(buf);
            checked_add(acc.actual_bytes, ggml_backend_buffer_get_size(buf));
        }
    } catch (...) {
        acc.overflow = true; // partial sums retained, flagged
    }
    return acc;
}

xkv_backend_accounting xkv_backend_calculate_accounting(
    const std::vector<std::shared_ptr<const xkv_backend_allocation>> & allocations) {
    return accounting_from_const(allocations);
}

xkv_backend_accounting xkv_backend_calculate_accounting(
    const std::vector<std::shared_ptr<xkv_backend_allocation>> & allocations) {
    try {
        std::vector<std::shared_ptr<const xkv_backend_allocation>> const_view;
        const_view.reserve(allocations.size());
        for (const auto & a : allocations) {
            const_view.push_back(a);
        }
        return accounting_from_const(const_view);
    } catch (...) {
        xkv_backend_accounting acc;
        acc.overflow = true;
        return acc;
    }
}

// ---------------------------------------------------------------------------
// Capability: consults the ACTUAL backend / buft / device, never a global flag

// ---------------------------------------------------------------------------
// Peak accounting: live plus retired-but-pinned (COW). combined dedups
// shared IDs/buffers across both lists; combined.actual_bytes is the peak.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Cross-tensor pack (device-native A row gather into a new immutable COW
// destination; the pinned source is never mutated in place)
// ---------------------------------------------------------------------------

static uint64_t pack_provenance_for(const codec_desc & dst_desc, uint64_t src_id,
                                    const uint32_t * rows, size_t n_rows) {
    uint64_t h = 14695981039346656037ULL;
    auto mix_u64 = [&](uint64_t v) {
        for (int b = 0; b < 8; ++b) {
            h ^= (uint8_t) (v >> (b * 8));
            h *= 0x100000001b3ULL;
        }
    };
    mix_u64(dst_desc.fingerprint());
    mix_u64(src_id);
    mix_u64((uint64_t) n_rows);
    for (size_t i = 0; i < n_rows; ++i) {
        mix_u64((uint64_t) rows[i]);
    }
    return h == 0 ? 1 : h;
}

bool xkv_backend_batch_builder::pack_impl(
    ggml_backend_t backend,
    ggml_backend_buffer_type_t buft,
    const std::vector<xkv_backend_pack_request> & requests,
    const xkv_backend_batch_config & config,
    xkv_allocation_id_generator & id_gen,
    xkv_backend_pack_result & out_local,
    std::string * err,
    const xkv_backend_store_reservation * reservation) {
    if (buft == nullptr) {
        set_err(err, "xkv_backend pack: null buffer type");
        return false;
    }
    if (requests.empty()) {
        set_err(err, "xkv_backend pack: empty batch");
        return false;
    }

    // ---- Phase 0: validate everything (zero mutation) ----
    struct validated_pack {
        const xkv_backend_pack_request * req = nullptr;
        size_t exact = 0;
        uint64_t src_rows = 0;
        size_t row_bytes = 0;
        size_t src_nbytes = 0;
        size_t src_nb1 = 0;
        int64_t padded_cols = 0;
    };
    std::vector<validated_pack> vs;
    vs.reserve(requests.size());
    ggml_xkv_residency res0 = GGML_XKV_RES_REFERENCE_HOST;
    bool have_res = false;
    uint64_t total_copies = 0;

    for (size_t i = 0; i < requests.size(); ++i) {
        const auto & rq = requests[i];
        auto fail = [&](const std::string & m) -> bool {
            if (err) {
                *err = "xkv_backend pack request " + std::to_string(i) + ": " + m;
            }
            return false;
        };
        const auto & src = rq.src;
        if (!src) {
            return fail("null source handle");
        }
        if (!src->is_immutable()) {
            return fail("source not immutable");
        }
        ggml_tensor * src_t = src->get_tensor();
        if (src_t == nullptr || src->get_buffer() == nullptr) {
            return fail("source has empty tensor/buffer");
        }
        const codec_desc & sd = src->get_desc();
        if (sd.orient != orientation::token_major) {
            return fail("only token-major A streams pack (B stays shared; landmarks attach exact)");
        }
        if (src->get_owning_buft() != buft) {
            return fail("source buffer-type mismatch with pack batch");
        }
        if (!have_res) {
            res0 = src->get_residency();
            have_res = true;
        } else if (src->get_residency() != res0) {
            return fail("mixed residency in pack batch");
        }
        char exact_err[128] = {};
        const size_t src_exact = ggml_xkv_exact_bytes(sd.type, (int64_t) sd.padded_shape.cols,
                                                      (int64_t) sd.logical_shape.rows,
                                                      exact_err, sizeof(exact_err));
        if (src_exact == 0 || src_exact != src->get_padded_bytes()) {
            return fail("source exact-bytes mismatch");
        }
        if (ggml_nbytes(src_t) != src_exact) {
            return fail("source tensor size mismatch");
        }
        const size_t rb = ggml_row_size(sd.type, (int64_t) sd.padded_shape.cols);
        if (rb == 0 || rb != (size_t) sd.row_stride_bytes || src_t->nb[1] != rb) {
            return fail("source row stride mismatch");
        }
        if (sd.logical_shape.rows == 0 ||
            sd.logical_shape.rows > (uint64_t) std::numeric_limits<int64_t>::max()) {
            return fail("source shape out of range");
        }
        if (sd.padded_shape.cols == 0 ||
            sd.padded_shape.cols > (uint64_t) std::numeric_limits<int64_t>::max()) {
            return fail("source width out of range");
        }
        if (rq.surviving_rows.empty()) {
            return fail("empty surviving rows");
        }
        if ((uint64_t) rq.surviving_rows.size() > sd.logical_shape.rows) {
            return fail("more survivors than source rows");
        }
        for (size_t k = 0; k < rq.surviving_rows.size(); ++k) {
            if ((uint64_t) rq.surviving_rows[k] >= sd.logical_shape.rows) {
                return fail("row out of range");
            }
            if (k > 0 && rq.surviving_rows[k] <= rq.surviving_rows[k - 1]) {
                return fail("rows must be strictly ascending");
            }
        }
        std::string derr;
        if (!rq.dst_desc.validate(&derr)) {
            return fail("invalid dst descriptor: " + derr);
        }
        const codec_desc & dd = rq.dst_desc;
        if (dd.type != sd.type || dd.orient != sd.orient ||
            dd.padded_shape.cols != sd.padded_shape.cols ||
            dd.row_stride_bytes != sd.row_stride_bytes) {
            return fail("dst must match src type/orient/padding/stride (byte-preserving)");
        }
        if (dd.padded_shape.cols > (uint64_t) std::numeric_limits<int64_t>::max() ||
            dd.logical_shape.rows != (uint64_t) rq.surviving_rows.size() ||
            dd.padded_shape.rows != (uint64_t) rq.surviving_rows.size()) {
            return fail("dst rows must equal survivor count");
        }
        uint64_t dst_expected = 0;
        try {
            dst_expected = encoded_matrix_bytes(dd);
        } catch (const std::exception & e) {
            return fail(std::string("dst byte size: ") + e.what());
        } catch (...) {
            return fail("dst byte size failed");
        }
        uint64_t want = 0;
        if (!mul_ok_u64((uint64_t) rq.surviving_rows.size(), (uint64_t) rb, want) ||
            want != dst_expected ||
            dst_expected > (uint64_t) std::numeric_limits<size_t>::max()) {
            return fail("dst byte size mismatch");
        }
        uint64_t next_total = 0;
        if (!add_ok_u64(total_copies, (uint64_t) rq.surviving_rows.size(), next_total)) {
            return fail("copy count overflow");
        }
        total_copies = next_total;
        validated_pack v;
        v.req = &rq;
        v.exact = (size_t) dst_expected;
        v.src_rows = sd.logical_shape.rows;
        v.row_bytes = rb;
        v.src_nbytes = src_exact;
        v.src_nb1 = rb;
        v.padded_cols = (int64_t) sd.padded_shape.cols;
        vs.push_back(v);
    }

    // ---- Phase 1: one shared dst store (single ctx, single buffer) ----
    std::shared_ptr<xkv_backend_batch_storage> storage;
    std::vector<ggml_tensor *> dst_tensors(requests.size(), nullptr);
    {
        uint64_t t = 0, mem = 0;
        if (!mul_ok_u64((uint64_t) requests.size(), (uint64_t) ggml_tensor_overhead(), t) ||
            !add_ok_u64(t, 64, mem) ||
            mem > (uint64_t) std::numeric_limits<size_t>::max()) {
            set_err(err, "xkv_backend pack: context size overflow");
            return false;
        }
        std::unique_ptr<ggml_context, ggml_context_deleter> ctx_guard;
        {
            ggml_init_params ip = {};
            ip.mem_size = (size_t) mem;
            ip.no_alloc = true;
            ggml_context * ctx = ggml_init(ip);
            if (ctx == nullptr) {
                set_err(err, "xkv_backend pack: ggml_init failed");
                return false;
            }
            ctx_guard.reset(ctx);
        }
        for (size_t i = 0; i < requests.size(); ++i) {
            const codec_desc & dd = requests[i].dst_desc;
            ggml_tensor * dt = ggml_new_tensor_2d(ctx_guard.get(), dd.type,
                                                  (int64_t) dd.padded_shape.cols,
                                                  (int64_t) dd.logical_shape.rows);
            if (dt == nullptr) {
                set_err(err, "xkv_backend pack: dst tensor create failed");
                return false;
            }
            if (ggml_nbytes(dt) != vs[i].exact) {
                set_err(err, "xkv_backend pack: dst tensor size mismatch");
                return false;
        }
            dst_tensors[i] = dt;
        }
        if (reservation != nullptr) {
            uint64_t estimate = 0;
            for (size_t i = 0; i < requests.size(); ++i) {
                const size_t es = ggml_backend_buft_get_alloc_size(buft, dst_tensors[i]);
                if (es == 0) {
                    set_err(err, "xkv_backend pack: preflight slice query failed");
                    return false;
            }
                uint64_t next_est = 0;
                if (!add_ok_u64(estimate, (uint64_t) es, next_est)) {
                    set_err(err, "xkv_backend pack: preflight estimate overflow");
                    return false;
        }
                estimate = next_est;
            }
            if (estimate > reservation->reserved_bytes) {
                set_err(err, "xkv_backend pack: preflight estimate " + std::to_string(estimate) +
                             " exceeds reserved " + std::to_string(reservation->reserved_bytes));
                return false;
            }
            if (reservation->cap_bytes != 0 && estimate > reservation->cap_bytes) {
                set_err(err, "xkv_backend pack: preflight estimate " + std::to_string(estimate) +
                             " exceeds cap " + std::to_string(reservation->cap_bytes));
                return false;
            }
        }
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_guard.get(), buft);
        if (buf == nullptr) {
            set_err(err, "xkv_backend pack: shared dst buffer alloc failed");
            return false;
        }
        std::unique_ptr<ggml_backend_buffer, ggml_buffer_deleter> buf_guard(buf);
        storage = std::shared_ptr<xkv_backend_batch_storage>(new xkv_backend_batch_storage());
        storage->ctx = ctx_guard.release();
        storage->buffer = buf_guard.release();
        storage->buffer_bytes = ggml_backend_buffer_get_size(storage->buffer);
        if (reservation != nullptr) {
            const uint64_t actual = (uint64_t) storage->buffer_bytes;
            if (actual > reservation->reserved_bytes) {
                set_err(err, "xkv_backend pack: actual buffer " + std::to_string(actual) +
                             " exceeds reserved " + std::to_string(reservation->reserved_bytes));
                return false;
            }
            if (reservation->cap_bytes != 0 && actual > reservation->cap_bytes) {
                set_err(err, "xkv_backend pack: actual buffer " + std::to_string(actual) +
                             " exceeds cap " + std::to_string(reservation->cap_bytes));
                return false;
            }
        }
    }
    // ---- Phase 2: assemble off-side dst allocations (IDs burn on later failure) ----
    std::vector<std::shared_ptr<xkv_backend_allocation>> created;
    created.reserve(requests.size());
    xkv_backend_batch_stats stats;
    std::string owner_name;
    int owner_dev_type = -1;
    if (backend != nullptr) {
        const char * bname = ggml_backend_name(backend);
        if (bname != nullptr) {
            owner_name = bname;
        }
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        if (dev == nullptr) {
            dev = ggml_backend_buft_get_device(buft);
        }
        if (dev != nullptr) {
            owner_dev_type = (int) ggml_backend_dev_type(dev);
        }
    } else {
        ggml_backend_dev_t bdev = ggml_backend_buft_get_device(buft);
        if (bdev != nullptr) {
            owner_dev_type = (int) ggml_backend_dev_type(bdev);
        }
    }
    for (size_t i = 0; i < requests.size(); ++i) {
        if ((int) i == config.inject_alloc_fail_at) {
            set_err(err, "xkv_backend pack: injected alloc failure at request " + std::to_string(i));
                return false;
        }
        uint64_t alloc_id = 0;
        if (!id_gen.allocate_id(alloc_id, err)) {
            if (err && err->empty()) {
                *err = "xkv_backend pack: ID overflow at request " + std::to_string(i);
            } else if (err) {
                *err = "xkv_backend pack request " + std::to_string(i) + ": " + *err;
            }
                return false;
        }
        const validated_pack & v = vs[i];
        auto alloc = std::shared_ptr<xkv_backend_allocation>(new xkv_backend_allocation());
        alloc->allocation_id_ = alloc_id;
        alloc->residency_ = res0;
        alloc->desc_ = v.req->dst_desc;
        alloc->descriptor_fingerprint_ = v.req->dst_desc.fingerprint();
        alloc->checksum_ = FNV_OFFSET;
        alloc->checksum_bound_ = false; // device-packed: provenance instead (no D2H)
        alloc->pack_provenance_ = pack_provenance_for(v.req->dst_desc,
                                                      v.req->src->get_allocation_id(),
                                                      v.req->surviving_rows.data(),
                                                      v.req->surviving_rows.size());
        alloc->logical_bytes_ = xkv_compute_logical_bytes(v.req->dst_desc.type,
                                                           v.req->dst_desc.logical_shape.rows,
                                                           v.req->dst_desc.logical_shape.cols);
        alloc->padded_bytes_ = v.exact;
        if (alloc->logical_bytes_ == 0) {
            set_err(err, "xkv_backend pack: unsupported dst type");
                return false;
        }
        alloc->storage_ = storage;
        alloc->tensor_ = dst_tensors[i];
        const size_t slice = ggml_backend_buft_get_alloc_size(buft, alloc->tensor_);
        if (slice == 0) {
            set_err(err, "xkv_backend pack: dst slice query failed");
                return false;
        }
        alloc->slice_bytes_ = slice;
        alloc->owning_buft_ = buft;
        alloc->owning_backend_name_ = owner_name;
        alloc->owning_dev_type_ = owner_dev_type;
        alloc->owning_backend_null_ = (backend == nullptr);
        created.push_back(std::move(alloc));
    }

    // ---- Phase 3: queue every row copy (cross-tensor views), then ONE sync ----
    // Views are metadata only; the pinned source is never written. Same-backend
    // queue order keeps every copy in submission order; one trailing
    // synchronize covers the batch. Null backend uses inline sync copies.
    std::unique_ptr<ggml_context, ggml_context_deleter> vctx_guard;
    {
        uint64_t t2 = 0, vmem = 0;
        if (!mul_ok_u64(total_copies, 2, t2) ||
            !mul_ok_u64(t2, (uint64_t) ggml_tensor_overhead(), vmem) ||
            !add_ok_u64(vmem, 64, vmem) ||
            vmem > (uint64_t) std::numeric_limits<size_t>::max()) {
            set_err(err, "xkv_backend pack: view context size overflow");
            return false;
        }
        ggml_init_params vip = {};
        vip.mem_size = (size_t) vmem;
        vip.no_alloc = true;
        ggml_context * vctx = ggml_init(vip);
        if (vctx == nullptr) {
            set_err(err, "xkv_backend pack: view context init failed");
            return false;
        }
        vctx_guard.reset(vctx);
    }
    bool submitted = false;
    detail::submit_fence fence{backend, &submitted};
    uint64_t k = 0; // flattened copy ordinal in queue order
    for (size_t i = 0; i < requests.size(); ++i) {
        const validated_pack & v = vs[i];
        ggml_tensor * src_t = v.req->src->get_tensor();
        ggml_tensor * dst_t = dst_tensors[i];
        if (dst_t->nb[1] != v.row_bytes) {
            set_err(err, "xkv_backend pack request " + std::to_string(i) + ": dst stride mismatch");
            return false;
        }
        for (size_t j = 0; j < v.req->surviving_rows.size(); ++j) {
            const uint32_t r = v.req->surviving_rows[j];
            uint64_t src_off = 0, dst_off = 0, src_end = 0, dst_end = 0;
            if (!mul_ok_u64((uint64_t) r, (uint64_t) v.src_nb1, src_off) ||
                !mul_ok_u64((uint64_t) j, (uint64_t) v.row_bytes, dst_off) ||
                !add_ok_u64(src_off, (uint64_t) v.row_bytes, src_end) ||
                src_end > (uint64_t) v.src_nbytes ||
                !add_ok_u64(dst_off, (uint64_t) v.row_bytes, dst_end) ||
                dst_end > (uint64_t) v.exact) {
                set_err(err, "xkv_backend pack request " + std::to_string(i) + ": row range overflow");
                return false;
            }
            ggml_tensor * sv = ggml_view_2d(vctx_guard.get(), src_t, v.padded_cols, 1,
                                            v.src_nb1, (size_t) src_off);
            ggml_tensor * dv = ggml_view_2d(vctx_guard.get(), dst_t, v.padded_cols, 1,
                                            v.row_bytes, (size_t) dst_off);
            if (sv == nullptr || dv == nullptr) {
                set_err(err, "xkv_backend pack request " + std::to_string(i) + ": view create failed");
                return false;
            }
            if (!ggml_are_same_shape(sv, dv) || ggml_nbytes(sv) != v.row_bytes ||
                ggml_nbytes(dv) != v.row_bytes || sv->nb[0] != src_t->nb[0] ||
                sv->nb[1] != src_t->nb[1] || dv->nb[0] != dst_t->nb[0] ||
                dv->nb[1] != dst_t->nb[1]) {
                set_err(err, "xkv_backend pack request " + std::to_string(i) + ": view layout mismatch");
                return false;
            }
            if (config.inject_copy_fail_at >= 0 && (uint64_t) config.inject_copy_fail_at == k) {
                set_err(err, "xkv_backend pack: injected copy failure at copy " + std::to_string(k));
                return false; // fence drains earlier copies before destruction
            }
            if (backend != nullptr) {
                ggml_backend_tensor_copy_async(backend, backend, sv, dv);
                submitted = true;
                if (config.inject_post_submit_fail_at >= 0 &&
                    (uint64_t) config.inject_post_submit_fail_at == k) {
                    set_err(err, "xkv_backend pack: injected post-submit failure at copy " +
                                 std::to_string(k));
                    return false; // fence drains
                }
            } else {
                ggml_backend_tensor_copy(sv, dv);
            }
            ++k;
        }
    }
    if (backend != nullptr) {
        ggml_backend_synchronize(backend);
        stats.sync_count++;
        submitted = false; // flushed: disarm the fence
    }

    // ---- Phase 4: atomically publish (discarded on any failure above) ----
    out_local.handles = std::move(created);
    for (auto & h : out_local.handles) {
        h->mark_immutable();
    }
    out_local.stats = stats;
    return true;
}

bool xkv_backend_pack_batch(
    ggml_backend_t backend,
    ggml_backend_buffer_type_t buft,
    const std::vector<xkv_backend_pack_request> & requests,
    const xkv_backend_batch_config & config,
    xkv_allocation_id_generator & id_gen,
    xkv_backend_pack_result & out,
    std::string * err,
    const xkv_backend_store_reservation * reservation) {
    // Failure-atomic: out assigned ONLY on success.
    try {
        xkv_backend_pack_result local;
        std::string lerr;
        if (!xkv_backend_batch_builder::pack_impl(backend, buft, requests, config,
                                                  id_gen, local, &lerr, reservation)) {
            set_err(err, lerr);
            return false;
        }
        out = std::move(local);
        return true;
    } catch (const std::bad_alloc &) {
        set_err(err, "xkv_backend pack: out of memory (outputs unchanged)");
        return false;
    } catch (const std::exception & e) {
        set_err(err, std::string("xkv_backend pack: failed: ") + e.what() + " (outputs unchanged)");
        return false;
    } catch (...) {
        set_err(err, "xkv_backend pack: failed (unknown; outputs unchanged)");
        return false;
    }
}

// ---------------------------------------------------------------------------
// Device-tensor adopt (DAG-computed tensors become immutable allocations)
// ---------------------------------------------------------------------------

bool xkv_backend_batch_builder::adopt_impl(
    ggml_backend_t backend,
    ggml_backend_buffer_type_t buft,
    const std::vector<xkv_backend_device_stream> & streams,
    const xkv_backend_batch_config & config,
    xkv_allocation_id_generator & id_gen,
    xkv_backend_adopt_result & out_local,
    std::string * err,
    const xkv_backend_store_reservation * reservation) {
    if (buft == nullptr) {
        set_err(err, "xkv_backend adopt: null buffer type");
        return false;
    }
    if (streams.empty()) {
        set_err(err, "xkv_backend adopt: empty batch");
        return false;
    }

    // Pre-flight owner identity validation BEFORE any allocation/graph/ID consumption:
    if (config.backend_owner && config.backend_owner.get() != backend) {
        set_err(err, "xkv_backend adopt: backend != config.backend_owner.get()");
        return false;
    }

    if (config.verify_readback || config.verify_checksum) {
        set_err(err, "xkv_backend adopt: verify flags refused (no host bytes to compare against)");
        return false;
    }
    // Effective residency mirrors build (host buft adapts truthfully).
    const bool buft_is_host = ggml_backend_buft_is_host(buft);
    ggml_xkv_residency effective_res = config.residency;
    if (!config.enforce_residency && buft_is_host && effective_res == GGML_XKV_RES_DEVICE_OWNED) {
        effective_res = GGML_XKV_RES_REFERENCE_HOST;
    }

    // ---- Phase 0: validate everything (zero mutation, zero copies) ----
    struct validated_adopt {
        const xkv_backend_device_stream * s = nullptr;
        size_t exact = 0;
    };
    std::vector<validated_adopt> vs;
    vs.reserve(streams.size());
    for (size_t i = 0; i < streams.size(); ++i) {
        const auto & st = streams[i];
        auto fail = [&](const std::string & m) -> bool {
            if (err) {
                *err = "xkv_backend adopt stream " + std::to_string(i) + ": " + m;
            }
            return false;
        };
        if (st.tensor == nullptr) {
            return fail("null source tensor");
        }
        if (st.provenance == 0) {
            return fail("zero provenance tag (audit tag required)");
        }
        std::string derr;
        if (!st.desc.validate(&derr)) {
            return fail("invalid descriptor: " + derr);
        }
        if (!ggml_xkv_codec_supported(st.desc.type)) {
            return fail("unsupported codec");
        }
        if (st.desc.type != st.tensor->type) {
            return fail("tensor type mismatch with descriptor");
        }
        if (!ggml_is_contiguous(st.tensor)) {
            return fail("source tensor must be contiguous");
        }
        uint64_t expected = 0;
        try {
            expected = encoded_matrix_bytes(st.desc);
        } catch (const std::exception & e) {
            return fail(std::string("byte size: ") + e.what());
        } catch (...) {
            return fail("byte size failed");
        }
        if (expected == 0 || expected > (uint64_t) std::numeric_limits<size_t>::max()) {
            return fail("byte size out of range");
        }
        if (st.desc.padded_shape.cols > (uint64_t) std::numeric_limits<int64_t>::max() ||
            st.desc.logical_shape.rows > (uint64_t) std::numeric_limits<int64_t>::max()) {
            return fail("shape out of int64 range");
        }
        char exact_err[128] = {};
        const size_t exact = ggml_xkv_exact_bytes(st.desc.type,
                                                  (int64_t) st.desc.padded_shape.cols,
                                                  (int64_t) st.desc.logical_shape.rows,
                                                  exact_err, sizeof(exact_err));
        if (exact == 0 || (uint64_t) exact != expected) {
            return fail("exact-bytes mismatch with descriptor bytes");
        }
        if (ggml_nbytes(st.tensor) != exact) {
            return fail("tensor size mismatch with descriptor");
        }
        if (!ggml_xkv_residency_supported(st.desc.type, effective_res)) {
            return fail("residency unsupported for codec");
        }
        if (config.enforce_residency) {
            if (effective_res == GGML_XKV_RES_DEVICE_OWNED && buft_is_host) {
                return fail("device-owned residency on host buft");
            }
            if (effective_res == GGML_XKV_RES_REFERENCE_HOST && !buft_is_host) {
                return fail("host residency on device buft");
            }
        }
        if (effective_res == GGML_XKV_RES_DEVICE_OWNED && buft_is_host) {
            return fail("device-owned on host buft refused");
        }
        if (effective_res == GGML_XKV_RES_DEVICE_OWNED && backend != nullptr &&
            !ggml_backend_supports_buft(backend, buft)) {
            return fail("backend does not support buft");
        }
        validated_adopt v;
        v.s = &st;
        v.exact = exact;
        vs.push_back(v);
    }

    // ---- Phase 1: one shared dst store (single ctx, single buffer) ----
    std::shared_ptr<xkv_backend_batch_storage> storage;
    std::vector<ggml_tensor *> dst_tensors(streams.size(), nullptr);
    {
        uint64_t t = 0, mem = 0;
        if (!mul_ok_u64((uint64_t) streams.size(), (uint64_t) ggml_tensor_overhead(), t) ||
            !add_ok_u64(t, 64, mem) ||
            mem > (uint64_t) std::numeric_limits<size_t>::max()) {
            set_err(err, "xkv_backend adopt: context size overflow");
            return false;
        }
        std::unique_ptr<ggml_context, ggml_context_deleter> ctx_guard;
        {
            ggml_init_params ip = {};
            ip.mem_size = (size_t) mem;
            ip.no_alloc = true;
            ggml_context * ctx = ggml_init(ip);
            if (ctx == nullptr) {
                set_err(err, "xkv_backend adopt: ggml_init failed");
                return false;
            }
            ctx_guard.reset(ctx);
        }
        for (size_t i = 0; i < streams.size(); ++i) {
            const codec_desc & dd = streams[i].desc;
            ggml_tensor * dt = ggml_new_tensor_2d(ctx_guard.get(), dd.type,
                                                  (int64_t) dd.padded_shape.cols,
                                                  (int64_t) dd.logical_shape.rows);
            if (dt == nullptr) {
                set_err(err, "xkv_backend adopt: dst tensor create failed");
                return false;
            }
            if (ggml_nbytes(dt) != vs[i].exact) {
                set_err(err, "xkv_backend adopt: dst tensor size mismatch");
                return false;
            }
            dst_tensors[i] = dt;
        }
        if (reservation != nullptr) {
            uint64_t estimate = 0;
            for (size_t i = 0; i < streams.size(); ++i) {
                const size_t es = ggml_backend_buft_get_alloc_size(buft, dst_tensors[i]);
                if (es == 0) {
                    set_err(err, "xkv_backend adopt: preflight slice query failed");
                    return false;
                }
                uint64_t next_est = 0;
                if (!add_ok_u64(estimate, (uint64_t) es, next_est)) {
                    set_err(err, "xkv_backend adopt: preflight estimate overflow");
                    return false;
                }
                estimate = next_est;
            }
            if (estimate > reservation->reserved_bytes) {
                set_err(err, "xkv_backend adopt: preflight estimate " + std::to_string(estimate) +
                             " exceeds reserved " + std::to_string(reservation->reserved_bytes));
                return false;
            }
            if (reservation->cap_bytes != 0 && estimate > reservation->cap_bytes) {
                set_err(err, "xkv_backend adopt: preflight estimate " + std::to_string(estimate) +
                             " exceeds cap " + std::to_string(reservation->cap_bytes));
                return false;
            }
        }
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_guard.get(), buft);
        if (buf == nullptr) {
            set_err(err, "xkv_backend adopt: shared dst buffer alloc failed");
            return false;
        }
        std::unique_ptr<ggml_backend_buffer, ggml_buffer_deleter> buf_guard(buf);
        storage = std::shared_ptr<xkv_backend_batch_storage>(new xkv_backend_batch_storage());
        storage->ctx = ctx_guard.release();
        storage->buffer = buf_guard.release();
        storage->buffer_bytes = ggml_backend_buffer_get_size(storage->buffer);
        if (reservation != nullptr) {
            const uint64_t actual = (uint64_t) storage->buffer_bytes;
            if (actual > reservation->reserved_bytes) {
                set_err(err, "xkv_backend adopt: actual buffer " + std::to_string(actual) +
                             " exceeds reserved " + std::to_string(reservation->reserved_bytes));
                return false;
            }
            if (reservation->cap_bytes != 0 && actual > reservation->cap_bytes) {
                set_err(err, "xkv_backend adopt: actual buffer " + std::to_string(actual) +
                             " exceeds cap " + std::to_string(reservation->cap_bytes));
                return false;
            }
        }
    }

    // ---- Phase 2: assemble off-side dst allocations (IDs burn on later failure) ----
    std::vector<std::shared_ptr<xkv_backend_allocation>> created;
    created.reserve(streams.size());
    xkv_backend_batch_stats stats;
    std::string owner_name;
    int owner_dev_type = -1;
    if (backend != nullptr) {
        const char * bname = ggml_backend_name(backend);
        if (bname != nullptr) {
            owner_name = bname;
        }
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        if (dev == nullptr) {
            dev = ggml_backend_buft_get_device(buft);
        }
        if (dev != nullptr) {
            owner_dev_type = (int) ggml_backend_dev_type(dev);
        }
    } else {
        ggml_backend_dev_t bdev = ggml_backend_buft_get_device(buft);
        if (bdev != nullptr) {
            owner_dev_type = (int) ggml_backend_dev_type(bdev);
        }
    }
    for (size_t i = 0; i < streams.size(); ++i) {
        if ((int) i == config.inject_alloc_fail_at) {
            set_err(err, "xkv_backend adopt: injected alloc failure at stream " + std::to_string(i));
            return false;
        }
        uint64_t alloc_id = 0;
        if (!id_gen.allocate_id(alloc_id, err)) {
            if (err && err->empty()) {
                *err = "xkv_backend adopt: ID overflow at stream " + std::to_string(i);
            } else if (err) {
                *err = "xkv_backend adopt stream " + std::to_string(i) + ": " + *err;
            }
            return false;
        }
        const validated_adopt & v = vs[i];
        auto alloc = std::shared_ptr<xkv_backend_allocation>(new xkv_backend_allocation());
        alloc->allocation_id_ = alloc_id;
        alloc->residency_ = effective_res;
        alloc->desc_ = v.s->desc;
        alloc->descriptor_fingerprint_ = v.s->desc.fingerprint();
        alloc->checksum_ = FNV_OFFSET;
        alloc->checksum_bound_ = false; // device-born: provenance instead (no D2H)
        alloc->pack_provenance_ = v.s->provenance;
        alloc->logical_bytes_ = xkv_compute_logical_bytes(v.s->desc.type,
                                                           v.s->desc.logical_shape.rows,
                                                           v.s->desc.logical_shape.cols);
        alloc->padded_bytes_ = v.exact;
        if (alloc->logical_bytes_ == 0) {
            set_err(err, "xkv_backend adopt: unsupported stream type");
            return false;
        }
        alloc->storage_ = storage;
        alloc->tensor_ = dst_tensors[i];
        const size_t slice = ggml_backend_buft_get_alloc_size(buft, alloc->tensor_);
        if (slice == 0) {
            set_err(err, "xkv_backend adopt: dst slice query failed");
            return false;
        }
        alloc->slice_bytes_ = slice;
        alloc->owning_buft_ = buft;
        alloc->owning_backend_name_ = owner_name;
        alloc->owning_dev_type_ = owner_dev_type;
        alloc->owning_backend_null_ = (backend == nullptr);
        created.push_back(std::move(alloc));
    }

    // ---- Phase 3: queue full-tensor copies, then ONE sync ----
    // Same-backend queue order; one trailing synchronize covers the batch.
    // Null backend uses inline sync copies. No views needed (whole tensors).
    bool submitted = false;
    detail::submit_fence fence{backend, &submitted};
    uint64_t k = 0; // copy ordinal == stream index (one copy per stream)
    for (size_t i = 0; i < streams.size(); ++i) {
        const validated_adopt & v = vs[i];
        ggml_tensor * src_t = v.s->tensor;
        ggml_tensor * dst_t = dst_tensors[i];
        if (!ggml_are_same_shape(src_t, dst_t) || ggml_nbytes(src_t) != v.exact ||
            ggml_nbytes(dst_t) != v.exact) {
            set_err(err, "xkv_backend adopt stream " + std::to_string(i) + ": shape drift");
            return false;
        }
        if (config.inject_copy_fail_at >= 0 && (uint64_t) config.inject_copy_fail_at == k) {
            set_err(err, "xkv_backend adopt: injected copy failure at copy " + std::to_string(k));
            return false; // fence drains earlier copies before destruction
        }
        if (backend != nullptr) {
            ggml_backend_tensor_copy_async(backend, backend, src_t, dst_t);
            submitted = true;
            if (config.inject_post_submit_fail_at >= 0 &&
                (uint64_t) config.inject_post_submit_fail_at == k) {
                set_err(err, "xkv_backend adopt: injected post-submit failure at copy " +
                             std::to_string(k));
                return false; // fence drains
            }
        } else {
            ggml_backend_tensor_copy(src_t, dst_t);
        }
        ++k;
    }
    if (backend != nullptr) {
        ggml_backend_synchronize(backend);
        stats.sync_count++;
        submitted = false; // flushed: disarm the fence
    }

    // ---- Phase 4: atomically publish (discarded on any failure above) ----
    // No host sources exist, so no releasable indices or release callbacks.
    out_local.handles = std::move(created);
    for (auto & h : out_local.handles) {
        h->mark_immutable();
    }
    out_local.stats = stats;
    // If caller provided a real backend_owner, verify identity and copy the owner.
    // NEVER manufacture ownership with a no-op deleter from a raw pointer!
    if (config.backend_owner) {
        if (config.backend_owner.get() != backend) {
            set_err(err, "xkv_backend: backend != config.backend_owner.get()");
            return false;
        }
        out_local.executor = config.backend_owner;
        if (storage) {
            storage->backend_executor = config.backend_owner;
        }
    }
    return true;
}

bool xkv_backend_adopt_device_tensors(
    ggml_backend_t backend,
    ggml_backend_buffer_type_t buft,
    const std::vector<xkv_backend_device_stream> & streams,
    const xkv_backend_batch_config & config,
    xkv_allocation_id_generator & id_gen,
    xkv_backend_adopt_result & out,
    std::string * err,
    const xkv_backend_store_reservation * reservation) {
    // Failure-atomic: out assigned ONLY on success.
    try {
        xkv_backend_adopt_result local;
        std::string lerr;
        if (!xkv_backend_batch_builder::adopt_impl(backend, buft, streams, config,
                                                   id_gen, local, &lerr, reservation)) {
            set_err(err, lerr);
            return false;
        }
        out = std::move(local);
        return true;
    } catch (const std::bad_alloc &) {
        set_err(err, "xkv_backend adopt: out of memory (outputs unchanged)");
        return false;
    } catch (const std::exception & e) {
        set_err(err, std::string("xkv_backend adopt: failed: ") + e.what() + " (outputs unchanged)");
        return false;
    } catch (...) {
        set_err(err, "xkv_backend adopt: failed (unknown; outputs unchanged)");
        return false;
    }
}

bool xkv_backend_batch_builder::take_impl(
    ggml_backend_t backend,
    ggml_backend_buffer_type_t buft,
    std::shared_ptr<xkv_backend_batch_storage> storage,
    const std::vector<xkv_backend_device_stream> & streams,
    const xkv_backend_batch_config & config,
    xkv_allocation_id_generator & id_gen,
    xkv_backend_adopt_result & out_local,
    std::string * err,
    const xkv_backend_store_reservation * reservation) {
    if (!backend || !buft || !storage || !storage->ctx || !storage->buffer) {
        set_err(err, "xkv_backend take: null backend/buft/storage");
        return false;
    }
    if (storage.use_count() != 1) {
        set_err(err, "xkv_backend take: storage must have unique ownership");
        return false;
    }
    if (streams.empty()) {
        set_err(err, "xkv_backend take: empty batch");
        return false;
    }
    if (!config.backend_owner || config.backend_owner.get() != backend) {
        set_err(err, "xkv_backend take: real matching backend_owner required");
        return false;
    }
    if (config.verify_readback || config.verify_checksum || config.inject_upload_fail_at >= 0 ||
        config.inject_readback_fail_at >= 0 || config.inject_checksum_fail_at >= 0 ||
        config.inject_post_submit_fail_at >= 0 || config.inject_copy_fail_at >= 0) {
        set_err(err, "xkv_backend take: copy/readback options are invalid for ownership transfer");
        return false;
    }
    if (ggml_backend_buffer_get_type(storage->buffer) != buft ||
        !ggml_backend_supports_buft(backend, buft)) {
        set_err(err, "xkv_backend take: storage placement mismatch");
        return false;
    }
    const size_t actual_buffer_bytes = ggml_backend_buffer_get_size(storage->buffer);
    if (actual_buffer_bytes == 0 || storage->buffer_bytes != actual_buffer_bytes) {
        set_err(err, "xkv_backend take: invalid storage byte accounting");
        return false;
    }
    if (reservation &&
        ((uint64_t)actual_buffer_bytes > reservation->reserved_bytes ||
         (reservation->cap_bytes != 0 && (uint64_t)actual_buffer_bytes > reservation->cap_bytes))) {
        set_err(err, "xkv_backend take: actual buffer exceeds store reservation");
        return false;
    }

    const bool buft_is_host = ggml_backend_buft_is_host(buft);
    ggml_xkv_residency effective_res = config.residency;
    if (!config.enforce_residency && buft_is_host && effective_res == GGML_XKV_RES_DEVICE_OWNED) {
        effective_res = GGML_XKV_RES_REFERENCE_HOST;
    }
    if (!ggml_xkv_residency_supported(streams[0].desc.type, effective_res) ||
        (effective_res == GGML_XKV_RES_DEVICE_OWNED && buft_is_host) ||
        (config.enforce_residency && effective_res == GGML_XKV_RES_REFERENCE_HOST && !buft_is_host)) {
        set_err(err, "xkv_backend take: residency/placement mismatch");
        return false;
    }

    struct validated_take {
        const xkv_backend_device_stream * stream = nullptr;
        size_t exact = 0;
        size_t slice = 0;
        size_t logical = 0;
    };
    std::vector<validated_take> validated;
    std::vector<ggml_tensor *> seen;
    validated.reserve(streams.size());
    seen.reserve(streams.size());
    for (size_t i = 0; i < streams.size(); ++i) {
        const auto & stream = streams[i];
        auto fail = [&](const std::string & message) {
            set_err(err, "xkv_backend take stream " + std::to_string(i) + ": " + message);
            return false;
        };
        if (!stream.tensor || stream.tensor->buffer != storage->buffer) return fail("tensor not owned by storage");
        if (std::find(seen.begin(), seen.end(), stream.tensor) != seen.end()) return fail("duplicate tensor");
        seen.push_back(stream.tensor);
        if (stream.provenance == 0) return fail("zero provenance tag");
        std::string desc_err;
        if (!stream.desc.validate(&desc_err)) return fail("invalid descriptor: " + desc_err);
        if (!ggml_xkv_codec_supported(stream.desc.type) ||
            !ggml_xkv_residency_supported(stream.desc.type, effective_res)) {
            return fail("codec/residency unsupported");
        }
        if (stream.tensor->type != stream.desc.type || !ggml_is_contiguous(stream.tensor) ||
            stream.desc.padded_shape.cols > (uint64_t)INT64_MAX ||
            stream.desc.logical_shape.rows > (uint64_t)INT64_MAX ||
            stream.tensor->ne[0] != (int64_t)stream.desc.padded_shape.cols ||
            stream.tensor->ne[1] != (int64_t)stream.desc.logical_shape.rows ||
            stream.tensor->ne[2] != 1 || stream.tensor->ne[3] != 1) {
            return fail("tensor shape/type/layout mismatch");
        }
        uint64_t expected = 0;
        try {
            expected = encoded_matrix_bytes(stream.desc);
        } catch (const std::exception & e) {
            return fail(std::string("byte size: ") + e.what());
        }
        if (expected == 0 || expected > SIZE_MAX || ggml_nbytes(stream.tensor) != (size_t)expected) {
            return fail("tensor byte size mismatch");
        }
        const size_t slice = ggml_backend_buft_get_alloc_size(buft, stream.tensor);
        if (slice == 0) return fail("allocation slice query failed");
        const size_t logical = xkv_compute_logical_bytes(
            stream.desc.type,
            stream.desc.logical_shape.rows,
            stream.desc.logical_shape.cols);
        if (logical == 0) return fail("logical byte calculation failed");
        validated.push_back({&stream, (size_t)expected, slice, logical});
    }

    // Validate exact tensor membership: storage->ctx must contain exactly the batch's tensors
    size_t ctx_tensor_count = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(storage->ctx); t != nullptr; t = ggml_get_next_tensor(storage->ctx, t)) {
        if (std::find(seen.begin(), seen.end(), t) == seen.end()) {
            set_err(err, "xkv_backend take: context contains tensor not in stream batch");
            return false;
        }
        ctx_tensor_count++;
    }
    if (ctx_tensor_count != streams.size()) {
        set_err(err, "xkv_backend take: tensor count mismatch between context and batch");
        return false;
    }

    std::string owner_name;
    if (const char * name = ggml_backend_name(backend)) owner_name = name;
    int owner_dev_type = -1;
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (!dev) dev = ggml_backend_buft_get_device(buft);
    if (dev) owner_dev_type = (int)ggml_backend_dev_type(dev);

    std::vector<std::shared_ptr<xkv_backend_allocation>> created;
    created.reserve(streams.size());
    for (size_t i = 0; i < streams.size(); ++i) {
        if ((int)i == config.inject_alloc_fail_at) {
            set_err(err, "xkv_backend take: injected allocation failure at stream " + std::to_string(i));
            return false;
        }
        uint64_t allocation_id = 0;
        if (!id_gen.allocate_id(allocation_id, err)) return false;
        const validated_take & value = validated[i];
        auto allocation = std::shared_ptr<xkv_backend_allocation>(new xkv_backend_allocation());
        allocation->allocation_id_ = allocation_id;
        allocation->residency_ = effective_res;
        allocation->desc_ = value.stream->desc;
        allocation->descriptor_fingerprint_ = value.stream->desc.fingerprint();
        allocation->checksum_ = FNV_OFFSET;
        allocation->checksum_bound_ = false;
        allocation->pack_provenance_ = value.stream->provenance;
        allocation->logical_bytes_ = value.logical;
        allocation->padded_bytes_ = value.exact;
        allocation->slice_bytes_ = value.slice;
        allocation->storage_ = storage;
        allocation->tensor_ = value.stream->tensor;
        allocation->owning_buft_ = buft;
        allocation->owning_backend_name_ = owner_name;
        allocation->owning_dev_type_ = owner_dev_type;
        allocation->owning_backend_null_ = false;
        allocation->mark_immutable();
        created.push_back(std::move(allocation));
    }

    storage->backend_executor = config.backend_owner;
    out_local.handles = std::move(created);
    out_local.stats = {};
    out_local.executor = config.backend_owner;
    return true;
}

bool xkv_backend_take_device_tensors(
    ggml_backend_t backend,
    ggml_backend_buffer_type_t buft,
    std::shared_ptr<xkv_backend_batch_storage> storage,
    const std::vector<xkv_backend_device_stream> & streams,
    const xkv_backend_batch_config & config,
    xkv_allocation_id_generator & id_gen,
    xkv_backend_adopt_result & out,
    std::string * err,
    const xkv_backend_store_reservation * reservation) {
    try {
        xkv_backend_adopt_result local;
        std::string local_err;
        if (!xkv_backend_batch_builder::take_impl(backend, buft, std::move(storage), streams,
                config, id_gen, local, &local_err, reservation)) {
            set_err(err, local_err);
            return false;
        }
        out = std::move(local);
        return true;
    } catch (const std::bad_alloc &) {
        set_err(err, "xkv_backend take: out of memory (outputs unchanged)");
        return false;
    } catch (const std::exception & e) {
        set_err(err, std::string("xkv_backend take: failed: ") + e.what() + " (outputs unchanged)");
        return false;
    } catch (...) {
        set_err(err, "xkv_backend take: failed (unknown; outputs unchanged)");
        return false;
    }
}

// ---------------------------------------------------------------------------
// State-restore import: host bytes -> shared device handles, atomically.
// The import never touches the live store; the store publishes after.
// ---------------------------------------------------------------------------

static bool import_impl(
    ggml_backend_t backend,
    ggml_backend_buffer_type_t buft,
    const std::vector<xkv_backend_import_stream> & streams,
    const xkv_backend_batch_config & config,
    xkv_allocation_id_generator & id_gen,
    xkv_backend_import_result & out_local,
    std::string * err,
    const xkv_backend_store_reservation * reservation) {
    if (buft == nullptr) {
        set_err(err, "xkv_backend import: null buffer type");
        return false;
    }
    if (streams.empty()) {
        set_err(err, "xkv_backend import: empty batch");
        return false;
    }
    // Phase 0: validate descriptors/sizes and verify checksums pre-upload.
    // Zero mutation, zero upload; any mismatch fails the whole import.
    for (size_t i = 0; i < streams.size(); ++i) {
        const auto & s = streams[i];
        auto fail = [&](const std::string & m) -> bool {
            if (err) {
                *err = "xkv_backend import stream " + std::to_string(i) + ": " + m;
            }
            return false;
        };
        std::string derr;
        if (!s.desc.validate(&derr)) {
            return fail("invalid descriptor: " + derr);
        }
        if (!ggml_xkv_codec_supported(s.desc.type)) {
            return fail("unsupported codec");
        }
        if (s.desc.logical_shape.rows == 0 || s.desc.logical_shape.cols == 0 ||
            s.desc.padded_shape.rows == 0 || s.desc.padded_shape.cols == 0) {
            return fail("zero-shape stream rejected");
        }
        uint64_t expected = 0;
        try {
            expected = encoded_matrix_bytes(s.desc);
        } catch (const std::exception & e) {
            return fail(std::string("byte size: ") + e.what());
        } catch (...) {
            return fail("byte size failed");
        }
        if (expected == 0 || (uint64_t) s.size != expected) {
            return fail("byte size mismatch");
        }
        if (s.data == nullptr) {
            return fail("null stream data");
        }
        char exact_err[128] = {};
        const size_t exact = ggml_xkv_exact_bytes(s.desc.type,
                                                  (int64_t) s.desc.padded_shape.cols,
                                                  (int64_t) s.desc.logical_shape.rows,
                                                  exact_err, sizeof(exact_err));
        if (exact == 0 || (uint64_t) exact != expected) {
            return fail("exact-bytes mismatch");
        }
        uint64_t got = FNV_OFFSET;
        if ((s.desc_bytes == nullptr) != (s.desc_size == 0)) {
            return fail("desc_bytes pointer/size mismatch");
        }
        if (s.desc_bytes != nullptr) {
            // Envelope form: chained FNV(desc_bytes || bytes), matching the
            // state image checksum. Absent desc_bytes: plain FNV(bytes).
            got = xkv_backend_checksum_seeded(s.desc_bytes, s.desc_size, got);
        }
        got = xkv_backend_checksum_seeded(s.data, s.size, got);
        if (got != s.expected_checksum) {
            return fail("checksum mismatch: persisted bytes do not match stored checksum");
        }
    }
    // Phase 1: dedup by key in first-occurrence order. Same key requires
    // same descriptor/size/checksum (identity of one state allocation).
    std::vector<uint64_t> uniq_keys;
    std::vector<size_t> uniq_stream;
    std::vector<size_t> slot_of(streams.size(), 0);
    uniq_keys.reserve(streams.size());
    uniq_stream.reserve(streams.size());
    for (size_t i = 0; i < streams.size(); ++i) {
        const uint64_t key = streams[i].dedup_key;
        size_t slot = uniq_keys.size();
        if (key != 0) {
            for (size_t u = 0; u < uniq_keys.size(); ++u) {
                if (uniq_keys[u] == key) {
                    slot = u;
                    break;
                }
            }
        }
        if (slot == uniq_keys.size()) {
            uniq_keys.push_back(key);
            uniq_stream.push_back(i);
        } else {
            const auto & first = streams[uniq_stream[slot]];
            const auto & dup = streams[i];
            if (!(first.desc == dup.desc) || first.size != dup.size ||
                first.expected_checksum != dup.expected_checksum) {
                if (err) {
                    *err = "xkv_backend import stream " + std::to_string(i) +
                           ": dedup-key collision across different streams";
                }
                return false;
            }
        }
        slot_of[i] = slot;
    }
    // Phase 2: one atomic builder upload of the unique set (reservation
    // enforced pre/post-alloc inside build). Raw borrowings: no auto-release;
    // the caller clears persisted bytes after publish via releasable indices.
    xkv_backend_batch_builder builder(&id_gen);
    for (size_t u = 0; u < uniq_stream.size(); ++u) {
        const auto & s = streams[uniq_stream[u]];
        builder.add_stream(s.desc, s.data, s.size);
    }
    xkv_backend_batch_result bres;
    if (!builder.build(backend, buft, config, bres, err, reservation)) {
        return false;
    }
    // Phase 3: fan shared handles out order-matched (shared_ptr copies keep
    // B identity shared on device too).
    std::vector<std::shared_ptr<xkv_backend_allocation>> mapped(streams.size());
    for (size_t i = 0; i < streams.size(); ++i) {
        mapped[i] = bres.handles[slot_of[i]];
    }
    out_local.handles = std::move(mapped);
    out_local.stats = bres.stats;
    return true;
}

bool xkv_backend_import_batch(
    ggml_backend_t backend,
    ggml_backend_buffer_type_t buft,
    const std::vector<xkv_backend_import_stream> & streams,
    const xkv_backend_batch_config & config,
    xkv_allocation_id_generator & id_gen,
    xkv_backend_import_result & out,
    std::string * err,
    const xkv_backend_store_reservation * reservation) {
    // Failure-atomic: out assigned ONLY on success; live store untouched.
    try {
        xkv_backend_import_result local;
        std::string lerr;
        if (!import_impl(backend, buft, streams, config, id_gen, local, &lerr, reservation)) {
            set_err(err, lerr);
            return false;
        }
        out = std::move(local);
        return true;
    } catch (const std::bad_alloc &) {
        set_err(err, "xkv_backend import: out of memory (outputs unchanged)");
        return false;
    } catch (const std::exception & e) {
        set_err(err, std::string("xkv_backend import: failed: ") + e.what() + " (outputs unchanged)");
        return false;
    } catch (...) {
        set_err(err, "xkv_backend import: failed (unknown; outputs unchanged)");
        return false;
    }
}
static xkv_backend_peak_accounting peak_from_const(
    const std::vector<std::shared_ptr<const xkv_backend_allocation>> & live,
    const std::vector<std::shared_ptr<const xkv_backend_allocation>> & pinned) noexcept {
    xkv_backend_peak_accounting peak;
    peak.live = accounting_from_const(live);
    peak.pinned_retired = accounting_from_const(pinned);
    try {
        std::vector<std::shared_ptr<const xkv_backend_allocation>> both;
        both.reserve(live.size() + pinned.size());
        for (const auto & a : live) {
            both.push_back(a);
        }
        for (const auto & a : pinned) {
            both.push_back(a);
        }
        peak.combined = accounting_from_const(both);
        if (peak.live.overflow || peak.pinned_retired.overflow) {
            peak.combined.overflow = true;
        }
    } catch (...) {
        peak.combined.overflow = true;
    }
    return peak;
}

xkv_backend_peak_accounting xkv_backend_calculate_peak_accounting(
    const std::vector<std::shared_ptr<const xkv_backend_allocation>> & live,
    const std::vector<std::shared_ptr<const xkv_backend_allocation>> & pinned_retired) {
    return peak_from_const(live, pinned_retired);
}

xkv_backend_peak_accounting xkv_backend_calculate_peak_accounting(
    const std::vector<std::shared_ptr<xkv_backend_allocation>> & live,
    const std::vector<std::shared_ptr<xkv_backend_allocation>> & pinned_retired) {
    try {
        std::vector<std::shared_ptr<const xkv_backend_allocation>> live_view;
        std::vector<std::shared_ptr<const xkv_backend_allocation>> pinned_view;
        live_view.reserve(live.size());
        pinned_view.reserve(pinned_retired.size());
        for (const auto & a : live) {
            live_view.push_back(a);
        }
        for (const auto & a : pinned_retired) {
            pinned_view.push_back(a);
        }
        return peak_from_const(live_view, pinned_view);
    } catch (...) {
        xkv_backend_peak_accounting peak;
        peak.combined.overflow = true;
        return peak;
    }
}

// ---------------------------------------------------------------------------
// Whole-bundle verification for store adopt-or-refuse (no mutation, no D2H)
// ---------------------------------------------------------------------------

static bool bundle_verify_const(
    const std::vector<std::shared_ptr<const xkv_backend_allocation>> & handles,
    ggml_xkv_residency expected_residency,
    bool allow_unbound_packed,
    std::string * err) {
    if (handles.empty()) {
        set_err(err, "xkv_backend bundle: empty handle set");
        return false;
    }
    ggml_backend_buffer_type_t buft0 = nullptr;
    bool have_buft = false;
    for (size_t i = 0; i < handles.size(); ++i) {
        const auto & h = handles[i];
        auto fail = [&](const std::string & m) -> bool {
            if (err) {
                *err = "xkv_backend bundle handle " + std::to_string(i) + ": " + m;
            }
            return false;
        };
        if (!h) {
            return fail("null handle");
        }
        if (!h->is_immutable()) {
            return fail("not immutable");
        }
        if (h->get_residency() != expected_residency) {
            return fail("residency mismatch with bundle");
        }
        if (h->get_tensor() == nullptr || h->get_buffer() == nullptr) {
            return fail("empty tensor/buffer");
        }
        if (!have_buft) {
            buft0 = h->get_owning_buft();
            have_buft = true;
            if (buft0 == nullptr) {
                return fail("null owning buffer type");
            }
        } else if (h->get_owning_buft() != buft0) {
            return fail("buffer-type mismatch within bundle");
        }
        const codec_desc & d = h->get_desc();
        std::string derr;
        if (!d.validate(&derr)) {
            return fail("invalid descriptor: " + derr);
        }
        if (d.padded_shape.cols > (uint64_t) std::numeric_limits<int64_t>::max() ||
            d.logical_shape.rows > (uint64_t) std::numeric_limits<int64_t>::max()) {
            return fail("shape out of int64 range");
        }
        if (h->get_descriptor_fingerprint() != d.fingerprint()) {
            return fail("descriptor fingerprint mismatch");
        }
        char exact_err[128] = {};
        const size_t exact = ggml_xkv_exact_bytes(d.type, (int64_t) d.padded_shape.cols,
                                                  (int64_t) d.logical_shape.rows,
                                                  exact_err, sizeof(exact_err));
        if (exact == 0 || exact != h->get_padded_bytes() ||
            ggml_nbytes(h->get_tensor()) != exact) {
            return fail("exact-bytes mismatch");
        }
        // Bound handles were byte-verified at upload; the value travels with
        // the handle (re-verifiable on readback). Packed handles carry only
        // provenance (no D2H allowed) and pass solely by allow-flag + tag.
        if (!h->is_checksum_bound() &&
            (!allow_unbound_packed || h->get_pack_provenance() == 0)) {
            return fail("unbound checksum without pack provenance");
        }
    }
    return true;
}

bool xkv_backend_bundle_verify(
    const std::vector<std::shared_ptr<const xkv_backend_allocation>> & handles,
    ggml_xkv_residency expected_residency,
    bool allow_unbound_packed,
    std::string * err) {
    try {
        return bundle_verify_const(handles, expected_residency, allow_unbound_packed, err);
    } catch (const std::exception & e) {
        set_err(err, std::string("xkv_backend bundle: failed: ") + e.what());
        return false;
    } catch (...) {
        set_err(err, "xkv_backend bundle: failed (unknown)");
        return false;
    }
}

bool xkv_backend_bundle_verify(
    const std::vector<std::shared_ptr<xkv_backend_allocation>> & handles,
    ggml_xkv_residency expected_residency,
    bool allow_unbound_packed,
    std::string * err) {
    try {
        std::vector<std::shared_ptr<const xkv_backend_allocation>> const_view;
        const_view.reserve(handles.size());
        for (const auto & h : handles) {
            const_view.push_back(h);
        }
        return bundle_verify_const(const_view, expected_residency, allow_unbound_packed, err);
    } catch (const std::bad_alloc &) {
        set_err(err, "xkv_backend bundle: out of memory");
        return false;
    } catch (const std::exception & e) {
        set_err(err, std::string("xkv_backend bundle: failed: ") + e.what());
        return false;
    } catch (...) {
        set_err(err, "xkv_backend bundle: failed (unknown)");
        return false;
    }
}
// ---------------------------------------------------------------------------

bool xkv_backend_supports_residency(
    ggml_backend_t backend,
    ggml_backend_buffer_type_t buft,
    enum ggml_type type,
    ggml_xkv_residency residency,
    std::string * err) {
    if (buft == nullptr) {
        set_err(err, "xkv_backend: null buffer type");
        return false;
    }
    if (!ggml_xkv_codec_supported(type)) {
        set_err(err, "xkv_backend: unsupported codec");
        return false;
    }
    if (!ggml_xkv_residency_supported(type, residency)) {
        set_err(err, "xkv_backend: residency unsupported for codec in this build");
        return false;
    }
    const bool buft_is_host = ggml_backend_buft_is_host(buft);
    if (residency == GGML_XKV_RES_DEVICE_OWNED) {
        if (buft_is_host) {
            set_err(err, "xkv_backend: device-owned residency on host buft");
            return false;
        }
        if (backend == nullptr) {
            set_err(err, "xkv_backend: device-owned residency requires a backend");
            return false;
        }
        if (!ggml_backend_supports_buft(backend, buft)) {
            set_err(err, "xkv_backend: backend does not support buft");
            return false;
        }
        // A CPU device can never back device-owned residency truthfully.
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        if (dev == nullptr) {
            dev = ggml_backend_buft_get_device(buft);
        }
        if (dev != nullptr && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            set_err(err, "xkv_backend: device-owned residency on CPU device");
            return false;
        }
    } else {
        // REFERENCE_HOST: requires a host buft and backend support when backend given.
        if (!buft_is_host) {
            set_err(err, "xkv_backend: host residency on device buft");
            return false;
        }
        if (backend != nullptr && !ggml_backend_supports_buft(backend, buft)) {
            set_err(err, "xkv_backend: backend does not support host buft");
            return false;
        }
    }
    // Live probe against the ACTUAL backend/buft pair (not just type claims):
    // a minimal allocation must succeed. Op-dispatch support for the XKV
    // reconstruct kernels is additionally validated per graph at execution
    // time via ggml_backend_supports_op.
    {
        const size_t align = ggml_backend_buft_get_alignment(buft);
        const size_t probe = align == 0 ? 1 : align;
        ggml_backend_buffer_t probe_buf = ggml_backend_buft_alloc_buffer(buft, probe);
        if (probe_buf == nullptr) {
            set_err(err, "xkv_backend: live allocation probe on buft failed");
            return false;
        }
        ggml_backend_buffer_free(probe_buf);
    }
    return true;
}

bool xkv_backend_is_production_vulkan(
    ggml_backend_t backend,
    ggml_backend_buffer_type_t buft,
    std::string * err) {
    if (backend == nullptr || buft == nullptr) {
        set_err(err, "xkv_backend: null backend/buft");
        return false;
    }
    if (ggml_backend_buft_is_host(buft)) {
        return false;
    }
    if (!ggml_backend_supports_buft(backend, buft)) {
        return false;
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (dev == nullptr) {
        dev = ggml_backend_buft_get_device(buft);
    }
    if (dev == nullptr) {
        return false;
    }
    if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
        return false;
    }
    const char * name = ggml_backend_name(backend);
    if (name == nullptr || std::string(name).find("Vulkan") == std::string::npos) {
        // Fall back to the device description / backend registry name.
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        const char * reg_name = reg ? ggml_backend_reg_name(reg) : nullptr;
        if (reg_name == nullptr || std::string(reg_name).find("Vulkan") == std::string::npos) {
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Helper to create a true shared_ptr<ggml_backend> with ggml_backend_free deleter
// ---------------------------------------------------------------------------

std::shared_ptr<struct ggml_backend> xkv_backend_make_owner(ggml_backend_t backend) {
    if (backend == nullptr) {
        return nullptr;
    }
    return std::shared_ptr<struct ggml_backend>(backend, ggml_backend_free);
}

std::shared_ptr<xkv_backend_batch_result> xkv_backend_import_result::make_batch_result(
    ggml_backend_t backend,
    std::shared_ptr<struct ggml_backend> owner) const {
    // Check if any imported handle has DEVICE_OWNED residency
    bool has_device = false;
    for (const auto & h : this->handles) {
        if (h && h->get_residency() == GGML_XKV_RES_DEVICE_OWNED) {
            has_device = true;
            break;
        }
    }
    // If DEVICE_OWNED, require non-null owner with owner.get() == backend.
    // NEVER manufacture ownership with a double-freeing or fake shared_ptr!
    if (has_device) {
        if (!owner || owner.get() != backend) {
            return nullptr;
        }
    } else if (owner && backend && owner.get() != backend) {
        return nullptr;
    }

    auto res = std::make_shared<xkv_backend_batch_result>();
    res->success_ = true;
    res->committed_ = true;
    res->handles = this->handles;
    res->stats = this->stats;
    if (owner) {
        res->executor = std::move(owner);
    }
    return res;
}

std::shared_ptr<xkv_backend_batch_result> xkv_backend_adopt_result::make_batch_result(
    ggml_backend_t backend,
    std::shared_ptr<struct ggml_backend> owner) const {
    bool has_device = false;
    for (const auto & h : this->handles) {
        if (h && h->get_residency() == GGML_XKV_RES_DEVICE_OWNED) {
            has_device = true;
            break;
        }
    }
    if (has_device) {
        if (!owner || owner.get() != backend) {
            return nullptr;
        }
    } else if (owner && backend && owner.get() != backend) {
        return nullptr;
    }

    auto res = std::make_shared<xkv_backend_batch_result>();
    res->success_ = true;
    res->committed_ = true;
    res->handles = this->handles;
    res->stats = this->stats;
    if (owner) {
        res->executor = std::move(owner);
    }
    return res;
}

// ---------------------------------------------------------------------------
// Safe hot backend bulk readback keyed by physical hot slot
// ---------------------------------------------------------------------------

bool xkv_backend_hot_readback_batch(
    ggml_backend_t backend,
    const xkv_backend_hot_readback_request & req,
    xkv_backend_hot_readback_result & out,
    std::string * err) {
    out = xkv_backend_hot_readback_result();
    uint64_t total = 0;
    if (!mul_ok_u64((uint64_t) req.n_slots, (uint64_t) req.row_stride_bytes, total) ||
        total > (uint64_t) std::numeric_limits<size_t>::max()) {
        set_err(err, "xkv_backend hot readback: output size overflow");
        return false;
    }
    std::vector<uint8_t> dst((size_t) total);
    xkv_backend_batch_stats stats;
    if (!xkv_backend_hot_readback_span(backend, req, dst.data(), dst.size(), stats, err)) {
        return false;
    }
    out.bytes = std::move(dst);
    out.stats = stats;
    return true;
}

bool xkv_backend_hot_readback_span(
    ggml_backend_t backend,
    const xkv_backend_hot_readback_request & req,
    uint8_t * dst_bytes,
    size_t dst_capacity_bytes,
    xkv_backend_batch_stats & out_stats,
    std::string * err) {
    // Failure-atomic: out assigned ONLY on success.
    auto fail = [&](const std::string & m) -> bool {
        set_err(err, "xkv_backend hot readback: " + m);
        return false;
    };
    if (req.hot_tensor == nullptr) {
        return fail("null hot tensor");
    }
    if (req.row_stride_bytes == 0) {
        return fail("zero row stride");
    }
    if (req.hot_capacity == 0) {
        return fail("zero hot capacity");
    }
    if (req.n_slots == 0) {
        return fail("zero slots requested");
    }
    if (req.physical_slots == nullptr) {
        return fail("null physical slots array");
    }
    if (dst_bytes == nullptr) {
        return fail("null destination buffer");
    }
    const size_t tensor_nb1 = (size_t) req.hot_tensor->nb[1];
    if (tensor_nb1 != req.row_stride_bytes) {
        return fail("tensor stride (" + std::to_string(tensor_nb1) +
                    ") != requested row stride (" + std::to_string(req.row_stride_bytes) + ")");
    }
    const size_t tensor_bytes = ggml_nbytes(req.hot_tensor);
    uint64_t total_out = 0;
    if (!mul_ok_u64((uint64_t) req.n_slots, (uint64_t) req.row_stride_bytes, total_out) ||
        total_out > (uint64_t) std::numeric_limits<size_t>::max() ||
        (size_t) total_out > dst_capacity_bytes) {
        return fail("destination capacity smaller than requested bytes");
    }
    // Slot bounds check + uniqueness check (deterministic failure before any copy).
    for (uint32_t i = 0; i < req.n_slots; ++i) {
        const uint32_t slot = req.physical_slots[i];
        if (slot >= req.hot_capacity) {
            return fail("physical slot " + std::to_string(slot) + " >= capacity " +
                        std::to_string(req.hot_capacity));
        }
        uint64_t end = 0;
        if (!mul_ok_u64((uint64_t) slot, (uint64_t) tensor_nb1, end) ||
            !add_ok_u64(end, (uint64_t) req.row_stride_bytes, end) ||
            end > (uint64_t) tensor_bytes) {
            return fail("slot byte range exceeds tensor bound");
        }
        for (uint32_t j = 0; j < i; ++j) {
            if (req.physical_slots[j] == slot) {
                return fail("duplicate slot " + std::to_string(slot) + " in readback batch");
            }
        }
    }
    try {
        bool submitted = false;
        detail::submit_fence fence{backend, &submitted};
        for (uint32_t i = 0; i < req.n_slots; ++i) {
            const uint32_t slot = req.physical_slots[i];
            const size_t off = (size_t) slot * tensor_nb1;
            uint8_t * out_row = dst_bytes + (size_t) i * req.row_stride_bytes;
            if (backend != nullptr) {
                ggml_backend_tensor_get_async(backend, req.hot_tensor, out_row, off, req.row_stride_bytes);
                submitted = true;
            } else {
                ggml_backend_tensor_get(req.hot_tensor, out_row, off, req.row_stride_bytes);
            }
        }
        xkv_backend_batch_stats stats;
        if (backend != nullptr) {
            ggml_backend_synchronize(backend);
            stats.sync_count++;
            submitted = false; // flushed
        }
        out_stats = stats;
        return true;
    } catch (const std::bad_alloc &) {
        return fail("out of memory (outputs unchanged)");
    } catch (const std::exception & e) {
        return fail(std::string("failed: ") + e.what() + " (outputs unchanged)");
    } catch (...) {
        return fail("failed (unknown; outputs unchanged)");
    }
}

// ---------------------------------------------------------------------------
// Native selected-K fetch for TriAttention over DEVICE_OWNED handles
// ---------------------------------------------------------------------------

bool xkv_backend_tri_fetch_selected_k(
    ggml_backend_t backend,
    const xkv_backend_tri_fetch_handles & handles,
    const uint32_t * seg_rows,
    uint32_t n_rows,
    uint32_t head_dim,
    uint32_t kv_head,
    uint32_t n_kv_heads,
    uint32_t feature_offset_k,
    uint32_t feature_dim_k,
    float * dst_pre_rope,
    size_t dst_capacity_elements,
    std::string * err) {
    // Reference wrapper allocates scratch explicitly on behalf of test callers
    const size_t req_meta_bytes = (size_t)(4 * n_rows + n_rows + 8 + 5) * sizeof(int32_t);
    const size_t req_d2h_floats = (size_t)n_rows * (head_dim + head_dim);
    std::vector<uint8_t> meta_vec(req_meta_bytes);
    std::vector<float> d2h_vec(req_d2h_floats);
    xkv_backend_tri_fetch_scratch scratch;
    scratch.meta_buf = meta_vec.data();
    scratch.meta_buf_bytes = meta_vec.size();
    scratch.d2h_staging = d2h_vec.data();
    scratch.d2h_staging_floats = d2h_vec.size();
    xkv_backend_batch_stats stats;
    return xkv_backend_tri_fetch_selected_k_carved(
        backend, handles, seg_rows, n_rows, head_dim, kv_head, n_kv_heads,
        feature_offset_k, feature_dim_k, dst_pre_rope, dst_capacity_elements,
        &scratch, stats, err);
}

bool xkv_backend_tri_fetch_selected_k_carved(
    ggml_backend_t backend,
    const xkv_backend_tri_fetch_handles & handles,
    const uint32_t * seg_rows,
    uint32_t n_rows,
    uint32_t head_dim,
    uint32_t kv_head,
    uint32_t n_kv_heads,
    uint32_t feature_offset_k,
    uint32_t feature_dim_k,
    float * dst_pre_rope,
    size_t dst_capacity_elements,
    const xkv_backend_tri_fetch_scratch * scratch,
    xkv_backend_batch_stats & out_stats,
    std::string * err) {
    auto fail = [&](const std::string & m) -> bool {
        set_err(err, "xkv_backend tri fetch: " + m);
        return false;
    };
    if (n_rows == 0) {
        return true;
    }
    if (dst_pre_rope == nullptr) {
        return fail("null dst pointer");
    }
    if (seg_rows == nullptr) {
        return fail("null seg_rows pointer");
    }
    if (head_dim == 0 || (head_dim & 1u) != 0) {
        return fail("invalid head_dim (must be even nonzero)");
    }
    if (head_dim > 1024) {
        return fail("head_dim exceeds v1 op bound");
    }
    uint64_t req_elems = 0;
    if (!mul_ok_u64((uint64_t) n_rows, (uint64_t) head_dim, req_elems) ||
        req_elems > (uint64_t) std::numeric_limits<size_t>::max() ||
        (size_t) req_elems > dst_capacity_elements) {
        return fail("destination capacity too small for requested elements");
    }
    if (n_kv_heads > 0 && kv_head >= n_kv_heads) {
        return fail("kv_head out of range");
    }
    uint64_t head_start = 0;
    if (!mul_ok_u64((uint64_t) kv_head, (uint64_t) head_dim, head_start) ||
        !add_ok_u64(head_start, (uint64_t) feature_offset_k, head_start)) {
        return fail("head feature offset overflow");
    }
    if (head_dim > feature_dim_k || head_start + head_dim > (uint64_t) feature_offset_k + feature_dim_k) {
        return fail("head feature span exceeds layer feature_dim_k");
    }
    if (!handles.a_k || !handles.b_k) {
        return fail("missing a_k or b_k handle");
    }
    if (!handles.a_k->is_immutable() || !handles.b_k->is_immutable()) {
        return fail("handles must be immutable");
    }
    ggml_tensor * ak_t = handles.a_k->get_tensor();
    ggml_tensor * bk_t = handles.b_k->get_tensor();
    if (ak_t == nullptr || bk_t == nullptr) {
        return fail("empty a_k or b_k tensor");
    }

    // If device-owned, verify backend non-null and matching owning backend
    const bool is_device = handles.a_k->get_residency() == GGML_XKV_RES_DEVICE_OWNED;
    if (is_device) {
        if (backend == nullptr) {
            return fail("device-owned handles require non-null owning backend");
        }
        if (!detail::validate_readback_backend(*handles.a_k, backend, 0, err)) {
            return false;
        }
    }

    const codec_desc & dak = handles.a_k->get_desc();
    const codec_desc & dbk = handles.b_k->get_desc();
    if (dak.orient != orientation::token_major || dbk.orient != orientation::feature_major_transposed) {
        return fail("orientation mismatch (a_k must be token_major, b_k feature_major_transposed)");
    }
    if (dak.logical_shape.cols != dbk.logical_shape.cols) {
        return fail("rank mismatch between a_k and b_k");
    }
    const uint32_t rank_k = (uint32_t) dak.logical_shape.cols;
    const uint32_t n_seg_rows = (uint32_t) dak.logical_shape.rows;
    for (uint32_t i = 0; i < n_rows; ++i) {
        if (seg_rows[i] >= n_seg_rows) {
            return fail("row index " + std::to_string(seg_rows[i]) + " out of segment bound " +
                        std::to_string(n_seg_rows));
        }
    }
    if (head_start + head_dim > dbk.logical_shape.rows) {
        return fail("head feature span exceeds b_k rows");
    }

    try {
        const uint32_t n_sel = n_rows;
        const uint32_t dim_k = head_dim;
        const uint32_t dim_v = head_dim;
        const uint32_t rank_v = rank_k;

        ggml_init_params ip = {};
        ip.mem_size = ggml_tensor_overhead() * 24 + ggml_graph_overhead_custom(24, false);
        ip.no_alloc = true;
        ggml_context * ctx = ggml_init(ip);
        if (ctx == nullptr) {
            return fail("ggml_init failed");
        }
        std::unique_ptr<ggml_context, ggml_context_deleter> ctx_guard(ctx);

        ggml_xkv_reconstruct_params p = {};
        p.version = GGML_XKV_VERSION;
        p.n_sel = n_sel;
        p.n_groups = 1;
        p.rank_k = rank_k;
        p.rank_v = rank_v;
        p.dim_k = dim_k;
        p.dim_v = dim_v;
        p.rotary_dim = 0;
        p.rope_mode = GGML_XKV_ROPE_HALF;
        p.seed_k = (uint32_t) dak.seed;
        p.seed_v = (uint32_t) dak.seed;
        p.fp_combined = ggml_xkv_fp_combined(dak.type, dbk.type, dak.type, dbk.type, p.seed_k, p.seed_v);

        ggml_tensor * a_v_t = nullptr;
        ggml_tensor * b_v_t = nullptr;
        ggml_backend_buffer_t dummy_v_buf = nullptr;
        std::unique_ptr<ggml_backend_buffer, ggml_buffer_deleter> dummy_v_guard;
        if (handles.a_v && handles.b_v && handles.a_v->get_tensor() && handles.b_v->get_tensor()) {
            a_v_t = handles.a_v->get_tensor();
            b_v_t = handles.b_v->get_tensor();
        } else {
            ggml_backend_buffer_type_t buft = handles.a_k->get_owning_buft();
            if (buft == nullptr) {
                return fail("null owning buft on a_k");
            }
            a_v_t = ggml_new_tensor_2d(ctx, dak.type, (int64_t) dak.padded_shape.cols, (int64_t) dak.logical_shape.rows);
            b_v_t = ggml_new_tensor_2d(ctx, dbk.type, (int64_t) dbk.padded_shape.cols, (int64_t) dim_v);
            dummy_v_buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
            if (dummy_v_buf == nullptr) {
                return fail("alloc dummy V tensors failed");
            }
            dummy_v_guard.reset(dummy_v_buf);
            ggml_backend_buffer_clear(dummy_v_buf, 0);
        }

        // Zero-heap carved metadata buffers
        const size_t req_meta_bytes = (size_t)(4 * n_sel + n_sel + 8 + 5) * sizeof(int32_t);
        int32_t * meta_ptr = nullptr;
        if (!scratch || !scratch->meta_buf || scratch->meta_buf_bytes < req_meta_bytes) {
            return fail("preflight_oom: meta_buf scratch missing or smaller than required bytes");
        }
        meta_ptr = (int32_t *)scratch->meta_buf;
        int32_t * refs_data = meta_ptr;
        int32_t * pos_data  = refs_data + 4 * n_sel;
        int32_t * gm_data   = pos_data + n_sel;
        int32_t * lm_data   = gm_data + 8;

        for (uint32_t i = 0; i < n_sel; ++i) {
            refs_data[4 * i + 0] = (int32_t) seg_rows[i];
            refs_data[4 * i + 1] = 0;
            refs_data[4 * i + 2] = 0;
            refs_data[4 * i + 3] = 0;
            pos_data[i] = 0;
        }
        gm_data[0] = 1; gm_data[1] = 1; gm_data[2] = 0; gm_data[3] = (int32_t)dim_k;
        gm_data[4] = 0; gm_data[5] = (int32_t)dim_v; gm_data[6] = (int32_t)rank_k; gm_data[7] = (int32_t)rank_v;
        lm_data[0] = (int32_t)head_start; lm_data[1] = (int32_t)dim_k; lm_data[2] = 0; lm_data[3] = (int32_t)dim_v; lm_data[4] = 1;

        ggml_tensor * refs_t = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 4, n_sel);
        ggml_tensor * pos_t  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_sel);
        ggml_tensor * gm_t   = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 8, 1);
        ggml_tensor * lm_t   = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 5, 1);
        ggml_tensor * rt_t   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 0);

        ggml_tensor * out_t = ggml_xkv_reconstruct(ctx, ak_t, bk_t, a_v_t, b_v_t, refs_t, pos_t, gm_t, lm_t, rt_t, &p);
        if (out_t == nullptr) {
            return fail("ggml_xkv_reconstruct op create failed");
        }
        if (backend != nullptr && !ggml_backend_supports_op(backend, out_t)) {
            return fail("backend does not support reconstruct op");
        }

        ggml_backend_buffer_type_t buft = handles.a_k->get_owning_buft();
        ggml_backend_buffer_t meta_buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (meta_buf == nullptr) {
            return fail("alloc metadata tensors failed");
        }
        std::unique_ptr<ggml_backend_buffer, ggml_buffer_deleter> meta_guard(meta_buf);

        if (backend != nullptr) {
            // Queued async sets for metadata tensors + single synchronize
            ggml_backend_tensor_set_async(backend, refs_t, refs_data, 0, 4 * n_sel * sizeof(int32_t));
            ggml_backend_tensor_set_async(backend, pos_t, pos_data, 0, n_sel * sizeof(int32_t));
            ggml_backend_tensor_set_async(backend, gm_t, gm_data, 0, 8 * sizeof(int32_t));
            ggml_backend_tensor_set_async(backend, lm_t, lm_data, 0, 5 * sizeof(int32_t));
        } else {
            ggml_backend_tensor_set(refs_t, refs_data, 0, 4 * n_sel * sizeof(int32_t));
            ggml_backend_tensor_set(pos_t, pos_data, 0, n_sel * sizeof(int32_t));
            ggml_backend_tensor_set(gm_t, gm_data, 0, 8 * sizeof(int32_t));
            ggml_backend_tensor_set(lm_t, lm_data, 0, 5 * sizeof(int32_t));
        }

        ggml_cgraph * g = ggml_new_graph_custom(ctx, 24, false);
        ggml_build_forward_expand(g, out_t);

        if (backend != nullptr) {
            if (ggml_backend_graph_compute_async(backend, g) != GGML_STATUS_SUCCESS) {
                return fail("async graph compute failed");
            }
        }

        const size_t out_stride_bytes = (size_t) (dim_k + dim_v) * sizeof(float);
        const size_t k_row_bytes = (size_t) dim_k * sizeof(float);
        const size_t req_d2h_bytes = n_sel * out_stride_bytes;
        uint8_t * d2h_ptr = nullptr;
        if (!scratch || !scratch->d2h_staging || scratch->d2h_staging_floats * sizeof(float) < req_d2h_bytes) {
            return fail("preflight_oom: d2h_staging scratch missing or smaller than required bytes");
        }
        d2h_ptr = (uint8_t *)scratch->d2h_staging;

        if (backend != nullptr) {
            // Queue async get, then ONE synchronize for entire compute + D2H
            ggml_backend_tensor_get_async(backend, out_t, d2h_ptr, 0, req_d2h_bytes);
            ggml_backend_synchronize(backend);
            out_stats.sync_count = 1;
        } else {
            char ora_err[256] = {};
            if (!ggml_xkv_reconstruct_oracle(
                    ak_t->data, ak_t->type, ak_t->ne[0], ak_t->ne[1], ak_t->nb[0], ak_t->nb[1],
                    bk_t->data, bk_t->type, bk_t->ne[0], bk_t->ne[1], bk_t->nb[0], bk_t->nb[1],
                    a_v_t->data, a_v_t->type, a_v_t->ne[0], a_v_t->ne[1], a_v_t->nb[0], a_v_t->nb[1],
                    b_v_t->data, b_v_t->type, b_v_t->ne[0], b_v_t->ne[1], b_v_t->nb[0], b_v_t->nb[1],
                    refs_data, pos_data, gm_data, lm_data, 1,
                    nullptr, 0, &p, (float*)d2h_ptr, (int64_t)(dim_k + dim_v), (int64_t)n_sel,
                    sizeof(float), (size_t)(dim_k + dim_v) * sizeof(float), ora_err, sizeof(ora_err))) {
                return fail(std::string("reconstruct oracle failed: ") + ora_err);
            }
            out_stats.sync_count = 0;
        }

        for (uint32_t i = 0; i < n_sel; ++i) {
            const uint8_t * src_k_row = d2h_ptr + (size_t) i * out_stride_bytes;
            float * dst_k_row = dst_pre_rope + (size_t) i * head_dim;
            memcpy(dst_k_row, src_k_row, k_row_bytes);
        }
        return true;
    } catch (const std::bad_alloc &) {
        return fail("out of memory");
    } catch (const std::exception & e) {
        return fail(std::string("failed: ") + e.what());
    } catch (...) {
        return fail("failed (unknown)");
    }
}

} // namespace llama_xkv

// ---------------------------------------------------------------------------
// Selected-hot canonicalize helper (dequant + inv WHT + inv RoPE)
// ---------------------------------------------------------------------------

namespace llama_xkv {

bool xkv_backend_hot_canonicalize_batch(
    ggml_backend_t backend,
    const xkv_backend_hot_canonicalize_request & req,
    float * dst_canonical,
    size_t dst_capacity_elements,
    xkv_backend_batch_stats & out_stats,
    std::string * err) {
    // Reference wrapper allocates scratch explicitly for test callers
    const size_t req_meta_bytes = (size_t)req.n_rows * sizeof(int32_t);
    const size_t total_floats = (size_t)req.n_rows * req.head_dim * req.n_kv_heads;
    std::vector<uint8_t> meta_vec(req_meta_bytes);
    std::vector<float> d2h_vec(total_floats);
    xkv_backend_hot_canonicalize_scratch scratch;
    scratch.meta_buf = meta_vec.data();
    scratch.meta_buf_bytes = meta_vec.size();
    scratch.d2h_staging = d2h_vec.data();
    scratch.d2h_staging_floats = d2h_vec.size();
    return xkv_backend_hot_canonicalize_carved(
        backend, req, dst_canonical, dst_capacity_elements,
        &scratch, out_stats, err);
}

bool xkv_backend_hot_canonicalize_carved(
    ggml_backend_t backend,
    const xkv_backend_hot_canonicalize_request & req,
    float * dst_canonical,
    size_t dst_capacity_elements,
    const xkv_backend_hot_canonicalize_scratch * scratch,
    xkv_backend_batch_stats & out_stats,
    std::string * err) {
    auto fail = [&](const std::string & m) -> bool {
        set_err(err, "xkv_backend hot canonicalize: " + m);
        return false;
    };
    if (req.n_rows == 0) {
        return true;
    }
    if (!req.hot_kv || !req.physical_slots || !req.storage_positions || !dst_canonical) {
        return fail("null required inputs");
    }
    if (req.head_dim == 0 || req.n_kv_heads == 0 || req.kv_head >= req.n_kv_heads) {
        return fail("invalid head dimensions");
    }
    uint64_t req_elems = 0;
    if (!mul_ok_u64((uint64_t)req.n_rows, (uint64_t)req.head_dim, req_elems) ||
        (size_t)req_elems > dst_capacity_elements) {
        return fail("dst capacity too small");
    }
    for (uint32_t i = 0; i < req.n_rows; ++i) {
        if (req.physical_slots[i] >= req.hot_capacity) {
            return fail("physical slot out of bounds");
        }
    }
    const uint32_t pad_hd = req.padded_head_dim > 0 ? req.padded_head_dim : req.head_dim;

    // Prepare canonicalize op params
    ggml_xkv_canonicalize_params p = {};
    p.version = GGML_XKV_FACTOR_VERSION;
    p.n_rows = req.n_rows;
    p.n_layers = 1;
    p.n_heads = req.n_kv_heads;
    p.head_dim = req.head_dim;
    p.padded_head_dim = pad_hd;
    p.total_feat = req.head_dim * req.n_kv_heads;
    p.rotary_dim = req.rotary_dim;
    p.rope_mode = req.rope_mode;
    p.input_type = (uint32_t)req.hot_kv->type;
    p.is_k = req.is_k ? 1 : 0;
    p.hadamard_dim = req.hadamard_dim;

    const bool is_host_buf = req.hot_kv->buffer && ggml_backend_buffer_is_host(req.hot_kv->buffer);

    // If not host and no backend supplied, refuse immediately (never deref device ptr)
    if (!is_host_buf && backend == nullptr) {
        return fail("device-resident hot tensor requires non-null owning backend");
    }

    // Carve or fallback host scratch for i_rows [n_rows * sizeof(int32_t)]
    const size_t req_meta_bytes = (size_t)req.n_rows * sizeof(int32_t);
    int32_t * i_rows_ptr = nullptr;
    if (!scratch || !scratch->meta_buf || scratch->meta_buf_bytes < req_meta_bytes) {
        return fail("preflight_oom: meta_buf scratch missing or smaller than required bytes");
    }
    i_rows_ptr = (int32_t*)scratch->meta_buf;
    for (uint32_t i = 0; i < req.n_rows; ++i) {
        i_rows_ptr[i] = (int32_t)req.physical_slots[i];
    }

    // Carve or fallback staging buffer for full_out [n_rows * total_feat floats]
    const size_t total_floats = (size_t)req.n_rows * p.total_feat;
    float * full_out_ptr = nullptr;
    if (!scratch || !scratch->d2h_staging || scratch->d2h_staging_floats < total_floats) {
        return fail("preflight_oom: d2h_staging scratch missing or smaller than required floats");
    }
    full_out_ptr = scratch->d2h_staging;

    if (backend != nullptr && !is_host_buf) {
        // REAL BACKEND-NATIVE PATH:
        // 1. Build metadata tensors in small context
        // 2. Allocate metadata in backend buft
        // 3. Queue async sets for metadata
        // 4. Run ggml_xkv_canonicalize on device
        // 5. One sync/D2H into full_out
        try {
            ggml_init_params ip = {};
            ip.mem_size = ggml_tensor_overhead() * 16 + ggml_graph_overhead_custom(16, false);
            ip.no_alloc = true;
            ggml_context * ctx = ggml_init(ip);
            if (!ctx) return fail("ggml_init failed");
            std::unique_ptr<ggml_context, ggml_context_deleter> ctx_guard(ctx);

            ggml_tensor * rows_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, req.n_rows);
            ggml_tensor * pos_t  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, req.n_rows);
            const uint32_t fc = req.rotary_dim / 2;
            ggml_tensor * rope_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, fc ? (int64_t)fc * 2 : 0);
            ggml_tensor * had_t  = req.hadamard_dim ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, req.hadamard_dim, req.hadamard_dim) : ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 0);
            ggml_tensor * st_t   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);

            ggml_tensor * out_t = ggml_xkv_canonicalize(ctx, (struct ggml_tensor *)req.hot_kv, rows_t, pos_t, rope_t, had_t, st_t, &p);
            if (!out_t) return fail("canonicalize op build failed");
            if (!ggml_backend_supports_op(backend, out_t)) return fail("backend does not support canonicalize op");

            ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
            ggml_backend_buffer_t mbuf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
            if (!mbuf) return fail("metadata buffer alloc failed");
            std::unique_ptr<ggml_backend_buffer, ggml_buffer_deleter> mbuf_guard(mbuf);

            // Queue metadata uploads async
            ggml_backend_tensor_set_async(backend, rows_t, i_rows_ptr, 0, req.n_rows * sizeof(int32_t));
            ggml_backend_tensor_set_async(backend, pos_t, req.storage_positions, 0, req.n_rows * sizeof(int32_t));
            if (req.rope_tables && req.rope_nelements > 0) {
                ggml_backend_tensor_set_async(backend, rope_t, req.rope_tables, 0, req.rope_nelements * sizeof(float));
            }
            if (req.hadamard && req.hadamard_dim > 0) {
                ggml_backend_tensor_set_async(backend, had_t, req.hadamard, 0, (size_t)req.hadamard_dim * req.hadamard_dim * sizeof(float));
            }
            int32_t zero_status = 0;
            ggml_backend_tensor_set_async(backend, st_t, &zero_status, 0, sizeof(int32_t));

            ggml_cgraph * g = ggml_new_graph_custom(ctx, 16, false);
            ggml_build_forward_expand(g, out_t);
            if (ggml_backend_graph_compute_async(backend, g) != GGML_STATUS_SUCCESS) {
                return fail("async device canonicalize graph compute failed");
            }

            // Queue async get, then ONE synchronize for entire compute + D2H
            ggml_backend_tensor_get_async(backend, out_t, full_out_ptr, 0, total_floats * sizeof(float));
            ggml_backend_synchronize(backend);
            out_stats.sync_count = 1;

            // Stride-extract requested head
            const size_t head_feat_off = (size_t)req.kv_head * req.head_dim;
            for (uint32_t r = 0; r < req.n_rows; ++r) {
                const float * src_head = full_out_ptr + (size_t)r * p.total_feat + head_feat_off;
                float * dst_head = dst_canonical + (size_t)r * req.head_dim;
                std::memcpy(dst_head, src_head, (size_t)req.head_dim * sizeof(float));
            }
            return true;
        } catch (const std::exception & e) {
            return fail(std::string("backend canonicalize failed: ") + e.what());
        }
    }

    char ora_err[256] = {};
    // Oracle CPU execution: dequant + inv WHT + inv RoPE
    if (!ggml_xkv_canonicalize_cpu_oracle(
            req.hot_kv->data, req.hot_kv->type,
            i_rows_ptr, req.storage_positions, /*pos_is_64=*/0, req.n_rows,
            req.rope_tables, req.rope_nelements,
            req.hadamard, req.hadamard_dim,
            &p, full_out_ptr,
            ora_err, sizeof(ora_err))) {
        return fail(std::string("oracle failed: ") + ora_err);
    }

    // Stride-extract only the requested kv_head:
    // full_out_ptr layout: for each row r in 0..n_rows-1, row span is [p.total_feat] floats.
    // kv_head features start at kv_head * head_dim and span head_dim floats.
    const size_t head_feat_off = (size_t)req.kv_head * req.head_dim;
    for (uint32_t r = 0; r < req.n_rows; ++r) {
        const float * src_head = full_out_ptr + (size_t)r * p.total_feat + head_feat_off;
        float * dst_head = dst_canonical + (size_t)r * req.head_dim;
        std::memcpy(dst_head, src_head, (size_t)req.head_dim * sizeof(float));
    }

    out_stats.sync_count = 0; // CPU oracle path
    return true;
}

} // namespace llama_xkv
