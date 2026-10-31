#include <ggml-backend.h>
#include <ggml-cuda.h>
#include "tp5-meta-submit-ready.h"
#include <ggml-relay.h>
#include <gguf.h>
#include <llama.h>
#include "llama-memory-hybrid-idx.h"
// The selected original libllama exports these stable fork MTP staging APIs;
// only this isolated one-object candidate uses them.
void llama_set_embeddings_nextn(llama_context * ctx, bool value, bool masked);
float * llama_get_embeddings_nextn_ith(llama_context * ctx, int32_t index);
#include <tq/relay/ledger.h>
#include <tq/sampling/shortlist.h>
#include <tq/scheduling/admission.h>
#include <tq/text/stream.h>
#include <tq/tp5/runtime.h>

#include "ordered-fold.h"
#include "dw-thru.h"
#include "dw_event_trace.h"
#include "dw_device_wait.h"

#include <algorithm>
#include <dlfcn.h>
#include <sched.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <exception>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>

namespace tq::tp5 {
namespace {
using clock_type = std::chrono::steady_clock;

std::uint64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(clock_type::now().time_since_epoch()).count();
}

// ASYNC-PROTOCOL: host waits on device completion must busy-wait, not sleep
// on an interrupt/signal (HIP default is ScheduleAuto -> may block). Resolved
// at runtime from the already-linked HIP runtime; no new link dependency.
void tp5_force_hip_spin_wait() noexcept {
    const char * env = std::getenv("TP5_HIP_SCHEDULE_SPIN");
    if (env && std::string(env) == "0") return;
    using set_flags_t = int (*)(unsigned);
    using set_dev_t = int (*)(int);
    using count_t = int (*)(int *);
    void * self = dlopen("libamdhip64.so", RTLD_NOW | RTLD_NOLOAD);
    if (!self) self = dlopen("libamdhip64.so.7", RTLD_NOW | RTLD_NOLOAD);
    if (!self) self = dlopen("libamdhip64.so.6", RTLD_NOW | RTLD_NOLOAD);
    void * scope = self ? self : RTLD_DEFAULT;
    auto set_flags = reinterpret_cast<set_flags_t>(dlsym(scope, "hipSetDeviceFlags"));
    auto set_dev = reinterpret_cast<set_dev_t>(dlsym(scope, "hipSetDevice"));
    auto count = reinterpret_cast<count_t>(dlsym(scope, "hipGetDeviceCount"));
    if (!set_flags || !set_dev || !count) {
        std::fprintf(stderr, "TP5_HIP_SCHEDULE_SPIN unavailable (symbols not found)\n");
        return;
    }
    int n = 0;
    if (count(&n) != 0) n = 0;
    int ok = 0;
    for (int d = 0; d < n; ++d) {
        if (set_dev(d) == 0 && set_flags(0x1 /* hipDeviceScheduleSpin */) == 0) ++ok;
    }
    if (n > 0) set_dev(0);
    std::fprintf(stderr, "TP5_HIP_SCHEDULE_SPIN devices=%d spin_set=%d\n", n, ok);
}

void pause_cpu() noexcept {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#else
    std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

// ASYNC-PROTOCOL / CPU budget: busy-waiting is confined to ONE dedicated
// physical core per window (worker on one hyperthread, relay courier on its
// sibling). Every other thread of this process is excluded from those CPUs,
// and all remaining CPUs stay free for ngram / other work.
// TP5_SPIN_CORES: unset = auto (last physical cores), "0" = no pinning,
// or explicit "worker:courier,worker:courier" per window, e.g. "21:43,20:42".
void require(bool ok, const char * why);
struct tp5_spin_slot { int worker = -1; int courier = -1; };
std::vector<tp5_spin_slot> g_tp5_spin_slots;

int tp5_sibling_of(int cpu) {
    const std::string path = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/thread_siblings_list";
    if (FILE * f = std::fopen(path.c_str(), "r")) {
        char buf[64] = {};
        const bool got = std::fgets(buf, sizeof(buf), f) != nullptr;
        std::fclose(f);
        if (got) {
            int a = -1, b = -1;
            if (std::sscanf(buf, "%d%*[,-]%d", &a, &b) == 2) return a == cpu ? b : a;
        }
    }
    return cpu;
}

void tp5_plan_spin_cores(std::size_t windows) {
    g_tp5_spin_slots.clear();
    const char * env = std::getenv("TP5_SPIN_CORES");
    if (env && std::string(env) == "0") {
        std::fprintf(stderr, "TP5_SPIN_CORES disabled\n");
        return;
    }
    const long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    if (env && *env) {
        std::string spec(env);
        std::size_t pos = 0;
        while (pos < spec.size()) {
            const auto end = spec.find(',', pos);
            const std::string item = spec.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
            tp5_spin_slot slot;
            require(std::sscanf(item.c_str(), "%d:%d", &slot.worker, &slot.courier) == 2 &&
                        slot.worker >= 0 && slot.courier >= 0 && slot.worker < ncpu && slot.courier < ncpu,
                    "TP5_SPIN_CORES must be worker:courier[,worker:courier]");
            g_tp5_spin_slots.push_back(slot);
            if (end == std::string::npos) break;
            pos = end + 1;
        }
        require(g_tp5_spin_slots.size() == windows, "TP5_SPIN_CORES needs one worker:courier pair per window");
    } else {
        // Highest-numbered CPUs; skip CPUs already claimed as a sibling.
        std::vector<bool> used(std::size_t(ncpu), false);
        for (long cpu = ncpu - 1; cpu >= 0 && g_tp5_spin_slots.size() < windows; --cpu) {
            const int sib = tp5_sibling_of(int(cpu));
            const int lo = std::min(int(cpu), sib), hi = std::max(int(cpu), sib);
            if (used[std::size_t(lo)] || used[std::size_t(hi)]) continue;
            used[std::size_t(lo)] = used[std::size_t(hi)] = true;
            g_tp5_spin_slots.push_back({ lo, hi });
        }
        require(g_tp5_spin_slots.size() == windows, "not enough CPUs for dedicated spin cores");
    }
    // Exclude spin CPUs from this thread; threads created later inherit it.
    cpu_set_t rest;
    CPU_ZERO(&rest);
    for (long cpu = 0; cpu < ncpu; ++cpu) CPU_SET(int(cpu), &rest);
    for (const auto & slot : g_tp5_spin_slots) {
        CPU_CLR(slot.worker, &rest);
        CPU_CLR(slot.courier, &rest);
    }
    require(CPU_COUNT(&rest) > 0, "spin cores leave no CPU for the rest of the process");
    require(sched_setaffinity(0, sizeof(rest), &rest) == 0, "cannot restrict process threads away from spin cores");
    for (std::size_t w = 0; w < g_tp5_spin_slots.size(); ++w) {
        std::fprintf(stderr, "TP5_SPIN_CORES window=%zu worker_cpu=%d courier_cpu=%d others_cpus=%d\n",
            w, g_tp5_spin_slots[w].worker, g_tp5_spin_slots[w].courier, CPU_COUNT(&rest));
    }
}

thread_local int tl_tp5_window = -1;

// Busy-wait segments of a window worker migrate onto the window's dedicated
// CPU and return to the shared mask afterwards, so host prepare / draft work
// (which may fork CPU thread teams) never inherits a one-CPU mask.
class scoped_spin_cpu {
  public:
    scoped_spin_cpu() {
        if (tl_tp5_window < 0 || std::size_t(tl_tp5_window) >= g_tp5_spin_slots.size()) return;
        if (sched_getaffinity(0, sizeof(saved_), &saved_) != 0) return;
        cpu_set_t one;
        CPU_ZERO(&one);
        CPU_SET(g_tp5_spin_slots[std::size_t(tl_tp5_window)].worker, &one);
        active_ = sched_setaffinity(0, sizeof(one), &one) == 0;
    }
    ~scoped_spin_cpu() {
        if (active_) sched_setaffinity(0, sizeof(saved_), &saved_);
    }
    scoped_spin_cpu(const scoped_spin_cpu &) = delete;
    scoped_spin_cpu & operator=(const scoped_spin_cpu &) = delete;
  private:
    cpu_set_t saved_{};
    bool active_ = false;
};

void tp5_pin_this_thread(std::size_t window, bool courier) {
    if (window >= g_tp5_spin_slots.size()) return;
    cpu_set_t one;
    CPU_ZERO(&one);
    CPU_SET(courier ? g_tp5_spin_slots[window].courier : g_tp5_spin_slots[window].worker, &one);
    require(sched_setaffinity(0, sizeof(one), &one) == 0, "cannot pin spin thread");
}

// ASYNC-PROTOCOL (2026-10-27): FIFO ticket spin lock. No futex sleep, no
// scheduler wake ("doorbell"): a waiter busy-waits on a shared counter and
// enters the moment the previous holder's release store becomes visible.
class spin_ticket_mutex {
  public:
    void lock() noexcept {
        const std::uint32_t mine = next_.fetch_add(1, std::memory_order_relaxed);
        if (serving_.load(std::memory_order_acquire) == mine) return;
        scoped_spin_cpu dedicated;
        while (serving_.load(std::memory_order_acquire) != mine) pause_cpu();
    }
    bool try_lock() noexcept {
        std::uint32_t s = serving_.load(std::memory_order_acquire);
        std::uint32_t n = s;
        return next_.compare_exchange_strong(n, s + 1, std::memory_order_acquire, std::memory_order_relaxed);
    }
    void unlock() noexcept { serving_.fetch_add(1, std::memory_order_release); }
  private:
    alignas(64) std::atomic<std::uint32_t> next_{0};
    alignas(64) std::atomic<std::uint32_t> serving_{0};
};
using submission_mutex_type = spin_ticket_mutex;

[[noreturn]] void retain_and_exit(const char * why) {
    std::fprintf(stderr, "TP5_RETAINED_IN_FLIGHT: %s; native owners are not released\n", why);
    std::fflush(stderr);
    std::_Exit(125);
}

void require(bool ok, const char * why) {
    if (!ok) {
        throw std::runtime_error(why);
    }
}

bool is_swift_model(const std::string & path) {
    // Inspect public GGUF metadata before choosing device placement. File names
    // and a failed single-GPU allocation cannot select the model's contract.
    std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(
        gguf_init_from_file(path.c_str(), {true, nullptr}), gguf_free);
    require(metadata != nullptr, "cannot read model metadata");
    const auto key = gguf_find_key(metadata.get(), "general.architecture");
    require(key >= 0 && gguf_get_kv_type(metadata.get(), key) == GGUF_TYPE_STRING,
            "model has no valid architecture metadata");
    return std::string(gguf_get_val_str(metadata.get(), key)) == "qwen4exp";
}

void check_ledger(relay::ledger_result value) {
    if (value != relay::ledger_result::ok) {
        throw std::runtime_error(std::string("relay ownership violation: ") + relay::to_string(value));
    }
}

options validate(const options & value) {
    options out = value;
    require(!out.model.empty(), "model path is empty");
    require(out.execution == mode::relay || out.execution == mode::reference, "unknown execution mode");
    require(out.wire == wire_type::f32 || out.wire == wire_type::f16, "unknown relay wire format");
    require(out.execution == mode::relay || out.wire == wire_type::f32,
            "F16 relay wire requires relay mode");
    require(!out.prompts.empty() && out.prompts.size() <= 4, "one through four requests are required");
    require(out.windows >= 1 && out.windows <= 2 && out.windows <= out.prompts.size(), "invalid window count");
    if (!out.hc_owner_offsets.empty()) {
        require(out.execution == mode::relay && out.hc_owner_offsets.size() == out.windows &&
                    out.prompts.size() == out.windows,
                "HC owner offsets require relay mode, one offset and one request per window");
        for (const auto offset : out.hc_owner_offsets) {
            require(offset < 5, "HC owner offset must be in [0,4]");
        }
    }
    require(out.predict && out.predict <= 4096, "predict must be in [1,4096]");
    require(out.context_per_request >= 32 && out.context_per_request <= 262144, "context must be in [32,262144]");
    require(out.threads && out.threads <= GGML_MAX_N_THREADS, "invalid CPU thread count");
    require(out.deadline_seconds && out.deadline_seconds <= 3600, "invalid execution deadline");
    require(out.delay_window < out.windows && out.delay_publish_ms <= 10000, "invalid controlled publish delay");
    require(out.cancel_request >= -1 && out.cancel_request < static_cast<std::int32_t>(out.prompts.size()),
            "invalid cancellation request");
    require(out.cancel_after > 0 && out.cancel_after <= out.predict, "invalid cancellation position");
    require(out.abort_window >= -1 && out.abort_window < static_cast<std::int32_t>(out.windows),
            "invalid abort window");
    require(out.abort_epoch > 0 && (out.abort_window >= 0 || out.abort_epoch == 1), "invalid abort epoch");
    if (!out.teacher_tokens.empty()) {
        require(out.teacher_tokens.size() == out.prompts.size(), "teacher request count differs");
        for (const auto & row : out.teacher_tokens) {
            require(row.size() == out.predict, "teacher trajectory must provide predict tokens per request");
        }
    }
    for (const auto & prompt : out.prompts) {
        require(!prompt.empty() && prompt.size() <= 1024 * 1024, "prompt size must be in [1,1048576]");
    }
    return out;
}

void configure_hc_owner_layout(const options & opts) {
    // Set the physical layout before backend initialization or any model load.
    // An environment-only request is rejected instead of silently mirroring
    // weights without a matching private execution plan.
    const char * hc_layout = std::getenv("GGML_TP5_HC_CONTEXT_OWNER");
    if (opts.hc_owner_offsets.empty()) {
        require(hc_layout == nullptr || std::string(hc_layout) == "0",
                "GGML_TP5_HC_CONTEXT_OWNER requires explicit --hc-owner-offsets (or unset/0)");
    } else {
        require(hc_layout == nullptr || std::string(hc_layout) == "1",
                "--hc-owner-offsets conflicts with GGML_TP5_HC_CONTEXT_OWNER; unset it or use 1");
        require(::setenv("GGML_TP5_HC_CONTEXT_OWNER", "1", 1) == 0,
                "cannot enable the HC context-owner weight layout before model loading");
    }
}

struct backend_owner {
    backend_owner() {
        tp5_force_hip_spin_wait();
        ggml_backend_load_all();
        llama_backend_init();
    }

    ~backend_owner() { llama_backend_free(); }
};

struct model_deleter {
    void operator()(llama_model * p) const { llama_model_free(p); }
};

struct context_deleter {
    void operator()(llama_context * p) const { llama_free(p); }
};

using model_ptr   = std::unique_ptr<llama_model, model_deleter>;
using context_ptr = std::unique_ptr<llama_context, context_deleter>;

struct failure_state {
    std::atomic<bool>  stopped{ false };
    std::mutex         mutex;
    std::exception_ptr first;

    void fail(std::exception_ptr error) {
        std::lock_guard<std::mutex> lock(mutex);
        if (!first) {
            first = std::move(error);
        }
        stopped.store(true, std::memory_order_release);
    }
};

struct channel_state {
    relay::channel_ledger       ledger;
    std::optional<relay::lease> held;
    std::uint64_t               epoch = 0;
    std::uint64_t               elements = 0;
    std::uint64_t               wire_bytes = 0;
    std::uint32_t               computed_mask = 0;

    channel_state(std::uint64_t run, std::uint64_t plan, std::uint64_t channel) : ledger(run, plan, channel, 5, 1) {}
};

class mtp_phase_pair {
  public:
    mtp_phase_pair(std::size_t participants, bool enabled) :
        participants_(participants), enabled_(enabled) {}

    bool meet(const std::atomic<bool> & stopped) {
        if (!enabled_) return true;
        std::unique_lock<std::mutex> lock(mutex_);
        if (cancelled_ || stopped.load()) return false;
        const auto generation = generation_;
        ++arrivals_;
        if (arrivals_ == participants_) {
            arrivals_ = 0;
            ++generation_;
            changed_.notify_all();
            return true;
        }
        // ASYNC-PROTOCOL: no condvar sleep (1 ms scheduler doorbell); spin.
        while (generation_ == generation && !cancelled_ && !stopped.load()) {
            lock.unlock();
            pause_cpu();
            lock.lock();
        }
        if (stopped.load()) {
            cancelled_ = true;
            changed_.notify_all();
        }
        return !cancelled_;
    }

    void retire() noexcept {
        if (!enabled_) return;
        std::lock_guard<std::mutex> lock(mutex_);
        if (participants_ > 0) --participants_;
        if (arrivals_ == participants_) {
            arrivals_ = 0;
            ++generation_;
        }
        changed_.notify_all();
    }

    void cancel() noexcept {
        if (!enabled_) return;
        std::lock_guard<std::mutex> lock(mutex_);
        cancelled_ = true;
        changed_.notify_all();
    }

  private:
    std::mutex mutex_;
    std::condition_variable changed_;
    std::size_t participants_;
    std::size_t arrivals_ = 0;
    std::uint64_t generation_ = 0;
    const bool enabled_;
    bool cancelled_ = false;
};


// DUALWIN-MUTEX-MEET (TP5_MTP_GATE_METRICS, default off): counters-only
// instrumentation of the submission gate and phase_pair::meet() waits.
// Enabled only by the exact value "1"; read once at static initialization
// (namespace scope, before main), later env changes are not observed.
// Off path adds one bool branch per measured site.
static const bool gate_metrics_req = [] {
    const char * env = std::getenv("TP5_MTP_GATE_METRICS");
    return env != nullptr && std::strcmp(env, "1") == 0;
}();

// DUALWIN-COMBO (TP5_COURIER_HITS, default off): per-window courier hit
// counters + Q8 hit observability. Enabled only by the exact value "1";
// read once at static initialization (namespace scope, before main), later
// env changes are not observed. Off path adds one bool branch per site.
// Q8 hits reuse the existing fusion_stats.q8_cache_hits counter read through
// ggml_backend_cuda_fusion_count(backend, "q8_cache_hits") at the per-window
// dump; no new Q8 counter is added. Observable only on directly-launched
// (non-replayed) cycles; not exercised under CUDA-graph capture.
static const bool courier_hits_req = [] {
    const char * env = std::getenv("TP5_COURIER_HITS");
    return env != nullptr && std::strcmp(env, "1") == 0;
}();

// DUALWIN-OVERLAP-METER (GGML_TP5_DW_OVERLAP_METER, default off): real
// wall-clock dual-window overlap + lock-hold metering. Host in-flight
// intervals reuse dw_thru_counter verbatim (begin at decode submission,
// end at completion-wait return); lock-hold metering reuses the existing
// gate_metrics counters (no second same-kind host counter). Device-side
// HIP event metering lives in the relay provider. Enabled only by the
// exact value "1"; read once at static initialization (namespace scope,
// before main), later env changes are not observed. Off path adds one
// bool branch per site.
static const bool overlap_meter_req = [] {
    const char * env = std::getenv("GGML_TP5_DW_OVERLAP_METER");
    return env != nullptr && std::strcmp(env, "1") == 0;
}();
static dw_thru_counter dw_overlap_meter;  // host in-flight only, never device evidence

// DUALWIN-BACKEND-EVENT: the Meta backend forwards event record/wait to its
// first simple child. Its identity and stream relationship to relay work
// have not been verified. Successful target decode submits record
// asynchronously; drain synchronizes the event on the host and retains
// llama_synchronize. Exact GGML_TP5_DW_BACKEND_EVENT=1 enables this at
// static initialization; other values create no event and print no receipt.
// CPU or unsupported children can return nullptr. Draft (NextN) contexts are
// not wired. Forward-skip honesty: a Meta forward silently skips a child
// whose interface member is null. In this tree that cannot hit the gated
// sites: an event exists only if the first child implemented event_new, and
// the only event-capable first children are the HIP/CUDA backends, which
// register event_record AND event_wait together (ggml-cuda.cu backend iface
// registers both under one gate condition); CPU children register neither,
// so their event is null and every site is skipped by the event-null guard
// before any counter moves. A future first child implementing event_new
// without event_record/event_wait would be counted as issued while the
// forward silently skipped — the receipt counters mean "issued", the
// device-side meaning stays with the rocprof route.
static const bool dw_backend_event_req = [] {
    const char * env = std::getenv("GGML_TP5_DW_BACKEND_EVENT");
    return env != nullptr && std::strcmp(env, "1") == 0;
}();
// DUALWIN-DEVICE-WAIT (GGML_TP5_DW_DEVICE_WAIT=1): gate parsing, peer
// selection and the wait/decision state machine live in dw_device_wait.h
// (shared verbatim with the fork runtime; see there for the acyclicity and
// no-unrecorded-wait reasoning). Under this gate the runtime issues real
// ggml_backend_event_wait calls at the consumption points: this window's
// first-child stream waits for the PEER window's most recent unwaited
// record, so the following host synchronize transitively covers the peer's
// latest recorded device work too (device-side dual completion). It only
// ever delays consumption behind work the peer already submitted; it never
// weakens an exit predicate. It requires GGML_TP5_DW_BACKEND_EVENT=1 (the
// events only exist under that gate); with the device-wait gate alone every
// site is a counted no-op.
static const bool dw_device_wait_req = dw_device_wait::gate();
struct gate_metrics {
    // Whole-decode submission-mutex hold (outer gate, target decodes).
    std::uint64_t hold_ns = 0;
    std::uint64_t holds   = 0;
    // Replay decode gate hold (MTP replay path).
    std::uint64_t replay_hold_ns = 0;
    std::uint64_t replay_holds   = 0;
    // Submit-only-gate enqueue hold (Meta submit-ready -> submitted).
    std::uint64_t enqueue_ns = 0;
    std::uint64_t enqueues   = 0;
    // phase_pair::meet() waits, before/after target.
    std::uint64_t meet_before_ns = 0, meet_before_calls = 0;
    std::uint64_t meet_after_ns  = 0, meet_after_calls  = 0;
    // Submit-only gate: time spent waiting for the runtime-wide ticket lock.
    // Always acquired, including under GGML_TP5_DW_SERIALCUT=1.
    std::uint64_t enqueue_wait_ns = 0;
    std::uint64_t enqueue_waits   = 0;
};

// DUALWIN-SERIALCUT: canary for the native five-rank submission. The
// runtime-wide serial gate and every provider-level submit mutex are always
// taken, so this detector is structurally zero.
// Updated only when TP5_MTP_GATE_METRICS=1.
struct mtp_submit_observation {
    std::atomic<unsigned>      active{ 0 };
    std::atomic<unsigned>      peak{ 0 };
    std::atomic<std::uint64_t> concurrent_events{ 0 };
    void enter() noexcept {
        const unsigned now = active.fetch_add(1, std::memory_order_acq_rel) + 1;
        unsigned seen = peak.load(std::memory_order_relaxed);
        while (seen < now && !peak.compare_exchange_weak(seen, now, std::memory_order_relaxed)) {
        }
        if (now > 1) concurrent_events.fetch_add(1, std::memory_order_relaxed);
    }
    void leave() noexcept { active.fetch_sub(1, std::memory_order_acq_rel); }
};

class mtp_submit_ready_scope {
  public:
    mtp_submit_ready_scope(bool enabled, bool release_after_submit, mtp_phase_pair & pair,
                          std::unique_lock<submission_mutex_type> & submission,
                          const std::atomic<bool> & stopped, unsigned window,
                          std::uint64_t call, std::uint64_t & waits, std::uint64_t & hits,
                          std::uint64_t & submitted_hits)
        : enabled_(enabled), release_after_submit_(release_after_submit), pair_(pair),
          submission_(submission), stopped_(stopped), window_(window), call_(call),
          waits_(waits), hits_(hits), submitted_hits_(submitted_hits) {
        require(!release_after_submit_ || enabled_, "post-submit unlock requires the ready pair");
        if (enabled_) {
            require(submission_.owns_lock(), "ready pair requires the existing submission gate");
            require(ggml_backend_meta_set_submit_ready_hook(&arrive, this),
                    "nested Meta submit-ready hook");
            if (release_after_submit_ &&
                    !ggml_backend_meta_set_submitted_hook(&submitted, this)) {
                ggml_backend_meta_set_submit_ready_hook(nullptr, nullptr);
                require(false, "nested Meta submitted hook");
            }
        }
    }
    ~mtp_submit_ready_scope() {
        if (release_after_submit_) ggml_backend_meta_set_submitted_hook(nullptr, nullptr);
        if (enabled_) ggml_backend_meta_set_submit_ready_hook(nullptr, nullptr);
    }
    void check(int decoded) {
        if (enabled_ && decoded == 0) {
            require(called_ == 1, "target decode did not visit exactly one Meta submit-ready point");
            require(!release_after_submit_ || submitted_ == 1,
                    "target decode did not visit exactly one successful Meta submit point");
        }
    }
  private:
    static bool arrive(void * opaque) {
        auto & self = *static_cast<mtp_submit_ready_scope *>(opaque);
        if (++self.called_ != 1 || !self.submission_.owns_lock()) {
            self.pair_.cancel();
            return false;
        }
        const auto begin = now_ns();
        self.submission_.unlock();
        bool met;
        try {
            met = self.pair_.meet(self.stopped_);
        } catch (...) {
            self.submission_.lock();
            throw;
        }
        self.submission_.lock();
        const auto released = now_ns();
        self.waits_ += released - begin;
        ++self.hits_;
        if (std::getenv("TP5_MTP_READY_PAIR_TRACE")) {
            std::fprintf(stderr,"TP5_MTP_READY window=%u call=%llu arrive_ns=%llu release_ns=%llu\n",
                self.window_,(unsigned long long)self.call_,
                (unsigned long long)begin,(unsigned long long)released);
        }
        return met && !self.stopped_.load();
    }
    static bool submitted(void * opaque) {
        auto & self = *static_cast<mtp_submit_ready_scope *>(opaque);
        if (self.called_ != 1 || ++self.submitted_ != 1 || !self.submission_.owns_lock()) {
            self.pair_.cancel();
            return false;
        }
        // The native provider has accepted and sealed all five private graphs.
        // Only this context's output copies remain; context synchronization and
        // the after-target pair still precede output reads, draft work and reuse.
        const auto released = now_ns();
        self.submission_.unlock();
        ++self.submitted_hits_;
        if (std::getenv("TP5_MTP_READY_PAIR_TRACE")) {
            std::fprintf(stderr,"TP5_MTP_SUBMITTED window=%u call=%llu release_ns=%llu\n",
                self.window_,(unsigned long long)self.call_,(unsigned long long)released);
        }
        return !self.stopped_.load();
    }
    bool enabled_;
    bool release_after_submit_;
    mtp_phase_pair & pair_;
    std::unique_lock<submission_mutex_type> & submission_;
    const std::atomic<bool> & stopped_;
    unsigned window_;
    std::uint64_t call_;
    std::uint64_t & waits_;
    std::uint64_t & hits_;
    std::uint64_t & submitted_hits_;
    unsigned called_ = 0;
    unsigned submitted_ = 0;
};

// ASYNC-PROTOCOL: optional submit-only gate. The cross-window lock covers only
// the native five-rank enqueue (Meta submit-ready -> submitted), i.e. the
// asynchronous send itself; host prepare of one window never blocks the other.
class mtp_submit_only_gate {
  public:
    mtp_submit_only_gate(bool enabled,
                         std::unique_lock<submission_mutex_type> & submission,
                         std::uint64_t & queue_ns, gate_metrics * gm,
                         mtp_submit_observation * observation, bool trim)
        : enabled_(enabled), trim_(trim), submission_(submission), queue_ns_(queue_ns),
          gm_(gm), observation_(observation) {
        if (!enabled_) return;
        require(!submission_.owns_lock(), "submit-only gate must start unlocked");
        require(ggml_backend_meta_set_submit_ready_hook(&arrive, this), "nested Meta submit-ready hook");
        if (!ggml_backend_meta_set_submitted_hook(&submitted, this)) {
            ggml_backend_meta_set_submit_ready_hook(nullptr, nullptr);
            require(false, "nested Meta submitted hook");
        }
    }
    ~mtp_submit_only_gate() {
        if (!enabled_) return;
        if (observed_) observation_->leave();
        // DUALWIN-SERIALCUT-TRIM (trim_ only): an abort between arrive() and
        // submitted() still charges the captured wait interval exactly once
        // (the r4 baseline charged it inside arrive()'s hold; with trim_ the
        // success path charges it after the release in submitted(), so this
        // branch only fires on the abort path). trim_ off = r4 accounting
        // untouched: arrive() already charged under the lock.
        if (trim_ && arrived_ && !submitted_) {
            queue_ns_ += pending_wait_ns_;
            if (gm_) {
                gm_->enqueue_wait_ns += pending_wait_ns_;
                ++gm_->enqueue_waits;
            }
        }
        ggml_backend_meta_set_submitted_hook(nullptr, nullptr);
        ggml_backend_meta_set_submit_ready_hook(nullptr, nullptr);
    }
    void check(int decoded) const {
        if (enabled_ && decoded == 0) {
            require(arrived_ == 1 && submitted_ == 1, "submit-only gate did not see exactly one native submit");
        }
    }
  private:
    static bool arrive(void * opaque) {
        auto & self = *static_cast<mtp_submit_only_gate *>(opaque);
        if (++self.arrived_ != 1 || self.submission_.owns_lock()) return false;
        const auto begin = now_ns();
        // Default async protocol: one process-wide ticket lock serializes
        // both windows' five-rank enqueues. GGML_TP5_DW_SERIALCUT=1 keeps it:
        // the global five-rank submission lock is always taken.
        self.submission_.lock();
        // DUALWIN-OVERLAP-METER: with lock-hold metering on, the hold
        // boundary starts at the acquire itself — before any in-lock work —
        // so the metered hold covers everything executed inside the lock
        // (the wait-accounting included); TRIM's relocation then shows up as
        // a measured hold reduction instead of hiding outside the meter.
        // Metering off (gm_ null) takes no timestamp: r4 behavior unchanged.
        const auto acquired = now_ns();
        if (self.gm_) self.enqueue_acquired_ns_ = acquired;
        if (self.trim_) {
            // DUALWIN-SERIALCUT-TRIM: only the interval capture stays inside
            // the ticket lock; the counter accumulation runs after the release
            // in submitted() below (per-worker thread-local state).
            self.pending_wait_ns_ = acquired - begin;
        } else {
            // r4 baseline verbatim: accumulate while holding the ticket lock.
            self.queue_ns_ += acquired - begin;
            if (self.gm_) {
                self.gm_->enqueue_wait_ns += acquired - begin;
                ++self.gm_->enqueue_waits;
            }
        }
        if (self.observation_) {
            self.observation_->enter();
            self.observed_ = true;
        }
        return true;
    }
    static bool submitted(void * opaque) {
        auto & self = *static_cast<mtp_submit_only_gate *>(opaque);
        if (self.arrived_ != 1 || ++self.submitted_ != 1 ||
                !self.submission_.owns_lock()) {
            return false;
        }
        if (self.gm_) {
            self.gm_->enqueue_ns += now_ns() - self.enqueue_acquired_ns_;
            ++self.gm_->enqueues;
        }
        if (self.observed_) {
            self.observation_->leave();
            self.observed_ = false;
        }
        self.submission_.unlock();
        if (self.trim_) {
            // DUALWIN-SERIALCUT-TRIM: ticket-lock work moved out of the hold —
            // the wait counter accumulation runs after the release.
            self.queue_ns_ += self.pending_wait_ns_;
            if (self.gm_) {
                self.gm_->enqueue_wait_ns += self.pending_wait_ns_;
                ++self.gm_->enqueue_waits;
            }
        }
        return true;
    }
    bool enabled_;
    bool trim_ = false;
    std::unique_lock<submission_mutex_type> & submission_;
    std::uint64_t & queue_ns_;
    gate_metrics * gm_ = nullptr;
    mtp_submit_observation * observation_ = nullptr;
    std::uint64_t enqueue_acquired_ns_ = 0;
    std::uint64_t pending_wait_ns_ = 0;
    bool observed_ = false;
    unsigned arrived_ = 0;
    unsigned submitted_ = 0;
};

struct relay_timing_row {
    std::uint64_t epoch, plan, ready_ns, reduced_ns, published_ns, elements;
    std::uint32_t channel, computed_mask;
};

struct cycle_timing_row {
    std::uint64_t call, rows, position, queue_ns, submit_ns, submitted_ns, waited_ns, committed_ns;
};

struct service_window {
    ggml_backend_t                                                                    backend = nullptr;
    std::map<std::pair<std::uint64_t, std::uint32_t>, std::unique_ptr<channel_state>> channels;
    std::atomic<std::uint64_t>                                                        input_retired{ 0 };
    // DUALWIN-COMBO (TP5_COURIER_HITS=1): per-window courier hit counters.
    // polls: progress() entries; events: events taken into service; ready:
    // INPUT_READY fully handled (incremented after reduce+publish succeed);
    // released: OUTPUT_RELEASED fully retired (incremented after success);
    // busy: polls that found no event; waits: delayed-publish deferrals.
    // Only these counters touch the off path: writes are behind
    // courier_hits_req at each site.
    std::atomic<std::uint64_t> hits_polls{ 0 };
    std::atomic<std::uint64_t> hits_events{ 0 };
    std::atomic<std::uint64_t> hits_ready{ 0 };
    std::atomic<std::uint64_t> hits_released{ 0 };
    std::atomic<std::uint64_t> hits_busy{ 0 };
    std::atomic<std::uint64_t> hits_waits{ 0 };
    std::atomic<std::uint64_t>                                                        output_retired{ 0 };
    bool                                                                              delayed = false;
    std::optional<ggml_relay_event_v2>                                                   delayed_event;
    clock_type::time_point                                                            delay_until{};
    // Only the service thread writes these; copy them after stop()/join().
    std::array<std::uint64_t, 5> rank_input_packed_payload_bytes{};
    std::array<std::uint64_t, 5> rank_output_consumed_payload_bytes{};
    std::uint64_t cpu_input_payload_bytes = 0;
    std::uint64_t cpu_output_payload_bytes = 0;
    // DUALWIN-COURIER (GGML_TP5_DW_COURIER=1): per-epoch demand samples and
    // the retirement race they observe. Written only by this window's worker.
    // demand_* values are PER-EPOCH DELTAS against demand_base_*; the worker
    // only leaves the wait loop once retired == cpu_reductions, so the
    // previous sample's baseline is a fully drained epoch.
    std::uint64_t demand_epochs = 0;
    std::uint64_t demand_hits = 0;
    std::uint64_t demand_misses = 0;
    std::uint64_t demand_base_published = 0;
    std::uint64_t demand_base_retired = 0;
    std::uint64_t demand_delta_published_sum = 0;
    std::uint64_t demand_delta_retired_sum = 0;
    std::uint64_t demand_pending_sum = 0;
    std::uint64_t demand_pending_max = 0;
    std::array<std::uint64_t, 8> demand_pending_hist{};
    std::uint64_t demand_drain_ns = 0;
    std::uint64_t demand_drain_passes = 0;
    std::uint64_t demand_drain_max = 0;
    std::uint64_t courier_batch_extra = 0;
    std::uint64_t courier_polls = 0;
    std::uint64_t courier_idle = 0;
    std::uint64_t courier_input = 0;
    std::uint64_t courier_output = 0;
    std::uint64_t courier_deferred = 0;
    std::uint64_t courier_errors = 0;
    // GGML_TP5_DW_COURIER_DUAL=1: independent dual completion verification at
    // the logits consumption points. Written only by this window's worker
    // while its dual wait is parked; read after the workers join for the
    // exit summary. double_complete_ok counts epochs whose consumption
    // proved BOTH completions; device_late/courier_late record the observed
    // late completion from strict differential sightings; violations count
    // protocol breaches or failures observed while the wait was parked.
    std::uint64_t dual_double_ok    = 0;
    std::uint64_t dual_device_late  = 0;
    std::uint64_t dual_courier_late = 0;
    std::uint64_t dual_violations   = 0;
    // r7 production handoff state: a window whose dual verification latched
    // (bounded handoff budget exhausted, or a dual scope breach) consumes
    // every later epoch through the exact shipped stock single-completion
    // loop. Written only by this window's worker; read after the workers
    // join for the exit summary.
    bool          dual_latched         = false;
    const char *  dual_latch_reason    = nullptr;
    std::uint64_t dual_timeouts        = 0;
    std::uint64_t dual_fallback_epochs = 0;
    // DUALWIN event-overlap trace (GGML_TP5_DW_EVENT_TRACE=1): host
    // correlation records for the proven dual-consumption epochs. Single
    // writer (this window's dual-wait worker); read after the workers join.
    dw_event_trace::buf trace;
    std::vector<relay_timing_row> timings;
};

// This C object owns the A/B collaboration. Neither A nor the ledger calls a
// sibling service. The reducer receives provider facts and sends ledger facts.
class relay_service {
  public:
    relay_service(const options &                     opts,
                  const std::vector<ggml_backend_t> & backends,
                  failure_state &                     failure,
                  std::uint64_t                       run) :
        options_(opts),
        failure_(failure),
        run_(run) {
        for (auto backend : backends) {
            auto item     = std::make_unique<service_window>();
            item->backend = backend;
            if (std::getenv("TP5_MTP_TIMELINE_PREFIX")) {
                item->timings.reserve(131072);
            }
            windows_.push_back(std::move(item));
        }
        // ASYNC-PROTOCOL: one CPU courier per window (default). A single shared
        // service thread serializes window A's fold/publish with window B's,
        // so one window's T sits on the other's critical path.
        // DUALWIN-COURIER: the exact value "1" turns on the demand/retirement
        // counters, bounded batch drain and worker lock-yield. It also forces
        // one courier per window (the batch path assumes a private lane).
        // Unset keeps the shipped default and the strict 0/1 validation.
        const char * dw_env = std::getenv("GGML_TP5_DW_COURIER");
        courier_metrics_ = dw_env != nullptr && std::strcmp(dw_env, "1") == 0;
        const char * ab_env = std::getenv("TP5_DW_COURIER_ABLATE");
        courier_ablate_ = courier_metrics_ && ab_env != nullptr && std::strcmp(ab_env, "1") == 0;
        // GGML_TP5_DW_COURIER_DUAL: the exact value "1" gates the independent
        // dual completion verification at the logits consumption points;
        // unset, "0", "2" or any other spelling keeps the shipped single
        // conjunction wait untouched.
        const char * dual_env = std::getenv("GGML_TP5_DW_COURIER_DUAL");
        courier_dual_ = dual_env != nullptr && std::strcmp(dual_env, "1") == 0;
        const char * per_env = std::getenv("TP5_RELAY_SERVICE_PER_WINDOW");
        require(!per_env || std::string(per_env) == "0" || std::string(per_env) == "1",
                "TP5_RELAY_SERVICE_PER_WINDOW must be 0 or 1");
        const bool per_window = courier_metrics_ || !per_env || std::string(per_env) == "1";
        const std::size_t lanes = per_window ? windows_.size() : 1;
        for (std::size_t lane = 0; lane < lanes; ++lane) {
            threads_.emplace_back([this, lane, lanes] { loop(lane, lanes); });
        }
        while (ready_.load(std::memory_order_acquire) != lanes) {
            pause_cpu();
        }
        std::fprintf(stderr, "TP5_RELAY_SERVICE threads=%zu windows=%zu dw_courier=%d\n", lanes, windows_.size(),
                     int(courier_metrics_));
    }

    ~relay_service() { stop(); }

    void abort_all() noexcept {
        for (const auto & item : windows_) {
            ggml_backend_relay_abort(item->backend, 1);
        }
    }

    void stop() noexcept {
        stop_.store(true, std::memory_order_release);
        for (auto & t : threads_) {
            if (t.joinable()) {
                t.join();
            }
        }
    }

    bool metrics_enabled() const noexcept { return courier_metrics_; }

    bool ablate() const noexcept { return courier_ablate_; }

    // One demand sample per epoch, taken by the worker after the first snapshot
    // that reports finished. cpu_reductions is final there (every publish
    // precedes its channel's generation store, which precedes every consumer
    // ack, which precedes the rank tails that make finished true), but
    // output_retired is read AFTER the snapshot returns and the courier may
    // advance it in between: the sample is "at the first post-finished
    // observation", not an instantaneous read at the finish instant.
    // Both values are provider/courier LIFETIME counters (begin() never resets
    // cpu_reductions), so per-epoch values are deltas against a baseline that
    // is only ever a FULLY DRAINED point: epoch_end() sets it to
    // (published, published) after the wait loop proved retired == published.
    // The sample itself must not move the baseline - at that moment the epoch
    // may still have unretired outputs, and folding them into the next epoch's
    // delta would attribute this epoch's tail to the next one.
    void epoch_demand(std::size_t index, const ggml_relay_status & state) {
        if (!courier_metrics_) return;
        auto &              w         = *windows_[index];
        const std::uint64_t published = state.cpu_reductions;
        const std::uint64_t retired   = w.output_retired.load(std::memory_order_acquire);
        require(retired <= published, "courier retired more outputs than were published");
        require(published >= w.demand_base_published && retired >= w.demand_base_retired,
                "relay lifetime counters moved backwards");
        const std::uint64_t delta_published = published - w.demand_base_published;
        const std::uint64_t delta_retired   = retired - w.demand_base_retired;
        require(delta_retired <= delta_published, "courier retired more outputs than this epoch published");
        const std::uint64_t pending = delta_published - delta_retired;
        ++w.demand_epochs;
        w.demand_delta_published_sum += delta_published;
        w.demand_delta_retired_sum += delta_retired;
        w.demand_pending_sum += pending;
        // pending == 0 is the hit bucket and is recorded in hist[0].
        w.demand_pending_max = std::max(w.demand_pending_max, pending);
        ++w.demand_pending_hist[std::min<std::size_t>(pending, 7)];
        if (pending == 0) {
            ++w.demand_hits;
        } else {
            ++w.demand_misses;
        }
    }
    // End of one epoch: the wait loop just proved retired == published, so this
    // is the only place the delta baseline may advance. Both baselines are set
    // to the same drained value, which keeps the next epoch's delta_retired
    // bounded by its delta_published.
    void epoch_end(std::size_t index, std::uint64_t drained) {
        if (!courier_metrics_) return;
        auto & w = *windows_[index];
        require(drained >= w.demand_base_published && drained >= w.demand_base_retired,
                "relay lifetime counters moved backwards");
        w.demand_base_published = drained;
        w.demand_base_retired   = drained;
    }

    void epoch_drain(std::size_t index, std::uint64_t drain_ns, std::uint64_t drain_passes) {
        if (!courier_metrics_) return;
        auto & w = *windows_[index];
        w.demand_drain_ns += drain_ns;
        w.demand_drain_passes += drain_passes;
        w.demand_drain_max = std::max(w.demand_drain_max, drain_passes);
    }

    // Worker-side re-check after the gated wait loop. The gated loop stops
    // snapshotting once finished, so a provider failure recorded after that
    // point is only visible through a fresh snapshot. snapshot() uses
    // try_to_lock and returns false on ordinary contention, so retry the same
    // way the wait loops do and fail closed only on a real failure.
    void epoch_verify(std::size_t index, ggml_backend_t backend) const {
        if (!courier_metrics_) return;
        ggml_relay_status status{};
        while (!ggml_backend_relay_snapshot(backend, &status)) {
            require(!failure_.stopped.load(), "relay stopped before the post-drain check");
            pause_cpu();
        }
        require(!status.failed && !failure_.stopped.load(), "native relay epoch failed after drain");
        require(status.finished && retired(index, status), "native relay epoch was not drained after finish");
    }

    // GGML_TP5_DW_COURIER_DUAL=1 (exact value; unset/"0"/"2"/"1x" stay OFF):
    // independent dual completion verification at the logits consumption
    // points. The two completions are INDEPENDENT signals already exposed by
    // the existing runtime: the DEVICE tail (status.finished, driven by the
    // relay provider's per-lane tail event queries inside
    // ggml_backend_relay_snapshot) and the CPU COURIER retirement
    // (retired(): output_retired == status.cpu_reductions). No new device
    // event is created and no GPU kernel waits on its own tail.
    //
    // Event query first: every pass refreshes the relay snapshot BEFORE any
    // courier counter is consulted. A lost snapshot try-lock is ordinary
    // contention, never completion; status.failed and a stopped failure
    // state propagate explicitly through require() and are never treated as
    // completion either.
    //
    // Not a busy wait: the dual wait parks through a bounded pause_cpu
    // phase and then sleeps in short slices instead of spinning unbounded
    // the way the ungated loops spin.
    //
    // The exit condition stays exactly the shipped conjunction (both flags
    // hold on the values of one pass), so the dual gate never invents an
    // earlier exit and never blocks an exit the OFF path would take; it adds
    // the per-epoch account rows, the invariant checks and the fresh
    // re-verification below. With the gate off, every path here is untouched.
    static constexpr unsigned                  kDualPausePasses = 128;
    static constexpr std::chrono::microseconds kDualSliceSleep{50};

    bool dual_enabled() const noexcept { return courier_dual_; }

    static void dual_backoff(unsigned & passes) noexcept {
        if (passes < kDualPausePasses) {
            ++passes;
            pause_cpu();
            return;
        }
        std::this_thread::sleep_for(kDualSliceSleep);
    }

    // Monotonic relay facts while a dual wait is parked: the epoch and both
    // lifetime counters (cpu_reductions, output_retired) never move
    // backwards, and the courier never retires more than was published.
    // Every breach is counted first, then fails explicitly.
    void dual_scope_check(std::size_t index, const ggml_relay_status & state, bool & seen,
                          std::uint64_t & prev_epoch, std::uint64_t & prev_published,
                          std::uint64_t & prev_retired) {
        auto &              w         = *windows_[index];
        const std::uint64_t published = state.cpu_reductions;
        const std::uint64_t retired   = w.output_retired.load(std::memory_order_acquire);
        if (seen && (state.epoch < prev_epoch || published < prev_published || retired < prev_retired)) {
            ++w.dual_violations;
            require(false, "relay epoch or lifetime counters moved backwards during the dual completion wait");
        }
        seen = true;
        if (retired > published) {
            ++w.dual_violations;
            require(false, "courier retired more outputs than were published");
        }
        prev_epoch     = state.epoch;
        prev_published = published;
        prev_retired   = retired;
    }

    // One account row per epoch whose consumption verified BOTH completions.
    // first_witness: 1 = the device tail was observed finished first (the
    // courier retired late), 2 = the courier was observed retired first (the
    // device tail finished late), 0 = both flags first seen on one pass, so
    // the order is not resolvable at this sampling and only the ok counter
    // moves.
    void dual_epoch_row(std::size_t index, int first_witness) {
        auto & w = *windows_[index];
        ++w.dual_double_ok;
        if (first_witness == 1) {
            ++w.dual_courier_late;
        } else if (first_witness == 2) {
            ++w.dual_device_late;
        }
    }
    // r7 bounded production handoff: one wall-clock budget per consumed
    // epoch. The dual machinery must never hold a production run hostage —
    // when the budget runs out the window latches the shipped stock
    // fallback (counted, one stderr line) and the epoch still completes
    // through the exact shipped single-completion loop. The exit predicate
    // never weakens: both arms consume only a proven finished && retired
    // epoch.
    static constexpr std::uint64_t kDualHandoffNs = 10'000'000'000ull; // 10 s

    // The exact shipped single-completion wait (the gate-off arm): event
    // query first, snapshot contention retries via pause_cpu, relay
    // failures and stops propagate through require, exit only on the
    // shipped finished && retired conjunction. Every call is one epoch
    // consumed through the stock fallback arm (counted).
    void stock_wait(std::size_t index, ggml_backend_t backend) {
        ++windows_[index]->dual_fallback_epochs;
        ggml_relay_status status{};
        for (;;) {
            if (!ggml_backend_relay_snapshot(backend, &status)) {
                pause_cpu();
                continue;
            }
            require(!status.failed && !failure_.stopped.load(), "native relay epoch failed");
            if (status.finished && retired(index, status)) {
                break;
            }
            pause_cpu();
        }
    }

    // The exact shipped metrics drain arm (the gate-off consumption point):
    // one demand sample per epoch at the first finished observation, drain
    // accounting, epoch_end baseline advance and the fresh epoch_verify
    // re-check. The bounded dual handoff degrades here, so the shipped
    // metrics contract completes exactly once per epoch.
    void stock_wait_metrics(std::size_t index, ggml_backend_t backend) {
        ++windows_[index]->dual_fallback_epochs;
        ggml_relay_status status        {};
        const bool        keep_snapshot = courier_ablate_;
        bool              finished      = false;
        std::uint64_t     drain_start = 0, drain_passes = 0;
        for (;;) {
            if (!finished) {
                if (!ggml_backend_relay_snapshot(backend, &status)) {
                    pause_cpu();
                    continue;
                }
                require(!status.failed && !failure_.stopped.load(), "native relay epoch failed");
                if (!status.finished) {
                    pause_cpu();
                    continue;
                }
                finished = true;
                epoch_demand(index, status);
                drain_start = now_ns();
            }
            if (keep_snapshot && !ggml_backend_relay_snapshot(backend, &status)) {
                pause_cpu();
                continue;
            }
            require(!status.failed && !failure_.stopped.load(), "native relay epoch failed");
            if (retired(index, status)) {
                break;
            }
            ++drain_passes;
            pause_cpu();
        }
        epoch_drain(index, now_ns() - drain_start, drain_passes);
        epoch_end(index, status.cpu_reductions);
        epoch_verify(index, backend);
    }

    // Latch this window's dual verification onto the shipped stock
    // fallback: one stderr line per window, never silent.
    void dual_fallback(std::size_t index, const char * reason) {
        auto & w = *windows_[index];
        if (!w.dual_latched) {
            w.dual_latched      = true;
            w.dual_latch_reason = reason;
            std::fprintf(stderr,
                         "TP5_DW_COURIER_DUAL_FALLBACK window=%zu reason=%s\n",
                         index, reason);
        }
    }

    bool dual_latched(std::size_t index) const noexcept { return windows_[index]->dual_latched; }

    // Dual wait for the metrics-disabled arm: same snapshot/retry/require
    // shape and the same exit conjunction as the ungated loop, plus the dual
    // account rows; ends with a fresh independent re-verification (the same
    // shape as the epoch_verify contract) right before the caller consumes.
    void dual_wait_plain(std::size_t index, ggml_backend_t backend, const char * what) {
        if (dual_latched(index)) {
            stock_wait(index, backend);
            return;
        }
        const std::uint64_t deadline = now_ns() + kDualHandoffNs;
        // DUALWIN event-overlap trace (GGML_TP5_DW_EVENT_TRACE=1): host
        // correlation timestamps only (see dw_event_trace.h). Zero stamps
        // when the gate is off; a fallback/timeout/latched epoch pushes
        // nothing (the push happens after the ok row below).
        const bool          dw_tracing  = dw_event_trace::gate();
        const std::uint64_t dw_entry_ns = dw_tracing ? now_ns() : 0;
        std::uint64_t       dw_dev_ns = 0, dw_cour_ns = 0, dw_consumed_ns = 0;
        bool                dw_dev_seen = false, dw_cour_seen = false;
        ggml_relay_status status        {};
        int               first_witness = 0;
        unsigned          backoff       = 0;
        bool              seen          = false;
        std::uint64_t     prev_epoch = 0, prev_published = 0, prev_retired = 0;
        for (;;) {
            // r7 bounded handoff: on timeout the window latches the shipped
            // stock fallback (counted) and this epoch still finishes through
            // the exact stock single-completion loop — the exit predicate
            // never weakens, only the dual machinery's own overhead stops.
            if (now_ns() >= deadline) {
                ++windows_[index]->dual_timeouts;
                dual_fallback(index, "dual completion handoff timed out");
                stock_wait(index, backend);
                return;
            }
            // Event query first.
            if (!ggml_backend_relay_snapshot(backend, &status)) {
                dual_backoff(backoff);
                continue;
            }
            if (status.failed || failure_.stopped.load()) {
                require(false, (std::string(what) + ": native relay epoch failed during the dual completion wait").c_str());
            }
            dual_scope_check(index, status, seen, prev_epoch, prev_published, prev_retired);
            const bool device_now  = status.finished;
            const bool courier_now = retired(index, status);
            if (dw_tracing) {
                if (!dw_dev_seen && device_now) {
                    dw_dev_seen = true;
                    dw_dev_ns   = now_ns();
                }
                if (!dw_cour_seen && courier_now) {
                    dw_cour_seen = true;
                    dw_cour_ns   = now_ns();
                }
            }
            if (!first_witness) {
                if (device_now && !courier_now) {
                    // Strict device-first: the tail is observed finished and
                    // the retirement equality still fails at the later
                    // retired() read of this pass.
                    first_witness = 1;
                } else if (courier_now && !device_now) {
                    // Provisional courier-first: the snapshot's finished flag
                    // was captured before the retired() read, so a strictly
                    // later unfinished event query must confirm it.
                    first_witness = -1;
                }
            } else if (first_witness == -1) {
                first_witness = (courier_now && !device_now) ? 2 : 0;
            }
            if (device_now && courier_now) {
                if (dw_tracing) dw_consumed_ns = now_ns();
                break;
            }
            dual_backoff(backoff);
        }
        // Fresh independent re-verification immediately before the logits
        // read: one new event query must still show BOTH completions. r7
        // bounded: a re-verify that cannot confirm latches the shipped
        // stock fallback and the epoch still completes through the exact
        // stock single-completion loop — production never aborts on the
        // dual machinery's own check; the main-loop proof already held.
        status = {};
        for (;;) {
            if (ggml_backend_relay_snapshot(backend, &status)) {
                break;
            }
            if (failure_.stopped.load()) {
                ++windows_[index]->dual_violations;
                require(false, "relay stopped before the dual completion re-verification");
            }
            if (now_ns() >= deadline) {
                ++windows_[index]->dual_timeouts;
                dual_fallback(index, "dual completion re-verification timed out");
                stock_wait(index, backend);
                return;
            }
            dual_backoff(backoff);
        }
        require(!status.failed && !failure_.stopped.load(),
                (std::string(what) + ": native relay failed before the dual completion re-verification").c_str());
        if (!(status.finished && retired(index, status))) {
            // A failed re-check counts as a violation sighting before the
            // explicit fail-closed require (same pattern as the stopped
            // path); a holding re-check never counts.
            ++windows_[index]->dual_violations;
            require(false, (std::string(what) + ": dual completion did not hold at the fresh consumption re-verification").c_str());
        }
        dual_epoch_row(index, first_witness == -1 ? 0 : first_witness);
        if (dw_tracing) {
            dw_event_trace::rec r;
            r.relay_epoch = status.epoch;
            r.entry_ns    = dw_entry_ns;
            r.dev_ns      = dw_dev_seen ? dw_dev_ns : dw_consumed_ns;
            r.cour_ns     = dw_cour_seen ? dw_cour_ns : dw_consumed_ns;
            r.consumed_ns = dw_consumed_ns;
            r.order       = first_witness == -1 ? 0 : first_witness;
            windows_[index]->trace.push(r);
        }
    }

    // Dual wait for the metrics-enabled arm: preserves the shipped one
    // demand-sample-per-epoch contract (sample at the FIRST finished
    // observation, drain accounting, epoch_end baseline advance and the
    // fresh epoch_verify re-check) while polling event-first and keeping the
    // dual account rows.
    void dual_wait_metrics(std::size_t index, ggml_backend_t backend, const char * what) {
        if (dual_latched(index)) {
            stock_wait_metrics(index, backend);
            return;
        }
        const std::uint64_t deadline = now_ns() + kDualHandoffNs;
        // DUALWIN event-overlap trace (GGML_TP5_DW_EVENT_TRACE=1): host
        // correlation timestamps only (see dw_event_trace.h). The device
        // stamp is the finished-flip (demand) instant; the courier stamp is
        // the first drain-pass retirement sighting (or the same demand
        // instant when same_tick holds). A fallback epoch pushes nothing.
        const bool          dw_tracing  = dw_event_trace::gate();
        const std::uint64_t dw_entry_ns = dw_tracing ? now_ns() : 0;
        std::uint64_t       dw_dev_ns = 0, dw_cour_ns = 0, dw_consumed_ns = 0;
        bool                dw_dev_seen = false, dw_cour_seen = false;
        ggml_relay_status status        {};
        const bool        keep_snapshot = courier_ablate_;
        bool              finished      = false;
        int               first_witness = 0;
        bool              same_tick     = false;
        unsigned          backoff       = 0;
        std::uint64_t     drain_start = 0, drain_passes = 0;
        bool              seen          = false;
        std::uint64_t     prev_epoch = 0, prev_published = 0, prev_retired = 0;
        for (;;) {
            // r7 bounded handoff (pre-demand only): on timeout the window
            // latches the shipped stock fallback and the epoch still runs
            // through the exact stock metrics drain. The demand sample has
            // not run yet, so the shipped contract completes exactly once.
            if (!finished && now_ns() >= deadline) {
                ++windows_[index]->dual_timeouts;
                dual_fallback(index, "dual completion handoff timed out");
                stock_wait_metrics(index, backend);
                return;
            }
            if (!finished || keep_snapshot) {
                // Event query first.
                if (!ggml_backend_relay_snapshot(backend, &status)) {
                    dual_backoff(backoff);
                    continue;
                }
                if (status.failed || failure_.stopped.load()) {
                    require(false, (std::string(what) + ": native relay epoch failed during the dual completion wait").c_str());
                }
                dual_scope_check(index, status, seen, prev_epoch, prev_published, prev_retired);
                if (!finished) {
                    if (!status.finished) {
                        dual_backoff(backoff);
                        continue;
                    }
                    finished = true;
                    // Same-pass courier sighting at the demand instant is not
                    // an order witness (the retired() read races the
                    // snapshot's finished capture): it counts only in
                    // double_complete_ok, matching the plain arm's rule.
                    // Captured at the exact demand-sample instant, before
                    // any further courier progress can be observed: true
                    // means both flags were seen on one pass, which counts
                    // only as double_complete_ok (no order witness).
                    same_tick = retired(index, status);
                    epoch_demand(index, status);
                    drain_start = now_ns();
                    if (dw_tracing) {
                        dw_dev_seen = true;
                        dw_dev_ns   = now_ns();
                        if (same_tick) {
                            dw_cour_seen = true;
                            dw_cour_ns   = dw_dev_ns;
                        }
                    }
                }
            }
            // Live failure check on every pass: the OFF loop re-checks it in
            // the frozen-snapshot drain too, so a stop during the drain must
            // propagate instead of parking this worker.
            if (failure_.stopped.load()) {
                ++windows_[index]->dual_violations;
                require(false, (std::string(what) + ": stopped during the dual completion drain").c_str());
            }
            if (finished && retired(index, status)) {
                if (!first_witness) {
                    // Same-tick sighting (courier already retired at the
                    // sample pass) records no order witness, only
                    // double_complete_ok; a later retirement flip is the
                    // strict courier_late witness.
                    first_witness = same_tick ? 0 : 1;
                }
                if (dw_tracing) {
                    if (!dw_cour_seen) {
                        dw_cour_seen = true;
                        dw_cour_ns   = now_ns();
                    }
                    dw_consumed_ns = now_ns();
                }
                break;
            }
            ++drain_passes;
            dual_backoff(backoff);
        }
        epoch_drain(index, now_ns() - drain_start, drain_passes);
        // status holds the finished snapshot whose cpu_reductions is final;
        // that drained value is the only point where the baseline moves.
        epoch_end(index, status.cpu_reductions);
        epoch_verify(index, backend);
        // The ok row commits only after the fresh epoch_verify held: a
        // failed re-check must not leave the epoch counted as ok.
        dual_epoch_row(index, first_witness);
        if (dw_tracing) {
            dw_event_trace::rec r;
            r.relay_epoch = status.epoch;
            r.entry_ns    = dw_entry_ns;
            r.dev_ns      = dw_dev_seen ? dw_dev_ns : dw_consumed_ns;
            r.cour_ns     = dw_cour_seen ? dw_cour_ns : dw_consumed_ns;
            r.consumed_ns = dw_consumed_ns;
            r.order       = first_witness;
            windows_[index]->trace.push(r);
        }
    }

    // GGML_TP5_DW_COURIER_DUAL=1 exit summary, one line per window: epochs
    // whose consumption proved BOTH independent completions
    // (double_complete_ok), the observed late completion per epoch
    // (device_late: device tail finished after the courier retired;
    // courier_late: courier retired after the device tail finished) and the
    // protocol violations and failures observed while the dual waits were
    // parked. Requires the stopped service (same contract as
    // dump_courier_metrics); no-op when the dual gate is off, so OFF runs
    // print nothing new.
    void dump_dual_completions_summary() const {
        if (!courier_dual_) return;
        require(!running(), "dual completion summary requires the stopped service");
        for (std::size_t i = 0; i < windows_.size(); ++i) {
            const auto & w = *windows_[i];
            std::fprintf(stderr,
                         "TP5_DW_COURIER_DUAL window=%zu double_complete_ok=%llu device_late=%llu courier_late=%llu "
                         "violations=%llu timeouts=%llu fallback_epochs=%llu latched=%d reason=%s\n",
                         i, (unsigned long long) w.dual_double_ok, (unsigned long long) w.dual_device_late,
                         (unsigned long long) w.dual_courier_late, (unsigned long long) w.dual_violations,
                         (unsigned long long) w.dual_timeouts, (unsigned long long) w.dual_fallback_epochs,
                         int(w.dual_latched), w.dual_latch_reason ? w.dual_latch_reason : "none");
        }
        // DUALWIN event-overlap trace summary: per-window WIN line with the
        // first detail epochs, then cross-window host-interval overlap
        // lines. Host correlation metadata only — never device evidence.
        if (dw_event_trace::gate()) {
            dw_event_trace::banner();
            for (std::size_t i = 0; i < windows_.size(); ++i) {
                windows_[i]->trace.dump_win(i);
            }
            for (std::size_t i = 0; i < windows_.size(); ++i) {
                for (std::size_t j = i + 1; j < windows_.size(); ++j) {
                    windows_[i]->trace.dump_overlap(i, j, windows_[j]->trace);
                }
            }
        }
    }

    // Per-window demand/retirement distribution; emitted once after every
    // courier lane is joined (callers invoke it after workers join).
    void dump_courier_metrics() const {
        require(!running(), "courier metrics require the stopped service");
        for (std::size_t i = 0; i < windows_.size(); ++i) {
            const auto & w = *windows_[i];
            const double hit_rate = w.demand_epochs ? double(w.demand_hits) / double(w.demand_epochs) : 0.0;
            const double event_hit_rate = w.demand_delta_published_sum ?
                                              double(w.demand_delta_retired_sum) / double(w.demand_delta_published_sum) :
                                              0.0;
            std::fprintf(stderr,
                         "TP5_DW_COURIER_HIT window=%zu lanes=%zu ablate=%d epochs=%llu hits=%llu misses=%llu "
                         "hit_rate=%.6f d_published=%llu d_retired_before=%llu event_hit_rate=%.6f "
                         "pending_sum=%llu pending_max=%llu hist=%llu/%llu/%llu/%llu/%llu/%llu/%llu/%llu "
                         "drain_ns=%llu drain_passes=%llu drain_max=%llu batch_extra=%llu "
                         "polls=%llu idle=%llu serviced_in=%llu serviced_out=%llu deferred=%llu errors=%llu\n",
                         i, threads_.size(), int(courier_ablate_),
                         (unsigned long long) w.demand_epochs, (unsigned long long) w.demand_hits,
                         (unsigned long long) w.demand_misses, hit_rate,
                         (unsigned long long) w.demand_delta_published_sum,
                         (unsigned long long) w.demand_delta_retired_sum, event_hit_rate,
                         (unsigned long long) w.demand_pending_sum, (unsigned long long) w.demand_pending_max,
                         (unsigned long long) w.demand_pending_hist[0], (unsigned long long) w.demand_pending_hist[1],
                         (unsigned long long) w.demand_pending_hist[2], (unsigned long long) w.demand_pending_hist[3],
                         (unsigned long long) w.demand_pending_hist[4], (unsigned long long) w.demand_pending_hist[5],
                         (unsigned long long) w.demand_pending_hist[6], (unsigned long long) w.demand_pending_hist[7],
                         (unsigned long long) w.demand_drain_ns, (unsigned long long) w.demand_drain_passes,
                         (unsigned long long) w.demand_drain_max, (unsigned long long) w.courier_batch_extra,
                         (unsigned long long) w.courier_polls, (unsigned long long) w.courier_idle,
                         (unsigned long long) w.courier_input, (unsigned long long) w.courier_output,
                         (unsigned long long) w.courier_deferred, (unsigned long long) w.courier_errors);
        }
    }

    bool retired(std::size_t index, const ggml_relay_status & state) const {
        return windows_[index]->output_retired.load(std::memory_order_acquire) == state.cpu_reductions;
    }

    std::uint64_t inputs(std::size_t index) const { return windows_[index]->input_retired.load(); }

    std::uint64_t outputs(std::size_t index) const { return windows_[index]->output_retired.load(); }

    void finish_ledgers(bool failed) {
        stop();
        for (const auto & window : windows_) {
            if (window->delayed_event) {
                require(failed, "normal close has an unpublished delayed input");
                ggml_backend_relay_publish_v2(window->backend, &*window->delayed_event, sizeof(*window->delayed_event));
                window->delayed_event.reset();
            }
            for (auto & entry : window->channels) {
                auto & owner = *entry.second;
                if (failed) {
                    owner.ledger.fail("run aborted after native drain");
                } else {
                    require(!owner.held.has_value(), "normal close still has a live output reader");
                }
                require(owner.ledger.mark_drained(true), "cannot record proved native drain");
            }
        }
    }
    void write_timings(const std::string & prefix) const {
        require(!running(), "timing output requires a stopped service");
        for (std::size_t i = 0; i < windows_.size(); ++i) {
            const auto path = prefix + ".relay-" + std::to_string(i) + ".csv";
            auto * out = std::fopen(path.c_str(), "wx");
            require(out != nullptr, "cannot create relay timing output");
            std::fprintf(out, "epoch,plan,channel,elements,mask,ready_ns,reduced_ns,published_ns\n");
            for (const auto & r : windows_[i]->timings) {
                std::fprintf(out, "%llu,%llu,%u,%llu,%u,%llu,%llu,%llu\n",
                    (unsigned long long)r.epoch, (unsigned long long)r.plan, r.channel,
                    (unsigned long long)r.elements, r.computed_mask,
                    (unsigned long long)r.ready_ns, (unsigned long long)r.reduced_ns,
                    (unsigned long long)r.published_ns);
            }
            const bool written = !std::ferror(out);
            const int closed = std::fclose(out);
            require(written && closed == 0, "relay timing output failed");
        }
    }

    void copy_wire_metrics(std::size_t index, window_result & result) const {
        require(!running(), "wire metrics require the stopped CPU service");
        const auto & window = *windows_[index];
        result.rank_input_packed_payload_bytes = window.rank_input_packed_payload_bytes;
        result.rank_output_consumed_payload_bytes = window.rank_output_consumed_payload_bytes;
        result.cpu_input_payload_bytes = window.cpu_input_payload_bytes;
        result.cpu_output_payload_bytes = window.cpu_output_payload_bytes;
    }

    // DUALWIN-COMBO (TP5_COURIER_HITS=1): one stderr line per window after
    // stop()/join(). ready/released count COMPLETED reductions/retirements
    // (incremented after the work succeeds). Q8 hits come from the existing
    // fusion_stats counter via the proc-address path; q8_scope reports only
    // whether that API lookup resolved — it says nothing about replay status.
    void dump_courier_metrics(std::size_t index) const {
        require(!running(), "courier metrics require the stopped service");
        const auto & window = *windows_[index];
        std::uint64_t q8_hits  = 0;
        bool          q8_valid = false;
        if (window.backend) {
            if (auto * dev = ggml_backend_get_device(window.backend)) {
                if (auto * reg = ggml_backend_dev_backend_reg(dev)) {
                    using fusion_count_fn = int64_t (*)(ggml_backend_t, const char *);
                    const auto fn = reinterpret_cast<fusion_count_fn>(
                            ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_fusion_count"));
                    if (fn) {
                        q8_hits  = static_cast<std::uint64_t>(fn(window.backend, "q8_cache_hits"));
                        q8_valid = true;
                    }
                }
            }
        }
        std::fprintf(stderr,
                     "TP5_COURIER_METRICS window=%zu polls=%llu events=%llu ready=%llu released=%llu busy=%llu waits=%llu q8_cache_hits=%llu q8_scope=%s\n",
                     index, static_cast<unsigned long long>(window.hits_polls.load()),
                     static_cast<unsigned long long>(window.hits_events.load()),
                     static_cast<unsigned long long>(window.hits_ready.load()),
                     static_cast<unsigned long long>(window.hits_released.load()),
                     static_cast<unsigned long long>(window.hits_busy.load()),
                     static_cast<unsigned long long>(window.hits_waits.load()),
                     static_cast<unsigned long long>(q8_hits), q8_valid ? "resolved" : "unavailable");
    }

  private:
    static void add_payload(std::uint64_t & total, std::uint64_t bytes) {
        require(bytes <= UINT64_MAX - total, "wire payload byte counter overflow");
        total += bytes;
    }

    // Returns true iff exactly one provider event was fully serviced. Empty,
    // deferred and error-return passes return false so a batch stops there.
    bool progress(std::size_t index) {
        auto &           window = *windows_[index];
        if (courier_hits_req) window.hits_polls.fetch_add(1, std::memory_order_relaxed);
        if (courier_metrics_) ++window.courier_polls;
        ggml_relay_event_v2 event{};
        int              polled = 0;
        if (window.delayed_event) {
            if (clock_type::now() < window.delay_until && !failure_.stopped.load()) {
                if (courier_metrics_) ++window.courier_deferred;
                return false;
            }
            event = *window.delayed_event;
            window.delayed_event.reset();
            polled = 1;
        } else {
            polled = ggml_backend_relay_poll_v2(window.backend, &event, sizeof(event));
        }
        if (polled < 0) {
            if (courier_metrics_) ++window.courier_errors;
            if (!failure_.stopped.load()) {
                throw std::runtime_error("native relay progress failed");
            }
            return false;
        }
        if (!polled) {
            if (courier_hits_req) window.hits_busy.fetch_add(1, std::memory_order_relaxed);
            if (courier_metrics_) ++window.courier_idle;
            return false;
        }
        if (event.kind == GGML_RELAY_INPUT_READY && !window.delayed && options_.delay_publish_ms &&
            index == options_.delay_window) {
            // Retain this explicit provider borrow while servicing other windows.
            window.delayed       = true;
            window.delayed_event = event;
            window.delay_until   = clock_type::now() + std::chrono::milliseconds(options_.delay_publish_ms);
            if (courier_hits_req) window.hits_waits.fetch_add(1, std::memory_order_relaxed);
            if (courier_metrics_) ++window.courier_deferred;
            return false;
        }
        if (courier_hits_req) window.hits_events.fetch_add(1, std::memory_order_relaxed);
        const bool borrowed = event.kind == GGML_RELAY_INPUT_READY;
        const bool timed = window.timings.capacity() != 0 && borrowed;
        const auto ready_ns = timed ? now_ns() : 0;
        bool       returned = false;
        try {
            require(
                event.abi_version == GGML_RELAY_EVENT_V2_ABI_VERSION && event.ranks == 5 &&
                    event.epoch && event.plan_generation,
                "relay event v2 ABI or identity differs");
            const auto expected_wire = options_.wire == wire_type::f16 ? GGML_RELAY_WIRE_F16 : GGML_RELAY_WIRE_F32;
            const std::uint64_t width = options_.wire == wire_type::f16 ? 2 : 4;
            require(event.wire_type == expected_wire && (event.computed_mask & ~std::uint32_t(31)) == 0,
                    "relay wire format or computed mask differs");
            require(event.elements > 0 && event.elements <= SIZE_MAX / width &&
                        event.elements <= UINT64_MAX / width && event.wire_bytes == event.elements * width,
                    "relay wire payload extent is invalid");
            auto key = std::make_pair(event.plan_generation, event.channel);
            if (event.kind == GGML_RELAY_INPUT_READY) {
                auto & owner = window.channels[key];
                if (!owner) {
                    owner = std::make_unique<channel_state>(run_, event.plan_generation, event.channel);
                }
                require(!owner->held && event.epoch > owner->epoch, "relay input epoch is stale or has live readers");
                auto reservation = owner->ledger.reserve(index, event.epoch, event.channel, event.plan_generation);
                require(reservation.code == relay::reserve_code::accepted && reservation.granted.has_value(),
                        "relay bank is not reusable");
                owner->held          = reservation.granted;
                owner->epoch         = event.epoch;
                owner->elements      = event.elements;
                owner->wire_bytes    = event.wire_bytes;
                owner->computed_mask = event.computed_mask;
                require(event.output != nullptr, "relay output is absent");
                for (std::uint32_t rank = 0; rank < 5; ++rank) {
                    require(event.inputs[rank] != nullptr, "relay input is absent");
                    check_ledger(owner->ledger.producer_ready(*owner->held, rank));
                }
                if (options_.abort_window == static_cast<std::int32_t>(index) &&
                    event.channel == options_.abort_channel && event.epoch == options_.abort_epoch) {
                    throw std::runtime_error("TP5_INJECTED_ABORT window=" + std::to_string(index) +
                                             " channel=" + std::to_string(event.channel) +
                                             " epoch=" + std::to_string(event.epoch));
                }
                require(!failure_.stopped.load(), "relay stopped before reduction");
                detail::fold_result reduced;
                if (event.wire_type == GGML_RELAY_WIRE_F16) {
                    const std::uint16_t * inputs[5];
                    for (std::size_t rank = 0; rank < 5; ++rank) {
                        inputs[rank] = static_cast<const std::uint16_t *>(event.inputs[rank]);
                    }
                    reduced = detail::ordered_fold_f16(inputs, static_cast<std::uint16_t *>(event.output),
                                                       std::size_t(event.elements), event.computed_mask);
                } else {
                    const float * inputs[5];
                    for (std::size_t rank = 0; rank < 5; ++rank) {
                        inputs[rank] = static_cast<const float *>(event.inputs[rank]);
                    }
                    reduced = detail::ordered_fold_f32_masked(inputs, static_cast<float *>(event.output),
                                                              std::size_t(event.elements), event.computed_mask);
                }
                const auto reduced_ns = timed ? now_ns() : 0;
                require(reduced != detail::fold_result::nonfinite_input, "nonfinite rank partial");
                require(reduced != detail::fold_result::nonfinite_sum, "nonfinite CPU reduction");
                require(reduced != detail::fold_result::nonfinite_pack, "CPU reduction overflows F16 wire");
                require(reduced == detail::fold_result::ok, "invalid relay computed mask");
                for (std::size_t rank = 0; rank < 5; ++rank) {
                    if (event.computed_mask & (1u << rank)) {
                        add_payload(window.rank_input_packed_payload_bytes[rank], event.wire_bytes);
                        add_payload(window.cpu_input_payload_bytes, event.wire_bytes);
                    }
                }
                add_payload(window.cpu_output_payload_bytes, event.wire_bytes);
                check_ledger(owner->ledger.reduce_done(*owner->held));
                window.input_retired.fetch_add(1, std::memory_order_release);
                const bool published = ggml_backend_relay_publish_v2(window.backend, &event, sizeof(event));
                returned             = true;
                require(published, "relay publication was rejected");
                check_ledger(owner->ledger.output_published(*owner->held));
                if (timed) {
                    require(window.timings.size() < window.timings.capacity(), "relay timing capacity exhausted");
                    window.timings.push_back({event.epoch, event.plan_generation, ready_ns, reduced_ns,
                        now_ns(), event.elements, event.channel, event.computed_mask});
                }
                if (courier_hits_req) window.hits_ready.fetch_add(1, std::memory_order_relaxed);
                if (courier_metrics_) ++window.courier_input;
            } else if (event.kind == GGML_RELAY_OUTPUT_RELEASED) {
                auto found = window.channels.find(key);
                require(found != window.channels.end() && found->second->held.has_value(),
                        "output has no matching CPU lease");
                auto & owner = *found->second;
                require(owner.epoch == event.epoch && owner.elements == event.elements &&
                            owner.wire_bytes == event.wire_bytes && owner.computed_mask == event.computed_mask,
                        "output retirement has the wrong epoch or wire payload");
                for (std::uint32_t rank = 0; rank < 5; ++rank) {
                    check_ledger(owner.ledger.consumer_done(*owner.held, rank));
                    add_payload(window.rank_output_consumed_payload_bytes[rank], event.wire_bytes);
                }
                owner.held.reset();
                window.output_retired.fetch_add(1, std::memory_order_release);
                if (courier_hits_req) window.hits_released.fetch_add(1, std::memory_order_relaxed);
                if (courier_metrics_) ++window.courier_output;
            } else {
                throw std::runtime_error("unknown relay event kind");
            }
        } catch (...) {
            failure_.fail(std::current_exception());
            abort_all();
            // Return the borrowed view after all CPU reads/writes have ceased.
            // A failed publish only releases this borrow; it cannot publish data.
            if (borrowed && !returned) {
                ggml_backend_relay_publish_v2(window.backend, &event, sizeof(event));
            }
            throw;
        }
        return true;
    }

    bool running() const noexcept {
        for (const auto & t : threads_) {
            if (t.joinable()) return true;
        }
        return false;
    }

    void loop(std::size_t lane, std::size_t lanes) noexcept {
        // One courier per window -> its window's dedicated sibling CPU.
        // A single shared courier (lanes==1) lands on window 0's courier CPU.
        try {
            tp5_pin_this_thread(lane, true);
        } catch (...) {
            failure_.fail(std::current_exception());
        }
        ready_.fetch_add(1, std::memory_order_acq_rel);
        while (!stop_.load(std::memory_order_acquire)) {
            for (std::size_t i = lane; i < windows_.size(); i += lanes) {
                // DUALWIN-COURIER: service up to kCourierPassEvents events per
                // wake so a backlog of k costs one pass instead of k. Only under
                // the gate, which forces a private lane per window; ablation
                // keeps one event per pass while retaining every counter, so
                // both arms are comparable in one binary.
                try {
                    if (!courier_metrics_ || courier_ablate_) {
                        (void) progress(i);
                    } else {
                        for (std::size_t n = 0; n < kCourierPassEvents; ++n) {
                            if (!progress(i)) break;
                            if (n) ++windows_[i]->courier_batch_extra;
                        }
                    }
                } catch (...) {
                    failure_.fail(std::current_exception());
                    abort_all();
                }
            }
            pause_cpu();
        }
    }

    static constexpr std::size_t                 kCourierPassEvents = 32;
    bool                                         courier_metrics_ = false;
    bool                                         courier_ablate_ = false;
    bool                                         courier_dual_ = false;
    const options &                              options_;
    failure_state &                              failure_;
    std::uint64_t                                run_;
    std::vector<std::unique_ptr<service_window>> windows_;
    std::atomic<std::size_t>                     ready_{ 0 };
    std::atomic<bool>                            stop_{ false };
    std::vector<std::thread>                     threads_;
};

struct row_batch {
    llama_batch batch{};

    explicit row_batch(std::uint32_t capacity) { batch = llama_batch_init(capacity, 0, 1); }

    ~row_batch() { llama_batch_free(batch); }

    row_batch(const row_batch &)             = delete;
    row_batch & operator=(const row_batch &) = delete;

    void clear() { batch.n_tokens = 0; }

    void push(llama_token token, std::uint64_t position, llama_seq_id seq, bool output) {
        auto i             = batch.n_tokens++;
        batch.token[i]     = token;
        batch.pos[i]       = static_cast<llama_pos>(position);
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = seq;
        batch.logits[i]    = output;
    }
};

std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & prompt) {
    auto needed =
        llama_tokenize(vocab, prompt.data(), static_cast<std::int32_t>(prompt.size()), nullptr, 0, true, false);
    require(needed != INT32_MIN, "tokenizer token count overflow");
    if (needed < 0) {
        needed = -needed;
    }
    require(needed > 0, "prompt has no tokens");
    std::vector<llama_token> out(needed);
    const auto               count =
        llama_tokenize(vocab, prompt.data(), static_cast<std::int32_t>(prompt.size()), out.data(), needed, true, false);
    require(count == needed, "tokenizer size changed");
    return out;
}

std::string piece(const llama_vocab * vocab, llama_token token) {
    std::string result(256, '\0');
    auto size = llama_token_to_piece(vocab, token, result.data(), static_cast<std::int32_t>(result.size()), 0, false);
    if (size < 0) {
        require(size != INT32_MIN, "token piece size overflow");
        result.resize(-size);
        size = llama_token_to_piece(vocab, token, result.data(), static_cast<std::int32_t>(result.size()), 0, false);
    }
    require(size >= 0 && static_cast<std::size_t>(size) <= result.size(), "cannot detokenize accepted token");
    result.resize(size);
    return result;
}

}  // namespace

run_result run(const options & requested) {
    // Independent TP5 real target T4 qualification. Default original full512 path
    // unaffected. Requires exactly one owner/request per independent window.
    const char * t4_env = std::getenv("GGML_TP5_MTP_VERIFY_PROBE");
    const bool mtp_t4_probe = t4_env && std::string(t4_env) == "1";
    const char * mtp_path = std::getenv("GGML_TP5_REAL_TRAINED_MTP_GGUF");
    const bool mtp_draft_smoke = mtp_path && mtp_path[0];
    const char * depth_env = std::getenv("GGML_TP5_MTP_DRAFT_DEPTH");
    require(!depth_env || (std::string(depth_env)=="1" ||
                           std::string(depth_env)=="2" ||
                           std::string(depth_env)=="3"),
            "real MTP target draft depth must be 1..3");
    const std::uint32_t mtp_depth = depth_env ? std::uint32_t(std::atoi(depth_env)) : 1;
    require(!depth_env || mtp_draft_smoke,
            "MTP multistep depth requires genuine trained NextN draft");
    // Compare IDENTICAL MTP-aware target/draft models, same optimizer binary
    // and algorithm with only device-native checkpoint mode changed.
    const char * cp_mode = std::getenv("GGML_TP5_MTP_ON_DEVICE_CHECKPOINT");
    require(!cp_mode || std::string(cp_mode) == "0" ||
            std::string(cp_mode) == "1", "MTP checkpoint mode must be 0 or 1");
    const bool mtp_device_snapshot = cp_mode && std::string(cp_mode) == "1";
    const char * ring_env=std::getenv("GGML_TP5_MTP_RS_RING");
    require(!ring_env || std::string(ring_env)=="0" ||
            std::string(ring_env)=="1", "bounded Qwen MTP native ring flag must be 0 or 1");
    const bool mtp_native_trim = ring_env && std::string(ring_env)=="1";
    const bool mtp_state_trace = std::getenv("GGML_TP5_MTP_PLE_KV_TRACE") &&
        std::string(std::getenv("GGML_TP5_MTP_PLE_KV_TRACE"))=="1";
    const char * draft_output_env = std::getenv("TP5_MTP_LAST_DRAFT_LOGITS");
    require(!draft_output_env || std::string(draft_output_env) == "0" ||
            std::string(draft_output_env) == "1", "MTP last draft logits must be 0 or 1");
    const bool last_draft_logits = draft_output_env && std::string(draft_output_env) == "1";
    const char * prompt_catchup_env = std::getenv("TP5_MTP_PROMPT_CATCHUP");
    require(!prompt_catchup_env || std::string(prompt_catchup_env) == "0" ||
            std::string(prompt_catchup_env) == "1", "MTP prompt catch-up must be 0 or 1");
    const bool prompt_catchup = prompt_catchup_env && std::string(prompt_catchup_env) == "1";
    require(!prompt_catchup || mtp_draft_smoke, "prompt catch-up requires a trained MTP context");
    const char * prompt_audit_env = std::getenv("TP5_MTP_PROMPT_AUDIT_PREFIX");
    const std::string prompt_audit_prefix = prompt_audit_env ? prompt_audit_env : "";
    require(!mtp_native_trim || (mtp_draft_smoke && mtp_depth==2 && !mtp_device_snapshot),
            "native two-slot GDN ring requires real trained NextN depth2, N3 main target and host output");
    const llama_state_seq_flags checkpoint_flags =
        mtp_device_snapshot ? LLAMA_STATE_SEQ_FLAGS_ON_DEVICE : LLAMA_STATE_SEQ_FLAGS_NONE;
    require(!mtp_device_snapshot || mtp_draft_smoke,
            "MTP device snapshot requires genuine trained MTP decoding");
    require(!mtp_draft_smoke || mtp_t4_probe,
            "actual trained MTP smoke requires genuine T4 target verification mode");
    const options                   opts = validate(requested);
    const char * pair_env = std::getenv("TP5_MTP_READY_PAIR");
    require(!pair_env || std::string(pair_env) == "0" || std::string(pair_env) == "1",
            "MTP phase pair must be 0 or 1");
    // ASYNC-PROTOCOL (default ON in this candidate): fixed structure + async
    // send + busy-wait. Same-phase meet() barriers contradict the dual-window
    // proof (they force lockstep T with T), so TP5_MTP_READY_PAIR is ignored.
    const char * async_env = std::getenv("TP5_MTP_ASYNC_PROTOCOL");
    require(!async_env || std::string(async_env) == "0" || std::string(async_env) == "1",
            "MTP async protocol must be 0 or 1");
    const bool async_protocol = !async_env || std::string(async_env) == "1";
    const bool ready_pair_requested = pair_env && std::string(pair_env) == "1";
    const bool paired_phases = ready_pair_requested && !async_protocol;
    const char * gate_env = std::getenv("TP5_MTP_SUBMIT_ONLY_GATE");
    require(!gate_env || std::string(gate_env) == "0" || std::string(gate_env) == "1",
            "MTP submit-only gate must be 0 or 1");
    // Reuse the existing submit-only protocol; never bypass the five-rank gate.
    // The paired legacy protocol retains its original whole-decode locking.
    const char * serialcut_env = std::getenv("GGML_TP5_DW_SERIALCUT");
    const bool serialcut = serialcut_env && std::strcmp(serialcut_env, "1") == 0;
    // DUALWIN-SERIALCUT: exact "1" narrows the cross-window submission ticket
    // lock from the whole decode to the native five-rank enqueue (the
    // submit-only gate), even when TP5_MTP_SUBMIT_ONLY_GATE=0 explicitly
    // disables it. The process-wide ticket lock itself is always taken
    // (arrive/submitted): no lock is removed or bypassed; the provider-level
    // submit mutex and the per-decode "exactly one native submit" check stay.
    // serialcut with TP5_MTP_ASYNC_PROTOCOL=0 is inert. Unset/0/any other
    // value keeps the cumulative baseline branch byte-for-byte equivalent.
    const bool serialcut_gate = async_protocol && serialcut;
    const bool submit_only_gate = async_protocol && (serialcut || !gate_env || std::string(gate_env) == "1");
    const char * trim_env = std::getenv("GGML_TP5_DW_SERIALCUT_TRIM");
    // DUALWIN-SERIALCUT-TRIM: requires GGML_TP5_DW_SERIALCUT=1. No
    // validation here (matches the r4 serialcut gate convention): unset,
    // "0" and any other value are simply off.
    const bool trim_req = trim_env && std::strcmp(trim_env, "1") == 0;
    const bool trim_active = serialcut && trim_req;
    if (trim_active) {
        std::fprintf(stderr, "TP5_DW_SERIALCUT_TRIM serialcut=1 trim_active=1\n");
    }
    // DUALWIN-OVERLAP-METER: lock-hold metering reuses the existing
    // gate_metrics counters — GGML_TP5_DW_OVERLAP_METER=1 implies the
    // TP5_MTP_GATE_METRICS collection (same counters, same fields); no
    // second same-kind host counter exists.
    const bool gate_metrics_on = gate_metrics_req || overlap_meter_req;
    std::fprintf(stderr, "TP5_MTP_ASYNC_PROTOCOL enabled=%d ready_pair_requested=%d ready_pair_active=%d submit_only_gate=%d lock=fifo_spin\n",
        int(async_protocol), int(ready_pair_requested), int(paired_phases), int(submit_only_gate));
    if (serialcut) {
        std::fprintf(stderr, "TP5_DW_SERIALCUT submit_only_gate=%d overrides_explicit_off=%d async_protocol=%d\n",
            int(serialcut_gate),
            int(serialcut_gate && async_protocol && gate_env && std::string(gate_env) == "0"),
            int(async_protocol));
    }
    if (gate_metrics_req) {
        std::fprintf(stderr, "TP5_GATE_METRICS enabled=1 flag=TP5_MTP_GATE_METRICS\n");
    }
    if (gate_metrics_on && !gate_metrics_req) {
        std::fprintf(stderr, "TP5_GATE_METRICS enabled=1 flag=GGML_TP5_DW_OVERLAP_METER\n");
    }
    const char * unlock_env = std::getenv("TP5_MTP_POST_SUBMIT_UNLOCK");
    require(!unlock_env || std::string(unlock_env) == "0" || std::string(unlock_env) == "1",
            "MTP post-submit unlock must be 0 or 1");
    const bool post_submit_unlock = !async_protocol && unlock_env && std::string(unlock_env) == "1";
    require(!post_submit_unlock || paired_phases,
            "MTP post-submit unlock requires paired private native snapshot windows");
    require(!paired_phases || (mtp_draft_smoke && opts.windows == 2 && mtp_native_trim),
            "MTP phase pair requires two private native snapshot windows");
    require(prompt_audit_prefix.empty() || (prompt_catchup && opts.capture_logits),
            "prompt hidden audit is only valid in the numerical short run");
    require(!mtp_t4_probe || (opts.prompts.size() == 2 && opts.windows == 2),
            "MTP native T4 probe requires two distinct private windows");
    configure_hc_owner_layout(opts);
    tp5_plan_spin_cores(opts.windows);
    const bool                      swift = is_swift_model(opts.model);
    require(swift || opts.hc_owner_offsets.empty(), "HC context-owner offsets require the Swift HC97 model");
    require(swift || opts.execution != mode::relay || opts.context_per_request <= 512,
            "the Qwen3-8B profile supports at most 512 input/generation tokens per request");
    require(!swift || opts.context_per_request == 262144,
            "the Swift profile requires --ctx-size 262144; context shrinking is not supported");
    backend_owner                   backends;
    std::vector<ggml_backend_dev_t> devices;
    for (std::size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }
        const auto   registry = ggml_backend_dev_backend_reg(dev);
        const char * name     = ggml_backend_reg_name(registry);
        if (std::string(name) == "ROCm") {
            devices.push_back(dev);
        }
    }
    if (opts.execution == mode::relay || swift) {
        require(devices.size() == 5, "TP5 requires exactly five HIP GPU devices");
    } else {
        require(!devices.empty(), "the decoded-F32 reference requires a HIP GPU");
        if (devices.size() > 1) {
            devices.resize(1);
        }
    }
    devices.push_back(nullptr);
    auto model_params          = llama_model_default_params();
    model_params.devices       = devices.data();
    model_params.n_gpu_layers  = 99;
    model_params.split_mode    = opts.execution == mode::relay || swift ? LLAMA_SPLIT_MODE_TENSOR : LLAMA_SPLIT_MODE_NONE;
    model_params.offload_input = true;
    model_ptr model(llama_model_load_from_file(opts.model.c_str(), model_params));
    require(model != nullptr, "model load failed");
    const auto vocab   = llama_model_get_vocab(model.get());
    const auto n_vocab = llama_vocab_n_tokens(vocab);
    require(n_vocab > 0, "invalid model vocabulary");
    if (opts.capture_logits) {
        const auto bytes = std::uint64_t(opts.prompts.size()) * opts.predict * std::uint64_t(n_vocab) * sizeof(float);
        require(bytes <= 1024ULL * 1024 * 1024, "requested full logits exceed the 1 GiB bound");
    }
    run_result result;
    result.execution_mode = opts.execution == mode::relay ? "tp5-relay" :
                            swift ? "cpu-staged-five-gpu-reference" : "single-gpu-reference";
    // The opt-in loader converts HC weights once while preserving the source GGUF.
    // A successful load with this flag requires the complete 194-tensor profile.
    const char * hc_q8_weights = std::getenv("GGML_TP5_HC_Q8_WEIGHTS");
    const bool load_hc_q8 = hc_q8_weights && std::string(hc_q8_weights) == "1";
    const char * hc_q4_weights = std::getenv("GGML_TP5_HC_Q4_WEIGHTS");
    const bool load_hc_q4 = hc_q4_weights && std::string(hc_q4_weights) == "1";
    require(!(load_hc_q8 && load_hc_q4), "HC Q8 and Q4 weight modes are mutually exclusive");
    result.numeric_policy = opts.execution == mode::relay ?
                            (load_hc_q4 ? "original-gguf-load-time-hc-q4_0-weights-native-backend-activations-ordered-f32" :
                             load_hc_q8 ? "original-gguf-load-time-hc-q8_0-weights-native-backend-activations-ordered-f32" :
                                          "original-weights-native-backend-activations-ordered-f32") :
                            swift ? "mixed-original-weights-f32-activations-ordered-f32" :
                                    "q8_0-decoded-weights-f32-activations-ordered-f32";
    result.wire_format = opts.execution == mode::relay ? (opts.wire == wire_type::f16 ? "f16" : "f32") : "none";
    result.hc_weight_layout = opts.hc_owner_offsets.empty() ? "loader-default" : "mirrored-hc97";
    if (opts.execution == mode::relay && opts.wire == wire_type::f16) {
        result.numeric_policy += "-reduction-f16-wire";
    }
    result.vocabulary     = n_vocab;
    char description[256]{};
    llama_model_desc(model.get(), description, sizeof(description));
    result.model_description = description;
    result.requests.resize(opts.prompts.size());
    result.windows.resize(opts.windows);
    std::vector<std::vector<std::size_t>> assignments(opts.windows);
    scheduling::admission_budget          budget(std::uint64_t(opts.context_per_request) * opts.prompts.size(),
                                                 opts.prompts.size());
    std::vector<scheduling::reservation>  reservations;
    for (std::size_t i = 0; i < opts.prompts.size(); ++i) {
        auto & request        = result.requests[i];
        request.id            = i;
        request.window        = i % opts.windows;
        request.prompt_tokens = tokenize(vocab, opts.prompts[i]);
        require(request.prompt_tokens.size() + opts.predict <= opts.context_per_request,
                "prompt plus generation exceeds per-request context");
        if (!opts.teacher_tokens.empty()) {
            for (auto token : opts.teacher_tokens[i]) {
                require(token >= 0 && token < n_vocab, "teacher token outside vocabulary");
            }
        }
        auto held = budget.try_acquire(i + 1, opts.context_per_request);
        require(held.has_value(), "request reservation failed");
        reservations.push_back(*held);
        assignments[request.window].push_back(i);
    }
    std::vector<context_ptr>    contexts;
    std::vector<ggml_backend_t> native;
    // DUALWIN-BACKEND-EVENT per-window receipts (GGML_TP5_DW_BACKEND_EVENT=1):
    // one real ggml backend event per window, created after the window's
    // relay lane resolves; counters below are touched only while the gate is
    // on. event==nullptr (CPU or non-event child) keeps every site a counted
    // no-op — CPU-equivalent to OFF.
    struct dw_backend_event_state {
        ggml_backend_event_t event = nullptr;
        // DUALWIN-DEVICE-WAIT: this window's first-child device, resolved
        // once at event creation (single-threaded setup, before decode
        // workers run) through the public meta accessor; waits are issued
        // only when the peer resolved to the SAME device.
        ggml_backend_dev_t device = nullptr;
        std::uint64_t record_calls = 0, record_ok = 0, sync_calls = 0,
                      free_calls = 0, skipped_unavailable = 0;
        // DUALWIN-DEVICE-WAIT (GGML_TP5_DW_DEVICE_WAIT=1): wait counters.
        // These are touched only by this window's own decode worker and read
        // after the workers join. wait_peer_consumed is this window's
        // watermark of the peer's published count, so each record is waited
        // at most once and no wait ever targets an unrecorded event. (The
        // one cross-thread counter — the peer's release-published
        // availability — lives in the dw_wait_peer_available side array
        // below: std::atomic members would delete the struct's move
        // constructor and break vector resize.)
        std::uint64_t wait_calls = 0, wait_ok = 0,
                      wait_skipped_unrecorded = 0, wait_skipped_unavailable = 0,
                      wait_peer_consumed = 0;
                      // DUALWIN event-point trace stamps (host correlation
                      // metadata only; wait_calls counts the real device
                      // waits under GGML_TP5_DW_DEVICE_WAIT=1).
                      dw_event_trace::evpt ev;
    };
    std::vector<dw_backend_event_state> dw_backend_events;
    // DUALWIN-DEVICE-WAIT: per-window peer publication counters. The array
    // is allocated ONLY under the device-wait gate (the BACKEND_EVENT gate
    // alone must not gain a new allocation), once here (single-threaded
    // setup, before decode workers run) and never reallocated, so the record
    // site of window w publishes release into slot w while the PEER window's
    // consumption site reads it with acquire — the happens-before edge that
    // makes every issued wait refer to a record the peer's host already
    // returned from.
    std::unique_ptr<std::atomic<std::uint64_t>[]> dw_wait_peer_available;
    if (dw_device_wait_req) {
        dw_backend_events.resize(opts.windows);
        dw_wait_peer_available.reset(new std::atomic<std::uint64_t>[opts.windows] {});
    } else if (dw_backend_event_req) {
        dw_backend_events.resize(opts.windows);
    }
    for (std::uint32_t i = 0; i < opts.windows; ++i) {
        auto params      = llama_context_default_params();
        params.n_ctx     = opts.context_per_request * assignments[i].size();
        params.n_seq_max = assignments[i].size();
        params.n_batch   = mtp_t4_probe ? 4u : uint32_t(assignments[i].size());
        params.n_ubatch  = params.n_batch;
        params.n_threads = params.n_threads_batch = opts.threads;
        // Two full Swift contexts must fit alongside the shared model.
        const bool native_q4_kv = swift && opts.execution == mode::relay && opts.flash_attention;
        params.type_k = params.type_v = native_q4_kv ? GGML_TYPE_Q4_0 :
                                        swift ? GGML_TYPE_F16 : GGML_TYPE_F32;
        params.flash_attn_type        = opts.flash_attention ? LLAMA_FLASH_ATTN_TYPE_ENABLED :
                                                             LLAMA_FLASH_ATTN_TYPE_DISABLED;
        params.kv_unified             = true;
        params.tp5_relay              = opts.execution == mode::relay;
        params.n_rs_seq                = mtp_native_trim ? 2u : 0u;
        params.quantized_matmul_f32   = opts.execution != mode::relay;
        context_ptr context(llama_init_from_model(model.get(), params));
        require(context != nullptr, "context preparation failed");
        require(!mtp_native_trim || llama_n_rs_seq(context.get())==2,
                "target Qwen MTP depth2 native GDN ring was not admitted as n_rs_seq2");
        auto & dimensions = result.windows[i];
        dimensions.context_tokens = llama_n_ctx(context.get());
        dimensions.sequence_tokens = llama_n_ctx_seq(context.get());
        dimensions.kv_capacity_tokens = llama_n_ctx_kv(context.get());
        require(dimensions.context_tokens >= params.n_ctx &&
                    dimensions.sequence_tokens >= opts.context_per_request &&
                    dimensions.kv_capacity_tokens >= opts.context_per_request,
                "context preparation shrank the requested context or KV capacity");
        const bool flash_attention = llama_get_flash_attention(context.get());
        require(flash_attention == opts.flash_attention, "context did not honor the requested Flash Attention setting");
        result.flash_attention = flash_attention;
        if (opts.execution == mode::relay) {
            auto lane = llama_tp5_relay_backend(context.get());
            require(lane != nullptr, "context did not expose the requested TP5 relay capability");
            require(ggml_backend_relay_set_wire_type(
                        lane, opts.wire == wire_type::f16 ? GGML_RELAY_WIRE_F16 : GGML_RELAY_WIRE_F32),
                    "context rejected the requested relay wire format before decode");
            if (!opts.hc_owner_offsets.empty()) {
                require(ggml_backend_relay_set_hc_owner_offset(lane, opts.hc_owner_offsets[i]),
                        "context rejected the HC owner offset before its first decode");
                result.windows[i].hc_owner_offset = static_cast<std::int32_t>(opts.hc_owner_offsets[i]);
                result.hc_owner_offsets.push_back(opts.hc_owner_offsets[i]);
            }
            native.push_back(lane);
            // DUALWIN-BACKEND-EVENT: one real ggml backend event per window,
            // created from the window's Meta device (which forwards to the
            // first simple child device — real HIP/CUDA). nullptr (CPU or
            // no-event child) keeps every site a counted no-op.
            // DUALWIN-DEVICE-WAIT: the window's first-child device is
            // resolved here once (single-threaded setup) so the wait sites
            // can compare devices without touching event internals.
            if ((dw_backend_event_req || dw_device_wait_req) &&
                i < dw_backend_events.size() && dw_backend_events[i].event == nullptr) {
                auto & dw_be     = dw_backend_events[i];
                auto * dw_be_dev = ggml_backend_get_device(native.back());
                dw_be.device     = ggml_backend_meta_first_child_device(dw_be_dev);
                if (dw_backend_event_req) {
                    dw_be.event = dw_be_dev != nullptr ? ggml_backend_event_new(dw_be_dev) : nullptr;
                    if (dw_be.event == nullptr) {
                        ++dw_be.skipped_unavailable;
                    }
                }
            }
        }
        if (mtp_draft_smoke) {
            // Unmasked: export one genuine 10240-wide residual per token,
            // including all four prefill rows without requested vocab logits.
            llama_set_embeddings_nextn(context.get(), true, false);
        }
        contexts.push_back(std::move(context));
        result.windows[i].id = i;
    }

    // Real trained Strata Qwen4Exp MTP head, now converted to native NextN GGUF
    // with EXACT model token embeddings/LM head Q3_K/Q5_K source bytes.
    // Load on a distinct physical GPU, without fifth Meta rank scheduling the
    // draft graph or touching the two target contexts' KV/recurrent states.
    model_ptr trained_mtp_model;
    model_ptr trained_mtp_model_secondary;
    std::vector<context_ptr> trained_mtp_contexts;
    if (mtp_draft_smoke) {
        const char * split_draft_env = std::getenv("TP5_MTP_SPLIT_DRAFT_DEVICES");
        require(!split_draft_env || std::string(split_draft_env) == "0" ||
                std::string(split_draft_env) == "1", "split draft devices must be 0 or 1");
        const bool split_draft_devices = split_draft_env && std::string(split_draft_env) == "1";
        require(!split_draft_devices || (opts.windows == 2 && devices.size() >= 5 && devices[3]),
                "split draft devices requires the two private windows and ROCm3");
        std::array<ggml_backend_dev_t,2> ddevices{ devices[0], nullptr };
        auto mp = llama_model_default_params();
        mp.devices = ddevices.data();
        mp.n_gpu_layers = 99;
        mp.split_mode = LLAMA_SPLIT_MODE_NONE;
        mp.offload_input = true;
        mp.load_mtp = true;
        // The selected main trunk has 194 Q8-converted HC matrices. The
        // separate trained MTP-only model has exactly ONE own BF16 HC block
        // (blk.48), and MUST retain those trained source BF16 bytes. This
        // model-loader option is read fresh in every loader constructor;
        // change it in a strictly synchronous/no-thread interval only.
        // The original main trunk is already fully loaded and owns its
        // immutable physical Q8 descriptors. Restore the original env before
        // any target/draft compute or CPU relay service starts.
        struct scoped_hc_q8_loader_policy {
            std::string previous;
            bool had = false;
            scoped_hc_q8_loader_policy() {
                const char * p = std::getenv("GGML_TP5_HC_Q8_WEIGHTS");
                had = p != nullptr;
                previous = p ? p : "";
                require(::setenv("GGML_TP5_HC_Q8_WEIGHTS", "0", 1)==0,
                        "could not temporarily select trained BF16 HC MTP-only loader");
            }
            ~scoped_hc_q8_loader_policy() {
                if (had) ::setenv("GGML_TP5_HC_Q8_WEIGHTS", previous.c_str(), 1);
                else ::unsetenv("GGML_TP5_HC_Q8_WEIGHTS");
            }
        };
        {
            scoped_hc_q8_loader_policy mtp_only_weights;
            trained_mtp_model.reset(llama_model_load_from_file(mtp_path, mp));
            if (split_draft_devices) {
                ggml_backend_dev_props props{};
                ggml_backend_dev_get_props(devices[3], &props);
                std::fprintf(stderr,
                    "TP5_MTP_DRAFT_SECOND_MODEL name=%s bdf=%s free_before_bytes=%zu total_bytes=%zu\n",
                    props.name, props.device_id ? props.device_id : "", props.memory_free, props.memory_total);
                std::array<ggml_backend_dev_t,2> secondary_devices{ devices[3], nullptr };
                auto secondary_params = mp;
                secondary_params.devices = secondary_devices.data();
                secondary_params.main_gpu = 0;
                trained_mtp_model_secondary.reset(llama_model_load_from_file(mtp_path, secondary_params));
                require(trained_mtp_model_secondary != nullptr &&
                        llama_model_n_layer_nextn(trained_mtp_model_secondary.get()) == 1,
                        "second private MTP model failed to load on ROCm3");
            }
        }
        require(trained_mtp_model != nullptr && llama_model_n_layer_nextn(trained_mtp_model.get()) == 1,
                "real native trained MTP-only Qwen4Exp model could not be loaded");
        require(std::getenv("GGML_TP5_HC_Q8_WEIGHTS") &&
                std::string(std::getenv("GGML_TP5_HC_Q8_WEIGHTS"))=="1",
                "main Q8 HC loader policy not restored before native MTP compute");
        trained_mtp_contexts.reserve(opts.windows);
        for (std::uint32_t wi=0; wi<opts.windows; ++wi) {
            auto p=llama_context_default_params();
            p.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
            p.ctx_other = contexts[wi].get();
            p.n_ctx = opts.context_per_request;
            p.n_seq_max = 1;
            p.n_batch = 4;
            p.n_ubatch = 4;
            p.n_threads = p.n_threads_batch = opts.threads;
            p.type_k = p.type_v = GGML_TYPE_Q4_0;
            p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
            const std::uint32_t draft_device = split_draft_devices && wi == 1 ? 3u : 0u;
            auto * draft_model = draft_device == 3 ? trained_mtp_model_secondary.get() : trained_mtp_model.get();
            context_ptr dctx(llama_init_from_model(draft_model, p));
            require(dctx != nullptr && llama_n_ctx_kv(dctx.get()) >= opts.context_per_request,
                    "real MTP draft context failed to allocate full independent 262144 KV");
            llama_set_embeddings_nextn(dctx.get(), true, true);
            ggml_backend_dev_props props{};
            ggml_backend_dev_get_props(devices[draft_device], &props);
            std::fprintf(stderr,
                "TP5_MTP_DRAFT_ASSIGN window=%u index=%u name=%s bdf=%s model=%p context=%p kv_capacity=%u model_copies=%u free_after_bytes=%zu\n",
                wi, draft_device, props.name, props.device_id ? props.device_id : "",
                (void *) draft_model, (void *) dctx.get(), llama_n_ctx_kv(dctx.get()),
                split_draft_devices ? 2u : 1u, props.memory_free);
            trained_mtp_contexts.push_back(std::move(dctx));
        }
        require(trained_mtp_contexts.size()==opts.windows,
                "real MTP trained draft and target window count disagree");
    }
    failure_state                  failure;
    std::unique_ptr<relay_service> service;
    if (opts.execution == mode::relay) {
        service = std::make_unique<relay_service>(opts, native, failure, now_ns() | 1);
    }
    result.started_ns = now_ns();
    // Count submitted epochs until native work and CPU readers retire.
    std::atomic<std::uint64_t> active{ 0 }, maximum{ 0 };
    const char * timeline_env = std::getenv("TP5_MTP_TIMELINE_PREFIX");
    const std::string timeline_prefix = timeline_env ? timeline_env : "";
    std::vector<std::vector<cycle_timing_row>> cycle_timings(opts.windows);
    if (!timeline_prefix.empty()) {
        for (auto & rows : cycle_timings) rows.reserve(1024);
    }
    mtp_phase_pair phase_pair(opts.windows, paired_phases);
    submission_mutex_type      submission_mutex;
    mtp_submit_observation     submit_observation;
    std::atomic<std::uint32_t> workers_done{ 0 };
    std::atomic<bool>          watchdog_stop{ false };
    std::vector<std::thread>   workers;
    workers.reserve(opts.windows);
    std::thread watchdog([&] {
        const auto deadline           = clock_type::now() + std::chrono::seconds(opts.deadline_seconds);
        bool       aborted            = false;
        auto       retention_deadline = deadline + std::chrono::seconds(45);
        while (!watchdog_stop.load(std::memory_order_acquire)) {
            if (!aborted && (clock_type::now() >= deadline || failure.stopped.load())) {
                if (!failure.stopped.load()) {
                    failure.fail(std::make_exception_ptr(std::runtime_error("TP5 execution deadline exceeded")));
                }
                if (service) {
                    service->abort_all();
                }
                aborted            = true;
                retention_deadline = clock_type::now() + std::chrono::seconds(45);
            }
            if (aborted && workers_done.load() != opts.windows && clock_type::now() >= retention_deadline) {
                retain_and_exit("decode or native drain did not return within the abort grace period");
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    });
    for (std::uint32_t wi = 0; wi < opts.windows; ++wi) {
        try {
            workers.emplace_back([&, wi] {
                tl_tp5_window = int(wi);
                bool epoch_active = false;
                try {
                    auto                                 ctx     = contexts[wi].get();
                    auto &                               metrics = result.windows[wi];
                    const auto &                         ids     = assignments[wi];
                    std::vector<std::uint64_t>           consumed(ids.size(), 0);
                    std::vector<text::generation_stream> streams(ids.size());
                    row_batch                            batch(mtp_t4_probe ? 4u : uint32_t(ids.size()));
                    // A single REAL trained MTP speculative round is verified
                    // with target batch [previously emitted target token,
                    // trained MTP proposal], then either committed by the
                    // target or fully rolled back (KV + recurrent).
                    std::vector<llama_token>            pending_mtp_tokens;
                    std::uint64_t                       mtp_proposal_tokens = 0;
                    std::uint64_t                       mtp_accepted_draft_tokens = 0;
                    std::uint64_t                       mtp_verify_rounds = 0;
                    std::uint64_t                       mtp_accept_rounds = 0;
                    std::uint64_t                       mtp_rollback_rounds = 0;
                    const auto diag_begin=now_ns();
                    std::uint64_t diag_draft_first=0,diag_draft_extra=0,diag_state=0,diag_queue=0;
                    std::uint64_t diag_first_count=0,diag_extra_count=0,diag_state_count=0;
                    std::uint64_t diag_state_only_count=0;
                    bool draft_pending = false;
                    std::uint64_t draft_phase_drains = 0;
                    std::uint64_t pair_before_ns=0, pair_after_ns=0, pair_meets=0;
                    std::uint64_t pair_ready_ns = 0, pair_ready_hits = 0, pair_submitted_hits = 0;
                    gate_metrics gm; // DUALWIN-MUTEX-MEET gate metrics
                    std::vector<float> draft_prompt_previous_h(10240, 0.0f);
                    std::size_t draft_prompt_rows=0;
                    std::uint64_t draft_prompt_ns=0;
                    auto timed_seq_rm = [&](llama_memory_t memory, llama_seq_id seq, llama_pos p0, llama_pos p1) {
                        const auto start=now_ns();
                        const bool result=llama_memory_seq_rm(memory,seq,p0,p1);
                        diag_state+=now_ns()-start; ++diag_state_count;
                        return result;
                    };
                    std::vector<std::uint8_t>           target_state_checkpoint;
                    std::vector<std::size_t>             local_rows;
                    local_rows.reserve(ids.size());
                    for (auto id : ids) {
                        result.requests[id].admitted_ns = now_ns();
                    }
                    while (!failure.stopped.load()) {
                        batch.clear();
                        local_rows.clear();
                        std::uint32_t output_rows = 0;
                        std::int32_t single_output_row = -1;
                        bool verify_mtp_this_epoch = false;
                        if (mtp_draft_smoke && !pending_mtp_tokens.empty() &&
                            ids.size()==1 &&
                            result.requests[ids[0]].tokens.size()+1 < opts.predict) {
                            auto & rr = result.requests[ids[0]];
                            require(!rr.tokens.empty() &&
                                    consumed[0] == rr.prompt_tokens.size()+rr.tokens.size()-1,
                                    "MTP N4 target verification cursor misaligned");
                            require(rr.stop_reason.empty(), "MTP N4 after stopped request");
                            require(pending_mtp_tokens.size()<=3 &&
                                    pending_mtp_tokens.size()+1<=4,
                                    "MTP N4 too many draft inputs for true native TP5");
                            require(pending_mtp_tokens.size()+1<=opts.predict-rr.tokens.size(),
                                    "MTP N4 would generate more true tokens than remaining budget");
                            for (const auto tok : pending_mtp_tokens)
                                require(tok>=0 && std::uint32_t(tok)<n_vocab,
                                        "MTP prediction ID outside real vocab");

                            // Save BOTH real attention KV and hybrid GDN recurrent
                            // state. The native two-slot GDN ring failed full
                            // 248320 logits quality (KL 7.31), so use the ONLY
                            // qualified full host sequence snapshot.
                            std::size_t state_sz=0;
                            if(!mtp_native_trim) {
                                state_sz=llama_state_seq_get_size_ext(ctx,0,checkpoint_flags);
                                require(state_sz>0 && state_sz<=(512ull<<20),
                                        "MTP valid full GDN+KV checkpoint unavailable");
                                target_state_checkpoint.resize(state_sz);
                                const auto written=llama_state_seq_get_data_ext(
                                    ctx,target_state_checkpoint.data(),state_sz,0,checkpoint_flags);
                                require(written==state_sz,
                                        "MTP full target state snapshot failed");
                            }
                            batch.push(rr.tokens.back().emitted_token,consumed[0],0,true);
                            local_rows.push_back(0);
                            for (std::size_t i=0;i<pending_mtp_tokens.size();++i) {
                                batch.push(pending_mtp_tokens[i],consumed[0]+i+1,0,true);
                                local_rows.push_back(0);
                            }
                            output_rows=std::uint32_t(batch.batch.n_tokens);
                            single_output_row=-1;
                            verify_mtp_this_epoch=true;
                            mtp_proposal_tokens+=pending_mtp_tokens.size();
                            if (mtp_state_trace) {
                                std::fprintf(stderr,
                                    "TP5_REAL_MTP_TARGET_MULTI_BEGIN window=%u N=%u draft_n=%zu source_pos=%llu snapshot_bytes=%zu draft0=%d\n",
                                    wi,unsigned(output_rows),pending_mtp_tokens.size(),
                                    (unsigned long long)consumed[0],state_sz,int(pending_mtp_tokens[0]));
                                std::fflush(stderr);
                            }
                        }
                        if (mtp_t4_probe && ids.size() == 1 && consumed[0] == 0) {
                            auto & initial = result.requests[ids[0]];
                            require(initial.prompt_tokens.size() >= 5,
                                    "MTP T4 verification probe requires prompt length >=5");
                            for (std::uint32_t n = 0; n < 4; ++n) {
                                batch.push(initial.prompt_tokens[n], n, 0, false);
                                local_rows.push_back(0);
                            }
                        }
                        for (std::size_t local = 0; local < ids.size(); ++local) {
                            if (verify_mtp_this_epoch ||
                                (mtp_t4_probe && consumed[local] == 0 && !local_rows.empty())) {
                                continue; // Actual T4 prefill or T2 verification already assembled.
                            }
                            auto & request = result.requests[ids[local]];
                            if (!request.stop_reason.empty()) {
                                continue;
                            }
                            llama_token input;
                            if (consumed[local] < request.prompt_tokens.size()) {
                                input = request.prompt_tokens[consumed[local]];
                            } else {
                                require(!request.tokens.empty(), "request has no accepted pending token");
                                input = request.tokens.back().emitted_token;
                            }
                            const bool output = consumed[local] + 1 >= request.prompt_tokens.size();
                            batch.push(input, consumed[local], local, output);
                            if (output) {
                                ++output_rows;
                                single_output_row = batch.batch.n_tokens - 1;
                            }
                            local_rows.push_back(local);
                        }
                        if (!batch.batch.n_tokens) {
                            break;
                        }
                        if (paired_phases) {
                            const auto begin = now_ns();
                            if (draft_pending) {
                                llama_synchronize(trained_mtp_contexts.at(wi).get());
                                draft_pending = false;
                                ++draft_phase_drains;
                            }
                            require(phase_pair.meet(failure.stopped), "MTP phase pair stopped before target");
                            pair_before_ns += now_ns() - begin;
                            ++pair_meets;
                            if (gate_metrics_on) {
                                gm.meet_before_ns += now_ns() - begin;
                                ++gm.meet_before_calls;
                            }
                        }
                        const auto cycle_queue_start = timeline_prefix.empty() ? 0 : now_ns();
                        const auto cycle_rows = batch.batch.n_tokens;
                        const auto cycle_position = batch.batch.pos[0];
                        // Keep each window's five-rank submission together.
                        // Completion waits and CPU relay work stay outside this gate.
                        // Default async protocol: this process-wide ticket lock
                        // serializes both windows' enqueues. It is always taken:
                        // GGML_TP5_DW_SERIALCUT=1 does not remove it, and the
                        // window's private Meta relay provider additionally holds
                        // its own submit mutex across all five ranks.
                        std::unique_lock<submission_mutex_type> submission_lock(submission_mutex, std::defer_lock);
                        std::uint64_t gate_hold_start = 0;
                        if (service && !submit_only_gate) {
                            const auto queue_start=now_ns();
                            submission_lock.lock();
                            diag_queue+=now_ns()-queue_start;
                            if (gate_metrics_req) {
                                // Approximate hold interval: timestamp is taken
                                // inside the lock boundaries, so it excludes the
                                // hold before this point and may include unlocked
                                // gaps (ready-scope hooks release/re-take when
                                // paired_phases).
                                gate_hold_start = now_ns();
                            }
                        }
                        const auto current = active.fetch_add(1) + 1;
                        epoch_active = true;
                        auto       seen    = maximum.load();
                        while (seen < current && !maximum.compare_exchange_weak(seen, current)) {
                        }
                        if (!metrics.first_submit_ns) {
                            metrics.first_submit_ns = now_ns();
                        }
                        if (overlap_meter_req) {
                            dw_overlap_meter.begin(wi);
                        }
                        const auto submit_start = now_ns();
                        int decoded;
                        {
                            mtp_submit_ready_scope ready(paired_phases, post_submit_unlock, phase_pair, submission_lock,
                                failure.stopped, wi, metrics.decode_calls + 1, pair_ready_ns, pair_ready_hits,
                                pair_submitted_hits);
                            mtp_submit_only_gate gate(service && submit_only_gate,
                                submission_lock, diag_queue,
                                gate_metrics_on ? &gm : nullptr,
                                gate_metrics_on ? &submit_observation : nullptr,
                                trim_active);
                            decoded = llama_decode(ctx, batch.batch);
                            ready.check(decoded);
                            gate.check(decoded);
                        }
                        const auto submit_finished = now_ns();
                        metrics.decode_submit_ns += submit_finished - submit_start;
                        if (submission_lock.owns_lock()) {
                            submission_lock.unlock();
                        }
                        if (gate_metrics_on && gate_hold_start) {
                            gm.hold_ns += now_ns() - gate_hold_start;
                            ++gm.holds;
                        }
                        ++metrics.decode_calls;
                        metrics.input_rows += batch.batch.n_tokens;
                        require(decoded == 0, "real model decode failed");
                        // Record on the Meta backend's first simple child.
                        // Its stream relationship to relay work is unverified.
                        // The record also publishes (release) the peer-wait
                        // availability counter under the device-wait gate.
                        if (dw_backend_event_req && wi < dw_backend_events.size() &&
                            dw_backend_events[wi].event != nullptr) {
                            auto & dw_be = dw_backend_events[wi];
                            ++dw_be.record_calls;
                            const std::uint64_t dw_ev_t0 = dw_event_trace::gate() ? now_ns() : 0;
                            ggml_backend_event_record(dw_be.event, native[wi]);
                            ++dw_be.record_ok;
                            if (dw_device_wait_req) {
                                dw_wait_peer_available[wi].fetch_add(1, std::memory_order_release);
                            }
                            if (dw_ev_t0) {
                                if (dw_be.ev.record_first_ns == 0) dw_be.ev.record_first_ns = dw_ev_t0;
                                dw_be.ev.record_last_ns = dw_ev_t0;
                            }
                        }
                        // DUALWIN-DEVICE-WAIT (GGML_TP5_DW_DEVICE_WAIT=1,
                        // requires GGML_TP5_DW_BACKEND_EVENT=1): enqueue ONE
                        // device-side wait on this window's first-child stream
                        // for the peer window's most recent unwaited record —
                        // the cross-window device dependency the r7 wire and
                        // the r8 trace documented as missing (wait_calls 0).
                        // Gate parsing, peer selection, the release/acquire
                        // pairing, the watermark and the acyclicity argument
                        // live in dw_device_wait.h; the call site only feeds
                        // it the peer state and issues the non-blocking
                        // enqueue it asks for. The wait binds at enqueue time
                        // to a record the peer's host already returned from,
                        // so re-records cannot retarget it; it is issued only
                        // for the peer's event on the SAME first-child device
                        // and only against an unconsumed record; every skip
                        // is counted. It never weakens an exit predicate: the
                        // following host synchronize still waits for this
                        // stream to drain — now transitively covering the
                        // peer's latest recorded work as well. With either
                        // gate off this block is one false bool branch: stock
                        // behaviour.
                        if (dw_device_wait_req && wi < dw_backend_events.size()) {
                            auto & dw_self = dw_backend_events[wi];
                            // wait_calls counts REAL device-side waits: it is
                            // bumped only in the branch that actually calls
                            // ggml_backend_event_wait, never for skips — the
                            // receipt's wait_calls>0 is direct evidence that
                            // device-side waits were issued (the r8 field
                            // documented the no-wait gap by staying 0).
                            const std::uint32_t dw_n =
                                std::uint32_t(dw_backend_events.size());
                            const std::uint32_t dw_peer_wi = dw_device_wait::peer_of(wi, dw_n);
                            const dw_device_wait::context dw_ctx{
                                dw_backend_event_req,
                                dw_device_wait::has_peer(wi, dw_n),
                                dw_peer_wi < dw_n && dw_backend_events[dw_peer_wi].event != nullptr,
                                dw_self.event != nullptr,
                                // Same first-child device, resolved once at
                                // event creation through the public meta
                                // accessor (unknown device never waits).
                                dw_peer_wi < dw_n &&
                                    dw_backend_events[dw_peer_wi].device != nullptr &&
                                    dw_backend_events[dw_peer_wi].device ==
                                        dw_self.device,
                            };
                            const std::uint64_t dw_peer_avail =
                                dw_ctx.peer_event
                                    ? dw_wait_peer_available[dw_peer_wi].load(
                                          std::memory_order_acquire)
                                    : 0;
                            const dw_device_wait::action dw_act =
                                dw_device_wait::plan(dw_ctx, dw_peer_avail, dw_self.wait_peer_consumed);
                            if (dw_act == dw_device_wait::action::wait) {
                                const std::uint64_t dw_wait_t0 = dw_event_trace::gate() ? now_ns() : 0;
                                ggml_backend_event_wait(native[wi], dw_backend_events[dw_peer_wi].event);
                                ++dw_self.wait_calls;
                                if (dw_event_trace::gate()) ++dw_self.ev.wait_calls;
                                ++dw_self.wait_ok;
                                if (dw_wait_t0) {
                                    if (dw_self.ev.wait_first_ns == 0) dw_self.ev.wait_first_ns = dw_wait_t0;
                                    dw_self.ev.wait_last_ns = dw_wait_t0;
                                }
                            } else if (dw_act == dw_device_wait::action::skip_unavailable) {
                                ++dw_self.wait_skipped_unavailable;
                            } else if (dw_act == dw_device_wait::action::skip_unrecorded) {
                                ++dw_self.wait_skipped_unrecorded;
                            }
                        }
                        std::optional<scoped_spin_cpu> verify_spin;
                        verify_spin.emplace();
                        const auto wait_start = now_ns();
                        const float * single_output_logits = nullptr;
                        if (output_rows == 1) {
                            // This public getter synchronizes the context. Keep only
                            // its pointer here; read logits after native retirement.
                            single_output_logits = llama_get_logits_ith(ctx, single_output_row);
                        } else {
                            llama_synchronize(ctx);
                        }
                        if (service) {
                            ggml_relay_status status{};
                            if (!service->metrics_enabled()) {
                                if (service->dual_enabled()) {
                                    service->dual_wait_plain(wi, native[wi], "target");
                                } else
                                for (;;) {
                                    if (!ggml_backend_relay_snapshot(native[wi], &status)) {
                                        pause_cpu();
                                        continue;
                                    }
                                    require(!status.failed && !failure.stopped.load(), "native relay epoch failed");
                                    if (status.finished && service->retired(wi, status)) {
                                        break;
                                    }
                                    pause_cpu();
                                }
                                    } else if (service->dual_enabled()) {
                                        service->dual_wait_metrics(wi, native[wi], "target");
                                    } else {
                                // One demand sample per epoch at the first finished
                                // snapshot. Ablation keeps refreshing the snapshot
                                // through the drain (baseline mutex behaviour, i.e.
                                // F2 off); the gated arm stops snapshotting after
                                // finished so the courier can drain the tail.
                                const bool    keep_snapshot = service->ablate();
                                bool          finished    = false;
                                std::uint64_t drain_start = 0, drain_passes = 0;
                                for (;;) {
                                    if (!finished) {
                                        if (!ggml_backend_relay_snapshot(native[wi], &status)) {
                                            pause_cpu();
                                            continue;
                                        }
                                        require(!status.failed && !failure.stopped.load(),
                                                "native relay epoch failed");
                                        if (!status.finished) {
                                            pause_cpu();
                                            continue;
                                        }
                                        finished = true;
                                        service->epoch_demand(wi, status);
                                        drain_start = now_ns();
                                    }
                                    if (keep_snapshot && !ggml_backend_relay_snapshot(native[wi], &status)) {
                                        pause_cpu();
                                        continue;
                                    }
                                    require(!status.failed && !failure.stopped.load(), "native relay epoch failed");
                                    if (service->retired(wi, status)) {
                                        break;
                                    }
                                    ++drain_passes;
                                    pause_cpu();
                                }
                                service->epoch_drain(wi, now_ns() - drain_start, drain_passes);
                                // status holds the finished snapshot whose
                                // cpu_reductions is final; that drained value is
                                // the only point where the delta baseline moves.
                                service->epoch_end(wi, status.cpu_reductions);
                                service->epoch_verify(wi, native[wi]);
                            }
                        }
                        const auto wait_finished = now_ns();
                        if (overlap_meter_req) {
                            dw_overlap_meter.end(wi);
                        }
                        verify_spin.reset();
                        metrics.native_wait_ns += wait_finished - wait_start;
                        active.fetch_sub(1);
                        epoch_active = false;
                        require(!failure.stopped.load(), "request stopped before state commit");
                        if (paired_phases) {
                            const auto begin = now_ns();
                            require(phase_pair.meet(failure.stopped), "MTP phase pair stopped after target");
                            pair_after_ns += now_ns() - begin;
                            if (gate_metrics_on) {
                                gm.meet_after_ns += now_ns() - begin;
                                ++gm.meet_after_calls;
                            }
                        }
                        if (verify_mtp_this_epoch) {
                            const std::size_t n=pending_mtp_tokens.size();
                            require(n>=1&&n<=3&&local_rows.size()==n+1&&output_rows==n+1,
                                "MTP multistep must have exactly 1+N true target output rows");
                            std::size_t accepted=0;
                            for (;accepted<n;++accepted) {
                                const float * logits=llama_get_logits_ith(ctx,int(accepted));
                                require(logits!=nullptr,"MTP multi target logits missing");
                                const auto greedy=tq::sampling::select_finite_greedy(logits,n_vocab);
                                require(greedy.usable,"MTP multi target logits nonfinite");
                                if(greedy.selected.id!=pending_mtp_tokens[accepted])break;
                            }
                            const bool full_accept=accepted==n;
                            ++mtp_verify_rounds;
                            mtp_accepted_draft_tokens+=accepted;
                            if(full_accept)++mtp_accept_rounds;
                            else ++mtp_rollback_rounds;
                            if (mtp_state_trace) {
                                std::fprintf(stderr,
                                    "TP5_REAL_MTP_MULTI_VERDICT window=%u N=%zu accepted_prefix=%zu full=%u source_pos=%llu\n",
                                    wi,n+1,accepted,unsigned(full_accept),
                                    (unsigned long long)consumed[0]);
                                std::fflush(stderr);
                            }

                            // The trained single-NextN head drafted and cached
                            // predicted tokens at positions > source x. Drop
                            // ALL predicted draft KV, retaining only the true
                            // emitted source token x at consumed[0]. The next
                            // actual target output rows will re-enter the
                            // trained head with REAL target hidden states.
                            auto * dctx=trained_mtp_contexts.at(wi).get();
                            require(timed_seq_rm(llama_get_memory(dctx),0,
                                       static_cast<llama_pos>(consumed[0]+1),-1),
                                    "MTP draft attention KV speculative tail trim failed");
                            pending_mtp_tokens.clear();

                            if (!full_accept) {
                                if(mtp_native_trim) {
                                    // The actual N3 target processed TRUE x at
                                    // pos p plus up to 2 tentative trained
                                    // NextN proposals y0,y1. Only the accepted
                                    // prefix [x, y0..y(accepted-1)] is VALID.
                                    // Rewind INVALID suffix beginning
                                    // p+1+accepted, at most TWO GDN steps;
                                    // preserve the model's already-computed
                                    // recurrent GDN/PLE/attention/indexer state
                                    // for valid prefix. No 118MB state copy and
                                    // no costly second target GPU decode.
                                    const llama_pos invalid_from=static_cast<llama_pos>(
                                        consumed[0]+1+accepted);
                                    auto debug_recurrent = [&](const char * stage) {
                                        if (!mtp_state_trace) return;
                                        auto * mem = dynamic_cast<llama_memory_hybrid_idx *>(llama_get_memory(ctx));
                                        require(mem != nullptr,
                                            "MTP ring-state trace requires Qwen4Exp hybrid_idx");
                                        const auto & hist=mem->ple_hist_get(0);
                                        auto * rec=mem->get_mem_recr();
                                        const int tail=rec->tails[0];
                                        const auto last_pos=tail>=0 ? rec->cells[tail].pos : -1;
                                        const int ht1=hist.toks.empty() ? -1 : int(hist.toks.back());
                                        const int ht2=hist.toks.size()<2 ? -1 : int(hist.toks[hist.toks.size()-2]);
                                        const auto idxpos=mem->get_mem_idx() ?
                                            mem->get_mem_idx()->seq_pos_max(0) : -1;
                                        const auto attpos=mem->get_mem_attn()->seq_pos_max(0);
                                        std::fprintf(stderr,
                                            "TP5_REAL_MTP_RECURRENT_PLE_KV_TRACE stage=%s window=%u source_pos=%llu accepted_drafts=%zu invalid_pos=%d GDN_tail_pos=%d replay_len=%u rs_idx=%u PLE_next_pos=%d PLE_count=%zu PLE_recent2=[%d,%d] attention_pos_max=%d indexer_pos_max=%d\n",
                                            stage,wi,(unsigned long long)consumed[0],
                                            accepted,int(invalid_from),int(last_pos),
                                            rec->replay_len.empty()?0:rec->replay_len[0],
                                            rec->rs_idx.empty()?0:rec->rs_idx[0],
                                            int(hist.next_pos),hist.toks.size(),ht2,ht1,
                                            int(attpos),int(idxpos));
                                        std::fflush(stderr);
                                    };
                                    debug_recurrent("before_invalid_suffix_trim");
                                    require(timed_seq_rm(
                                                llama_get_memory(ctx),0,invalid_from,-1),
                                            "bounded native Qwen recurrent/PLE/attention/indexer speculative suffix trim failed");
                                    debug_recurrent("after_invalid_suffix_trim");
                                    const auto valid_rows=accepted+1;
                                    local_rows.resize(valid_rows);
                                    batch.batch.n_tokens=int32_t(valid_rows);
                                    output_rows=uint32_t(valid_rows);
                                    if(valid_rows==1) {
                                        single_output_row=0;
                                        single_output_logits=llama_get_logits_ith(ctx,0);
                                        require(single_output_logits,
                                                "native ring missing verified target N3 row0");
                                    }
                                    if (mtp_state_trace) {
                                        std::fprintf(stderr,
                                            "TP5_REAL_MTP_MULTI_NATIVE_SUFFIX_TRIM window=%u accepted_drafts=%zu valid_rows=%zu removed_invalid_start=%d rollback_steps=%zu host_checkpoint_bytes=0 extra_target_replay=0\n",
                                            wi,accepted,valid_rows,int(invalid_from),n-accepted);
                                        std::fflush(stderr);
                                    }
                                } else {
                                // The target N4 executed 1+N speculative inputs.
                                // Only the first [x,d0,...,d_{k-1}] were
                                // verified as a legitimate chain. Keep copies
                                // of those original verifier logits BEFORE the
                                // full restore; replay overwrites host output.
                                const std::size_t replay_n=accepted+1;
                                std::vector<std::vector<float>> verifier_rows(replay_n);
                                for (std::size_t j=0;j<replay_n;++j) {
                                    const float * t=llama_get_logits_ith(ctx,int(j));
                                    require(t!=nullptr,"MTP rejected prefix verifier logits missing");
                                    verifier_rows[j].assign(t,t+n_vocab);
                                }
                                require(!target_state_checkpoint.empty(),
                                        "MTP multi full target checkpoint missing");
                                const std::size_t restored=llama_state_seq_set_data_ext(
                                    ctx,target_state_checkpoint.data(),
                                    target_state_checkpoint.size(),0,checkpoint_flags);
                                require(restored==target_state_checkpoint.size(),
                                        "MTP multi full GDN+attention KV restore failed");
                                target_state_checkpoint.clear();
                                std::vector<llama_token> verified_inputs(replay_n);
                                std::vector<llama_pos> verified_positions(replay_n);
                                for(std::size_t j=0;j<replay_n;++j){
                                    verified_inputs[j]=batch.batch.token[j];
                                    verified_positions[j]=batch.batch.pos[j];
                                }
                                batch.clear();
                                for(std::size_t j=0;j<replay_n;++j)
                                    batch.push(verified_inputs[j],verified_positions[j],0,true);
                                local_rows.resize(replay_n);
                                output_rows=std::uint32_t(replay_n);
                                single_output_row=replay_n==1 ? 0 : -1;
                                std::unique_lock<submission_mutex_type> replay_lock(
                                    submission_mutex,std::defer_lock);
                                if(service && !submit_only_gate)replay_lock.lock();
                                const std::uint64_t gm_replay_hold_start = (gate_metrics_on && service && !submit_only_gate) ? now_ns() : 0;
                                const auto active_count=active.fetch_add(1)+1;
                                epoch_active=true;
                                auto seen=maximum.load();
                                while(seen<active_count &&
                                      !maximum.compare_exchange_weak(seen,active_count)){}
                                if (overlap_meter_req) {
                                    dw_overlap_meter.begin(wi);
                                }
                                const auto begin_replay=now_ns();
                                int ret;
                                {
                                    mtp_submit_only_gate replay_gate(service && submit_only_gate,
                                        replay_lock, diag_queue,
                                        gate_metrics_on ? &gm : nullptr,
                                        gate_metrics_on ? &submit_observation : nullptr,
                                        trim_active);
                                    ret=llama_decode(ctx,batch.batch);
                                    replay_gate.check(ret);
                                }
                                metrics.decode_submit_ns+=now_ns()-begin_replay;
                                if(replay_lock.owns_lock())replay_lock.unlock();
                                if (gate_metrics_on && gm_replay_hold_start) {
                                    gm.replay_hold_ns += now_ns() - gm_replay_hold_start;
                                    ++gm.replay_holds;
                                }
                                ++metrics.decode_calls;
                                metrics.input_rows+=batch.batch.n_tokens;
                                require(ret==0,"MTP multi verified-target-prefix replay failed");
                                // Record on the Meta backend's first simple
                                // child; its relationship to replay work is
                                // unverified. The record also publishes
                                // (release) the peer-wait availability counter
                                // under the device-wait gate.
                                if (dw_backend_event_req && wi < dw_backend_events.size() &&
                                    dw_backend_events[wi].event != nullptr) {
                                    auto & dw_be = dw_backend_events[wi];
                                    ++dw_be.record_calls;
                                    const std::uint64_t dw_ev_t0 = dw_event_trace::gate() ? now_ns() : 0;
                                    ggml_backend_event_record(dw_be.event, native[wi]);
                                    ++dw_be.record_ok;
                                    if (dw_device_wait_req) {
                                        dw_wait_peer_available[wi].fetch_add(1, std::memory_order_release);
                                    }
                                    if (dw_ev_t0) {
                                        if (dw_be.ev.record_first_ns == 0) dw_be.ev.record_first_ns = dw_ev_t0;
                                        dw_be.ev.record_last_ns = dw_ev_t0;
                                    }
                                }
                                // DUALWIN-DEVICE-WAIT: same contract as the
                                // target site above — dw_device_wait.h owns
                                // the decision (binding at enqueue time per
                                // the CUDA/HIP event contract, same
                                // first-child device only, skips counted,
                                // exits never weakened); the replay epoch is
                                // its own relay epoch, so its consumption
                                // covers the peer's latest recorded work the
                                // same way, and the host synchronize below
                                // still drains this stream afterwards.
                                if (dw_device_wait_req && wi < dw_backend_events.size()) {
                                    auto & dw_self = dw_backend_events[wi];
                                    // wait_calls counts REAL device-side waits
                                    // (bumped only where the call is actually
                                    // issued), never skips — same contract as
                                    // the target site above.
                                    const std::uint32_t dw_n =
                                        std::uint32_t(dw_backend_events.size());
                                    const std::uint32_t dw_peer_wi = dw_device_wait::peer_of(wi, dw_n);
                                    const dw_device_wait::context dw_ctx{
                                        dw_backend_event_req,
                                        dw_device_wait::has_peer(wi, dw_n),
                                        dw_peer_wi < dw_n && dw_backend_events[dw_peer_wi].event != nullptr,
                                        dw_self.event != nullptr,
                                        // Same first-child device, resolved once
                                        // at event creation through the public
                                        // meta accessor (unknown device never
                                        // waits).
                                        dw_peer_wi < dw_n &&
                                            dw_backend_events[dw_peer_wi].device != nullptr &&
                                            dw_backend_events[dw_peer_wi].device ==
                                                dw_self.device,
                                    };
                                    const std::uint64_t dw_peer_avail =
                                        dw_ctx.peer_event
                                            ? dw_wait_peer_available[dw_peer_wi].load(
                                                  std::memory_order_acquire)
                                            : 0;
                                    const dw_device_wait::action dw_act =
                                        dw_device_wait::plan(dw_ctx, dw_peer_avail, dw_self.wait_peer_consumed);
                                    if (dw_act == dw_device_wait::action::wait) {
                                        const std::uint64_t dw_wait_t0 = dw_event_trace::gate() ? now_ns() : 0;
                                        ggml_backend_event_wait(native[wi], dw_backend_events[dw_peer_wi].event);
                                        ++dw_self.wait_calls;
                                        if (dw_event_trace::gate()) ++dw_self.ev.wait_calls;
                                        ++dw_self.wait_ok;
                                        if (dw_wait_t0) {
                                            if (dw_self.ev.wait_first_ns == 0) dw_self.ev.wait_first_ns = dw_wait_t0;
                                            dw_self.ev.wait_last_ns = dw_wait_t0;
                                        }
                                    } else if (dw_act == dw_device_wait::action::skip_unavailable) {
                                        ++dw_self.wait_skipped_unavailable;
                                    } else if (dw_act == dw_device_wait::action::skip_unrecorded) {
                                        ++dw_self.wait_skipped_unrecorded;
                                    }
                                }
                                std::optional<scoped_spin_cpu> replay_spin;
                                replay_spin.emplace();
                                llama_synchronize(ctx);
                                if (overlap_meter_req) {
                                    dw_overlap_meter.end(wi);
                                }
                                if(service) {
                                    ggml_relay_status status{};
                                    if (!service->metrics_enabled()) {
                                        if (service->dual_enabled()) {
                                            service->dual_wait_plain(wi, native[wi], "replay");
                                        } else
                                        // Ablation arm: exact baseline loop (F2 off — the
                                        // worker keeps snapshotting, i.e. taking the
                                        // provider mutex, through the drain).
                                        for(;;) {
                                            if(!ggml_backend_relay_snapshot(native[wi],&status)){
                                                pause_cpu();continue;
                                            }
                                            require(!status.failed&&!failure.stopped.load(),
                                                "native relay failed MTP multi prefix replay");
                                            if(status.finished&&service->retired(wi,status))break;
                                            pause_cpu();
                                        }
                                    } else if (service->dual_enabled()) {
                                        service->dual_wait_metrics(wi, native[wi], "replay");
                                    } else {
                                        // Same one-sample-per-epoch rule as the target
                                        // wait: the replay is its own relay epoch.
                                        // Ablation samples too: keep_snapshot
                                        // refreshes the snapshot through the drain
                                        // (baseline mutex behaviour), so both arms
                                        // sample every epoch; arms differ in F1+F2.
                                        const bool keep_snapshot = service->ablate();
                                        bool finished=false;
                                        std::uint64_t drain_start=0,drain_passes=0;
                                        for(;;) {
                                            if(!finished) {
                                                if(!ggml_backend_relay_snapshot(native[wi],&status)){
                                                    pause_cpu();continue;
                                                }
                                                require(!status.failed&&!failure.stopped.load(),
                                                    "native relay failed MTP multi prefix replay");
                                                if(!status.finished){pause_cpu();continue;}
                                                finished=true;
                                                service->epoch_demand(wi,status);
                                                drain_start=now_ns();
                                            }
                                            if (keep_snapshot && !ggml_backend_relay_snapshot(native[wi], &status)) {
                                                pause_cpu();continue;
                                            }
                                            require(!status.failed&&!failure.stopped.load(),
                                                "native relay failed MTP multi prefix replay");
                                            if(service->retired(wi,status))break;
                                            ++drain_passes;
                                            pause_cpu();
                                        }
                                        service->epoch_drain(wi,now_ns()-drain_start,drain_passes);
                                        // status holds the finished snapshot whose
                                        // cpu_reductions is final; that drained value
                                        // is the only point where the baseline moves.
                                        service->epoch_end(wi, status.cpu_reductions);
                                        service->epoch_verify(wi, native[wi]);
                                    }
                                }
                                active.fetch_sub(1);
                                epoch_active=false;
                                replay_spin.reset();

                                for(std::size_t j=0;j<replay_n;++j) {
                                    const float * replay=llama_get_logits_ith(ctx,int(j));
                                    require(replay!=nullptr,
                                        "MTP multi genuine replay missing original target logits");
                                    const auto greedy=tq::sampling::select_finite_greedy(replay,n_vocab);
                                    require(greedy.usable,
                                        "MTP multi replay has nonfinite target logits");
                                    auto & ref=verifier_rows[j];
                                    const auto v=tq::sampling::select_finite_greedy(ref.data(),n_vocab);
                                    require(v.usable,
                                        "MTP multi target original verifier row nonfinite");
                                    if(greedy.selected.id!=v.selected.id) {
                                        double e2=0,b2=0,maxabs=0;
                                        for(std::uint32_t t=0;t<n_vocab;++t) {
                                            const double a=ref[t],b=replay[t];
                                            require(std::isfinite(a)&&std::isfinite(b),
                                                "MTP multi target/replay nonfinite F32");
                                            e2+=(a-b)*(a-b);b2+=a*a;
                                            maxabs=std::max(maxabs,std::abs(a-b));
                                        }
                                        const double rel=std::sqrt(e2/std::max(b2,1e-30));
                                        std::vector<int> orderA(n_vocab),orderB(n_vocab);
                                        for(std::uint32_t t=0;t<n_vocab;++t){
                                            orderA[t]=int(t);orderB[t]=int(t);
                                        }
                                        std::partial_sort(orderA.begin(),orderA.begin()+10,orderA.end(),
                                            [&](int a,int b){return ref[a]>ref[b];});
                                        std::partial_sort(orderB.begin(),orderB.begin()+10,orderB.end(),
                                            [&](int a,int b){return replay[a]>replay[b];});
                                        int overlap=0;
                                        for(int a=0;a<10;++a)for(int b=0;b<10;++b)
                                            if(orderA[a]==orderB[b])++overlap;
                                        const double amax=ref[orderA[0]];
                                        const double bmax=replay[orderB[0]];
                                        double Za=0,Zb=0,mean_delta=0;
                                        for(std::uint32_t t=0;t<n_vocab;++t){
                                            const double w=std::exp(double(ref[t])-amax);
                                            Za+=w;
                                            Zb+=std::exp(double(replay[t])-bmax);
                                            mean_delta+=w*(double(ref[t])-double(replay[t]));
                                        }
                                        const double KL=mean_delta/Za +
                                            (bmax+std::log(Zb))-(amax+std::log(Za));
                                        if (mtp_state_trace) {
                                            std::fprintf(stderr,
                                                "TP5_REAL_MTP_MULTI_NUMERIC window=%u row=%zu original_top1=%d replay_top1=%d relL2=%.9g KL=%.9g top10=%d maxabs=%.9g\n",
                                                wi,j,int(v.selected.id),
                                                int(greedy.selected.id),rel,KL,overlap,maxabs);
                                            std::fflush(stderr);
                                        }
                                        require(std::isfinite(KL)&&KL>=-1e-6&&KL<=0.05&&
                                                rel<=0.05&&overlap>=9,
                                            "MTP multi verifier/replay F32 distribution materially differs");
                                    }
                                    if(j<accepted)
                                        require(greedy.selected.id==v.selected.id,
                                            "MTP multi accepted-prefix token changed during rollback replay");
                                }
                                if(replay_n==1)
                                    single_output_logits=llama_get_logits_ith(ctx,0);
                                if (mtp_state_trace) {
                                    std::fprintf(stderr,
                                        "TP5_REAL_MTP_MULTI_REJECT_ROLLBACK_PASS window=%u matched_prefix=%zu replay_N=%zu restored_bytes=%zu\n",
                                        wi,accepted,replay_n,restored);
                                    std::fflush(stderr);
                                }
                                }
                            } else {
                                target_state_checkpoint.clear();
                                if (mtp_state_trace) {
                                    std::fprintf(stderr,
                                        "TP5_REAL_MTP_MULTI_ACCEPT window=%u n_drafts=%zu committed_outputs=%zu\n",
                                        wi,n,n+1);
                                    std::fflush(stderr);
                                }
                            }
                        }
                        const auto commit_start = now_ns();
                        for (std::size_t row = 0; row < local_rows.size(); ++row) {
                            const auto local          = local_rows[row];
                            auto &     request        = result.requests[ids[local]];
                            const auto input_position = consumed[local]++;
                            if (prompt_catchup && input_position < request.prompt_tokens.size()) {
                                require(ids.size() == 1 && local == 0 &&
                                        input_position == draft_prompt_rows &&
                                        batch.batch.pos[row] == static_cast<llama_pos>(input_position) &&
                                        batch.batch.token[row] == request.prompt_tokens[input_position],
                                        "draft prompt rows must match the private prompt in order");
                                const auto prompt_start=now_ns();
                                constexpr int wide=10240;
                                const float * target_h=llama_get_embeddings_nextn_ith(ctx, int32_t(row));
                                require(target_h != nullptr, "prompt target hidden row is missing");
                                for (int j=0;j<wide;++j)
                                    require(std::isfinite(target_h[j]), "prompt target hidden is nonfinite");
                                if (!prompt_audit_prefix.empty()) {
                                    const auto save_row = [&](const char * kind, const float * data) {
                                        const auto path=prompt_audit_prefix+"-"+kind+"-w"+std::to_string(wi)+".f32le";
                                        FILE * file=std::fopen(path.c_str(), input_position == 0 ? "wbx" : "ab");
                                        require(file != nullptr, "could not open prompt hidden audit");
                                        const bool written=std::fwrite(data, sizeof(float), wide, file) == wide;
                                        const bool closed=std::fclose(file) == 0;
                                        require(written && closed, "prompt hidden audit write failed");
                                    };
                                    save_row("input", draft_prompt_previous_h.data());
                                    save_row("target", target_h);
                                }
                                auto * dctx=trained_mtp_contexts.at(wi).get();
                                llama_batch prompt_batch=llama_batch_init(1,wide,1);
                                prompt_batch.token=static_cast<llama_token *>(std::malloc(sizeof(llama_token)));
                                require(prompt_batch.token != nullptr, "prompt draft token allocation failed");
                                prompt_batch.n_tokens=1;
                                prompt_batch.token[0]=batch.batch.token[row];
                                prompt_batch.pos[0]=static_cast<llama_pos>(input_position);
                                prompt_batch.n_seq_id[0]=1;
                                prompt_batch.seq_id[0][0]=0;
                                prompt_batch.logits[0]=0;
                                std::memcpy(prompt_batch.embd,draft_prompt_previous_h.data(),wide*sizeof(float));
                                const int rc=llama_decode(dctx,prompt_batch);
                                std::free(prompt_batch.token);
                                prompt_batch.token=nullptr;
                                llama_batch_free(prompt_batch);
                                require(rc == 0, "draft prompt catch-up failed");
                                draft_pending = true;
                                std::memcpy(draft_prompt_previous_h.data(),target_h,wide*sizeof(float));
                                ++draft_prompt_rows;
                                if (draft_prompt_rows == request.prompt_tokens.size()) {
                                    llama_synchronize(dctx);
                                    draft_pending = false;
                                    const auto kv_min=llama_memory_seq_pos_min(llama_get_memory(dctx),0);
                                    const auto kv_max=llama_memory_seq_pos_max(llama_get_memory(dctx),0);
                                    require(kv_min == 0 && kv_max+1 == static_cast<llama_pos>(draft_prompt_rows),
                                            "draft prompt KV is incomplete before the first generated input");
                                    std::fprintf(stderr,
                                        "TP5_MTP_PROMPT_CATCHUP window=%u rows=%zu kv_min=%d kv_max=%d initial_hidden_zero=1 next_input_pos=%zu\n",
                                        wi,draft_prompt_rows,int(kv_min),int(kv_max),draft_prompt_rows);
                                }
                                draft_prompt_ns+=now_ns()-prompt_start;
                            }
                            if (!batch.batch.logits[row]) {
                                continue;
                            }
                            const float * logits =
                                output_rows == 1 && row == static_cast<std::size_t>(single_output_row) ?
                                    single_output_logits : llama_get_logits_ith(ctx, row);
                            require(logits != nullptr, "full logits row is missing");
                            const auto greedy = tq::sampling::select_finite_greedy(logits, n_vocab);
                            require(greedy.usable, "nonfinite full model logits");
                            const llama_token selected = greedy.selected.id;
                            if (mtp_draft_smoke && request.tokens.size() + 1 < opts.predict) {
                                require(row < static_cast<std::size_t>(batch.batch.n_tokens),
                                        "MTP output row must belong to actual target decode");
                                // This hidden row corresponds to the last real prompt
                                // token; the newly target-sampled id is the NEXT token
                                // to feed the trained MTP block (shift-by-one).
                                const float * real_h = llama_get_embeddings_nextn_ith(ctx, int32_t(row));
                                require(real_h != nullptr, "actual target did not export its wide NextN residual");
                                constexpr int wide = 10240;
                                double hnorm = 0;
                                for (int j=0;j<wide;++j) {
                                    require(std::isfinite(real_h[j]), "target MTP wide residual is nonfinite");
                                    hnorm += double(real_h[j])*real_h[j];
                                }
                                require(hnorm > 0, "actual target residual must be nontrivial");
                                auto * dctx = trained_mtp_contexts.at(wi).get();
                                const bool last_target_row=row+1==local_rows.size();
                                const bool draft_output = !last_draft_logits || last_target_row;
                                const auto first_start=now_ns();
                                llama_batch dft_batch = llama_batch_init(1, wide, 1);
                                dft_batch.token = static_cast<llama_token *>(std::malloc(sizeof(llama_token)));
                                require(dft_batch.token != nullptr, "actual MTP token input allocation failed");
                                dft_batch.n_tokens = 1;
                                dft_batch.token[0] = selected;
                                dft_batch.pos[0] = static_cast<llama_pos>(consumed[local]);
                                dft_batch.n_seq_id[0] = 1;
                                dft_batch.seq_id[0][0] = 0;
                                dft_batch.logits[0] = draft_output ? 1 : 0;
                                std::memcpy(dft_batch.embd, real_h, wide*sizeof(float));
                                const int dc = llama_decode(dctx, dft_batch);
                                std::free(dft_batch.token);
                                dft_batch.token = nullptr;
                                llama_batch_free(dft_batch);
                                require(dc == 0, "actual MTP draft decode from target residual failed");
                                draft_pending = true;
                                // These rows only update draft KV. Their outputs are unused.
                                if (!draft_output) {
                                    diag_draft_first+=now_ns()-first_start; ++diag_first_count;
                                    ++diag_state_only_count;
                                } else {
                                    const auto * dl = llama_get_logits_ith(dctx, 0);
                                    const auto * dh = llama_get_embeddings_nextn_ith(dctx, 0);
                                    require(dl && dh, "actual MTP draft missing logits/wide output");
                                    draft_pending = false;
                                    auto dg = tq::sampling::select_finite_greedy(dl,n_vocab);
                                    require(dg.usable, "actual trained MTP logits must be finite");
                                    double dsum=0;
                                    for (int j=0;j<wide;++j) {
                                        require(std::isfinite(dh[j]),"actual MTP output residual nonfinite");
                                        dsum+=double(dh[j])*dh[j];
                                    }
                                    require(dsum>0,"actual trained MTP output residual empty");
                                    // Each emitted real target token's ACTUAL
                                    // corresponding h_nextn is paired with that
                                    // selected token at shifted position. When
                                    // the target accepted two outputs in one
                                    // batch, process BOTH true output tokens
                                    // sequentially in the MTP context; the last
                                    // proposal supersedes the earlier proposal.
                                    diag_draft_first+=now_ns()-first_start; ++diag_first_count;
                                    if(last_target_row){
                                        pending_mtp_tokens.clear();
                                        const auto produced_after_this=request.tokens.size()+1;
                                        const std::size_t remaining=opts.predict-produced_after_this;
                                        const auto n_propose=std::min<std::size_t>(
                                            mtp_depth,remaining>1?remaining-1:0);
                                        if(n_propose>0){
                                            pending_mtp_tokens.push_back(dg.selected.id);
                                            std::vector<float> prev_draft_h(dh,dh+wide);
                                            auto next_token=dg.selected.id;
                                            for(std::size_t extra=1;extra<n_propose;++extra){
                                                const auto extra_start=now_ns();
                                                llama_batch extra_batch=llama_batch_init(1,wide,1);
                                                extra_batch.token=static_cast<llama_token *>(
                                                    std::malloc(sizeof(llama_token)));
                                                require(extra_batch.token!=nullptr,
                                                    "MTP multi extra draft token input malloc failed");
                                                extra_batch.n_tokens=1;
                                                extra_batch.token[0]=next_token;
                                                extra_batch.pos[0]=static_cast<llama_pos>(
                                                    consumed[local]+extra);
                                                extra_batch.n_seq_id[0]=1;
                                                extra_batch.seq_id[0][0]=0;
                                                extra_batch.logits[0]=1;
                                                std::memcpy(extra_batch.embd,prev_draft_h.data(),
                                                    wide*sizeof(float));
                                                const int extra_rc=llama_decode(dctx,extra_batch);
                                                std::free(extra_batch.token);
                                                extra_batch.token=nullptr;
                                                llama_batch_free(extra_batch);
                                                require(extra_rc==0,
                                                    "real trained multi-draft MTP decode failed");
                                                const float * more=llama_get_logits_ith(dctx,0);
                                                const float * h_more=llama_get_embeddings_nextn_ith(dctx,0);
                                                require(more&&h_more,
                                                    "multi-draft NextN missing true logits/hidden");
                                                const auto greedy_more=tq::sampling::select_finite_greedy(
                                                    more,n_vocab);
                                                require(greedy_more.usable,
                                                    "multi-draft NextN full logits nonfinite");
                                                for(int j=0;j<wide;++j)
                                                    require(std::isfinite(h_more[j]),
                                                        "multi-draft NextN hidden nonfinite");
                                                pending_mtp_tokens.push_back(greedy_more.selected.id);
                                                next_token=greedy_more.selected.id;
                                                std::memcpy(prev_draft_h.data(),h_more,
                                                    wide*sizeof(float));
                                                diag_draft_extra+=now_ns()-extra_start; ++diag_extra_count;
                                            }
                                            if (mtp_state_trace) {
                                                std::fprintf(stderr,
                                                    "TP5_REAL_MTP_MULTI_DRAFT window=%u depth=%zu source_pos=%llu draft0=%d draft_last=%d\n",
                                                    wi,pending_mtp_tokens.size(),
                                                    (unsigned long long)consumed[local],
                                                    int(pending_mtp_tokens.front()),
                                                    int(pending_mtp_tokens.back()));
                                                std::fflush(stderr);
                                            }
                                        }
                                    }
                                    if (mtp_state_trace) {
                                        std::fprintf(stderr,
                                            "TP5_REAL_TRAINED_MTP_FROM_TARGET window=%u seq=0 target_last_input=%d next_token=%d next_pos=%llu draft_proposal=%d target_h_l2sq=%.8g draft_h_l2sq=%.8g\n",
                                            wi, int(batch.batch.token[row]), int(selected),
                                            (unsigned long long)consumed[local],int(dg.selected.id),hnorm,dsum);
                                        std::fflush(stderr);
                                    }
                                }
                            }
                            token_record record;
                            record.input_token    = batch.batch.token[row];
                            record.sampled_token  = selected;
                            record.emitted_token  = opts.teacher_tokens.empty() ?
                                                        selected :
                                                        opts.teacher_tokens[ids[local]][request.tokens.size()];
                            record.input_position = input_position;
                            if (opts.capture_logits) {
                                record.logits.assign(logits, logits + n_vocab);
                            }
                            // There is exactly one owner/worker for this request. No
                            // other window can mutate its input, sampler or cursor.
                            record.committed_ns = now_ns();
                            request.tokens.push_back(std::move(record));
                            auto delta = streams[local].push(piece(vocab, request.tokens.back().emitted_token));
                            request.text += delta.text;
                            if (opts.cancel_request == static_cast<std::int32_t>(ids[local]) &&
                                request.tokens.size() >= opts.cancel_after) {
                                request.stop_reason = "cancelled";
                            } else if (!opts.ignore_eos &&
                                       llama_vocab_is_eog(vocab, request.tokens.back().emitted_token)) {
                                request.stop_reason = "eos";
                            } else if (request.tokens.size() == opts.predict) {
                                request.stop_reason = "budget";
                            }
                            if (!request.stop_reason.empty()) {
                                request.text += streams[local].finish().text;
                                request.finished_ns = now_ns();
                            }
                        }
                        metrics.last_finish_ns = now_ns();
                        metrics.result_commit_ns += metrics.last_finish_ns - commit_start;
                        if (!timeline_prefix.empty()) {
                            auto & rows = cycle_timings[wi];
                            require(rows.size() < rows.capacity(), "cycle timing capacity exhausted");
                            rows.push_back({metrics.decode_calls, (std::uint64_t)cycle_rows,
                                (std::uint64_t)cycle_position, cycle_queue_start, submit_start,
                                submit_finished, wait_finished, metrics.last_finish_ns});
                        }
                    }
                    std::fprintf(stderr,
                        "TP5_DIAG_PHASE window=%u elapsed_ns=%llu draft_first_ns=%llu draft_extra_ns=%llu state_trim_ns=%llu submission_queue_ns=%llu target_submit_ns=%llu native_wait_ns=%llu commit_including_draft_ns=%llu draft_first_calls=%llu draft_extra_calls=%llu state_trim_calls=%llu\n",
                        wi,(unsigned long long)(now_ns()-diag_begin),
                        (unsigned long long)diag_draft_first,(unsigned long long)diag_draft_extra,
                        (unsigned long long)diag_state,(unsigned long long)diag_queue,
                        (unsigned long long)metrics.decode_submit_ns,(unsigned long long)metrics.native_wait_ns,
                        (unsigned long long)metrics.result_commit_ns,(unsigned long long)diag_first_count,
                        (unsigned long long)diag_extra_count,(unsigned long long)diag_state_count);
                    if (gate_metrics_on) {
                        // Outer hold is an approximation: timestamps are taken
                        // inside the lock boundaries (start after acquisition,
                        // stop after unlock), so the interval excludes the hold
                        // before the start timestamp and may include unlocked
                        // gaps (ready-scope hooks release/re-take when
                        // paired_phases). meet_* nonzero only when paired_phases.
                        std::fprintf(stderr,
                            "TP5_GATE_METRICS window=%u decode_calls=%llu submission_holds=%llu submission_hold_ns=%llu submission_queue_ns=%llu replay_holds=%llu replay_hold_ns=%llu enqueue_holds=%llu enqueue_hold_ns=%llu enqueue_wait_calls=%llu enqueue_wait_ns=%llu meet_before_calls=%llu meet_before_ns=%llu meet_after_calls=%llu meet_after_ns=%llu ready_hits=%llu ready_wait_ns=%llu paired_phases=%u submit_only_gate=%u async_protocol=%u serialcut_gate=%u\n",
                            wi, (unsigned long long)metrics.decode_calls,
                            (unsigned long long)gm.holds, (unsigned long long)gm.hold_ns,
                            (unsigned long long)diag_queue,
                            (unsigned long long)gm.replay_holds, (unsigned long long)gm.replay_hold_ns,
                            (unsigned long long)gm.enqueues, (unsigned long long)gm.enqueue_ns,
                            (unsigned long long)gm.enqueue_waits, (unsigned long long)gm.enqueue_wait_ns,
                            (unsigned long long)gm.meet_before_calls, (unsigned long long)gm.meet_before_ns,
                            (unsigned long long)gm.meet_after_calls, (unsigned long long)gm.meet_after_ns,
                            (unsigned long long)pair_ready_hits, (unsigned long long)pair_ready_ns,
                            unsigned(paired_phases), unsigned(submit_only_gate), unsigned(async_protocol),
                            unsigned(serialcut_gate));
                    }
                    if (paired_phases) {
                        std::fprintf(stderr, "TP5_MTP_READY_PAIR_TOTAL window=%u meets=%llu before_ns=%llu after_ns=%llu\n",
                            wi, (unsigned long long)pair_meets, (unsigned long long)pair_before_ns,
                            (unsigned long long)pair_after_ns);
                    }
                    if (paired_phases) {
                        if (draft_pending) {
                            llama_synchronize(trained_mtp_contexts.at(wi).get());
                            draft_pending = false;
                            ++draft_phase_drains;
                        }
                        std::fprintf(stderr, "TP5_MTP_DRAFT_PHASE_DRAIN_TOTAL window=%u count=%llu\n",
                            wi, (unsigned long long) draft_phase_drains);
                        require(pair_ready_hits == metrics.decode_calls, "Meta submit-ready count mismatch");
                        require(!post_submit_unlock || pair_submitted_hits == metrics.decode_calls,
                                "Meta submitted count mismatch");
                        std::fprintf(stderr,"TP5_MTP_SUBMITTED_TOTAL window=%u enabled=%u calls=%llu\n",
                            wi,unsigned(post_submit_unlock),(unsigned long long)pair_submitted_hits);
                        std::fprintf(stderr,"TP5_MTP_READY_TOTAL window=%u calls=%llu wait_ns=%llu\n",
                            wi,(unsigned long long)pair_ready_hits,(unsigned long long)pair_ready_ns);
                    }
                    if (mtp_draft_smoke) {
                        if (prompt_catchup) {
                            require(ids.size() == 1 &&
                                    draft_prompt_rows == result.requests[ids[0]].prompt_tokens.size(),
                                    "request ended with missing draft prompt rows");
                            std::fprintf(stderr,
                                "TP5_MTP_PROMPT_TOTAL window=%u rows=%zu elapsed_ns=%llu\n",
                                wi,draft_prompt_rows,(unsigned long long)draft_prompt_ns);
                        }
                        std::fprintf(stderr,
                            "TP5_MTP_DRAFT_OUTPUT_TOTAL window=%u state_only_calls=%llu output_calls=%llu\n",
                            wi,(unsigned long long)diag_state_only_count,
                            (unsigned long long)(diag_first_count-diag_state_only_count));
                        std::fprintf(stderr,
                            "TP5_REAL_MTP_REPEATED_TOTAL window=%u verified_rounds=%llu accepted_rounds=%llu rejected_rollback_rounds=%llu emitted=%zu target_decode_calls=%llu\n",
                            wi,(unsigned long long)mtp_verify_rounds,
                            (unsigned long long)mtp_accept_rounds,
                            (unsigned long long)mtp_rollback_rounds,
                            result.requests[ids[0]].tokens.size(),
                            (unsigned long long)metrics.decode_calls);
                        std::fprintf(stderr,
                            "TP5_REAL_MTP_MULTI_TOTAL window=%u draft_depth=%u drafted_tokens=%llu accepted_draft_tokens=%llu accepted_fraction=%.6f\n",
                            wi,mtp_depth,(unsigned long long)mtp_proposal_tokens,
                            (unsigned long long)mtp_accepted_draft_tokens,
                            mtp_proposal_tokens ?
                                double(mtp_accepted_draft_tokens)/mtp_proposal_tokens : 0.0);
                        std::fflush(stderr);
                    }
                } catch (...) {
                    if (epoch_active) {
                        active.fetch_sub(1);
                        epoch_active = false;
                    }
                    failure.fail(std::current_exception());
                    if (service) {
                        service->abort_all();
                    }
                }
                phase_pair.retire();
                workers_done.fetch_add(1, std::memory_order_release);
            });
        } catch (...) {
            failure.fail(std::current_exception());
            phase_pair.cancel();
            if (service) {
                service->abort_all();
            }
            break;
        }
    }
    for (auto & worker : workers) {
        worker.join();
    }
    if (gate_metrics_req) {
        // DUALWIN-SERIALCUT acceptance signal: the runtime-wide ticket gate and
        // every provider-level submit mutex are always taken, so the native
        // submit section can never overlap (peak=1, events=0). The serialcut
        // gate only narrows the lock hold to the native five-rank enqueue; it
        // removes no lock. The canary is expected to stay zero.
        std::fprintf(stderr,
            "TP5_SERIALCUT_OBSERVATION windows=%u peak_concurrent_submits=%u concurrent_submit_events=%llu serialcut_gate=%d async_protocol=%d\n",
            unsigned(opts.windows), submit_observation.peak.load(std::memory_order_relaxed),
            (unsigned long long)submit_observation.concurrent_events.load(std::memory_order_relaxed),
            int(serialcut_gate), int(async_protocol));
    }
    if (overlap_meter_req) {
        // Host in-flight overlap + lock-hold receipt: the counter's own
        // TP5_DW_THRU_CTR / TP5_DW_THRU_ORDER lines are dumped verbatim;
        // device-side HIP event metering is dumped by the relay providers.
        std::fprintf(stderr, "TP5_DW_OVERLAP_METER enabled=1 flag=GGML_TP5_DW_OVERLAP_METER host_counter=1\n");
        dw_overlap_meter.dump();
    }
    // Keep the CPU service alive while native abort/tail work exits.
    if (service) {
        const auto deadline = clock_type::now() + std::chrono::seconds(45);
        for (std::uint32_t wi = 0; wi < opts.windows; ++wi) {
            ggml_relay_status status{};
            for (;;) {
                if (!ggml_backend_relay_snapshot(native[wi], &status)) {
                    if (clock_type::now() >= deadline) {
                        retain_and_exit("native status remained unavailable");
                    }
                    pause_cpu();
                    continue;
                }
                if (status.finished && (status.failed || service->retired(wi, status))) {
                    break;
                }
                if (clock_type::now() >= deadline) {
                    retain_and_exit("native completion or CPU reader retirement is unproved");
                }
                pause_cpu();
            }
            if (overlap_meter_req) {
                // Device-evidence attribution: joins the relay provider's
                // TP5_DW_HIP_OVERLAP lines (same physical device per rank,
                // PCI BDF as the join key) to this window.
                std::fprintf(stderr, "TP5_DW_HIP_OVERLAP_MAP window=%u native=%p",
                    wi, (void *) native[wi]);
                for (std::uint32_t r = 0; r < status.ranks && r < GGML_RELAY_RANKS; ++r) {
                    std::fprintf(stderr, " pci_bdf%u=%s", r, status.pci_bdf[r]);
                }
                std::fprintf(stderr, "\n");
            }
            auto & metrics              = result.windows[wi];
            if (status.failed || failure.stopped.load()) {
                std::fprintf(stderr,
                             "TP5_NATIVE_DRAIN window=%u epoch=%llu attempted=%u accepted=%u unknown=%u completed=%u sealed=%u finished=%u error=%d detail=%.*s\n",
                             wi, static_cast<unsigned long long>(status.epoch), status.attempted_mask,
                             status.accepted_mask, status.unknown_mask, status.completed_mask,
                             unsigned(status.sealed), unsigned(status.finished), status.error_code,
                             int(sizeof(status.error)), status.error);
            }
            metrics.epochs              = status.epoch;
            metrics.channels            = status.channels;
            metrics.cpu_reductions      = status.cpu_reductions;
            metrics.input_retirements   = service->inputs(wi);
            metrics.output_retirements  = service->outputs(wi);
            metrics.hip_runtime_version = status.hip_runtime_version;
            metrics.native_drained      = status.finished;
            for (std::size_t rank = 0; rank < 5; ++rank) {
                metrics.rank_graphs[rank]          = status.rank_graphs[rank];
                metrics.rank_compute_nodes[rank]   = status.rank_compute_nodes[rank];
                metrics.rank_relay_nodes[rank]     = status.rank_relay_nodes[rank];
                metrics.rank_published_bytes[rank] = status.rank_published_bytes[rank];
                metrics.rank_consumed_bytes[rank]  = status.rank_consumed_bytes[rank];
                metrics.pci_bdf[rank]              = status.pci_bdf[rank];
            }
            if (courier_hits_req) service->dump_courier_metrics(wi);
        }
        // finish_ledgers() starts with stop(), so by the time it returns the
        // workers and every courier lane are joined even on the failure path.
        try {
            service->finish_ledgers(failure.stopped.load());
            if (!timeline_prefix.empty()) service->write_timings(timeline_prefix);
            for (std::size_t wi = 0; wi < opts.windows; ++wi) {
                service->copy_wire_metrics(wi, result.windows[wi]);
            }
        } catch (...) {
            failure.fail(std::current_exception());
        }
        // Dumps sit after the ledger try/catch (finish_ledgers' stop() has run,
        // so workers and courier lanes are joined in every path, including the
        // failure path). Reads cannot race a courier write, and each window
        // prints exactly once.
        if (service->metrics_enabled()) service->dump_courier_metrics();
        service->dump_dual_completions_summary();
        for (std::uint32_t wi = 0; wi < opts.windows; ++wi) {
            if (courier_hits_req) service->dump_courier_metrics(wi);
        }
    }
    watchdog_stop.store(true, std::memory_order_release);
    watchdog.join();
    for (std::uint32_t wi = 0; wi < opts.windows; ++wi) {
        // Synchronize the first-child event on the host before the existing
        // llama_synchronize; their stream and coverage relationship is unverified.
        if (dw_backend_event_req && wi < dw_backend_events.size() &&
            dw_backend_events[wi].event != nullptr) {
            ++dw_backend_events[wi].sync_calls;
            const std::uint64_t dw_ev_t0 = dw_event_trace::gate() ? now_ns() : 0;
            ggml_backend_event_synchronize(dw_backend_events[wi].event);
            if (dw_ev_t0) {
                auto & dw_ev = dw_backend_events[wi].ev;
                const std::uint64_t dw_ev_done = now_ns();
                if (dw_ev.sync_first_ns == 0) dw_ev.sync_first_ns = dw_ev_t0;
                dw_ev.sync_last_ns = dw_ev_t0;
                dw_ev.sync_ns += dw_ev_done - dw_ev_t0;
                if (dw_ev_done - dw_ev_t0 > dw_ev.sync_max_ns) dw_ev.sync_max_ns = dw_ev_done - dw_ev_t0;
            }
        }
        llama_synchronize(contexts[wi].get());
        for (std::size_t local = 0; local < assignments[wi].size(); ++local) {
            require(llama_memory_seq_rm(llama_get_memory(contexts[wi].get()), local, -1, -1),
                    "request state removal failed");
        }
        llama_synchronize(contexts[wi].get());
    }
    if (dw_backend_event_req) {
        for (std::uint32_t wi = 0; wi < dw_backend_events.size(); ++wi) {
            if (dw_backend_events[wi].event != nullptr) {
                ggml_backend_event_free(dw_backend_events[wi].event);
                dw_backend_events[wi].event = nullptr;
                ++dw_backend_events[wi].free_calls;
            }
        }
    }
    if (dw_backend_event_req) {
        // Receipts count calls after drain and free. The first child's stream
        // relationship to relay work is unverified; these are not overlap or
        // device completion evidence. record_ok counts returned record calls.
        std::fprintf(stderr, "TP5_DW_BACKEND_EVENT enabled=1 flag=GGML_TP5_DW_BACKEND_EVENT windows=%u\n",
            unsigned(opts.windows));
        for (std::uint32_t wi = 0; wi < dw_backend_events.size(); ++wi) {
            const auto & dw_be = dw_backend_events[wi];
            std::fprintf(stderr,
                "TP5_DW_BACKEND_EVENT window=%u created=%d record_calls=%llu record_ok=%llu sync_calls=%llu free_calls=%llu skipped_unavailable=%llu\n",
                unsigned(wi), int(dw_be.event != nullptr || dw_be.free_calls != 0),
                (unsigned long long)dw_be.record_calls, (unsigned long long)dw_be.record_ok,
                (unsigned long long)dw_be.sync_calls,
                (unsigned long long)dw_be.free_calls, (unsigned long long)dw_be.skipped_unavailable);
        }
        // DUALWIN event-point trace: per-window stamps around the real
        // record/synchronize call sites. Host correlation metadata only;
        // wait_calls counts the real device-side waits issued under
        // GGML_TP5_DW_DEVICE_WAIT=1 (zero when that gate is off).
        if (dw_event_trace::gate()) {
            dw_event_trace::banner();
            for (std::uint32_t wi = 0; wi < dw_backend_events.size(); ++wi) {
                const auto & dw_be = dw_backend_events[wi];
                std::fprintf(stderr,
                    "TP5_DW_EVENT_TRACE_EVENTS window=%u record_calls=%llu record_ok=%llu record_first_ns=%llu record_last_ns=%llu sync_calls=%llu sync_first_ns=%llu sync_last_ns=%llu sync_ns=%llu sync_max_ns=%llu wait_calls=%llu\n",
                    unsigned(wi), (unsigned long long)dw_be.record_calls, (unsigned long long)dw_be.record_ok,
                    (unsigned long long)dw_be.ev.record_first_ns, (unsigned long long)dw_be.ev.record_last_ns,
                    (unsigned long long)dw_be.sync_calls, (unsigned long long)dw_be.ev.sync_first_ns,
                    (unsigned long long)dw_be.ev.sync_last_ns, (unsigned long long)dw_be.ev.sync_ns,
                    (unsigned long long)dw_be.ev.sync_max_ns, (unsigned long long)dw_be.ev.wait_calls);
            }
        }
    }
    if (dw_device_wait_req) {
        // DUALWIN-DEVICE-WAIT receipt (GGML_TP5_DW_DEVICE_WAIT=1): per-window
        // device-side event wait counters and host stamps. wait_ok counts
        // ggml_backend_event_wait calls actually issued on this window's
        // first-child stream for the peer window's event (same first-child
        // device, unconsumed record only); every skip is counted, never
        // silent. The first child's stream relationship to relay work is
        // unverified; these are not overlap or device completion evidence.
        // With GGML_TP5_DW_BACKEND_EVENT unset no event exists and every
        // counter stays zero except the skips.
        std::fprintf(stderr, "TP5_DW_DEVICE_WAIT enabled=1 flag=GGML_TP5_DW_DEVICE_WAIT windows=%u\n",
            unsigned(opts.windows));
        for (std::uint32_t wi = 0; wi < dw_backend_events.size(); ++wi) {
            const auto & dw_be = dw_backend_events[wi];
            std::fprintf(stderr,
                "TP5_DW_DEVICE_WAIT window=%u wait_calls=%llu wait_ok=%llu wait_skip_unrecorded=%llu wait_skip_unavailable=%llu wait_first_ns=%llu wait_last_ns=%llu\n",
                unsigned(wi), (unsigned long long)dw_be.wait_calls, (unsigned long long)dw_be.wait_ok,
                (unsigned long long)dw_be.wait_skipped_unrecorded,
                (unsigned long long)dw_be.wait_skipped_unavailable,
                (unsigned long long)dw_be.ev.wait_first_ns, (unsigned long long)dw_be.ev.wait_last_ns);
        }
    }
    contexts.clear();
    for (const auto & held : reservations) {
        require(budget.release(held), "state budget release failed");
    }
    result.reserved_tokens_after_close = budget.reserved_tokens();
    result.active_requests_after_close = budget.active_count();
    result.max_simultaneous_windows     = maximum.load();
    result.finished_ns                 = now_ns();
    if (!timeline_prefix.empty()) {
        for (std::size_t i = 0; i < cycle_timings.size(); ++i) {
            const auto path = timeline_prefix + ".cycles-" + std::to_string(i) + ".csv";
            auto * out = std::fopen(path.c_str(), "wx");
            require(out != nullptr, "cannot create cycle timing output");
            std::fprintf(out, "call,rows,position,queue_ns,submit_ns,submitted_ns,waited_ns,committed_ns\n");
            for (const auto & r : cycle_timings[i]) {
                std::fprintf(out, "%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu\n",
                    (unsigned long long)r.call, (unsigned long long)r.rows,
                    (unsigned long long)r.position, (unsigned long long)r.queue_ns,
                    (unsigned long long)r.submit_ns, (unsigned long long)r.submitted_ns,
                    (unsigned long long)r.waited_ns, (unsigned long long)r.committed_ns);
            }
            const bool written = !std::ferror(out);
            const int closed = std::fclose(out);
            require(written && closed == 0, "cycle timing output failed");
        }
    }
    if (failure.first) {
        std::fprintf(stderr, "TP5_FAILURE_CLEANUP reserved_tokens=%llu active_requests=%u\n",
                     static_cast<unsigned long long>(result.reserved_tokens_after_close),
                     result.active_requests_after_close);
        std::rethrow_exception(failure.first);
    }
    for (const auto & request : result.requests) {
        require(!request.stop_reason.empty(), "request did not complete");
    }
    return result;
}
}  // namespace tq::tp5
