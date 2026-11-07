#include <ggml-backend.h>
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
    mtp_submit_only_gate(bool enabled, std::unique_lock<submission_mutex_type> & submission,
                         std::uint64_t & queue_ns)
        : enabled_(enabled), submission_(submission), queue_ns_(queue_ns) {
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
        self.submission_.lock();
        self.queue_ns_ += now_ns() - begin;
        return true;
    }
    static bool submitted(void * opaque) {
        auto & self = *static_cast<mtp_submit_only_gate *>(opaque);
        if (self.arrived_ != 1 || ++self.submitted_ != 1 || !self.submission_.owns_lock()) return false;
        self.submission_.unlock();
        return true;
    }
    bool enabled_;
    std::unique_lock<submission_mutex_type> & submission_;
    std::uint64_t & queue_ns_;
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
    std::atomic<std::uint64_t>                                                        output_retired{ 0 };
    bool                                                                              delayed = false;
    std::optional<ggml_relay_event_v2>                                                   delayed_event;
    clock_type::time_point                                                            delay_until{};
    // Only the service thread writes these; copy them after stop()/join().
    std::array<std::uint64_t, 5> rank_input_packed_payload_bytes{};
    std::array<std::uint64_t, 5> rank_output_consumed_payload_bytes{};
    std::uint64_t cpu_input_payload_bytes = 0;
    std::uint64_t cpu_output_payload_bytes = 0;
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
        const char * per_env = std::getenv("TP5_RELAY_SERVICE_PER_WINDOW");
        require(!per_env || std::string(per_env) == "0" || std::string(per_env) == "1",
                "TP5_RELAY_SERVICE_PER_WINDOW must be 0 or 1");
        const bool per_window = !per_env || std::string(per_env) == "1";
        const std::size_t lanes = per_window ? windows_.size() : 1;
        for (std::size_t lane = 0; lane < lanes; ++lane) {
            threads_.emplace_back([this, lane, lanes] { loop(lane, lanes); });
        }
        while (ready_.load(std::memory_order_acquire) != lanes) {
            pause_cpu();
        }
        std::fprintf(stderr, "TP5_RELAY_SERVICE threads=%zu windows=%zu\n", lanes, windows_.size());
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

  private:
    static void add_payload(std::uint64_t & total, std::uint64_t bytes) {
        require(bytes <= UINT64_MAX - total, "wire payload byte counter overflow");
        total += bytes;
    }

    void progress(std::size_t index) {
        auto &           window = *windows_[index];
        ggml_relay_event_v2 event{};
        int              polled = 0;
        if (window.delayed_event) {
            if (clock_type::now() < window.delay_until && !failure_.stopped.load()) {
                return;
            }
            event = *window.delayed_event;
            window.delayed_event.reset();
            polled = 1;
        } else {
            polled = ggml_backend_relay_poll_v2(window.backend, &event, sizeof(event));
        }
        if (polled < 0) {
            if (!failure_.stopped.load()) {
                throw std::runtime_error("native relay progress failed");
            }
            return;
        }
        if (!polled) {
            return;
        }
        if (event.kind == GGML_RELAY_INPUT_READY && !window.delayed && options_.delay_publish_ms &&
            index == options_.delay_window) {
            // Retain this explicit provider borrow while servicing other windows.
            window.delayed       = true;
            window.delayed_event = event;
            window.delay_until   = clock_type::now() + std::chrono::milliseconds(options_.delay_publish_ms);
            return;
        }
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
                try {
                    progress(i);
                } catch (...) {
                    failure_.fail(std::current_exception());
                    abort_all();
                }
            }
            pause_cpu();
        }
    }

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
    const bool submit_only_gate = async_protocol && (!gate_env || std::string(gate_env) == "1");
    std::fprintf(stderr, "TP5_MTP_ASYNC_PROTOCOL enabled=%d ready_pair_requested=%d ready_pair_active=%d submit_only_gate=%d lock=fifo_spin\n",
        int(async_protocol), int(ready_pair_requested), int(paired_phases), int(submit_only_gate));
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
                        }
                        const auto cycle_queue_start = timeline_prefix.empty() ? 0 : now_ns();
                        const auto cycle_rows = batch.batch.n_tokens;
                        const auto cycle_position = batch.batch.pos[0];
                        // Keep each window's five-rank submission together.
                        // Completion waits and CPU relay work stay outside this gate.
                        std::unique_lock<submission_mutex_type> submission_lock(submission_mutex, std::defer_lock);
                        if (service && !submit_only_gate) {
                            const auto queue_start=now_ns();
                            submission_lock.lock();
                            diag_queue+=now_ns()-queue_start;
                        }
                        const auto current = active.fetch_add(1) + 1;
                        epoch_active = true;
                        auto       seen    = maximum.load();
                        while (seen < current && !maximum.compare_exchange_weak(seen, current)) {
                        }
                        if (!metrics.first_submit_ns) {
                            metrics.first_submit_ns = now_ns();
                        }
                        const auto submit_start = now_ns();
                        int decoded;
                        {
                            mtp_submit_ready_scope ready(paired_phases, post_submit_unlock, phase_pair, submission_lock,
                                failure.stopped, wi, metrics.decode_calls + 1, pair_ready_ns, pair_ready_hits,
                                pair_submitted_hits);
                            mtp_submit_only_gate gate(service && submit_only_gate, submission_lock, diag_queue);
                            decoded = llama_decode(ctx, batch.batch);
                            ready.check(decoded);
                            gate.check(decoded);
                        }
                        const auto submit_finished = now_ns();
                        metrics.decode_submit_ns += submit_finished - submit_start;
                        if (submission_lock.owns_lock()) {
                            submission_lock.unlock();
                        }
                        ++metrics.decode_calls;
                        metrics.input_rows += batch.batch.n_tokens;
                        require(decoded == 0, "real model decode failed");
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
                        }
                        const auto wait_finished = now_ns();
                        verify_spin.reset();
                        metrics.native_wait_ns += wait_finished - wait_start;
                        active.fetch_sub(1);
                        epoch_active = false;
                        require(!failure.stopped.load(), "request stopped before state commit");
                        if (paired_phases) {
                            const auto begin = now_ns();
                            require(phase_pair.meet(failure.stopped), "MTP phase pair stopped after target");
                            pair_after_ns += now_ns() - begin;
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
                                const auto active_count=active.fetch_add(1)+1;
                                epoch_active=true;
                                auto seen=maximum.load();
                                while(seen<active_count &&
                                      !maximum.compare_exchange_weak(seen,active_count)){}
                                const auto begin_replay=now_ns();
                                int ret;
                                {
                                    mtp_submit_only_gate replay_gate(service && submit_only_gate, replay_lock, diag_queue);
                                    ret=llama_decode(ctx,batch.batch);
                                    replay_gate.check(ret);
                                }
                                metrics.decode_submit_ns+=now_ns()-begin_replay;
                                if(replay_lock.owns_lock())replay_lock.unlock();
                                ++metrics.decode_calls;
                                metrics.input_rows+=batch.batch.n_tokens;
                                require(ret==0,"MTP multi verified-target-prefix replay failed");
                                std::optional<scoped_spin_cpu> replay_spin;
                                replay_spin.emplace();
                                llama_synchronize(ctx);
                                if(service) {
                                    ggml_relay_status status{};
                                    for(;;) {
                                        if(!ggml_backend_relay_snapshot(native[wi],&status)){
                                            pause_cpu();continue;
                                        }
                                        require(!status.failed&&!failure.stopped.load(),
                                            "native relay failed MTP multi prefix replay");
                                        if(status.finished&&service->retired(wi,status))break;
                                        pause_cpu();
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
    // Keep the independent CPU service alive while native abort/tail work exits.
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
        }
        try {
            service->finish_ledgers(failure.stopped.load());
            if (!timeline_prefix.empty()) service->write_timings(timeline_prefix);
            for (std::size_t wi = 0; wi < opts.windows; ++wi) {
                service->copy_wire_metrics(wi, result.windows[wi]);
            }
        } catch (...) {
            failure.fail(std::current_exception());
        }
    }
    watchdog_stop.store(true, std::memory_order_release);
    watchdog.join();
    for (std::uint32_t wi = 0; wi < opts.windows; ++wi) {
        llama_synchronize(contexts[wi].get());
        for (std::size_t local = 0; local < assignments[wi].size(); ++local) {
            require(llama_memory_seq_rm(llama_get_memory(contexts[wi].get()), local, -1, -1),
                    "request state removal failed");
        }
        llama_synchronize(contexts[wi].get());
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
