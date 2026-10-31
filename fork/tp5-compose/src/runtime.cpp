#include <ggml-backend.h>
#include <ggml-relay.h>
#include <gguf.h>
#include <llama.h>
#include <tq/relay/ledger.h>
#include <tq/sampling/shortlist.h>
#include <tq/scheduling/admission.h>
#include <tq/text/stream.h>
#include <tq/tp5/runtime.h>

#include "ordered-fold.h"
#include "dw_event_trace.h"
#include "dw_compose_event.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

void pause_cpu() noexcept {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#else
    std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

// DUALWIN-COMPOSE-EVENT: gate semantics, ON-path contract and honesty
// notes live in dw_compose_event.h (exact "1", read once at static
// initialization; OFF keeps every site an untouched stock branch).
static const bool dw_compose_event_req = dw_compose_event::gate();

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
    std::uint64_t courier_polls = 0;
    std::uint64_t courier_idle = 0;
    std::uint64_t courier_input = 0;
    std::uint64_t courier_output = 0;
    std::uint64_t courier_deferred = 0;
    std::uint64_t courier_errors = 0;
    // Per-epoch demand samples and the retirement race they observe. Written
    // only by this window's worker (epoch_demand/epoch_drain), read after join.
    // demand_* values are PER-EPOCH DELTAS against demand_base_*; the worker
    // only leaves the wait loop once retired == cpu_reductions, so the previous
    // sample's baseline is a fully drained epoch.
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
    // Batching evidence for the gated lane.
    std::uint64_t courier_batch_extra = 0;
    // GGML_TP5_DW_COURIER_DUAL=1: independent dual completion verification at
    // the logits consumption point. Written only by this window's worker
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
            windows_.push_back(std::move(item));
        }
        const char * dw_env = std::getenv("GGML_TP5_DW_COURIER");
        courier_metrics_ = dw_env != nullptr && std::strcmp(dw_env, "1") == 0;
        const char * ab_env = std::getenv("TP5_DW_COURIER_ABLATE");
        courier_ablate_ = courier_metrics_ && ab_env != nullptr && std::strcmp(ab_env, "1") == 0;
        // GGML_TP5_DW_COURIER_DUAL: the exact value "1" gates the independent
        // dual completion verification at the logits consumption point;
        // unset, "0", "2" or any other spelling keeps the shipped single
        // conjunction wait untouched.
        const char * dual_env = std::getenv("GGML_TP5_DW_COURIER_DUAL");
        courier_dual_ = dual_env != nullptr && std::strcmp(dual_env, "1") == 0;
        const char * pw_env = std::getenv("TP5_RELAY_SERVICE_PER_WINDOW");
        per_window_ = courier_metrics_ || (pw_env != nullptr && std::string(pw_env) == "1");
        if (per_window_) {
            for (std::size_t lane = 0; lane < windows_.size(); ++lane) {
                try {
                    threads_.emplace_back([this, lane] { loop_per_window(lane); });
                } catch (...) {
                    stop();
                    throw;
                }
            }
            while (lanes_ready_.load(std::memory_order_acquire) != windows_.size()) {
                pause_cpu();
            }
        } else {
            thread_ = std::thread([this] { loop(); });
            while (!ready_.load(std::memory_order_acquire)) {
                pause_cpu();
            }
        }
    }

    ~relay_service() { stop(); }

    void abort_all() noexcept {
        for (const auto & item : windows_) {
            ggml_backend_relay_abort(item->backend, 1);
        }
    }

    void stop() noexcept {
        stop_.store(true, std::memory_order_release);
        if (thread_.joinable()) {
            thread_.join();
        }
        for (auto & t : threads_) {
            if (t.joinable()) {
                t.join();
            }
        }
        if (courier_metrics_ && !metrics_dumped_) {
            metrics_dumped_ = true;
            dump_courier_metrics();
        }
    }

    // Per-window demand/retirement distribution. Emitted once, after every
    // courier lane is joined, so no counter can still be written.
    void dump_courier_metrics() const {
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

    void epoch_drain(std::size_t index, std::uint64_t drain_ns, std::uint64_t drain_passes) {
        if (!courier_metrics_) return;
        auto & w = *windows_[index];
        w.demand_drain_ns += drain_ns;
        w.demand_drain_passes += drain_passes;
        w.demand_drain_max = std::max(w.demand_drain_max, drain_passes);
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

    // Worker-side re-check after the gated wait loop. The gated loop stops
    // snapshotting once finished, so a provider failure recorded after that
    // point is only visible through a fresh snapshot. snapshot() uses
    // try_to_lock and returns false on ordinary contention, so retry the same
    // way the wait loops do and fail closed only on a real failure.
    void epoch_verify(std::size_t index, ggml_backend_t backend) const {
        if (!courier_metrics_) return;
        ggml_relay_status status{};
        while (!ggml_backend_relay_snapshot(backend, &status)) {
            require(!failure_.stopped.load(), "native relay stopped before the post-drain check");
            pause_cpu();
        }
        require(!status.failed && !failure_.stopped.load(), "native relay epoch failed after drain");
        require(status.finished && retired(index, status), "native relay epoch was not drained after finish");
    }

    bool metrics_enabled() const noexcept { return courier_metrics_; }

    bool ablate() const noexcept { return courier_ablate_; }

    bool retired(std::size_t index, const ggml_relay_status & state) const {
        return windows_[index]->output_retired.load(std::memory_order_acquire) == state.cpu_reductions;
    }

    std::uint64_t inputs(std::size_t index) const { return windows_[index]->input_retired.load(); }

    std::uint64_t outputs(std::size_t index) const { return windows_[index]->output_retired.load(); }
    // GGML_TP5_DW_COURIER_DUAL=1 (exact value; unset/"0"/"2"/"1x" stay OFF):
    // independent dual completion verification at the logits consumption
    // point. The two completions are INDEPENDENT signals already exposed by
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
    // parked. Dumped once from finish_ledgers() next to the courier metrics;
    // no-op when the dual gate is off, so OFF runs print nothing new.
    void dump_dual_completions_summary() const {
        if (!courier_dual_) return;
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


    void finish_ledgers(bool failed) {
        stop();
        // Dual completion summary: printed here, at the single real
        // end-of-run point, rather than inside stop() — a constructor spawn
        // failure also calls stop() and must not print an all-zero summary.
        if (courier_dual_ && !dual_dumped_) {
            dual_dumped_ = true;
            dump_dual_completions_summary();
        }
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
    void copy_wire_metrics(std::size_t index, window_result & result) const {
        require(!thread_.joinable(), "wire metrics require the stopped CPU service");
        for (const auto & t : threads_) {
            require(!t.joinable(), "wire metrics require the stopped CPU service");
        }
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

    // Returns true iff exactly one provider event was fully serviced. Empty,
    // deferred and error-return passes return false so a batch stops there.
    bool progress(std::size_t index) {
        auto &           window = *windows_[index];
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
            if (courier_metrics_) ++window.courier_idle;
            return false;
        }
        if (event.kind == GGML_RELAY_INPUT_READY && !window.delayed && options_.delay_publish_ms &&
            index == options_.delay_window) {
            // Retain this explicit provider borrow while servicing other windows.
            window.delayed       = true;
            window.delayed_event = event;
            window.delay_until   = clock_type::now() + std::chrono::milliseconds(options_.delay_publish_ms);
            if (courier_metrics_) ++window.courier_deferred;
            return false;
        }
        const bool borrowed = event.kind == GGML_RELAY_INPUT_READY;
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

    void loop() noexcept {
        ready_.store(true, std::memory_order_release);
        while (!stop_.load(std::memory_order_acquire)) {
            for (std::size_t i = 0; i < windows_.size(); ++i) {
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

    // Gated lane: service up to kCourierPassEvents events per wake so a backlog
    // of k costs one pass instead of k. Ablation keeps F1 off (one event per
    // pass) while retaining every counter; F2 is switched off on the worker
    // side (keep_snapshot), so both arms differ in F1+F2 only.
    void loop_per_window(std::size_t lane) noexcept {
        lanes_ready_.fetch_add(1, std::memory_order_acq_rel);
        while (!stop_.load(std::memory_order_acquire)) {
            try {
                if (!courier_metrics_ || courier_ablate_) {
                    (void) progress(lane);
                } else {
                    for (std::size_t n = 0; n < kCourierPassEvents; ++n) {
                        if (!progress(lane)) break;
                        if (n) ++windows_[lane]->courier_batch_extra;
                    }
                }
            } catch (...) {
                failure_.fail(std::current_exception());
                abort_all();
            }
            pause_cpu();
        }
    }

    static constexpr std::size_t                 kCourierPassEvents = 32;
    const options &                              options_;
    bool                                         courier_metrics_ = false;
    bool                                         courier_ablate_ = false;
    bool                                         courier_dual_ = false;
    bool                                         metrics_dumped_ = false;
    failure_state &                              failure_;
    bool                                         dual_dumped_ = false;
    std::uint64_t                                run_;
    std::vector<std::unique_ptr<service_window>> windows_;
    std::atomic<bool>                            ready_{ false }, stop_{ false };
    std::thread                                  thread_;
    std::vector<std::thread>                     threads_;
    std::atomic<std::size_t>                     lanes_ready_{ 0 };
    bool                                         per_window_ = false;
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
    const options                   opts = validate(requested);
    configure_hc_owner_layout(opts);
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
    // DUALWIN-COMPOSE-EVENT per-window state: counters only, touched under
    // the gate by this window's own decode worker and read after the workers
    // join. There is no per-window event, no peer publication array and no
    // peer wait: each window owns a private Meta relay provider, its own
    // context, KV and requests, and the model weights are read-only, so no
    // writable state is shared between windows. A symmetric per-epoch peer
    // event wait would therefore be a same-phase rendezvous with no data
    // dependency (and, through the Meta forwarder, only a rank-0 wait, not a
    // five-rank barrier). The epoch's only real cross-boundary dependency is
    // this window's own logits D2H copy, covered by the output-copy events
    // recorded below and host-completed after the relay drain.
    struct dw_compose_event_state {
        // Host-synchronization replacement counters (GGML_TP5_DW_COMPOSE_EVENT
        // only). output_event_syncs counts epochs whose host completion barrier
        // was the output-copy events recorded after llama_decode instead of the
        // scheduler-wide llama_synchronize; host_sync_replaced is the same
        // count under its contract name (one stock host synchronize skipped per
        // count); fallback_no_output_barrier counts ON-path epochs that had to
        // keep the stock synchronize because no sound output-copy barrier was
        // available. None of these is device-completion evidence.
        std::uint64_t output_event_syncs = 0, host_sync_replaced = 0,
                      fallback_no_output_barrier = 0, no_sync_logits_calls = 0;
    };
    std::vector<dw_compose_event_state> dw_compose_events;
    if (dw_compose_event_req && opts.execution == mode::relay) {
        dw_compose_events.resize(opts.windows);
    }
    for (std::uint32_t i = 0; i < opts.windows; ++i) {
        auto params      = llama_context_default_params();
        params.n_ctx     = opts.context_per_request * assignments[i].size();
        params.n_seq_max = assignments[i].size();
        params.n_batch   = assignments[i].size();
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
        params.quantized_matmul_f32   = opts.execution != mode::relay;
        context_ptr context(llama_init_from_model(model.get(), params));
        require(context != nullptr, "context preparation failed");
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
            // DUALWIN-COMPOSE-EVENT: nothing per-window is created here. The
            // output-copy events are owned by the Meta backend and allocated
            // lazily by ggml_backend_meta_output_record on the exact child
            // streams the logits D2H readback uses; see
            // llama_tp5_output_copy_record.
        }
        contexts.push_back(std::move(context));
        result.windows[i].id = i;
    }
    failure_state                  failure;
    std::unique_ptr<relay_service> service;
    if (opts.execution == mode::relay) {
        service = std::make_unique<relay_service>(opts, native, failure, now_ns() | 1);
    }
    result.started_ns = now_ns();
    // Count submitted epochs until native work and CPU readers retire.
    std::atomic<std::uint64_t> active{ 0 }, maximum{ 0 };
    std::mutex                 submission_mutex;
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
                bool epoch_active = false;
                try {
                    auto                                 ctx     = contexts[wi].get();
                    auto &                               metrics = result.windows[wi];
                    const auto &                         ids     = assignments[wi];
                    std::vector<std::uint64_t>           consumed(ids.size(), 0);
                    std::vector<text::generation_stream> streams(ids.size());
                    row_batch                            batch(ids.size());
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
                        for (std::size_t local = 0; local < ids.size(); ++local) {
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
                        // Keep each window's five-rank submission together.
                        // Completion waits and CPU relay work stay outside this gate.
                        std::unique_lock<std::mutex> submission_lock(submission_mutex, std::defer_lock);
                        if (service) {
                            submission_lock.lock();
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
                        const int decoded = llama_decode(ctx, batch.batch);
                        metrics.decode_submit_ns += now_ns() - submit_start;
                        if (submission_lock.owns_lock()) {
                            submission_lock.unlock();
                        }
                        ++metrics.decode_calls;
                        metrics.input_rows += batch.batch.n_tokens;
                        require(decoded == 0, "real model decode failed");

                        // DUALWIN-COMPOSE-EVENT: record the OUTPUT-COPY
                        // completion events immediately after llama_decode
                        // returned. decode has already enqueued the asynchronous
                        // logits D2H copy on the child stream(s) its Meta
                        // readback uses; a record issued now on exactly those
                        // streams is ordered behind the copy, so the paired wait
                        // later proves the host logits row is readable. ggml
                        // reports how many streams were covered (one for a
                        // mirrored output, every contributing rank for a split
                        // output); 0 means no sound barrier exists and the stock
                        // synchronize below stays. These events are the ONLY
                        // completion proof for the logits read — the relay's
                        // finished/retired counters do not cover the scheduler's
                        // D2H copy and are never substituted for it.
                        const std::size_t dw_copy_streams =
                            (dw_compose_event_req && wi < dw_compose_events.size()) ?
                                llama_tp5_output_copy_record(ctx) :
                                0;
                        const bool dw_output_barrier = dw_copy_streams > 0;
                        const auto wait_start = now_ns();
                        const float * single_output_logits = nullptr;
                        // DUALWIN-COMPOSE-EVENT: with the gate ON the per-epoch
                        // HOST synchronization is REPLACED, not augmented, when
                        // the output-copy events recorded above exist.
                        //
                        // Proof chain (no relay counter is used): those events
                        // were recorded on exactly the child streams the Meta
                        // D2H readback of the logits tensor uses, after
                        // llama_decode returned, so each record is ordered behind
                        // that asynchronous copy on the same stream. Waiting on
                        // them (below, after the relay drain) proves the host
                        // logits row is readable.
                        //
                        // Removed: the scheduler-wide
                        // ggml_backend_sched_synchronize performed by
                        // llama_synchronize / llama_get_logits_ith. Kept: the
                        // unchanged relay dual-completion drain below, still
                        // required by this runtime's own exit predicate. The
                        // output wait is deferred past the drain
                        // so the copies complete while the host services the
                        // relay instead of blocking first.
                        if (!dw_output_barrier) {
                            if (dw_compose_event_req && wi < dw_compose_events.size()) {
                                // Gate ON but no sound output barrier: keep the
                                // exact stock synchronize, fail-closed, counted.
                                // No part of this epoch is proven readable by an
                                // event, so the stock barrier stays exactly as in
                                // the OFF arm; it is never deferred to a later
                                // consumer.
                                ++dw_compose_events[wi].fallback_no_output_barrier;
                            }
                            if (output_rows == 1) {
                                // This public getter synchronizes the context. Keep only
                                // its pointer here; read logits after native retirement.
                                single_output_logits = llama_get_logits_ith(ctx, single_output_row);
                            } else {
                                llama_synchronize(ctx);
                            }
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
                                bool          finished      = false;
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
                        // DUALWIN-COMPOSE-EVENT: the epoch's relay drain above
                        // is the runtime's exit predicate and is unchanged. Only
                        // now, immediately before the host reads logits, do we
                        // host-complete the output-copy events recorded right
                        // after decode: their completion proves the async D2H
                        // copy of this epoch finished, which is the barrier the
                        // replaced llama_synchronize used to provide. The wait
                        // is placed after the drain so the drain's host work can
                        // proceed while any remaining copy latency elapses;
                        // whether that overlap actually saves time is not
                        // measured and is not claimed.
                        if (dw_output_barrier) {
                            auto & dw_be = dw_compose_events[wi];
                            ++dw_be.output_event_syncs;
                            ++dw_be.host_sync_replaced;
                            const std::size_t dw_proven =
                                llama_tp5_output_event_wait(ctx);
                            require(dw_proven == dw_copy_streams,
                                    "output-copy event barrier did not cover every recorded stream");
                        }
                        metrics.native_wait_ns += now_ns() - wait_start;
                        active.fetch_sub(1);
                        epoch_active = false;
                        require(!failure.stopped.load(), "request stopped before state commit");
                        const auto commit_start = now_ns();
                        for (std::size_t row = 0; row < local_rows.size(); ++row) {
                            const auto local          = local_rows[row];
                            auto &     request        = result.requests[ids[local]];
                            const auto input_position = consumed[local]++;
                            if (!batch.batch.logits[row]) {
                                continue;
                            }
                            // DUALWIN-COMPOSE-EVENT: with the event barrier
                            // above, every row is resolved without a further
                            // host synchronize; the barrier already proved this
                            // epoch's output copy completed. OFF and the
                            // counted fallback keep the stock syncing getter.
                            const bool single_row_hit =
                                output_rows == 1 && row == static_cast<std::size_t>(single_output_row);
                            const float * logits =
                                single_row_hit && single_output_logits != nullptr ?
                                    single_output_logits :
                                    (dw_output_barrier ?
                                         llama_get_logits_ith_no_sync(ctx, row) :
                                         llama_get_logits_ith(ctx, row));
                            if (dw_output_barrier) {
                                ++dw_compose_events[wi].no_sync_logits_calls;
                            }
                            require(logits != nullptr, "full logits row is missing");
                            const auto greedy = tq::sampling::select_finite_greedy(logits, n_vocab);
                            require(greedy.usable, "nonfinite full model logits");
                            const llama_token selected = greedy.selected.id;
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
                workers_done.fetch_add(1, std::memory_order_release);
            });
        } catch (...) {
            failure.fail(std::current_exception());
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
    if (dw_compose_event_req) {
        // Receipt counts host calls only. Any output-copy event still owned by
        // a context is released by ggml_backend_meta_free when the context is
        // destroyed below; on the success path every epoch's wait already
        // released them, and the two llama_synchronize calls per window above
        // drained the streams before that destruction. There is no per-window
        // peer event, no peer wait and no blocking event synchronize here: the
        // peer layer was a same-phase rendezvous between windows that share no
        // writable state.
        // host_sync_replaced>0 states that this window's per-epoch host
        // synchronization was the recorded OUTPUT-COPY events instead of the
        // scheduler-wide llama_synchronize. It does NOT prove on real hardware
        // that each event shares a stream with the copy it is meant to cover —
        // that rests on the cited ggml/llama code paths and still needs the
        // RUNBOOK's device trace. fallback_no_output_barrier counts epochs that
        // kept the stock synchronize because no sound output-copy barrier was
        // available; no_sync_logits_calls counts rows resolved without a
        // synchronizing getter.
        std::fprintf(stderr,
            "TP5_DW_COMPOSE_EVENT enabled=1 flag=GGML_TP5_DW_COMPOSE_EVENT"
            " windows=%u\n", unsigned(opts.windows));
        for (std::uint32_t wi = 0; wi < dw_compose_events.size();
             ++wi) {
            const auto & dw_be = dw_compose_events[wi];
            std::fprintf(stderr,
                "TP5_DW_COMPOSE_EVENT window=%u epochs=%llu"
                " output_event_syncs=%llu host_sync_replaced=%llu"
                " fallback_no_output_barrier=%llu no_sync_logits_calls=%llu\n",
                unsigned(wi),
                (unsigned long long)result.windows[wi].decode_calls,
                (unsigned long long)dw_be.output_event_syncs,
                (unsigned long long)dw_be.host_sync_replaced,
                (unsigned long long)dw_be.fallback_no_output_barrier,
                (unsigned long long)dw_be.no_sync_logits_calls);
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
