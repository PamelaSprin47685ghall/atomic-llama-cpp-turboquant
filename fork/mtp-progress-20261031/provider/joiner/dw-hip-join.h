#pragma once

// DUALWIN-HIP-JOIN — offline joiner for device-stream overlap evidence.
//
// This header is the CONSUMER half of the HIP-ready throughput-prove chain:
//
//   relay.hip  (GGML_TP5_DW_OVERLAP_METER=1, GGML_TP5_DW_HIP_SPANS=1)
//     -> emits per-rank aggregates   TP5_DW_HIP_OVERLAP  (existing)
//        and per-span device times   TP5_DW_HIP_SPANS / TP5_DW_HIP_SPAN (new)
//   runtime    (GGML_TP5_DW_HIP_SPANS=1)
//     -> emits per-window join keys  TP5_DW_HIP_OVERLAP_MAP (existing line,
//        now also emitted under the HIP_SPANS gate)
//   dw-hip-overlap-join (this header)
//     -> parses the combined evidence log, joins windows to device spans,
//        and emits TP5_DW_HIP_OVERLAP_JOINED* verdict lines.
//
//   - Every begin_ns/end_ns is a DEVICE timestamp on that span's physical
//     device, measured by the relay's HIP event pair relative to that
//     device's anchor event. It is device stream residency — the span
//     covers the drain of earlier queued work on that stream — never host
//     wall clock and never kernel-execution-only.
//   - Each physical device has an INDEPENDENT device clock. Timestamps
//     from different PCIs are never subtracted, unioned, intersected or
//     compared in any way: there is no shared clock to do it against, and
//     this joiner refuses to fabricate one. Every emitted number is
//     single-clock: it is computed inside exactly one PCI's timestamp set.
//   - Cross-window device overlap exists ONLY as the same-PCI overlap of
//     two windows' span sets on that PCI, and only when every provider on
//     that PCI used the shared per-device epoch (shared_epoch=1). With
//     per-provider anchors (shared_epoch=0) even the same-PCI cross-
//     provider numbers would compare unrelated clocks: refused.
//   - scope=device_stream on every emitted line, with the clock boundary
//     machine-readable (pair_scope=same_pci_shared_epoch for proven
//     window pairs, cross_provider_unattributed for provider-scoped
//     diagnostics, or not_computable when no pair of either kind exists).
//
// Failure policy (fail-closed, never swallowed):
//   - Any TP5_DW_HIP_* line that does not parse exactly is a hard error
//     naming the line number. Unrelated lines are skipped (a combined log
//     interleaves many producers).
//   - dropped > 0 (relay ring overflow) or errors > 0 (relay HIP failure)
//     refuses the verdict: incomplete device evidence never yields a
//     number. exit code 1, reason on stderr.
//   - Aggregate cross-check: when a TP5_DW_HIP_OVERLAP aggregate and the
//     corresponding spans are both present and dropped == 0, the span
//     COUNT must equal the aggregate exactly (integer fact). The DURATION
//     total is deliberately not re-derived from the spans: total_ns and
//     end_ns-begin_ns are rounded to ns through different subtractions of
//     float event math, so an exact equality check would false-fail on
//     sub-microsecond spans. The rounding is documented, not hidden; the
//     count check still catches truncated or duplicated span logs.
//   - PCI/rank consistency: one (provider, rank) maps to exactly one PCI
//     string across all its lines; one PCI string maps to one rank inside
//     a provider. Violations are hard errors.
//   - Provider mapping (TP5_DW_HIP_OVERLAP_MAP provider=) is the ONLY
//     proven provider->window link, and it is all-or-nothing:
//       * missing: MAP lines either ALL carry provider= or NONE do; a mix
//         is refused, and when provider= is present on every MAP line but
//         a span provider has no matching provider=, the mapping is
//         incomplete and refused (never downgraded to shared);
//       * duplicate: a second provider= key on one MAP line is a parse
//         error; two windows claiming the same provider= pointer are
//         refused (canonical pointer text, so 0xABC and 0xabc collide);
//       * conflict: a provider matching one window's provider= AND a
//         DIFFERENT window's native= is ambiguous and refused;
//     Old logs without provider= remain shared attribution: usable
//     diagnostics, but NO window claim, NO _JOINED_PAIR row and NO
//     pair_scope=same_pci_shared_epoch — the old unattributed logs cannot
//     pass a window-overlap gate. Provider-span interval intersection is
//     device-event in-flight overlap; it is NEVER kernel parallelism.
//   - Window attribution is PROVEN from the log, never inferred:
//     pointer mode (no MAP carries provider=; every span provider matches
//     some MAP native pointer) or provider mode (every MAP line carries
//     provider= naming its window's relay provider, and every span
//     provider matches one). Mixed evidence is an error. Otherwise:
//     shared mode — used only for old MAP forms that carry no provider=
//     at all, so the log ties no provider to any window and nothing is
//     attributed to a window (window rows carry no busy numbers). A PCI
//     legitimately appears in several windows' mapped PCI sets when those
//     windows share the physical device (dual-window TP5 runs every window
//     on every rank's GPU): that is topology, not evidence corruption,
//     and never a refusal.
//   - Shared mode still measures what the log proves — as DIAGNOSTICS on
//     their own tag, deliberately kept apart from the window-pair
//     contract: per-provider per-PCI rows, and the same-PCI overlap of
//     two DISTINCT providers' span sets under the shared per-device
//     epoch, emitted as _JOINED_PROVIDER_PAIR (provider pointers inline)
//     with pair_scope=cross_provider_unattributed. The original
//     _JOINED_PAIR tag and pair_scope=same_pci_shared_epoch are reserved
//     for PROVEN window attribution (pointer/provider mode) — a
//     window-overlap gate reading the original pair tag/scope can never
//     be satisfied by unattributed evidence. A PCI with fewer than two
//     distinct span-carrying providers yields NO provider pair
//     (fail-closed). Overlap means device-event-time interval
//     intersection of submit envelopes, never a kernel-level
//     co-execution claim.
//
// Known non-goals (documented, not hidden):
//   - Same-stream serialization inside one provider is a stream-order
//     property of the hardware, not re-verified here: the timestamps are
//     float-rounded to ns and a strict end_i <= begin_{i+1} assertion
//     would false-fail on sub-microsecond rounding. Sequence numbers are
//     still required to be strictly increasing per (provider, rank).
//   - Cross-device concurrency (two different GPUs busy at the same host
//     moment) is NOT computed and CANNOT be computed from device event
//     clocks; it needs a host-side correlation (e.g. rocprof) that this
//     chain does not provide. Nothing in the output may be read as a
//     cross-device claim.

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace dw_hip_join {

struct parse_error : std::runtime_error {
    unsigned long long line;
    parse_error(const std::string & what, unsigned long long line_no)
        : std::runtime_error(what), line(line_no) {}
};

struct span_rec {
    std::string provider;
    unsigned rank = 0;
    std::string pci;
    unsigned long long seq = 0;
    unsigned long long begin_ns = 0;
    unsigned long long end_ns = 0;
};

struct map_rec {
    unsigned window = 0;
    std::string native;                 // %p text as emitted, e.g. 0x55...
    bool native_null = false;           // (nil)
    std::string provider;               // provider=%p (newer runtime): this window's relay provider
    bool provider_present = false;      // provider= seen on this MAP line
    std::vector<std::string> pci;       // pci_bdf%u=%s in emit order
};

struct agg_rec {
    std::string provider;
    unsigned rank = 0;
    std::string pci;
    unsigned long long spans = 0;
    unsigned long long total_ns = 0;
    unsigned long long skipped_pending = 0;
    unsigned long long skipped_mode = 0;
    unsigned long long errors = 0;
};

struct spans_hdr_rec {
    std::string provider;
    unsigned rank = 0;
    std::string pci;
    unsigned long long n = 0;
    unsigned long long dropped = 0;
    unsigned long long errors = 0;      // relay HIP failure count; 0 = absent (older emitter)
    int shared_epoch = -1;              // -1: field absent (older emitter)
};

struct model {
    std::vector<span_rec> spans;
    std::vector<map_rec> map;
    std::vector<agg_rec> aggs;
    std::vector<spans_hdr_rec> spans_hdrs;

    // Feed one line. Throws parse_error for malformed TP5_DW_HIP_* lines;
    // returns false for lines outside the evidence families (skipped).
    bool feed_line(const char * line, unsigned long long line_no);
    // Cross-checks after all lines are fed. Throws std::runtime_error.
    void finalize() const;
};

namespace detail {

inline bool is_space(char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; }

// Splits on whitespace; empty tokens are never produced.
inline std::vector<std::string> tokenize(const char * line) {
    std::vector<std::string> out;
    const char * p = line;
    while (*p != '\0') {
        while (is_space(*p)) {
            ++p;
        }
        if (*p == '\0') {
            break;
        }
        const char * start = p;
        while (*p != '\0' && !is_space(*p)) {
            ++p;
        }
        out.emplace_back(start, static_cast<std::size_t>(p - start));
    }
    return out;
}

// "key=value" split at the FIRST '='; throws when there is none.
inline std::pair<std::string, std::string> split_kv(const std::string & tok, unsigned long long line_no) {
    const std::string::size_type eq = tok.find('=');
    if (eq == std::string::npos || eq == 0) {
        throw parse_error("malformed token (expected key=value): " + tok, line_no);
    }
    return { tok.substr(0, eq), tok.substr(eq + 1) };
}

inline unsigned long long parse_ull(const std::string & v, const char * key, unsigned long long line_no) {
    if (v.empty()) {
        throw parse_error(std::string("empty integer for ") + key, line_no);
    }
    errno = 0;
    char * end = nullptr;
    const unsigned long long value = std::strtoull(v.c_str(), &end, 10);
    if (errno != 0 || end == nullptr || *end != '\0') {
        throw parse_error(std::string("bad integer for ") + key + ": " + v, line_no);
    }
    return value;
}

inline unsigned parse_uint(const std::string & v, const char * key, unsigned long long line_no) {
    const unsigned long long value = parse_ull(v, key, line_no);
    if (value > 0xffffffffULL) {
        throw parse_error(std::string("out-of-range unsigned for ") + key + ": " + v, line_no);
    }
    return static_cast<unsigned>(value);
}

// "%p" text: glibc prints 0x-prefixed lowercase hex, or (nil). Canonicalize
// to a lowercase digit string without the 0x prefix so pointer text from
// different %p widths still compares equal.
inline std::string canon_pointer(const std::string & v, const char * key, unsigned long long line_no, bool & is_null) {
    is_null = false;
    if (v == "(nil)" || v == "(0x0)") {
        is_null = true;
        return std::string();
    }
    std::string s = v;
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (s.rfind("0x", 0) == 0) {
        s = s.substr(2);
    }
    if (s.empty() || s.find_first_not_of("0123456789abcdef") != std::string::npos) {
        throw parse_error(std::string("bad pointer for ") + key + ": " + v, line_no);
    }
    return s;
}

// Requires the next tokens to be exactly `expected` in order.
inline void expect_literals(const std::vector<std::string> & toks, size_t & i,
                            const char * const * expected, size_t n, unsigned long long line_no) {
    for (size_t k = 0; k < n; ++k, ++i) {
        if (i >= toks.size() || toks[i] != expected[k]) {
            throw parse_error(std::string("expected literal ") + expected[k], line_no);
        }
    }
}

struct kv_map {
    std::vector<std::pair<std::string, std::string>> items;
    bool has(const char * key) const {
        for (const auto & kv : items) {
            if (kv.first == key) {
                return true;
            }
        }
        return false;
    }
    const std::string & at(const char * key) const {
        for (const auto & kv : items) {
            if (kv.first == key) {
                return kv.second;
            }
        }
        throw std::runtime_error(std::string("missing key ") + key);
    }
};

// Strict key=value parse of tokens [i, end): every token must be key=value,
// no key may repeat, no unknown key allowed.
inline kv_map parse_kv_strict(const std::vector<std::string> & toks, size_t i,
                              const char * const * known, size_t n_known, unsigned long long line_no) {
    kv_map out;
    for (; i < toks.size(); ++i) {
        const auto kv = split_kv(toks[i], line_no);
        bool known_key = false;
        for (size_t k = 0; k < n_known; ++k) {
            if (kv.first == known[k]) {
                known_key = true;
                break;
            }
        }
        if (!known_key) {
            throw parse_error("unknown key: " + kv.first, line_no);
        }
        for (const auto & prev : out.items) {
            if (prev.first == kv.first) {
                throw parse_error("duplicate key: " + kv.first, line_no);
            }
        }
        out.items.push_back(kv);
    }
    return out;
}

inline std::string u64(unsigned long long v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%llu", v);
    return std::string(buf);
}

// Merges closed-open intervals; returns the union measure.
inline unsigned long long union_measure(std::vector<std::pair<unsigned long long, unsigned long long>> v) {
    std::sort(v.begin(), v.end());
    unsigned long long total = 0;
    unsigned long long cur_b = 0, cur_e = 0;
    bool open = false;
    for (const auto & iv : v) {
        if (iv.second <= iv.first) {
            continue;  // zero-length spans occupy nothing
        }
        if (!open || iv.first > cur_e) {
            if (open) {
                total += cur_e - cur_b;
            }
            cur_b = iv.first;
            cur_e = iv.second;
            open = true;
        } else if (iv.second > cur_e) {
            cur_e = iv.second;
        }
    }
    if (open) {
        total += cur_e - cur_b;
    }
    return total;
}

// Intersection measure of two already-merged sorted interval lists.
inline unsigned long long intersect_measure(const std::vector<std::pair<unsigned long long, unsigned long long>> & a,
                                            const std::vector<std::pair<unsigned long long, unsigned long long>> & b) {
    size_t i = 0, j = 0;
    unsigned long long total = 0;
    while (i < a.size() && j < b.size()) {
        const unsigned long long lo = a[i].first > b[j].first ? a[i].first : b[j].first;
        const unsigned long long hi = a[i].second < b[j].second ? a[i].second : b[j].second;
        if (hi > lo) {
            total += hi - lo;
        }
        if (a[i].second <= b[j].second) {
            ++i;
        } else {
            ++j;
        }
    }
    return total;
}

using intervals = std::vector<std::pair<unsigned long long, unsigned long long>>;

inline intervals merge_intervals(intervals v) {
    std::sort(v.begin(), v.end());
    intervals out;
    for (const auto & iv : v) {
        if (iv.second <= iv.first) {
            continue;
        }
        if (!out.empty() && iv.first <= out.back().second) {
            if (iv.second > out.back().second) {
                out.back().second = iv.second;
            }
        } else {
            out.push_back(iv);
        }
    }
    return out;
}

}  // namespace detail

inline bool model::feed_line(const char * line, unsigned long long line_no) {
    const std::vector<std::string> toks = detail::tokenize(line);
    if (toks.empty()) {
        return false;
    }
    if (toks[0] == "TP5_DW_HIP_SPAN") {
        static const char * const known[] = { "provider", "rank", "pci", "seq", "begin_ns", "end_ns" };
        const detail::kv_map kv = detail::parse_kv_strict(toks, 1, known, 6, line_no);
        span_rec s;
        bool null_ptr = false;
        s.provider = detail::canon_pointer(kv.at("provider"), "provider", line_no, null_ptr);
        if (null_ptr) {
            throw parse_error("null provider pointer in TP5_DW_HIP_SPAN", line_no);
        }
        s.rank     = detail::parse_uint(kv.at("rank"), "rank", line_no);
        s.pci      = kv.at("pci");
        s.seq      = detail::parse_ull(kv.at("seq"), "seq", line_no);
        s.begin_ns = detail::parse_ull(kv.at("begin_ns"), "begin_ns", line_no);
        s.end_ns   = detail::parse_ull(kv.at("end_ns"), "end_ns", line_no);
        if (s.end_ns < s.begin_ns) {
            throw parse_error("device timestamps reversed (end_ns < begin_ns)", line_no);
        }
        spans.push_back(std::move(s));
        return true;
    }
    if (toks[0] == "TP5_DW_HIP_SPANS") {
        // Two producer forms share this tag (both fail-closed strict):
        //   relay.hip per-provider/rank header:
        //     TP5_DW_HIP_SPANS provider=%p rank=%u pci=%s n=%llu dropped=%llu errors=%llu shared_epoch=1
        //   runtime.cpp per-run banner (no provider):
        //     TP5_DW_HIP_SPANS enabled=1 flag=GGML_TP5_DW_HIP_SPANS join_map=1 scope=device_stream device_only=1
        // The banner's enabled= is OPTIONAL (a tail may omit it) and is
        // accepted ONLY on this banner form. The header's errors= is parsed
        // (never ignored) and refused by analyze() when nonzero, exactly
        // like dropped=. Span body, map and aggregate lines stay strict.
        if (toks.size() < 2) {
            throw parse_error("TP5_DW_HIP_SPANS without any key=value fields", line_no);
        }
        bool header_form;
        {
            const auto kv0 = detail::split_kv(toks[1], line_no);
            header_form = kv0.first == "provider";
        }
        static const char * const hdr_known[] = { "provider", "rank", "pci", "n", "dropped", "errors", "shared_epoch" };
        static const char * const banner_known[] = { "enabled", "flag", "join_map", "scope", "device_only" };
        const detail::kv_map kv = header_form
            ? detail::parse_kv_strict(toks, 1, hdr_known, 7, line_no)
            : detail::parse_kv_strict(toks, 1, banner_known, 5, line_no);
        if (!header_form) {
            // Banner: informational gate summary, stored nowhere. Validate
            // every present key against the producer contract.
            if (kv.has("enabled")) {
                const unsigned long long e = detail::parse_ull(kv.at("enabled"), "enabled", line_no);
                if (e != 0 && e != 1) {
                    throw parse_error("enabled must be 0 or 1", line_no);
                }
            }
            if (kv.has("join_map")) {
                const unsigned long long j = detail::parse_ull(kv.at("join_map"), "join_map", line_no);
                if (j != 0 && j != 1) {
                    throw parse_error("join_map must be 0 or 1", line_no);
                }
            }
            if (kv.has("flag") && kv.at("flag") != "GGML_TP5_DW_HIP_SPANS") {
                throw parse_error("flag must be GGML_TP5_DW_HIP_SPANS", line_no);
            }
            if (kv.has("scope") && kv.at("scope") != "device_stream") {
                throw parse_error("scope must be device_stream", line_no);
            }
            if (kv.has("device_only") && kv.at("device_only") != "1") {
                throw parse_error("device_only must be 1", line_no);
            }
            return true;
        }
        spans_hdr_rec h;
        bool null_ptr = false;
        h.provider = detail::canon_pointer(kv.at("provider"), "provider", line_no, null_ptr);
        if (null_ptr) {
            throw parse_error("null provider pointer in TP5_DW_HIP_SPANS", line_no);
        }
        h.rank         = detail::parse_uint(kv.at("rank"), "rank", line_no);
        h.pci          = kv.at("pci");
        h.n            = detail::parse_ull(kv.at("n"), "n", line_no);
        h.dropped      = detail::parse_ull(kv.at("dropped"), "dropped", line_no);
        h.errors       = kv.has("errors") ? detail::parse_ull(kv.at("errors"), "errors", line_no) : 0;
        h.shared_epoch = static_cast<int>(detail::parse_ull(kv.at("shared_epoch"), "shared_epoch", line_no));
        if (h.shared_epoch != 0 && h.shared_epoch != 1) {
            throw parse_error("shared_epoch must be 0 or 1", line_no);
        }
        spans_hdrs.push_back(std::move(h));
        return true;
    }
    if (toks[0] == "TP5_DW_HIP_OVERLAP_MAP") {
        // Format (runtime.cpp): TP5_DW_HIP_OVERLAP_MAP window=%u native=%p
        // [provider=%p] [pci_bdf%u=%s]*  -- keys in fixed order, pci_bdf
        // entries variable. provider= (newer runtimes) names the window's
        // relay provider and is the ONLY key that proves which window a
        // span provider belongs to; MAP lines either all carry it or none
        // does (checked in analyze(), never inferred).
        if (toks.size() < 3) {
            throw parse_error("TP5_DW_HIP_OVERLAP_MAP too short", line_no);
        }
        map_rec m;
        {
            const auto kv = detail::split_kv(toks[1], line_no);
            if (kv.first != "window") {
                throw parse_error("expected window= in TP5_DW_HIP_OVERLAP_MAP", line_no);
            }
            m.window = detail::parse_uint(kv.second, "window", line_no);
        }
        {
            const auto kv = detail::split_kv(toks[2], line_no);
            if (kv.first != "native") {
                throw parse_error("expected native= in TP5_DW_HIP_OVERLAP_MAP", line_no);
            }
            m.native = detail::canon_pointer(kv.second, "native", line_no, m.native_null);
        }
        unsigned long long seen_mask = 0;
        for (size_t i = 3; i < toks.size(); ++i) {
            const auto kv = detail::split_kv(toks[i], line_no);
            if (kv.first == "provider") {
                if (m.provider_present) {
                    throw parse_error("duplicate key in TP5_DW_HIP_OVERLAP_MAP: provider", line_no);
                }
                bool null_provider = false;
                m.provider = detail::canon_pointer(kv.second, "provider", line_no, null_provider);
                if (null_provider) {
                    throw parse_error("null provider pointer in TP5_DW_HIP_OVERLAP_MAP", line_no);
                }
                m.provider_present = true;
                continue;
            }
            if (kv.first.rfind("pci_bdf", 0) != 0) {
                throw parse_error("unknown key in TP5_DW_HIP_OVERLAP_MAP: " + kv.first, line_no);
            }
            const unsigned long long rank = detail::parse_ull(kv.first.substr(7), "pci_bdf rank", line_no);
            if (rank >= 64) {
                throw parse_error("pci_bdf rank out of supported range", line_no);
            }
            const unsigned long long bit = 1ULL << rank;
            if (seen_mask & bit) {
                throw parse_error("duplicate pci_bdf rank in TP5_DW_HIP_OVERLAP_MAP", line_no);
            }
            seen_mask |= bit;
            if (kv.second.empty()) {
                throw parse_error("empty pci_bdf value", line_no);
            }
            m.pci.push_back(kv.second);
        }
        if (m.pci.empty()) {
            throw parse_error("TP5_DW_HIP_OVERLAP_MAP without any pci_bdf entry", line_no);
        }
        for (const auto & prev : map) {
            if (prev.window == m.window) {
                throw parse_error("duplicate window in TP5_DW_HIP_OVERLAP_MAP", line_no);
            }
        }
        map.push_back(std::move(m));
        return true;
    }
    if (toks[0] == "TP5_DW_HIP_OVERLAP") {
        // Relay aggregate line; the trailing literals are part of the
        // contract and are checked verbatim.
        static const char * const known[] = { "provider", "device", "rank", "pci", "active", "spans",
                                              "total_ns", "max_ns", "skipped_pending", "skipped_mode",
                                              "errors" };
        // Find where the trailing literals start.
        static const char * const tail[] = { "coverage=enqueue_to_completion_stream_span", "clock=hip_event", "device_only=1" };
        size_t tail_pos = toks.size();
        for (size_t i = 1; i + 3 <= toks.size(); ++i) {
            if (toks[i] == tail[0] && toks[i + 1] == tail[1] && toks[i + 2] == tail[2]) {
                tail_pos = i;
                break;
            }
        }
        if (tail_pos == toks.size()) {
            throw parse_error("TP5_DW_HIP_OVERLAP missing coverage/clock/device_only trailer", line_no);
        }
        // Strict kv parse of tokens [1, tail_pos); the trailing literal
        // tokens were verified above and are not key=value pairs.
        detail::kv_map kv2;
        for (size_t i = 1; i < tail_pos; ++i) {
            const auto kv3 = detail::split_kv(toks[i], line_no);
            bool known_key = false;
            for (size_t k = 0; k < 11; ++k) {
                if (kv3.first == known[k]) {
                    known_key = true;
                    break;
                }
            }
            if (!known_key) {
                throw parse_error("unknown key in TP5_DW_HIP_OVERLAP: " + kv3.first, line_no);
            }
            for (const auto & prev : kv2.items) {
                if (prev.first == kv3.first) {
                    throw parse_error("duplicate key in TP5_DW_HIP_OVERLAP: " + kv3.first, line_no);
                }
            }
            kv2.items.push_back(kv3);
        }
        agg_rec a;
        bool null_ptr = false;
        a.provider = detail::canon_pointer(kv2.at("provider"), "provider", line_no, null_ptr);
        if (null_ptr) {
            throw parse_error("null provider pointer in TP5_DW_HIP_OVERLAP", line_no);
        }
        (void) detail::parse_ull(kv2.at("device"), "device", line_no);  // validated, not joined on
        a.rank            = detail::parse_uint(kv2.at("rank"), "rank", line_no);
        a.pci             = kv2.at("pci");
        a.spans           = detail::parse_ull(kv2.at("spans"), "spans", line_no);
        a.total_ns        = detail::parse_ull(kv2.at("total_ns"), "total_ns", line_no);
        a.skipped_pending = detail::parse_ull(kv2.at("skipped_pending"), "skipped_pending", line_no);
        a.skipped_mode    = detail::parse_ull(kv2.at("skipped_mode"), "skipped_mode", line_no);
        a.errors          = detail::parse_ull(kv2.at("errors"), "errors", line_no);
        aggs.push_back(std::move(a));
        return true;
    }
    if (toks[0].rfind("TP5_DW_HIP_", 0) == 0) {
        // A TP5_DW_HIP_* line this joiner does not implement: the evidence
        // families above are the contract; anything else in that namespace
        // is a protocol change and must fail loudly, never silently skip.
        throw parse_error("unknown TP5_DW_HIP_* line: " + toks[0], line_no);
    }
    return false;  // unrelated producer line: skipped by design
}

inline void model::finalize() const {
    // One PCI string per (provider, rank) across every line family.
    std::map<std::pair<std::string, unsigned>, std::string> pci_by_provider_rank;
    auto note_pci = [&](const std::string & provider, unsigned rank, const std::string & pci,
                        const char * what, unsigned long long line_no) {
        const auto key = std::make_pair(provider, rank);
        const auto it = pci_by_provider_rank.find(key);
        if (it == pci_by_provider_rank.end()) {
            pci_by_provider_rank.emplace(key, pci);
        } else if (it->second != pci) {
            throw std::runtime_error(std::string("conflicting pci for provider/rank in ") + what +
                                     " at line " + detail::u64(line_no));
        }
    };
    for (const auto & s : spans) {
        note_pci(s.provider, s.rank, s.pci, "TP5_DW_HIP_SPAN", 0);
    }
    for (const auto & h : spans_hdrs) {
        note_pci(h.provider, h.rank, h.pci, "TP5_DW_HIP_SPANS", 0);
    }
    for (const auto & a : aggs) {
        note_pci(a.provider, a.rank, a.pci, "TP5_DW_HIP_OVERLAP", 0);
    }
    // One rank per (provider, pci): a rank is a lane on one device.
    std::map<std::pair<std::string, std::string>, unsigned> rank_by_provider_pci;
    for (const auto & s : spans) {
        const auto key = std::make_pair(s.provider, s.pci);
        const auto it = rank_by_provider_pci.find(key);
        if (it == rank_by_provider_pci.end()) {
            rank_by_provider_pci.emplace(key, s.rank);
        } else if (it->second != s.rank) {
            throw std::runtime_error("conflicting rank for provider/pci in TP5_DW_HIP_SPAN");
        }
    }
    // Sequence numbers strictly increasing per (provider, rank).
    std::map<std::pair<std::string, unsigned>, unsigned long long> last_seq;
    for (const auto & s : spans) {
        const auto key = std::make_pair(s.provider, s.rank);
        const auto it = last_seq.find(key);
        if (it != last_seq.end() && s.seq <= it->second) {
            throw std::runtime_error("non-increasing seq for provider/rank in TP5_DW_HIP_SPAN");
        }
        last_seq[key] = s.seq;
    }
    // Sequence numbers are exactly 0..count-1 per (provider, rank): the
    // producer numbers harvested spans densely. A missing or repeated
    // sequence number means lost or duplicated evidence even when
    // dropped=0, so it is refused, never tolerated.
    {
        std::map<std::pair<std::string, unsigned>, std::set<unsigned long long>> seqs;
        for (const auto & s : spans) {
            seqs[{s.provider, s.rank}].insert(s.seq);
        }
        for (const auto & sp : seqs) {
            const auto & set = sp.second;
            if (*set.begin() != 0 || *set.rbegin() != set.size() - 1) {
                throw std::runtime_error("seq gap or non-zero base for provider " + sp.first.first +
                                         " rank " + detail::u64(sp.first.second) + " (count " +
                                         detail::u64(set.size()) + ", first " + detail::u64(*set.begin()) +
                                         ", last " + detail::u64(*set.rbegin()) + ")");
            }
        }
    }
    // Header n must equal the number of span lines for that (provider, rank).
    for (const auto & h : spans_hdrs) {
        unsigned long long count = 0;
        for (const auto & s : spans) {
            if (s.provider == h.provider && s.rank == h.rank) {
                ++count;
            }
        }
        if (count != h.n) {
            throw std::runtime_error("TP5_DW_HIP_SPANS n=" + detail::u64(h.n) + " but " + detail::u64(count) +
                                     " span lines present for provider/rank");
        }
    }
    // Aggregate cross-check: span COUNT must equal the aggregate exactly
    // (integer fact). Durations are deliberately NOT compared: the relay's
    // total_ns is elapsed(begin,end) while each span's end_ns-begin_ns is a
    // difference of two elapsed(anchor,event) values; both are rounded to ns
    // through different float subtractions, so exact equality would
    // false-fail on sub-microsecond spans. The count check still catches
    // truncated or duplicated span logs. With drops the aggregate counts
    // spans the log does not contain; that mismatch is rejected by the
    // dropped>0 verdict refusal instead of being hidden here.
    for (const auto & a : aggs) {
        const auto h = std::find_if(spans_hdrs.begin(), spans_hdrs.end(), [&](const spans_hdr_rec & h) {
            return h.provider == a.provider && h.rank == a.rank;
        });
        if (h == spans_hdrs.end() || h->dropped != 0) {
            continue;
        }
        unsigned long long count = 0;
        for (const auto & s : spans) {
            if (s.provider == a.provider && s.rank == a.rank) {
                ++count;
            }
        }
        if (count != a.spans) {
            throw std::runtime_error("aggregate mismatch for provider/rank: span lines " + detail::u64(count) +
                                     " vs aggregate spans " + detail::u64(a.spans));
        }
    }
}

// Joined verdict: SINGLE-CLOCK numbers only. Every measure below is computed
// inside exactly one PCI's device timestamp set. There is no cross-device
// number anywhere: different GPUs have independent event clocks and their
// intervals are never combined. Cross-window overlap exists only per shared
// PCI (window-attributed modes — pointer or provider — with shared_epoch=1
// enforced; in shared mode the same math is emitted as cross-provider pairs,
// labeled by provider index, never claiming a window).
struct verdict {
    enum class mode { pointer, provider, shared };
    mode attribution;
    struct window_pci_row {
        std::string pci;
        unsigned long long spans = 0;
        unsigned long long busy_ns = 0;
        unsigned long long wall_ns = 0;
        double busy_frac = 0.0;
    };
    struct window_row {
        unsigned window;
        std::vector<std::string> pci;                  // mapped set (MAP order)
        std::vector<window_pci_row> pci_busy;          // pointer mode only
    };
    struct pair_pci_row {
        unsigned a, b;
        std::string pci;                               // present in both maps
        unsigned long long spans_a = 0;
        unsigned long long spans_b = 0;
        unsigned long long overlap_ns = 0;             // same-device intersection
    };
    struct provider_pair_row {
        std::string provider_a;                        // canonical, distinct
        std::string provider_b;
        std::string pci;                               // shared physical device
        unsigned long long spans_a = 0;
        unsigned long long spans_b = 0;
        unsigned long long overlap_ns = 0;             // same-device intersection
    };
    struct provider_pci_row {
        std::string pci;
        unsigned long long spans = 0;
        unsigned long long busy_ns = 0;                // union over this provider's spans on pci
        unsigned long long wall_ns = 0;                // [min begin, max end], same clock
        double busy_frac = 0.0;
    };
    struct provider_row {
        unsigned index = 0;                            // deterministic label: pointer-sorted ordinal
        std::string provider;                          // canonical provider pointer text
        std::vector<provider_pci_row> pci_busy;
    };
    struct pci_row {
        std::string pci;
        unsigned long long spans = 0;
        unsigned long long busy_ns = 0;                // union over all spans on pci
        unsigned long long wall_ns = 0;                // [min begin, max end], same clock
        double busy_frac = 0.0;
    };
    std::vector<window_row> windows;
    std::vector<provider_row> providers;   // shared mode only
    std::vector<pair_pci_row> pairs;       // window ids, pointer/provider mode only
    std::vector<provider_pair_row> provider_pairs;  // shared mode only, provider-scoped
    std::vector<pci_row> pcis;
};

// Computes the verdict from a finalized model. Throws std::runtime_error on
// any refusal condition (each is a hard failure, never a degraded number).
inline verdict analyze(const model & m) {
    if (m.map.empty()) {
        throw std::runtime_error("no TP5_DW_HIP_OVERLAP_MAP lines: no window join keys");
    }
    if (m.spans.empty()) {
        throw std::runtime_error("no TP5_DW_HIP_SPAN lines: no device spans");
    }
    // Refuse on relay-side incompleteness.
    for (const auto & h : m.spans_hdrs) {
        if (h.dropped != 0) {
            throw std::runtime_error("refusing verdict: dropped=" + detail::u64(h.dropped) +
                                     " spans lost to ring overflow (provider " + h.provider + " rank " +
                                     detail::u64(h.rank) + ")");
        }
    }
    for (const auto & h : m.spans_hdrs) {
        if (h.errors != 0) {
            throw std::runtime_error("refusing verdict: errors=" + detail::u64(h.errors) +
                                     " relay HIP failures on the span path (provider " + h.provider +
                                     " rank " + detail::u64(h.rank) + ")");
        }
    }
    for (const auto & a : m.aggs) {
        if (a.errors != 0) {
            throw std::runtime_error("refusing verdict: relay meter errors=" + detail::u64(a.errors) +
                                     " (provider " + a.provider + " rank " + detail::u64(a.rank) + ")");
        }
        if (a.skipped_mode != 0) {
            throw std::runtime_error("refusing verdict: skipped_mode=" + detail::u64(a.skipped_mode) +
                                     " (capture/replay submits bypass direct metering; provider " + a.provider +
                                     " rank " + detail::u64(a.rank) + ")");
        }
    }
    // Window attribution.
    std::map<std::string, const map_rec *> by_native;
    std::map<std::string, const map_rec *> by_provider;
    bool any_provider_key = false, all_provider_keys = true;
    for (const auto & w : m.map) {
        if (!w.native_null) {
            const bool fresh = by_native.emplace(w.native, &w).second;
            if (!fresh) {
                throw std::runtime_error("two windows claim the same native pointer");
            }
        }
        if (w.provider_present) {
            any_provider_key = true;
            const bool fresh = by_provider.emplace(w.provider, &w).second;
            if (!fresh) {
                throw std::runtime_error("duplicate provider mapping: two windows claim the same provider pointer");
            }
        } else {
            all_provider_keys = false;
        }
    }
    if (any_provider_key && !all_provider_keys) {
        throw std::runtime_error("mixed TP5_DW_HIP_OVERLAP_MAP forms: some lines carry provider=, others do not");
    }
    // A span provider is attributed to a window only when the log proves
    // the link: provider= (the MAP names its window's relay provider) or,
    // on older MAP forms, the native pointer itself. A provider matching
    // both keys of DIFFERENT windows is ambiguous and refused. When the
    // MAP carries provider=, every span provider MUST map to one of those
    // entries: an unmapped provider is a broken identity chain (missing
    // mapping) and refuses — it is never silently downgraded to shared.
    // Only old MAP forms without any provider= fall back to shared
    // diagnostics when no native pointer matches; nothing is ever guessed
    // from ordering, priority, slot or creation order.
    bool all_matched = !m.spans.empty();
    bool none_matched = true;
    std::string first_unmapped;
    for (const auto & s : m.spans) {
        const auto pit = by_provider.find(s.provider);
        const auto nit = by_native.find(s.provider);
        const map_rec * w = nullptr;
        if (pit != by_provider.end()) {
            w = pit->second;
            if (nit != by_native.end() && nit->second != pit->second) {
                throw std::runtime_error("ambiguous window attribution: provider " + s.provider +
                                         " matches one window's provider= and another window's native=");
            }
        } else if (!any_provider_key && nit != by_native.end()) {
            w = nit->second;
        }
        const bool matched = w != nullptr;
        if (!matched && first_unmapped.empty()) {
            first_unmapped = s.provider;
        }
        all_matched = all_matched && matched;
        none_matched = none_matched && !matched;
    }
    if (any_provider_key && !all_matched) {
        throw std::runtime_error("incomplete provider mapping: span provider " + first_unmapped +
                                 " has no TP5_DW_HIP_OVERLAP_MAP provider= entry");
    }
    if (!all_matched && !none_matched) {
        throw std::runtime_error("mixed window attribution: some span providers match a mapped window key, others do not");
    }
    verdict v;
    v.attribution = !all_matched ? verdict::mode::shared
                  : (any_provider_key ? verdict::mode::provider : verdict::mode::pointer);
    // Cross-provider comparability on a shared PCI: the providers' anchors
    // must be the same per-device epoch. shared_epoch=0 or absent is a hard
    // refusal — otherwise same-PCI cross-provider overlap would compare
    // unrelated clocks.
    {
        std::map<std::string, std::map<std::string, unsigned>> providers_by_pci;
        for (const auto & s : m.spans) {
            ++providers_by_pci[s.pci][s.provider];
        }
        std::map<std::pair<std::string, unsigned>, int> epoch_by_provider_rank;
        for (const auto & h : m.spans_hdrs) {
            epoch_by_provider_rank[{h.provider, h.rank}] = h.shared_epoch;
        }
        for (const auto & s : m.spans) {
            const auto & provs = providers_by_pci[s.pci];
            if (provs.size() < 2) {
                continue;
            }
            const auto it = epoch_by_provider_rank.find({s.provider, s.rank});
            const int epoch = it == epoch_by_provider_rank.end() ? -1 : it->second;
            if (epoch != 1) {
                throw std::runtime_error("refusing verdict: pci " + s.pci +
                                         " carries spans from multiple providers but shared_epoch!=1" +
                                         " (provider " + s.provider + " rank " + detail::u64(s.rank) +
                                         "); cross-provider timestamps are not comparable");
            }
        }
    }
    // Spans per provider (always provable: the provider is on every span
    // line) and per window (only under proven attribution).
    std::map<unsigned, std::map<std::string, detail::intervals>> wpci;
    std::map<std::string, std::map<std::string, detail::intervals>> ppci;
    for (const auto & s : m.spans) {
        ppci[s.provider][s.pci].push_back({s.begin_ns, s.end_ns});
        if (v.attribution != verdict::mode::shared) {
            const std::map<std::string, const map_rec *> & key_map =
                v.attribution == verdict::mode::provider ? by_provider : by_native;
            const auto it = key_map.find(s.provider);
            if (it == key_map.end()) {
                throw std::runtime_error("unreachable: span provider lost between mode decision and attribution");
            }
            const map_rec & w = *it->second;
            bool owns = false;
            for (const auto & p : w.pci) {
                owns = owns || (p == s.pci);
            }
            if (!owns) {
                throw std::runtime_error("span pci " + s.pci + " not in window " + detail::u64(w.window) +
                                         "'s mapped pci set");
            }
            wpci[w.window][s.pci].push_back({s.begin_ns, s.end_ns});
        }
    }
    // Per-PCI rows: every number inside one device clock.
    std::map<std::string, detail::intervals> by_pci;
    for (const auto & s : m.spans) {
        by_pci[s.pci].push_back({s.begin_ns, s.end_ns});
    }
    std::map<std::string, const verdict::pci_row *> pci_lookup;
    for (const auto & pp : by_pci) {
        verdict::pci_row row;
        row.pci = pp.first;
        row.spans = pp.second.size();
        unsigned long long wall_b = pp.second.front().first;
        unsigned long long wall_e = pp.second.front().second;
        for (const auto & iv : pp.second) {
            wall_b = iv.first < wall_b ? iv.first : wall_b;
            wall_e = iv.second > wall_e ? iv.second : wall_e;
        }
        row.wall_ns = wall_e - wall_b;
        if (row.wall_ns == 0) {
            throw std::runtime_error("degenerate evidence: pci " + pp.first + " has wall_ns == 0");
        }
        row.busy_ns = detail::union_measure(detail::merge_intervals(pp.second));
        row.busy_frac = static_cast<double>(row.busy_ns) / static_cast<double>(row.wall_ns);
        v.pcis.push_back(std::move(row));
    }
    for (const auto & row : v.pcis) {
        pci_lookup.emplace(row.pci, &row);
    }
    // Per-window rows: only under proven attribution (pointer or provider
    // mode). Shared mode attributes nothing to a window, so its window
    // rows carry the mapped PCI set alone — never a busy number.
    if (v.attribution != verdict::mode::shared) {
        for (const auto & w : m.map) {
            verdict::window_row row;
            row.window = w.window;
            row.pci = w.pci;
            for (const auto & p : w.pci) {
                verdict::window_pci_row pr;
                pr.pci = p;
                const auto pit = wpci[w.window].find(p);
                if (pit != wpci[w.window].end()) {
                    pr.spans = pit->second.size();
                    pr.busy_ns = detail::union_measure(detail::merge_intervals(pit->second));
                }
                const auto lit = pci_lookup.find(p);
                if (lit != pci_lookup.end()) {
                    pr.wall_ns = lit->second->wall_ns;
                    if (pr.wall_ns != 0) {
                        pr.busy_frac = static_cast<double>(pr.busy_ns) / static_cast<double>(pr.wall_ns);
                    }
                }
                row.pci_busy.push_back(std::move(pr));
            }
            v.windows.push_back(std::move(row));
        }
    } else {
        for (const auto & w : m.map) {
            verdict::window_row row;
            row.window = w.window;
            row.pci = w.pci;
            v.windows.push_back(std::move(row));
        }
    }
    // Shared mode: per-provider rows under a deterministic pointer-sorted
    // ordinal. The ordinal is a labeling index into the emitted PROVIDER
    // table; it is never a window claim and implies nothing about
    // ordering, priority or creation order.
    if (v.attribution == verdict::mode::shared) {
        unsigned idx = 0;
        for (const auto & pp : ppci) {
            verdict::provider_row row;
            row.index = idx++;
            row.provider = pp.first;
            for (const auto & cp : pp.second) {
                verdict::provider_pci_row pr;
                pr.pci = cp.first;
                const auto & ivs = cp.second;
                pr.spans = ivs.size();
                unsigned long long wall_b = ivs.front().first;
                unsigned long long wall_e = ivs.front().second;
                for (const auto & iv : ivs) {
                    wall_b = iv.first < wall_b ? iv.first : wall_b;
                    wall_e = iv.second > wall_e ? iv.second : wall_e;
                }
                pr.wall_ns = wall_e - wall_b;
                if (pr.wall_ns == 0) {
                    throw std::runtime_error("degenerate evidence: provider " + pp.first + " pci " +
                                             cp.first + " has wall_ns == 0");
                }
                pr.busy_ns = detail::union_measure(detail::merge_intervals(ivs));
                pr.busy_frac = static_cast<double>(pr.busy_ns) / static_cast<double>(pr.wall_ns);
                row.pci_busy.push_back(std::move(pr));
            }
            v.providers.push_back(std::move(row));
        }
    }
    // Same-PCI overlap of two attributed span sets — the original
    // definition, unchanged: merge each side's spans on ONE pci (one
    // device clock, shared per-device epoch enforced above) and measure
    // the intersection. Window-attributed modes pair windows over their
    // mapped PCI sets: those are the ONLY rows on the original
    // _JOINED_PAIR tag and the ONLY thing pair_scope=same_pci_shared_epoch
    // ever describes. Shared mode emits its cross-provider pairs on a
    // SEPARATE tag (_JOINED_PROVIDER_PAIR) with the provider pointers
    // inline: the same single-clock math, but provider-scoped — it never
    // claims a window and never satisfies a window-overlap gate.
    if (v.attribution == verdict::mode::shared) {
        std::map<std::string, std::vector<std::pair<const std::string *, const detail::intervals *>>> providers_by_pci;
        for (const auto & pp : ppci) {
            for (const auto & cp : pp.second) {
                providers_by_pci[cp.first].push_back({&pp.first, &cp.second});
            }
        }
        for (const auto & bp : providers_by_pci) {
            const auto & lst = bp.second;  // ascending provider ordinal
            for (size_t i = 0; i < lst.size(); ++i) {
                for (size_t j = i + 1; j < lst.size(); ++j) {
                    // ppci keys are distinct providers, so i != j are
                    // distinct providers by construction; a provider is
                    // never paired with itself.
                    verdict::provider_pair_row pr;
                    pr.provider_a = *lst[i].first;
                    pr.provider_b = *lst[j].first;
                    pr.pci = bp.first;
                    pr.spans_a = lst[i].second->size();
                    pr.spans_b = lst[j].second->size();
                    pr.overlap_ns = detail::intersect_measure(detail::merge_intervals(*lst[i].second),
                                                              detail::merge_intervals(*lst[j].second));
                    v.provider_pairs.push_back(std::move(pr));
                }
            }
        }
    } else {
        for (size_t i = 0; i < m.map.size(); ++i) {
            for (size_t j = i + 1; j < m.map.size(); ++j) {
                const map_rec & wa = m.map[i];
                const map_rec & wb = m.map[j];
                for (const auto & pa : wa.pci) {
                    for (const auto & pb : wb.pci) {
                        if (pa != pb) {
                            continue;
                        }
                        verdict::pair_pci_row pr;
                        pr.a = wa.window;
                        pr.b = wb.window;
                        pr.pci = pa;
                        const auto ia = wpci[wa.window].find(pa);
                        const auto ib = wpci[wb.window].find(pb);
                        detail::intervals va, vb;
                        if (ia != wpci[wa.window].end()) {
                            pr.spans_a = ia->second.size();
                            va = detail::merge_intervals(ia->second);
                        }
                        if (ib != wpci[wb.window].end()) {
                            pr.spans_b = ib->second.size();
                            vb = detail::merge_intervals(ib->second);
                        }
                        pr.overlap_ns = detail::intersect_measure(va, vb);
                        v.pairs.push_back(std::move(pr));
                    }
                }
            }
        }
    }
    return v;
}

// Prints the verdict in the machine-readable TP5_DW_HIP_OVERLAP_JOINED*
// format. Returns the number of lines printed. Every numeric field is a
// single-device-clock quantity. PAIR lines exist only when at least one
// same-PCI pair was measured; with none, pair_scope=not_computable and
// no pair of any kind is claimed.
inline unsigned long long print_verdict(const verdict & v, std::FILE * out) {
    unsigned long long n = 0;
    const char * mode = v.attribution == verdict::mode::pointer ? "pointer"
                      : v.attribution == verdict::mode::provider ? "provider" : "shared";
    // pair_scope labels the family of the emitted rows: _JOINED_PAIR rows
    // (window-attributed, pointer/provider mode only) are
    // same_pci_shared_epoch; _JOINED_PROVIDER_PAIR rows (shared mode) are
    // cross_provider_unattributed — provider-scoped diagnostics that
    // never claim a window and never satisfy a window-overlap gate; with
    // no rows of either kind nothing is claimed (not_computable).
    const char * pair_scope = !v.pairs.empty() ? "same_pci_shared_epoch"
                            : !v.provider_pairs.empty() ? "cross_provider_unattributed"
                            : "not_computable";
    std::fprintf(out, "TP5_DW_HIP_OVERLAP_JOINED mode=%s windows=%zu pcis=%zu pair_scope=%s scope=device_stream clock=hip_event device_only=1\n",
                 mode, v.windows.size(), v.pcis.size(), pair_scope);
    ++n;
    for (const auto & w : v.windows) {
        std::fprintf(out, "TP5_DW_HIP_OVERLAP_JOINED_WINDOW window=%u pci_count=%zu\n",
                     w.window, w.pci.size());
        ++n;
        for (const auto & pr : w.pci_busy) {
            std::fprintf(out, "TP5_DW_HIP_OVERLAP_JOINED_WINDOW_PCI window=%u pci=%s spans=%llu busy_ns=%llu wall_ns=%llu busy_frac=%.6f\n",
                         w.window, pr.pci.c_str(), pr.spans, pr.busy_ns, pr.wall_ns, pr.busy_frac);
            ++n;
        }
    }
    for (const auto & pr : v.providers) {
        std::fprintf(out, "TP5_DW_HIP_OVERLAP_JOINED_PROVIDER index=%u provider=0x%s pcis=%zu\n",
                     pr.index, pr.provider.c_str(), pr.pci_busy.size());
        ++n;
        for (const auto & cr : pr.pci_busy) {
            std::fprintf(out, "TP5_DW_HIP_OVERLAP_JOINED_PROVIDER_PCI provider_index=%u pci=%s spans=%llu busy_ns=%llu wall_ns=%llu busy_frac=%.6f\n",
                         pr.index, cr.pci.c_str(), cr.spans, cr.busy_ns, cr.wall_ns, cr.busy_frac);
            ++n;
        }
    }
    for (const auto & p : v.pairs) {
        std::fprintf(out, "TP5_DW_HIP_OVERLAP_JOINED_PAIR a=%u b=%u pci=%s spans_a=%llu spans_b=%llu overlap_ns=%llu\n",
                     p.a, p.b, p.pci.c_str(), p.spans_a, p.spans_b, p.overlap_ns);
        ++n;
    }
    // Provider-scoped diagnostics (shared mode ONLY): deliberately NOT on
    // the _JOINED_PAIR tag and NOT under pair_scope=same_pci_shared_epoch,
    // so a window-overlap gate cannot read them as window pairs. The two
    // providers are identified by their canonical pointers, inline.
    for (const auto & p : v.provider_pairs) {
        std::fprintf(out, "TP5_DW_HIP_OVERLAP_JOINED_PROVIDER_PAIR provider_a=0x%s provider_b=0x%s pci=%s spans_a=%llu spans_b=%llu overlap_ns=%llu\n",
                     p.provider_a.c_str(), p.provider_b.c_str(), p.pci.c_str(),
                     p.spans_a, p.spans_b, p.overlap_ns);
        ++n;
    }
    for (const auto & c : v.pcis) {
        std::fprintf(out, "TP5_DW_HIP_OVERLAP_JOINED_PCI pci=%s spans=%llu busy_ns=%llu wall_ns=%llu busy_frac=%.6f\n",
                     c.pci.c_str(), c.spans, c.busy_ns, c.wall_ns, c.busy_frac);
        ++n;
    }
    return n;
}

}  // namespace dw_hip_join
