// llama-flashprefill-state.h — versioned binary policy envelope for FlashPrefill KV state.
//
// A framed identity header written ahead of the legacy KV bytes whenever the
// FlashPrefill policy is enabled, for both full-context state
// (state_write_data) and per-sequence state (state_seq_write_data). OFF writes
// unchanged legacy bytes (no envelope) and reads the unchanged legacy path.
//
// On-wire framing (total kFramedBytes = 4 + marker + 64):
//   [u32 LE marker length][marker bytes][fixed 64-byte body]
// The length-prefixed marker comes FIRST for one reason: the legacy full-state
// reader opens with read_string, which allocates vector<char>(len) BEFORE any
// bounds check. A raw 64-byte body starting with magic 0x46505330 would be
// misread as a ~1.18 GB string length (allocation bomb / OOM). With framing, an
// OFF/old reader's read_string instead gets a SMALL length, allocates a few
// bytes, reads the distinct marker, and fails cleanly at its arch-string
// mismatch — no unbounded allocation, OFF read code byte-identical.
// Enabled readers never use read_string: they take one bounded fixed-size
// read and validate length + marker bytes + body in memory (see
// decode_framed_envelope), rejecting legacy/raw/foreign bytes before any KV
// mutation. There is deliberately NO compatibility shim for the unreleased raw
// 64-byte prefix format (never deployed): those bytes fail the marker-length
// check and are rejected as foreign.
//
// What the envelope carries (identity only, bounded fixed-size fields):
//   - algorithm/schema versions (LLAMA_FLASHPREFILL_VERSION / CONFIG_VERSION)
//   - scope discriminator (full vs seq; cross-scope reuse is rejected)
//   - config policy fingerprint (every approximation field + model.desc +
//     empty adapter domain, via llama_flashprefill_fingerprint)
//   - 128-bit process nonce (fixed random bytes per process; see below)
//   - per-context serial (unique within the process, nonzero only when enabled)
//   - adapter generation (bumped on every effective LoRA/cvec mutation)
//
// What it deliberately does NOT carry: no pooled means, no plans, no GPU
// pointers, no KV contents, no prompt text, no sequence/reader ids. Derived
// GPU work is rebuilt per graph after restore, never persisted.
//
// Identity semantics (conservative, documented):
//   - model.desc() is an arch/type/params string, NOT a weight hash. It
//     cannot prove weights, and a bare per-process counter restarts at 1 in
//     every process — so the counter alone can NEVER authorize a restore.
//     The 128-bit process nonce (random per process, fixed for its lifetime)
//     makes cross-process/cross-restart collisions probabilistically
//     impossible: same nonce + same serial == same live context in this
//     process (within-context RAM/checkpoint round trips allowed); anything
//     else is rejected (durable same-identity reuse stays rejected until an
//     explicitly verifiable model/adapter identity exists — none today).
//     The nonce is a probabilistic anti-collision value, clearly labeled as
//     such: NOT a secret, NOT a security boundary, just unguessable restart
//     discrimination. Compared exact, like every other field.
//   - Adapter set/remove/scale/cvec mutations bump the generation, so blobs
//     saved under older adapters never validate afterwards. FlashPrefill
//     never deletes KV: resident history keeps baseline semantics (the server
//     owns per-slot LoRA/prompt compatibility); isolation comes from the
//     retired generation plus graph/scheduler re-reservation, with derived
//     means rebuilt per graph. A saturated (UINT64_MAX) generation is
//     always-invalid, like CellGeneration.
//   - The original Tri/RERoT fingerprints are preserved verbatim: this
//     envelope wraps OUTSIDE them (written first, validated first), and the
//     legacy arch/model and RERoT magic/version/caps/triple checks run
//     unchanged afterwards.
//
// Read discipline: validate the envelope BEFORE consuming any KV bytes or
// mutating context/memory state. Missing header (legacy OFF bytes fed to an
// enabled reader), unknown version/scope, or any identity mismatch throws
// (caller converts to a 0 return); the caller must re-prefill under the
// current policy instead of reusing the state.
//
// This header is pure: no llama.h / llama-io.h / KV dependency, no
// allocation on any encode/decode/match/cache_key path (describe() returns a
// string). Context integration serializes via one kFramedBytes stack buffer
// plus a single io.write / io.read call per envelope.

#pragma once

#include "../include/llama-flashprefill.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>

namespace llama_flashprefill_state {

// Fixed envelope identity. 'FPS0' avoids io_magic (0xaf143cd8), the RERoT
// magic ('RROT' 0x52524f54), and the session/seq file magics.
constexpr uint32_t kMagic   = 0x46505330u;
constexpr uint32_t kVersion = 1u;

constexpr uint32_t kScopeFull = 0u;
constexpr uint32_t kScopeSeq  = 1u;

// Exact on-wire byte count: 6 x u32 + 5 x u64. Fixed at v1; a version bump
// must define a new size, never reinterpret these bytes.
constexpr size_t kEnvelopeBytes = 64u;

struct envelope {
    uint32_t magic            = kMagic;
    uint32_t version          = kVersion;
    uint32_t scope            = kScopeFull;
    uint32_t fp_version       = LLAMA_FLASHPREFILL_VERSION;
    uint32_t config_version   = LLAMA_FLASHPREFILL_CONFIG_VERSION;
    uint32_t reserved         = 0;
    uint64_t policy_fp        = 0;
    uint64_t proc_nonce0      = 0; // process nonce, word 0 (exact-compared)
    uint64_t proc_nonce1      = 0; // process nonce, word 1 (exact-compared)
    uint64_t context_serial   = 0;
    uint64_t adapter_gen      = 0;
};

constexpr size_t envelope_nbytes() { return kEnvelopeBytes; }

// Framing marker: short, nonzero, arch-incompatible (uppercase + FP prefix;
// no llm_arch_name is uppercase or carries this prefix, so an OFF reader's
// read_string yields it verbatim and fails cleanly at its arch mismatch with
// only a tiny allocation). The LENGTH word doubles as the seq-path tripwire:
// OFF memory readers see it as their first count word (see the reader matrix
// in llama-context.cpp), and the distinct outer seq magics reject enabled
// blobs before any memory reader runs — the marker is defense in depth there.
constexpr std::string_view kFrameMarker = "FLASHPREFILL_STATE_V1";

// Total framed on-wire bytes: u32 LE marker length + marker + 64-byte body.
constexpr size_t kFramedBytes = 4u + kFrameMarker.size() + kEnvelopeBytes;

constexpr size_t framed_nbytes() { return kFramedBytes; }

// Framed codec over byte buffers (pure, bounded, allocation-free). The 64-byte
// body codec above is unchanged; these add/remove the length-prefixed marker.
// encode: single kFramedBytes write or fail (short buffer, or the same body
// validity bar as encode_envelope — never persists what load must reject).
// decode: fail on short input, marker-length mismatch (legacy OFF bytes,
// foreign state, or the unreleased raw-64B prefix whose magic reads as a
// ~1.18 GB length), marker-bytes mismatch, or body invalidity. First failure
// wins with a bounded diagnostic. Either function leaves its out-buffer
// untouched on failure (encode validates the body before storing anything).
bool encode_framed_envelope(const envelope & env, uint8_t * dst, size_t dst_size, size_t * out_n = nullptr);
bool decode_framed_envelope(const uint8_t * src, size_t src_size, envelope * out, size_t * out_n, std::string * error);

// Build the envelope a writer persists. Pure field fill; the caller supplies
// the already-computed policy fingerprint, the fixed process nonce words, the
// live context serial, and the current adapter generation. Fingerprint,
// nonce words, serial, and generation must all be nonzero; MAX adapter_gen is
// sticky-invalid and must never be written by a fresh save — match() rejects
// it either way.
envelope make_envelope(
        uint32_t scope, uint64_t policy_fp,
        uint64_t proc_nonce0, uint64_t proc_nonce1,
        uint64_t context_serial, uint64_t adapter_gen);

// Bounds-checked host-order codec over the fixed field order. encode() writes
// exactly kEnvelopeBytes or fails (NULL dst / short buffer). decode() fails
// on short buffers, magic mismatch ("missing policy header"), unknown
// version/scope, schema skew, or zero policy_fp/serial/adapter_gen. First
// failure wins; *error holds a bounded diagnostic (no prompt/KV content).
bool encode_envelope(const envelope & env, uint8_t * dst, size_t dst_size, size_t * out_n = nullptr);
bool decode_envelope(const uint8_t * src, size_t src_size, envelope * out, size_t * out_n, std::string * error);

// Fail-closed comparison against the live context's expectations. Checks, in
// order: scope, schema versions, policy fingerprint, context serial
// (same-context only), adapter generation (either side MAX == mismatch).
// Returns true on full match; false with *error naming the offending domain
// plus the re-prefill path. Pure: no IO, no state access.
bool match_envelope(
        const envelope & stored,
        uint32_t         expected_scope,
        uint64_t         expected_policy_fp,
        uint64_t         expected_nonce0,
        uint64_t         expected_nonce1,
        uint64_t         expected_serial,
        uint64_t         expected_adapter_gen,
        std::string *    error);

// Persistent cache-key getter (StatePolicy-owned, ServerRouting-called).
// FNV-1a 64 mix over a domain separator plus the full envelope identity
// (policy_fp, serial, adapter_gen). Same inputs == same key; any identity
// difference == different key (modulo 64-bit hash collisions, which are
// rejected authoritatively by match_envelope afterwards — the key is a
// fast-path stamp, the envelope is the authority).
//
// Process-local IN-MEMORY use only (RAM stamps): the serial (and the process
// nonce it stands with) make the key meaningless across processes. NEVER
// serialize this key as restore authority — no file sidecars, no on-disk fast
// paths. Durable authority is the envelope inside the state bytes, validated
// by match_envelope on every load.
uint64_t state_cache_key(uint64_t policy_fp, uint64_t context_serial, uint64_t adapter_gen);

// Bounded one-line diagnostic for logs ("FPS0 v1 scope=full policy=0x…aw…"
// style hex). Never includes prompt text, KV bytes, or ids beyond the three
// identity integers.
std::string describe_envelope(const envelope & env);

} // namespace llama_flashprefill_state
