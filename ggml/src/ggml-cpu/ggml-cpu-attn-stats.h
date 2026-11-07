#pragma once

//
// ggml-cpu-attn-stats.h
//
// The CPU attention work counters, shared by the kernels that measure the work
// (ops.cpp soft_max_ext, ggml-cpu.c mul_mat) and the backend object that owns
// them (ggml-cpu.cpp ggml_backend_cpu_context).
//
// Why this is a header and not a field on ggml_compute_params: the params
// struct is built per thread from the plan and has no per-context identity,
// and ggml_threadpool is shared and outlives a context. The only per-context
// CPU object is ggml_backend_cpu_context, so the counters live there and the
// kernels reach them through the opaque `cpu_ctx` pointer the plan carries.
//
// This header is included from both C and C++ translation units (ggml-cpu.c
// is C), so it is plain C: a struct of three plain uint64_t and a helper that
// takes them. The atomics that make it thread-safe live in the owner
// (ggml-cpu.cpp), which is C++.
//
// The counters are MEASURED, never derived: the kernels add the loop bound
// they actually executed, so a trim that is not executed does not count and a
// caller cannot satisfy them by writing an op parameter.
//
//   live_elems : elements the measured attention reductions processed (the
//                loop bound the kernel executed)
//   pad_elems  : the extent those same rows cover. A TRUE_LEN_REDUCE node's
//                tensor is shaped at the padded extent, so its ne00 is the
//                value. An ordered attention node runs on a tensor the bind
//                patched to the runtime width, so its extent IS the live
//                length and live == pad: the narrowed mask/K/V removed the
//                padded work instead of skipping it.
//   rows       : executed runtime-length reduction rows
//
// `live_elems == pad_elems` means no padded work ran: either the ordered
// lane (narrowed tensors) or an unbound graph. `live_elems < pad_elems`
// means a TRUE_LEN_REDUCE node reduced fewer elements than its tensor's
// extent. `rows * runtime_n_kv == live_elems` is the self-consistency
// identity a test checks. Row counting is restricted to runtime-length
// reductions: the generic mul_mat counts a node only when it carries
// TRUE_LEN_REDUCE, and the ordered softmax counts the rows it executes. An
// unflagged mul_mat (the LM head, the projections) is not counted at all. A
// generic unflagged softmax still adds its elements to live and pad but no
// row; on the ordered n_kv lane the only softmax is the ordered attention
// one, so it does not pollute the identity there.
//

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_cpu_attn_stats {
    uint64_t live_elems;
    uint64_t pad_elems;
    uint64_t rows;
};

// Accumulate one chunk of measured work. The owner supplies the thread-safe
// implementation; the kernels never touch the storage directly.
void ggml_cpu_attn_stats_add(struct ggml_cpu_attn_stats * stats,
                             uint64_t live, uint64_t padded, uint64_t n_rows);

#ifdef __cplusplus
}
#endif
