#pragma once
#include <stdint.h>
#include <stdbool.h>

#if defined(__VXE__) || defined(__VXE2__)
#include <vecintrin.h>
#endif

#ifdef _MSC_VER
#define NOINLINE __declspec(noinline)
#else
#define NOINLINE __attribute__((__noinline__))
#endif

#ifdef __cplusplus
extern "C" {
#endif

bool llamafile_sgemm(const struct ggml_compute_params * params, int64_t, int64_t, int64_t,
                     const void *, int64_t, const void *, int64_t, void *, int64_t,
                     int, int, int);

// Reproduce a padded reduction without reading omitted K/V elements.
bool llamafile_sgemm_masked_tail(const struct ggml_compute_params * params,
                                int64_t m, int64_t n, int64_t padded_k, int64_t live_k,
                                const void * a, int64_t lda, const void * b, int64_t ldb,
                                void * c, int64_t ldc, int type);

#ifdef __cplusplus
}
#endif
