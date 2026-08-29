// Dense linear algebra for the offline probes.
//
// Deliberately self-contained: these binaries answer structural questions about
// the weights (spectrum, rank, log-det) and adding a BLAS/LAPACK dependency to a
// llama.cpp fork to do it once per layer is not worth the build surface. Sizes
// are n_embd = 2048 and n_ff = 512 class, which cyclic Jacobi handles in seconds.
//
// Accumulation is double throughout. The quantities being measured -- tail
// eigenvalues, log-determinant ratios -- are exactly the ones f32 accumulation
// destroys.

#pragma once

#include <cstdint>
#include <vector>

namespace wxq {

// C[M,N] = A[M,K] * B[K,N]
void gemm_AB(float * C, const float * A, const float * B, int64_t M, int64_t K, int64_t N);

// C[M,N] = A[M,K] * B[N,K]^T
void gemm_ABt(float * C, const float * A, const float * B, int64_t M, int64_t K, int64_t N);

// C[K,K] = A[M,K]^T * A[M,K], computed on the upper triangle then mirrored
void syrk_AtA(float * C, const float * A, int64_t M, int64_t K);

// A[d,d] += X^T X over the upper triangle only; X is [n, d] row-major with row
// stride ld. Rows are handed out interleaved because row i touches only columns
// j >= i, so a contiguous split would leave the last thread with nothing to do.
// The lower half stays untouched -- mirror it at the consumer (write_calib does).
void accum_cov(double * A, const float * X, size_t ld, int64_t n, int64_t d, int nth);

// Cyclic Jacobi, eigenvalues only, returned descending. `a` is [n,n] row-major and
// is destroyed.
void eig_sym(std::vector<double> & a, int64_t n, std::vector<double> & ev);

// In-place Cholesky over the upper triangle of `a` [d,d]; false on a
// non-positive pivot, which is the caller's signal that the matrix is not
// positive definite (rank deficient calibration, most likely).
bool chol_upper(std::vector<double> & a, int64_t d);

} // namespace wxq
