#include "linalg.h"
#include "parallel.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <thread>

namespace wxq {

void gemm_AB(float * C, const float * A, const float * B, int64_t M, int64_t K, int64_t N) {
    parallel_for(M, [=](int64_t t, int64_t step) {
        for (int64_t i = t; i < M; i += step) {
            float * Ci = C + i*N;
            std::memset(Ci, 0, (size_t) N * sizeof(float));
            for (int64_t k = 0; k < K; ++k) {
                const float a = A[i*K + k];
                if (a == 0.0f) {
                    continue;
                }
                const float * Bk = B + k*N;
                for (int64_t j = 0; j < N; ++j) {
                    Ci[j] += a * Bk[j];
                }
            }
        }
    });
}

void gemm_ABt(float * C, const float * A, const float * B, int64_t M, int64_t K, int64_t N) {
    parallel_for(M, [=](int64_t t, int64_t step) {
        for (int64_t i = t; i < M; i += step) {
            const float * Ai = A + i*K;
            for (int64_t j = 0; j < N; ++j) {
                const float * Bj = B + j*K;
                double s = 0.0;
                for (int64_t k = 0; k < K; ++k) {
                    s += (double) Ai[k] * (double) Bj[k];
                }
                C[i*N + j] = (float) s;
            }
        }
    });
}

void syrk_AtA(float * C, const float * A, int64_t M, int64_t K) {
    parallel_for(K, [=](int64_t t, int64_t step) {
        for (int64_t i = t; i < K; i += step) {
            for (int64_t j = i; j < K; ++j) {
                double s = 0.0;
                for (int64_t m = 0; m < M; ++m) {
                    s += (double) A[m*K + i] * (double) A[m*K + j];
                }
                C[i*K + j] = (float) s;
            }
        }
    });
    for (int64_t i = 0; i < K; ++i) {
        for (int64_t j = 0; j < i; ++j) {
            C[i*K + j] = C[j*K + i];
        }
    }
}

void accum_cov(double * A, const float * X, size_t ld, int64_t n, int64_t d, int nth) {
    auto worker = [=](int t) {
        for (int64_t i = t; i < d; i += nth) {
            double * Ai = A + (size_t) i * (size_t) d;
            for (int64_t k = 0; k < n; ++k) {
                const float * xk = X + (size_t) k * ld;
                const double  xi = (double) xk[i];
                if (xi == 0.0) {
                    continue;
                }
                for (int64_t j = i; j < d; ++j) {
                    Ai[j] += xi * (double) xk[j];
                }
            }
        }
    };
    if (nth <= 1) {
        worker(0);
        return;
    }
    std::vector<std::thread> pool;
    pool.reserve(nth);
    for (int t = 0; t < nth; ++t) {
        pool.emplace_back(worker, t);
    }
    for (auto & th : pool) {
        th.join();
    }
}

void eig_sym(std::vector<double> & a, int64_t n, std::vector<double> & ev) {
    for (int sweep = 0; sweep < 30; ++sweep) {
        double off = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            for (int64_t j = i+1; j < n; ++j) {
                off += a[i*n+j]*a[i*n+j];
            }
        }
        if (off <= 1e-22) {
            break;
        }
        for (int64_t p = 0; p < n-1; ++p) {
            for (int64_t q = p+1; q < n; ++q) {
                const double apq = a[p*n+q];
                if (std::fabs(apq) < 1e-18) {
                    continue;
                }
                const double app = a[p*n+p], aqq = a[q*n+q];
                const double theta = 0.5*(aqq - app)/apq;
                const double tt = (theta >= 0.0 ? 1.0 : -1.0)/(std::fabs(theta) + std::sqrt(theta*theta + 1.0));
                const double c = 1.0/std::sqrt(tt*tt + 1.0), s = tt*c;
                for (int64_t k = 0; k < n; ++k) {
                    const double akp = a[k*n+p], akq = a[k*n+q];
                    a[k*n+p] = c*akp - s*akq;
                    a[k*n+q] = s*akp + c*akq;
                }
                for (int64_t k = 0; k < n; ++k) {
                    const double apk = a[p*n+k], aqk = a[q*n+k];
                    a[p*n+k] = c*apk - s*aqk;
                    a[q*n+k] = s*apk + c*aqk;
                }
            }
        }
    }
    ev.resize(n);
    for (int64_t i = 0; i < n; ++i) {
        ev[i] = a[i*n+i];
    }
    std::sort(ev.begin(), ev.end(), std::greater<double>());
}

bool chol_upper(std::vector<double> & a, int64_t d) {
    for (int64_t k = 0; k < d; ++k) {
        double * ak = a.data() + k*d;
        if (!(ak[k] > 0.0)) {
            return false;
        }
        const double inv = 1.0/std::sqrt(ak[k]);
        for (int64_t j = k; j < d; ++j) {
            ak[j] *= inv;
        }
        parallel_for(d, [&, k](int64_t t, int64_t step) {
            for (int64_t i = k+1+t; i < d; i += step) {
                double * ai = a.data() + i*d;
                const double aki = ak[i];
                if (aki == 0.0) {
                    continue;
                }
                for (int64_t j = i; j < d; ++j) {
                    ai[j] -= aki*ak[j];
                }
            }
        });
    }
    return true;
}

} // namespace wxq
