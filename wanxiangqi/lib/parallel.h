// Strided thread fan-out for the offline probes.
//
// Every body is called as body(t, step) and is expected to walk t, t+step,
// t+2*step, ... The stride, rather than a contiguous split, is what makes
// triangular work (row i touches only columns j >= i) balance across threads.
//
// These tools are offline analysis binaries with one hot loop at a time, so a
// process-wide thread count set once from argv is the right granularity. There is
// no nesting and no pool: each call spawns and joins.

#pragma once

#include <cstdint>
#include <functional>

namespace wxq {

void set_num_threads(int n);
int  num_threads();

void parallel_for(int64_t n, const std::function<void(int64_t t, int64_t step)> & body);

} // namespace wxq
