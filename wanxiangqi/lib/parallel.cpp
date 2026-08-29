#include "parallel.h"

#include <thread>
#include <vector>

namespace wxq {

static int g_nth = 1;

void set_num_threads(int n) { g_nth = n > 0 ? n : 1; }
int  num_threads()         { return g_nth; }

void parallel_for(int64_t n, const std::function<void(int64_t, int64_t)> & body) {
    if (g_nth <= 1 || n <= 1) {
        body(0, 1);
        return;
    }
    std::vector<std::thread> pool;
    pool.reserve(g_nth);
    for (int t = 0; t < g_nth; ++t) {
        pool.emplace_back(body, (int64_t) t, (int64_t) g_nth);
    }
    for (auto & th : pool) {
        th.join();
    }
}

} // namespace wxq
