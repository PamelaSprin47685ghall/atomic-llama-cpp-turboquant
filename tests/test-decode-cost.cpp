#include "llama-decode-cost.h"
#include <cassert>

static int64_t fake_now = 100;
static int calls = 0;
extern "C" int64_t ggml_time_us() { ++calls; return fake_now; }

int main(int argc, char **) {
    {
        llama_decode_cost c("fixture", nullptr, 1, 0);
        fake_now = 110; c.mark(0);
        fake_now = 120; c.mark(1);
        fake_now = 125; c.mark(2);
        fake_now = 127; c.mark(3);
        fake_now = 200; c.mark(4);
        fake_now = 210;
    }
    if (argc == 1) assert(calls == 0);
    else assert(calls == 7);
}
