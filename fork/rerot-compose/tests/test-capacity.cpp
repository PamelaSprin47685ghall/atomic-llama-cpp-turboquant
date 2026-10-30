#include "capacity.h"

#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

void require(bool valid, const char * message) {
    if (!valid) { throw std::runtime_error(message); }
}

template <typename Error, typename Operation>
void rejects(Operation operation, const char * message) {
    try { operation(); }
    catch (const Error &) { return; }
    throw std::runtime_error(message);
}

} // namespace

int main() {
    try {
        tq::rerot_compose::episode_capacity budget(3);
        // Three base keys; two workers publish 2+4 and 1+3 keys; synthesis
        // publishes 2+5. There are 20 public keys and three private ends.
        budget.add_stage(2, 4);
        budget.add_stage(1, 3);
        budget.add_stage(2, 5);
        require(budget.physical_tokens() == 23, "lost a physically owned private end");
        require(budget.reader_tokens() == 21, "peer private ends entered the reader budget");
        budget.require(23, 21);
        rejects<std::invalid_argument>([&] { budget.require(22, 21); },
                                       "accepted a short physical pool");
        rejects<std::invalid_argument>([&] { budget.require(23, 20); },
                                       "accepted a short reader context");
        rejects<std::invalid_argument>([&] { budget.require(21, 23); },
                                       "exchanged physical and logical capacity");

        const auto largest = std::numeric_limits<std::uint64_t>::max();
        tq::rerot_compose::episode_capacity full(largest - 1);
        full.add_stage(0, 0);
        require(full.physical_tokens() == largest && full.reader_tokens() == largest,
                "rejected the largest representable budget");
        rejects<std::overflow_error>([&] { full.add_stage(0, 0); },
                                    "private ends overflowed silently");
        require(full.physical_tokens() == largest && full.reader_tokens() == largest,
                "a rejected update changed the budget");
        rejects<std::overflow_error>([&] { full.add_stage(largest, 1); },
                                    "public keys overflowed silently");
        require(full.physical_tokens() == largest, "overflow damaged prior capacity");
        std::cout << "PASS: independent physical/reader limits and atomic overflow rejection\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
