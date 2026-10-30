#include <tq/relay/ledger.h>

#include <stdexcept>

int main() {
    using namespace tq::relay;
    channel_ledger ledger(1, 2, 3, 1, 1);
    const auto held = ledger.reserve(4, 1, 5, 1);
    if (held.code != reserve_code::accepted || !held.granted || ledger.active_bank_count() != 1) {
        throw std::runtime_error("installed relay reservation failed");
    }
    const auto & token = *held.granted;
    if (ledger.producer_ready(token, 0) != ledger_result::ok ||
        ledger.reduce_done(token) != ledger_result::ok ||
        ledger.output_published(token) != ledger_result::ok ||
        ledger.reserve(6, 1, 5, 1).code != reserve_code::wait_for_capacity ||
        ledger.consumer_done(token, 0) != ledger_result::ok ||
        ledger.reserve(6, 1, 5, 1).code != reserve_code::accepted) {
        throw std::runtime_error("installed relay bank lifecycle failed");
    }
}
