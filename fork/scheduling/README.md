# TQScheduling

This C++17 library provides independent admission and Prefill-selection objects. It has no model, GPU, network, or sibling-library dependency. C composes their public value contracts with the model engine.

## Admission budgets

`<tq/scheduling/admission.h>` provides `admission_budget`. It accounts for a fixed token capacity and a fixed number of resident request slots. It does not allocate model memory, measure device memory, evict a cache, or select a prompt.

Reserve a complete request's upper token bound before starting its model work. A causal text runner that emits at most `N` tokens needs `prompt_tokens + N - 1` input rows. A different consumer must calculate the bound for its own protocol.

```cpp
#include <tq/scheduling/admission.h>

int main() {
    tq::scheduling::admission_budget budget(8192, 4);
    auto held = budget.try_acquire(42, 2048);
    if (!held) { return 1; }
    // A model consumer must release its state before this call.
    return budget.release(*held) ? 0 : 2;
}
```

`try_acquire` returns no reservation when tokens or slots are currently full. Zero demand, demand above total capacity, and an already active request ID throw `std::invalid_argument`. Request IDs may use the complete `uint64_t` range. All failures preserve accounting.

Each reservation has an owner identity and a monotonically increasing generation. A stale copy cannot release a new occupant of the same slot. A reservation from another budget cannot release a matching slot. Destruction and copying do not release capacity: release is explicit because the budget cannot know when the consumer has finished with model state. Destroyed owners are not confused with later instances. Generation exhaustion throws before state changes.

Serialize calls on one budget. Independent budgets share no mutable state. The budget cannot be copied or moved; reservation values can be copied or moved. A moved-from reservation is rejected by `release`.

The installed `TQScheduling::scheduling` target provides both admission and Prefill selection. These objects are independent: C decides when a request is eligible, acquires its reservation, and then gives active request snapshots to the selector. The batch runner is a real consumer with FIFO admission, fixed resident slots, and explicit model-state release.

## Prefill selection

`prefill_selector` selects the next prompt or generation work from an ordered slot snapshot. It owns the id of the prompt it is currently serving. IDs identify the caller's requests; they need not be model-engine sequence indices.

An unfinished prompt keeps its turn. Otherwise the first prompt in slot order
wins. Generation resumes when no prompt remains. This is the old fork's policy:
it is not FIFO, and a stream of new prompts can starve generation.

```sh
cmake -S fork/scheduling -B build-scheduling -DCMAKE_INSTALL_PREFIX="$PWD/install-scheduling"
cmake --build build-scheduling
ctest --test-dir build-scheduling --output-on-failure
cmake --install build-scheduling
```

A consumer uses `find_package(TQScheduling CONFIG REQUIRED)` and links
`TQScheduling::scheduling`. The selector header is `tq/scheduling/prefill.h`.
`choose()` reads a value snapshot, returns owned ids, and retains no reference
to either. Duplicate ids are rejected before selector state changes.

`fork/batch-compose` is a separate consumer that sends the selected work to a
real installed llama library. Installing this selector alone does not alter
the upstream server's schedule or memory use.
