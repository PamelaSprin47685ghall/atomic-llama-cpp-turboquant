# TQScheduling

This C++17 library selects the next prompt or generation work from an ordered
slot snapshot. It has no model, GPU, llama, network, or sibling-library dependency.
It owns one value: the id of the prompt it is currently serving.

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
`TQScheduling::scheduling`. The public header is `tq/scheduling/prefill.h`.
`choose()` reads a value snapshot, returns owned ids, and retains no reference
to either. Duplicate ids are rejected before selector state changes.

`fork/batch-compose` is a separate consumer that sends the selected work to a
real installed llama library. Installing this selector alone does not alter
the upstream server's schedule or memory use.
