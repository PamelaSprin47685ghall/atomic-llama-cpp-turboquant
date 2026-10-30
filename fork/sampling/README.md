# TQSampling

TQSampling is library B. It selects a small, owned set of logit values. It has
no llama or ggml dependency, model state, random generator, or sampler callback.

The bounded heap and AVX2 scan come from the fork's
`common/sampling-unique-argmax.h`. The public interface uses ordinary values so
the library can build and run without the upstream tree.

## Contract

`tq::sampling::select(logits, count, k)` returns `min(count, k)` candidates in
descending logit order. It accepts `2 <= k <= 128` and a unique finite maximum.
Negative infinity can mask tokens. NaN, positive infinity, an all-masked row,
a maximum tie, and invalid arguments return an empty result with `usable=false`.
The result owns its candidate values. It does not keep a pointer to the input.

The order of tokens with equal logits below the maximum is not specified. The
composition layer may use this result only when top-k is the first active filter
and the remaining chain has proved that the compact row preserves its behavior.
The library does not make this decision. In the current composition, only the
static greedy chain can use the result. Positive temperatures use the full row.

Preserve every returned candidate, including masks, and run the real sampler
chain. Replacing that chain with an argmax can change random-state consumption
when filters reduce the candidate count to one.

## Build and test

```sh
cmake -S fork/sampling -B build-tq-sampling -DCMAKE_BUILD_TYPE=Release
cmake --build build-tq-sampling --parallel
ctest --test-dir build-tq-sampling --output-on-failure
cmake --install build-tq-sampling --prefix "$PWD/build-tq-sampling/install"
```

Use `-DTQ_SAMPLING_SCALAR_ONLY=ON` in a separate build directory to test the
portable scan. The normal build checks CPU support before calling AVX2 code.
Neither build requires an upstream source directory.

A separate consumer can load the installed package:

```cmake
find_package(TQSampling CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE TQSampling::sampling)
```
