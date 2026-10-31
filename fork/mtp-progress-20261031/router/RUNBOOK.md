# Deferred CPU-only verification

**Do not execute while the current prohibition on compilation, tests, and syntax checks remains in force. GPU/HIP use is prohibited.** The commands below have not been executed for the candidate. Do not modify selected installations. Stop on build-slot exit 75 or 124.

The source tree is `/home/kunweiz/work/router-m2-alloc-scratch-20261030` at baseline `f1c5fafc0449b22969ddabfcc9c361940d94d033` plus `patch.diff`. CPU CMake, C++ compiler, make, and the repository's normal build dependencies are required. Future CPU-only configure and build, after explicit authorization:

```sh
src=/home/kunweiz/work/router-m2-alloc-scratch-20261030
build=/home/kunweiz/work/lead-router-m2-20261030/build-cpu
cmake -S "$src" -B "$build" -DGGML_CUDA=OFF -DGGML_HIP=OFF -DLLAMA_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
/home/kunweiz/.local/bin/build-slot cmake --build "$build" -j2
# On 75 or 124 stop immediately; do not retry.
```

For a CPU operator smoke, after successful build and only if `test-backend-ops` exists in `build-cpu/bin`:

```sh
/home/kunweiz/work/lead-router-m2-20261030/build-cpu/bin/test-backend-ops test -b CPU -o MUL_MAT_ID
```

The source `tests/test-backend-ops.cpp` contains the individual operator tests. Their CLI spelling and candidate binary have not been verified here; confirm the built binary's help when permission is granted. An individual operator pass is not an integrated router test.

The existing CPU router harness can be rebuilt against the candidate libraries for a **generic graph** smoke (it does not set the new retention flag and does not execute the Qwen4Exp model consumer):

```sh
cd /home/kunweiz/work/lead-router-m2-20261030
/home/kunweiz/.local/bin/build-slot c++ -std=c++17 -O2 \
  -I/home/kunweiz/work/router-m2-alloc-scratch-20261030/ggml/include router_cpu_reference.cpp \
  -L/home/kunweiz/work/lead-router-m2-20261030/build-cpu/bin \
  -Wl,-rpath,/home/kunweiz/work/lead-router-m2-20261030/build-cpu/bin \
  -lggml -lggml-cpu -lggml-base -o router_cpu_reference_candidate
./router_cpu_reference_candidate
```

These commands do not constitute a candidate acceptance test: a further real Qwen4Exp model-consumer harness must verify the actual opt-in path, both IDs and weights, true allocation extents, N1/N2/N3 row references and OFF-versus-pristine-stock outputs. Cross-warp and real fusion cannot be established with CPU-only tests, and must remain unverified while GPU is forbidden. The unrelated historical tiny model under `prefab-share-live-r8-080352` has two experts rather than 512 and cannot activate this candidate gate.
