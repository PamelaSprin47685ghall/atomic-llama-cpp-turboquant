find_package(Threads REQUIRED)

llama_add_compile_flags()

function(llama_build source)
    set(TEST_SOURCES ${source} ${ARGN})

    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    target_link_libraries(${TEST_TARGET} PRIVATE llama llama-common)
    if (LLAMA_TESTS_INSTALL)
        install(TARGETS ${TEST_TARGET} RUNTIME)
    endif()
endfunction()

function(llama_test target)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_NAME ${LLAMA_TEST_NAME})
    else()
        set(TEST_NAME ${target})
    endif()

    set(TEST_TARGET ${target})

    add_test(
        NAME ${TEST_NAME}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_NAME} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

function(llama_test_cmd target)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_NAME ${LLAMA_TEST_NAME})
    else()
        set(TEST_NAME ${target})
    endif()

    add_test(
        NAME ${TEST_NAME}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND ${target}
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_NAME} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

# Builds and runs a test source file.
# Optional args:
# - NAME: name of the executable & test target (defaults to the source file name without extension)
# - LABEL: label for the test (defaults to main)
# - ARGS: arguments to pass to the test executable
# - WORKING_DIRECTORY
function(llama_build_and_test source)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(TEST_SOURCES ${source} ${LLAMA_TEST_UNPARSED_ARGUMENTS})

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    if (LLAMA_TESTS_INSTALL)
        install(TARGETS ${TEST_TARGET} RUNTIME)
    endif()
    target_link_libraries(${TEST_TARGET} PRIVATE llama-common)

    add_test(
        NAME ${TEST_TARGET}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_TARGET} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

# the GGML_TP5_* value contract (exactly "1") as pure functions: no model, no
# context, no backend, so it runs in every CPU build
add_executable(test-union-binder-gate test-union-binder-gate.cpp)
add_test(NAME test-union-binder-gate COMMAND test-union-binder-gate)

# The SET_ROWS symbolic shape contract (src/llama-union-shape-guard.h) as one
# pure predicate: the accepted KV-cache write plus one refusal per guard group.
# No model, no context, no backend, no library: the header is ggml-free.
add_executable(test-union-shape-guard test-union-shape-guard.cpp)
target_include_directories(test-union-shape-guard PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME test-union-shape-guard COMMAND test-union-shape-guard)

# Stable topology digest (src/llama-topology-hash.h): pointer-, order- and
# process-independent identity of a ggml graph. Builds its graphs from ggml ops
# in a no_alloc context, so it needs no model file, no backend and no GPU. It
# links ggml only, because the digest lives in a header that includes ggml.h.
add_executable(test-topology-hash test-topology-hash.cpp)
target_link_libraries(test-topology-hash PRIVATE ggml)
add_test(NAME test-topology-hash COMMAND test-topology-hash)

# Single graph cache key (src/llama-topology-hash.h): llama_topology_key,
# llama_topology_key_of and llama_topology_same. Same shape as the digest test
# above - ggml ops in a no_alloc context, no model, no backend - so it is
# always registered and links ggml only.
add_executable(test-topology-cache-key test-topology-cache-key.cpp)
target_link_libraries(test-topology-cache-key PRIVATE ggml)
add_test(NAME test-topology-cache-key COMMAND test-topology-cache-key)

# Narrow real-CPU receipt contract: a synchronous CPU backend must publish a
# completed-success receipt with a real attempt id; wait() must report the
# same completed fact. No model and no GPU.
add_executable(test-sched-receipt test-sched-receipt.cpp)
target_link_libraries(test-sched-receipt PRIVATE ggml ggml-cpu)
target_include_directories(test-sched-receipt PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)
add_test(NAME test-sched-receipt COMMAND test-sched-receipt)

# the two decision predicates of the rs_z runtime lane (arming, decode-time
# row check) as pure functions: no model, no context, no backend, so the
# n_rs_seq refusal and the _cell/_envelope split are pinned in every CPU build
add_executable(test-rsz-lane-rows test-rsz-lane-rows.cpp)
# llama-nout-state.h includes llama-topology-hash.h, which includes ggml.h.
# The test links no library, so it must name the ggml include directory.
target_include_directories(test-rsz-lane-rows PRIVATE ${PROJECT_SOURCE_DIR}/ggml/include)
add_test(NAME test-rsz-lane-rows COMMAND test-rsz-lane-rows)

# Requires a caller-supplied tiny LLAMA GGUF; see nkv-cpu-probe/README.md.
add_executable(test-varlen-output-probe nkv-cpu-probe/varlen_output_probe.cpp)
target_link_libraries(test-varlen-output-probe PRIVATE llama)
set(LLAMA_VARLEN_TEST_MODEL "" CACHE FILEPATH "Tiny LLAMA GGUF for paired variable-length tests")
add_executable(test-ntok-strict test-ntok-strict.cpp)
target_link_libraries(test-ntok-strict PRIVATE llama)
add_executable(test-nkv-exact-width test-nkv-exact-width.cpp)
target_link_libraries(test-nkv-exact-width PRIVATE llama)
if (LLAMA_VARLEN_TEST_MODEL)
    add_test(NAME test-nkv-exact-width COMMAND test-nkv-exact-width "${LLAMA_VARLEN_TEST_MODEL}")
    set_property(TEST test-nkv-exact-width PROPERTY LABELS "model;cpu;nkv")
    add_test(NAME test-ntok-strict COMMAND test-ntok-strict "${LLAMA_VARLEN_TEST_MODEL}")
endif()
# Pillar 1: n_kv, n_outputs, rs_z and n_tokens as runtime parameters of ONE
# graph.
#
# Two registrations, both from one source file:
#   - test-union-four-param-hash   needs NO model: it hashes synthetic graphs
#     built at every combination of the four runtime parameters
#     (6 n_kv x 4 n_outputs x 6 n_rs x 7 token widths = 1008 combinations,
#     from kv_v[], out_v[], nrs_v[] and tok_v[] in part1()), asserts the
#     width-table hash does not move, asserts the SINGLE GRAPH CACHE KEY
#     (llama-topology-hash.h) is stable across token widths only when the token
#     lane's exclusion is passed, and calls union_topology_key_excl() and
#     union_bound_ntok() - the functions the binder calls - so a regression in
#     the plumbing, not only in the exclusion, turns it red. It is always
#     registered, so the invariant is checked on every host, model or not.
#
#     It also carries the FALSIFIABILITY arm: the registration fails unless
#     FOUR_LEAK_NEG is ok=1, because an equality sweep that cannot fail proves
#     nothing. A token width that leaks into the hash must be refused (a mask
#     whose row count disagrees with the armed token width), and the width must
#     reach the hash when the exclusion is not passed.
#   - test-union-four-param-matrix needs a model and reads LLAMA_NTOK_TEST_MODEL.
#     It runs the four-parameter cross matrix (one hash, zero rebuilds on the
#     binder arm, at least one rebuild per cell on the stock arm, bitwise
#     equality against the stock arm) and the mask refusal case. The test itself
#     treats a missing or unreadable model as an error (it exits 1 with
#     "FAIL: model load"), never as a skip; ctest only registers it when the
#     variable is set, the same idiom as test-nkv-exact-width.
#
#     Its bitwise denominator EXCLUDES the zero-output cell: with n_outputs = 0
#     both logits buffers are empty and memcmp on two empty ranges returns 0, so
#     counting that cell would be a comparison that compared nothing. The cell
#     is proven by its own FOUR_NOUT_ZERO line instead, and the count is
#     reported as nout_zero so the denominator is cells - nout_zero.
#
# This replaced the two overlapping matrices it grew out of
# (test-triparam-matrix and test-ntok-cross-matrix, both retired along with
# their sources). One matrix, one registration: two tests of the same claim is
# how they drift apart.
#
# The PASS_REGULAR_EXPRESSION names the whole summary line shape, including the
# stock arm's rebuild count: a run where the stock arm reused the graph
# (stock_rebuilds=0) must not satisfy it, because that would mean the two arms
# took the same path and the comparison proved nothing. CMake's
# PASS_REGULAR_EXPRESSION cannot compare two numbers (no backreferences), so
# this registration pins the shape and both counts to the expected 124 cells
# and bitwise=123/123 nout_zero=1; the equality of the two numbers,
# `rebuilds != stock_rebuilds`, `stock_rebuilds >= cells` and
# `bitwise denominator == cells - nout_zero` are checked in shell by
# tests/nkv-cpu-probe/gate-nkv-runtime-13022.sh, which is the only entry
# point: it builds its own run_dir, calls ctest once per test and writes
# GATE_SUMMARY. There is no separate runner script for this matrix.
#
# The mask census is pinned as a SHAPE only -- `mask_cells=<n>`,
# `mask_rows=<n>/<n>`, `mask_expect=<n>` and `mask_cells_all=<n>` -- and not
# to 124/124. The equalities `mask_cells == cells`, `mask_rows ok == checked`,
# `mask_rows checked == mask_expect`, `checked >= cells` and
# `mask_cells_all == cells` are the gate's shell arithmetic, which derives them
# from the line itself. Pinning the literal here would duplicate the schedule
# constant and redden ctest for a reason unrelated to masks on any future
# schedule change (and on a model with SWA, where more than one mask is
# registered per cell). mask_expect has no literal either: it is
# n_tokens * mask_filled summed over the cells, so it moves with the schedule
# for the same reason.
add_executable(test-union-four-param-matrix test-union-four-param-matrix.cpp)
target_link_libraries(test-union-four-param-matrix PRIVATE llama)
target_include_directories(test-union-four-param-matrix PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME test-union-four-param-hash COMMAND test-union-four-param-matrix --hash-only)
set_property(TEST test-union-four-param-hash PROPERTY LABELS "main;nkv;ntok")
if (LLAMA_NTOK_TEST_MODEL)
    add_test(NAME test-union-four-param-matrix COMMAND test-union-four-param-matrix --matrix-only "${LLAMA_NTOK_TEST_MODEL}")
    # PASS_REGULAR_EXPRESSION names the part-2 line only, and ctest decides
    # PASS from it while ignoring the exit code. The binary also runs part 3,
    # so the registration must reject a part-3 failure by name: a FAIL: line,
    # FOUR_END ok=0, or the tri-state arm reporting ok=0.
    set_tests_properties(test-union-four-param-matrix PROPERTIES LABELS "model;cpu;nkv;ntok"
        PASS_REGULAR_EXPRESSION "FOUR_MATRIX ok=1 cells=124 hash=[0-9a-f]+ rebuilds=0 stock_rebuilds=[1-9][0-9]* bitwise=123/123 nout_zero=1 mask_cells=[1-9][0-9]* mask_rows=[1-9][0-9]*/[1-9][0-9]* mask_expect=[1-9][0-9]* mask_cells_all=[1-9][0-9]*"
        FAIL_REGULAR_EXPRESSION "^FAIL:;FOUR_END ok=0;FOUR_MASK_UNPROVEN ok=0")
endif()
# The live n_kv lane (single-token decode, 300 steps crossing the 256-cell
# boundary, the hash re-taken over real binds) is a different claim from the
# cross matrix: it asserts that a bind happened at all and that the hash was
# re-taken, which the matrix does not. It keeps its own registration.

# The n_kv observation gate: one ctest name per binder arm plus one per
# negative arm, so each arm is its own ctest result and a missing arm cannot
# hide inside another's pass.
#
# The arm is injected HERE, not by the test: `cmake -E env` sets the variable
# for this one process, and test-nkv-exact-width reads it once and never
# writes it (see the file header of tests/test-nkv-exact-width.cpp). The
# variables the other arms must not inherit are cleared with `--unset=`.
# cmake 4 rejects the short `-u` form ("unknown option '-u'"), and a rejected
# `cmake -E env` makes ctest fail before the test process starts.
#
# GGML_TP5_UNION_BINDER is cleared here too, so the unset arm is unset even
# when the caller's environment carries a stale selector. The off/on arms
# append `GGML_TP5_UNION_BINDER=0/1` AFTER this prefix, and the later
# assignment wins.
#
#   p1-nkv-exact-unset    GGML_TP5_UNION_BINDER unset -> cases single,
#                         multi_seq, multi_stream, widths
#   p1-nkv-exact-off      GGML_TP5_UNION_BINDER=0     -> case off
#   p1-nkv-exact-on       GGML_TP5_UNION_BINDER=1     -> cases single,
#                         multi_seq, multi_stream, widths
#   p1-nkv-exact-negative-{mask,k,v}
#                         --negative omit-<root>, binder unset. The test MUST
#                         exit 0 AND print
#                             NKV_NEGATIVE ok=1 omitted=<root-name>
#                         after recording the named "FAIL: missing tensor"
#                         line for that root. A run that fails for any other
#                         reason exits 1 without that record, so it does not
#                         pass either.
#   p1-nkv-exact-negative-pad-tail
#                         --negative omit-pad-tail, binder unset. Same
#                         contract for the pad-tail observation: exit 0 AND
#                         NKV_NEGATIVE ok=1 omitted=reduce_pad_tail, after
#                         the named "FAIL: missing pad tail" line.
#
# Each arm writes its own NKV_ARM line, which the gate re-reads.
set(NKV_ARM_ENV ${CMAKE_COMMAND} -E env
    --unset=GGML_TP5_UNION_BINDER
    --unset=LLAMA_EXPERIMENTAL_VARLEN_NTOK
    --unset=LLAMA_GRAPH_REUSE_DISABLE
    --unset=GGML_TP5_RSZ_RT
    HIP_VISIBLE_DEVICES=
    ROCR_VISIBLE_DEVICES=)
if (LLAMA_VARLEN_TEST_MODEL)
    add_test(NAME p1-nkv-exact-unset
             COMMAND ${NKV_ARM_ENV} $<TARGET_FILE:test-nkv-exact-width> "${LLAMA_VARLEN_TEST_MODEL}")
    add_test(NAME p1-nkv-exact-off
             COMMAND ${NKV_ARM_ENV} GGML_TP5_UNION_BINDER=0
                     $<TARGET_FILE:test-nkv-exact-width> "${LLAMA_VARLEN_TEST_MODEL}")
    add_test(NAME p1-nkv-exact-on
             COMMAND ${NKV_ARM_ENV} GGML_TP5_UNION_BINDER=1
                     $<TARGET_FILE:test-nkv-exact-width> "${LLAMA_VARLEN_TEST_MODEL}")
    # The negative arms are NOT marked WILL_FAIL. A correct negative run exits
    # 0, so ctest reports it PASS, and the PASS is only meaningful because the
    # registration pins its REASON: PASS_REGULAR_EXPRESSION requires the
    # NKV_NEGATIVE line for this arm's own root, and FAIL_REGULAR_EXPRESSION
    # rejects the text a run prints when the omission went undetected. ctest
    # applies both to the combined stdout+stderr of the test.
    #
    # The root name in the regex is the tensor name the callback matches on
    # (tests/test-nkv-exact-width.cpp, nkv_root_name), not the --negative
    # spelling: omit-mask -> attn_inp_kq_mask.
    set(NKV_NEGATIVE_ROOT_attn_inp_kq_mask mask)
    set(NKV_NEGATIVE_ROOT_union_nkv_k      k)
    set(NKV_NEGATIVE_ROOT_union_nkv_v      v)
    foreach (tensor_name attn_inp_kq_mask union_nkv_k union_nkv_v)
        set(root ${NKV_NEGATIVE_ROOT_${tensor_name}})
        add_test(NAME p1-nkv-exact-negative-${root}
                 COMMAND ${NKV_ARM_ENV} $<TARGET_FILE:test-nkv-exact-width>
                         "${LLAMA_VARLEN_TEST_MODEL}" --negative omit-${root})
        set_property(TEST p1-nkv-exact-negative-${root}
                     PROPERTY
                     LABELS "model;cpu;nkv;negative"
                     PASS_REGULAR_EXPRESSION "NKV_NEGATIVE ok=1 omitted=${tensor_name}"
                     FAIL_REGULAR_EXPRESSION "negative run omitting ${tensor_name} passed")
    endforeach()
    # The pad-tail arm is not one of the three n_kv roots: it omits the
    # pad-tail observation of the consumer node, so its root name is
    # reduce_pad_tail and its named failure line is
    #   FAIL: missing pad tail at step=0 (negative: omitted reduce_pad_tail ...)
    # The record and the undetected-omission text use the same root name, so
    # the two regexes below are the same discipline as the three above.
    add_test(NAME p1-nkv-exact-negative-pad-tail
             COMMAND ${NKV_ARM_ENV} $<TARGET_FILE:test-nkv-exact-width>
                     "${LLAMA_VARLEN_TEST_MODEL}" --negative omit-pad-tail)
    set_property(TEST p1-nkv-exact-negative-pad-tail
                 PROPERTY
                 LABELS "model;cpu;nkv;negative"
                 PASS_REGULAR_EXPRESSION "NKV_NEGATIVE ok=1 omitted=reduce_pad_tail"
                 FAIL_REGULAR_EXPRESSION "negative run omitting reduce_pad_tail passed")
    set_property(TEST p1-nkv-exact-unset p1-nkv-exact-off p1-nkv-exact-on
                 PROPERTY LABELS "model;cpu;nkv;p1")
endif()
# Pillar 1: n_kv as a runtime parameter.
#
# Two registrations, both from one source file:
#   - p1-nkv-topology-hash  needs NO model: it hashes synthetic graphs
#     built at several n_kv widths and asserts the hash does not move. It is
#     always registered, so the topology-hash invariant is checked on every
#     host, model or not.
#   - p1-nkv-live-width     needs a model and reads LLAMA_VARLEN_TEST_MODEL.
#     The test itself treats a missing or unreadable model as an error (it
#     exits 1 with "FAIL: model load"), never as a skip; ctest only registers
#     it when the variable is set, the same idiom as test-nkv-exact-width.
#
# The p1-* names are the ones tests/nkv-cpu-probe/gate-p1-ctest-pack-13022.sh
# runs, one `ctest -R '^<name>$'` per parameter, so each parameter is its own
# ctest result. The older names stay registered as compatibility aliases:
# gate-nkv-runtime-13022.sh and gate-p1-merge-13022.sh run them by name, and
# removing them would break those gates. Both names run the same command.
add_executable(test-nkv-runtime-param test-nkv-runtime-param.cpp)
target_link_libraries(test-nkv-runtime-param PRIVATE llama)
target_include_directories(test-nkv-runtime-param PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME p1-nkv-topology-hash COMMAND test-nkv-runtime-param --hash-only)
set_property(TEST p1-nkv-topology-hash PROPERTY LABELS "main;nkv;p1")
add_test(NAME test-nkv-runtime-param-hash COMMAND test-nkv-runtime-param --hash-only)
set_property(TEST test-nkv-runtime-param-hash PROPERTY LABELS "main;nkv")
if (LLAMA_VARLEN_TEST_MODEL)
    add_test(NAME p1-nkv-live-width COMMAND test-nkv-runtime-param "${LLAMA_VARLEN_TEST_MODEL}" 300)
    set_property(TEST p1-nkv-live-width PROPERTY LABELS "model;cpu;nkv;p1")
    add_test(NAME test-nkv-runtime-param COMMAND test-nkv-runtime-param "${LLAMA_VARLEN_TEST_MODEL}" 300)
    set_property(TEST test-nkv-runtime-param PROPERTY LABELS "model;cpu;nkv")
endif()

# The FINAL gate of the exact n_kv lane: BOTH positive arms must produce
# bitwise identical logits to a stock context at every row of every step,
# while building exactly ONE graph for the whole decode against the stock
# arm's one-per-step. Arm 1 is a 300-step single-token decode (300 rows);
# arm 2 is a 65-step 4-token decode at positions 0..259 with all four rows
# marked (260 rows), so 560 rows are compared in total and the executed span
# crosses 256 in both arms. tests/test-union-binder-stock-equiv.cpp owns the
# claim; this registration exists so the gate script can run it through ctest
# with the same env -i discipline as the other model tests.
#
# Gated on LLAMA_VARLEN_TEST_MODEL, the same idiom as test-nkv-runtime-param:
# ctest only registers it when the variable names a model, and the test itself
# treats a missing or unreadable model as an error, never as a skip.
#
# PASS_REGULAR_EXPRESSION names BSE_END ok=1 only. Every numeric claim is
# checked in shell by tests/nkv-cpu-probe/gate-nkv-runtime-13022.sh, because
# CMake's PASS_REGULAR_EXPRESSION cannot compare two numbers: BSE_SINGLE and
# BSE_MULTI with on_decode_builds == 1, off_decode_builds >= steps,
# rebuild_fallbacks == 0 and pad_fallbacks == 0; BSE_BITWISE rows=a/b with
# a == b and b == the sum of both arms' rows; BSE_NEG differ > 0; and
# BSE_KERNEL_SEL. The regex here must not be widened to make a red run green:
# BSE_KERNEL_SEL is fail-closed on a tree without the kernel-selection
# decoupling, so a green BSE_END is the only honest form.
add_executable(test-union-binder-stock-equiv test-union-binder-stock-equiv.cpp)
target_link_libraries(test-union-binder-stock-equiv PRIVATE llama)
target_include_directories(test-union-binder-stock-equiv PRIVATE ${PROJECT_SOURCE_DIR}/src)
if (LLAMA_VARLEN_TEST_MODEL)
    add_test(NAME test-union-binder-stock-equiv COMMAND test-union-binder-stock-equiv "${LLAMA_VARLEN_TEST_MODEL}")
    # PASS_REGULAR_EXPRESSION ignores the exit code. BSE_END is printed one
    # statement before the return, so a crash after the print would pass; the
    # FAIL regex rejects the named failure lines as well.
    set_property(TEST test-union-binder-stock-equiv PROPERTY LABELS "model;cpu;nkv"
        PASS_REGULAR_EXPRESSION "BSE_END ok=1"
        FAIL_REGULAR_EXPRESSION "^FAIL:;BSE_END ok=0")
endif()

# G4: the fused-QKV fail-closed gate. The fixture is a GGUF whose per-layer
# attn_qkv.weight is the Q||K||V concatenation of the dense model's
# attn_q/attn_k/attn_v output rows (scripts/make-fused-qkv-fixture.py). On that
# fixture the token lane must REFUSE: the Q/K/V views are non-contiguous row
# slices of one fused matmul, so the shape pass fails closed, names the
# refusal, applies no in-place bind, disables the envelope, and reproduces the
# stock arm's logits bit for bit at every step. The test sets BOTH arms'
# environment itself (the variables are read once, in the llama_context
# constructor), so ctest needs no ENVIRONMENT property; only the fixture path
# is required.
#
# Gated on LLAMA_FUSED_QKV_TEST_MODEL, the same idiom as the other
# model-gated tests: ctest registers it only when the variable names a model,
# and the test itself treats a missing or unreadable model as an error, never
# as a skip. PASS_REGULAR_EXPRESSION names FUSED_QKV_REFUSAL ok=1 only; every
# counter on that line is checked inside the harness (it exits nonzero on any
# of them), because CMake's regex cannot compare numbers.
set(LLAMA_FUSED_QKV_TEST_MODEL "" CACHE FILEPATH "Fused-QKV LLAMA GGUF for the G4 fail-closed gate")
add_executable(test-fused-qkv-refusal test-fused-qkv-refusal.cpp)
target_link_libraries(test-fused-qkv-refusal PRIVATE llama)
if (LLAMA_FUSED_QKV_TEST_MODEL)
    add_test(NAME test-fused-qkv-refusal COMMAND $<TARGET_FILE:test-fused-qkv-refusal> "${LLAMA_FUSED_QKV_TEST_MODEL}")
    set_property(TEST test-fused-qkv-refusal PROPERTY LABELS "model;cpu;nkv"
        PASS_REGULAR_EXPRESSION "FUSED_QKV_REFUSAL ok=1"
        FAIL_REGULAR_EXPRESSION "^FAIL:;FUSED_QKV_REFUSAL ok=0")
endif()

# The report writer's unit tests are pure Python: they parse synthetic logs in
# the shapes test-nkv-exact-width prints and need neither a model nor a built
# binary. The three-arm runner (nkv-cpu-probe/run-nkv-exact-width-arms-13022.sh)
# is deliberately not registered here: it needs a model and a CPU build host.
find_package(Python3 COMPONENTS Interpreter QUIET)
if (Python3_Interpreter_FOUND)
    add_test(
        NAME test-nkv-exact-width-report
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/nkv-cpu-probe/test_nkv_exact_width_report.py)
    set_property(TEST test-nkv-exact-width-report PROPERTY LABELS "main;nkv")
endif()
# Fixed-token-width n_outputs tail gates. These read NOUT_TAIL_MODEL inside
# the probe (a missing model is an error, not a skip), so no configure-time
# model variable is needed; the gate script supplies the variable.
#
# p1-nout-tail-* are the pack gate's names; test-nout-tail-* stay registered
# because gate-p1-merge-13022.sh and reproduce_nout_tail_cpu.sh run them.
add_test(NAME p1-nout-tail-fixed COMMAND test-varlen-output-probe --gate)
add_test(NAME p1-nout-tail-instrumented COMMAND test-varlen-output-probe --tail-gate)
set_tests_properties(p1-nout-tail-fixed p1-nout-tail-instrumented PROPERTIES LABELS "model;cpu;nout;p1")
add_test(NAME test-nout-tail-fixed COMMAND test-varlen-output-probe --gate)
add_test(NAME test-nout-tail-instrumented COMMAND test-varlen-output-probe --tail-gate)
set_tests_properties(test-nout-tail-fixed test-nout-tail-instrumented PROPERTIES LABELS "model;cpu;nout")

# Union/varlen end-to-end driver: real llama_decode, multi-sequence schedule,
# JSON report. Needs a caller-supplied tiny LLAMA GGUF, like the probes above.
add_executable(test-union-varlen-e2e nkv-cpu-probe/union_varlen_e2e.cpp)
target_link_libraries(test-union-varlen-e2e PRIVATE llama)

# Padded-lane A/B gate: the exact lane (GGML_TP5_UNION_BINDER_EXACT unset) and
# the stock padded control lane (EXACT=0) on one model, so the two lanes are
# shown to be distinguishable and to produce the same logits. Needs a
# caller-supplied tiny LLAMA GGUF, like the probes above.
add_executable(test-pad-lane-ab-gate nkv-cpu-probe/pad_lane_ab_gate.cpp)
target_link_libraries(test-pad-lane-ab-gate PRIVATE llama)

# n_outputs runtime parameter lane: one graph topology across a cycle of active
# output counts, paired row comparison against the stock arm. Needs a
# caller-supplied tiny LLAMA GGUF, like the probes above.
add_executable(test-nout-runtime-probe nkv-cpu-probe/nout_runtime_probe.cpp)
target_link_libraries(test-nout-runtime-probe PRIVATE llama)
target_include_directories(test-nout-runtime-probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/nkv-cpu-probe)

# rs_z as a runtime parameter (GGML_TP5_RSZ_RUNTIME_PARAM). Needs a caller-
# supplied tiny recurrent (MAMBA/MAMBA2) GGUF; without one the test is
# registered but skipped (exit 77 -> SKIP_RETURN_CODE). It asserts one graph
# build across >= 4 distinct rs_z values, bitwise equality with the OFF arm,
# non-degenerate outputs, a named non-silent fallback for rs_z == -1, the
# envelope boundary, and the n_rs_seq refusal.
add_executable(test-rsz-runtime-param test-rsz-runtime-param.cpp)
target_link_libraries(test-rsz-runtime-param PRIVATE llama)
# same fixture, crossed with the n_kv lane of the same binder: a step that
# moves BOTH n_kv and rs_z must still bind in place and stay bitwise equal to
# stock, with the exact width and with the padded control. It reads the live
# llama_memory_recurrent state, so it needs the internal headers.
add_executable(test-rsz-union-combo test-rsz-union-combo.cpp)
target_link_libraries(test-rsz-union-combo PRIVATE llama)
target_include_directories(test-rsz-union-combo PRIVATE ${PROJECT_SOURCE_DIR}/src)
set(LLAMA_RSZ_TEST_MODEL "" CACHE FILEPATH "Tiny recurrent (MAMBA/MAMBA2) GGUF for the rs_z runtime-parameter tests")
if (LLAMA_RSZ_TEST_MODEL)
    add_test(NAME test-rsz-runtime-param COMMAND $<TARGET_FILE:test-rsz-runtime-param> "${LLAMA_RSZ_TEST_MODEL}")
    set_property(TEST test-rsz-runtime-param PROPERTY LABELS main)
    set_property(TEST test-rsz-runtime-param PROPERTY SKIP_RETURN_CODE 77)
    add_test(NAME test-rsz-union-combo COMMAND $<TARGET_FILE:test-rsz-union-combo> "${LLAMA_RSZ_TEST_MODEL}")
    set_property(TEST test-rsz-union-combo PROPERTY LABELS main)
    set_property(TEST test-rsz-union-combo PROPERTY SKIP_RETURN_CODE 77)
endif()

# the same rs_z lane on a HYBRID fixture (recurrent state AND an attention KV
# cache), so the n_kv width lane actually moves. Needs a caller-supplied tiny
# hybrid GGUF (QWEN35 / NEMOTRON_H / JAMBA); without one the test is
# registered but skipped (exit 77 -> SKIP_RETURN_CODE). See
# tests/nkv-cpu-probe/README.md for the fixture contract.
add_executable(test-rsz-hybrid-compose test-rsz-hybrid-compose.cpp)
target_link_libraries(test-rsz-hybrid-compose PRIVATE llama)
target_include_directories(test-rsz-hybrid-compose PRIVATE ${PROJECT_SOURCE_DIR}/src)
set(LLAMA_RSZ_HYBRID_TEST_MODEL "" CACHE FILEPATH "Tiny hybrid (recurrent + attention KV) GGUF for test-rsz-hybrid-compose")
if (LLAMA_RSZ_HYBRID_TEST_MODEL)
    add_test(NAME test-rsz-hybrid-compose COMMAND $<TARGET_FILE:test-rsz-hybrid-compose> "${LLAMA_RSZ_HYBRID_TEST_MODEL}")
    set_property(TEST test-rsz-hybrid-compose PROPERTY LABELS main)
    set_property(TEST test-rsz-hybrid-compose PROPERTY SKIP_RETURN_CODE 77)
endif()

# the rs_z selection write through llm_graph_input_rs::set_input(): the null
# no-op, the write, the rewrite after a moved rs_z, the ring row
# rs_idx*mem_size + rs_z, the mixed-depth refusal with can_reuse() false, and
# the null row on a live context. A minimal llama_memory_recurrent (one layer,
# two cells) and a real recurrent context over it, no model file and no backend
# device (the CPU backend is registered by llama_backend_init). Runs in every
# CPU build; it needs no fixture variable.
add_executable(test-rsz-selection-write test-rsz-selection-write.cpp)
target_link_libraries(test-rsz-selection-write PRIVATE llama)
target_include_directories(test-rsz-selection-write PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME test-rsz-selection-write COMMAND test-rsz-selection-write)
set_property(TEST test-rsz-selection-write PROPERTY LABELS "cpu;rsz")
set_property(TEST test-rsz-selection-write PROPERTY PASS_REGULAR_EXPRESSION "RSZ_SELECTION_WRITE_TEST PASS")

# the same write driven through BOTH hybrid set_input() sites
# (llm_graph_input_mem_hybrid, llm_graph_input_mem_hybrid_k): the write, the
# rewrite after a move, cross-site agreement, the ring row, the mixed-depth
# refusal, and the null no-op. A minimal llama_memory_hybrid (one attention
# layer, one recurrent layer, two cells) and a real hybrid context over it, no
# model file and no backend device. The iswa site is not driven: the lane never
# arms on an iswa context (union_is_hybrid_iswa() excludes it from rsz_gate_ok).
# Runs in every CPU build; it needs no fixture variable.
add_executable(test-rsz-hybrid-selection-write test-rsz-hybrid-selection-write.cpp)
target_link_libraries(test-rsz-hybrid-selection-write PRIVATE llama)
target_include_directories(test-rsz-hybrid-selection-write PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME test-rsz-hybrid-selection-write COMMAND test-rsz-hybrid-selection-write)
set_property(TEST test-rsz-hybrid-selection-write PROPERTY LABELS "cpu;rsz")
set_property(TEST test-rsz-hybrid-selection-write PROPERTY PASS_REGULAR_EXPRESSION "RSZ_HYBRID_SELECTION_WRITE_TEST PASS")

# rs_z crossed with the n_kv width on the SAME steps, on the hybrid fixture:
# topology invariance (no rebuild while both parameters move), both parameters
# observed to move, the selection row read back from the live graph, and
# bitwise equality with stock. Same fixture variable and skip contract as
# test-rsz-hybrid-compose.
add_executable(test-rsz-nkv-rsz-topology test-rsz-nkv-rsz-topology.cpp)
target_link_libraries(test-rsz-nkv-rsz-topology PRIVATE llama)
target_include_directories(test-rsz-nkv-rsz-topology PRIVATE ${PROJECT_SOURCE_DIR}/src)
if (LLAMA_RSZ_HYBRID_TEST_MODEL)
    add_test(NAME test-rsz-nkv-rsz-topology COMMAND $<TARGET_FILE:test-rsz-nkv-rsz-topology> "${LLAMA_RSZ_HYBRID_TEST_MODEL}")
    set_property(TEST test-rsz-nkv-rsz-topology PROPERTY LABELS main)
    set_property(TEST test-rsz-nkv-rsz-topology PROPERTY SKIP_RETURN_CODE 77)
endif()

# the recurrent rollback ring crossed with the tp5-relay path, driven through
# the REAL llm_graph_context::build_rs() and llm_graph_input_rs::set_input() /
# can_reuse(): arms R0 (no ring), R1 (ring, group 2), R2 (stock refusal with a
# positive control), R3 (mixed depths refused, can_reuse() false), and the
# mem_size x n_rs_seq row-arithmetic sweep. The arms differ by constructor
# arguments and rs_row_armed. The only environment variable the file touches is
# LLAMA_GDN_REPLAY, which it unsets in its own process before constructing the
# memory so that gdn_replay is false whatever the caller exported. Runs in
# every CPU build; it needs no fixture variable and no GPU.
add_executable(test-rs-ring-relay test-rs-ring-relay.cpp)
target_link_libraries(test-rs-ring-relay PRIVATE llama)
target_include_directories(test-rs-ring-relay PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME test-rs-ring-relay COMMAND test-rs-ring-relay)
set_property(TEST test-rs-ring-relay PROPERTY LABELS "cpu;rsz")
set_property(TEST test-rs-ring-relay PROPERTY PASS_REGULAR_EXPRESSION "RS_RING_RELAY_TEST PASS")

# the ticket's literal requirement: putting rs_z into a reuse key or a
# topology key must be DETECTED. K1 armed + moved rs_z reuses and the row
# follows; K2 the stock form (s_zero == nullptr) must still key on rs_z; K3a
# the index tensor's VALUE does not move the topology hash, K3b a widened
# state tensor does, K3c the width table excludes exactly the dimension it
# names. A minimal llama_memory_recurrent and a synthetic graph, no model file
# and no backend device. Runs in every CPU build.
add_executable(test-rsz-key-injection test-rsz-key-injection.cpp)
target_link_libraries(test-rsz-key-injection PRIVATE llama)
target_include_directories(test-rsz-key-injection PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME test-rsz-key-injection COMMAND test-rsz-key-injection)
set_property(TEST test-rsz-key-injection PROPERTY LABELS "cpu;rsz")
set_property(TEST test-rsz-key-injection PROPERTY PASS_REGULAR_EXPRESSION "RSZ_KEY_INJECTION_TEST PASS")

# the relay lane's one-graph reuse decision across a moving rs_z, pinned where
# it is reachable without a relay backend: llm_graph_input_rs::can_reuse()
# (R1 armed + moved rs_z reuses), llm_graph_params::allow_reuse() for n_tokens
# (R3), llm_graph_input_out_ids::can_reuse() for n_outputs (R4), and the
# stock-form control (R5). There is no n_kv arm: the constraint is stated in the
# test header rather than asserted, because no movable llama_kv_cache_context is
# constructible without a llama_context. The three-parameter one-graph claim
# lives on the union binder (test-rsz-nkv-rsz-topology, test-triparam-matrix).
# Runs in every CPU build.
add_executable(test-rsz-relay-onegraph test-rsz-relay-onegraph.cpp)
target_link_libraries(test-rsz-relay-onegraph PRIVATE llama)
target_include_directories(test-rsz-relay-onegraph PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME test-rsz-relay-onegraph COMMAND test-rsz-relay-onegraph)
set_property(TEST test-rsz-relay-onegraph PROPERTY LABELS "cpu;rsz")
set_property(TEST test-rsz-relay-onegraph PROPERTY PASS_REGULAR_EXPRESSION "RSZ_RELAY_ONEGRAPH_TEST PASS")

# TP5 pillar-5 plan/frame/pool policy. Header-only, no llama/ggml library and
# no device: the K=8 plan cache, the five-shape frame LRU, the exact pool
# budget, the per-window channel slots and the legacy switch defaults.
add_executable(test-relay-plan-policy test-relay-plan-policy.cpp)
target_compile_features(test-relay-plan-policy PRIVATE cxx_std_17)
target_include_directories(test-relay-plan-policy PRIVATE ${PROJECT_SOURCE_DIR}/ggml/include)
add_test(NAME test-relay-plan-policy COMMAND test-relay-plan-policy)
set_property(TEST test-relay-plan-policy PROPERTY LABELS "cpu;tp5")
set_property(TEST test-relay-plan-policy PROPERTY PASS_REGULAR_EXPRESSION "RELAY_PLAN_POLICY_TEST PASS")

# TP5 relay verification counters. Header-only, no llama/ggml library and no
# device: requested/resolved/executed stay nested and a submission that did
# not request verification counts nothing.
add_executable(test-relay-verify-counters test-relay-verify-counters.cpp)
target_compile_features(test-relay-verify-counters PRIVATE cxx_std_17)
target_include_directories(test-relay-verify-counters PRIVATE ${PROJECT_SOURCE_DIR}/ggml/include)
add_test(NAME test-relay-verify-counters COMMAND test-relay-verify-counters)
set_property(TEST test-relay-verify-counters PROPERTY LABELS "cpu;tp5")
set_property(TEST test-relay-verify-counters PROPERTY PASS_REGULAR_EXPRESSION "RELAY_VERIFY_COUNTERS_TEST PASS")

# TP5 llama_free relay retain policy. Header-only, no llama/ggml library and
# no device: epoch 0 and the prepare window, borrow, completion tokens, output
# releases, failed-provider exemption, contention, unsupported ABI fallback and
# epoch/channel mismatches.
add_executable(test-tp5-relay-free-decision test-tp5-relay-free-decision.cpp)
target_compile_features(test-tp5-relay-free-decision PRIVATE cxx_std_17)
target_include_directories(test-tp5-relay-free-decision PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/ggml/include
    ${PROJECT_SOURCE_DIR}/ggml/src)
add_test(NAME test-tp5-relay-free-decision COMMAND test-tp5-relay-free-decision)
set_property(TEST test-tp5-relay-free-decision PROPERTY LABELS "cpu;tp5")
set_property(TEST test-tp5-relay-free-decision PROPERTY PASS_REGULAR_EXPRESSION "RELAY_FREE_DECISION_TEST PASS")

# TP5 llama-side per-shape plan cache policy. Header-only, no model/backend:
# same-shape hit, alloc_gen isolation, flush, LRU eviction, rekey and the
# dual-window independence of two cache instances.
add_executable(test-shape-cache test-shape-cache.cpp)
target_compile_features(test-shape-cache PRIVATE cxx_std_17)
target_include_directories(test-shape-cache PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME test-shape-cache COMMAND test-shape-cache)
set_property(TEST test-shape-cache PROPERTY LABELS "cpu;tp5")
set_property(TEST test-shape-cache PROPERTY PASS_REGULAR_EXPRESSION "SHAPE_CACHE_TEST PASS")

# llama_kv_cells journal borrow ownership. Header-only, no model/backend:
# a snapshot copy must not keep the source's cell_journal pointer, and a
# moved-from object must not record into the journal it no longer owns.
add_executable(test-kv-cells-journal-ownership test-kv-cells-journal-ownership.cpp)
target_compile_features(test-kv-cells-journal-ownership PRIVATE cxx_std_17)
target_include_directories(test-kv-cells-journal-ownership PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/ggml/include)
# ggml supplies ggml_abort for the GGML_ASSERTs in the header.
target_link_libraries(test-kv-cells-journal-ownership PRIVATE ggml)
add_test(NAME test-kv-cells-journal-ownership COMMAND test-kv-cells-journal-ownership)
set_property(TEST test-kv-cells-journal-ownership PROPERTY LABELS "cpu;tp5")
set_property(TEST test-kv-cells-journal-ownership PROPERTY PASS_REGULAR_EXPRESSION "KV_CELLS_JOURNAL_OWNERSHIP_TEST PASS")

# llama_kv_cells position lookup. Header-only, no model/backend: the
# conservative pos_max_ fast path must never turn a hit into a miss, and every
# non-hit must keep the original scan semantics after removals, shifts,
# snapshots, copies and journal replay.
add_executable(test-kv-cells-pos-lookup test-kv-cells-pos-lookup.cpp)
target_compile_features(test-kv-cells-pos-lookup PRIVATE cxx_std_17)
target_include_directories(test-kv-cells-pos-lookup PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/ggml/include)
# ggml supplies ggml_abort for the GGML_ASSERTs in the header.
target_link_libraries(test-kv-cells-pos-lookup PRIVATE ggml)
add_test(NAME test-kv-cells-pos-lookup COMMAND test-kv-cells-pos-lookup)
set_property(TEST test-kv-cells-pos-lookup PROPERTY LABELS "cpu;tp5")
set_property(TEST test-kv-cells-pos-lookup PROPERTY PASS_REGULAR_EXPRESSION "KV_CELLS_POS_LOOKUP_TEST PASS")

if (UNIX)
    add_executable(test-ntok-onegraph test-ntok-onegraph.cpp)
    target_link_libraries(test-ntok-onegraph PRIVATE llama)
    set(LLAMA_NTOK_TEST_MODEL "" CACHE FILEPATH "Dense LLAMA GGUF for the variable-token test")
    set(LLAMA_NTOK_TEST_ROUNDS "50" CACHE STRING "Alternating-width test rounds")
    # Narrow real-CPU status test for llama_kv_cache::update: NO_UPDATE,
    # APPLIED K-shift, and a real K-shift graph-allocation failure that must
    # return -3 and restore the pre-shift cells.
    add_executable(test-kv-update-status test-kv-update-status.cpp)
    target_link_libraries(test-kv-update-status PRIVATE llama)
    target_include_directories(test-kv-update-status PRIVATE
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/ggml/src
        ${PROJECT_SOURCE_DIR}/ggml/include)
    if (LLAMA_NTOK_TEST_MODEL)
        add_test(NAME test-kv-update-status COMMAND $<TARGET_FILE:test-kv-update-status> "${LLAMA_NTOK_TEST_MODEL}")
        set_property(TEST test-kv-update-status PROPERTY LABELS "main;cpu;tp5")
        set_property(TEST test-kv-update-status PROPERTY PASS_REGULAR_EXPRESSION "KV_UPDATE_STATUS_TEST PASS")
        set_property(TEST test-kv-update-status PROPERTY SKIP_RETURN_CODE 77)
    endif()
    if (LLAMA_NTOK_TEST_MODEL)
        # The n_outputs probe sets both binder arms itself, so it needs no
        # environment and no extra switch: the same caller-supplied GGUF is the
        # only requirement.
        add_test(NAME test-nout-runtime-probe COMMAND $<TARGET_FILE:test-nout-runtime-probe>
            "${LLAMA_NTOK_TEST_MODEL}")
        set_property(TEST test-nout-runtime-probe PROPERTY LABELS main)
        add_test(NAME test-ntok-onegraph COMMAND $<TARGET_FILE:test-ntok-onegraph> "${LLAMA_NTOK_TEST_MODEL}" "${LLAMA_NTOK_TEST_ROUNDS}")
        # the paired fixed/variable varlen probes: the same fixture, one run at
        # a fixed token width and one at a variable width, each writing its ON
        # and OFF arms to a file for the comparison the probe performs itself.
        foreach(mode fixed variable)
            add_test(NAME test-varlen-output-${mode} COMMAND $<TARGET_FILE:test-varlen-output-probe>
                "${LLAMA_NTOK_TEST_MODEL}" "${mode}"
                "${CMAKE_CURRENT_BINARY_DIR}/varlen-${mode}-on.bin"
                "${CMAKE_CURRENT_BINARY_DIR}/varlen-${mode}-off.bin")
            set_property(TEST test-varlen-output-${mode} PROPERTY ENVIRONMENT "LLAMA_EXPERIMENTAL_VARLEN_NTOK=1")
            set_property(TEST test-varlen-output-${mode} PROPERTY LABELS main)
        endforeach()
        set_property(TEST test-ntok-onegraph PROPERTY LABELS main)
    endif()
    # Token-lane bind/replay regression: ONE context binds the token and n_kv
    # axes in place on every step after the record build, builds no graph after
    # its first decode, and reproduces the per-length rebuild baseline bit for
    # bit. The baseline is a fresh context per captured length with
    # LLAMA_GRAPH_REUSE_DISABLE=1, so "it rebuilt" is measured, not assumed.
    # Needs a model, so it is registered only when LLAMA_NTOK_TEST_MODEL is
    # set, like test-ntok-onegraph above.
    add_executable(test-ntok-bind-replay test-ntok-bind-replay.cpp)
    target_link_libraries(test-ntok-bind-replay PRIVATE llama)
    if (LLAMA_NTOK_TEST_MODEL)
        add_test(NAME test-ntok-bind-replay COMMAND $<TARGET_FILE:test-ntok-bind-replay> "${LLAMA_NTOK_TEST_MODEL}")
        set_property(TEST test-ntok-bind-replay PROPERTY LABELS "model;cpu;nkv")
    endif()
    # Length sweep: drives llama_decode through a growing width schedule on the
    # binder arm and asserts the topology key never moves, every step built at
    # most one graph, and the sweep built at least one new graph. It asserts
    # neither rebuild_count == 0 nor that the graph was ever reused: on a
    # growing-length schedule allow_reuse() compares n_tokens and
    # can_reuse_kq_mask() compares the mask width against the used span, so
    # reuse is unreachable (see the test's header comment). Needs a model, so
    # it is registered only when LLAMA_NTOK_TEST_MODEL is set, like
    # test-ntok-onegraph above.
    add_executable(test-topology-key-sweep test-topology-key-sweep.cpp)
    target_link_libraries(test-topology-key-sweep PRIVATE llama)
    if (LLAMA_NTOK_TEST_MODEL)
        add_test(NAME test-topology-key-sweep COMMAND $<TARGET_FILE:test-topology-key-sweep> "${LLAMA_NTOK_TEST_MODEL}" "${LLAMA_NTOK_TEST_ROUNDS}")
        set_property(TEST test-topology-key-sweep PROPERTY LABELS main)
    endif()
    set(LLAMA_UNION_VARLEN_E2E_STEPS "64" CACHE STRING "Union/varlen end-to-end decode steps")
    # Report path. Empty keeps the build-tree default, so a plain configure is
    # unchanged; the 13022 runner sets it so the JSON lands in its RUN_DIR.
    set(LLAMA_UNION_VARLEN_E2E_JSON "" CACHE FILEPATH "Union/varlen e2e JSON report path")
    # Explicit request for the end-to-end test. The test needs a caller-supplied
    # GGUF, so it cannot be registered without one. Leaving this OFF keeps a
    # plain configure unchanged; turning it ON turns a missing model path into a
    # configure-time error instead of a test that silently never registers.
    set(LLAMA_UNION_VARLEN_E2E OFF CACHE BOOL
        "Register test-union-varlen-e2e; requires LLAMA_NTOK_TEST_MODEL")
    if (LLAMA_UNION_VARLEN_E2E AND NOT LLAMA_NTOK_TEST_MODEL)
        message(FATAL_ERROR
            "LLAMA_UNION_VARLEN_E2E=ON requires LLAMA_NTOK_TEST_MODEL to name a "
            "dense LLAMA GGUF, but LLAMA_NTOK_TEST_MODEL is empty. Set both, or "
            "leave LLAMA_UNION_VARLEN_E2E=OFF to skip the test.")
    endif()
    if (LLAMA_UNION_VARLEN_E2E_JSON)
        set(LLAMA_UNION_VARLEN_E2E_JSON_PATH "${LLAMA_UNION_VARLEN_E2E_JSON}")
    else()
        set(LLAMA_UNION_VARLEN_E2E_JSON_PATH "${CMAKE_CURRENT_BINARY_DIR}/union-varlen-e2e.json")
    endif()
    if (LLAMA_UNION_VARLEN_E2E)
        add_test(NAME test-union-varlen-e2e COMMAND $<TARGET_FILE:test-union-varlen-e2e>
            "${LLAMA_NTOK_TEST_MODEL}"
            "${LLAMA_UNION_VARLEN_E2E_JSON_PATH}"
            "${LLAMA_UNION_VARLEN_E2E_STEPS}")
        set_property(TEST test-union-varlen-e2e PROPERTY LABELS main)
    endif()
    set(LLAMA_PAD_LANE_AB_GATE_STEPS "32" CACHE STRING "Padded-lane A/B gate decode steps")
    # Report path. Empty keeps the build-tree default, so a plain configure is
    # unchanged; the 13022 runner sets it so the JSON lands in its RUN_DIR.
    set(LLAMA_PAD_LANE_AB_GATE_JSON "" CACHE FILEPATH "Padded-lane A/B gate JSON report path")
    # Explicit request for the gate runner. It needs a caller-supplied GGUF, so
    # it cannot be registered without one. Leaving this OFF keeps a plain
    # configure unchanged; turning it ON turns a missing model path into a
    # configure-time error instead of a test that silently never registers.
    set(LLAMA_PAD_LANE_AB_GATE OFF CACHE BOOL
        "Register test-pad-lane-ab-gate; requires LLAMA_NTOK_TEST_MODEL")
    if (LLAMA_PAD_LANE_AB_GATE AND NOT LLAMA_NTOK_TEST_MODEL)
        message(FATAL_ERROR
            "LLAMA_PAD_LANE_AB_GATE=ON requires LLAMA_NTOK_TEST_MODEL to name a "
            "dense LLAMA GGUF, but LLAMA_NTOK_TEST_MODEL is empty. Set both, or "
            "leave LLAMA_PAD_LANE_AB_GATE=OFF to skip the test.")
    endif()
    if (LLAMA_PAD_LANE_AB_GATE_JSON)
        set(LLAMA_PAD_LANE_AB_GATE_JSON_PATH "${LLAMA_PAD_LANE_AB_GATE_JSON}")
    else()
        set(LLAMA_PAD_LANE_AB_GATE_JSON_PATH "${CMAKE_CURRENT_BINARY_DIR}/pad-lane-ab-gate.json")
    endif()
    if (LLAMA_PAD_LANE_AB_GATE)
        add_test(NAME test-pad-lane-ab-gate COMMAND $<TARGET_FILE:test-pad-lane-ab-gate>
            "${LLAMA_NTOK_TEST_MODEL}"
            "${LLAMA_PAD_LANE_AB_GATE_JSON_PATH}"
            "${LLAMA_PAD_LANE_AB_GATE_STEPS}")
        set_property(TEST test-pad-lane-ab-gate PROPERTY LABELS main)
    endif()
endif()
# The token lane's own kernel-level A/B gate (tests/nkv-cpu-probe/pad_lane_ab_gate.cpp)
# is registered above. It is NOT the four-parameter matrix: it compares the
# exact lane with the stock padded control lane on one model and proves the two
# lanes are distinguishable. The four-parameter cross matrix is
# test-union-four-param-matrix, registered near the top of this file, and it is
# the single registration of that claim.
# Fixture test for the aggregate union/onegraph ctest gate summarizer
# (nkv-cpu-probe/union_ctest_gate.py). It feeds the summarizer synthetic ctest
# listings, synthetic JUnit XML and synthetic per-test artifacts, and asserts
# the aggregate JSON and its fail-closed paths. No GGUF model, no compiled
# binary and no build tree are needed: it is plain python3, stdlib only, so it
# adds nothing to the build beyond what LLAMA_BUILD_TESTS=ON already does.
# Registered under the same explicit switch as test-union-varlen-e2e.
if (LLAMA_UNION_VARLEN_E2E)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if (Python3_Interpreter_FOUND)
        add_test(NAME test-union-ctest-gate
                 COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/nkv-cpu-probe/test_union_ctest_gate.py)
        set_property(TEST test-union-ctest-gate PROPERTY LABELS main)
    endif()
endif()
# build test-tokenizer-0 target once and add many tests
llama_build(test-tokenizer-0.cpp)

llama_test(test-tokenizer-0 NAME test-tokenizer-0-bert-bge          ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-bert-bge.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-command-r         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-command-r.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-coder    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-deepseek-coder.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-llm      ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-deepseek-llm.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-falcon            ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-falcon.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-gemma-4           ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gemma-4.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-gpt-2             ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-llama-bpe         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-llama-spm         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-spm.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-mpt               ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-mpt.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-phi-3             ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-phi-3.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-qwen2             ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-qwen2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-qwen35            ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-qwen35.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-refact            ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-refact.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-starcoder         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-starcoder.gguf)

if (NOT WIN32)
    llama_test_cmd(
        ${CMAKE_CURRENT_SOURCE_DIR}/test-tokenizers-repo.sh
        NAME test-tokenizers-ggml-vocabs
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        ARGS https://huggingface.co/ggml-org/vocabs ${PROJECT_SOURCE_DIR}/models/ggml-vocabs
    )
endif()

if (LLAMA_LLGUIDANCE)
    llama_build_and_test(test-grammar-llguidance.cpp ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf)
endif ()

llama_build(test-recurrent-state-rollback.cpp)
add_executable(test-union-binder-rsz test-union-binder-rsz.cpp)
target_link_libraries(test-union-binder-rsz PRIVATE llama)
if (LLAMA_TESTS_INSTALL)
    install(TARGETS test-union-binder-rsz RUNTIME)
endif()
target_include_directories(test-union-binder-rsz PRIVATE ${PROJECT_SOURCE_DIR}/src)
# this test uses only llama.h / ggml.h and the internal headers, so it links
# libllama directly and does not pull in llama-common (fewer TUs to build)
add_executable(test-rsz-multiseq-decode test-rsz-multiseq-decode.cpp)
target_link_libraries(test-rsz-multiseq-decode PRIVATE llama)
target_include_directories(test-rsz-multiseq-decode PRIVATE ${PROJECT_SOURCE_DIR}/src)

# Pillar 1: the rs_z lane as ctest gates, one name per parameter.
#
# p1-rsz-runtime-param      model: test-union-binder-rsz on a tiny recurrent
#                           MAMBA fixture (LLAMA_RSZ_TEST_MODEL). The test
#                           prints RSZ_BINDER_TEST PASS last; the gate re-reads
#                           that line so a truncated run cannot pass.
# p1-rsz-selection-write    no model: the selection-row write contract, run
#                           through the real llm_graph_input_rs helper.
# p1-rsz-nkv-topology       model: the rs_z lane crossed with the n_kv width
#                           lane on a tiny hybrid fixture
#                           (LLAMA_RSZ_HYBRID_TEST_MODEL).
#
# Each model test exits 77 when its model variable is unset, so ctest reports
# SKIP rather than a vacuous PASS; the pack gate maps that to unproven.
#
# The three sources are registered unconditionally so a missing fixture is a
# visible SKIP, not a missing test. A target that does not exist would make
# `ctest -R '^<name>$'` fail loudly, which is also correct, but a SKIP names
# the reason.
set(LLAMA_RSZ_TEST_MODEL "" CACHE FILEPATH "Tiny recurrent MAMBA GGUF for the rs_z runtime-parameter test")
set(LLAMA_RSZ_HYBRID_TEST_MODEL "" CACHE FILEPATH "Tiny hybrid (recurrent + attention KV) GGUF for the rs_z x n_kv topology test")

# The three targets are declared once, above, by the rs_z lane block. These
# are the pack gate's p1-* names for the same binaries; a second
# add_executable() for the same target would be a duplicate declaration.
add_test(NAME p1-rsz-runtime-param COMMAND test-rsz-runtime-param "${LLAMA_RSZ_TEST_MODEL}")
set_property(TEST p1-rsz-runtime-param PROPERTY LABELS "model;cpu;rsz;p1")
set_property(TEST p1-rsz-runtime-param PROPERTY SKIP_RETURN_CODE 77)

add_test(NAME p1-rsz-selection-write COMMAND test-rsz-selection-write)
set_property(TEST p1-rsz-selection-write PROPERTY LABELS "main;cpu;rsz;p1")

add_test(NAME p1-rsz-nkv-topology COMMAND test-rsz-nkv-rsz-topology "${LLAMA_RSZ_HYBRID_TEST_MODEL}")
set_property(TEST p1-rsz-nkv-topology PROPERTY LABELS "model;cpu;rsz;p1")
set_property(TEST p1-rsz-nkv-topology PROPERTY SKIP_RETURN_CODE 77)
# The two model-driven rs_z tests were built but never registered, so a green
# ctest said nothing about them. Both take exactly one tiny recurrent GGUF
# (tests/rsz/make_tiny_rsz_mamba.py) and exit 2 on a missing argument, so they
# are registered only when LLAMA_RSZ_TEST_MODEL is set -- the same variable and
# the same contract as test-rsz-runtime-param above. Their final lines are
# "RSZ_BINDER_TEST PASS" and "RSZ_MULTISEQ_TEST PASS"; a run that prints
# "FAIL n" also exits 1, so the regex cannot pass on a failing run.
if (LLAMA_RSZ_TEST_MODEL)
    add_test(NAME test-union-binder-rsz COMMAND $<TARGET_FILE:test-union-binder-rsz> "${LLAMA_RSZ_TEST_MODEL}")
    set_property(TEST test-union-binder-rsz PROPERTY LABELS "model;cpu;rsz")
    set_property(TEST test-union-binder-rsz PROPERTY PASS_REGULAR_EXPRESSION "RSZ_BINDER_TEST PASS")
    add_test(NAME test-rsz-multiseq-decode COMMAND $<TARGET_FILE:test-rsz-multiseq-decode> "${LLAMA_RSZ_TEST_MODEL}")
    set_property(TEST test-rsz-multiseq-decode PROPERTY LABELS "model;cpu;rsz")
    set_property(TEST test-rsz-multiseq-decode PROPERTY PASS_REGULAR_EXPRESSION "RSZ_MULTISEQ_TEST PASS")
endif()

llama_build_and_test(test-gdn-ingredient-replay.cpp LABEL "turbo")

if (NOT WIN32 OR NOT BUILD_SHARED_LIBS)
    # these tests are disabled on Windows because they use internal functions not exported with LLAMA_API (when building with shared libraries)
    llama_build_and_test(test-sampling.cpp)
    llama_build_and_test(test-reasoning-budget.cpp)
    llama_build_and_test(test-grammar-parser.cpp)
    llama_build_and_test(test-grammar-integration.cpp)
    llama_build_and_test(test-llama-grammar.cpp)
    llama_build_and_test(test-batch-alloc.cpp)
    llama_build_and_test(test-tp5-ubatch-pool.cpp LABEL "main")
    llama_build_and_test(test-chat.cpp WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    target_include_directories(test-chat PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    target_link_libraries(test-chat PRIVATE server-context)
    # TODO: disabled on loongarch64 because the ggml-ci node lacks Python 3.8
    if (NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES "loongarch64")
        llama_build_and_test(test-json-schema-to-grammar.cpp   WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
        target_include_directories(test-json-schema-to-grammar PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    endif()

    if (NOT GGML_BACKEND_DL)
        llama_build(test-quantize-stats.cpp)
    endif()

    llama_build(test-gbnf-validator.cpp)

    # build test-tokenizer-1-bpe target once and add many tests
    llama_build(test-tokenizer-1-bpe.cpp)

    # TODO: disabled due to slowness
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-aquila    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-aquila.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-falcon    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-falcon.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-gpt-2     ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-2.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-gpt-neox  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-neox.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-llama-bpe ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf --ignore-merges)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-mpt       ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-mpt.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-refact    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-refact.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-starcoder ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-starcoder.gguf)

    # build test-tokenizer-1-spm target once and add many tests
    llama_build(test-tokenizer-1-spm.cpp)

    llama_test(test-tokenizer-1-spm  NAME test-tokenizer-1-llama-spm ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-spm.gguf)
    #llama_test(test-tokenizer-1-spm  NAME test-tokenizer-1-baichuan  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-baichuan.gguf)

    # llama_build_and_test(test-double-float.cpp) # SLOW

    llama_build_and_test(test-llama-archs.cpp)

    set(MODEL_DIR "${CMAKE_CURRENT_BINARY_DIR}/test-models/")
    file(MAKE_DIRECTORY "${MODEL_DIR}")

    llama_test(
        test-llama-archs
        NAME test-generate-models
        LABEL main
        ARGS -o "${MODEL_DIR}"
    )
    set_tests_properties(test-generate-models PROPERTIES
        FIXTURES_SETUP generate-models
    )

    llama_test(
        test-recurrent-state-rollback
        LABEL main
        ARGS -m "${MODEL_DIR}/qwen35-dense.gguf"
    )
    set_tests_properties(test-recurrent-state-rollback PROPERTIES
        FIXTURES_REQUIRED generate-models
    )
endif()

llama_build_and_test(test-chat-peg-parser.cpp peg-parser/simple-tokenize.cpp)
llama_build_and_test(test-jinja.cpp)
llama_test(test-jinja NAME test-jinja-py ARGS -py LABEL python)
llama_build_and_test(test-chat-auto-parser.cpp WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
llama_build_and_test(test-chat-template.cpp)
llama_build_and_test(test-log.cpp)

if (LLAMA_BUILD_TOOLS AND NOT EMSCRIPTEN)
    # decision API library (tools/decision); kept outside the WIN32 shared block on purpose
    llama_build_and_test(test-decision-json.cpp ARGS ${PROJECT_SOURCE_DIR}/tests/decision/golden)
    llama_build_and_test(test-decision-request.cpp)
    llama_build_and_test(test-decision-calib.cpp ARGS ${PROJECT_SOURCE_DIR}/tests/decision)
    llama_build_and_test(test-decision-router.cpp)
    llama_build_and_test(test-decision-reference.cpp ARGS ${PROJECT_SOURCE_DIR}/tests/decision/golden/router_cases.jsonl)
    # clef prompt against encode_record of the model repo (tests/clef/gen_golden.py), with the vocabulary of
    # Cloudflare/clef-flash (convert_hf_to_gguf.py --vocab-only; models/ggml-vocab-qwen35.gguf has the Qwen3 vocabulary)
    add_test(NAME test-decision-clef-vocab
             COMMAND ${CMAKE_COMMAND} -E tar xzf ${PROJECT_SOURCE_DIR}/tests/clef/ggml-vocab-clef.tar.gz
             WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    set_tests_properties(test-decision-clef-vocab PROPERTIES FIXTURES_SETUP clef-vocab LABELS main)
    llama_build_and_test(test-decision-clef.cpp ARGS ${CMAKE_CURRENT_BINARY_DIR}/ggml-vocab-clef.gguf ${PROJECT_SOURCE_DIR}/tests/clef/golden/prompts.jsonl)
    set_tests_properties(test-decision-clef PROPERTIES FIXTURES_REQUIRED clef-vocab)
    foreach (test_target test-decision-json test-decision-request test-decision-calib test-decision-router test-decision-reference test-decision-clef)
        target_link_libraries(${test_target} PRIVATE decision)
    endforeach()

    # llama-laya-cli -m <checkpoint dir>: GGUF cache (converted, then cache hit), Python bytes, same output as -m GGUF
    add_test(NAME test-decision-checkpoint-cli
             COMMAND ${CMAKE_COMMAND} -DCLI=$<TARGET_FILE:llama-laya-cli> -DFIXTURES=${PROJECT_SOURCE_DIR}/tests/laya/convert
                     "-DINPUTS=${PROJECT_SOURCE_DIR}/tests/laya/demo_input.json|${PROJECT_SOURCE_DIR}/tests/laya/golden/score_en.json"
                     -DWORK=${CMAKE_CURRENT_BINARY_DIR}/decision-checkpoint-cli
                     -P ${PROJECT_SOURCE_DIR}/tests/decision/checkpoint-cli.cmake)
    set_tests_properties(test-decision-checkpoint-cli PROPERTIES LABELS main)

    # laya HF checkpoint -> GGUF converter (tools/laya/laya-convert): tiny fixtures vs Python golden sha256
    llama_build_and_test(test-laya-convert.cpp ${PROJECT_SOURCE_DIR}/examples/gguf-hash/deps/sha256/sha256.c
                         ARGS ${PROJECT_SOURCE_DIR}/tests/laya/convert ${CMAKE_CURRENT_BINARY_DIR}/laya-convert-scratch)
    target_link_libraries     (test-laya-convert PRIVATE laya-convert)
    target_include_directories(test-laya-convert PRIVATE ${PROJECT_SOURCE_DIR}/vendor ${PROJECT_SOURCE_DIR}/examples/gguf-hash/deps)

    # the same cases against a fresh Python conversion (exit 77 = skip without torch / transformers; LAYA_REF_PYTHON picks the interpreter)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if (Python3_Interpreter_FOUND)
        add_test(NAME test-laya-convert-py
                 COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tests/laya/make_tiny_hf_laya.py --check $<TARGET_FILE:llama-laya-convert>)
        set_tests_properties(test-laya-convert-py PROPERTIES LABELS python SKIP_RETURN_CODE 77)
    endif()

    # laya GPU backends (Phase 4b): exit code 77 = no GPU / iGPU device in the build (skipped)
    llama_build_and_test(test-laya-backend-precision.cpp LABEL laya)
    target_link_libraries(test-laya-backend-precision PRIVATE laya)
    set_tests_properties(test-laya-backend-precision PROPERTIES SKIP_RETURN_CODE 77)

    # the replay test needs the tiny random laya GGUF, which a Python fixture generates (numpy, pyyaml:
    # tools/server/tests/requirements.txt); -DPython3_EXECUTABLE=<venv>/bin/python picks the interpreter
    llama_build(test-laya-backend-parity.cpp)
    target_link_libraries(test-laya-backend-parity PRIVATE laya)
    set(LAYA_TINY_PY_OK OFF)
    if (Python3_Interpreter_FOUND)
        # the fixture imports numpy and gguf-py (which needs pyyaml): without them the test is not registered
        execute_process(COMMAND ${Python3_EXECUTABLE} -c "import numpy, yaml"
                        RESULT_VARIABLE LAYA_TINY_PY_RC OUTPUT_QUIET ERROR_QUIET)
        if (LAYA_TINY_PY_RC EQUAL 0)
            set(LAYA_TINY_PY_OK ON)
        else()
            message(STATUS "test-laya-backend-parity: ${Python3_EXECUTABLE} has no numpy / pyyaml, not registered")
        endif()
    endif()
    if (LAYA_TINY_PY_OK)
        set(LAYA_TINY_GGUF "${CMAKE_CURRENT_BINARY_DIR}/laya-tiny-backend.gguf")
        add_test(NAME test-laya-tiny-gguf
                 COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tests/decision/make_tiny_laya.py ${LAYA_TINY_GGUF})
        set_tests_properties(test-laya-tiny-gguf PROPERTIES FIXTURES_SETUP laya-tiny LABELS laya)
        llama_test(test-laya-backend-parity LABEL laya
                   ARGS ${LAYA_TINY_GGUF} ${PROJECT_SOURCE_DIR}/tests/laya/parity/tiers.json)
        set_tests_properties(test-laya-backend-parity PROPERTIES SKIP_RETURN_CODE 77 FIXTURES_REQUIRED laya-tiny)
    endif()
endif()

llama_build_and_test(
    test-peg-parser.cpp
    peg-parser/simple-tokenize.cpp
    peg-parser/test-basic.cpp
    peg-parser/test-gbnf-generation.cpp
    peg-parser/test-json-parser.cpp
    peg-parser/test-json-serialization.cpp
    peg-parser/test-python-dict-parser.cpp
    peg-parser/test-unicode.cpp
    peg-parser/tests.h
)

if (NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES "s390x")
    set(MODEL_NAME "tinyllamas/stories15M-q4_0.gguf")
    set(MODEL_HASH "SHA256=66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739")
else()
    set(MODEL_NAME "tinyllamas/stories15M-be.Q4_0.gguf")
    set(MODEL_HASH "SHA256=9aec857937849d976f30397e97eb1cabb53eb9dcb1ce4611ba8247fb5f44c65d")
endif()
set(MODEL_DEST "${CMAKE_BINARY_DIR}/${MODEL_NAME}")

add_test(NAME test-download-model COMMAND ${CMAKE_COMMAND}
    -DDEST=${MODEL_DEST}
    -DNAME=${MODEL_NAME}
    -DHASH=${MODEL_HASH}
    -P ${CMAKE_SOURCE_DIR}/cmake/download-models.cmake
)
set_tests_properties(test-download-model PROPERTIES FIXTURES_SETUP test-download-model)

llama_build_and_test(test-thread-safety.cpp ARGS -m "${MODEL_DEST}" -ngl 99 -p "The meaning of life is" -n 128 -c 256 -ub 32 -np 4 -t 2)
set_tests_properties(test-thread-safety PROPERTIES FIXTURES_REQUIRED test-download-model)

llama_build_and_test(test-arg-parser.cpp)
if (NOT BUILD_SHARED_LIBS)
    llama_build_and_test(test-kv-stream-plan.cpp ${PROJECT_SOURCE_DIR}/src/llama-kv-stream-plan.cpp)
    target_include_directories(test-kv-stream-plan PRIVATE ${PROJECT_SOURCE_DIR}/src)
    llama_build_and_test(test-kv-stream-config.cpp ${PROJECT_SOURCE_DIR}/src/llama-kv-stream-config.cpp)
    target_include_directories(test-kv-stream-config PRIVATE ${PROJECT_SOURCE_DIR}/src)
endif()
llama_build_and_test(test-kv-stream-softmax.cpp llama-kv-stream-softmax.cpp)
if (GGML_CUDA)
    llama_build_and_test(
        test-kv-stream-bench-config.cpp
        ${PROJECT_SOURCE_DIR}/tools/kv-stream-bench/kv-stream-bench-config.cpp)
    target_include_directories(test-kv-stream-bench-config PRIVATE ${PROJECT_SOURCE_DIR}/tools/kv-stream-bench)
    llama_build_and_test(test-kv-stream-cuda-buffer.cpp LABEL cuda)
    target_include_directories(test-kv-stream-cuda-buffer PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src/ggml-cuda)
    # The test exits 77 when no CUDA device is present (e.g. a CUDA build run
    # on a GPU-less CI runner). ctest only treats that as a skip when
    # SKIP_RETURN_CODE says so; without it the skip is recorded as a failure.
    set_tests_properties(test-kv-stream-cuda-buffer PROPERTIES SKIP_RETURN_CODE 77)
    llama_build_and_test(test-kv-stream-cuda-attn.cpp LABEL cuda)
    target_include_directories(test-kv-stream-cuda-attn PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src/ggml-cuda)
    llama_build_and_test(test-kv-stream-cuda-set-rows.cpp LABEL cuda)
    target_include_directories(test-kv-stream-cuda-set-rows PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src/ggml-cuda)
endif()
llama_build_and_test(test-model-resolution.cpp)
# the test serves its repos from an httplib server, and the library links it privately
target_link_libraries(test-model-resolution PRIVATE cpp-httplib)
llama_build_and_test(test-speculative-adaptive.cpp)

if (NOT LLAMA_SANITIZE_ADDRESS AND NOT GGML_SCHED_NO_REALLOC)
  # TODO: repair known memory leaks
  llama_build_and_test(test-opt.cpp)
endif()
llama_build_and_test(test-backend-ops.cpp)
if (NOT LLAMA_USE_SYSTEM_GGML)
    llama_build(test-moe-cache.cpp)
    if (GGML_CUDA)
        # The test compares the cache's budget grants against cudaMemGetInfo,
        # the same query the cache uses. On UMA devices (GB10, Jetson) the
        # backend device memory API reports /proc/meminfo instead, so the
        # reserve calculation must match the cache's own view.
        find_package(CUDAToolkit QUIET)
        if (CUDAToolkit_FOUND)
            target_compile_definitions(test-moe-cache PRIVATE GGML_CUDA=1)
            target_include_directories(test-moe-cache PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
            target_link_libraries(test-moe-cache PRIVATE CUDA::cudart)
        endif()
    endif()
    llama_test(test-moe-cache LABEL "main;backend")
    # The test exits 77 when no CUDA device is present. ctest only treats that
    # as a skip when SKIP_RETURN_CODE says so; without it, every CPU-only
    # runner records the skip as a failure.
    set_tests_properties(test-moe-cache PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
endif()

llama_build_and_test(test-moe-cache-fit.cpp)

# Scheduler MoE-cache session policy on the real CPU scheduler: default
# session replacement, OFF, dormant, and the unproven refusal. The provider is
# a real session implementation; every assertion reads the scheduler's own
# observable mode/budget/session state.
llama_build_and_test(test-moe-cache-session.cpp LABEL "cpu")
set_tests_properties(test-moe-cache-session PROPERTIES
    TIMEOUT 60
    SKIP_RETURN_CODE 77
    PASS_REGULAR_EXPRESSION "MOE_SESSION_TEST PASS")

# Explicit ON is a requirement at the real llama_init_from_model entry: a dense
# model must fail context creation instead of silently resolving to OFF. AUTO
# and the unspecified default may resolve to OFF. Model-gated; skips without
# LLAMA_TEST_MODEL.
llama_build_and_test(test-moe-cache-context-policy.cpp LABEL "cpu")
set_tests_properties(test-moe-cache-context-policy PROPERTIES
    TIMEOUT 120
    SKIP_RETURN_CODE 77
    PASS_REGULAR_EXPRESSION "MOE_CONTEXT_POLICY_TEST PASS")

if (TARGET ggml-cpu AND NOT LLAMA_USE_SYSTEM_GGML AND NOT GGML_BACKEND_DL AND (NOT WIN32 OR NOT BUILD_SHARED_LIBS))
    add_executable(test-cpu-attn-pad test-cpu-attn-pad.cpp)
    target_link_libraries(test-cpu-attn-pad PRIVATE ggml-cpu ggml-base)
    # One binary, four arms. Each arm prints its PASS line only after every
    # assertion in it returned true, so the line is the arm's own verdict. A
    # binary that exits 0 without running the arm -- an early return, a
    # skipped loop, a guard that turned the body into a no-op -- cannot print
    # it, and the registration below fails. Without the regex, exit 0 alone
    # would be enough to pass.
    #
    # The expressions are unanchored substrings on purpose: CMake applies them
    # with cmsys::RegularExpression, whose `$` matches end-of-string rather
    # than end-of-line, so an anchor would silently never match a line inside
    # a multi-line output.
    #
    # The default arm prints two lines; the registration names the second, the
    # one that covers the 608-width sweep. The first line (the padded helper's
    # live lanes) is printed just above it in the same run.
    add_test(NAME test-cpu-attn-pad COMMAND test-cpu-attn-pad)
    add_test(NAME test-cpu-attn-f16-kqv COMMAND test-cpu-attn-pad --f16-kqv)
    add_test(NAME test-cpu-attn-true-len COMMAND test-cpu-attn-pad --true-len)
    add_test(NAME test-cpu-attn-true-len-counters COMMAND test-cpu-attn-pad --counters)
    add_test(NAME test-cpu-attn-masked-gemm COMMAND test-cpu-attn-pad --masked-gemm)
    set_tests_properties(test-cpu-attn-pad PROPERTIES LABELS "main;cpu;attn"
        PASS_REGULAR_EXPRESSION "PASS: 608 widths, three reference widths each, guarded active rows")
    set_tests_properties(test-cpu-attn-f16-kqv PROPERTIES LABELS "main;cpu;attn"
        PASS_REGULAR_EXPRESSION "PASS: F16 ordered kqv rows match reference width bitwise")
    set_tests_properties(test-cpu-attn-true-len PROPERTIES LABELS "main;cpu;attn"
        PASS_REGULAR_EXPRESSION "PASS: TRUE_LEN_REDUCE matches a live-width graph bitwise")
    set_tests_properties(test-cpu-attn-true-len-counters PROPERTIES LABELS "main;cpu;attn"
        PASS_REGULAR_EXPRESSION "PASS: TRUE_LEN_REDUCE counters follow the runtime length")
    set_tests_properties(test-cpu-attn-masked-gemm PROPERTIES LABELS "main;cpu;attn"
        PASS_REGULAR_EXPRESSION "PASS: masked-tail SGEMM reduces over the live length only")
    add_executable(test-cpu-attn-mask test-cpu-attn-mask.cpp)
    target_link_libraries(test-cpu-attn-mask PRIVATE ggml-cpu ggml-base)
    if (MSVC)
        target_compile_options(test-cpu-attn-pad PRIVATE /fp:strict)
        target_compile_options(test-cpu-attn-mask PRIVATE /fp:strict)
    elseif (CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang|IntelLLVM)$")
        target_compile_options(test-cpu-attn-pad PRIVATE -fno-fast-math -ffp-contract=off)
        target_compile_options(test-cpu-attn-mask PRIVATE -fno-fast-math -ffp-contract=off)
    endif()
    add_test(NAME test-cpu-attn-mask COMMAND test-cpu-attn-mask)
    add_test(NAME test-cpu-attn-mask-widths COMMAND test-cpu-attn-mask --widths)
    add_test(NAME test-cpu-attn-mask-kq-dispatch COMMAND test-cpu-attn-mask --kq-dispatch)
    add_executable(test-ntok-replan test-ntok-replan.cpp)
    target_link_libraries(test-ntok-replan PRIVATE ggml ggml-cpu)
    add_test(NAME test-ntok-replan COMMAND test-ntok-replan)
    add_test(NAME test-cpu-attn-ordered-kq COMMAND test-cpu-attn-mask --ordered-kq)
    # The union binder narrows a live graph in place (ne[], nb[], view_offs).
    # This test guards that patch with three arms, none of which marks a node:
    #   STOCK   built at the envelope then patched COMPLETELY to width n must
    #           equal a graph built at n. A complete patch is an identity by
    #           construction, so this arm guards the patch rather than proving
    #           equality; the equality argument is ANALYSIS-EQUALITY.md section C.
    #   STRIDE  ne[] narrowed but nb[] left at the envelope stride must be
    #           DETECTED (it runs in a child process: ggml asserts on the
    #           inconsistent layout).
    #   OLD     two widths straddling a tinyBLAS tile class must differ
    #           somewhere, so the comparison is not blind to a kernel change.
    add_executable(test-cpu-kernel-sel-len-indep test-cpu-kernel-sel-len-indep.cpp)
    target_link_libraries(test-cpu-kernel-sel-len-indep PRIVATE ggml-cpu ggml-base)
    if (MSVC)
        target_compile_options(test-cpu-kernel-sel-len-indep PRIVATE /fp:strict)
    elseif (CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang|IntelLLVM)$")
        target_compile_options(test-cpu-kernel-sel-len-indep PRIVATE -fno-fast-math -ffp-contract=off)
    endif()
    add_test(NAME test-cpu-kernel-sel-len-indep COMMAND test-cpu-kernel-sel-len-indep)
endif()

# One ctest entry for the whole union mainline: the four-parameter matrix, the
# binder-vs-stock equality, the n_tokens trajectory and the kernel-selection
# probe, read in one place by nkv-cpu-probe/union-mainline-gate-r24o.sh. The
# script runs the four binaries directly (never `ctest -V`: ctest's
# "<index>: " prefix defeats its anchored greps) and enforces every numeric
# equality that CMake's PASS_REGULAR_EXPRESSION cannot compare.
#
# The block sits here, after the last of the four targets is declared, and not
# next to the four-parameter matrix registration near the top of the file:
# `if (TARGET ...)` is evaluated in file order, and $<TARGET_FILE:...> of a
# target that has not been declared yet is a generate-time error. At the matrix
# registration only test-union-four-param-matrix and
# test-union-binder-stock-equiv exist; test-ntok-onegraph is declared later
# under if (UNIX) and test-cpu-kernel-sel-len-indep under the ggml-cpu guard.
# The guard is still needed: a configuration that skips either of those two
# blocks cannot run the aggregate gate, and not registering it is better than
# failing to configure.
#
# LLAMA_NTOK_TEST_ROUNDS is set above, inside if (UNIX), with a cache default
# of 50. The script's own ${6:-2} fallback applies only when it is called by
# hand with an empty sixth argument.
if (LLAMA_NTOK_TEST_MODEL)
    if (TARGET test-ntok-onegraph AND TARGET test-cpu-kernel-sel-len-indep)
        add_test(NAME test-union-mainline-gate
                 COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/nkv-cpu-probe/union-mainline-gate-r24o.sh
                         $<TARGET_FILE:test-union-four-param-matrix>
                         $<TARGET_FILE:test-union-binder-stock-equiv>
                         $<TARGET_FILE:test-ntok-onegraph>
                         $<TARGET_FILE:test-cpu-kernel-sel-len-indep>
                         "${LLAMA_NTOK_TEST_MODEL}"
                         "${LLAMA_NTOK_TEST_ROUNDS}")
        set_tests_properties(test-union-mainline-gate PROPERTIES
            LABELS "model;cpu;union"
            TIMEOUT 1800
            PASS_REGULAR_EXPRESSION "UNION_MAINLINE_GATE ok=1")
    endif()
endif()

# The shell half of the tri-state gate contract, model-free: it feeds
# synthetic logs to the real check_four_matrix / check_bse / check_ntok /
# check_kernel_sel and asserts PASS, FAIL and UNPROVEN for each false-pass
# shape. No GGUF, no build, no ctest. The script needs bash, so the
# registration is UNIX-only.
#
# No PASS_REGULAR_EXPRESSION: the script asserts every case and exits
# non-zero on any mismatch. A PASS regex would make ctest ignore that exit
# status, which is the defect this test exists to catch.
if (UNIX)
    add_test(NAME test-union-gate-verdict-selftest
             COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/nkv-cpu-probe/union-gate-verdict-selftest.sh)
    set_tests_properties(test-union-gate-verdict-selftest PROPERTIES
        LABELS "cpu;union"
        TIMEOUT 300)
endif()

add_executable(test-flash-attn-bias test-flash-attn-bias.cpp)
target_link_libraries(test-flash-attn-bias PRIVATE ggml)
add_test(NAME test-flash-attn-bias COMMAND test-flash-attn-bias)

add_executable(test-flash-attn-generic-hash test-flash-attn-generic-hash.cpp)
target_link_libraries(test-flash-attn-generic-hash PRIVATE ggml)
add_test(NAME test-flash-attn-generic-hash COMMAND test-flash-attn-generic-hash)

llama_build_and_test(test-model-load-cancel.cpp LABEL "model")
llama_build_and_test(test-autorelease.cpp       LABEL "model")
llama_build_and_test(test-backend-sampler.cpp   LABEL "model")

# Test for state restore with fragmented KV cache
# Requires a model, uses same args pattern as test-thread-safety
llama_build_and_test(test-state-restore-fragmented.cpp LABEL "model" ARGS -m "${MODEL_DEST}")
set_tests_properties(test-state-restore-fragmented PROPERTIES FIXTURES_REQUIRED test-download-model)

# Test state save/load functionality
llama_build_and_test(test-save-load-state.cpp LABEL "model" ARGS -m "${MODEL_DEST}")
set_tests_properties(test-save-load-state PROPERTIES FIXTURES_REQUIRED test-download-model)

if (APPLE)
    llama_build(test-rset-release.cpp)
endif()

if (NOT GGML_BACKEND_DL)
    # these tests use the backends directly and cannot be built with dynamic loading
    llama_build_and_test(test-barrier.cpp)
    llama_build_and_test(test-quantize-fns.cpp)
    llama_build_and_test(test-quantize-perf.cpp)
    llama_build_and_test(test-rope.cpp)
    llama_build_and_test(test-col2im-1d.cpp)
endif()

# libmtmd
set(LLAMA_TEST_NAME test-mtmd-c-api)
llama_build_and_test(test-mtmd-c-api.c)
target_link_libraries(${LLAMA_TEST_NAME} PRIVATE mtmd)
unset(LLAMA_TEST_NAME)

# GGUF model data fetcher library for tests that need real model metadata
# Only compile when cpp-httplib has SSL support (CPPHTTPLIB_OPENSSL_SUPPORT)
if (TARGET cpp-httplib)
    get_target_property(_cpp_httplib_defs cpp-httplib INTERFACE_COMPILE_DEFINITIONS)
    if (_cpp_httplib_defs MATCHES "CPPHTTPLIB_OPENSSL_SUPPORT")
        add_library(gguf-model-data STATIC gguf-model-data.cpp)
        target_link_libraries(gguf-model-data PRIVATE llama-common cpp-httplib)
        target_include_directories(gguf-model-data PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

        add_executable(test-gguf-model-data test-gguf-model-data.cpp)
        target_link_libraries(test-gguf-model-data PRIVATE gguf-model-data llama-common)
        llama_test(test-gguf-model-data LABEL "model")

        # test-quant-type-selection requires gguf-model-data for remote model metadata
        llama_build_and_test(test-quant-type-selection.cpp LABEL "model")
        target_link_libraries(test-quant-type-selection PRIVATE gguf-model-data)
        target_compile_definitions(test-quant-type-selection PRIVATE
            SNAPSHOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/snapshots")
    endif()
endif()

# dummy executable - not installed
get_filename_component(TEST_TARGET test-c.c NAME_WE)
add_executable(${TEST_TARGET} test-c.c)
target_link_libraries(${TEST_TARGET} PRIVATE llama)

# TurboQuant WHT round-trip test (links ggml turbo codec symbols via llama)
llama_build_and_test(test-turbo-quant.c LABEL "turbo")

if (NOT LLAMA_USE_SYSTEM_GGML)
    # Needs non-public ggml-impl.h
    llama_build_and_test(test-gguf.cpp)

    # Needs non-public ggml{,-backend}-impl.h
    llama_build_and_test(test-alloc.cpp)
endif()

llama_build(test-export-graph-ops.cpp)
if (TARGET gguf-model-data)
    target_link_libraries(test-export-graph-ops PRIVATE gguf-model-data)
    target_compile_definitions(test-export-graph-ops PRIVATE LLAMA_HF_FETCH)
endif()

# TP5 WS4 scheduler submission-boundary CPU test. It drives the real
# scheduler's split input copy through a fake backend with its own host buffer
# type, so the scheduler must call cpy_tensor_async and then ask the optional
# submission-state capability. Covers success, rejected, and fatal (unknown)
# submissions, no replay after fatal, and copy-slot ownership. No GPU.
add_executable(test-tp5-staging-scheduler test-tp5-staging-scheduler.cpp)
target_link_libraries(test-tp5-staging-scheduler PRIVATE ggml ggml-cpu)
add_test(NAME test-tp5-staging-scheduler COMMAND test-tp5-staging-scheduler)
set_property(TEST test-tp5-staging-scheduler PROPERTY LABELS "cpu;tp5")
set_property(TEST test-tp5-staging-scheduler PROPERTY PASS_REGULAR_EXPRESSION "TP5_STAGING_SCHED_TEST PASS")

# TP5 staging submit-gate / note_source lock ownership. The meta fallback holds
# the submit gate while it calls the backend set path, and a CUDA simple tensor
# on that path calls note_source on the same thread. The test pins that the
# same-thread note does not try_lock the owned gate, does not drop it, and does
# not take mutex_ while a fatal publication waits for the gate. CPU only.
add_executable(test-tp5-staging-note-source-lock test-tp5-staging-note-source-lock.cpp)
target_link_libraries(test-tp5-staging-note-source-lock PRIVATE ggml ggml-cpu)
add_test(NAME test-tp5-staging-note-source-lock COMMAND test-tp5-staging-note-source-lock)
set_property(TEST test-tp5-staging-note-source-lock PROPERTY LABELS "cpu;tp5;staging")
set_property(TEST test-tp5-staging-note-source-lock PROPERTY PASS_REGULAR_EXPRESSION "TP5_STAGING_NOTE_SOURCE_LOCK PASS")
set_property(TEST test-tp5-staging-note-source-lock PROPERTY TIMEOUT 60)

# TP5 staging pool slot lifecycle. Header-only pool with injected host/event
# primitives: lease reuse, pending retire, no_slot attribution, per-kind and
# per-window counters, union kinds, and partial/fatal fail-stop. CPU only.
add_executable(test-tp5-staging-pool test-tp5-staging-pool.cpp)
target_link_libraries(test-tp5-staging-pool PRIVATE ggml ggml-cpu)
add_test(NAME test-tp5-staging-pool COMMAND test-tp5-staging-pool)
set_property(TEST test-tp5-staging-pool PROPERTY LABELS "cpu;tp5;staging")
set_property(TEST test-tp5-staging-pool PROPERTY PASS_REGULAR_EXPRESSION "45/45 passed")
set_property(TEST test-tp5-staging-pool PROPERTY TIMEOUT 60)

# TP5 staging Meta partial-enqueue CPU test. A real Meta backend over CPU
# child devices receives an injected v2 staging api table whose enqueue queues
# rank 0 and then cleanly rejects rank 1. The Meta layer must call
# mark_partial, not return a silent success, and the next set must hit the
# fatal submit gate. No GPU.
add_executable(test-tp5-staging-meta-partial test-tp5-staging-meta-partial.cpp)
target_link_libraries(test-tp5-staging-meta-partial PRIVATE ggml ggml-cpu)
add_test(NAME test-tp5-staging-meta-partial COMMAND test-tp5-staging-meta-partial)
set_property(TEST test-tp5-staging-meta-partial PROPERTY LABELS "cpu;tp5;staging")
set_property(TEST test-tp5-staging-meta-partial PROPERTY PASS_REGULAR_EXPRESSION "TP5_STAGING_META_PARTIAL_TEST PASS")

# TP5 staging synchronous source identity. The scheduler's blocking input
# fallback reaches the Meta buffer through set_tensor, which only sees the
# internal copy destination (no input flag with n_copies == 1, decorated
# name). The buffer cpy_tensor entry carries the original source identity.
# The test covers the source kind, mirrored bytes, a non-input source, a
# direct set without source identity, and a partial mirrored submission that
# must not be replayed. CPU only.
add_executable(test-tp5-staging-sync-src test-tp5-staging-sync-src.cpp)
target_link_libraries(test-tp5-staging-sync-src PRIVATE ggml ggml-cpu)
add_test(NAME test-tp5-staging-sync-src COMMAND test-tp5-staging-sync-src)
set_property(TEST test-tp5-staging-sync-src PROPERTY LABELS "cpu;tp5;staging")
set_property(TEST test-tp5-staging-sync-src PROPERTY PASS_REGULAR_EXPRESSION "TP5_STAGING_SYNC_SRC_TEST PASS")

# TP5 staging rejection-summary diagnostics. The first example of each
# (lane, reason) class is a self-owned copy: the source tensor is renamed and
# freed before the dump. CPU only.
add_executable(test-tp5-staging-reject-summary test-tp5-staging-reject-summary.cpp)
target_link_libraries(test-tp5-staging-reject-summary PRIVATE ggml ggml-cpu)
add_test(NAME test-tp5-staging-reject-summary COMMAND test-tp5-staging-reject-summary)
set_property(TEST test-tp5-staging-reject-summary PROPERTY LABELS "cpu;tp5;staging")
set_property(TEST test-tp5-staging-reject-summary PROPERTY PASS_REGULAR_EXPRESSION "TP5_STAGING_REJECT_TEST PASS")

# TP5 staging rejection-summary concurrency. Independent Meta backends record
# rejections from several threads while a dump thread reads the summary. The
# hot path is one relaxed atomic plus a one-time example publication; the dump
# holds no lock across its print. CPU only.
add_executable(test-tp5-staging-reject-concurrent test-tp5-staging-reject-concurrent.cpp)
target_link_libraries(test-tp5-staging-reject-concurrent PRIVATE ggml ggml-cpu)
add_test(NAME test-tp5-staging-reject-concurrent COMMAND test-tp5-staging-reject-concurrent)
set_property(TEST test-tp5-staging-reject-concurrent PROPERTY LABELS "cpu;tp5;staging")
set_property(TEST test-tp5-staging-reject-concurrent PROPERTY PASS_REGULAR_EXPRESSION "TP5_STAGING_REJECT_CONCURRENT_TEST PASS")

# TP5 WS4 pending-context/model ownership test. llama_free retains the whole
# context (and defers llama_model_free) while an async submission is unproven;
# a clean context/model takes the normal destruction path. The context part
# needs LLAMA_TEST_MODEL; the model retention part always runs.
add_executable(test-tp5-context-retain test-tp5-context-retain.cpp)
target_link_libraries(test-tp5-context-retain PRIVATE llama)
target_include_directories(test-tp5-context-retain PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME test-tp5-context-retain COMMAND test-tp5-context-retain)
set_property(TEST test-tp5-context-retain PROPERTY LABELS "cpu;tp5")
set_property(TEST test-tp5-context-retain PROPERTY PASS_REGULAR_EXPRESSION "TP5_CONTEXT_RETAIN_TEST PASS")

# TP5 factor-prefab sub-envelope KQ-mask regression. A ubatch with
# n_tokens < n_env runs on an n_env-row envelope; the mask fill must write only
# the real rows and the wrapper must pad the rest. The test compares stock,
# exact-envelope and padded-envelope logits for one real token sequence, and
# requires the padded arm's n_sub_envelope/n_pad_rows counters with no stock
# fallback. Needs LLAMA_TEST_MODEL; without it the arm is skipped.
add_executable(test-tp5-prefab-subenvelope test-tp5-prefab-subenvelope.cpp)
target_link_libraries(test-tp5-prefab-subenvelope PRIVATE llama)
add_test(NAME test-tp5-prefab-subenvelope COMMAND test-tp5-prefab-subenvelope)
set_property(TEST test-tp5-prefab-subenvelope PROPERTY LABELS "cpu;tp5;prefab")
set_property(TEST test-tp5-prefab-subenvelope PROPERTY PASS_REGULAR_EXPRESSION "PREFAB_SUBENVELOPE_TEST PASS")
set_property(TEST test-tp5-prefab-subenvelope PROPERTY FAIL_REGULAR_EXPRESSION "^FAIL:")

# Meta host<->shard virtual-address copy test. CPU Meta buffers can report
# is_host=true while the top-level tensor data is a sentinel; the generic copy
# must route through the real simple-tensor bytes. Covers host<->Meta non-view
# and non-zero cross-rank views, different splits, empty and alias semantics.
add_executable(test-meta-host-view-copy test-meta-host-view-copy.cpp)
target_link_libraries(test-meta-host-view-copy PRIVATE llama)
add_test(NAME test-meta-host-view-copy COMMAND test-meta-host-view-copy)
set_property(TEST test-meta-host-view-copy PROPERTY LABELS "cpu;meta")
set_property(TEST test-meta-host-view-copy PROPERTY PASS_REGULAR_EXPRESSION "META_HOST_VIEW_COPY_TEST PASS")
set_property(TEST test-meta-host-view-copy PROPERTY FAIL_REGULAR_EXPRESSION "FAIL ")
# TP5 staging owner-runtime retention. Real QWEN4EXP target/draft contexts
# (self-generated CPU fixtures) drive the per-run retainer: old busy + new run,
# release failure, delayed completion and record capacity. The staging
# provider is a real pool with controllable events. The test takes the target
# and draft GGUF paths; register it only when the fixture paths are known.
add_executable(test-tp5-staging-owner-runtime test-tp5-staging-owner-runtime.cpp)
target_link_libraries(test-tp5-staging-owner-runtime PRIVATE llama)
target_include_directories(test-tp5-staging-owner-runtime PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/ggml/include
    ${PROJECT_SOURCE_DIR}/ggml/src
    ${PROJECT_SOURCE_DIR}/fork/tp5-compose/src)
if (MTP_DRAFTONLY_TARGET AND MTP_DRAFTONLY_DRAFT)
    add_test(NAME test-tp5-staging-owner-runtime
             COMMAND test-tp5-staging-owner-runtime "${MTP_DRAFTONLY_TARGET}" "${MTP_DRAFTONLY_DRAFT}")
    set_property(TEST test-tp5-staging-owner-runtime PROPERTY LABELS "model;cpu;tp5;staging")
    set_property(TEST test-tp5-staging-owner-runtime PROPERTY PASS_REGULAR_EXPRESSION "TP5_STAGING_OWNER_RUNTIME_TEST PASS")
    set_property(TEST test-tp5-staging-owner-runtime PROPERTY TIMEOUT 300)
endif()
if (LLAMA_TP5_TEST_HOOKS)
    add_executable(test-tp5-context-cleanup-concurrent test-tp5-context-cleanup-concurrent.cpp)
    target_link_libraries(test-tp5-context-cleanup-concurrent PRIVATE llama)
    target_include_directories(test-tp5-context-cleanup-concurrent PRIVATE
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/ggml/src
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-cuda
        ${PROJECT_SOURCE_DIR}/fork/tp5-compose/src)
endif()
