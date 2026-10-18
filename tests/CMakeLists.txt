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

# The probe runner must fail on failed checks, independently of GPU/model
# availability. Its tests use only the Python standard library.
find_package(Python3 QUIET COMPONENTS Interpreter)
if (Python3_Interpreter_FOUND)
    foreach(check nanbeige-audit nanbeige-report rerot-semantic-smoke tp5-gap-tools tp5-perf-model tp5-match-model)
        add_test(NAME test-${check}
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test-${check}.py)
        set_tests_properties(test-${check} PROPERTIES LABELS "python" TIMEOUT 30)
    endforeach()
    add_test(NAME test-rerot-dag-reference
        COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/scripts/rerot-dag-reference.py)
    set_tests_properties(test-rerot-dag-reference PROPERTIES LABELS "python" TIMEOUT 60)
    add_test(NAME test-gpu-recovery-mock
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test-gpu-recovery-mock.py)
    set_tests_properties(test-gpu-recovery-mock PROPERTIES LABELS "recovery;python" TIMEOUT 30)
endif()

# These tests need no model and never submit GPU work. Keep assertions enabled
# in optimized builds: they check exact selected tokens and RNG continuation.
add_executable(test-greedy-unique-host test-greedy-unique-host.cpp)
target_include_directories(test-greedy-unique-host PRIVATE ${PROJECT_SOURCE_DIR}/common)
target_link_libraries(test-greedy-unique-host PRIVATE llama ggml-base)
target_compile_features(test-greedy-unique-host PRIVATE cxx_std_17)
add_executable(test-decode-cost test-decode-cost.cpp)
target_include_directories(test-decode-cost PRIVATE ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/ggml/include)
target_compile_features(test-decode-cost PRIVATE cxx_std_17)
if (MSVC)
    target_compile_options(test-greedy-unique-host PRIVATE /UNDEBUG)
    target_compile_options(test-decode-cost PRIVATE /UNDEBUG)
else()
    target_compile_options(test-greedy-unique-host PRIVATE -UNDEBUG)
    target_compile_options(test-decode-cost PRIVATE -UNDEBUG)
endif()
llama_test(test-greedy-unique-host LABEL "sampling;cpu")
llama_test(test-decode-cost NAME test-decode-cost-off LABEL "performance;cpu")
set_tests_properties(test-decode-cost-off PROPERTIES ENVIRONMENT "GGML_DECODE_COST=0")
llama_test(test-decode-cost NAME test-decode-cost-on LABEL "performance;cpu" ARGS enabled)
set_tests_properties(test-decode-cost-on PROPERTIES ENVIRONMENT "GGML_DECODE_COST=1")

# build test-tokenizer-0 target once and add many tests
add_executable(test-predefined-capacity test-predefined-capacity.cpp)
target_link_libraries(test-predefined-capacity PRIVATE ggml-base)
llama_test(test-predefined-capacity LABEL "predefined;cpu")

add_executable(test-mtp-workspace test-mtp-workspace.cpp)
target_link_libraries(test-mtp-workspace PRIVATE llama llama-common)
llama_test(test-mtp-workspace LABEL "predefined;cpu")

add_test(NAME test-gpu-recovery COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/test-gpu-recovery.sh)
set_tests_properties(test-gpu-recovery PROPERTIES LABELS "recovery;cpu" TIMEOUT 15)

if (UNIX AND LLAMA_BUILD_COMMON AND LLAMA_BUILD_TOOLS AND LLAMA_BUILD_SERVER)
    add_executable(test-server-signal test-server-signal.cpp)
    target_include_directories(test-server-signal PRIVATE ${PROJECT_SOURCE_DIR}/tools/server ${PROJECT_SOURCE_DIR}/tools/mtmd)
    target_link_libraries(test-server-signal PRIVATE server-context)
    target_compile_features(test-server-signal PRIVATE cxx_std_17)
    llama_test(test-server-signal LABEL "server;cpu")
    set_tests_properties(test-server-signal PROPERTIES TIMEOUT 10)
endif()

add_executable(test-target-capacity test-target-capacity.cpp)
target_include_directories(test-target-capacity PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(test-target-capacity PRIVATE llama llama-common ggml-base)
llama_test(test-target-capacity LABEL "predefined;cpu")

add_executable(test-predefined-coverage test-predefined-coverage.cpp)
target_compile_features(test-predefined-coverage PRIVATE cxx_std_17)
llama_test(test-predefined-coverage LABEL "predefined;cpu")

add_executable(test-tp5-row-program test-tp5-row-program.cpp)
target_compile_features(test-tp5-row-program PRIVATE cxx_std_17)
llama_test(test-tp5-row-program LABEL "predefined;cpu")

add_executable(test-tp5-rank-local-readback test-tp5-rank-local-readback.cpp)
target_link_libraries(test-tp5-rank-local-readback PRIVATE ggml)
llama_test(test-tp5-rank-local-readback LABEL "predefined;cpu")

add_executable(test-turbo-quant test-turbo-quant.c)
target_include_directories(test-turbo-quant PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)
target_link_libraries(test-turbo-quant PRIVATE ggml-base)
if (NOT MSVC)
    target_link_libraries(test-turbo-quant PRIVATE m)
endif()
llama_test(test-turbo-quant)

add_executable(test-turbo-config test-turbo-config.cpp)
target_link_libraries(test-turbo-config PRIVATE ggml-base)
llama_test(test-turbo-config)

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

llama_build_and_test(test-gdn-ingredient-replay.cpp LABEL "turbo")

if (NOT WIN32 OR NOT BUILD_SHARED_LIBS)
    # these tests are disabled on Windows because they use internal functions not exported with LLAMA_API (when building with shared libraries)
    llama_build_and_test(test-sampling.cpp)
    llama_build_and_test(test-reasoning-budget.cpp)
    llama_build_and_test(test-grammar-parser.cpp)
    llama_build_and_test(test-grammar-integration.cpp)
    llama_build_and_test(test-llama-grammar.cpp)
    llama_build_and_test(test-batch-alloc.cpp)
    llama_build_and_test(test-kv-cells.cpp)
    llama_build_and_test(test-qsa-pooled-cache.cpp)
    llama_build_and_test(test-model-lazy-mmap.cpp)
    llama_build_and_test(test-rerot-view.cpp)
    target_include_directories(test-rerot-view PRIVATE ${PROJECT_SOURCE_DIR}/src)
    llama_build_and_test(test-rerot-span-expand.cpp)
    target_include_directories(test-rerot-span-expand PRIVATE ${PROJECT_SOURCE_DIR}/src)
    llama_build_and_test(test-rerot-q-prep.cpp)
    target_include_directories(test-rerot-q-prep PRIVATE ${PROJECT_SOURCE_DIR}/src)
    target_link_libraries(test-rerot-q-prep PRIVATE ggml-base llama)
    llama_build_and_test(test-rerot-ddvr.cpp)
    target_include_directories(test-rerot-ddvr PRIVATE ${PROJECT_SOURCE_DIR}/src)
    llama_build_and_test(test-rerot-math.cpp)
    llama_build_and_test(test-rerot-shared-block.cpp)
    target_include_directories(test-rerot-shared-block PRIVATE ${PROJECT_SOURCE_DIR}/src)
    target_link_libraries(test-rerot-shared-block PRIVATE ggml-base)
    # C10 / Q3 two-reader shared-KV prototype: CPU golden + optional glslc;
    # GPU dispatch deferred (skips cleanly when no device). No 5-GPU requirement.
    llama_build_and_test(test-rerot-q3-shared-kv-2reader.cpp)
    target_include_directories(test-rerot-q3-shared-kv-2reader PRIVATE ${PROJECT_SOURCE_DIR}/src)
    target_link_libraries(test-rerot-q3-shared-kv-2reader PRIVATE ggml-base)
    target_include_directories(test-rerot-math PRIVATE ${PROJECT_SOURCE_DIR}/src)
    # [Q7] F32 gate compares against ggml's own PQ2_0 block layout/decode
    # (ggml-common.h / ggml-quants.c pointers are declared there).
    target_include_directories(test-rerot-math PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)
    target_link_libraries(test-rerot-math PRIVATE ggml-base)
    # [Q9] joint sampler gate compares against the production
    # llama_sampler_* chain (same ordering / tie-break contract).
    target_link_libraries(test-rerot-math PRIVATE llama)
    llama_build_and_test(test-rerot-recurrent.cpp)
    target_include_directories(test-rerot-recurrent PRIVATE ${PROJECT_SOURCE_DIR}/src)
    target_link_libraries(test-rerot-recurrent PRIVATE llama)
    llama_build_and_test(test-rerot-profile.cpp)
    target_include_directories(test-rerot-profile PRIVATE ${PROJECT_SOURCE_DIR}/src)
    target_link_libraries(test-rerot-profile PRIVATE llama)
    llama_build_and_test(test-rerot-attn.cpp)
    target_include_directories(test-rerot-attn PRIVATE ${PROJECT_SOURCE_DIR}/src)
    target_link_libraries(test-rerot-attn PRIVATE llama)
    # Opt-in real-model teacher-forcing diagnostic; no model is downloaded
    # and this is not registered among the hermetic correctness tests.
    llama_build(test-rerot-model-batch.cpp)
    llama_build(test-rerot-model-single.cpp)
    llama_build(test-rerot-model-permute.cpp)
    llama_build(test-native-determinism.cpp)
    llama_build_and_test(test-rerot-parser.cpp
        ${PROJECT_SOURCE_DIR}/tools/server/server-rerot.cpp
        ${PROJECT_SOURCE_DIR}/tools/server/server-rerot-todo.cpp
        ${PROJECT_SOURCE_DIR}/tools/server/server-rerot-thinking.cpp
        ${PROJECT_SOURCE_DIR}/tools/server/server-rerot-mindmap.cpp)
    # MM-R1 mindmap wire parser (strict incremental parser, no llama deps)
    llama_build_and_test(test-rerot-mindmap-parser.cpp ${PROJECT_SOURCE_DIR}/tools/server/server-rerot-mindmap.cpp)
    target_include_directories(test-rerot-mindmap-parser PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    target_include_directories(test-rerot-parser PRIVATE
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/tools/server)
    llama_build_and_test(test-rerot-runtime.cpp
        ${PROJECT_SOURCE_DIR}/tools/server/server-rerot.cpp
        ${PROJECT_SOURCE_DIR}/tools/server/server-rerot-todo.cpp
        ${PROJECT_SOURCE_DIR}/tools/server/server-rerot-thinking.cpp
        ${PROJECT_SOURCE_DIR}/tools/server/server-rerot-mindmap.cpp)
    llama_build_and_test(test-rerot-todo.cpp
        ${PROJECT_SOURCE_DIR}/tools/server/server-rerot-todo.cpp)
    target_include_directories(test-rerot-todo PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    # Goal wire: flat HTML sub-goal list, GBNF-constrained, </ul>-terminated
    llama_build_and_test(test-rerot-thinking.cpp
        ${PROJECT_SOURCE_DIR}/tools/server/server-rerot-thinking.cpp)
    target_include_directories(test-rerot-thinking PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    target_include_directories(test-rerot-runtime PRIVATE
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/tools/server
        ${PROJECT_SOURCE_DIR}/tools/mtmd)
    target_link_libraries(test-rerot-runtime PRIVATE llama llama-common server-context)
    llama_build_and_test(test-triattention-score.cpp)
    target_include_directories(test-triattention-score PRIVATE ${PROJECT_SOURCE_DIR}/src)
    llama_build_and_test(test-triattention-kv.cpp)
    target_include_directories(test-triattention-kv PRIVATE ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/ggml/src)
    llama_build_and_test(test-server-triattention.cpp)
    target_include_directories(test-server-triattention PRIVATE ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/tools/server)

    foreach(component routing select state)
        llama_build_and_test(test-flashprefill-${component}.cpp)
        target_include_directories(test-flashprefill-${component} PRIVATE ${PROJECT_SOURCE_DIR}/src)
    endforeach()

    # The ATTN contract test exercises only ggml graph/wire/backend APIs. Keep
    # it independent of llama-common so Vulkan kernel regressions can be built
    # and run without first rebuilding the entire libllama dependency graph.
    add_executable(test-flashprefill-attn test-flashprefill-attn.cpp)
    target_link_libraries(test-flashprefill-attn PRIVATE ggml)
    target_include_directories(test-flashprefill-attn PRIVATE ${PROJECT_SOURCE_DIR}/src)
    if (LLAMA_TESTS_INSTALL)
        install(TARGETS test-flashprefill-attn RUNTIME)
    endif()
    add_test(NAME test-flashprefill-attn COMMAND $<TARGET_FILE:test-flashprefill-attn>)
    set_property(TEST test-flashprefill-attn PROPERTY LABELS main)
    # test-flashprefill-state exercises the real RERoT runtime (chains B/E/G:
    # episode codec, context shift, pen suspend/resume) — same extra source
    # and include dir as test-rerot-runtime.
    target_sources(test-flashprefill-state PRIVATE
        ${PROJECT_SOURCE_DIR}/tools/server/server-rerot-todo.cpp
        ${PROJECT_SOURCE_DIR}/tools/server/server-rerot-thinking.cpp
        ${PROJECT_SOURCE_DIR}/tools/server/server-rerot.cpp
        ${PROJECT_SOURCE_DIR}/tools/server/server-rerot-mindmap.cpp)
    target_include_directories(test-flashprefill-state PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    llama_build_and_test(test-xkv-memory-model.cpp)
    target_include_directories(test-xkv-memory-model PRIVATE ${PROJECT_SOURCE_DIR}/src)

    # TP5 Vulkan collective mesh test (TP5.md 22.2): needs >=2 Vulkan
    # devices at RUNTIME; registered only when the Vulkan backend is built.
    # ctest runs it with --devices auto and the binary SKIPs cleanly when
    # fewer than two Vulkan devices are present.
    if (GGML_VULKAN)
        add_executable(test-vulkan-tp5-mesh test-vulkan-tp5-mesh.cpp)
        target_include_directories(test-vulkan-tp5-mesh PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)
        target_link_libraries(test-vulkan-tp5-mesh PRIVATE ggml)
        # Explicit --run and an idle-device audit are required; never run from ctest.
        add_executable(test-vulkan-tp5-relay test-vulkan-tp5-relay.cpp)
        target_include_directories(test-vulkan-tp5-relay PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src/ggml-vulkan)
        target_link_libraries(test-vulkan-tp5-relay PRIVATE ggml ggml-vulkan)
        if (LLAMA_TESTS_INSTALL)
            install(TARGETS test-vulkan-tp5-mesh RUNTIME)
        endif()
        add_test(NAME test-vulkan-tp5-mesh COMMAND $<TARGET_FILE:test-vulkan-tp5-mesh>)
        set_property(TEST test-vulkan-tp5-mesh PROPERTY LABELS main)
        set_property(TEST test-vulkan-tp5-mesh PROPERTY SKIP_RETURN_CODE 77)
    endif()
    if (GGML_VULKAN)
        add_executable(test-vulkan-command-replay test-vulkan-command-replay.cpp)
        target_link_libraries(test-vulkan-command-replay PRIVATE ggml)
        if (LLAMA_TESTS_INSTALL)
            install(TARGETS test-vulkan-command-replay RUNTIME)
        endif()
        add_test(NAME test-vulkan-command-replay COMMAND $<TARGET_FILE:test-vulkan-command-replay>)
        set_property(TEST test-vulkan-command-replay PROPERTY LABELS main)
    endif()
    if (GGML_VULKAN)
        add_executable(test-vulkan-sparse-attn test-vulkan-sparse-attn.cpp)
        target_link_libraries(test-vulkan-sparse-attn PRIVATE ggml)
        if (LLAMA_TESTS_INSTALL)
            install(TARGETS test-vulkan-sparse-attn RUNTIME)
        endif()
        add_test(NAME test-vulkan-sparse-attn COMMAND $<TARGET_FILE:test-vulkan-sparse-attn>)
        set_property(TEST test-vulkan-sparse-attn PROPERTY LABELS main)
    endif()
    if (GGML_VULKAN)
        add_executable(test-vulkan-gdn-multistep test-vulkan-gdn-multistep.cpp)
        target_link_libraries(test-vulkan-gdn-multistep PRIVATE ggml)
        if (LLAMA_TESTS_INSTALL)
            install(TARGETS test-vulkan-gdn-multistep RUNTIME)
        endif()
        add_test(NAME test-vulkan-gdn-multistep COMMAND $<TARGET_FILE:test-vulkan-gdn-multistep>)
        set_property(TEST test-vulkan-gdn-multistep PROPERTY LABELS main)

        add_executable(test-vulkan-flash-attn-capacity test-vulkan-flash-attn-capacity.cpp)
        target_link_libraries(test-vulkan-flash-attn-capacity PRIVATE ggml)
        if (LLAMA_TESTS_INSTALL)
            install(TARGETS test-vulkan-flash-attn-capacity RUNTIME)
        endif()
        add_test(NAME test-vulkan-flash-attn-capacity COMMAND $<TARGET_FILE:test-vulkan-flash-attn-capacity>)
        set_property(TEST test-vulkan-flash-attn-capacity PROPERTY LABELS main)
    endif()
    llama_build_and_test(test-xkv-codec.cpp)
    target_include_directories(test-xkv-codec PRIVATE ${PROJECT_SOURCE_DIR}/src)
    llama_build_and_test(test-xkv-factor.cpp)
    target_include_directories(test-xkv-factor PRIVATE ${PROJECT_SOURCE_DIR}/src)
    target_compile_options(test-xkv-factor PRIVATE -g -O0)
    llama_build_and_test(test-xkv-graph-ref.cpp)
    target_include_directories(test-xkv-graph-ref PRIVATE ${PROJECT_SOURCE_DIR}/src)
    llama_build_and_test(test-xkv-graph-runtime.cpp)
    target_include_directories(test-xkv-graph-runtime PRIVATE ${PROJECT_SOURCE_DIR}/src)
    llama_build_and_test(test-xkv-hot.cpp)
    target_include_directories(test-xkv-hot PRIVATE ${PROJECT_SOURCE_DIR}/src)
    llama_build_and_test(test-xkv-hot-cache.cpp)
    target_include_directories(test-xkv-hot-cache PRIVATE ${PROJECT_SOURCE_DIR}/src)
    llama_build_and_test(test-xkv-cache.cpp)
    target_include_directories(test-xkv-cache PRIVATE ${PROJECT_SOURCE_DIR}/src)
    llama_build_and_test(test-xkv-admission.cpp)
    target_include_directories(test-xkv-admission PRIVATE ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/tools/server ${PROJECT_SOURCE_DIR}/tools/mtmd)
    target_link_libraries(test-xkv-admission PRIVATE llama llama-common)
    llama_build_and_test(test-xkv-fit-residency.cpp)
    target_link_libraries(test-xkv-fit-residency PRIVATE llama-common)
    llama_build_and_test(test-xkv-metrics.cpp)
    target_include_directories(test-xkv-metrics PRIVATE ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/tools/server ${PROJECT_SOURCE_DIR}/tools/mtmd)
    target_link_libraries(test-xkv-metrics PRIVATE llama llama-common)
    llama_build_and_test(test-xkv-canonical.cpp)
    target_include_directories(test-xkv-canonical PRIVATE ${PROJECT_SOURCE_DIR}/src)
    llama_build_and_test(test-xkv-landmark.cpp)
    target_include_directories(test-xkv-landmark PRIVATE ${PROJECT_SOURCE_DIR}/src)
    llama_build_and_test(test-xkv-reader.cpp)
    target_include_directories(test-xkv-reader PRIVATE ${PROJECT_SOURCE_DIR}/src)
    llama_build_and_test(test-xkv-runtime.cpp)
    target_include_directories(test-xkv-runtime PRIVATE ${PROJECT_SOURCE_DIR}/src)
    llama_build_and_test(test-xkv-state.cpp)
    target_include_directories(test-xkv-state PRIVATE ${PROJECT_SOURCE_DIR}/src)
    llama_build_and_test(test-xkv-transaction.cpp)
    target_include_directories(test-xkv-transaction PRIVATE ${PROJECT_SOURCE_DIR}/src)
    llama_build_and_test(test-xkv-tri.cpp)
    target_include_directories(test-xkv-tri PRIVATE ${PROJECT_SOURCE_DIR}/src)
    llama_build_and_test(test-xkv-reconstruct.cpp)
    target_include_directories(test-xkv-reconstruct PRIVATE ${PROJECT_SOURCE_DIR}/src)
    llama_build_and_test(test-xkv-backend.cpp)
    target_include_directories(test-xkv-backend PRIVATE ${PROJECT_SOURCE_DIR}/src)
    llama_build_and_test(test-xkv-vulkan-attention.cpp)
    target_include_directories(test-xkv-vulkan-attention PRIVATE ${PROJECT_SOURCE_DIR}/src)
    llama_build_and_test(test-xkv-matrix.cpp)
    target_include_directories(test-xkv-matrix PRIVATE ${PROJECT_SOURCE_DIR}/src)
    llama_build_and_test(test-xkv-fault-matrix.cpp)
    target_include_directories(test-xkv-fault-matrix PRIVATE ${PROJECT_SOURCE_DIR}/src)

    llama_build_and_test(test-xkv-factor-encoding-size.cpp)
    target_include_directories(test-xkv-factor-encoding-size PRIVATE ${PROJECT_SOURCE_DIR}/src)
    llama_build_and_test(test-xkv-no-dense-mirror.cpp)
    target_include_directories(test-xkv-no-dense-mirror PRIVATE ${PROJECT_SOURCE_DIR}/src)
    llama_build_and_test(test-xkv-workspace-bound.cpp)
    target_include_directories(test-xkv-workspace-bound PRIVATE ${PROJECT_SOURCE_DIR}/src)
    llama_build_and_test(test-xkv-quantized-pack.cpp)
    target_include_directories(test-xkv-quantized-pack PRIVATE ${PROJECT_SOURCE_DIR}/src)
    llama_build_and_test(test-xkv-state-code-stream.cpp)
    target_include_directories(test-xkv-state-code-stream PRIVATE ${PROJECT_SOURCE_DIR}/src)
    llama_build_and_test(test-xkv-vulkan-factorize.cpp)
    target_include_directories(test-xkv-vulkan-factorize PRIVATE ${PROJECT_SOURCE_DIR}/src)
    llama_build_and_test(test-xkv-vulkan-landmark.cpp)
    target_include_directories(test-xkv-vulkan-landmark PRIVATE ${PROJECT_SOURCE_DIR}/src)
    add_executable(test-xkv-vulkan-landmark-build test-xkv-vulkan-landmark-build.cpp)
    target_include_directories(test-xkv-vulkan-landmark-build PRIVATE ${PROJECT_SOURCE_DIR}/src)
    target_link_libraries(test-xkv-vulkan-landmark-build PRIVATE llama-common)
    llama_build_and_test(test-xkv-turbo-shift.cpp)
    target_include_directories(test-xkv-turbo-shift PRIVATE ${PROJECT_SOURCE_DIR}/src)
    llama_build_and_test(test-xkv-native-seal.cpp)
    target_include_directories(test-xkv-native-seal PRIVATE ${PROJECT_SOURCE_DIR}/src)

    llama_build_and_test(test-backend-memmove.cpp)
    llama_build_and_test(test-vulkan-moe-threshold.cpp)
    # CPU-only transport/fusion composition contract; no Vulkan initialization.
    llama_build_and_test(test-vulkan-tp5-output-liveness.cpp)
    target_include_directories(test-vulkan-tp5-output-liveness PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src/ggml-vulkan)
    llama_build_and_test(test-chat.cpp WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    target_include_directories(test-chat PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    target_link_libraries(test-chat PRIVATE server-context)
    llama_build_and_test(test-server-task.cpp)
    target_include_directories(test-server-task PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    target_link_libraries(test-server-task PRIVATE server-context)
    set_tests_properties(test-server-task PROPERTIES TIMEOUT 30)
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

    # TP5 plan: pure CPU role tables / slice legality (TP5.md T02, 22.1)
    llama_build_and_test(test-tp5-plan.cpp)
    target_include_directories(test-tp5-plan PRIVATE ${PROJECT_SOURCE_DIR}/src /usr/include/libdrm)

    # TP5 meta backend CPU reduction boundaries and linear region deferral
    llama_build_and_test(test-meta-reduce-boundary.cpp)
    target_include_directories(test-meta-reduce-boundary PRIVATE ${PROJECT_SOURCE_DIR}/ggml/include ${PROJECT_SOURCE_DIR}/ggml/src)

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

# G2 indent-aware GBNF extension (opt-in; the plain grammar path is unchanged)
llama_build_and_test(test-grammar-indent.cpp)
target_include_directories(test-grammar-indent PRIVATE ${PROJECT_SOURCE_DIR}/src)
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
    llama_build_and_test(test-flash-attn-precision.cpp)
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

if (NOT LLAMA_USE_SYSTEM_GGML)
    # Needs non-public ggml-impl.h
    llama_build_and_test(test-gguf.cpp)

    # Needs non-public ggml{,-backend}-impl.h
    llama_build_and_test(test-alloc.cpp)

    llama_build_and_test(test-spec-direct-hidden.cpp)
endif()

llama_build(test-export-graph-ops.cpp)
if (TARGET gguf-model-data)
    target_link_libraries(test-export-graph-ops PRIVATE gguf-model-data)
    target_compile_definitions(test-export-graph-ops PRIVATE LLAMA_HF_FETCH)
endif()
