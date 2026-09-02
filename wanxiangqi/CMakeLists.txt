# Wanxiangqi fork build registration.
#
# Keep implementation source under this directory and leave the main
# tree with only minimal integration hooks.

if (LLAMA_BUILD_TOOLS)
include(FetchContent)
include(GNUInstallDirs)

set(CPP_TIKTOKEN_INSTALL OFF CACHE BOOL "" FORCE)
set(CPP_TIKTOKEN_TESTING OFF CACHE BOOL "" FORCE)
set(PCRE2_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(PCRE2_BUILD_PCRE2GREP OFF CACHE BOOL "" FORCE)
if (DEFINED CMAKE_POLICY_VERSION_MINIMUM)
    set(WANXIANGQI_PARENT_CMAKE_POLICY_VERSION_MINIMUM_DEFINED TRUE)
    set(WANXIANGQI_PARENT_CMAKE_POLICY_VERSION_MINIMUM "${CMAKE_POLICY_VERSION_MINIMUM}")
endif()
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
set(WANXIANGQI_PARENT_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(wanxiangqi_cpp_tiktoken
    GIT_REPOSITORY https://github.com/gh-markt/cpp-tiktoken.git
    GIT_TAG 9323db528d52e48900c75ce197c3251085b18480
    GIT_SHALLOW FALSE
    GIT_SUBMODULES_RECURSE TRUE)
FetchContent_MakeAvailable(wanxiangqi_cpp_tiktoken)
set_target_properties(tiktoken PROPERTIES POSITION_INDEPENDENT_CODE ON)
if (TARGET pcre2-8-static)
    set_target_properties(pcre2-8-static PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
set(BUILD_SHARED_LIBS ${WANXIANGQI_PARENT_BUILD_SHARED_LIBS} CACHE BOOL "" FORCE)
if (WANXIANGQI_PARENT_CMAKE_POLICY_VERSION_MINIMUM_DEFINED)
    set(CMAKE_POLICY_VERSION_MINIMUM "${WANXIANGQI_PARENT_CMAKE_POLICY_VERSION_MINIMUM}")
else()
    unset(CMAKE_POLICY_VERSION_MINIMUM)
endif()

add_library(wanxiangqi-o200k STATIC
    common/o200k-codec.cpp
    common/o200k-codec.h)
set_target_properties(wanxiangqi-o200k PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(wanxiangqi-o200k PRIVATE cxx_std_20)
find_package(Threads REQUIRED)
target_link_libraries(wanxiangqi-o200k PRIVATE tiktoken Threads::Threads)
target_include_directories(wanxiangqi-o200k PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/common)

file(MAKE_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tokenizers")
configure_file(
    "${wanxiangqi_cpp_tiktoken_SOURCE_DIR}/o200k_base.tiktoken"
    "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tokenizers/o200k_base.tiktoken"
    COPYONLY)

add_executable(wanxiangqi-repetition-envelope-tool EXCLUDE_FROM_ALL
    common/derive-repetition-envelope.cpp
    common/repetition-corpus.cpp
    common/repetition-corpus.h)
target_compile_features(wanxiangqi-repetition-envelope-tool PRIVATE cxx_std_17)
target_link_libraries(wanxiangqi-repetition-envelope-tool PRIVATE wanxiangqi-o200k)

set(WANXIANGQI_REPETITION_GENERATED_DIR "${CMAKE_CURRENT_SOURCE_DIR}/generated")
set(WANXIANGQI_REPETITION_ENVELOPE "${WANXIANGQI_REPETITION_GENERATED_DIR}/repetition-envelope.h")

# The envelope is a checked-in artifact. Normal configure/build paths only consume it.
# Refresh it explicitly with: cmake --build <build-dir> --target wanxiangqi-repetition-envelope-update
add_custom_target(wanxiangqi-repetition-envelope-update
    COMMAND ${CMAKE_COMMAND}
        -DWANXIANGQI_REPETITION_ROOT=${PROJECT_SOURCE_DIR}
        -DWANXIANGQI_REPETITION_MANIFEST=${CMAKE_CURRENT_BINARY_DIR}/repetition-corpus.txt
        -DWANXIANGQI_REPETITION_OUTPUT=${WANXIANGQI_REPETITION_ENVELOPE}
        -DWANXIANGQI_REPETITION_TOOL=$<TARGET_FILE:wanxiangqi-repetition-envelope-tool>
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/update-repetition-envelope.cmake
    DEPENDS wanxiangqi-repetition-envelope-tool
    VERBATIM)

add_library(wanxiangqi-repetition-guard STATIC
    common/repetition-guard.cpp
    common/repetition-guard.h
    "${WANXIANGQI_REPETITION_ENVELOPE}")
target_compile_features(wanxiangqi-repetition-guard PRIVATE cxx_std_17)
target_link_libraries(wanxiangqi-repetition-guard PRIVATE wanxiangqi-o200k)
target_include_directories(wanxiangqi-repetition-guard PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    "${WANXIANGQI_REPETITION_GENERATED_DIR}")

if (NOT TARGET llama-perplexity)
    add_subdirectory(${PROJECT_SOURCE_DIR}/tools/perplexity ${CMAKE_CURRENT_BINARY_DIR}/perplexity)
endif()

# Shared layer for the compression probes. Every tool below reads the same trie
# and the same WXQCAL03 calibration dump; those formats and the dense linear
# algebra over them are defined once, here, so a format change cannot land in one
# reader and miss another.
add_library(wanxiangqi-probe-lib STATIC
    lib/calib-dump.cpp
    lib/calib-dump.h
    lib/gguf-slab.cpp
    lib/gguf-slab.h
    lib/linalg.cpp
    lib/linalg.h
    lib/parallel.cpp
    lib/parallel.h
    lib/trie.cpp
    lib/trie.h)
set_target_properties(wanxiangqi-probe-lib PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(wanxiangqi-probe-lib PRIVATE cxx_std_17)
target_link_libraries(wanxiangqi-probe-lib PUBLIC ggml)
target_include_directories(wanxiangqi-probe-lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/lib)

# Trie-driven MoE calibration collector: routing counts + expert-input second
# moment, gathered by walking the committed request-token trie depth first.
add_executable(wanxiangqi-trie-calib calib/trie-calib.cpp)
target_compile_features(wanxiangqi-trie-calib PRIVATE cxx_std_17)
target_link_libraries(wanxiangqi-trie-calib PRIVATE wanxiangqi-probe-lib llama-common llama ggml ${CMAKE_THREAD_LIBS_INIT})
install(TARGETS wanxiangqi-trie-calib RUNTIME)

# Trie-driven TriAttention calibration collector: pre-RoPE Q statistics
# (complex mean, mean magnitude per frequency band) for all full-attention
# layers, gathered by walking the request-token trie depth first.
add_executable(wanxiangqi-trie-triattention-calib calib/trie-triattention-calib.cpp)
target_compile_features(wanxiangqi-trie-triattention-calib PRIVATE cxx_std_17)
target_include_directories(wanxiangqi-trie-triattention-calib PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(wanxiangqi-trie-triattention-calib PRIVATE wanxiangqi-probe-lib llama-common llama ggml ${CMAKE_THREAD_LIBS_INIT})
install(TARGETS wanxiangqi-trie-triattention-calib RUNTIME)

# Offline structural probe: reads GGUF weights directly, never runs the model.
add_executable(wanxiangqi-calib-probe calib/calib-probe.cpp)
target_compile_features(wanxiangqi-calib-probe PRIVATE cxx_std_17)
target_link_libraries(wanxiangqi-calib-probe PRIVATE wanxiangqi-probe-lib ggml ${CMAKE_THREAD_LIBS_INIT})
install(TARGETS wanxiangqi-calib-probe RUNTIME)

# Does QMoE's sub-1-bit ternary result transfer to a modern top-k MoE?
add_executable(wanxiangqi-qmoe-probe calib/qmoe-probe.cpp)
target_compile_features(wanxiangqi-qmoe-probe PRIVATE cxx_std_17)
target_link_libraries(wanxiangqi-qmoe-probe PRIVATE wanxiangqi-probe-lib ggml ${CMAKE_THREAD_LIBS_INIT})
install(TARGETS wanxiangqi-qmoe-probe RUNTIME)

# Greedy generation for the behavioural collapse check. llama-cli segfaults
# before it logs anything in this build; the tools here load the same model fine.
add_executable(wanxiangqi-gen probe/wxq-gen.cpp)
target_compile_features(wanxiangqi-gen PRIVATE cxx_std_17)
target_link_libraries(wanxiangqi-gen PRIVATE llama-common llama ggml ${CMAKE_THREAD_LIBS_INIT})
install(TARGETS wanxiangqi-gen RUNTIME)

# Applies a linear input transform to the expert weights in place: y = W(Mx)
# is the same as y = (WM)x, so the tail ablation needs no graph change. Shapes
# and types are preserved, so the output is a copy of the source with only the
# gate/up slabs overwritten at their existing offsets.
add_executable(wanxiangqi-bake probe/wxq-bake.cpp)
target_compile_features(wanxiangqi-bake PRIVATE cxx_std_17)
target_link_libraries(wanxiangqi-bake PRIVATE wanxiangqi-probe-lib ggml ${CMAKE_THREAD_LIBS_INIT})
install(TARGETS wanxiangqi-bake RUNTIME)

# Raw activation sampler. Covariance cannot see manifold dimension: a thin
# curved manifold in high ambient dimension has near-full-rank covariance, so
# intrinsic dimension needs samples, not moments.
add_executable(wanxiangqi-manifold probe/wxq-manifold.cpp)
target_compile_features(wanxiangqi-manifold PRIVATE cxx_std_17)
target_link_libraries(wanxiangqi-manifold PRIVATE wanxiangqi-probe-lib llama-common llama ggml ${CMAKE_THREAD_LIBS_INIT})
install(TARGETS wanxiangqi-manifold RUNTIME)

# Per-position KL against a reference dump, walked over the calibration trie.
# tools/perplexity --kl-divergence cannot do the trie population, per-position
# output, or role-weighted aggregation, all of which are needed here.
add_executable(wanxiangqi-kl probe/wxq-kl.cpp)
target_compile_features(wanxiangqi-kl PRIVATE cxx_std_17)
target_link_libraries(wanxiangqi-kl PRIVATE wanxiangqi-probe-lib llama-common llama ggml ${CMAKE_THREAD_LIBS_INIT})
install(TARGETS wanxiangqi-kl RUNTIME)

if (TARGET server-context)
    target_sources(server-context PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/server/server-corpus-dump.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/server/server-corpus-dump.h
        ${CMAKE_CURRENT_SOURCE_DIR}/server/integration.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/server/integration.h)
    target_include_directories(server-context PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/server
        ${CMAKE_CURRENT_SOURCE_DIR}/common)
    target_link_libraries(server-context PRIVATE wanxiangqi-repetition-guard)
endif()

install(FILES "${wanxiangqi_cpp_tiktoken_SOURCE_DIR}/o200k_base.tiktoken"
    DESTINATION "${CMAKE_INSTALL_BINDIR}/tokenizers")
endif()

if (LLAMA_BUILD_TESTS AND NOT CMAKE_JS_VERSION AND TARGET server-context)
    llama_build_and_test(${CMAKE_CURRENT_SOURCE_DIR}/tests/test-o200k-codec.cpp)
    target_include_directories(test-o200k-codec PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/common)
    target_link_libraries(test-o200k-codec PRIVATE wanxiangqi-o200k)

    llama_build_and_test(${CMAKE_CURRENT_SOURCE_DIR}/tests/test-server-corpus-dump.cpp)
    target_include_directories(test-server-corpus-dump PRIVATE
        ${PROJECT_SOURCE_DIR}/tools/server
        ${CMAKE_CURRENT_SOURCE_DIR}/server)
    target_link_libraries(test-server-corpus-dump PRIVATE server-context)

    llama_build_and_test(${CMAKE_CURRENT_SOURCE_DIR}/tests/test-repetition-guard.cpp)
    target_sources(test-repetition-guard PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/common/repetition-corpus.cpp)
    target_include_directories(test-repetition-guard PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/common
        "${WANXIANGQI_REPETITION_GENERATED_DIR}")
    target_link_libraries(test-repetition-guard PRIVATE wanxiangqi-repetition-guard)
endif()
