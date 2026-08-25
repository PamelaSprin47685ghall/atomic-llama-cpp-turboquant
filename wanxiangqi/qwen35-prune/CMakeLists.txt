set(TARGET llama-qwen35-prune)

# Fork-isolated calibration executable. The stock tools/imatrix source stays
# byte-identical to upstream; Qwen3.5 trie/GDN/ENP collection lives
# here with the pruning pipeline that consumes it.
add_library(qwen35-prune-enp STATIC enp-geometric-coreset.cpp enp-geometric-coreset.h)
target_include_directories(qwen35-prune-enp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(qwen35-prune-enp PUBLIC cxx_std_17)

add_library(qwen35-prune-expert-replacement STATIC expert-replacement.cpp expert-replacement.h)
target_include_directories(qwen35-prune-expert-replacement PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(qwen35-prune-expert-replacement PUBLIC cxx_std_17)

add_executable(llama-qwen35-imatrix imatrix.cpp)
target_link_libraries(llama-qwen35-imatrix PRIVATE qwen35-prune-enp llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(llama-qwen35-imatrix PRIVATE cxx_std_17)

# File authentication scans a 23+ GB source repeatedly during plan/apply/verify.
# Use OpenSSL's hardware-accelerated SHA-256 when available, with the bundled
# scalar implementation retained as a portable fallback.
find_package(OpenSSL QUIET COMPONENTS Crypto)

add_library(qwen35-prune-q2 STATIC q2-opt-signed.cpp q2-opt-signed.h)
target_link_libraries(qwen35-prune-q2 PUBLIC ggml)
target_include_directories(qwen35-prune-q2
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ../../ggml/src)
target_compile_features(qwen35-prune-q2 PUBLIC cxx_std_17)

add_library(qwen35-prune-gdn STATIC
    gdn-adjoint.cpp
    gdn-adjoint.h
    gdn-geometry-plan.cpp
    gdn-geometry-plan.h
    gdn-joint-replay.cpp
    gdn-joint-replay.h
    gdn-function-projection.cpp
    gdn-function-projection.h
    gdn-observable-risk.cpp
    gdn-observable-risk.h)
target_include_directories(qwen35-prune-gdn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(qwen35-prune-gdn PUBLIC cxx_std_17)

add_executable(test-q2-opt-signed test-q2-opt-signed.cpp)
target_link_libraries(test-q2-opt-signed PRIVATE qwen35-prune-q2 ggml)
target_compile_features(test-q2-opt-signed PRIVATE cxx_std_17)

add_executable(test-q2-opt-backend test-q2-opt-backend.cpp)
target_link_libraries(test-q2-opt-backend PRIVATE qwen35-prune-q2 ggml)
target_compile_features(test-q2-opt-backend PRIVATE cxx_std_17)

add_executable(test-gdn-function-projection test-gdn-function-projection.cpp)
target_link_libraries(test-gdn-function-projection PRIVATE qwen35-prune-gdn)
target_compile_features(test-gdn-function-projection PRIVATE cxx_std_17)

add_executable(test-gdn-observable-risk test-gdn-observable-risk.cpp)
target_link_libraries(test-gdn-observable-risk PRIVATE qwen35-prune-gdn)
target_compile_features(test-gdn-observable-risk PRIVATE cxx_std_17)

add_executable(test-gdn-adjoint test-gdn-adjoint.cpp)
target_link_libraries(test-gdn-adjoint PRIVATE qwen35-prune-gdn)
target_compile_features(test-gdn-adjoint PRIVATE cxx_std_17)

add_executable(test-gdn-geometry-plan test-gdn-geometry-plan.cpp)
target_link_libraries(test-gdn-geometry-plan PRIVATE qwen35-prune-gdn)
target_compile_features(test-gdn-geometry-plan PRIVATE cxx_std_17)

add_executable(test-gdn-joint-replay test-gdn-joint-replay.cpp)
target_link_libraries(test-gdn-joint-replay PRIVATE qwen35-prune-gdn)
target_compile_features(test-gdn-joint-replay PRIVATE cxx_std_17)

add_executable(test-enp-geometric-coreset test-enp-geometric-coreset.cpp)
target_link_libraries(test-enp-geometric-coreset PRIVATE qwen35-prune-enp)
target_compile_features(test-enp-geometric-coreset PRIVATE cxx_std_17)

add_executable(test-expert-replacement test-expert-replacement.cpp)
target_link_libraries(test-expert-replacement PRIVATE qwen35-prune-expert-replacement)
target_compile_features(test-expert-replacement PRIVATE cxx_std_17)

if (LLAMA_BUILD_TESTS)
    add_test(NAME q2-opt-signed COMMAND test-q2-opt-signed)
    add_test(NAME q2-opt-backend COMMAND test-q2-opt-backend)
    add_test(NAME gdn-function-projection COMMAND test-gdn-function-projection)
    add_test(NAME gdn-observable-risk COMMAND test-gdn-observable-risk)
    add_test(NAME gdn-adjoint COMMAND test-gdn-adjoint)
    add_test(NAME gdn-geometry-plan COMMAND test-gdn-geometry-plan)
    add_test(NAME gdn-joint-replay COMMAND test-gdn-joint-replay)
    add_test(NAME enp-geometric-coreset COMMAND test-enp-geometric-coreset)
    add_test(NAME expert-replacement COMMAND test-expert-replacement)
endif()

add_executable(${TARGET}
    qwen35-prune.cpp
    ../../examples/gguf-hash/deps/sha256/sha256.c
)
target_include_directories(${TARGET} PRIVATE
    ../../examples/gguf-hash/deps
    ../../src
    ../../vendor
)
target_link_libraries(${TARGET} PRIVATE qwen35-prune-enp qwen35-prune-expert-replacement qwen35-prune-q2 qwen35-prune-gdn llama-common llama ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # ENP coreset scoring is dominated by fixed-size dot products.  Enable
    # OpenMP SIMD directives without pulling in the OpenMP runtime/threading
    # model; expert-level parallelism remains the existing std::thread pool.
    target_compile_options(${TARGET} PRIVATE -fopenmp-simd)
endif()

string(TOUPPER "${CMAKE_BUILD_TYPE}" QWEN35_PRUNE_BUILD_TYPE_UPPER)
set(QWEN35_PRUNE_EFFECTIVE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${QWEN35_PRUNE_BUILD_TYPE_UPPER}}")
string(STRIP "${QWEN35_PRUNE_EFFECTIVE_CXX_FLAGS}" QWEN35_PRUNE_EFFECTIVE_CXX_FLAGS)

set(QWEN35_PRUNE_WORKTREE_DIRTY 0)
if (Git_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} status --porcelain --untracked-files=normal
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE QWEN35_PRUNE_GIT_STATUS
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE QWEN35_PRUNE_GIT_STATUS_RESULT)
    if (QWEN35_PRUNE_GIT_STATUS_RESULT EQUAL 0 AND NOT QWEN35_PRUNE_GIT_STATUS STREQUAL "")
        set(QWEN35_PRUNE_WORKTREE_DIRTY 1)
    endif()
endif()
target_compile_definitions(${TARGET} PRIVATE
    QWEN35_PRUNE_CXX_FLAGS="${QWEN35_PRUNE_EFFECTIVE_CXX_FLAGS}"
    QWEN35_PRUNE_GIT_COMMIT="${LLAMA_BUILD_COMMIT}"
    QWEN35_PRUNE_WORKTREE_DIRTY=${QWEN35_PRUNE_WORKTREE_DIRTY})

if (OpenSSL_FOUND AND TARGET OpenSSL::Crypto)
    target_link_libraries(${TARGET} PRIVATE OpenSSL::Crypto)
    target_compile_definitions(${TARGET} PRIVATE QWEN35_PRUNE_USE_OPENSSL_SHA=1)
endif()

if (LLAMA_TOOLS_INSTALL)
    install(TARGETS llama-qwen35-imatrix ${TARGET} test-q2-opt-signed test-q2-opt-backend test-gdn-function-projection test-gdn-observable-risk test-gdn-adjoint test-gdn-geometry-plan test-gdn-joint-replay test-enp-geometric-coreset test-expert-replacement RUNTIME)
endif()
