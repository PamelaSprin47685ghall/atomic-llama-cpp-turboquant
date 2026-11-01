set(TARGET llama-tp5-router-fixture)
add_executable(${TARGET} tp5-router-fixture.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE llama ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# In-tree builds also compile the contract check, which drives the real
# Qwen4Exp loader (src/models/qwen4exp.cpp) over the generated file. That needs
# the private src/ headers, so it is only available here -- a stock tree that
# merely copies this directory keeps the generator working without it.
if (TARGET llama AND EXISTS "${PROJECT_SOURCE_DIR}/src/models/qwen4exp.cpp")
    target_sources(${TARGET} PRIVATE router-fixture-contract.cpp)
    target_include_directories(${TARGET} PRIVATE "${PROJECT_SOURCE_DIR}/src")
    target_compile_definitions(${TARGET} PRIVATE TP5_FIXTURE_CONTRACT)
endif()

set(TARGET llama-tp5-router-verify)
add_executable(${TARGET} tp5-router-verify.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE llama ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# tp5-router-scratch-diag.h is #included by tp5-router-verify.cpp from this same
# directory (L1 triage only, GGML_TP5_ROUTER_SCRATCH_DIAG=1, never a gate); it needs
# no extra include path or source entry.

# No private headers for the verifier: the executed decode graph is not reachable
# through llama.h, and the only in-tree accessor (get_gf_res_reserve) is the
# reserve-sized graph, not the one that runs. The L1 gate reports that gap instead
# of auditing a graph that is not executed.
