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

add_subdirectory(qwen35-prune)

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
