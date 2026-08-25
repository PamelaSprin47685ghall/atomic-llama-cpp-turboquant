# Wanxiangqi fork build registration.
#
# Keep implementation source under this directory and leave the main
# tree with only minimal integration hooks.

if (LLAMA_BUILD_TOOLS)
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
            ${CMAKE_CURRENT_SOURCE_DIR}/server)
    endif()
endif()

if (LLAMA_BUILD_TESTS AND NOT CMAKE_JS_VERSION AND TARGET server-context)
    llama_build_and_test(${CMAKE_CURRENT_SOURCE_DIR}/tests/test-server-corpus-dump.cpp)
    target_include_directories(test-server-corpus-dump PRIVATE
        ${PROJECT_SOURCE_DIR}/tools/server
        ${CMAKE_CURRENT_SOURCE_DIR}/server)
    target_link_libraries(test-server-corpus-dump PRIVATE server-context)
endif()
