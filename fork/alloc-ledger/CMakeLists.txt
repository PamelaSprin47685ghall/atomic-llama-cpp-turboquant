cmake_minimum_required(VERSION 3.16)
project(TQAllocLedger VERSION 0.1.0 LANGUAGES CXX)

include(CTest)

option(TQ_ALLOC_SANITIZE "Build with AddressSanitizer and UndefinedBehaviorSanitizer." OFF)
option(TQ_ALLOC_BUILD_TESTS "Build the allocation ledger tests." ${BUILD_TESTING})

if(TQ_ALLOC_SANITIZE)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer
                        -fno-sanitize-recover=all)
    add_link_options(-fsanitize=address,undefined)
endif()

add_library(tq_alloc STATIC src/arena.cpp src/plan_cache.cpp src/domain.cpp)
add_library(TQAlloc::alloc ALIAS tq_alloc)
set_target_properties(tq_alloc PROPERTIES
    OUTPUT_NAME tq-alloc
    POSITION_INDEPENDENT_CODE ON)
target_compile_features(tq_alloc PUBLIC cxx_std_17)
target_include_directories(tq_alloc PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(tq_alloc PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

if(TQ_ALLOC_BUILD_TESTS)
    enable_testing()
    foreach(name arena plan-cache domain events)
        add_executable(tq_alloc_${name}_tests tests/${name}-test.cpp)
        target_link_libraries(tq_alloc_${name}_tests PRIVATE TQAlloc::alloc)
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(tq_alloc_${name}_tests PRIVATE
                                   -Wall -Wextra -Wpedantic -Werror)
        endif()
        add_test(NAME tq_alloc.${name} COMMAND tq_alloc_${name}_tests)
    endforeach()
endif()
