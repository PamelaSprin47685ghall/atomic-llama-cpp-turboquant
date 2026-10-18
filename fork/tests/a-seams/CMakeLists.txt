cmake_minimum_required(VERSION 3.16)
project(TQASeams LANGUAGES CXX)
enable_testing()

if(NOT TARGET llama)
    find_package(llama CONFIG REQUIRED)
endif()

add_executable(test-compute-memory compute-memory.cpp)
target_compile_features(test-compute-memory PRIVATE cxx_std_17)
target_link_libraries(test-compute-memory PRIVATE ggml::ggml ggml::ggml-cpu)
add_test(NAME compute-memory COMMAND test-compute-memory)

add_executable(test-flash-attn-precision flash-attn-precision.cpp)
target_compile_features(test-flash-attn-precision PRIVATE cxx_std_17)
target_link_libraries(test-flash-attn-precision PRIVATE ggml::ggml ggml::ggml-cpu)
add_test(NAME flash-attn-precision COMMAND test-flash-attn-precision)
