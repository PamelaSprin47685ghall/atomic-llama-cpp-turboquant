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

add_executable(test-meta-transfers test-meta-transfers.cpp)
target_compile_features(test-meta-transfers PRIVATE cxx_std_17)
target_link_libraries(test-meta-transfers PRIVATE ggml::ggml ggml::ggml-cpu)
add_test(NAME meta-transfers COMMAND test-meta-transfers)

if(TARGET ggml::ggml-vulkan)
    add_executable(test-vulkan-wht vulkan-wht.cpp)
    target_compile_features(test-vulkan-wht PRIVATE cxx_std_17)
    target_link_libraries(test-vulkan-wht PRIVATE ggml::ggml ggml::ggml-cpu ggml::ggml-vulkan)
    add_test(NAME vulkan-wht COMMAND test-vulkan-wht)
endif()
