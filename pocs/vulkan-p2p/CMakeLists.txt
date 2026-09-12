set(TARGET test-vulkan-p2p-allreduce)
add_executable(${TARGET} test-vulkan-p2p-allreduce.cpp)
target_include_directories(${TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/common)
target_link_libraries(${TARGET} PRIVATE ${CMAKE_THREAD_LIBS_INIT})

find_package(Vulkan REQUIRED)
target_link_libraries(${TARGET} PRIVATE Vulkan::Vulkan)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
