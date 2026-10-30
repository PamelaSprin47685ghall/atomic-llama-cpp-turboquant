cmake_minimum_required(VERSION 3.16)
project(TQBatchCompose VERSION 0.2.0 LANGUAGES CXX)

include(GNUInstallDirs)
if(NOT TARGET llama)
    find_package(llama CONFIG REQUIRED)
endif()
if(NOT TARGET TQScheduling::scheduling)
    find_package(TQScheduling CONFIG REQUIRED)
endif()
if(NOT TARGET TQText::text)
    find_package(TQText CONFIG REQUIRED)
endif()

add_executable(tq_batch_compose src/main.cpp)
set_target_properties(tq_batch_compose PROPERTIES OUTPUT_NAME tq-batch-compose)
if(UNIX AND NOT APPLE)
    set_target_properties(tq_batch_compose PROPERTIES INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
endif()
target_compile_features(tq_batch_compose PRIVATE cxx_std_17)
target_link_libraries(tq_batch_compose PRIVATE llama TQScheduling::scheduling TQText::text)
install(TARGETS tq_batch_compose RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
install(FILES LICENSE DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/TQBatchCompose")

set(TQ_BATCH_COMPOSE_TEST_MODEL "" CACHE FILEPATH "Real GGUF model for the integration test.")
include(CTest)
if(BUILD_TESTING)
    add_test(NAME tq-batch-compose-help COMMAND tq_batch_compose --help)
endif()
set(TQ_BATCH_COMPOSE_TEST_GPU_LAYERS "0" CACHE STRING "GPU layers requested by the integration test.")
if(TQ_BATCH_COMPOSE_TEST_MODEL)
    enable_testing()
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(NAME tq_batch_compose.real_model
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/real-batch.py"
            --binary "$<TARGET_FILE:tq_batch_compose>"
            --model "${TQ_BATCH_COMPOSE_TEST_MODEL}"
            --gpu-layers "${TQ_BATCH_COMPOSE_TEST_GPU_LAYERS}"
            --output-dir "${CMAKE_CURRENT_BINARY_DIR}/real-model-test")
    add_test(NAME tq_batch_compose.admission
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_admission.py"
            --binary "$<TARGET_FILE:tq_batch_compose>"
            --model "${TQ_BATCH_COMPOSE_TEST_MODEL}"
            --gpu-layers "${TQ_BATCH_COMPOSE_TEST_GPU_LAYERS}"
            --output-dir "${CMAKE_CURRENT_BINARY_DIR}/admission-test")
    add_test(NAME tq_batch_compose.streaming
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_streaming.py"
            --binary "$<TARGET_FILE:tq_batch_compose>"
            --model "${TQ_BATCH_COMPOSE_TEST_MODEL}"
            --gpu-layers "${TQ_BATCH_COMPOSE_TEST_GPU_LAYERS}"
            --output-dir "${CMAKE_CURRENT_BINARY_DIR}/streaming-test")
endif()
