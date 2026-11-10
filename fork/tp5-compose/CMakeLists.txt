cmake_minimum_required(VERSION 3.21)
project(TQTP5Compose VERSION 0.1.0 LANGUAGES C CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
find_package(Threads REQUIRED)
if(NOT TARGET llama)
    find_package(llama CONFIG REQUIRED)
endif()
if(NOT TARGET TQRelay::relay)
    find_package(TQRelay CONFIG REQUIRED)
endif()
if(NOT TARGET TQSampling::sampling)
    find_package(TQSampling CONFIG REQUIRED)
endif()
if(NOT TARGET TQScheduling::scheduling)
    find_package(TQScheduling CONFIG REQUIRED)
endif()
if(NOT TARGET TQText::text)
    find_package(TQText CONFIG REQUIRED)
endif()

add_library(tq_tp5_runtime STATIC src/runtime.cpp src/ordered-fold.cpp)
add_library(TQTP5Compose::runtime ALIAS tq_tp5_runtime)
set_target_properties(tq_tp5_runtime PROPERTIES EXPORT_NAME runtime)
target_compile_features(tq_tp5_runtime PUBLIC cxx_std_17)
target_compile_options(tq_tp5_runtime PRIVATE -fno-fast-math -ffp-contract=off)
target_include_directories(tq_tp5_runtime PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
# In-memory lease records come from the read-only F2 ledger library. Only its
# value types are used here; the C adapter owns the external schema.
target_include_directories(tq_tp5_runtime PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../alloc-ledger/include)
target_link_libraries(tq_tp5_runtime PRIVATE llama TQRelay::relay
    TQSampling::sampling TQScheduling::scheduling TQText::text Threads::Threads
    ${CMAKE_DL_LIBS})

add_executable(tq_tp5_compose src/main.cpp)
set_target_properties(tq_tp5_compose PROPERTIES OUTPUT_NAME tq-tp5-compose)
target_link_libraries(tq_tp5_compose PRIVATE TQTP5Compose::runtime)
if(UNIX AND NOT APPLE)
    set_target_properties(tq_tp5_compose PROPERTIES INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
endif()
install(TARGETS tq_tp5_runtime tq_tp5_compose EXPORT TQTP5ComposeTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
install(DIRECTORY include/ DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
configure_package_config_file(cmake/TQTP5ComposeConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/TQTP5ComposeConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/TQTP5Compose")
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/TQTP5ComposeConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}" COMPATIBILITY SameMajorVersion)
install(EXPORT TQTP5ComposeTargets NAMESPACE TQTP5Compose::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/TQTP5Compose")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/TQTP5ComposeConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/TQTP5ComposeConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/TQTP5Compose")
install(FILES LICENSE DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/TQTP5Compose")
install(FILES README.md DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/TQTP5Compose")
install(PROGRAMS tests/real-tp5.py DESTINATION "${CMAKE_INSTALL_DATADIR}/TQTP5Compose" RENAME real-tp5.py)
include(CTest)
option(TQ_TP5_BUILD_HIP_TESTS "Build and run the real Q8/F32 and attention HIP arithmetic oracles" OFF)
if(BUILD_TESTING)
    add_test(NAME tq-tp5-compose-help COMMAND tq_tp5_compose --help)
    add_executable(tq_tp5_ordered_fold_tests tests/test-ordered-fold.cpp src/ordered-fold.cpp)
    target_compile_features(tq_tp5_ordered_fold_tests PRIVATE cxx_std_17)
    target_compile_options(tq_tp5_ordered_fold_tests PRIVATE -fno-fast-math -ffp-contract=off)
    target_include_directories(tq_tp5_ordered_fold_tests PRIVATE src)
    target_link_libraries(tq_tp5_ordered_fold_tests PRIVATE llama)
    add_test(NAME tq-tp5-ordered-f32-fold COMMAND tq_tp5_ordered_fold_tests)
    if(UNIX AND NOT APPLE)
        add_library(tp5_fail_malloc SHARED tests/fail-malloc.c)
        target_link_libraries(tp5_fail_malloc PRIVATE ${CMAKE_DL_LIBS})
    endif()
    add_executable(tq_tp5_host_pool_tests tests/test-host-pool.cpp)
    target_compile_features(tq_tp5_host_pool_tests PRIVATE cxx_std_17)
    target_include_directories(tq_tp5_host_pool_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/../alloc-ledger/include)
    target_compile_definitions(tq_tp5_host_pool_tests PRIVATE
        TP5_TEST_VOCAB_DEFAULT="${CMAKE_CURRENT_SOURCE_DIR}/../../models/ggml-vocab-command-r.gguf")
    target_link_libraries(tq_tp5_host_pool_tests PRIVATE llama ${CMAKE_DL_LIBS})
    if(UNIX AND NOT APPLE)
        # Wrap with cmake -E env: ENVIRONMENT genex expansion is not
        # guaranteed at the declared CMake 3.21 minimum.
        add_test(NAME tq-tp5-host-pool COMMAND ${CMAKE_COMMAND} -E env
            "LD_PRELOAD=$<TARGET_FILE:tp5_fail_malloc>"
            "TP5_REQUIRE_FAIL_MALLOC=1"
            $<TARGET_FILE:tq_tp5_host_pool_tests>)
    else()
        add_test(NAME tq-tp5-host-pool COMMAND tq_tp5_host_pool_tests)
    endif()
    add_executable(tq_tp5_mtp_checkpoint_tests tests/test-mtp-state-checkpoint.cpp)
    target_compile_features(tq_tp5_mtp_checkpoint_tests PRIVATE cxx_std_17)
    target_include_directories(tq_tp5_mtp_checkpoint_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    add_test(NAME tq-tp5-mtp-state-checkpoint COMMAND tq_tp5_mtp_checkpoint_tests)
    add_executable(tq_tp5_mtp_state_real_tests tests/test-mtp-state-real.cpp)
    target_compile_features(tq_tp5_mtp_state_real_tests PRIVATE cxx_std_17)
    target_link_libraries(tq_tp5_mtp_state_real_tests PRIVATE llama)
    add_test(NAME tq-tp5-mtp-state-real COMMAND tq_tp5_mtp_state_real_tests)
    set_tests_properties(tq-tp5-mtp-state-real PROPERTIES SKIP_RETURN_CODE 77)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND AND EXISTS
            "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/rsz/make_tiny_rsz_mamba.py")
        set(tq_tp5_tiny_mamba_model "${CMAKE_CURRENT_BINARY_DIR}/tiny-mamba.gguf")
        add_custom_command(
            OUTPUT "${tq_tp5_tiny_mamba_model}"
            COMMAND ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/rsz/make_tiny_rsz_mamba.py"
                "${tq_tp5_tiny_mamba_model}"
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/rsz/make_tiny_rsz_mamba.py"
            COMMENT "Generating tiny Mamba fixture for the checkpoint test")
        add_custom_target(tq_tp5_tiny_mamba_fixture ALL DEPENDS "${tq_tp5_tiny_mamba_model}")
        add_dependencies(tq_tp5_mtp_state_real_tests tq_tp5_tiny_mamba_fixture)
        set_tests_properties(tq-tp5-mtp-state-real PROPERTIES
            ENVIRONMENT "TP5_TEST_MODEL=${tq_tp5_tiny_mamba_model}")
    endif()
    add_executable(tq_tp5_mtp_greedy_cache_tests tests/test-mtp-greedy-cache.cpp)
    target_compile_features(tq_tp5_mtp_greedy_cache_tests PRIVATE cxx_std_17)
    target_include_directories(tq_tp5_mtp_greedy_cache_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(tq_tp5_mtp_greedy_cache_tests PRIVATE TQSampling::sampling)
    add_test(NAME tq-tp5-mtp-greedy-cache COMMAND tq_tp5_mtp_greedy_cache_tests)
    add_executable(tq_tp5_relay_channel_map_tests tests/test-relay-channel-map.cpp)
    target_compile_features(tq_tp5_relay_channel_map_tests PRIVATE cxx_std_17)
    target_include_directories(tq_tp5_relay_channel_map_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(tq_tp5_relay_channel_map_tests PRIVATE TQRelay::relay)
    add_test(NAME tq-tp5-relay-channel-map COMMAND tq_tp5_relay_channel_map_tests)
    add_executable(tq_tp5_commit_stop_tests tests/test-commit-stop.cpp)
    target_compile_features(tq_tp5_commit_stop_tests PRIVATE cxx_std_17)
    target_include_directories(tq_tp5_commit_stop_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(tq_tp5_commit_stop_tests PRIVATE TQText::text)
    add_test(NAME tq-tp5-commit-stop COMMAND tq_tp5_commit_stop_tests)
    if(TQ_TP5_BUILD_HIP_TESTS)
        add_executable(tq_tp5_q8_f32_tests tests/test-q8-f32.cpp)
        target_compile_features(tq_tp5_q8_f32_tests PRIVATE cxx_std_17)
        target_link_libraries(tq_tp5_q8_f32_tests PRIVATE llama)
        add_test(NAME tq-tp5-q8-f32-oracle COMMAND tq_tp5_q8_f32_tests)
        set_tests_properties(tq-tp5-q8-f32-oracle PROPERTIES TIMEOUT 120)
        add_executable(tq_tp5_flash_f32_tests tests/test-flash-f32.cpp)
        target_compile_features(tq_tp5_flash_f32_tests PRIVATE cxx_std_17)
        target_compile_options(tq_tp5_flash_f32_tests PRIVATE -fno-fast-math -ffp-contract=off)
        target_link_libraries(tq_tp5_flash_f32_tests PRIVATE llama)
        add_test(NAME tq-tp5-flash-f32-oracle COMMAND tq_tp5_flash_f32_tests)
        set_tests_properties(tq-tp5-flash-f32-oracle PROPERTIES TIMEOUT 120)
    endif()
endif()
