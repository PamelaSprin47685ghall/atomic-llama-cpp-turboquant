cmake_minimum_required(VERSION 3.21)
project(TQRelayProbe VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)

set(TQ_RELAY_PROBE_ARCHS "gfx1030" CACHE STRING "HIP architectures for the probe")
option(TQ_RELAY_PROBE_BUILD_TESTS "Build the probe self-checks that need no GPU." ON)
option(TQ_RELAY_PROBE_BUILD_HIP "Build HIP tools when the ROCm toolchain is available." ON)

add_library(tq_relay_probe_common STATIC src/evidence.cpp)
target_compile_features(tq_relay_probe_common PUBLIC cxx_std_17)
target_include_directories(tq_relay_probe_common PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/tq-relay-probe>)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(tq_relay_probe_common PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

if(TQ_RELAY_PROBE_BUILD_TESTS)
    include(CTest)
    add_executable(tq_relay_probe_config_tests tests/config-test.cpp)
    target_link_libraries(tq_relay_probe_config_tests PRIVATE tq_relay_probe_common)
    add_test(NAME tq_relay_probe.config COMMAND tq_relay_probe_config_tests)
    # The evidence validator is the operator's pre-ship check; its self-test
    # runs here too so the pipeline proves the tool before anyone relies on
    # it. Python is already a fork-tooling dependency (build.py, guards).
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(NAME tq_relay_probe.validator
             COMMAND ${Python3_EXECUTABLE}
                     ${CMAKE_CURRENT_SOURCE_DIR}/validate-report.py --self-test)
endif()

if(NOT TQ_RELAY_PROBE_BUILD_HIP)
    message(STATUS "HIP tools disabled; building CPU evidence checks only")
    return()
endif()
find_package(hip QUIET)
if(NOT hip_FOUND)
    message(STATUS "HIP not found; the relay platform probe executable is not built")
    return()
endif()
enable_language(HIP)

# Native HIP compilation supplies the device link flags. Link the host
# runtime as a real imported library as well. Its directory can be an
# implicit HIP compiler search path, which CMake omits from automatic
# RPATH generation even though the OS loader does not know that path.
set(tq_relay_probe_hip_runtime_dir "$<TARGET_FILE_DIR:hip::amdhip64>")

add_executable(tq-relay-probe src/main.cpp)
set_source_files_properties(src/main.cpp PROPERTIES LANGUAGE HIP)
target_link_libraries(tq-relay-probe PRIVATE tq_relay_probe_common hip::host)
set_target_properties(tq-relay-probe PROPERTIES
    HIP_ARCHITECTURES "${TQ_RELAY_PROBE_ARCHS}"
    BUILD_RPATH "${tq_relay_probe_hip_runtime_dir}"
    INSTALL_RPATH "${tq_relay_probe_hip_runtime_dir}"
    INSTALL_RPATH_USE_LINK_PATH TRUE)
target_compile_features(tq-relay-probe PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(tq-relay-probe PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Clock-rate calibration utility for the operator: re-derives the clock64()
# rate the probe's spin budgets are anchored to, on the machine it will
# actually run on. Diagnostic only; produces no evidence files.
add_executable(tq-clock-rate tools/tq-clock-rate.cpp)
set_source_files_properties(tools/tq-clock-rate.cpp PROPERTIES LANGUAGE HIP)
target_link_libraries(tq-clock-rate PRIVATE hip::host)
set_target_properties(tq-clock-rate PROPERTIES
    HIP_ARCHITECTURES "${TQ_RELAY_PROBE_ARCHS}"
    BUILD_RPATH "${tq_relay_probe_hip_runtime_dir}"
    INSTALL_RPATH "${tq_relay_probe_hip_runtime_dir}"
    INSTALL_RPATH_USE_LINK_PATH TRUE)
target_compile_features(tq-clock-rate PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(tq-clock-rate PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(TQ_RELAY_PROBE_BUILD_TESTS)
    add_test(NAME tq_relay_probe.cli
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cli.py
                     $<TARGET_FILE:tq-relay-probe> $<TARGET_FILE:tq-clock-rate>)
endif()

install(TARGETS tq-relay-probe RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
install(TARGETS tq-clock-rate RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
# The evidence validator is part of the operator's pre-ship contract
# (README: archive evidence only after it validates); ship it with the
# binaries so an installed tree can stand alone. It is already executable
# in the repository; PROGRAMS preserves that mode.
install(PROGRAMS validate-report.py DESTINATION "${CMAKE_INSTALL_BINDIR}")
install(FILES LICENSE DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/TQRelayProbe")
install(FILES README.md DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/TQRelayProbe")
