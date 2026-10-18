cmake_minimum_required(VERSION 3.16)
project(TQRelayProbe VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)

set(TQ_RELAY_PROBE_ARCHS "gfx1030" CACHE STRING "HIP architectures for the probe")
option(TQ_RELAY_PROBE_BUILD_TESTS "Build the probe self-checks that need no GPU." ON)

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
endif()

find_package(hip QUIET)
if(NOT hip_FOUND)
    message(STATUS "HIP not found; the relay platform probe executable is not built")
    return()
endif()
enable_language(HIP)

add_executable(tq-relay-probe src/main.cpp)
set_source_files_properties(src/main.cpp PROPERTIES LANGUAGE HIP)
target_link_libraries(tq-relay-probe PRIVATE tq_relay_probe_common hip::device)
set_target_properties(tq-relay-probe PROPERTIES
    HIP_ARCHITECTURES "${TQ_RELAY_PROBE_ARCHS}")
target_compile_features(tq-relay-probe PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(tq-relay-probe PRIVATE -Wall -Wextra -Wpedantic)
endif()

install(TARGETS tq-relay-probe RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
install(FILES LICENSE DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/TQRelayProbe")
install(FILES README.md DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/TQRelayProbe")
