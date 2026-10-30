cmake_minimum_required(VERSION 3.16)
project(TQRelay VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CTest)

add_library(tq_relay STATIC src/ledger.cpp src/submissions.cpp src/plan.cpp)
add_library(TQRelay::relay ALIAS tq_relay)
set_target_properties(tq_relay PROPERTIES
    EXPORT_NAME relay
    OUTPUT_NAME tq-relay
    POSITION_INDEPENDENT_CODE ON)
target_compile_features(tq_relay PUBLIC cxx_std_17)
target_include_directories(tq_relay PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -Werror)
endif()

if(BUILD_TESTING)
    add_executable(tq_relay_ledger_tests tests/ledger-test.cpp)
    target_link_libraries(tq_relay_ledger_tests PRIVATE TQRelay::relay)
    add_test(NAME tq_relay.ledger COMMAND tq_relay_ledger_tests)

    add_executable(tq_relay_submissions_tests tests/submissions-test.cpp)
    target_link_libraries(tq_relay_submissions_tests PRIVATE TQRelay::relay)
    add_test(NAME tq_relay.submissions COMMAND tq_relay_submissions_tests)

    add_executable(tq_relay_plan_tests tests/plan-test.cpp)
    target_link_libraries(tq_relay_plan_tests PRIVATE TQRelay::relay)
    add_test(NAME tq_relay.plan COMMAND tq_relay_plan_tests)

    add_executable(relay_plan_dump tests/relay-plan-dump.cpp)
    target_link_libraries(relay_plan_dump PRIVATE TQRelay::relay)

    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(NAME tq_relay.python_crosscheck
            COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/crosscheck-plan.py
                    $<TARGET_FILE:relay_plan_dump>
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/python-reference)
        set_tests_properties(tq_relay.python_crosscheck PROPERTIES SKIP_RETURN_CODE 77)
    endif()
endif()

set(TQ_RELAY_PACKAGE_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/TQRelay")
configure_package_config_file(cmake/TQRelayConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/TQRelayConfig.cmake"
    INSTALL_DESTINATION "${TQ_RELAY_PACKAGE_DIR}")
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/TQRelayConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion)
install(TARGETS tq_relay EXPORT TQRelayTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")
install(DIRECTORY include/ DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(EXPORT TQRelayTargets
    FILE TQRelayTargets.cmake
    NAMESPACE TQRelay::
    DESTINATION "${TQ_RELAY_PACKAGE_DIR}")
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/TQRelayConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/TQRelayConfigVersion.cmake"
    DESTINATION "${TQ_RELAY_PACKAGE_DIR}")
install(FILES LICENSE DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/TQRelay")
install(FILES README.md DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/TQRelay")
