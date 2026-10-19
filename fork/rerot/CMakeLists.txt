cmake_minimum_required(VERSION 3.16)
project(TQRerot VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CTest)

option(TQ_REROT_BUILD_TESTS "Build the independent RERoT tests" ${BUILD_TESTING})

add_library(tq_rerot src/plan.cpp src/view.cpp src/episode.cpp)
add_library(TQRerot::rerot ALIAS tq_rerot)
set_target_properties(tq_rerot PROPERTIES
    EXPORT_NAME rerot
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON)
target_compile_features(tq_rerot PUBLIC cxx_std_17)
target_include_directories(tq_rerot PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

install(TARGETS tq_rerot EXPORT TQRerotTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES LICENSE DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/TQRerot)
install(FILES README.md DESTINATION ${CMAKE_INSTALL_DOCDIR})
install(EXPORT TQRerotTargets FILE TQRerotTargets.cmake
    NAMESPACE TQRerot:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TQRerot)
configure_package_config_file(cmake/TQRerotConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/TQRerotConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TQRerot)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/TQRerotConfigVersion.cmake
    VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/TQRerotConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/TQRerotConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TQRerot)

if(TQ_REROT_BUILD_TESTS)
    enable_testing()
    add_executable(tq-rerot-test tests/test-rerot.cpp)
    target_link_libraries(tq-rerot-test PRIVATE TQRerot::rerot)
    set_target_properties(tq-rerot-test PROPERTIES CXX_EXTENSIONS OFF)
    foreach(test_name IN ITEMS dag transaction bounds oracle)
        add_test(NAME rerot-${test_name} COMMAND tq-rerot-test ${test_name})
    endforeach()
endif()
