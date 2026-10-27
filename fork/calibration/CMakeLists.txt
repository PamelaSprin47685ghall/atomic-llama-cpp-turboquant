cmake_minimum_required(VERSION 3.16)
project(TQCalibration VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CTest)

add_library(tq_calibration STATIC src/reader.cpp)
add_library(TQCalibration::calibration ALIAS tq_calibration)
set_target_properties(tq_calibration PROPERTIES EXPORT_NAME calibration OUTPUT_NAME tq-calibration POSITION_INDEPENDENT_CODE ON)
target_compile_features(tq_calibration PUBLIC cxx_std_17)
target_include_directories(tq_calibration PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
if(BUILD_TESTING)
    add_executable(tq_calibration_test tests/test-reader.cpp)
    target_link_libraries(tq_calibration_test PRIVATE TQCalibration::calibration)
    add_test(NAME tq_calibration.format COMMAND tq_calibration_test)
endif()

set(TQ_CALIBRATION_PACKAGE_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/TQCalibration")
configure_package_config_file(cmake/TQCalibrationConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/TQCalibrationConfig.cmake"
    INSTALL_DESTINATION "${TQ_CALIBRATION_PACKAGE_DIR}")
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/TQCalibrationConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}" COMPATIBILITY SameMajorVersion)
install(TARGETS tq_calibration EXPORT TQCalibrationTargets ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")
install(DIRECTORY include/ DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(EXPORT TQCalibrationTargets FILE TQCalibrationTargets.cmake NAMESPACE TQCalibration:: DESTINATION "${TQ_CALIBRATION_PACKAGE_DIR}")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/TQCalibrationConfig.cmake" "${CMAKE_CURRENT_BINARY_DIR}/TQCalibrationConfigVersion.cmake"
    DESTINATION "${TQ_CALIBRATION_PACKAGE_DIR}")
install(FILES LICENSE DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/TQCalibration")
install(FILES README.md DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/TQCalibration")
