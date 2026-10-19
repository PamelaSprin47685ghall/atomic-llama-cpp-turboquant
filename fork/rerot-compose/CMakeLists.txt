cmake_minimum_required(VERSION 3.16)
project(TQRerotCompose VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
find_package(llama CONFIG REQUIRED)
find_package(TQRerot CONFIG REQUIRED)
find_package(TQRelay CONFIG REQUIRED)
find_package(TQText CONFIG REQUIRED)

add_executable(tq_rerot_compose src/main.cpp src/native.cpp)
set_target_properties(tq_rerot_compose PROPERTIES OUTPUT_NAME tq-rerot-compose)
target_compile_features(tq_rerot_compose PRIVATE cxx_std_17)
target_link_libraries(tq_rerot_compose PRIVATE llama TQRerot::rerot TQRelay::relay TQText::text)
if(UNIX AND NOT APPLE)
    set_target_properties(tq_rerot_compose PROPERTIES INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
endif()
install(TARGETS tq_rerot_compose RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
install(FILES README.md DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/TQRerotCompose")
install(FILES LICENSE DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/TQRerotCompose")

include(CTest)
if(BUILD_TESTING)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(NAME tq_rerot_compose.cli
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cli.py"
                --binary "$<TARGET_FILE:tq_rerot_compose>")
endif()
