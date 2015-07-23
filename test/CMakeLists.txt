project(aruco-test)

include(ExternalProject)
ExternalProject_Add(googletest
  URL https://googletest.googlecode.com/files/gtest-1.7.0.zip
  SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/gtest
  BUILD_IN_SOURCE TRUE
  CMAKE_ARGS -DBUILD_SHARED_LIBS=TRUE
  INSTALL_COMMAND "" )

link_directories(${CMAKE_CURRENT_BINARY_DIR}/gtest/)

add_executable(core_tests core_tests.cpp)
add_dependencies(core_tests googletest)
target_compile_options(core_tests PRIVATE -std=c++11)
target_include_directories(core_tests PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/gtest/include)
target_link_libraries(core_tests PRIVATE aruco gtest_main gtest)

add_executable(perf_tests perf_tests.cpp)
add_dependencies(perf_tests googletest)
target_compile_options(perf_tests PRIVATE 
    -std=c++11 
    -DARUCO_VERSION="${PROJECT_VERSION}")
    
target_include_directories(perf_tests PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/gtest/include)
target_link_libraries(perf_tests PRIVATE aruco gtest_main gtest)
