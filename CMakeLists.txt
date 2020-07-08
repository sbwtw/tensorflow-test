cmake_minimum_required(VERSION 3.17)
project(tf_ios)

set(CMAKE_CXX_STANDARD 14)

find_package(PkgConfig)

pkg_check_modules(TensorFlow REQUIRED tensorflow)

link_directories(${TensorFlow_LIBRARY_DIRS})
include_directories(${TensorFlow_INCLUDE_DIRS})
add_compile_definitions(${TensorFlow_CFLAGS_OTHER})
add_executable(tfios library.cpp)
target_link_libraries(tfios ${TensorFlow_LIBRARIES})