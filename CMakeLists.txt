cmake_minimum_required(VERSION 3.16)
project(transformer LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 REQUIRED NO_MODULE)

add_subdirectory(src)
add_subdirectory(example)

include(CTest)
if(BUILD_TESTING)
    add_subdirectory(test)
endif()
