
cmake_minimum_required(VERSION 3.16)

project(HPC-K-Means DESCRIPTION "COMS4040A HPC Project: K-Means clustering on CUDA and MPI" LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS Off)

add_subdirectory(lib/fmt)

add_executable(kmeans-serial
    serial/src/main.cpp serial/src/support.cpp
    src/nvector.cpp
)
target_include_directories(kmeans-serial PUBLIC include serial/include PUBLIC lib/fmt/include)
target_link_directories(kmeans-serial PRIVATE src serial/src lib/fmt/src)
target_link_libraries(kmeans-serial fmt)

