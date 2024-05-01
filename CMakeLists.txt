
cmake_minimum_required(VERSION 3.16)

project(HPC-K-Means DESCRIPTION "COMS4040A HPC Project: K-Means clustering on CUDA and MPI" LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS Off)

add_subdirectory(lib/fmt)

