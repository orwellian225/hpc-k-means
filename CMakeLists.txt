
cmake_minimum_required(VERSION 3.16)

project(HPC-K-Means DESCRIPTION "COMS4040A HPC Project: K-Means clustering on CUDA and MPI" LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS Off)

add_subdirectory(lib/fmt)
find_package(MPI REQUIRED)

add_executable(kmeans-serial
    src/serial/main.cpp src/serial/kmeans.cpp
    src/nvector.cpp src/support.cpp
)
target_include_directories(kmeans-serial PUBLIC include PUBLIC lib/fmt/include)
target_link_directories(kmeans-serial PRIVATE src lib/fmt/src)
target_link_libraries(kmeans-serial fmt)

add_executable(kmeans-mpi
    src/mpi/main.cpp src/mpi/kmeans.cpp
    src/nvector.cpp src/support.cpp
)
target_include_directories(kmeans-mpi PUBLIC include PUBLIC lib/fmt/include)
target_link_directories(kmeans-mpi PRIVATE src lib/fmt/src)
target_link_libraries(kmeans-mpi fmt MPI::MPI_CXX)
