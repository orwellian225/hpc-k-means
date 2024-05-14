#include <chrono>
#include <ratio>
#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <stddef.h>
#include <string>
#include <vector>

#include <mpi.h>

#include <fmt/core.h>

#include "mpi/kmeans.hpp"
#include "nvector.hpp"
#include "support.hpp"

int main(int argc, char **argv) {
    if (argc != 7) { 
        fmt::println(stderr, "Incorrect arguments - {} != 7", argc);
        fmt::println(stderr, "\tNumber of dimensions");
        fmt::println(stderr, "\tNumber of points");
        fmt::println(stderr, "\tInput data file");
        fmt::println(stderr, "\tOutput data file");
        fmt::println(stderr, "\tNumber of classes");
        fmt::println(stderr, "\tMax number of iterations");
        exit(EXIT_FAILURE);
    }

    uint8_t num_dimensions = std::atoi(argv[1]);
    uint32_t num_points = std::atoi(argv[2]);
    std::string infile_path = std::string(argv[3]);
    std::string outfile_path = std::string(argv[4]);
    uint32_t num_classes = std::atoi(argv[5]);
    uint32_t max_iterations = std::atoi(argv[6]);

    int32_t world_size, world_rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    bool infile_exists;
    if (world_rank == 0) {
        infile_exists = access(infile_path.c_str(), R_OK) != 0;
    }

    MPI_Bcast(&infile_exists, 1, MPI_CXX_BOOL, world_rank, MPI_COMM_WORLD);
    if (infile_exists) {
        fmt::println(stderr, "Cannot find file {}", infile_path);
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    double start = MPI_Wtime();

    float *centroids = new float[num_classes * num_dimensions];
    float *points = new float[num_points * num_dimensions];
    
    if (world_rank == 0) {
        std::vector<NVector> points_vec = load_points(num_points, num_dimensions, infile_path);
        std::vector<NVector> generated_centroids = kmeansplusplus_centroids(num_classes, num_dimensions, points_vec);
        // send to all nodes

        for (size_t i = 0; i < num_points; ++i)
            for (size_t d = 0; d < num_dimensions; ++d)
                points[num_dimensions * i + d] = points_vec[i][d];

        for (size_t i = 0; i < num_classes; ++i) 
            for (size_t d = 0; d < num_dimensions; ++d)
                centroids[num_dimensions * i + d] = generated_centroids[i][d];
    } 

    MPI_Bcast(centroids, num_classes * num_dimensions, MPI_FLOAT, 0, MPI_COMM_WORLD);
    std::vector<uint32_t> classifications = classify_kmeans(num_dimensions, num_points, num_classes, points, centroids, max_iterations, world_rank, world_size, MPI_COMM_WORLD);
    double end = MPI_Wtime();


    std::vector<NVector> points_vec, centroid_vec;
    for (size_t i = 0; i < num_points; ++i) {
        NVector temp(num_dimensions, 0.);
        for (uint8_t d = 0; d < num_dimensions; ++d)
            temp[d] = points[num_dimensions * i + d];

        points_vec.push_back(temp);
    }

    for (size_t i = 0; i < num_classes; ++i) {
        NVector temp(num_dimensions, 0.);
        for (uint8_t d = 0; d < num_dimensions; ++d)
            temp[d] = centroids[num_dimensions * i + d];

        centroid_vec.push_back(temp);
    }

    if (world_rank == 0) {
        double duration = (end - start) * 1000; // MPI_Wtime is in seconds
        save_classification(points_vec, centroid_vec, classifications, outfile_path);
        fmt::println("Time: {:.2f} ms", duration);
    }

    MPI_Finalize();
    return 0;
}