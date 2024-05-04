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
        fmt::println(stderr, "Incorrect arguments");
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

    std::vector<NVector> points = load_points(num_points, num_dimensions, infile_path);
    std::vector<NVector> centroids;

    double start = MPI_Wtime();

    float *merged_dimension_data = new float[num_classes * num_dimensions];
    
    if (world_rank == 0) {
        std::vector<NVector> generated_centroids = kmeansplusplus_centroids(num_classes, num_dimensions, points);
        // send to all nodes

        for (size_t i = 0; i < num_classes * num_dimensions; i += num_dimensions) {
            for (uint8_t d = 0; d < num_dimensions; ++d)
                merged_dimension_data[i + d] = generated_centroids[i / num_dimensions].data[d];
        }

    } 

    MPI_Bcast(merged_dimension_data, num_classes * num_dimensions, MPI_FLOAT, 0, MPI_COMM_WORLD);

    for (size_t i = 0; i < num_classes * num_dimensions; i += num_dimensions) {
        centroids.push_back(NVector(num_dimensions, 0.0));
        for (uint8_t d = 0; d < num_dimensions; ++d) {
            centroids[i / num_dimensions][d] = merged_dimension_data[i + d];
        }
    }
    std::vector<uint32_t> classifications = classify_kmeans(points, centroids, max_iterations, world_rank, world_size, MPI_COMM_WORLD);
    double end = MPI_Wtime();

    if (world_rank == 0) {
        double duration = end - start;
        // save_classification(points, centroids, classifications, outfile_path);
        fmt::println("Time: {:.2f} ms", duration);
    }

    MPI_Finalize();
    return 0;
}