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
#include "support.hpp"

int main(int argc, char **argv) {
    if (argc != 8) {
        fmt::println(stderr, "Incorrect arguments");
        fmt::println(stderr, "\tNumber of dimensions");
        fmt::println(stderr, "\tNumber of points");
        fmt::println(stderr, "\tInput data file");
        fmt::println(stderr, "\tOutput data file");
        fmt::println(stderr, "\tNumber of classes");
        fmt::println(stderr, "\tMax number of iterations");
        fmt::println(stderr, "\tRandom seed");
        exit(EXIT_FAILURE);
    }

    uint8_t num_dimensions = std::atoi(argv[1]);
    uint32_t num_points = std::atoi(argv[2]);
    std::string infile_path = std::string(argv[3]);
    std::string outfile_path = std::string(argv[4]);
    uint32_t num_classes = std::atoi(argv[5]);
    uint32_t max_iterations = std::atoi(argv[6]);
    uint32_t seed = std::stoi(argv[7]);

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

    float *points = new float[num_points * num_dimensions];
    float *centroids = new float[num_classes * num_dimensions];
    uint32_t *classes = new uint32_t[num_points];

    TimeBreakdown timer = {};

    if (world_rank == 0) {
        load_points(num_points, num_dimensions, points, infile_path);
    }

    double global_start = MPI_Wtime();

    double init_start = MPI_Wtime();
    if (world_rank == 0) {
        init_centroids(
            num_points, num_classes, num_dimensions, 
            points, centroids,
            seed
        );
    } 
    double init_end = MPI_Wtime();
    timer.initilize_time_ms = (init_end - init_start) * 1000;

    MPI_Bcast(centroids, num_classes * num_dimensions, MPI_FLOAT, 0, MPI_COMM_WORLD);
    classify_kmeans(
        num_dimensions, num_points, num_classes, 
        points, centroids, classes, 
        max_iterations, &timer,
        world_rank, world_size, MPI_COMM_WORLD
    );
    double global_end = MPI_Wtime();


    if (world_rank == 0) {
        double duration = (global_end - global_start) * 1000; // MPI_Wtime is in seconds
        save_classifications(
            num_points, num_classes, num_dimensions,
            points, centroids, classes,
            outfile_path
        );
        fmt::println(
                "{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}", 
                duration, timer.initilize_time_ms, 
                timer.cumulative_classify_time_ms, timer.cumulative_update_time_ms, 
                timer.final_classify_time_ms
        );
    }

    MPI_Finalize();
    return 0;
}
