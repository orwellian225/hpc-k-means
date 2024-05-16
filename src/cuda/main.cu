#include <stdio.h>
#include <unistd.h>

#include <fmt/core.h>

#include "support.hpp"
#include "cuda/kmeans.cuh"

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

    if (access(infile_path.c_str(), R_OK) != 0) {
        fmt::println(stderr, "Cannot find file {}", infile_path);
        exit(EXIT_FAILURE);
    }


    cudaEvent_t start, end;
    float duration_ms;

    float *points = new float[num_points * num_dimensions];
    float *centroids = new float[num_classes * num_dimensions];
    uint32_t *classes = new uint32_t[num_points];

    load_points(num_points, num_dimensions, points, infile_path);

    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start, 0);
    init_centroids(
        num_points, num_classes, num_dimensions, 
        points, centroids,
        seed
    );

    classify_kmeans(
        num_dimensions, num_points, num_classes,
        points, centroids, classes,
        max_iterations
    );

    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&duration_ms, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    save_classifications(
        num_points, num_classes, num_dimensions,
        points, centroids, classes,
        outfile_path
    );
    fmt::println("{:.4f}", duration_ms);
}