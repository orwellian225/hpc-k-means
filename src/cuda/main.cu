#include <chrono>
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

    float *points = new float[num_points * num_dimensions];
    float *centroids = new float[num_classes * num_dimensions];
    uint32_t *classes = new uint32_t[num_points];

    TimeBreakdown timer = {};

    load_points(num_points, num_dimensions, points, infile_path);

    auto global_start = std::chrono::high_resolution_clock::now();

    auto initilize_start = std::chrono::high_resolution_clock::now();
    init_centroids(
        num_points, num_classes, num_dimensions, 
        points, centroids,
        seed
    );
    auto initilize_end = std::chrono::high_resolution_clock::now();
    timer.initilize_time_ms = std::chrono::duration<float, std::milli>(initilize_end - initilize_start).count();

    classify_kmeans(
        num_dimensions, num_points, num_classes,
        points, centroids, classes,
        max_iterations, &timer
    );

    auto global_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration(global_end - global_start);

    save_classifications(
        num_points, num_classes, num_dimensions,
        points, centroids, classes,
        outfile_path
    );
    fmt::println(
            "{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}", 
            duration.count(), timer.initilize_time_ms, 
            timer.cumulative_classify_time_ms, timer.cumulative_update_time_ms, 
            timer.final_classify_time_ms
    );
}
