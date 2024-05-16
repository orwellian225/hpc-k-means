#include <stdio.h>
#include <unistd.h>

#include <fmt/core.h>

#include "nvector.hpp"
#include "support.hpp"
#include "cuda/kmeans.cuh"

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

    if (access(infile_path.c_str(), R_OK) != 0) {
        fmt::println(stderr, "Cannot find file {}", infile_path);
        exit(EXIT_FAILURE);
    }


    cudaEvent_t start, end;
    float duration_ms;

    float *centroids = new float[num_classes * num_dimensions];
    float *points = new float[num_points * num_dimensions];
    std::vector<NVector> points_loaded_vec = load_points(num_points, num_dimensions, infile_path);

    for (size_t i = 0; i < num_points; ++i)
        for (size_t d = 0; d < num_dimensions; ++d)
            points[num_dimensions * i + d] = points_loaded_vec[i][d];

    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start, 0);
    init_centroids(
        num_points, num_classes, num_dimensions, 
        points, centroids,
        0
    );

    std::vector<uint32_t> classifications = classify_kmeans(num_dimensions, num_points, num_classes, points, centroids, max_iterations);

    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&duration_ms, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

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

    save_classification(points_vec, centroid_vec, classifications, outfile_path);
    fmt::println("Time: {:.2f} ms", duration_ms);
}