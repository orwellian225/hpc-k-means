#include <chrono>
#include <ratio>
#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <stddef.h>
#include <string>
#include <vector>

#include <fmt/core.h>

#include "nvector.hpp"
#include "support.hpp"
#include "serial/kmeans.hpp"

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


    float *points = new float[num_dimensions * num_points];
    float *centroids = new float[num_dimensions * num_classes];
    uint32_t *classes = new uint32_t[num_points];
    load_points(num_points, num_dimensions, points, infile_path);

    auto start = std::chrono::high_resolution_clock::now();
    
    init_centroids(
        num_points, num_classes, num_dimensions, 
        points, centroids,
        0
    );

    kmeans(
        num_points, num_classes, num_dimensions, 
        points, centroids, classes, 
        max_iterations, nullptr
    );

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float, std::milli> duration(end - start);

    std::vector<NVector> centroid_vec, points_vec;
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

    std::vector<uint32_t> classifications(classes, classes + num_points);
    save_classification(points_vec, centroid_vec, classifications, outfile_path);
    fmt::println("Time: {:.2f} ms", duration.count());

    return 0;
}
