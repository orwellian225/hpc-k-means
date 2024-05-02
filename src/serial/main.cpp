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

    std::vector<NVector> points = load_points(num_points, num_dimensions, infile_path);
    float max_range = points.size() - points.size() / 2., min_range = -max_range;

    auto start = std::chrono::high_resolution_clock::now();
    /* std::vector<NVector> centroids = random_centroids(num_classes, num_dimensions, min_range, max_range); */
    std::vector<NVector> centroids = kmeansplusplus_centroids(num_classes, num_dimensions, points);
    std::vector<point_class> classifications = classify_kmeans(points, centroids, max_iterations);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float, std::milli> duration(end - start);
    save_classification(points, centroids, classifications, outfile_path);
    fmt::println("Time: {:.2f} ms", duration.count());

    return 0;
}
