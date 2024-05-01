#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <string>
#include <vector>

#include <fmt/core.h>

#include "nvector.hpp"
#include "support.hpp"
#include "kmeans.hpp"

int main(int argc, char **argv) {

    if (argc != 6) {
        fmt::println(stderr, "Incorrect arguments");
        fmt::println(stderr, "\tNumber of dimensions");
        fmt::println(stderr, "\tNumber of points");
        fmt::println(stderr, "\tInput data file");
        fmt::println(stderr, "\tNumber of classes");
        fmt::println(stderr, "\tMax number of iterations");
        exit(EXIT_FAILURE);
    }

    uint8_t num_dimensions = std::atoi(argv[1]);
    uint32_t num_points = std::atoi(argv[2]);
    std::string infile_path = std::string(argv[3]);
    uint32_t num_classes = std::atoi(argv[4]);
    uint32_t max_iterations = std::atoi(argv[5]);


    FILE *infile = fopen(infile_path.c_str(), "r+");
    if (infile == nullptr) {
        fmt::println(stderr, "Failed to open file {}", infile_path);
        exit(EXIT_FAILURE);
    }

    std::vector<NVector> points = load_points(num_points, num_dimensions, infile);

    kmeans(points, num_classes, max_iterations);

    fclose(infile);

    return 0;
}