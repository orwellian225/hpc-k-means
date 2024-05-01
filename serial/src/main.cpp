#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <string>
#include <vector>

#include <fmt/core.h>

#include "nvector.hpp"
#include "support.hpp"

int main(int argc, char **argv) {

    if (argc != 4) {
        fmt::println(stderr, "Incorrect arguments");
        fmt::println(stderr, "\tNumber of dimensions");
        fmt::println(stderr, "\tNumber of points");
        fmt::println(stderr, "\tInput data file");
        exit(EXIT_FAILURE);
    }

    uint8_t num_dimensions = std::atoi(argv[1]);
    uint32_t num_points = std::atoi(argv[2]);
    std::string infile_path = std::string(argv[3]);


    FILE *infile = fopen(infile_path.c_str(), "r+");
    if (infile == nullptr) {
        fmt::println(stderr, "Failed to open file {}", infile_path);
        exit(EXIT_FAILURE);
    }

    std::vector<NVector> points = load_points(num_points, num_dimensions, infile);


    for (uint32_t i = 0; i < num_points; ++i) {
        fmt::println("Point {}: {}", i, points[i].to_string());
    }

    fclose(infile);

    return 0;
}