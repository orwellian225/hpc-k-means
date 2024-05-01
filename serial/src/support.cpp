#include <stdio.h>
#include <vector>

#include <fmt/core.h>

#include "nvector.hpp"
#include "support.hpp"

std::vector<NVector> load_points(uint32_t num_points, uint8_t num_dimensions, FILE *infile) {
    NVector *points;

    char linebuffer[1024];
    size_t delimiter_pos;
    std::string line, token;

    points = new NVector[num_points];
    for (uint32_t i = 0; i < num_points; ++i) {
        if (feof(infile)) {
            fmt::println(stderr, "Not enough points in specified file");
            fmt::println(stderr, "\tOnly found {} of {}", i, num_points);
            exit(EXIT_FAILURE);
        }

        fgets(linebuffer, 1024, infile);
        line = std::string(linebuffer);
        line.pop_back(); // remove newline char

        float *values = new float[num_dimensions];
        for (uint8_t d = 0; d < num_dimensions; ++d) {
            delimiter_pos = line.find(",");
            token = line.substr(0, delimiter_pos);
            line.erase(0, delimiter_pos + 1);
            values[d] = std::stof(token);
        }

        points[i].num_dimensions = num_dimensions;
        points[i].data = values;
    }

    return std::vector<NVector>(points, points + num_points);
}