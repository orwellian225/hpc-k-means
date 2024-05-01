#include <stdio.h>
#include <vector>
#include <string>

#include <fmt/core.h>

#include "nvector.hpp"
#include "support.hpp"

std::vector<NVector> load_points(uint32_t num_points, uint8_t num_dimensions, std::string infile_path) {
    FILE *infile = fopen(infile_path.c_str(), "r+");
    if (infile == nullptr) {
        fmt::println(stderr, "Failed to open file {}", infile_path);
        exit(EXIT_FAILURE);
    }

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

    fclose(infile);

    return std::vector<NVector>(points, points + num_points);
}

void save_classification(const std::vector<NVector>& points, const std::vector<NVector>& centroids, const std::vector<std::pair<size_t, uint32_t>>& classifications, std::string outfile_path) {
    FILE *outfile = fopen(outfile_path.c_str(), "w+");
    if (outfile == nullptr) {
        fmt::println(stderr, "Failed to open file {}", outfile_path);
        exit(EXIT_FAILURE);
    }

    for (auto point: points)
        fmt::println(outfile, "{}", point.to_csv_string());
    fmt::println(outfile, "");
    for (auto centroid: centroids)
        fmt::println(outfile, "{}", centroid.to_csv_string());
    fmt::println(outfile, "");
    for (auto classification: classifications)
        fmt::println(outfile, "{},{}", classification.first, classification.second);

    fclose(outfile);
}