#include <stdio.h>
#include <vector>
#include <string>
#include <random>

#include <fmt/core.h>

#include "nvector.hpp"
#include "support.hpp"

float nvec_distance(const float *nvec_a, const float *nvec_b, const uint8_t dimension) {
    float sum = 0.;

    for (uint8_t d = 0; d < dimension; ++d)
        sum += (nvec_a[d] - nvec_b[d]) * (nvec_a[d] - nvec_b[d]);

    return std::sqrt(sum);
}

void load_points(
    const uint32_t num_points, const uint8_t dimension, 
    float *points,
    std::string infile_path
) {
    FILE *infile = fopen(infile_path.c_str(), "r+");
    if (infile == nullptr) {
        fmt::println(stderr, "Failed to open file {}", infile_path);
        exit(EXIT_FAILURE);
    }

    char linebuffer[1024];
    size_t delimiter_pos;
    std::string line, token;

    for (uint32_t p = 0; p < num_points; ++p) {
        if (feof(infile)) {
            fmt::println(stderr, "Not enough points in specified file");
            fmt::println(stderr, "\tOnly found {} of {}", p, num_points);
            exit(EXIT_FAILURE);
        }

        fgets(linebuffer, 1024, infile);
        line = std::string(linebuffer);
        line.pop_back(); // remove newline char

        for (uint8_t d = 0; d < dimension; ++d) {
            delimiter_pos = line.find(",");
            token = line.substr(0, delimiter_pos);
            line.erase(0, delimiter_pos + 1);
            points[dimension * p + d] = std::stof(token);
        }
    }

    fclose(infile);
}

void save_classification(const std::vector<NVector>& points, const std::vector<NVector>& centroids, const std::vector<uint32_t>& classifications, std::string outfile_path) {
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
    for (uint32_t p = 0; p < classifications.size(); ++p)
        fmt::println(outfile, "{},{}", p, classifications[p]);

    fclose(outfile);
}

void init_centroids(
    const uint32_t num_points, const uint32_t num_classes, const uint8_t dimension,
    const float *points, float *centroids,
    const uint32_t seed
) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<uint32_t> uniform_dist(0, num_points);

    uint32_t *selected_points = new uint32_t[num_classes];
    selected_points[0] = uniform_dist(rng);
    uint32_t most_recent_centroid = 0;

    for (uint8_t d = 0; d < dimension; ++d)
        centroids[dimension * 0 + d] = points[selected_points[0]];

    float *probabilites = new float[num_points];
    for (uint32_t k = 1; k < num_classes; ++k) {

        for (uint32_t p = 0; p < num_points; ++p) {

            bool selected_point_before = false;
            for (uint32_t sp = 0; sp < most_recent_centroid; ++sp) {
                if (p == sp) {
                    probabilites[p] = 0.;
                    selected_point_before = true;
                    break;
                }
            }

            if (selected_point_before)
                continue;

            uint32_t closest_centroid = 0;
            float closest_distance = nvec_distance(&points[dimension * p], &centroids[dimension * 0], dimension);
            probabilites[p] = closest_distance;
            float next_distance;
            for (uint32_t k = 0; k < most_recent_centroid; ++k) {
                next_distance = nvec_distance(&points[dimension * p], &centroids[dimension * k], dimension);
                if (next_distance < closest_distance) {
                    closest_centroid = k;
                    closest_distance = next_distance;
                    probabilites[p] = next_distance;
                }
            }
        }

        float sum_distances_sqr = 0.;
        for (uint32_t p = 0; p < num_points; ++p)
            sum_distances_sqr += probabilites[p] * probabilites[p];

        for (uint32_t p = 0; p < num_points; ++p)
            probabilites[p] / sum_distances_sqr;

        std::discrete_distribution distance_dist(probabilites, probabilites + num_points);
        ++most_recent_centroid;
        selected_points[most_recent_centroid] = distance_dist(rng);

        for (uint8_t d = 0; d < dimension; ++d)
            centroids[dimension * most_recent_centroid + d] = points[selected_points[most_recent_centroid]];
    }
    delete[] probabilites;

    delete[] selected_points;
}