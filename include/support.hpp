#pragma once

#include <string>
#include <vector>

#include "nvector.hpp"

float nvec_distance(const float *nvec_a, const float *nvec_b, const uint8_t dimension);

std::vector<NVector> load_points(uint32_t num_points, uint8_t num_dimensions, std::string infile_path);
void save_classification(const std::vector<NVector>& points, const std::vector<NVector>& centroids, const std::vector<uint32_t>& classifications, std::string outfile_path);

void init_centroids(
    const uint32_t num_points, const uint32_t num_classes, const uint8_t dimension,
    const float *points, float *centroids,
    const uint32_t seed
);

struct TimeBreakdown {
    float initilize_time_ms;
    float cumulative_classify_time_ms;
    float cumulative_update_time_ms;
    float final_classify_time_ms;
};