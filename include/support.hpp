#pragma once

#include <string>
#include <vector>

float nvec_distance(const float *nvec_a, const float *nvec_b, const uint8_t dimension);
std::string nvec_to_csv_string(const float *nvec, const uint8_t dimension);

void save_classifications(
    const uint32_t num_points, const uint32_t num_classes, const uint8_t dimension,
    const float *points, const float *centroids, const uint32_t *classes, 
    std::string outfile_path
);

void load_points(
    const uint32_t num_points, const uint8_t dimension, 
    float *points,
    std::string infile_path
);

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