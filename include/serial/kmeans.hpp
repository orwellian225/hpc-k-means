#pragma once

#include <vector>

#include "nvector.hpp"

std::vector<NVector> random_centroids(uint32_t num_centroids, uint8_t num_dimensions, float min, float max);
std::vector<NVector> kmeansplusplus_centroids(uint32_t num_centroids, uint8_t num_dimensions, const std::vector<NVector>& points);
std::vector<uint32_t> classify_kmeans(const std::vector<NVector>& points, std::vector<NVector>& centroids, uint32_t max_iterations);
