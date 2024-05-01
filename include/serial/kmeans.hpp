#pragma once

#include <vector>

#include "nvector.hpp"

std::vector<NVector> random_centroids(uint32_t num_centroids, uint8_t num_dimensions, float min, float max);
void classify_kmeans(const std::vector<NVector>& points, uint32_t num_classes, uint32_t max_iterations);