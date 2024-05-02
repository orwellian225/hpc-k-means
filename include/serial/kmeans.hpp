#pragma once

#include <vector>

#include "nvector.hpp"

// size_t -> Index of the point
// uint32_t -> Index of centroid => class of point
typedef std::pair<size_t, uint32_t> point_class;

std::vector<NVector> random_centroids(uint32_t num_centroids, uint8_t num_dimensions, float min, float max);
std::vector<NVector> kmeansplusplus_centroids(uint32_t num_centroids, uint8_t num_dimensions, const std::vector<NVector>& points);
std::vector<point_class> classify_kmeans(const std::vector<NVector>& points, std::vector<NVector>& centroids, uint32_t max_iterations);
