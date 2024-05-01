#pragma once

#include <vector>

#include "nvector.hpp"

void kmeans(const std::vector<NVector>& points, uint32_t num_classes, uint32_t max_iterations);