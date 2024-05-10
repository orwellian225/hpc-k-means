#pragma once

#include <string>
#include <vector>

#include "nvector.hpp"

std::vector<NVector> load_points(uint32_t num_points, uint8_t num_dimensions, std::string infile_path);
void save_classification(const std::vector<NVector>& points, const std::vector<NVector>& centroids, const std::vector<uint32_t>& classifications, std::string outfile_path);