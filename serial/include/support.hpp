#pragma once

#include <stdio.h>
#include <vector>

#include <nvector.hpp>

std::vector<NVector> load_points(uint32_t num_points, uint8_t num_dimensions, FILE *infile);