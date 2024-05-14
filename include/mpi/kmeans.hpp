#pragma once

#include <vector>

#include <mpi.h>

#include "nvector.hpp"

std::vector<NVector> kmeansplusplus_centroids(
    uint32_t num_centroids, uint8_t num_dimensions, 
    const std::vector<NVector>& points
);

std::vector<uint32_t> classify_kmeans(
    const uint8_t dimension, const uint32_t num_points, const uint32_t num_classes,
    const float *points,
    float *centroids,
    uint32_t max_iterations,
    const int32_t rank, const int32_t size, const MPI_Comm comm
);