#pragma once

#include <vector>

#include <mpi.h>

#include "nvector.hpp"

typedef std::pair<size_t, uint32_t> PointClass;

std::vector<NVector> kmeansplusplus_centroids(
    uint32_t num_centroids, uint8_t num_dimensions, 
    const std::vector<NVector>& points
);

std::vector<PointClass> classify_kmeans(
    const std::vector<NVector>& points, 
    std::vector<NVector> centroids, 
    uint32_t max_iterations,
    const int32_t rank, const int32_t size, const MPI_Comm comm
);