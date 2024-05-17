#pragma once

#include <mpi.h>

#include "support.hpp"

void classify_kmeans(
    const uint8_t dimension, const uint32_t num_points, const uint32_t num_classes,
    const float *points, float *centroids, uint32_t *classes,
    uint32_t max_iterations, TimeBreakdown *timer,
    const int32_t rank, const int32_t size, const MPI_Comm comm
);
