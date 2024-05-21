#pragma once

#include "support.hpp"

void handle_cuda_error(cudaError_t error);

__device__ float d_vec_distance(float *vec_1, float *vec_2, uint8_t dimension);
__global__ void classify(cudaTextureObject_t points, float *centroids, uint32_t *classifications);

void classify_kmeans(
    const uint8_t dimension, const uint32_t num_points, const uint32_t num_classes,
    const float *points, float *centroids, uint32_t *classes,
    uint32_t max_iterations, TimeBreakdown *timer
);
