#pragma once

#include <stdint.h>

#include "support.hpp"

float nvec_distance(const float *nvec_a, const float *nvec_b, const uint8_t dimension);
void classify_points(
    const uint32_t num_points, const uint32_t num_classes, const uint8_t dimension, 
    const float *points, float *centroids, uint32_t *classes
);
void kmeans(
    const uint32_t num_points, const uint32_t num_classes, const uint8_t dimension, 
    const float *points, float *centroids, uint32_t *classes, 
    const uint32_t max_iterations, TimeBreakdown *timer
);