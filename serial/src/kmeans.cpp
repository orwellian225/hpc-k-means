#include <vector>
#include <random>

#include <fmt/core.h>

#include "kmeans.hpp"
#include "nvector.hpp"

std::vector<NVector> random_centroids(uint32_t num_centroids, uint8_t num_dimensions, float min, float max) {
    std::random_device device;
    std::mt19937 rng(device());
    std::uniform_real_distribution<float> centroid_distribution(min, max);

    NVector *temp_centroids = new NVector[num_centroids];
    for (uint32_t k = 0; k < num_centroids; ++k) {
        float *values = new float[num_dimensions];
        for (uint8_t d = 0; d < num_dimensions; ++d) {
            values[d] = centroid_distribution(rng);
        }

        temp_centroids[k].num_dimensions = num_dimensions;
        temp_centroids[k].data = values;
    }
    return std::vector<NVector>(temp_centroids, temp_centroids + num_centroids);
}

void classify_kmeans(const std::vector<NVector>& points, uint32_t num_classes, uint32_t max_iterations) {
    uint8_t num_dimensions = points[0].num_dimensions;

    float max_range = points.size() - points.size() / 2., min_range = -max_range;
    std::vector<NVector> centroids = random_centroids(num_classes, num_dimensions, min_range, max_range);

}