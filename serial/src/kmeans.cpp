#include <vector>
#include <random>

#include <fmt/core.h>

#include "kmeans.hpp"
#include "nvector.hpp"

void kmeans(const std::vector<NVector>& points, uint32_t num_classes, uint32_t max_iterations) {
    uint8_t num_dimensions = points[0].num_dimensions;

    std::vector<NVector> centroids;
    float max_range = points.size() - points.size() / 2.;
    float min_range = -max_range;

    std::random_device device;
    std::mt19937 rng(device());
    std::uniform_real_distribution<float> centroid_distribution(min_range, max_range);

    NVector *temp_centroids = new NVector[num_classes];
    for (uint32_t k = 0; k < num_classes; ++k) {
        float *values = new float[num_dimensions];
        for (uint8_t d = 0; d < num_dimensions; ++d) {
            values[d] = centroid_distribution(rng);
        }

        temp_centroids[k].num_dimensions = num_dimensions;
        temp_centroids[k].data = values;
    }
    centroids = std::vector<NVector>(temp_centroids, temp_centroids + num_classes);

    for (uint32_t i = 0; i < centroids.size(); ++i) {
        fmt::println("Centroid {}: {}", i, centroids[i].to_string());
    }

}