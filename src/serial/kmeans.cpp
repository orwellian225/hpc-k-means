#include <vector>
#include <random>
#include <tuple>

#include <fmt/core.h>

#include "serial/kmeans.hpp"
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


std::vector<point_class> classify_kmeans(const std::vector<NVector>& points, std::vector<NVector>& centroids, uint32_t max_iterations) {
    uint8_t num_dimensions = points[0].num_dimensions;
    uint32_t num_classes = centroids.size();

    std::vector<NVector> previous_centroids;
    for (uint32_t k = 0; k < num_classes; ++k)
        previous_centroids.push_back(NVector(centroids[k]));

    std::vector<point_class> classifications;

    for (uint32_t i = 0; i < max_iterations; ++i) {
        classifications.clear();

        for (size_t p = 0; p < points.size(); ++p) {
            uint32_t closest_class = 0;
            for (uint32_t k = 0; k < num_classes; ++k) {
                if (points[p].distance_to(centroids[k]) < points[p].distance_to(centroids[closest_class])) {
                    closest_class = k;
                }
            }

            classifications.push_back({ p, closest_class });
        }

        // reset centroids
        for (uint32_t k = 0; k < num_classes; ++k)
            for (uint8_t d = 0; d < num_dimensions; ++d)
                centroids[k][d] = 0.0;

        // update centroids to mean of all points in the class
        std::vector<float> points_per_class(num_classes, 0.);
        for (size_t p = 0; p < points.size(); ++p) {
            uint32_t current_class = classifications[p].second;
            centroids[current_class] += points[p]; 
            points_per_class[current_class] += 1.;
        }
        for (uint32_t k = 0; k < num_classes; ++k)
            centroids[k] /= points_per_class[k];

        // Convergence check
        bool converged = true;
        for (uint32_t k = 0; k < num_classes; ++k) {
            converged = converged && centroids[k].distance_to(previous_centroids[k]) < 1e-3;
        }

        if (converged)
            break;

        previous_centroids.clear();
        for (uint32_t k = 0; k < num_classes; ++k)
            previous_centroids.push_back(NVector(centroids[k]));
    }

    return classifications;
}