#include <algorithm>
#include <iterator>
#include <numeric>
#include <vector>
#include <random>
#include <tuple>

#include <fmt/core.h>

#include "fmt/base.h"
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

std::vector<NVector> kmeansplusplus_centroids(uint32_t num_centroids, uint8_t num_dimensions, const std::vector<NVector> &points) {
    std::random_device device;
    std::mt19937 rng(device());
    std::uniform_int_distribution<size_t> points_uniform_distribution(0, points.size());

    std::vector<NVector> centroids;

    std::vector<size_t> selected_points;
    selected_points.push_back(points_uniform_distribution(rng));
    size_t most_recent_centroid = 0;

    centroids.push_back(NVector(points[selected_points[0]]));

    for (size_t i = 1; i < num_centroids; ++i) {

        std::vector<float> probabilities(points.size());
        for (size_t p = 0; p < points.size(); ++p) {

            bool selected_point_before = false;
            for (auto sp: selected_points) {
                if (sp == p) {
                    probabilities[p] = 0.;
                    selected_point_before = true;
                    break;
                }
            }
            if (selected_point_before)
                continue;
            

            size_t closest_centroid = 0;
            float distance_to_max = points[p].distance_to(centroids[closest_centroid]);
            probabilities[p] = distance_to_max;
            for (size_t j = 0; j < most_recent_centroid; ++j) {
                float distance_to_current = points[p].distance_to(centroids[j]);
                if (distance_to_current < distance_to_max) {
                    closest_centroid = j;
                    probabilities[p] = distance_to_current;
                }
            }
        }

        float sum_sqr_distances = 0;
        for (auto d: probabilities)
            sum_sqr_distances += d*d;
        for (auto d: probabilities)
            d /= sum_sqr_distances;

        std::discrete_distribution<size_t> points_distance_distribution(probabilities.begin(), probabilities.end());
        ++most_recent_centroid;
        selected_points.push_back(points_distance_distribution(rng));

        centroids.push_back(points[selected_points[most_recent_centroid]]);
    }

    return centroids;
}


std::vector<uint32_t> classify_kmeans(const std::vector<NVector>& points, std::vector<NVector>& centroids, uint32_t max_iterations) {
    uint8_t num_dimensions = points[0].num_dimensions;
    uint32_t num_classes = centroids.size();

    std::vector<NVector> previous_centroids;
    for (uint32_t k = 0; k < num_classes; ++k)
        previous_centroids.push_back(NVector(centroids[k]));

    std::vector<uint32_t> classifications;

    for (uint32_t i = 0; i < max_iterations; ++i) {
        // fmt::print("\rIteration {: <7} of {: <7}", i, max_iterations);
        // fflush(stdout);

        classifications.clear();

        for (size_t p = 0; p < points.size(); ++p) {
            uint32_t closest_class = 0;
            for (uint32_t k = 0; k < num_classes; ++k) {
                if (points[p].distance_to(centroids[k]) < points[p].distance_to(centroids[closest_class])) {
                    closest_class = k;
                }
            }

            classifications.push_back(closest_class);
        }

        // reset centroids
        for (uint32_t k = 0; k < num_classes; ++k)
            for (uint8_t d = 0; d < num_dimensions; ++d) 
                centroids[k][d] = 0.0;

        // update centroids to mean of all points in the class
        std::vector<float> points_per_class(num_classes, 0.);
        for (size_t p = 0; p < points.size(); ++p) {
            uint32_t current_class = classifications[p];
            centroids[current_class] += points[p]; 
            points_per_class[current_class] += 1.;
        }

        for (uint32_t k = 0; k < num_classes; ++k)
            centroids[k] /= points_per_class[k] != 0 ? points_per_class[k] : 1;

        // Convergence check
        bool converged = true;
        for (uint32_t k = 0; k < num_classes; ++k) {
            /* fmt::println("({}) -> ({}) = {}", centroids[k].to_string(), previous_centroids[k].to_string(), centroids[k].distance_to(previous_centroids[k])); */
            converged = converged && centroids[k].distance_to(previous_centroids[k]) < 1e-3;
        }

        if (converged)
            break;

        previous_centroids.clear();
        for (uint32_t k = 0; k < num_classes; ++k)
            previous_centroids.push_back(NVector(centroids[k]));
    }

    fmt::println("");
    return classifications;
}
