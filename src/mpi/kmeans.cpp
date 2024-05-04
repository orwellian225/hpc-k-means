#include <algorithm>
#include <iterator>
#include <numeric>
#include <vector>
#include <random>
#include <tuple>
#include <mpi.h>

#include "mpi/kmeans.hpp"
#include "nvector.hpp"

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