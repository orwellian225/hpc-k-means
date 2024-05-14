#include <algorithm>
#include <iterator>
#include <numeric>
#include <vector>
#include <random>
#include <tuple>
#include <mpi.h>
#include <stdio.h>

#include <fmt/core.h>

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

float vec_distance(const float *vec_1, const float *vec_2, uint8_t dimension) {

    float sum = 0;
    for (uint8_t d = 0; d < dimension; ++d)
        sum = (vec_1[d] - vec_2[d]) * (vec_1[d] - vec_2[d]);

    return std::sqrt(sum);
}

std::vector<uint32_t> classify_kmeans(
    const uint8_t dimension, const uint32_t num_points, const uint32_t num_classes,
    const float *points,
    float *centroids,
    uint32_t max_iterations,
    const int32_t rank, const int32_t size, const MPI_Comm comm
) {
    MPI_Datatype nvec_row_t, nvec_col_t;
    MPI_Op nvec_row_sum;

    MPI_Type_contiguous(dimension, MPI_FLOAT, &nvec_row_t);
    MPI_Type_vector(num_classes, 1, dimension, MPI_FLOAT, &nvec_col_t);

    MPI_Type_commit(&nvec_row_t);
    MPI_Type_commit(&nvec_col_t);

    uint32_t num_local_points = num_points / size + (rank < (num_points % size));
    int32_t *process_point_distribution = new int32_t[size];
    int32_t *process_point_displacements = new int32_t[size];
    for (uint32_t i = 0; i < size; ++i) {
        process_point_distribution[i] = num_points / size + (i < (num_points % size));
        process_point_displacements[i] = i > 0 ? process_point_displacements[i - 1] + process_point_distribution[i] : 0;
    }

    float *local_points = new float[dimension * num_local_points];
    MPI_Scatterv(points, process_point_distribution, process_point_displacements, nvec_row_t, local_points, num_local_points, nvec_row_t, 0, comm);

    uint32_t *local_classifications = new uint32_t[num_local_points];
    uint32_t *local_num_classifications_per_class = new uint32_t[num_classes];
    uint32_t *num_classifications_per_class = new uint32_t[num_classes];

    float *local_new_centroids = new float[num_classes * dimension];
    float *new_centroids = new float[num_classes * dimension];

    // fmt::println("{} Initial", rank);
    // for (uint32_t k = 0; k < num_classes; ++k) {
    //     fmt::print("\tCentroid {} ", k);
    //     for (uint8_t d = 0; d < dimension; ++d)
    //         fmt::print("{} ", centroids[dimension * k + d]);
    //     fmt::println("");
    // }

    // fmt::println("{} Points", rank);
    // for (uint32_t p = 0; p < num_local_points; ++p) {
    //     fmt::print("\tPoint {} ", p);
    //     for (uint8_t d = 0; d < dimension; ++d)
    //         fmt::print("{} ", local_points[dimension * p + d]);
    //     fmt::println("");
    // }

    // fmt::println("{} Centroids", rank);
    // for (uint32_t k = 0; k < num_classes; ++k) {
    //     fmt::print("\tCentroid {} ", k);
    //     for (uint8_t d = 0; d < dimension; ++d)
    //         fmt::print("{} ", centroids[dimension * k + d]);
    //     fmt::println("");
    // }

    for (uint32_t iteration = 0; iteration < max_iterations; ++iteration) {
        // Classify points

        // reset local classes
        for (uint32_t k = 0; k < num_classes; ++k) {
            local_num_classifications_per_class[k] = 0;
            for (uint8_t d = 0; d < dimension; ++d)
                local_new_centroids[dimension * k + d] = 0;
        }

        for (uint32_t i = 0; i < num_local_points; ++i) {
            uint32_t closest_centroid = 0;
            float closest_distance = vec_distance(&local_points[dimension * i], &centroids[dimension * closest_centroid], dimension);

            for (uint32_t j = 1; j < num_classes; ++j) {
                float next_distance = vec_distance(&local_points[dimension * i], &centroids[dimension * j], dimension);
                if (next_distance < closest_distance) {
                    closest_distance = next_distance;
                    closest_centroid = j;
                }
            }

            ++local_num_classifications_per_class[closest_centroid];
            local_classifications[i] = closest_centroid;
        }

        // Update centroids
        for (uint32_t p = 0; p < num_local_points; ++p) {
            uint32_t point_class = local_classifications[p];
            for (uint8_t d = 0; d < dimension; ++d) {
                local_new_centroids[dimension * point_class + d] += local_points[dimension * p + d];
            }
        }

        MPI_Reduce(local_new_centroids, new_centroids, num_classes * dimension, MPI_FLOAT, MPI_SUM, 0, comm);
        MPI_Reduce(local_num_classifications_per_class, num_classifications_per_class, num_classes, MPI_INT, MPI_SUM, 0, comm);

        if (rank == 0) {
            for (uint32_t k = 0; k < num_classes; ++k)
                for (uint8_t d = 0; d < dimension; ++d)
                    new_centroids[dimension * k + d] /= num_classifications_per_class[k];
        }

        MPI_Bcast(new_centroids, num_classes, nvec_row_t, 0, comm);

        bool all_centroids_converged = true;
        for (uint32_t k = 0; k < num_classes; ++k)
            all_centroids_converged = all_centroids_converged && vec_distance(&new_centroids[dimension * k], &centroids[dimension * k], dimension) < 1e-3;

        if (all_centroids_converged)
            break;

        for (uint32_t i = 0; i < num_classes * dimension; ++i)
            centroids[i] = new_centroids[i];

    }

    // Last Classification
    for (uint32_t i = 0; i < num_local_points; ++i) {
        uint32_t closest_centroid = 0;
        float closest_distance = vec_distance(&local_points[dimension * i], &centroids[dimension * closest_centroid], dimension);

        for (uint32_t j = 1; j < num_classes; ++j) {
            float next_distance = vec_distance(&local_points[dimension * i], &centroids[dimension * j], dimension);
            if (next_distance < closest_distance) {
                closest_distance = next_distance;
                closest_centroid = j;
            }
        }

        ++local_num_classifications_per_class[closest_centroid];
        local_classifications[i] = closest_centroid;
    }

    uint32_t *classifications = new uint32_t[num_points];
    MPI_Gatherv(local_classifications, num_local_points, MPI_INT, classifications, process_point_distribution, process_point_displacements, MPI_INT, 0, comm);

    std::vector<uint32_t> results;
    if (rank == 0) 
        results = std::vector<uint32_t>(classifications, classifications + num_points);

    delete[] process_point_distribution, process_point_displacements;
    delete[] local_classifications;
    delete[] local_num_classifications_per_class, num_classifications_per_class;
    delete[] local_new_centroids, new_centroids;

    MPI_Type_free(&nvec_row_t);

    return results;
}