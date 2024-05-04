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

std::vector<uint32_t> classify_kmeans(
    const std::vector<NVector>& points, 
    std::vector<NVector>& centroids, 
    uint32_t max_iterations,
    const int32_t comm_rank, const int32_t comm_size, const MPI_Comm comm
) {
    const uint8_t num_dimensions = points[0].num_dimensions;
    const uint32_t num_classes = centroids.size();

    uint32_t points_span = points.size() / comm_size;
    uint32_t points_spare = points.size() % comm_size;
    uint32_t points_start_idx = comm_rank * points_span + comm_rank * (comm_rank - 1 < points_spare);
    uint32_t points_end_idx = points_start_idx + points_span + (comm_rank < points_spare) - 1;
    uint32_t rank_points_span = points_end_idx - points_start_idx + 1;
    /* Example of above
    *   points.size = 7, comm_size = 4, points_span = 7 / 4 = 1, points_spare = 7 % 4 = 3
    *   rank = 0 start_idx = 0 * 1 + 0 * (0 - 1 < 3) = 0
    *            end_idx = 0 + 
    *   rank = 1 start_idx = 
    *            end_idx = 
    *   rank = 2 start_idx = 
    *            end_idx = 
    *   rank = 3 start_idx = 
    *            end_idx = 
    */

    uint32_t current_iteration = 0;

    uint32_t *local_classifications = new uint32_t[rank_points_span];
    uint32_t *local_class_count = new uint32_t[num_classes];
    float **local_dimensional_sums = new float*[num_dimensions];
    for (uint8_t d = 0; d < num_dimensions; ++d)
        local_dimensional_sums[d] = new float[num_classes];

    uint32_t *global_classifications;
    uint32_t *global_class_count = new uint32_t[num_classes];
    float **global_dimensional_sums = new float*[num_dimensions];
    for (uint8_t d = 0; d < num_dimensions; ++d)
        global_dimensional_sums[d] = new float[num_classes];
    if (comm_rank == 0)
        global_classifications = new uint32_t[points.size()];

    float *merged_dimension_data = new float[num_classes * num_dimensions];

    while (true) {

        for (uint32_t k = 0; k < num_classes; ++k) {
            local_class_count[k] = 0;
            for (uint8_t d = 0; d < num_dimensions; ++d)
                local_dimensional_sums[d][k] = 0.;
        }

        // Phase 1 - classification
        for (uint32_t p = points_start_idx; p <= points_end_idx; ++p) {
            uint32_t closest_class = 0;
            for (uint32_t k = 0; k < num_classes; ++k) {
                if (points[p].distance_to(centroids[k]) < points[p].distance_to(centroids[closest_class])) {
                    closest_class = k;
                }
            }

            local_classifications[p] = closest_class;
        }

        for (uint32_t p = points_start_idx; p <= points_end_idx; ++p) {
            for (uint8_t d = 0; d < num_dimensions; ++d)
                local_dimensional_sums[d][local_classifications[p]] += points[p][d];
            local_class_count[local_classifications[p]] += 1;
        }

        MPI_Reduce(local_class_count, global_class_count, num_classes, MPI_UINT32_T, MPI_SUM, 0, comm_rank);
        for (uint8_t d = 0; d < num_dimensions; ++d)
            MPI_Reduce(local_dimensional_sums[d], global_dimensional_sums[d], num_classes, MPI_FLOAT, MPI_SUM, 0, comm);
        // Phase 2 - update + convergance || iteration check

        bool converged = true;
        std::vector<NVector> new_centroids;
        if (comm_rank == 0) {
            for (uint32_t k = 0; k < num_classes; ++k) {
                new_centroids.push_back(NVector(num_dimensions, 0.0));
                for (uint8_t d = 0; d < num_dimensions; ++d)
                    new_centroids[k][d] = global_dimensional_sums[d][k] / global_class_count[k];

                converged = converged && centroids[k].distance_to(new_centroids[k]) < 1e-3;
            }
        }

        MPI_Bcast(&converged, 1, MPI_CXX_BOOL, 0, comm);

        if (converged || current_iteration > max_iterations) {
            MPI_Allgather(local_classifications, rank_points_span, MPI_UINT32_T, global_classifications, points.size(), MPI_UINT32_T, comm);

            std::vector<uint32_t> classifications;
            for (uint32_t p = 0; p < points.size(); ++p) {
                classifications.push_back(global_classifications[p]);
            }

            delete[] local_class_count, local_classifications;
            delete[] global_class_count, global_classifications;
            for (uint8_t d = 0; d < num_dimensions; ++d) {
                delete[] local_dimensional_sums[d];
                delete[] global_dimensional_sums[d];
            }
            delete[] local_dimensional_sums, global_dimensional_sums;
            delete[] merged_dimension_data;

            return classifications;
        }

        // Broadcast the centroids
        if (comm_rank == 0) {
            std::vector<NVector> generated_centroids = kmeansplusplus_centroids(num_classes, num_dimensions, points);
            // send to all nodes

            for (size_t i = 0; i < num_classes * num_dimensions; i += num_dimensions) {
                for (uint8_t d = 0; d < num_dimensions; ++d)
                    merged_dimension_data[i + d] = generated_centroids[i / num_dimensions].data[d];
            }

        } 

        MPI_Bcast(merged_dimension_data, num_classes * num_dimensions, MPI_FLOAT, 0, MPI_COMM_WORLD);

        for (size_t i = 0; i < num_classes * num_dimensions; i += num_dimensions) {
            centroids.push_back(NVector(num_dimensions, 0.0));
            for (uint8_t d = 0; d < num_dimensions; ++d) {
                centroids[i / num_dimensions][d] = merged_dimension_data[i + d];
            }
        }

        ++current_iteration;
    }
}