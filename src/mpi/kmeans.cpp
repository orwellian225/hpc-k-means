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
#include "support.hpp"

void classify_kmeans(
    const uint8_t dimension, const uint32_t num_points, const uint32_t num_classes,
    const float *points, float *centroids, uint32_t *classes,
    uint32_t max_iterations, TimeBreakdown *timer,
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

    timer->cumulative_update_time_ms = 0.;
    timer->cumulative_classify_time_ms = 0.;

    float *local_points = new float[dimension * num_local_points];
    MPI_Scatterv(points, process_point_distribution, process_point_displacements, nvec_row_t, local_points, num_local_points, nvec_row_t, 0, comm);

    uint32_t *local_classifications = new uint32_t[num_local_points];
    uint32_t *local_num_classifications_per_class = new uint32_t[num_classes];
    uint32_t *num_classifications_per_class = new uint32_t[num_classes];

    float *local_new_centroids = new float[num_classes * dimension];
    float *new_centroids = new float[num_classes * dimension];

    for (uint32_t iteration = 0; iteration < max_iterations; ++iteration) {
        // Classify points
        double classify_start = MPI_Wtime();

        // reset local classes
        for (uint32_t k = 0; k < num_classes; ++k) {
            local_num_classifications_per_class[k] = 0;
            for (uint8_t d = 0; d < dimension; ++d)
                local_new_centroids[dimension * k + d] = 0;
        }

        for (uint32_t i = 0; i < num_local_points; ++i) {
            uint32_t closest_centroid = 0;
            float closest_distance = nvec_distance(&local_points[dimension * i], &centroids[dimension * closest_centroid], dimension);

            for (uint32_t k = 0; k < num_classes; ++k) {
                float next_distance = nvec_distance(&local_points[dimension * i], &centroids[dimension * k], dimension);
                if (next_distance < closest_distance) {
                    closest_distance = next_distance;
                    closest_centroid = k;
                }
            }

            ++local_num_classifications_per_class[closest_centroid];
            local_classifications[i] = closest_centroid;
        }
        double classify_end = MPI_Wtime();
        timer->cumulative_classify_time_ms += (classify_end - classify_start) * 1000;

        // Update centroids
        double update_start = MPI_Wtime();
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
        double update_end = MPI_Wtime();
        timer->cumulative_update_time_ms += (update_end - update_start) * 1000;

        bool all_centroids_converged = true;
        for (uint32_t k = 0; k < num_classes; ++k) {
            all_centroids_converged = all_centroids_converged && nvec_distance(&new_centroids[dimension * k], &centroids[dimension * k], dimension) < 1e-3;
        }

        if (all_centroids_converged)
            break;

        for (uint32_t i = 0; i < num_classes * dimension; ++i)
            centroids[i] = new_centroids[i];

    }

    // Last Classification
    double last_classify_start = MPI_Wtime();
    for (uint32_t i = 0; i < num_local_points; ++i) {
        uint32_t closest_centroid = 0;
        float closest_distance = nvec_distance(&local_points[dimension * i], &centroids[dimension * closest_centroid], dimension);

        for (uint32_t j = 1; j < num_classes; ++j) {
            float next_distance = nvec_distance(&local_points[dimension * i], &centroids[dimension * j], dimension);
            if (next_distance < closest_distance) {
                closest_distance = next_distance;
                closest_centroid = j;
            }
        }

        ++local_num_classifications_per_class[closest_centroid];
        local_classifications[i] = closest_centroid;
    }

    MPI_Gatherv(local_classifications, num_local_points, MPI_INT, classes, process_point_distribution, process_point_displacements, MPI_INT, 0, comm);
    double last_classify_end = MPI_Wtime();
    timer->final_classify_time_ms = (last_classify_end - last_classify_start) * 1000;

    delete[] process_point_distribution;
    delete[] process_point_displacements;
    delete[] local_classifications;
    delete[] local_num_classifications_per_class;
    delete[] num_classifications_per_class;
    delete[] local_new_centroids;
    delete[] new_centroids;

    MPI_Type_free(&nvec_row_t);
}
