#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>

#include "support.hpp"
#include "serial/kmeans.hpp"

void classify_points(
    const uint32_t num_points, const uint32_t num_classes, const uint8_t dimension, 
    const float *points, float *centroids, uint32_t *classes
) {
    uint32_t closest_class;
    float closest_distance, next_distance;
    for (uint32_t p = 0; p < num_points; ++p) {
        closest_class = 0;
        closest_distance = nvec_distance(&points[dimension * p], &centroids[dimension * closest_class], dimension);

        for (uint32_t k = 0; k < num_classes; ++k) {
            next_distance = nvec_distance(&points[dimension * p], &centroids[dimension * k], dimension);
            if (next_distance < closest_distance) {
                closest_distance = next_distance;
                closest_class = k;
            }
        }

        classes[p] = closest_class;
    }
}

/**
 *  Unsupervised Machine Learning
 *      Classify each point to a centroid, and then update the centroids to be the average of all its classifed points
 * 
 *  Input:
 *      num_points      -> The number of points in the dataset
 *      num_classes     -> The number of classes
 *      dimension       -> The dimension of each datapoint i.e. number of features
 *      points          -> The points in the data
 *      centroids       -> The previouisly initialized centroids
 *      classes         -> The unintialized but declared array of the class for each point
 *      max_iterations  -> The total number of times to execute the centroid updates
 *      timer           -> The time breakdown of each section of work
 * 
 *  Output:
 *      centroids   -> The final centroids
 *      classes     -> The final classes of each point
*/
void kmeans(
    const uint32_t num_points, const uint32_t num_classes, const uint8_t dimension, 
    const float *points, float *centroids, uint32_t *classes, 
    const uint32_t max_iterations, TimeBreakdown *timer
) {
    uint32_t *class_counts = new uint32_t[num_classes];
    float *new_centroids = new float[dimension * num_classes];

    timer->cumulative_update_time_ms = 0.;
    timer->cumulative_classify_time_ms = 0.;
    for (uint32_t iteration = 0; iteration < max_iterations; ++iteration) {

        // Classify the points
        auto classify_start = std::chrono::high_resolution_clock::now();
        classify_points(
            num_points, num_classes, dimension,
            points, centroids, classes
        );
        auto classify_end = std::chrono::high_resolution_clock::now();
        timer->cumulative_classify_time_ms += std::chrono::duration<float, std::milli>(classify_end - classify_start).count();

        // Update the centroids
        auto update_start = std::chrono::high_resolution_clock::now();
        std::memset(new_centroids, 0, dimension * num_classes * sizeof(float));
        std::memset(class_counts, 0, num_classes * sizeof(uint32_t));
        for (uint32_t p = 0; p < num_points; ++p) {
            ++class_counts[classes[p]];
            for (uint8_t d = 0; d < dimension; ++d)
                new_centroids[dimension * classes[p] + d] += points[dimension * p + d];
        }

        for (uint32_t k = 0; k < num_classes; ++k)
            for (uint8_t d = 0; d < dimension; ++d)
                new_centroids[dimension * k + d] /= class_counts[k];
        auto update_end = std::chrono::high_resolution_clock::now();
        timer->cumulative_update_time_ms += std::chrono::duration<float, std::milli>(update_end - update_start).count();

        bool converged = true;
        for (uint32_t k = 0; k < num_classes; ++k)
            converged = converged && nvec_distance(&centroids[dimension * k], &new_centroids[dimension * k], dimension) < 1e-3;

        if (converged)
            break;

        memcpy(centroids, new_centroids, num_classes * dimension * sizeof(float));
    }


    auto last_classify_start = std::chrono::high_resolution_clock::now();
    classify_points(
        num_points, num_classes, dimension,
        points, centroids, classes
    );
    auto last_classify_end = std::chrono::high_resolution_clock::now();
    timer->final_classify_time_ms = std::chrono::duration<float, std::milli>(last_classify_end - last_classify_start).count();


    delete[] new_centroids;
    delete[] class_counts;
}
