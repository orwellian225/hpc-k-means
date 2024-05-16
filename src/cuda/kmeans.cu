#include <vector>
#include <random>

#include <fmt/core.h>

#include "cuda/kmeans.cuh"
#include "nvector.hpp"

void handle_cuda_error(cudaError_t error) {
    if (error == cudaSuccess)
        return;
    
    fmt::println(stderr, "CUDA Error:");
    fmt::println(stderr, "\t{}", cudaGetErrorString(error));
    cudaDeviceReset();
    exit(EXIT_FAILURE);
}

std::vector<NVector> kmeansplusplus_centroids(uint32_t num_centroids, uint8_t num_dimensions, const std::vector<NVector> &points) { 
    std::mt19937 rng(0);
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

float h_vec_distance(float *vec_1, float *vec_2, uint8_t dimension) {
    float sum = 0;
    for (uint8_t d = 0; d < dimension; ++d)
        sum += (vec_1[d] - vec_2[d]) * (vec_1[d] - vec_2[d]);

    return std::sqrt(sum);
}

__constant__ uint32_t d_num_points;
__constant__ uint32_t d_num_classes;
__constant__ uint8_t d_dimension;

__device__ float d_vec_distance(float *vec_1, float *vec_2, uint8_t dimension) {
    float sum = 0;
    for (uint8_t d = 0; d < dimension; ++d)
        sum += (vec_1[d] - vec_2[d]) * (vec_1[d] - vec_2[d]);

    return sqrt(sum);
}

__global__ void classify(cudaTextureObject_t points, float *centroids, uint32_t *classifications) {
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (point_idx >= d_num_points) {
        return;
    }

    // Bring centroids to shared memory
    extern __shared__ float shared_centroids[];
    if (threadIdx.x < d_num_classes * d_dimension) {
        shared_centroids[threadIdx.x] = centroids[threadIdx.x];
    }

    __syncthreads();

    float *point = new float[d_dimension];
    for (uint8_t d = 0; d < d_dimension; ++d)
        point[d] = tex1Dfetch<float>(points, d_dimension * point_idx + d);

    uint32_t closest_centroid = 0;
    float closest_distance = d_vec_distance(point, &shared_centroids[d_dimension * closest_centroid], d_dimension);
    float next_distance;

    for (uint32_t k = 0; k < d_num_classes; ++k) {
        next_distance = d_vec_distance(point, &shared_centroids[d_dimension * k], d_dimension);
        if (next_distance < closest_distance) {
            closest_distance = next_distance;
            closest_centroid = k;
        }
    }

    classifications[point_idx] = closest_centroid; 
    delete[] point;
}

std::vector<uint32_t> classify_kmeans(
    const uint8_t dimension, const uint32_t num_points, const uint32_t num_classes,
    const float *points,
    float *centroids,
    uint32_t max_iterations
) {

    // Send points to texture memory
    float *d_points;
    handle_cuda_error(cudaMalloc(&d_points, num_points * dimension * sizeof(float)));
    handle_cuda_error(cudaMemcpy(d_points, points, num_points * dimension * sizeof(float), cudaMemcpyHostToDevice));
    
    cudaResourceDesc points_res_desc;
    memset(&points_res_desc, 0, sizeof(points_res_desc));
    points_res_desc.resType = cudaResourceTypeLinear;
    points_res_desc.res.linear.devPtr = d_points;
    points_res_desc.res.linear.desc.f = cudaChannelFormatKindFloat;
    points_res_desc.res.linear.sizeInBytes = num_points * dimension * sizeof(float);
    points_res_desc.res.linear.desc.x = 32;

    cudaTextureDesc points_tex_desc;
    memset(&points_tex_desc, 0, sizeof(points_tex_desc));
    points_tex_desc.readMode = cudaReadModeElementType;

    cudaTextureObject_t points_tex;
    handle_cuda_error(cudaCreateTextureObject(&points_tex, &points_res_desc, &points_tex_desc, nullptr));

    const size_t block_size = 1024; 
    const size_t grid_size = num_points / block_size + 1;

    // handle_cuda_error(cudaMalloc(&d_num_points, sizeof(uint32_t)));
    // handle_cuda_error(cudaMalloc(&d_num_classes, sizeof(uint32_t)));
    // handle_cuda_error(cudaMalloc(&d_dimension, sizeof(uint8_t)));

    handle_cuda_error(cudaMemcpyToSymbol(d_num_points, &num_points, sizeof(num_points)));
    handle_cuda_error(cudaMemcpyToSymbol(d_num_classes, &num_classes, sizeof(num_classes)));
    handle_cuda_error(cudaMemcpyToSymbol(d_dimension, &dimension, sizeof(dimension)));

    uint32_t *classifications = new uint32_t[num_points];
    uint32_t *d_classifications;
    handle_cuda_error(cudaMalloc(&d_classifications, num_points * sizeof(uint32_t)));

    float *d_centroids;
    handle_cuda_error(cudaMalloc(&d_centroids, num_classes * dimension * sizeof(float)));

    float *new_centroids = new float[dimension * num_classes];
    uint32_t *d_class_counts, *class_counts = new uint32_t[num_classes];
    handle_cuda_error(cudaMalloc(&d_class_counts, num_points * sizeof(uint32_t)));
    // In loop
    for (uint32_t iteration = 0; iteration < max_iterations; ++iteration) {
        // send centroids to GPU
        handle_cuda_error(cudaMemcpy(d_centroids, centroids, num_classes * dimension * sizeof(float), cudaMemcpyHostToDevice));
        classify<<<grid_size, block_size, num_classes * dimension * sizeof(float)>>>(points_tex, d_centroids, d_classifications);
        cudaDeviceSynchronize();
        handle_cuda_error(cudaMemcpy(classifications, d_classifications, num_points * sizeof(uint32_t), cudaMemcpyDeviceToHost));

        // Manual reduction
        memset(new_centroids, 0, num_classes * dimension * sizeof(float));
        memset(class_counts, 0, num_classes * sizeof(uint32_t));
        for (uint32_t p = 0; p < num_points; ++p) {
            for (uint8_t d = 0; d < dimension; ++d)
                new_centroids[dimension * classifications[p] + d] += points[dimension * p + d];
            class_counts[classifications[p]] += 1;
        }

        for (uint32_t k = 0; k < num_classes ; ++k)
            for (uint8_t d = 0; d < dimension; ++d)
                new_centroids[dimension * k + d] /= class_counts[k];

        // Check convergence
        bool all_centroids_converged = true;
        for (uint32_t k = 0; k < num_classes; ++k) {
            all_centroids_converged = all_centroids_converged && h_vec_distance(&new_centroids[dimension * k], &centroids[dimension * k], dimension) < 1e-3;
        }

        if (all_centroids_converged)
            break;

        // update centroids
        for (uint32_t i = 0; i < num_classes * dimension; ++i)
            centroids[i] = new_centroids[i];
    }

    // Perform last classification
    handle_cuda_error(cudaMemcpy(d_centroids, centroids, num_classes * dimension * sizeof(float), cudaMemcpyHostToDevice));
    classify<<<grid_size, block_size, num_classes * dimension * sizeof(float)>>>(points_tex, d_centroids, d_classifications);
    cudaDeviceSynchronize();
    handle_cuda_error(cudaMemcpy(classifications, d_classifications, num_points * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    handle_cuda_error(cudaDestroyTextureObject(points_tex));
    handle_cuda_error(cudaFree(d_points));

    handle_cuda_error(cudaFree(d_class_counts));
    handle_cuda_error(cudaFree(d_classifications));

    delete[] new_centroids, class_counts;

    // return last classification
    std::vector<uint32_t> results;
    results = std::vector<uint32_t>(classifications, classifications + num_points);
    return results;
}