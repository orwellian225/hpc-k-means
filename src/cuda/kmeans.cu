#include <chrono>
#include <vector>
#include <random>

#include <fmt/core.h>

#include "cuda/kmeans.cuh"
#include "support.hpp"

void handle_cuda_error(cudaError_t error) {
    if (error == cudaSuccess)
        return;
    
    fmt::println(stderr, "CUDA Error:");
    fmt::println(stderr, "\t{}", cudaGetErrorString(error));
    cudaDeviceReset();
    exit(EXIT_FAILURE);
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

void classify_kmeans(
    const uint8_t dimension, const uint32_t num_points, const uint32_t num_classes,
    const float *points, float *centroids, uint32_t *classes,
    uint32_t max_iterations, TimeBreakdown *timer
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

    cudaEvent_t classify_start;
    cudaEvent_t classify_end;

    cudaEventCreate(&classify_start);
    cudaEventCreate(&classify_end);

    handle_cuda_error(cudaMemcpyToSymbol(d_num_points, &num_points, sizeof(num_points)));
    handle_cuda_error(cudaMemcpyToSymbol(d_num_classes, &num_classes, sizeof(num_classes)));
    handle_cuda_error(cudaMemcpyToSymbol(d_dimension, &dimension, sizeof(dimension)));

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
        cudaEventRecord(classify_start);
        classify<<<grid_size, block_size, num_classes * dimension * sizeof(float)>>>(points_tex, d_centroids, d_classifications);
        cudaEventRecord(classify_end);

        cudaEventSynchronize(classify_end);
        float classify_duration = 0;
        cudaEventElapsedTime(&classify_duration, classify_start, classify_end);
        timer->cumulative_classify_time_ms += classify_duration;

        handle_cuda_error(cudaMemcpy(classes, d_classifications, num_points * sizeof(uint32_t), cudaMemcpyDeviceToHost));

        // Manual reduction
        auto update_start = std::chrono::high_resolution_clock::now();
        memset(new_centroids, 0, num_classes * dimension * sizeof(float));
        memset(class_counts, 0, num_classes * sizeof(uint32_t));
        for (uint32_t p = 0; p < num_points; ++p) {
            for (uint8_t d = 0; d < dimension; ++d)
                new_centroids[dimension * classes[p] + d] += points[dimension * p + d];
            class_counts[classes[p]] += 1;
        }

        for (uint32_t k = 0; k < num_classes ; ++k)
            for (uint8_t d = 0; d < dimension; ++d)
                new_centroids[dimension * k + d] /= class_counts[k];
        auto update_end = std::chrono::high_resolution_clock::now();
        timer->cumulative_update_time_ms += std::chrono::duration<float, std::milli>(update_end - update_start).count();

        // Check convergence
        bool all_centroids_converged = true;
        for (uint32_t k = 0; k < num_classes; ++k) {
            all_centroids_converged = all_centroids_converged && nvec_distance(&new_centroids[dimension * k], &centroids[dimension * k], dimension) < 1e-3;
        }

        if (all_centroids_converged)
            break;

        // update centroids
        for (uint32_t i = 0; i < num_classes * dimension; ++i)
            centroids[i] = new_centroids[i];
    }

    // Perform last classification
    handle_cuda_error(cudaMemcpy(d_centroids, centroids, num_classes * dimension * sizeof(float), cudaMemcpyHostToDevice));
    cudaEventRecord(classify_start);
    classify<<<grid_size, block_size, num_classes * dimension * sizeof(float)>>>(points_tex, d_centroids, d_classifications);
    cudaEventRecord(classify_end);
    handle_cuda_error(cudaMemcpy(classes, d_classifications, num_points * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    cudaEventSynchronize(classify_end);
    float classify_duration = 0;
    cudaEventElapsedTime(&classify_duration, classify_start, classify_end);
    timer->final_classify_time_ms = classify_duration;

    handle_cuda_error(cudaDestroyTextureObject(points_tex));
    handle_cuda_error(cudaFree(d_points));

    handle_cuda_error(cudaFree(d_class_counts));
    handle_cuda_error(cudaFree(d_classifications));

    delete[] new_centroids, class_counts;
}
