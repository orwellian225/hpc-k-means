#include <vector>
#include <random>

#include "cuda/kmeans.cuh"
#include "nvector.hpp"

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
    int point_idx = blockIdx.x * blockDim.x * threadIdx.x;

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
    float closest_distance = d_vec_distance(point, &centroids[0], d_dimension);
    float next_distance;

    for (uint32_t k = 0; k < d_num_classes; ++k) {
        next_distance = d_vec_distance(point, &centroids[k], d_dimension);
        if (next_distance < closest_distance) {
            closest_centroid = k;
        }
    }

    classifications[point_idx] = closest_centroid; 
}

__global__ void update_centroids(cudaTextureObject_t points, float *centroids, uint32_t *class_counts, uint32_t *classifications) {
    int point_idx = blockIdx.x * blockDim.x * threadIdx.x;

    if (point_idx >= d_num_points) {
        return;
    }

    extern __shared__ uint32_t shared_classifications[];
    shared_classifications[threadIdx.x] = classifications[point_idx];

    __syncthreads();

    if (threadIdx.x == 0) {
        float *local_centroid_sums = new float[d_dimension * d_num_classes];
        uint32_t *local_class_counts = new uint32_t[d_num_classes];

        for (uint32_t j = 0; j < blockDim.x; ++j) {
            for (uint8_t d = 0; d < d_dimension; ++d)
                local_centroid_sums[d_dimension * shared_classifications[j] + d] += tex1Dfetch<float>(points, d_dimension * j + d);

            local_class_counts[shared_classifications[j]] += 1;
        }

        for (uint32_t k = 0; k < d_num_classes; ++k) {
            for (uint8_t d = 0; d < d_dimension; ++d)
                atomicAdd(&centroids[d_dimension * k + d], local_centroid_sums[d_dimension * k + d]);
            atomicAdd(&class_counts[k], local_class_counts[k]);
        }
    }

    __syncthreads();

    if (point_idx < d_num_classes) {
        for (uint8_t d = 0; d < d_dimension; ++d)
            centroids[d_dimension * point_idx + d] = centroids[d_dimension * point_idx + d] / class_counts[point_idx];
    }
}

std::vector<uint32_t> classify_kmeans(
    const uint8_t dimension, const uint32_t num_points, const uint32_t num_classes,
    const float *points,
    float *centroids,
    uint32_t max_iterations
) {

    // Send points to texture memory
    float *d_points;
    cudaMalloc(&d_points, num_points * dimension * sizeof(float));
    
    cudaResourceDesc points_res_desc;
    memset(&points_res_desc, 0, sizeof(points_res_desc));
    points_res_desc.resType = cudaResourceTypeLinear;
    points_res_desc.res.linear.devPtr = d_points;
    points_res_desc.res.linear.desc.f = cudaChannelFormatKindFloat;
    points_res_desc.res.linear.sizeInBytes = num_points * dimension * sizeof(float);

    cudaTextureDesc points_tex_desc;
    memset(&points_tex_desc, 0, sizeof(points_tex_desc));
    points_tex_desc.readMode = cudaReadModeElementType;

    cudaTextureObject_t points_tex;
    cudaCreateTextureObject(&points_tex, &points_res_desc, &points_tex_desc, nullptr);

    const size_t block_size = 1024; 
    const size_t grid_size = num_points / block_size + 1;

    uint32_t *d_num_points, *d_num_classes;
    uint8_t *d_dimension;
    cudaMalloc(&d_num_points, sizeof(uint32_t));
    cudaMalloc(&d_num_classes, sizeof(uint32_t));
    cudaMalloc(&d_dimension, sizeof(uint8_t));

    cudaMemcpyToSymbol(d_num_points, &num_points, sizeof(uint32_t), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_num_classes, &num_classes, sizeof(uint32_t), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_dimension, &dimension, sizeof(uint8_t), 0, cudaMemcpyHostToDevice);

    uint32_t *classifications = new uint32_t[num_points];
    uint32_t *d_classifications;
    cudaMalloc(&d_classifications, num_points * sizeof(uint32_t));

    float *d_centroids;
    cudaMalloc(&d_centroids, num_classes * dimension * sizeof(float));

    float *new_centroids = new float[dimension * num_classes];
    uint32_t *d_class_counts;
    cudaMalloc(&d_class_counts, num_points * sizeof(uint32_t));
    // In loop
    for (uint32_t iteration = 0; iteration < max_iterations; ++iteration) {
        // send centroids to GPU
        cudaMemcpy(&d_centroids, &centroids, num_classes * dimension * sizeof(float), cudaMemcpyHostToDevice);

        // Perform classification
        classify<<<grid_size, block_size, num_classes * dimension * sizeof(float)>>>(points_tex, d_centroids, d_classifications);
        // cudaMemcpy(classifications, d_classifications, num_points * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        // update centroids with gpu reduction
        // reset d_centroids and d_class_counts so they can be used to sum the new values
        cudaMemset(&d_centroids, 0, dimension * num_classes * sizeof(float));
        cudaMemset(&d_class_counts, 0, num_classes * sizeof(uint32_t));
        update_centroids<<<grid_size, block_size, block_size * sizeof(uint32_t)>>>(points_tex, d_centroids, d_class_counts, d_classifications);

        cudaMemcpy(&new_centroids, &d_centroids, dimension * num_classes * sizeof(float), cudaMemcpyDeviceToHost);
        
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
    cudaMemcpy(&d_centroids, &centroids, num_classes * dimension * sizeof(float), cudaMemcpyHostToDevice);
    classify<<<grid_size, block_size, num_classes * dimension * sizeof(float)>>>(points_tex, d_centroids, d_classifications);
    cudaMemcpy(classifications, d_classifications, num_points * sizeof(uint32_t), cudaMemcpyDeviceToHost);


    cudaDestroyTextureObject(points_tex);
    cudaFree(&d_points);

    // return last classification
    std::vector<uint32_t> results;
    results = std::vector<uint32_t>(classifications, classifications + num_points);
    return results;
}