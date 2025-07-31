// ScalarInnerProduct.cpp
#include "ScalarInnerProduct.h"
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <cstdint>

// Convert bfloat16 to float32
float ScalarInnerProduct::bfloat16_to_float(bfloat16_t bf16) {
    // bfloat16 is the upper 16 bits of a float32
    // We need to shift it to the upper half and zero the lower half
    uint32_t f32_bits = static_cast<uint32_t>(bf16) << 16;
    float result;
    std::memcpy(&result, &f32_bits, sizeof(float));
    return result;
}

std::vector<std::vector<float>> ScalarInnerProduct::compute(
    const std::vector<std::vector<bfloat16_t>>& centroids,
    const std::vector<std::vector<bfloat16_t>>& data
) {
    // Input validation
    if (centroids.empty() || data.empty()) {
        throw std::invalid_argument("Input vectors cannot be empty");
    }
    
    // Check that all centroids have the same dimension
    size_t centroid_dim = centroids[0].size();
    for (const auto& centroid : centroids) {
        if (centroid.size() != centroid_dim) {
            throw std::invalid_argument("All centroids must have the same dimension");
        }
    }
    
    // Check that all data vectors have the same dimension
    size_t data_dim = data[0].size();
    for (const auto& vec : data) {
        if (vec.size() != data_dim) {
            throw std::invalid_argument("All data vectors must have the same dimension");
        }
    }
    
    // Check dimension compatibility
    if (centroid_dim != data_dim) {
        throw std::invalid_argument("Centroids and data vectors must have the same dimension");
    }
    
    // Initialize output matrix
    size_t num_centroids = centroids.size();
    size_t num_data = data.size();
    std::vector<std::vector<float>> output(num_centroids, std::vector<float>(num_data, 0.0f));
    
    // Compute inner products
    for (size_t i = 0; i < num_centroids; ++i) {
        for (size_t j = 0; j < num_data; ++j) {
            float inner_product = 0.0f;
            for (size_t k = 0; k < centroid_dim; ++k) {
                // Convert bfloat16 values to float32 before multiplication
                float centroid_val = bfloat16_to_float(centroids[i][k]);
                float data_val = bfloat16_to_float(data[j][k]);
                inner_product += centroid_val * data_val;
            }
            output[i][j] = inner_product;
        }
    }
    
    return output;
}

void ScalarInnerProduct::printMatrix(const std::vector<std::vector<float>>& matrix) {
    for (const auto& row : matrix) {
        for (float val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}
