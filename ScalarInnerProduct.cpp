// ScalarInnerProduct.cpp
#include "ScalarInnerProduct.h"
#include <iomanip>
#include <limits>
#include <string>
#include <algorithm>
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
    if (matrix.empty()) {
        std::cout << "Matrix is empty." << std::endl;
        return;
    }
    
    const size_t rows = matrix.size();
    const size_t cols = matrix[0].size();
    
    // Print header with matrix dimensions
    std::cout << "\n╔══════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║                    Inner Product Matrix                  ║" << std::endl;
    std::cout << "║                 " << rows << " x " << cols << " (Centroids × Data)               ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════╝" << std::endl;
    
    // Set precision to 3 decimal places
    std::cout << std::fixed << std::setprecision(3);
    
    // Print column headers (data vector indices)
    std::cout << "   ";  // Space for row labels
    for (size_t j = 0; j < cols; ++j) {
        std::cout << std::setw(10) << ("D" + std::to_string(j));
    }
    std::cout << std::endl;
    
    // Print separator line
    std::cout << "    ┌──";
    for (size_t j = 0; j < cols; ++j) {
        std::cout << "──────────";
    }
    std::cout << "┐" << std::endl;
    
    // Print matrix rows with row labels (centroid indices)
    for (size_t i = 0; i < rows; ++i) {
        std::cout << "C" << std::setw(2) << i << " │ ";
        
        for (size_t j = 0; j < cols; ++j) {
            // Color coding for visualization (optional)
            float val = matrix[i][j];
            if (val > 0.800) {
                std::cout << "\033[1;32m";  // Green for high similarity
            } else if (val > 0.500) {
                std::cout << "\033[1;33m";  // Yellow for medium similarity
            } else if (val < 0.200) {
                std::cout << "\033[1;31m";  // Red for low similarity
            }
            
            std::cout << std::setw(9) << val << " ";
            std::cout << "\033[0m";  // Reset color
        }
        std::cout << " │" << std::endl;
    }
    
    // Print bottom border
    std::cout << "    └──";
    for (size_t j = 0; j < cols; ++j) {
        std::cout << "──────────";
    }
    std::cout << "┘" << std::endl;
    
    // Print summary statistics
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();
    float sum = 0.0f;
    size_t total_elements = 0;
    
    for (const auto& row : matrix) {
        for (float val : row) {
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
            sum += val;
            total_elements++;
        }
    }
    
    float avg_val = sum / total_elements;
    
    std::cout << "\n📊 Matrix Statistics:" << std::endl;
    std::cout << "   • Minimum value: " << std::setw(8) << min_val << std::endl;
    std::cout << "   • Maximum value: " << std::setw(8) << max_val << std::endl;
    std::cout << "   • Average value: " << std::setw(8) << avg_val << std::endl;
    std::cout << "   • Range:         " << std::setw(8) << (max_val - min_val) << std::endl;
    
    // Legend for color coding
    std::cout << "\n🎨 Color Legend:" << std::endl;
    std::cout << "   \033[1;32m■\033[0m Green:  High similarity (> 0.800)" << std::endl;
    std::cout << "   \033[1;33m■\033[0m Yellow: Medium similarity (0.500 - 0.800)" << std::endl;
    std::cout << "   \033[1;31m■\033[0m Red:    Low similarity (< 0.200)" << std::endl;
    std::cout << "   ■ White:  Normal similarity" << std::endl;
    
    std::cout << std::endl;
}
