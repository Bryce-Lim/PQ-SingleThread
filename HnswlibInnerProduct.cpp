#include "HnswlibInnerProduct.h"
#include <iostream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <immintrin.h>

// Constructor
HnswlibInnerProduct::HnswlibInnerProduct() {
    reset_timers();
}

// Convert bfloat16 to float32
float HnswlibInnerProduct::bfloat16_to_float(bfloat16_t bf16) {
    uint32_t f32_bits = static_cast<uint32_t>(bf16) << 16;
    float result;
    std::memcpy(&result, &f32_bits, sizeof(float));
    return result;
}

// Reset timing counters
void HnswlibInnerProduct::reset_timers() {
    total_compute_time = std::chrono::duration<double>::zero();
    conversion_time = std::chrono::duration<double>::zero();
    actual_compute_time = std::chrono::duration<double>::zero();
}

// Timing getter methods
double HnswlibInnerProduct::get_total_compute_time_ms() const { 
    return total_compute_time.count() * 1000.0; 
}

double HnswlibInnerProduct::get_conversion_time_ms() const { 
    return conversion_time.count() * 1000.0; 
}

double HnswlibInnerProduct::get_actual_compute_time_ms() const { 
    return actual_compute_time.count() * 1000.0; 
}

// Print timing statistics
void HnswlibInnerProduct::print_timing_stats() const {
    std::cout << "\n=== Hnswlib Inner Product Timing Statistics ===\n";
    std::cout << std::fixed << std::setprecision(3);

    std::cout << " Total compute time:        " << std::setw(8) << get_total_compute_time_ms() << " ms\n";
    std::cout << " - Conversion time:         " << std::setw(8) << get_conversion_time_ms() << " ms\n";
    std::cout << " - Actual compute time:     " << std::setw(8) << get_actual_compute_time_ms() << " ms\n";

    std::cout << "================================================\n\n";
}

// Optimized AVX512 inner product computation
static float avx512_inner_product(const float* a, const float* b, size_t dim) {
    __m512 sum = _mm512_setzero_ps();
    size_t i = 0;
    
    // Process 16 elements at a time
    for (; i + 15 < dim; i += 16) {
        __m512 va = _mm512_loadu_ps(&a[i]);
        __m512 vb = _mm512_loadu_ps(&b[i]);
        sum = _mm512_fmadd_ps(va, vb, sum);
    }
    
    // Horizontal sum of the 16 elements in sum
    float result = _mm512_reduce_add_ps(sum);
    
    // Handle remaining elements
    for (; i < dim; ++i) {
        result += a[i] * b[i];
    }
    
    return result;
}

// Fallback AVX2 inner product computation for systems without AVX512
static float avx2_inner_product(const float* a, const float* b, size_t dim) {
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;
    
    // Process 8 elements at a time
    for (; i + 7 < dim; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        sum = _mm256_fmadd_ps(va, vb, sum);
    }
    
    // Horizontal sum of the 8 elements in sum
    __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum, 0), _mm256_extractf128_ps(sum, 1));
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float result = _mm_cvtss_f32(sum128);
    
    // Handle remaining elements
    for (; i < dim; ++i) {
        result += a[i] * b[i];
    }
    
    return result;
}

// Main compute function
std::vector<std::vector<float>> HnswlibInnerProduct::compute(
    const std::vector<std::vector<bfloat16_t>>& centroids,
    const std::vector<std::vector<bfloat16_t>>& data
) {
    auto start_total = std::chrono::high_resolution_clock::now();
    
    // Input validation
    if (centroids.empty() || data.empty()) {
        throw std::invalid_argument("Input vectors cannot be empty");
    }
    
    size_t num_centroids = centroids.size();
    size_t num_data = data.size();
    size_t dim = centroids[0].size();
    
    auto start_conversion = std::chrono::high_resolution_clock::now();
    
    // Convert centroids to float32 (only once)
    std::vector<std::vector<float>> centroids_float(num_centroids, std::vector<float>(dim));
    for (size_t i = 0; i < num_centroids; ++i) {
        for (size_t k = 0; k < dim; ++k) {
            centroids_float[i][k] = bfloat16_to_float(centroids[i][k]);
        }
    }
    
    // Convert data to float32 (only once)
    std::vector<std::vector<float>> data_float(num_data, std::vector<float>(dim));
    for (size_t j = 0; j < num_data; ++j) {
        for (size_t k = 0; k < dim; ++k) {
            data_float[j][k] = bfloat16_to_float(data[j][k]);
        }
    }
    
    auto end_conversion = std::chrono::high_resolution_clock::now();
    conversion_time += end_conversion - start_conversion;
    
    // Initialize output matrix
    std::vector<std::vector<float>> output(num_centroids, std::vector<float>(num_data));
    
    auto start_compute = std::chrono::high_resolution_clock::now();
    
    // Check if AVX512 is available
    bool has_avx512 = __builtin_cpu_supports("avx512f");
    
    // Compute inner products using optimized SIMD
    for (size_t i = 0; i < num_centroids; ++i) {
        for (size_t j = 0; j < num_data; ++j) {
            if (has_avx512) {
                output[i][j] = avx512_inner_product(centroids_float[i].data(), data_float[j].data(), dim);
            } else {
                output[i][j] = avx2_inner_product(centroids_float[i].data(), data_float[j].data(), dim);
            }
        }
    }
    
    auto end_compute = std::chrono::high_resolution_clock::now();
    actual_compute_time += end_compute - start_compute;
    
    auto end_total = std::chrono::high_resolution_clock::now();
    total_compute_time += end_total - start_total;
    
    return output;
}

// Print matrix function (reusing from ScalarInnerProduct)
void HnswlibInnerProduct::printMatrix(const std::vector<std::vector<float>>& matrix) {
    if (matrix.empty()) {
        std::cout << "Matrix is empty." << std::endl;
        return;
    }
    
    const size_t rows = matrix.size();
    const size_t cols = matrix[0].size();
    
    // Print header with matrix dimensions
    std::cout << "\n╔══════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║                  Hnswlib Inner Product Matrix            ║" << std::endl;
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
