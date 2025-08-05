#ifndef HNSWLIB_INNER_PRODUCT_H
#define HNSWLIB_INNER_PRODUCT_H

#include <vector>
#include <cstring>
#include <cstdint>
#include <chrono>
#include "hnswlib/hnswlib.h"

typedef uint16_t bfloat16_t;

class HnswlibInnerProduct {
private:
    // Timing instance variables
    std::chrono::duration<double> total_compute_time;
    std::chrono::duration<double> conversion_time;
    std::chrono::duration<double> actual_compute_time;

    /**
     * Convert bfloat16 to float32
     */
    static float bfloat16_to_float(bfloat16_t bf16);

public:
    // Constructor
    HnswlibInnerProduct();

    /**
     * Computes the inner product matrix between centroids and data vectors using hnswlib's optimized functions.
     * 
     * @param centroids: Vector of centroid vectors (rows) in bfloat16_t
     * @param data: Vector of data vectors (columns) in bfloat16_t
     * @return Matrix where output[row][col] = inner product of centroids[row] and data[col] in float32
     */
    std::vector<std::vector<float>> compute(
        const std::vector<std::vector<bfloat16_t>>& centroids,
        const std::vector<std::vector<bfloat16_t>>& data
    );

    // Timing methods
    void reset_timers();
    void print_timing_stats() const;

    // Individual timing getters (in milliseconds)
    double get_total_compute_time_ms() const;
    double get_conversion_time_ms() const;
    double get_actual_compute_time_ms() const;

    /**
     * Helper function to print a matrix
     */
    void printMatrix(const std::vector<std::vector<float>>& matrix);
};

#endif // HNSWLIB_INNER_PRODUCT_H
