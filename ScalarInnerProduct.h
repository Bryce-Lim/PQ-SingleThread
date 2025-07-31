// ScalarInnerProduct.h
#ifndef SCALAR_INNER_PRODUCT_H
#define SCALAR_INNER_PRODUCT_H

#include <vector>
#include <cstring>
#include <cstdint>

typedef uint16_t bfloat16_t;

class ScalarInnerProduct {
private:
    /**
     * Convert bfloat16 to float32
     */
    static float bfloat16_to_float(bfloat16_t bf16);

public:
    /**
     * Computes the inner product matrix between centroids and data vectors.
     * 
     * @param centroids: Vector of centroid vectors (rows) in bfloat16_t
     * @param data: Vector of data vectors (columns) in bfloat16_t
     * @return Matrix where output[row][col] = inner product of centroids[row] and data[col] in float32
     */
    std::vector<std::vector<float>> compute(
        const std::vector<std::vector<bfloat16_t>>& centroids,
        const std::vector<std::vector<bfloat16_t>>& data
    );
    
    /**
     * Helper function to print a matrix
     */
    void printMatrix(const std::vector<std::vector<float>>& matrix);
};

#endif // SCALAR_INNER_PRODUCT_H
