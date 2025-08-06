#include "AMXInnerProductBF16.h"
#include "ScalarInnerProduct.h"
#include "HnswlibInnerProduct.h"
#include "arrow/api.h"
#include "arrow/io/api.h"
#include "arrow/ipc/api.h"
#include "parquet/arrow/reader.h"
#include <algorithm>
#include <random>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <chrono>
#include <cstring>

typedef uint16_t bfloat16_t;

// Define these constants based on your data
const int dim = 1024;             // Adjust to your embedding dimension
const int max_elements = 1000; // Maximum number of vectors to load
const int num_centroids = 256;
const int rounds = 1;
const std::string dataroot = "/mnt/ceph/district9/dataset/openai/openai_large_5m/"; // Set your data directory

// Convert float32 to bfloat16
static bfloat16_t float_to_bfloat16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(float));
    
    // Round to nearest even and truncate to bfloat16
    uint32_t rounding_bias = 0x00007FFF + ((bits >> 16) & 1);
    return static_cast<bfloat16_t>((bits + rounding_bias) >> 16);
}

// Convert bfloat16 to float32 for debugging/printing
static float bfloat16_to_float(bfloat16_t bf16) {
    uint32_t f32_bits = static_cast<uint32_t>(bf16) << 16;
    float result;
    std::memcpy(&result, &f32_bits, sizeof(float));
    return result;
}

static void differenceAnalyzer(const std::vector<std::vector<float>>& scalar_results, 
                              const std::vector<std::vector<float>>& comparison_results, 
                              const std::string& comparison_name)
{
    std::vector<std::vector<float>> subtract_results(scalar_results.size(), std::vector<float>(scalar_results[0].size()));
    float average_error = 0.0f;
    float max_error = 0.0f;

    for (size_t i = 0; i < scalar_results.size(); i++)
    {
        for (size_t j = 0; j < scalar_results[0].size(); j++)
        {
            subtract_results[i][j] = std::fabs(comparison_results[i][j] - scalar_results[i][j]);
            average_error += subtract_results[i][j];
            if (subtract_results[i][j] > max_error)
            {
                max_error = subtract_results[i][j];
            }
        }
    }

    average_error = average_error / (scalar_results.size() * scalar_results[0].size());
    std::cout << "Average difference between Scalar and " << comparison_name << " -- " << average_error << std::endl;
    std::cout << "Largest difference between Scalar and " << comparison_name << " -- " << max_error << std::endl;
}

int main()
{
    // Start Timer for Initialization
    auto init_start = std::chrono::high_resolution_clock::now();

    // Reading parquet files (0, 1 - 1m size)
    std::vector<std::vector<float>> data_float;  // Temporary float storage
    data_float.reserve(max_elements);

    int cnt = 0;
    size_t partition_size = 500000;

    for (int file_idx = 0; file_idx < 2; file_idx++)
    {
        auto pool = arrow::default_memory_pool();
        std::shared_ptr<arrow::io::RandomAccessFile> input;

        std::string path = dataroot + "train-0";
        path += std::to_string(file_idx);
        path += "-of-10.parquet";

        auto maybe_input = arrow::io::ReadableFile::Open(path);
        if (!maybe_input.ok())
        {
            std::cerr << "Error opening file: " << maybe_input.status().ToString() << std::endl;
            return -1;
        }
        input = maybe_input.ValueUnsafe();

        std::unique_ptr<parquet::arrow::FileReader> arrow_reader;
        auto status = parquet::arrow::OpenFile(input, pool, &arrow_reader);
        if (!status.ok())
        {
            std::cerr << "Error opening parquet file: " << status.ToString() << std::endl;
            return -2;
        }

        std::shared_ptr<arrow::Table> table;
        status = arrow_reader->ReadTable(&table);
        if (!status.ok())
        {
            std::cerr << "Error reading table: " << status.ToString() << std::endl;
            return -3;
        }

        auto emb_col = table->column(1);
        if (emb_col->chunks().size() != 1)
        {
            std::cout << "Multiple chunks found: " << emb_col->chunks().size() << std::endl;
        }

        for (auto &arr : emb_col->chunks())
        {
            auto val = std::static_pointer_cast<arrow::DoubleArray>(
                std::static_pointer_cast<arrow::ListArray>(arr)->values());

            for (int i = 0; i < partition_size && data_float.size() < max_elements; i++)
            {
                std::vector<float> vec(dim);
                for (int j = 0; j < dim; j++)
                {
                    vec[j] = (float)val->Value(i * dim + j);
                }
                data_float.push_back(vec);
            }
        }
        cnt++;
    }

    // Normalize vectors (still in float)
    for (auto &emb : data_float)
    {
        float mag = 0;
        for (int d = 0; d < dim; d++)
        {
            mag += emb[d] * emb[d];
        }
        mag = sqrt(mag);

        if (mag > 0)
        { // Avoid division by zero
            for (int d = 0; d < dim; d++)
            {
                emb[d] /= mag;
            }
        }
    }

    // Sample random centroids (still in float)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<std::vector<float>> random_centroids_float;
    std::sample(data_float.begin(), data_float.end(), std::back_inserter(random_centroids_float), num_centroids, gen);

    // Convert data and centroids to bfloat16
    std::vector<std::vector<bfloat16_t>> data_bf16;
    data_bf16.reserve(data_float.size());
    for (const auto& vec_float : data_float) {
        std::vector<bfloat16_t> vec_bf16(vec_float.size());
        for (size_t i = 0; i < vec_float.size(); i++) {
            vec_bf16[i] = float_to_bfloat16(vec_float[i]);
        }
        data_bf16.push_back(vec_bf16);
    }

    std::vector<std::vector<bfloat16_t>> random_centroids_bf16;
    random_centroids_bf16.reserve(random_centroids_float.size());
    for (const auto& vec_float : random_centroids_float) {
        std::vector<bfloat16_t> vec_bf16(vec_float.size());
        for (size_t i = 0; i < vec_float.size(); i++) {
            vec_bf16[i] = float_to_bfloat16(vec_float[i]);
        }
        random_centroids_bf16.push_back(vec_bf16);
    }

    // Clean up float vectors to save memory
    data_float.clear();
    data_float.shrink_to_fit();
    random_centroids_float.clear();
    random_centroids_float.shrink_to_fit();

    std::cout << "Successfully loaded " << data_bf16.size() << " vectors of dimension " << data_bf16[0].size() << std::endl;
    std::cout << "Sampled " << random_centroids_bf16.size() << " random centroids" << std::endl;
    std::cout << "All data converted to bfloat16 format" << std::endl;

    auto init_end = std::chrono::high_resolution_clock::now();

    auto init_duration = std::chrono::duration_cast<std::chrono::microseconds>(init_end - init_start);
    std::cout << "Preprocessing / Initialization took: " << init_duration.count() << " microseconds\n" << std::endl;

    // Create copies for AMX (which might modify the input)
    std::vector<std::vector<bfloat16_t>> centroids_copy = random_centroids_bf16;
    std::vector<std::vector<bfloat16_t>> data_copy = data_bf16;

    // ===== SCALAR COMPUTATION =====
    std::cout << "=== Running Scalar Computation ===" << std::endl;
    auto scalar_start = std::chrono::high_resolution_clock::now();
    ScalarInnerProduct scalar_calculator;
    std::vector<std::vector<float>> scalar_results = scalar_calculator.compute(random_centroids_bf16, data_bf16);
    auto scalar_end = std::chrono::high_resolution_clock::now();
    auto scalar_duration = std::chrono::duration_cast<std::chrono::microseconds>(scalar_end - scalar_start);
    std::cout << "Scalar Calculation function took: " << scalar_duration.count() << " microseconds" << std::endl;

    // ===== HNSWLIB COMPUTATION =====
    std::cout << "\n=== Running Hnswlib-optimized Computation ===" << std::endl;
    HnswlibInnerProduct hnswlib_calculator;
    long hnswlib_total_time = 0;
    std::vector<std::vector<float>> hnswlib_results;
    
    for (int i = 0; i < rounds; i++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        hnswlib_calculator.reset_timers();
        hnswlib_results = hnswlib_calculator.compute(random_centroids_bf16, data_bf16);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        hnswlib_total_time += duration.count();
        std::cout << "Hnswlib Calculation function took: " << duration.count() << " microseconds" << std::endl;
    }

    // ===== AMX COMPUTATION =====
    std::cout << "\n=== Running AMX Computation ===" << std::endl;
    AMXInnerProductBF16 amx_calculator;
    long AMX_total_time = 0;
    std::vector<std::vector<float>> AMX_results;
    for (int i = 0; i < rounds; i++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        if (amx_calculator.initialize())
        {
            amx_calculator.reset_timers();
            AMX_results = amx_calculator.compute_inner_products(centroids_copy, data_copy);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        AMX_total_time += duration.count();
        std::cout << "AMX Calculation function took: " << duration.count() << " microseconds" << std::endl;
    }

    // ===== PERFORMANCE COMPARISON =====
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "                PERFORMANCE COMPARISON" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    std::cout << "Scalar runtime:                    " << std::setw(10) << scalar_duration.count() << " μs" << std::endl;
    std::cout << "Hnswlib AVERAGE runtime:          " << std::setw(10) << hnswlib_total_time / rounds << " μs" << std::endl;
    std::cout << "AMX AVERAGE runtime:              " << std::setw(10) << AMX_total_time / rounds << " μs" << std::endl;

    // Calculate speedups
    double hnswlib_speedup = static_cast<double>(scalar_duration.count()) / (hnswlib_total_time / rounds);
    double amx_speedup = static_cast<double>(scalar_duration.count()) / (AMX_total_time / rounds);
    double hnswlib_vs_amx = static_cast<double>(hnswlib_total_time / rounds) / (AMX_total_time / rounds);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\nSpeedup vs Scalar:" << std::endl;
    std::cout << "  Hnswlib:                         " << std::setw(10) << hnswlib_speedup << "x" << std::endl;
    std::cout << "  AMX:                             " << std::setw(10) << amx_speedup << "x" << std::endl;
    std::cout << "\nHnswlib vs AMX ratio:             " << std::setw(10) << hnswlib_vs_amx << "x" << std::endl;

    if (hnswlib_vs_amx > 1.0) {
        std::cout << "  (AMX is " << hnswlib_vs_amx << "x faster than Hnswlib)" << std::endl;
    } else {
        std::cout << "  (Hnswlib is " << (1.0/hnswlib_vs_amx) << "x faster than AMX)" << std::endl;
    }

    std::cout << std::string(60, '=') << std::endl;

    // Print detailed timing stats
    std::cout << "\n=== DETAILED TIMING BREAKDOWN ===" << std::endl;
    hnswlib_calculator.print_timing_stats();
    amx_calculator.print_timing_stats();

    // ===== ACCURACY COMPARISON =====
    std::cout << "\n=== ACCURACY COMPARISON ===" << std::endl;
    differenceAnalyzer(scalar_results, hnswlib_results, "Hnswlib");
    differenceAnalyzer(scalar_results, AMX_results, "AMX");

    // Additional metrics
    std::cout << "\n=== PROBLEM SIZE METRICS ===" << std::endl;
    std::cout << "Matrix size:                       " << random_centroids_bf16.size() << " x " << data_bf16.size() << std::endl;
    std::cout << "Total computations:                " << random_centroids_bf16.size() * data_bf16.size() << std::endl;
    std::cout << "Vector dimension:                  " << data_bf16[0].size() << std::endl;
    std::cout << "Total multiply-adds:               " << random_centroids_bf16.size() * data_bf16.size() * data_bf16[0].size() << std::endl;

    // Throughput calculations
    long long total_ops = static_cast<long long>(random_centroids_bf16.size()) * data_bf16.size() * data_bf16[0].size();
    
    std::cout << "\n=== THROUGHPUT (GFLOPS) ===" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    
    double scalar_gflops = (total_ops * 2.0) / (scalar_duration.count() * 1e-6) / 1e9;  // *2 for multiply-add
    double hnswlib_gflops = (total_ops * 2.0) / ((hnswlib_total_time / rounds) * 1e-6) / 1e9;
    double amx_gflops = (total_ops * 2.0) / ((AMX_total_time / rounds) * 1e-6) / 1e9;
    
    std::cout << "Scalar throughput:                 " << std::setw(10) << scalar_gflops << " GFLOPS" << std::endl;
    std::cout << "Hnswlib throughput:                " << std::setw(10) << hnswlib_gflops << " GFLOPS" << std::endl;
    std::cout << "AMX throughput:                    " << std::setw(10) << amx_gflops << " GFLOPS" << std::endl;

    // Uncomment to print results for debugging
    // std::cout << "\n=== SAMPLE RESULTS ===" << std::endl;
    // hnswlib_calculator.printMatrix(AMX_results);
    // scalar_calculator.printMatrix(scalar_results);

    return 0;
}
