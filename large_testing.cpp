#include "AMXInnerProductBF16.h"
#include "ScalarInnerProduct.h"
#include "arrow/api.h"
#include "arrow/io/api.h"
#include "arrow/ipc/api.h"
#include "parquet/arrow/reader.h"
#include <algorithm>
#include <random>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cstring>

typedef uint16_t bfloat16_t;

// Define these constants based on your data
const int dim = 128;             // Adjust to your embedding dimension
const int max_elements = 960000; // Maximum number of vectors to load
const int num_centroids = 16;
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

static void differenceAnalyzer(std::vector<std::vector<float>> scalar_results, std::vector<std::vector<float>> AMX_results)
{
    std::vector<std::vector<float>> subtract_results(scalar_results.size(), std::vector<float>(scalar_results[0].size()));
    float average_error = 0.0f;
    float max_error = 0.0f;

    for (int i = 0; i < scalar_results.size(); i++)
    {
        for (int j = 0; j < scalar_results[0].size(); j++)
        {
            subtract_results[i][j] = std::fabs(AMX_results[i][j] - scalar_results[i][j]);
            average_error += subtract_results[i][j];
            if (subtract_results[i][j] > max_error)
            {
                max_error = subtract_results[i][j];
            }
        }
    }

    average_error = average_error / (scalar_results.size() * scalar_results[0].size());
    std::cout << "Average difference between Scalar and AMX -- " << average_error << std::endl;
    std::cout << "Largest difference between Scalar and AMX -- " << max_error << std::endl;
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

    // AMX computation
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

    // Scalar computation
    auto scalar_start = std::chrono::high_resolution_clock::now();
    ScalarInnerProduct scalar_calculator;
    std::vector<std::vector<float>> scalar_results = scalar_calculator.compute(random_centroids_bf16, data_bf16);
    auto scalar_end = std::chrono::high_resolution_clock::now();

    // Print timing information
    std::cout << "AVERAGE AMX Calculation function took: " << AMX_total_time / rounds << " microseconds" << std::endl;

    auto scalar_duration = std::chrono::duration_cast<std::chrono::microseconds>(scalar_end - scalar_start);
    std::cout << "Scalar Calculation function took: " << scalar_duration.count() << " microseconds" << std::endl;
    std::cout << "Scalar runtime + Preprocessing took: " << scalar_duration.count() + init_duration.count() << " microseconds\n"
              << std::endl;

    // Uncomment to see detailed timing stats
    amx_calculator.print_timing_stats();
    
    differenceAnalyzer(scalar_results, AMX_results);

    // Uncomment to print results for debugging
    // scalar_calculator.printMatrix(AMX_results);
    // scalar_calculator.printMatrix(scalar_results);

    return 0;
}
