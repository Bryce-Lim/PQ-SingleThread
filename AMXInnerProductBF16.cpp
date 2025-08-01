//==============================================================//
// Author: Bryce Lim, 2025                                      //
// SPDX-License-Identifier: MIT                                 //
//                                                              //
// This code is based in part on Intel's 2022 starter code:     //
// https://github.com/intel/AMX-TMUL-Code-Samples               //
// Licensed under the MIT License.                              //
//                                                              //
// Modifications and extensions by Bryce Lim, 2025              //
//==============================================================//

#include "AMXInnerProductBF16.h"
#include <iostream>
#include <stdio.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <stdlib.h>
#include <cstring>
#include <iomanip>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <vector>
#include <immintrin.h>
#include <cstdint>

// Constructor
AMXInnerProductBF16::AMXInnerProductBF16() : amx_initialized(false)
{
    reset_timers();
}

// Destructor
AMXInnerProductBF16::~AMXInnerProductBF16()
{
    if (amx_initialized)
    {
        _tile_release();
    }
}

// Initialize AMX functionality
bool AMXInnerProductBF16::initialize()
{
    amx_initialized = true;
    return true;
}

// Timing getter methods
double AMXInnerProductBF16::get_total_compute_time_ms() const { return total_compute_time.count() * 1000.0; }
double AMXInnerProductBF16::get_padding_time_ms() const { return padding_time.count() * 1000.0; }
double AMXInnerProductBF16::get_conversion_time_ms() const { return conversion_time.count() * 1000.0; }
double AMXInnerProductBF16::get_chunking_time_ms() const { return chunking_time.count() * 1000.0; }
double AMXInnerProductBF16::get_multiplication_time_ms() const { return multiplication_time.count() * 1000.0; }
double AMXInnerProductBF16::get_tile_setup_time_ms() const { return tile_setup_time.count() * 1000.0; }
double AMXInnerProductBF16::get_actual_amx_time_ms() const { return actual_amx_time.count() * 1000.0; }
double AMXInnerProductBF16::get_tile_load_time_ms() const { return tile_load_time.count() * 1000.0; }

// Reset all timing counters
void AMXInnerProductBF16::reset_timers()
{
    total_compute_time = std::chrono::duration<double>::zero();
    padding_time = std::chrono::duration<double>::zero();
    conversion_time = std::chrono::duration<double>::zero();
    chunking_time = std::chrono::duration<double>::zero();
    multiplication_time = std::chrono::duration<double>::zero();
    tile_setup_time = std::chrono::duration<double>::zero();
    actual_amx_time = std::chrono::duration<double>::zero();
    tile_load_time = std::chrono::duration<double>::zero();
}

// Print comprehensive timing statistics
void AMXInnerProductBF16::print_timing_stats() const
{
    std::cout << "\n=== AMX Inner Product Timing Statistics ===\n";
    std::cout << std::fixed << std::setprecision(3);

    std::cout << " Total compute time:        " << std::setw(8) << get_total_compute_time_ms() << " ms\n";
    std::cout << " - Padding time:            " << std::setw(8) << get_padding_time_ms() << " ms\n";
    std::cout << " - Chunking time:           " << std::setw(8) << get_chunking_time_ms() << " ms\n";
    std::cout << "   - Multiplication time:   " << std::setw(8) << get_multiplication_time_ms() << " ms\n";
    std::cout << "     - Conversion time:     " << std::setw(8) << get_conversion_time_ms() << " ms\n";
    std::cout << "     - Result merging time: " << std::setw(8) << get_tile_setup_time_ms() << " ms\n";
    std::cout << "     - Actual AMX time:     " << std::setw(8) << get_actual_amx_time_ms() << " ms\n";
    std::cout << "     - Tile load time:      " << std::setw(8) << get_tile_load_time_ms() << " ms\n";

    std::cout << "===========================================\n\n";
}

// Initialize tile config
void AMXInnerProductBF16::init_tile_config(__tilecfg *tileinfo)
{
    int i;
    tileinfo->palette_id = 1;
    tileinfo->start_row = 0;

    // Tile 1: accumulator (float32)
    tileinfo->colsb[0] = MAX_SIZE * sizeof(float);
    tileinfo->rows[0] = MAX_SIZE;

    // Tiles 2,3: bfloat16 operands
    for (i = 1; i < 4; ++i)
    {
        tileinfo->colsb[i] = MAX_COLS * sizeof(bfloat16_t);
        tileinfo->rows[i] = MAX_SIZE;
    }

    _tile_loadconfig(tileinfo);
}

// Padding Vectors
void AMXInnerProductBF16::padVectors(std::vector<std::vector<bfloat16_t>> &vectors)
{
    // 1. Reserve outer vector capacity upfront
    int padded_size = (vectors.size() + 15) & ~15;
    vectors.reserve(padded_size);
    vectors.resize(padded_size);

    // 2. Optimize inner vector padding
    for (auto &vec : vectors)
    {
        if (vec.empty())
        {
            vec.resize(MAX_COLS);
        }
        else
        {
            size_t current_size = vec.size();
            size_t padded_inner_size = ((current_size + MAX_COLS - 1) / MAX_COLS) * MAX_COLS;
            vec.resize(padded_inner_size);
        }
    }
}

// Main public interface for computing inner products
std::vector<std::vector<float>> AMXInnerProductBF16::compute_inner_products(std::vector<std::vector<bfloat16_t>> &centroids, std::vector<std::vector<bfloat16_t>> &data)
{
    auto start_total = std::chrono::high_resolution_clock::now();

    if (!amx_initialized)
    {
        throw std::runtime_error("AMX not initialized. Call initialize() first.");
    }

    // Time padding
    auto start_padding = std::chrono::high_resolution_clock::now();
    padVectors(centroids);
    padVectors(data);
    auto end_padding = std::chrono::high_resolution_clock::now();
    padding_time += end_padding - start_padding;

    // Prepare result matrix
    std::vector<std::vector<float>> results_agg(centroids.size(), std::vector<float>(data.size()));

    // Perform the computation (timing handled inside chunking)
    main_multiply(results_agg, centroids, data);

    auto end_total = std::chrono::high_resolution_clock::now();
    total_compute_time += end_total - start_total;

    return results_agg;
}

void AMXInnerProductBF16::main_multiply(std::vector<std::vector<float>> &results_agg, std::vector<std::vector<bfloat16_t>> &centroids, std::vector<std::vector<bfloat16_t>> &data)
{
    int centroid_height = centroids.size() / MAX_SIZE;
    int data_height = data[0].size() / MAX_COLS;

    auto start_chunking = std::chrono::high_resolution_clock::now();

    float results_chunk[MAX_SIZE * MAX_SIZE];
    std::vector<std::vector<bfloat16_t>> centroid_chunk;

    // Chunk and format centroids
    auto start_conversion = std::chrono::high_resolution_clock::now();
    centroid_format(centroids, centroid_chunk);
    auto end_conversion = std::chrono::high_resolution_clock::now();
    conversion_time += end_conversion - start_conversion;

    std::cout << "Centroid formatting done!\n";

    // Tile init!
    __tilecfg tile_data = {0};
    init_tile_config(&tile_data);

    int id = 0;
    int centroid_id = 0;
    int chunk_index = 0;

    std::vector<bfloat16_t> data_chunk(MAX_SIZE * MAX_COLS);
    for (int offset = 0; offset < data.size(); offset += MAX_SIZE)
    {
        for (int d_offset = 0; d_offset < data[0].size(); d_offset += MAX_COLS)
        {
            // Chunk and format data
            auto start_conversion = std::chrono::high_resolution_clock::now();
            data_format(data, data_chunk, offset, d_offset);
            auto end_conversion = std::chrono::high_resolution_clock::now();
            conversion_time += end_conversion - start_conversion;

	    std::cout << "Data formatting done!\n";

            // Multiply using AMX
            for (int i = 0; i < centroid_height; ++i)
            {
                _tile_zero(1);
		std::cout << "Tile 1 loaded!\n";
                _tile_loadd(2, centroid_chunk[centroid_id].data(), STRIDE);
		std::cout << "Tile 2 (centroid) loaded!\n";
                _tile_loadd(3, data_chunk.data(), STRIDE);
		std::cout << "Tile 3 (data) loaded!\n";

                _tile_dpbf16ps(1, 3, 2);
		std::cout << "Tile 1 += Tile 3 * Tile 2 done!\n";
                _tile_stored(1, results_chunk, STRIDE);
		std::cout << "Stored in tile 1!\n";

		std::cout << "AMX Multiplication done!\n";
                // Merge results (same as before)
                int col_offset = (id / data_height) * MAX_SIZE;
                for (int row = 0; row < MAX_SIZE; ++row)
                {
                    float *chunk_row = &results_chunk[row * MAX_SIZE];
                    float *agg_row = &results_agg[i * MAX_SIZE + row][col_offset];

                    int col = 0;
                    for (; col <= MAX_SIZE - 16; col += 16)
                    {
                        __m512 chunk_vec1 = _mm512_loadu_ps(&chunk_row[col]);
                        __m512 agg_vec1 = _mm512_loadu_ps(&agg_row[col]);
                        __m512 result1 = _mm512_add_ps(agg_vec1, chunk_vec1);
                        _mm512_storeu_ps(&agg_row[col], result1);
                    }
                }
		std::cout << "Results merging done!\n";
                centroid_id = (centroid_id + 1) % centroid_chunk.size();
            }
            chunk_index++;
            id++;
        }
    }
}

void AMXInnerProductBF16::centroid_format(std::vector<std::vector<bfloat16_t>> &centroids, std::vector<std::vector<bfloat16_t>> &centroid_chunk) {
    std::vector<bfloat16_t> chunk(MAX_COLS * MAX_SIZE);
    for (int offset = 0; offset < centroids.size(); offset += MAX_SIZE)
    {
        for (int d_offset = 0; d_offset < centroids[0].size(); d_offset += MAX_COLS)
        {
            int k = 0;
            for (int i = 0; i < MAX_COLS; i += 2)
            {
                for (int j = 0; j < MAX_SIZE; j++)
                {
                    chunk[k++] = centroids[offset + j][d_offset + i];
                    chunk[k++] = centroids[offset + j][d_offset + i + 1];
                }
            }
            centroid_chunk.push_back(chunk);
        }
    }
}

void AMXInnerProductBF16::data_format(std::vector<std::vector<bfloat16_t>> &data, std::vector<bfloat16_t> &data_chunk, int data_num, int element_num)
{
    for (int i = 0; i < MAX_SIZE; i++)
    {
        std::memcpy(&data_chunk[i * MAX_COLS], &data[data_num + i][element_num], MAX_COLS * sizeof(bfloat16_t));
    }
}

bfloat16_t AMXInnerProductBF16::float_to_bfloat16(float f)
{
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(float));

    // Round to nearest even and truncate to bfloat16
    uint32_t rounding_bias = 0x00007FFF + ((bits >> 16) & 1);
    return static_cast<bfloat16_t>((bits + rounding_bias) >> 16);
}

float AMXInnerProductBF16::bfloat16_to_float(bfloat16_t bf16)
{
    uint32_t bits = static_cast<uint32_t>(bf16) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(float));
    return result;
}

void AMXInnerProductBF16::print_bfloat16_vectors(const std::vector<std::vector<bfloat16_t>> &vecs)
{
    std::cout << "Vector of vectors (bfloat16):\n";

    for (size_t i = 0; i < vecs.size(); ++i)
    {
        std::cout << "Vector " << i << ":\t[";
        for (size_t j = 0; j < vecs[i].size(); ++j)
        {
            std::cout << bfloat16_to_float(vecs[i][j]);
            if (j < vecs[i].size() - 1)
            {
                std::cout << ", ";
            }
        }
        std::cout << "]\n";
    }
    std::cout << "]\n";
}
