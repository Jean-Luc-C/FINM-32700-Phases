#include "kernels_optimized.h"
#include <stdexcept>
#include <cstring>

void multiply_mm_tiled(const double* matrixA, int rowsA, int colsA, const double* matrixB, int rowsB, int colsB, double* result, int tile_size) {
    if (!matrixA || !matrixB || !result) {
        throw std::invalid_argument("Null pointer passed to multiply_mm_tiled");
    }
    if (colsA != rowsB) {
        throw std::invalid_argument("Incompatible dimensions for matrix multiplication");
    }

    // Zero out result
    std::memset(result, 0, rowsA * colsB * sizeof(double));

    // Tiled multiplication: iterate over tiles of the output
    for (int i0 = 0; i0 < rowsA; i0 += tile_size) {
        for (int j0 = 0; j0 < colsB; j0 += tile_size) {
            for (int k0 = 0; k0 < colsA; k0 += tile_size) {

                // Multiply within the tile
                int i_max = std::min(i0 + tile_size, rowsA);
                int j_max = std::min(j0 + tile_size, colsB);
                int k_max = std::min(k0 + tile_size, colsA);

                for (int i = i0; i < i_max; i++) {
                    for (int k = k0; k < k_max; k++) {
                        double a_ik = matrixA[i * colsA + k];
                        for (int j = j0; j < j_max; j++) {
                            result[i * colsB + j] += a_ik * matrixB[k * colsB + j];
                        }
                    }
                }

            }
        }
    }
}