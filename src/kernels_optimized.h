#pragma once

// Tiled/blocked matrix-matrix multiplication for better cache reuse
void multiply_mm_tiled(const double* matrixA, int rowsA, int colsA, const double* matrixB, int rowsB, int colsB, double* result, int tile_size = 32);