#include <iostream>
#include <cmath>
#include "kernels.h"
#include "kernels_optimized.h"

const double TOLERANCE = 1e-9;

bool check_equal(const double* actual, const double* expected, int size) {
    for (int i = 0; i < size; i++) {
        if (std::abs(actual[i] - expected[i]) > TOLERANCE) {
            return false;
        }
    }
    return true;
}

void test_mv_row_major() {
    // 2x3 matrix, row-major: [[1, 2, 3], [4, 5, 6]]
    double* matrix = new double[6]{1, 2, 3, 4, 5, 6};
    double* vector = new double[3]{1, 2, 3};
    double* result = new double[2];
    double expected[] = {14, 32}; // [1*1+2*2+3*3, 4*1+5*2+6*3]

    multiply_mv_row_major(matrix, 2, 3, vector, result);

    if (check_equal(result, expected, 2)) {
        std::cout << "PASS: multiply_mv_row_major" << std::endl;
    } else {
        std::cout << "FAIL: multiply_mv_row_major" << std::endl;
    }

    delete[] matrix;
    delete[] vector;
    delete[] result;
}

void test_mv_col_major() {
    // Same 2x3 matrix, column-major: columns stored contiguously
    // [[1, 2, 3], [4, 5, 6]] -> stored as [1, 4, 2, 5, 3, 6]
    double* matrix = new double[6]{1, 4, 2, 5, 3, 6};
    double* vector = new double[3]{1, 2, 3};
    double* result = new double[2];
    double expected[] = {14, 32};

    multiply_mv_col_major(matrix, 2, 3, vector, result);

    if (check_equal(result, expected, 2)) {
        std::cout << "PASS: multiply_mv_col_major" << std::endl;
    } else {
        std::cout << "FAIL: multiply_mv_col_major" << std::endl;
    }

    delete[] matrix;
    delete[] vector;
    delete[] result;
}

void test_mm_naive() {
    // A: 2x3 [[1, 2, 3], [4, 5, 6]]
    // B: 3x2 [[7, 8], [9, 10], [11, 12]]
    double* A = new double[6]{1, 2, 3, 4, 5, 6};
    double* B = new double[6]{7, 8, 9, 10, 11, 12};
    double* result = new double[4];
    double expected[] = {58, 64, 139, 154};

    multiply_mm_naive(A, 2, 3, B, 3, 2, result);

    if (check_equal(result, expected, 4)) {
        std::cout << "PASS: multiply_mm_naive" << std::endl;
    } else {
        std::cout << "FAIL: multiply_mm_naive" << std::endl;
    }

    delete[] A;
    delete[] B;
    delete[] result;
}

void test_mm_transposed_b() {
    // A: 2x3 [[1, 2, 3], [4, 5, 6]]
    // B: 3x2 [[7, 8], [9, 10], [11, 12]]
    // B transposed: 2x3 [[7, 9, 11], [8, 10, 12]]
    double* A = new double[6]{1, 2, 3, 4, 5, 6};
    double* B_transposed = new double[6]{7, 9, 11, 8, 10, 12};
    double* result = new double[4];
    double expected[] = {58, 64, 139, 154};

    multiply_mm_transposed_b(A, 2, 3, B_transposed, 2, 3, result);

    if (check_equal(result, expected, 4)) {
        std::cout << "PASS: multiply_mm_transposed_b" << std::endl;
    } else {
        std::cout << "FAIL: multiply_mm_transposed_b" << std::endl;
    }

    delete[] A;
    delete[] B_transposed;
    delete[] result;
}

void test_mm_tiled() {
    // A: 2x3 [[1, 2, 3], [4, 5, 6]]
    // B: 3x2 [[7, 8], [9, 10], [11, 12]]
    double* A = new double[6]{1, 2, 3, 4, 5, 6};
    double* B = new double[6]{7, 8, 9, 10, 11, 12};
    double* result = new double[4];
    double expected[] = {58, 64, 139, 154};

    multiply_mm_tiled(A, 2, 3, B, 3, 2, result, 2);

    if (check_equal(result, expected, 4)) {
        std::cout << "PASS: multiply_mm_tiled" << std::endl;
    } else {
        std::cout << "FAIL: multiply_mm_tiled" << std::endl;
    }

    delete[] A;
    delete[] B;
    delete[] result;
}

int main() {
    std::cout << "Running kernel tests..." << std::endl;
    test_mv_row_major();
    test_mv_col_major();
    test_mm_naive();
    test_mm_transposed_b();
    test_mm_tiled();
    return 0;
}