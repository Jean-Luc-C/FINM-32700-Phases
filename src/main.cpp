#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cstring>
#ifdef _WIN32
#include <malloc.h>
#else
#include <cstdlib>
#endif
#include "kernels.h"
#include "benchmark.h"

const int ALIGNMENT = 64; // bytes

// Fill array with random values between 0 and 1
void fill_random(double* arr, int size) {
    for (int i = 0; i < size; i++) {
        arr[i] = static_cast<double>(rand()) / RAND_MAX;
    }
}

void benchmark_mv(int rows, int cols, int iterations) {
    double* matrix_rm = new double[rows * cols];
    double* matrix_cm = new double[rows * cols];
    double* vec = new double[cols];
    double* result = new double[rows];

    fill_random(matrix_rm, rows * cols);
    fill_random(vec, cols);

    // Build column-major version of the same matrix
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix_cm[j * rows + i] = matrix_rm[i * cols + j];
        }
    }

    std::string label = std::to_string(rows) + "x" + std::to_string(cols);

    auto r1 = run_benchmark("MV Row-Major    (" + label + ")", [&]() {
        multiply_mv_row_major(matrix_rm, rows, cols, vec, result);
    }, iterations);
    print_result("MV Row-Major    (" + label + ")", r1);

    auto r2 = run_benchmark("MV Col-Major    (" + label + ")", [&]() {
        multiply_mv_col_major(matrix_cm, rows, cols, vec, result);
    }, iterations);
    print_result("MV Col-Major    (" + label + ")", r2);

    delete[] matrix_rm;
    delete[] matrix_cm;
    delete[] vec;
    delete[] result;
}

void benchmark_mm(int n, int iterations) {
    double* A = new double[n * n];
    double* B = new double[n * n];
    double* Bt = new double[n * n];
    double* result = new double[n * n];

    fill_random(A, n * n);
    fill_random(B, n * n);

    // Build transposed B
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Bt[j * n + i] = B[i * n + j];
        }
    }

    std::string label = std::to_string(n) + "x" + std::to_string(n);

    auto r1 = run_benchmark("MM Naive        (" + label + ")", [&]() {
        multiply_mm_naive(A, n, n, B, n, n, result);
    }, iterations);
    print_result("MM Naive        (" + label + ")", r1);

    auto r2 = run_benchmark("MM Transposed-B (" + label + ")", [&]() {
        multiply_mm_transposed_b(A, n, n, Bt, n, n, result);
    }, iterations);
    print_result("MM Transposed-B (" + label + ")", r2);

    delete[] A;
    delete[] B;
    delete[] Bt;
    delete[] result;
}

#ifdef _WIN32
void benchmark_mm_aligned(int n, int iterations) {
    size_t size = n * n * sizeof(double);

    // Unaligned (standard new)
    double* uA = new double[n * n];
    double* uB = new double[n * n];
    double* uBt = new double[n * n];
    double* uResult = new double[n * n];

    // Aligned to 64-byte boundary
    double* aA = static_cast<double*>(_aligned_malloc(size, ALIGNMENT));
    double* aB = static_cast<double*>(_aligned_malloc(size, ALIGNMENT));
    double* aBt = static_cast<double*>(_aligned_malloc(size, ALIGNMENT));
    double* aResult = static_cast<double*>(_aligned_malloc(size, ALIGNMENT));

    fill_random(uA, n * n);
    fill_random(uB, n * n);
    std::memcpy(aA, uA, size);
    std::memcpy(aB, uB, size);

    // Build transposed B for both
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            uBt[j * n + i] = uB[i * n + j];
            aBt[j * n + i] = aB[i * n + j];
        }
    }

    std::string label = std::to_string(n) + "x" + std::to_string(n);

    auto r1 = run_benchmark("MM Naive Unaligned  (" + label + ")", [&]() {
        multiply_mm_naive(uA, n, n, uB, n, n, uResult);
    }, iterations);
    print_result("MM Naive Unaligned  (" + label + ")", r1);

    auto r2 = run_benchmark("MM Naive Aligned    (" + label + ")", [&]() {
        multiply_mm_naive(aA, n, n, aB, n, n, aResult);
    }, iterations);
    print_result("MM Naive Aligned    (" + label + ")", r2);

    auto r3 = run_benchmark("MM Trans-B Unaligned(" + label + ")", [&]() {
        multiply_mm_transposed_b(uA, n, n, uBt, n, n, uResult);
    }, iterations);
    print_result("MM Trans-B Unaligned(" + label + ")", r3);

    auto r4 = run_benchmark("MM Trans-B Aligned  (" + label + ")", [&]() {
        multiply_mm_transposed_b(aA, n, n, aBt, n, n, aResult);
    }, iterations);
    print_result("MM Trans-B Aligned  (" + label + ")", r4);

    delete[] uA;
    delete[] uB;
    delete[] uBt;
    delete[] uResult;
    _aligned_free(aA);
    _aligned_free(aB);
    _aligned_free(aBt);
    _aligned_free(aResult);
}
#endif

// Sums every nth element of a large array to isolate cache stride effects
void benchmark_stride_access(int stride, int iterations) {
    const int total_elements = 1024 * 1024 * 8; // ~64MB of doubles
    int elements_accessed = total_elements / stride;
    double* data = new double[total_elements];
    fill_random(data, total_elements);

    volatile double sink = 0.0; // prevent optimizer from removing the loop

    std::string label = "Stride " + std::to_string(stride);

    auto result = run_benchmark(label, [&]() {
        double sum = 0.0;
        for (int i = 0; i < total_elements; i += stride) {
            sum += data[i];
        }
        sink = sum;
    }, iterations);

    double ns_per_element = (result.mean_ms * 1e6) / elements_accessed;

    std::cout << std::left << std::setw(12) << label
              << "mean: " << std::right << std::setw(10) << std::fixed << std::setprecision(3) << result.mean_ms << " ms"
              << "   elements: " << std::setw(10) << elements_accessed
              << "   ns/element: " << std::setw(8) << std::setprecision(2) << ns_per_element
              << std::endl;

    delete[] data;
}

int main(int argc, char* argv[]) {
    srand(42);

    bool mm_only = false;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--mm-only") {
            mm_only = true;
        }
    }

    if (!mm_only) {
        std::cout << "=== Matrix-Vector Benchmarks ===" << std::endl;
        benchmark_mv(64, 64, 1000);
        benchmark_mv(256, 256, 100);
        benchmark_mv(1024, 1024, 50);
        benchmark_mv(4096, 4096, 10);
        std::cout << std::endl;
    }

    std::cout << "=== Matrix-Matrix Benchmarks ===" << std::endl;
    benchmark_mm(64, 100);
    benchmark_mm(256, 10);
    benchmark_mm(512, 5);
    benchmark_mm(1024, 3);

    if (!mm_only) {
        #ifdef _WIN32
        std::cout << std::endl;
        std::cout << "=== Aligned vs Unaligned Memory ===" << std::endl;
        benchmark_mm_aligned(256, 10);
        benchmark_mm_aligned(512, 5);
        benchmark_mm_aligned(1024, 3);
        #endif

        std::cout << std::endl;
        std::cout << "=== Cache Stride Analysis ===" << std::endl;
        std::cout << "(Summing every Nth element of a 64MB array)" << std::endl;
        benchmark_stride_access(1, 20);
        benchmark_stride_access(2, 20);
        benchmark_stride_access(4, 20);
        benchmark_stride_access(8, 20);
        benchmark_stride_access(16, 20);
        benchmark_stride_access(32, 20);
        benchmark_stride_access(64, 20);
        benchmark_stride_access(128, 20);
    }

    return 0;
}