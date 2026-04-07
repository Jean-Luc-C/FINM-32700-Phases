#include "benchmark.h"
#include <chrono>
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>

BenchmarkResult run_benchmark(const std::string& name, std::function<void()> fn, int iterations) {
    std::vector<double> times(iterations);

    for (int i = 0; i < iterations; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        fn();
        auto end = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration<double, std::milli>(end - start).count();
    }

    double sum = 0.0;
    for (double t : times) sum += t;
    double mean = sum / iterations;

    double sq_sum = 0.0;
    for (double t : times) sq_sum += (t - mean) * (t - mean);
    double stddev = std::sqrt(sq_sum / iterations);

    return {mean, stddev, iterations};
}

void print_result(const std::string& name, const BenchmarkResult& result) {
    std::cout << std::left << std::setw(40) << name
              << "mean: " << std::right << std::setw(10) << std::fixed << std::setprecision(3) << result.mean_ms << " ms"
              << "   stddev: " << std::setw(10) << result.stddev_ms << " ms"
              << "   (" << result.iterations << " runs)"
              << std::endl;
}