#pragma once

#include <functional>
#include <string>

struct BenchmarkResult {
    double mean_ms;
    double stddev_ms;
    int iterations;
};

BenchmarkResult run_benchmark(const std::string& name, std::function<void()> fn, int iterations = 10);

void print_result(const std::string& name, const BenchmarkResult& result);