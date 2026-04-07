# High-Performance Linear Algebra Kernels

**FINM 32700 — Advanced Computing, Phase 1**

## Team

- Jean-Luc Choiseul

## Project Structure

```
src/
├── kernels.h / kernels.cpp               # Baseline kernel implementations
├── kernels_optimized.h / kernels_optimized.cpp  # Tiled matrix multiplication
├── benchmark.h / benchmark.cpp            # Timing and statistics utilities
├── tests.cpp                              # Correctness tests
└── main.cpp                               # Benchmark driver
report/
└── report.pdf                             # Performance analysis report
Makefile
```

## Build & Run

Requires `g++` with C++17 support (tested on MSYS2 MinGW/UCRT64).

| Command | Description |
|---------|-------------|
| `make test` | Builds and runs correctness tests for all kernel functions |
| `make run` | Builds and runs full benchmark suite with `-O3` optimization |
| `make noopt` | Builds and runs benchmarks with `-O0` (no optimization) |
| `make clean` | Removes all compiled binaries and profiling output |

### Profiling (WSL)

gprof does not produce valid output on MSYS2/MinGW. To profile, use WSL:

```bash
g++ -O0 -pg -g -std=c++17 -o main_profile src/main.cpp src/kernels.cpp src/kernels_optimized.cpp src/benchmark.cpp
./main_profile --mm-only
gprof main_profile gmon.out > profile_output.txt
```

