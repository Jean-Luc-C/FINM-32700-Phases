# High-Performance Linear Algebra Kernels

**FINM 32700 — Advanced Computing, Phase 1**

## Team

- Jean-Luc Choiseul

## Project Structure

```
src/
├── kernels.h / kernels.cpp                         # Baseline kernel implementations
├── kernels_optimized.h / kernels_optimized.cpp     # Tiled matrix multiplication
├── benchmark.h / benchmark.cpp                     # Timing and statistics utilities
├── tests.cpp                                       # Correctness tests
└── main.cpp                                        # Benchmark driver
report/
└── report.pdf                                      # Performance analysis report
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

## Discussion Questions
 
### 1. Pointers vs References
 
Pointers can be null, reassigned, and used with arithmetic. References are like pointers that are bound to specific values, and can't be reassigned. In numerical code, pointers are necessary for dynamically allocated arrays where the caller controls allocation and deallocation. References are better for passing values that must always exist.
 
### 2. Row-Major vs Column-Major and Cache Locality
 
Column-major outperformed row-major for matrix-vector multiplication because the loop structure streams through memory sequentially. For matrix-matrix multiplication, the naive version strides through columns of B, causing cache misses. Transposed-B converts these into sequential row accesses, yielding a 1.7x speedup at 1024x1024 under `-O0` and 6.9x under `-O3`.
 
### 3. CPU Caches and Locality
 
Temporal locality means reusing recently accessed data. Spatial locality means accessing nearby data. The tiled implementation exploits both: tiles of A are reused across multiple tiles of B (temporal), and elements within each tile are accessed sequentially (spatial). The stride benchmark confirmed spatial locality, with cost per element roughly tripling between stride 1 and stride 16 as cache line utilization dropped.
 
### 4. Memory Alignment
 
Alignment places data at addresses that are multiples of a boundary (e.g., 64 bytes). Our experiments showed no consistent improvement from 64-byte alignment over standard `new`, because modern x86 handles unaligned double access efficiently and the default allocator already provides sufficient alignment. The bottleneck was cache access patterns, not alignment.
 
### 5. Compiler Optimizations and Inlining
 
Transposed-B saw a 5.1x speedup from `-O0` to `-O3`, while naive saw only 1.3x. The compiler can vectorize sequential access patterns but cannot fix cache-unfriendly access orders. Cache-aware code and compiler optimization provide multiplicative gains together. Potential drawbacks of aggressive optimization include longer compile times, and harder debugging (variables get optimized out, code is reordered)
 
### 6. Profiling and Bottlenecks
 
gprof (run via WSL) showed the two multiply kernels accounted for 99.7% of execution time: `multiply_mm_naive` at 58.7% and `multiply_mm_transposed_b` at 41.0%. Everything else was negligible, confirming that optimization should target the kernels, which motivated the tiled implementation.
 
### 7. Teamwork Reflection
 
This project was completed individually. The assignment structure would be very open ended for a team since probably the easiest part, the base kernel functions, was the only part with a prescribed way of splitting up the work. I liked working by myself I think having teammates wouldn't have really added much. Although there would have been more brainstorming with another teammate, which would have been nice.
