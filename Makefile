CXX = g++
CXXFLAGS_RELEASE = -O3 -std=c++17
CXXFLAGS_DEBUG = -O0 -pg -g -std=c++17

# Default target
all: main.exe tests.exe

# Benchmark binary (release)
main.exe: src/main.cpp src/kernels.cpp src/kernels_optimized.cpp src/benchmark.cpp
	$(CXX) $(CXXFLAGS_RELEASE) -o $@ $^

# Test binary
tests.exe: src/tests.cpp src/kernels.cpp src/kernels_optimized.cpp
	$(CXX) $(CXXFLAGS_RELEASE) -o $@ $^

# Benchmark with no optimization (for inlining comparison)
noopt: src/main.cpp src/kernels.cpp src/kernels_optimized.cpp src/benchmark.cpp
	$(CXX) $(CXXFLAGS_DEBUG) -o main_noopt.exe $^
	./main_noopt.exe

test: tests.exe
	./tests.exe

run: main.exe
	./main.exe

clean:
	rm -f *.exe *.out

.PHONY: all test run clean noopt