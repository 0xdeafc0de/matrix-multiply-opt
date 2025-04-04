# Blocked Matrix Multiplication Optimizer

A cache-efficient, blocked matrix multiplication algorithm in C, with optional OpenMP parallelization.
It allows benchmarking across different block sizes to analyze performance impacts based on CPU cache behavior.

Compare single-threaded and OpenMP-parallelized blocked matrix multiplication using different block sizes.
## Build

```bash
make         # single-threaded
make omp     # OpenMP version (macOS-safe)
```

## Run
```bash
./blocked_matmul 1024              # run single-threaded with 1024x1024 matrix 
./blocked_matmul 1024  perf.csv    # run single-threaded with 1024x1024 matrix + write results to CSV
./blocked_matmul_omp 1024 perf.csv # run OpenMP version + write results to CSV
```
## Output
```bash
~/matrix-multiply-opt > ./blocked_matmul 1024
Testing block sizes (matrix size: 1024 x 1024):
BlockSize	Time (ms)
-----------------------------
4         	345.65
8         	308.12
16        	306.62
32        	355.95
64        	560.61
128       	812.95
~/matrix-multiply-opt >

~/matrix-multiply-opt > ./blocked_matmul_omp 1024
Using 12 OpenMP threads
Testing block sizes (matrix size: 1024 x 1024):
BlockSize	Time (ms)
-----------------------------
4         	64.14
8         	44.13
16        	47.04
32        	53.34
64        	87.58
128       	122.82
~/matrix-multiply-opt >

```
## CSV file (when option provide)
```bash
$ cat out.csv 
block_size,time_ms
4,345.65
8,308.12
16,306.62
32,355.95
64,560.61
128,812.95
```

## Requirements
macOS (with libomp: brew install libomp)
clang, make
