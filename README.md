# Blocked Matrix Multiplication Optimizer

Compare single-threaded and OpenMP-parallelized blocked matrix multiplication using different block sizes.

## Build

```bash
make         # single-threaded
make omp     # OpenMP version (macOS-safe)
```

## Run
```bash
./blocked_matmul 1024              # run single-threaded with 1024x1024 matrix
./blocked_matmul_omp 1024 perf.csv # run OpenMP version + write results to CSV
```
## Output
```bash
Testing block sizes (matrix size: 1024 x 1024):
BlockSize	Time (ms)
-----------------------------
4         	343.50
8         	322.85
16        	305.89
32        	358.36
64        	570.44
128       	804.72
```
## CSV file (when option provide)
```bash
$ cat out.csv 
block_size,time_ms
4,343.50
8,322.85
16,305.89
32,358.36
64,570.44
128,804.72
```

## Requirements
macOS (with libomp: brew install libomp)
clang, make
