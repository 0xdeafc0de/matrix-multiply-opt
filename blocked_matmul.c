#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#ifdef USE_OPENMP
#include <omp.h>
#endif

/*
 * Logic to Choose Max Block Size (e.g. with L1 data cache size of 32 KB)
 *   Each float is 4 bytes, so:
 *     * 1 block of size B x B = B^2 * 4 bytes
 *     * 3 blocks = 3 * B^2 * 4 <= 32 * 1024 bytes
 *     * B <= sqrt((32*1024)/(3*4))
 *     * B <= 52
 *
 */

#define MAX_BLOCK 128     // Max block size to test
#define MIN_BLOCK 4       // Starting block size

float** allocate_matrix(int n) {
    float** mat = (float**)malloc(n * sizeof(float*));
    for (int i = 0; i < n; i++) {
        mat[i] = (float*)calloc(n, sizeof(float));  // zero-initialize
    }
    return mat;
}

void free_matrix(float** mat, int n) {
    for (int i = 0; i < n; i++) {
        free(mat[i]);
    }
    free(mat);
}

void fill_random(float** mat, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            mat[i][j] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;  // [-1, 1]
        }
    }
}

void zero_matrix(float** mat, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            mat[i][j] = 0.0f;
}

long time_diff_ns(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);
}

// Blocked matrix multiplication: C = A * B
void blocked_matrix_multiply(float** A, float** B, float** C, int n, int block_size) {
#ifdef USE_OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int block_row = 0; block_row < n; block_row += block_size) {
        for (int block_col = 0; block_col < n; block_col += block_size) {
            for (int block_k = 0; block_k < n; block_k += block_size) {
                // Multiply block A[block_row][block_k] with B[block_k][block_col] and add to C[block_row][block_col]
                for (int row = block_row; row < block_row + block_size && row < n; row++) {
                    for (int col = block_col; col < block_col + block_size && col < n; col++) {
                        float sum = C[row][col];
                        for (int k_index = block_k; k_index < block_k + block_size && k_index < n; k_index++) {
                            sum += A[row][k_index] * B[k_index][col];
                        }
                        C[row][col] = sum;
                    }
                }
            }
        }
    }
}

void print_usage(const char* prog) {
    printf("Usage: %s <matrix_size> [csv_output_file]\n", prog);
    printf("Example: %s 1024 perf.csv\n", prog);
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }
    int n = atoi(argv[1]);
    const char* csv_file = argc >= 3 ? argv[2] : NULL;

    FILE* csv = NULL;
    if (csv_file) {
        csv = fopen(csv_file, "w");
        if (!csv) {
            perror("Error opening CSV file");
            return 1;
        }
        fprintf(csv, "block_size,time_ms\n");
    }

    srand((unsigned int)time(NULL));

    float** A = allocate_matrix(n);
    float** B = allocate_matrix(n);
    float** C = allocate_matrix(n);

    fill_random(A, n);
    fill_random(B, n);

#ifdef USE_OPENMP
    printf("Using %d OpenMP threads\n", omp_get_max_threads());
#endif
    printf("Testing block sizes (matrix size: %d x %d):\n", n, n);
    printf("BlockSize\tTime (ms)\n");
    printf("-----------------------------\n");

    for (int block_size = MIN_BLOCK; block_size <= MAX_BLOCK; block_size *= 2) {
        zero_matrix(C, n);
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        blocked_matrix_multiply(A, B, C, n, block_size);

        clock_gettime(CLOCK_MONOTONIC, &end);
        long elapsed_ns = time_diff_ns(start, end);
        double time_ms = elapsed_ns / 1e6;

        printf("%-10d\t%.2f\n", block_size, time_ms);
        if (csv) {
            fprintf(csv, "%d,%.2f\n", block_size, time_ms);
        }
    }

    if (csv) {
        fclose(csv);
    }

    free_matrix(A, n);
    free_matrix(B, n);
    free_matrix(C, n);

    return 0;
}

