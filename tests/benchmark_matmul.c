#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>

/* ---------- timing helpers ---------- */

#if defined(__linux__) || defined(__APPLE__)
static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
#else
static double get_time_sec(void) {
    return (double)clock() / CLOCKS_PER_SEC;
}
#endif

/* ---------- naive kernel (original i-j-p loop, always compiled) ---------- */

static void matmul_float_naive(const float *A, int64_t M, int64_t K,
                                const float *B, int64_t N,
                                float *C)
{
    for (int64_t i = 0; i < M; i++) {
        for (int64_t j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int64_t p = 0; p < K; p++) {
                sum += A[i * K + p] * B[p * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

/* ---------- optimized kernel ---------- */

#include "matmul_kernel.h"
#define HAVE_OPTIMIZED_KERNEL 1

/* ---------- helpers ---------- */

static void fill_random(float *data, int64_t n)
{
    for (int64_t i = 0; i < n; i++) {
        data[i] = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
    }
}

static int verify_results(const float *C_ref, const float *C_test,
                           int64_t M, int64_t N, float tol)
{
    for (int64_t i = 0; i < M * N; i++) {
        float diff = fabsf(C_ref[i] - C_test[i]);
        if (diff > tol) {
            fprintf(stderr, "  MISMATCH at [%ld]: ref=%f test=%f diff=%f\n",
                    (long)i, C_ref[i], C_test[i], diff);
            return 0;
        }
    }
    return 1;
}

/* ---------- benchmark runner ---------- */

typedef struct {
    double avg_ms;
    double min_ms;
    double max_ms;
    double gflops;
} bench_result;

static bench_result run_bench(void (*fn)(const float*, int64_t, int64_t,
                                          const float*, int64_t, float*),
                               const float *A, int64_t M, int64_t K,
                               const float *B, int64_t N, float *C,
                               int n_warmup, int n_runs)
{
    bench_result r = {0};
    double flops = 2.0 * M * N * K;

    /* warmup */
    for (int w = 0; w < n_warmup; w++) {
        fn(A, M, K, B, N, C);
    }

    double total = 0.0;
    r.min_ms = 1e30;
    r.max_ms = 0.0;

    for (int run = 0; run < n_runs; run++) {
        double t0 = get_time_sec();
        fn(A, M, K, B, N, C);
        double t1 = get_time_sec();
        double elapsed_ms = (t1 - t0) * 1000.0;
        total += elapsed_ms;
        if (elapsed_ms < r.min_ms) r.min_ms = elapsed_ms;
        if (elapsed_ms > r.max_ms) r.max_ms = elapsed_ms;
    }

    r.avg_ms = total / n_runs;
    r.gflops = flops / (r.min_ms / 1000.0) / 1e9;
    return r;
}

/* ---------- main ---------- */

int main(void)
{
    int sizes[][3] = {
        {4, 4, 4},
        {32, 32, 32},
        {64, 64, 64},
        {128, 128, 128},
        {256, 256, 256},
        {512, 512, 512},
        {1024, 1024, 1024},
    };
    int n_sizes = sizeof(sizes) / sizeof(sizes[0]);

    printf("================================================================\n");
    printf("  MatMul Benchmark (float)\n");
#ifdef HAVE_OPTIMIZED_KERNEL
    printf("  Optimized kernel: YES");
#ifdef CONNXR_USE_AVX2
    printf(" [AVX2+FMA]");
#elif defined(CONNXR_USE_SSE4)
    printf(" [SSE4.1]");
#else
    printf(" [scalar tiled]");
#endif
#ifdef CONNXR_USE_OMP
    printf(" [OpenMP]");
#endif
    printf("\n");
#else
    printf("  Optimized kernel: NO (naive-vs-naive baseline)\n");
#endif
    printf("================================================================\n");
    printf("%-12s | %-10s %-10s %-10s | %-10s %-10s %-10s | %-8s %-8s\n",
           "Size", "Naive avg", "min", "GFLOPS",
           "Opt avg", "min", "GFLOPS", "Speedup", "Verify");
    printf("%-12s-+-%10s-%10s-%10s-+-%10s-%10s-%10s-+-%8s-%8s\n",
           "------------", "----------", "----------", "----------",
           "----------", "----------", "----------", "--------", "--------");

    srand(42);

    for (int s = 0; s < n_sizes; s++) {
        int64_t M = sizes[s][0];
        int64_t K = sizes[s][1];
        int64_t N = sizes[s][2];

        float *A       = (float*)malloc(M * K * sizeof(float));
        float *B       = (float*)malloc(K * N * sizeof(float));
        float *C_naive = (float*)malloc(M * N * sizeof(float));
        float *C_opt   = (float*)malloc(M * N * sizeof(float));

        fill_random(A, M * K);
        fill_random(B, K * N);

        /* adaptive run count: more runs for small matrices */
        int n_runs = (M <= 64) ? 100 : (M <= 256) ? 20 : 5;
        int n_warmup = (M <= 64) ? 10 : 2;

        bench_result naive_r = run_bench(matmul_float_naive,
                                          A, M, K, B, N, C_naive,
                                          n_warmup, n_runs);

#ifdef HAVE_OPTIMIZED_KERNEL
        bench_result opt_r = run_bench(matmul_float,
                                        A, M, K, B, N, C_opt,
                                        n_warmup, n_runs);
#else
        bench_result opt_r = run_bench(matmul_float_optimized,
                                        A, M, K, B, N, C_opt,
                                        n_warmup, n_runs);
#endif

        /* Verify correctness */
        float tol = (M > 256) ? 0.01f : 0.001f;
        int ok = verify_results(C_naive, C_opt, M, N, tol);
        double speedup = naive_r.min_ms / opt_r.min_ms;

        char size_str[48];
        snprintf(size_str, sizeof(size_str), "%ldx%ldx%ld",
                 (long)M, (long)K, (long)N);

        printf("%-12s | %8.3f ms %8.3f ms %8.3f   | %8.3f ms %8.3f ms %8.3f   | %7.2fx  %s\n",
               size_str,
               naive_r.avg_ms, naive_r.min_ms, naive_r.gflops,
               opt_r.avg_ms, opt_r.min_ms, opt_r.gflops,
               speedup, ok ? "OK" : "FAIL");

        free(A);
        free(B);
        free(C_naive);
        free(C_opt);
    }

    printf("================================================================\n");
    return 0;
}
