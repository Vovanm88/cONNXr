#ifndef MATMUL_KERNEL_H
#define MATMUL_KERNEL_H

#include <stddef.h>
#include <stdint.h>

/*
 * Optimized matrix multiplication kernels for cONNXr.
 *
 * C[M x N] = A[M x K] * B[K x N],  all row-major.
 *
 * Optimizations (compile-time selectable):
 *   - Cache-friendly i-k-j loop order with tiling
 *   - SSE4.1 vectorization   (define CONNXR_USE_SSE4)
 *   - AVX2 + FMA             (define CONNXR_USE_AVX2)
 *   - OpenMP parallelism     (define CONNXR_USE_OMP)
 *   - BLAS delegation        (define CONNXR_USE_BLAS)
 */

/* float: C = A * B */
void matmul_float(const float *A, int64_t M, int64_t K,
                  const float *B, int64_t N,
                  float *C);

/* double: C = A * B */
void matmul_double(const double *A, int64_t M, int64_t K,
                   const double *B, int64_t N,
                   double *C);

/* int32: C = A * B */
void matmul_int32(const int32_t *A, int64_t M, int64_t K,
                  const int32_t *B, int64_t N,
                  int32_t *C);

/* int64: C = A * B */
void matmul_int64(const int64_t *A, int64_t M, int64_t K,
                  const int64_t *B, int64_t N,
                  int64_t *C);

/* uint64: C = A * B */
void matmul_uint64(const uint64_t *A, int64_t M, int64_t K,
                   const uint64_t *B, int64_t N,
                   uint64_t *C);

#endif /* MATMUL_KERNEL_H */
