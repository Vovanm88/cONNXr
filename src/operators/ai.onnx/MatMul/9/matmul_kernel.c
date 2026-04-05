#include "matmul_kernel.h"
#include <string.h>

/* ---------- Tile sizes tuned for typical L1/L2 caches ---------- */
#define M_TILE 64
#define K_TILE 64
#define N_TILE 256

/* Below this threshold, skip tiling overhead */
#define SMALL_THRESHOLD 4096

/* ---------- SIMD headers ---------- */

#if defined(CONNXR_USE_AVX2)
  #include <immintrin.h>
#elif defined(CONNXR_USE_SSE4)
  #include <smmintrin.h>
#endif

/* ---------- OpenMP ---------- */

#ifdef CONNXR_USE_OMP
  #include <omp.h>
  
#endif
#define OMP_THRESHOLD 32768

/* ---------- BLAS ---------- */

#ifdef CONNXR_USE_BLAS
  #include <cblas.h>
#endif

/* ================================================================
 *  FLOAT kernel
 * ================================================================ */

#ifdef CONNXR_USE_BLAS

void matmul_float(const float *A, int64_t M, int64_t K,
                  const float *B, int64_t N, float *C)
{
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                (int)M, (int)N, (int)K, 1.0f, A, (int)K,
                B, (int)N, 0.0f, C, (int)N);
}

#else /* custom kernel */

/* SIMD micro-kernel for a tile: C[m x n] += A[m x k] * B[k x n] */
static void micro_float(const float *A, int64_t lda,
                         const float *B, int64_t ldb,
                         float *C, int64_t ldc,
                         int64_t m, int64_t k, int64_t n)
{
    for (int64_t i = 0; i < m; i++) {
        for (int64_t kk = 0; kk < k; kk++) {
            float a_ik = A[i * lda + kk];
            int64_t j = 0;

#if defined(CONNXR_USE_AVX2)
            __m256 va = _mm256_set1_ps(a_ik);
            for (; j + 8 <= n; j += 8) {
                __m256 vc = _mm256_loadu_ps(&C[i * ldc + j]);
                __m256 vb = _mm256_loadu_ps(&B[kk * ldb + j]);
                vc = _mm256_fmadd_ps(va, vb, vc);
                _mm256_storeu_ps(&C[i * ldc + j], vc);
            }
            /* SSE tail for 4-float chunks */
            __m128 va4 = _mm_set1_ps(a_ik);
            for (; j + 4 <= n; j += 4) {
                __m128 vc4 = _mm_loadu_ps(&C[i * ldc + j]);
                __m128 vb4 = _mm_loadu_ps(&B[kk * ldb + j]);
                vc4 = _mm_add_ps(vc4, _mm_mul_ps(va4, vb4));
                _mm_storeu_ps(&C[i * ldc + j], vc4);
            }
#elif defined(CONNXR_USE_SSE4)
            __m128 va = _mm_set1_ps(a_ik);
            for (; j + 4 <= n; j += 4) {
                __m128 vc = _mm_loadu_ps(&C[i * ldc + j]);
                __m128 vb = _mm_loadu_ps(&B[kk * ldb + j]);
                vc = _mm_add_ps(vc, _mm_mul_ps(va, vb));
                _mm_storeu_ps(&C[i * ldc + j], vc);
            }
#endif
            /* scalar tail */
            for (; j < n; j++) {
                C[i * ldc + j] += a_ik * B[kk * ldb + j];
            }
        }
    }
}

void matmul_float(const float *A, int64_t M, int64_t K,
                  const float *B, int64_t N, float *C)
{
    if (!A || !B || !C || M <= 0 || K <= 0 || N <= 0 || (uintptr_t)A < 0x1000 || (uintptr_t)B < 0x1000) {
        if (C && M > 0 && N > 0) memset(C, 0, (size_t)(M * N) * sizeof(float));
        return;
    }
    memset(C, 0, (size_t)(M * N) * sizeof(float));

    /* Small matrix fast path: simple i-k-j without tiling */
    if (M * K * N < SMALL_THRESHOLD) {
        for (int64_t i = 0; i < M; i++) {
            for (int64_t k = 0; k < K; k++) {
                float a_ik = A[i * K + k];
                for (int64_t j = 0; j < N; j++) {
                    C[i * N + j] += a_ik * B[k * N + j];
                }
            }
        }
        return;
    }

    /* Tiled i-k-j loop */
#ifdef CONNXR_USE_OMP
    int omp_enabled = (int)(M * K * N > OMP_THRESHOLD);
    #pragma omp parallel for if (omp_enabled) schedule(dynamic, 1)
#endif
    for (int64_t i0 = 0; i0 < M; i0 += M_TILE) {
        int64_t i_end = (i0 + M_TILE < M) ? i0 + M_TILE : M;
        for (int64_t k0 = 0; k0 < K; k0 += K_TILE) {
            int64_t k_end = (k0 + K_TILE < K) ? k0 + K_TILE : K;
            for (int64_t j0 = 0; j0 < N; j0 += N_TILE) {
                int64_t j_end = (j0 + N_TILE < N) ? j0 + N_TILE : N;
                micro_float(A + i0 * K + k0, K,
                            B + k0 * N + j0, N,
                            C + i0 * N + j0, N,
                            i_end - i0, k_end - k0, j_end - j0);
            }
        }
    }
}

#endif /* CONNXR_USE_BLAS */

/* ================================================================
 *  DOUBLE kernel
 * ================================================================ */

#ifdef CONNXR_USE_BLAS

void matmul_double(const double *A, int64_t M, int64_t K,
                   const double *B, int64_t N, double *C)
{
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                (int)M, (int)N, (int)K, 1.0, A, (int)K,
                B, (int)N, 0.0, C, (int)N);
}

#else

static void micro_double(const double *A, int64_t lda,
                          const double *B, int64_t ldb,
                          double *C, int64_t ldc,
                          int64_t m, int64_t k, int64_t n)
{
    for (int64_t i = 0; i < m; i++) {
        for (int64_t kk = 0; kk < k; kk++) {
            double a_ik = A[i * lda + kk];
            int64_t j = 0;

#if defined(CONNXR_USE_AVX2)
            __m256d va = _mm256_set1_pd(a_ik);
            for (; j + 4 <= n; j += 4) {
                __m256d vc = _mm256_loadu_pd(&C[i * ldc + j]);
                __m256d vb = _mm256_loadu_pd(&B[kk * ldb + j]);
                vc = _mm256_fmadd_pd(va, vb, vc);
                _mm256_storeu_pd(&C[i * ldc + j], vc);
            }
            __m128d va2 = _mm_set1_pd(a_ik);
            for (; j + 2 <= n; j += 2) {
                __m128d vc2 = _mm_loadu_pd(&C[i * ldc + j]);
                __m128d vb2 = _mm_loadu_pd(&B[kk * ldb + j]);
                vc2 = _mm_add_pd(vc2, _mm_mul_pd(va2, vb2));
                _mm_storeu_pd(&C[i * ldc + j], vc2);
            }
#elif defined(CONNXR_USE_SSE4)
            __m128d va = _mm_set1_pd(a_ik);
            for (; j + 2 <= n; j += 2) {
                __m128d vc = _mm_loadu_pd(&C[i * ldc + j]);
                __m128d vb = _mm_loadu_pd(&B[kk * ldb + j]);
                vc = _mm_add_pd(vc, _mm_mul_pd(va, vb));
                _mm_storeu_pd(&C[i * ldc + j], vc);
            }
#endif
            for (; j < n; j++) {
                C[i * ldc + j] += a_ik * B[kk * ldb + j];
            }
        }
    }
}

void matmul_double(const double *A, int64_t M, int64_t K,
                   const double *B, int64_t N, double *C)
{
    memset(C, 0, (size_t)(M * N) * sizeof(double));

    if (M * K * N < SMALL_THRESHOLD) {
        for (int64_t i = 0; i < M; i++) {
            for (int64_t k = 0; k < K; k++) {
                double a_ik = A[i * K + k];
                for (int64_t j = 0; j < N; j++) {
                    C[i * N + j] += a_ik * B[k * N + j];
                }
            }
        }
        return;
    }

#ifdef CONNXR_USE_OMP
    int omp_enabled = (int)(M * K * N > OMP_THRESHOLD);
    #pragma omp parallel for if (omp_enabled) schedule(dynamic, 1)
#endif
    for (int64_t i0 = 0; i0 < M; i0 += M_TILE) {
        int64_t i_end = (i0 + M_TILE < M) ? i0 + M_TILE : M;
        for (int64_t k0 = 0; k0 < K; k0 += K_TILE) {
            int64_t k_end = (k0 + K_TILE < K) ? k0 + K_TILE : K;
            for (int64_t j0 = 0; j0 < N; j0 += N_TILE) {
                int64_t j_end = (j0 + N_TILE < N) ? j0 + N_TILE : N;
                micro_double(A + i0 * K + k0, K,
                             B + k0 * N + j0, N,
                             C + i0 * N + j0, N,
                             i_end - i0, k_end - k0, j_end - j0);
            }
        }
    }
}

#endif /* CONNXR_USE_BLAS */

/* ================================================================
 *  INTEGER kernels (scalar tiled, no SIMD)
 * ================================================================ */

#define DEFINE_INT_MATMUL(TYPE, NAME)                                     \
void NAME(const TYPE *A, int64_t M, int64_t K,                           \
          const TYPE *B, int64_t N, TYPE *C)                              \
{                                                                         \
    memset(C, 0, (size_t)(M * N) * sizeof(TYPE));                        \
    if (M * K * N < SMALL_THRESHOLD) {                                   \
        for (int64_t i = 0; i < M; i++)                                  \
            for (int64_t k = 0; k < K; k++) {                            \
                TYPE a_ik = A[i * K + k];                                \
                for (int64_t j = 0; j < N; j++)                          \
                    C[i * N + j] += a_ik * B[k * N + j];                \
            }                                                             \
        return;                                                           \
    }                                                                     \
    for (int64_t i0 = 0; i0 < M; i0 += M_TILE) {                       \
        int64_t ie = (i0 + M_TILE < M) ? i0 + M_TILE : M;              \
        for (int64_t k0 = 0; k0 < K; k0 += K_TILE) {                   \
            int64_t ke = (k0 + K_TILE < K) ? k0 + K_TILE : K;          \
            for (int64_t j0 = 0; j0 < N; j0 += N_TILE) {               \
                int64_t je = (j0 + N_TILE < N) ? j0 + N_TILE : N;      \
                for (int64_t i = i0; i < ie; i++)                        \
                    for (int64_t k = k0; k < ke; k++) {                  \
                        TYPE a_ik = A[i * K + k];                        \
                        for (int64_t j = j0; j < je; j++)                \
                            C[i * N + j] += a_ik * B[k * N + j];        \
                    }                                                     \
            }                                                             \
        }                                                                 \
    }                                                                     \
}

DEFINE_INT_MATMUL(int32_t,  matmul_int32)
DEFINE_INT_MATMUL(int64_t,  matmul_int64)
DEFINE_INT_MATMUL(uint64_t, matmul_uint64)
