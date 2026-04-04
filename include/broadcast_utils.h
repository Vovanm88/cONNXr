#ifndef BROADCAST_UTILS_H
#define BROADCAST_UTILS_H

#include <stdint.h>
#include <string.h>
#include <stdlib.h>

/*
 * Numpy-style broadcasting utilities for cONNXr operators.
 *
 * Given shapes A[n_a] and B[n_b], computes output shape and provides
 * index mapping from flat output index to flat input indices.
 */

#define BROADCAST_MAX_DIMS 8

typedef struct {
    int64_t n_dims;
    int64_t dims[BROADCAST_MAX_DIMS];
    int64_t strides_a[BROADCAST_MAX_DIMS];
    int64_t strides_b[BROADCAST_MAX_DIMS];
    int64_t strides_out[BROADCAST_MAX_DIMS];
    int64_t total;
} broadcast_ctx;

/* Compute broadcast shape and strides. Returns 0 on success, -1 on incompatible shapes. */
static inline int broadcast_init(broadcast_ctx *bc,
                                  int64_t n_a, const int64_t *dims_a,
                                  int64_t n_b, const int64_t *dims_b)
{
    bc->n_dims = (n_a > n_b) ? n_a : n_b;
    int64_t nd = bc->n_dims;

    /* Compute output dims and per-tensor strides */
    int64_t stride_a = 1, stride_b = 1, stride_out = 1;
    for (int64_t i = nd - 1; i >= 0; i--) {
        int64_t da = (i >= nd - n_a) ? dims_a[i - (nd - n_a)] : 1;
        int64_t db = (i >= nd - n_b) ? dims_b[i - (nd - n_b)] : 1;
        if (da != db && da != 1 && db != 1) return -1;
        bc->dims[i] = (da > db) ? da : db;
        bc->strides_a[i] = (da == 1) ? 0 : stride_a;
        bc->strides_b[i] = (db == 1) ? 0 : stride_b;
        bc->strides_out[i] = stride_out;
        stride_a *= da;
        stride_b *= db;
        stride_out *= bc->dims[i];
    }
    bc->total = stride_out;
    return 0;
}

/* Convert flat output index to flat A and B indices */
static inline void broadcast_indices(const broadcast_ctx *bc,
                                      int64_t out_idx,
                                      int64_t *a_idx, int64_t *b_idx)
{
    int64_t ai = 0, bi = 0;
    for (int64_t d = 0; d < bc->n_dims; d++) {
        int64_t coord = (out_idx / bc->strides_out[d]) % bc->dims[d];
        ai += coord * bc->strides_a[d];
        bi += coord * bc->strides_b[d];
    }
    *a_idx = ai;
    *b_idx = bi;
}

/* Ternary broadcast (for Where: condition, X, Y) */
typedef struct {
    int64_t n_dims;
    int64_t dims[BROADCAST_MAX_DIMS];
    int64_t strides_c[BROADCAST_MAX_DIMS];
    int64_t strides_x[BROADCAST_MAX_DIMS];
    int64_t strides_y[BROADCAST_MAX_DIMS];
    int64_t strides_out[BROADCAST_MAX_DIMS];
    int64_t total;
} broadcast3_ctx;

static inline int broadcast3_init(broadcast3_ctx *bc,
                                   int64_t nc, const int64_t *dc,
                                   int64_t nx, const int64_t *dx,
                                   int64_t ny, const int64_t *dy)
{
    int64_t nd = nc;
    if (nx > nd) nd = nx;
    if (ny > nd) nd = ny;
    bc->n_dims = nd;

    int64_t sc = 1, sx = 1, sy = 1, so = 1;
    for (int64_t i = nd - 1; i >= 0; i--) {
        int64_t dci = (i >= nd - nc) ? dc[i - (nd - nc)] : 1;
        int64_t dxi = (i >= nd - nx) ? dx[i - (nd - nx)] : 1;
        int64_t dyi = (i >= nd - ny) ? dy[i - (nd - ny)] : 1;
        int64_t m = dci; if (dxi > m) m = dxi; if (dyi > m) m = dyi;
        bc->dims[i] = m;
        bc->strides_c[i] = (dci == 1) ? 0 : sc;
        bc->strides_x[i] = (dxi == 1) ? 0 : sx;
        bc->strides_y[i] = (dyi == 1) ? 0 : sy;
        bc->strides_out[i] = so;
        sc *= dci; sx *= dxi; sy *= dyi; so *= m;
    }
    bc->total = so;
    return 0;
}

static inline void broadcast3_indices(const broadcast3_ctx *bc,
                                       int64_t out_idx,
                                       int64_t *c_idx, int64_t *x_idx, int64_t *y_idx)
{
    int64_t ci = 0, xi = 0, yi = 0;
    for (int64_t d = 0; d < bc->n_dims; d++) {
        int64_t coord = (out_idx / bc->strides_out[d]) % bc->dims[d];
        ci += coord * bc->strides_c[d];
        xi += coord * bc->strides_x[d];
        yi += coord * bc->strides_y[d];
    }
    *c_idx = ci; *x_idx = xi; *y_idx = yi;
}

/* Compute total number of elements from dims */
static inline int64_t shape_total(int64_t n_dims, const int64_t *dims)
{
    int64_t total = 1;
    for (int64_t i = 0; i < n_dims; i++) total *= dims[i];
    return total;
}

/* Allocate and compute broadcast output dims. Caller must free. */
static inline int64_t *broadcast_shape(int64_t n_a, const int64_t *da,
                                        int64_t n_b, const int64_t *db,
                                        int64_t *out_ndims)
{
    int64_t nd = (n_a > n_b) ? n_a : n_b;
    int64_t *out = (int64_t*)malloc(nd * sizeof(int64_t));
    for (int64_t i = nd - 1; i >= 0; i--) {
        int64_t a = (i >= nd - n_a) ? da[i - (nd - n_a)] : 1;
        int64_t b = (i >= nd - n_b) ? db[i - (nd - n_b)] : 1;
        out[i] = (a > b) ? a : b;
    }
    *out_ndims = nd;
    return out;
}

#endif /* BROADCAST_UTILS_H */
