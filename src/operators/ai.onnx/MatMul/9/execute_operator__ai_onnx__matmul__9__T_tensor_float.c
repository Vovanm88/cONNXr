#include "operator__ai_onnx__matmul__9.h"
#include "matmul_kernel.h"
#include "tracing.h"
#include "utils.h"
#include <string.h>
#include <stdint.h>

operator_status
execute_operator__ai_onnx__matmul__9__T_tensor_float(
    node_context *ctx
)
{
    TRACE_ENTRY(1);
    TRACE_NODE(2, true, ctx->onnx_node);

    Onnx__TensorProto *i_A = searchInputByName(ctx, 0);
    Onnx__TensorProto *i_B = searchInputByName(ctx, 1);
    Onnx__TensorProto *o_Y = searchOutputByName(ctx, 0);

    TRACE_TENSOR(2, true, i_A);
    TRACE_TENSOR(2, true, i_B);
    TRACE_TENSOR(2, true, o_Y);

    /* Effective shapes (promote 1D to 2D) */
    int64_t ndA = i_A->n_dims, ndB = i_B->n_dims;
    int64_t M, K_a, K_b, N;

    if (ndA == 1) { M = 1; K_a = i_A->dims[0]; }
    else { M = i_A->dims[ndA - 2]; K_a = i_A->dims[ndA - 1]; }

    if (ndB == 1) { K_b = i_B->dims[0]; N = 1; }
    else { K_b = i_B->dims[ndB - 2]; N = i_B->dims[ndB - 1]; }

    int64_t K = K_a;

    /* Simple 2D case (most common, uses optimized kernel) */
    if (ndA <= 2 && ndB <= 2) {
        matmul_float(i_A->float_data, M, K, i_B->float_data, N, o_Y->float_data);
        TRACE_EXIT(1);
        return OP_OK;
    }

    /* N-D batched matmul with broadcasting */
    int64_t max_nd = (ndA > ndB) ? ndA : ndB;
    int64_t batch_nd = max_nd - 2;

    /* Compute batch dimensions and total batch size */
    int64_t batch_dims[batch_nd > 0 ? batch_nd : 1];
    int64_t batch_total = 1;
    for (int64_t i = 0; i < batch_nd; i++) {
        int64_t da = (i < ndA - 2) ? i_A->dims[i + (ndA - max_nd)] : 1;
        int64_t db = (i < ndB - 2) ? i_B->dims[i + (ndB - max_nd)] : 1;
        batch_dims[i] = (da > db) ? da : db;
        batch_total *= batch_dims[i];
    }

    /* Compute strides for batch dims */
    int64_t stride_a[batch_nd > 0 ? batch_nd : 1];
    int64_t stride_b[batch_nd > 0 ? batch_nd : 1];
    int64_t stride_o[batch_nd > 0 ? batch_nd : 1];
    int64_t sa = M * K, sb = K * N, so = M * N;

    for (int64_t i = batch_nd - 1; i >= 0; i--) {
        int64_t da = (i < ndA - 2) ? i_A->dims[i + (ndA - max_nd)] : 1;
        int64_t db = (i < ndB - 2) ? i_B->dims[i + (ndB - max_nd)] : 1;
        stride_a[i] = (da > 1) ? sa : 0;
        stride_b[i] = (db > 1) ? sb : 0;
        stride_o[i] = so;
        sa *= da; sb *= db; so *= batch_dims[i];
    }

    /* Run batched matmuls */
    for (int64_t batch = 0; batch < batch_total; batch++) {
        int64_t a_off = 0, b_off = 0, o_off = 0;
        int64_t rem = batch;
        for (int64_t d = 0; d < batch_nd; d++) {
            int64_t coord = rem / (batch_total / batch_dims[d]);
            /* recompute properly */
            int64_t sz = 1;
            for (int64_t dd = d + 1; dd < batch_nd; dd++) sz *= batch_dims[dd];
            coord = (rem / sz) % batch_dims[d];
            a_off += coord * stride_a[d];
            b_off += coord * stride_b[d];
            o_off += coord * stride_o[d];
        }
        matmul_float(i_A->float_data + a_off, M, K,
                     i_B->float_data + b_off, N,
                     o_Y->float_data + o_off);
    }

    TRACE_EXIT(1);
    return OP_OK;
}