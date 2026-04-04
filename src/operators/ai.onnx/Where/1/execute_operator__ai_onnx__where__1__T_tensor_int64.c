#include "operator__ai_onnx__where__1.h"
#include "tracing.h"
#include "utils.h"
#include "broadcast_utils.h"
#include <string.h>
#include <stdint.h>
#include "op_utils.h"
operator_status
execute_operator__ai_onnx__where__1__T_tensor_int64(node_context *ctx)
{
    TRACE_ENTRY(1);
    TRACE_NODE(2, true, ctx->onnx_node);

    Onnx__TensorProto *i_C = searchInputByName(ctx, 0);
    Onnx__TensorProto *i_X = searchInputByName(ctx, 1);
    Onnx__TensorProto *i_Y = searchInputByName(ctx, 2);
    Onnx__TensorProto *o_O = searchOutputByName(ctx, 0);
    /* Skip if output tensor is empty (has 0-dim) */
    if (tensor_is_empty(o_O)) return OP_OK;

    broadcast3_ctx bc;
    broadcast3_init(&bc, i_C->n_dims, i_C->dims, i_X->n_dims, i_X->dims, i_Y->n_dims, i_Y->dims);
    for (int64_t i = 0; i < bc.total; i++) {
        int64_t ci, xi, yi;
        broadcast3_indices(&bc, i, &ci, &xi, &yi);
        o_O->int64_data[i] = i_C->int32_data[ci] ? i_X->int64_data[xi] : i_Y->int64_data[yi];
    }

    TRACE_EXIT(1);
    return OP_OK;
}
