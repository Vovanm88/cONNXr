#include "operator__ai_onnx__where__1.h"
#include "tracing.h"
#include "utils.h"
#include "broadcast_utils.h"
#include <string.h>
#include <stdint.h>
operator_status
execute_operator__ai_onnx__where__1__T_tensor_float(node_context *ctx)
{
    TRACE_ENTRY(1);
    TRACE_NODE(2, true, ctx->onnx_node);

    Onnx__TensorProto *i_C = searchInputByName(ctx, 0);
    Onnx__TensorProto *i_X = searchInputByName(ctx, 1);
    Onnx__TensorProto *i_Y = searchInputByName(ctx, 2);
    Onnx__TensorProto *o_O = searchOutputByName(ctx, 0);
    broadcast3_ctx bc;
    broadcast3_init(&bc, i_C->n_dims, i_C->dims, i_X->n_dims, i_X->dims, i_Y->n_dims, i_Y->dims);
    for (int64_t i = 0; i < bc.total; i++) {
        int64_t ci, xi, yi;
        broadcast3_indices(&bc, i, &ci, &xi, &yi);
        o_O->float_data[i] = i_C->int32_data[ci] ? i_X->float_data[xi] : i_Y->float_data[yi];
    }

    TRACE_EXIT(1);
    return OP_OK;
}
