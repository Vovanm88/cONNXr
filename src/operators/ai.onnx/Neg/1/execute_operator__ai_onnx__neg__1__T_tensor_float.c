#include "operator__ai_onnx__neg__1.h"
#include "tracing.h"
#include "utils.h"
#include <string.h>
#include <stdint.h>
#include "op_utils.h"
operator_status
execute_operator__ai_onnx__neg__1__T_tensor_float(node_context *ctx)
{
    TRACE_ENTRY(1);
    TRACE_NODE(2, true, ctx->onnx_node);

    Onnx__TensorProto *i_X = searchInputByName(ctx, 0);
    if (!i_X || tensor_is_empty(i_X)) return OP_OK;
    Onnx__TensorProto *o_Y = searchOutputByName(ctx, 0);
    if (!i_X->float_data || !o_Y->float_data) return OP_OK;
    /* Skip if output tensor is empty (has 0-dim) */
    if (tensor_is_empty(o_Y)) return OP_OK;

    for (int64_t i = 0; i < (int64_t)o_Y->n_float_data; i++) {
        float x = i_X->float_data[i];
        o_Y->float_data[i] = (-x);
    }

    TRACE_EXIT(1);
    return OP_OK;
}
