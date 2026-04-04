#include "operator__ai_onnx__isnan__1.h"
#include "tracing.h"
#include "utils.h"
#include <math.h>
#include <string.h>
#include <stdint.h>
operator_status
execute_operator__ai_onnx__isnan__1__T_tensor_float(node_context *ctx)
{
    TRACE_ENTRY(1);
    TRACE_NODE(2, true, ctx->onnx_node);

    Onnx__TensorProto *i_X = searchInputByName(ctx, 0);
    Onnx__TensorProto *o_Y = searchOutputByName(ctx, 0);
    for (int64_t i = 0; i < (int64_t)i_X->n_float_data; i++) {
        o_Y->int32_data[i] = isnan(i_X->float_data[i]) ? 1 : 0;
    }

    TRACE_EXIT(1);
    return OP_OK;
}
