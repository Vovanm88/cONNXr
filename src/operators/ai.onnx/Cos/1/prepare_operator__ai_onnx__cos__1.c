#include "operator__ai_onnx__cos__1.h"
#include "tracing.h"
#include "utils.h"
#include <math.h>

operator_status
prepare_operator__ai_onnx__cos__1(node_context *ctx)
{
    TRACE_ENTRY(1);
    TRACE_NODE(2, true, ctx->onnx_node);

    Onnx__TensorProto *i_X = searchInputByName(ctx, 0);
    Onnx__TensorProto *o_Y = searchOutputByName(ctx, 0);
    o_Y->has_raw_data = 0;
    o_Y->data_type = i_X->data_type;
    o_Y->n_dims = i_X->n_dims;
    o_Y->dims = ARRAYDUP(i_X->dims, i_X->n_dims);
    mallocTensorData(o_Y);
    ctx->executer = resolve_operator__ai_onnx__cos__1(ctx);

    TRACE_EXIT(1);
    return OP_OK;
}
