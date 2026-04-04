#include "operator__ai_onnx__where__1.h"
#include "tracing.h"
#include "utils.h"
#include "broadcast_utils.h"

operator_status
prepare_operator__ai_onnx__where__1(node_context *ctx)
{
    TRACE_ENTRY(1);
    TRACE_NODE(2, true, ctx->onnx_node);

    Onnx__TensorProto *i_C = searchInputByName(ctx, 0);
    Onnx__TensorProto *i_X = searchInputByName(ctx, 1);
    Onnx__TensorProto *i_Y = searchInputByName(ctx, 2);
    Onnx__TensorProto *o_O = searchOutputByName(ctx, 0);
    o_O->has_raw_data = 0;
    o_O->data_type = i_X->data_type;
    /* broadcast all 3 shapes: first broadcast C with X, then result with Y */
    int64_t nd1;
    int64_t *s1 = broadcast_shape(i_C->n_dims, i_C->dims, i_X->n_dims, i_X->dims, &nd1);
    int64_t nd2;
    o_O->dims = broadcast_shape(nd1, s1, i_Y->n_dims, i_Y->dims, &nd2);
    o_O->n_dims = nd2;
    free(s1);
    mallocTensorData(o_O);
    ctx->executer = resolve_operator__ai_onnx__where__1(ctx);

    TRACE_EXIT(1);
    return OP_OK;
}
