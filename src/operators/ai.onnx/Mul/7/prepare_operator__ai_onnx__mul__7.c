#include "operator__ai_onnx__mul__7.h"
#include "tracing.h"
#include "utils.h"
#include "broadcast_utils.h"

operator_status
prepare_operator__ai_onnx__mul__7(
    node_context *ctx
)
{
    TRACE_ENTRY(1);
    TRACE_NODE(2, true, ctx->onnx_node);

    Onnx__TensorProto *i_A = searchInputByName(ctx, 0);
    Onnx__TensorProto *i_B = searchInputByName(ctx, 1);

    TRACE_TENSOR(2, true, i_A);
    TRACE_TENSOR(2, true, i_B);

    Onnx__TensorProto *o_C = searchOutputByName(ctx, 0);

    o_C->has_raw_data = 0;
    o_C->data_type = i_A->data_type;
    int64_t nd;
    o_C->dims = broadcast_shape(i_A->n_dims, i_A->dims, i_B->n_dims, i_B->dims, &nd);
    o_C->n_dims = nd;

    mallocTensorData(o_C);

    TRACE_TENSOR(2, true, o_C);

    ctx->executer = resolve_operator__ai_onnx__mul__7(ctx);

    TRACE_EXIT(1);
    return OP_OK;
}
