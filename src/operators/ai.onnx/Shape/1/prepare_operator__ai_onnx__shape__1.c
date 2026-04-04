#include "operator__ai_onnx__shape__1.h"
#include "tracing.h"
#include "utils.h"

operator_status
prepare_operator__ai_onnx__shape__1(node_context *ctx)
{
    TRACE_ENTRY(1);
    TRACE_NODE(2, true, ctx->onnx_node);

    Onnx__TensorProto *i_data = searchInputByName(ctx, 0);
    Onnx__TensorProto *o_shape = searchOutputByName(ctx, 0);
    o_shape->has_raw_data = 0;
    o_shape->data_type = ONNX__TENSOR_PROTO__DATA_TYPE__INT64;
    o_shape->n_dims = 1;
    o_shape->dims = malloc(sizeof(int64_t));
    o_shape->dims[0] = i_data->n_dims;
    mallocTensorData(o_shape);
    /* Fill immediately - Shape is a metadata op */
    for (int64_t i = 0; i < (int64_t)i_data->n_dims; i++) {
        o_shape->int64_data[i] = i_data->dims[i];
    }
    ctx->executer = resolve_operator__ai_onnx__shape__1(ctx);

    TRACE_EXIT(1);
    return OP_OK;
}
