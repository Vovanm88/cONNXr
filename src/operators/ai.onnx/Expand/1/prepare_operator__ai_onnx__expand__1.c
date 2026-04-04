#include "operator__ai_onnx__expand__1.h"
#include "tracing.h"
#include "utils.h"
#include "broadcast_utils.h"

operator_status
prepare_operator__ai_onnx__expand__1(node_context *ctx)
{
    TRACE_ENTRY(1);
    TRACE_NODE(2, true, ctx->onnx_node);

    Onnx__TensorProto *i_input = searchInputByName(ctx, 0);
    Onnx__TensorProto *i_shape = searchInputByName(ctx, 1);
    Onnx__TensorProto *o_output = searchOutputByName(ctx, 0);
    o_output->has_raw_data = 0;
    o_output->data_type = i_input->data_type;
    /* Output shape is broadcast of input shape and target shape */
    int64_t nd;
    o_output->dims = broadcast_shape(i_input->n_dims, i_input->dims,
                                      i_shape->n_int64_data, i_shape->int64_data, &nd);
    o_output->n_dims = nd;
    mallocTensorData(o_output);
    ctx->executer = resolve_operator__ai_onnx__expand__1(ctx);

    TRACE_EXIT(1);
    return OP_OK;
}
