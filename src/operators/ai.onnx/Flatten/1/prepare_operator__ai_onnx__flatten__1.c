#include "operator__ai_onnx__flatten__1.h"
#include "tracing.h"
#include "utils.h"
#include <string.h>

operator_status
prepare_operator__ai_onnx__flatten__1(node_context *ctx)
{
    TRACE_ENTRY(1);
    TRACE_NODE(2, true, ctx->onnx_node);

    Onnx__TensorProto *i_input = searchInputByName(ctx, 0);
    Onnx__TensorProto *o_output = searchOutputByName(ctx, 0);
    Onnx__AttributeProto *a_axis = searchAttributeNyName(
        ctx->onnx_node->n_attribute, ctx->onnx_node->attribute, "axis");
    int64_t axis = a_axis ? a_axis->i : 1;
    if (axis < 0) axis += i_input->n_dims;
    int64_t outer = 1, inner = 1;
    for (int64_t i = 0; i < axis; i++) outer *= i_input->dims[i];
    for (int64_t i = axis; i < (int64_t)i_input->n_dims; i++) inner *= i_input->dims[i];
    o_output->has_raw_data = 0;
    o_output->data_type = i_input->data_type;
    o_output->n_dims = 2;
    o_output->dims = malloc(2 * sizeof(int64_t));
    o_output->dims[0] = outer;
    o_output->dims[1] = inner;
    mallocTensorData(o_output);
    ctx->executer = resolve_operator__ai_onnx__flatten__1(ctx);

    TRACE_EXIT(1);
    return OP_OK;
}
