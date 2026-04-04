#include "operator__ai_onnx__gather__1.h"
#include "tracing.h"
#include "utils.h"

operator_status
prepare_operator__ai_onnx__gather__1(node_context *ctx)
{
    TRACE_ENTRY(1);
    TRACE_NODE(2, true, ctx->onnx_node);

    Onnx__TensorProto *i_data = searchInputByName(ctx, 0);
    Onnx__TensorProto *i_indices = searchInputByName(ctx, 1);
    Onnx__TensorProto *o_output = searchOutputByName(ctx, 0);
    Onnx__AttributeProto *a_axis = searchAttributeNyName(
        ctx->onnx_node->n_attribute, ctx->onnx_node->attribute, "axis");
    int64_t axis = a_axis ? a_axis->i : 0;
    if (axis < 0) axis += i_data->n_dims;
    /* output shape: data.shape[:axis] + indices.shape + data.shape[axis+1:] */
    int64_t out_ndims = i_data->n_dims - 1 + i_indices->n_dims;
    o_output->has_raw_data = 0;
    o_output->data_type = i_data->data_type;
    o_output->n_dims = out_ndims;
    o_output->dims = malloc(out_ndims * sizeof(int64_t));
    int64_t d = 0;
    for (int64_t i = 0; i < axis; i++) o_output->dims[d++] = i_data->dims[i];
    for (int64_t i = 0; i < (int64_t)i_indices->n_dims; i++) o_output->dims[d++] = i_indices->dims[i];
    for (int64_t i = axis + 1; i < (int64_t)i_data->n_dims; i++) o_output->dims[d++] = i_data->dims[i];
    mallocTensorData(o_output);
    ctx->executer = resolve_operator__ai_onnx__gather__1(ctx);

    TRACE_EXIT(1);
    return OP_OK;
}
