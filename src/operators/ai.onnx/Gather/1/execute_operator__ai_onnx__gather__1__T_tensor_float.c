#include "operator__ai_onnx__gather__1.h"
#include "tracing.h"
#include "utils.h"
#include <string.h>
#include <stdint.h>
operator_status
execute_operator__ai_onnx__gather__1__T_tensor_float(node_context *ctx)
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
    int64_t axis_dim = i_data->dims[axis];
    int64_t outer = 1, inner = 1;
    for (int64_t i = 0; i < axis; i++) outer *= i_data->dims[i];
    for (int64_t i = axis + 1; i < (int64_t)i_data->n_dims; i++) inner *= i_data->dims[i];
    int64_t n_idx = 1;
    for (int64_t i = 0; i < (int64_t)i_indices->n_dims; i++) n_idx *= i_indices->dims[i];
    int64_t out_pos = 0;
    for (int64_t o = 0; o < outer; o++) {
        for (int64_t idx = 0; idx < n_idx; idx++) {
            int64_t g = i_indices->int64_data ? i_indices->int64_data[idx] : i_indices->int32_data[idx];
            if (g < 0) g += axis_dim;
            for (int64_t in = 0; in < inner; in++) {
                o_output->float_data[out_pos++] = i_data->float_data[(o * axis_dim + g) * inner + in];
            }
        }
    }

    TRACE_EXIT(1);
    return OP_OK;
}
