#include "operator__ai_onnx__gather__1.h"
#include "tracing.h"
#include "utils.h"
#include <stdint.h>
#include "op_utils.h"
operator_status execute_operator__ai_onnx__gather__1__T_tensor_int64(node_context *ctx) {
    Onnx__TensorProto *i_data = searchInputByName(ctx, 0);
    Onnx__TensorProto *i_indices = searchInputByName(ctx, 1);
    Onnx__TensorProto *o_output = searchOutputByName(ctx, 0);
    /* Skip if output tensor is empty (has 0-dim) */
    if (tensor_is_empty(o_output)) return OP_OK;

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
    if (i_data->has_raw_data) convertRawDataOfTensorProto(i_data);
    if (i_indices->has_raw_data) convertRawDataOfTensorProto(i_indices);
    if (!i_data->int64_data) return OP_ENOSYS;
    int64_t out_pos = 0;
    for (int64_t o = 0; o < outer; o++) {
        for (int64_t idx = 0; idx < n_idx; idx++) {
            int64_t g;
            if (i_indices->int64_data) g = i_indices->int64_data[idx];
            else if (i_indices->int32_data) g = i_indices->int32_data[idx];
            else g = 0;
            if (g < 0) g += axis_dim;
            for (int64_t in = 0; in < inner; in++) {
                o_output->int64_data[out_pos++] = i_data->int64_data[(o * axis_dim + g) * inner + in];
            }
        }
    }
    return OP_OK;
}
