#include "operator__ai_onnx__gather__1.h"
#include "tracing.h"
#include "utils.h"
#include <string.h>
#include <stdint.h>
#include <stdio.h>

#define GATHER_TYPED(TYPE, DATA_FIELD)                                       \
do {                                                                          \
    TYPE *src = i_data->DATA_FIELD;                                          \
    TYPE *dst = o_output->DATA_FIELD;                                        \
    if (!src || !dst) return OP_ENOSYS;                                      \
    int64_t out_pos = 0;                                                     \
    for (int64_t o = 0; o < outer; o++)                                      \
        for (int64_t idx = 0; idx < n_idx; idx++) {                          \
            int64_t g = get_index(i_indices, idx);                           \
            if (g < 0) g += axis_dim;                                        \
            if (g < 0 || g >= axis_dim) g = 0;                              \
            for (int64_t in = 0; in < inner; in++)                           \
                dst[out_pos++] = src[(o * axis_dim + g) * inner + in];       \
        }                                                                     \
} while(0)

static inline int64_t get_index(Onnx__TensorProto *t, int64_t i) {
    if (t->int64_data) return t->int64_data[i];
    if (t->int32_data) return t->int32_data[i];
    return 0;
}

operator_status
execute_operator__ai_onnx__gather__1__T_tensor_float(node_context *ctx)
{
    Onnx__TensorProto *i_data = searchInputByName(ctx, 0);
    Onnx__TensorProto *i_indices = searchInputByName(ctx, 1);
    Onnx__TensorProto *o_output = searchOutputByName(ctx, 0);
    if (!i_data || !i_indices || !o_output) return OP_EINVAL;
    if (!i_data->dims || i_data->n_dims == 0) return OP_EINVAL;

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

    switch (i_data->data_type) {
        case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
            GATHER_TYPED(float, float_data); break;
        case ONNX__TENSOR_PROTO__DATA_TYPE__INT32:
        case ONNX__TENSOR_PROTO__DATA_TYPE__BOOL:
            GATHER_TYPED(int32_t, int32_data); break;
        case ONNX__TENSOR_PROTO__DATA_TYPE__INT64:
            GATHER_TYPED(int64_t, int64_data); break;
        case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE:
            GATHER_TYPED(double, double_data); break;
        default:
            return OP_ENOSYS;
    }
    return OP_OK;
}
