#include "operator__ai_onnx__unsqueeze__1.h"
#include "tracing.h"
#include "utils.h"
#include <string.h>
#include <stdint.h>
#include "op_utils.h"

operator_status
execute_operator__ai_onnx__unsqueeze__1__T_tensor_float(node_context *ctx)
{
    Onnx__TensorProto *i_data = searchInputByName(ctx, 0);
    Onnx__TensorProto *o_expanded = searchOutputByName(ctx, 0);
    if (!i_data || !o_expanded) return OP_EINVAL;
    if (tensor_is_empty(o_expanded) || tensor_is_empty(i_data)) return OP_OK;

    // Получаем axes, как в prepare
    Onnx__AttributeProto *a_axes = searchAttributeNyName(
        ctx->onnx_node->n_attribute, ctx->onnx_node->attribute, "axes");
    int64_t *axes = NULL;
    int64_t n_axes = 0;
    Onnx__TensorProto *i_axes = NULL;
    if (a_axes) {
        axes = a_axes->ints;
        n_axes = a_axes->n_ints;
    } else {
        i_axes = searchInputByName(ctx, 1);
        if (i_axes) { axes = i_axes->int64_data; n_axes = i_axes->n_int64_data; }
    }
    int64_t out_ndims = i_data->n_dims + n_axes;

    // Нормализуем negative axes
    int64_t *norm_axes = malloc(n_axes * sizeof(int64_t));
    for (int64_t i = 0; i < n_axes; i++) {
        norm_axes[i] = axes[i] < 0 ? axes[i] + out_ndims : axes[i];
    }
    // Сортируем axes
    for (int64_t i = 0; i < n_axes - 1; i++)
        for (int64_t j = i + 1; j < n_axes; j++)
            if (norm_axes[i] > norm_axes[j]) { int64_t t = norm_axes[i]; norm_axes[i] = norm_axes[j]; norm_axes[j] = t; }
    // Строим output dims: вставляем 1s at axes positions
    int64_t *new_dims = malloc(out_ndims * sizeof(int64_t));
    int64_t src = 0;
    for (int64_t d = 0; d < out_ndims; d++) {
        int is_axis = 0;
        for (int64_t a = 0; a < n_axes; a++) { if (norm_axes[a] == d) { is_axis = 1; break; } }
        new_dims[d] = is_axis ? 1 : i_data->dims[src++];
    }
    free(norm_axes);

    // Проверяем, изменились ли dims
    int dims_changed = (o_expanded->n_dims != out_ndims);
    if (!dims_changed) {
        for (int64_t d = 0; d < out_ndims; d++) {
            if (o_expanded->dims[d] != new_dims[d]) {
                dims_changed = 1;
                break;
            }
        }
    }

    if (dims_changed) {
        // Пересчитываем dims
        free(o_expanded->dims);
        o_expanded->dims = new_dims;
        o_expanded->n_dims = out_ndims;
        // Перевыделяем data
        freeTensorData(o_expanded);
        mallocTensorData(o_expanded);
    } else {
        free(new_dims);
    }

    #define SAFE_COPY(FIELD, N_FIELD, TYPE) \
        if (i_data->FIELD && o_expanded->FIELD) { \
            size_t n = (i_data->N_FIELD < o_expanded->N_FIELD) ? i_data->N_FIELD : o_expanded->N_FIELD; \
            memcpy(o_expanded->FIELD, i_data->FIELD, n * sizeof(TYPE)); \
        }

    switch (o_expanded->data_type) {
        case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:  SAFE_COPY(float_data, n_float_data, float); break;
        case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE: SAFE_COPY(double_data, n_double_data, double); break;
        case ONNX__TENSOR_PROTO__DATA_TYPE__INT32:
        case ONNX__TENSOR_PROTO__DATA_TYPE__BOOL:
        case ONNX__TENSOR_PROTO__DATA_TYPE__UINT8:  SAFE_COPY(int32_data, n_int32_data, int32_t); break;
        case ONNX__TENSOR_PROTO__DATA_TYPE__INT64:   SAFE_COPY(int64_data, n_int64_data, int64_t); break;
        case ONNX__TENSOR_PROTO__DATA_TYPE__UINT64:  SAFE_COPY(uint64_data, n_uint64_data, uint64_t); break;
        default: SAFE_COPY(float_data, n_float_data, float); break;
    }
    #undef SAFE_COPY
    return OP_OK;
}
