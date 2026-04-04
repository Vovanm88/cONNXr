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
