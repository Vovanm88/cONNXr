#include "operator__ai_onnx__flatten__1.h"
#include "tracing.h"
#include "utils.h"
#include <string.h>
#include <stdint.h>
#include "op_utils.h"

operator_status
execute_operator__ai_onnx__flatten__1__T_tensor_float(node_context *ctx)
{
    Onnx__TensorProto *i_input = searchInputByName(ctx, 0);
    Onnx__TensorProto *o_output = searchOutputByName(ctx, 0);
    if (tensor_is_empty(o_output) || tensor_is_empty(i_input)) return OP_OK;

    #define SAFE_COPY(FIELD, N_FIELD, TYPE) \
        if (i_input->FIELD && o_output->FIELD) { \
            size_t n = (i_input->N_FIELD < o_output->N_FIELD) ? i_input->N_FIELD : o_output->N_FIELD; \
            memcpy(o_output->FIELD, i_input->FIELD, n * sizeof(TYPE)); \
        }

    switch (o_output->data_type) {
        case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:  SAFE_COPY(float_data, n_float_data, float); break;
        case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE: SAFE_COPY(double_data, n_double_data, double); break;
        case ONNX__TENSOR_PROTO__DATA_TYPE__INT32:
        case ONNX__TENSOR_PROTO__DATA_TYPE__BOOL:   SAFE_COPY(int32_data, n_int32_data, int32_t); break;
        case ONNX__TENSOR_PROTO__DATA_TYPE__INT64:   SAFE_COPY(int64_data, n_int64_data, int64_t); break;
        default: SAFE_COPY(float_data, n_float_data, float); break;
    }
    #undef SAFE_COPY
    return OP_OK;
}
