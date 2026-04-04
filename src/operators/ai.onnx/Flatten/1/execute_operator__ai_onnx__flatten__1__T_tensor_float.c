#include "operator__ai_onnx__flatten__1.h"
#include "tracing.h"
#include "utils.h"
#include <string.h>
#include <stdint.h>

operator_status
execute_operator__ai_onnx__flatten__1__T_tensor_float(node_context *ctx)
{
    Onnx__TensorProto *i_input = searchInputByName(ctx, 0);
    Onnx__TensorProto *o_output = searchOutputByName(ctx, 0);

    /* Copy data based on actual type */
    switch (o_output->data_type) {
        case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
            memcpy(o_output->float_data, i_input->float_data, o_output->n_float_data * sizeof(float)); break;
        case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE:
            memcpy(o_output->double_data, i_input->double_data, o_output->n_double_data * sizeof(double)); break;
        case ONNX__TENSOR_PROTO__DATA_TYPE__INT32:
        case ONNX__TENSOR_PROTO__DATA_TYPE__BOOL:
            memcpy(o_output->int32_data, i_input->int32_data, o_output->n_int32_data * sizeof(int32_t)); break;
        case ONNX__TENSOR_PROTO__DATA_TYPE__INT64:
            memcpy(o_output->int64_data, i_input->int64_data, o_output->n_int64_data * sizeof(int64_t)); break;
        default: break;
    }
    return OP_OK;
}
