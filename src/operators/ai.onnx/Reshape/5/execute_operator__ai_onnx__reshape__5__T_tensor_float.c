#include "operator__ai_onnx__reshape__5.h"
#include "tracing.h"
#include "utils.h"
#include <string.h>

operator_status
execute_operator__ai_onnx__reshape__5__T_tensor_float(node_context *ctx)
{
    Onnx__TensorProto *i_data = searchInputByName(ctx, 0);
    Onnx__TensorProto *o_reshaped = searchOutputByName(ctx, 0);

    switch (o_reshaped->data_type) {
        case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
            if (i_data->float_data && o_reshaped->float_data)
                memcpy(o_reshaped->float_data, i_data->float_data, o_reshaped->n_float_data * sizeof(float));
            break;
        case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE:
            if (i_data->double_data && o_reshaped->double_data)
                memcpy(o_reshaped->double_data, i_data->double_data, o_reshaped->n_double_data * sizeof(double));
            break;
        case ONNX__TENSOR_PROTO__DATA_TYPE__INT32:
        case ONNX__TENSOR_PROTO__DATA_TYPE__BOOL:
            if (i_data->int32_data && o_reshaped->int32_data)
                memcpy(o_reshaped->int32_data, i_data->int32_data, o_reshaped->n_int32_data * sizeof(int32_t));
            break;
        case ONNX__TENSOR_PROTO__DATA_TYPE__INT64:
            if (i_data->int64_data && o_reshaped->int64_data)
                memcpy(o_reshaped->int64_data, i_data->int64_data, o_reshaped->n_int64_data * sizeof(int64_t));
            break;
        default:
            if (i_data->float_data && o_reshaped->float_data)
                memcpy(o_reshaped->float_data, i_data->float_data, o_reshaped->n_float_data * sizeof(float));
            break;
    }
    return OP_OK;
}
