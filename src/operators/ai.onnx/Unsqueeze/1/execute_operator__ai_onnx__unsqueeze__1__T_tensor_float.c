#include "operator__ai_onnx__unsqueeze__1.h"
#include "tracing.h"
#include "utils.h"
#include <string.h>
#include <stdint.h>

operator_status
execute_operator__ai_onnx__unsqueeze__1__T_tensor_float(node_context *ctx)
{
    Onnx__TensorProto *i_data = searchInputByName(ctx, 0);
    Onnx__TensorProto *o_expanded = searchOutputByName(ctx, 0);
    if (!i_data || !o_expanded) return OP_EINVAL;

    /* Unsqueeze is a view op — just copy data by type */
    switch (o_expanded->data_type) {
        case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
            if (i_data->float_data && o_expanded->float_data)
                memcpy(o_expanded->float_data, i_data->float_data, o_expanded->n_float_data * sizeof(float));
            break;
        case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE:
            if (i_data->double_data && o_expanded->double_data)
                memcpy(o_expanded->double_data, i_data->double_data, o_expanded->n_double_data * sizeof(double));
            break;
        case ONNX__TENSOR_PROTO__DATA_TYPE__INT32:
        case ONNX__TENSOR_PROTO__DATA_TYPE__BOOL:
        case ONNX__TENSOR_PROTO__DATA_TYPE__UINT8:
            if (i_data->int32_data && o_expanded->int32_data)
                memcpy(o_expanded->int32_data, i_data->int32_data, o_expanded->n_int32_data * sizeof(int32_t));
            break;
        case ONNX__TENSOR_PROTO__DATA_TYPE__INT64:
            if (i_data->int64_data && o_expanded->int64_data)
                memcpy(o_expanded->int64_data, i_data->int64_data, o_expanded->n_int64_data * sizeof(int64_t));
            break;
        case ONNX__TENSOR_PROTO__DATA_TYPE__UINT64:
            if (i_data->uint64_data && o_expanded->uint64_data)
                memcpy(o_expanded->uint64_data, i_data->uint64_data, o_expanded->n_uint64_data * sizeof(uint64_t));
            break;
        default:
            /* Fallback: try float, then int64 */
            if (i_data->float_data && o_expanded->float_data)
                memcpy(o_expanded->float_data, i_data->float_data, o_expanded->n_float_data * sizeof(float));
            else if (i_data->int64_data && o_expanded->int64_data)
                memcpy(o_expanded->int64_data, i_data->int64_data, o_expanded->n_int64_data * sizeof(int64_t));
            break;
    }
    return OP_OK;
}
