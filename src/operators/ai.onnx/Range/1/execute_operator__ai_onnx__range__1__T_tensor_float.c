#include "operator__ai_onnx__range__1.h"
#include "tracing.h"
#include "utils.h"
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include "op_utils.h"

operator_status
execute_operator__ai_onnx__range__1__T_tensor_float(node_context *ctx)
{
    Onnx__TensorProto *i_start = searchInputByName(ctx, 0);
    if (!i_start || tensor_is_empty(i_start)) return OP_OK;
    Onnx__TensorProto *i_limit = searchInputByName(ctx, 1);
    Onnx__TensorProto *i_delta = searchInputByName(ctx, 2);
    Onnx__TensorProto *o_output = searchOutputByName(ctx, 0);

    if (!i_start || !i_limit || !i_delta) return OP_EINVAL;

    /* Skip if output tensor is empty (has 0-dim) */
    if (tensor_is_empty(o_output)) return OP_OK;

    if (i_start->data_type == ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT) {
        float s = i_start->float_data[0], l = i_limit->float_data[0], d = i_delta->float_data[0];
        int64_t n = (d != 0) ? (int64_t)ceilf((l - s) / d) : 0;
        if (n < 0) n = 0;
        o_output->dims[0] = n;
        mallocTensorData(o_output);
        for (int64_t i = 0; i < n; i++) o_output->float_data[i] = s + i * d;
    } else if (i_start->data_type == ONNX__TENSOR_PROTO__DATA_TYPE__INT32) {
        int32_t s = i_start->int32_data[0], l = i_limit->int32_data[0], d = i_delta->int32_data[0];
        int64_t n = (d != 0) ? (int64_t)((l - s + d - (d > 0 ? 1 : -1)) / d) : 0;
        if (n < 0) n = 0;
        o_output->dims[0] = n;
        mallocTensorData(o_output);
        for (int64_t i = 0; i < n; i++) o_output->int32_data[i] = s + (int32_t)i * d;
    } else if (i_start->data_type == ONNX__TENSOR_PROTO__DATA_TYPE__INT64) {
        if (!i_start->int64_data || !i_limit->int64_data || !i_delta->int64_data) return OP_EINVAL;
        int64_t s = i_start->int64_data[0], l = i_limit->int64_data[0], d = i_delta->int64_data[0];
        int64_t n = (d != 0) ? (l - s + d - (d > 0 ? 1 : -1)) / d : 0;
        if (n < 0) n = 0;
        o_output->dims[0] = n;
        mallocTensorData(o_output);
        for (int64_t i = 0; i < n; i++) o_output->int64_data[i] = s + i * d;
    }
    return OP_OK;
}
