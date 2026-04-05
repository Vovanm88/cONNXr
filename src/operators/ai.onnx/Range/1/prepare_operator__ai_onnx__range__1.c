#include "operator__ai_onnx__range__1.h"
#include "op_utils.h"
#include "tracing.h"
#include "utils.h"
#include <math.h>
#include <stdint.h>
#include <inttypes.h>

operator_status
prepare_operator__ai_onnx__range__1(node_context *ctx)
{
    TRACE_ENTRY(1);
    TRACE_NODE(2, true, ctx->onnx_node);
    // print name of node
    Onnx__TensorProto *i_start = searchInputByName(ctx, 0);
    Onnx__TensorProto *i_limit = searchInputByName(ctx, 1);
    Onnx__TensorProto *i_delta = searchInputByName(ctx, 2);
    Onnx__TensorProto *o_output = searchOutputByName(ctx, 0);

    o_output->has_raw_data = 0;
    o_output->data_type = i_start->data_type;
    o_output->n_dims = 1;
    o_output->dims = malloc(sizeof(int64_t));
    o_output->dims[0] = 0;
    
    // Вычисляем размерность, если данные не пустые
    if (tensor_has_data(i_start) && tensor_has_data(i_limit) && tensor_has_data(i_delta)) {

        int64_t n = 0;
        if (i_start->data_type == ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT) {
            float s = i_start->float_data[0], l = i_limit->float_data[0], d = i_delta->float_data[0];
            n = (d != 0) ? (int64_t)ceilf((l - s) / d) : 0;
        } else if (i_start->data_type == ONNX__TENSOR_PROTO__DATA_TYPE__INT32) {
            int32_t s = i_start->int32_data[0], l = i_limit->int32_data[0], d = i_delta->int32_data[0];
            n = (d != 0) ? (int64_t)((l - s + d - (d > 0 ? 1 : -1)) / d) : 0;
        } else if (i_start->data_type == ONNX__TENSOR_PROTO__DATA_TYPE__INT64) {
            int64_t s = i_start->int64_data[0], l = i_limit->int64_data[0], d = i_delta->int64_data[0];
            n = (d != 0) ? (l - s + d - (d > 0 ? 1 : -1)) / d : 0;
        }else{
            printf("Unsupported data type for Range operator: %d\n", i_start->data_type);
        }
        if (n < 0) n = 0;
        o_output->dims[0] = n;
    }

    ctx->executer = (operator_executer)&execute_operator__ai_onnx__range__1__T_tensor_float;
    TRACE_EXIT(1);
    return OP_OK;
}
