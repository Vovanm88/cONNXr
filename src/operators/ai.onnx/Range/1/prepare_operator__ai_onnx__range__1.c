#include "operator__ai_onnx__range__1.h"
#include "tracing.h"
#include "utils.h"
#include <math.h>

operator_status
prepare_operator__ai_onnx__range__1(node_context *ctx)
{
    TRACE_ENTRY(1);
    TRACE_NODE(2, true, ctx->onnx_node);

    Onnx__TensorProto *i_start = searchInputByName(ctx, 0);
    Onnx__TensorProto *i_limit = searchInputByName(ctx, 1);
    Onnx__TensorProto *i_delta = searchInputByName(ctx, 2);
    printf("\ninputs found\n");
    Onnx__TensorProto *o_output = searchOutputByName(ctx, 0);
    printf("\noutputs found\n");

    o_output->has_raw_data = 0;
    o_output->data_type = i_start->data_type;
    o_output->n_dims = 1;
    o_output->dims = malloc(sizeof(int64_t));
    printf("\nmalloced\n");
    ctx->executer = (operator_executer)&execute_operator__ai_onnx__range__1__T_tensor_float;
    printf("\n executer set\n");
    TRACE_EXIT(1);
    return OP_OK;
}
