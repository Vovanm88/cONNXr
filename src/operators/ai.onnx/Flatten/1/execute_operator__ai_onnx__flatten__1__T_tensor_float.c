#include "operator__ai_onnx__flatten__1.h"
#include "tracing.h"
#include "utils.h"
#include <string.h>
#include <string.h>
#include <stdint.h>
operator_status
execute_operator__ai_onnx__flatten__1__T_tensor_float(node_context *ctx)
{
    TRACE_ENTRY(1);
    TRACE_NODE(2, true, ctx->onnx_node);

    Onnx__TensorProto *i_input = searchInputByName(ctx, 0);
    Onnx__TensorProto *o_output = searchOutputByName(ctx, 0);
    memcpy(o_output->float_data, i_input->float_data, o_output->n_float_data * sizeof(float));

    TRACE_EXIT(1);
    return OP_OK;
}
