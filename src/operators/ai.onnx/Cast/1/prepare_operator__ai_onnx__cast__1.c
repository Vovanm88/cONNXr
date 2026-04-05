#include "operator__ai_onnx__cast__1.h"
#include "tracing.h"
#include "utils.h"

operator_status
prepare_operator__ai_onnx__cast__1(node_context *ctx)
{
    TRACE_ENTRY(1);
    TRACE_NODE(2, true, ctx->onnx_node);

    Onnx__TensorProto *i_input = searchInputByName(ctx, 0);
    Onnx__TensorProto *o_output = searchOutputByName(ctx, 0);
    Onnx__AttributeProto *a_to = searchAttributeNyName(
        ctx->onnx_node->n_attribute, ctx->onnx_node->attribute, "to");
    int64_t to_type = a_to ? a_to->i : i_input->data_type;
    o_output->has_raw_data = 0;
    o_output->data_type = to_type;
    o_output->n_dims = i_input->n_dims;
    o_output->dims = ARRAYDUP(i_input->dims, i_input->n_dims);
    mallocTensorData(o_output);
    /* Cast is special: we handle all type combos in one execute function */
    ctx->executer = (operator_executer)&execute_operator__ai_onnx__cast__1__T_tensor_float;

    /* If input data is available, perform the cast during prepare */
    operator_status exec_status = execute_operator__ai_onnx__cast__1__T_tensor_float(ctx);
    if (exec_status != OP_OK) return exec_status;

    TRACE_EXIT(1);
    return OP_OK;
}
