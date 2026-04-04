#include "operator__ai_onnx__concat__1.h"
#include "tracing.h"
#include "utils.h"
#include "op_utils.h"
#include <string.h>

operator_status
prepare_operator__ai_onnx__concat__1(node_context *ctx)
{
    TRACE_ENTRY(1);
    TRACE_NODE(2, true, ctx->onnx_node);

    Onnx__TensorProto *i_first = searchInputByName(ctx, 0);
    Onnx__TensorProto *o_concat = searchOutputByName(ctx, 0);
    Onnx__AttributeProto *a_axis = searchAttributeNyName(
        ctx->onnx_node->n_attribute, ctx->onnx_node->attribute, "axis");
    int64_t axis = a_axis ? a_axis->i : 0;
    if (axis < 0) axis += i_first->n_dims;
    o_concat->has_raw_data = 0;
    o_concat->data_type = i_first->data_type;
    o_concat->n_dims = i_first->n_dims;
    o_concat->dims = ARRAYDUP(i_first->dims, i_first->n_dims);
    /* Sum the axis dimension across all inputs */
    int64_t axis_total = i_first->dims[axis];
    for (int64_t inp = 1; inp < (int64_t)ctx->onnx_node->n_input; inp++) {
        Onnx__TensorProto *t = searchInputByName(ctx, inp);
        if (t) axis_total += t->dims[axis];
    }
    o_concat->dims[axis] = axis_total;
    mallocTensorData(o_concat);
    ctx->executer = resolve_operator__ai_onnx__concat__1(ctx);

    TRACE_EXIT(1);
    return OP_OK;
}
