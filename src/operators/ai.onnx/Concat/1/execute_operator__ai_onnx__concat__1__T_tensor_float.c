#include "operator__ai_onnx__concat__1.h"
#include "tracing.h"
#include "utils.h"
#include <string.h>
#include <string.h>
#include <stdint.h>
operator_status
execute_operator__ai_onnx__concat__1__T_tensor_float(node_context *ctx)
{
    TRACE_ENTRY(1);
    TRACE_NODE(2, true, ctx->onnx_node);

    Onnx__TensorProto *o_concat = searchOutputByName(ctx, 0);
    Onnx__AttributeProto *a_axis = searchAttributeNyName(
        ctx->onnx_node->n_attribute, ctx->onnx_node->attribute, "axis");
    Onnx__TensorProto *i_first = searchInputByName(ctx, 0);
    int64_t axis = a_axis ? a_axis->i : 0;
    if (axis < 0) axis += (int64_t)i_first->n_dims;
    /* Compute outer_size (product of dims before axis) and inner_size (after axis) */
    int64_t outer = 1, inner = 1;
    for (int64_t d = 0; d < axis; d++) outer *= o_concat->dims[d];
    for (int64_t d = axis + 1; d < (int64_t)o_concat->n_dims; d++) inner *= o_concat->dims[d];
    int64_t out_offset = 0;
    for (int64_t inp = 0; inp < (int64_t)ctx->onnx_node->n_input; inp++) {
        Onnx__TensorProto *t = searchInputByName(ctx, inp);
        if (!t) continue;
        int64_t chunk = t->dims[axis] * inner;
        for (int64_t o = 0; o < outer; o++) {
            memcpy(&o_concat->float_data[o * o_concat->dims[axis] * inner + out_offset],
                   &t->float_data[o * chunk],
                   chunk * sizeof(float));
        }
        out_offset += chunk;
    }

    TRACE_EXIT(1);
    return OP_OK;
}
