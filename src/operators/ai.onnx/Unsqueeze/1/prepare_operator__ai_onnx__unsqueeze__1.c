#include "operator__ai_onnx__unsqueeze__1.h"
#include "tracing.h"
#include "utils.h"
#include <string.h>

operator_status
prepare_operator__ai_onnx__unsqueeze__1(node_context *ctx)
{
    TRACE_ENTRY(1);
    TRACE_NODE(2, true, ctx->onnx_node);

    Onnx__TensorProto *i_data = searchInputByName(ctx, 0);
    Onnx__TensorProto *o_expanded = searchOutputByName(ctx, 0);
    Onnx__AttributeProto *a_axes = searchAttributeNyName(
        ctx->onnx_node->n_attribute, ctx->onnx_node->attribute, "axes");
    /* axes come from attribute (opset 11) or input (opset 13) */
    int64_t *axes = NULL;
    int64_t n_axes = 0;
    Onnx__TensorProto *i_axes = NULL;
    if (a_axes) {
        axes = a_axes->ints;
        n_axes = a_axes->n_ints;
    } else {
        i_axes = searchInputByName(ctx, 1);
        if (i_axes) { axes = i_axes->int64_data; n_axes = i_axes->n_int64_data; }
    }
    int64_t out_ndims = i_data->n_dims + n_axes;
    o_expanded->has_raw_data = 0;
    o_expanded->data_type = i_data->data_type;
    o_expanded->n_dims = out_ndims;
    o_expanded->dims = malloc(out_ndims * sizeof(int64_t));
    /* Normalize negative axes */
    int64_t *norm_axes = malloc(n_axes * sizeof(int64_t));
    for (int64_t i = 0; i < n_axes; i++) {
        norm_axes[i] = axes[i] < 0 ? axes[i] + out_ndims : axes[i];
    }
    /* Sort axes */
    for (int64_t i = 0; i < n_axes - 1; i++)
        for (int64_t j = i + 1; j < n_axes; j++)
            if (norm_axes[i] > norm_axes[j]) { int64_t t = norm_axes[i]; norm_axes[i] = norm_axes[j]; norm_axes[j] = t; }
    /* Build output dims: insert 1s at axes positions */
    int64_t src = 0;
    for (int64_t d = 0; d < out_ndims; d++) {
        int is_axis = 0;
        for (int64_t a = 0; a < n_axes; a++) { if (norm_axes[a] == d) { is_axis = 1; break; } }
        o_expanded->dims[d] = is_axis ? 1 : i_data->dims[src++];
    }
    free(norm_axes);
    mallocTensorData(o_expanded);
    ctx->executer = resolve_operator__ai_onnx__unsqueeze__1(ctx);

    TRACE_EXIT(1);
    return OP_OK;
}
