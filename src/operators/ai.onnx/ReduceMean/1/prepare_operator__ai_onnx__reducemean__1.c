#include "operator__ai_onnx__reducemean__1.h"
#include "tracing.h"
#include "utils.h"

operator_status
prepare_operator__ai_onnx__reducemean__1(node_context *ctx)
{
    TRACE_ENTRY(1);
    TRACE_NODE(2, true, ctx->onnx_node);

    Onnx__TensorProto *i_data = searchInputByName(ctx, 0);
    Onnx__TensorProto *o_reduced = searchOutputByName(ctx, 0);
    Onnx__AttributeProto *a_axes = searchAttributeNyName(
        ctx->onnx_node->n_attribute, ctx->onnx_node->attribute, "axes");
    Onnx__AttributeProto *a_keepdims = searchAttributeNyName(
        ctx->onnx_node->n_attribute, ctx->onnx_node->attribute, "keepdims");
    int64_t keepdims = a_keepdims ? a_keepdims->i : 1;
    int64_t ndims = i_data->n_dims;
    /* Determine which axes to reduce */
    int8_t *reduce = calloc(ndims, sizeof(int8_t));
    if (a_axes && a_axes->n_ints > 0) {
        for (int64_t i = 0; i < (int64_t)a_axes->n_ints; i++) {
            int64_t ax = a_axes->ints[i];
            if (ax < 0) ax += ndims;
            reduce[ax] = 1;
        }
    } else {
        for (int64_t i = 0; i < ndims; i++) reduce[i] = 1;
    }
    o_reduced->has_raw_data = 0;
    o_reduced->data_type = i_data->data_type;
    if (keepdims) {
        o_reduced->n_dims = ndims;
        o_reduced->dims = malloc(ndims * sizeof(int64_t));
        for (int64_t i = 0; i < ndims; i++) o_reduced->dims[i] = reduce[i] ? 1 : i_data->dims[i];
    } else {
        int64_t out_nd = 0;
        for (int64_t i = 0; i < ndims; i++) if (!reduce[i]) out_nd++;
        if (out_nd == 0) out_nd = 1; /* scalar output */
        o_reduced->n_dims = out_nd;
        o_reduced->dims = malloc(out_nd * sizeof(int64_t));
        int64_t d = 0;
        for (int64_t i = 0; i < ndims; i++) if (!reduce[i]) o_reduced->dims[d++] = i_data->dims[i];
        if (d == 0) o_reduced->dims[0] = 1;
    }
    free(reduce);
    mallocTensorData(o_reduced);
    ctx->executer = resolve_operator__ai_onnx__reducemean__1(ctx);

    TRACE_EXIT(1);
    return OP_OK;
}
