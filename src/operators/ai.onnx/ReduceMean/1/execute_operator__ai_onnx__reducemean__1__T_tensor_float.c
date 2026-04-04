#include "operator__ai_onnx__reducemean__1.h"
#include "tracing.h"
#include "utils.h"
#include <string.h>
#include <stdint.h>
#include "op_utils.h"
operator_status
execute_operator__ai_onnx__reducemean__1__T_tensor_float(node_context *ctx)
{
    TRACE_ENTRY(1);
    TRACE_NODE(2, true, ctx->onnx_node);

    Onnx__TensorProto *i_data = searchInputByName(ctx, 0);
    if (!i_data || tensor_is_empty(i_data)) return OP_OK;
    if (!tensor_has_data(searchInputByName(ctx, 0))) return OP_OK;
    Onnx__TensorProto *o_reduced = searchOutputByName(ctx, 0);
    /* Skip if output tensor is empty (has 0-dim) */
    if (tensor_is_empty(o_reduced)) return OP_OK;
    if (!tensor_has_data(searchInputByName(ctx, 0))) return OP_OK;

    Onnx__AttributeProto *a_axes = searchAttributeNyName(
        ctx->onnx_node->n_attribute, ctx->onnx_node->attribute, "axes");
    Onnx__AttributeProto *a_keepdims = searchAttributeNyName(
        ctx->onnx_node->n_attribute, ctx->onnx_node->attribute, "keepdims");
    int64_t keepdims = a_keepdims ? a_keepdims->i : 1;
    int64_t ndims = i_data->n_dims;
    int8_t *reduce = calloc(ndims, sizeof(int8_t));
    if (a_axes && a_axes->n_ints > 0) {
        for (int64_t i = 0; i < (int64_t)a_axes->n_ints; i++) {
            int64_t ax = a_axes->ints[i]; if (ax < 0) ax += ndims; reduce[ax] = 1;
        }
    } else { for (int64_t i = 0; i < ndims; i++) reduce[i] = 1; }
    /* Compute strides */
    int64_t *strides = malloc(ndims * sizeof(int64_t));
    strides[ndims - 1] = 1;
    for (int64_t i = ndims - 2; i >= 0; i--) strides[i] = strides[i+1] * i_data->dims[i+1];
    int64_t out_total = 1;
    for (int64_t i = 0; i < (int64_t)o_reduced->n_dims; i++) out_total *= o_reduced->dims[i];
    /* Zero output */
    for (int64_t i = 0; i < out_total; i++) o_reduced->float_data[i] = 0;
    /* Compute number of elements being reduced */
    int64_t reduce_count = 1;
    for (int64_t i = 0; i < ndims; i++) if (reduce[i]) reduce_count *= i_data->dims[i];
    /* Iterate all input elements, accumulate into output */
    int64_t in_total = 1;
    for (int64_t i = 0; i < ndims; i++) in_total *= i_data->dims[i];
    int64_t *out_strides = malloc(o_reduced->n_dims * sizeof(int64_t));
    if (o_reduced->n_dims > 0) {
        out_strides[o_reduced->n_dims - 1] = 1;
        for (int64_t i = (int64_t)o_reduced->n_dims - 2; i >= 0; i--)
            out_strides[i] = out_strides[i+1] * o_reduced->dims[i+1];
    }
    for (int64_t flat = 0; flat < in_total; flat++) {
        /* Compute multi-dim coords from flat input index */
        int64_t rem = flat, out_idx = 0, od = 0;
        for (int64_t d = 0; d < ndims; d++) {
            int64_t coord = rem / strides[d];
            rem %= strides[d];
            if (!reduce[d] || keepdims) {
                int64_t oc = reduce[d] ? 0 : coord;
                if (od < (int64_t)o_reduced->n_dims) out_idx += oc * out_strides[od];
                od++;
            }
        }
        o_reduced->float_data[out_idx] += i_data->float_data[flat];
    }
    for (int64_t i = 0; i < out_total; i++) o_reduced->float_data[i] /= reduce_count;
    free(reduce); free(strides); free(out_strides);

    TRACE_EXIT(1);
    return OP_OK;
}
