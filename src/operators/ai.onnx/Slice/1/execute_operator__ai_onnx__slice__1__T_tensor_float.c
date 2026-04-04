#include "operator__ai_onnx__slice__1.h"
#include "tracing.h"
#include "utils.h"
#include <string.h>
#include <stdint.h>
operator_status
execute_operator__ai_onnx__slice__1__T_tensor_float(node_context *ctx)
{
    TRACE_ENTRY(1);
    TRACE_NODE(2, true, ctx->onnx_node);

    Onnx__TensorProto *i_data = searchInputByName(ctx, 0);
    Onnx__TensorProto *o_output = searchOutputByName(ctx, 0);
    context_operator__ai_onnx__slice__1 *op_ctx = ctx->executer_context;
    int64_t ndims = i_data->n_dims;
    /* Compute strides for input */
    int64_t *in_strides = malloc(ndims * sizeof(int64_t));
    in_strides[ndims - 1] = 1;
    for (int64_t i = ndims - 2; i >= 0; i--) in_strides[i] = in_strides[i+1] * i_data->dims[i+1];
    /* Iterate output and map back to input */
    int64_t total = 1;
    for (int64_t i = 0; i < ndims; i++) total *= o_output->dims[i];
    int64_t *out_strides = malloc(ndims * sizeof(int64_t));
    out_strides[ndims - 1] = 1;
    for (int64_t i = ndims - 2; i >= 0; i--) out_strides[i] = out_strides[i+1] * o_output->dims[i+1];
    for (int64_t flat = 0; flat < total; flat++) {
        int64_t in_idx = 0;
        int64_t rem = flat;
        for (int64_t d = 0; d < ndims; d++) {
            int64_t coord = rem / out_strides[d];
            rem %= out_strides[d];
            in_idx += (op_ctx->starts[d] + coord * op_ctx->steps[d]) * in_strides[d];
        }
        o_output->float_data[flat] = i_data->float_data[in_idx];
    }
    free(in_strides);
    free(out_strides);

    TRACE_EXIT(1);
    return OP_OK;
}
