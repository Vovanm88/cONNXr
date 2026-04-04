#include "operator__ai_onnx__slice__1.h"
#include "tracing.h"
#include "utils.h"
#include <stdint.h>
#include <stdlib.h>
#include "op_utils.h"

#define SLICE_TYPED(TYPE, FIELD) \
    if (i_data->FIELD && o_output->FIELD) { \
        int64_t in_total = tensor_numel(i_data); \
        for (int64_t flat = 0; flat < total; flat++) { \
            int64_t in_idx = 0, rem = flat; \
            for (int64_t d = 0; d < ndims; d++) { \
                int64_t coord = rem / out_strides[d]; rem %= out_strides[d]; \
                in_idx += (op_ctx->starts[d] + coord * op_ctx->steps[d]) * in_strides[d]; \
            } \
            if (in_idx >= 0 && in_idx < in_total) \
                ((TYPE*)o_output->FIELD)[flat] = ((TYPE*)i_data->FIELD)[in_idx]; \
        } \
    }

operator_status
execute_operator__ai_onnx__slice__1__T_tensor_float(node_context *ctx)
{
    Onnx__TensorProto *i_data = searchInputByName(ctx, 0);
    Onnx__TensorProto *o_output = searchOutputByName(ctx, 0);
    if (tensor_is_empty(o_output) || tensor_is_empty(i_data)) return OP_OK;
    if (!i_data || !o_output) return OP_OK;

    context_operator__ai_onnx__slice__1 *op_ctx = ctx->executer_context;
    if (!op_ctx) return OP_OK;
    int64_t ndims = i_data->n_dims;
    int64_t *in_strides = malloc(ndims * sizeof(int64_t));
    int64_t *out_strides = malloc(ndims * sizeof(int64_t));
    in_strides[ndims - 1] = 1;
    out_strides[ndims - 1] = 1;
    for (int64_t i = ndims - 2; i >= 0; i--) {
        in_strides[i] = in_strides[i+1] * i_data->dims[i+1];
        out_strides[i] = out_strides[i+1] * o_output->dims[i+1];
    }
    int64_t total = 1;
    for (int64_t i = 0; i < ndims; i++) total *= o_output->dims[i];

    switch (i_data->data_type) {
        case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT: SLICE_TYPED(float, float_data); break;
        case ONNX__TENSOR_PROTO__DATA_TYPE__INT64: SLICE_TYPED(int64_t, int64_data); break;
        case ONNX__TENSOR_PROTO__DATA_TYPE__INT32:
        case ONNX__TENSOR_PROTO__DATA_TYPE__BOOL: SLICE_TYPED(int32_t, int32_data); break;
        case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE: SLICE_TYPED(double, double_data); break;
        default: SLICE_TYPED(float, float_data); break;
    }
    free(in_strides); free(out_strides);
    return OP_OK;
}
