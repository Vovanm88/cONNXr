#include "operator__ai_onnx__cast__1.h"
#include "tracing.h"
#include "utils.h"
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include "op_utils.h"

operator_status
execute_operator__ai_onnx__cast__1__T_tensor_float(node_context *ctx)
{
    Onnx__TensorProto *i_in = searchInputByName(ctx, 0);
    if (!i_in || tensor_is_empty(i_in)) return OP_OK;
    Onnx__TensorProto *o_out = searchOutputByName(ctx, 0);
    /* Skip if output tensor is empty (has 0-dim) */
    if (tensor_is_empty(o_out)) return OP_OK;

    int64_t n = 1;
    for (int64_t d = 0; d < (int64_t)i_in->n_dims; d++) n *= i_in->dims[d];
    if (n == 0) return OP_OK;
    if (n < 0 || n > 1000000000LL) {
        fprintf(stderr, "Cast: suspicious n=%lld from %zu dims:", (long long)n, i_in->n_dims);
        for (int64_t d = 0; d < (int64_t)i_in->n_dims; d++) fprintf(stderr, " %lld", (long long)i_in->dims[d]);
        fprintf(stderr, "\n");
        return OP_EINVAL;
    }

    double *tmp = malloc(n * sizeof(double));
    if (!tmp) return OP_ENOMEM;

    switch (i_in->data_type) {
        case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
            for (int64_t i = 0; i < n; i++) tmp[i] = i_in->float_data[i]; break;
        case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE:
            for (int64_t i = 0; i < n; i++) tmp[i] = i_in->double_data[i]; break;
        case ONNX__TENSOR_PROTO__DATA_TYPE__INT32:
        case ONNX__TENSOR_PROTO__DATA_TYPE__BOOL:
            for (int64_t i = 0; i < n; i++) tmp[i] = i_in->int32_data[i]; break;
        case ONNX__TENSOR_PROTO__DATA_TYPE__INT64:
            for (int64_t i = 0; i < n; i++) tmp[i] = (double)i_in->int64_data[i]; break;
        default: for (int64_t i = 0; i < n; i++) tmp[i] = 0; break;
    }

    switch (o_out->data_type) {
        case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
            for (int64_t i = 0; i < n; i++) o_out->float_data[i] = (float)tmp[i]; break;
        case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE:
            for (int64_t i = 0; i < n; i++) o_out->double_data[i] = tmp[i]; break;
        case ONNX__TENSOR_PROTO__DATA_TYPE__INT32:
        case ONNX__TENSOR_PROTO__DATA_TYPE__BOOL:
            for (int64_t i = 0; i < n; i++) o_out->int32_data[i] = (int32_t)tmp[i]; break;
        case ONNX__TENSOR_PROTO__DATA_TYPE__INT64:
            for (int64_t i = 0; i < n; i++) o_out->int64_data[i] = (int64_t)tmp[i]; break;
        default: break;
    }
    free(tmp);
    return OP_OK;
}
