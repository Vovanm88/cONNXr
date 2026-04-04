#include "operator__ai_onnx__cast__1.h"
#include "tracing.h"
#include "utils.h"
#include <string.h>
#include <stdint.h>
operator_status
execute_operator__ai_onnx__cast__1__T_tensor_float(node_context *ctx)
{
    TRACE_ENTRY(1);
    TRACE_NODE(2, true, ctx->onnx_node);

    Onnx__TensorProto *i_in = searchInputByName(ctx, 0);
    Onnx__TensorProto *o_out = searchOutputByName(ctx, 0);
    int64_t n = 1;
    for (int64_t d = 0; d < (int64_t)i_in->n_dims; d++) n *= i_in->dims[d];
    /* Get source data as doubles first */
    double *tmp = malloc(n * sizeof(double));
    printf("\n n: %lld\n", n);
    printf("i_in data type: %d\n", i_in->data_type);
    switch (i_in->data_type) {
        case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
            for (int64_t i = 0; i < n; i++) tmp[i] = i_in->float_data[i]; break;
        case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE:
            for (int64_t i = 0; i < n; i++) tmp[i] = i_in->double_data[i]; break;
        case ONNX__TENSOR_PROTO__DATA_TYPE__INT32:
            for (int64_t i = 0; i < n; i++) tmp[i] = i_in->int32_data[i]; break;
        case ONNX__TENSOR_PROTO__DATA_TYPE__INT64:
            for (int64_t i = 0; i < n; i++) tmp[i] = i_in->int64_data[i]; break;
        default: break;
    }
    /* Write to destination type */
    switch (o_out->data_type) {
        case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
            for (int64_t i = 0; i < n; i++) o_out->float_data[i] = (float)tmp[i]; break;
        case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE:
            for (int64_t i = 0; i < n; i++) o_out->double_data[i] = tmp[i]; break;
        case ONNX__TENSOR_PROTO__DATA_TYPE__INT32:
            for (int64_t i = 0; i < n; i++) o_out->int32_data[i] = (int32_t)tmp[i]; break;
        case ONNX__TENSOR_PROTO__DATA_TYPE__INT64:
            for (int64_t i = 0; i < n; i++) o_out->int64_data[i] = (int64_t)tmp[i]; break;
        default: break;
    }
    free(tmp);

    TRACE_EXIT(1);
    return OP_OK;
}
