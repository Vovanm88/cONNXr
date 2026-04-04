#include "operator__ai_onnx__where__1.h"
#include "operators/operator_stub.h"
#include <stdint.h>
#include <stdio.h>
operator_executer resolve_operator__ai_onnx__where__1(node_context *ctx) {
    operator_executer executer = NULL;
    uint32_t T = 0;
    /* Dispatch on input[1] (X), not input[0] (condition/bool) */
    if (ctx->inputs[1]) { T = ctx->inputs[1]->data_type; }
    switch (T) {
    case 0:
    case ONNX__TENSOR_PROTO__DATA_TYPE__BOOL:
    case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT: { executer = (operator_executer)&execute_operator__ai_onnx__where__1__T_tensor_float; break; }
    case ONNX__TENSOR_PROTO__DATA_TYPE__INT64: { executer = (operator_executer)&execute_operator__ai_onnx__where__1__T_tensor_int64; break; }
    default:
        fprintf(stderr, "no matching type for operator__ai_onnx__where__1 with type %d\n", T);
        break;
    }
    if (!executer) { executer = &operator_stub; }
    return executer;
}
