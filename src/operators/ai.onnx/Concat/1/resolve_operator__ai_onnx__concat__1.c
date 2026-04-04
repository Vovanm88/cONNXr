#include "operator__ai_onnx__concat__1.h"
#include "operators/operator_stub.h"
#include <stdint.h>
#include <stdio.h>
operator_executer resolve_operator__ai_onnx__concat__1(node_context *ctx) {
    operator_executer executer = NULL;
    uint32_t T = 0;
    if (ctx->inputs[0]) { T = ctx->inputs[0]->data_type; }
    switch (T) {
    case 0:
    case ONNX__TENSOR_PROTO__DATA_TYPE__BOOL:
    case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT: { executer = (operator_executer)&execute_operator__ai_onnx__concat__1__T_tensor_float; break; }
    case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE: { executer = (operator_executer)&execute_operator__ai_onnx__concat__1__T_tensor_double; break; }
    case ONNX__TENSOR_PROTO__DATA_TYPE__INT32: { executer = (operator_executer)&execute_operator__ai_onnx__concat__1__T_tensor_int32; break; }
    case ONNX__TENSOR_PROTO__DATA_TYPE__INT64: { executer = (operator_executer)&execute_operator__ai_onnx__concat__1__T_tensor_int64; break; }
    default:
        fprintf(stderr, "no matching type for operator__ai_onnx__concat__1 with type %d\n", T);
        break;
    }
    if (!executer) { executer = &operator_stub; }
    return executer;
}
