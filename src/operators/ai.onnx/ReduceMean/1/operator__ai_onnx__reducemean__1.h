#ifndef OPERATOR_OPERATOR__AI_ONNX__REDUCEMEAN__1_H
#define OPERATOR_OPERATOR__AI_ONNX__REDUCEMEAN__1_H
#include "operators/operator.h"
#include "operators/operator_stub.h"
#include "operators/operator_info.h"

operator_status prepare_operator__ai_onnx__reducemean__1(node_context *ctx);
extern operator_info info_operator__ai_onnx__reducemean__1;

typedef struct {
// no attributes
} context_operator__ai_onnx__reducemean__1;

operator_executer resolve_operator__ai_onnx__reducemean__1(node_context *ctx);

operator_status execute_operator__ai_onnx__reducemean__1__T_tensor_float(node_context *ctx);

operator_status execute_operator__ai_onnx__reducemean__1__T_tensor_double(node_context *ctx);

#endif
