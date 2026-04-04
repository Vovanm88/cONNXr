#ifndef OPERATOR_OPERATOR__AI_ONNX__AND__1_H
#define OPERATOR_OPERATOR__AI_ONNX__AND__1_H
#include "operators/operator.h"
#include "operators/operator_stub.h"
#include "operators/operator_info.h"
typedef struct { } context_operator__ai_onnx__and__1;
operator_status prepare_operator__ai_onnx__and__1(node_context *ctx);
extern operator_info info_operator__ai_onnx__and__1;
operator_executer resolve_operator__ai_onnx__and__1(node_context *ctx);
operator_status execute_operator__ai_onnx__and__1__T_tensor_float(node_context *ctx);
#endif
