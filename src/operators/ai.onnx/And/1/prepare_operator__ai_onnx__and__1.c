#include "operator__ai_onnx__and__1.h"
#include "tracing.h"
#include "utils.h"
#include "broadcast_utils.h"
operator_status prepare_operator__ai_onnx__and__1(node_context *ctx) {
    TRACE_ENTRY(1); TRACE_NODE(2, true, ctx->onnx_node);
    Onnx__TensorProto *i_A = searchInputByName(ctx, 0);
    Onnx__TensorProto *i_B = searchInputByName(ctx, 1);
    Onnx__TensorProto *o_C = searchOutputByName(ctx, 0);
    o_C->has_raw_data = 0;
    o_C->data_type = ONNX__TENSOR_PROTO__DATA_TYPE__BOOL;
    int64_t nd;
    o_C->dims = broadcast_shape(i_A->n_dims, i_A->dims, i_B->n_dims, i_B->dims, &nd);
    o_C->n_dims = nd;
    mallocTensorData(o_C);
    ctx->executer = (operator_executer)&execute_operator__ai_onnx__and__1__T_tensor_float;
    TRACE_EXIT(1); return OP_OK;
}
