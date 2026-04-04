#include "operator__ai_onnx__and__1.h"
#include "tracing.h"
#include "utils.h"
#include "broadcast_utils.h"
#include "op_utils.h"
operator_status execute_operator__ai_onnx__and__1__T_tensor_float(node_context *ctx) {
    TRACE_ENTRY(1); TRACE_NODE(2, true, ctx->onnx_node);
    Onnx__TensorProto *i_A = searchInputByName(ctx, 0);
    if (!i_A || tensor_is_empty(i_A)) return OP_OK;
    Onnx__TensorProto *i_B = searchInputByName(ctx, 1);
    Onnx__TensorProto *o_C = searchOutputByName(ctx, 0);
    if (!i_A || !i_B || !o_C || !i_A->int32_data || !i_B->int32_data || !o_C->int32_data)
    /* Skip if output tensor is empty (has 0-dim) */
    if (tensor_is_empty(o_C)) return OP_OK;

        return OP_OK;
    broadcast_ctx bc;
    broadcast_init(&bc, i_A->n_dims, i_A->dims, i_B->n_dims, i_B->dims);
    for (int64_t i = 0; i < bc.total; i++) {
        int64_t ai, bi;
        broadcast_indices(&bc, i, &ai, &bi);
        o_C->int32_data[i] = (i_A->int32_data[ai] && i_B->int32_data[bi]) ? 1 : 0;
    }
    TRACE_EXIT(1); return OP_OK;
}
