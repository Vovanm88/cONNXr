#include "operator__ai_onnx__add__7.h"
#include "tracing.h"
#include "utils.h"
#include "broadcast_utils.h"

operator_status
execute_operator__ai_onnx__add__7__T_tensor_float(
    node_context *ctx
)
{
    TRACE_ENTRY(1);
    TRACE_NODE(2, true, ctx->onnx_node);

    Onnx__TensorProto *i_A = searchInputByName(ctx, 0);
    Onnx__TensorProto *i_B = searchInputByName(ctx, 1);
    Onnx__TensorProto *o_C = searchOutputByName(ctx, 0);

    TRACE_TENSOR(2, true, i_A);
    TRACE_TENSOR(2, true, i_B);
    TRACE_TENSOR(2, true, o_C);

    broadcast_ctx bc;
    broadcast_init(&bc, i_A->n_dims, i_A->dims, i_B->n_dims, i_B->dims);
    for (int64_t i = 0; i < bc.total; i++) {
        int64_t ai, bi;
        broadcast_indices(&bc, i, &ai, &bi);
        o_C->float_data[i] = i_A->float_data[ai] + i_B->float_data[bi];
    }

    TRACE_EXIT(1);
    return OP_OK;
}
