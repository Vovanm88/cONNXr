#include "operator__ai_onnx__lessorequal__1.h"
#include "tracing.h"
#include "utils.h"
#include "broadcast_utils.h"

operator_status execute_operator__ai_onnx__lessorequal__1__T_tensor_int64(node_context *ctx) {
    Onnx__TensorProto *i_A = searchInputByName(ctx, 0);
    Onnx__TensorProto *i_B = searchInputByName(ctx, 1);
    Onnx__TensorProto *o_C = searchOutputByName(ctx, 0);
    broadcast_ctx bc;
    broadcast_init(&bc, i_A->n_dims, i_A->dims, i_B->n_dims, i_B->dims);
    for (int64_t i = 0; i < bc.total; i++) {
        int64_t ai, bi;
        broadcast_indices(&bc, i, &ai, &bi);
        o_C->int32_data[i] = (i_A->int64_data[ai] <= i_B->int64_data[bi]) ? 1 : 0;
    }
    return OP_OK;
}
