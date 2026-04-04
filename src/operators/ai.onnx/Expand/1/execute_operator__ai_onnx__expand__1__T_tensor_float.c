#include "operator__ai_onnx__expand__1.h"
#include "tracing.h"
#include "utils.h"
#include "broadcast_utils.h"
#include <string.h>
#include <stdint.h>
operator_status
execute_operator__ai_onnx__expand__1__T_tensor_float(node_context *ctx)
{
    TRACE_ENTRY(1);
    TRACE_NODE(2, true, ctx->onnx_node);

    Onnx__TensorProto *i_input = searchInputByName(ctx, 0);
    Onnx__TensorProto *o_output = searchOutputByName(ctx, 0);
    /* Broadcast input to output shape */
    broadcast_ctx bc;
    broadcast_init(&bc, i_input->n_dims, i_input->dims, o_output->n_dims, o_output->dims);
    for (int64_t i = 0; i < (int64_t)o_output->n_float_data; i++) {
        int64_t ai, bi;
        broadcast_indices(&bc, i, &ai, &bi);
        o_output->float_data[i] = i_input->float_data[ai];
    }

    TRACE_EXIT(1);
    return OP_OK;
}
