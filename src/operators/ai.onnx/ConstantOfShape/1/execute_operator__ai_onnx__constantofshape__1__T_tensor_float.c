#include "operator__ai_onnx__constantofshape__1.h"
#include "tracing.h"
#include "utils.h"
#include <string.h>
#include <stdint.h>
operator_status
execute_operator__ai_onnx__constantofshape__1__T_tensor_float(node_context *ctx)
{
    TRACE_ENTRY(1);
    TRACE_NODE(2, true, ctx->onnx_node);

    /* Already filled in prepare */

    TRACE_EXIT(1);
    return OP_OK;
}
