#include "operator__ai_onnx__slice__1.h"
#include "tracing.h"
#include "utils.h"
#include <stdlib.h>
void free_operator__ai_onnx__slice__1(node_context *ctx) {
    TRACE_ENTRY(1);
    context_operator__ai_onnx__slice__1 *op_ctx = ctx->executer_context;
    if (op_ctx) {
        free(op_ctx->starts);
        free(op_ctx->steps);
        free(op_ctx);
    }
    TRACE_EXIT(1);
}
