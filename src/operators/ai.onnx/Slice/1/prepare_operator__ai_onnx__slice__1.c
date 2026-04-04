#include "operator__ai_onnx__slice__1.h"
#include "tracing.h"
#include "utils.h"

operator_status
prepare_operator__ai_onnx__slice__1(node_context *ctx)
{
    TRACE_ENTRY(1);
    TRACE_NODE(2, true, ctx->onnx_node);

    Onnx__TensorProto *i_data = searchInputByName(ctx, 0);
    Onnx__TensorProto *i_starts = searchInputByName(ctx, 1);
    Onnx__TensorProto *i_ends = searchInputByName(ctx, 2);
    Onnx__TensorProto *i_axes = searchInputByName(ctx, 3);
    Onnx__TensorProto *i_steps = searchInputByName(ctx, 4);
    Onnx__TensorProto *o_output = searchOutputByName(ctx, 0);
    int64_t ndims = i_data->n_dims;
    /* Default: all axes, step 1 */
    int64_t *starts = calloc(ndims, sizeof(int64_t));
    int64_t *ends = malloc(ndims * sizeof(int64_t));
    int64_t *steps = malloc(ndims * sizeof(int64_t));
    for (int64_t i = 0; i < ndims; i++) { ends[i] = i_data->dims[i]; steps[i] = 1; }
    int64_t n_slices = i_starts->n_int64_data;
    for (int64_t i = 0; i < n_slices; i++) {
        int64_t ax = i_axes ? i_axes->int64_data[i] : i;
        if (ax < 0) ax += ndims;
        int64_t step = (i_steps && i < (int64_t)i_steps->n_int64_data) ? i_steps->int64_data[i] : 1;
        int64_t start = i_starts->int64_data[i];
        int64_t end = i_ends->int64_data[i];
        int64_t dim = i_data->dims[ax];
        if (start < 0) start += dim;
        if (end < 0) end += dim;
        if (start < 0) start = 0;
        if (end < 0) end = 0;
        if (step > 0) { if (start > dim) start = dim; if (end > dim) end = dim; }
        else { if (start >= dim) start = dim - 1; if (end < -1) end = -1; }
        starts[ax] = start;
        ends[ax] = end;
        steps[ax] = step;
    }
    o_output->has_raw_data = 0;
    o_output->data_type = i_data->data_type;
    o_output->n_dims = ndims;
    o_output->dims = malloc(ndims * sizeof(int64_t));
    for (int64_t i = 0; i < ndims; i++) {
        int64_t s = steps[i];
        int64_t len = (s > 0) ? (ends[i] - starts[i] + s - 1) / s : (starts[i] - ends[i] - s - 1) / (-s);
        if (len < 0) len = 0;
        o_output->dims[i] = len;
    }
    mallocTensorData(o_output);
    /* Store slice params in context */
    context_operator__ai_onnx__slice__1 *op_ctx = malloc(sizeof(context_operator__ai_onnx__slice__1));
    op_ctx->starts = starts;
    op_ctx->steps = steps;
    free(ends);
    ctx->executer = resolve_operator__ai_onnx__slice__1(ctx);
    ctx->executer_context = op_ctx;

    TRACE_EXIT(1);
    return OP_OK;
}
