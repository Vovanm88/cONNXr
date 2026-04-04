#include "operator__ai_onnx__constantofshape__1.h"
#include "tracing.h"
#include "utils.h"

operator_status
prepare_operator__ai_onnx__constantofshape__1(node_context *ctx)
{
    TRACE_ENTRY(1);
    TRACE_NODE(2, true, ctx->onnx_node);

    Onnx__TensorProto *i_shape = searchInputByName(ctx, 0);
    Onnx__TensorProto *o_output = searchOutputByName(ctx, 0);
    Onnx__AttributeProto *a_value = searchAttributeNyName(
        ctx->onnx_node->n_attribute, ctx->onnx_node->attribute, "value");
    /* Default: float 0 */
    int32_t dtype = ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT;
    if (a_value && a_value->t) {
        dtype = a_value->t->data_type;
        convertRawDataOfTensorProto(a_value->t);
    }
    o_output->has_raw_data = 0;
    o_output->data_type = dtype;
    o_output->n_dims = i_shape->n_int64_data;
    o_output->dims = malloc(o_output->n_dims * sizeof(int64_t));
    for (int64_t i = 0; i < (int64_t)o_output->n_dims; i++) o_output->dims[i] = i_shape->int64_data[i];
    mallocTensorData(o_output);
    /* Fill with value */
    int64_t total = 1;
    for (int64_t i = 0; i < (int64_t)o_output->n_dims; i++) total *= o_output->dims[i];
    if (dtype == ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT) {
        float v = (a_value && a_value->t && a_value->t->n_float_data > 0) ? a_value->t->float_data[0] : 0.0f;
        for (int64_t i = 0; i < total; i++) o_output->float_data[i] = v;
    } else if (dtype == ONNX__TENSOR_PROTO__DATA_TYPE__INT32) {
        int32_t v = (a_value && a_value->t && a_value->t->n_int32_data > 0) ? a_value->t->int32_data[0] : 0;
        for (int64_t i = 0; i < total; i++) o_output->int32_data[i] = v;
    } else if (dtype == ONNX__TENSOR_PROTO__DATA_TYPE__INT64) {
        int64_t v = (a_value && a_value->t && a_value->t->n_int64_data > 0) ? a_value->t->int64_data[0] : 0;
        for (int64_t i = 0; i < total; i++) o_output->int64_data[i] = v;
    } else if (dtype == ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE) {
        double v = (a_value && a_value->t && a_value->t->n_double_data > 0) ? a_value->t->double_data[0] : 0.0;
        for (int64_t i = 0; i < total; i++) o_output->double_data[i] = v;
    }
    ctx->executer = (operator_executer)&execute_operator__ai_onnx__constantofshape__1__T_tensor_float;

    TRACE_EXIT(1);
    return OP_OK;
}
