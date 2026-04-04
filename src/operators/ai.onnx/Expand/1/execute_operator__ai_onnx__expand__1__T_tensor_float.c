#include "operator__ai_onnx__expand__1.h"
#include "tracing.h"
#include "utils.h"
#include "broadcast_utils.h"
#include <stdint.h>
#include "op_utils.h"

#define EXPAND_TYPED(TYPE, FIELD, N_FIELD) \
    if (i_input->FIELD && o_output->FIELD) { \
        for (int64_t i = 0; i < (int64_t)o_output->N_FIELD; i++) { \
            int64_t ai, bi; broadcast_indices(&bc, i, &ai, &bi); \
            o_output->FIELD[i] = i_input->FIELD[ai]; \
        } \
    }

operator_status
execute_operator__ai_onnx__expand__1__T_tensor_float(node_context *ctx)
{
    Onnx__TensorProto *i_input = searchInputByName(ctx, 0);
    if (!i_input || tensor_is_empty(i_input)) return OP_OK;
    Onnx__TensorProto *o_output = searchOutputByName(ctx, 0);
    /* Skip if output tensor is empty (has 0-dim) */
    if (tensor_is_empty(o_output)) return OP_OK;

    broadcast_ctx bc;
    broadcast_init(&bc, i_input->n_dims, i_input->dims, o_output->n_dims, o_output->dims);

    switch (i_input->data_type) {
        case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT: EXPAND_TYPED(float, float_data, n_float_data); break;
        case ONNX__TENSOR_PROTO__DATA_TYPE__INT64: EXPAND_TYPED(int64_t, int64_data, n_int64_data); break;
        case ONNX__TENSOR_PROTO__DATA_TYPE__INT32:
        case ONNX__TENSOR_PROTO__DATA_TYPE__BOOL: EXPAND_TYPED(int32_t, int32_data, n_int32_data); break;
        case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE: EXPAND_TYPED(double, double_data, n_double_data); break;
        default: break;
    }
    return OP_OK;
}
