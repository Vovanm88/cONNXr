#include "operator__ai_onnx__mul__7.h"
#include "tracing.h"
#include "utils.h"
#include "broadcast_utils.h"
#include "op_utils.h"

#define MUL_TYPED(TYPE, FIELD) \
    if (i_A->FIELD && i_B->FIELD && o_C->FIELD) { \
        for (int64_t i = 0; i < bc.total; i++) { \
            int64_t ai, bi; broadcast_indices(&bc, i, &ai, &bi); \
            o_C->FIELD[i] = i_A->FIELD[ai] * i_B->FIELD[bi]; \
        } \
    }

operator_status
execute_operator__ai_onnx__mul__7__T_tensor_float(node_context *ctx)
{
    Onnx__TensorProto *i_A = searchInputByName(ctx, 0);
    if (!i_A || tensor_is_empty(i_A)) return OP_OK;
    Onnx__TensorProto *i_B = searchInputByName(ctx, 1);
    Onnx__TensorProto *o_C = searchOutputByName(ctx, 0);
    if (!i_A || !i_B || !o_C) return OP_OK;
    /* Skip if output tensor is empty (has 0-dim) */
    if (tensor_is_empty(o_C)) return OP_OK;

    broadcast_ctx bc;
    broadcast_init(&bc, i_A->n_dims, i_A->dims, i_B->n_dims, i_B->dims);

    switch (i_A->data_type) {
        case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT: MUL_TYPED(float, float_data); break;
        case ONNX__TENSOR_PROTO__DATA_TYPE__INT64: MUL_TYPED(int64_t, int64_data); break;
        case ONNX__TENSOR_PROTO__DATA_TYPE__INT32: MUL_TYPED(int32_t, int32_data); break;
        case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE: MUL_TYPED(double, double_data); break;
        default: MUL_TYPED(float, float_data); break;
    }
    return OP_OK;
}
