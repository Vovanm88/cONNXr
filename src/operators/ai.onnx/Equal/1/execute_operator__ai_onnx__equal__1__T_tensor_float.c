#include "operator__ai_onnx__equal__1.h"
#include "tracing.h"
#include "utils.h"
#include "broadcast_utils.h"
#include <string.h>
#include <stdint.h>

#define EQUAL_TYPED(TYPE, FIELD) \
    for (int64_t i = 0; i < bc.total; i++) { \
        int64_t ai, bi; broadcast_indices(&bc, i, &ai, &bi); \
        o_C->int32_data[i] = (i_A->FIELD[ai] == i_B->FIELD[bi]) ? 1 : 0; \
    }

operator_status
execute_operator__ai_onnx__equal__1__T_tensor_float(node_context *ctx)
{
    Onnx__TensorProto *i_A = searchInputByName(ctx, 0);
    Onnx__TensorProto *i_B = searchInputByName(ctx, 1);
    Onnx__TensorProto *o_C = searchOutputByName(ctx, 0);
    broadcast_ctx bc;
    broadcast_init(&bc, i_A->n_dims, i_A->dims, i_B->n_dims, i_B->dims);

    switch (i_A->data_type) {
        case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT: EQUAL_TYPED(float, float_data); break;
        case ONNX__TENSOR_PROTO__DATA_TYPE__INT64: EQUAL_TYPED(int64_t, int64_data); break;
        case ONNX__TENSOR_PROTO__DATA_TYPE__INT32:
        case ONNX__TENSOR_PROTO__DATA_TYPE__BOOL: EQUAL_TYPED(int32_t, int32_data); break;
        case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE: EQUAL_TYPED(double, double_data); break;
        default: EQUAL_TYPED(float, float_data); break;
    }
    return OP_OK;
}
