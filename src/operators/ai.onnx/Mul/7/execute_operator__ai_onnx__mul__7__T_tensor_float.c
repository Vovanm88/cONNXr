#include "operator__ai_onnx__mul__7.h"
#include "tracing.h"
#include "utils.h"
#include "broadcast_utils.h"
#include "op_utils.h"

#define TILE_SIZE 4096

#define MUL_TYPED(TYPE, FIELD) \
    if (i_A->FIELD && i_B->FIELD && o_C->FIELD) { \
        int64_t ai_indices[TILE_SIZE]; \
        int64_t bi_indices[TILE_SIZE]; \
        for (int64_t tile_start = 0; tile_start < bc.total; tile_start += TILE_SIZE) { \
            int64_t tile_end = (tile_start + TILE_SIZE < bc.total) ? tile_start + TILE_SIZE : bc.total; \
            int64_t tile_size = tile_end - tile_start; \
            for (int64_t i = 0; i < tile_size; i++) { \
                broadcast_indices(&bc, tile_start + i, &ai_indices[i], &bi_indices[i]); \
            } \
            for (int64_t i = 0; i < tile_size; i++) { \
                o_C->FIELD[tile_start + i] = i_A->FIELD[ai_indices[i]] * i_B->FIELD[bi_indices[i]]; \
            } \
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
    int bc_result = broadcast_init(&bc, i_A->n_dims, i_A->dims, i_B->n_dims, i_B->dims);
    if (bc_result != 0) {
        printf("ERROR: Broadcast init failed!\n");
        return OP_OK;
    }

    switch (i_A->data_type) {
        case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT: MUL_TYPED(float, float_data); break;
        case ONNX__TENSOR_PROTO__DATA_TYPE__INT64: MUL_TYPED(int64_t, int64_data); break;
        case ONNX__TENSOR_PROTO__DATA_TYPE__INT32: MUL_TYPED(int32_t, int32_data); break;
        case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE: MUL_TYPED(double, double_data); break;
        default: MUL_TYPED(float, float_data); break;
    }
    return OP_OK;
}
