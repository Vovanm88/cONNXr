#ifndef OP_UTILS_H
#define OP_UTILS_H

#include "onnx.pb-c.h"
#include <stdint.h>

/*
 * Check if a tensor has zero total elements (any dim is 0).
 * Returns 1 if empty, 0 if non-empty.
 */
static inline int tensor_is_empty(Onnx__TensorProto *t) {
    if (!t || !t->dims) return 1;
    for (size_t i = 0; i < t->n_dims; i++) {
        if (t->dims[i] == 0) return 1;
    }
    return 0;
}

/*
 * Get total number of elements in a tensor.
 */
static inline int64_t tensor_numel(Onnx__TensorProto *t) {
    if (!t || !t->dims || t->n_dims == 0) return 1; /* scalar = 1 element */
    int64_t n = 1;
    for (size_t i = 0; i < t->n_dims; i++) {
        n *= t->dims[i];
    }
    return n;
}

/*
 * Check if a tensor's data pointer is valid for its type.
 */
static inline int tensor_has_data(Onnx__TensorProto *t) {
    if (!t) return 0;
    switch (t->data_type) {
        case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:  return t->float_data != NULL;
        case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE: return t->double_data != NULL;
        case ONNX__TENSOR_PROTO__DATA_TYPE__INT32:
        case ONNX__TENSOR_PROTO__DATA_TYPE__BOOL:
        case ONNX__TENSOR_PROTO__DATA_TYPE__UINT8:  return t->int32_data != NULL;
        case ONNX__TENSOR_PROTO__DATA_TYPE__INT64:  return t->int64_data != NULL;
        case ONNX__TENSOR_PROTO__DATA_TYPE__UINT64: return t->uint64_data != NULL;
        default: return (t->float_data || t->int32_data || t->int64_data || t->double_data);
    }
}

#endif /* OP_UTILS_H */
