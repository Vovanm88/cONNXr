#include "operator__ai_onnx__and__1.h"
#include <stddef.h>
static uint32_t ct[] = { ONNX__TENSOR_PROTO__DATA_TYPE__BOOL };
static operator_info_tensor ins[] = {
    { .name = "A", .optional = false, .constraint = "T", .n_types = 1, .types = ct },
    { .name = "B", .optional = false, .constraint = "T", .n_types = 1, .types = ct },
};
static operator_info_tensor outs[] = {
    { .name = "C", .optional = false, .constraint = "T", .n_types = 1, .types = ct },
};
static operator_info_constraint constraints[] = { { .name = "T" } };
operator_info info_operator__ai_onnx__and__1 = {
    .name = "And", .range_input = {2,2}, .range_output = {1,1},
    .n_input = 2, .input = ins, .n_output = 1, .output = outs,
    .n_constraint = 1, .constraint = constraints,
};
