#include "operator__ai_onnx__unsqueeze__1.h"
#include <stddef.h>
static uint32_t constraint_types[] = {
    ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT,
    ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE,
    ONNX__TENSOR_PROTO__DATA_TYPE__INT32,
    ONNX__TENSOR_PROTO__DATA_TYPE__INT64,
};
static operator_info_tensor inputs[] = {
    { .name = "input", .optional = false, .constraint = "T", .n_types = 4, .types = constraint_types },
    { .name = "input2", .optional = true, .constraint = "T", .n_types = 4, .types = constraint_types },
    { .name = "input3", .optional = true, .constraint = "T", .n_types = 4, .types = constraint_types },
    { .name = "input4", .optional = true, .constraint = "T", .n_types = 4, .types = constraint_types },
    { .name = "input5", .optional = true, .constraint = "T", .n_types = 4, .types = constraint_types },
};
static operator_info_tensor outputs[] = {
    { .name = "output", .optional = false, .constraint = "T", .n_types = 4, .types = constraint_types },
};
static operator_info_constraint constraints[] = {
    { .name = "T" },
};
operator_info info_operator__ai_onnx__unsqueeze__1 = {
    .name = "Unsqueeze",
    .range_input = { 1, 5 },
    .range_output = { 1, 1 },
    .n_attribute = 0,
    .attribute = NULL,
    .n_input = 1,
    .input = inputs,
    .n_output = 1,
    .output = outputs,
    .n_constraint = 1,
    .constraint = constraints,
};
