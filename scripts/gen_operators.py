#!/usr/bin/env python3
"""Generate all 22 ONNX operator implementations for cONNXr."""
import os

BASE = "src/operators/ai.onnx"

def mkdir(p):
    os.makedirs(p, exist_ok=True)

def write(path, content):
    with open(path, 'w') as f:
        f.write(content)

# ── operator definitions ──────────────────────────────────────────
# (Name, lowercase, version, inputs, outputs, attributes_code, prepare_code, execute_float_code, extra_includes, resolve_types, info_n_inputs, info_n_outputs)

OPS = []

# ═══════════════════ UNARY ELEMENT-WISE ═══════════════════

for name, lc, func, inc in [
    ("Cos", "cos", "cosf(x)", "<math.h>"),
    ("Sin", "sin", "sinf(x)", "<math.h>"),
    ("Sqrt", "sqrt", "sqrtf(x)", "<math.h>"),
    ("Neg", "neg", "(-x)", ""),
]:
    # Use version 1 so it matches any opset
    OPS.append({
        "Name": name, "lc": lc, "ver": 1,
        "prepare": f"""
    Onnx__TensorProto *i_X = searchInputByName(ctx, 0);
    Onnx__TensorProto *o_Y = searchOutputByName(ctx, 0);
    o_Y->has_raw_data = 0;
    o_Y->data_type = i_X->data_type;
    o_Y->n_dims = i_X->n_dims;
    o_Y->dims = ARRAYDUP(i_X->dims, i_X->n_dims);
    mallocTensorData(o_Y);
    ctx->executer = resolve_operator__ai_onnx__{lc}__13(ctx);
""",
        "exec_float": f"""
    Onnx__TensorProto *i_X = searchInputByName(ctx, 0);
    Onnx__TensorProto *o_Y = searchOutputByName(ctx, 0);
    for (int64_t i = 0; i < (int64_t)o_Y->n_float_data; i++) {{
        float x = i_X->float_data[i];
        o_Y->float_data[i] = {func};
    }}
""",
        "extra_inc": inc,
        "n_in": 1, "n_out": 1,
        "resolve_types": ["FLOAT", "DOUBLE"],
        "exec_double": f"""
    Onnx__TensorProto *i_X = searchInputByName(ctx, 0);
    Onnx__TensorProto *o_Y = searchOutputByName(ctx, 0);
    for (int64_t i = 0; i < (int64_t)o_Y->n_double_data; i++) {{
        double x = i_X->double_data[i];
        o_Y->double_data[i] = (double)({func.replace('cosf','cos').replace('sinf','sin').replace('sqrtf','sqrt')});
    }}
""",
    })

# ═══════════════════ IsNaN ═══════════════════
OPS.append({
    "Name": "IsNaN", "lc": "isnan", "ver": 1,
    "prepare": """
    Onnx__TensorProto *i_X = searchInputByName(ctx, 0);
    Onnx__TensorProto *o_Y = searchOutputByName(ctx, 0);
    o_Y->has_raw_data = 0;
    o_Y->data_type = ONNX__TENSOR_PROTO__DATA_TYPE__BOOL;
    o_Y->n_dims = i_X->n_dims;
    o_Y->dims = ARRAYDUP(i_X->dims, i_X->n_dims);
    mallocTensorData(o_Y);
    ctx->executer = resolve_operator__ai_onnx__isnan__13(ctx);
""",
    "exec_float": """
    Onnx__TensorProto *i_X = searchInputByName(ctx, 0);
    Onnx__TensorProto *o_Y = searchOutputByName(ctx, 0);
    for (int64_t i = 0; i < (int64_t)i_X->n_float_data; i++) {
        o_Y->int32_data[i] = isnan(i_X->float_data[i]) ? 1 : 0;
    }
""",
    "extra_inc": "<math.h>",
    "n_in": 1, "n_out": 1,
    "resolve_types": ["FLOAT"],
})

# ═══════════════════ BINARY ELEMENT-WISE WITH BROADCASTING ═══════════════════

for name, lc, op_expr in [
    ("Div", "div", "a / b"),
    ("Pow", "pow", "powf(a, b)"),
]:
    OPS.append({
        "Name": name, "lc": lc, "ver": 1,
        "prepare": f"""
    Onnx__TensorProto *i_A = searchInputByName(ctx, 0);
    Onnx__TensorProto *i_B = searchInputByName(ctx, 1);
    Onnx__TensorProto *o_C = searchOutputByName(ctx, 0);
    o_C->has_raw_data = 0;
    o_C->data_type = i_A->data_type;
    int64_t nd;
    o_C->dims = broadcast_shape(i_A->n_dims, i_A->dims, i_B->n_dims, i_B->dims, &nd);
    o_C->n_dims = nd;
    mallocTensorData(o_C);
    ctx->executer = resolve_operator__ai_onnx__{lc}__13(ctx);
""",
        "exec_float": f"""
    Onnx__TensorProto *i_A = searchInputByName(ctx, 0);
    Onnx__TensorProto *i_B = searchInputByName(ctx, 1);
    Onnx__TensorProto *o_C = searchOutputByName(ctx, 0);
    broadcast_ctx bc;
    broadcast_init(&bc, i_A->n_dims, i_A->dims, i_B->n_dims, i_B->dims);
    for (int64_t i = 0; i < bc.total; i++) {{
        int64_t ai, bi;
        broadcast_indices(&bc, i, &ai, &bi);
        float a = i_A->float_data[ai];
        float b = i_B->float_data[bi];
        o_C->float_data[i] = {op_expr};
    }}
""",
        "extra_inc": "<math.h>",
        "n_in": 2, "n_out": 1,
        "resolve_types": ["FLOAT", "DOUBLE", "INT32", "INT64"],
        "use_broadcast": True,
    })

# ═══════════════════ COMPARISON OPS (output bool) ═══════════════════

for name, lc, op_expr in [
    ("Equal", "equal", "a == b"),
    ("LessOrEqual", "lessorequal", "a <= b"),
]:
    OPS.append({
        "Name": name, "lc": lc, "ver": 1,
        "prepare": f"""
    Onnx__TensorProto *i_A = searchInputByName(ctx, 0);
    Onnx__TensorProto *i_B = searchInputByName(ctx, 1);
    Onnx__TensorProto *o_C = searchOutputByName(ctx, 0);
    o_C->has_raw_data = 0;
    o_C->data_type = ONNX__TENSOR_PROTO__DATA_TYPE__BOOL;
    int64_t nd;
    o_C->dims = broadcast_shape(i_A->n_dims, i_A->dims, i_B->n_dims, i_B->dims, &nd);
    o_C->n_dims = nd;
    mallocTensorData(o_C);
    ctx->executer = resolve_operator__ai_onnx__{lc}__13(ctx);
""",
        "exec_float": f"""
    Onnx__TensorProto *i_A = searchInputByName(ctx, 0);
    Onnx__TensorProto *i_B = searchInputByName(ctx, 1);
    Onnx__TensorProto *o_C = searchOutputByName(ctx, 0);
    broadcast_ctx bc;
    broadcast_init(&bc, i_A->n_dims, i_A->dims, i_B->n_dims, i_B->dims);
    for (int64_t i = 0; i < bc.total; i++) {{
        int64_t ai, bi;
        broadcast_indices(&bc, i, &ai, &bi);
        float a = i_A->float_data[ai];
        float b = i_B->float_data[bi];
        o_C->int32_data[i] = ({op_expr}) ? 1 : 0;
    }}
""",
        "extra_inc": "",
        "n_in": 2, "n_out": 1,
        "resolve_types": ["FLOAT", "DOUBLE", "INT32", "INT64"],
        "use_broadcast": True,
    })

# ═══════════════════ Where ═══════════════════
OPS.append({
    "Name": "Where", "lc": "where", "ver": 1,
    "prepare": """
    Onnx__TensorProto *i_C = searchInputByName(ctx, 0);
    Onnx__TensorProto *i_X = searchInputByName(ctx, 1);
    Onnx__TensorProto *i_Y = searchInputByName(ctx, 2);
    Onnx__TensorProto *o_O = searchOutputByName(ctx, 0);
    o_O->has_raw_data = 0;
    o_O->data_type = i_X->data_type;
    /* broadcast all 3 shapes: first broadcast C with X, then result with Y */
    int64_t nd1;
    int64_t *s1 = broadcast_shape(i_C->n_dims, i_C->dims, i_X->n_dims, i_X->dims, &nd1);
    int64_t nd2;
    o_O->dims = broadcast_shape(nd1, s1, i_Y->n_dims, i_Y->dims, &nd2);
    o_O->n_dims = nd2;
    free(s1);
    mallocTensorData(o_O);
    ctx->executer = resolve_operator__ai_onnx__where__13(ctx);
""",
    "exec_float": """
    Onnx__TensorProto *i_C = searchInputByName(ctx, 0);
    Onnx__TensorProto *i_X = searchInputByName(ctx, 1);
    Onnx__TensorProto *i_Y = searchInputByName(ctx, 2);
    Onnx__TensorProto *o_O = searchOutputByName(ctx, 0);
    broadcast3_ctx bc;
    broadcast3_init(&bc, i_C->n_dims, i_C->dims, i_X->n_dims, i_X->dims, i_Y->n_dims, i_Y->dims);
    for (int64_t i = 0; i < bc.total; i++) {
        int64_t ci, xi, yi;
        broadcast3_indices(&bc, i, &ci, &xi, &yi);
        o_O->float_data[i] = i_C->int32_data[ci] ? i_X->float_data[xi] : i_Y->float_data[yi];
    }
""",
    "extra_inc": "",
    "n_in": 3, "n_out": 1,
    "resolve_types": ["FLOAT", "INT64"],
    "use_broadcast": True,
    "exec_int64": """
    Onnx__TensorProto *i_C = searchInputByName(ctx, 0);
    Onnx__TensorProto *i_X = searchInputByName(ctx, 1);
    Onnx__TensorProto *i_Y = searchInputByName(ctx, 2);
    Onnx__TensorProto *o_O = searchOutputByName(ctx, 0);
    broadcast3_ctx bc;
    broadcast3_init(&bc, i_C->n_dims, i_C->dims, i_X->n_dims, i_X->dims, i_Y->n_dims, i_Y->dims);
    for (int64_t i = 0; i < bc.total; i++) {
        int64_t ci, xi, yi;
        broadcast3_indices(&bc, i, &ci, &xi, &yi);
        o_O->int64_data[i] = i_C->int32_data[ci] ? i_X->int64_data[xi] : i_Y->int64_data[yi];
    }
""",
})

# ═══════════════════ Shape ═══════════════════
OPS.append({
    "Name": "Shape", "lc": "shape", "ver": 1,
    "prepare": """
    Onnx__TensorProto *i_data = searchInputByName(ctx, 0);
    Onnx__TensorProto *o_shape = searchOutputByName(ctx, 0);
    o_shape->has_raw_data = 0;
    o_shape->data_type = ONNX__TENSOR_PROTO__DATA_TYPE__INT64;
    o_shape->n_dims = 1;
    o_shape->dims = malloc(sizeof(int64_t));
    o_shape->dims[0] = i_data->n_dims;
    mallocTensorData(o_shape);
    /* Fill immediately - Shape is a metadata op */
    for (int64_t i = 0; i < (int64_t)i_data->n_dims; i++) {
        o_shape->int64_data[i] = i_data->dims[i];
    }
    ctx->executer = resolve_operator__ai_onnx__shape__13(ctx);
""",
    "exec_float": """
    /* Already computed in prepare */
""",
    "extra_inc": "",
    "n_in": 1, "n_out": 1,
    "resolve_types": ["FLOAT", "DOUBLE", "INT32", "INT64"],
    "no_type_dispatch": True,
})

# ═══════════════════ Flatten ═══════════════════
OPS.append({
    "Name": "Flatten", "lc": "flatten", "ver": 1,
    "prepare": """
    Onnx__TensorProto *i_input = searchInputByName(ctx, 0);
    Onnx__TensorProto *o_output = searchOutputByName(ctx, 0);
    Onnx__AttributeProto *a_axis = searchAttributeNyName(
        ctx->onnx_node->n_attribute, ctx->onnx_node->attribute, "axis");
    int64_t axis = a_axis ? a_axis->i : 1;
    if (axis < 0) axis += i_input->n_dims;
    int64_t outer = 1, inner = 1;
    for (int64_t i = 0; i < axis; i++) outer *= i_input->dims[i];
    for (int64_t i = axis; i < (int64_t)i_input->n_dims; i++) inner *= i_input->dims[i];
    o_output->has_raw_data = 0;
    o_output->data_type = i_input->data_type;
    o_output->n_dims = 2;
    o_output->dims = malloc(2 * sizeof(int64_t));
    o_output->dims[0] = outer;
    o_output->dims[1] = inner;
    mallocTensorData(o_output);
    ctx->executer = resolve_operator__ai_onnx__flatten__13(ctx);
""",
    "exec_float": """
    Onnx__TensorProto *i_input = searchInputByName(ctx, 0);
    Onnx__TensorProto *o_output = searchOutputByName(ctx, 0);
    memcpy(o_output->float_data, i_input->float_data, o_output->n_float_data * sizeof(float));
""",
    "extra_inc": "<string.h>",
    "n_in": 1, "n_out": 1,
    "resolve_types": ["FLOAT", "DOUBLE", "INT32", "INT64"],
})

# ═══════════════════ Unsqueeze ═══════════════════
OPS.append({
    "Name": "Unsqueeze", "lc": "unsqueeze", "ver": 1,
    "prepare": """
    Onnx__TensorProto *i_data = searchInputByName(ctx, 0);
    Onnx__TensorProto *o_expanded = searchOutputByName(ctx, 0);
    Onnx__AttributeProto *a_axes = searchAttributeNyName(
        ctx->onnx_node->n_attribute, ctx->onnx_node->attribute, "axes");
    /* axes come from attribute (opset 11) or input (opset 13) */
    int64_t *axes = NULL;
    int64_t n_axes = 0;
    Onnx__TensorProto *i_axes = NULL;
    if (a_axes) {
        axes = a_axes->ints;
        n_axes = a_axes->n_ints;
    } else {
        i_axes = searchInputByName(ctx, 1);
        if (i_axes) { axes = i_axes->int64_data; n_axes = i_axes->n_int64_data; }
    }
    int64_t out_ndims = i_data->n_dims + n_axes;
    o_expanded->has_raw_data = 0;
    o_expanded->data_type = i_data->data_type;
    o_expanded->n_dims = out_ndims;
    o_expanded->dims = malloc(out_ndims * sizeof(int64_t));
    /* Normalize negative axes */
    int64_t *norm_axes = malloc(n_axes * sizeof(int64_t));
    for (int64_t i = 0; i < n_axes; i++) {
        norm_axes[i] = axes[i] < 0 ? axes[i] + out_ndims : axes[i];
    }
    /* Sort axes */
    for (int64_t i = 0; i < n_axes - 1; i++)
        for (int64_t j = i + 1; j < n_axes; j++)
            if (norm_axes[i] > norm_axes[j]) { int64_t t = norm_axes[i]; norm_axes[i] = norm_axes[j]; norm_axes[j] = t; }
    /* Build output dims: insert 1s at axes positions */
    int64_t src = 0;
    for (int64_t d = 0; d < out_ndims; d++) {
        int is_axis = 0;
        for (int64_t a = 0; a < n_axes; a++) { if (norm_axes[a] == d) { is_axis = 1; break; } }
        o_expanded->dims[d] = is_axis ? 1 : i_data->dims[src++];
    }
    free(norm_axes);
    mallocTensorData(o_expanded);
    ctx->executer = resolve_operator__ai_onnx__unsqueeze__11(ctx);
""",
    "exec_float": """
    Onnx__TensorProto *i_data = searchInputByName(ctx, 0);
    Onnx__TensorProto *o_expanded = searchOutputByName(ctx, 0);
    memcpy(o_expanded->float_data, i_data->float_data, o_expanded->n_float_data * sizeof(float));
""",
    "extra_inc": "<string.h>",
    "n_in": 1, "n_out": 1,
    "resolve_types": ["FLOAT", "DOUBLE", "INT32", "INT64"],
})

# ═══════════════════ Expand ═══════════════════
OPS.append({
    "Name": "Expand", "lc": "expand", "ver": 1,
    "prepare": """
    Onnx__TensorProto *i_input = searchInputByName(ctx, 0);
    Onnx__TensorProto *i_shape = searchInputByName(ctx, 1);
    Onnx__TensorProto *o_output = searchOutputByName(ctx, 0);
    o_output->has_raw_data = 0;
    o_output->data_type = i_input->data_type;
    /* Output shape is broadcast of input shape and target shape */
    int64_t nd;
    o_output->dims = broadcast_shape(i_input->n_dims, i_input->dims,
                                      i_shape->n_int64_data, i_shape->int64_data, &nd);
    o_output->n_dims = nd;
    mallocTensorData(o_output);
    ctx->executer = resolve_operator__ai_onnx__expand__13(ctx);
""",
    "exec_float": """
    Onnx__TensorProto *i_input = searchInputByName(ctx, 0);
    Onnx__TensorProto *o_output = searchOutputByName(ctx, 0);
    /* Broadcast input to output shape */
    broadcast_ctx bc;
    broadcast_init(&bc, i_input->n_dims, i_input->dims, o_output->n_dims, o_output->dims);
    for (int64_t i = 0; i < (int64_t)o_output->n_float_data; i++) {
        int64_t ai, bi;
        broadcast_indices(&bc, i, &ai, &bi);
        o_output->float_data[i] = i_input->float_data[ai];
    }
""",
    "extra_inc": "",
    "n_in": 2, "n_out": 1,
    "resolve_types": ["FLOAT", "INT32", "INT64"],
    "use_broadcast": True,
})

# ═══════════════════ Cast ═══════════════════
OPS.append({
    "Name": "Cast", "lc": "cast", "ver": 1,
    "prepare": """
    Onnx__TensorProto *i_input = searchInputByName(ctx, 0);
    Onnx__TensorProto *o_output = searchOutputByName(ctx, 0);
    Onnx__AttributeProto *a_to = searchAttributeNyName(
        ctx->onnx_node->n_attribute, ctx->onnx_node->attribute, "to");
    int64_t to_type = a_to ? a_to->i : i_input->data_type;
    o_output->has_raw_data = 0;
    o_output->data_type = to_type;
    o_output->n_dims = i_input->n_dims;
    o_output->dims = ARRAYDUP(i_input->dims, i_input->n_dims);
    mallocTensorData(o_output);
    /* Cast is special: we handle all type combos in one execute function */
    ctx->executer = (operator_executer)&execute_operator__ai_onnx__cast__13__T_tensor_float;
""",
    "exec_float": """
    Onnx__TensorProto *i_in = searchInputByName(ctx, 0);
    Onnx__TensorProto *o_out = searchOutputByName(ctx, 0);
    int64_t n = 1;
    for (int64_t d = 0; d < (int64_t)i_in->n_dims; d++) n *= i_in->dims[d];
    /* Get source data as doubles first */
    double *tmp = malloc(n * sizeof(double));
    switch (i_in->data_type) {
        case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
            for (int64_t i = 0; i < n; i++) tmp[i] = i_in->float_data[i]; break;
        case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE:
            for (int64_t i = 0; i < n; i++) tmp[i] = i_in->double_data[i]; break;
        case ONNX__TENSOR_PROTO__DATA_TYPE__INT32:
            for (int64_t i = 0; i < n; i++) tmp[i] = i_in->int32_data[i]; break;
        case ONNX__TENSOR_PROTO__DATA_TYPE__INT64:
            for (int64_t i = 0; i < n; i++) tmp[i] = (double)i_in->int64_data[i]; break;
        default: break;
    }
    /* Write to destination type */
    switch (o_out->data_type) {
        case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
            for (int64_t i = 0; i < n; i++) o_out->float_data[i] = (float)tmp[i]; break;
        case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE:
            for (int64_t i = 0; i < n; i++) o_out->double_data[i] = tmp[i]; break;
        case ONNX__TENSOR_PROTO__DATA_TYPE__INT32:
            for (int64_t i = 0; i < n; i++) o_out->int32_data[i] = (int32_t)tmp[i]; break;
        case ONNX__TENSOR_PROTO__DATA_TYPE__INT64:
            for (int64_t i = 0; i < n; i++) o_out->int64_data[i] = (int64_t)tmp[i]; break;
        default: break;
    }
    free(tmp);
""",
    "extra_inc": "",
    "n_in": 1, "n_out": 1,
    "resolve_types": ["FLOAT"],
    "no_type_dispatch": True,
})

# ═══════════════════ Concat ═══════════════════
OPS.append({
    "Name": "Concat", "lc": "concat", "ver": 1,
    "prepare": """
    Onnx__TensorProto *i_first = searchInputByName(ctx, 0);
    Onnx__TensorProto *o_concat = searchOutputByName(ctx, 0);
    Onnx__AttributeProto *a_axis = searchAttributeNyName(
        ctx->onnx_node->n_attribute, ctx->onnx_node->attribute, "axis");
    int64_t axis = a_axis ? a_axis->i : 0;
    if (axis < 0) axis += i_first->n_dims;
    o_concat->has_raw_data = 0;
    o_concat->data_type = i_first->data_type;
    o_concat->n_dims = i_first->n_dims;
    o_concat->dims = ARRAYDUP(i_first->dims, i_first->n_dims);
    /* Sum the axis dimension across all inputs */
    int64_t axis_total = i_first->dims[axis];
    for (int64_t inp = 1; inp < (int64_t)ctx->onnx_node->n_input; inp++) {
        Onnx__TensorProto *t = searchInputByName(ctx, inp);
        if (t) axis_total += t->dims[axis];
    }
    o_concat->dims[axis] = axis_total;
    mallocTensorData(o_concat);
    ctx->executer = resolve_operator__ai_onnx__concat__13(ctx);
""",
    "exec_float": """
    Onnx__TensorProto *o_concat = searchOutputByName(ctx, 0);
    Onnx__AttributeProto *a_axis = searchAttributeNyName(
        ctx->onnx_node->n_attribute, ctx->onnx_node->attribute, "axis");
    Onnx__TensorProto *i_first = searchInputByName(ctx, 0);
    int64_t axis = a_axis ? a_axis->i : 0;
    if (axis < 0) axis += (int64_t)i_first->n_dims;
    /* Compute outer_size (product of dims before axis) and inner_size (after axis) */
    int64_t outer = 1, inner = 1;
    for (int64_t d = 0; d < axis; d++) outer *= o_concat->dims[d];
    for (int64_t d = axis + 1; d < (int64_t)o_concat->n_dims; d++) inner *= o_concat->dims[d];
    int64_t out_offset = 0;
    for (int64_t inp = 0; inp < (int64_t)ctx->onnx_node->n_input; inp++) {
        Onnx__TensorProto *t = searchInputByName(ctx, inp);
        if (!t) continue;
        int64_t chunk = t->dims[axis] * inner;
        for (int64_t o = 0; o < outer; o++) {
            memcpy(&o_concat->float_data[o * o_concat->dims[axis] * inner + out_offset],
                   &t->float_data[o * chunk],
                   chunk * sizeof(float));
        }
        out_offset += chunk;
    }
""",
    "extra_inc": "<string.h>",
    "n_in": 2, "n_out": 1,
    "resolve_types": ["FLOAT", "DOUBLE", "INT32", "INT64"],
})

# ═══════════════════ Gather ═══════════════════
OPS.append({
    "Name": "Gather", "lc": "gather", "ver": 1,
    "prepare": """
    Onnx__TensorProto *i_data = searchInputByName(ctx, 0);
    Onnx__TensorProto *i_indices = searchInputByName(ctx, 1);
    Onnx__TensorProto *o_output = searchOutputByName(ctx, 0);
    Onnx__AttributeProto *a_axis = searchAttributeNyName(
        ctx->onnx_node->n_attribute, ctx->onnx_node->attribute, "axis");
    int64_t axis = a_axis ? a_axis->i : 0;
    if (axis < 0) axis += i_data->n_dims;
    /* output shape: data.shape[:axis] + indices.shape + data.shape[axis+1:] */
    int64_t out_ndims = i_data->n_dims - 1 + i_indices->n_dims;
    o_output->has_raw_data = 0;
    o_output->data_type = i_data->data_type;
    o_output->n_dims = out_ndims;
    o_output->dims = malloc(out_ndims * sizeof(int64_t));
    int64_t d = 0;
    for (int64_t i = 0; i < axis; i++) o_output->dims[d++] = i_data->dims[i];
    for (int64_t i = 0; i < (int64_t)i_indices->n_dims; i++) o_output->dims[d++] = i_indices->dims[i];
    for (int64_t i = axis + 1; i < (int64_t)i_data->n_dims; i++) o_output->dims[d++] = i_data->dims[i];
    mallocTensorData(o_output);
    ctx->executer = resolve_operator__ai_onnx__gather__13(ctx);
""",
    "exec_float": """
    Onnx__TensorProto *i_data = searchInputByName(ctx, 0);
    Onnx__TensorProto *i_indices = searchInputByName(ctx, 1);
    Onnx__TensorProto *o_output = searchOutputByName(ctx, 0);
    Onnx__AttributeProto *a_axis = searchAttributeNyName(
        ctx->onnx_node->n_attribute, ctx->onnx_node->attribute, "axis");
    int64_t axis = a_axis ? a_axis->i : 0;
    if (axis < 0) axis += i_data->n_dims;
    int64_t axis_dim = i_data->dims[axis];
    int64_t outer = 1, inner = 1;
    for (int64_t i = 0; i < axis; i++) outer *= i_data->dims[i];
    for (int64_t i = axis + 1; i < (int64_t)i_data->n_dims; i++) inner *= i_data->dims[i];
    int64_t n_idx = 1;
    for (int64_t i = 0; i < (int64_t)i_indices->n_dims; i++) n_idx *= i_indices->dims[i];
    int64_t out_pos = 0;
    for (int64_t o = 0; o < outer; o++) {
        for (int64_t idx = 0; idx < n_idx; idx++) {
            int64_t g = i_indices->int64_data ? i_indices->int64_data[idx] : i_indices->int32_data[idx];
            if (g < 0) g += axis_dim;
            for (int64_t in = 0; in < inner; in++) {
                o_output->float_data[out_pos++] = i_data->float_data[(o * axis_dim + g) * inner + in];
            }
        }
    }
""",
    "extra_inc": "",
    "n_in": 2, "n_out": 1,
    "resolve_types": ["FLOAT", "DOUBLE", "INT32", "INT64"],
})

# ═══════════════════ Slice ═══════════════════
OPS.append({
    "Name": "Slice", "lc": "slice", "ver": 1,
    "prepare": """
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
    context_operator__ai_onnx__slice__13 *op_ctx = malloc(sizeof(context_operator__ai_onnx__slice__13));
    op_ctx->starts = starts;
    op_ctx->steps = steps;
    free(ends);
    ctx->executer = resolve_operator__ai_onnx__slice__13(ctx);
    ctx->executer_context = op_ctx;
""",
    "exec_float": """
    Onnx__TensorProto *i_data = searchInputByName(ctx, 0);
    Onnx__TensorProto *o_output = searchOutputByName(ctx, 0);
    context_operator__ai_onnx__slice__13 *op_ctx = ctx->executer_context;
    int64_t ndims = i_data->n_dims;
    /* Compute strides for input */
    int64_t *in_strides = malloc(ndims * sizeof(int64_t));
    in_strides[ndims - 1] = 1;
    for (int64_t i = ndims - 2; i >= 0; i--) in_strides[i] = in_strides[i+1] * i_data->dims[i+1];
    /* Iterate output and map back to input */
    int64_t total = 1;
    for (int64_t i = 0; i < ndims; i++) total *= o_output->dims[i];
    int64_t *out_strides = malloc(ndims * sizeof(int64_t));
    out_strides[ndims - 1] = 1;
    for (int64_t i = ndims - 2; i >= 0; i--) out_strides[i] = out_strides[i+1] * o_output->dims[i+1];
    for (int64_t flat = 0; flat < total; flat++) {
        int64_t in_idx = 0;
        int64_t rem = flat;
        for (int64_t d = 0; d < ndims; d++) {
            int64_t coord = rem / out_strides[d];
            rem %= out_strides[d];
            in_idx += (op_ctx->starts[d] + coord * op_ctx->steps[d]) * in_strides[d];
        }
        o_output->float_data[flat] = i_data->float_data[in_idx];
    }
    free(in_strides);
    free(out_strides);
""",
    "extra_inc": "",
    "n_in": 3, "n_out": 1,
    "resolve_types": ["FLOAT", "DOUBLE", "INT32", "INT64"],
    "context_fields": "    int64_t *starts;\n    int64_t *steps;\n",
})

# ═══════════════════ ConstantOfShape ═══════════════════
OPS.append({
    "Name": "ConstantOfShape", "lc": "constantofshape", "ver": 1,
    "prepare": """
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
    ctx->executer = (operator_executer)&execute_operator__ai_onnx__constantofshape__9__T_tensor_float;
""",
    "exec_float": """
    /* Already filled in prepare */
""",
    "extra_inc": "",
    "n_in": 1, "n_out": 1,
    "resolve_types": ["FLOAT"],
    "no_type_dispatch": True,
})

# ═══════════════════ Range ═══════════════════
OPS.append({
    "Name": "Range", "lc": "range", "ver": 1,
    "prepare": """
    Onnx__TensorProto *i_start = searchInputByName(ctx, 0);
    Onnx__TensorProto *i_limit = searchInputByName(ctx, 1);
    Onnx__TensorProto *i_delta = searchInputByName(ctx, 2);
    Onnx__TensorProto *o_output = searchOutputByName(ctx, 0);
    o_output->has_raw_data = 0;
    o_output->data_type = i_start->data_type;
    o_output->n_dims = 1;
    o_output->dims = malloc(sizeof(int64_t));
    /* Compute number of elements: max(0, ceil((limit - start) / delta)) */
    if (i_start->data_type == ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT) {
        float s = i_start->float_data[0], l = i_limit->float_data[0], d = i_delta->float_data[0];
        int64_t n = (int64_t)ceilf((l - s) / d);
        if (n < 0) n = 0;
        o_output->dims[0] = n;
        mallocTensorData(o_output);
        for (int64_t i = 0; i < n; i++) o_output->float_data[i] = s + i * d;
    } else if (i_start->data_type == ONNX__TENSOR_PROTO__DATA_TYPE__INT32) {
        int32_t s = i_start->int32_data[0], l = i_limit->int32_data[0], d = i_delta->int32_data[0];
        int64_t n = (int64_t)((l - s + d - (d > 0 ? 1 : -1)) / d);
        if (n < 0) n = 0;
        o_output->dims[0] = n;
        mallocTensorData(o_output);
        for (int64_t i = 0; i < n; i++) o_output->int32_data[i] = s + (int32_t)i * d;
    } else if (i_start->data_type == ONNX__TENSOR_PROTO__DATA_TYPE__INT64) {
        int64_t s = i_start->int64_data[0], l = i_limit->int64_data[0], d = i_delta->int64_data[0];
        int64_t n = (l - s + d - (d > 0 ? 1 : -1)) / d;
        if (n < 0) n = 0;
        o_output->dims[0] = n;
        mallocTensorData(o_output);
        for (int64_t i = 0; i < n; i++) o_output->int64_data[i] = s + i * d;
    } else {
        o_output->dims[0] = 0;
        mallocTensorData(o_output);
    }
    ctx->executer = (operator_executer)&execute_operator__ai_onnx__range__11__T_tensor_float;
""",
    "exec_float": """
    /* Already computed in prepare */
""",
    "extra_inc": "<math.h>",
    "n_in": 3, "n_out": 1,
    "resolve_types": ["FLOAT"],
    "no_type_dispatch": True,
})

# ═══════════════════ ReduceMean ═══════════════════
OPS.append({
    "Name": "ReduceMean", "lc": "reducemean", "ver": 1,
    "prepare": """
    Onnx__TensorProto *i_data = searchInputByName(ctx, 0);
    Onnx__TensorProto *o_reduced = searchOutputByName(ctx, 0);
    Onnx__AttributeProto *a_axes = searchAttributeNyName(
        ctx->onnx_node->n_attribute, ctx->onnx_node->attribute, "axes");
    Onnx__AttributeProto *a_keepdims = searchAttributeNyName(
        ctx->onnx_node->n_attribute, ctx->onnx_node->attribute, "keepdims");
    int64_t keepdims = a_keepdims ? a_keepdims->i : 1;
    int64_t ndims = i_data->n_dims;
    /* Determine which axes to reduce */
    int8_t *reduce = calloc(ndims, sizeof(int8_t));
    if (a_axes && a_axes->n_ints > 0) {
        for (int64_t i = 0; i < (int64_t)a_axes->n_ints; i++) {
            int64_t ax = a_axes->ints[i];
            if (ax < 0) ax += ndims;
            reduce[ax] = 1;
        }
    } else {
        for (int64_t i = 0; i < ndims; i++) reduce[i] = 1;
    }
    o_reduced->has_raw_data = 0;
    o_reduced->data_type = i_data->data_type;
    if (keepdims) {
        o_reduced->n_dims = ndims;
        o_reduced->dims = malloc(ndims * sizeof(int64_t));
        for (int64_t i = 0; i < ndims; i++) o_reduced->dims[i] = reduce[i] ? 1 : i_data->dims[i];
    } else {
        int64_t out_nd = 0;
        for (int64_t i = 0; i < ndims; i++) if (!reduce[i]) out_nd++;
        if (out_nd == 0) out_nd = 1; /* scalar output */
        o_reduced->n_dims = out_nd;
        o_reduced->dims = malloc(out_nd * sizeof(int64_t));
        int64_t d = 0;
        for (int64_t i = 0; i < ndims; i++) if (!reduce[i]) o_reduced->dims[d++] = i_data->dims[i];
        if (d == 0) o_reduced->dims[0] = 1;
    }
    free(reduce);
    mallocTensorData(o_reduced);
    ctx->executer = resolve_operator__ai_onnx__reducemean__13(ctx);
""",
    "exec_float": """
    Onnx__TensorProto *i_data = searchInputByName(ctx, 0);
    Onnx__TensorProto *o_reduced = searchOutputByName(ctx, 0);
    Onnx__AttributeProto *a_axes = searchAttributeNyName(
        ctx->onnx_node->n_attribute, ctx->onnx_node->attribute, "axes");
    Onnx__AttributeProto *a_keepdims = searchAttributeNyName(
        ctx->onnx_node->n_attribute, ctx->onnx_node->attribute, "keepdims");
    int64_t keepdims = a_keepdims ? a_keepdims->i : 1;
    int64_t ndims = i_data->n_dims;
    int8_t *reduce = calloc(ndims, sizeof(int8_t));
    if (a_axes && a_axes->n_ints > 0) {
        for (int64_t i = 0; i < (int64_t)a_axes->n_ints; i++) {
            int64_t ax = a_axes->ints[i]; if (ax < 0) ax += ndims; reduce[ax] = 1;
        }
    } else { for (int64_t i = 0; i < ndims; i++) reduce[i] = 1; }
    /* Compute strides */
    int64_t *strides = malloc(ndims * sizeof(int64_t));
    strides[ndims - 1] = 1;
    for (int64_t i = ndims - 2; i >= 0; i--) strides[i] = strides[i+1] * i_data->dims[i+1];
    int64_t out_total = 1;
    for (int64_t i = 0; i < (int64_t)o_reduced->n_dims; i++) out_total *= o_reduced->dims[i];
    /* Zero output */
    for (int64_t i = 0; i < out_total; i++) o_reduced->float_data[i] = 0;
    /* Compute number of elements being reduced */
    int64_t reduce_count = 1;
    for (int64_t i = 0; i < ndims; i++) if (reduce[i]) reduce_count *= i_data->dims[i];
    /* Iterate all input elements, accumulate into output */
    int64_t in_total = 1;
    for (int64_t i = 0; i < ndims; i++) in_total *= i_data->dims[i];
    int64_t *out_strides = malloc(o_reduced->n_dims * sizeof(int64_t));
    if (o_reduced->n_dims > 0) {
        out_strides[o_reduced->n_dims - 1] = 1;
        for (int64_t i = (int64_t)o_reduced->n_dims - 2; i >= 0; i--)
            out_strides[i] = out_strides[i+1] * o_reduced->dims[i+1];
    }
    for (int64_t flat = 0; flat < in_total; flat++) {
        /* Compute multi-dim coords from flat input index */
        int64_t rem = flat, out_idx = 0, od = 0;
        for (int64_t d = 0; d < ndims; d++) {
            int64_t coord = rem / strides[d];
            rem %= strides[d];
            if (!reduce[d] || keepdims) {
                int64_t oc = reduce[d] ? 0 : coord;
                if (od < (int64_t)o_reduced->n_dims) out_idx += oc * out_strides[od];
                od++;
            }
        }
        o_reduced->float_data[out_idx] += i_data->float_data[flat];
    }
    for (int64_t i = 0; i < out_total; i++) o_reduced->float_data[i] /= reduce_count;
    free(reduce); free(strides); free(out_strides);
""",
    "extra_inc": "",
    "n_in": 1, "n_out": 1,
    "resolve_types": ["FLOAT", "DOUBLE"],
})


# ══════════════════════════════════════════════════════════════
#  CODE GENERATION
# ══════════════════════════════════════════════════════════════

TYPE_MAP = {
    "FLOAT": ("float", "float_data", "n_float_data"),
    "DOUBLE": ("double", "double_data", "n_double_data"),
    "INT32": ("int32_t", "int32_data", "n_int32_data"),
    "INT64": ("int64_t", "int64_data", "n_int64_data"),
}

def gen_header(op):
    name, lc, ver = op["Name"], op["lc"], op["ver"]
    ctx_fields = op.get("context_fields", "// no attributes\n")
    types = op.get("resolve_types", ["FLOAT"])
    lines = [f"#ifndef OPERATOR_OPERATOR__AI_ONNX__{lc.upper()}__{ver}_H"]
    lines.append(f"#define OPERATOR_OPERATOR__AI_ONNX__{lc.upper()}__{ver}_H")
    lines.append('#include "operators/operator.h"')
    lines.append('#include "operators/operator_stub.h"')
    lines.append('#include "operators/operator_info.h"')
    lines.append(f"\noperator_status prepare_operator__ai_onnx__{lc}__{ver}(node_context *ctx);")
    lines.append(f"extern operator_info info_operator__ai_onnx__{lc}__{ver};")
    lines.append(f"\ntypedef struct {{\n{ctx_fields}}} context_operator__ai_onnx__{lc}__{ver};")
    lines.append(f"\noperator_executer resolve_operator__ai_onnx__{lc}__{ver}(node_context *ctx);")
    for t in types:
        lines.append(f"\noperator_status execute_operator__ai_onnx__{lc}__{ver}__T_tensor_{t.lower()}(node_context *ctx);")
    lines.append(f"\n#endif")
    return "\n".join(lines) + "\n"

def gen_prepare(op):
    name, lc, ver = op["Name"], op["lc"], op["ver"]
    inc = op.get("extra_inc", "")
    inc_line = f'#include {inc}\n' if inc else ''
    use_bc = op.get("use_broadcast", False)
    bc_inc = '#include "broadcast_utils.h"\n' if use_bc else ''
    return f"""#include "operator__ai_onnx__{lc}__{ver}.h"
#include "tracing.h"
#include "utils.h"
{inc_line}{bc_inc}
operator_status
prepare_operator__ai_onnx__{lc}__{ver}(node_context *ctx)
{{
    TRACE_ENTRY(1);
    TRACE_NODE(2, true, ctx->onnx_node);
{op["prepare"]}
    TRACE_EXIT(1);
    return OP_OK;
}}
"""

def gen_resolve(op):
    name, lc, ver = op["Name"], op["lc"], op["ver"]
    types = op.get("resolve_types", ["FLOAT"])
    if op.get("no_type_dispatch"):
        return f"""#include "operator__ai_onnx__{lc}__{ver}.h"
#include "operators/operator_stub.h"
#include <stdint.h>
operator_executer resolve_operator__ai_onnx__{lc}__{ver}(node_context *ctx) {{
    return (operator_executer)&execute_operator__ai_onnx__{lc}__{ver}__T_tensor_float;
}}
"""
    cases = []
    for t in types:
        cases.append(f'    case ONNX__TENSOR_PROTO__DATA_TYPE__{t}: {{ executer = (operator_executer)&execute_operator__ai_onnx__{lc}__{ver}__T_tensor_{t.lower()}; break; }}')
    return f"""#include "operator__ai_onnx__{lc}__{ver}.h"
#include "operators/operator_stub.h"
#include <stdint.h>
#include <stdio.h>
operator_executer resolve_operator__ai_onnx__{lc}__{ver}(node_context *ctx) {{
    operator_executer executer = NULL;
    uint32_t T = 0;
    if (ctx->inputs[0]) {{ T = ctx->inputs[0]->data_type; }}
    switch (T) {{
    case 0:
{chr(10).join(cases)}
    default:
        fprintf(stderr, "no matching type for operator__ai_onnx__{lc}__{ver} with type %d\\n", T);
        break;
    }}
    if (!executer) {{ executer = &operator_stub; }}
    return executer;
}}
"""

def gen_execute(op, dtype="FLOAT"):
    lc, ver = op["lc"], op["ver"]
    inc = op.get("extra_inc", "")
    inc_line = f'#include {inc}\n' if inc else ''
    use_bc = op.get("use_broadcast", False)
    bc_inc = '#include "broadcast_utils.h"\n' if use_bc else ''

    # Get the code for this type
    if dtype == "FLOAT":
        code = op.get("exec_float", "")
    elif dtype == "DOUBLE":
        code = op.get("exec_double", "    /* not implemented */\n    return OP_ENOSYS;\n")
        if "not implemented" in code:
            return f"""#include "operator__ai_onnx__{lc}__{ver}.h"
#include "tracing.h"
#include "utils.h"
operator_status execute_operator__ai_onnx__{lc}__{ver}__T_tensor_{dtype.lower()}(node_context *ctx) {{
    return OP_ENOSYS;
}}
"""
    elif dtype == "INT32":
        code = op.get("exec_int32", None)
        if code is None:
            return f"""#include "operator__ai_onnx__{lc}__{ver}.h"
#include "tracing.h"
#include "utils.h"
operator_status execute_operator__ai_onnx__{lc}__{ver}__T_tensor_int32(node_context *ctx) {{
    return OP_ENOSYS;
}}
"""
    elif dtype == "INT64":
        code = op.get("exec_int64", None)
        if code is None:
            return f"""#include "operator__ai_onnx__{lc}__{ver}.h"
#include "tracing.h"
#include "utils.h"
operator_status execute_operator__ai_onnx__{lc}__{ver}__T_tensor_int64(node_context *ctx) {{
    return OP_ENOSYS;
}}
"""
    else:
        return f"""#include "operator__ai_onnx__{lc}__{ver}.h"
#include "tracing.h"
#include "utils.h"
operator_status execute_operator__ai_onnx__{lc}__{ver}__T_tensor_{dtype.lower()}(node_context *ctx) {{
    return OP_ENOSYS;
}}
"""

    return f"""#include "operator__ai_onnx__{lc}__{ver}.h"
#include "tracing.h"
#include "utils.h"
{inc_line}{bc_inc}#include <string.h>
#include <stdint.h>
operator_status
execute_operator__ai_onnx__{lc}__{ver}__T_tensor_{dtype.lower()}(node_context *ctx)
{{
    TRACE_ENTRY(1);
    TRACE_NODE(2, true, ctx->onnx_node);
{code}
    TRACE_EXIT(1);
    return OP_OK;
}}
"""

def gen_info(op):
    lc, ver = op["lc"], op["ver"]
    n_in, n_out = op["n_in"], op["n_out"]
    return f"""#include "operator__ai_onnx__{lc}__{ver}.h"
#include <stddef.h>
static uint32_t constraint_types[] = {{
    ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT,
    ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE,
    ONNX__TENSOR_PROTO__DATA_TYPE__INT32,
    ONNX__TENSOR_PROTO__DATA_TYPE__INT64,
}};
static operator_info_tensor inputs[] = {{
    {{ .name = "input", .optional = false, .constraint = "T", .n_types = 4, .types = constraint_types }},
    {{ .name = "input2", .optional = true, .constraint = "T", .n_types = 4, .types = constraint_types }},
    {{ .name = "input3", .optional = true, .constraint = "T", .n_types = 4, .types = constraint_types }},
    {{ .name = "input4", .optional = true, .constraint = "T", .n_types = 4, .types = constraint_types }},
    {{ .name = "input5", .optional = true, .constraint = "T", .n_types = 4, .types = constraint_types }},
}};
static operator_info_tensor outputs[] = {{
    {{ .name = "output", .optional = false, .constraint = "T", .n_types = 4, .types = constraint_types }},
}};
static operator_info_constraint constraints[] = {{
    {{ .name = "T" }},
}};
operator_info info_operator__ai_onnx__{lc}__{ver} = {{
    .name = "{op['Name']}",
    .range_input = {{ {n_in}, 5 }},
    .range_output = {{ {n_out}, {n_out} }},
    .n_attribute = 0,
    .attribute = NULL,
    .n_input = {n_in},
    .input = inputs,
    .n_output = {n_out},
    .output = outputs,
    .n_constraint = 1,
    .constraint = constraints,
}};
"""

def gen_opversion(op):
    lc, ver = op["lc"], op["ver"]
    return f"""#include "operator__ai_onnx__{lc}__{ver}.h"
#include "operators/operator_set.h"
operator_set_opversion opversion_operator__ai_onnx__{lc}__{ver} = {{
    .version = {ver},
    .preparer = prepare_operator__ai_onnx__{lc}__{ver},
    .info = &info_operator__ai_onnx__{lc}__{ver}
}};
"""

def gen_opname(op):
    lc, ver = op["lc"], op["ver"]
    return f"""#include "operators/operator_set.h"
extern operator_set_opversion opversion_operator__ai_onnx__{lc}__{ver};
operator_set_opname opname_operator__ai_onnx__{lc} = {{
    .name = "{op['Name']}",
    .opversions = {{
        &opversion_operator__ai_onnx__{lc}__{ver},
        NULL
    }}
}};
"""

def gen_free(op):
    lc, ver = op["lc"], op["ver"]
    free_code = ""
    if op.get("context_fields"):
        free_code = f"""    context_operator__ai_onnx__{lc}__{ver} *op_ctx = ctx->executer_context;
    if (op_ctx) {{
        free(op_ctx->starts);
        free(op_ctx->steps);
        free(op_ctx);
    }}"""
    return f"""#include "operator__ai_onnx__{lc}__{ver}.h"
#include "tracing.h"
#include "utils.h"
#include <stdlib.h>
void free_operator__ai_onnx__{lc}__{ver}(node_context *ctx) {{
    TRACE_ENTRY(1);
{free_code}
    TRACE_EXIT(1);
}}
"""

# ── Generate all files ──
for op in OPS:
    name, lc, ver = op["Name"], op["lc"], op["ver"]
    types = op.get("resolve_types", ["FLOAT"])
    dirpath = os.path.join(BASE, name, str(ver))
    mkdir(dirpath)

    # Header
    write(os.path.join(dirpath, f"operator__ai_onnx__{lc}__{ver}.h"), gen_header(op))
    # Prepare
    write(os.path.join(dirpath, f"prepare_operator__ai_onnx__{lc}__{ver}.c"), gen_prepare(op))
    # Resolve
    write(os.path.join(dirpath, f"resolve_operator__ai_onnx__{lc}__{ver}.c"), gen_resolve(op))
    # Execute files for each type
    for t in types:
        write(os.path.join(dirpath, f"execute_operator__ai_onnx__{lc}__{ver}__T_tensor_{t.lower()}.c"), gen_execute(op, t))
    # Info
    write(os.path.join(dirpath, f"info_operator__ai_onnx__{lc}__{ver}.c"), gen_info(op))
    # Opversion
    write(os.path.join(dirpath, f"opversion_operator__ai_onnx__{lc}__{ver}.c"), gen_opversion(op))
    # Free
    write(os.path.join(dirpath, f"free_operator__ai_onnx__{lc}__{ver}.c"), gen_free(op))
    # Opname (top level)
    write(os.path.join(BASE, name, f"opname_operator__ai_onnx__{lc}.c"), gen_opname(op))

    print(f"  Generated: {name} v{ver} ({len(types)} types)")

print(f"\nGenerated {len(OPS)} operators")
