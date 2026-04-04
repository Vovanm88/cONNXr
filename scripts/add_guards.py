#!/usr/bin/env python3
"""Add empty-tensor guards to all operator executors."""
import os, re, glob

GUARD_INCLUDE = '#include "op_utils.h"\n'

# Pattern: after the last searchInputByName/searchOutputByName call,
# add a guard that checks if output is empty
def add_guard(filepath):
    with open(filepath) as f:
        content = f.read()

    # Skip if already guarded
    if 'tensor_is_empty' in content or 'op_utils.h' in content:
        return False

    # Skip if it's a delegating executor (calls another executor)
    if re.search(r'return execute_operator__', content):
        return False

    # Skip stubs
    if 'return OP_ENOSYS;' in content and content.count('\n') < 10:
        return False

    # Find the last searchOutputByName line
    lines = content.split('\n')
    insert_idx = -1
    for i, line in enumerate(lines):
        if 'searchOutputByName' in line or 'searchInputByName' in line:
            insert_idx = i + 1

    if insert_idx == -1:
        return False  # No search calls found

    # Skip past any existing null checks right after
    while insert_idx < len(lines) and ('if (!' in lines[insert_idx] or lines[insert_idx].strip() == ''):
        insert_idx += 1

    # Determine what to check based on operator type
    op_name = os.path.basename(filepath)

    # Find output variable name
    out_var = None
    for line in lines:
        m = re.search(r'Onnx__TensorProto \*(\w+)\s*=\s*searchOutputByName', line)
        if m:
            out_var = m.group(1)
            break

    if not out_var:
        return False

    # Build guard
    guard = f'    /* Skip if output tensor is empty (has 0-dim) */\n'
    guard += f'    if (tensor_is_empty({out_var})) return OP_OK;\n'

    # Add include if not present
    if GUARD_INCLUDE.strip() not in content:
        # Add after the last existing #include
        last_include = -1
        for i, line in enumerate(lines):
            if line.startswith('#include'):
                last_include = i
        if last_include >= 0:
            lines.insert(last_include + 1, GUARD_INCLUDE.strip())
            insert_idx += 1  # shift

    lines.insert(insert_idx, guard)

    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))
    return True

# Process all executor files
count = 0
for pattern in ['src/operators/ai.onnx/*/[0-9]*/execute_operator__*__T_tensor_float.c',
                 'src/operators/ai.onnx/*/[0-9][0-9]*/execute_operator__*__T_tensor_float.c']:
    for filepath in sorted(glob.glob(pattern)):
        if add_guard(filepath):
            print(f'  Guarded: {os.path.basename(filepath)}')
            count += 1

# Also guard specific non-float executors that have real implementations
for filepath in sorted(glob.glob('src/operators/ai.onnx/*/[0-9]*/execute_operator__*__T_tensor_int64.c')):
    if 'return execute_operator' not in open(filepath).read():
        if add_guard(filepath):
            print(f'  Guarded: {os.path.basename(filepath)}')
            count += 1

# Guard the And v1 executor
for filepath in sorted(glob.glob('src/operators/ai.onnx/And/1/execute_operator__*__T_tensor_float.c')):
    if add_guard(filepath):
        print(f'  Guarded: {os.path.basename(filepath)}')
        count += 1

print(f'\nGuarded {count} executor files')
