#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "onnx.pb-c.h"
#include "utils.h"
#include "tracing.h"
#include "inference.h"
#include "operators/operator_set.h"

// Won't be global in the future
node_context all_context[MAX_NUM_OF_NODES];
int _populatedIdx = 0;

void resolve(Onnx__ModelProto *model,
             Onnx__TensorProto **inputs,
             int nInputs)
{
  TRACE_ENTRY(1);
  printf("resolve: start\n");
  /* Resolving operators and input/outputs. Has to be moved outside of infeference */
  _populatedIdx = -1;

  TRACE_FATAL(0, model->graph->n_node > MAX_NUM_OF_NODES, "The number of nodes of the model is greater than the hardcoded one");
  printf("I\n");
  printf("resolve: processing %zu nodes\n", model->graph->n_node);
  for (int nodeIdx = 0; nodeIdx < model->graph->n_node; nodeIdx++)
  {
    if (nodeIdx % 500 == 0 || nodeIdx > 7415) printf("resolve: node %d\n", nodeIdx);
    all_context[nodeIdx].onnx_node = model->graph->node[nodeIdx];


    // Search the inputs for a node
    
    all_context[nodeIdx].inputs = malloc(sizeof(Onnx__TensorProto*) * model->graph->node[nodeIdx]->n_input);
    if (!all_context[nodeIdx].inputs) { fprintf(stderr, "OOM: inputs at node %d\n", nodeIdx); return; }
    for (int i = 0; i < model->graph->node[nodeIdx]->n_input; i++)
    {
      all_context[nodeIdx].inputs[i] = searchTensorProtoByName(model, inputs, nInputs, model->graph->node[nodeIdx]->input[i]);
      if (all_context[nodeIdx].inputs[i] && all_context[nodeIdx].inputs[i]->has_raw_data){
        TRACE(1, true, "input %s has raw data", all_context[nodeIdx].inputs[i]->name);
        convertRawDataOfTensorProto(all_context[nodeIdx].inputs[i]);
      }
    }
    all_context[nodeIdx].outputs = malloc(sizeof(Onnx__TensorProto*) * model->graph->node[nodeIdx]->n_output);
    if (!all_context[nodeIdx].outputs) { fprintf(stderr, "OOM: outputs at node %d\n", nodeIdx); return; }
    for (int i = 0; i < model->graph->node[nodeIdx]->n_output; i++)
    {
      all_context[nodeIdx].outputs[i] = malloc(sizeof(Onnx__TensorProto));
      if (!all_context[nodeIdx].outputs[i]) { fprintf(stderr, "OOM: output tensor at node %d\n", nodeIdx); return; }
      init_tensor_proto(all_context[nodeIdx].outputs[i]);
      all_context[nodeIdx].outputs[i]->name = strdup(model->graph->node[nodeIdx]->output[i]);
      all_context[nodeIdx].outputs[i]->data_type = 1;
    }
    /*** Prototyping ***/
    // Check model->opset_import->has_version must be True
    // More than 1 opset can be imported. Iterate n_opset_import
    // model->opset_import[0]->version
    // TODO Hackish temporal solution. Use opset 12.
    
    size_t version = 23;
    char *op_type = model->graph->node[nodeIdx]->op_type;
    if (!op_type) {
        fprintf(stderr, "FATAL: node %d has NULL op_type\n", nodeIdx);
        return;
    }
    /* Validate pointer is readable */
    volatile char test_byte = op_type[0];
    (void)test_byte;
    printf("searching for preparer of operator %s version %zu\n", op_type, version);
    operator_preparer prepare = operator_set_find_preparer(model->graph->node[nodeIdx]->op_type, version);
    
    if (!prepare) {
        fprintf(stderr, "FATAL: No prepare for operator '%s' v%zu at node %d\n", op_type, version, nodeIdx);
        return;
    }
    operator_status prep_status = prepare(&all_context[nodeIdx]);
    if (prep_status != OP_OK) {
        fprintf(stderr, "WARNING: prepare returned %d for operator '%s' at node %d\n", prep_status, op_type, nodeIdx);
    }
    _populatedIdx++;
  }
  printf("resolve: done (%d nodes)\n", _populatedIdx + 1);
  TRACE_EXIT(1);
}

Onnx__TensorProto** inference(Onnx__ModelProto *model, Onnx__TensorProto **inputs, int nInputs)
{
  TRACE_ENTRY(1);
  printf("inference: running %zu nodes\n", model->graph->n_node);
  for (int nodeIdx = 0; nodeIdx < model->graph->n_node; nodeIdx++)
  {
    if (nodeIdx % 500 == 0) printf("inference: node %d/%zu op=%s\n", nodeIdx, model->graph->n_node, model->graph->node[nodeIdx]->op_type);
    TRACE(1, true, "Running node %d, operator=%s", nodeIdx, model->graph->node[nodeIdx]->op_type);
    all_context[nodeIdx].executer(&all_context[nodeIdx]);
  }

  printf("inference: done\n");
  TRACE_EXIT(1);
  return 0;
}
