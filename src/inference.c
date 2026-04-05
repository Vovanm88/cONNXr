#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
static volatile int _current_node = -1;
static volatile const char *_current_op = "?";
static void segv_handler(int sig) {
    fprintf(stderr, "\n*** SIGSEGV at node %d op=%s ***\n", _current_node, _current_op);
    _exit(139);
}
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
  _populatedIdx = -1;
  /* Convert raw_data for all initializers up front */
  for (int i = 0; i < model->graph->n_initializer; i++) {
    if (model->graph->initializer[i]->has_raw_data) {
      convertRawDataOfTensorProto(model->graph->initializer[i]);
    }
  }
  printf("resolve: converted %zu initializers, processing %zu nodes\n", model->graph->n_initializer, model->graph->n_node);
  fflush(stdout);
  for (int nodeIdx = 0; nodeIdx < model->graph->n_node; nodeIdx++)
  {
    all_context[nodeIdx].onnx_node = model->graph->node[nodeIdx];
    if (nodeIdx % 100 == 0 || nodeIdx < 5) { printf("resolve: node %d/%zu op=%s\n", nodeIdx, model->graph->n_node, model->graph->node[nodeIdx]->op_type); fflush(stdout); }


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
      /* data_type is set by each operator's prepare function */
    }
    /*** Prototyping ***/
    // Check model->opset_import->has_version must be True
    // More than 1 opset can be imported. Iterate n_opset_import
    // model->opset_import[0]->version
    // TODO Hackish temporal solution. Use opset 12.
    
    size_t version = 23;
    char *op_type = model->graph->node[nodeIdx]->op_type;
    if (nodeIdx % 100 ==0)
      printf("resolve: node %d/%zu op=%s\n", nodeIdx, model->graph->n_node, op_type ? op_type : "NULL");
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
  fflush(stdout);
}

Onnx__TensorProto** inference(Onnx__ModelProto *model, Onnx__TensorProto **inputs, int nInputs)
{
  signal(SIGSEGV, segv_handler);
  printf("inference: running %zu nodes\n", model->graph->n_node);
  fflush(stdout);
  for (int nodeIdx = 0; nodeIdx < model->graph->n_node; nodeIdx++)
  {
    printf("inference: node %d/%zu op=%s\n", nodeIdx, model->graph->n_node, model->graph->node[nodeIdx]->op_type); fflush(stdout);
    if (!all_context[nodeIdx].executer) { fprintf(stderr, "FATAL: node %d has NULL executer\n", nodeIdx); return 0; }
    _current_node = nodeIdx;
    _current_op = model->graph->node[nodeIdx]->op_type;

    operator_status st = all_context[nodeIdx].executer(&all_context[nodeIdx]);
    if (st != OP_OK) {
        fprintf(stderr, "ERROR: node %d op=%s returned status %d\n", nodeIdx, model->graph->node[nodeIdx]->op_type, st);
    }
    
    /* Print output dimensions */
    for (int oi = 0; oi < model->graph->node[nodeIdx]->n_output; oi++) {
        Onnx__TensorProto *out = all_context[nodeIdx].outputs[oi];
        if (out) {
            printf("  output[%d] '%s': n_dims=%zu", oi, out->name, out->n_dims);
            printf(" | [");
            if (out->n_dims>0){
              for (size_t di = 0; di < out->n_dims; di++) {
                  if(di+1<out->n_dims)
                    printf("%ld, ", out->dims[di]);
                  else
                    printf("%ld", out->dims[di]);
              }
            }
            printf("]\n");
        }
    }
    /* Validate output dims aren't corrupted */
    for (int oi = 0; oi < model->graph->node[nodeIdx]->n_output; oi++) {
        Onnx__TensorProto *out = all_context[nodeIdx].outputs[oi];
        if (out) {
            for (size_t di = 0; di < out->n_dims; di++) {
                if (out->dims[di] < 0 || out->dims[di] > 1000000000LL) {
                    out->dims[di] = 0; /* clamp corrupt dim */
                }
            }
        }
    }
  }

  printf("inference: done\n");
  fflush(stdout);
  return 0;
}
