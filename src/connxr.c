#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <dirent.h>
#include <sys/stat.h>

#include "onnx.pb-c.h"
#include "tracing.h"
#include "inference.h"
#include "utils.h"

#define MAX_INPUTS 100

int main(int argc, char **argv){

  /* Modified to support multiple inputs from a directory containing .pb files.
     Usage: connxr model.onnx input_dir/ [--dump-file]
     Where input_dir contains .pb files named after the input tensor names.
  */
  if (argc <  3){
    fprintf(stderr, "not enough arguments! %s model.onnx input_dir/ [--dump-file]\n", argv[0]);
    return 1;
  }
  printf("Loading model %s...", argv[1]);
  Onnx__ModelProto *model = openOnnxFile(argv[1]);
  if (model != NULL){printf("ok!\n");}
  // TRACE_MODEL(2, true, model);

  // Load all .pb files from input directory
  char *input_dir = argv[2];
  DIR *dir = opendir(input_dir);
  if (!dir) {
    fprintf(stderr, "Cannot open input directory %s\n", input_dir);
    return 1;
  }

  Onnx__TensorProto *inputs[MAX_INPUTS];
  int num_inputs = 0;

  struct dirent *entry;
  while ((entry = readdir(dir)) != NULL && num_inputs < MAX_INPUTS) {
    if (strstr(entry->d_name, ".pb") != NULL) {
      char filepath[1024];
      snprintf(filepath, sizeof(filepath), "%s/%s", input_dir, entry->d_name);
      printf("Loading input %s...", filepath);
      Onnx__TensorProto *tensor = openTensorProtoFile(filepath);
      if (tensor != NULL) {
        printf("ok!\n");
        // TRACE_TENSOR(2, true, tensor);
        convertRawDataOfTensorProto(tensor);
        // Set name from filename (remove .pb extension)
        char *name = strdup(entry->d_name);
        char *dot = strrchr(name, '.');
        if (dot) *dot = '\0';
        tensor->name = name;
        inputs[num_inputs++] = tensor;
      } else {
        printf("failed!\n");
      }
    }
  }
  closedir(dir);

  if (num_inputs == 0) {
    fprintf(stderr, "No .pb files found in %s\n", input_dir);
    return 1;
  }

  printf("Loaded %d inputs\n", num_inputs);

  clock_t start, end;
  double cpu_time_used;

  printf("Resolving model...\n");
  printf("Before resolve\n");
  resolve(model, inputs, num_inputs);
  printf("After resolve\n");
  printf("Running inference on %s model...\n", model->graph->name);
  printf("Before inference\n");
  start = clock();
  inference(model, inputs, num_inputs);
  printf("After inference\n");
  end = clock();
  printf("finished!\n");

  // TODO Is CLOCKS_PER_SEC ok to use?
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  printf("Predicted in %f cycles or %f seconds\n", (double) (end - start), cpu_time_used);

  /* Print the last output which should be the model output */
  for (int i = 0; i < all_context[_populatedIdx].outputs[0]->n_float_data; i++){
    //printf("n_float_data[%d] = %f\n", i, all_context[_populatedIdx].outputs[0]->float_data[i]);
  }

  if ((argc == 4) && !strcmp(argv[3], "--dump-file")){
    printf("Writing dump file with intermediate outputs\n");
    //int max_print = 10;
    FILE *fp = fopen("dump.txt", "w+");
    for (int i = 0; i < _populatedIdx + 1; i++){
      fprintf(fp, "name=%s\n", all_context[i].outputs[0]->name);
      fprintf(fp, "shape=");
      for (int dim_index = 0; dim_index < all_context[i].outputs[0]->n_dims; dim_index++){
        fprintf(fp, "%" PRId64 ",", all_context[i].outputs[0]->dims[dim_index]);
      }
      fprintf(fp, "\n");
      //int float_to_print = all_context[i].outputs[0]->n_float_data > max_print ? max_print : all_context[i].outputs[0]->n_float_data;
      fprintf(fp, "tensor=");
      /* TODO: Just implemented for float */
      for (int data_index = 0; data_index < all_context[i].outputs[0]->n_float_data; data_index++){
        fprintf(fp, "%f,", all_context[i].outputs[0]->float_data[data_index]);
      }
      fprintf(fp, "\n");
    }
    fclose(fp);
  }
}
