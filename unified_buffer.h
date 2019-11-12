#pragma once
#include "parameters.h"

void u_buffer_ini();
void u_buffer_free();
void setupArray_a(char* array_a_host_input, int array_a_colsize_host_input, int array_a_rowsize_host_input);
void setupArray_b(char* array_b_host_input, int array_b_colsize_host_input, int array_b_rowsize_host_input);
void setupResult(char op);
void result_activate();
__device__ char feed_data_h(int row_num);
__device__ char feed_data_v(int col_num);
__global__ void _collect_result();
int array_a_colsize_host;
int array_a_rowsize_host;
int array_b_colsize_host;
int array_b_rowsize_host;
int result_colsize_host;
int result_rowsize_host;
char* result_host;
