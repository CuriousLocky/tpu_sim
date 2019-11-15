#pragma once
#include "inttypes.h"
void tpu_ini();
void tpu_free();
void read_host_memory(char* data_arr, int data_arr_colsize, int data_arr_rowsize);
void read_weights(char* weight_arr, int weight_arr_colsize, int weight_arr_rowsize);
void matrix_multiply();
void matrix_convolution();
void activate();
void write_host_memory(int32_t* host_pos);