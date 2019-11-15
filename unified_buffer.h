﻿#pragma once
#include "parameters.h"

void u_buffer_ini();
void u_buffer_free();
void setupArray_a(char* array_a_host_input, int array_a_colsize_host_input, int array_a_rowsize_host_input);
void setupArray_b(char* array_b_host_input, int array_b_colsize_host_input, int array_b_rowsize_host_input);
void setupResult(char op);
void collect_result();
void change_size_a(int new_rowsize, int new_colsize);
void change_size_b(int new_rowsize, int new_colsize);
void change_size_result(int new_rowsize, int new_colsize);

__device__ char feed_data_h(int row_num);
__device__ char feed_data_v(int col_num);
__global__ void _collect_result();