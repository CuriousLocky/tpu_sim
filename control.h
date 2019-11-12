#pragma once
void tpu_ini();
void tpu_free();
void read_host_memory(char* data_arr, int data_arr_rowsize, int data_arr_colsize);
void read_weights(char* weight_arr, int weight_arr_rowsize, int weight_arr_colsize);
void matrix_multiply();
void activate();
void write_host_memory(char* host_pos);