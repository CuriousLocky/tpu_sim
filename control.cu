#define sys_array_size 256
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

extern void sys_arr_ini();
extern void u_buffer_ini();
extern void sys_arr_free();
extern void u_buffer_free();
extern void setupArray_a(char*, int, int);
extern void setupArray_b(char*, int, int);
extern int array_a_colsize_host, array_a_rowsize_host, array_b_colsize_host, array_b_rowsize_host;
extern void heart_beat();
extern void result_shift();
extern void flush_sys_arr();
extern void result_activate();
extern char* result_host;
extern int result_rowsize_host, result_colsize_host;

void tpu_ini()
{
	sys_arr_ini();
	u_buffer_ini();
}

void tpu_free()
{
	sys_arr_free();
	u_buffer_free();
}

void read_host_memory(char* data_arr, int data_arr_colsize, int data_arr_rowsize)
{
	setupArray_a(data_arr, data_arr_colsize, data_arr_rowsize);
}

void read_weights(char* weight_arr, int weight_arr_colsize, int weight_arr_rowsize)
{
	setupArray_b(weight_arr, weight_arr_colsize, weight_arr_rowsize);
}

void matrix_multiply()
{
	int beat_cycles = array_a_colsize_host + array_a_rowsize_host + array_b_colsize_host - 2;
	for (int i = 0; i < beat_cycles; i++)
		heart_beat();
	for (int i = 0; i < sys_array_size; i++)
		result_shift();
	flush_sys_arr();
}

void activate()
{
	result_activate();
}

void write_host_memory(char* host_pos)
{
	cudaMemcpy(host_pos, result_host, result_rowsize_host*result_colsize_host * sizeof(char), cudaMemcpyHostToDevice);
}

