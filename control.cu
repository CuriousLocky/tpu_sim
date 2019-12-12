#include "parameters.h"
#include "sys_arr.h"
#include "unified_buffer.h"
#include "accumulators.h"
#include "simulation.h"

extern int array_a_colsize_host, array_a_rowsize_host, array_b_colsize_host, array_b_rowsize_host;
extern char* array_a_host, *array_b_host;
extern __device__ char* array_a, *array_b;
extern int32_t* result_host;
extern int result_rowsize_host, result_colsize_host;


void tpu_ini()
{
	sys_arr_ini();
	u_buffer_ini();
	accumulators_ini();
}

void tpu_free()
{
	sys_arr_free();
	u_buffer_free();
	accumulators_free();
}

void read_host_memory(char* data_arr, int data_arr_colsize, int data_arr_rowsize)
{
	U_buffer_trans(data_arr_colsize * data_arr_rowsize);
	setupArray_a(data_arr, data_arr_colsize, data_arr_rowsize);
}

void read_weights(char* weight_arr, int weight_arr_colsize, int weight_arr_rowsize)
{
	U_buffer_trans(weight_arr_colsize * weight_arr_rowsize);
	setupArray_b(weight_arr, weight_arr_colsize, weight_arr_rowsize);
}

void matrix_multiply()
{
	setupResult(MAT_MUL);
	reset_count << <1, 1 >> > ();
	flush_accumulators();
	flush_sys_arr();
	weight_load(array_b_colsize_host * array_b_rowsize_host);
	in_buffer_load(array_a_colsize_host * array_a_rowsize_host);
	int beat_cycles = array_a_colsize_host + array_a_rowsize_host + array_b_rowsize_host - 1;
	for (int i = 0; i < beat_cycles; i++) {
		weight_load(256);
		in_buffer_load(256);
		heart_beat();
		sys_arr_cycle();
	}
}

void matrix_convolution()
{
	if (array_b_colsize_host > array_a_colsize_host || array_b_rowsize_host > array_a_rowsize_host) {
		printf("cannot perform convolution, size incompatible\n");
		return;
	}
	if (array_b_colsize_host == array_a_colsize_host && array_b_colsize_host == array_a_colsize_host) {
		change_size_a(array_a_colsize_host*array_a_rowsize_host, 1);
		change_size_b(1, array_b_colsize_host*array_b_rowsize_host);
		matrix_multiply();
		return;
	}
	int new_a_rowsize = array_a_colsize_host * array_a_rowsize_host;
	int new_a_colsize = (array_a_colsize_host - array_b_colsize_host + 1)*(array_a_rowsize_host - array_b_rowsize_host + 1);
	if (new_a_colsize > sys_array_size) {
		printf("cannot perform convolution, size too large\n");
		return;
	}
	int result_rowsize = array_a_rowsize_host - array_b_rowsize_host + 1;
	int result_colsize = array_a_colsize_host - array_b_colsize_host + 1;
	char* activation_buffer = (char*)malloc(sizeof(char)*array_b_rowsize_host * array_b_colsize_host);
	cudaMemcpy(activation_buffer, array_b_host, sizeof(char)*array_b_colsize_host*array_b_rowsize_host, cudaMemcpyDeviceToHost);
	char* b_buffer = (char*)malloc(sizeof(char)*new_a_rowsize);
	cudaMemcpy(b_buffer, array_a_host, sizeof(char)*new_a_rowsize, cudaMemcpyDeviceToHost);
	char* a_buffer = (char*)malloc(sizeof(char)*new_a_rowsize);
	for (int i = 0; i < result_colsize; i++) {
		for (int j = 0; j < result_rowsize; j++) {
			memset(a_buffer, 0, sizeof(char)*new_a_rowsize);
			for (int k1 = 0; k1 < array_b_colsize_host; k1++) {
				for (int k2 = 0; k2 < array_b_rowsize_host; k2++) {
					*(a_buffer + i * array_a_rowsize_host + j + k1 * array_a_rowsize_host + k2) = activation_buffer[k1*array_b_rowsize_host + k2];
				}
			}
			cudaMemcpy(array_a_host + (i *result_rowsize+j)*new_a_rowsize, a_buffer, sizeof(char)*new_a_rowsize, cudaMemcpyHostToDevice);
		}
	}
	change_size_a(new_a_rowsize, new_a_colsize);
	setupArray_b(b_buffer, new_a_rowsize, 1);
	free(b_buffer);
	free(a_buffer);
	free(activation_buffer);
	matrix_multiply();
	change_size_result(1, result_colsize*result_rowsize);
}

void activate()
{
	result_activate();
}

void write_host_memory(int32_t* host_pos)
{
	collect_result();
	in_buffer_load(result_colsize_host * result_rowsize_host*sizeof(uint32_t));
	flush_accumulators();
	U_buffer_trans(result_colsize_host * result_rowsize_host);
	cudaMemcpy(host_pos, result_host, result_rowsize_host*result_colsize_host * sizeof(int32_t), cudaMemcpyDeviceToHost);
}