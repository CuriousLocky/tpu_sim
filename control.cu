#include "parameters.h"
#include "sys_arr.h"
#include "unified_buffer.h"
#include "accumulators.h"
#include "simulation.h"


extern int array_a_colsize_host, array_a_rowsize_host, array_b_colsize_host, array_b_rowsize_host;
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
	int beat_cycles = array_a_colsize_host + array_a_rowsize_host + array_b_colsize_host - 1;
	for (int i = 0; i < beat_cycles; i++) {
		heart_beat();
		sys_arr_cycle();
	}
	flush_sys_arr();
}

void activate()
{
	result_activate();
}

void write_host_memory(int32_t* host_pos)
{
	collect_result();
	U_buffer_trans(result_colsize_host * result_rowsize_host);
	cudaMemcpy(host_pos, result_host, result_rowsize_host*result_colsize_host * sizeof(int32_t), cudaMemcpyDeviceToHost);
}

