#include "parameters.h"

#define U_BUFFER_SIZE 96*1024*256

__device__ char* u_buffer;
__device__ char* array_a;
char* array_a_host;
__device__ char* array_b;
char* array_b_host;
__device__ int array_a_colsize;
int array_a_colsize_host;
__device__ int array_a_rowsize;
int array_a_rowsize_host;
__device__ int array_b_colsize;
int array_b_colsize_host;
__device__ int array_b_rowsize;
int array_b_rowsize_host;

__device__ int32_t* result;
int32_t* result_host;
__device__ int result_colsize;
int result_colsize_host;
__device__ int result_rowsize;
int result_rowsize_host;

__device__ int cycle_count = 0;

extern __device__ Cell* sys_arr;
extern __device__ uint32_t* accumulators;

void u_buffer_ini()
{
	cudaMalloc((void**)& array_a_host, sizeof(char) * U_BUFFER_SIZE);
	cudaMemcpyToSymbol(array_a, &array_a_host, sizeof(char*));
	cudaMemcpyToSymbol(u_buffer, &array_a_host, sizeof(char*));
	cudaMemset(array_a_host, 0, sizeof(char)*U_BUFFER_SIZE);
}

void u_buffer_free()
{
	cudaFree(array_a_host);
}

void setupArray_a(char* array_a_host_input, int array_a_colsize_host_input, int array_a_rowsize_host_input)
{
	if (array_a_colsize_host_input * array_a_rowsize_host_input >= U_BUFFER_SIZE) {
		printf("array a size is too big\n");
		exit(0);
	}
	
	array_a_colsize_host = array_a_colsize_host_input;
	array_a_rowsize_host = array_a_rowsize_host_input;
	cudaMemcpy(array_a_host, array_a_host_input, sizeof(char)*array_a_colsize_host_input*array_a_rowsize_host_input, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(array_a_colsize, &array_a_colsize_host, sizeof(int));
	cudaMemcpyToSymbol(array_a_rowsize, &array_a_rowsize_host, sizeof(int));
}

void setupArray_b(char* array_b_host_input, int array_b_colsize_host_input, int array_b_rowsize_host_input)
{
	if (array_b_colsize_host_input * array_b_rowsize_host_input >= U_BUFFER_SIZE-array_a_colsize_host*array_a_rowsize_host) {
		printf("array b size is too big\n");
		exit(0);
	}

	array_b_colsize_host = array_b_colsize_host_input;
	array_b_rowsize_host = array_b_rowsize_host_input;
	array_b_host = array_a_host + array_a_colsize_host * array_a_rowsize_host;
	cudaMemcpy(array_b_host, array_b_host_input, sizeof(char) * array_b_colsize_host_input * array_b_rowsize_host_input, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(array_b, &array_b_host, sizeof(char*));
	cudaMemcpyToSymbol(array_b_colsize, &array_b_colsize_host, sizeof(int));
	cudaMemcpyToSymbol(array_b_rowsize, &array_b_rowsize_host, sizeof(int));
}

inline int32_t* align(char* in)
{
	char last_2 = ((uint64_t)in) & 3;
	if (last_2 == 0)
		return (int32_t*)in;
	else
		return (int32_t*)(in + 4 - last_2);
}

void setupResult(char op)
{
	char* result_unaligned = array_b_host + array_b_colsize_host * array_b_rowsize_host;
	result_host = align(result_unaligned);
	if (op == MAT_MUL) {
		if (array_a_rowsize_host != array_b_colsize_host) {
			printf("Cannot apply multiplication, size unmatched\n");
			exit(0);
		}
		result_colsize_host = array_a_colsize_host;
		result_rowsize_host = array_b_rowsize_host;
	}
	else if (op == MAT_CON) {
		if (array_a_rowsize_host < array_b_rowsize_host && array_a_colsize_host < array_b_colsize_host) {
			printf("Cannot apply convolution, filter size too small\n");
			exit(0);
		}
		result_colsize_host = array_a_colsize_host - array_b_colsize_host;
		result_rowsize_host = array_a_rowsize_host - array_b_rowsize_host;
	}
	else {
		printf("Operation not supported\n");
		exit(0);
	}

	if ((result_colsize_host * result_rowsize_host)*(sizeof(int32_t)/sizeof(char)) > U_BUFFER_SIZE - array_a_colsize_host * array_a_rowsize_host - array_b_colsize_host * array_b_rowsize_host) {
		printf("Buffer size insufficient\n");
		exit(0);
	}

	cudaMemcpyToSymbol(result, &result_host, sizeof(char*));
	cudaMemcpyToSymbol(result_colsize, &result_colsize_host, sizeof(int));
	cudaMemcpyToSymbol(result_rowsize, &result_rowsize_host, sizeof(int));
}

__device__ char feed_data_h(int row_num)
{
	if (row_num > cycle_count || cycle_count-row_num>=array_a_rowsize)
		return 0;
	else
		return array_a[row_num*array_a_rowsize + cycle_count - row_num];
}

__device__ char feed_data_v(int col_num)
{
	if (col_num > cycle_count || cycle_count - col_num >= array_b_colsize)
		return 0;
	else
		return array_b[(cycle_count - col_num)*array_b_rowsize+col_num];
}

__global__ void _collect_result()
{
	result[blockIdx.y*result_rowsize + blockIdx.x] = accumulators[blockIdx.y*sys_array_size + blockIdx.x];
}

void collect_result()
{
	_collect_result << <dim3(result_rowsize_host, result_colsize_host), 1 >> > ();
}

void change_size_a(int new_rowsize, int new_colsize)
{
	array_a_rowsize_host = new_rowsize;
	array_a_colsize_host = new_colsize;
	cudaMemcpyToSymbol(array_a_rowsize, &array_a_rowsize_host, sizeof(int));
	cudaMemcpyToSymbol(array_a_colsize, &array_a_colsize_host, sizeof(int));
}

void change_size_b(int new_rowsize, int new_colsize)
{
	array_b_rowsize_host = new_rowsize;
	array_b_colsize_host = new_colsize;
	cudaMemcpyToSymbol(array_b_rowsize, &array_b_rowsize_host, sizeof(int));
	cudaMemcpyToSymbol(array_b_colsize, &array_b_colsize_host, sizeof(int));
}

void change_size_result(int new_rowsize, int new_colsize)
{
	result_rowsize_host = new_rowsize;
	result_colsize_host = new_colsize;
	cudaMemcpyToSymbol(result_rowsize, &result_rowsize_host, sizeof(int));
	cudaMemcpyToSymbol(result_colsize, &result_colsize_host, sizeof(int));
}