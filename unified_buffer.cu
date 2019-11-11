#include "parameters.h"
#include "simulation.h"

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

__device__ char* result;
char* result_host;
__device__ int result_colsize;
int result_colsize_host;
__device__ int result_rowsize;
int result_rowsize_host;

void u_buffer_ini()
{
	cudaMalloc((void**)& array_a_host, sizeof(char) * U_BUFFER_SIZE);
	cudaMemcpyToSymbol(&u_buffer, &array_a_host, sizeof(char*), 0, cudaMemcpyHostToDevice);
	cudaMemset(u_buffer, 0, U_BUFFER_SIZE);
}

void u_buffer_free()
{
	cudaFree(u_buffer);
}

void setupArray_a(char* array_a_host_input, int array_a_colsize_host_input, int array_a_rowsize_host_input)
{
	if (array_a_colsize_host_input * array_a_rowsize_host_input >= U_BUFFER_SIZE) {
		printf("array a size is too big\n");
		exit(0);
	}
	array_a_colsize_host = array_a_colsize_host_input;
	array_a_rowsize_host = array_a_rowsize_host_input;
	cudaMemcpyToSymbol(u_buffer, array_a_host_input, sizeof(char)*array_a_colsize_host_input*array_a_rowsize_host_input, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(&array_a, &array_a_host, sizeof(char*), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(&array_a_colsize, &array_a_colsize_host, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(&array_a_rowsize, &array_a_rowsize_host, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(&array_a, &u_buffer, sizeof(char*), 0, cudaMemcpyDeviceToDevice);
	U_buffer_trans(array_a_colsize_host * array_a_rowsize_host);
}

void setupArray_b(char* array_b_host_input, int array_b_colsize_host_input, int array_b_rowsize_host_input)
{
	if (array_b_colsize_host_input * array_b_rowsize_host_input >= U_BUFFER_SIZE-array_a_colsize_host*array_a_rowsize_host) {
		printf("array b size is too big\n");
		exit(0);
	}

	array_b_host = array_a_host + array_a_colsize_host * array_a_rowsize_host;
	array_b_colsize_host = array_b_colsize_host_input;
	array_b_rowsize_host = array_b_rowsize_host_input;
	cudaMemcpyToSymbol(array_b_host, array_b_host_input, sizeof(char) * array_b_colsize_host_input * array_b_rowsize_host_input, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(&array_b, &array_b_host, sizeof(char*), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(&array_b_colsize, &array_b_colsize_host, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(&array_b_rowsize, &array_b_rowsize_host, sizeof(int), cudaMemcpyHostToDevice);
	U_buffer_trans(array_b_colsize_host * array_b_rowsize_host);
}

void setupResult(char op)
{
	result_host = array_b_host + array_b_colsize_host * array_b_rowsize_host;
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

	if (result_colsize_host * result_rowsize_host > U_BUFFER_SIZE - array_a_colsize_host * array_a_rowsize_host - array_b_colsize_host * array_b_rowsize_host) {
		printf("Buffer size insufficient\n");
		exit(0);
	}

	cudaMemcpyToSymbol(&result, &result_host, sizeof(char*), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(&result_colsize, &result_rowsize, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(&result_rowsize, &result_rowsize_host, sizeof(int), 0, cudaMemcpyHostToDevice);
}