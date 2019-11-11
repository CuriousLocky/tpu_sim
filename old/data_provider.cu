#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

extern unsigned int sys_arr_size;
extern unsigned int data_provider_size;
extern __device__ int** data_provider_h;//[sys_arr_size];
extern __device__ int** data_provider_v;//[sys_arr_size];

__device__ int data_provider_count = 0;

int** data_provider_h_buffer;
int** data_provider_v_buffer;

void flush_data_provider();
void data_provider_free();

void data_provider_ini()
{
	data_provider_h_buffer = (int**)malloc(sizeof(int*) * sys_arr_size);
	data_provider_v_buffer = (int**)malloc(sizeof(int*) * sys_arr_size);
	for (int i = 0; i < sys_arr_size; i++)
	{
		cudaMalloc((void**)& (data_provider_h_buffer[i]), sizeof(int) * data_provider_size);
		cudaMalloc((void**)& (data_provider_v_buffer[i]), sizeof(int) * data_provider_size);
	}
	flush_data_provider();

	int** temp_h, ** temp_v;
	cudaMalloc((void**)& temp_h, sizeof(int*) * sys_arr_size);
	cudaMalloc((void**)& temp_v, sizeof(int*) * sys_arr_size);
	cudaMemcpy(temp_h, data_provider_h_buffer, sizeof(int*) * sys_arr_size, cudaMemcpyHostToDevice);
	cudaMemcpy(temp_v, data_provider_v_buffer, sizeof(int*) * sys_arr_size, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(data_provider_h, &temp_h, sizeof(int**), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(data_provider_v, &temp_v, sizeof(int**), 0, cudaMemcpyHostToDevice);
	
	printf("Data provider modules successfully initialized, size %d\n", data_provider_size);
}

__global__ void _flush_dp_count()
{
	data_provider_count = 0;
}
void flush_data_provider()
{
	for (int i = 0; i < sys_arr_size; i++)
	{
		cudaMemset(data_provider_h_buffer[i], 0, sizeof(int) * data_provider_size);
		cudaMemset(data_provider_v_buffer[i], 0, sizeof(int) * data_provider_size);
	}
	_flush_dp_count << <1, 1 >> > ();
}

void data_provider_free()
{
	for (int i = 0; i < sys_arr_size; i++)
	{
		cudaFree(data_provider_h_buffer[i]);
		cudaFree(data_provider_v_buffer[i]);
	}
	free(data_provider_h_buffer);
	free(data_provider_v_buffer);
	cudaFree(data_provider_h);
	cudaFree(data_provider_v);
}

__global__ void _prepare_data_h(int size, int* array) //damn C++, does not support "int array[][size]"
{
	int x = blockIdx.x;
	int y = blockIdx.y;
	data_provider_h[y][x+y] = array[y * size + x];
}

__global__ void _prepare_data_v(int size, int* array)
{
	int x = blockIdx.x;
	int y = blockIdx.y;
	data_provider_v[x][x+y] = array[y * size + x];
}

void prepare_data(int r_col_size, int r_row_size, int common_size, int* A, int* B)
{
	int* temp_A;
	printf("%d\n", cudaMalloc((void**)& temp_A, sizeof(int) * r_col_size * common_size));
	printf("%d\n", cudaMemcpy(temp_A, A, sizeof(int) * r_col_size*common_size, cudaMemcpyHostToDevice));
	int* temp_B;
	cudaMalloc((void**)& temp_B, sizeof(int) * r_row_size * common_size);
	cudaMemcpy(temp_B, B, sizeof(int) * r_row_size * common_size, cudaMemcpyHostToDevice);
	_prepare_data_h << <dim3(common_size, r_col_size), 1 >> > (common_size, temp_A);
	_prepare_data_v << <dim3(r_row_size, common_size), 1 >> > (r_row_size, temp_B);
	cudaFree(temp_A);
	cudaFree(temp_B);
}

__device__ int feed_data_h(int row_num)
{
	return data_provider_h[row_num][data_provider_count];
}

__device__ int feed_data_v(int col_num)
{
	return data_provider_v[col_num][data_provider_count];
}

__global__ void _dp_count_update()
{
	data_provider_count++;
	return;
}
void dp_count_update()
{
	_dp_count_update << <1, 1 >> > ();
}