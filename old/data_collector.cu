#include "cuda_runtime.h"

#ifndef __CUDACC__ 
#define __CUDACC__
#endif
#include "device_launch_parameters.h"
#include "device_functions.h"

#include <stdio.h>
#include <stdlib.h>
#include "cell_def.h"

extern unsigned int sys_arr_size;
extern unsigned int data_collector_size;
extern __device__ int** data_collector;
extern __device__ cell* sys_arr;

void data_collector_ini();
void data_collector_free();
void flush_data_collector();
void dc_update();

int** data_collector_host;
__device__ int dc_count = 0;
int dc_count_h = 1;


__global__ void _data_collector_ini(unsigned int size)
{
	cudaMalloc((void**)& data_collector[blockIdx.x], sizeof(int) * size);
}
void data_collector_ini()
{
	cudaMalloc((void**)& data_collector_host, sizeof(int*) * sys_arr_size);
	cudaMemcpyToSymbol(data_collector, &data_collector_host, sizeof(int**));
	_data_collector_ini << <sys_arr_size, 1 >> > (data_collector_size);
	printf("data_collector module initialized.\n");
	flush_data_collector();
}

__global__ void _data_collector_free()
{
	int* lane = data_collector[blockIdx.x];
	cudaFree(lane);
}
void data_collector_free()
{	
	_data_collector_free << <sys_arr_size, 1 >> > ();
	cudaFree(data_collector_host);
}

__global__ void _flush_data_collector()
{
	data_collector[blockIdx.x][blockIdx.y] = 0;
}
void flush_data_collector()
{
	_flush_data_collector << <dim3(sys_arr_size, data_collector_size), 1 >> > ();
}

__global__ void _dc_update()
{
	dc_count++;
}
void dc_update()
{
	_dc_update << <1, 1 >> > ();
}

__global__ void _shift_down()
{
	int temp = data_collector[blockIdx.x][threadIdx.x];
	__syncthreads();
	data_collector[blockIdx.x][threadIdx.x + 1] = temp;
}
__global__ void _collect_data()
{
	data_collector[blockIdx.x][0] = sys_arr[gridDim.x*(gridDim.x-1) + blockIdx.x].result;
}
void collect_data()
{
	_shift_down << <sys_arr_size, dc_count_h >> > ();
	cudaDeviceSynchronize();
	_collect_data << <sys_arr_size, 1 >> > ();
	_dc_update << <1, 1 >> > ();
	dc_count_h++;
}


__global__ void _form_result(int* result_device)
{
	int x = blockIdx.x;
	int y = blockIdx.y;
	result_device[x + y * gridDim.x] = data_collector[x][y];
}
int* form_result(unsigned int result_row_size, unsigned int result_col_size)
{
	int* result = (int*)malloc(sizeof(int) * result_row_size * result_col_size);
	int* result_device;
	cudaMalloc((void**)& result_device, sizeof(int) * result_row_size * result_col_size);
	_form_result << <dim3(result_row_size, result_col_size), 1 >> > (result_device);
	cudaMemcpy(result, result_device, sizeof(int) * result_row_size * result_col_size, cudaMemcpyDeviceToHost);
	cudaFree(result_device);
	return result;
}