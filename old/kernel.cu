#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#ifndef __CUDACC__ 
#define __CUDACC__
#endif
#include "device_functions.h"

#include <stdio.h>
#include <stdlib.h>

#include "parameters.h"
#include "data_collector.h"
#include "data_provider.h"

void sys_arr_ini();
void flush_sys_arr();
void sys_arr_free();

void sys_arr_ini()
{
	cell* sys_arr_buffer;
	cudaMalloc((void**)&sys_arr_buffer, sizeof(cell) * sys_arr_size * sys_arr_size);
	cudaMemcpyToSymbol(sys_arr, &sys_arr_buffer, sizeof(cell*), 0, cudaMemcpyHostToDevice);
	flush_sys_arr();
	printf("Systolic array successfully initialized, size %d * %d\n", sys_arr_size, sys_arr_size);
	data_provider_ini();
	data_collector_ini();
	//return sys_arr;
}

__global__ void _flush_sys_arr()
{
	int x = blockIdx.x;
	int y = blockIdx.y;
	sys_arr[x + gridDim.x * y].x = 0;
	sys_arr[x + gridDim.x * y].y = 0;
	sys_arr[x + gridDim.x * y].x_output = 0;
	sys_arr[x + gridDim.x * y].y_output = 0;
	sys_arr[x + gridDim.x * y].result = 0;
}

void flush_sys_arr()
{
	_flush_sys_arr <<<grid, 1 >>> ();
}

__global__ void _heart_beat()
{
	int x;
	int y;

	if (blockIdx.x == 0)
		x = feed_data_h(blockIdx.y);
	else
		x = sys_arr[blockIdx.x-1 + blockIdx.y * gridDim.x].x_output;

	if (blockIdx.y == 0)
		y = feed_data_v(blockIdx.x);
	else
		y = sys_arr[blockIdx.x + (blockIdx.y-1) * gridDim.x].y_output;

	sys_arr[blockIdx.x + blockIdx.y * gridDim.x].result += x * y;
	sys_arr[blockIdx.x + blockIdx.y * gridDim.x].x = x;
	sys_arr[blockIdx.x + blockIdx.y * gridDim.x].y = y;
}

__global__ void _cell_update()
{
	sys_arr[blockIdx.x + blockIdx.y * gridDim.x].x_output = sys_arr[blockIdx.x + blockIdx.y * gridDim.x].x;
	sys_arr[blockIdx.x + blockIdx.y * gridDim.x].y_output = sys_arr[blockIdx.x + blockIdx.y * gridDim.x].y;
}
__global__ void _result_shift()
{
	int result;
	if (threadIdx.x == 0)
		result = 0;
	else
		result = sys_arr[blockIdx.x + (threadIdx.x-1) * gridDim.x].result;
	__syncthreads();
	sys_arr[blockIdx.x + threadIdx.x * gridDim.x].result = result;
}

void sys_arr_free()
{
	cell* temp;
	cudaMemcpyToSymbol(sys_arr, &temp, sizeof(cell*), 0, cudaMemcpyDeviceToHost);
	cudaFree(temp);
	data_provider_free();
	data_collector_free();
}

int main()
{
	sys_arr_ini();
	int A[5][3] = {
		3, 2, 3,
		1, 4, 4,
		5, 6, 7,
		2, 9, 7,
		1, 2, 1
	};
	int B[3][4] = {
		1, 2, 3, 4,
		5, 6, 7, 8,
		4, 3, 2, 1
	};
	prepare_data(5, 4, 3, (int*)A, (int*)B);
	for (int i = 0; i < (5-1+3-1+4); i++)
	{
		_heart_beat << <grid, 1 >> > ();
		cudaDeviceSynchronize();
		_cell_update << <grid, 1 >> > ();
		dp_count_update();
		cudaDeviceSynchronize();
	}
	for (int i = 0; i < sys_arr_size-1; i++)
	{
		_result_shift << <sys_arr_size, sys_arr_size >> > ();
		collect_data();
	}

	int* result = form_result(10, 10);
	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < 10; j++)
			printf("%d ", result[i * 10 + j]);
		printf("\n");
	}

	free(result);
	sys_arr_free();
}