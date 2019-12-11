#include "parameters.h"
#include "unified_buffer.h"
#include "accumulators.h"

Cell* sys_arr_host;
__device__ Cell* sys_arr;

void flush_sys_arr();
void sys_arr_free();

extern __device__ int cycle_count;
extern int result_rowsize_host;
extern int result_colsize_host;
extern __device__ int result_rowsize;
extern __device__ int result_colsize;
extern __device__ char* result;

void sys_arr_ini()
{
	cudaMalloc((void**)&sys_arr_host, sizeof(Cell) * sys_array_size * sys_array_size);
	cudaMemcpyToSymbol(sys_arr, &sys_arr_host, sizeof(Cell*));
	flush_sys_arr();
	printf("Systolic array successfully initialized, size %d * %d\n", sys_array_size, sys_array_size);
}

void flush_sys_arr()
{
	cudaMemset(sys_arr_host, 0, sizeof(Cell) * sys_array_size * sys_array_size);
}

void sys_arr_free()
{
	cudaFree(sys_arr_host);
}

__global__ void _heart_beat()
{
	int x;
	int y;

	if (blockIdx.x == 0)
		x = feed_data_h(blockIdx.y);
	else
		x = sys_arr[blockIdx.x - 1 + blockIdx.y * gridDim.x].x_output;

	if (blockIdx.y == 0)
		y = feed_data_v(blockIdx.x);
	else
		y = sys_arr[blockIdx.x + (blockIdx.y - 1) * gridDim.x].y_output;
	int16_t result = x * y;
	accumulate(blockIdx.x, blockIdx.y, result);
	sys_arr[blockIdx.x + blockIdx.y * gridDim.x].x = x;
	sys_arr[blockIdx.x + blockIdx.y * gridDim.x].y = y;
}

__global__ void _cell_update()
{
	sys_arr[blockIdx.x + blockIdx.y * gridDim.x].x_output = sys_arr[blockIdx.x + blockIdx.y * gridDim.x].x;
	sys_arr[blockIdx.x + blockIdx.y * gridDim.x].y_output = sys_arr[blockIdx.x + blockIdx.y * gridDim.x].y;
}

__global__ void increase_count()
{
	cycle_count++;
}

__global__ void reset_count()
{
	cycle_count = 0;
}

void heart_beat()
{
	_heart_beat <<< grid, 1 >>> ();
	cudaDeviceSynchronize();
	_cell_update << <grid, 1 >> > ();
	cudaDeviceSynchronize();
	increase_count << <1, 1 >> > ();
}
