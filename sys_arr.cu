#include "parameters.h"
#include "unified_buffer.h"

Cell* sys_arr_host;
__device__ Cell* sys_arr;
dim3 grid(sys_array_size, sys_array_size);

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
	sys_arr[blockIdx.x + blockIdx.y * gridDim.x].result += x * y;
	sys_arr[blockIdx.x + blockIdx.y * gridDim.x].x = x;
	sys_arr[blockIdx.x + blockIdx.y * gridDim.x].y = y;
}

__global__ void _cell_update()
{
	sys_arr[blockIdx.x + blockIdx.y * gridDim.x].result_output = sys_arr[blockIdx.x + blockIdx.y * gridDim.x].result;
	sys_arr[blockIdx.x + blockIdx.y * gridDim.x].x_output = sys_arr[blockIdx.x + blockIdx.y * gridDim.x].x;
	sys_arr[blockIdx.x + blockIdx.y * gridDim.x].y_output = sys_arr[blockIdx.x + blockIdx.y * gridDim.x].y;
}

__global__ void _result_shift()
{
	if (blockIdx.y == 0)
		sys_arr[blockIdx.x + blockIdx.y * sys_array_size].result = 0;
	else
		sys_arr[blockIdx.x + blockIdx.y * sys_array_size].result = sys_arr[blockIdx.x + (blockIdx.y-1) * sys_array_size].result_output;
	
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

void result_shift()
{
	_collect_result << <dim3(result_rowsize_host, result_colsize_host), 1 >> > ();
	cudaDeviceSynchronize();
	_result_shift <<< grid, 1 >>> ();
	cudaDeviceSynchronize();
	_cell_update << <grid, 1 >> > ();
	cudaDeviceSynchronize();
}