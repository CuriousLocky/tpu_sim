#include "parameters.h"
#include "simulation.h"

void flush_sys_arr();
void sys_arr_free();
void sys_arr_ini()
{
	cudaMalloc((void**)&sys_arr_host, sizeof(Cell) * sys_array_size * sys_array_size);
	cudaMemcpyToSymbol(sys_arr, &sys_arr_host, sizeof(Cell*), 0, cudaMemcpyHostToDevice);
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
	sys_arr[blockIdx.x + blockIdx.y * gridDim.x].x_output = sys_arr[blockIdx.x + blockIdx.y * gridDim.x].x;
	sys_arr[blockIdx.x + blockIdx.y * gridDim.x].y_output = sys_arr[blockIdx.x + blockIdx.y * gridDim.x].y;
}

void heart_beat()
{
	_heart_beat <<< grid, 1 >>> ();
	cudaDeviceSynchronize();
	_cell_update <<< grid, 1 >>> ();
	sys_arr_cycle();
	cudaDeviceSynchronize();
}