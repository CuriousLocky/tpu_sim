#include "parameters.h"
__device__ int32_t* accumulators;
int32_t* accumulators_host;

void accumulators_ini()
{
	cudaMalloc(&accumulators, sizeof(int32_t) * sys_array_size * sys_array_size);
	cudaMemcpyToSymbol(accumulators, &accumulators, sizeof(int32_t*));
}

void flush_accumulators()
{
	cudaMemset(accumulators_host, 0, sizeof(int32_t)*sys_array_size*sys_array_size);
}

void accumulators_free()
{
	cudaFree(accumulators_host);
}

__device__ void accumulate(int x, int y, int16_t result)
{
	accumulators[y*sys_array_size + x] += result;
}

__global__ void _result_activate()
{
	if (accumulators[blockIdx.x + blockIdx.y*sys_array_size] < 0)
		accumulators[blockIdx.x + blockIdx.y*sys_array_size] = 0;
}

void result_activate()
{
	_result_activate << <grid, 1 >> > ();
}