#ifndef __parameters_H
#define __parameters_H

#include "cell_def.h"
unsigned int sys_arr_size = 256;
dim3 grid(sys_arr_size, sys_arr_size);
unsigned int data_provider_size = 512;
unsigned int data_collector_size = 512;
__device__ cell* sys_arr;
__device__ int** data_provider_h;
__device__ int** data_provider_v;
__device__ int** data_collector;

#endif