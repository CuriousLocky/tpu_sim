#pragma once
#ifndef cuda_h
	#define cuda_h
	#include "cuda.h"
	#include "cuda_runtime.h"
	#include "device_launch_parameters.h"
	#include "stdio.h"
	#include "stdlib.h"
#endif

#define MAT_MUL 1
#define MAT_CON 2

typedef struct
{
	char x;
	char x_output;
	char y;
	char y_output;
	char result;
}Cell;

unsigned int sys_array_size = 256;
dim3 grid(sys_array_size, sys_array_size);
unsigned int data_provider_depth = 512;
unsigned int data_collector_depth = 512;

Cell* sys_arr_host;

__device__ Cell* sys_arr;
__device__ char** data_provider_h;
__device__ char** data_provider_h;
__device__ char** data_collector;