//#pragma once
#ifndef PARA_H
#define PARA_H

#define __CUDACC__RTC__
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdio.h"
#include "stdlib.h"
#include "inttypes.h"
#include "string.h"

#define MAT_MUL 1
#define MAT_CON 2
#define sys_array_size 256
#define grid dim3(sys_array_size,sys_array_size)

typedef struct
{
	char x;
	char x_output;
	char y;
	char y_output;
	char result;
	char result_output;
}Cell;

#endif