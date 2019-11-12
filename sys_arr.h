#pragma once
#include "parameters.h"
void sys_arr_ini();
void flush_sys_arr();
void sys_arr_free();
void heart_beat();
void result_shift();

Cell* sys_arr_host;
__device__ Cell* sys_arr;