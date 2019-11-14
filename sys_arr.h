#pragma once
#include "parameters.h"
void sys_arr_ini();
void flush_sys_arr();
void sys_arr_free();
void heart_beat();

__global__ void reset_count();