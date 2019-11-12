#include "parameters.h"
#include "inttypes.h"

const uint64_t PCIe_bandwidth = 14LL * 1024 * 1024 * 1024;
const uint64_t U_Buffer_bandwidth = 10LL * 1024 * 1024 * 1024;
const uint64_t DDR3_bandwidth = 30LL * 1024 * 1024 * 1024;
const uint64_t sys_arr_frequency = 700LL * 1024 * 1024;

double data_transfer_time = 0;

uint64_t sys_arr_cycles = 0;

void sys_arr_cycle()
{
	sys_arr_cycles++;
}

void U_buffer_trans(size_t size)
{
	data_transfer_time += ((double)size) / U_Buffer_bandwidth;
}