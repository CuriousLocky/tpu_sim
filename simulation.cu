#include "parameters.h"
#include "inttypes.h"

const uint64_t PCIe_bandwidth = 14LL * 1024 * 1024 * 1024;
const uint64_t U_Buffer_bandwidth = 10LL * 1024 * 1024 * 1024;
const uint64_t DDR3_bandwidth = 30LL * 1024 * 1024 * 1024;
const uint64_t sys_arr_frequency = 700LL * 1024 * 1024;

uint64_t data_trans = 0;
uint64_t sys_arr_cycles = 0;

void sys_arr_cycle()
{
	sys_arr_cycles++;
}

void U_buffer_trans(size_t size)
{
	data_trans += size;
}

void resetSimInfo()
{
	data_trans = 0;
	sys_arr_cycles = 0;
}

void printSimInfo()
{
	double data_transfer_time = ((double)data_trans) / U_Buffer_bandwidth;
	double total_time = data_transfer_time + sys_arr_cycles / sys_arr_frequency;
	printf("TPU_sim info:\n");
	printf("\tsystolic array cycles:\t%lld\n", sys_arr_cycles);
	printf("\tdata throughput:\t%lld\n", data_trans);
	printf("\ttotal estimated time:\t%f\n", total_time);
	printf("\ttime on data transfer:\t%f\n", data_transfer_time);
}