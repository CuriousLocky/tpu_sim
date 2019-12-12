#include "parameters.h"
#include "inttypes.h"

const uint64_t PCIe_bandwidth = 14LL * 1024;
const uint64_t U_Buffer_bandwidth = 10LL * 1024;
const uint64_t DDR3_bandwidth = 30LL * 1024;
const uint64_t sys_arr_frequency = 700LL;
const uint64_t in_buffer_bandwidth = 167LL * 1024;

uint64_t data_trans = 0;
uint64_t weight_trans = 0;
uint64_t sys_arr_cycles = 0;
uint64_t in_buffer_trans = 0;

void sys_arr_cycle()
{
	sys_arr_cycles++;
}

void U_buffer_trans(size_t size)
{
	data_trans += size;
}

void weight_load(size_t size)
{
	weight_trans += size;
}

void in_buffer_load(size_t size)
{
	in_buffer_trans += size;
}

void resetSimInfo()
{
	data_trans = 0;
	sys_arr_cycles = 0;
	in_buffer_trans = 0;
	weight_trans = 0;
}

void printSimInfo()
{
	double data_transfer_time = ((double)data_trans) / U_Buffer_bandwidth;
	double weight_load_time = ((double)weight_trans) / DDR3_bandwidth;
	double cycle_time = ((double)sys_arr_cycles) / sys_arr_frequency;
	double in_buffer_trans_time = ((double)in_buffer_trans) / in_buffer_bandwidth;
	double total_time = data_transfer_time + weight_load_time + cycle_time + in_buffer_trans_time;
	printf("TPU_sim info:\n");
	printf("\tsystolic array cycles:\t%lld\n", sys_arr_cycles);
	printf("\tdata throughput:\t%lld\n", data_trans);
	printf("\ttotal estimated time:\t%f\n", total_time);
	printf("\ttime on matrix multiply unit cycles:\t%f\n", cycle_time);
	printf("\ttime on weight load: \t%f\n", weight_load_time);
	printf("\ttime on data load and result read: \t%f\n", in_buffer_trans_time);
	printf("\ttime on communication between host and device:\t%f\n", data_transfer_time);
}