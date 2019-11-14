#include "parameters.h"
void accumulators_ini();
void accumulators_free();
void flush_accumulators();
void result_activate();
__device__ void accumulate(int, int, int16_t);