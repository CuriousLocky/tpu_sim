void data_provider_ini();
void flush_data_provider();
void data_provider_free();
void prepare_data(int r_col_size, int r_row_size, int common_size, int* A, int* B);
__device__ int feed_data_h(int row_num);
__device__ int feed_data_v(int row_num);
void dp_count_update();