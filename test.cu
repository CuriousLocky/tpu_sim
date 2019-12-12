#include "control.h"
#include "simulation.h"
#include "stdio.h"
#include "stdlib.h"
#include "inttypes.h"
#include "string.h"


int main()
{
	tpu_ini();
	char* A = (char*)malloc(sizeof(char) * 512 * 512);
	char* B = (char*)malloc(sizeof(char) * 512 * 512);
	int32_t* result = (int32_t*)malloc(sizeof(int32_t) * 512 * 512);
	char input_buffer[128];
	while (1) {
		int size1, size2, size3;
		fgets(input_buffer, 128, stdin);
		if (strcmp(input_buffer, "clear\n") == 0) {
			resetSimInfo();
			continue;
		}
		sscanf(input_buffer, "%d %d %d\n", &size1, &size2, &size3);
		for (int x = 0; x < size1; x++) {
			for (int y = 0; y < size2; y++) {
				A[x*size1 + y] = rand() % 128;
			}
		}
		for (int x = 0; x < size2; x++) {
			for (int y = 0; y < size3; y++) {
				B[x * size2 + y] = rand() % 128;
			}
		}

		for (int i = 0; i < 100; i++){
			read_host_memory(A, size1, size2);
			read_weights(B, size2, size3);
			matrix_multiply();
			activate();
			write_host_memory(result);
		}
		printSimInfo();
	}
	
}