#include "control.h"
#include "simulation.h"
#include "stdio.h"
#include "stdlib.h"
#include "inttypes.h"

int main()
{
	char A[] = {
		3,2,3,
		1,4,4,
		5,6,7,
		2,9,7,
		1,2,1
	};
	char B[] = {
		1,2,3,4,
		5,6,7,8,
		4,3,2,1
	};
	int32_t result[20];
	tpu_ini();
	read_host_memory(A, 5, 3);
	read_weights(B, 3, 4);
	matrix_multiply();
	activate();
	write_host_memory(result);

	printf("computation result:\n");
	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 4; j++)
			printf("%d ", result[i*4+j]);
		printf("\n");
	}
	printf("\n");

	printSimInfo();

	char X[] = {
		0,0,0,
		0,0,0,
		0,0,0
	};
	char Y[] = {
		0,0,0,
		0,0,0,
		0,0,0
	};

	read_host_memory(X, 3, 3);
	read_weights(Y, 3, 3);
	int32_t R[6];
	matrix_convolution();
	write_host_memory(R);
	for (int i = 0; i < 1; i++) {
		for (int j = 0; j < 1; j++)
			printf("%d ", R[i * 1 + j]);
		printf("\n");
	}

	printSimInfo();
}