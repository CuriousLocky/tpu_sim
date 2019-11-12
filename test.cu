#include "control.h"
#include "stdio.h"
#include "stdlib.h"

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
	char result[15];
	tpu_ini();
	read_host_memory(A, 3, 5);
	read_weights(B, 4, 3);
	matrix_multiply();
	activate();
	write_host_memory(result);
	for (int i = 0; i < 5; i++) {
		for (int j = 0; i < 3; i++)
			printf("%d ", result[i*3+j]);
		printf("\n");
	}
}