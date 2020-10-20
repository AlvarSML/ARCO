
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h> 

cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);


int getSPcores(cudaDeviceProp devProp)
{
	int cores = 0;
	int mp = devProp.multiProcessorCount;
	switch (devProp.major) {
	case 2: // Fermi
		if (devProp.minor == 1) cores = mp * 48;
		else cores = mp * 32;
		break;
	case 3: // Kepler
		cores = mp * 192;
		break;
	case 5: // Maxwell
		cores = mp * 128;
		break;
	case 6: // Pascal
		if ((devProp.minor == 1) || (devProp.minor == 2)) cores = mp * 128;
		else if (devProp.minor == 0) cores = mp * 64;
		else printf("Unknown device type\n");
		break;
	case 7: // Volta and Turing
		if ((devProp.minor == 0) || (devProp.minor == 5)) cores = mp * 64;
		else printf("Unknown device type\n");
		break;
	case 8: // Ampere
		if (devProp.minor == 0) cores = mp * 64;
		else printf("Unknown device type\n");
		break;
	default:
		printf("Unknown device type\n");
		break;
	}
	printf("Hay %i nucleos\n", cores);
	return cores;
}
__global__ void addKernel(int* a, int* b)
{
	int i = threadIdx.x;
	a[i] += b[i];
}

__global__ void invertirArray(int* dest, int* org, unsigned int n) {
	for (int i = n - 1; i >= 0; i--) {
		dest[(n - 1) - i] = org[i];
	}
}

int main()
{
	int deviceID = 0, ncores, nvec;
	int* hst_matriz = NULL;
	int* dev_matriz = NULL;
	int* dev_matriz_inver = NULL;
	int deviceCount;

	cudaGetDeviceCount(&deviceCount);
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, deviceID);

	ncores = getSPcores(props);

	printf("Numero de valores del vector(max %i): ", ncores);
	scanf("%i", &nvec);

	hst_matriz = (int*)malloc(nvec * sizeof(int));
	cudaMallocManaged((void**)&dev_matriz, nvec * sizeof(int));
	cudaMallocManaged((void**)&dev_matriz_inver, nvec * sizeof(int));

	// Generacion del array
	srand(time(NULL));
	for (int i = 0; i < nvec; i++) {
		hst_matriz[i] = rand() % 10;
	}

	int* arr = (int*)malloc(nvec * sizeof(int));

	cudaMemcpy((void**)dev_matriz, hst_matriz, nvec * sizeof(int), cudaMemcpyHostToDevice);
	invertirArray <<<1,1>>> (dev_matriz_inver, dev_matriz, nvec);
	addKernel <<<1,nvec>>> (dev_matriz, dev_matriz_inver);
	cudaMemcpy((void**)hst_matriz,dev_matriz,nvec*sizeof(int),cudaMemcpyDeviceToHost);

	printf("Resultado final:");
	for (int i = 0; i < nvec;i++) {
		printf("[%i]",hst_matriz[i]);
	}
	
}

