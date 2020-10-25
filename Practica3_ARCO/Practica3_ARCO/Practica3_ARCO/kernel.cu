#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h> 

#define MAXTHR 10 // MAXIMO de hilos / bloque

cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);

void showSpecs() {
	cudaDeviceProp deviceProp;
	int deviceID = 0;
	cudaGetDeviceProperties(&deviceProp, deviceID);
	int cores = 0;
	int SM = deviceProp.multiProcessorCount;
	int major = deviceProp.major;
	int minor = deviceProp.minor;
	int mp = deviceProp.multiProcessorCount;

	switch (major) {
	case 2: // Fermi
		if (minor == 1) cores = mp * 48;
		else cores = mp * 32;
		break;
	case 3: // Kepler
		cores = mp * 192;
		break;
	case 5: // Maxwell
		cores = mp * 128;
		break;
	case 6: // Pascal
		if ((minor == 1) || (minor == 2)) cores = mp * 128;
		else if (minor == 0) cores = mp * 64;
		else printf("Unknown device type\n");
		break;
	case 7: // Volta and Turing
		if ((minor == 0) || (minor == 5)) cores = mp * 64;
		else printf("Unknown device type\n");
		break;
	case 8: // Ampere
		if (minor == 0) cores = mp * 64;
		else printf("Unknown device type\n");
		break;
	default:
		printf("Unknown device type\n");
		break;
	}
	printf("\nDEVICE %d: %s\n", deviceID, deviceProp.name);
	printf("> Capacidad de Computo \t \t: %d.%d\n", major, minor);
	printf("> No. MultiProcesadores \t: %d \n", SM);
	printf("> No. Nucleos CUDA (%dx%d) \t: %d \n", cores, SM, cores * SM);
	// printf("> Memoria Global (total) \t: %u MiB\n", deviceProp.totalGlobalMem/Mebi);

	printf("***************************************************\n");
	printf("MAX Hilos por bloque: %d\n", deviceProp.maxThreadsPerBlock);
	printf("MAX BLOCK SIZE\n");
	printf(" [x -> %d]\n [y -> %d]\n [z -> %d]\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
	printf("MAX GRID SIZE\n");
	printf(" [x -> %d]\n [y -> %d]\n [z -> %d]\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
	printf("***************************************************\n");
}

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
	return cores;
}
__global__ void sumaKernel(int* a, int* b, int* c, unsigned int maxThr)
{
	// El index = id del hilo en el bloque + num de hilos por bloque * nid de bloque
	// Evitar que los hilos extra no entren
	// Parametro nhilos totales
	int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < maxThr) a[i] = b[i] + c[i];
}

__global__ void invertirArray(int* dest, int* org, unsigned int n) {
	// Evitar que los hilos extra no entren
	// Parametro nhilos totales (if)
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	// Solo si no supero el numero maximo de hilos
	// asi se evita modificar memoria externa
	if (i < n) dest[(n - 1) - i] = org[i];
}

int main()
{
	int deviceID = 0, ncores, nthr;
	int* hst_matriz = NULL;
	int* dev_matriz = NULL;
	int* dev_matriz_inver = NULL;
	int* dev_resultado = NULL;
	int deviceCount;

	int nbloques;

	cudaGetDeviceCount(&deviceCount);
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, deviceID);

	ncores = getSPcores(props);

	//Mostrar caracteristicas
	showSpecs();

	printf("Numero de valores del vector:");
	scanf("%i", &nthr);

	hst_matriz = (int*)malloc(nthr * sizeof(int));
	// cudaMallocManaged no funciona en algunas versiones
	// - genera una memoria unificada
	cudaMalloc((void**)&dev_matriz, nthr * sizeof(int));
	cudaMalloc((void**)&dev_resultado, nthr * sizeof(int));
	cudaMalloc((void**)&dev_matriz_inver, nthr * sizeof(int));

	// Obtencion del limite de bloques
	nbloques = ceil(nthr / MAXTHR);

	// Generacion del array
	srand(time(NULL));
	for (int i = 0; i < nthr; i++) {
		hst_matriz[i] = rand() % 10;
	}

	printf("Matriz original:\n");
	for (int i = 0; i < nthr; i++) {
		printf("[%i]", hst_matriz[i]);
	}
	printf("\n");

	cudaMemcpy(dev_matriz, hst_matriz, nthr * sizeof(int), cudaMemcpyHostToDevice);
	invertirArray << <nbloques, MAXTHR >> > (dev_matriz_inver, dev_matriz, nthr);
	cudaMemcpy(hst_matriz, dev_matriz_inver, nthr * sizeof(int), cudaMemcpyDeviceToHost);

	printf("Matriz invertida:\n");
	for (int i = 0; i < nthr; i++) {
		printf("[%i]", hst_matriz[i]);
	}
	printf("\n");

	sumaKernel << <nbloques, MAXTHR >> > (dev_resultado, dev_matriz_inver, dev_matriz,nthr);
	cudaMemcpy(hst_matriz, dev_resultado, nthr * sizeof(int), cudaMemcpyDeviceToHost);


	printf("Resultado final:\n");
	for (int i = 0; i < nthr; i++) {
		printf("[%i]", hst_matriz[i]);
	}
	printf("\n");

	free(hst_matriz);
	cudaFree(dev_resultado);
	cudaFree(dev_matriz);
	cudaFree(dev_matriz_inver);
}