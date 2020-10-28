// includes
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
// defines
#define I 10 // tamaño de la matriz filas
#define J 9 // tamaño de la matriz columnas

// declaracion de funciones

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

// GLOBAL: funcion llamada desde el host y ejecutada en el device (kernel)
__global__ void fImpares(int* A, int* C)
{
	// indice de fila
	int fila = threadIdx.y;
	// indice de columna 
	int columna = threadIdx.x;

	// Calculamos la suma:
	// C[fila][columna] = A[fila][columna] + B[fila][columna] =
	// Para ello convertimos los indices de 'fila' y columna' en un indice lineal
	int myID = columna + fila * blockDim.x;
	// Filas impares
	if (columna % 2 != 0) {
		C[myID] = 0;
	}
	else {
		C[myID] = A[myID];
	}

}

// MAIN: rutina principal ejecutada en el host
int main(int argc, char** argv)
{
	// declaraciones
	int* hst_A, * hst_C;
	int* dev_A, * dev_C;

	// reserva en el host.
	hst_A = (int*)malloc(I * J * sizeof(int));
	hst_C = (int*)malloc(I * J * sizeof(int));

	// reserva en el device
	cudaMalloc((void**)&dev_A, I * J * sizeof(int));
	cudaMalloc((void**)&dev_C, I * J * sizeof(int));

	// incializacion
	showSpecs();
	for (int i = 0; i < I * J; i++)
	{
		hst_A[i] = i;

	}

	// copia de datos
	cudaMemcpy(dev_A, hst_A, I * J * sizeof(int), cudaMemcpyHostToDevice);

	// dimensiones del kernel
	//Nbloques(1) es un solo bloque.
	dim3 Nbloques(1);
	dim3 hilosB(J, I);

	// llamada al kernel bidimensional de NxN hilos
	fImpares << <Nbloques, hilosB >> > (dev_A, dev_C);

	// recogida de datos
	cudaMemcpy(hst_C, dev_C, I * J * sizeof(int), cudaMemcpyDeviceToHost);

	// impresion de resultados
	printf("MATRIZ A:\n");
	for (int x = 0; x < I; x++)
	{
		for (int y = 0; y < J; y++)
		{
			printf("%3i ", hst_A[y + x * J]); //* el numero de filas.
		}
		printf("\n");
	}

	//MATRIZ SOLUCIÓN
	printf("MATRIZ SOLUCION:\n");
	for (int x = 0; x < I; x++)
	{
		for (int y = 0; y < J; y++)
		{
			printf("%3i ", hst_C[y + x * J]); //* el numero de filas.
		}
		printf("\n");
	}

	// salida
	time_t fecha;
	time(&fecha);
	printf("***************************************************\n");
	printf("Programa ejecutado el: %s\n", ctime(&fecha));
	printf("<pulsa [INTRO] para finalizar>");
	getchar();
	return 0;
}