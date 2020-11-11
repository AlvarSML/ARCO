// includes
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>

#define N 10	//Columnas == Valores por array
#define M 5		//FIlas == numero de arrays

int getSPcores()
{
	cudaDeviceProp devProp;
	int deviceID = 0;
	cudaGetDeviceProperties(&devProp, deviceID);
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

// Desplaza una posicion hacia abajo las filas de un array
// n y m son las dimensiones del array
__global__ void desplazarFila(int* arrOrg, int* arrDest, int m) {

	int fila = threadIdx.y;
	int columna = threadIdx.x;
	int hilosFila = blockDim.x;

	int id = columna + fila * hilosFila;
	
	int idDest;

	if (fila + 1 >= m) {
		idDest = columna;
	}
	else {
		idDest = (fila + 1) * hilosFila + columna;
	}
		
	arrDest[idDest] = arrOrg[id];
}

// Programa que invierte una matriz de enteros
// hst1 -> dev1
// dev1 -> inversion -> dev2
// dev2 -> hst2
// se muestra hst1 y hst2
int main(int argc, char** argv)
{
	// Declaracio
	int* hst_1, * hst_2;
	int* dev_1, * dev_2;

	// Reserva de memoria
	hst_1 = (int*)malloc(N * M * sizeof(int));
	hst_2 = (int*)malloc(N * M * sizeof(int));

	cudaMalloc((void**)&dev_1, N * M * sizeof(int));
	cudaMalloc((void**)&dev_2, N * M * sizeof(int));

	// Inicializacion del array origen
	int fila = 0;
	int contColumna = 0;
	for (int i = 0; i < (N * M); i++) {
		if (contColumna >= N) {
			fila++;
			contColumna = 0;
		}

		hst_1[i] = fila;
		contColumna++;
	}

	dim3 NHilos(N, M);

	cudaMemcpy(dev_1, hst_1, N * M * sizeof(int), cudaMemcpyHostToDevice);

	desplazarFila << <1, NHilos >> > (dev_1, dev_2, M);

	cudaMemcpy(hst_2, dev_2, N * M * sizeof(int), cudaMemcpyDeviceToHost);

	// Visualizacion de hst_1
	fila = 0;
	contColumna = 0;
	for (int i = 0; i < (N * M); i++) {
		if (contColumna >= N) {
			fila++;
			contColumna = 0;
			printf("\n");
		}
		printf("%i", hst_1[i]);
		contColumna++;
	}
	printf("\n\n");

	// Visualizacion de hst_2
	fila = 0;
	contColumna = 0;
	for (int i = 0; i < (N * M); i++) {
		if (contColumna >= N) {
			fila++;
			contColumna = 0;
			printf("\n");
		}
		printf("%i", hst_2[i]);
		contColumna++;
	}
}