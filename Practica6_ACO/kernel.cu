#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
// defines
#define N 16 // numero de terminos que se van a sumar

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

// GLOBAL: funcion llamada desde el host y ejecutada en el device (kernel)
__global__ void reduccion(double* datos, double* suma)
// Funcion que suma los primeros N numeros naturales
{
	// indice local de cada hilo -> kernel con un solo bloque de N hilos
	int myID = threadIdx.x;

	// Rellenamos el vector de datos
	double dato = 1 / ((double)myID + 1);
	// si es par o impar
	if (myID % 2) { 
		dato *= -1;
	}

	printf("Dato:%lf\n", dato);

	datos[myID] = dato;

	// sincronizamos para evitar riesgos de tipo RAW
	__syncthreads();

	// ******************
	// REDUCCION PARALELA
	// ******************
	int salto = N / 2;
	// realizamos log2(N) iteraciones
	while (salto > 0)
	{
		// en cada paso solo trabajan la mitad de los hilos
		if (myID < salto)
		{
			datos[myID] = datos[myID] + datos[myID + salto];
		}
		// sincronizamos los hilos evitar riesgos de tipo RAW
		__syncthreads();
		salto = salto / 2;
	}
	// ******************
	// Solo el hilo no.'0' escribe el resultado final
	if (myID == 0)
	{
		*suma = datos[0];
		printf("*suma:%lf\n", *suma);
		printf("suma:%lf\n", suma);
	}
}

int main(int argc, char** argv)
{
	double* dev_datos;
	double* hst_suma;
	double* dev_suma;


	// Terminos tiene que ser potencia de 2
	int terminos;
	int max = getSPcores();

	// Visualizacion de datos GPU
	showSpecs();

	printf(">> Introduce el numero de terminos (potencia de 2): ");
	scanf("%i", &terminos);

	printf("< Lanzamiento con 1 bloque de %i hilos",terminos);

	hst_suma = (double*)malloc(sizeof(double));
	cudaMalloc((void**)&dev_datos, terminos * sizeof(double));
	// No es del todo necesario
	cudaMalloc((void**)&dev_suma, sizeof(double));

	reduccion << <1, terminos >> > (dev_datos, dev_suma);

	cudaMemcpy(hst_suma, dev_suma, sizeof(double), cudaMemcpyDeviceToHost);

	

	printf("> La solucion calculada es: %lf\n", *hst_suma);
	printf("> Referencia ln(2) : 0,6931472\n");
	printf("> Error relativo: %lf %",(*hst_suma - 0,6931472)/100);
	printf("> Tiempo de ejecucion")

	// liberacion de memoria
	free(hst_suma);
	cudaFree(dev_suma);
	cudaFree(dev_datos);

}
