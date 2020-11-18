/*
* ARQUITECTURA DE COMPUTADORES
* 2º Grado en Ingenieria Informatica
* Curso 2020/21
*
* ENTREGA no.2 Ordenación por rango
*
* EQUIPO:   G6 ARCO 103
* MIEMBROS: Gonzalez Martinez Sergio
*           Arnaiz Lopez Lucia
*           San Martin Liendo Alvar
*
*/
///////////////////////////////////////////////////////////////////////////
// includes
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

__host__ void propiedades_Device(int deviceID = 0)
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, deviceID);
	// calculo del numero de cores (SP)
	int cudaCores = 0;
	int SM = deviceProp.multiProcessorCount;
	int major = deviceProp.major;
	int minor = deviceProp.minor;
	const char* ArchName;
	switch (major)
	{
	case 1:
		//TESLA
		ArchName = "TESLA";
		cudaCores = 8;
		break;
	case 2:
		//FERMI
		ArchName = "FERMI";
		if (minor == 0)
			cudaCores = 32;
		else
			cudaCores = 48;
		break;
	case 3:
		//KEPLER
		ArchName = "KEPLER";
		cudaCores = 192;
		break;
	case 5:
		//MAXWELL
		ArchName = "MAXWELL";
		cudaCores = 128;
		break;
	case 6:
		//PASCAL
		ArchName = "PASCAL";
		cudaCores = 64;
		break;
	case 7:
		//VOLTA (7.0) TURING (7.5)
		cudaCores = 64;
		if (minor == 0)
			ArchName = "VOLTA";
		else
			ArchName = "TURING";
		break;
	case 8:
		//AMPERE
		ArchName = "AMPERE";
		cudaCores = 64;
		break;
	default:
		//ARQUITECTURA DESCONOCIDA
		ArchName = "DESCONOCIDA";
		cudaCores = 0;
		printf("!!!!!dispositivo desconocido!!!!!\n");
	}
	// presentacion de propiedades
	printf("***************************************************\n");
	printf("DEVICE %d: %s\n", deviceID, deviceProp.name);
	printf("***************************************************\n");
	printf("> Capacidad de Computo            \t: %d.%d\n", major, minor);
	printf("> Arquitectura CUDA               \t: %s\n", ArchName);
	printf("> No. de MultiProcesadores        \t: %d\n", SM);
	printf("> No. de CUDA Cores (%dx%d)       \t: %d\n", cudaCores, SM, cudaCores * SM);
	printf("> Memoria Global (total)          \t: %zu MiB\n", deviceProp.totalGlobalMem / (1024 * 1024));
	printf("> No. maximo de Hilos (por bloque)\t: %d\n", deviceProp.maxThreadsPerBlock);
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

__global__ void ordenarPorRango(int* arrOrg, int* arrDest, int n) {
	int myID = threadIdx.x;
	int rango = 0;
	int valor = arrOrg[myID];

	for (int i = 0; i < n; i++) {
		if (valor > arrOrg[i] || (valor == arrOrg[i] && i > myID)) {
			rango++;
		}
	}

	//rintf("Rango del %i = %i\n", valor, rango);

	arrDest[rango] = valor;
}

int main(int argc, char** argv)
{
	int n;
	int* hst_datos, * hst_ordenado;
	int* dev_datos, * dev_ordenado;

	do {
		printf("Numeros a ordenar(0-50): ");
		scanf("%i", &n);
	} while (n < 0 && n > 50);

	hst_datos = (int*)malloc(n * sizeof(int));
	hst_ordenado = (int*)malloc(n * sizeof(int));

	cudaMalloc((void**)&dev_datos, n * sizeof(int));
	cudaMalloc((void**)&dev_ordenado, n * sizeof(int));

	srand(time(NULL));
	for (int i = 0; i < n;i++) {
		hst_datos[i] = rand() % 10;
		hst_ordenado[i] = 0;
		printf("[%i]", hst_datos[i]);
	}
	printf("\n");
	printf("\n");

	cudaMemcpy(dev_datos, hst_datos, n * sizeof(int), cudaMemcpyHostToDevice);

	ordenarPorRango << <1, n >> > (dev_datos, dev_ordenado, n);

	cudaMemcpy(hst_ordenado, dev_ordenado, n * sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < n; i++) {
		printf("[%i]",hst_ordenado[i]);
	}
	printf("\n");

	// SALIDA DEL PROGRAMA
	char infoName[1024];
	char infoUser[1024];
	DWORD  longitud;
	GetComputerName(infoName, &longitud);
	GetUserName(infoUser, &longitud);
	time_t fecha;
	time(&fecha);
	printf("\n***************************************************\n");
	printf("Programa ejecutado el dia: %s", ctime(&fecha));
	printf("Maquina: %s\n", infoName);
	printf("Usuario: %s\n", infoUser);
	printf("<pulsa [INTRO] para finalizar>");
	getchar();
	printf("<Pulsa intro para terminar>");
	return 0;



}