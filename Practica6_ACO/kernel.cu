/*
* ARQUITECTURA DE COMPUTADORES
* 2º Grado en Ingenieria Informatica
* Curso 2020/21
*
* ENTREGA no.1 Reduccion Paralela
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
	char* ArchName;
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

// GLOBAL: funcion llamada desde el host y ejecutada en el device (kernel)
__global__ void reduccion(double* datos, double* suma, int N)
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

	// Se puede poner un if con el signo
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
	}
}

int main(int argc, char** argv)
{
	double* dev_datos;

	double* hst_suma;
	double* dev_suma;

	// declaracion de eventos
	cudaEvent_t start;
	cudaEvent_t stop;
	// creacion de eventos
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Terminos tiene que ser potencia de 2
	int terminos;
	int max = getSPcores();

	// Visualizacion de datos GPU
	propiedades_Device(0);

	printf(">> Introduce el numero de terminos (potencia de 2): ");
	scanf("%i", &terminos);
	fflush(stdout);

	printf("> Lanzamiento con 1 bloque de %i hilos\n",terminos);
	fflush(stdout);

	hst_suma = (double*)malloc(sizeof(double));
	cudaMalloc((void**)&dev_datos, terminos * sizeof(double));
	// No es del todo necesario
	cudaMalloc((void**)&dev_suma, sizeof(double));

	// marca de inicio
	cudaEventRecord(start, 0);

	reduccion << <1, terminos >> > (dev_datos, dev_suma, terminos);

	cudaMemcpy(hst_suma, dev_suma, sizeof(double), cudaMemcpyDeviceToHost);
	// Depende le la ocasion hay que temporizar o no la copia de datos
	cudaEventRecord(stop, 0);
	// sincronizacion GPU-CPU
	cudaEventSynchronize(stop);
	
	printf("> La solucion calculada es: %lf\n", *hst_suma);
	printf("> Referencia ln(2) : 0.6931472\n");
	printf("> Error relativo: %lf %%\n",100 *((*hst_suma - 0.6931472)/ 0.6931472));
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	// impresion de resultados
	printf("> Tiempo de ejecucion: %f ms\n", elapsedTime);
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
	return 0;
	printf("<Pulsa intro para terminar>");
	// liberacion de memoria
	free(hst_suma);
	cudaFree(dev_suma);
	cudaFree(dev_datos);

}
