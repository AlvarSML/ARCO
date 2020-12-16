/*
* ARQUITECTURA DE COMPUTADORES
* 2º Grado en Ingenieria Informatica
* Curso 2020/21
*
* EXAMEN DE PRÁCTICAS
*
* EQUIPO:
* MIEMBROS: San Martin Liendo Alvar
*           Gonzalez Martinez Sergio
*           Arnaiz Lopez Lucia
*
*/
///////////////////////////////////////////////////////////////////////////
// includes
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>


// Funcion que imprime las propiedades del device
__host__ void propiedades_Device(int deviceID);
__global__ void reduccion(double* datos, double* suma, unsigned int N);
////////////////////////////////////////////////////////////////////


// MAIN
int main(int argc, char** argv)
{
	// Dispositivo CUDA
	int currentDevice;
	cudaGetDevice(&currentDevice);
	propiedades_Device(currentDevice);

	// Vectores de valores
	double* dev_datos;
	double* hst_suma;
	double* dev_suma;

	// Para que el usuario repita el programa de foram indefinida
	unsigned char continuar = 1;

	// Terminos, que se van a calcular tiene que ser potencia de 2 (se asegura)
	unsigned int terminos;

	// Obtiene el numero maximo de hilos por bloque de la GPU 0
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	int maxCores = deviceProp.maxThreadsPerBlock;

	// declaracion de eventos
	cudaEvent_t start;
	cudaEvent_t stop;

	// creacion de eventos
	cudaEventCreate(&start);
	cudaEventCreate(&stop);		

	// Visualizacion de datos GPU
	// propiedades_Device(0);

	printf(">> Introduce la potencia de 2 del numero de terminos (2^x)): ");
	scanf("%i", &terminos);
	fflush(stdout);

	terminos = pow(2,terminos);

	int bloques = ceil((float)terminos / maxCores);

	printf("> Lanzamiento con %i bloques de %i hilos\n",bloques,maxCores);
	fflush(stdout);

	hst_suma = (double*)malloc(sizeof(double));
	cudaMalloc((void**)&dev_datos, terminos * sizeof(double));
	// No es del todo necesario
	cudaMalloc((void**)&dev_suma, sizeof(double));

	// marca de inicio
	cudaEventRecord(start, 0);
	
	// Si hay menos terminos que hilos maximos por bloque, nos ahorramos lanzar hilos insuficientes
	reduccion << <bloques, maxCores >> > (dev_datos, dev_suma, terminos);
	

	cudaMemcpy(hst_suma, dev_suma, sizeof(double), cudaMemcpyDeviceToHost);
	// Depende le la ocasion hay que temporizar o no la copia de datos
	cudaEventRecord(stop, 0);
	// sincronizacion GPU-CPU
	cudaEventSynchronize(stop);

	printf("> La solucion calculada es: %lf\n", *hst_suma);
	printf("> La solucion para pi es: %lf\n", (*hst_suma * 4));
	printf("> Referencia pi : 3,14159265\n");
	printf("> Error relativo: %lf %%\n", 100 * ((3.14159265- (*hst_suma*4)) / 3.14159265));
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	// impresion de resultados
	printf("> Tiempo de ejecucion: %f ms\n", elapsedTime);



	// Informacion del Sistema
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

	// SALIDA DEL PROGRAMA
	printf("<pulsa [INTRO] para finalizar>");
	getchar();
	return 0;
}

__global__ void reduccion(double* datos, double* suma, unsigned int N)
// Funcion que suma los primeros N numeros naturales
{
	// indice local de cada hilo -> kernel con un solo bloque de N hilos
	// Adaptado a multiples bloques
	unsigned int myID = threadIdx.x + blockDim.x * blockIdx.x;
	if (myID < N) {
		// Rellenamos el vector de datos M = 2 * n - 1
		double dato = 1 / (2 * ((double)myID + 1) - 1);

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
		unsigned int salto = N / 2;
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
}


__global__ void reduccionSuma(double* datos, double* suma, unsigned int N) {
	// N -> numero de bloques

	unsigned int myID = threadIdx.x;
	if (myID < N) {
		// sincronizamos para evitar riesgos de tipo RAW
		__syncthreads();

		// ******************
		// REDUCCION PARALELA
		// ******************
		unsigned int salto = N / 2;
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
}

////////////////////////////////////////////////////////////////////
__host__ void propiedades_Device(int deviceID)
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, deviceID);
	int SM = deviceProp.multiProcessorCount;
	int maxThreads = deviceProp.maxThreadsPerBlock;
	int major = deviceProp.major;
	int minor = deviceProp.minor;
	const char* ArchName;
	// calculo del numero de cores (SP)
	int cudaCores = 0;
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
	printf("> No. maximo de Hilos (por bloque)\t: %d\n", maxThreads);
	printf("***************************************************\n");
}