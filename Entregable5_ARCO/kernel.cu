///////////////////////////////////////////////////////////////////////////
// includes
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <math.h>
#ifdef __linux__
#include <sys/time.h>
typedef struct timeval event;
#else
#include <windows.h>
typedef LARGE_INTEGER event;
#endif
///////////////////////////////////////////////////////////////////////////

// declaracion de funciones
// HOST: funcion llamada desde el host y ejecutada en el host
__host__ void setEvent(event* ev)
/* Descripcion: Genera un evento de tiempo */
{
#ifdef __linux__
	gettimeofday(ev, NULL);
#else
	QueryPerformanceCounter(ev);
#endif
}
__host__ double eventDiff(event* first, event* last)
/* Descripcion: Devuelve la diferencia de tiempo (en ms) entre dos eventos */
{
#ifdef __linux__
	return
		((double)(last->tv_sec + (double)last->tv_usec / 1000000) -
			(double)(first->tv_sec + (double)first->tv_usec / 1000000)) * 1000.0;
#else
	event freq;
	QueryPerformanceFrequency(&freq);
	return ((double)(last->QuadPart - first->QuadPart) / (double)freq.QuadPart) * 1000.0;
#endif
}

__global__ void ordenarPorRango(int* arrOrg, int* arrDest, int n) {
	int myID = threadIdx.x + blockIdx.x * blockDim.x;
	int rango = 0;
	int valor = arrOrg[myID];

	if (myID < n) {
		for (int i = 0; i < n; i++) {
			if (valor > arrOrg[i] || (valor == arrOrg[i] && i > myID)) {
				rango++;
			}
		}
	}
	

	//rintf("Rango del %i = %i\n", valor, rango);

	arrDest[rango] = valor;
}

__host__ void ordenarPorRangoCPU(int* arrOrg, int* arrDest, int n) {
	// Se sustituye el id por j
	int rango = 0;

	for (int j = 0; j < n; j++) {
		for (int i = 0; i < n; i++) {
			if (arrOrg[j] > arrOrg[i] || (arrOrg[j] == arrOrg[i] && i > j)) {
				rango++;
			}
		}

		arrDest[rango] = arrOrg[j];

		rango = 0;
	}

}

// Orden == potencia de 2 hasta la que calcular
double calcularCPU(int orden) {
	int* hst_A, * hst_B;
	
	int len = pow(2, orden);

	// reserva en el host
	hst_A = (int*)malloc(len * sizeof(float));
	hst_B = (int*)malloc(len * sizeof(float));

	// inicializacion
	srand((int)time(NULL));
	for (int i = 0; i < len; i++)
	{
		hst_A[i] = rand() % 51;
	}

	// La funcion eventDiff() calcula la diferencia de tiempo (en milisegundos) entre dos eventos.
	event start; // variable para almacenar el evento de tiempo inicial.
	event stop; // variable para almacenar el evento de tiempo final.
	double t_ms;


	// Iniciamos el contador
	setEvent(&start); // marca de evento inicial

	ordenarPorRangoCPU(hst_A, hst_B, len);

	// Paramos el contador
	setEvent(&stop);// marca de evento final

	// Intervalos de tiempo
	t_ms = eventDiff(&start, &stop); // diferencia de tiempo en ms
	//printf("En CPU con %i valores tarda %lf\n", len, t_ms);
	return t_ms;
}

// Orden == potencia de 2 hasta la que calcular
double calcularGPU(int orden) {
	int len = pow(2,orden);
	int* hst_A,* dev_A,* dev_B;

	// reserva en el host
	hst_A = (int*)malloc(len * sizeof(float));
	cudaMalloc((void**)&dev_B, len * sizeof(int));
	cudaMalloc((void**)&dev_A, len * sizeof(int));

	// Numero de bloques
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	int cores = deviceProp.maxThreadsPerBlock;

	int bloques = ceil(len / cores);

	// declaracion de eventos
	cudaEvent_t startDev;
	cudaEvent_t stopDev;

	// creacion de eventos
	cudaEventCreate(&startDev);
	cudaEventCreate(&stopDev);

	// inicializacion
	srand((int)time(NULL));
	for (int i = 0; i < len; i++)
	{
		hst_A[i] = rand() % 51;
	}

	cudaMemcpy(dev_A, hst_A, len * sizeof(int), cudaMemcpyHostToDevice);

	cudaEventRecord(startDev, 0);
	ordenarPorRango << <bloques, cores >> > (dev_A, dev_B, len);
	cudaEventRecord(stopDev, 0);

	// sincronizacion GPU-CPU
	cudaEventSynchronize(stopDev);

	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, startDev, stopDev);
	//printf("En GPU con %i valores tarda %lf\n", len, elapsedTime);
	return (double)elapsedTime;

}


///////////////////////////////////////////////////////////////////////////
// MAIN: rutina principal ejecutada en el host
int main(int argc, char** argv)
{
	int orden;
	const unsigned char OFFSET = 5;

	printf("Hasta que potencia quieres calcular (desde 2^5 (32))[1-n]: ");
	scanf("%i",&orden);

	double* tiemposCPU = (double*)malloc(orden * sizeof(double));
	double* tiemposGPU = (double*)malloc(orden * sizeof(double));

	for (int i = 0; i < orden; i++) {
		tiemposCPU[i] = calcularCPU((i + OFFSET));
		tiemposGPU[i] = calcularGPU((i + OFFSET));
	}

	int potencia;

	printf("   N   ");
	for (int i = 0; i < orden; i++) {
		potencia = pow(2,(i + OFFSET));
		printf("    %d    ",potencia);
	}
	printf("\n");

	printf("  CPU  ");
	for (int i = 0; i < orden; i++) {
		printf("%.8lf ", tiemposCPU[i]);
	}
	printf("\n");

	printf("  GPU  ");
	for (int i = 0; i < orden; i++) {
		printf("%.8lf ", tiemposGPU[i]);
	}
	printf("\n");

	// salida
	printf("\npulsa INTRO para finalizar...");
	fflush(stdin);
	getchar();
	return 0;
}