/*
* ARQUITECTURA DE COMPUTADORES
* 2º Grado en Ingenieria Informatica
* Curso 2020/21
*
* ENTREGA no.3 Gráficos en CUDA.
*
* EQUIPO:   G6 ARCO 103
* MIEMBROS: Gonzalez Martinez Sergio
*           Arnaiz Lopez Lucia
*           San Martin Liendo Alvar
*
*/
///////////////////////////////////////////////////////////////////////////
// includes
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "gpu_bitmap.h"
// defines
#define TCASILLA 4
#define ANCHO 128* TCASILLA // Dimension horizontal
#define ALTO 128*TCASILLA // Dimension vertical
// GLOBAL: funcion llamada desde el host y ejecutada en el device (kernel)
__global__ void kernel(unsigned char* imagen)
{
	// ** Kernel bidimensional multibloque **
	//
	// coordenada horizontal de cada hilo
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	// coordenada vertical de cada hilo
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	// indice global de cada hilo (indice lineal para acceder a la memoria)
	int myID = x + y * blockDim.x * gridDim.x;
	// cada hilo obtiene la posicion de su pixel
	int miPixel = myID * 4;
	int idBloque = blockIdx.x / TCASILLA + blockIdx.y / TCASILLA;
	if (idBloque % 2 == 0) {
		imagen[miPixel + 0] = 0; // canal R
		imagen[miPixel + 1] = 0; // canal G
		imagen[miPixel + 2] = 0; // canal B
		imagen[miPixel + 3] = 0; // canal alf
	// cada hilo rellena los 4 canales de su pixel con un valor

	}
	else {
		imagen[miPixel + 0] = 255; // canal R
		imagen[miPixel + 1] = 255; // canal G
		imagen[miPixel + 2] = 255; // canal B
		imagen[miPixel + 3] = 255; // canal alf
	}

}
// MAIN: rutina principal ejecutada en el host
int main(int argc, char** argv)
{
	// Declaracion del bitmap:
	// Inicializacion de la estructura RenderGPU
	RenderGPU foto(ANCHO, ALTO);
	// Tamaño del bitmap en bytes
	size_t size = foto.image_size();
	// Asignacion y reserva de la memoria en el host (framebuffer)
	unsigned char* host_bitmap = foto.get_ptr();
	// Reserva en el device
	unsigned char* dev_bitmap;
	// declaracion de eventos
	cudaEvent_t start;
	cudaEvent_t stop;
	// creacion de eventos
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaMalloc((void**)&dev_bitmap, size);
	// Lanzamos un kernel bidimensional con bloques de 256 hilos (16x16)
	dim3 hilosB(16, 16);
	// Calculamos el numero de bloques necesario (un hilo por cada pixel)
	dim3 Nbloques(ANCHO / 16, ALTO / 16);
	// marca de inicio
	cudaEventRecord(start, 0);
	// Generamos el bitmap
	kernel << <Nbloques, hilosB >> > (dev_bitmap);
	// Copiamos los datos desde la GPU hasta el framebuffer para visualizarlos
	cudaMemcpy(host_bitmap, dev_bitmap, size, cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);
	// sincronizacion GPU-CPU
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	// impresion de resultados
	printf("> Tiempo de ejecucion: %f ms\n", elapsedTime);
	// Visualizacion y salida
	printf("\n...pulsa [ESC] para finalizar...");
	foto.display_and_exit();

	return 0;
}