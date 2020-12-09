/*
* ARQUITECTURA DE COMPUTADORES
* 2º Grado en Ingenieria Informatica
* Curso 2020/21
*
* ENTREGA no.3 Procesamiento de imágenes
*
* EQUIPO:   G6 ARCO 103
* MIEMBROS: Gonzalez Martinez Sergio
*           Arnaiz Lopez Lucia
*           San Martin Liendo Alvar
*
*/
/////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////
// includes
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "gpu_bitmap.h"
#include <math.h>
#include <time.h>
// HOST: funcion llamada desde el host y ejecutada en el host
__host__ void leerBMP_RGBA(const char* nombre, int* w, int* h, unsigned char** imagen);
int getSPcores();
__host__ void propiedades_Device(int deviceID);
// Funcion que lee un archivo de tipo BMP:
// -> entrada: nombre del archivo
// <- salida : ancho de la imagen en pixeles
// <- salida : alto de la imagen en pixeles
// <- salida : puntero al array de datos de la imagen en formato RGBA
///////////////////////////////////////////////////////////////////////////

__global__ void escalaGrises(int len, unsigned char* imagen) {
	// coordenada horizontal de cada hilo
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	// coordenada vertical de cada hilo
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	// indice global de cada hilo (indice lineal para acceder a la memoria)
	int myID = x + y * blockDim.x * gridDim.x;

	int miPixel = myID * 4;
		
	if (myID <= len) {
		//Y = 0, 299×R + 0, 587×G + 0, 114×B
		unsigned char R, G, B;

		R = imagen[miPixel + 0];
		G = imagen[miPixel + 1];
		B = imagen[miPixel + 2];

		float gris = R * 0.299 + G * 0.587 + B * 0.114;

		imagen[miPixel + 0] = gris;
		imagen[miPixel + 1] = gris;
		imagen[miPixel + 2] = gris;
	}

}

// MAIN: rutina principal ejecutada en el host
int main(int argc, char** argv)
{
	// Info de la GPU
	propiedades_Device(0);

	// Eventos
	// declaracion de eventos
	cudaEvent_t start;
	cudaEvent_t stop;
	// creacion de eventos
	cudaEventCreate(&start);
	cudaEventCreate(&stop);


	// Leemos el archivo BMP
	unsigned char* host_color;
	int ancho, alto;
	leerBMP_RGBA("imagen.bmp", &ancho, &alto, &host_color);


	// Declaracion del bitmap RGBA:
	// Inicializacion de la estructura RenderGPU
	RenderGPU foto(ancho, alto);

	// Tamaño del bitmap en bytes
	size_t img_size = foto.image_size();

	// Asignacion y reserva de la memoria en el host (framebuffer)
	unsigned char* host_bitmap = foto.get_ptr();

	// Numero de nucleos cuda
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	int cores = deviceProp.maxThreadsPerBlock;
	
	// Se obtiene su raiz
	int dimension = sqrt(cores);
	float Bancho = ceil((float)ancho / (float)dimension);
	float Balto = ceil((float)alto / (float)dimension);


	// Dimensiones del kernel
	dim3 Nhilos(dimension, dimension);
	dim3 Nbloques(Bancho,Balto);

	// Reserva de memoria en la CPU para la imagen
	unsigned char* dev_imagen;
	cudaMalloc((void**)&dev_imagen, img_size * sizeof(unsigned char));
	cudaMemcpy(dev_imagen, host_color, img_size * sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaEventRecord(start, 0);
	
	escalaGrises << <Nbloques, Nhilos >> > (ancho*alto, dev_imagen);

	cudaEventRecord(stop, 0);
	// Recogemos la imagen de la CPU
	cudaMemcpy(host_bitmap, dev_imagen, img_size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	

	// Salida
	printf("\n...pulsa [ESC] para finalizar...");
	
	// Fin
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
	// Visualizacion
	foto.display_and_exit();

	getchar();	

	return 0;
}

// Funcion que lee un archivo de tipo BMP:
__host__ void leerBMP_RGBA(const char* nombre, int* w, int* h, unsigned char** imagen)
{
	// Lectura del archivo .BMP
	FILE* archivo;

	// Abrimos el archivo en modo solo lectura binario
	if ((archivo = fopen(nombre, "rb")) == NULL)
	{
		printf("\nERROR ABRIENDO EL ARCHIVO %s...", nombre);
		// salida
		printf("\npulsa [INTRO] para finalizar");
		getchar();
		exit(1);
	}
	printf("> Archivo [%s] abierto:\n", nombre);
	// En Windows, la cabecera tiene un tamaño de 54 bytes:
	// 14 bytes (BMP header) + 40 bytes (DIB header)

	// BMP HEADER
	// Extraemos cada campo y lo almacenamos en una variable del tipo adecuado
	// posicion 0x00 -> Tipo de archivo: "BM" (leemos 2 bytes)
	unsigned char tipo[2];
	fread(tipo, 1, 2, archivo);
	// Comprobamos que es un archivo BMP
	if (tipo[0] != 'B' || tipo[1] != 'M')
	{
		printf("\nERROR: EL ARCHIVO %s NO ES DE TIPO BMP...", nombre);
		// salida
		printf("\npulsa [INTRO] para finalizar");
		getchar();
		exit(1);
	}
	// posicion 0x02 -> Tamaño del archivo .bmp (leemos 4 bytes)
	unsigned int file_size;
	fread(&file_size, 4, 1, archivo);

	// posicion 0x06 -> Campo reservado (leemos 2 bytes)
	// posicion 0x08 -> Campo reservado (leemos 2 bytes)
	unsigned char buffer[4];
	fread(buffer, 1, 4, archivo);

	// posicion 0x0A -> Offset a los datos de imagen (leemos 4 bytes)
	unsigned int offset;
	fread(&offset, 4, 1, archivo);

	// imprimimos los datos
	printf(" \nDatos de la cabecera BMP\n");
	printf("> Tipo de archivo : %c%c\n", tipo[0], tipo[1]);
	printf("> Tamano del archivo : %u KiB\n", file_size / 1024);
	printf("> Offset de datos : %u bytes\n", offset);
	// DIB HEADER
	// Extraemos cada campo y lo almacenamos en una variable del tipo adecuado
	// posicion 0x0E -> Tamaño de la cabecera DIB (BITMAPINFOHEADER) (leemos 4 bytes)
	unsigned int header_size;
	fread(&header_size, 4, 1, archivo);

	// posicion 0x12 -> Ancho de la imagen (leemos 4 bytes)
	unsigned int ancho;
	fread(&ancho, 4, 1, archivo);

	// posicion 0x16 -> Alto de la imagen (leemos 4 bytes)
	unsigned int alto;
	fread(&alto, 4, 1, archivo);

	// posicion 0x1A -> Numero de planos de color (leemos 2 bytes)
	unsigned short int planos;
	fread(&planos, 2, 1, archivo);

	// posicion 0x1C -> Profundidad de color (leemos 2 bytes)
	unsigned short int color_depth;
	fread(&color_depth, 2, 1, archivo);

	// posicion 0x1E -> Tipo de compresion (leemos 4 bytes)
	unsigned int compresion;
	fread(&compresion, 4, 1, archivo);

	// imprimimos los datos
	printf(" \nDatos de la cabecera DIB\n");
	printf("> Tamano de la cabecera: %u bytes\n", header_size);
	printf("> Ancho de la imagen : %u pixeles\n", ancho);
	printf("> Alto de la imagen : %u pixeles\n", alto);
	printf("> Planos de color : %u\n", planos);
	printf("> Profundidad de color : %u bits/pixel\n", color_depth);
	printf("> Tipo de compresion : %s\n", (compresion == 0) ? "none" : "unknown");
	// LEEMOS LOS DATOS DEL ARCHIVO
	// Calculamos espacio para una imagen de tipo RGBA:
	size_t img_size = ancho * alto * 4;
	// Reserva para almacenar los datos del bitmap
	unsigned char* datos = (unsigned char*)malloc(img_size);;
	// Desplazamos el puntero FILE hasta el comienzo de los datos de imagen: 0 + offset
	fseek(archivo, offset, SEEK_SET);
	// Leemos pixel a pixel, reordenamos (BGR -> RGB) e insertamos canal alfa
	unsigned int pixel_size = color_depth / 8;
	for (unsigned int i = 0; i < ancho * alto; i++)
	{
		fread(buffer, 1, pixel_size, archivo); // leemos el pixel i
		datos[i * 4 + 0] = buffer[2]; // escribimos canal R
		datos[i * 4 + 1] = buffer[1]; // escribimos canal G
		datos[i * 4 + 2] = buffer[0]; // escribimos canal B
		datos[i * 4 + 3] = buffer[3]; // escribimos canal alfa (si lo hay)
	}
	// Cerramos el archivo
	fclose(archivo);
	// PARAMETROS DE SALIDA
	// Ancho de la imagen en pixeles
	*w = ancho;
	// Alto de la imagen en pixeles
	*h = alto;
	// Puntero al array de datos RGBA
	*imagen = datos;
	// Salida
	return;
}

// Obtiene el numero de nucleos
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