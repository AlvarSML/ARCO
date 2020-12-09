///////////////////////////////////////////////////////////////////////////
/*
* ARQUITECTURA DE COMPUTADORES
* 2º Grado en Ingenieria Informatica
*
* GPU_BITMAP.H
* Archivo de cabecera para la impresión grafica por pantalla con OpenGL
*/
///////////////////////////////////////////////////////////////////////////
/*
Este archivo de cabecera debe ir acompañado de los siguientes archivos:

<1> glut.h
<2> glext.h
-> Comprobar que se encuentran junto al fichero fuente.

<3> glut32.lib
<4> glut64.lib
-> Comprobar que se encuentran junto al fichero fuente.

<5> glut32.dll
<6> glut64.dll
-> Comprobar que se encuentran junto al fichero ejecutable.
*/

#ifndef __GPU_BITMAP_H__
#define __GPU_BITMAP_H__
/*
On 64-bit Windows, we need to prevent GLUT from automatically linking against
glut32. We do this by defining GLUT_NO_LIB_PRAGMA. This means that we need to
manually add opengl32.lib and glut64.lib back to the link using pragmas.
Alternatively, this could be done on the compilation/link command-line, but
we chose this so that compilation is identical between 32- and 64-bit Windows.
*/
#ifdef _WIN64
#define GLUT_NO_LIB_PRAGMA
#pragma comment (lib, "opengl32.lib")  /* link with Microsoft OpenGL lib */
#pragma comment (lib, "glut64.lib")    /* link with Win64 GLUT lib */
#endif //_WIN64


#ifdef _WIN32
/* On Windows, include local copy of glut.h and glext.h */
#include "glut.h"
#include "glext.h"

#define GET_PROC_ADDRESS( str ) wglGetProcAddress( str )
#endif //WIN32


#ifdef __APPLE__
/* On OSX include the system's copy of glut.h, glext.h */
// Added by Mario Bartolome due nomenclature changes on Apple's OpenGL and GL Frameworks.
// OSX framework names no longer coincide with *nix.
#include <GLUT/glut.h>
#include <OpenGL/glext.h>

#define GET_PROC_ADDRESS( str ) glXGetProcAddress( (const GLubyte *)str )
#endif //OSX


#ifdef __linux__
/* On Linux, include the system's copy of glut.h, glext.h, and glx.h */
#include <GL/glut.h>
#include <GL/glext.h>
#include <GL/glx.h>
#endif //*nix

#endif  // __GPU_BITMAP_H__

struct RenderGPU
{
	// Framebuffer
	unsigned char *pixels;
	// Dimensiones de la ventana
	int x, y;
	// Tamaño del framebuffer (host)
	long size;
	// Parametros para animacion con CUDA
	int trigger;
	int anime;
	// Memoria del device
	unsigned char *device_memory;
	// Funcion de llamada al kernel
	void(*rendering_kernel)(unsigned char*, unsigned char*, long);

	// Constructor
	RenderGPU(int width, int height)
	{
		// Reserva para cuatro canales (RGBA)
		pixels = (unsigned char*)malloc(width * height * 4);
		x = width;
		y = height;
		size = x * y * 4;
		// temporizador para las animaciones (en ms)
		trigger = 0;
		anime = false;
	}

	// Destructor
	~RenderGPU()
	{
		free(pixels);
	}

	// Puntero al framebuffer
	unsigned char* get_ptr(void) const
	{
		return pixels;
	}

	// Tamaño del framebuffer
	long image_size(void) const
	{
		return x * y * 4;
	}

	// Establece el temporizador para las animaciones (tiempo en ms)
	void set_timer(int tiempo)
	{
		trigger = tiempo;
		anime = true;
	}

	// static method used for glut callbacks
	// Puntero a la estructura RenderGPU
	static RenderGPU** get_bitmap_ptr(void)
	{
		static RenderGPU *gBitmap;
		return &gBitmap;
	}

	// static method used for glut callbacks
	// Tecla de salida: [esc]
	static void Key(unsigned char key, int x, int y)
	{
		switch (key)
		{
		case 27:
			// Cerramos ventana y salimos del main()
			exit(0);
		}
	}

	// static method used for glut callbacks
	// Dibuja la imagen
	static void Draw(void)
	{
		RenderGPU* bitmap = *(get_bitmap_ptr());
		glClearColor(0.0, 1.0, 0.0, 1.0);
		glClear(GL_COLOR_BUFFER_BIT);
		glDrawPixels(bitmap->x, bitmap->y, GL_RGBA, GL_UNSIGNED_BYTE, bitmap->pixels);
		// Flush drawing commands and swap
		glutSwapBuffers();
		// En caso de animaciones lanzamos un temporizador para generar el nuevo frame
		if (bitmap->anime)
			glutTimerFunc(bitmap->trigger, newFrame, 1);
	}

	// static method used for glut callbacks
	// Genera el nuevo frame de una animacion lanzando un kernel
	static void newFrame(int value)
	{
		RenderGPU* bitmap = *(get_bitmap_ptr());
		// lanzamos el kernel generador del frame
		bitmap->rendering_kernel(bitmap->pixels, bitmap->device_memory, bitmap->size);
		// Redibujamos el bitmap: callback to Draw()
		glutPostRedisplay();
	}

	// Animacion y salida
	void anime_and_exit(void(*f)(unsigned char*, unsigned char*, long))
	{
		RenderGPU** bitmap = get_bitmap_ptr();
		*bitmap = this;
		rendering_kernel = f;
		// a bug in the Windows GLUT implementation prevents us from
		// passing zero arguments to glutInit()
		int c = 1;
		char* dummy = (char*)"none";
		// generamos el bitmap inicial lanzando un kernel
		rendering_kernel(pixels, device_memory, size);
		// iniciamos el bucle de openGL
		glutInit(&c, &dummy);
		glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
		glutInitWindowSize(x, y);
		glutCreateWindow("Anime: Arquitectura de Computadores");
		glutKeyboardFunc(Key);
		glutDisplayFunc(Draw);
		glutMainLoop();
	}

	// Visualizacion y salida
	void display_and_exit(void)
	{
		RenderGPU** bitmap = get_bitmap_ptr();
		*bitmap = this;
		// a bug in the Windows GLUT implementation prevents us from
		// passing zero arguments to glutInit()
		int c = 1;
		char* dummy = (char*)"none";
		// iniciamos el bucle de openGL
		glutInit(&c, &dummy);
		glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
		glutInitWindowSize(x, y);
		glutCreateWindow("Display: Arquitectura de Computadores");
		glutKeyboardFunc(Key);
		glutDisplayFunc(Draw);
		glutMainLoop();
	}
};