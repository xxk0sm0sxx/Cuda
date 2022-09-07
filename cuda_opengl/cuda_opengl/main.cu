#define GL_GLEXT_PROTOTYPES
#include "camera.h"
#include "stdio.h"
#include "GL/glew.h"
#include "gl/freeglut.h"
#include <iostream>
#define ENABLE_CUDA 1
#ifdef ENABLE_CUDA
#include "cuda.h"
#include "cuda_gl_interop.h"
#include "kernel.cuh"
#include "AntTweakBar.h"
#endif
/* LIBRARIES */
#ifdef _M_IX86
#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "AntTweakBar.lib")
#elif _M_X64
#pragma comment(lib, "glew64.lib")
#pragma comment(lib, "freeglut64.lib")
#pragma comment(lib, "AntTweakBar64.lib")
#endif
/* Defines */
#define W_WIDTH 1024
#define W_HEIGHT 768
#define NULL 0
//Macro to calculate the index of the 3D array
#define HIndex(x, y, z) ((z)*N*N + (y)*N + (x))
static void HandleError(cudaError_t err,
	const char *file,
	int line) {
	if (err != cudaSuccess) {
		printf("%s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
/* Structs */
struct sVec {
	float x, y, z;
	sVec(float _x = 0, float _y = 0, float _z = 0) :
		x(_x), y(_y), z(_z) {}

	sVec& operator* (int r) {
		x = x * r;
		y = y * r;
		z = z * r;
		return *this;
	}
};
struct sColor {
	unsigned char r, g, b;
	sColor(unsigned char _r = 0, unsigned char _g = 0, unsigned char _b = 0) :
		r(_r), g(_g), b(_b) {}
};
struct sBlock {
	sVec vt[36];
};
/* Global Variables*/
Camera                 camera;
GLuint                 bufferObj;
GLuint                 colorObj;
#ifdef ENABLE_CUDA
cudaGraphicsResource*  resource;
#endif
float3* devPtr;
int worldTime = 0;
int maxWorldTime = 0;
bool isPaused = true;
int slowdownCounter = 0;
int slowdown = 5;
static void compute_lookat();
static void run_cuda();
//*************************************************************/
//  GLUT Keyboard function
//*************************************************************/
static void key_func(unsigned char key, int x, int y) {
	TwEventKeyboardGLUT(key, x, y);
	switch (key){
		/* ESCAPE key quits the app */
	case 27:
		// clean up OpenGL and CUDA
#ifdef ENABLE_CUDA
		HANDLE_ERROR(cudaGraphicsUnregisterResource(resource));
#endif
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glDeleteBuffers(1, &bufferObj);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glDeleteBuffers(1, &colorObj);
#ifdef ENABLE_CUDA
		cudaFree(dCube1);
		cudaFree(dCube2);
		free(hCube);
#endif
		TwTerminate();
		exit(0);
		break;
		/* SPACEBAR key pause/unpause the simulation */
	case 32:
		isPaused = !isPaused;
		break;
		/* - key zooms out the camera*/
	case '-':
		camera.eyeRadius -= 0.1f * N;
		compute_lookat();
		break;
		/* + key zooms in the camera */
	case '=':
		camera.eyeRadius += 0.1f * N;
		compute_lookat();
		break;
	}
}
//*************************************************************/
//  GLUT Special Keyboard function
//*************************************************************/
void SpecialKeyboard(int key, int x, int y)
{
	TwEventSpecialGLUT(key, x, y);
	/* Right key advances simulation by one frame*/
	if (key == GLUT_KEY_RIGHT) {
		if (isPaused)
			run_cuda();
	}
}
//*************************************************************/
//  GLUT Mouse Click function
//*************************************************************/
bool leftButtonClicked = false;
int xPrev = 0, yPrev = 0;
void MouseClick(int button, int state, int x, int y)
{
	if (TwEventMouseButtonGLUT(button, state, x, y)) return;
	switch (button)
	{
		/* Drag camera */
	case GLUT_LEFT_BUTTON:
		if (state == GLUT_DOWN)
			leftButtonClicked = true;
		else
			leftButtonClicked = false;
		break;
	}
}
//*************************************************************/
//  GLUT Mouse Motion function
//*************************************************************/
void MouseMotion(int x, int y)
{
	/* Drag camera */
	TwEventMouseMotionGLUT(x, y);
	if (leftButtonClicked && xPrev != 0 && yPrev != 0)
	{
		int xDiff = x - xPrev;
		int yDiff = y - yPrev;

		camera.eyeAlpha -= xDiff / 20.0f;
		camera.eyeBeta += yDiff / 20.0f;

		if (camera.eyeAlpha > PI_TWICE)
			camera.eyeAlpha -= PI_TWICE;
		else
		if (camera.eyeAlpha < 0)
			camera.eyeAlpha += PI_TWICE;

		if (camera.eyeBeta > PI_TWICE)
			camera.eyeBeta -= PI_TWICE;
		else
		if (camera.eyeBeta < 0)
			camera.eyeBeta += PI_TWICE;

		compute_lookat();
	}
	xPrev = x;
	yPrev = y;
}
//*************************************************************/
//  GLUT Mouse Passive function
//*************************************************************/
void MousePassiveMotion(int x, int y)
{
	/* Drag camera */
	TwEventMouseMotionGLUT(x, y);
	xPrev = x;
	yPrev = y;
}
//*************************************************************/
//  GLUT Mouse Wheel function
//*************************************************************/
void MouseWheel(int wheel, int direction, int x, int y)
{
	/* Mouse wheel zooms in and out the camera*/
	switch (direction)
	{
	case 1:
		camera.eyeRadius += 0.1f * N;
		break;
	case -1:
		camera.eyeRadius -= 0.1f * N;
		break;
	}
	compute_lookat();
}
//*************************************************************/
//  GLUT Reshape function
//*************************************************************/
static void reshape_func(int w, int h) {
	TwWindowSize(w, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glViewport(0, 0, (GLsizei)w, (GLsizei)h);
	float aspect = (float)w / (float)h;
	gluPerspective(45.0f, aspect, 0.1f, 1000.0f);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}
//*************************************************************/
//  Update gluLookAt based on camera
//*************************************************************/
static void compute_lookat() {
	camera.ComputeEye();
	glLoadIdentity();
	gluLookAt(camera.eye.x, camera.eye.y, camera.eye.z,
		camera.center.x, camera.center.y, camera.center.z,
		camera.upVec.x, camera.upVec.y, camera.upVec.z);
}
//*************************************************************/
//  GLUT Draw function
//*************************************************************/
static void draw_func(void) {
	/* Run cuda and draw graphical representation */
	if (!isPaused && worldTime < maxWorldTime && ++slowdownCounter > slowdown)
	{
		run_cuda();
		slowdownCounter = 0;
	}
	reshape_func(W_WIDTH, W_HEIGHT);
	compute_lookat();
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glClearColor(255, 255, 255, 255);
	glBindBuffer(GL_ARRAY_BUFFER, bufferObj);
	glVertexPointer(3, GL_FLOAT, 0, 0);
	glBindBuffer(GL_ARRAY_BUFFER, colorObj);
	glColorPointer(3, GL_UNSIGNED_BYTE, 0, 0);
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
	glDrawArrays(GL_TRIANGLES, 0, N * N * N * 36);
	glDisableClientState(GL_COLOR_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
	TwDraw();
	glutSwapBuffers();
}
//*************************************************************/
//  Run kernel once
//*************************************************************/
static void run_cuda() {
#ifdef ENABLE_CUDA
	/* Map to cuda buffer */
	size_t size;
	HANDLE_ERROR(cudaGraphicsMapResources(1, &resource, NULL));
	HANDLE_ERROR(
		cudaGraphicsResourceGetMappedPointer((void**)&devPtr,
		&size,
		resource)
		);
	//Copies initial state to device
	cudaMemcpy(dCube1, hCube, sizeof(bool)* N * N * N, cudaMemcpyHostToDevice);
	dim3 dimBlock(blockSize, blockSize, blockSize);
	dim3 dimGrid(N / dimBlock.x + 1, N / dimBlock.y + 1, N / dimBlock.z + 1);
	//Perform simulation ON dCube1 and writes the result TO dCube2 to prevent synchronization problems between blocks
	simulate_draw_cube << <dimGrid, dimBlock >> >(dCube1, dCube2, N, blockSize, devPtr);
	cudaMemcpy(dCube1, dCube2, sizeof(bool)* N * N * N, cudaMemcpyDeviceToDevice);
	//Result is in dCube1. Copy result back to host array.
	cudaMemcpy(hCube, dCube1, sizeof(bool)*N*N*N, cudaMemcpyDeviceToHost);
	HANDLE_ERROR(cudaGraphicsUnmapResources(1, &resource, NULL));
	/* Advance world time */
	++worldTime;
#endif
}
//*************************************************************/
//  Initialize opengl graphical representation
//*************************************************************/
static void graphic_output(unsigned N, unsigned T) {
	maxWorldTime = (int)T;
	if (N > 8)
		blockSize = 8; //Fixed blockSize as 8 for optimal performance
	else
		blockSize = (N / 2) > 0 ? (N / 2) : 1; //If N is less than 8, make sure blockSize is N/2 or at least 1.
	/* Build initial vertex data */
	sBlock* vertex;
	vertex = new sBlock[N * N * N];
	for (float z = 0; z < N; ++z) {
		for (float y = 0; y < N; ++y) {
			for (float x = 0; x < N; ++x) {
				int idx = (int)(z * N * N + y * N + x);
				float size = 0.5f;
				int i = 0;
				/* front */
				vertex[idx].vt[i++] = sVec(x + size, y + size, z + size);
				vertex[idx].vt[i++] = sVec(x - size, y + size, z + size);
				vertex[idx].vt[i++] = sVec(x - size, y - size, z + size);
				vertex[idx].vt[i++] = sVec(x - size, y - size, z + size);
				vertex[idx].vt[i++] = sVec(x + size, y - size, z + size);
				vertex[idx].vt[i++] = sVec(x + size, y + size, z + size);
				/* right */
				vertex[idx].vt[i++] = sVec(x + size, y + size, z - size);
				vertex[idx].vt[i++] = sVec(x + size, y + size, z + size);
				vertex[idx].vt[i++] = sVec(x + size, y - size, z + size);
				vertex[idx].vt[i++] = sVec(x + size, y - size, z + size);
				vertex[idx].vt[i++] = sVec(x + size, y - size, z - size);
				vertex[idx].vt[i++] = sVec(x + size, y + size, z - size);
				/* top */
				vertex[idx].vt[i++] = sVec(x + size, y + size, z - size);
				vertex[idx].vt[i++] = sVec(x - size, y + size, z - size);
				vertex[idx].vt[i++] = sVec(x - size, y + size, z + size);
				vertex[idx].vt[i++] = sVec(x - size, y + size, z + size);
				vertex[idx].vt[i++] = sVec(x + size, y + size, z + size);
				vertex[idx].vt[i++] = sVec(x + size, y + size, z - size);
				/* back */
				vertex[idx].vt[i++] = sVec(x - size, y + size, z - size);
				vertex[idx].vt[i++] = sVec(x + size, y + size, z - size);
				vertex[idx].vt[i++] = sVec(x + size, y - size, z - size);
				vertex[idx].vt[i++] = sVec(x + size, y - size, z - size);
				vertex[idx].vt[i++] = sVec(x - size, y - size, z - size);
				vertex[idx].vt[i++] = sVec(x - size, y + size, z - size);
				/* left */
				vertex[idx].vt[i++] = sVec(x - size, y + size, z + size);
				vertex[idx].vt[i++] = sVec(x - size, y + size, z - size);
				vertex[idx].vt[i++] = sVec(x - size, y - size, z - size);
				vertex[idx].vt[i++] = sVec(x - size, y - size, z - size);
				vertex[idx].vt[i++] = sVec(x - size, y - size, z + size);
				vertex[idx].vt[i++] = sVec(x - size, y + size, z + size);
				/* bottom */
				vertex[idx].vt[i++] = sVec(x - size, y - size, z - size);
				vertex[idx].vt[i++] = sVec(x + size, y - size, z - size);
				vertex[idx].vt[i++] = sVec(x + size, y - size, z + size);
				vertex[idx].vt[i++] = sVec(x + size, y - size, z + size);
				vertex[idx].vt[i++] = sVec(x - size, y - size, z + size);
				vertex[idx].vt[i++] = sVec(x - size, y - size, z - size);
			}
		}
	}
	/* Initialize color data */
	sColor* color;
	color = new sColor[N * N * N * 36];
	for (int x = 0; x < (int)(N*N*N); ++x) {
		for (int i = 0; i < 6; ++i) {
			for (int j = 0; j < 6; ++j) {
				int start = x * 36 + i * 6;
				switch (i) {
				case 0:
					color[start + j] = sColor(0, 192, 255);
					break;
				case 1:
					color[start + j] = sColor(0, 128, 255);
					break;
				case 2:
					color[start + j] = sColor(0, 255, 255);
					break;
				case 3:
					color[start + j] = sColor(0, 160, 255);
					break;
				case 4:
					color[start + j] = sColor(0, 224, 255);
					break;
				case 5:
					color[start + j] = sColor(0, 96, 255);
					break;
				}
			}
		}
	}
	glewInit();
	/* Generate VBO buffer */
	glGenBuffers(1, &bufferObj);
	glBindBuffer(GL_ARRAY_BUFFER, bufferObj);
	/* Copy vertex data*/
	glBufferData(GL_ARRAY_BUFFER, sizeof(sBlock)* N * N * N,
		&vertex[0], GL_DYNAMIC_DRAW_ARB);
	/* Copy color buffer*/
	glGenBuffers(1, &colorObj);
	glBindBuffer(GL_ARRAY_BUFFER, colorObj);
	/* Copy color data*/
	glBufferData(GL_ARRAY_BUFFER, N * N * N * 36 * sizeof(sColor),
		&color[0], GL_DYNAMIC_DRAW_ARB);
	delete[] vertex;
	delete[] color;
	/* Register cuda to VBO */
#ifdef ENABLE_CUDA
	HANDLE_ERROR(
		cudaGraphicsGLRegisterBuffer(&resource,
		bufferObj,
		cudaGraphicsMapFlagsNone));
#endif
	/* Init view matrix */
	glEnable(GL_CULL_FACE);
	reshape_func(W_WIDTH, W_HEIGHT);
	camera.center = glm::vec3(N / 2, N / 2, N / 2);
	camera.eyeRadius = (float)N * 3;
	compute_lookat();
	// set up GLUT and kick off main loop
	glutKeyboardFunc(key_func);
	glutDisplayFunc(draw_func);
	glutIdleFunc(draw_func);
	glutReshapeFunc(reshape_func);
	glutSpecialFunc(SpecialKeyboard);
	glutMouseFunc(MouseClick);
	glutMotionFunc(MouseMotion);
	glutMouseWheelFunc(MouseWheel);
	glutPassiveMotionFunc(MousePassiveMotion);
	glEnable(GL_DEPTH_TEST);
	/* Initialize kernel */
	hCube = (bool*)malloc(sizeof(bool)* N * N * N);
	for (unsigned z = 0; z < N; ++z) {
		for (unsigned y = 0; y < N; ++y) {
			for (unsigned x = 0; x < N; ++x) {
				hCube[HIndex(x, y, z)] = 1;
			}
		}
	}
#ifdef ENABLE_CUDA
	cudaMalloc(&dCube1, sizeof(bool)* N * N * N);
	cudaMalloc(&dCube2, sizeof(bool)* N * N * N);
#endif
	glutMainLoop();
}
//*************************************************************/
//  MAIN function
//*************************************************************/
int main(int argc, char **argv) {
	if (argc < 3) {
		std::cout << "usage: <size of N> <number of cycles> [-demo]" << std::endl;
		return 0;
	}
	N = atoi(argv[1]);
	unsigned T = atoi(argv[2]);
	/* Run graphical demo if -demo is entered */
	if (argc == 4 && strcmp(argv[3], "-demo") == 0) {
#ifdef ENABLE_CUDA
		cudaDeviceProp  prop;
		int             dev;
		memset(&prop, 0, sizeof(cudaDeviceProp));
		prop.major = 1;
		prop.minor = 0;
		HANDLE_ERROR(cudaChooseDevice(&dev, &prop));
		HANDLE_ERROR(cudaGLSetGLDevice(dev));
#endif
		// these GLUT calls need to be made before the other GL calls
		glutInit(&argc, argv);
		glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
		glutInitWindowSize(W_WIDTH, W_HEIGHT);
		glutCreateWindow("Cuda");
		TwInit(TW_OPENGL, NULL);
		TwBar *myBar;
		myBar = TwNewBar("Parameters");
		TwDefine("Parameters color='0 180 255' alpha=85 fontSize=3 position='10 10' size='250 200' valuesWidth=100 ");
		TwAddVarRO(myBar, "N", TW_TYPE_INT32, &N, "");
		TwAddVarRO(myBar, "Time", TW_TYPE_INT32, &worldTime, "");
		TwAddVarRW(myBar, "Slowdown", TW_TYPE_INT32, &slowdown, "min = 1 max=10 step=1");
		TwAddVarRW(myBar, "Paused", TW_TYPE_BOOL8, &isPaused, "");
		graphic_output(N, T);
	}
	else {
		/* Run console output */
		console_output(N, T);
	}
	return 0;
}