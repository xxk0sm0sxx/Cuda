#ifndef KERNEL_H
#define KERNEL_H

#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "device_launch_parameters.h"

__global__ void simulate_draw_cube(bool *oricube, bool *newcube, int N, int blockSize, float3* vbo);
__global__ void simulate(bool *oricube, bool *newcube, int N, int blockSize);
void PrintCube(bool* cube);
void console_output(unsigned N, unsigned T);

//Cube size and size of each thread block
extern unsigned N;
extern unsigned blockSize;
extern bool *hCube;
extern bool *dCube1;
extern bool *dCube2;

#endif