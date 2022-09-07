#include "kernel.cuh"

#include <iostream>
/* Global Variables*/
unsigned N;
unsigned blockSize;
bool *hCube;
bool *dCube1;
bool *dCube2;
//Macro to calculate the index of the 3D array
#define HIndex(x, y, z) ((z)*N*N + (y)*N + (x))
//*************************************************************/
//  Simulate one frame of the world and write triangles into VBO
//*************************************************************/
__global__ void simulate_draw_cube(bool *oricube, bool *newcube, int N, int blockSize, float3* vbo) {
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int z = threadIdx.z + blockDim.z * blockIdx.z;
	//Did not use shared memory as I found after testing that it was twice as slow as only using global alone
	//Accumulates amount of cubes around each index
	int count = 0;
	//Make sure within boundaries
	if (x >= 0 && y >= 0 && z >= 0 && x < N && y < N && z < N) {
		//For each cube around each index, add 1
		if (x > 0 && oricube[HIndex(x - 1, y, z)] == 1) {
			++count;
		}
		if (y > 0 && oricube[HIndex(x, y - 1, z)] == 1) {
			++count;
		}
		if (z > 0 && oricube[HIndex(x, y, z - 1)] == 1) {
			++count;
		}
		if (x < (N - 1) && oricube[HIndex(x + 1, y, z)] == 1) {
			++count;
		}
		if (y < (N - 1) && oricube[HIndex(x, y + 1, z)] == 1) {
			++count;
		}
		if (z < (N - 1) && oricube[HIndex(x, y, z + 1)] == 1) {
			++count;
		}
		//Simulate world behaviour
		if (oricube[HIndex(x, y, z)] == 1) { //index has cube 
			if (count % 2 == 0) { //Surrounded by even cubes
				newcube[HIndex(x, y, z)] = 0; //becomes hole
			}
			else {
				newcube[HIndex(x, y, z)] = 1; //remains as cube
			}
		}
		else { //index has hole
			if (count != 0 && count % 2 == 0) { //Surrounded by even cubes
				newcube[HIndex(x, y, z)] = 1; //Fill hole
			}
			else {
				newcube[HIndex(x, y, z)] = 0; //remains as hole
			}
		}
		/* Write VBO */
		if (newcube[HIndex(x, y, z)] == 0) {
			for (int i = 0; i < 36; ++i) {
				/* Hole */
				vbo[HIndex(x, y, z) * 36 + i] = make_float3(0, 0, 0);
			}
		}
		else {
			/* Write into VBO for cube at position */
			int idx = HIndex(x, y, z);
			int i = 0;
			float size = 0.5f;
			vbo[idx * 36 + i++] = make_float3(x + size, y + size, z + size);
			vbo[idx * 36 + i++] = make_float3(x - size, y + size, z + size);
			vbo[idx * 36 + i++] = make_float3(x - size, y - size, z + size);
			vbo[idx * 36 + i++] = make_float3(x - size, y - size, z + size);
			vbo[idx * 36 + i++] = make_float3(x + size, y - size, z + size);
			vbo[idx * 36 + i++] = make_float3(x + size, y + size, z + size);

			vbo[idx * 36 + i++] = make_float3(x + size, y + size, z - size);
			vbo[idx * 36 + i++] = make_float3(x + size, y + size, z + size);
			vbo[idx * 36 + i++] = make_float3(x + size, y - size, z + size);
			vbo[idx * 36 + i++] = make_float3(x + size, y - size, z + size);
			vbo[idx * 36 + i++] = make_float3(x + size, y - size, z - size);
			vbo[idx * 36 + i++] = make_float3(x + size, y + size, z - size);

			vbo[idx * 36 + i++] = make_float3(x + size, y + size, z - size);
			vbo[idx * 36 + i++] = make_float3(x - size, y + size, z - size);
			vbo[idx * 36 + i++] = make_float3(x - size, y + size, z + size);
			vbo[idx * 36 + i++] = make_float3(x - size, y + size, z + size);
			vbo[idx * 36 + i++] = make_float3(x + size, y + size, z + size);
			vbo[idx * 36 + i++] = make_float3(x + size, y + size, z - size);

			vbo[idx * 36 + i++] = make_float3(x - size, y + size, z - size);
			vbo[idx * 36 + i++] = make_float3(x + size, y + size, z - size);
			vbo[idx * 36 + i++] = make_float3(x + size, y - size, z - size);
			vbo[idx * 36 + i++] = make_float3(x + size, y - size, z - size);
			vbo[idx * 36 + i++] = make_float3(x - size, y - size, z - size);
			vbo[idx * 36 + i++] = make_float3(x - size, y + size, z - size);

			vbo[idx * 36 + i++] = make_float3(x - size, y + size, z + size);
			vbo[idx * 36 + i++] = make_float3(x - size, y + size, z - size);
			vbo[idx * 36 + i++] = make_float3(x - size, y - size, z - size);
			vbo[idx * 36 + i++] = make_float3(x - size, y - size, z - size);
			vbo[idx * 36 + i++] = make_float3(x - size, y - size, z + size);
			vbo[idx * 36 + i++] = make_float3(x - size, y + size, z + size);

			vbo[idx * 36 + i++] = make_float3(x - size, y - size, z - size);
			vbo[idx * 36 + i++] = make_float3(x + size, y - size, z - size);
			vbo[idx * 36 + i++] = make_float3(x + size, y - size, z + size);
			vbo[idx * 36 + i++] = make_float3(x + size, y - size, z + size);
			vbo[idx * 36 + i++] = make_float3(x - size, y - size, z + size);
			vbo[idx * 36 + i++] = make_float3(x - size, y - size, z - size);
		}
	}
}
//*************************************************************/
//  Simulate one frame of the world
//*************************************************************/
__global__ void simulate(bool *oricube, bool *newcube, int N, int blockSize) {
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int z = threadIdx.z + blockDim.z * blockIdx.z;
	//Did not use shared memory as I found after testing that it was twice as slow as only using global alone
	//Accumulates amount of cubes around each index
	int count = 0;
	//Make sure within boundaries
	if (x >= 0 && y >= 0 && z >= 0 && x < N && y < N && z < N) {
		//For each cube around each index, add 1
		if (x > 0 && oricube[HIndex(x - 1, y, z)] == 1) {
			++count;
		}
		if (y > 0 && oricube[HIndex(x, y - 1, z)] == 1) {
			++count;
		}
		if (z > 0 && oricube[HIndex(x, y, z - 1)] == 1) {
			++count;
		}
		if (x < (N - 1) && oricube[HIndex(x + 1, y, z)] == 1) {
			++count;
		}
		if (y < (N - 1) && oricube[HIndex(x, y + 1, z)] == 1) {
			++count;
		}
		if (z < (N - 1) && oricube[HIndex(x, y, z + 1)] == 1) {
			++count;
		}
		//Simulate world behaviour
		if (oricube[HIndex(x, y, z)] == 1) { //index has cube
			if (count % 2 == 0) { //Surrounded by even cubes
				newcube[HIndex(x, y, z)] = 0; //becomes hole
			}
			else {
				newcube[HIndex(x, y, z)] = 1; //remains as cube
			}
		}
		else { //index has hole
			if (count != 0 && count % 2 == 0) { //Surrounded by even cubes
				newcube[HIndex(x, y, z)] = 1; //Fill hole
			}
			else {
				newcube[HIndex(x, y, z)] = 0; //remains as hole
			}
		}
	}
}
//*************************************************************/
//  Print cube
//*************************************************************/
void PrintCube(bool* cube) {
	for (unsigned x = 0; x < N; ++x) {
		for (unsigned y = 0; y < N; ++y) {
			for (unsigned z = 0; z < N; ++z) {
				if (cube[HIndex(x, y, z)] == 1)
					std::cout << x << " " << y << " " << z << std::endl;
			}
		}
	}
}
//*************************************************************/
//  Do console output
//*************************************************************/
void console_output(unsigned N, unsigned T) {
	if (N > 8)
		blockSize = 8; //Fixed blockSize as 8 for optimal performance
	else
		blockSize = (N / 2) > 0 ? (N / 2) : 1; //If N is less than 8, make sure blockSize is N/2 or at least 1.
	//Array for host side
	bool *hCube;
	hCube = (bool*)malloc(sizeof(bool)* N * N * N);
	for (unsigned z = 0; z < N; ++z) {
		for (unsigned y = 0; y < N; ++y) {
			for (unsigned x = 0; x < N; ++x) {
				hCube[HIndex(x, y, z)] = 1;
			}
		}
	}
	//two arrays for device side
	bool *dCube1;
	bool *dCube2;
	cudaMalloc(&dCube1, sizeof(bool)* N * N * N);
	cudaMalloc(&dCube2, sizeof(bool)* N * N * N);
	dim3 dimBlock(blockSize, blockSize, blockSize);
	dim3 dimGrid(N / dimBlock.x + 1, N / dimBlock.y + 1, N / dimBlock.z + 1);
	//Copies initial state to device
	cudaMemcpy(dCube1, hCube, sizeof(bool)* N * N * N, cudaMemcpyHostToDevice);
	for (unsigned i = 0; i < T; ++i) {
		//Perform simulation ON dCube1 and writes the result TO dCube2 to prevent synchronization problems between blocks
		//We are going to run 2 simulation for each loop
		simulate << <dimGrid, dimBlock >> >(dCube1, dCube2, N, blockSize);
		++i;
		if (i < T) {
			//To avoid doing a memcopy from dCube2 to dCube1 and performing the simulation again, we do the simulation ON dCube2 and writes the result TO dCube1.
			//Hence utilizing both blocks of arrays to speed up. Note that i is incremented twice.
			simulate << <dimGrid, dimBlock >> >(dCube2, dCube1, N, blockSize);
		}
		else {
			//Odd number of Ts. In this case memcopy dCube2's result to dCube1 and exit the loop.
			cudaMemcpy(dCube1, dCube2, sizeof(bool)* N * N * N, cudaMemcpyDeviceToDevice);
		}
	}
	//Result is in dCube1. Copy result back to host array.
	cudaMemcpy(hCube, dCube1, sizeof(bool)*N*N*N, cudaMemcpyDeviceToHost);
	PrintCube(hCube);
	cudaFree(dCube1);
	cudaFree(dCube2);
	free(hCube);
}