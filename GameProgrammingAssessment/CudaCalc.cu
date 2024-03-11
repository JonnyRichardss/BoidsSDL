#include "CudaCalc.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"
#include "Global_Flags.h"
#include "Boid.h"

int NUMBLOCKS = NUM_BOIDS;
int NUMTHREADS = 1024;
//I previously tried converting these to half2 so that I could do vector operations on them but I can't figure out how to convert to half2
//weird results pop out of the builtin conversion functions
struct CudaNeighbourStruct {
	int id;
	float2 position;
	int neighbours[NUM_BOIDS / NEIGHBOUR_STORAGE_DIV]; //neighbours is defined to overwrite itself if over 1/8 of the boids are nearby
	//int neighbours[1024];
	unsigned int currentIdx = 0;
};
struct CudaBoidStruct {
	int neighbours[NUM_BOIDS / NEIGHBOUR_STORAGE_DIV];
	int id;
	float2 position;
	float2 velocity;
	float2 aligVec;
	float2 sepVec;
	float2 cohesVec;
};
//(as from my previous AMP code) most of this adapted from
//https://github.com/SebLague/Boids/blob/master/Assets/Scripts/BoidCompute.compute
//not anymore but *some* basics of the way it's calculated on the CPU are
__global__ void GPUNeighbourCalc(CudaNeighbourStruct* boids, int size, float sqrVisDist) {
	int itest = blockIdx.x;
	
	if (itest > size) return;//think this is never true but oh well
	for (int jtest = threadIdx.x; jtest < size; jtest += blockDim.x) {
		float2 offset = make_float2(boids[itest].position.x - boids[jtest].position.x, boids[itest].position.y - boids[jtest].position.y);
		float sqrDist = offset.x * offset.x + offset.y * offset.y;
		if (sqrDist < sqrVisDist) {
			unsigned int index = atomicInc(&boids[itest].currentIdx, (NUM_BOIDS / NEIGHBOUR_STORAGE_DIV));
			boids[itest].neighbours[index] = boids[jtest].id;
		}
	}
}
/*
* previously I tried calculating the vectors on the GPU but i couldn't actually figure it out - the calculations were just wrong (eg cohesion sent all of them to the screen center)
__global__ void GPUVecCalc(CudaBoidStruct* boids,int size) {
	int itest = blockIdx.x;
	int jtest = threadIdx.x;
	if (itest > size) return;
	float2 offset = make_float2(boids[itest].position.x - boids[jtest].position.x, boids[itest].position.y - boids[jtest].position.y);
	atomicAdd(&boids[itest].aligVec.x, boids[jtest].velocity.x);
	atomicAdd(&boids[itest].aligVec.y, boids[jtest].velocity.y);
	atomicAdd(&boids[itest].cohesVec.x, boids[jtest].position.x);
	atomicAdd(&boids[itest].cohesVec.y, boids[jtest].position.y);
	atomicAdd(&boids[itest].sepVec.x, offset.x);
	atomicAdd(&boids[itest].sepVec.y,offset.y);
}
*/
namespace JRCudaCalc {

	void MakeNeighbourStructs(CudaNeighbourStruct* output, std::vector<Boid*>& input) {
		int size = input.size();
		for (int i = 0; i < size; i++) {
			Vector2 position = input[i]->GetPos();
			output[i].id = i;
			output[i].position = make_float2(position.x, position.y);
		}
	}
	void UnMakeNeighbourStructs(std::vector<Boid*>& output, CudaNeighbourStruct* input) {
		int size = output.size();
		for (int i = 0; i < size; i++) {
			int idx = static_cast<int>(input[i].currentIdx);
			for (int j = 0; j < std::min(idx,NUM_BOIDS/NEIGHBOUR_STORAGE_DIV); j++) {
				if (i == input[i].neighbours[j])continue;
				output[i]->Neighbours.push_back(output[input[i].neighbours[j]]);
			}
			output[i]->hasNeighbours = true;
		}
	}
	static CudaNeighbourStruct* gpuBoids;
	static size_t arraySize;
	void GetNeighboursCUDA(std::vector<Boid*>& AllBoids)
	{
		//GameLogging::GetInstance()->DebugLog("OI");
		int size = AllBoids.size();
		CudaNeighbourStruct* boids = new CudaNeighbourStruct[size];
		MakeNeighbourStructs(boids, AllBoids);
		//allocate and copy
		//CudaBoidStruct* gpuBoids;
		
		cudaMemcpy(gpuBoids, boids, arraySize, cudaMemcpyHostToDevice);
		//do calc
		float sqrVisDist = BOID_VISION_DISTANCE * BOID_VISION_DISTANCE;
		float sqrAvoidDist = BOID_AVOID_DISTANCE * BOID_AVOID_DISTANCE;
		GPUNeighbourCalc <<<NUMBLOCKS, NUMTHREADS>>> (gpuBoids, size,sqrVisDist);
		//copy back
		cudaMemcpy(boids, gpuBoids, arraySize, cudaMemcpyDeviceToHost);
		UnMakeNeighbourStructs(AllBoids, boids);
		//free all
		
		delete[] boids;
	}
	void Init(int size) {
		arraySize = sizeof(CudaNeighbourStruct) * size;
		cudaMalloc((void**)&gpuBoids, arraySize);
	}
	void Clear() {
		cudaFree(gpuBoids);
	}

}

