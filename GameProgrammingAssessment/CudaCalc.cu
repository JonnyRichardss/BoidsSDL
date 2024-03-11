#include "CudaCalc.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"
#include "Global_Flags.h"
#include "Boid.h"

int NUMBLOCKS = 1024;
int NUMTHREADS = 1024;
//I previously tried converting these to half2 so that I could do vector operations on them but I can't figure out how to convert to half2
//weird results pop out of the builtin conversion functions
struct CudaBoidStruct {
	int id;
	float2 position;
	int neighbours[NUM_BOIDS];
	//int neighbours[1024];
	unsigned int currentIdx = 0;
};
//(as from my previous AMP code) most of this adapted from
//https://github.com/SebLague/Boids/blob/master/Assets/Scripts/BoidCompute.compute
__global__ void GPUNeighbourCalc(CudaBoidStruct* boids, int size, float sqrVisDist) {
	int itest = blockIdx.x;
	int jtest = threadIdx.x;

		//if (boids[itest].id == boids[jtest].id) return;
		float2 offset = make_float2(boids[itest].position.x - boids[jtest].position.x, boids[itest].position.y - boids[jtest].position.y);
		float sqrDist = offset.x * offset.x + offset.y * offset.y;
		if (sqrDist < sqrVisDist) {
			unsigned int index = atomicInc(&boids[itest].currentIdx,NUM_BOIDS);
			boids[itest].neighbours[index] = boids[jtest].id;
		}
}
namespace JRCudaCalc {

	void MakeStructs(CudaBoidStruct* output, std::vector<Boid*>& input) {
		int size = input.size();
		for (int i = 0; i < size; i++) {
			Vector2 position = input[i]->GetPos();
			output[i].id = i;
			output[i].position = make_float2(position.x, position.y);
		}
	}
	void UnMakeStructs(std::vector<Boid*>& output, CudaBoidStruct* input) {
		int size = output.size();
		for (int i = 0; i < size; i++) {
			for (int j = 0; j < input[i].currentIdx; j++) {
				if (i == input[i].neighbours[j])continue;
				output[i]->Neighbours.push_back(output[input[i].neighbours[j]]);
			}
			output[i]->hasNeighbours = true;
		}
	}
	static CudaBoidStruct* gpuBoids;
	static size_t arraySize;
	void GetNeighboursCUDA(std::vector<Boid*>& AllBoids)
	{
		//GameLogging::GetInstance()->DebugLog("OI");
		int size = AllBoids.size();
		CudaBoidStruct* boids = new CudaBoidStruct[size];
		MakeStructs(boids, AllBoids);
		//allocate and copy
		//CudaBoidStruct* gpuBoids;
		
		cudaMemcpy(gpuBoids, boids, arraySize, cudaMemcpyHostToDevice);
		//do calc
		float sqrVisDist = BOID_VISION_DISTANCE * BOID_VISION_DISTANCE;
		float sqrAvoidDist = BOID_AVOID_DISTANCE * BOID_AVOID_DISTANCE;
		GPUNeighbourCalc <<<NUMBLOCKS, NUMTHREADS>>> (gpuBoids, size,sqrVisDist);
		//copy back
		cudaMemcpy(boids, gpuBoids, arraySize, cudaMemcpyDeviceToHost);
		UnMakeStructs(AllBoids, boids);
		//free all
		
		delete[size] boids;
	}
	void Init(int size) {
		arraySize = sizeof(CudaBoidStruct) * size;
		cudaMalloc((void**)&gpuBoids, arraySize);
	}
	void Clear() {
		cudaFree(gpuBoids);
	}

}

