#include "CudaCalc.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Global_Flags.h"
#include "Boid.h"
int NUMBLOCKS = 1024;
int NUMTHREADS = 1024;
struct CudaBoidStruct {
	int id;
	float2 position;
	float2 velocity;
	float2 sepOutput;
	float2 aligOutput;
	float2 cohesOutput;
	int numNeighbours;
};
//(as from my previous AMP code) most of this adapted from
//https://github.com/SebLague/Boids/blob/master/Assets/Scripts/BoidCompute.compute
__global__ void GPUDoCalc(CudaBoidStruct* boids, int size, float sqrVisDist, float sqrAvoidDist) {
	int itest = blockIdx.x * blockDim.x + threadIdx.x;

	//for (int i = 0; i < size; i++) {
	for (int jtest = 0; jtest < size; jtest++) {
			//if (i == j) continue;
		if (itest == jtest) continue;
		float2 offset = make_float2(boids[itest].position.x - boids[jtest].position.x, boids[itest].position.y - boids[jtest].position.y);
		float sqrDist = offset.x * offset.x + offset.y * offset.y;
		if (sqrDist < sqrVisDist) {
			boids[itest].numNeighbours++;
			boids[itest].aligOutput.x += boids[jtest].velocity.x;
			boids[itest].aligOutput.y += boids[jtest].velocity.y;
			boids[itest].cohesOutput.x += boids[jtest].position.x;
			boids[itest].cohesOutput.y += boids[jtest].position.y;
		}
		if (sqrDist < sqrAvoidDist) {
			boids[itest].sepOutput.x -= offset.x;
			boids[itest].sepOutput.y -= offset.y;
		}
		}
	//}
}
namespace JRCudaCalc {

	void MakeStructs(CudaBoidStruct* output, std::vector<Boid*>& input) {
		int size = input.size();
		for (int i = 0; i < size; i++) {
			Vector2 position = input[i]->GetPos();
			Vector2 velocity = input[i]->GetVelo();
			output[i].id = i;
			output[i].position = make_float2(position.x, position.y);
			output[i].velocity = make_float2(velocity.x, velocity.y);
			output[i].sepOutput = make_float2(0, 0);
			output[i].aligOutput = make_float2(0, 0);
			output[i].cohesOutput = make_float2(0, 0);
			output[i].numNeighbours = 0;
		}
	}
	void UnMakeStructs(std::vector<Boid*>& output, CudaBoidStruct* input) {
		int size = output.size();
		for (int i = 0; i < size; i++) {
			output[i]->steerTarget = Vector2::zero();
			output[i]->numNeighbours = input[i].numNeighbours;
			output[i]->DoSeparation(Vector2(input[i].sepOutput.x, input[i].sepOutput.y));
			output[i]->DoAlignment(Vector2(input[i].aligOutput.x, input[i].aligOutput.y));
			output[i]->DoCohesion(Vector2(input[i].cohesOutput.x, input[i].cohesOutput.y));
			output[i]->hasNeighbours = true;
		}
	}
	void DoCalc(std::vector<Boid*>& AllBoids)
	{
		GameLogging::GetInstance()->DebugLog("OI");
		int size = AllBoids.size();
		CudaBoidStruct* boids = new CudaBoidStruct[size];
		MakeStructs(boids, AllBoids);
		//allocate and copy
		CudaBoidStruct* gpuBoids;
		size_t arraySize = sizeof(CudaBoidStruct) * size;
		cudaMalloc((void**)&gpuBoids, arraySize);
		cudaMemcpy(gpuBoids, boids, arraySize, cudaMemcpyHostToDevice);
		//do calc
		float sqrVisDist = BOID_VISION_DISTANCE * BOID_VISION_DISTANCE;
		float sqrAvoidDist = BOID_AVOID_DISTANCE * BOID_AVOID_DISTANCE;
		GPUDoCalc <<<NUMBLOCKS, NUMTHREADS>>> (gpuBoids, size,sqrVisDist , sqrAvoidDist);
		//copy back
		cudaMemcpy(boids, gpuBoids, arraySize, cudaMemcpyDeviceToHost);
		UnMakeStructs(AllBoids, boids);
		//free all
		cudaFree(gpuBoids);
		delete[size] boids;
	}
}

