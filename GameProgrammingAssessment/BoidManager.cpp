#include "BoidManager.h"
#include "GameMath.h"
#include "GameClock.h"
#include "CudaCalc.cuh"
BoidManager::BoidManager(GameScene* scene)
{
	for (int i = 0; i < NUM_BOIDS-1; i++) { 
		AllBoids.push_back(new Boid());
		AllBoids[i]->SetOwner(scene);
		AllBoids[i]->SetManager(this);
	}
	JRCudaCalc::Init(AllBoids.size() + 1);
}

BoidManager::~BoidManager()
{
	JRCudaCalc::Clear();
}

void BoidManager::PopulateNeighbours()
{
	//GameClock::GetInstance()->TickProfilingSpecial("GPUSTART");
	JRCudaCalc::GetNeighboursCUDA(AllBoids);
	//GameClock::GetInstance()->TickProfilingSpecial("GPUDONE");
}



