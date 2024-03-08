#include "BoidManager.h"
#include "GameMath.h"
#define _SILENCE_AMP_DEPRECATION_WARNINGS
//#include <amp.h>
#include "GameClock.h"
#include "CudaCalc.cuh"
BoidManager::BoidManager(GameScene* scene)
{
	for (int i = 0; i < NUM_BOIDS; i++) {
		AllBoids.push_back(new Boid());
		AllBoids[i]->SetOwner(scene);
		AllBoids[i]->SetManager(this);
	}

}

BoidManager::~BoidManager()
{
}

void BoidManager::PopulateNeighbours()
{
	//most of this adapted from
	//https://github.com/SebLague/Boids/blob/master/Assets/Scripts/BoidCompute.compute
	//int size = AllBoids.size();
	//BoidInfo* boidsInfos = new BoidInfo[size];
	//MakeStructs(boidsInfos);


	//THIS IS ALL CPPAMP CODE
	/*
	
	concurrency::array_view<BoidInfo, 1>boids(size, boidsInfos);
	//concurrency::array_view<BoidInfo, 1>boidsOut(size, boidsInfos);
	//boidsOut.discard_data();
	float visDist = BOID_VISION_DISTANCE;
	float avoidDist = BOID_AVOID_DISTANCE;
	concurrency::parallel_for_each(
		boids.extent,
		[=](concurrency::index<1> i) restrict(amp) {
			for (int j = 0; j < size; j++) {
				if (boids[i].id == j)
					continue;
				float offsetX = boids[i].posX - boids[j].posX;
				float offsetY = boids[i].posY - boids[j].posY;
				//Vector2 offset = boids[i].position - boids[j].position;
				float sqrDistance = offsetX * offsetX + offsetY * offsetY; // sqrt(a^2 + b^2) skipping sqrt step
				if (sqrDistance < visDist * visDist) {
					boids[i].numNeighbours++;
					boids[i].aligX -= boids[j].velX;
					boids[i].aligY -= boids[j].velY;
					boids[i].cohesX += boids[j].posX;
					boids[i].cohesY += boids[j].posY;
				}
				if (sqrDistance < avoidDist * avoidDist) {
					//float invSqr = 1.0 / sqrDistance;
					float invSqr = 1;
					boids[i].sepX -= offsetX * invSqr;
					boids[i].sepY -= offsetY * invSqr;
				}
			}
		}
		
	);
	


	for (int i = 0; i < size;i++) {
		Boid* b = AllBoids[i];
		b->steerTarget = Vector2::zero();
		b->ParseStruct(boids[i]);
	}
	*/


	//GameClock::GetInstance()->TickProfilingSpecial("GPU DONE");
	//delete[size] boidsInfos;


	JRCudaCalc::DoCalc(AllBoids);
}

void BoidManager::MakeStructs(BoidInfo* boids) {
	for (int i = 0; i < AllBoids.size();i++) {
		Boid* b = AllBoids[i];
		Vector2 position = b->GetPos();
		Vector2 velocity = b->GetVelo();
		boids[i].id = i;
		boids[i].posX = position.x;
		boids[i].posY = position.y;
		boids[i].velX = velocity.x;
		boids[i].velY = velocity.y;
	}
}

