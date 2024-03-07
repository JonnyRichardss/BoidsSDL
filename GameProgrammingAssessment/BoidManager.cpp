#include "BoidManager.h"
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

