#include "BoidScene.h"
#include "IncludeGameObjects.h"
#include "BoidManager.h"
BoidScene::BoidScene()
{
	name = "Boids Scene";
}

BoidScene::~BoidScene()
{
}
static BoidManager* manager;
void BoidScene::CreateObjects()
{
	manager = new BoidManager(this);

	//addobjs
	for(Boid* b : manager->AllBoids)
		UpdateQueue.push_back(b);
}
