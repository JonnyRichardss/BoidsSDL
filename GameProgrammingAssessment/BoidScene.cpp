#include "BoidScene.h"
#include "IncludeGameObjects.h"
BoidScene::BoidScene()
{
	name = "Boids Scene";
}

BoidScene::~BoidScene()
{
}

void BoidScene::CreateObjects()
{
	//addobjs
	Boid* testBoid = new Boid();
	UpdateQueue.push_back(testBoid);
}
