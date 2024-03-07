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
static Boid* brian;
void BoidScene::CreateObjects()
{
	manager = new BoidManager(this);
	brian = new Boid();
	brian->MakeBrian();
	brian->SetManager(manager);
	brian->SetOwner(this);
	manager->AllBoids.push_back(brian);
	//addobjs
	for(Boid* b : manager->AllBoids)
		UpdateQueue.push_back(b);
	RenderEngine::GetInstance()->SetBrian(brian);
	//UpdateQueue.push_back(brian);
	
}
