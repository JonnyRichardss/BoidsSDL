#pragma once
#ifndef USE_BOIDMANAGER
#define USE_BOIDMANAGER
#include "Boid.h"
#include <vector>

class BoidManager
{
public:
	BoidManager(GameScene* scene);
	~BoidManager();
	std::vector<Boid*> AllBoids;
	void PopulateNeighbours();
};
#endif // !USE_BOIDMANAGER


