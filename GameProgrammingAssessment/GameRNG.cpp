#include "GameRNG.h"
#include <cstdlib>
#include <ctime>
static RNG* instance;
int RNG::randi(int min, int max)
{
	return (int)randf(min, max);
}

float RNG::randf(float min, float max)
{
	float diff = max - min;
	float normalisedRand = (float)rand() / (float)RAND_MAX;
	return normalisedRand * diff + min;
}
void RNG::Seed() {
	srand((unsigned int)time(NULL));
}
RNG::RNG() 
{
	
}
RNG::~RNG()
{
}
