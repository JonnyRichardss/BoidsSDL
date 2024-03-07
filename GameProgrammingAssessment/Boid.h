#pragma once
#ifndef USE_BOID
#define USE_BOID
#include "GameObject.h"
class Boid : public GameObject
{
public:
	void Init();
	void InitVisuals();
protected:
	void Update();
	float heading;
	void DoRotation();
};
#endif // !USE_BOID


