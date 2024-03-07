#pragma once
#ifndef USE_BOID
#define USE_BOID
#include "GameObject.h"
class BoidManager; //fwd def
class Boid : public GameObject
{
public:
	void Init();
	void InitVisuals();
	void SetManager(BoidManager* newManager);
protected:
	std::vector<Boid*> Neighbours;
	BoidManager* manager;
	void Update();
	void DoRotation();
	void ScreenWrap();
	void SteerTowards(Vector2 target);
	std::vector<Boid*> GetVisibleBoids();
	Vector2 GetBoidVec(Boid* other);
	void DoSeparation(Vector2& target);
	void DoAlignment(Vector2& target);
	void DoCohesion(Vector2& target);
};
#endif // !USE_BOID


