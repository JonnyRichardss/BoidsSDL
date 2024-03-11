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
	void MakeBrian();
	void DrawBrianDebug();

	
	std::vector<Boid*> Neighbours;
	bool hasNeighbours = false;
protected:
	Vector2 steerTarget;
	void DoSeparation(Vector2 vec);
	void DoAlignment(Vector2 vec);
	void DoCohesion(Vector2 vec);
	int numNeighbours = 0;
	BoidManager* manager;
	bool isBrian = false;
	void Update();
	void DoRotation();
	void ScreenWrap();
	void Calc();
	void SteerTowards(Vector2 target);
	std::vector<Boid*> GetVisibleBoids();
	Vector2 GetBoidVec(Boid* other);

	Vector2 ConvertToTarget(Vector2 vector);
	
	
};
#endif // !USE_BOID


