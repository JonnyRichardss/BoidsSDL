#pragma once
#ifndef USE_BOID
#define USE_BOID
#include "GameObject.h"
class BoidManager; //fwd def
struct BoidInfo {
	int id;
	//Vector2 position;
	float posX;
	float posY;
	//Vector2 velocity;
	float velX;
	float velY;
	//Vector2 SepVec;
	float sepX=0;
	float sepY=0;
	//Vector2 AligVec;
	float aligX=0;
	float aligY=0;
	//Vector2 CohesVec;
	float cohesX=0;
	float cohesY=0;
	int numNeighbours=0;
};
class Boid : public GameObject
{
public:
	void Init();
	void InitVisuals();
	void SetManager(BoidManager* newManager);
	void MakeBrian();
	void DrawBrianDebug();
	void ParseStruct(BoidInfo info);
	Vector2 steerTarget;
protected:
	
	std::vector<Boid*> Neighbours;
	int numNeighbours = 0;
	BoidManager* manager;
	bool hasNeighbours = false;
	bool isBrian = false;
	void Update();
	void DoRotation();
	void ScreenWrap();
	void CPUCalc();
	void SteerTowards(Vector2 target);
	std::vector<Boid*> GetVisibleBoids();
	Vector2 GetBoidVec(Boid* other);
	void DoSeparation(Vector2 vec);
	void DoAlignment(Vector2 vec);
	void DoCohesion(Vector2 vec);
	
};
#endif // !USE_BOID


