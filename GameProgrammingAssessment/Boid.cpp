#include "Boid.h"
#include <string>
#include "SDL.h"
#include "SDL_image.h"
#include "GameRNG.h"
#include "BoidScene.h"
#include "BoidManager.h"
#include <math.h>
void Boid::Init()
{
	BoundingBox = Vector2(BOID_SIZE, BOID_SIZE * 2);
	shown = true;
	facing = RNG::randf(0, 2 * M_PI);
	velocity = Vector2::zero();
	position = Vector2(RNG::randi(0, GAME_MAX_X * 2) - GAME_MAX_X, RNG::randi(0, GAME_MAX_Y * 2) - GAME_MAX_Y); //maybe ill eventually fix my random functions to behave with -ve values
	DoRotation();
	steerTarget = Vector2::zero();
	//dont know its necessary here but TODO a way for a gameobject to access its parent scene
	//maybe just a call to engine to register OR access currentScene through engine
}

void Boid::InitVisuals()
{
	//this is hardcoded for now - I need a generic way to load textures (probably in RenderableComponent) but i only added it to spritesheet so ill just use this for now
	std::string filename, fileformat;
	filename = "boid";
	fileformat = ".png";
	std::string basePath = std::string(BASE_ASSET_PATH) + filename;
	std::string imgPath = basePath + fileformat;
	std::string dimPath = basePath + SPRITE_INFO_FORMAT;
	SDL_Surface* Surf = IMG_Load(imgPath.c_str());
	if (Surf == nullptr) {
		logging->Log(SDL_GetError());
		return;
	}
	SDL_Texture* Tex = SDL_CreateTextureFromSurface(RenderEngine::GetInstance()->GetRenderContext(), Surf);
	SDL_FreeSurface(Surf);
	SDL_SetTextureColorMod(Tex,RNG::randi(0,0),RNG::randi(0,128),RNG::randi(128,255));
	if (isBrian) {
		SDL_SetTextureColorMod(Tex, 255,0,0);
	}
	visuals->UpdateTexture(Tex);

	SDL_Rect DefaultRect = BBtoDestRect();
	visuals->UpdateDestPos(&DefaultRect);

}

void Boid::SetManager(BoidManager* newManager)
{
	manager = newManager;
}

void Boid::MakeBrian()
{
	isBrian = true;

}

void Boid::Update()
{
	
	if (GPU_CALC) {
		if (!hasNeighbours) {
			manager->PopulateNeighbours();
		}
		hasNeighbours = false;
	}
	else {
		Neighbours = GetVisibleBoids();
		
	}
	numNeighbours = Neighbours.size();
	Calc();
	SteerTowards(steerTarget);
	ScreenWrap();
	DoRotation();
	Neighbours.clear();
}

void Boid::DoRotation()
{
	velocity = Vector2(sin(facing) * BOID_SPEED, cos(facing)*BOID_SPEED);
	//facing = Vector2::AngleBetweenRAD(Vector2::up(), velocity);
}

void Boid::ScreenWrap()
{
	if (position.x > GAME_MAX_X) {
		position.x = -GAME_MAX_X;
	}
	else if (position.x < -GAME_MAX_X) {
		position.x = GAME_MAX_X;
	}
	if (position.y > GAME_MAX_Y) {
		position.y = -GAME_MAX_Y;
	}
	else if (position.y < -GAME_MAX_Y) {
		position.y = GAME_MAX_Y;
	}
}

void Boid::Calc()
{
	steerTarget = Vector2::zero();
	Vector2 aligVec = Vector2::zero();
	Vector2 sepVec = Vector2::zero();
	Vector2 cohesVec = Vector2::zero();
	for (Boid* b : Neighbours) {
		
		aligVec -= b->GetVelo();
		cohesVec += b->GetPos();
		sepVec -= GetBoidVec(b);
	}
	DoSeparation(sepVec);
	DoAlignment(aligVec);
	DoCohesion(cohesVec);

}

void Boid::SteerTowards(Vector2 target)
{
	facing += Vector2::AngleBetweenRAD(velocity, target) * BOID_STEER_MULTIPLIER;
	//actually steer in velocity thanks


}

std::vector<Boid*> Boid::GetVisibleBoids()
{
	std::vector<Boid*> output;
	for (Boid* b : manager->AllBoids) {
		Vector2 vec = GetBoidVec(b);
		if (b == this)
			continue;
		if (vec.GetMagnitude() < BOID_VISION_DISTANCE) {
			output.push_back(b);
		}
	}
	//numNeighbours = output.size();
	return output;
}

Vector2 Boid::GetBoidVec(Boid* other)
{
	return position - other->position;
}
Vector2 Boid::ConvertToTarget(Vector2 vector) {
	Vector2 v = vector.Normalise() * BOID_SPEED - velocity;
	return v.Normalise() * std::min(v.GetMagnitude(), BOID_STEER_MULTIPLIER);
}
void Boid::DoSeparation(Vector2 vec)
{
	/*
	if (Neighbours.empty())
		return;
	Vector2 OppositeHeading = Vector2::zero();
	for (Boid* b : Neighbours) {
		OppositeHeading -= GetBoidVec(b);
	}
	OppositeHeading *= 1.0f / (float)Neighbours.size();
	target += OppositeHeading * BOID_SEPARATION_STRENGTH;
	*/
	if (numNeighbours == 0)
		return;
	vec *= (1.0 / numNeighbours);
	vec *= BOID_SEPARATION_STRENGTH;
	steerTarget += vec;
}

void Boid::DoAlignment(Vector2 vec)
{
	/*
	if (Neighbours.empty())
		return;
	Vector2 LocalAverageHeading = Vector2::zero();
	for (Boid* b : Neighbours) {
		LocalAverageHeading -= b->velocity;
	}
	LocalAverageHeading = LocalAverageHeading.Normalise();
	target += LocalAverageHeading * BOID_ALIGNMENT_STRENGTH;
	*/
	if (numNeighbours == 0)
		return;
	vec = vec.Normalise();
	vec *= BOID_ALIGNMENT_STRENGTH;
	steerTarget += vec;
}

void Boid::DoCohesion(Vector2 vec)
{
	/*
	if (Neighbours.empty())
		return;
	Vector2 LocalCentre = Vector2::zero();
	for (Boid* b : Neighbours) {
		LocalCentre += b->position;
	}
	LocalCentre *=  1.0f / (float)Neighbours.size();

	target += (position - LocalCentre) * BOID_COHESION_STRENGTH;
	*/
	if (numNeighbours == 0)
		return;
	vec *= (1.0 / numNeighbours);
	vec = position - vec;
	vec *= BOID_COHESION_STRENGTH;
	steerTarget += vec;
}

void Boid::DrawBrianDebug()
{
	float minAngle = facing - BOID_VISION_ANGLE;
	float maxAngle = facing + BOID_VISION_ANGLE;
	float angleRange = maxAngle - minAngle;
	float currentAngle = minAngle;
	int maxi = 20;
	SDL_SetRenderDrawColor(renderContext, 255, 255, 255, 255);
	for (int i = 0; i < maxi; i++) {
		Vector2 windowPos = GameToWindowCoords(position);
		Vector2 circlePos = GameToWindowCoords(Vector2((cos(currentAngle) * BOID_VISION_DISTANCE + position.x), (sin(currentAngle) * BOID_VISION_DISTANCE + position.y)));
		SDL_RenderDrawLine(renderContext,windowPos.x,windowPos.y,circlePos.x,circlePos.y);
		currentAngle += angleRange / maxi;
	}
}


