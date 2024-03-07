#pragma once
#ifndef USE_RENDERABLE
#define USE_RENDERABLE
#include "SDL.h"
#include "GameLogging.h"
struct Vector2; //fwd def - i dont want to include math here necessarily
class RenderableComponent
{
public:
	RenderableComponent();
	RenderableComponent(SDL_Texture* _texture, SDL_Rect* _destination_pos);
	RenderableComponent(SDL_Texture* _texture, SDL_Rect* _destination_pos, SDL_Rect* _source_pos);
	RenderableComponent(SDL_Texture* _texture, SDL_Rect* _destination_pos, int _layer);
	RenderableComponent(SDL_Texture* _texture, SDL_Rect* _destination_pos, SDL_Rect* _source_pos,int _layer);
	virtual ~RenderableComponent();
	SDL_Rect* GetSourcePos();
	SDL_Rect* GetDestPos();
	SDL_Texture* GetTexture();
	SDL_Point* GetCentrePoint();
	SDL_RendererFlip GetFlip();
	int GetLayer();
	double GetAngle();
	
	void UpdateLayer(int newLayer);
	void UpdateSourcePos(SDL_Rect* newPos);
	void UpdateDestPos(SDL_Rect* newPos);
	void UpdateTexture(SDL_Texture* newTex);
	void UpdateCentrePoint(Vector2 newPoint);
	void UpdateAngleRAD(double newAngle);
	void UpdateAngleDEG(double newAngle);
	void UpdateFlip(SDL_RendererFlip newFlip);
	bool operator > (const RenderableComponent& other);
	bool operator < (const RenderableComponent& other);
protected:
	int layer;
	double angle;
	SDL_Point* centrePoint; // bear in mind this is in local coordinates and is relative to the size of dest rect
	
	SDL_Rect* source_pos;
	SDL_Texture* texture;
	SDL_Rect* destination_pos;
	GameLogging* logging;

	SDL_RendererFlip flip; //added for completeness - i wont use it for now
};
#endif
