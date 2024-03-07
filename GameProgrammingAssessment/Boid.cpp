#include "Boid.h"
#include <string>
#include "SDL.h"
#include "SDL_image.h"
#include "GameRNG.h"
void Boid::Init()
{
	BoundingBox = Vector2(5, 10);
	shown = true;
	heading = RNG::randf(0, 2 * M_PI);
	velocity = Vector2::zero();
	DoRotation();
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
	visuals->UpdateTexture(Tex);

	SDL_Rect DefaultRect = BBtoDestRect();
	visuals->UpdateDestPos(&DefaultRect);

}

void Boid::Update()
{
	heading+=0.25f;
	DoRotation();
}

void Boid::DoRotation()
{
	velocity = Vector2(sin(heading) * BOID_SPEED, cos(heading)*BOID_SPEED);
}
