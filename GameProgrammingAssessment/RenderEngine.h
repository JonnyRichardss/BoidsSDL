
#ifndef USE_RENDER
#define USE_RENDER
#include"SDL.h"
#include "GameClock.h"
#include "SDL_ttf.h"
#include "RenderableComponent.h"
#include <vector>
#include "GameLogging.h"
class Boid;
class RenderEngine
{
public:
	static RenderEngine* GetInstance();
	
	SDL_Window* GetWindow();
	SDL_Renderer* GetRenderContext();
	void RenderFrame();
	void ToggleFullscreen();

	void Enqueue(RenderableComponent* object);
	void SetBrian(Boid* _brian);
private:
	RenderEngine();
	~RenderEngine();
	std::vector<RenderableComponent*> RenderQueue;
	GameLogging* logging;
	GameClock* clock;
	SDL_Window* window;
	SDL_Renderer* renderContext;
	SDL_DisplayMode mode;

};
#endif // !USE_RENDER
