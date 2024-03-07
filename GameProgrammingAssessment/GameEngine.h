#ifndef USE_ENGINE
#define USE_ENGINE
#include "SDL.h"
#include "GameClock.h"
#include "RenderEngine.h"
#include "GameObject.h"
#include "FPSCounter.h"
#include "Global_Flags.h"
#include "GameLogging.h"
#include "GameScene.h"

#include <vector>
class GameEngine
{
public:
	static GameEngine* GetInstance();
	void StartLoop();
	
private:
	GameEngine();
	~GameEngine();
	GameLogging* logging;
	RenderEngine* renderer;
	GameClock* clock;
	FPSCounter* FPS;
	GameScene* ActiveScene;
	std::vector<GameScene*> AllScenes;
	void CreateScenes();
	void SwitchScene(int SceneIndex);
	void ProcessEvents();

	void FPSUpdate();


	void GameLoop();

};
#endif // !USE_ENGINE
