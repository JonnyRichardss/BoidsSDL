#include "GameEngine.h"
//#include "IncludeGameObjects.h"
#include "IncludeScenes.h"
#include "GameRNG.h"
static GameEngine* _instance;
GameEngine::GameEngine()
{
    logging = GameLogging::GetInstance();
    if (SDL_Init(SDL_INIT_EVERYTHING) != 0) {
        logging->Log("Failed to initialise SDL: exiting.");
        ENGINE_QUIT_FLAG = true;
    }
    clock = GameClock::GetInstance();
    renderer = RenderEngine::GetInstance();
    logging->Log("Initialised game engine.");
    RNG::Seed();
}

GameEngine::~GameEngine()
{
}

GameEngine* GameEngine::GetInstance()
{
    if (_instance == nullptr)
        _instance = new GameEngine();
    return _instance;
}

void GameEngine::StartLoop()
{
    FPS = new FPSCounter();
    FPS->Init();
    CreateScenes();
    GameLoop();
}

 void GameEngine::CreateScenes()
{
     //push back scenes
     GameScene* scene0 = new BoidScene();
     AllScenes.push_back(scene0);
     for (auto Scene : AllScenes) {
         Scene->CreateObjects();
         logging->FileLog("Created object queues for all scenes.");
     }
     if (AllScenes.size() == 0) {
         logging->Log("Scene list empty!! Exiting!");
         ENGINE_QUIT_FLAG = true;
         return;
     }
     ActiveScene = AllScenes[0];
}

void GameEngine::SwitchScene(int SceneIndex)
{
    if (SceneIndex < 0 || SceneIndex >= AllScenes.size()) {
        logging->Log("Failed to switch to scene with index: " + std::to_string(SceneIndex));
        return;
    }
    ActiveScene = AllScenes[SceneIndex];
    logging->Log("Switched to scene with index: " + std::to_string(SceneIndex));
}

void GameEngine::ProcessEvents()
{
    SDL_Event event;
    while (SDL_PollEvent(&event) != 0) {
        if (event.type == SDL_KEYDOWN) {
            switch (event.key.keysym.sym) {
            case SDLK_F11:
                renderer->ToggleFullscreen();
                break;
            case SDLK_ESCAPE:
                ENGINE_QUIT_FLAG = true;
                logging->Log("Recieved ESC input");
                break;
            case SDLK_F8:
                FPS->ToggleVisibility();
                break;
            case SDLK_SPACE:
                //dothings
                break;
            }
        }
        if (event.type == SDL_QUIT) {
            logging->Log("Recieved SDL Quit");
            ENGINE_QUIT_FLAG = true;
        }
    }
}

void GameEngine::FPSUpdate() {
    RenderableComponent* newComponent = nullptr;
    if (FPS->UpdateAndRender(newComponent)) {
        renderer->Enqueue(newComponent);
    }
}

void GameEngine::GameLoop() {
    while (!ENGINE_QUIT_FLAG) {
        clock->TickProfiling(STARTPHASE);
        ProcessEvents();
        clock->TickProfiling(INPUTPHASE);
        ActiveScene->Update();
        FPSUpdate();
        clock->TickProfiling(UPDATEPHASE);
        renderer->RenderFrame();
        if (DEBUG_DRAW_BB)
           ActiveScene->DrawBBs();
        clock->TickProfiling(RENDERPHASE);
        clock->Tick();
        if (DO_BATCH_LOGGING)
            logging->SaveLogFile();
    }
    logging->Log("Exiting Program");
    SDL_Quit();
}


