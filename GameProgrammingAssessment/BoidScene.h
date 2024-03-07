#ifndef USE_BOIDSCENE
#define USE_BOIDSCENE
#include "GameScene.h"
class BoidScene : public GameScene
{
public:
	BoidScene();
	~BoidScene();
	void CreateObjects();
};

#endif // !USE_BOIDSCENE

