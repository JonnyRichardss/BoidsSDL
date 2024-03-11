#ifndef USE_GAMEGLOBALS
#define USE_GAMEGLOBALS
//SOME OF THESE CAN PROBABLY BE MACROS SO THEY COMPILE OUT THE CODE THEY POINT TO BUT IM NOT ENTIRELY SURE THATS REALLY NECESSARY
static constexpr int FRAME_CAP = 60;

static constexpr int WINDOW_WIDTH = 640;
static constexpr int WINDOW_HEIGHT = 480;
static constexpr int GAME_MAX_X = 800;
static constexpr int GAME_MAX_Y = 600;

static const char* LOG_FOLDER_PATH = "Logs/";
static const char* LOGFILE_NAME = "latest.log";

static const char* BASE_ASSET_PATH = "Assets/";
static const char* SPRITE_INFO_FORMAT = ".spritedims";

static constexpr bool DEBUG_DRAW_BB = false;
static constexpr bool DEBUG_EXTRA_LOGGING = true; //designed for when using print debugging so I can (possibly) leave the logs in

static constexpr bool CONSOLE_LOG_DEFAULT = true;
static constexpr bool DO_FILE_LOGGING = true;
static constexpr bool DO_BATCH_LOGGING = true;
static constexpr bool VERBOSE_CONSOLE = false;

static constexpr bool DO_PROFILING = true;

static bool ENGINE_QUIT_FLAG = false; //be aware this doesn't work the way you think it does

static constexpr float BOID_SIZE = 10.0f;
static constexpr float BOID_SPEED = 4.0f;
static constexpr int NUM_BOIDS = 1024;

static constexpr float BOID_VISION_DISTANCE = 50.0f;
static constexpr float BOID_AVOID_DISTANCE = 40.0f;// will try change later
static constexpr float BOID_VISION_ANGLE = 3.141592653589;
static constexpr float BOID_STEER_MULTIPLIER = 0.05f;
static constexpr float BOID_SEPARATION_STRENGTH = 1.0f;
static constexpr float BOID_ALIGNMENT_STRENGTH = 1.0f;
static constexpr float BOID_COHESION_STRENGTH =1.0f;

static constexpr bool GPU_CALC = true;
static constexpr bool BRIAN_DEBUG = false;


enum WaitMethods {BUSY,SDL,THREAD};
static constexpr WaitMethods GF_WAIT_METHOD = BUSY; //SDL seems to under-sleep and THREAD seems to over-sleep
#endif // !USE_GAMEGLOBALS
