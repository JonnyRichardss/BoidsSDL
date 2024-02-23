#ifndef USE_GAMEGLOBALS
#define USE_GAMEGLOBALS

static const int FRAME_CAP = 60;

static const int WINDOW_WIDTH = 640;
static const int WINDOW_HEIGHT = 480;
static const int GAME_MAX_X = 400;
static const int GAME_MAX_Y = 300;

static const char* LOG_FOLDER_PATH = "Logs/";
static const char* LOGFILE_NAME = "latest.log";

static const char* BASE_ASSET_PATH = "Assets/";
static const char* SPRITE_INFO_FORMAT = ".spritedims";

static const bool DEBUG_DRAW_BB = false;

static const bool CONSOLE_LOG_DEFAULT = true;
static const bool DO_FILE_LOGGING = true;
static const bool VERBOSE_CONSOLE = false;
enum WaitMethods {BUSY,SDL,THREAD};
static const WaitMethods GF_WAIT_METHOD = BUSY; //SDL seems to under-sleep and THREAD seems to over-sleep
#endif // !USE_GAMEGLOBALS
