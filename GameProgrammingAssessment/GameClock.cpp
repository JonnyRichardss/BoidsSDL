#include "GameClock.h"
#include "SDL.h"
#include "Global_Flags.h"
#include <thread>
static GameClock* _instance;
GameClock::GameClock() : ENGINE_START_TP(std::chrono::high_resolution_clock::now())
{
	logging = GameLogging::GetInstance();
	last_frame_tp = ENGINE_START_TP;
	frame_start_tp = ENGINE_START_TP;
	input_tp = ENGINE_START_TP;
	update_tp = ENGINE_START_TP;
	render_tp = ENGINE_START_TP;
	framecounter = 0;
	frametime_ns = 0ns;
	target_ns = 0ns;
	unused_ns = 0ns;
	SetFPSLimit(FRAME_CAP);
	logging->Log("Initialised game clock.");
}

GameClock::~GameClock()
{
}


GameClock* GameClock::GetInstance()
{
	if (_instance == nullptr) {
		_instance = new GameClock();
	}
	return _instance;
}

void GameClock::Tick()
{
	framecounter++;
	auto frameProcessTime = target_ns - GetRemainingBudget();
	EnforceLimit();
	frametime_ns = TimeSinceLastFrame();
	last_frame_tp = std::chrono::high_resolution_clock::now();
	std::string logString = "Frame ";
	logString.append(std::to_string(GetFrameCount()) + " - ");
	logString.append(std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(frameProcessTime).count()) + "ms - ");
	//logString.append(std::to_string(GetFPS()) + " - ");
	logString.append(std::to_string((int)(GetBudgetPercent())) + "% ");
	
	logging->Log(logString);
	if (DO_PROFILING) {
		std::string profileLog = "Profiling:\n";
		profileLog.append("INPUT - "+std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(input_tp - frame_start_tp).count())+"\u00B5s\n");
		profileLog.append("UPDATE - " + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(update_tp - input_tp).count()) + "\u00B5s\n");
		profileLog.append("RENDER - " + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(render_tp - update_tp).count()) + "\u00B5s\n");
		//UNLESS I REWORK PROFILING SPECIAL TPS *HAVE* TO BE IN UPDATE
		//also i cant use this for what i wanted to LOL
		//HAVE ENUM SPECIAL TP PHASE
		//MAYBE SPECIAL TP STRUCT??????
		//SEEMS LOGICAL

		if (!special_tps.empty()) {
			for (int i = 0; i < special_tps.size(); i++) {
				if (i == 0)
					profileLog.append(special_tp_names[i] + " - " + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(special_tps[i] - update_tp).count()) + "\u00B5s\n");
				else
					profileLog.append(special_tp_names[i] + " - " + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(special_tps[i] - special_tps[i-1]).count()) + "\u00B5s\n");
			}
			special_tps.clear();
			special_tp_names.clear();
		}
		logging->Log(profileLog);
	}
	
}

std::chrono::nanoseconds GameClock::GetFrametime()
{
	return frametime_ns;
}

long long GameClock::GetFrameCount()
{
	return framecounter;
}

float GameClock::GetBudgetPercent()
{
	if (target_ns == 0ns)
		return 100.0f;
	else
		return roundf((1- (float)unused_ns.count() / (float)target_ns.count()) * 100 );
}

std::chrono::nanoseconds GameClock::GetRemainingBudget()
{
	return target_ns - TimeSinceLastFrame();
}

std::chrono::high_resolution_clock::time_point GameClock::GetTimePoint()
{
	return last_frame_tp;
}

int GameClock::GetFPS()
{
	if (frametime_ns == 0ns)
		return 1000000000;
	else
		return 1000000000/frametime_ns.count();
}

void GameClock::TickProfiling(ProfilerPhases phase)
{
	if (!DO_PROFILING)
		return;
	auto TP = std::chrono::high_resolution_clock::now();
	switch (phase) {
		case STARTPHASE:
			frame_start_tp = TP;
			break;
		case INPUTPHASE:
			input_tp = TP;
			break;
		case UPDATEPHASE:
			update_tp = TP;
			break;
		case RENDERPHASE:
			render_tp = TP;
			break;
		default:
			throw "Invalid profile phase";
			//pretty sure the enum prevents invalid values being set but this isnt hurting anyone so
			break;
	}
}
void GameClock::TickProfilingSpecial(std::string name) {
	special_tps.push_back(std::chrono::high_resolution_clock::now());
	special_tp_names.push_back(name);
}

void GameClock::SetFPSLimit(int newLimit)
{
	if (newLimit == 0) 
		target_ns = 0ns;
	else
		target_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(1s) / newLimit;
}

void GameClock::EnforceLimit()
{
	unused_ns = GetRemainingBudget();
	if (unused_ns <= 0ns)
		return;
	else
		WaitFor(unused_ns);
}

std::chrono::nanoseconds GameClock::TimeSinceLastFrame()
{
	return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - last_frame_tp);
}

void GameClock::WaitFor(std::chrono::nanoseconds wait_ns)
{
	switch (GF_WAIT_METHOD) {
	case SDL:
		SDL_Delay(std::chrono::duration_cast<std::chrono::milliseconds>(wait_ns).count());
		break;
	case THREAD:
		std::this_thread::sleep_for(wait_ns);
		break;
	case BUSY:
	{
		auto wait_until = std::chrono::high_resolution_clock::now() + wait_ns;
		while (std::chrono::high_resolution_clock::now() < wait_until) {
			//DoThingsWhileWaiting()
			//NOTE TO SELF - DONT ALLOW DISABLING FRAME LIMITER IF YOU IMPLEMENT FUNCTIONALITY IN HERE
		}
		break;
	}
	default:
		throw "Invalid Wait Method";
		//pretty sure the enum prevents invalid values being set but this isnt hurting anyone so
		break;
	}
}
