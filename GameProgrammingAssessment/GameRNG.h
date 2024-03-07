#pragma once
#ifndef USE_RNG
#define USE_RNG

class RNG
{
public:
	/*
	* Provides a random integer between min and max (both inclusive)
	*/
	static int randi(int min, int max);
	/*
	* Provides a random integer between min and max (both inclusive)
	*/
	static float randf(float min, float max);
	static void Seed();
private:
	RNG();// DO NOT CALL MANUALLY, USE STATIC FUNCTIONS
	~RNG();
};
#endif // !USE_RNG

