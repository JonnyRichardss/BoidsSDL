#ifndef USE_CUDACALC
#define USE_CUDACALC
#include <vector>
class Boid; //fwd dec
namespace JRCudaCalc {
	void DoCalc(std::vector<Boid*>& AllBoids);
}
#endif // !USE_CUDACALC


