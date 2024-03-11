#ifndef USE_CUDACALC
#define USE_CUDACALC
#include <vector>
class Boid; //fwd dec
namespace JRCudaCalc {
	void GetNeighboursCUDA(std::vector<Boid*>& AllBoids);
	void Init(int size);
	void Clear();
}
#endif // !USE_CUDACALC


