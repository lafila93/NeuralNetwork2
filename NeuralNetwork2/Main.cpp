#include "Math.h"
#include "NeuralNetwork.h"
#include <iostream>

using namespace std;
using namespace NN;

typedef Math::Vector<double> Vector;
typedef Math::Matrix<double> Matrix;

Math::Vector<double> func(const Math::Vector<double>& vec) {
	Math::Vector<double> ret(4);
	ret[0] = vec[0] + vec[1];
	ret[1] = vec[0] - vec[1];
	ret[2] = vec[0] * vec[1];
	ret[3] = (vec[0] > vec[1] ? 1 : 0);
	return ret;
};

int main() {
	NeuralNetwork nn(vector<int>({ 2,7,4 }));
	nn.randomizeWeightsAndBiases(1337);
	nn.setFunctions(new LeakyReLu(), new Identity(), new Quadratic());

	srand(int(1));
	for (int j = 0; j < 10000; j++) {
		double cost = 0.0;
		for (int i = 0; i < 50; i++) {
			Math::Vector<double> vin(vector<double>({ rand() % 10000 / 1666.67 - 3, rand() % 10000 / 1666.67 - 3 }));
			Math::Vector<double> vout = func(vin);

			cost += nn.train(vin, vout);
		}
		cout << cost << endl;
		nn.applyTrainingsBatch(0.03);

		Math::Vector<double> vin(vector<double>({ rand() % 10000 / 1666.67 - 3, rand() % 10000 / 1666.67 - 3 }));
		cout << nn.computeOutput(vin) << " vs " << func(vin) << endl;
	}

	return 0;
}