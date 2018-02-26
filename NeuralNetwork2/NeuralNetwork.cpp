#include "NeuralNetwork.h"
#include <random>
#include <time.h>

using namespace std;

namespace NN {

Vector Sigmoid::func(const Vector& vec) {
	Vector ret(vec.getSize());
	for (int i = 0; i < vec.getSize(); i++)
		ret[i] = 1.0 / (1.0 + exp(-vec[i]));
	return ret;
};

Vector Sigmoid::derivedFunc(const Vector& vec) {
	Vector ret = func(vec);
	Vector one(vec.getSize());
	for (int i = 0; i < vec.getSize(); i++)
		one[i] = 1.0;
	return Math::hadamard(ret, one - ret);
};

Vector ReLu::func(const Vector& vec) {
	Vector ret(vec.getSize());
	for (int i = 0; i < vec.getSize(); i++)
		ret[i] = vec[i] > 0 ? vec[i] : 0;
	return ret;
};

Vector ReLu::derivedFunc(const Vector& vec) {
	Vector ret(vec.getSize());
	for (int i = 0; i < vec.getSize(); i++)
		ret[i] = vec[i] > 0 ? 1 : 0;
	return ret;
};

Vector LeakyReLu::func(const Vector& vec) {
	Vector ret(vec.getSize());
	for (int i = 0; i < vec.getSize(); i++)
		ret[i] = vec[i] > 0 ? vec[i] : -0.01 * vec[i];
	return ret;
};

Vector LeakyReLu::derivedFunc(const Vector& vec) {
	Vector ret(vec.getSize());
	for (int i = 0; i < vec.getSize(); i++)
		ret[i] = vec[i] > 0 ? 1 : -0.01;
	return ret;
};

Vector Identity::func(const Vector& vec) {
	return vec;
};

Vector Identity::derivedFunc(const Vector& vec) {
	Vector ret(vec.getSize());
	for (int i = 0; i < vec.getSize(); i++)
		ret[i] = 1.0;
	return ret;
};

double Quadratic::func(const Vector& lhs, const Vector& rhs) {
	double ret = 0.0;
	for (int i = 0; i < lhs.getSize(); i++)
		ret += pow(lhs[i] - rhs[i], 2) * 0.5;
	return ret;
};

Vector Quadratic::derivedFunc(const Vector& lhs, const Vector& rhs) {
	return (lhs - rhs);
};

TrainingsData::TrainingsData(const std::vector<Vector>& z, const std::vector<Vector>& a, const Vector& y) {
	z_ = z;
	a_ = a;
	y_ = y;
};

const vector<Vector>& TrainingsData::getZ() const {
	return z_;
};

const vector<Vector>& TrainingsData::getA() const {
	return a_;
};

const Vector& TrainingsData::getY() const {
	return y_;
};

NeuralNetwork::NeuralNetwork() {};

NeuralNetwork::NeuralNetwork(const std::vector<int>& layer) {
	for (const int& i : layer)
		addLayer(i);
};

NeuralNetwork::~NeuralNetwork() {
	if (hiddenLayerActivationFunction_)
		delete hiddenLayerActivationFunction_;
	if (outputLayerActivationFunction_)
		delete outputLayerActivationFunction_;
	if (costFunction_)
		delete costFunction_;
}

void NeuralNetwork::addLayer(const int& i) {
	z_.push_back(Vector(i));
	a_.push_back(Vector(i));
	if (z_.size() > 1) {
		b_.push_back(Vector(i));
		w_.push_back(Matrix(i, z_[z_.size() - 2].getSize()));
	}
};

void NeuralNetwork::randomizeWeightsAndBiases(const int& seed) {
	if (seed >= 0)
		srand(seed);
	else
		srand(int(time(0)));

	for (Matrix& m : w_) {
		for (int i = 0; i < m.getNumRows(); i++) {
			for (int j = 0; j < m.getNumColumns(); j++)
				m[i][j] = double(rand() % 10000) / 1666.67 - 3.0;
		}
	}

	for (Vector& v : b_) {
		for (int i = 0; i < v.getSize(); i++)
			v[i] = double(rand() % 10000) / 1666.67 - 3.0;
	}
}

void NeuralNetwork::setFunctions(ActivationFunction* hidden, ActivationFunction* output, CostFunction* cost) {
	hiddenLayerActivationFunction_ = hidden;
	outputLayerActivationFunction_ = output;
	costFunction_ = cost;
};


void NeuralNetwork::setInput(const Vector& vec) {
	z_[0] = vec;
	a_[0] = vec;
}

const Vector& NeuralNetwork::computeOutput(const Vector& i) {
	setInput(i);
	for (int i = 1; i < z_.size(); i++) {
		z_[i] = w_[i - 1] * a_[i - 1] + b_[i - 1];
		if (i < z_.size() - 1)
			a_[i] = hiddenLayerActivationFunction_->func(z_[i]);
		else
			a_[i] = outputLayerActivationFunction_->func(z_[i]);
	}
	return a_.back();
};

double NeuralNetwork::train(const Vector& in, const Vector& out) {
	computeOutput(in);
	trainingsData_.push_back(TrainingsData(z_, a_, out));
	return costFunction_->func(a_.back(), out);
}

void NeuralNetwork::applyTrainingsBatch(const double& learnRate) {
	vector<Vector> biasGradients;
	for (int i = 0; i < b_.size(); i++)
		biasGradients.push_back(Vector(b_[i].getSize()));
	vector<Matrix> edgeGradients;
	for (int i = 0; i < w_.size(); i++)
		edgeGradients.push_back(Matrix(w_[i].getNumRows(), w_[i].getNumColumns()));

	for (const TrainingsData& td : trainingsData_) {
		vector<Vector> bg;
		vector<Matrix> eg;
		backpropagate(td, bg, eg);
		for (int i = 0; i < biasGradients.size(); i++)
			biasGradients[i] += bg[i];
		for (int i = 0; i < edgeGradients.size(); i++)
			edgeGradients[i] += eg[i];
	}

	for (int i = 0; i < biasGradients.size(); i++)
		b_[i] -= (biasGradients[i] * (learnRate / trainingsData_.size()));
	for (int i = 0; i < edgeGradients.size(); i++)
		w_[i] -= (edgeGradients[i] * (learnRate / trainingsData_.size()));

	trainingsData_.clear();
};

void NeuralNetwork::backpropagate(const TrainingsData& td, vector<Vector>& bg, vector<Matrix>& eg) {
	const vector<Vector>& a = td.getA();
	const vector<Vector>& z = td.getZ();

	vector<Vector> delta = a;

	delta.back() = Math::hadamard(costFunction_->derivedFunc(a.back(), td.getY()), outputLayerActivationFunction_->derivedFunc(z.back()));
	for (int i = delta.size() - 2; i > 0; i--)
		delta[i] = Math::hadamard(Math::transpose(w_[i]) * delta[i + 1], hiddenLayerActivationFunction_->derivedFunc(z[i]));

	for (int i = 0; i < b_.size(); i++)
		bg.push_back(delta[i + 1]);

	for (int i = 0; i < w_.size(); i++)
		eg.push_back(Math::outerProduct(delta[i + 1], a[i]));
};

}