#pragma once

#include "Math.h"
#include <vector>

namespace NN {

typedef Math::Vector<double> Vector;
typedef Math::Matrix<double> Matrix;

class ActivationFunction {
public:
	virtual Vector func(const Vector&) = 0;
	virtual Vector derivedFunc(const Vector&) = 0;
};

class Sigmoid : public ActivationFunction {
public:
	Vector func(const Vector&);
	Vector derivedFunc(const Vector&);
};

class ReLu : public ActivationFunction {
public:
	Vector func(const Vector&);
	Vector derivedFunc(const Vector&);
};

class LeakyReLu : public ActivationFunction {
public:
	Vector func(const Vector&);
	Vector derivedFunc(const Vector&);
};

class Identity : public ActivationFunction {
public:
	Vector func(const Vector&);
	Vector derivedFunc(const Vector&);
};

class CostFunction {
public:
	virtual double func(const Vector&, const Vector&) = 0;
	virtual Vector derivedFunc(const Vector&, const Vector&) = 0;
};

class Quadratic : public CostFunction {
public:
	double func(const Vector&, const Vector&);
	Vector derivedFunc(const Vector&, const Vector&);
};

class TrainingsData {
public:
	TrainingsData() {};
	TrainingsData(const std::vector<Vector>& z, const std::vector<Vector>& a, const Vector& y);
	const std::vector<Vector>& getZ() const;
	const std::vector<Vector>& getA() const;
	const Vector& getY() const;
private:
	std::vector<Vector> z_;
	std::vector<Vector> a_;
	Vector y_;
};

class NeuralNetwork {
public:
	NeuralNetwork();
	NeuralNetwork(const std::vector<int>&);
	~NeuralNetwork();

	void addLayer(const int&);
	void randomizeWeightsAndBiases(const int& seed = -1);
	void setFunctions(ActivationFunction* hidden, ActivationFunction* output, CostFunction* cost);
	const Vector& computeOutput(const Vector&);
	double train(const Vector&, const Vector&);
	void applyTrainingsBatch(const double&);
private:
	std::vector<Vector> z_; //w*(a-1)+b
	std::vector<Vector> a_; //sigma(z)
	std::vector<Vector> b_; //biases
	std::vector<Matrix> w_; //weights

	ActivationFunction* hiddenLayerActivationFunction_ = NULL;
	ActivationFunction* outputLayerActivationFunction_ = NULL;
	CostFunction* costFunction_ = NULL;

	std::vector<TrainingsData> trainingsData_;

	void setInput(const Vector&);
	void backpropagate(const TrainingsData&, std::vector<Vector>&, std::vector<Matrix>&);
};

}