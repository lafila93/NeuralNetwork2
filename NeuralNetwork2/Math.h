#pragma once

#include <vector>
#include <iostream>

namespace Math {

template <class T> class Vector {
public:
	Vector();
	Vector(const int&);
	Vector(const std::vector<T>&);

	int getSize() const;
	const bool& isValid() const;

	T& operator[](const int&);
	const T& operator[](const int&) const;
	Vector<T> operator+(const Vector<T>&) const;
	Vector<T> operator+=(const Vector<T>&);
	Vector<T> operator-(const Vector<T>&) const;
	Vector<T> operator-=(const Vector<T>&);
	Vector<T> operator*(const T&) const;
private:
	bool valid_{ false };
	std::vector<T> data_;
};

template <class T> class Matrix {
public:
	Matrix();
	Matrix(const int&, const int&);
	Matrix(const std::vector<Vector<T>>&);

	const int& getNumRows() const;
	const int& getNumColumns() const;

	Vector<T>& operator[](const int&);
	const Vector<T>& operator[](const int&) const;
	Matrix<T> operator+(const Matrix<T>&) const;
	Matrix<T> operator+=(const Matrix<T>&);
	Matrix<T> operator-(const Matrix<T>&) const;
	Matrix<T> operator-=(const Matrix<T>&);
	Matrix<T> operator*(const T&) const;
	Vector<T> operator*(const Vector<T>&) const;
	Matrix<T> operator*(const Matrix<T>&) const;
private:
	bool valid_{ false };
	std::vector<Vector<T>> data_;
	int numRows_{ 0 }, numCols_{ 0 };
};

template <class T> T negate(const T&);
template <class T> T skalar(const Vector<T>&, const Vector<T>&);
template <class T> Vector<T> hadamard(const Vector<T>&, const Vector<T>&);
template <class T> Matrix<T> outerProduct(const Vector<T>&, const Vector<T>&);
template <class T> std::ostream& operator<<(std::ostream&, const Vector<T>&);
template <class T> std::ostream& operator<<(std::ostream&, const Matrix<T>&);
template <class T> Matrix<T> transpose(const Matrix<T>&);

/*********************************************/
//					Vector
/*********************************************/

template <class T> Vector<T>::Vector() {
};

template <class T> Vector<T>::Vector(const int& i) {
	data_ = std::vector<T>(i, T(0));
	valid_ = true;
};

template <class T> Vector<T>::Vector(const std::vector<T>& vec) {
	data_ = vec;
	valid_ = true;
};

template <class T> int Vector<T>::getSize() const {
	return data_.size();
};

template <class T> const bool& Vector<T>::isValid() const {
	return valid_;
};

template <class T> T& Vector<T>::operator[](const int& i) {
	return data_[i];
};

template <class T> const T& Vector<T>::operator[](const int& i) const {
	return data_[i];
};

template <class T> Vector<T> Vector<T>::operator+(const Vector<T>& rhs) const {
	if (data_.size() != rhs.getSize())
		exit(1);
	Vector<T> ret(data_.size());
	for (int i = 0; i < data_.size(); i++)
		ret[i] = data_[i] + rhs[i];
	return ret;
};

template <class T> Vector<T> Vector<T>::operator+=(const Vector<T>& rhs) {
	*this = (*this + rhs);
	return *this;
};

template <class T> Vector<T> Vector<T>::operator-(const Vector<T>& rhs) const {
	return (*this + negate(rhs));
};

template <class T> Vector<T> Vector<T>::operator-=(const Vector<T>& rhs) {
	*this = (*this - rhs);
	return *this;
};

template <class T> Vector<T> Vector<T>::operator*(const T& s) const {
	Vector<T> ret = *this;
	for (int i = 0; i < getSize(); i++)
		ret[i] *= s;
	return ret;
};

/*********************************************/
//					Matrix
/*********************************************/

template <class T> Matrix<T>::Matrix() {
};

template <class T> Matrix<T>::Matrix(const int& row, const int& col) {
	data_ = std::vector<Vector<T>>(row, Vector<T>(col));
	numRows_ = row;
	numCols_ = col;
	valid_ = true;
};

template <class T> Matrix<T>::Matrix(const std::vector<Vector<T>>& vecs) {
	data_ = vecs;
	numRows_ = vecs.size();
	numCols_ = vecs[0].getSize();
	valid_ = true;
};

template <class T> const int& Matrix<T>::getNumRows() const {
	return numRows_;
};

template <class T> const int& Matrix<T>::getNumColumns() const {
	return numCols_;
};

template <class T> Vector<T>& Matrix<T>::operator[](const int& i) {
	return data_[i];
};

template <class T> const Vector<T>& Matrix<T>::operator[](const int& i) const {
	return data_[i];
};

template <class T> Matrix<T> Matrix<T>::operator+(const Matrix<T>& rhs) const {
	if (getNumRows() != rhs.getNumRows() || getNumColumns() != rhs.getNumColumns())
		exit(1);
	Matrix<T> ret(getNumRows(), getNumColumns());
	for (int i = 0; i < getNumRows(); i++)
		ret[i] = data_[i] + rhs[i];
	return ret;
};

template <class T> Matrix<T> Matrix<T>::operator+=(const Matrix<T>& rhs) {
	*this = *this + rhs;
	return *this;
};

template <class T> Matrix<T> Matrix<T>::operator-(const Matrix<T>& rhs) const {
	return (*this + negate(rhs));
};

template <class T> Matrix<T> Matrix<T>::operator-=(const Matrix<T>& rhs) {
	*this = (*this - rhs);
	return *this;
};

template <class T> Vector<T> Matrix<T>::operator*(const Vector<T>& vec) const {
	Vector<T> ret(getNumRows());
	for (int i = 0; i < getNumRows(); i++)
		ret[i] = skalar(data_[i], vec);
	return ret;
};

template <class T> Matrix<T> Matrix<T>::operator*(const T& s) const {
	Matrix<T> ret = *this;
	for (int i = 0; i < getNumRows(); i++) {
		Vector<T>& vec = ret[i];
		vec = vec * s;
	}
	return ret;
};

template <class T> Matrix<T> Matrix<T>::operator*(const Matrix<T>& rhs) const {
	if (getNumColumns() != rhs.getNumRows())
		exit(1);
	Matrix<T> ret(getNumRows(), rhs.getNumColumns());
	for (int i = 0; i < getNumRows(); i++) {
		for (int j = 0; j < rhs.getNumColumns(); j++) {
			T& val = ret[i][j];
			for (int k = 0; k < getNumColumns(); k++)
				val += data_[i][k] * rhs[k][j];
		}
	}
	return ret;
};

/*********************************************/
//					Classless
/*********************************************/

template <class T> T negate(const T& obj) {
	return obj * (-1);
};

template <class T> T skalar(const Vector<T>& lhs, const Vector<T>& rhs) {
	if (lhs.getSize() != rhs.getSize())
		exit(1);
	T ret = 0;
	for (int i = 0; i < lhs.getSize(); i++)
		ret += lhs[i] * rhs[i];
	return ret;
};

template <class T> Vector<T> hadamard(const Vector<T>& lhs, const Vector<T>& rhs) {
	if (lhs.getSize() != rhs.getSize())
		exit(1);
	Vector<T> ret(lhs.getSize());
	for (int i = 0; i < lhs.getSize(); i++)
		ret[i] = lhs[i] * rhs[i];
	return ret;
};

template <class T> Matrix<T> outerProduct(const Vector<T>& lhs, const Vector<T>& rhs) {
	Matrix<T> ret(lhs.getSize(), rhs.getSize());
	for (int i = 0; i < lhs.getSize(); i++) {
		for (int j = 0; j < rhs.getSize(); j++)
			ret[i][j] = lhs[i] * rhs[j];
	}
	return ret;
};

template <class T> std::ostream& operator<<(std::ostream& os, const Vector<T>& vec) {
	os << "(";
	for (int i = 0; i < vec.getSize(); i++) {
		os << vec[i];
		if (i < vec.getSize() - 1)
			os << ", ";
	}
	os << ")";
	return os;
};

template <class T> std::ostream& operator<<(std::ostream& os, const Matrix<T>& mat) {
	for (int i = 0; i < mat.getNumRows(); i++) {
		os << "|" << mat[i] << "|";
		if (i < mat.getNumRows() - 1)
			os << endl;
	}
	return os;
};

template <class T> Matrix<T> transpose(const Matrix<T>& mat) {
	Matrix<T> ret(mat.getNumColumns(), mat.getNumRows());
	for (int i = 0; i < mat.getNumRows(); i++) {
		for (int j = 0; j < mat.getNumColumns(); j++)
			ret[j][i] = mat[i][j];
	}
	return ret;
};

}