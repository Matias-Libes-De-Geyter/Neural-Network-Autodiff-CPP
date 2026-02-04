#include "Tensor.hpp"
#include <sstream> // for ostringstream
#include <string>

#ifndef FUNCS
#define FUNCS

// Tensor print
inline void print(Tensor& A) {
	const size_t rows = A.rows();
	const size_t cols = A.cols();
	const double* data = A.data();

	std::ostringstream stream;
	stream << "[";
	for (size_t i = 0; i < rows; i++) {
		stream << "[";
		for (size_t j = 0; j < cols; j++) {
			stream << data[i*cols + j];
			if (j < cols - 1) stream << ", ";
		}
		stream << "]";
		if (i < rows - 1) stream << ", " << std::endl;
	}
	stream << "]" << std::endl;

	std::cout << stream.str();
}

inline void print(const std::string& text, Tensor& A) {
	std::cout << text;
	print(A);
}

// Gradient print
inline void printgrad(Tensor& A) {
	const size_t rows = A.rows();
	const size_t cols = A.cols();
	const double* grad = A.gradient();

	std::ostringstream stream;
	stream << "[";
	for (size_t i = 0; i < rows; i++) {
		stream << "[";
		for (size_t j = 0; j < cols; j++) {
			stream << grad[i * cols + j];
			if (j < cols - 1) stream << ", ";
		}
		stream << "]";
		if (i < rows - 1) stream << ", " << std::endl;
	}
	stream << "]" << std::endl;

	std::cout << stream.str();
}

// Single print
template<typename T>
typename std::enable_if<!std::is_same<T, Tensor>::value, void>::type
inline print(const T& arg) { std::cout << arg << std::endl; }

#endif // !FUNCS