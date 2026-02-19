#include "Tensor.hpp"
#include <sstream> // for ostringstream
#include <string>

#ifndef FUNCS
#define FUNCS

inline void tabulation(std::ostringstream& stream, const int n) {
	for (int i = 0; i < n; i++)
		stream << " ";
}

inline void printReccursive(std::ostringstream& stream, std::vector<size_t> dims, size_t I, const Scalar* data, const bool& is_first, const bool& is_last, int indent) {
	if (dims.size() <= 2) {
		size_t rows = dims[0];
		size_t cols = dims[1];

		indent += 2;
		stream << "[";
		for (size_t i = 0; i < rows; i++) {
			if (i != 0) tabulation(stream, indent);
			stream << "[";
			for (size_t j = 0; j < cols; j++) {
				stream << data[I*rows*cols + i * cols + j];
				if (j < cols - 1) stream << ", ";
			}
			stream << "]";
			if (i < rows - 1) stream << ", " << std::endl;
		}
		stream << "]";
		if (!is_last) stream << "," << std::endl << std::endl;
		if (is_first) tabulation(stream, indent-1);
	}
	else {
		stream << "[";
		indent += 1;
		size_t first = dims[0];
		dims.erase(dims.begin());
		for (size_t i = 0; i < first; i++)
			printReccursive(stream, dims, I * first + i, data, (i==0), (i==(first - 1)), indent);
		stream << "]";
		if (!is_last) {
			stream << "," << std::endl << std::endl;
			tabulation(stream, indent);
		}
	}
}


inline void print(Tensor& A) {

	std::ostringstream stream;
	stream << " ";

	const Scalar* data = A.data();
	
	printReccursive(stream, A.dims(), 0, data, true, false, 0);

	std::cout << stream.str();
}

// Tensor print
inline void oldprint(Tensor& A) {
	const size_t rows = A.rows();
	const size_t cols = A.cols();
	const Scalar* data = A.data();

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
	const Scalar* grad = A.gradient();

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