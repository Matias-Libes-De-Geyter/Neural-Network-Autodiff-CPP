#include "Tensor.hpp"
#include <fstream> // for ifstream

#ifndef DATASET
#define DATASET

// ======== DATASET LOADER ======== //
inline int reverseInt(int i) { // to little-endian
	unsigned char c1, c2, c3, c4;
	c1 = i & 255;
	c2 = (i >> 8) & 255;
	c3 = (i >> 16) & 255;
	c4 = (i >> 24) & 255;
	return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}
inline std::vector<int> readMNIST(const std::string& imageFile, const std::string& labelFile, std::vector<Scalar>& images, std::vector<Scalar>& labels) {
	std::ifstream imgFile(imageFile, std::ios::binary);
	std::ifstream lblFile(labelFile, std::ios::binary);

	int magicNumber, numImages, numRows, numCols;
	imgFile.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
	magicNumber = reverseInt(magicNumber);
	imgFile.read(reinterpret_cast<char*>(&numImages), sizeof(numImages));
	numImages = reverseInt(numImages);
	imgFile.read(reinterpret_cast<char*>(&numRows), sizeof(numRows));
	numRows = reverseInt(numRows);
	imgFile.read(reinterpret_cast<char*>(&numCols), sizeof(numCols));
	numCols = reverseInt(numCols);

	int labelMagicNumber, numLabels;
	lblFile.read(reinterpret_cast<char*>(&labelMagicNumber), sizeof(labelMagicNumber));
	labelMagicNumber = reverseInt(labelMagicNumber);
	lblFile.read(reinterpret_cast<char*>(&numLabels), sizeof(numLabels));
	numLabels = reverseInt(numLabels);

	int C = std::min(numImages, numLabels);
	images = std::vector<Scalar>(C * numRows * numCols);
	labels = std::vector<Scalar>(1 * C);

	for (int i = 0; i < C; ++i) {
		for (int j = 0; j < numRows * numCols; ++j) {
			unsigned char pixel;
			imgFile.read(reinterpret_cast<char*>(&pixel), sizeof(pixel));
			images[i * numRows * numCols + j] = static_cast<Scalar>(pixel) / 255.0;
		}
		unsigned char label;
		lblFile.read(reinterpret_cast<char*>(&label), sizeof(label));
		labels[i] = static_cast<Scalar>(label);
	}

	return { numRows, numCols, C };
}


// ======== DATASET ======== //
struct Dataset {
	std::vector<TensorPtr> x;
	std::vector<TensorPtr> y;
};
inline Dataset DataLoader(const int& batch_number, const int& batch_size, const std::string& dataset_type) {

	std::vector<Scalar> images;
	std::vector<Scalar> labels;
	std::string ImagesFile;
	std::string LabelsFile;
	if (dataset_type == "train") {
		ImagesFile = "C:/Users/saïtama/Documents/Documents/Programming/--=NeuralNetworks=--/-= Datasets/MNIST/train-images.idx3-ubyte";
		LabelsFile = "C:/Users/saïtama/Documents/Documents/Programming/--=NeuralNetworks=--/-= Datasets/MNIST/train-labels.idx1-ubyte";
	}
	else if (dataset_type == "validation") {
		ImagesFile = "C:/Users/saïtama/Documents/Documents/Programming/--=NeuralNetworks=--/-= Datasets/MNIST/t10k-images.idx3-ubyte";
		LabelsFile = "C:/Users/saïtama/Documents/Documents/Programming/--=NeuralNetworks=--/-= Datasets/MNIST/t10k-labels.idx1-ubyte";
	}
	else
		print("Dataset type is wrong");

	std::vector<int> size = readMNIST(ImagesFile, LabelsFile, images, labels);
	const int nrows = size[0];
	const int ncols = size[1];
	const int C = size[2];

	Dataset data;
	data.x.reserve(batch_number);
	data.y.reserve(batch_number);

	for (int n = 0; n < batch_number; n++) {
		std::vector<Scalar> batch_x(&images[batch_size * nrows * ncols * n], &images[batch_size * nrows * ncols * (n + 1)]);
		std::vector<Scalar> batch_y(&labels[batch_size * n], &labels[batch_size * (n + 1)]);
		
		TensorPtr X = std::make_shared<Tensor>(batch_size, nrows * ncols, batch_x, "IMG", false);
		TensorPtr Y = std::make_shared<Tensor>(batch_size, 1, batch_y, "LBL", false);

		data.x.emplace_back(X);
		data.y.emplace_back(Y);
	}

	return data;
};

#endif