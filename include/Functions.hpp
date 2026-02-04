#include "Tensor.hpp"
#include "Utilities.hpp"
#include <cmath> // for the std::pow of MSELoss

#ifndef FUNCTIONS
#define FUNCTIONS

// ============================= OPERATIONS

class matmul_BW : public BW_Function {
public:
	matmul_BW(const TensorPtr A, const TensorPtr B, const TensorPtr out, std::string name) {
		inputs_.push_back(A);
		inputs_.push_back(B);
		outputs_.push_back(out);
		name_ = name;
	};

	// mul will have to be improve with row-first i think
	void backward() override {
		if (inputs_[0]->requires_grad()) {
			double* lgrad = inputs_[0]->gradient();
			const double* out_grad = outputs_[0]->gradient();
			const double* rdata = inputs_[1]->data();

			for (size_t i = 0; i < inputs_[0]->rows(); i++)
				for (size_t j = 0; j < inputs_[0]->cols(); j++) {
					size_t ij = i * inputs_[0]->cols() + j;
					for (size_t k = 0; k < inputs_[1]->cols(); k++)
						lgrad[ij] += out_grad[i * inputs_[1]->cols() + k] * rdata[j * inputs_[1]->cols() + k];
				}
		}

		if (inputs_[1]->requires_grad()) {
			double* rgrad = inputs_[1]->gradient();
			const double* out_grad = outputs_[0]->gradient();
			const double* ldata = inputs_[0]->data();

			for (size_t i = 0; i < inputs_[1]->rows(); i++)
				for (size_t j = 0; j < inputs_[1]->cols(); j++) {
					size_t ij = i * inputs_[1]->cols() + j;
					for (size_t k = 0; k < inputs_[0]->rows(); k++)
						rgrad[ij] += out_grad[k * inputs_[1]->cols() + j] * ldata[k * inputs_[1]->rows() + i];
				}
		}
	};
};

inline const TensorPtr matmul(const TensorPtr A, const TensorPtr B, std::string name) {
	assert(A->cols() == B->rows());

	const TensorPtr output = std::make_shared<Tensor>(A->rows(), B->cols(), A->requires_grad() || B->requires_grad());
	output->set_fn(std::make_shared<matmul_BW>(A, B, output, name));

	double* output_data = output->data();
	const double* A_data = A->data();
	const double* B_data = B->data();

	// same, next step will be row first
	for (size_t i = 0; i < output->rows(); i++) {
		for (size_t j = 0; j < output->cols(); j++) {
			size_t ij = i * output->cols() + j;
			for (int k = 0; k < A->cols(); k++)
				output_data[ij] += A_data[i * A->cols() + k] * B_data[k * B->cols() + j];
		}
	}

	// RVO
	return output;
};


class matadd_BW : public BW_Function {
public:
	matadd_BW(const TensorPtr A, const TensorPtr B, const TensorPtr out, std::string name) {
		inputs_.push_back(A);
		inputs_.push_back(B);
		outputs_.push_back(out);
		name_ = name;
	};

	void backward() override {
		if (inputs_[0]->requires_grad()) {
			double* lgrad = inputs_[0]->gradient();
			const double* out_grad = outputs_[0]->gradient();

			for (size_t ij = 0; ij < inputs_[0]->size(); ij++)
				lgrad[ij] += out_grad[ij];
		}

		if (inputs_[1]->requires_grad()) {
			double* rgrad = inputs_[1]->gradient();
			const double* out_grad = outputs_[0]->gradient();

			for (size_t ij = 0; ij < inputs_[1]->size(); ij++)
				rgrad[ij] += out_grad[ij];
		}
	};
};

inline const TensorPtr matadd(const TensorPtr A, const TensorPtr B, std::string name) {
	assert(A->rows() == B->rows());
	assert(A->cols() == B->cols());

	const TensorPtr output = std::make_shared<Tensor>(A->rows(), A->cols(), A->requires_grad() || B->requires_grad());
	output->set_fn(std::make_shared<matadd_BW>(A, B, output, name));

	double* output_data = output->data();
	const double* A_data = A->data();
	const double* B_data = B->data();

	for (size_t ij = 0; ij < output->size(); ij++)
		output_data[ij] = A_data[ij] + B_data[ij];

	// RVO
	return output;
};


class mul_BW : public BW_Function {
private:
	double value;
public:
	mul_BW(const TensorPtr A, const double& b, const TensorPtr out, std::string name) {
		inputs_.push_back(A);
		value = b;
		outputs_.push_back(out);
		name_ = name;
	};

	void backward() override {
		if (inputs_[0]->requires_grad()) {
			double* lgrad = inputs_[0]->gradient();
			const double* out_grad = outputs_[0]->gradient();

			for (size_t ij = 0; ij < inputs_[0]->size(); ij++)
				lgrad[ij] += out_grad[ij] * value;
		}
	};
};

inline const TensorPtr mul(const TensorPtr A, const double& b, std::string name) {
	const TensorPtr output = std::make_shared<Tensor>(A->rows(), A->cols(), A->requires_grad());
	output->set_fn(std::make_shared<mul_BW>(A, b, output, name));

	double* output_data = output->data();
	const double* A_data = A->data();

	for (size_t ij = 0; ij < output->size(); ij++)
		output_data[ij] = A_data[ij] * b;

	// RVO
	return output;
};


// ======================= ACTIVATION:

class ReLU_BW : public BW_Function {
public:
	ReLU_BW(const TensorPtr A, const TensorPtr out, std::string name) {
		inputs_.push_back(A);
		outputs_.push_back(out);
		name_ = name;
	};

	void backward() override {
		if (inputs_[0]->requires_grad()) {
			double* grad = inputs_[0]->gradient();
			const double* out_grad = outputs_[0]->gradient();
			const double* data = inputs_[0]->data();

			for (size_t ij = 0; ij < inputs_[0]->size(); ij++)
				grad[ij] += out_grad[ij] * (data[ij] >= 0 ? 1 : 0);
		}
	};
};

inline const TensorPtr ReLU(const TensorPtr A, std::string name) {
	const TensorPtr output = std::make_shared<Tensor>(A->rows(), A->cols(), A->requires_grad());
	output->set_fn(std::make_shared<ReLU_BW>(A, output, name));

	double* output_data = output->data();
	const double* A_data = A->data();

	for (size_t ij = 0; ij < output->size(); ij++)
		output_data[ij] = std::max(A_data[ij], 0.0);

	// RVO
	return output;
};


class softmax_BW : public BW_Function {
public:
	softmax_BW(const TensorPtr A, const TensorPtr out, std::string name) {
		inputs_.push_back(A);
		outputs_.push_back(out);
		name_ = name;
	};

	void backward() override {
		if (inputs_[0]->requires_grad()) {
			double* grad = inputs_[0]->gradient();
			const double* out_grad = outputs_[0]->gradient();
			const double* out_data = outputs_[0]->data();

			for (size_t i = 0; i < outputs_[0]->rows(); i++) {

				double product = 0;
				for (size_t k = 0; k < outputs_[0]->cols(); k++)
					product += out_grad[i * outputs_[0]->cols() + k] * out_data[i * outputs_[0]->cols() + k];

				for (size_t j = 0; j < outputs_[0]->cols(); j++)
					grad[i * outputs_[0]->cols() + j] += out_data[i * outputs_[0]->cols() + j] * (out_grad[i * outputs_[0]->cols() + j] - product);
			}
		}
	};
};

inline const void softmax(double* output, const double* input, const int& nrows, const int& ncols) {

	std::vector<double> tmp;
	tmp.resize(ncols);

	// for each row
	for (size_t i = 0; i < nrows; i++) {
		const double* A_row = input + i * ncols;
		double* output_row = output + i * ncols;

		// max
		double max = A_row[0];
		for (size_t j = 0; j < ncols; j++) {
			double value = A_row[j];
			if (value > max)
				max = value;
		}

		// sum of the exps
		double sum = 0.0;
		for (size_t j = 0; j < ncols; j++) {
			tmp[j] = std::exp(A_row[j] - max);
			sum += tmp[j];
		}

		// value exp(x_i)/sum
		for (size_t j = 0; j < ncols; j++)
			output_row[j] = tmp[j] / sum;
	}

};

inline const TensorPtr softmax(const TensorPtr A, std::string name) {
	const TensorPtr output = std::make_shared<Tensor>(A->rows(), A->cols(), A->requires_grad());
	output->set_fn(std::make_shared<softmax_BW>(A, output, name));

	double* output_data = output->data();
	const double* A_data = A->data();

	softmax(output_data, A_data, A->rows(), A->cols());

	// RVO
	return output;
};

// ============================== LOSS

class loss_BW : public BW_Function {
public:
	loss_BW(const TensorPtr input, const TensorPtr target, const TensorPtr output, std::string name) {
		inputs_.push_back(input);
		inputs_.push_back(target);
		outputs_.push_back(output);
	};

	void backward() override {
		if (inputs_[0]->requires_grad()) {
			double* lgrad = inputs_[0]->gradient();
			const double* ldata = inputs_[0]->data();
			const double* rdata = inputs_[1]->data();

			for (size_t ij = 0; ij < inputs_[0]->size(); ij++)
				lgrad[ij] += (ldata[ij] - rdata[ij]) * inputs_[0]->size();
		}
		if (inputs_[1]->requires_grad()) {
			double* rgrad = inputs_[1]->gradient();
			const double* ldata = inputs_[0]->data();
			const double* rdata = inputs_[1]->data();

			for (size_t ij = 0; ij < inputs_[0]->size(); ij++)
				rgrad[ij] += (rdata[ij] - ldata[ij]) * 2 / inputs_[0]->size();
		}
	};
};

inline const TensorPtr MSELoss(const TensorPtr input, const TensorPtr target, std::string name) {
	assert(input->rows() == target->rows());
	assert(input->cols() == target->cols());

	const TensorPtr output = std::make_shared<Tensor>(1, 1, input->requires_grad() || target->requires_grad());
	output->set_fn(std::make_shared<loss_BW>(input, target, output, name));

	double* output_data = output->data();
	const double* input_data = input->data();
	const double* target_data = target->data();

	for (size_t ij = 0; ij < target->size(); ij++)
		output_data[0] += std::pow(input_data[ij] - target_data[ij], 2);

	// mean loss reduction (not sum).
	output_data[0] /= target->size(); // if i add : 'output_data[0] /= target->size();' alors le rajouter dans le backward aussi

	// RVO
	return output;
};


class CELoss_BW : public BW_Function {
public:
	CELoss_BW(const TensorPtr input, const TensorPtr target_logits, const TensorPtr output, std::string name) {
		inputs_.push_back(input);
		inputs_.push_back(target_logits);
		outputs_.push_back(output);
	};

	void backward() override {
		if (inputs_[0]->requires_grad()) {
			double* lgrad = inputs_[0]->gradient();
			const double* ldata = inputs_[0]->data();
			const double* rdata = inputs_[1]->data();

			// should be changed to temp buffer later on
			double* softldata = new double[inputs_[0]->size()];

			softmax(softldata, ldata, inputs_[0]->rows(), inputs_[0]->cols());

			// basically : grad = softmax(input) - hot_one(target_logits)
			for (size_t i = 0; i < inputs_[0]->rows(); i++)
				for (size_t j = 0; j < inputs_[0]->cols(); j++)
					lgrad[i * inputs_[0]->cols() + j] += (softldata[i * inputs_[0]->cols() + j] - (j == rdata[i] ? 1 : 0)) / inputs_[1]->size();

			delete[] softldata;
		}
		if (inputs_[1]->requires_grad())
			print("This is the target, having a gradient here is weird.");
	};
};

inline const TensorPtr CrossEntropyLoss(const TensorPtr input, const TensorPtr target_logits, std::string name) {
	assert(input->rows() == target_logits->rows());
	assert(target_logits->size() == target_logits->rows());

	const TensorPtr output = std::make_shared<Tensor>(1, 1, input->requires_grad() || target_logits->requires_grad());
	output->set_fn(std::make_shared<CELoss_BW>(input, target_logits, output, name));

	double* output_data = output->data();
	const double* input_data = input->data();
	const double* target_data = target_logits->data();

	// LogSoftmax : logSM(i, j) = x(i, j) - max(i) - log(sum(i)) with sum(i) the sum of exps
	double* LogSoftmaxed_input = new double[input->size()];
	for (size_t i = 0; i < input->rows(); i++) {
		double max = 0;
		double sum = 0;
		for (size_t j = 0; j < input->cols(); j++)
			if (max < input_data[i * input->cols() + j])
				max = input_data[i * input->cols() + j];

		for (size_t j = 0; j < input->cols(); j++)
			sum += std::exp(input_data[i * input->cols() + j] - max);

		for (size_t j = 0; j < input->cols(); j++)
			LogSoftmaxed_input[i * input->cols() + j] = input_data[i * input->cols() + j] - max - std::log(sum); // NLLLoss of LogSoftmax
	}

	// NLLLoss : L(x, class) = -x[class]
	for (size_t i = 0; i < target_logits->size(); i++)
		output_data[0] -= LogSoftmaxed_input[i * input->cols() + static_cast<int>(target_data[i])];
	delete[] LogSoftmaxed_input;

	// normalizing
	output_data[0] /= target_logits->size();

	// RVO, no copy
	return output;
};


// To add : nn.CrossEntropyLoss (avec NLLoss ?), nn.Dropout, nn.Embedding, nn.LayerNorm, nn.Conv2D, nn.MaxPool2D

#endif // !FUNCTIONS