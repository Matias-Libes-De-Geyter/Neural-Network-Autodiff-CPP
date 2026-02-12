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

		children_.push_back(A->get_fn());
		children_.push_back(B->get_fn());
	};

	void backward() override {
		const size_t M = inputs_[0]->rows();
		const size_t K = inputs_[0]->cols();
		const size_t N = inputs_[1]->cols();

		if (inputs_[0]->requires_grad()) {

			Scalar* lgrad = inputs_[0]->gradient();
			const Scalar* out_grad = outputs_[0]->gradient();
			const Scalar* rdata = inputs_[1]->data();

			for (size_t i = 0; i < M; i++)
				for (size_t j = 0; j < K; j++) {
					size_t ij = i * K + j;
					for (size_t k = 0; k < N; k++)
						lgrad[ij] += out_grad[i * N + k] * rdata[j * N + k];
				}
		}

		// row-first implanted
		if (inputs_[1]->requires_grad()) {
			Scalar* rgrad = inputs_[1]->gradient();
			const Scalar* out_grad = outputs_[0]->gradient();
			const Scalar* ldata = inputs_[0]->data();

			for (size_t k = 0; k < M; k++) {
				const Scalar* out_col = out_grad + k * N;
				for (size_t i = 0; i < K; i++) {
					Scalar data = ldata[k * K + i];
					Scalar* rgrad_col = rgrad + i * N;
					for (size_t j = 0; j < N; j++)
						rgrad_col[j] += out_col[j] * data;
				}
			}
		}

	};
};

inline const TensorPtr matmul(const TensorPtr A, const TensorPtr B, std::string name) {
	assert(A->cols() == B->rows());

	const size_t M = A->rows();
	const size_t K = A->cols();
	const size_t N = B->cols();

	const TensorPtr output = std::make_shared<Tensor>(M, N, A->requires_grad() || B->requires_grad());
	output->set_fn(std::make_shared<matmul_BW>(A, B, output, name));

	Scalar* output_data = output->data();
	const Scalar* A_data = A->data();
	const Scalar* B_data = B->data();

	// I guess it's faster because the pointer is always already in the good place, no need to search k*B_cols each time we search for k*B_cols+j
	for (size_t i = 0; i < M; i++) {
		for (size_t k = 0; k < K; k++) {
			Scalar data = A_data[i * K + k];
			const Scalar* B_col = B_data + k * N;
			Scalar* out_row = output_data + i * N;
			for (size_t j = 0; j < N; j++)
				out_row[j] += data * B_col[j];
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

		children_.push_back(A->get_fn());
		children_.push_back(B->get_fn());
	};

	void backward() override {
		if (inputs_[0]->requires_grad()) {
			Scalar* lgrad = inputs_[0]->gradient();
			const Scalar* out_grad = outputs_[0]->gradient();

			for (size_t ij = 0; ij < inputs_[0]->size(); ij++)
				lgrad[ij] += out_grad[ij];
		}

		if (inputs_[1]->requires_grad()) {
			Scalar* rgrad = inputs_[1]->gradient();
			const Scalar* out_grad = outputs_[0]->gradient();

			for (size_t ij = 0; ij < inputs_[1]->size(); ij++)
				rgrad[ij] += out_grad[ij];
		}
	};
};

inline const TensorPtr matadd(const TensorPtr A, const TensorPtr B, std::string name) {
	assert(A->rows() == B->rows());
	assert(A->cols() == B->cols());

	const size_t M = A->rows();
	const size_t N = A->cols();

	const TensorPtr output = std::make_shared<Tensor>(M, N, A->requires_grad() || B->requires_grad());
	output->set_fn(std::make_shared<matadd_BW>(A, B, output, name));

	Scalar* output_data = output->data();
	const Scalar* A_data = A->data();
	const Scalar* B_data = B->data();

	for (size_t ij = 0; ij < M*N; ij++)
		output_data[ij] = A_data[ij] + B_data[ij];

	// RVO
	return output;
};


class mul_BW : public BW_Function {
private:
	Scalar value;
public:
	mul_BW(const TensorPtr A, const Scalar& b, const TensorPtr out, std::string name) {
		inputs_.push_back(A);
		value = b;
		outputs_.push_back(out);
		name_ = name;

		children_.push_back(A->get_fn());
	};

	void backward() override {
		if (inputs_[0]->requires_grad()) {
			Scalar* lgrad = inputs_[0]->gradient();
			const Scalar* out_grad = outputs_[0]->gradient();

			for (size_t ij = 0; ij < inputs_[0]->size(); ij++)
				lgrad[ij] += out_grad[ij] * value;
		}
	};
};

inline const TensorPtr mul(const TensorPtr A, const Scalar& b, std::string name) {
	const TensorPtr output = std::make_shared<Tensor>(A->rows(), A->cols(), A->requires_grad());
	output->set_fn(std::make_shared<mul_BW>(A, b, output, name));

	Scalar* output_data = output->data();
	const Scalar* A_data = A->data();

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

		children_.push_back(A->get_fn());
	};

	void backward() override {
		if (inputs_[0]->requires_grad()) {
			Scalar* grad = inputs_[0]->gradient();
			const Scalar* out_grad = outputs_[0]->gradient();
			const Scalar* data = inputs_[0]->data();

			for (size_t ij = 0; ij < inputs_[0]->size(); ij++)
				grad[ij] += (data[ij] >= 0 ? out_grad[ij] : 0);
		}
	};
};

inline const TensorPtr ReLU(const TensorPtr A, std::string name) {
	const TensorPtr output = std::make_shared<Tensor>(A->rows(), A->cols(), A->requires_grad());
	output->set_fn(std::make_shared<ReLU_BW>(A, output, name));

	Scalar* output_data = output->data();
	const Scalar* A_data = A->data();

	for (size_t ij = 0; ij < output->size(); ij++)
		output_data[ij] = std::max(A_data[ij], static_cast<Scalar>(0));

	// RVO
	return output;
};


class softmax_BW : public BW_Function {
public:
	softmax_BW(const TensorPtr A, const TensorPtr out, std::string name) {
		inputs_.push_back(A);
		outputs_.push_back(out);
		name_ = name;

		children_.push_back(A->get_fn());
	};

	void backward() override {
		if (inputs_[0]->requires_grad()) {
			Scalar* grad = inputs_[0]->gradient();
			const Scalar* out_grad = outputs_[0]->gradient();
			const Scalar* out_data = outputs_[0]->data();

			const size_t M = outputs_[0]->rows();
			const size_t N = outputs_[0]->cols();

			for (size_t i = 0; i < M; i++) {

				Scalar product = 0;
				for (size_t k = 0; k < M; k++)
					product += out_grad[i * M + k] * out_data[i * M + k];

				for (size_t j = 0; j < M; j++)
					grad[i * M + j] += out_data[i * M + j] * (out_grad[i * M + j] - product);
			}
		}
	};
};

inline const void softmax(Scalar* output, const Scalar* input, const int& nrows, const int& ncols) {

	std::vector<Scalar> tmp;
	tmp.resize(ncols);

	// for each row
	for (size_t i = 0; i < nrows; i++) {
		const Scalar* A_row = input + i * ncols;
		Scalar* output_row = output + i * ncols;

		// max
		Scalar max = A_row[0];
		for (size_t j = 0; j < ncols; j++) {
			Scalar value = A_row[j];
			if (value > max)
				max = value;
		}

		// sum of the exps
		Scalar sum = 0.0;
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
	const size_t M = A->rows();
	const size_t N = A->cols();

	const TensorPtr output = std::make_shared<Tensor>(M, N, A->requires_grad());
	output->set_fn(std::make_shared<softmax_BW>(A, output, name));

	Scalar* output_data = output->data();
	const Scalar* A_data = A->data();

	softmax(output_data, A_data, M, N);

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

		children_.push_back(input->get_fn());
		children_.push_back(target->get_fn());
	};

	void backward() override {
		if (inputs_[0]->requires_grad()) {
			Scalar* lgrad = inputs_[0]->gradient();
			const Scalar* ldata = inputs_[0]->data();
			const Scalar* rdata = inputs_[1]->data();

			for (size_t ij = 0; ij < inputs_[0]->size(); ij++)
				lgrad[ij] += (ldata[ij] - rdata[ij]) * inputs_[0]->size();
		}
		if (inputs_[1]->requires_grad()) {
			Scalar* rgrad = inputs_[1]->gradient();
			const Scalar* ldata = inputs_[0]->data();
			const Scalar* rdata = inputs_[1]->data();

			for (size_t ij = 0; ij < inputs_[0]->size(); ij++)
				rgrad[ij] += (rdata[ij] - ldata[ij]) * 2 / inputs_[0]->size();
		}
	};
};

inline const TensorPtr MSELoss(const TensorPtr input, const TensorPtr target, std::string name) {
	assert(input->rows() == target->rows());
	assert(input->cols() == target->cols());

	const Scalar size = target->size();

	const TensorPtr output = std::make_shared<Tensor>(1, 1, input->requires_grad() || target->requires_grad());
	output->set_fn(std::make_shared<loss_BW>(input, target, output, name));

	Scalar* output_data = output->data();
	const Scalar* input_data = input->data();
	const Scalar* target_data = target->data();

	for (size_t ij = 0; ij < size; ij++)
		output_data[0] += std::pow(input_data[ij] - target_data[ij], 2);

	// mean loss reduction.
	output_data[0] /= size;

	// RVO
	return output;
};


class CELoss_BW : public BW_Function {
private:
	std::vector<Scalar> temp_;
public:
	CELoss_BW(const TensorPtr input, const TensorPtr target_logits, const TensorPtr output, std::string name) {
		inputs_.push_back(input);
		inputs_.push_back(target_logits);
		outputs_.push_back(output);

		temp_.reserve(input->size());
		children_.push_back(input->get_fn());
		children_.push_back(target_logits->get_fn());
	};

	void backward() override {
		if (inputs_[0]->requires_grad()) {
			Scalar* lgrad = inputs_[0]->gradient();
			const Scalar* ldata = inputs_[0]->data();
			const Scalar* rdata = inputs_[1]->data();

			const size_t M = inputs_[0]->rows();
			const size_t N = inputs_[0]->cols();

			softmax(temp_.data(), ldata, M, N);

			// basically : grad = softmax(input) - hot_one(target_logits)
			for (size_t i = 0; i < M; i++)
				for (size_t j = 0; j < N; j++)
					lgrad[i * N + j] += (temp_[i * N + j] - (j == rdata[i] ? 1 : 0)) / M;
		}
		if (inputs_[1]->requires_grad())
			print("This is the target, having a gradient here is unusual.");
	};
};

inline const TensorPtr CrossEntropyLoss(const TensorPtr input, const TensorPtr target_logits, std::string name) {
	assert(input->rows() == target_logits->rows());
	assert(target_logits->size() == target_logits->rows());

	const size_t M = input->rows();
	const size_t N = input->cols();

	const TensorPtr output = std::make_shared<Tensor>(1, 1, input->requires_grad() || target_logits->requires_grad());
	output->set_fn(std::make_shared<CELoss_BW>(input, target_logits, output, name));

	Scalar* output_data = output->data();
	const Scalar* input_data = input->data();
	const Scalar* target_data = target_logits->data();

	// LogSoftmax : logSM(i, j) = x(i, j) - max(i) - log(sum(i)) with sum(i) the sum of exps
	Scalar* LogSoftmaxed_input = new Scalar[M * N];
	for (size_t i = 0; i < M; i++) {
		const Scalar* input_data_col = input_data + i * N;
		Scalar* logsfinput_col = LogSoftmaxed_input + i * N;
		Scalar max = 0, sum = 0;

		for (size_t j = 0; j < N; j++)
			if (max < input_data_col[j])
				max = input_data_col[j];

		for (size_t j = 0; j < N; j++)
			sum += std::exp(input_data_col[j] - max);

		for (size_t j = 0; j < N; j++)
			logsfinput_col[j] = input_data_col[j] - max - std::log(sum); // NLLLoss of LogSoftmax
	}

	// NLLLoss : L(x, class) = -x[class]
	for (size_t i = 0; i < M; i++)
		output_data[0] -= LogSoftmaxed_input[i * N + static_cast<int>(target_data[i])];
	delete[] LogSoftmaxed_input;

	// normalizing
	output_data[0] /= M;

	// RVO, no copy
	return output;
};


// To add : nn.CrossEntropyLoss (avec NLLoss ?), nn.Dropout, nn.Embedding, nn.LayerNorm, nn.Conv2D, nn.MaxPool2D

#endif // !FUNCTIONS