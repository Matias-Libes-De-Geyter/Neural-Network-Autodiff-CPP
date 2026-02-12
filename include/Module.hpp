#include "Tensor.hpp"
#include "Functions.hpp"
#include <random>

#ifndef MODULE
#define MODULE

std::random_device rd;
std::mt19937 gen(rd());

class Module {
public:
	virtual const TensorPtr forward(const TensorPtr input) { return input; };
	virtual const std::vector<TensorPtr> parameters() {};
	virtual const void zero_grad() {};
};

class Linear : public Module {
private:
	TensorPtr weight;
public:
	Linear() {};

	Linear(const size_t in_features, const size_t out_features) : weight(std::make_shared<Tensor>(in_features, out_features, true)) {
		const Scalar k = 1.0 / std::sqrt(in_features);
		std::uniform_real_distribution<Scalar> random_value(-k, k);

		Scalar* weight_value = weight->data();
		for (int ij = 0; ij < weight->size(); ij++)
			weight_value[ij] = random_value(gen);
	};

	const TensorPtr forward(const TensorPtr input) override {
		return matmul(input, weight, "forward");
	};

	const std::vector<TensorPtr> parameters() override { return { weight }; };
};

// disgusting, to improve
class FFNN : public Module {
private:
	std::vector<Linear> blocks;
	std::vector<int> dims_;
	int length_;

	TensorPtr next_hidden;
	TensorPtr next;
public:
	FFNN(std::vector<int> dims) : dims_(dims), length_(dims.size()) {
		for (int i = 0; i < dims_.size() - 1; i++)
			blocks.push_back(Linear(dims_[i], dims_[i + 1]));
	};
	const TensorPtr forward(const TensorPtr input) override {
		
		next = input;
		for (int i = 0; i < length_ - 2; i++) {
			next_hidden = blocks[i].forward(next);
			next = ReLU(next_hidden, "relu i");
		}
		next_hidden = blocks[length_ - 2].forward(next);

		return next_hidden; // logits
	};
	const std::vector<TensorPtr> parameters() override {
		std::vector<TensorPtr> params;
		for (int i = 0; i < dims_.size() - 1; i++)
			params.push_back(blocks[i].parameters()[0]);

		return params;
	};
};

class Simple_Resnet : public Module {
private:
	std::vector<FFNN> blocks;
	std::vector<std::vector<int>> dims_;
	int length_;

	TensorPtr next_hidden;
	TensorPtr next_hidden_relu;
	TensorPtr next;
public:
	Simple_Resnet(std::vector<std::vector<int>> dims) : dims_(dims), length_(dims.size()) {

		for (int i = 0; i < length_; i++)
			blocks.push_back(FFNN(dims_[i]));
	};
	const TensorPtr forward(const TensorPtr input) override {

		next = input;
		for (int i = 0; i < length_ - 1; i++) {
			next_hidden = blocks[i].forward(next);
			next_hidden_relu = ReLU(next_hidden, "relu");
			next = matadd(next, next_hidden_relu, "matadd");
		}
		next_hidden = blocks[length_ - 1].forward(next);

		return next_hidden; // logits
	};
	const std::vector<TensorPtr> parameters() override {
		std::vector<TensorPtr> params = blocks[0].parameters();

		for (int i = 1; i < length_; i++) {
			std::vector<TensorPtr> _params = blocks[i].parameters();
			params.insert(params.end(), _params.begin(), _params.end());
		}

		return params;
	};
};

#endif // !MODULE