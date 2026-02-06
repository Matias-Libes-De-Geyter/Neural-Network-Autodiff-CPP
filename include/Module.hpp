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

// disgusting, to improve with one hidden tensor and more dynamic approach
class FFNN : public Module {
private:
	std::vector<Linear> blocks;
	std::vector<int> dims_;

	TensorPtr h_1;
	TensorPtr h_2;
	TensorPtr x_1;
	TensorPtr x_2;
	TensorPtr x_3;
public:
	FFNN(std::vector<int> dims) : dims_(dims) {
		for (int i = 0; i < dims_.size() - 1; i++)
			blocks.push_back(Linear(dims_[i], dims_[i + 1]));
	};
	const TensorPtr forward(const TensorPtr input) override {

		x_1 = blocks[0].forward(input);
		h_1 = ReLU(x_1, "nom 1");
		x_2 = blocks[1].forward(h_1);
		h_2 = ReLU(x_2, "nom 2");
		x_3 = blocks[2].forward(h_2);

		return x_3; // logits
	};
	const std::vector<TensorPtr> parameters() override {
		std::vector<TensorPtr> params;
		for (int i = 0; i < dims_.size() - 1; i++)
			params.push_back(blocks[i].parameters()[0]);

		return params;
	};
};

#endif // !MODULE