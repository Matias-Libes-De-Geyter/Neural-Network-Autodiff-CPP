#include "Tensor.hpp"

class Optimizer {
public:
	virtual void zero_grad() {};
	virtual void step(const Scalar& lr) {};
};

class SGD : public Optimizer {
private:
	std::vector<TensorPtr> parameters_;
public:
	SGD(std::vector<TensorPtr> parameters) : parameters_(parameters) {};

	inline void zero_grad() override {
		for (TensorPtr parameter : parameters_)
			parameter->zero_grad();
	};

	inline void step(const Scalar& lr) override {
		for (TensorPtr parameter : parameters_) {
			if (parameter->requires_grad()) {
				Scalar* params_data = parameter->data();
				Scalar* params_grad = parameter->gradient();
				for (int ij = 0; ij < parameter->size(); ij++)
					params_data[ij] -= lr * params_grad[ij];
			}
		}
	};
};