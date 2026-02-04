#include "Tensor.hpp"

class Optimizer {
public:
	virtual void zero_grad() {};
	virtual void step(const double& lr) {};
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

	inline void step(const double& lr) override {
		for (TensorPtr parameter : parameters_) {
			if (parameter->requires_grad()) {
				double* params_data = parameter->data();
				double* params_grad = parameter->gradient();
				for (int ij = 0; ij < parameter->size(); ij++)
					params_data[ij] -= lr * params_grad[ij];
			}
		}
	};
};