// X must have Function object which all ops inherits from. Forward method with grad_fn allocation
// X "autograd keeps a record of data (tensors) & all executed operations (along with the resulting new tensors)
//  |_ in a directed acyclic graph (DAG) consisting of Function objects"
#include "include/Tensor.hpp"
#include "include/Functions.hpp"
#include "include/Module.hpp"
#include "include/Utilities.hpp"
#include "include/Optimizer.hpp"
#include "include/DataLoader.hpp"

const int batch_number = 100;
const int batch_size = 32;
const int epochs = 10;
const int image_size = 28 * 28;
const int label_number = 10;

#include <chrono>

int main() {
	std::vector<Scalar> A_data(210);
	std::vector<Scalar> B_data(210);
	std::vector<Scalar> C_data(210);

	for (int i = 0; i < 210; ++i) {
		A_data[i] = static_cast<Scalar>(i + 1);
		B_data[i] = static_cast<Scalar>(-i);
		C_data[i] = static_cast<Scalar>(-i);
	}

	// Let's not focus on broadcastable dimension but identical dimension
	TensorPtr A = std::make_shared<Tensor>(std::vector<size_t>{7, 5, 2, 3}, A_data, "A", true);
	TensorPtr B = std::make_shared<Tensor>(std::vector<size_t>{7, 5, 2, 3}, B_data, "B", true);
	TensorPtr C = std::make_shared<Tensor>(std::vector<size_t>{7, 5, 3, 2}, C_data, "C", true);

	TensorPtr X = matadd(A, B, "X");
	TensorPtr Y = matmul(A, C, "Y");
	print(*X);
	print(*Y);


}

/*int second_main() {
	auto start = std::chrono::high_resolution_clock::now();

	Dataset train = DataLoader(batch_number, batch_size, "train");

	FFNN model({ image_size, 256, 128, label_number });

	SGD optim(model.parameters());

	for (int epoch = 0; epoch < epochs; epoch++) {
		Scalar total_loss = 0;
		for (int batch = 0; batch < batch_number; batch++) {
			TensorPtr logits = model.forward(train.x[batch]);

			TensorPtr loss = CrossEntropyLoss(logits, train.y[batch], "celoss");

			total_loss += loss->data()[0];

			optim.zero_grad();

			loss->backward();

			optim.step(0.05);
		}
		print(total_loss / batch_number);
	}
	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	print(duration / 1000.f);

}*/