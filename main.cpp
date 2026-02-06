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
	auto start = std::chrono::high_resolution_clock::now();

	Dataset train = DataLoader(batch_number, batch_size, "train");

	FFNN mmodel({ image_size, 256, 128, label_number });
	Linear model(image_size, label_number);

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
	print(duration/1000.f);

}