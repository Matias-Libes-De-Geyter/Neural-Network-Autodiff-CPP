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
const int epochs = 5;
const int image_size = 28 * 28;
const int label_number = 10;

int main() {

	Dataset train = DataLoader(batch_number, batch_size, "train");

	FFNN mmodel({ image_size, 256, 128, label_number });
	Linear model(image_size, label_number);

	SGD optim(model.parameters());

							// nécessairement ncols output = C (nb_classes) for logits in CELoss
	/*TensorPtr A = std::make_shared<Tensor>(5, 3, std::vector<double>{ 0.1, 0.5, 0.1,
																	  0.3, 0.9, 0.6,
																	  0.1, 0.7, 0.3,
																	  0.8, 0.1, 0.9,
																	  0.1, 0.5, 0.7 }, "A", true);
	TensorPtr Y = std::make_shared<Tensor>(5, 1, std::vector<double>{ 0, 2, 1, 2, 2 }, "Y", false);

	TensorPtr CELoss = CrossEntropyLoss(A, Y, "cel");
	print(*CELoss);
	CELoss->backward();
	printgrad(*A);*/

	for (int epoch = 0; epoch < epochs; epoch++) {
		double total_loss = 0;
		for (int batch = 0; batch < batch_number; batch++) {
			TensorPtr logits = model.forward(train.x[batch]);
			TensorPtr loss = CrossEntropyLoss(logits, train.y[batch], "celoss");

			total_loss += loss->data()[0];

			optim.zero_grad();
			loss->backward();

			optim.step(0.01);
		}
		print(total_loss / batch_number);
	}
}

/*int backward_with_residual_test() {
	TensorPtr X = std::make_shared<Tensor>(2, 3, std::vector<double>{ 1, 2, 3,
																	  4, 5, 6 }, "X", true);
	TensorPtr W = std::make_shared<Tensor>(3, 3, std::vector<double>{ 10, 20,  10,
																	  10, 20,  50,
																	  30, 100, 40 }, "W", true);

	TensorPtr A = ReLU(X, "A"); // 2x3 = 2x3
	print("A: ", *A);

	TensorPtr B = matmul(A, W, "B"); // 2x3 * 3x3 = 2x3
	print("B: ", *B);

	TensorPtr C = ReLU(B, "C"); // 2x3 = 2x3
	print("C: ", *C);

	TensorPtr D = matadd(A, C, "D"); // 2x3 + 2x3 = 2x3
	print("D: ", *D);

	TensorPtr output = ReLU(D, "output"); // 2x3 = 2x3
	print("output: ", *output);

	print("\n\n");
	
	output->ones_grad();
	printgrad(*output);
	output->backward();

	print("dD, dC: ");
	printgrad(*D);
	printgrad(*C);

	print("dB: ");
	printgrad(*B);

	print("dA, dW: ");
	printgrad(*A);
	printgrad(*W);

	print("dX: ");
	printgrad(*X);

	// --> Good result: dX = ( (41, 81, 171) , (41, 81, 171) )

}

int backward_dfs_kahn_test() {
	TensorPtr A = std::make_shared<Tensor>(2, 3, std::vector<double>{ 1, 2, 3, 4, 5, 6 }, "A", true);
	TensorPtr B = std::make_shared<Tensor>(3, 4, std::vector<double>{ 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0 }, "B", true);
	TensorPtr C = std::make_shared<Tensor>(4, 2, std::vector<double>{ 1, 2, 3, 4, 5, 6, 7, 8 }, "C", true);
	TensorPtr D = std::make_shared<Tensor>(2, 3, std::vector<double>{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 }, "D", true);

	TensorPtr E = matmul(A, B, "E");          // 2x3 * 3x4 = 2x4
	print("E: ", *E);

	TensorPtr F = matmul(E, C, "F");          // 2x4 * 4x2 = 2x2
	print("F: ", *F);

	TensorPtr G = matmul(D, B, "G");          // 2x3 * 3x4 = 2x4
	print("G: ", *G);

	TensorPtr H = matmul(F, D, "H");          // 2x2 * 2x3 = 2x3
	print("H: ", *H);

	TensorPtr I = matmul(G, C, "I");          // 2x4 * 4x2 = 2x2
	print("I: ", *I);

	TensorPtr J = matmul(H, B, "J");          // 2x3 * 3x4 = 2x4
	print("J: ", *J);

	TensorPtr K = matmul(I, H, "K");          // 2x2 * 2x3 = 2x3
	print("K: ", *K);

	TensorPtr L = matmul(J, C, "L");          // 2x4 * 4x2 = 2x2
	print("K: ", *L);

	TensorPtr output = matmul(L, K, "output"); // 2x2 * 2x3 = 2x3
	print("\nFinal output: ", *output);

	print("\nWe verified output has the two correct children. Let's try backward: \n");

	output->backward();

};

struct vertex {
public:
	std::string index;
	std::vector<vertex*> children;
	vertex(std::string index_, std::vector<vertex*> children_) : index(index_), children(children_) {};
	bool operator==(const vertex& other) {
		return index == other.index;
	}
};

int dfs_kahn_test() {

	vertex a("a", std::vector<vertex*>{});
	vertex b("b", std::vector<vertex*>{});
	vertex c("c", std::vector<vertex*>{});

	// Second level
	vertex d("d", std::vector<vertex*>{ &a, &b });
	vertex e("e", std::vector<vertex*>{ &a });
	vertex f("f", std::vector<vertex*>{ &c });

	// Third level
	vertex g("g", std::vector<vertex*>{ &d, &e });
	vertex h("h", std::vector<vertex*>{ &b, &f });
	vertex i("i", std::vector<vertex*>{ &e });
	vertex i_prime("i_prime", std::vector<vertex*>{ &i });

	// Fourth level
	vertex j("j", std::vector<vertex*>{ &g, &h });
	vertex k("k", std::vector<vertex*>{ &i, &f });
	vertex l("l", std::vector<vertex*>{ &d });

	// Fifth level
	vertex m("m", std::vector<vertex*>{ &j, &k, &l });
	vertex n("n", std::vector<vertex*>{ &k, &h });

	// Sixth level (final)
	vertex o("o", std::vector<vertex*>{ &m, &n });

	std::vector<vertex*> stack{ &o };
	std::vector<vertex*> V{ &o };
	std::unordered_map<std::string, int> nbParents{ {"o", 0} };
	while (!stack.empty()) {
		vertex* x = stack.back(); stack.pop_back();

		for (vertex* child : x->children) {
			if (nbParents[child->index] == 0) { // A SURVEILLER
				stack.push_back(child);
				V.push_back(child);
			}
			nbParents[child->index]++; // initialisation to 0 by operator[] ?
		}
	}
	for (vertex* element : V) {
		std::cout << element->index << ": " << nbParents[element->index] << " | ";
	} std::cout << std::endl;

	std::vector<vertex*> S;
	std::vector<vertex*> L;

	// Pas besoin car le premier élément de S est nécessairement que le dernier vertex.
	for (vertex* v : V)
		if (nbParents[v->index] == 0)
			S.push_back(v);

	while (!S.empty()) {
		vertex* n = S.back();
		S.pop_back();
		L.push_back(n);

		for (vertex* children : n->children)
			if (--nbParents[children->index] == 0)
				S.push_back(children);
	}

	assert(L.size() == V.size());
	for (vertex* element : L) {
		std::cout << element->index << ", ";
	} std::cout << std::endl;

};


int matmul_linear_test() {
	Tensor A(2, 3, { 0.1, 0.2, 0.3, 0.9, 0.8, 0.7 }, true);
	Tensor B(3, 2, { 1, 2, 5, 6, 9, 10 }, true);

	print("A: ", A);
	print("");
	print("B: ", B);
	print("");

	A.zero_grad();
	B.zero_grad();

	//		VERY CAREFUL: MAKE_SHARED ACTUALLY CREATES A FULL COPY OF A

	print("");
	print("========");
	print("");

	TensorPtr V = std::make_shared<Tensor>(2, 3, std::vector<double>{ 0.1, 0.2, 0.3, 0.8, 0.8, 0.5 }, true);
	Linear model(3, 6);
	print(*model.parameters()[0]);
	model.zero_grad();
	V->zero_grad();

	TensorPtr sortie = model.forward(V);
	print(*sortie);
	sortie->ones_grad();

	print("before backwards");
	printgrad(*model.parameters()[0]);
	printgrad(*V);
	printgrad(*sortie);

	sortie->backward();
	print("after backwards");
	printgrad(*model.parameters()[0]);
	printgrad(*V);
	printgrad(*sortie);

	// Linear backwards works

	print("");
	print("========");
	print("");

	TensorPtr input_dense = std::make_shared<Tensor>(3, 2, std::vector<double>{ 0.1, 0.2, 0.3, 0.8, 0.8, 0.5 }, true);
	DenseBlock dmodel(2, 1);
	dmodel.zero_grad();
	input_dense->zero_grad();

	TensorPtr doutput = dmodel.forward(input_dense);
	doutput->ones_grad();

	print(*doutput);
	print(*dmodel.parameters()[0]);

	print("before backwards");
	printgrad(*dmodel.parameters()[0]);
	printgrad(*input_dense);
	printgrad(*doutput);

	doutput->backward();
	print("after backwards");
	printgrad(*dmodel.parameters()[0]);
	printgrad(*input_dense);
	printgrad(*doutput);

	// DenseBlock backwards works (if backward is called in ReLU)
};*/