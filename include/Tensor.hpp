#include <iostream>
#include <memory> // for shared_ptr and make_shared
#include <cassert> // for assert()
#include <vector> // for  vectors
#include <unordered_map> // for DAG

#ifndef TENSOR
#define TENSOR

#pragma GCC diagnostic ignored "-Wreturn-type"

class Tensor;
class BW_Function;
using TensorPtr = std::shared_ptr<Tensor>;
using FunctionPtr = std::shared_ptr<BW_Function>;

using Scalar = float;

class BW_Function {
protected:
    std::vector<FunctionPtr> children_;
    std::vector<TensorPtr> inputs_;
    std::vector<TensorPtr> outputs_;
    std::string name_;
public:
    virtual std::string getname() { return name_; };
    virtual void setname(std::string name) { name_ = name; };

    virtual const std::vector<FunctionPtr> children();
    virtual void backward() {};
};

class Tensor {
private:
    size_t nrows_, ncols_;
    size_t size_;
    std::vector<Scalar> data_;
    std::vector<Scalar> gradient_;
    FunctionPtr grad_fn_;
    bool requires_grad_;
public:
    // constructor
    Tensor() {}
    Tensor(size_t row, size_t col, bool requires_grad) : nrows_(row), ncols_(col), size_(row* col), requires_grad_(requires_grad) {
        grad_fn_ = std::make_shared<BW_Function>(); // added
        grad_fn_->setname("no name"); // added
        size_ = nrows_ * ncols_;
        data_.resize(size_, 0.0);
        gradient_.resize(size_, 0.0);
    }
    Tensor(size_t row, size_t col, std::vector<Scalar> data, std::string name, bool requires_grad) : nrows_(row), ncols_(col), size_(row* col), requires_grad_(requires_grad), data_(data) {
        assert(data.size() == size_);

        grad_fn_ = std::make_shared<BW_Function>(); // added
        grad_fn_->setname(name); // added
        gradient_.resize(size_, 0.0);
    } // zero_grad() could take care of grad, but initialized at 0 for the input which is not within any model->zero_grad() scope.

    void zero_grad() {
        std::fill(gradient_.begin(), gradient_.end(), 0.0);
    }
    void ones_grad() {
        std::fill(gradient_.begin(), gradient_.end(), 1.0);
    }

    // retrieving data
    const size_t size() const { return size_; }
    const size_t rows() const { return nrows_; }
    const size_t cols() const { return ncols_; }
    Scalar* data() { return data_.data(); }
    const Scalar* data() const { return data_.data(); }
    Scalar* gradient() { return gradient_.data(); }
    const Scalar* gradient() const { return gradient_.data(); }
    void set_fn(FunctionPtr fn) { grad_fn_ = fn; }
    const FunctionPtr get_fn() const { return grad_fn_; }
    const bool requires_grad() const { return requires_grad_; }


    const void backward() const {
        std::vector<FunctionPtr> stack{ grad_fn_ };
        std::vector<FunctionPtr> nodes{ grad_fn_ };
        std::unordered_map<FunctionPtr, int> nb_parents;

        // generating the graph via DFS
        while (!stack.empty()) {
            std::vector<FunctionPtr> current_fn_children = stack.back()->children(); // limits the calls of children() without storing it to an intermediate variable
            stack.pop_back();

            for (FunctionPtr child : current_fn_children) {
                if (nb_parents[child] == 0) { // A SURVEILLER. PTET FAIRE DFS ET COMPTAGE DES PARENTS EN DEUX PARTIES.
                    stack.push_back(child);
                    nodes.push_back(child);
                }
                nb_parents[child]++;
            }
        }

        /*for (FunctionPtr node : nodes) {
            std::cout << node->getname() << " : " << nb_parents[node] << std::endl;
        } std::cout << std::endl << std::endl;*/

        std::unordered_map<FunctionPtr, int> nb_parents_bis = nb_parents;

        // filling the operation heap via Kahn's Algorithm
        std::vector<FunctionPtr> topo_ordered_nodes;
        stack.push_back(grad_fn_); // the only node without parent is the output here. Might want to watch this
        while (!stack.empty()) {
            FunctionPtr current_node = stack.back();
            stack.pop_back();

            topo_ordered_nodes.push_back(current_node);

            for (FunctionPtr child : current_node->children())
                if (--nb_parents[child] == 0)
                    stack.push_back(child);
        }

        /*for (FunctionPtr node : topo_ordered_nodes) {
            std::cout << node->getname() << " : " << nb_parents_bis[node] << std::endl;
        }*/

        // now that we found the correct topological reverse order of the graph, we call all the backward functions in the correct order
        for (FunctionPtr node : topo_ordered_nodes) {
            //std::cout << node->getname();
            node->backward();
        }
    }
};

// for later: here I copy the children / I should in constructors append children directly
inline const std::vector<FunctionPtr> BW_Function::children() {
    return children_;
};

#endif // !TENSOR