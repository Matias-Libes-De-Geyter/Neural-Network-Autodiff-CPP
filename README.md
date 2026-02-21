# Neural-Network-Autodiff-CPP
This program aims at integrating a lightweight reverse-mode automatic differentiation (AD) for Neural Networks. It's also intended to be working when implementing complex residual networks such as Resnet, or Transformer.

# Commit from 06-02-2026 | Processing time with the same 'main.exe' with MNIST.
Initial: 8.74s - 8.78s - 8.8s \\
Children filled in constructor rather than in the BW_Function::children() method: 7.87s - 8.1s - 7.99s \\
Row first in matmul & bw_matmul: 8.2s - 8.35s - 8.39s \\
Row first + optimised pointers only matmul: 6.96s - 6.55s - 6.71s \\
Row first + optimised pointers in matmul & bw_matmul: 5.07s - 5.57s - 6.1s \\
Getting the number of rows/cols at the beginning of the functions: 1.32s - 1.32s - 1.31s

# Commit #1/#2 from 12-02-2026 | Making backward pass work for a linear feed forward neural network.
The backward pass didn't work, since on of the variables (i) in the matmul_bw went across the wrong dimension. \\
After having results with a MLP, I tried to implement a residual network with only matmul, add and ReLU nodes. Good results : loss of 0.16 in 10 epochs (with three MatNet layers : (ReLU(Linear(x))^3)), compared to 0.34 with only a Linear(x). \\
Nonetheless I have to quickly improve the parameters() functions, along with the backward topo order calculation, if I can do it only once or smth...?

# Commit #3 from 12-02-2026 | Please Github make me commit with my actual github account

# Commit #1 from 19-02-2026 | From Matrices to Tensors
I replaces the row/col sizes from Tensor to std::vector\<size_t\> for real handling (useful for CNNs later on). \\
Only implemented matmul, matadd and linear functions for now. Next will be the other functions, including CELoss and MSELoss.

# Commit #1 from 21-02-2026 | Fully Tensorised the code
Replaced row/col usage by dims, for all implemented functions. Normal training for main works fine