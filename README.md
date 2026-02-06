# Neural-Network-Autodiff-CPP
This program aims at integrating a lightweight reverse-mode automatic differentiation (AD) for Neural Networks. It's also intended to be working when implementing complex residual networks such as Resnet, or Transformer.

# Processing time with the same 'main.exe' with MNIST.
Initial: 8.74s - 8.78s - 8.8s
Children filled in constructor rather than in the BW_Function::children() method: 7.87s - 8.1s - 7.99s
Row first in matmul & bw_matmul: 8.2s - 8.35s - 8.39s
Row first + optimised pointers only matmul: 6.96s - 6.55s - 6.71s
Row first + optimised pointers in matmul & bw_matmul: 5.07s - 5.57s - 6.1s
Getting the number of rows/cols at the beginning of the functions: 1.32s - 1.32s - 1.31s