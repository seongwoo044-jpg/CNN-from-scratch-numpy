# CNN from Scratch with NumPy

This is a Convolutional Neural Network (CNN) project implemented from the ground up using only Python and NumPy, without any frameworks such as PyTorch or TensorFlow.

## Key Features & Implementation Details

* **im2col/col2im base**
  * I improved the speed of Convolution and Pooling operations by converting $N \times C \times H \times W$ tensors into matrices.
* **Backpropagation & Chain Rule calculation**
  * I calculated the differentiation of the backpropagation formulas for each layer. (Conv2D, ReLU, MaxPool2D, FC, SoftmaxWithLoss)
* **He Normal Initialization**
  * I applied He Normal Initialization($\sqrt{2/n}$) that optimized at ReLU function.

## Stack
* Python 3
* NumPy
* Scikit-Learn (MNIST data sets)

## Results
* **Dataset:** MNIST Hand-written Digits (28x28)
* **Optimization:** Mini-batch SGD (Batch size: 100, Iterations: 1,000)
* **Performance:** 88.4% accuracy about 1000 test datas after 1,000 iterations of training
