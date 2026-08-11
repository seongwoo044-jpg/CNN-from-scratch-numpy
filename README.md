# CNN from Scratch with NumPy

This project is a Convolutional Neural Network (CNN) implemented from the ground up using only Python and NumPy, without any frameworks such as PyTorch or TensorFlow.

## Key Features & Implementation Details

* **im2col/col2im base**
  * Transformed high-dimensional tensors ($N \times C \times H \times W$) into 2D matrices to accelerate Convolution and Spatial Pooling operations via optimized matrix multiplication (`np.dot`).
* **Backpropagation & Chain Rule calculation**
  * Derived and implemented explicit `Forward` and `Backward` passes for all custom layers: `Conv2D`, `ReLU`, `MaxPool2D`, `FC`, and `SoftmaxWithLoss`.
* **He Normal Initialization**
  * Applied He Normal Initialization($\sqrt{2/n}$) customized for ReLU activation functions
* **Dynamic Network Architecture (`_build_network`)**
  * Programmed an adaptive architecture setup that dynamically constructs pooling, multi-stage convolutional layers, and linear classifier dimensions based on input tensor shapes.

## Key Mathematical Derivation (Backpropagation)

This project implemented backpropagation by deriving partial derivative formulas based on the Chain Rule.

* **`Conv2D` Layer Gradient:**

  4D Tensor Index Form:

  $$\frac{\partial L}{\partial W_{out\_c, in\_c, h, w}} = \sum_{N} \sum_{out\_h} \sum_{out\_w} X_{n, in\_c, h + out\_h, w + out\_w} \cdot \frac{\partial L}{\partial Z_{n, out\_c, out\_h, out\_w}}$$
  
  $$\frac{\partial L}{\partial X_{n, in\_c, h + out\_h, w + out\_w}} = \sum_{out\_c} W_{out\_c, in\_c, h, w} \cdot \frac{\partial L}{\partial Z_{n, out\_c, out\_h, out\_w}}$$

  Using `im2col` to $X$ & Reshaping $\delta$ (= $\frac{\partial L}{\partial Z}$) and $W$:
  * $X \rightarrow X_{\text{col}}$
  * $\delta \rightarrow \delta_{\text{col}}$
  * $T \rightarrow T_{\text{col}}$

  $$\frac{\partial L}{\partial W} = \delta_{\text{col}}^T \cdot X_{\text{col}} \quad \xrightarrow{\text{reshape}} \quad (\text{Shape: } out\_c, in\_c, h, w)$$

  $$\frac{\partial L}{\partial X} = W_{\text{col}} \cdot \delta_{\text{col}} \quad \xrightarrow{\text{col2im}} \quad (\text{Shape: } N, in\_c, H, W)$$


* **`SoftmaxWithLoss` Loss & Gradient:**

$$y_i = \text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}$$

$$\frac{\partial y_i}{\partial z_j} = \begin{cases} y_i (1 - y_i) & \text{if } i = j \\\\ -y_i y_j & \text{if } i \neq j \end{cases}$$

$$L = -\sum_{i} t_i log (y_i)$$

$$\frac{\partial L}{\partial z_i} = -\sum_{j} \frac{t_j}{y_j} \frac{\partial y_j}{\partial z_i} = y_i - t_i$$


## Stack
* Python 3
* NumPy
* Scikit-Learn (`fetch_openml` for loading MNIST)


## Results
* **Dataset:** MNIST Hand-written Digits ($1 \times 28 \times 28$)
* **Optimization:** Mini-batch SGD (Batch size: 100, Iterations: 1,000)
* **Performance:** Acheived **90.50% accuracy** on 1000 test samples after 1,000 iterations of training


## What I Learned
* Implementing backpropagation helped me understand neural network training as a chain of partial derivatives and the chain rule.
* Implementing `im2col` and `col2im` was more challenging than the mathematical derivation because of the required tensor reshaping and indexing.
* I initially considered implementing convolution with nested loops, but switched to `im2col` to take advantage of NumPy's vectorized matrix operations.
* Through this implementation, I was able to understand the overall flow of CNN training, from input data and convolution to loss calculation and parameter updates.
* The main goal of this project was not to maximize accuracy, but to understand how a CNN works internally by implementing its core components from scratch.
