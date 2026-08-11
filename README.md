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

 ## Mathematical Derivation (Backpropagation)

This project implemented backpropagation by deriving partial derivative formulas based on the Chain Rule.

* **'Conv2D' Layer Gradient:**

* **'FC' Layer Gradient:**

* **`SoftmaxWithLoss` Loss & Gradient:**
     $$
     y_i = \text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}
     $$
     
     $$
     \frac{\partial y_i}{\partial z_j} = \begin{cases} y_i (1 - y_i) & \text{if } i = j \\ -y_i y_j & \text{if } i \neq j \end{cases}
     $$
     
     $$
     L = -\sum_{i} t_i \log(y_i)
     $$
     
     $$
     \frac{\partial L}{\partial z_i} = -\sum_{j} \frac{t_j}{y_j} \frac{\partial y_j}{\partial z_i} = y_i - t_i
     $$

## Stack
* Python 3
* NumPy
* Scikit-Learn (`fetch_openml` for loading MNIST)

## Results
* **Dataset:** MNIST Hand-written Digits ($1 \times 28 \times 28$)
* **Optimization:** Mini-batch SGD (Batch size: 100, Iterations: 1,000)
* **Performance:** Acheived **90.50% accuracy** on 1000 test samples after 1,000 iterations of training
