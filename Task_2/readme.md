# Assignment: Neural Network for MNIST Digit Classification

## 1. Objective
Implement a simple Feedforward Neural Network in PyTorch to classify handwritten digits. This assignment leverages PyTorch's high-level APIs like `torch.nn`, `torch.optim`, and `torchvision` modules to streamline data loading, network definition, and training.

---

## 2. Problem Statement
The objective is to accurately identify $28 \times 28$ pixel grayscale images of handwritten digits (0-9) from the popular MNIST dataset. 

### The Network Structure
The input is a flattened 784-dimensional vector (28*28). The model maps this input to a 10-class probability distribution using a hidden layer, capturing essential non-linear patterns.

---

## 3. Implementation Requirements
To successfully complete this task, the codebase uses the following key components:

* **Data Preparation:** Use `torchvision.datasets.MNIST` to download and load the training and test datasets. Apply `torchvision.transforms` to convert the images to PyTorch Tensors and normalize them with a mean and standard deviation of `0.5`. Organize the data into batches using `DataLoader`.
* **Neural Network Architecture:**
    * Create a `SimpleNN` class inheriting from `nn.Module`.
    * Include an `nn.Linear` hidden layer with 128 units, followed by a ReLU (`nn.ReLU()`) activation function.
    * Use an `nn.Linear` output layer mapping the 128 features to 10 output classes.
* **The Custom Training Loop:**
    * *Loss Function:* Use Cross-Entropy Loss (`nn.CrossEntropyLoss`).
    * *Optimizer:* Use the Adam optimizer (`optim.Adam`) with a learning rate of `0.003`.
    * Train for 3 epochs, running forward passes, calculating loss, triggering `loss.backward()`, and stepping the optimizer.
* **Evaluation:** After training, put the model in evaluation mode (`model.eval()`) and disable gradients (`torch.no_grad()`). Run the test dataset through the model and calculate the percentage accuracy.

---

## 4. Setup & Environment

### Prerequisites
Ensure the local environment has Python 3.x and the required libraries installed:

```bash
pip install torch torchvision
```
