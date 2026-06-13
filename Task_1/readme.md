# Task 1: Linear Regression from Scratch using PyTorch

## Overview
This project implements a simple linear regression model in PyTorch from the ground up. It demonstrates the fundamental concepts of machine learning, including data generation, forward propagation, loss calculation, backpropagation, and gradient descent, without relying on high-level APIs like `torch.nn` or `torch.optim`.

## The Task
The goal of this task is to find the mathematical relationship between input `X` and output `Y`. 
We generate synthetic data based on the equation:
$$Y = 3X + 8 + \text{noise}$$

The model aims to learn the weight ($w = 3$) and bias ($b = 8$) by iteratively updating parameters based on the calculated error.

## Key Concepts Demonstrated
1. **Synthetic Data Generation:** Creating continuous data using `torch.rand` and adding realistic noise using `torch.randn`.
2. **Autograd:** Manually initializing `weight` and `bias` tensors with `requires_grad=True` to let PyTorch automatically track operations.
3. **Training Loop:**
   - **Forward Pass:** Calculating predictions ($y = wX + b$).
   - **Loss Function:** Implementing Mean Squared Error (MSE) manually.
   - **Backpropagation:** Computing the gradients using `loss.backward()`.
   - **Optimizer:** Manually updating weights using Gradient Descent and zeroing the gradients.
4. **Visualization:** Plotting the final learned regression line over the generated data using `matplotlib`.

## Prerequisites
To run this project, make sure you have the following installed:
- Python 3.x
- PyTorch
- Matplotlib

You can install the dependencies using pip:
```bash
pip install torch matplotlib
```

How to run
```bash
python main.py
```
