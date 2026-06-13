# Assignment: Linear Regression using PyTorch

## 1. Objective
Implement a simple linear regression model in PyTorch from the ground up. This assignment focuses on the core mathematical and programmatic mechanics of machine learning—including data generation, forward propagation, loss calculation, backpropagation, and gradient descent—without relying on high-level APIs like `torch.nn` or `torch.optim`.

---

## 2. Problem Statement
The objective is to find the underlying mathematical relationship between an input tensor $X$ and an output tensor $Y$. 

### The Target Function
Synthetic data must be generated based on the following linear equation:

$$Y = 3X + 8 + \text{noise}$$

The model must iteratively update its parameters to learn the true weight ($w = 3$) and bias ($b = 8$) by minimizing the prediction error.

---

## 3. Implementation Requirements
To successfully complete this task, the codebase must implement the following four pillars:

* **Synthetic Data Generation:** Create continuous data using `torch.rand` and introduce realistic variance using `torch.randn`.
* **Autograd & Parameter Tracking:** Manually initialize `weight` and `bias` tensors, explicitly setting `requires_grad=True` to enable PyTorch's gradient tracking.
* **The Custom Training Loop:**
    * *Forward Pass:* Calculate predictions using the linear formula ($y\_pred = wX + b$).
    * *Loss Function:* Manually code the Mean Squared Error (MSE) formula.
    * *Backpropagation:* Trigger gradient computation using `loss.backward()`.
    * *Optimization:* Update the weights using vanilla Gradient Descent and manually zero out the gradients after each iteration.
* **Visualization:** Use `matplotlib` to plot the final learned regression line against the original dataset to visually verify model accuracy.

---

## 4. Setup & Environment

### Prerequisites
Ensure the local environment has Python 3.x and the required libraries installed:

```bash
pip install torch matplotlib