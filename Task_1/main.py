import torch 
import matplotlib.pyplot as plt

X = torch.rand(100,1) * 10
noise = torch.randn(100,1) * 0.2

Y= 3 * X + 8 + noise


weight = torch.randn(1, requires_grad=True)
bias = torch.randn(1, requires_grad=True)

print(f"Initial weight: {weight.item():.4f}")
print(f"Initial bias: {bias.item():.4f}")

learning_rate = 0.01

for epoch in range(1000):

    # Forward pass
    predictions = X * weight + bias

    # Loss Function
    loss = torch.mean((predictions - Y) ** 2)

    # Backpropagation
    loss.backward()

    # Optimizer
    with torch.no_grad():
        weight -= learning_rate * weight.grad
        bias -= learning_rate * bias.grad

        weight.grad.zero_()
        bias.grad.zero_()

    if (epoch + 1) % 100 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/1000 | Loss (Error): {loss.item():.4f} | w: {weight.item():.4f}, b: {bias.item():.4f}")

print("\n--- Training Finished! ---")
print(f"Final learned weight (w): {weight.item():.4f} (Goal: 3.0)")
print(f"Final learned bias (b): {bias.item():.4f} (Goal: 8.0)")


X_numpy = X.numpy()
Y_numpy = Y.numpy()
predictions_numpy = predictions.detach().numpy()

plt.scatter(X_numpy, Y_numpy, color='red', label='Real data', alpha=0.5)

plt.plot(X_numpy, predictions_numpy, color='blue', label='Learned line', linewidth=3)

plt.xlabel('Input (X)')
plt.ylabel('Output (Y)')
plt.legend()
plt.title('Results')
plt.show()