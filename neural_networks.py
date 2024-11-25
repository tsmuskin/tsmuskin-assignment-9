import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr # learning rate
        self.activation_fn = activation # activation function

        # TODO: define layers and initialize weights
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b2 = np.zeros((1, output_dim))


    def forward(self, X):
        # TODO: forward pass, apply layers to input X
        # TODO: store activations for visualization
        self.Z1 = np.dot(X, self.W1) + self.b1
        if self.activation_fn == "tanh":
            self.A1 = np.tanh(self.Z1)
        elif self.activation_fn == "relu":
            self.A1 = np.maximum(0, self.Z1)
        elif self.activation_fn == "sigmoid":
            self.A1 = 1 / (1 + np.exp(-self.Z1))
        
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = 1 / (1 + np.exp(-self.Z2))  # Sigmoid for binary output
        return self.A2

    def backward(self, X, y):
        # TODO: compute gradients using chain rule

        # TODO: update weights with gradient descent

        # TODO: store gradients for visualization
        m = X.shape[0]  # Number of samples

        # Output layer gradients
        dZ2 = self.A2 - y
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        # Hidden layer gradients
        if self.activation_fn == "tanh":
            dA1 = np.dot(dZ2, self.W2.T) * (1 - np.power(self.A1, 2))
        elif self.activation_fn == "relu":
            dA1 = np.dot(dZ2, self.W2.T) * (self.A1 > 0)
        elif self.activation_fn == "sigmoid":
            dA1 = np.dot(dZ2, self.W2.T) * self.A1 * (1 - self.A1)
        dW1 = np.dot(X.T, dA1) / m
        db1 = np.sum(dA1, axis=0, keepdims=True) / m

        # Update weights and biases
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

        # pass

def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y


# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # perform training steps by calling forward and backward function
    for _ in range(10):
        # Perform a training step
        mlp.forward(X)
        mlp.backward(X, y)

    # -------------------------------
    # 1. Hidden Space (3D Scatter & Hyperplane)
    # -------------------------------
    hidden_features = mlp.A1
    hidden_features /= np.max(np.abs(hidden_features), axis=0)
    ax_hidden.scatter(
        hidden_features[:, 0],
        hidden_features[:, 1],
        hidden_features[:, 2],
        c=y.ravel(),
        cmap='bwr',
        alpha=max(0, min(1, 0.7))
    )
    # ax_hidden.set_title("Hidden Space (3D)")
    # ax_hidden.set_xlim([-1.5, 1.5])
    # ax_hidden.set_ylim([-1.5, 1.5])
    # ax_hidden.set_zlim([-1.5, 1.5])

    # Plot decision boundary as a plane
    # x = np.linspace(-1.5, 1.5, 20)
    # y = np.linspace(-1.5, 1.5, 20)
    # X1, X2 = np.meshgrid(x, y)
    xx, yy = np.meshgrid(np.linspace(-1, 1, 30), np.linspace(-1, 1, 30))
    zz = -(mlp.W2[0] * xx + mlp.W2[1] * yy + mlp.b2[0]) / mlp.W2[2]  # Assuming W2 is [3 x n_classes]

    # Plot decision plane
    ax_hidden.plot_surface(xx, yy, zz, alpha=max(0, min(1, 0.3)), color='green')

    # Set titles and limits
    ax_hidden.set_xlim([-1, 1])
    ax_hidden.set_ylim([-1, 1])
    ax_hidden.set_zlim([-1, 1])
    ax_hidden.set_title(f"Hidden Space (Step {frame *10})")

    

    xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Forward pass through the MLP for grid predictions
    preds = mlp.forward(grid).reshape(xx.shape)
    binary_preds = (preds > 0.5).astype(int)

    # Plot the decision boundary dynamically
    ax_input.contourf(xx, yy, binary_preds, levels=[0, 0.5, 1], cmap='bwr', alpha = max(0, min(1, 0.3)))

    # Scatter plot of data points
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolor='k')

    # Title and limits
    ax_input.set_title(f"Input Space (Step {frame * 10})")
    ax_input.set_xlim([-3, 3])
    ax_input.set_ylim([-3, 3])
    # print(f"Frame {frame}, W1: {mlp.W1}, b1: {mlp.b1}")
    print(f"Predictions at frame {frame}: {preds[:5]}")
    # print(f"Gradient magnitudes: {np.linalg.norm(dW1)}, {np.linalg.norm(dW2)}")
    

        # Input layer nodes
    print("Input data shape:", X.shape)
    print("W1 shape (Input to Hidden):", mlp.W1.shape)
    print("W2 shape (Hidden to Output):", mlp.W2.shape)
    # Input layer nodes
    for i, (x, y) in enumerate(zip([0.2] * mlp.W1.shape[0], np.linspace(0.2, 0.8, mlp.W1.shape[0]))):
        ax_gradient.scatter(x, y, s=100, color='blue', label=f"Input {i + 1}" if frame == 0 else None)
        ax_gradient.text(x - 0.05, y, f"x{i + 1}", ha='center', fontsize=15)  # Label input nodes

    # Hidden layer nodes
    for j, (x, y) in enumerate(zip([0.5] * mlp.W1.shape[1], np.linspace(0.2, 0.8, mlp.W1.shape[1]))):
        ax_gradient.scatter(x, y, s=100, color='orange', label=f"Hidden {j + 1}" if frame == 0 else None)
        ax_gradient.text(x - 0.05, y, f"h{j + 1}", ha='center', fontsize=15)  # Label hidden nodes

    # Output layer nodes
    for k, (x, y) in enumerate(zip([0.8] * mlp.W2.shape[1], [0.5])):  # Single output node at y=0.5
        ax_gradient.scatter(x, y, s=100, color='red', label=f"Output {k + 1}" if frame == 0 else None)
        ax_gradient.text(x - 0.05, y, "y", ha='center', fontsize=15)  # Label output node

    # Draw edges with weight intensities
    for i in range(mlp.W1.shape[0]):  # Input to Hidden
        for j in range(mlp.W1.shape[1]):
            weight = mlp.W1[i, j]
            ax_gradient.plot(
                [0.2, 0.5],  # Horizontal coordinates: Input to Hidden
                [np.linspace(0.2, 0.8, mlp.W1.shape[0])[i], np.linspace(0.2, 0.8, mlp.W1.shape[1])[j]],  # Vertical
                color='gray' if weight == 0 else 'black',
                alpha=max(0, min(1, abs(weight))),
            )

    for j in range(mlp.W2.shape[0]):  # Hidden to Output
        for k in range(mlp.W2.shape[1]):
            weight = mlp.W2[j, k]
            ax_gradient.plot(
                [0.5, 0.8],  # Horizontal coordinates: Hidden to Output
                [np.linspace(0.2, 0.8, mlp.W1.shape[1])[j], 0.5],  # Vertical
                color='gray' if weight == 0 else 'black',
                alpha=max(0, min(1, abs(weight))),
            )

    # Add titles
    ax_gradient.set_title(f"Gradient Space (Step {frame * 10})")

    # Adjust axis limits to fit the left-to-right layout
    ax_gradient.set_xlim(0, .95)
    ax_gradient.set_ylim(0, .95)



def visualize(activation, lr, step_num):
    # fig, (ax_input, ax_hidden, ax_gradient) = plt.subplots(1, 3, figsize=(15, 5))
    X, y = generate_data()
    y_binary = (y > 0).astype(int)
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num//10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)