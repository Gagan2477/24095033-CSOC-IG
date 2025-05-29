import numpy as np

# Activation Functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

# Binary Cross Entropy Loss
def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-8
    return -np.mean(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))

# Neural Network Class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, 1) * 0.01
        self.b2 = np.zeros((1, 1))

    def forward(self, X):
        self.Z1 = X * self.W1 + self.b1
        self.A1 = relu(self.Z1)
        self.Z2 = self.A1 * self.W2 + self.b2
        self.A2 = sigmoid(self.Z2)
        return self.A2

    def backward(self, X, y, learning_rate=0.01):
        m = X.shape[0]
        dZ2 = self.A2 - y
        dW2 = (self.A1.T * dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = dZ2 * self.W2.T
        dZ1 = dA1 * relu_derivative(self.Z1)
        dW1 = (X.T * dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def train(self, X, y, epochs=1000, lr=0.01):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = binary_cross_entropy(y, y_pred)
            self.backward(X, y, learning_rate=lr)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        probs = self.forward(X)
        return (probs > 0.5).astype(int)

