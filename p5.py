import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, input_size, output_size, hidden_size):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.hidden_layer_input_weights = np.random.rand(self.hidden_size, self.input_size)
        self.hidden_layer_bias = np.zeros((self.hidden_size, 1))
        self.hidden_layer_output_weights = np.random.rand(self.output_size, self.hidden_size)

        self.output_bias = np.zeros((self.output_size, 1))
        self.learning_rate = 0.01

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, inputs, target, epochs):
        losses = []

        for epoch in range(epochs):
            hidden_layer_input = np.dot(self.hidden_layer_input_weights, inputs) + self.hidden_layer_bias
            hidden_layer_output = self.sigmoid(hidden_layer_input)

            output_layer_input = np.dot(self.hidden_layer_output_weights, hidden_layer_output) + self.output_bias
            output_layer_output = self.sigmoid(output_layer_input)

            error = target - output_layer_output
            losses.append(np.mean(error**2))

            output_layer_error = error * self.sigmoid_derivative(output_layer_output)
            hidden_layer_error = np.dot(self.hidden_layer_output_weights.T, output_layer_error) * self.sigmoid_derivative(hidden_layer_output)

            self.hidden_layer_output_weights += self.learning_rate * np.dot(output_layer_error, hidden_layer_output.T)
            self.output_bias += self.learning_rate * np.sum(output_layer_error, axis=1, keepdims=True)

            self.hidden_layer_input_weights += self.learning_rate * np.dot(hidden_layer_error, inputs.T)
            self.hidden_layer_bias += self.learning_rate * np.sum(hidden_layer_error, axis=1, keepdims=True)

        return losses

    def predict(self, inputs):
        hidden_inputs = np.dot(self.hidden_layer_input_weights, inputs) + self.hidden_layer_bias
        hidden_output = self.sigmoid(hidden_inputs)

        output_input = np.dot(self.hidden_layer_output_weights, hidden_output) + self.output_bias
        output_output = self.sigmoid(output_input)

        return output_output

np.random.seed(42)
X = np.random.rand(2, 100)
y = (X[0, :] + X[1, :]).reshape(1, -1)

X = (X - np.mean(X)) / np.std(X)
y = (y - np.mean(y)) / np.std(y)

input_size = X.shape[0]
hidden_size = 4
output_size = 1

nn = NeuralNetwork(input_size, output_size, hidden_size)
losses = nn.train(X, y, 1000)

plt.plot(losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.show()
