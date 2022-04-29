import numpy as np
class Neural_net():
    def __init__(self, input_size, output_size, hidden_layers, hidden_size, learning_rate):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        self.weights_grad = []
        self.biases_grad = []
        self.initialize_weights()
        self.initialize_biases()

    def initialize_weights(self):
        for i in range(self.hidden_layers):
            if i == 0:
                self.weights.append(np.random.randn(self.input_size, self.hidden_size))
            else:
                self.weights.append(np.random.randn(self.hidden_size, self.hidden_size))
        self.weights.append(np.random.randn(self.hidden_size, self.output_size))
    def initialize_biases(self):
        for i in range(self.hidden_layers):
            if i == 0:
                self.biases.append(np.random.randn(self.hidden_size))
            else:
                self.biases.append(np.random.randn(self.hidden_size))
        self.biases.append(np.random.randn(self.output_size))
    def forward(self, x):
        self.input = x
        self.hidden = []
        self.output = []
        for i in range(self.hidden_layers):
            if i == 0:
                self.hidden.append(np.dot(self.input, self.weights[i]) + self.biases[i])
            else:
                self.hidden.append(np.dot(self.hidden[i-1], self.weights[i]) + self.biases[i])
        self.output.append(np.dot(self.hidden[self.hidden_layers-1], self.weights[self.hidden_layers]) + self.biases[self.hidden_layers])
        return self.output[0]
    def backward(self, y):
        self.error = y - self.output[0]
        self.delta = self.error * self.output[0] * (1 - self.output[0])
        self.weights_grad.append(np.dot(self.hidden[self.hidden_layers-1].T, self.delta))
        self.biases_grad.append(self.delta)
        for i in range(self.hidden_layers-1, -1, -1):
            self.delta = np.dot(self.delta, self.weights[i+1].T) * self.hidden[i] * (1 - self.hidden[i])
            self.weights_grad.append(np.dot(self.input.T, self.delta))
            self.biases_grad.append(self.delta)
        self.weights_grad.reverse()
        self.biases_grad.reverse()
    def update_weights(self):
        for i in range(self.hidden_layers+1):
            self.weights[i] -= self.learning_rate * self.weights_grad[i]
            self.biases[i] -= self.learning_rate * self.biases_grad[i]
    def train(self, x, y):
        self.forward(x)
        self.backward(y)
        self.update_weights()

