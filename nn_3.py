import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def dsigmoid(y):
    return y * (1 - y)

def softmax(x):
    exp_vals = np.exp(x - np.max(x))  # Subtracting max(x) for numerical stability
    return exp_vals / exp_vals.sum(axis=0)

def cross_entropy(predictions, targets):
    epsilon = 1e-10  # Small value to avoid division by zero
    predictions = np.clip(predictions, epsilon, 1.0 - epsilon)  # Clip to prevent log(0) errors
    ce = -np.sum(targets * np.log(predictions))  # Compute cross-entropy
    return ce

class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.layers = [input_nodes] + hidden_nodes + [output_nodes]
        self.num_layers = len(self.layers)

        self.weights = [np.random.rand(self.layers[i + 1], self.layers[i]) - 0.5 for i in range(self.num_layers - 1)]
        self.biases = [np.random.rand(self.layers[i + 1], 1) - 0.5 for i in range(self.num_layers - 1)]

        self.learning_rate = 0.05

    def feedforward(self, input_array):
        activations = np.array(input_array, ndmin=2).T
        for i in range(self.num_layers - 1):
            activations = np.dot(self.weights[i], activations) + self.biases[i]
            activations = sigmoid(activations)
        return activations.flatten()

    def train(self, input_array, target_array):
        activations = np.array(input_array, ndmin=2).T
        layer_activations = [activations]

        for i in range(self.num_layers - 1):
            activations = np.dot(self.weights[i], activations) + self.biases[i]
            activations = sigmoid(activations)
            layer_activations.append(activations)

        targets = np.array(target_array, ndmin=2).T
        errors = [targets - activations]

        for i in range(self.num_layers - 2, -1, -1):
            gradient = dsigmoid(layer_activations[i + 1])
            gradient *= errors[-1]
            gradient *= self.learning_rate

            deltas = np.dot(gradient, layer_activations[i].T)
            self.weights[i] += deltas
            self.biases[i] += gradient

            weights_t = np.transpose(self.weights[i])
            errors.append(np.dot(weights_t, errors[-1]))

        errors.reverse()

        return cross_entropy(softmax(activations), targets)
