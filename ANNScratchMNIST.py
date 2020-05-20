import random
import numpy as np


class Network(object):

    def __init__(self, sizes):
        """"A network can be initialised with an array called sizes. For example if sizes was [2, 3, 1], then it would
            be a 3-layer network, with the first layer containing 2 neurons, the second layer 3 neurons, and the third layer
            1 neuron.
            Currently biases and weights are randomly generated from a standard normal distribution with mean 0 and variance 1.
            The first layer is the input layer, so it has no biases."""
        self.amountLayers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward_output(self, a):
        """"Returns the output vector of the network, if the vector a is the input"""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def feedforward(self, a):
        """"Returns two arrays. The first is the list of all activation_vectors ordered by layer.
        The second is the list of all weighted-input-sum_vectors ordered by layer"""
        activation = a
        activations = [a]
        weighted_input_sums = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            weighted_input_sums.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        return activations, weighted_input_sums

    def SGD(self, training_data, epochs, mini_batch_size, learning_rate):
        """"Training data is a list of tuples (x,y) representing the training inputs and the desired outputs.
        The training data is shuffled and divided into mini_batches"""
        training_data = list(training_data)

        for i in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, len(training_data), mini_batch_size)]
            for mini_batch in mini_batches:
                batch_gradient_bias, batch_gradient_weights = self.calculate_batch_gradient(mini_batch)
                self.gradient_descent(batch_gradient_bias, batch_gradient_weights, learning_rate)
            print("Epoch {} complete \n".format(i))

    def calculate_batch_gradient(self, mini_batch):
        """"Mini_batch is a list of tuples (x,y) representing a batch of training inputs and desired outputs."""
        # initialize GRADIENT sum as empty list of pdb vectors and empty list of pdw vectors
        batch_gradient_bias = [np.zeros(b.shape) for b in self.biases]
        batch_gradient_weights = [np.zeros(w.shape) for w in self.weights]

        # for each training example in the mini-batch, calculate update for all weights and biases in the network
        # the gradients of all training examples in the mini batch will be added up in the batch_gradient
        for i in range(len(mini_batch)):
            # feedforward
            activations, weighted_input_sums = self.feedforward(mini_batch[i][0])

            # pdbv = (per-layer) vector of partial derivatives of the loss function with respect to the bias
            # pdwv = (per-layer) vector of partial derivatives of the loss function with respect to the weight
            pdbv = sigmoid_derivative(weighted_input_sums[-1]) * (activations[-1] - mini_batch[i][1])
            pdwv = np.dot(pdbv, np.transpose(activations[-2]))

            # pdbv and pdwv that were just initialized are now added to the list of partial derivatives
            # this list is called the gradient
            gradient_bias = [pdbv]
            gradient_weights = [pdwv]

            # start from 2, because the pdbv and pdwv of the last layer are already calculated and added
            for k in range(2, self.amountLayers):
                pdbv = sigmoid_derivative(weighted_input_sums[-k]) * np.dot(self.weights[-k+1].transpose(), pdbv)
                pdwv = np.dot(pdbv, np.transpose(activations[-k-1]))
                gradient_bias.append(pdbv)
                gradient_weights.append(pdwv)

            # pdb/pdw vectors are added in order from last layer to first layer. Reverse for later purposes
            gradient_bias.reverse()
            gradient_weights.reverse()

            # Add each pdbv and pdwv of the gradient to the corresponding vector of the batch_gradient
            for pdb_vector in range(len(gradient_bias)):
                batch_gradient_bias[pdb_vector] += gradient_bias[pdb_vector]
            for pdw_vector in range(len(gradient_weights)):
                batch_gradient_weights[pdw_vector] += gradient_weights[pdw_vector]

        # Finally, the sum that was added up in the batch_gradient is divided by the amount of training examples from
        # the mini-batch to get the average gradient over the mini-batch
        batch_gradient_bias = [pdb / len(mini_batch) for pdb in batch_gradient_bias]
        batch_gradient_weights = [pdw / len(mini_batch) for pdw in batch_gradient_weights]

        return batch_gradient_bias, batch_gradient_weights

    def gradient_descent(self, gradient_bias, gradient_weights, learning_rate):
        update_step_biases = [learning_rate * (-1) * pdbv for pdbv in gradient_bias]
        update_step_weights = [learning_rate * (-1) * pdwv for pdwv in gradient_weights]
        for elem in range(self.amountLayers - 1):
            self.biases[elem] += update_step_biases[elem]
            self.weights[elem] += update_step_weights[elem]

    def MSE(self, output_activation, target_activation):
        return (output_activation - target_activation)**2

    def MSE_derivative(self, output_activation, target_activation):
        """"Note that is the PARTIAL derivative with respect to the activation"""
        return (output_activation - target_activation)


# Some math functions
def sigmoid(z):
    """"The sigmoid function"""
    return 1.0/(1.0+np.exp(-z))


def sigmoid_derivative(z):
    """"Derivative of the sigmoid function"""
    return sigmoid(z)*(1-sigmoid(z))
