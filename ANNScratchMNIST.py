import random
import numpy as np
import matplotlib.pyplot as plt


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
        """"Performs stochastic gradient descent (SGD) on the network, using mini batches. Training data is a list of
        tuples (x,y) representing the training inputs and the desired outputs. The training data is shuffled and
        divided into mini_batches. For each mini_batch gradient descent is performed on all weights and biases in the
        network. Gradient descent will be performed for all mini batch each epoch. Also, metrics are computed, printed,
        and shown in a plot"""
        training_data = list(training_data)

        # Initial prints, start showing initial metrics before training
        initial_loss, initial_accuracy = self.calculate_metrics(training_data)
        print_initial_metrics("Stochastic Gradient Descent (SGD)", initial_loss, initial_accuracy)
        losses = [initial_loss]
        accuracies = [initial_accuracy]
        epochs_axis = [0]

        # Repeat the process every epoch
        for i in range(epochs):
            # shuffle the training_data and divide into mini_batches
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, len(training_data), mini_batch_size)]

            # calculate the average batch gradient for weights and biases and use it to perform GRADIENT DESCENT
            for mini_batch in mini_batches:
                batch_gradient_bias, batch_gradient_weights = self.calculate_batch_gradient(mini_batch)
                self.gradient_descent(batch_gradient_bias, batch_gradient_weights, learning_rate)

            # METRICS calculation
            loss, accuracy = self.calculate_metrics(training_data)
            losses.append(loss)
            accuracies.append(accuracy)
            epochs_axis.append(i + 1)

            # PRINT in console
            print_training_metrics(i + 1, loss, accuracy)

        # PLOT the metrics
        plot_metrics(epochs_axis, losses, accuracies)

    def calculate_metrics(self, training_data):
        """"Returns the metrics loss and accuracy for all training_data after an epoch"""
        correct_predictions = 0
        los_sum = 0
        for training_example in range(len(training_data)):
            if self.predict(training_data[training_example][0], training_data[training_example][1]):
                correct_predictions += 1
            los_sum += self.MSE(self.feedforward_output(training_data[training_example][0]),
                                training_data[training_example][1])

        loss = los_sum / len(training_data)
        accuracy = correct_predictions / len(training_data)

        return loss, accuracy

    def predict(self, input_vector, target_vector):
        """"Returns True if the highest activated neuron of the output equals the highest activated neuron of the
        target. In other words, returns True if the prediction is correct, else False. Note that this only works
        for output layers with a sigmoid, relu or softmax activation function (not for thanh)"""
        output_vector = list(self.feedforward_output(input_vector))
        highest = max(output_vector)
        indx = output_vector.index(highest)
        target_indx = list(target_vector).index(1)

        return indx == target_indx

    def calculate_batch_gradient(self, mini_batch):
        """"Returns the average gradient over the batch of all the biases and all the weights in the network.
        Mini_batch is a list of tuples (x,y) representing a batch of training inputs and desired outputs."""
        # GRADIENT biases initialized as empty list of pdb vectors and GRADIENT weights empty list of pdw vectors
        batch_gradient_bias = [np.zeros(b.shape) for b in self.biases]
        batch_gradient_weights = [np.zeros(w.shape) for w in self.weights]

        # for each training example in the mini-batch, calculate update for all weights and biases in the network
        # the gradients of all training examples in the mini batch will be added up in the batch_gradient
        for i in range(len(mini_batch)):
            # FEEDFORWARD
            activations, weighted_input_sums = self.feedforward(mini_batch[i][0])

            # pdbv = (per-layer) vector of partial derivatives of the loss function with respect to the bias
            # pdwv = (per-layer) vector of partial derivatives of the loss function with respect to the weight
            pdbv = sigmoid_derivative(weighted_input_sums[-1]) * self.MSE_derivative(activations[-1], mini_batch[i][1])
            pdwv = np.dot(pdbv, np.transpose(activations[-2]))

            # pdbv and pdwv that were just initialized are now added to the list of partial derivatives
            # this list is called the gradient
            gradient_bias = [pdbv]
            gradient_weights = [pdwv]

            # BACKPROPAGATION
            # start from 2, because the pdbv and pdwv of the last layer are already calculated and added.
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
        """"Updates all the weights and biases in the network by taking a step in the negative gradient direction
        of the loss function with respect to each particular weight and bias in the network. The step is multiplied
        by the learning rate in order to control the step size"""
        # STEP SIZE calculation
        update_step_biases = [learning_rate * (-1) * pdbv for pdbv in gradient_bias]
        update_step_weights = [learning_rate * (-1) * pdwv for pdwv in gradient_weights]

        # update all weights and biases in the network, done by (per-layer) vector-wise addition
        for elem in range(self.amountLayers - 1):
            self.biases[elem] += update_step_biases[elem]
            self.weights[elem] += update_step_weights[elem]

    def MSE(self, output_activation, target_activation):
        """"Returns the value of the loss function Mean Squared Error for one training example"""
        return np.sum((output_activation-target_activation)**2) / len(output_activation)

    def MSE_derivative(self, output_activation, target_activation):
        """"Returns the (partial) derivative of the Mean Squared Error loss function.
        Note that is the PARTIAL derivative with respect to the activation"""
        return output_activation - target_activation

    # --------------------------------------------- OTHER TRAINING METHODS ---------------------------------------------
    def hebbian(self, hebbian_type, training_data, epochs, learning_rate):
        """"Depending on hebbian_type that was specified:
        Performs Hebbian learning on the network that was inspired by an analogy to logic. The truth table of the
        imply operator is used to define this learning rule.
        OR
        Performs competitive Hebbian learning on the network by decreasing weights of connections where the pre-
        synaptic neuron fails to excite the post-synaptic neuron, but the post-synaptic neuron is firing nonetheless
        under the influence of other pre-synaptic neurons. This means that the increase of strength of certain synapses
        onto the post-synaptic neuron is accompanied by the simultaneous decrease of strength of other synapses onto
        the same post-synaptic neuron: Spatial competition between convergent afferents.
        OR
        Performs BCM Theory, which is focused on synaptic modification that results in temporal competition between
        input patterns rather than a spatial competition between different synapses. Whether the synaptic strength
        increases or decreases depends on the magnitude of the post-synaptic response as compared with a variable
        modification threshold."""
        training_data = list(training_data)
        quit_training = False

        # Initial prints, start showing initial metrics before training
        initial_loss, initial_accuracy = self.calculate_metrics(training_data)
        print_initial_metrics("Hebbian (" + hebbian_type + ") learning rule", initial_loss, initial_accuracy)
        losses = [initial_loss]
        accuracies = [initial_accuracy]
        epochs_axis = [0]

        # Only used for BCM: Initialize the list that contains average activations over all epochs as empty vectors
        activations, weighted_input_sums = self.feedforward(training_data[1][0])
        average_act_list = [np.zeros(act.shape) for act in activations]

        for i in range(epochs):
            random.shuffle(training_data)

            # the average activations list over all training examples in an epoch
            # is initialized as a list with empty activation vectors
            avg_acts = [np.zeros(act.shape) for act in activations]

            for training_example in range(len(training_data)):
                # FEEDFORWARD
                activations, weighted_input_sums = self.feedforward(training_data[training_example][0])

                # Add every activation vector to the corresponding average activation vector
                for act_idx in range(len(avg_acts)):
                    avg_acts[act_idx] += activations[act_idx]

            # calculate every average activation vector by dividing by the amount of training examples
            avg_acts = [act / len(training_data) for act in avg_acts]

            # Add every avg_activation vector to the corresponding average (over epochs) activation vector
            for avg_act_vec_idx in range(len(average_act_list)):
                average_act_list[avg_act_vec_idx] += avg_acts[avg_act_vec_idx]

            # Moving Average: calculate every average (over epochs) activation vector by dividing the amount of epochs
            average_act_list = [avg_act_vec / (i + 1) for avg_act_vec in average_act_list]

            # A hebbian type learning rule is used to update the weights of the network
            for w_idx in range(len(self.weights)):
                # Define pre-synaptic activation and post-synaptic activation
                pre_act = avg_acts[w_idx]
                pre_act_row = np.transpose(pre_act)
                post_act = avg_acts[w_idx + 1]
                average_post_act = average_act_list[w_idx + 1]

                # HEBBIAN IMPLY LEARNING RULE
                if hebbian_type == "imply":
                    self.weights[w_idx] = (2 * (np.dot(post_act, pre_act_row)) -
                                           np.dot(np.ones(post_act.shape), pre_act_row)) \
                                          * learning_rate + self.weights[w_idx]
                # HEBBIAN COMPETITIVE LEARNING RULE
                elif hebbian_type == "competitive":
                    self.weights[w_idx] = (np.dot(post_act, pre_act_row) - np.abs(np.dot(post_act, pre_act_row))) \
                                          * learning_rate + self.weights[w_idx]
                # BCM THEORY
                elif hebbian_type == "BCM":
                    p = 2
                    epsilon = 0.1
                    # FIXME: Currently, I take the average output activation over time. I should take the average
                    # activation per neuron of the input layer, and calculate activations for other layers from that
                    threshold = ((average_post_act / learning_rate)**p) * average_post_act
                    delta_weight = np.dot((post_act - threshold), pre_act_row) + (1 - epsilon) * self.weights[w_idx]
                    self.weights[w_idx] += delta_weight
                # INVALID LEARNING RULE
                else:
                    print(hebbian_type + " is not a valid Hebbian learning method. Please choose specify one of the "
                                         "following: \n \"imply\", \"competitive\", \"BCM\"")
                    quit_training = True
                    break

            # --- end of for loop trough layers of the network --- #

            # only used when invalid hebbian_type is specified
            if quit_training:
                break

            # METRICS calculation
            loss, accuracy = self.calculate_metrics(training_data)
            losses.append(loss)
            accuracies.append(accuracy)
            epochs_axis.append(i + 1)

            # PRINT in console
            print_training_metrics(i + 1, loss, accuracy)

        # PLOT the metrics
        plot_metrics(epochs_axis, losses, accuracies)


# Some math functions
def sigmoid(z):
    """"Returns the sigmoid function"""
    return 1.0/(1.0+np.exp(-z))


def sigmoid_derivative(z):
    """"Returns the derivative of the sigmoid function"""
    return sigmoid(z)*(1-sigmoid(z))


# Some printing and plotting functions
def print_initial_metrics(method, initial_loss, initial_accuracy):
    print("Start Training: 0 epochs complete")
    print("Training method: " + method)
    print("Initial loss: {}".format(initial_loss))
    print("Initial accuracy: {} \n".format(initial_accuracy))


def print_training_metrics(epoch, loss, accuracy):
    print("Epoch {} complete".format(epoch))
    print("Loss: {}".format(loss))
    print("Accuracy: {} \n".format(accuracy))


def plot_metrics(horizontal_axis, losses, accuracies):
    plt.plot(horizontal_axis, losses)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()
    plt.plot(horizontal_axis, accuracies)
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.show()
