# Goal : implement a multilayer perceptron (MLP) to classify image data
# Source : code available in the slides
# Description : 
#      backpropagation and
#      the mini-batch gradient descent algorithm used (e.g., SGD).
#      follow the equations that are presented in the lecture slides
#      Use Numpy
#      MLP :
#          Constructor:
#             Inputs:
#              - the activation function(e.g., ReLU), 
#              - the number of hidden layers (e.g., 2) 
#              - and the number of units in the hidden layers (e.g., [64, 64])

#              should initialize the weights and biases (with an initializer of your choice)
#
#          2 methods:
#             - fit 
#                takes the training data (i.e., X and y)—as well as other hyperparameters (e.g., the
#                learning rate and number of gradient descent iterations)—as input. 
#                This function should train your model by modifying the model parameters.

#             – predict 
#                takes a set of input points (i.e., X) as input and outputs predictions (i.e., ˆy) for these points.

#             - evaluate acc to evaluate the model accuracy. 
#                takes the true labels (i.e., y), and target labels (i.e., ˆy) as input, and output the accuracy score.

#             - use any Python libraries you like to tune the hyper-parameters (ex. grid search)


import numpy as np
from random import random
np.set_printoptions(precision=5)


class MLP(object):

    """A Multilayer Perceptron class. Inspired by the code on the following github repository links :

    https://github.com/musikalkemist/DeepLearningForAudioWithPython/blob/master/6-%20Implementing%20a%20neural%20network%20from%20scratch/code/mlp.py
    https://github.com/musikalkemist/DeepLearningForAudioWithPython/blob/master/8-%20Training%20a%20neural%20network:%20Implementing%20back%20propagation%20from%20scratch/code/mlp.py


    """

    def __init__(self, num_inputs=3, hidden_layers=[3, 3], num_outputs=2, activation_function='sigmoid'):
        """Constructor for the MLP. Takes the number of units in the input layer,
            a variable number of units in the hidden layers, and number of outputs
        Args:
            num_inputs (int): Number of units in the input layer
            hidden_layers (list): A list of ints for the hidden layers
            num_outputs (int): Number of units in the output layer
            activation (str): Activation function
        """

        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs
        if activation_function == "sigmoid":
            self.activation_function = self.sigmoid
        elif activation_function == "relu":
            self.activation_function = self.relu
        else:
            raise ValueError("Invalid activation function")
     
        # create a generic representation of the layers
        self.layers = [num_inputs] + hidden_layers + [num_outputs]

        # create random connection weights for the layers
        weights = []
        for i in range(len(self.layers)-1):
            w = np.random.rand(self.layers[i], self.layers[i+1])
            weights.append(w)
        self.weights = weights

        # create random biases for the layers
        biases = []
        for i in range(len(self.layers)-1):
            b = np.zeros((1, self.layers[i+1]))
            biases.append(b)
        self.biases = biases

        # save derivatives per layer
        derivatives = []
        for i in range(len(self.layers) - 1):
            dW= np.zeros((self.layers[i], self.layers[i + 1]))
            db= np.zeros((1, self.layers[i + 1]))
            derivatives.append({"weights": dW, "biases": db})
        self.derivatives = derivatives

        # save activations per layer
        activations = []
        for i in range(len(self.layers)):
            a = np.zeros(self.layers[i])
            activations.append(a)
        self.activations = activations



    def forward_propagate(self, X):
        """Computes forward propagation of the network based on input signals.
        Args:
            X (ndarray): Input signals
        Returns:
            Y_hat (ndarray): Output values
        """

        # the input layer activation is just the input itself
        activations = X

        # save the activations for backpropogation
        self.activations[0] = activations.reshape(activations.shape[0],-1).T               # need to reshape for backpropogation

        # iterate through the network layers except the last one
        for i, (w, b) in enumerate(zip(self.weights[:-1], self.biases[:-1])):

            # calculate matrix multiplication between previous activation and weight matrix
            net_inputs = np.dot(activations, w) + b

            # apply sigmoid activation function
            activations = self.activation_function(net_inputs)

            # save the activations for backpropogation
            self.activations[i+1] = activations

        # calculate net inputs for last layer
        net_inputs = np.dot(activations, self.weights[-1]) + self.biases[-1]

        # apply softmax activation function to last layer net inputs
        activations = self.softmax(net_inputs)

        # save the last activations for backpropogation
        self.activations[-1] = activations

        # return output layer activation
        return activations
    
    def back_propagate(self, y, Y_hat, verbose=False):
        """Backpropogates an error signal.
        Args:
            Y_hat: predicted y values
        Returns:
            y: actual y values
        """

        error = y - Y_hat

        # iterate backwards through the network layers
        for i in reversed(range(len(self.derivatives))):

            # get activation for previous layer
            activations = self.activations[i+1]

            if i == len(self.derivatives) - 1:
                # apply softmax derivative function for last layer
                # delta = error * self._softmax_derivative(activations)
                delta = error * self._softmax_derivative(activations)
            else:
                # apply sigmoid derivative function for other layers
                delta = error * self._sigmoid_derivative(activations)
            
            # get activations for current layer
            current_activations = self.activations[i]

            # reshape activations for dot product operation
            current_activations_reshaped = current_activations.reshape(current_activations.shape[0], -1).T

            # save derivative after applying matrix multiplication
            self.derivatives[i]["weights"] = np.dot(current_activations_reshaped, delta)
            self.derivatives[i]["biases"] = np.sum(delta, axis=0, keepdims=True)


            # backpropogate the next error
            error = np.dot(delta, self.weights[i].T)

            if verbose: 
                print("Derivative for W{}: {}".format(i, np.round(self.derivatives[i]["weights"], 5)))
                print("Derivative for B{}: {}".format(i, np.round(self.derivatives[i]["biases"], 5)))
        
        return error
    
    def fit(self, inputs, targets, epochs, learning_rate):
        """Trains model running forward prop and backprop
        Args:
            inputs (ndarray): X
            targets (ndarray): Y
            epochs (int): Num. epochs we want to train the network for
            learning_rate (float): Step to apply to gradient descent
        """
        # now enter the training loop
        for i in range(epochs):
            sum_errors = 0

            # iterate through all the training data
            for j, input in enumerate(inputs):
                target = targets[j]

                # activate the network!
                output = self.forward_propagate(input)

                self.back_propagate(target,output)

                # now perform gradient descent on the derivatives
                # (this will update the weights
                # self.stochastic_gradient_descent(input, output, epochs,learning_rate)
                self.gradient_descent(learning_rate)

                # keep track of the MSE for reporting later
                sum_errors += self._mse(target, output)

            # Epoch complete, report the training error
            print("Error: {} at epoch {}".format(sum_errors / len(inputs), i+1))

        print("Training complete!")
        print("=====")

    # def fit(self, X, y, lr, num_iterations, batch_size):
    #     num_examples = X.shape[0]
    #     num_batches = num_examples // batch_size
    #     for i in range(num_iterations):
    #         indices = np.random.permutation(num_examples)
    #         for j in range(num_batches):
    #             start_idx = j * batch_size
    #             end_idx = start_idx + batch_size
    #             batch_indices = indices[start_idx:end_idx]
    #             X_batch = X[batch_indices]
    #             y_batch = y[batch_indices]
    #             Y_hat = self.forward_propagate(X_batch)
    #             self.back_propagate(y_batch, Y_hat)

    


    def predict(self, X):
        Y_hat = self.forward_propagate(X)
        return np.argmax(Y_hat, axis=1)
    
    
    def gradient_descent(self, learningRate=1):
        """Learns by descending the gradient
        Args:
            learningRate (float): How fast to learn.
        """
        # update the weights by stepping down the gradient
        for i in range(len(self.weights)):
            weights = self.weights[i]
            # print("Original W{} {}".format(i, weights))
            biases = self.biases[i]
            # print("Original B{} {}".format(i, biases))
            deriv_weights = self.derivatives[i]["weights"]
            deriv_biases = self.derivatives[i]["biases"]
            weights += deriv_weights * learningRate
            biases += deriv_biases * learningRate
            # print("Updated W{} {}".format(i, weights))
            # print("Updated B{} {}".format(i, biases))
    
    def stochastic_gradient_descent(self, X, y, epochs, learning_rate=1):
        """
        Performs stochastic gradient descent to train the network.

        Args:
            X (numpy array): the input data to the network.
            y (numpy array): the target output of the network.
            epochs (int): the number of epochs to train for.
            learning_rate (float): the learning rate to use for gradient descent.
        """
        num_samples = X.shape[0]
        for epoch in range(epochs):
            for i in range(num_samples):
                # Select a random sample from the dataset
                idx = np.random.randint(num_samples)
                x_i = X[idx, :]
                y_i = y[idx, :]

                # Compute the derivatives using backpropagation for this sample
                self.back_propagate(x_i, y_i)

                # Update the weights and biases using the computed derivatives and learning rate
                for j in range(len(self.layers) -1):
                    weights = self.weights[j]
                    biases = self.biases[j]
                    weights_deriv = self.derivatives[j]["weights"]
                    biases_deriv = self.derivatives[j]["biases"]
                    weights += learning_rate * weights_deriv
                    biases += learning_rate * biases_deriv


    def sigmoid(self, x):
        """Sigmoid activation function
        Args:
            x (float): Value to be processed
        Returns:
            y (float): Output
        """
        
        y = 1.0 / (1 + np.exp(-x))
        return y
    
    def relu (self, x):
        return max(0, x)


    def softmax(self,x):
        """
        Compute softmax values for each row of x.
        """
        # Subtract the maximum value from each element to improve numerical stability
        x -= np.max(x, axis=1, keepdims=True)
        
        # Compute the exponential of each element
        exp_x = np.exp(x)
        
        # Compute the sum of the exponential values for each row
        sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
        
        # Compute the softmax probabilities by dividing each exponential value by the sum of exponential values for the row
        softmax_probs = exp_x / sum_exp_x
    
        return softmax_probs

    
    def _sigmoid_derivative(self, x):
        """Sigmoid derivative function
        Args:
            x (float): Value to be processed
        Returns:
            y (float): Output
        """
        return x * (1.0 - x)
    
    def _softmax_derivative(self, activations):
        """
        Calculate the derivative of the softmax activation function for the output layer.
        Args:
            activations: numpy array of activations from the output layer
        Returns:
            numpy array of derivatives of the softmax function
        """
        softmax = np.exp(activations) / np.sum(np.exp(activations), axis=1, keepdims=True)
        return softmax * (1 - softmax)
    
    def _mse(self, target, output):
        """Mean Squared Error loss function
        Args:
            target (ndarray): The ground trut
            output (ndarray): The predicted values
        Returns:
            (float): Output
        """
        return np.average((target - output) ** 2)


if __name__ == "__main__":

    # # create a Multilayer Perceptron
    # mlp = MLP(5, [5,6], 10)

    # # set random values for network's input
    # # inputs = np.random.rand(mlp.num_inputs)
    # inputs = np.array([0.1, 0.2,1,2,3])
    # target = np.array([0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2])

    # # perform forward propagation
    # output = mlp.forward_propagate(inputs)

    # mlp.back_propagate(target,output, verbose=False)

    # mlp.gradient_descent()

    # print("Network inputs: {}".format(inputs))
    # print("Network activation: {}".format(output))

    if __name__ == "__main__":

        # create a dataset to train a network for the sum operation
        items = np.array([[random()/2 for _ in range(2)] for _ in range(1000)])
        targets = np.array([[i[0] + i[1]] for i in items])

        # create a Multilayer Perceptron with one hidden layer
        mlp = MLP(2, [5], 1)

        # train network
        mlp.fit(items, targets, 50, 0.1)

        # create dummy data
        input = np.array([0.3, 0.1])
        target = np.array([0.4])

        # get a prediction
        output = mlp.predict(input)

        print()
        print("Our network believes that {} + {} is equal to {}".format(input[0], input[1], output[0]))