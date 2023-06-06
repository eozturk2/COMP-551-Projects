import tensorflow as tf
import numpy as np


def loadCIFAR10(normalize=True):
    # Load the CIFAR-10 dataset
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

    # Vectorize the dataset
    train_images = train_images.reshape((50000, 3072))
    test_images = test_images.reshape((10000, 3072))

    # Normalize the dataset
    train_images = train_images.astype('float32') / 255
    test_images = test_images.astype('float32') / 255

    # Compute the mean and standard deviation of the dataset
    mean = train_images.mean(axis=0)
    std = train_images.std(axis=0)

    # Normalize the dataset using the mean and standard deviation
    if normalize:
        train_images = (train_images - mean) / std
        test_images = (test_images - mean) / std

    return train_images, train_labels, test_images, test_labels


def ReLU(Z):
    return np.maximum(0, Z)


def leakyReLU(Z, leak):
    return np.maximum(np.zeros(Z.size), Z) + leak * np.minimum(np.zeros(Z.size), Z)


def tanh(Z):
    return (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))


def softmax(x):
    """Compute softmax values for each row of x."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)


def evaluate_acc(y_predicted, y_actual):
    y_predicted = np.clip(y_predicted, 1e-15, 1 - 1e-15)
    return -np.mean(np.sum(y_actual * np.log(y_predicted), axis=1))


class MLP:
    def __init__(self, activation, layer_layout, add_bias=False, bias=0):
        """

        :param activation: Activation function to be used for layers. Note that the output layer
        is always activated with softmax since this is a classification task.
        :param layer_layout: Python array representing the topology

        Example:
        3072 input neurons -> 256 layer 1 neurons -> 256 layer 2 neurons -> 10 output neurons
        = [3072, 256, 256, 10]
        """
        self.activation = activation
        self.layer_layout = layer_layout
        self.weights = []  # regular list, not np.array
        self.biases = []  # regular list, not np.array
        self.add_bias = add_bias
        self.bias = bias
        self.initialize_weights_and_biases()

    def initialize_weights_and_biases(self):
        for i in range(len(self.layer_layout) - 1):
            input_size = self.layer_layout[i]
            output_size = self.layer_layout[i + 1]
            layer_weights = np.random.rand(input_size, output_size)
            self.weights.append(layer_weights)

            if self.add_bias:
                layer_biases = np.zeros(output_size)
                self.biases.append(layer_biases)

    def forward(self, X):
        # Initialize input as X
        layer_input = X
        layer_output = None
        # Loop through layers
        for i, layer_weights in enumerate(self.weights):
            # Apply weights to input
            print("Input to layer " + str(i))
            print(layer_input.shape)
            print(layer_input[-1])

            print("Layer weights " + str(i))
            print(layer_weights.shape)
            print(layer_weights[-1])
            layer_output = np.dot(layer_input, layer_weights)

            # Add bias if requested
            if self.add_bias:
                layer_output += self.biases[i]

            # Apply activation function (except for last layer)
            if i < len(self.layer_layout) - 2:
                layer_output = self.activation(layer_output)
            else:
                # Apply softmax activation to last layer
                layer_output = softmax(layer_output)

            # Update input to be output of activation function
            layer_input = layer_output

            # print("Output from layer " + str(i))
            # print(layer_output.shape)
            # print(layer_output[-1])

        # Return output of last layer (predicted class probabilities)
        print()
        print()
        for output in layer_output:
            print(output)

        return layer_output

    def backward(self, X, Y, Y_hat, lr):
        
        m = X.shape[0]
        delta = [None] * len(self.weights)

        # Compute delta for the output layer
        delta[-1] = Y_hat - Y

        # Compute delta for the hidden layers
        for l in range(len(delta)-2, -1, -1):
            delta[l] = np.dot(delta[l+1], self.weights[l+1].T) * self.activation_derivative(self.layer_input[l+1])

        # Compute gradients for weights and biases
        dW = [None] * len(self.weights)
        db = [None] * len(self.biases)

        dW[0] = np.dot(X.T, delta[0])/m
        for l in range(1, len(self.weights)):
            dW[l] = np.dot(self.layer_output[l-1].T, delta[l])/m
        if self.add_bias:
            db = [np.mean(delta[l], axis=0) for l in range(len(delta))]

        # Update weights and biases
        self.weights = [w - lr*dW[i] for i,w in enumerate(self.weights)]
        self.biases = [b - lr*db[i] for i,b in enumerate(self.biases)]
    
    def activation_derivative(self, Z):
        if self.activation == ReLU:
            return np.where(Z > 0, 1, 0)
        elif self.activation == leakyReLU:
            return np.where(Z > 0, 1, self.bias)
        elif self.activation == tanh:
            return 1 - np.square(np.tanh(Z))
        elif self.activation == softmax:
            # derivative of softmax is a bit more complex
            pass


    def fit(self, X, y, lr, num_iterations, batch_size):
        num_examples = X.shape[0]
        if batch_size > 0:
            num_batches = num_examples // batch_size
        else:
            num_batches = 1
        for i in range(num_iterations):
            indices = np.random.permutation(num_examples)
            for j in range(num_batches):
                start_idx = j * batch_size
                end_idx = start_idx + batch_size
                batch_indices = indices[start_idx:end_idx]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]
                Y_hat = self.forward(X_batch)
                self.backward(X_batch, y_batch, Y_hat, lr)

    def predict(self, X):
        Y_hat = self.forward(X)
        return np.argmax(Y_hat, axis=1)
