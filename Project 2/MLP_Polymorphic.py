import numpy as np
from tensorflow.keras import utils, datasets


class MLP:

    def __init__(self, activation, layer_layout):
        self.layer_layout = layer_layout
        self.input_size = layer_layout[0]
        self.output_size = layer_layout[-1]
        hidden_layer_count = len(layer_layout) - 2

        if 0 <= hidden_layer_count < 3:
            self.hidden_layers = hidden_layer_count
        elif 0 > hidden_layer_count:
            raise AttributeError("Hidden layer count must be >= 0")
        else:
            raise AttributeError("For this assignment, only up to two layers are supported. Sorry!")

        self.activation = activation
        self.initializeNetwork()
        self.activation_cache = {}

    def initializeNetwork(self):
        if self.hidden_layers == 0:
            self.initializeSimple(self.input_size, self.output_size)
        elif self.hidden_layers == 1:
            self.initializeOneLayer(self.input_size, self.output_size)
        elif self.hidden_layers == 2:
            self.initializeTwoLayers(self.input_size, self.output_size)

    def fit(self, training_input, training_labels, num_epochs, learning_rate,
            validation_input=None, validation_labels=None):
        valin = validation_input
        vallab = validation_labels
        if self.hidden_layers == 0:
            self.fitSimple(training_input, training_labels, num_epochs, learning_rate,
                           validation_input=valin, validation_labels=vallab)
        elif self.hidden_layers == 1:
            self.fitOneLayer(training_input, training_labels, num_epochs, learning_rate,
                             validation_input=valin, validation_labels=vallab)
        elif self.hidden_layers == 2:
            self.fitTwoLayers(training_input, training_labels, num_epochs, learning_rate,
                              validation_input=valin, validation_labels=vallab)

    def predict(self):
        if self.hidden_layers == 0:
            self.predictSimple(self.input_size, self.output_size)
        elif self.hidden_layers == 1:
            self.predictOneLayer(self.input_size, self.output_size)
        elif self.hidden_layers == 2:
            self.predictTwoLayers(self.input_size, self.output_size)

    def initializeSimple(self, input_size, output_size):
        # Initialize weights and biases
        self.weights = np.random.randn(output_size, input_size) * 0.01
        self.biases = np.zeros((output_size, 1))

    def initializeOneLayer(self, input_size, output_size):
        W1 = np.random.randn(input_size, self.layer_layout[1]) * 0.01
        b1 = np.zeros(self.layer_layout[1])
        W2 = np.random.randn(self.layer_layout[1], output_size)
        b2 = np.zeros(output_size)
        self.weights = [W1, W2]
        self.biases = [b1, b2]

    def initializeTwoLayers(self, input_size, output_size):
        pass

    def fitSimple(self, training_input, training_labels, num_epochs, learning_rate,
                  validation_input=None, validation_labels=None):
        for epoch in range(num_epochs):
            total_loss = 0.0
            total_accuracy = 0.0
            for idx, example in enumerate(training_input):
                # Forward propagation
                forward_prop = softmax(np.dot(self.weights, example))

                # Backward propagation and adjustments
                dErrdZ = forward_prop - training_labels[idx]
                dZdW = example
                dZdW = dZdW.reshape(dZdW.size, 1)
                dErrdZ = dErrdZ.reshape(1, dErrdZ.size)
                dErrdW = np.dot(dZdW, dErrdZ)
                dErrdB = np.sum(dErrdZ, axis=1, keepdims=True)
                self.weights -= (learning_rate * dErrdW.T)
                self.biases -= (learning_rate * dErrdB.T)

                total_loss += evaluate_acc(forward_prop, training_labels[idx])
                total_accuracy += evaluate_accuracy(forward_prop, training_labels[idx])

            avg_accuracy = total_accuracy / len(training_input)
            avg_loss = total_loss / len(training_input)
            print("Epoch", epoch, "Average Loss", avg_loss, "Average Accuracy", avg_accuracy)

            if validation_input is not None and validation_labels is not None:
                total_accuracy = 0.0
                for idx, example in enumerate(validation_input):
                    forward_prop = softmax(np.dot(self.weights, example))
                    total_accuracy += evaluate_accuracy(forward_prop, validation_labels[idx])
                avg_val_loss = total_accuracy / len(validation_input)
                print("Validation accuracy", avg_val_loss)

        return forward_prop

    def fitOneLayer(self, training_input, training_labels, num_epochs, learning_rate,
                    validation_input=None, validation_labels=None):
        for epoch in range(num_epochs):
            total_loss = 0.0
            total_accuracy = 0.0
            for idx, example in enumerate(training_input):
                W1 = self.weights[0]
                W2 = self.weights[1]
                b1 = self.biases[0]
                b2 = self.biases[1]

                # Forward propagation
                Z1 = np.dot(W1.T, example) + b1
                A1 = self.activation(Z1)
                Z2 = np.dot(W2.T, A1) + b2
                A2 = softmax(Z2)

                self.activation_cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}

                # Back propagation
                delta_output = A2 - training_labels[idx]
                delta_hidden = (delta_output.dot(W2.T) * relu_derivative(A1))

                grad_W2 = A2.T.dot(delta_output)
                grad_B2 = np.sum(delta_output, axis=0)
                grad_W1 = A1.T.dot(delta_hidden)
                grad_B1 = np.sum(delta_hidden, axis=0)

                W1 -= learning_rate * (grad_W1 / len(training_input))
                b1 -= learning_rate * (grad_B1 / len(training_input))
                W2 -= learning_rate * (grad_W2 / len(training_input))
                b2 -= learning_rate * (grad_B2 / len(training_input))

                self.weights[0] = W1
                self.weights[1] = W2
                self.biases[0] = b1
                self.biases[1] = b2

                total_loss += evaluate_acc(A2, training_labels[idx])
                total_accuracy += evaluate_accuracy(A2, training_labels[idx])

            avg_accuracy = total_accuracy / len(training_input)
            avg_loss = total_loss / len(training_input)
            print("Epoch", epoch, "Average Loss", avg_loss, "Average Accuracy", avg_accuracy)

            if validation_input is not None and validation_labels is not None:
                total_accuracy = 0.0
                for idx, example in enumerate(validation_input):
                    Z1v = np.dot(self.weights[0].T, example) + self.biases[0]
                    A1v = self.activation(Z1v)
                    Z2v = np.dot(self.weights[1].T, A1v) + self.biases[1]
                    forward_prop = softmax(Z2v)
                    total_accuracy += evaluate_accuracy(forward_prop, validation_labels[idx])
                avg_val_loss = total_accuracy / len(validation_input)
                print("Validation accuracy", avg_val_loss)

    def fitTwoLayers(self, training_input, training_labels, num_epochs, learning_rate):
        pass

    def predictSimple(self, input_size, output_size):
        pass

    def predictOneLayer(self, input_size, output_size):
        pass

    def predictTwoLayers(self, input_size, output_size):
        pass


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


def evaluate_acc(y_predicted, y_real):
    smooth_factor = 1e-10
    y_predicted = np.clip(y_predicted, smooth_factor, 1 - smooth_factor)
    return np.mean(-np.sum(y_real * np.log(y_predicted)))


def evaluate_accuracy(y_predicted, y_real):
    y_pred_labels = np.argmax(y_predicted, axis=0)
    y_real_labels = np.argmax(y_real, axis=0)
    accuracy = np.mean(y_pred_labels == y_real_labels)
    return accuracy


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def loadCIFAR10(normalize=True):
    # Load the CIFAR-10 dataset
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # Convert labels to categorical
    train_labels = utils.to_categorical(train_labels, num_classes=10)
    test_labels = utils.to_categorical(test_labels, num_classes=10)

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

    # Fin

    return train_images, train_labels, test_images, test_labels
