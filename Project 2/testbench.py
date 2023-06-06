import perceptron
import tensorflow as tf
import time
from matplotlib import pyplot as plt
import numpy as np


# function begin
# batch: 16
# filter:
def fitWithHyperParams(train_images, train_labels, test_images, test_labels, filter, kernel, pool_size,
                       learning_rate, momentum, epochs, batch_size):
    # Construct the model as described by the lab handout and hyperparameters
    cnn = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filter, kernel, activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=pool_size),
        tf.keras.layers.Conv2D(filter, kernel, activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=pool_size),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Optimize with gradient descent
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)

    # Finally, compile the model
    cnn.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    training_times = []
    training_accuracies = []
    training_losses = []
    test_accuracies = []
    test_losses = []

    for i in range(epochs):
        start_time = time.time()
        performance = cnn.fit(train_images, train_labels, epochs=1, batch_size=batch_size,
                              validation_data=(test_images, test_labels))
        end_time = time.time()
        training_time = end_time - start_time
        training_times.append(training_time)

        training_accuracies.append(performance.history['accuracy'][0])
        training_losses.append(performance.history['loss'][0])
        test_accuracies.append(performance.history['val_accuracy'][0])
        test_losses.append(performance.history['val_loss'][0])

    return [np.array(training_times), np.array(training_accuracies),
            np.array(training_losses), np.array(test_accuracies), np.array(test_losses)]


if __name__ == "__main__":
    train_images, train_labels, test_images, test_labels = perceptron.loadCIFAR10()

    # Default parameters for CNN
    filter = 4
    kernel = (3, 3)
    pool = (2, 2)
    learning_rate = 0.0001
    momentum = 0.9
    epochs = 10
    batch_size = 512

    (train_images_cnn, train_labels_cnn), (test_images_cnn, test_labels_cnn) = tf.keras.datasets.cifar10.load_data()
    train_labels_cnn = tf.keras.utils.to_categorical(train_labels_cnn, num_classes=10)
    test_labels_cnn = tf.keras.utils.to_categorical(test_labels_cnn, num_classes=10)

    train_labels_cnn = train_labels_cnn.astype('float32')
    test_labels_cnn = test_labels_cnn.astype('float32')

    # Learning rate
    # create the plot with a single subplot
    # LR: 0.0001
    def tryLearningRates():
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)

        # loop over each dataset and plot it on the same subplot
        for i in range(-6, -1):
            # generate data for the current dataset
            # x, y = fitWithHyperParams(i)
            learning = (10 ** i)
            performanceData = fitWithHyperParams(train_images_cnn, train_labels_cnn, test_images_cnn,
                                                 test_labels_cnn, filter, kernel, pool, learning, momentum,
                                                 epochs, batch_size)

            t = np.cumsum(performanceData[0])

            # plot the data on the same subplot
            ax1.plot(t - t[0], 100 * performanceData[1], label=f"Learning rate: {learning}",
                     marker='o', linestyle='-')
            ax2.plot(t - t[0], 100 * performanceData[3], label=f"Learning rate: {learning}",
                     marker='o', linestyle='-')
            ax3.plot(t - t[0], performanceData[2], label=f"Learning rate: {learning}",
                     marker='o', linestyle='-')
            ax4.plot(t - t[0], performanceData[4], label=f"Learning rate: {learning}",
                     marker='o', linestyle='-')

        # add a title to the subplot
        ax1.set_title('Training accuracy')
        ax2.set_title('Test accuracy')
        ax3.set_title('Training loss')
        ax4.set_title('Test loss')

        # add labels for x and y axes
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Training accuracy (%)')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Training accuracy (%)')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Training accuracy (%)')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Training accuracy (%)')

        # add a legend for the plot
        ax1.legend()
        ax2.legend()
        ax3.legend()
        ax4.legend()

        # display the plot
        ax1.grid()
        ax2.grid()
        ax3.grid()
        ax4.grid()
        plt.show()


    # Filter size
    # 4 chosen
    def tryFilterSizes():
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)

        # loop over each dataset and plot it on the same subplot
        for i in range(0, 6):
            # generate data for the current dataset
            # x, y = fitWithHyperParams(i)
            filter_s = (2 ** i)
            performanceData = fitWithHyperParams(train_images_cnn, train_labels_cnn, test_images_cnn,
                                                 test_labels_cnn, filter, kernel, pool, learning_rate, momentum,
                                                 epochs, batch_size)

            t = np.cumsum(performanceData[0])

            # plot the data on the same subplot
            ax1.plot(t - t[0], 100 * performanceData[1], label=f"Filter size: {filter_s}",
                     marker='o', linestyle='-')
            ax2.plot(t - t[0], 100 * performanceData[3], label=f"Filter size: {filter_s}",
                     marker='o', linestyle='-')
            ax3.plot(t - t[0], performanceData[2], label=f"Filter size: {filter_s}",
                     marker='o', linestyle='-')
            ax4.plot(t - t[0], performanceData[4], label=f"Filter size: {filter_s}",
                     marker='o', linestyle='-')

        # add a title to the subplot
        ax1.set_title('Training accuracy')
        ax2.set_title('Test accuracy')
        ax3.set_title('Training loss')
        ax4.set_title('Test loss')

        # add labels for x and y axes
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Training accuracy (%)')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Training accuracy (%)')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Training loss')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Training loss')

        # add a legend for the plot
        ax1.legend()
        ax2.legend()
        ax3.legend()
        ax4.legend()

        # display the plot
        ax1.grid()
        ax2.grid()
        ax3.grid()
        ax4.grid()
        plt.show()


    # Batch size
    # 16 chosen
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)


    def tryBatchSizes():
        # loop over each dataset and plot it on the same subplot
        for i in range(0, 6):
            # generate data for the current dataset
            # x, y = fitWithHyperParams(i)
            batch_s = (2 ** i)
            performanceData = fitWithHyperParams(train_images_cnn, train_labels_cnn, test_images_cnn,
                                                 test_labels_cnn, filter, kernel, pool, learning_rate, momentum,
                                                 epochs, batch_s)

            t = np.cumsum(performanceData[0])

            # plot the data on the same subplot
            ax1.plot(t - t[0], 100 * performanceData[1], label=f"Batch size: {batch_s}",
                     marker='o', linestyle='-')
            ax2.plot(t - t[0], 100 * performanceData[3], label=f"Batch size: {batch_s}",
                     marker='o', linestyle='-')
            ax3.plot(t - t[0], performanceData[2], label=f"Batch size: {batch_s}",
                     marker='o', linestyle='-')
            ax4.plot(t - t[0], performanceData[4], label=f"Batch size: {batch_s}",
                     marker='o', linestyle='-')

        # add a title to the subplot
        ax1.set_title('Training accuracy')
        ax2.set_title('Test accuracy')
        ax3.set_title('Training loss')
        ax4.set_title('Test loss')

        # add labels for x and y axes
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Training accuracy (%)')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Training accuracy (%)')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Training loss')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Training loss')

        # add a legend for the plot
        ax1.legend()
        ax2.legend()
        ax3.legend()
        ax4.legend()

        # display the plot
        ax1.grid()
        ax2.grid()
        ax3.grid()
        ax4.grid()
        plt.show()
        # Pool size


    # Batch size = 16 and filter size = 32 gives almost 60% test accuracy
    def increaseFilterSize():
        for i in range(0, 1):
            # generate data for the current dataset
            # x, y = fitWithHyperParams(i)
            batch_s = 16
            performanceData = fitWithHyperParams(train_images_cnn, train_labels_cnn, test_images_cnn,
                                                 test_labels_cnn, 32, kernel, pool, learning_rate, momentum,
                                                 15, batch_s)

            t = np.cumsum(performanceData[0])

            # plot the data on the same subplot
            ax1.plot(t - t[0], 100 * performanceData[1], label=f"Batch size: {batch_s}",
                     marker='o', linestyle='-')
            ax2.plot(t - t[0], 100 * performanceData[3], label=f"Batch size: {batch_s}",
                     marker='o', linestyle='-')
            ax3.plot(t - t[0], performanceData[2], label=f"Batch size: {batch_s}",
                     marker='o', linestyle='-')
            ax4.plot(t - t[0], performanceData[4], label=f"Batch size: {batch_s}",
                     marker='o', linestyle='-')

        # add a title to the subplot
        ax1.set_title('Training accuracy')
        ax2.set_title('Test accuracy')
        ax3.set_title('Training loss')
        ax4.set_title('Test loss')

        # add labels for x and y axes
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Training accuracy (%)')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Training accuracy (%)')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Training loss')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Training loss')

        # add a legend for the plot
        ax1.legend()
        ax2.legend()
        ax3.legend()
        ax4.legend()

        # display the plot
        ax1.grid()
        ax2.grid()
        ax3.grid()
        ax4.grid()
        plt.show()


    # Kernel size: 3x3
    def tryKernel():
        for i in range(1, 6):
            # generate data for the current dataset
            # x, y = fitWithHyperParams(i)
            kern = (i, i)
            performanceData = fitWithHyperParams(train_images_cnn, train_labels_cnn, test_images_cnn,
                                                 test_labels_cnn, 32, kernel, pool, learning_rate, momentum,
                                                 3, 16)

            t = np.cumsum(performanceData[0])

            # plot the data on the same subplot
            ax1.plot(t - t[0], 100 * performanceData[1], label=f"Kernel size: {kern[0]}",
                     marker='o', linestyle='-')
            ax2.plot(t - t[0], 100 * performanceData[3], label=f"Kernel size: {kern[0]}",
                     marker='o', linestyle='-')
            ax3.plot(t - t[0], performanceData[2], label=f"Kernel size: {kern[0]}",
                     marker='o', linestyle='-')
            ax4.plot(t - t[0], performanceData[4], label=f"Kernel size: {kern[0]}",
                     marker='o', linestyle='-')

        # add a title to the subplot
        ax1.set_title('Training accuracy')
        ax2.set_title('Test accuracy')
        ax3.set_title('Training loss')
        ax4.set_title('Test loss')

        # add labels for x and y axes
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Training accuracy (%)')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Training accuracy (%)')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Training loss')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Training loss')

        # add a legend for the plot
        ax1.legend()
        ax2.legend()
        ax3.legend()
        ax4.legend()

        # display the plot
        ax1.grid()
        ax2.grid()
        ax3.grid()
        ax4.grid()
        plt.show()


    # Pool size: 4
    def tryPoolSizes():
        for i in range(1, 5):
            # generate data for the current dataset
            # x, y = fitWithHyperParams(i)
            pool = (5 - i, 5 - i)
            performanceData = fitWithHyperParams(train_images_cnn, train_labels_cnn, test_images_cnn,
                                                 test_labels_cnn, 32, (3, 3), pool, learning_rate, momentum,
                                                 3, 16)

            t = np.cumsum(performanceData[0])

            # plot the data on the same subplot
            ax1.plot(t - t[0], 100 * performanceData[1], label=f"Pool size: {pool[0]}",
                     marker='o', linestyle='-')
            ax2.plot(t - t[0], 100 * performanceData[3], label=f"Kernel size: {pool[0]}",
                     marker='o', linestyle='-')
            ax3.plot(t - t[0], performanceData[2], label=f"Kernel size: {pool[0]}",
                     marker='o', linestyle='-')
            ax4.plot(t - t[0], performanceData[4], label=f"Kernel size: {pool[0]}",
                     marker='o', linestyle='-')

        # add a title to the subplot
        ax1.set_title('Training accuracy')
        ax2.set_title('Test accuracy')
        ax3.set_title('Training loss')
        ax4.set_title('Test loss')

        # add labels for x and y axes
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Training accuracy (%)')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Training accuracy (%)')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Training loss')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Training loss')

        # add a legend for the plot
        ax1.legend()
        ax2.legend()
        ax3.legend()
        ax4.legend()

        # display the plot
        ax1.grid()
        ax2.grid()
        ax3.grid()
        ax4.grid()
        plt.show()


    # Momentum: 0.7
    def tryMomenta():
        for i in range(0, 10):
            # generate data for the current dataset
            # x, y = fitWithHyperParams(i)
            momentum = (i / 10)
            performanceData = fitWithHyperParams(train_images_cnn, train_labels_cnn, test_images_cnn,
                                                 test_labels_cnn, 32, (3, 3), (4, 4), learning_rate, momentum,
                                                 3, 16)
            print(momentum)

            t = np.cumsum(performanceData[0])

            # plot the data on the same subplot
            ax1.plot(t - t[0], 100 * performanceData[1], label=f"Momentum: {momentum}",
                     marker='o', linestyle='-')
            ax2.plot(t - t[0], 100 * performanceData[3], label=f"Momentum: {momentum}",
                     marker='o', linestyle='-')
            ax3.plot(t - t[0], performanceData[2], label=f"Momentum: {momentum}",
                     marker='o', linestyle='-')
            ax4.plot(t - t[0], performanceData[4], label=f"Momentum: {momentum}",
                     marker='o', linestyle='-')

        # add a title to the subplot
        ax1.set_title('Training accuracy')
        ax2.set_title('Test accuracy')
        ax3.set_title('Training loss')
        ax4.set_title('Test loss')

        # add labels for x and y axes
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Training accuracy (%)')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Training accuracy (%)')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Training loss')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Training loss')

        # add a legend for the plot
        ax1.legend()
        ax2.legend()
        ax3.legend()
        ax4.legend()

        # display the plot
        ax1.grid()
        ax2.grid()
        ax3.grid()
        ax4.grid()
        plt.show()

    increaseFilterSize()

    '''
    The very best parameters are:
    Batch size: 16
    Filter size: 32
    Kernel size: 3
    Pool size: 2
    Training epochs: 15
    Learning rate: 0.0001
    Momentum: 0.9
    
    Test accuracy gets to about 60% after 10-12 training epochs, at which point it tapers off
    
    '''
