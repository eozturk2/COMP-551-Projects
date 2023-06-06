import MLP_Polymorphic

if __name__ == "__main__":
    train_images, train_labels, test_images, test_labels = MLP_Polymorphic.loadCIFAR10()
    simple = MLP_Polymorphic.MLP(MLP_Polymorphic.softmax, [3072, 10])
    one_layer = MLP_Polymorphic.MLP(MLP_Polymorphic.relu, [3072, 256, 10])
    predictions = one_layer.fit(train_images, train_labels, 30, 0.000001,
                             validation_input=test_images,
                             validation_labels=test_labels)

