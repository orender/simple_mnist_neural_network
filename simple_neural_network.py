import matplotlib.pyplot as plt
import numpy as np

# Loads the images


def get_mnist():
    with np.load("mnist.npz") as f:
        mnist_images, mnist_labels = f["x_train"], f["y_train"]
    mnist_images = mnist_images.astype("float32") / 255
    mnist_images = np.reshape(mnist_images, (mnist_images.shape[0], mnist_images.shape[1] * mnist_images.shape[2]))
    mnist_labels = np.eye(10)[mnist_labels]
    return mnist_images, mnist_labels


"""
w = weights, b = bias, i = input, h = hidden, o = output, l = label
e.g. weights_Input_hidden = weights from input layer to hidden layer
"""

images, labels = get_mnist()

weights_Input_hidden = np.random.uniform(-0.5, 0.5, (100, 784))

weights_hidden_output = np.random.uniform(-0.5, 0.5, (10, 100))

bias_input_hidden = np.zeros((100, 1))

bias_hidden_output = np.zeros((10, 1))


learn_rate = 0.01

nr_correct = 0

epochs = 3

# this is the learning part of the neural network!!!
for epoch in range(epochs):
    for img, l in zip(images, labels):
        img.shape += (1,)
        l.shape += (1,)

        hidden_before_active = bias_input_hidden + weights_Input_hidden @ img
        hidden = 1 / (1 + np.exp(-hidden_before_active))

        output_before_active = bias_hidden_output + weights_hidden_output @ hidden
        output = 1 / (1 + np.exp(-output_before_active))

        e = 1 / len(output) * np.sum((output - l) ** 2, axis=0)
        nr_correct += int(np.argmax(output) == np.argmax(l))

        delta_o = output - l
        weights_hidden_output += -learn_rate * delta_o @ np.transpose(hidden)
        bias_hidden_output += -learn_rate * delta_o

        delta_h = np.transpose(weights_hidden_output) @ delta_o * (hidden * (1 - hidden))
        weights_Input_hidden += -learn_rate * delta_h @ np.transpose(img)
        bias_input_hidden += -learn_rate * delta_h

    print(f"Acc: {round((nr_correct / images.shape[0]) * 100, 2)}%")
    nr_correct = 0

# this is where the user can use the AI to see it function
while True:
    index = int(input("Enter a number (0 - 59999): "))
    img = images[index]
    plt.imshow(img.reshape(28, 28), cmap="Greys")

    img.shape += (1,)

    hidden_before_active = bias_input_hidden + weights_Input_hidden @ img.reshape(784, 1)
    h = 1 / (1 + np.exp(-hidden_before_active))

    o_pre = bias_hidden_output + weights_hidden_output @ h
    o = 1 / (1 + np.exp(-o_pre))

    plt.title(f"the number is a {o.argmax()}")
    plt.show()
