# Functional API for GANs

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from keras.datasets import mnist
from keras import models, layers
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train/255, x_test/255

"""
# MLP
net = models.Sequential([
                        layers.Flatten(),
                        layers.Dense(200, activation="relu"),
                        layers.Dense(80, activation="relu"),
                        layers.Dense(10, activation="softmax")
                        ])
"""

# (60000, 28, 28) ---> convolution should take 3-dimension as input
# print(x_train.shape)
x_train = x_train.reshape((len(x_train), 28, 28, 1))
x_test = x_test.reshape((len(x_test), 28, 28, 1))

"""
# Convolution
net = models.Sequential([
                        layers.Conv2D(32, (3,3), activation="relu", padding="same"),
                        layers.MaxPool2D((2,2)),
                        layers.Conv2D(64, (3,3), activation="relu", padding="same"),
                        layers.MaxPool2D((2,2)),
                        layers.Flatten(),
                        layers.Dense(80, activation="relu"),
                        layers.Dense(10, activation="softmax")
                        ])
"""

# Finctional API
input_layer = layers.Input(shape=(28, 28, 1))

x = layers.Conv2D(32, (3,3), strides=(2,2))(input_layer)
x = layers.LeakyReLU(negative_slope=0.1)(x)
# x = layers.MaxPool2D((2,2))(x)
x = layers.Conv2D(32, (3,3), strides=(2,2))(x)
x = layers.LeakyReLU(negative_slope=0.1)(x)
# x = layers.MaxPool2D((2,2))(x)
x = layers.Flatten()(x)
x = layers.Dense(80)(x)
x = layers.LeakyReLU(negative_slope=0.1)(x)
output_layer = layers.Dense(10, activation="softmax")(x)

net = models.Model(inputs = input_layer, outputs = output_layer)

net.summary()

net.compile(optimizer="adam",
            metrics=["accuracy"],
            loss="sparse_categorical_crossentropy")

H = net.fit(x_train, y_train, batch_size=16, epochs=5, validation_data=(x_test, y_test))

plt.style.use('ggplot')
plt.plot(H.history["accuracy"], label="train")
plt.plot(H.history["val_accuracy"], label="test")
plt.plot(H.history["loss"], label="train loss")
plt.plot(H.history["val_loss"], label="test loss")
plt.legend()
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.title('digit detection')
plt.show()