# Batch Normalization (BN)

from keras.datasets import cifar10
from keras import models, layers
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train, x_test = x_train/255, x_test/255

net = models.Sequential([
                        layers.Conv2D(32, (3,3), activation="relu", input_shape=(32, 32, 3)),
                        layers.BatchNormalization(),
                        layers.Conv2D(32, (3,3), activation="relu"),
                        layers.BatchNormalization(),
                        layers.MaxPool2D(),

                        layers.Conv2D(64, (3,3), activation="relu", input_shape=(32, 32, 3)),
                        layers.BatchNormalization(),
                        layers.Conv2D(64, (3,3), activation="relu"),
                        layers.BatchNormalization(),
                        layers.MaxPool2D(),

                        layers.Flatten(),
                        layers.Dense(512, activation="relu"),
                        layers.BatchNormalization(),
                        layers.Dense(10, activation="softmax")
                        ])

net.compile(optimizer="sgd",
            metrics=["accuracy"],
            loss=["sparse_categorical_crossentropy"])

H = net.fit(x_train, y_train, batch_size=16, epochs=25, validation_data=(x_test, y_test))

plt.style.use("ggplot")
plt.plot(H.history["accuracy"], label="train")
plt.plot(H.history["val_accuracy"], label="test")
plt.plot(H.history["loss"], label="train loss")
plt.plot(H.history["val_loss"], label="test loss")
plt.legend()
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.title("Batch Normalization") 
plt.savefig("With_BN.png")
plt.show()