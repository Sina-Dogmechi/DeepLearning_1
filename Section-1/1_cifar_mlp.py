# MLP --> Multi Layer Percepteron

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from keras import models, layers
from keras.datasets import cifar10
import matplotlib.pyplot as plt


def load_data_preprocessing():

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # print(x_train.shape)

    x_train , x_test = x_train/255, x_test/255

    return x_train, x_test, y_train, y_test 


def neural_network():
    
    net = models.Sequential([
                            layers.Flatten(),
                            layers.Dense(400, activation = "relu"),
                            layers.Dense(200, activation = "relu"),
                            layers.Dense(80, activation = "relu"),
                            layers.Dense(10, activation = "softmax"),
                        ])


    net.compile(optimizer="SGD",
                metrics= ["accuracy"],
                loss="sparse_categorical_crossentropy")
    
    return H


def show_results(H):

    plt.plot(H.history['accuracy'], label='train accuracy')
    plt.plot(H.history['val_accuracy'], label='test accuracy')
    plt.plot(H.history['loss'], label='train loss')
    plt.plot(H.history['val_loss'], label='test loss')
    plt.xlabel('epochs')
    plt.ylabel('accuracy/loss')
    plt.title('cifar classification')
    plt.legend()
    plt.show()


x_train, x_test, y_train, y_test = load_data_preprocessing()

net = neural_network()

H = net.fit(x_train, y_train, batch_size=32, validation_data=(x_test, y_test), epochs=15)

show_results(H)