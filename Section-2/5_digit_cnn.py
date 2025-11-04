import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras import layers, models
import matplotlib.pyplot as plt
plt.style.use("ggplot")


def load_data_preprocess(dataset):
    
    all_images = []
    all_labels = []

    for i, item in enumerate(glob.glob(dataset)):

        img = cv2.imread(item)
        img = cv2.resize(img, (32, 32))
        img = img/255.0

        all_images.append(img)

        label = item.split("\\")[-2]
        # print(label)
        all_labels.append(label)

        if i % 100 == 0:
            print("[Info] {}/2012 processed".format(i))

    all_images = np.array(all_images)
    # print(all_images.shape)

    lb = LabelBinarizer()
    all_labels = lb.fit_transform(all_labels)

    trainX, testX, trainy, testy = train_test_split(all_images, all_labels, test_size=0.2)

    return trainX, testX, trainy, testy



def miniCNN():

    net = models.Sequential([
                            layers.Conv2D(32, (3,3), activation="relu", padding="same" ,input_shape=(32, 32, 3)),
                            layers.MaxPool2D((2,2)),
                            layers.Conv2D(64, (3,3), activation="relu", padding="same"),
                            layers.MaxPool2D((2,2)),
                            layers.Flatten(),
                            layers.Dense(32, activation="relu"),
                            layers.Dense(9, activation="softmax")
                            ])

    net.compile(loss = "categorical_crossentropy",
                optimizer = "sgd",
                metrics = ["accuracy"])
    
    return net


def show_learning_curve(H):

    plt.plot(H.history["accuracy"], label="train accuracy")
    plt.plot(H.history["val_accuracy"], label="test accuracy")
    plt.plot(H.history["loss"], label="train loss")
    plt.plot(H.history["val_loss"], label="test loss")
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("loss/accuracy")
    plt.title("digits classification")
    plt.show()


trainX, testX, trainy, testy = load_data_preprocess("dataset\\*\\*\\*")
# print(trainy[0])
# print(trainX.shape)

net = miniCNN()
# print(net.summary())

# Training...
H = net.fit(x=trainX, y=trainy, epochs=20, batch_size=32, validation_data = (testX, testy))

show_learning_curve(H)

net.save("digit_classifier.keras")