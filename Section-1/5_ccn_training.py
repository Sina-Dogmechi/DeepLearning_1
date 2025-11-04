import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras import models, layers


data = []
labels = []

for i, item in enumerate(glob.glob("dataset\\fire_dataset\\*\\*")):

    img = cv2.imread(item)

    img = cv2.resize(img, (32, 32))
    img = img / 255

    data.append(img)

    label = item.split("\\")[-1].split(".")[0]
    labels.append(label)

    if i % 100 == 0:
        print(f"[INFO] {i}/{1000} processed")


data = np.array(data)
# print(data.shape)


le = LabelEncoder()
labels = le.fit_transform(labels)
labels = to_categorical(labels)


x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

net = models.Sequential([

                        layers.Conv2D(32, (3,3), activation = "relu", input_shape = (32, 32, 3)),
                        layers.MaxPool2D(),
                        layers.Conv2D(32, (3,3), activation = "relu"),
                        layers.MaxPool2D(),
                        layers.Flatten(),
                        layers.Dense(100, activation = "relu"),
                        layers.Dense(2, activation = "sigmoid")
                        ])

net.compile(optimizer='SGD',
            loss='binary_crossentropy',
            metrics=['accuracy'])

H = net.fit(x_train, y_train, batch_size=32, epochs=15, validation_data=(x_test, y_test))

loss, acc = net.evaluate(x_test, y_test)
print("loss: {:.2f}, acc: {:.2f}".format(loss, acc))

net.save("cnn.keras")


plt.style.use('ggplot')
plt.plot(H.history['accuracy'], label='train accuracy')
plt.plot(H.history['val_accuracy'], label='test accuracy')
plt.plot(H.history['loss'], label='train loss')
plt.plot(H.history['val_loss'], label='test loss')
plt.title('Fire/None fire dataset')
plt.xlabel('epochs')
plt.ylabel('accuracy/loss')
plt.legend()
plt.show()