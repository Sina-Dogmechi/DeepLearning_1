# Transfer Learning

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.applications import VGG16
from keras.optimizers import Adam, SGD
from keras import models, layers

data = []
labels = []

for item in glob.glob("dataset\\covid19-dataset\\*\\*"):

    img = cv2.imread(item)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    r_img = cv2.resize(img, (224, 224))
    data.append(r_img)

    label = item.split("\\")[-2]
    labels.append(label)


le = LabelBinarizer()
labels = le.fit_transform(labels)
labels = to_categorical(labels, 2)

data = np.array(data)/255

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.15, random_state=42)


aug = ImageDataGenerator(rotation_range=10,
                         fill_mode="nearest")

baseModel = VGG16(weights = "imagenet",
                  include_top = False,
                  input_tensor = layers.Input(shape=(224, 224, 3)))

for layer in baseModel.layers:
    
    layer.trainable = False


network = models.Sequential([
                            baseModel,
                            layers.MaxPool2D((4,4)),
                            layers.Flatten(),
                            layers.Dense(64, activation = "relu"),
                            # layers.Dropout(0.5),
                            layers.Dense(2, activation = "softmax")
                            ])

# opt = SGD(learning_rate = 0.001, weight_decay = 0.00025)
opt = Adam(learning_rate = 0.001, weight_decay = 0.001/25)

network.compile(optimizer = opt,
                loss = "binary_crossentropy",
                metrics = ["accuracy"])

H = network.fit(aug.flow(x_train, y_train, batch_size = 8),
                steps_per_epoch = len(x_train)//8,
                validation_data = (x_test, y_test),
                epochs = 25)

plt.style.use('ggplot')
plt.plot(np.arange(25), H.history["accuracy"], label = "acc")
plt.plot(np.arange(25), H.history["val_accuracy"], label = "val_acc")
plt.plot(np.arange(25), H.history["loss"], label = "loss")
plt.plot(np.arange(25), H.history["val_loss"], label = "val_loss")

plt.title("Covid 19")
plt.xlabel("epochs")
plt.ylabel("accuracy/loss")
plt.legend()
plt.show()
