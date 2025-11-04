# Batch Normalization (BN)
"""
vaghti augmentation ezafe mashe,
tabee 'sparse_categorical_crossentropy'
dorost kar nemikone, pas bayad khodemun
one-hot anjam bedim.
"""

from keras.datasets import cifar10
from keras import models, layers
import matplotlib.pyplot as plt
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer


aug = ImageDataGenerator(rotation_range = 20,
                         width_shift_range = 0.1,
                         height_shift_range = 0.1,
                         shear_range = 0.2,
                         zoom_range = 0.2,
                         vertical_flip = True,
                         fill_mode = "nearest")


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train, x_test = x_train/255, x_test/255

le = LabelBinarizer()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

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
            loss=["categorical_crossentropy"])

H = net.fit(aug.flow(x_train, y_train, batch_size=16),
            # condition for stop generator
            steps_per_epoch= len(x_train) // 32,
            epochs=25,
            validation_data=(x_test, y_test))

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