import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from mtcnn import MTCNN
import cv2
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from keras.utils import to_categorical
from keras import layers, models
import matplotlib.pyplot as plt

detector = MTCNN()

def detect_faces(img):

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = detector.detect_faces(rgb_img)[0]
    x, y, w, h = out["box"]

    return rgb_img[y:y+h, x:x+w]


all_faces = []
all_labels = []
for i, item in enumerate(glob.glob("dataset\\smile_dataset\\*\\*")):

    img = cv2.imread(item)
    try:
        face = detect_faces(img)

        """cv2.imshow("image", face)
        if cv2.waitKey(0) == ord("q"):
            break"""
        
        face = cv2.resize(face, (32, 32))
        face = face/255.0

        all_faces.append(face)

        label = item.split("\\")[-2]
        all_labels.append(label)
    except:
        pass

    if i % 100 == 0:
        print("[INFO] {}/4000 processed".format(i))

all_faces = np.array(all_faces)

lb = LabelBinarizer()
all_labels = lb.fit_transform(all_labels)

# le = LabelEncoder()
# all_labels_le = le.fit_transform(all_labels)
# all_labels_le = to_categorical(all_labels_le)

trainX, testX, trainy, testy = train_test_split(all_faces, all_labels, test_size=0.2)


net = models.Sequential([
                        layers.Conv2D(32, (3,3), activation="relu", padding="same" ,input_shape=(32, 32, 3)),
                        layers.Conv2D(32, (3,3), activation="relu", padding="same" ,input_shape=(32, 32, 3)),
                        layers.MaxPool2D((2,2)),

                        layers.Conv2D(64, (3,3), activation="relu", padding="same"),
                        layers.Conv2D(64, (3,3), activation="relu", padding="same"),
                        layers.MaxPool2D((2,2)),

                        layers.Flatten(),
                        layers.Dense(32, activation="relu"),
                        layers.Dense(1, activation="sigmoid")
                        ])

net.compile(loss = "binary_crossentropy",
            optimizer = "sgd",
            metrics = ["accuracy"])

H = net.fit(x=trainX, y=trainy, epochs=20, batch_size=32, validation_data = (testX, testy))

plt.plot(H.history["accuracy"], label="train accuracy")
plt.plot(H.history["val_accuracy"], label="test accuracy")
plt.plot(H.history["loss"], label="train loss")
plt.plot(H.history["val_loss"], label="test loss")
plt.legend()
plt.xlabel("epochs")
plt.ylabel("loss/accuracy")
plt.title("digits classification")
plt.show()