import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
import glob
from keras.models import load_model


model = load_model("cnn.keras")

for item in glob.glob("dataset\\test_images\\*"):

    img = cv2.imread(item)

    # r_img = cv2.resize(img, (32, 32)).flatten()
    r_img = cv2.resize(img, (32, 32))
    r_img = r_img/255.0
    r_img = np.array([r_img])
    output = model.predict(r_img)[0]
    max_output = np.argmax(output)

    category_name = ['fire', 'non fire']

    text = "{}: {:.2f} %".format(category_name[max_output], output[max_output]*100)
    cv2.putText(img, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
    cv2.imshow("image", img)
    cv2.waitKey(0)

cv2.destroyAllWindows()