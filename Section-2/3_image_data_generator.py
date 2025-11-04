import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# from keras.preprocessing.image import ImageDataGenerator
from keras.src.legacy.preprocessing.image import ImageDataGenerator  
import cv2
import numpy as np

img = cv2.imread("man.jfif")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.array([img])

aug = ImageDataGenerator(rotation_range = 20,
                         width_shift_range = 0.1,
                         height_shift_range = 0.1,
                         shear_range = 0.2,
                         zoom_range = 0.2,
                         vertical_flip = True,
                         fill_mode = "nearest")

imageGen = aug.flow(img, batch_size=1, save_to_dir="out", save_format="jpg", save_prefix="cv")

total = 0

for image in imageGen:
    total += 1

    if total == 10: break   