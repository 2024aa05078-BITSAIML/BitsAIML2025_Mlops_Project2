import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import shutil

IMG_SIZE = 224
RANDOM_STATE = 42


def preprocess_and_split(raw_data_dir, processed_data_dir):
    images = []
    labels = []

    for label, class_name in enumerate(["cats", "dogs"]):
        class_path = os.path.join(raw_data_dir, class_name)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0
            images.append(img)
            labels.append(label)

    X = np.array(images)
    y = np.array(labels)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp
    )

    np.save(os.path.join(processed_data_dir, "X_train.npy"), X_train)
    np.save(os.path.join(processed_data_dir, "y_train.npy"), y_train)
    np.save(os.path.join(processed_data_dir, "X_val.npy"), X_val)
    np.save(os.path.join(processed_data_dir, "y_val.npy"), y_val)
    np.save(os.path.join(processed_data_dir, "X_test.npy"), X_test)
    np.save(os.path.join(processed_data_dir, "y_test.npy"), y_test)
