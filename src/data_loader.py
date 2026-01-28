import numpy as np
import os


def load_data(processed_data_dir):
    X_train = np.load(os.path.join(processed_data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(processed_data_dir, "y_train.npy"))
    X_val = np.load(os.path.join(processed_data_dir, "X_val.npy"))
    y_val = np.load(os.path.join(processed_data_dir, "y_val.npy"))
    X_test = np.load(os.path.join(processed_data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(processed_data_dir, "y_test.npy"))

    return X_train, y_train, X_val, y_val, X_test, y_test
