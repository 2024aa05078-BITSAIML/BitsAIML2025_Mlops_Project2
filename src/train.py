import os
import mlflow
import mlflow.tensorflow
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from data_loader import load_data
from model import build_model


# -------------------------
# MLflow Configuration
# -------------------------
mlflow.set_experiment("Cats_vs_Dogs_Baseline")

PROCESSED_DATA_DIR = "data/processed"
MODEL_DIR = "models"


# -------------------------
# Utility: Confusion Matrix Plot
# -------------------------
def plot_confusion_matrix(cm, filename="confusion_matrix.png"):
    plt.figure(figsize=(4, 4))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# -------------------------
# Training Pipeline
# -------------------------
if __name__ == "__main__":

    # Load preprocessed data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(PROCESSED_DATA_DIR)

    # Hyperparameters
    learning_rate = 0.001
    epochs = 5
    batch_size = 32

    with mlflow.start_run():

        # Log parameters
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("input_size", "224x224 RGB")
        mlflow.log_param("augmentation", "rotation, shift, flip")

        # Data Augmentation (TRAIN ONLY)
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True
        )

        train_datagen.fit(X_train)

        # Build model
        model = build_model(learning_rate=learning_rate)

        # Train model
        history = model.fit(
            train_datagen.flow(X_train, y_train, batch_size=batch_size),
            validation_data=(X_val, y_val),
            epochs=epochs
        )

        # Evaluate on test set
        test_loss, test_accuracy = model.evaluate(X_test, y_test)

        # Log metrics
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_accuracy)

        # Predictions for confusion matrix
        y_pred = (model.predict(X_test) > 0.5).astype(int)

        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm)

        # Log confusion matrix
        mlflow.log_artifact("confusion_matrix.png")

        # Save model
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_DIR, "cats_dogs_model.h5")
        model.save(model_path)

        # Log model artifact
        mlflow.log_artifact(model_path)
