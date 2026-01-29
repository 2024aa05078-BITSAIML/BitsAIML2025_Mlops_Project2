import os
import tensorflow as tf
import mlflow
import mlflow.tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ===============================
# SAFE CONFIG (CPU)
# ===============================
IMG_SIZE = (128, 128)
BATCH_SIZE = 4
EPOCHS = 5
SEED = 42

# ===============================
# PATHS
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "processed")
MLRUNS_DIR = os.path.join(BASE_DIR, "..", "mlruns")

TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")

# ===============================
# MLflow
# ===============================
mlflow.set_tracking_uri(f"file:///{MLRUNS_DIR.replace(os.sep, '/')}")
mlflow.set_experiment("Bits_MLOps_Project2")

# ===============================
# DATA GENERATORS
# ===============================
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    seed=SEED
)

val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    seed=SEED
)

# ===============================
# MODEL
# ===============================
model = Sequential([
    Input(shape=(128, 128, 3)),

    Conv2D(16, 3, activation="relu"),
    MaxPooling2D(),

    Conv2D(32, 3, activation="relu"),
    MaxPooling2D(),

    Flatten(),
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ===============================
# TRAIN + LOG
# ===============================
with mlflow.start_run(run_name="cats_vs_dogs_cpu"):

    mlflow.log_param("img_size", IMG_SIZE)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("optimizer", "adam")

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen
    )

    mlflow.log_metric("val_accuracy", history.history["val_accuracy"][-1])
    mlflow.log_metric("val_loss", history.history["val_loss"][-1])

    mlflow.tensorflow.log_model(model, "model")

print("✅ Training completed successfully")
print("✅ MLflow run logged")
