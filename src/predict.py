import requests
import numpy as np
from tensorflow.keras.preprocessing import image
import json

# API endpoint
URL = "http://127.0.0.1:5001/invocations"

# Load and preprocess image
img = image.load_img("data/raw/cats/1.jpg", target_size=(128, 128))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Prepare payload
payload = {
    "inputs": img_array.tolist()
}

# Send request
response = requests.post(
    URL,
    data=json.dumps(payload),
    headers={"Content-Type": "application/json"}
)

print("Prediction:", response.json())
