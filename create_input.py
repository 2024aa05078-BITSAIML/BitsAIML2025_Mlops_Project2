import json
import numpy as np

# Model expects (1, 128, 128, 3)
x = np.random.rand(1, 128, 128, 3).astype("float32")

payload = {
    "instances": x.tolist()
}

with open("input.json", "w") as f:
    json.dump(payload, f)

print("input.json regenerated correctly")
