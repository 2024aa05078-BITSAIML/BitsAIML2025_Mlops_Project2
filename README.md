# BitsAIML2025 – MLOps Project (Module 1 & 2)

This repository contains an end-to-end **MLOps pipeline** implemented as part of the BITS AIML M.Tech coursework.
The project covers **data preprocessing, model training, experiment tracking, and model registration using MLflow**.

---

## Project Structure

```
BitsAIML2025_Mlops_Project2/
│
├── data/
│   ├── raw/                # Original dataset
│   ├── processed/          # Preprocessed train/val data
│
├── src/
│   ├── preprocess.py       # Module 1: Data preprocessing
│   ├── train.py            # Module 2: Model training + MLflow logging
│   ├── register_model.py   # Module 3: Model registration
│
├── mlruns/                 # MLflow experiment artifacts (auto-generated)
├── requirements.txt
└── README.md
```

---

## Environment Setup (Windows – PowerShell)

### Create and activate virtual environment

```powershell
python -m venv venv
venv\Scripts\activate
```

### Install dependencies

```powershell
pip install -r requirements.txt
```

> Note: Project is configured to run on **CPU-only systems**

---

## Module 1 – Data Preprocessing

### Objective

* Load raw image dataset
* Resize images
* Split into Train / Validation sets
* Save processed data to disk

### Run

```powershell
python src/preprocess.py
```

### Expected Output

```
Data preprocessing completed successfully!
```

Processed data will be available under:

```
data/processed/
```

---

## Module 2 – Model Training + MLflow Tracking

### Configuration (CPU Safe)

The training configuration is **intentionally kept small** to avoid CPU crashes:

```python
IMG_SIZE = (128, 128)
BATCH_SIZE = 4
EPOCHS = 5
```

### What happens in this module

* CNN model is trained
* Metrics logged to MLflow:

  * training accuracy
  * validation accuracy
  * training loss
  * validation loss
* Trained model logged as MLflow artifact

### Run

```powershell
python src/train.py
```

### Sample Output

```
val_accuracy: 0.76
val_loss: 0.49
```

### View MLflow UI

```powershell
mlflow ui
```

Open browser:

```
http://127.0.0.1:5000
```

---

## Module 3 – Model Registration (MLflow Model Registry)

### Objective

* Register best trained model
* Move model to **STAGING** stage

### Run

```powershell
python src/register_model.py
```

### Sample Output

```
Model registered as 'Bits_CNN_Model', version 1
Model moved to STAGING
```

> Warning related to model registry stages is expected due to MLflow deprecation notice

---

## Current Status

* ✔ Module 1: Data Preprocessing – Completed
* ✔ Module 2: Model Training & Tracking – Completed
* ✔ Module 3: Model Registration – Completed

---

## Next Steps (Planned)

* Module 4: Model Serving
* Module 5: CI/CD Integration
* Module 6: Monitoring & Logging

---

## Course Context

* **Program:** BITS Pilani – M.Tech AIML
* **Course:** MLOps
* **Semester:** SEM 3

---

This README is written to allow **fresh execution from scratch** on a new machine.
