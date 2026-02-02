import mlflow
from mlflow.tracking import MlflowClient

# -----------------------------
# Configuration
# -----------------------------
EXPERIMENT_NAME = "Bits_MLOps_Project2"
MODEL_NAME = "Bits_CNN_Model"

# -----------------------------
# Set MLflow tracking URI
# -----------------------------
mlflow.set_tracking_uri("file:./mlruns")
client = MlflowClient()

# -----------------------------
# Get latest successful run
# -----------------------------
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["attributes.start_time DESC"],
    max_results=1
)

if len(runs) == 0:
    raise Exception("❌ No runs found to register")

run_id = runs[0].info.run_id
print(f"✅ Using run_id: {run_id}")

# -----------------------------
# Register the model
# -----------------------------
model_uri = f"runs:/{run_id}/model"

result = mlflow.register_model(
    model_uri=model_uri,
    name=MODEL_NAME
)

print(f"✅ Model registered as '{MODEL_NAME}', version {result.version}")

# -----------------------------
# Move model to STAGING
# -----------------------------
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=result.version,
    stage="Staging",
    archive_existing_versions=True
)

print("🚀 Model moved to STAGING")
