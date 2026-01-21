import os
import joblib
import pandas as pd
import mlflow
from flask import Flask, request, jsonify
from mlflow.tracking import MlflowClient

app = Flask(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "ClassificationModel")

model = None
scaler = None


def load_model_and_scaler():
    """
    Loads:
    - latest registered model from MLflow Model Registry
    - scaler.pkl from the same run artifacts (preprocessing/scaler.pkl)
    """
    global model, scaler

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


    # Get latest version of the registered model
    latest_versions = client.get_latest_versions(REGISTERED_MODEL_NAME)
    if not latest_versions:
        raise RuntimeError(f"No versions found for model: {REGISTERED_MODEL_NAME}")

    latest = latest_versions[0]
    run_id = latest.run_id

    # Load model from registry
    model_uri = f"models:/{REGISTERED_MODEL_NAME}/{latest.version}"
    model = mlflow.pyfunc.load_model(model_uri)

    # Download scaler from the same run artifacts
    local_scaler_path = client.download_artifacts(run_id, "preprocessing/scaler.pkl")
    scaler = joblib.load(local_scaler_path)

    print(f"✅ Loaded model: {REGISTERED_MODEL_NAME} v{latest.version} (run_id={run_id})")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    global model, scaler

    if model is None or scaler is None:
        return jsonify({"error": "Model/scaler not loaded"}), 500

    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Invalid JSON payload"}), 400

    if "features" not in payload:
        return jsonify({"error": "Missing 'features' in request body"}), 400

    features = payload["features"]

    if not isinstance(features, list) or len(features) == 0:
        return jsonify({"error": "'features' must be a non-empty list"}), 400

    try:
        # Accept: list of dicts OR list of lists
        if isinstance(features[0], dict):
            df = pd.DataFrame(features)
        else:
            df = pd.DataFrame(features)

        X_scaled = scaler.transform(df)
        preds = model.predict(X_scaled)

        return jsonify({"predictions": preds.tolist()}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    load_model_and_scaler()
    app.run(host="0.0.0.0", port=8000, debug=True)
