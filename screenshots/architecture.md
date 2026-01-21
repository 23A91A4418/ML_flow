# ARCHITECTURE.md

## High-Level Architecture (MLflow + Dockerized Inference API)

This project is a simple end-to-end ML workflow that includes:

- Training a Logistic Regression model on the Breast Cancer dataset
- Tracking experiments (params, metrics, artifacts) using MLflow
- Registering the trained model in the MLflow Model Registry
- Serving the latest registered model using a Flask-based inference API
- Running all components using Docker Compose

---

## Components

### 1) MLflow Server (`mlflow_server`)
Purpose:
- Hosts the MLflow Tracking UI
- Stores experiments, runs, parameters, and metrics
- Stores artifacts (plots, reports, scaler)
- Maintains the Model Registry (registered models + versions)

Exposed Port:
- `5000` → MLflow UI available at: `http://localhost:5000`

Storage:
- Uses a SQLite backend store (mlflow.db)
- Uses a local artifacts folder (mounted volume)

---

### 2) Trainer Service (`trainer`)
Purpose:
- Runs the training script (`src/model_trainer.py`)
- Trains the model multiple times with different hyperparameters
- Logs parameters, metrics, and artifacts to MLflow
- Registers the trained model as `ClassificationModel`

Behavior:
- Runs once and exits after training is completed

---

### 3) Model API (`model_api`)
Purpose:
- Runs the inference server (`src/inference_api.py`)
- Loads the latest version of the model from MLflow Model Registry
- Downloads the scaler artifact from the same run
- Provides HTTP endpoints:
  - `GET /health`
  - `POST /predict`

Exposed Port:
- `8000` → API available at: `http://localhost:8000`

---

## Data and Model Flow

1. `trainer` starts and trains the model
2. `trainer` logs:
   - Parameters (C, penalty, dataset_name, etc.)
   - Metrics (accuracy, precision, recall, f1_score)
   - Artifacts (confusion matrix, classification report, scaler.pkl)
3. `trainer` registers the model in MLflow Model Registry as:
   - `ClassificationModel`
4. `model_api` loads:
   - Latest model version from MLflow Registry
   - `preprocessing/scaler.pkl` from MLflow run artifacts
5. Client calls the API for predictions using `/predict`

---

## Architecture Diagram

                 Docker Compose Network
+-----------------------------------------------------------------------+
|                        Docker Compose Network                          |
+-----------------------------------------------------------------------+

      +---------------------------+          HTTP           +---------------------------+
      |          trainer          |------------------------>|       mlflow_server       |
      |     (model_trainer.py)    |   (MLFLOW_TRACKING_URI) |     (Tracking + UI)       |
      +---------------------------+                         +---------------------------+
      | - trains model            |                         | - experiments             |
      | - logs params             |                         | - metrics                 |
      | - logs metrics            |                         | - artifacts               |
      | - logs artifacts          |                         | - model registry          |
      | - registers model         |                         +-------------+-------------+
      +---------------------------+                                       |
                                                                          |
                                                                          | HTTP (Registry + Artifacts)
                                                                          v
                                                           +---------------------------+
                                                           |         model_api          |
                                                           |     (inference_api.py)     |
                                                           +---------------------------+
                                                           | - loads latest model       |
                                                           | - downloads scaler.pkl     |
                                                           | - exposes /health          |
                                                           | - exposes /predict         |
                                                           +-------------+-------------+
                                                                         |
                                                                         | Port 8000
                                                                         v
                                                                Client / User Requests

---

## Docker Compose Service Dependency Order

- `mlflow_server` must start first
- `trainer` depends on `mlflow_server` (logs runs and registers model)
- `model_api` depends on both:
  - `mlflow_server` (loads model + scaler)
  - `trainer` (ensures model is registered before API tries to load it)

---

## Key Design Decisions

### Why MLflow?
- Central place to track experiments and compare results
- Logs everything needed to reproduce training
- Model Registry provides versioning so inference always uses the latest model

### Why log the scaler as an artifact?
- Ensures preprocessing is consistent between training and inference
- Prevents mismatch between training-time scaling and prediction-time scaling

### Why Docker Compose?
- Runs all services consistently on any machine
- Makes setup reproducible without manual installation steps
- Ensures trainer + API can communicate with MLflow reliably

---

## Ports Summary

- MLflow UI: `http://localhost:5000`
- Inference API: `http://localhost:8000`
