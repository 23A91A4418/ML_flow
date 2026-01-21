import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.data_processor import load_and_preprocess_data


def save_confusion_matrix_plot(y_true, y_pred, out_path="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # Add numbers inside matrix
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def train_model(dataset_name="breast_cancer", C=1.0, penalty="l2", random_state=42):
    # Load and preprocess data
    X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess_data(
        dataset_name=dataset_name,
        test_size=0.2,
        random_state=random_state
    )

    # Create model
    model = LogisticRegression(
        C=C,
        penalty=penalty,
        solver="liblinear",
        random_state=random_state,
        max_iter=200
    )

    # Start MLflow run
    with mlflow.start_run(run_name=f"logreg_C={C}_penalty={penalty}"):

        # Log parameters
        mlflow.log_param("dataset_name", dataset_name)
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("C", C)
        mlflow.log_param("penalty", penalty)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("max_iter", 200)

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # Create artifacts folder
        os.makedirs("temp_artifacts", exist_ok=True)

        # Classification report artifact
        report_text = classification_report(y_test, y_pred)
        report_path = os.path.join("temp_artifacts", "classification_report.txt")
        with open(report_path, "w") as f:
            f.write(report_text)
        mlflow.log_artifact(report_path, artifact_path="reports")

        # Confusion matrix plot artifact
        cm_path = os.path.join("temp_artifacts", "confusion_matrix.png")
        save_confusion_matrix_plot(y_test, y_pred, out_path=cm_path)
        mlflow.log_artifact(cm_path, artifact_path="plots")

        # Save scaler artifact
        scaler_path = os.path.join("temp_artifacts", "scaler.pkl")
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(scaler_path, artifact_path="preprocessing")

        # Log model + register it
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="ClassificationModel"
        )

        print(f"✅ Run logged: C={C}, penalty={penalty}, F1={f1:.4f}")
        return f1


if __name__ == "__main__":
    # MUST come from docker-compose environment
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise RuntimeError("❌ MLFLOW_TRACKING_URI is not set. Check docker-compose.yml")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("BreastCancer_Classification")

    # 3 runs with different hyperparameters
    train_model(C=0.1, penalty="l2")
    train_model(C=1.0, penalty="l2")
    train_model(C=10.0, penalty="l2")

    print("\n✅ Training runs completed.")
