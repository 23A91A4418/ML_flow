from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data(dataset_name="breast_cancer", test_size=0.2, random_state=42):
    """
    Loads a classification dataset and returns scaled train/test splits + scaler + feature names.
    """
    if dataset_name == "breast_cancer":
        data = load_breast_cancer(as_frame=True)
        X = data.data
        y = data.target
    else:
        raise ValueError("Only 'breast_cancer' dataset is supported right now.")

    # Train-test split (reproducible)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, list(X.columns)
