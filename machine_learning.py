"""
Trains a classification model to predict whether a chatbot conversation
will be SUCCESSFUL or FAILED, based on numeric conversation features.

"""

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)

FEATURE_COLUMNS = [
    "num_messages", "conversation_duration", "response_time",
    "user_message_length", "response_length", "satisfaction_score"
]
TARGET_COLUMN = "conversation_status"
MODEL_PATH = "models/chatbot_model.pkl"


def prepare_features(df: pd.DataFrame):
    data = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).copy()

    X = data[FEATURE_COLUMNS]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data[TARGET_COLUMN])  # Successful/Failed -> 1/0

    return X, y, label_encoder


def train_and_evaluate(df: pd.DataFrame, test_size: float = 0.25, random_state: int = 42) -> dict:
    """
    Trains Logistic Regression AND Random Forest, evaluates both on a
    held-out test set, and returns a comparison plus the better model.
    """
    X, y, label_encoder = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Feature scaling helps Logistic Regression converge properly
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=random_state)
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            predictions = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

        results[name] = {
            "accuracy": round(accuracy_score(y_test, predictions), 4),
            "precision": round(precision_score(y_test, predictions, zero_division=0), 4),
            "recall": round(recall_score(y_test, predictions, zero_division=0), 4),
            "f1_score": round(f1_score(y_test, predictions, zero_division=0), 4),
            "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
            "classification_report": classification_report(y_test, predictions,
                                                              target_names=label_encoder.classes_,
                                                              zero_division=0)
        }
        trained_models[name] = model

    # Pick the best model based on F1 score (balances precision and recall)
    best_model_name = max(results, key=lambda name: results[name]["f1_score"])
    best_model = trained_models[best_model_name]

    # Persist the best model + preprocessing objects to disk (pickle -> models/chatbot_model.pkl)
    bundle = {
        "model": best_model,
        "model_name": best_model_name,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "feature_columns": FEATURE_COLUMNS,
        "needs_scaling": best_model_name == "Logistic Regression"
    }
    try:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(bundle, f)
    except (IOError, OSError) as e:
        print(f"Warning: could not save model to disk: {e}")

    return {
        "results": results,
        "best_model_name": best_model_name,
        "test_size": len(X_test),
        "train_size": len(X_train),
        "feature_importances": (
            dict(zip(FEATURE_COLUMNS, trained_models["Random Forest"].feature_importances_.round(4)))
        )
    }


def predict_new_conversation(features: dict) -> dict:
    """
    Loads the saved model bundle and predicts success/failure for a
    single new conversation described by a `features` dict.
    """
    try:
        with open(MODEL_PATH, "rb") as f:
            bundle = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("No trained model found. Please train a model first on the ML page.")

    X_new = pd.DataFrame([[features[col] for col in bundle["feature_columns"]]],
                          columns=bundle["feature_columns"])

    if bundle["needs_scaling"]:
        X_new = bundle["scaler"].transform(X_new)

    prediction = bundle["model"].predict(X_new)[0]
    probability = None
    if hasattr(bundle["model"], "predict_proba"):
        proba_input = X_new
        probability = bundle["model"].predict_proba(proba_input)[0].max()

    predicted_label = bundle["label_encoder"].inverse_transform([prediction])[0]

    return {
        "predicted_status": predicted_label,
        "confidence": round(float(probability), 4) if probability is not None else None,
        "model_used": bundle["model_name"]
    }
