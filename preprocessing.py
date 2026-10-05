"""
modules/preprocessing.py
--------------------------
DATA PREPROCESSING module.

Handles missing values, duplicates, invalid values, text cleaning,
categorical encoding, and feature scaling. Every function returns a
NEW dataframe (does not silently mutate the caller's copy) and prints
before/after statistics so the Streamlit UI can display them.
"""

import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def get_missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a small summary dataframe of missing values per column."""
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    summary = pd.DataFrame({
        "column": missing.index,
        "missing_count": missing.values,
        "missing_percent": (missing.values / len(df) * 100).round(2)
    })
    return summary.reset_index(drop=True)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills missing numeric values with the column median and missing
    text values with a placeholder. Median is used instead of mean
    because it is less sensitive to outliers in response_time etc.
    """
    cleaned = df.copy()

    numeric_cols = cleaned.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if cleaned[col].isnull().any():
            median_value = cleaned[col].median()
            cleaned[col] = cleaned[col].fillna(median_value)

    text_cols = cleaned.select_dtypes(include=["object"]).columns
    for col in text_cols:
        cleaned[col] = cleaned[col].fillna("Unknown")
        cleaned[col] = cleaned[col].replace("", "Unknown")

    return cleaned


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Removes fully duplicated rows, keeping the first occurrence."""
    before = len(df)
    cleaned = df.drop_duplicates().reset_index(drop=True)
    after = len(cleaned)
    print(f"Removed {before - after} duplicate rows ({before} -> {after}).")
    return cleaned


def handle_invalid_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Corrects impossible/invalid numeric values, e.g. negative durations
    or response times, and clips satisfaction scores to the valid 1-5 range.
    """
    cleaned = df.copy()

    for col in ["conversation_duration", "response_time", "num_messages",
                "user_message_length", "response_length"]:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].apply(lambda x: x if pd.notnull(x) and x >= 0 else np.nan)
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())

    if "satisfaction_score" in cleaned.columns:
        cleaned["satisfaction_score"] = cleaned["satisfaction_score"].clip(lower=1, upper=5)

    return cleaned


def clean_text(text: str) -> str:
    """Basic text cleaning: lowercase, strip punctuation/extra whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Applies clean_text() to the user_message and chatbot_response columns."""
    cleaned = df.copy()
    for col in ["user_message", "chatbot_response"]:
        if col in cleaned.columns:
            cleaned[f"{col}_clean"] = cleaned[col].apply(clean_text)
    return cleaned


def encode_categorical(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    """
    Label-encodes categorical columns (e.g. topic, sentiment,
    conversation_status) so they can be used by ML models.
    Adds new "<col>_encoded" columns rather than overwriting originals,
    so the human-readable labels stay available for charts.
    """
    cleaned = df.copy()
    if columns is None:
        columns = ["topic", "sentiment", "conversation_status"]

    encoders = {}
    for col in columns:
        if col in cleaned.columns:
            encoder = LabelEncoder()
            cleaned[f"{col}_encoded"] = encoder.fit_transform(cleaned[col].astype(str))
            encoders[col] = encoder

    return cleaned, encoders


def scale_features(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    """
    Scales numeric feature columns to a 0-1 range using MinMaxScaler.
    Scaling helps distance/gradient based ML algorithms treat all
    features fairly regardless of their original units.
    """
    cleaned = df.copy()
    if columns is None:
        columns = ["conversation_duration", "response_time",
                   "user_message_length", "response_length", "num_messages"]
    columns = [c for c in columns if c in cleaned.columns]

    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(cleaned[columns])
    for i, col in enumerate(columns):
        cleaned[f"{col}_scaled"] = scaled_values[:, i]

    return cleaned, scaler


def full_preprocessing_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs the complete preprocessing pipeline in the correct order and
    returns the cleaned dataframe. This is the single function the
    Streamlit app calls for the "Data Preprocessing" page.
    """
    step1 = handle_missing_values(df)
    step2 = remove_duplicates(step1)
    step3 = handle_invalid_values(step2)
    step4 = clean_text_columns(step3)
    step5, _ = encode_categorical(step4)
    step6, _ = scale_features(step5)
    return step6
