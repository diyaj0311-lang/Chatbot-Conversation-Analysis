"""
modules/statistics.py
------------------------
STATISTICAL ANALYSIS module (uses NumPy heavily, Pandas for grouping).

Provides descriptive statistics (mean, median, mode, range, variance,
standard deviation, quartiles) and correlation analysis for the
numeric columns of the conversation dataset.
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

NUMERIC_COLUMNS = [
    "num_messages", "conversation_duration", "response_time",
    "user_message_length", "response_length", "satisfaction_score"
]


def describe_column(df: pd.DataFrame, column: str) -> dict:
    """
    Computes a full set of descriptive statistics for one numeric column
    using NumPy functions directly (rather than just df.describe()) so
    the NumPy requirement is explicitly demonstrated.
    """
    values = df[column].dropna().to_numpy()  # NumPy array

    mode_result = scipy_stats.mode(values, keepdims=True)

    result = {
        "column": column,
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "mode": float(mode_result.mode[0]),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "range": float(np.max(values) - np.min(values)),
        "variance": float(np.var(values, ddof=1)),
        "std_dev": float(np.std(values, ddof=1)),
        "q1": float(np.percentile(values, 25)),
        "q3": float(np.percentile(values, 75)),
        "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
    }
    return result


def describe_all(df: pd.DataFrame) -> pd.DataFrame:
    """Runs describe_column() for every key numeric column and returns a summary table."""
    rows = [describe_column(df, col) for col in NUMERIC_COLUMNS if col in df.columns]
    return pd.DataFrame(rows)


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Returns the Pearson correlation matrix for the numeric columns."""
    cols = [c for c in NUMERIC_COLUMNS if c in df.columns]
    return df[cols].corr(method="pearson")


def group_stats(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    """
    Uses Pandas groupby to compute mean/median/std of `value_col` for
    each category in `group_col` (e.g. average response_time per topic).
    """
    grouped = df.groupby(group_col)[value_col].agg(["mean", "median", "std", "count"]).reset_index()
    grouped = grouped.round(2)
    return grouped


def success_rate_by_category(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Computes the percentage of successful conversations per category."""
    grouped = df.groupby(group_col)["resolved"].mean().reset_index()
    grouped["success_percent"] = (grouped["resolved"] * 100).round(2)
    grouped = grouped.drop(columns=["resolved"])
    return grouped.sort_values("success_percent", ascending=False)
