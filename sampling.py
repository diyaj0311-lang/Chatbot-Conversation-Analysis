"""
modules/sampling.py
----------------------
SAMPLING & INFERENCE module.

Draws a simple random sample from the full ("population") dataset and
estimates population parameters from the sample, including a
confidence interval, so students can compare the sample statistic to
the true population statistic and see how close inference gets.
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def simple_random_sample(df: pd.DataFrame, sample_size: int = 100, random_state: int = 42) -> pd.DataFrame:
    """Draws a simple random sample (without replacement) from the dataset."""
    sample_size = min(sample_size, len(df))
    return df.sample(n=sample_size, random_state=random_state)


def compare_sample_to_population(df: pd.DataFrame, column: str, sample_size: int = 100) -> dict:
    """
    Compares the sample mean to the population (full dataset) mean for
    a given numeric column, and builds a 95% confidence interval for
    the population mean based on the sample.
    """
    population = df[column].dropna().to_numpy()
    sample_df = simple_random_sample(df, sample_size)
    sample = sample_df[column].dropna().to_numpy()

    population_mean = float(np.mean(population))
    sample_mean = float(np.mean(sample))
    sample_std = float(np.std(sample, ddof=1))
    n = len(sample)

    # 95% confidence interval for the population mean using the
    # t-distribution (appropriate since population std is unknown).
    confidence = 0.95
    standard_error = sample_std / np.sqrt(n)
    t_critical = scipy_stats.t.ppf((1 + confidence) / 2, df=n - 1)
    margin_of_error = t_critical * standard_error

    ci_lower = sample_mean - margin_of_error
    ci_upper = sample_mean + margin_of_error

    return {
        "column": column,
        "population_mean": round(population_mean, 4),
        "sample_size": n,
        "sample_mean": round(sample_mean, 4),
        "sample_std": round(sample_std, 4),
        "confidence_level": confidence,
        "margin_of_error": round(margin_of_error, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "population_mean_within_ci": bool(ci_lower <= population_mean <= ci_upper)
    }


def stratified_sample_by_topic(df: pd.DataFrame, fraction: float = 0.15, random_state: int = 42) -> pd.DataFrame:
    """
    Draws a stratified sample that preserves the proportion of each
    'topic' category -- useful when topics are imbalanced.
    """
    return df.groupby("topic", group_keys=False).apply(
        lambda x: x.sample(frac=fraction, random_state=random_state)
    )
