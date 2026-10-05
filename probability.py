"""
modules/probability.py
-------------------------
PROBABILITY module.

Calculates empirical probabilities directly from the dataset (classical
"favorable outcomes / total outcomes" definition) plus a couple of
conditional probabilities that are meaningful for chatbot performance.
"""

import pandas as pd


def probability_of_success(df: pd.DataFrame) -> float:
    """P(conversation is successful)."""
    return round(df["resolved"].mean(), 4)


def probability_of_satisfaction(df: pd.DataFrame, threshold: int = 4) -> float:
    """P(satisfaction_score >= threshold), default: satisfied = 4 or 5 out of 5."""
    satisfied = (df["satisfaction_score"] >= threshold).sum()
    return round(satisfied / len(df), 4)


def probability_of_failure_given_topic(df: pd.DataFrame, topic: str) -> float:
    """
    Conditional probability P(Failure | Topic = topic)
    = P(Failure AND Topic) / P(Topic)
    """
    topic_df = df[df["topic"] == topic]
    if len(topic_df) == 0:
        return 0.0
    failures = (topic_df["conversation_status"] == "Failed").sum()
    return round(failures / len(topic_df), 4)


def probability_by_all_topics(df: pd.DataFrame) -> pd.DataFrame:
    """Returns P(Failure | Topic) for every topic in the dataset, for easy charting."""
    topics = df["topic"].unique()
    rows = [{"topic": t, "p_failure_given_topic": probability_of_failure_given_topic(df, t)}
            for t in topics]
    result = pd.DataFrame(rows).sort_values("p_failure_given_topic", ascending=False)
    return result.reset_index(drop=True)


def probability_success_short_vs_long(df: pd.DataFrame, median_split: bool = True) -> dict:
    """
    Compares P(Success | short conversation) vs P(Success | long conversation),
    where 'short'/'long' is defined by a median split on num_messages.
    """
    threshold = df["num_messages"].median() if median_split else df["num_messages"].mean()

    short_convs = df[df["num_messages"] <= threshold]
    long_convs = df[df["num_messages"] > threshold]

    p_success_short = round(short_convs["resolved"].mean(), 4) if len(short_convs) else 0.0
    p_success_long = round(long_convs["resolved"].mean(), 4) if len(long_convs) else 0.0

    return {
        "threshold_messages": threshold,
        "p_success_short": p_success_short,
        "p_success_long": p_success_long,
        "short_count": len(short_convs),
        "long_count": len(long_convs)
    }


def sentiment_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Returns the empirical probability (relative frequency) of each sentiment class."""
    counts = df["sentiment"].value_counts(normalize=True).round(4)
    result = counts.reset_index()
    result.columns = ["sentiment", "probability"]
    return result
