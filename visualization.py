"""
modules/visualization.py
---------------------------
DATA VISUALIZATION module.

Every function builds and returns a Matplotlib Figure object (rather
than calling plt.show()) so the Streamlit app can render it directly
with st.pyplot(fig). Seaborn is used for styling and some plot types.
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, required for server/Streamlit use
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set_theme(style="whitegrid")


def plot_status_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["conversation_status"].value_counts()
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette="Set2", ax=ax, legend=False)
    ax.set_title("Conversation Status Distribution")
    ax.set_xlabel("Status")
    ax.set_ylabel("Number of Conversations")
    fig.tight_layout()
    return fig


def plot_satisfaction_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df["satisfaction_score"].dropna(), bins=5, kde=False, color="steelblue", ax=ax)
    ax.set_title("Satisfaction Score Distribution")
    ax.set_xlabel("Satisfaction Score (1-5)")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    return fig


def plot_avg_response_time_by_topic(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    grouped = df.groupby("topic")["response_time"].mean().sort_values(ascending=False)
    sns.barplot(x=grouped.values, y=grouped.index, hue=grouped.index, palette="viridis", ax=ax, legend=False)
    ax.set_title("Average Response Time by Topic")
    ax.set_xlabel("Average Response Time (seconds)")
    ax.set_ylabel("Topic")
    fig.tight_layout()
    return fig


def plot_conversations_by_topic(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df["topic"].value_counts()
    sns.barplot(x=counts.values, y=counts.index, hue=counts.index, palette="magma", ax=ax, legend=False)
    ax.set_title("Number of Conversations by Topic")
    ax.set_xlabel("Number of Conversations")
    ax.set_ylabel("Topic")
    fig.tight_layout()
    return fig


def plot_conversation_trend(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 4))
    trend_df = df.copy()
    trend_df["date"] = pd.to_datetime(trend_df["date"])
    daily_counts = trend_df.groupby(trend_df["date"].dt.date).size()
    ax.plot(daily_counts.index, daily_counts.values, marker="o", markersize=2, linewidth=1, color="darkorange")
    ax.set_title("Conversation Volume Trend Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Conversations")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_response_time_vs_satisfaction(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=df, x="response_time", y="satisfaction_score",
                     hue="conversation_status", alpha=0.6, ax=ax, palette="Set1")
    ax.set_title("Response Time vs Satisfaction Score")
    ax.set_xlabel("Response Time (seconds)")
    ax.set_ylabel("Satisfaction Score")
    ax.legend(title="Status")
    fig.tight_layout()
    return fig


def plot_length_vs_success(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=df, x="conversation_status", y="num_messages", hue="conversation_status",
                palette="pastel", ax=ax, legend=False)
    ax.set_title("Conversation Length vs Outcome")
    ax.set_xlabel("Conversation Status")
    ax.set_ylabel("Number of Messages")
    fig.tight_layout()
    return fig


def plot_sentiment_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["sentiment"].value_counts()
    colors = {"Positive": "#4CAF50", "Neutral": "#FFC107", "Negative": "#F44336"}
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
           colors=[colors.get(c, "#999999") for c in counts.index])
    ax.set_title("Sentiment Distribution")
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 6))
    numeric_cols = ["num_messages", "conversation_duration", "response_time",
                     "user_message_length", "response_length", "satisfaction_score"]
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap of Numeric Features")
    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm, class_labels):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(np.array(cm), annot=True, fmt="d", cmap="Blues",
                xticklabels=class_labels, yticklabels=class_labels, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    fig.tight_layout()
    return fig
