"""
modules/data_manager.py
-------------------------
FILE HANDLING + CRUD OPERATIONS module.

Responsible for reading/writing the conversation dataset (CSV/JSON) and
implementing Create, Read, Update, Delete operations on it. Also shows
a simple stack/queue based "recent activity" log using Python's
built-in `collections.deque`.
"""

import os
import json
import pandas as pd
from collections import deque

from utils.helpers import DatasetNotFoundError, InvalidDataError, InvalidInputError, validate_columns

DEFAULT_CSV_PATH = "data/conversations.csv"

# DATA STRUCTURE: a deque used as a fixed-size STACK of recent actions
# (most recent action is always at index 0). Useful for an "Activity Log"
# panel in the UI so the student can show off stacks meaningfully.
recent_activity_log = deque(maxlen=20)


def _log_action(action: str):
    recent_activity_log.appendleft(action)


# ---------------------------------------------------------------------
# READ
# ---------------------------------------------------------------------
def load_dataset(path: str = DEFAULT_CSV_PATH) -> pd.DataFrame:
    """
    Loads the conversation dataset from CSV.
    Demonstrates try/except/else/finally exception handling.
    """
    df = None
    try:
        if not os.path.isfile(path):
            raise DatasetNotFoundError(f"Dataset file not found at '{path}'.")
        df = pd.read_csv(path)
    except DatasetNotFoundError:
        raise
    except pd.errors.EmptyDataError:
        raise InvalidDataError("The dataset file exists but is empty.")
    except pd.errors.ParserError:
        raise InvalidDataError("The dataset file is not a valid CSV.")
    else:
        # Runs only if no exception occurred
        validate_columns(df)
        _log_action(f"Loaded dataset with {len(df)} rows from {path}")
    finally:
        # Always runs, regardless of success/failure -- good place for cleanup logging
        pass

    return df


def load_json(path: str) -> list:
    """Reads a JSON file and returns a Python list of records."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        raise DatasetNotFoundError(f"JSON file not found at '{path}'.")
    except json.JSONDecodeError:
        raise InvalidDataError(f"'{path}' is not valid JSON.")


# ---------------------------------------------------------------------
# CREATE
# ---------------------------------------------------------------------
def add_conversation(df: pd.DataFrame, record: dict) -> pd.DataFrame:
    """
    Adds a new conversation record (CREATE).
    `record` must be a dictionary with (at least) the core fields.
    """
    required_fields = ["topic", "user_message", "chatbot_response"]
    for field in required_fields:
        if field not in record or record[field] in (None, ""):
            raise InvalidInputError(f"Field '{field}' is required to add a conversation.")

    new_id = int(df["conversation_id"].max()) + 1 if len(df) else 1
    record.setdefault("conversation_id", new_id)
    record.setdefault("num_messages", 1)
    record.setdefault("conversation_duration", 0.0)
    record.setdefault("response_time", 0.0)
    record.setdefault("user_message_length", len(str(record.get("user_message", ""))))
    record.setdefault("response_length", len(str(record.get("chatbot_response", ""))))
    record.setdefault("satisfaction_score", 3)
    record.setdefault("sentiment", "Neutral")
    record.setdefault("conversation_status", "Successful")
    record.setdefault("resolved", True)
    record.setdefault("date", pd.Timestamp.now().strftime("%Y-%m-%d"))
    record.setdefault("user_id", "MANUAL_ENTRY")

    new_row = pd.DataFrame([record])
    updated_df = pd.concat([df, new_row], ignore_index=True)
    _log_action(f"Added conversation #{record['conversation_id']}")
    return updated_df


# ---------------------------------------------------------------------
# UPDATE
# ---------------------------------------------------------------------
def update_conversation(df: pd.DataFrame, conversation_id: int, updates: dict) -> pd.DataFrame:
    """Updates fields of an existing record identified by conversation_id (UPDATE)."""
    if conversation_id not in df["conversation_id"].values:
        raise InvalidInputError(f"Conversation ID {conversation_id} does not exist.")

    index = df.index[df["conversation_id"] == conversation_id][0]
    for key, value in updates.items():
        if key in df.columns:
            df.at[index, key] = value

    _log_action(f"Updated conversation #{conversation_id}")
    return df


# ---------------------------------------------------------------------
# DELETE
# ---------------------------------------------------------------------
def delete_conversation(df: pd.DataFrame, conversation_id: int) -> pd.DataFrame:
    """Removes a record identified by conversation_id (DELETE)."""
    if conversation_id not in df["conversation_id"].values:
        raise InvalidInputError(f"Conversation ID {conversation_id} does not exist.")

    updated_df = df[df["conversation_id"] != conversation_id].reset_index(drop=True)
    _log_action(f"Deleted conversation #{conversation_id}")
    return updated_df


# ---------------------------------------------------------------------
# SEARCH / FILTER
# ---------------------------------------------------------------------
def search_conversations(df: pd.DataFrame, topic: str = None, status: str = None,
                          keyword: str = None) -> pd.DataFrame:
    """Filters the dataset by topic, status and/or a keyword in the user message."""
    result = df.copy()
    if topic and topic != "All":
        result = result[result["topic"] == topic]
    if status and status != "All":
        result = result[result["conversation_status"] == status]
    if keyword:
        result = result[result["user_message"].astype(str).str.contains(keyword, case=False, na=False)]
    return result


# ---------------------------------------------------------------------
# SAVE / EXPORT
# ---------------------------------------------------------------------
def save_dataset(df: pd.DataFrame, path: str = DEFAULT_CSV_PATH):
    """Writes the current dataframe back to CSV."""
    try:
        df.to_csv(path, index=False)
        _log_action(f"Saved dataset ({len(df)} rows) to {path}")
        return True
    except (IOError, OSError) as e:
        print(f"Error saving dataset: {e}")
        return False


def export_to_json(df: pd.DataFrame, path: str = "data/conversations_export.json"):
    """Exports the dataframe to a JSON file."""
    try:
        df.to_json(path, orient="records", indent=2)
        return True
    except (IOError, OSError) as e:
        print(f"Error exporting to JSON: {e}")
        return False


def append_text_log(message: str, path: str = "data/activity_log.txt"):
    """Appends a plain-text log line -- demonstrates TXT file handling."""
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{pd.Timestamp.now()} - {message}\n")
        return True
    except (IOError, OSError) as e:
        print(f"Error writing text log: {e}")
        return False
