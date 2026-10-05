"""
generate_dataset.py
--------------------
Creates a realistic SYNTHETIC dataset of chatbot conversations for the
Chatbot Conversation Analysis System.

IMPORTANT: This data is artificially generated for academic demonstration.
It does NOT come from a real chatbot deployment or real users.

Run:
    python generate_dataset.py
Output:
    data/conversations.csv  (800 records)
"""

import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Fixing the seed makes the "random" data reproducible every time the
# script is run -- useful for grading/demo consistency.
random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------
# Reference data used to build believable rows (DATA STRUCTURES: lists)
# ---------------------------------------------------------------------
TOPICS = [
    "Billing", "Technical Support", "Account Management", "Product Info",
    "Order Tracking", "Refund Request", "Password Reset", "General Query",
    "Complaint", "Feature Request"
]

SAMPLE_USER_MESSAGES = {
    "Billing": ["Why was I charged twice this month?", "Can you explain my latest invoice?",
                "I need a copy of my last bill", "Is there a late payment fee?"],
    "Technical Support": ["The app keeps crashing on startup", "I cannot log into my account",
                           "The page is not loading properly", "My upload keeps failing"],
    "Account Management": ["How do I update my email address?", "I want to delete my account",
                            "Can I merge two accounts?", "How do I change my username?"],
    "Product Info": ["What plans do you offer?", "Does this plan include support?",
                     "What is the difference between the two tiers?", "Is there a free trial?"],
    "Order Tracking": ["Where is my order?", "My package has not arrived yet",
                       "Can you track order 45213?", "Why is my delivery delayed?"],
    "Refund Request": ["I want a refund for my last purchase", "How long do refunds take?",
                       "My refund has not been processed", "Can I get a partial refund?"],
    "Password Reset": ["I forgot my password", "The reset link is not working",
                       "How do I set a new password?", "I am locked out of my account"],
    "General Query": ["What are your working hours?", "How do I contact support?",
                      "Do you have a mobile app?", "Where are you located?"],
    "Complaint": ["Your service is too slow", "I am very unhappy with the support",
                 "This is the third time I am contacting you", "Nobody is responding to my emails"],
    "Feature Request": ["Can you add dark mode?", "Please support multiple languages",
                        "It would help to have export to Excel", "Can you add two factor login?"]
}

SAMPLE_RESPONSES = [
    "I understand your concern, let me check that for you.",
    "Thank you for reaching out, here is the information you need.",
    "I have escalated this issue to our support team.",
    "Your request has been processed successfully.",
    "I am sorry for the inconvenience, we are looking into it.",
    "Here is a quick guide that should resolve your issue.",
    "Unfortunately I do not have that information right now.",
    "Your issue has been resolved, is there anything else I can help with?",
    "Let me connect you with a human agent for further assistance.",
    "I could not fully resolve your issue, please try again later."
]

SENTIMENTS = ["Positive", "Neutral", "Negative"]


def random_date(start_days_ago=180):
    """Generate a random date within the last `start_days_ago` days."""
    days_back = random.randint(0, start_days_ago)
    return (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")


def generate_row(conversation_id):
    """Builds one synthetic conversation record as a dictionary
    (DATA STRUCTURES: dictionary used for a single record)."""
    topic = random.choice(TOPICS)
    user_message = random.choice(SAMPLE_USER_MESSAGES[topic])
    chatbot_response = random.choice(SAMPLE_RESPONSES)

    num_messages = int(np.random.poisson(6) + 1)          # NumPy random distribution
    conversation_duration = round(num_messages * np.random.uniform(15, 90), 2)  # seconds
    response_time = round(np.random.exponential(2.5), 2)  # seconds, exponential dist fits response time
    user_message_length = len(user_message)
    response_length = len(chatbot_response)

    # Satisfaction score influenced by response_time (faster => usually happier)
    base_satisfaction = np.clip(5 - (response_time / 3) + np.random.normal(0, 1), 1, 5)
    satisfaction_score = int(round(base_satisfaction))

    sentiment = random.choices(SENTIMENTS, weights=[0.45, 0.35, 0.20])[0]

    # Conversation is more likely to succeed when satisfaction is high and response_time is low
    success_probability = np.clip(
        0.5 + (satisfaction_score - 3) * 0.12 - (response_time - 2.5) * 0.05, 0.05, 0.95
    )
    resolved = np.random.rand() < success_probability
    conversation_status = "Successful" if resolved else "Failed"

    # Deliberately inject a few missing / messy values to demonstrate preprocessing
    if random.random() < 0.03:
        satisfaction_score = np.nan
    if random.random() < 0.02:
        response_time = np.nan
    if random.random() < 0.01:
        user_message = ""  # empty text value

    return {
        "conversation_id": conversation_id,
        "user_id": f"U{random.randint(1000, 1999)}",
        "date": random_date(),
        "topic": topic,
        "user_message": user_message,
        "chatbot_response": chatbot_response,
        "num_messages": num_messages,
        "conversation_duration": conversation_duration,
        "response_time": response_time,
        "user_message_length": user_message_length,
        "response_length": response_length,
        "satisfaction_score": satisfaction_score,
        "sentiment": sentiment,
        "conversation_status": conversation_status,
        "resolved": resolved
    }


def generate_dataset(n_records=800):
    records = [generate_row(i + 1) for i in range(n_records)]
    df = pd.DataFrame(records)

    # Inject a handful of exact duplicate rows to demonstrate de-duplication later
    duplicates = df.sample(10, random_state=1)
    df = pd.concat([df, duplicates], ignore_index=True)

    return df


if __name__ == "__main__":
    dataset = generate_dataset(800)
    dataset.to_csv("data/conversations.csv", index=False)
    print(f"Generated {len(dataset)} records -> data/conversations.csv")
