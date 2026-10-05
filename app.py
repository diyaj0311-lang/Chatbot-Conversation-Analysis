"""
app.py
--------
CHATBOT CONVERSATION ANALYSIS SYSTEM
Main Streamlit application entry point.

Run with:
    streamlit run app.py

This file wires together every custom module in /modules and /utils
into a single multi-page analytics application. It intentionally keeps
UI logic here and analytical logic in the modules, so the modules stay
reusable and testable on their own (e.g. from a Python shell or unit tests).
"""

import streamlit as st
import pandas as pd
from collections import Counter
import re

from modules import (data_manager, preprocessing, probability,
                      sampling, hypothesis_testing, statistics_analysis, visualization,
                      machine_learning)
from utils.helpers import (DatasetNotFoundError, InvalidDataError,
                            InvalidInputError, format_seconds, percentage)

st.set_page_config(page_title="Chatbot Conversation Analysis", layout="wide", page_icon="🤖")

DATA_PATH = "data/conversations.csv"

# ---------------------------------------------------------------------
# SESSION STATE (keeps the working dataframe alive between page switches)
# ---------------------------------------------------------------------
if "df" not in st.session_state:
    try:
        st.session_state.df = data_manager.load_dataset(DATA_PATH)
        st.session_state.load_error = None
    except (DatasetNotFoundError, InvalidDataError) as e:
        st.session_state.df = pd.DataFrame()
        st.session_state.load_error = str(e)

# ---------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------------------
st.sidebar.title(" Chatbot Conversation\nAnalysis System")
PAGES = [
    "Dashboard", "Dataset Management", "Conversation Analysis", "Text Analysis",
    "Data Preprocessing", "Statistical Analysis", "Probability",
    "Sampling & Inference", "Hypothesis Testing", "Machine Learning", "Export Reports"
]
page = st.sidebar.radio("Navigate to:", PAGES)

st.sidebar.markdown("---")
st.sidebar.caption("Recent activity")
for action in list(data_manager.recent_activity_log)[:6]:
    st.sidebar.caption(f"• {action}")

df = st.session_state.df

if st.session_state.load_error and page != "Dataset Management":
    st.error(f"Dataset problem: {st.session_state.load_error}. Go to 'Dataset Management' to upload/generate one.")
    st.stop()

# =======================================================================
# PAGE 1: DASHBOARD
# =======================================================================
if page == "Dashboard":
    st.title("📊 Dashboard")

    total = len(df)
    successful = int((df["conversation_status"] == "Successful").sum())
    failed = total - successful
    avg_response_time = df["response_time"].mean()
    avg_duration = df["conversation_duration"].mean()
    most_common_topic = df["topic"].mode()[0] if not df.empty else "N/A"
    avg_satisfaction = df["satisfaction_score"].mean()
    success_pct = percentage(successful, total)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Conversations", total)
    col2.metric("Successful", successful)
    col3.metric("Failed", failed)
    col4.metric("Success Rate", f"{success_pct}%")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Avg Response Time", format_seconds(avg_response_time))
    col6.metric("Avg Conversation Length", format_seconds(avg_duration))
    col7.metric("Most Common Topic", most_common_topic)
    col8.metric("Avg Satisfaction", f"{avg_satisfaction:.2f} / 5")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(visualization.plot_status_distribution(df))
    with c2:
        st.pyplot(visualization.plot_sentiment_distribution(df))

    st.pyplot(visualization.plot_conversation_trend(df))

# =======================================================================
# PAGE 2: DATASET MANAGEMENT (CRUD)
# =======================================================================
elif page == "Dataset Management":
    st.title("🗂️ Dataset Management (CRUD)")

    uploaded_file = st.file_uploader("Upload a CSV dataset to replace the current one", type=["csv"])
    if uploaded_file is not None:
        try:
            new_df = pd.read_csv(uploaded_file)
            from utils.helpers import validate_columns
            validate_columns(new_df)
            st.session_state.df = new_df
            df = new_df
            st.success(f"Uploaded dataset with {len(new_df)} rows.")
        except InvalidDataError as e:
            st.error(f"Upload rejected: {e}")
        except Exception as e:
            st.error(f"Could not read file: {e}")

    tab_view, tab_add, tab_update, tab_delete, tab_search = st.tabs(
        ["View", "Add", "Update", "Delete", "Search / Filter"]
    )

    with tab_view:
        st.dataframe(df, use_container_width=True, height=350)
        st.caption(f"{len(df)} total records")

    with tab_add:
        with st.form("add_form"):
            topic = st.selectbox("Topic", sorted(df["topic"].unique()) if not df.empty else ["General Query"])
            user_message = st.text_area("User message")
            chatbot_response = st.text_area("Chatbot response")
            satisfaction = st.slider("Satisfaction score", 1, 5, 3)
            status = st.selectbox("Conversation status", ["Successful", "Failed"])
            submitted = st.form_submit_button("Add conversation")
            if submitted:
                try:
                    record = {"topic": topic, "user_message": user_message,
                              "chatbot_response": chatbot_response,
                              "satisfaction_score": satisfaction,
                              "conversation_status": status,
                              "resolved": status == "Successful"}
                    st.session_state.df = data_manager.add_conversation(df, record)
                    st.success("Conversation added successfully.")
                except InvalidInputError as e:
                    st.error(str(e))

    with tab_update:
        if not df.empty:
            selected_id = st.selectbox("Select conversation ID to update", df["conversation_id"].tolist())
            new_status = st.selectbox("New status", ["Successful", "Failed"])
            new_satisfaction = st.slider("New satisfaction score", 1, 5, 3, key="upd_sat")
            if st.button("Update conversation"):
                try:
                    st.session_state.df = data_manager.update_conversation(
                        df, selected_id,
                        {"conversation_status": new_status, "resolved": new_status == "Successful",
                         "satisfaction_score": new_satisfaction}
                    )
                    st.success(f"Conversation #{selected_id} updated.")
                except InvalidInputError as e:
                    st.error(str(e))

    with tab_delete:
        if not df.empty:
            delete_id = st.selectbox("Select conversation ID to delete", df["conversation_id"].tolist(), key="del_id")
            if st.button("Delete conversation", type="primary"):
                try:
                    st.session_state.df = data_manager.delete_conversation(df, delete_id)
                    st.success(f"Conversation #{delete_id} deleted.")
                except InvalidInputError as e:
                    st.error(str(e))

    with tab_search:
        topic_filter = st.selectbox("Topic", ["All"] + sorted(df["topic"].unique().tolist()) if not df.empty else ["All"])
        status_filter = st.selectbox("Status", ["All", "Successful", "Failed"])
        keyword = st.text_input("Keyword in user message")
        results = data_manager.search_conversations(df, topic_filter, status_filter, keyword)
        st.dataframe(results, use_container_width=True)
        st.caption(f"{len(results)} matching records")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("💾 Save dataset to CSV"):
            data_manager.save_dataset(st.session_state.df, DATA_PATH)
            st.success("Saved to data/conversations.csv")
    with c2:
        if st.button("📤 Export as JSON"):
            data_manager.export_to_json(st.session_state.df)
            st.success("Exported to data/conversations_export.json")

# =======================================================================
# PAGE 3: CONVERSATION ANALYSIS
# =======================================================================
elif page == "Conversation Analysis":
    st.title("💬 Conversation Analysis")

    c1, c2, c3 = st.columns(3)
    c1.metric("Avg messages/conversation", round(df["num_messages"].mean(), 2))
    c2.metric("Avg conversation duration", format_seconds(df["conversation_duration"].mean()))
    c3.metric("Avg response time", format_seconds(df["response_time"].mean()))

    st.pyplot(visualization.plot_length_vs_success(df))
    st.pyplot(visualization.plot_response_time_vs_satisfaction(df))

    st.subheader("Outcome breakdown by topic")
    st.dataframe(statistics_analysis.success_rate_by_category(df, "topic"), use_container_width=True)

# =======================================================================
# PAGE 4: TEXT ANALYSIS
# =======================================================================
elif page == "Text Analysis":
    st.title("🔤 Text / Conversation Content Analysis")

    clean_df = preprocessing.clean_text_columns(df)

    # Frequently used words -- built with a Counter and a set of stopwords (DATA STRUCTURES)
    stopwords = {"the", "is", "a", "to", "i", "my", "you", "it", "do", "does", "how",
                 "can", "and", "of", "in", "for", "was", "am", "on", "this", "that"}
    all_words = " ".join(clean_df["user_message_clean"].dropna()).split()
    filtered_words = [w for w in all_words if w not in stopwords and len(w) > 2]
    word_counts = Counter(filtered_words)
    top_words = pd.DataFrame(word_counts.most_common(15), columns=["word", "frequency"])

    st.subheader("Most frequently used words in user messages")
    st.bar_chart(top_words.set_index("word"))

    st.subheader("Most frequently asked questions (exact repeats)")
    faq = df["user_message"].value_counts().head(10).reset_index()
    faq.columns = ["message", "times_asked"]
    st.dataframe(faq, use_container_width=True)

    unique_topics = set(df["topic"].unique())  # DATA STRUCTURE: set for unique categories
    st.subheader(f"Conversation topics ({len(unique_topics)} unique)")
    st.write(", ".join(sorted(unique_topics)))

    st.subheader("Message length statistics")
    c1, c2 = st.columns(2)
    c1.metric("Avg user message length", round(df["user_message_length"].mean(), 1))
    c2.metric("Avg chatbot response length", round(df["response_length"].mean(), 1))

# =======================================================================
# PAGE 5: DATA PREPROCESSING
# =======================================================================
elif page == "Data Preprocessing":
    st.title("🧹 Data Preprocessing")

    st.subheader("Before preprocessing")
    st.write(f"Rows: {len(df)} | Missing values: {int(df.isnull().sum().sum())} | "
             f"Duplicate rows: {int(df.duplicated().sum())}")
    st.dataframe(preprocessing.get_missing_value_summary(df), use_container_width=True)

    if st.button("Run full preprocessing pipeline"):
        cleaned = preprocessing.full_preprocessing_pipeline(df)
        st.session_state.df_clean = cleaned

        st.subheader("After preprocessing")
        st.write(f"Rows: {len(cleaned)} | Missing values: {int(cleaned.isnull().sum().sum())} | "
                 f"Duplicate rows: {int(cleaned.duplicated().sum())}")
        st.dataframe(cleaned.head(20), use_container_width=True)
        st.success("Missing values filled, duplicates removed, invalid values corrected, "
                    "text cleaned, categorical columns encoded, numeric features scaled.")

# =======================================================================
# PAGE 6: STATISTICAL ANALYSIS
# =======================================================================
elif page == "Statistical Analysis":
    st.title("📈 Statistical Analysis")

    summary = statistics_analysis.describe_all(df)
    st.dataframe(summary, use_container_width=True)

    st.markdown("""
    **How to read this:** `mean` and `median` show the "typical" value; a big gap between
    them signals skewed data. `std_dev` shows how spread out the values are — a high
    `response_time` std_dev means response speed is inconsistent. `q1`/`q3`/`iqr` describe
    the middle 50% of the data and help spot outliers.
    """)

    st.subheader("Correlation heatmap")
    st.pyplot(visualization.plot_correlation_heatmap(df))

    st.subheader("Average response time by topic")
    st.dataframe(statistics_analysis.group_stats(df, "topic", "response_time"), use_container_width=True)
    st.pyplot(visualization.plot_avg_response_time_by_topic(df))

# =======================================================================
# PAGE 7: PROBABILITY
# =======================================================================
elif page == "Probability":
    st.title("🎲 Probability Analysis")

    c1, c2 = st.columns(2)
    c1.metric("P(conversation successful)", probability.probability_of_success(df))
    c2.metric("P(user satisfied, score ≥ 4)", probability.probability_of_satisfaction(df))

    st.subheader("P(Failure | Topic)")
    st.dataframe(probability.probability_by_all_topics(df), use_container_width=True)

    st.subheader("Success probability: short vs long conversations")
    result = probability.probability_success_short_vs_long(df)
    c1, c2 = st.columns(2)
    c1.metric(f"P(Success | short, ≤{int(result['threshold_messages'])} msgs)", result["p_success_short"])
    c2.metric(f"P(Success | long, >{int(result['threshold_messages'])} msgs)", result["p_success_long"])

    st.subheader("Sentiment probability distribution")
    st.dataframe(probability.sentiment_distribution(df), use_container_width=True)

# =======================================================================
# PAGE 8: SAMPLING & INFERENCE
# =======================================================================
elif page == "Sampling & Inference":
    st.title("🎯 Sampling & Inference")

    column = st.selectbox("Numeric column to sample", statistics_analysis.NUMERIC_COLUMNS)
    sample_size = st.slider("Sample size", 20, min(500, len(df)), 100)

    if st.button("Draw sample and estimate population mean"):
        result = sampling.compare_sample_to_population(df, column, sample_size)
        c1, c2, c3 = st.columns(3)
        c1.metric("Population mean", result["population_mean"])
        c2.metric("Sample mean", result["sample_mean"])
        c3.metric("95% CI", f"[{result['ci_lower']}, {result['ci_upper']}]")

        if result["population_mean_within_ci"]:
            st.success("✅ The true population mean falls inside the sample's 95% confidence interval, "
                        "showing the sample is a reasonably good estimate of the population.")
        else:
            st.warning("⚠️ The population mean falls OUTSIDE the sample's confidence interval this time — "
                       "try a larger sample size for a more reliable estimate.")

# =======================================================================
# PAGE 9: HYPOTHESIS TESTING
# =======================================================================
elif page == "Hypothesis Testing":
    st.title("🧪 Hypothesis Testing")

    st.subheader("Test 1: Does response time affect conversation success?")
    ttest_result = hypothesis_testing.response_time_ttest(df)

    st.write(f"**Null Hypothesis (H0):** {ttest_result['null_hypothesis']}")
    st.write(f"**Alternative Hypothesis (H1):** {ttest_result['alt_hypothesis']}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Significance level (α)", ttest_result["alpha"])
    c2.metric("t-statistic", ttest_result["t_statistic"])
    c3.metric("p-value", ttest_result["p_value"])
    c4.metric("Decision", ttest_result["decision"])
    st.info(ttest_result["conclusion"])

    st.markdown("---")
    st.subheader("Test 2: Is conversation topic associated with outcome?")
    chi_result = hypothesis_testing.topic_success_chi_square(df)
    st.write(f"**Null Hypothesis (H0):** {chi_result['null_hypothesis']}")
    st.write(f"**Alternative Hypothesis (H1):** {chi_result['alt_hypothesis']}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Chi-square statistic", chi_result["chi2_statistic"])
    c2.metric("Degrees of freedom", chi_result["degrees_of_freedom"])
    c3.metric("p-value", chi_result["p_value"])
    c4.metric("Decision", chi_result["decision"])
    st.info(chi_result["conclusion"])

# =======================================================================
# PAGE 10: MACHINE LEARNING
# =======================================================================
elif page == "Machine Learning":
    st.title("🤖 Machine Learning: Predict Conversation Success")

    if st.button("Train models (Logistic Regression + Random Forest)"):
        with st.spinner("Training models..."):
            results = machine_learning.train_and_evaluate(df)
        st.session_state.ml_results = results
        st.success(f"Training complete. Best model: {results['best_model_name']}")

    if "ml_results" in st.session_state:
        results = st.session_state.ml_results
        for name, r in results["results"].items():
            st.subheader(name)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", r["accuracy"])
            c2.metric("Precision", r["precision"])
            c3.metric("Recall", r["recall"])
            c4.metric("F1-score", r["f1_score"])
            st.pyplot(visualization.plot_confusion_matrix(r["confusion_matrix"], ["Failed", "Successful"]))
            with st.expander("Full classification report"):
                st.text(r["classification_report"])

        st.subheader("Feature importance (Random Forest)")
        st.bar_chart(pd.Series(results["feature_importances"]))

        st.markdown("---")
        st.subheader("Predict a new conversation")
        c1, c2, c3 = st.columns(3)
        num_messages = c1.number_input("Number of messages", 1, 50, 5)
        duration = c2.number_input("Conversation duration (s)", 0.0, 2000.0, 200.0)
        response_time = c3.number_input("Avg response time (s)", 0.0, 30.0, 2.0)
        c4, c5, c6 = st.columns(3)
        msg_len = c4.number_input("User message length", 0, 500, 30)
        resp_len = c5.number_input("Response length", 0, 500, 50)
        satisfaction = c6.slider("Prior satisfaction score", 1, 5, 3)

        if st.button("Predict outcome"):
            try:
                pred = machine_learning.predict_new_conversation({
                    "num_messages": num_messages, "conversation_duration": duration,
                    "response_time": response_time, "user_message_length": msg_len,
                    "response_length": resp_len, "satisfaction_score": satisfaction
                })
                st.success(f"Predicted status: **{pred['predicted_status']}** "
                           f"(confidence: {pred['confidence']}, model: {pred['model_used']})")
            except FileNotFoundError as e:
                st.error(str(e))
    else:
        st.info("Click 'Train models' above to train and evaluate the classifiers.")

# =======================================================================
# PAGE 11: EXPORT REPORTS
# =======================================================================
elif page == "Export Reports":
    st.title("📄 Export Analysis Report")

    if st.button("Generate text report"):
        lines = []
        lines.append("CHATBOT CONVERSATION ANALYSIS - SUMMARY REPORT")
        lines.append("=" * 50)
        lines.append(f"Total conversations: {len(df)}")
        lines.append(f"Successful: {int((df['conversation_status']=='Successful').sum())}")
        lines.append(f"Failed: {int((df['conversation_status']=='Failed').sum())}")
        lines.append(f"Average response time: {df['response_time'].mean():.2f} seconds")
        lines.append(f"Average satisfaction score: {df['satisfaction_score'].mean():.2f} / 5")
        lines.append(f"Most common topic: {df['topic'].mode()[0]}")
        lines.append("")
        ttest = hypothesis_testing.response_time_ttest(df)
        lines.append("HYPOTHESIS TEST RESULT")
        lines.append(f"  {ttest['decision']} (p-value = {ttest['p_value']})")
        lines.append(f"  {ttest['conclusion']}")
        report_text = "\n".join(lines)

        st.text(report_text)
        st.download_button("⬇️ Download report (.txt)", report_text, file_name="analysis_report.txt")

    st.markdown("---")
    st.write("You can also export the raw or cleaned dataset from the **Dataset Management** page.")
