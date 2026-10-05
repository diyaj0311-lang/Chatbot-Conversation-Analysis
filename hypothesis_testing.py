import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def response_time_ttest(df: pd.DataFrame, alpha: float = 0.05) -> dict:

    success_group = df[df["conversation_status"] == "Successful"]["response_time"].dropna()
    failed_group = df[df["conversation_status"] == "Failed"]["response_time"].dropna()

    # Welch's t-test (equal_var=False) does not assume equal variances,
    # which is a safer default for real-world data.
    t_statistic, p_value = scipy_stats.ttest_ind(success_group, failed_group, equal_var=False)

    decision = "Reject H0" if p_value < alpha else "Fail to reject H0"
    if p_value < alpha:
        conclusion = ("There IS a statistically significant difference in average response time "
                       "between successful and failed conversations.")
    else:
        conclusion = ("There is NOT enough evidence to say response time significantly affects "
                       "whether a conversation succeeds or fails.")

    return {
        "test": "Independent two-sample t-test (Welch's)",
        "null_hypothesis": "H0: Mean response_time is the same for successful and failed conversations.",
        "alt_hypothesis": "H1: Mean response_time differs between successful and failed conversations.",
        "alpha": alpha,
        "success_group_mean": round(float(success_group.mean()), 4),
        "failed_group_mean": round(float(failed_group.mean()), 4),
        "t_statistic": round(float(t_statistic), 4),
        "p_value": round(float(p_value), 6),
        "decision": decision,
        "conclusion": conclusion
    }


def topic_success_chi_square(df: pd.DataFrame, alpha: float = 0.05) -> dict:
    """
    Chi-square test of independence: is 'topic' associated with
    'conversation_status' (Successful/Failed)?
    """
    contingency_table = pd.crosstab(df["topic"], df["conversation_status"])
    chi2_statistic, p_value, degrees_of_freedom, expected = scipy_stats.chi2_contingency(contingency_table)

    decision = "Reject H0" if p_value < alpha else "Fail to reject H0"
    if p_value < alpha:
        conclusion = "Conversation topic IS significantly associated with success/failure outcome."
    else:
        conclusion = "There is no significant association between conversation topic and outcome."

    return {
        "test": "Chi-square test of independence",
        "null_hypothesis": "H0: Topic and conversation outcome are independent.",
        "alt_hypothesis": "H1: Topic and conversation outcome are NOT independent.",
        "alpha": alpha,
        "chi2_statistic": round(float(chi2_statistic), 4),
        "degrees_of_freedom": int(degrees_of_freedom),
        "p_value": round(float(p_value), 6),
        "decision": decision,
        "conclusion": conclusion,
        "contingency_table": contingency_table
    }
