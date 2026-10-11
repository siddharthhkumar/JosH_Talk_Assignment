"""Optional Streamlit dashboard for analysis.py.

    streamlit run dashboard.py

Upload the real CSV, or explicitly choose the synthetic demo. Read-only: there
are no account-action controls by design.
"""

import pandas as pd
import streamlit as st

import analysis as A

st.set_page_config(page_title="Reviewer Quality Screening", layout="wide")
st.title("Transcription reviewer quality screening")
st.caption("Flags mean *sample this worker's transcripts for human review*. They are not a verdict. "
           "All thresholds are unvalidated hypotheses.")

source = st.sidebar.radio("Data source", ["Upload real CSV", "Synthetic demo (not real data)"])
cfg = A.Config()
with st.sidebar.expander("Thresholds (hypotheses)"):
    cfg.min_valid_tasks = st.number_input("Min valid tasks", 1, 1000, cfg.min_valid_tasks)
    cfg.fast_ratio_threshold = st.number_input("A: fast ratio <", 0.05, 2.0, cfg.fast_ratio_threshold, 0.05)
    cfg.char_rate_threshold = st.number_input("C: chars/s >", 1.0, 100.0, cfg.char_rate_threshold, 1.0)

raw = None
if source.startswith("Upload"):
    f = st.sidebar.file_uploader("Worker-performance CSV", type=["csv"])
    if f is None:
        st.info("No real dataset loaded. Upload the worker-performance CSV with columns: "
                + ", ".join(A.REQUIRED_COLUMNS))
        st.stop()
    raw = pd.read_csv(f, dtype=str, encoding="utf-8-sig")
    raw.attrs["is_synthetic"] = False
else:
    raw = A.make_synthetic_dataset()

try:
    res = A.run_pipeline(raw, cfg)
except A.DataError as exc:
    st.error(str(exc))
    st.stop()

if res["synthetic"]:
    st.error("SYNTHETIC DEMONSTRATION DATA: these are fictional workers, not Josh Talks transcribers.")
else:
    st.success("Real user-supplied dataset")

w, t, q = res["workers"], res["tasks"], res["quality"]
flagged = w["review_status"].isin(["review_recommended", "priority_review"])
c1, c2, c3, c4 = st.columns(4)
c1.metric("Workers analysed", len(w))
c2.metric("Review recommended", int((w["review_status"] == "review_recommended").sum()))
c3.metric("Priority review", int((w["review_status"] == "priority_review").sum()))
c4.metric("Insufficient data", int((w["review_status"] == "insufficient_data").sum()))

st.subheader("Signal distributions (task level)")
d1, d2 = st.columns(2)
d1.caption(f"Completion ratio (time taken / audio duration); hypothesis threshold < {cfg.fast_ratio_threshold}")
d1.bar_chart(pd.cut(t["completion_ratio"].dropna().clip(upper=5), bins=25).value_counts(sort=False)
             .rename(lambda i: f"{i.left:.1f}"))
d2.caption(f"Characters per second of time taken; hypothesis threshold > {cfg.char_rate_threshold:g}")
d2.bar_chart(pd.cut(t["char_rate"].dropna().clip(upper=60), bins=25).value_counts(sort=False)
             .rename(lambda i: f"{i.left:.0f}"))

st.subheader("Worker review table")
show_all = st.checkbox("Show all workers", value=False)
cols = ["user_id", "review_status", "risk_score", "n_valid_timing", "median_completion_ratio",
        "pct_fast_tasks", "pct_unedited", "pct_fast_and_unedited", "median_char_rate",
        "char_rate_robust_z", "reasons", "notes", "recommended_action"]
st.dataframe((w if show_all else w[flagged])[cols].astype({"review_status": str}), width="stretch")

st.subheader("Task-level investigation")
uid = st.selectbox("Worker", w["user_id"].tolist())
st.dataframe(
    t.loc[t["user_id"] == uid, ["recording_url", "duration", "time_taken_by_user", "completion_ratio",
                                "char_rate", "is_edited", "edit_flag_consistency", "text_similarity",
                                "whisper_text", "user_text"]],
    width="stretch",
)

with st.expander("Data-quality report"):
    st.json({k: (v if not isinstance(v, (pd.Series,)) else v.to_dict()) for k, v in q.items()})
