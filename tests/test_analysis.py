"""Tests for analysis.py. All fixtures are synthetic and built in memory."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analysis as A  # noqa: E402


def make_rows(user="u1", n=12, duration=10.0, taken=20.0, edited="Yes",
              whisper="नमस्ते आप कैसे हैं", user_text="नमस्ते आप कैसे हैं", cps=1.0):
    return [
        {
            "user_id": user,
            "recording_url": f"synthetic://{user}/{i}",
            "whisper_text": whisper,
            "user_text": user_text,
            "is_edited": edited,
            "duration": duration,
            "time_taken_by_user": taken,
            "segment_character_per_second": cps,
        }
        for i in range(n)
    ]


def frame(rows):
    df = pd.DataFrame(rows)
    df.attrs["is_synthetic"] = True
    return df


# ----- validation ---------------------------------------------------------- #
def test_missing_required_columns_reported_by_name():
    df = frame(make_rows()).drop(columns=["duration", "user_text"])
    with pytest.raises(A.DataError) as exc:
        A.run_pipeline(df)
    assert "duration" in str(exc.value) and "user_text" in str(exc.value)


def test_column_names_are_case_and_space_insensitive():
    df = frame(make_rows())
    df.columns = [f" {c.upper()} " for c in df.columns]
    result = A.run_pipeline(df)
    assert len(result["workers"]) == 1


def test_empty_dataset_raises_clear_error():
    df = pd.DataFrame(columns=A.REQUIRED_COLUMNS)
    with pytest.raises(A.DataError, match="no rows"):
        A.run_pipeline(df)


def test_missing_file_gives_actionable_message(tmp_path):
    with pytest.raises(A.DataError, match="--synthetic-demo"):
        A.load_dataset(tmp_path / "nope.csv")


def test_cli_without_input_exits_gracefully(capsys):
    assert A.main([]) == 2
    assert "no empirical analysis" in capsys.readouterr().err


def test_cli_reports_missing_columns(tmp_path, capsys):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"user_id": ["a"]}).to_csv(p, index=False)
    assert A.main(["--input", str(p), "--no-plots"]) == 2
    assert "missing required column" in capsys.readouterr().err


# ----- cleaning ------------------------------------------------------------ #
@pytest.mark.parametrize(
    "raw,expected",
    [("Yes", True), ("yes ", True), ("TRUE", True), ("1", True), (1, True), (True, True),
     ("No", False), ("false", False), ("0", False), (0, False), (False, False),
     ("maybe", pd.NA), (None, pd.NA), (np.nan, pd.NA)],
)
def test_is_edited_normalization(raw, expected):
    out = A.normalize_is_edited(raw)
    if expected is pd.NA:
        assert out is pd.NA
    else:
        assert out is expected


def test_invalid_and_zero_durations_excluded_from_ratios():
    rows = make_rows(n=3)
    rows[0]["duration"] = 0
    rows[1]["duration"] = "abc"
    rows[2]["duration"] = -5
    rows += make_rows(user="u1", n=1, duration=10, taken=5)
    rows[-1]["recording_url"] = "synthetic://u1/ok"
    res = A.run_pipeline(frame(rows))
    t = res["tasks"]
    assert t["completion_ratio"].notna().sum() == 1
    assert t["completion_ratio"].dropna().iloc[0] == pytest.approx(0.5)
    assert res["quality"]["duration_missing_or_nonpositive"] == 3
    assert res["quality"]["duration_unparseable"] == 1
    assert not t.loc[~t["valid_timing"], "is_fast_task"].any()


def test_zero_or_missing_time_taken_handled():
    rows = make_rows(n=2)
    rows[0]["time_taken_by_user"] = 0
    rows[1]["time_taken_by_user"] = None
    t = A.run_pipeline(frame(rows))["tasks"]
    assert t["completion_ratio"].isna().all()
    assert t["char_rate"].isna().all()
    assert np.isfinite(t["char_rate"].fillna(0)).all()


def test_missing_text_values_handled():
    rows = make_rows(n=2)
    rows[0]["user_text"] = None
    rows[1]["whisper_text"] = None
    t = A.run_pipeline(frame(rows))["tasks"]
    assert t["text_similarity"].isna().all()
    assert (t["edit_flag_consistency"] == "not_checkable").all()
    assert pd.isna(t.loc[0, "char_rate"])
    assert t.loc[1, "char_rate"] > 0


def test_duplicates_and_missing_user_ids_removed():
    rows = make_rows(n=2)
    rows.append(dict(rows[0]))
    rows.append({**rows[1], "user_id": None, "recording_url": "x"})
    res = A.run_pipeline(frame(rows))
    assert res["quality"]["duplicate_rows_removed"] == 1
    assert res["quality"]["rows_missing_user_id_removed"] == 1
    assert res["quality"]["rows_after_cleaning"] == 2


# ----- task metrics --------------------------------------------------------- #
def test_character_rate_counts_non_space_devanagari_chars():
    text = "नमस्ते आप"  # 'नमस्ते' = 6 code points, 'आप' = 2 -> 8 non-space chars
    assert A.count_non_space_chars(text) == 8
    t = A.run_pipeline(frame(make_rows(n=1, user_text=text, taken=4.0)))["tasks"]
    assert t.loc[0, "char_rate"] == pytest.approx(2.0)


def test_supplied_cps_compared_against_both_definitions():
    text = "abcdefghij"  # 10 chars
    rows = make_rows(n=4, user_text=text, whisper=text, duration=5.0, taken=10.0, cps=2.0)  # chars/duration
    q = A.run_pipeline(frame(rows))["quality"]
    assert q["share_matching_chars_per_audio_duration"] == 1.0
    assert q["share_matching_chars_per_time_taken"] == 0.0


def test_edit_flag_inconsistencies_detected():
    rows = make_rows(n=2, whisper="एक दो तीन", user_text="एक दो तीन")
    rows[0].update(is_edited="No", user_text="एक दो चार")
    rows[1].update(is_edited="Yes")
    t = A.run_pipeline(frame(rows))["tasks"]
    assert t.loc[0, "edit_flag_consistency"] == "flag_no_but_text_changed"
    assert t.loc[1, "edit_flag_consistency"] == "flag_yes_but_text_identical"


# ----- aggregation & risk --------------------------------------------------- #
def test_worker_aggregation_counts_and_denominators():
    rows = make_rows(user="w", n=10, duration=10, taken=20, edited="Yes")       # normal speed
    rows += make_rows(user="w", n=5, duration=10, taken=2, edited="No")         # fast + unedited
    for i, r in enumerate(rows):
        r["recording_url"] = f"synthetic://w/{i}"
    rows[0]["duration"] = 0                                                     # one invalid timing
    w = A.run_pipeline(frame(rows))["workers"].iloc[0]
    assert w["n_tasks_total"] == 15
    assert w["n_valid_timing"] == 14
    assert w["n_fast_tasks"] == 5
    assert w["pct_fast_tasks"] == pytest.approx(5 / 14)
    assert w["n_edit_flag_known"] == 15
    assert w["pct_unedited"] == pytest.approx(5 / 15)
    assert w["n_fast_and_unedited"] == 5


def test_minimum_sample_size_gives_insufficient_data():
    rows = make_rows(user="tiny", n=4, duration=10, taken=1, edited="No")  # very suspicious, but tiny
    for i, r in enumerate(rows):
        r["recording_url"] = f"synthetic://tiny/{i}"
    w = A.run_pipeline(frame(rows), A.Config(min_valid_tasks=10))["workers"].iloc[0]
    assert w["review_status"] == "insufficient_data"
    assert "provisional signals" in w["reasons"]

    w2 = A.run_pipeline(frame(rows), A.Config(min_valid_tasks=3))["workers"].iloc[0]
    assert w2["review_status"] != "insufficient_data"


def _unique(rows, user):
    for i, r in enumerate(rows):
        r["recording_url"] = f"synthetic://{user}/{i}"
    return rows


def test_risk_status_assignment():
    long = "क" * 30
    normal = _unique(make_rows("normal", 12, duration=10, taken=20, edited="Yes",
                               whisper=long + "x", user_text=long), "normal")
    # high no-edit but plausible timing -> must NOT be flagged on Signal B alone
    accurate = _unique(make_rows("accurate", 12, duration=10, taken=15, edited="No",
                                 whisper=long, user_text=long), "accurate")
    # fast only, sparse text (low char rate)
    fast = _unique(make_rows("fast", 12, duration=10, taken=3, edited="Yes",
                             whisper="हाँx", user_text="हाँ"), "fast")
    # fast + unedited + dense text -> A, B, C
    stamp = _unique(make_rows("stamp", 12, duration=10, taken=1, edited="No",
                              whisper=long, user_text=long), "stamp")
    res = A.run_pipeline(frame(normal + accurate + fast + stamp))
    status = dict(zip(res["workers"]["user_id"], res["workers"]["review_status"].astype(str)))
    assert status["normal"] == "normal"
    assert status["accurate"] == "normal"
    assert status["fast"] == "review_recommended"
    assert status["stamp"] == "priority_review"
    acc = res["workers"].set_index("user_id").loc["accurate"]
    assert "not flagged" in acc["notes"]
    stamp_row = res["workers"].set_index("user_id").loc["stamp"]
    assert stamp_row["signal_a_fast"] and stamp_row["signal_b_unedited_corroborated"]
    assert stamp_row["signal_c_high_throughput"]
    assert "no account action" in stamp_row["recommended_action"]


def test_thresholds_are_configurable():
    rows = _unique(make_rows("w", 12, duration=10, taken=6, edited="Yes"), "w")  # ratio 0.6
    default = A.run_pipeline(frame(rows))["workers"].iloc[0]
    strict = A.run_pipeline(frame(rows), A.Config(fast_ratio_threshold=0.7))["workers"].iloc[0]
    assert not default["signal_a_fast"]
    assert strict["signal_a_fast"]


def test_no_blocking_or_suspension_language_in_actions():
    res = A.run_pipeline(A.make_synthetic_dataset())
    actions = " ".join(res["workers"]["recommended_action"]).lower()
    assert "suspend" not in actions and "block" not in actions and "ban" not in actions


# ----- synthetic vs real labelling ------------------------------------------ #
def test_synthetic_outputs_are_labelled(tmp_path):
    res = A.run_pipeline(A.make_synthetic_dataset())
    assert res["synthetic"]
    assert (res["workers"]["data_source"] == A.SYNTHETIC_LABEL).all()
    assert res["workers"]["user_id"].str.startswith("SYN_").all()
    written = A.write_outputs(res, tmp_path, make_plots=True)
    assert written and all(p.name.startswith("SYNTHETIC_") for p in written)
    on_disk = pd.read_csv(tmp_path / "SYNTHETIC_worker_risk_summary.csv")
    assert (on_disk["data_source"] == A.SYNTHETIC_LABEL).all()


def test_real_csv_is_labelled_real_and_not_synthetic(tmp_path):
    rows = _unique(make_rows("r1", 12), "r1")
    p = tmp_path / "real.csv"
    pd.DataFrame(rows).to_csv(p, index=False, encoding="utf-8")
    out = tmp_path / "out"
    assert A.main(["--input", str(p), "--output-dir", str(out), "--no-plots"]) == 0
    df = pd.read_csv(out / "worker_risk_summary.csv")
    assert (df["data_source"] == A.REAL_LABEL).all()
    assert not any(f.name.startswith("SYNTHETIC_") for f in out.iterdir())


def test_synthetic_demo_covers_all_statuses():
    res = A.run_pipeline(A.make_synthetic_dataset())
    seen = set(res["workers"]["review_status"].astype(str))
    assert {"insufficient_data", "normal", "review_recommended", "priority_review"} <= seen


def test_charts_skipped_for_data_without_valid_values(tmp_path):
    rows = make_rows(n=2)
    for r in rows:
        r["duration"] = 0
        r["time_taken_by_user"] = 0
    res = A.run_pipeline(frame(rows))
    written = A.make_charts(res["tasks"], res["workers"], res["config"], tmp_path, True, "SYNTHETIC_")
    names = [p.name for p in written]
    assert not any("completion_ratio" in n or "char_rate" in n for n in names)
