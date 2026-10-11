"""
Josh Talks AI Product Task - Question 2
Detecting potentially low-quality human reviewers of Whisper-generated Hindi transcripts.

This script flags workers for HUMAN REVIEW. It never decides that a worker is
low quality, and it never recommends suspension or blocking. Every threshold
below is an unvalidated starting hypothesis, to be calibrated against reviewer
outcomes before anyone relies on it.

Usage
-----
    python analysis.py --input path/to/worker_tasks.csv      # real data
    python analysis.py --synthetic-demo                      # labelled demo only
    python analysis.py --help

Outputs go to ./outputs (real data) or ./outputs/synthetic_demo (demo).
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
import unicodedata
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "user_id",
    "recording_url",
    "whisper_text",
    "user_text",
    "is_edited",
    "duration",
    "time_taken_by_user",
    "segment_character_per_second",
]

SYNTHETIC_LABEL = "SYNTHETIC - NOT REAL JOSH TALKS DATA"
REAL_LABEL = "REAL - user-supplied dataset"

STATUS_ORDER = ["insufficient_data", "normal", "review_recommended", "priority_review"]

TRUE_TOKENS = {"yes", "y", "true", "t", "1", "1.0", "edited"}
FALSE_TOKENS = {"no", "n", "false", "f", "0", "0.0", "not edited", "unedited"}


# --------------------------------------------------------------------------- #
# Configuration: every threshold lives here, not scattered through the code.
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    # Sample-size requirements
    min_valid_tasks: int = 10           # tasks with valid timing needed before any status
    min_repeated_tasks: int = 3         # a signal needs at least this many suspicious tasks
    min_workers_for_mad: int = 5        # workers needed to compute a peer MAD baseline

    # Signal A - suspiciously fast completion (HYPOTHESIS, unvalidated)
    fast_ratio_threshold: float = 0.5   # time_taken / duration below this = fast task
    fast_share_threshold: float = 0.30  # share of a worker's tasks that are fast

    # Signal B - accepting AI output without edits (HYPOTHESIS, unvalidated)
    no_edit_share_threshold: float = 0.90       # high no-edit share (informational alone)
    fast_unedited_share_threshold: float = 0.30  # corroboration: fast AND unedited share
    material_similarity_threshold: float = 0.90  # text similarity below this = material change

    # Signal C - unusually high character throughput (HYPOTHESIS, unvalidated)
    char_rate_threshold: float = 15.0   # non-space characters per second of work
    high_rate_share_threshold: float = 0.30
    mad_z_threshold: float = 3.5        # robust z of worker median vs peer workers

    # Data-quality check on the supplied segment_character_per_second field
    cps_relative_tolerance: float = 0.20

    # Risk scoring
    weight_a: int = 1
    weight_b: int = 1
    weight_c: int = 1
    review_score: int = 1               # score >= this -> review_recommended
    priority_score: int = 2             # score >= this -> priority_review


# --------------------------------------------------------------------------- #
# Loading and validation
# --------------------------------------------------------------------------- #
class DataError(Exception):
    """Raised for problems a user can fix (missing file, missing columns, ...)."""


def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise DataError(
            f"Input file not found: {path}\n"
            "Provide the real worker-performance export, e.g.\n"
            "  python analysis.py --input data/worker_tasks.csv\n"
            "or run the labelled demo with: python analysis.py --synthetic-demo"
        )
    suffix = path.suffix.lower()
    try:
        if suffix in {".csv", ".txt"}:
            df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=True)
        elif suffix == ".tsv":
            df = pd.read_csv(path, sep="\t", encoding="utf-8-sig", dtype=str)
        elif suffix in {".xlsx", ".xls"}:
            df = pd.read_excel(path, dtype=str)  # requires openpyxl
        elif suffix in {".json", ".jsonl"}:
            df = pd.read_json(path, lines=suffix == ".jsonl", dtype=False)
        else:
            raise DataError(f"Unsupported file type '{suffix}'. Use CSV, TSV, XLSX or JSON.")
    except DataError:
        raise
    except Exception as exc:  # malformed file
        raise DataError(f"Could not parse {path}: {exc}") from exc
    return df


def validate_columns(df: pd.DataFrame) -> None:
    normalized = {c.strip().lower(): c for c in df.columns}
    missing = [c for c in REQUIRED_COLUMNS if c not in normalized]
    if missing:
        raise DataError(
            "Dataset is missing required column(s): "
            + ", ".join(missing)
            + "\nExpected columns: "
            + ", ".join(REQUIRED_COLUMNS)
            + "\nFound columns: "
            + ", ".join(map(str, df.columns))
        )


def normalize_is_edited(value) -> object:
    """Map Yes/No, True/False, 1/0 (any case) to True/False; anything else -> pd.NA."""
    if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NA:
        return pd.NA
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    token = str(value).strip().lower()
    if token in TRUE_TOKENS:
        return True
    if token in FALSE_TOKENS:
        return False
    return pd.NA


def count_non_space_chars(text) -> float:
    """Count characters excluding whitespace. Devanagari combining marks count as characters."""
    if not isinstance(text, str):
        return np.nan
    return float(len(re.sub(r"\s+", "", unicodedata.normalize("NFC", text))))


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", text).strip()


def text_similarity(a, b) -> float:
    """Character-level similarity in [0, 1]; NaN if either side is missing."""
    if not isinstance(a, str) or not isinstance(b, str):
        return np.nan
    a, b = _normalize_text(a), _normalize_text(b)
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Standardize columns, coerce types, drop exact duplicates. Never drops rows silently."""
    validate_columns(df)
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})[REQUIRED_COLUMNS].copy()
    quality = {"rows_input": int(len(df))}

    for col in ["user_id", "recording_url", "whisper_text", "user_text", "is_edited"]:
        df[col] = df[col].astype("object").where(df[col].notna(), None)
        df[col] = df[col].map(lambda v: v.strip() if isinstance(v, str) else v)
    df["user_id"] = df["user_id"].replace("", None)

    dup_mask = df.duplicated(keep="first")
    quality["duplicate_rows_removed"] = int(dup_mask.sum())
    df = df.loc[~dup_mask].reset_index(drop=True)

    missing_user = df["user_id"].isna()
    quality["rows_missing_user_id_removed"] = int(missing_user.sum())
    df = df.loc[~missing_user].reset_index(drop=True)
    df["user_id"] = df["user_id"].astype(str)

    for col in ["duration", "time_taken_by_user", "segment_character_per_second"]:
        raw = df[col]
        df[col] = pd.to_numeric(raw, errors="coerce").replace([np.inf, -np.inf], np.nan)
        quality[f"{col}_unparseable"] = int((raw.notna() & df[col].isna()).sum())

    df["is_edited_raw"] = df["is_edited"]
    df["is_edited"] = df["is_edited"].map(normalize_is_edited).astype("boolean")
    quality["is_edited_unrecognized_or_missing"] = int(df["is_edited"].isna().sum())
    quality["duration_missing_or_nonpositive"] = int((~(df["duration"] > 0)).sum())
    quality["time_taken_missing_or_nonpositive"] = int((~(df["time_taken_by_user"] > 0)).sum())
    quality["user_text_missing"] = int(df["user_text"].isna().sum())
    quality["whisper_text_missing"] = int(df["whisper_text"].isna().sum())
    quality["rows_after_cleaning"] = int(len(df))
    return df, quality


# --------------------------------------------------------------------------- #
# Task-level indicators
# --------------------------------------------------------------------------- #
def compute_task_metrics(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    t = df.copy()
    valid_duration = t["duration"] > 0
    valid_time = t["time_taken_by_user"] > 0
    t["valid_timing"] = valid_duration & valid_time

    # Signal A: completion ratio = time spent / audio length
    t["completion_ratio"] = np.where(
        t["valid_timing"], t["time_taken_by_user"] / t["duration"].where(valid_duration), np.nan
    )
    t["is_fast_task"] = t["completion_ratio"] < cfg.fast_ratio_threshold
    t.loc[~t["valid_timing"], "is_fast_task"] = False

    # Signal B: edit flag vs actual text change
    t["text_similarity"] = [text_similarity(w, u) for w, u in zip(t["whisper_text"], t["user_text"])]
    t["text_comparable"] = t["text_similarity"].notna()
    t["text_identical"] = t["text_similarity"] == 1.0
    t["material_change"] = t["text_similarity"] < cfg.material_similarity_threshold
    t["is_unedited"] = (t["is_edited"] == False).fillna(False).astype(bool)  # noqa: E712

    flag_no_but_changed = (t["is_edited"] == False) & t["text_comparable"] & ~t["text_identical"]  # noqa: E712
    flag_yes_but_same = (t["is_edited"] == True) & t["text_identical"]  # noqa: E712
    t["edit_flag_consistency"] = "consistent"
    t.loc[~t["text_comparable"] | t["is_edited"].isna(), "edit_flag_consistency"] = "not_checkable"
    t.loc[flag_no_but_changed.fillna(False).astype(bool), "edit_flag_consistency"] = "flag_no_but_text_changed"
    t.loc[flag_yes_but_same.fillna(False).astype(bool), "edit_flag_consistency"] = "flag_yes_but_text_identical"
    t["is_fast_and_unedited"] = t["is_fast_task"] & t["is_unedited"]

    # Signal C: recomputed character throughput per second of the worker's time
    t["user_chars"] = t["user_text"].map(count_non_space_chars)
    t["char_rate"] = np.where(
        valid_time & t["user_chars"].notna(),
        t["user_chars"] / t["time_taken_by_user"].where(valid_time),
        np.nan,
    )
    t["valid_char_rate"] = t["char_rate"].notna()
    t["is_high_rate_task"] = (t["char_rate"] > cfg.char_rate_threshold).fillna(False).astype(bool)

    # The definition of the supplied field is not documented; test both candidate denominators.
    t["chars_per_audio_second"] = np.where(
        valid_duration & t["user_chars"].notna(),
        t["user_chars"] / t["duration"].where(valid_duration),
        np.nan,
    )
    supplied = t["segment_character_per_second"]
    t["cps_rel_diff_vs_time"] = (supplied - t["char_rate"]).abs() / t["char_rate"].where(t["char_rate"] > 0)
    t["cps_rel_diff_vs_duration"] = (supplied - t["chars_per_audio_second"]).abs() / t[
        "chars_per_audio_second"
    ].where(t["chars_per_audio_second"] > 0)
    t["cps_matches_time_based"] = t["cps_rel_diff_vs_time"] <= cfg.cps_relative_tolerance
    t["cps_matches_duration_based"] = t["cps_rel_diff_vs_duration"] <= cfg.cps_relative_tolerance
    return t


def cps_discrepancy_summary(tasks: pd.DataFrame) -> dict:
    supplied = tasks["segment_character_per_second"].notna()
    n = int(supplied.sum())
    if n == 0:
        return {"rows_with_supplied_cps": 0}
    return {
        "rows_with_supplied_cps": n,
        "share_matching_chars_per_time_taken": round(float(tasks.loc[supplied, "cps_matches_time_based"].mean()), 3),
        "share_matching_chars_per_audio_duration": round(
            float(tasks.loc[supplied, "cps_matches_duration_based"].mean()), 3
        ),
        "median_rel_diff_vs_time_taken": _round(tasks.loc[supplied, "cps_rel_diff_vs_time"].median()),
        "median_rel_diff_vs_audio_duration": _round(tasks.loc[supplied, "cps_rel_diff_vs_duration"].median()),
    }


# --------------------------------------------------------------------------- #
# Worker-level aggregation and risk logic
# --------------------------------------------------------------------------- #
def _round(x, nd=3):
    return None if x is None or pd.isna(x) else round(float(x), nd)


def _safe_share(num, den):
    return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def aggregate_workers(tasks: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if tasks.empty:
        return pd.DataFrame()
    g = tasks.groupby("user_id", sort=True)
    timed = tasks[tasks["valid_timing"]]
    gt = timed.groupby("user_id")
    edit_known = tasks[tasks["is_edited"].notna()]
    ge = edit_known.groupby("user_id")
    comparable = tasks[tasks["text_comparable"]]
    gc = comparable.groupby("user_id")
    rated = tasks[tasks["valid_char_rate"]]
    gr = rated.groupby("user_id")

    w = pd.DataFrame(index=g.size().index)
    w["n_tasks_total"] = g.size()
    # Signal A
    w["n_valid_timing"] = gt.size()
    w["median_completion_ratio"] = gt["completion_ratio"].median()
    w["n_fast_tasks"] = gt["is_fast_task"].sum()
    # Signal B
    w["n_edit_flag_known"] = ge.size()
    w["n_unedited"] = ge["is_unedited"].sum()
    w["n_fast_and_unedited"] = gt["is_fast_and_unedited"].sum()
    w["n_text_comparable"] = gc.size()
    w["n_text_identical"] = gc["text_identical"].sum()
    w["n_material_change"] = gc["material_change"].sum()
    w["n_edit_flag_inconsistent"] = g["edit_flag_consistency"].agg(
        lambda s: int(s.isin(["flag_no_but_text_changed", "flag_yes_but_text_identical"]).sum())
    )
    # Signal C
    w["n_valid_char_rate"] = gr.size()
    w["median_char_rate"] = gr["char_rate"].median()
    w["p90_char_rate"] = gr["char_rate"].quantile(0.9)
    w["n_high_rate_tasks"] = gr["is_high_rate_task"].sum()

    count_cols = [c for c in w.columns if c.startswith("n_")]
    w[count_cols] = w[count_cols].fillna(0).astype(int)

    w["pct_fast_tasks"] = _safe_share(w["n_fast_tasks"], w["n_valid_timing"])
    w["pct_unedited"] = _safe_share(w["n_unedited"], w["n_edit_flag_known"])
    w["pct_fast_and_unedited"] = _safe_share(w["n_fast_and_unedited"], w["n_valid_timing"])
    w["pct_material_change"] = _safe_share(w["n_material_change"], w["n_text_comparable"])
    w["pct_high_rate_tasks"] = _safe_share(w["n_high_rate_tasks"], w["n_valid_char_rate"])

    # Peer-relative robust z (MAD) of each eligible worker's median char rate.
    eligible = w["n_valid_char_rate"] >= cfg.min_valid_tasks
    w["char_rate_robust_z"] = np.nan
    w["mad_baseline_workers"] = int(eligible.sum())
    if eligible.sum() >= cfg.min_workers_for_mad:
        med = w.loc[eligible, "median_char_rate"]
        center = med.median()
        mad = (med - center).abs().median()
        if mad > 0:
            w.loc[eligible, "char_rate_robust_z"] = 0.6745 * (med - center) / mad
    return w.reset_index().rename(columns={"index": "user_id"})


def assign_risk(workers: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if workers.empty:
        return workers
    w = workers.copy()
    statuses, scores, reasons, notes = [], [], [], []
    sig_a, sig_b, sig_c = [], [], []
    for _, r in w.iterrows():
        why, info = [], []
        a = (
            r["n_valid_timing"] > 0
            and r["pct_fast_tasks"] >= cfg.fast_share_threshold
            and r["n_fast_tasks"] >= cfg.min_repeated_tasks
        )
        if a:
            why.append(
                f"A: {r['n_fast_tasks']}/{r['n_valid_timing']} tasks ({r['pct_fast_tasks']:.0%}) finished in "
                f"< {cfg.fast_ratio_threshold}x audio length (median ratio {r['median_completion_ratio']:.2f})"
            )

        high_no_edit = r["n_edit_flag_known"] > 0 and r["pct_unedited"] >= cfg.no_edit_share_threshold
        b = bool(
            high_no_edit
            and r["pct_fast_and_unedited"] >= cfg.fast_unedited_share_threshold
            and r["n_fast_and_unedited"] >= cfg.min_repeated_tasks
        )
        if b:
            why.append(
                f"B: {r['pct_unedited']:.0%} of tasks unedited AND {r['n_fast_and_unedited']} tasks "
                f"were both unedited and fast (corroborated)"
            )
        elif high_no_edit:
            info.append(
                f"High no-edit share ({r['pct_unedited']:.0%}) without fast-completion corroboration - "
                "not flagged; Whisper output may simply have been accurate"
            )

        c_share = (
            r["n_valid_char_rate"] > 0
            and r["pct_high_rate_tasks"] >= cfg.high_rate_share_threshold
            and r["n_high_rate_tasks"] >= cfg.min_repeated_tasks
        )
        c_mad = pd.notna(r["char_rate_robust_z"]) and r["char_rate_robust_z"] >= cfg.mad_z_threshold
        c = bool(c_share or c_mad)
        if c:
            parts = []
            if c_share:
                parts.append(
                    f"{r['n_high_rate_tasks']}/{r['n_valid_char_rate']} tasks above "
                    f"{cfg.char_rate_threshold:g} chars/s"
                )
            if c_mad:
                parts.append(f"median rate {r['median_char_rate']:.1f} chars/s is a peer outlier "
                             f"(robust z {r['char_rate_robust_z']:.1f})")
            why.append("C: " + "; ".join(parts))

        if r["n_edit_flag_inconsistent"] > 0:
            info.append(f"{r['n_edit_flag_inconsistent']} task(s) where is_edited disagrees with the text diff "
                        "(data-quality check, not a risk signal)")

        score = cfg.weight_a * a + cfg.weight_b * b + cfg.weight_c * c
        if r["n_valid_timing"] < cfg.min_valid_tasks:
            status = "insufficient_data"
            why = [f"Only {r['n_valid_timing']} task(s) with valid timing; {cfg.min_valid_tasks} required"] + (
                ["(provisional signals present: " + " | ".join(why) + ")"] if why else []
            )
        elif score >= cfg.priority_score:
            status = "priority_review"
        elif score >= cfg.review_score:
            status = "review_recommended"
        else:
            status = "normal"
            why = why or ["No repeated warning pattern at current thresholds"]

        statuses.append(status)
        scores.append(int(score))
        reasons.append(" | ".join(why))
        notes.append(" | ".join(info))
        sig_a.append(bool(a))
        sig_b.append(bool(b))
        sig_c.append(bool(c))

    w["signal_a_fast"] = sig_a
    w["signal_b_unedited_corroborated"] = sig_b
    w["signal_c_high_throughput"] = sig_c
    w["risk_score"] = scores
    w["review_status"] = pd.Categorical(statuses, categories=STATUS_ORDER, ordered=True)
    w["reasons"] = reasons
    w["notes"] = notes
    w["recommended_action"] = w["review_status"].map(
        {
            "insufficient_data": "Keep collecting data; no action",
            "normal": "No action",
            "review_recommended": "Sample 5-10 transcripts for blind quality review",
            "priority_review": "Prioritise sampled quality review; no account action without reviewer evidence",
        }
    ).astype(str)
    return w.sort_values(["review_status", "risk_score", "user_id"], ascending=[False, False, True])


# --------------------------------------------------------------------------- #
# Synthetic data (in-memory only, for demos and tests)
# --------------------------------------------------------------------------- #
_SYN_SENTENCES = [
    "मैं कल बाज़ार गया था और वहाँ बहुत भीड़ थी",
    "हाँ हाँ बिल्कुल सही बात है",
    "हम्म",
    "मतलब आप ये कहना चाहते हैं कि बिज़नेस में नुकसान हुआ",
    "बच्चों की पढ़ाई के लिए हमने नया फ़ोन लिया",
    "गाँव में अभी भी बिजली की समस्या रहती है",
    "आपका दिन कैसा रहा",
    "मुझे लगता है कि मेहनत से सब कुछ हो सकता है",
]


def _perturb(text: str, rng: np.random.Generator) -> str:
    words = text.split()
    if len(words) > 1:
        i = int(rng.integers(len(words)))
        words[i] = words[i] + "ा"
    else:
        words.append("जी")
    return " ".join(words)


def make_synthetic_dataset(seed: int = 7) -> pd.DataFrame:
    """Return an in-memory SYNTHETIC dataset. Never written to disk as input data."""
    rng = np.random.default_rng(seed)
    # Speech density = characters spoken per second of audio. Long silences or fillers lower it.
    profiles = {
        # name: (n_tasks, completion-ratio range, P(unedited), speech-density range chars/s)
        "SYN_normal_01": (40, (1.2, 3.0), 0.55, (8, 13)),
        "SYN_normal_02": (35, (1.0, 2.5), 0.60, (8, 13)),
        "SYN_normal_03": (30, (1.3, 3.5), 0.50, (8, 13)),
        "SYN_normal_04": (25, (0.9, 2.2), 0.65, (8, 13)),
        "SYN_normal_05": (30, (1.1, 2.8), 0.45, (8, 13)),
        "SYN_accurate_whisper_06": (30, (0.9, 1.8), 0.95, (8, 13)),  # high no-edit, plausible time
        "SYN_fast_rubberstamp_07": (30, (0.1, 0.45), 0.97, (8, 13)),  # fast + unedited
        "SYN_fast_sparse_audio_08": (25, (0.25, 0.6), 0.40, (2, 4)),  # fast, but audio mostly silence
        "SYN_high_throughput_09": (30, (0.55, 0.9), 0.50, (14, 18)),  # not 'fast', but dense text
        "SYN_new_worker_10": (4, (0.1, 0.3), 1.0, (8, 13)),  # insufficient data
    }
    rows = []
    for uid, (n, (lo, hi), p_unedited, (d_lo, d_hi)) in profiles.items():
        for i in range(n):
            text = _SYN_SENTENCES[int(rng.integers(len(_SYN_SENTENCES)))]
            duration = round(max(count_non_space_chars(text) / float(rng.uniform(d_lo, d_hi)), 0.5), 2)
            ratio = float(rng.uniform(lo, hi))
            taken = round(max(duration * ratio, 0.2), 2)
            unedited = rng.random() < p_unedited
            whisper = text if unedited else _perturb(text, rng)
            user = text
            edited_repr = ["No", "FALSE", "0", "no"][i % 4] if unedited else ["Yes", "TRUE", "1", "yes"][i % 4]
            rows.append(
                {
                    "user_id": uid,
                    "recording_url": f"synthetic://{uid}/{i}.wav",
                    "whisper_text": whisper,
                    "user_text": user,
                    "is_edited": edited_repr,
                    "duration": duration,
                    "time_taken_by_user": taken,
                    # Simulates the supplied field as chars per AUDIO second (definition is undocumented)
                    "segment_character_per_second": round(count_non_space_chars(user) / duration, 3),
                }
            )
    df = pd.DataFrame(rows)
    # Deliberately malformed rows to exercise cleaning
    bad = [
        {**rows[0], "duration": 0},
        {**rows[1], "duration": "abc"},
        {**rows[2], "time_taken_by_user": None},
        {**rows[3], "user_text": None},
        {**rows[4], "is_edited": "maybe"},
        {**rows[5], "user_text": rows[5]["whisper_text"] + " जी", "is_edited": "No"},  # flag/text mismatch
    ]
    df = pd.concat([df, pd.DataFrame(bad), df.iloc[[10, 11]]], ignore_index=True)  # + 2 exact duplicates
    df.attrs["is_synthetic"] = True
    return df


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = "#2a78d6"
STATUS_COLORS = {
    "insufficient_data": "#a3a29c",
    "normal": "#0ca30c",
    "review_recommended": "#fab219",
    "priority_review": "#d03b3b",
}


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _finish(fig, title: str, synthetic: bool, path: Path):
    fig.patch.set_facecolor(SURFACE)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=INK, fontweight="bold")
    if synthetic:
        fig.text(
            0.01, 0.005,
            "SYNTHETIC DEMONSTRATION DATA - not real Josh Talks workers. Thresholds are unvalidated hypotheses.",
            color="#d03b3b", fontsize=9, fontweight="bold", ha="left", va="bottom",
        )
    else:
        fig.text(0.01, 0.005, "Thresholds are unvalidated hypotheses; flags mean 'review', not 'guilty'.",
                 color=INK_MUTED, fontsize=8, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(path, dpi=130)


def make_charts(tasks: pd.DataFrame, workers: pd.DataFrame, cfg: Config, out: Path,
                synthetic: bool, prefix: str = "") -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    saved: list[Path] = []
    tag = " [SYNTHETIC]" if synthetic else ""

    ratios = tasks["completion_ratio"].dropna()
    if len(ratios) > 0:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        clipped = ratios.clip(upper=5)
        ax.hist(clipped, bins=40, color=SERIES, edgecolor=SURFACE, linewidth=1)
        ax.axvline(cfg.fast_ratio_threshold, color=INK, linestyle="--", linewidth=1.5)
        ax.text(cfg.fast_ratio_threshold, ax.get_ylim()[1] * 0.95,
                f"  hypothesis: < {cfg.fast_ratio_threshold}x", color=INK, fontsize=9, va="top")
        ax.set_xlabel("time_taken_by_user / duration (values above 5 shown at 5)", color=INK_MUTED)
        ax.set_ylabel("Tasks", color=INK_MUTED)
        _style_axes(ax)
        p = out / f"{prefix}completion_ratio_distribution.png"
        _finish(fig, f"Task completion ratio (n={len(ratios)}){tag}", synthetic, p)
        plt.close(fig)
        saved.append(p)

    rates = tasks["char_rate"].dropna()
    if len(rates) > 0:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(rates.clip(upper=60), bins=40, color=SERIES, edgecolor=SURFACE, linewidth=1)
        ax.axvline(cfg.char_rate_threshold, color=INK, linestyle="--", linewidth=1.5)
        ax.text(cfg.char_rate_threshold, ax.get_ylim()[1] * 0.95,
                f"  hypothesis: > {cfg.char_rate_threshold:g} chars/s", color=INK, fontsize=9, va="top")
        ax.set_xlabel("Non-space characters in user_text per second of time taken (capped at 60)", color=INK_MUTED)
        ax.set_ylabel("Tasks", color=INK_MUTED)
        _style_axes(ax)
        p = out / f"{prefix}char_rate_distribution.png"
        _finish(fig, f"Character throughput per task (n={len(rates)}){tag}", synthetic, p)
        plt.close(fig)
        saved.append(p)

    if not workers.empty:
        w = workers.sort_values("user_id", ascending=False)
        panels = [
            ("pct_fast_tasks", "A: share of fast tasks", cfg.fast_share_threshold),
            ("pct_fast_and_unedited", "B: share fast AND unedited", cfg.fast_unedited_share_threshold),
            ("pct_high_rate_tasks", f"C: share > {cfg.char_rate_threshold:g} chars/s", cfg.high_rate_share_threshold),
        ]
        height = max(3.5, 0.32 * len(w) + 1.5)
        fig, axes = plt.subplots(1, 3, figsize=(13, height), sharey=True)
        colors = [STATUS_COLORS[str(s)] for s in w["review_status"]]
        for ax, (col, title, thr) in zip(axes, panels):
            vals = w[col].fillna(0).to_numpy() * 100
            ax.barh(w["user_id"], vals, color=colors, height=0.6)
            ax.axvline(thr * 100, color=INK, linestyle="--", linewidth=1.2)
            ax.set_xlim(0, 100)
            ax.set_title(title, fontsize=10, color=INK, loc="left")
            ax.set_xlabel("% of worker's valid tasks", color=INK_MUTED, fontsize=9)
            _style_axes(ax)
            ax.grid(axis="y", visible=False)
            ax.grid(axis="x", color=GRID, linewidth=0.8)
        from matplotlib.patches import Patch

        handles = [Patch(color=STATUS_COLORS[s], label=s) for s in STATUS_ORDER]
        fig.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 0.0), ncol=4, fontsize=8,
                   frameon=False)
        p = out / f"{prefix}worker_signal_comparison.png"
        _finish(fig, f"Worker-level warning signals (colour = review status, dashed = hypothesis threshold){tag}", synthetic, p)
        plt.close(fig)
        saved.append(p)

        counts = workers["review_status"].value_counts().reindex(STATUS_ORDER, fill_value=0)
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(counts.index, counts.values, color=[STATUS_COLORS[s] for s in counts.index], width=0.6)
        for bar, v in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(int(v)),
                    ha="center", va="bottom", color=INK, fontsize=10)
        ax.set_ylabel("Workers", color=INK_MUTED)
        _style_axes(ax)
        p = out / f"{prefix}workers_by_review_status.png"
        _finish(fig, f"Workers by review status (n={len(workers)}){tag}", synthetic, p)
        plt.close(fig)
        saved.append(p)
    return saved


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def run_pipeline(raw: pd.DataFrame, cfg: Config | None = None) -> dict:
    """Pure function: raw records in, task metrics / worker risk / quality report out."""
    cfg = cfg or Config()
    synthetic = bool(raw.attrs.get("is_synthetic", False))
    if raw.empty:
        raise DataError("Dataset contains no rows.")
    cleaned, quality = clean_dataset(raw)
    if cleaned.empty:
        raise DataError("No usable rows after cleaning (all rows were duplicates or missing user_id).")
    tasks = compute_task_metrics(cleaned, cfg)
    workers = assign_risk(aggregate_workers(tasks, cfg), cfg)
    label = SYNTHETIC_LABEL if synthetic else REAL_LABEL
    tasks.insert(0, "data_source", label)
    workers.insert(0, "data_source", label)
    quality.update(cps_discrepancy_summary(tasks))
    quality["edit_flag_consistency"] = tasks["edit_flag_consistency"].value_counts().to_dict()
    quality["data_source"] = label
    return {"tasks": tasks, "workers": workers, "quality": quality, "synthetic": synthetic, "config": cfg}


def write_outputs(result: dict, out_dir: Path, make_plots: bool = True) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    synthetic = result["synthetic"]
    prefix = "SYNTHETIC_" if synthetic else ""
    written = []
    task_cols = [
        "data_source", "user_id", "recording_url", "is_edited_raw", "is_edited", "duration",
        "time_taken_by_user", "valid_timing", "completion_ratio", "is_fast_task", "text_similarity",
        "material_change", "edit_flag_consistency", "is_fast_and_unedited", "user_chars", "char_rate",
        "is_high_rate_task", "segment_character_per_second", "chars_per_audio_second",
        "cps_matches_time_based", "cps_matches_duration_based",
    ]
    p = out_dir / f"{prefix}task_level_metrics.csv"
    result["tasks"][task_cols].to_csv(p, index=False, encoding="utf-8-sig")
    written.append(p)

    p = out_dir / f"{prefix}worker_risk_summary.csv"
    result["workers"].round(4).to_csv(p, index=False, encoding="utf-8-sig")
    written.append(p)

    p = out_dir / f"{prefix}data_quality_report.csv"
    flat = {k: (str(v) if isinstance(v, dict) else v) for k, v in result["quality"].items()}
    pd.DataFrame(sorted(flat.items()), columns=["check", "value"]).to_csv(p, index=False, encoding="utf-8-sig")
    written.append(p)

    p = out_dir / f"{prefix}config_used.csv"
    pd.DataFrame(sorted(asdict(result["config"]).items()), columns=["parameter", "value"]).to_csv(p, index=False)
    written.append(p)

    if make_plots:
        written += make_charts(result["tasks"], result["workers"], result["config"], out_dir, synthetic, prefix)
    return written


def print_summary(result: dict, written: list[Path]) -> None:
    q, w = result["quality"], result["workers"]
    banner = "=" * 72
    print(banner)
    print(f"DATA SOURCE: {q['data_source']}")
    if result["synthetic"]:
        print("!! These results describe SYNTHETIC test workers, not real Josh Talks transcribers. !!")
    print(banner)
    print(f"Rows in: {q['rows_input']} | after cleaning: {q['rows_after_cleaning']} | "
          f"duplicates removed: {q['duplicate_rows_removed']} | missing user_id removed: "
          f"{q['rows_missing_user_id_removed']}")
    print(f"Invalid/zero duration: {q['duration_missing_or_nonpositive']} | invalid/zero time taken: "
          f"{q['time_taken_missing_or_nonpositive']} | unrecognised is_edited: "
          f"{q['is_edited_unrecognized_or_missing']} | missing user_text: {q['user_text_missing']}")
    print(f"is_edited vs text diff: {q['edit_flag_consistency']}")
    if q.get("rows_with_supplied_cps"):
        print(f"segment_character_per_second matches chars/time_taken in "
              f"{q['share_matching_chars_per_time_taken']:.0%} of rows, chars/duration in "
              f"{q['share_matching_chars_per_audio_duration']:.0%}")
    print(f"Workers analysed: {len(w)}")
    for status, n in w["review_status"].value_counts().reindex(STATUS_ORDER, fill_value=0).items():
        print(f"  {status:<20} {n}")
    flagged = w[w["review_status"].isin(["review_recommended", "priority_review"])]
    for _, r in flagged.iterrows():
        print(f"  -> {r['user_id']} [{r['review_status']}]: {r['reasons']}")
    print("Flags mean 'sample for human review'. No automatic account action is implied.")
    print("Outputs:")
    for p in written:
        print(f"  {p}")


def build_config(args: argparse.Namespace) -> Config:
    cfg = Config()
    for f in fields(Config):
        val = getattr(args, f.name, None)
        if val is not None:
            setattr(cfg, f.name, val)
    return cfg


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--input", help="Path to the real worker-performance file (CSV/TSV/XLSX/JSON)")
    src.add_argument("--synthetic-demo", action="store_true",
                     help="Run on an in-memory SYNTHETIC dataset (clearly labelled; for demonstration only)")
    p.add_argument("--output-dir", default=None, help="Default: outputs/ (real) or outputs/synthetic_demo/")
    p.add_argument("--no-plots", action="store_true")
    for f in fields(Config):
        p.add_argument(f"--{f.name.replace('_', '-')}", dest=f.name,
                       type=int if f.type in ("int", int) else float, default=None,
                       help=f"(default {f.default})")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    here = Path(__file__).resolve().parent
    cfg = build_config(args)
    try:
        if args.synthetic_demo:
            raw = make_synthetic_dataset()
            out = Path(args.output_dir) if args.output_dir else here / "outputs" / "synthetic_demo"
        elif args.input:
            raw = load_dataset(args.input)
            raw.attrs["is_synthetic"] = False
            out = Path(args.output_dir) if args.output_dir else here / "outputs"
        else:
            print(
                "No dataset supplied. The real Josh Talks worker-performance dataset was not found in this\n"
                "project, so no empirical analysis has been performed.\n\n"
                "  Real data:      python analysis.py --input path/to/worker_tasks.csv\n"
                f"  Required cols:  {', '.join(REQUIRED_COLUMNS)}\n"
                "  Labelled demo:  python analysis.py --synthetic-demo",
                file=sys.stderr,
            )
            return 2
        result = run_pipeline(raw, cfg)
    except DataError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    written = write_outputs(result, out, make_plots=not args.no_plots)
    print_summary(result, written)
    return 0


if __name__ == "__main__":
    sys.exit(main())
