# Detecting Potentially Low-Quality Transcription Reviewers

**Josh Talks AI Product Task, Question 2: product and analysis report**

> **Read this first: status of the evidence.**
> No Josh Talks worker-performance dataset or task PDF was available in the working directory. **No real workers have been analysed, and no real worker has been identified as low quality.** This report describes (a) the detection logic, (b) an implementation that runs on the real export as soon as it is supplied, and (c) a demonstration on a clearly labelled **synthetic** dataset that exists only to show the pipeline working. Every threshold is an **unvalidated hypothesis**.

Labels used throughout:
**[Given]** facts from the task · **[Proposed]** product logic · **[Hypothesis]** unvalidated threshold · **[Synthetic]** demo output · **[Empirical]** real-data finding (none yet)

---

## 1. Executive summary

Josh Talks pays human reviewers to correct Whisper's Hindi transcripts. A reviewer who skips the audio and accepts Whisper's output looks productive but adds no quality, which leaves errors in the training data. I propose three interpretable warning signals computed from data the platform already logs:

| Signal | What it measures | Provisional trigger [Hypothesis] |
|---|---|---|
| **A: Fast completion** | time spent ÷ audio length | ≥30% of a worker's tasks below 0.5× |
| **B: Unedited acceptance** | share of tasks submitted without edits | ≥90% unedited **and** ≥30% of tasks both fast and unedited |
| **C: High throughput** | non-space characters ÷ time spent | ≥30% of tasks above 15 chars/s, **or** a peer outlier (robust z ≥ 3.5) |

Signals are aggregated per worker. A worker must have a minimum number of valid tasks (default 10) before getting any status. The output is one of four statuses (`insufficient_data`, `normal`, `review_recommended`, `priority_review`), each with plain-language reasons. **A flag only sends sampled work to human quality review.** The system never suspends or blocks anyone automatically.

## 2. Problem statement and business impact

- **[Given]** Workers review Whisper-generated Hindi transcripts. For each task the platform records `user_id`, `recording_url`, `whisper_text`, `user_text`, `is_edited`, `duration`, `time_taken_by_user` and `segment_character_per_second`.
- **[Given, from the public catalogue, context only]** The Hindi conversational data is spontaneous two-speaker speech recorded on mobile phones. Segments are short (under 1 s to about 15 s) and include fillers such as "हम्म" and "हाँ हाँ" and English words written in Devanagari. Short segments make per-task timing noisy. Fillers and silence mean characters per audio second vary widely.
- **Impact.** (1) *Data quality:* uncorrected Whisper errors become ground truth, so downstream models learn their own mistakes. (2) *Cost:* the company pays for reviews that added nothing. (3) *Fairness:* careless quality control can wrongly penalise good, fast workers and damage trust in the workforce. The design has to deal with both (1) and (3).

## 3. Available data and assumptions

| Item | Status |
|---|---|
| Worker-performance export | **Not available.** The pipeline accepts it via `--input`. |
| Task PDF | Received Product Task, July 2026 (Question 2, pages 6–9). The supplied PDF contains the assignment brief but no worker-task CSV or Excel dataset. |
| Hindi audio catalogue | Used only for context (segment length, speech style). **Not** treated as worker data. |
| Ground-truth quality labels | None. This is why no ML classifier is trained (see §8). |

Assumptions [Proposed], each checked or handled by the code:
1. `duration` and `time_taken_by_user` are in seconds. Rows where either is ≤0, missing or non-numeric are kept in the counts but excluded from ratios.
2. `is_edited` may arrive as Yes/No, True/False or 1/0. Unrecognised values become *unknown*, not *No*.
3. The assignment explicitly defines `segment_character_per_second` as non-space characters in `user_text` divided by `time_taken_by_user` (in seconds). Signal C uses characters divided by worker time. In addition, the pipeline's data-quality check independently compares the supplied field against both worker-time and audio-duration calculations to detect potential discrepancies or logging bugs in data exports (in the synthetic demonstration, the simulated field was generated against audio duration in 99% of rows to exercise this discrepancy check).
4. `time_taken_by_user` may capture only part of the workflow, for example excluding replays or including idle time. This uncertainty is the main reason the thresholds are hypotheses.

## 4. The three warning signals

### Signal A: Suspiciously fast completion
- **Formula:** `completion_ratio = time_taken_by_user / duration` (valid only when both are > 0).
- **Worker aggregates:** valid task count, median ratio, % of tasks below threshold, count of fast tasks.
- **Rationale:** checking a transcript properly means listening to the audio at least once, so time spent far below the audio length is hard to explain.
- **Provisional threshold [Hypothesis]:** ratio < 0.5 on ≥30% of tasks, with at least 3 such tasks.
- **Limitations:** 1.5–2× playback is legitimate. Very short clips such as "हाँ" can be verified almost instantly. Long silent stretches inflate `duration`. Timing may be logged incompletely. *Mitigation:* the signal fires only on repeated patterns, and Signal C acts as a counter-check (see the A/C note below).

### Signal B: Repeatedly accepting AI transcripts without edits
- **Metrics:** % of tasks with `is_edited = No`, count of unchanged tasks, count of tasks where text similarity between `whisper_text` and `user_text` is below 0.90 (a material change), and count of tasks where the `is_edited` flag **contradicts** the actual text diff.
- **Rationale:** a reviewer who never edits may not be reviewing at all.
- **Key safeguard [Proposed]:** **a high no-edit rate alone never raises a flag.** Whisper is often right, and a worker assigned clean audio *should* rarely edit. B counts only when it is corroborated: ≥90% unedited **and** ≥30% of tasks both unedited **and** fast. Uncorroborated high no-edit rates appear as an informational note.
- **Flag/text inconsistencies** (for example, `is_edited = No` but the text differs) are reported as a **data-quality** issue for engineering and are not counted against the worker.

### Signal C: Unusually high character throughput
- **Formula:** `char_rate = non_space_characters(user_text) / time_taken_by_user`. Devanagari vowel signs count as characters after Unicode NFC normalisation.
- **Two triggers [Hypothesis]:** (i) ≥30% of tasks above 15 chars/s (≥3 tasks), or (ii) the worker's median rate is a peer outlier, with robust z = 0.6745 × (median − peer median) / MAD ≥ 3.5. The outlier test runs only when at least 5 workers each have ≥10 valid tasks.
- **Rationale:** it measures how much verified text a worker produced per second of effort. 15 chars/s is close to or above normal conversational speech density, so sustained rates above it suggest the audio was not fully heard.
- **Limitations:** a high rate is **not** proof of inaccuracy, and fast typists and replay-free reviewers exist. The 15 chars/s figure is a starting point, not an established standard. It must be calibrated on the real distribution.

**How A and C relate.** Because `char_rate = (chars / duration) / completion_ratio`, both signals share `time_taken_by_user`, so they are **not independent**. C adds a correction for speech density. A fast review of a mostly silent clip trips A but not C, while a dense clip reviewed at 0.7× trips C but not A. When **both** fire, the worker was fast *and* produced a lot of text per second, which is the strongest timing-based pattern. Still, all automated signals are behavioural. **The only truly independent evidence is a human quality review of sampled transcripts**, which the workflow requires before any consequence.

## 5. Worker-level aggregation and risk scoring [Proposed]

1. Compute task-level indicators, then aggregate by `user_id` (over a rolling window in production).
2. Every worker metric is stored with its **denominator** (`n_valid_timing`, `n_edit_flag_known`, `n_text_comparable`, `n_valid_char_rate`).
3. Fewer than `min_valid_tasks` (default 10) valid tasks → `insufficient_data`, whatever the signals say. Provisional signals are still shown in the reasons so they can be watched.
4. `risk_score = wA·A + wB·B + wC·C` (default weights 1).
   `0 → normal`, `≥1 → review_recommended`, `≥2 → priority_review`.
5. Each worker gets a `reasons` string that cites the actual counts (e.g. "A: 21/25 tasks (84%) finished in < 0.5x audio length"), plus `notes` for informational and data-quality items.
6. All thresholds, minimum counts and weights live in one `Config` object and can be overridden on the CLI (e.g. `--fast-ratio-threshold 0.4`). Each run writes the config it used to `config_used.csv` for audit.

## 6. False-positive risks and safeguards

| Risk | Safeguard |
|---|---|
| Legitimate fast playback or skilled workers | Repeated-pattern thresholds; review, not penalty; calibrate thresholds on reviewer-verified outcomes |
| Easy or short clips assigned to some workers | Ratios normalise by duration; in production, compare workers within the same audio-difficulty or segment-length band |
| Accurate Whisper output → legitimately no edits | B never fires alone |
| Faulty timing or `is_edited` logging | Data-quality report: unparseable values, flag/text contradictions, `segment_character_per_second` definition check |
| New workers with few tasks | `insufficient_data` status |
| Threshold drift or bias against groups (e.g. device type, region) | Monitor flag rates and reviewer-confirmed precision by segment; appeals process |
| Over-reliance on automation | No automatic suspension; any account action needs reviewer evidence plus a human decision |

## 7. Proposed automated detection workflow

Capture → validate/normalise → task indicators → rolling worker aggregates → rules → status and reasons → **human review queue (blind-sample 5–10 transcripts, scored against the reference audio)** → escalate only when reviewers confirm quality failures → record outcomes → recalibrate thresholds → worker appeal path. Full design, Mermaid diagram, schema and controls: [`system_design.md`](system_design.md).

## 8. Engineering recommendations

1. **Log better timing:** active listening time, play/seek/replay events and playback speed. This would sharply cut false positives on Signal A.
2. **Standardize `segment_character_per_second`** calculation server-side to guarantee consistency with the defined formula (non-space characters divided by worker time).
3. **Derive `is_edited` from the text diff** at submit time, rather than relying on a separately set flag.
4. **Seed known-answer "gold" tasks** (e.g. deliberately corrupted Whisper output) into the queue. A worker who accepts a planted error gives direct, independent quality evidence. This is the single highest-value addition.
5. **Store reviewer verdicts** in a structured form. Only these labels can later justify a supervised model.
6. Run the analysis daily as a batch job and keep every output versioned.

## 9. Limitations

- **No real data was analysed.** All numbers in the demo are synthetic and say nothing about Josh Talks workers.
- Thresholds (0.5×, 90%, 15 chars/s, 30% shares, z ≥ 3.5, n ≥ 10) are hypotheses.
- All three signals are behavioural proxies built on the same timing field. None measures transcript accuracy directly.
- Audio difficulty, speaker overlap and noise are not observed in the given schema.
- Text similarity is character-level. It does not weigh semantically important edits more heavily than punctuation.

## 10. Validation plan and next steps

1. **Run on the real export** (`python analysis.py --input export.csv`). Inspect the distributions and the data-quality report, and verify whether the supplied `segment_character_per_second` matches the defined worker-time calculation or an audio-duration alternative.
2. **Recalibrate thresholds** from real distributions (e.g. bottom 5th percentile of completion ratio), stratified by segment length.
3. **Blind review study:** sample flagged *and* unflagged workers, have expert reviewers score transcripts against audio (WER/CER versus a reference), and measure precision, recall and false-positive rate per status.
4. **Gold-task pilot** on 2–5% of the queue.
5. **Iterate:** tune weights and thresholds against reviewer outcomes. Consider a supervised model only once enough labelled outcomes exist.
6. **Governance:** monitor flag-rate parity across worker segments, and run the appeals and reassessment process from the start.

### Appendix: synthetic demonstration [Synthetic]

`python analysis.py --synthetic-demo` builds 10 fictional workers (IDs prefixed `SYN_`) plus deliberately malformed rows: zero and non-numeric durations, missing time and text, an unrecognised `is_edited` value, a flag/text contradiction and duplicates. The pipeline cleaned them as intended: 2 duplicates removed, invalid rows excluded from ratios, 1 flag/text contradiction reported. The constructed profiles were classified as designed:

| Synthetic profile (constructed) | Status | Why |
|---|---|---|
| `SYN_fast_rubberstamp_07` (fast + unedited) | priority_review | A + B + C |
| `SYN_fast_sparse_audio_08` (fast on mostly silent audio) | review_recommended | A only; C correctly *not* triggered. This is the case where human review should clear the worker |
| `SYN_high_throughput_09` (dense speech, 0.55–0.9×) | review_recommended | C only |
| `SYN_accurate_whisper_06` (97% unedited, normal timing) | normal | B **not** flagged without corroboration |
| `SYN_new_worker_10` (4 tasks) | insufficient_data | below minimum sample |
| `SYN_normal_01…05` | normal | no repeated pattern |

This shows only that **the logic behaves as specified on cases built to test it**. It is not evidence about detection accuracy on real workers.
