# Executive Summary: Flagging Transcription Reviewers for Quality Review

**Josh Talks AI Product Task, Question 2**

> **Evidence status:** the real worker dataset was not provided, so **no real workers were analysed or identified.** What follows is a tested, ready-to-run method. The demo uses clearly labelled synthetic data, and every threshold is a hypothesis to be validated.

### The problem
Human reviewers are paid to check and correct Whisper's Hindi transcripts. Some may click "accept" without really listening. That leaves AI errors in the training data, wastes review spend, and is hard to see, because these workers look highly productive.

### The proposed solution
A transparent, rules-based screening tool that uses data the platform already records. It surfaces workers whose *repeated* behaviour justifies a closer look, then sends samples of their work to **human quality reviewers**. It is a triage tool, not a judge.

### The three warning signals
| | Signal | In plain words | Starting threshold (to be validated) |
|---|---|---|---|
| **A** | **Too fast** | Time spent compared with audio length | Under half the audio length on ≥30% of tasks |
| **B** | **Never edits *and* rushes** | Accepts AI text unchanged, and does so quickly | ≥90% unedited **plus** ≥30% both fast and unedited |
| **C** | **Too much text per second** | Characters produced per second of effort | Over 15 chars/s on ≥30% of tasks, or far above peers |

A high "no edits" rate **on its own is never flagged**, because Whisper is often simply correct.

### How suspicious workers are reviewed
1. Workers need at least **10 valid tasks** before any status is assigned. Below that, the status is "insufficient data".
2. Each worker is labelled **Normal**, **Review recommended** (1 signal) or **Priority review** (2+ signals), with a written reason quoting the actual counts.
3. Flagged workers have 5–10 transcripts **blind-sampled** and scored against the audio by quality reviewers.
4. Any consequence requires **confirmed quality failures** and a human decision. There is **no automatic blocking**, and workers can appeal.
5. Reviewer outcomes feed back to recalibrate the thresholds.

### Expected business value
- Cleaner training data, because fewer uncorrected AI errors become "ground truth".
- Review budget spent on people who actually review.
- Quality-control effort focused on a small, explained shortlist instead of random audits.
- Fair treatment of fast-but-good workers, which protects workforce trust.

### Key limitations
- Not yet run on real data. The thresholds are educated guesses.
- All three signals rely on the same timing field and measure *behaviour*, not *accuracy*. Human review supplies the accuracy evidence.
- Playback speed, clip difficulty and partial timing logs can produce false alarms, which is why flags lead only to review.

### Next steps to validate
1. Run the tool on the real export and check the data-quality report.
2. Set thresholds from the real distributions.
3. Run a blind review of flagged **and** unflagged workers to measure precision and false-positive rates.
4. Add hidden "known-answer" tasks with planted errors as independent quality evidence.
5. Log active listening time and playback speed to sharpen Signal A.
