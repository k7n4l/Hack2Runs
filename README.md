# Redrob Hackathon — Candidate Ranking System

A deterministic, explainable, rule-based ranker for the **Intelligent Candidate
Discovery & Ranking Challenge**, targeting the Redrob *Senior AI Engineer —
Founding Team* job description.

## Quick start

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

No installation step required — the project uses only the Python standard
library (see `requirements.txt`). Runs in **~15 seconds** on the full
100,000-candidate file (measured across repeated runs via the exact CLI
command above), peak memory **~1.7 GB**, well inside the spec's 5-minute /
16 GB / CPU-only / no-network budget.

### Sandbox / small-sample mode

```bash
python rank.py --candidates ./small_sample.jsonl --out ./demo_output.csv --top-n 50
```

`--top-n` controls how many ranked rows to produce and validate against. The
real competition submission always uses the default (100); a smaller value
is for the hosted sandbox demo, which only needs to prove the pipeline runs
end-to-end on a small sample (per `submission_spec.docx` Section 10.5).

## Why this isn't a keyword-count ranker

`job_description.docx` is explicit that the "right answer" is **not** "find
candidates whose skills section contains the most AI keywords" — that's a
trap deliberately built into the dataset. We confirmed this directly: of the
100,000 candidates, **1,737 have non-ML job titles (Backend Engineer, Data
Analyst, Cloud Engineer, etc.) while listing 3+ core ML-infrastructure
skills** (LangChain, FAISS, LoRA, etc.) with no career-history evidence of
ever using them. A naive "count AI skills" scorer — which is what
`sample_submission.csv` and the generic "master prompt" both do — ranks
these candidates at the top. This system doesn't.

Two corpus-wide checks anchor every score in something harder to fake than a
skills list:

1. **`skill_assessment_scores` cross-check.** Redrob's own platform
   assessment scores are compared against each candidate's *self-reported*
   proficiency for the same skill. Across the ~24,000 candidates with
   assessment data, **56.3% self-report "advanced"/"expert" on at least one
   skill while scoring below 50 on the platform's own test for it** (43.9%
   of all individually-checked skill instances). That gap is the system's
   main defense against stuffed skill lists, and it can't be gamed by adding
   more keywords.
2. **Career-history text.** Production/scale language and evaluation
   -framework language (NDCG, MRR, click-through, A/B testing, etc.) are
   checked directly in `career_history[].description`, not in the skills
   list — so the signal tracks what someone actually did, not what they
   listed.

## Architecture

```
candidate-ranking-system/
├── rank.py                  # single CLI entrypoint
├── requirements.txt         # intentionally empty — stdlib only
├── submission_metadata.yaml
├── README.md                 (this file)
├── data/
│   ├── sample_candidates.jsonl  # 50-row sample for the sandbox/demo mode above
│   └── submission.csv           # full real run's output, regenerated from current code
└── src/
    ├── loader.py             # streaming JSONL reader, skips corrupted lines
    ├── skill_taxonomy.py     # hand-classified 133-skill closed vocabulary
    ├── feature_extractor.py  # raw candidate -> 10 frozen scored features
    ├── disqualifiers.py      # JD-specific penalty checks (consolidated)
    ├── honeypot.py            # 3 logical-impossibility checks
    ├── scorer.py              # percentile normalization + weighted formula
    ├── reasoning.py           # fact-grounded 1-2 sentence explanations
    └── exporter.py            # CSV writer + spec self-validation
```

Note: the full `candidates.jsonl` (~465 MB, organizer-provided) is intentionally
**not** vendored into this repo — only the small `data/sample_candidates.jsonl`
used for sandbox-mode demos. Point `--candidates` at wherever you keep the real
file when running the full pipeline.

## Pipeline

1. **`loader.py`** streams `candidates.jsonl` line-by-line. Malformed JSON,
   non-object lines, and lines missing `candidate_id` are skipped and
   counted, never crash the run.
2. **`feature_extractor.py`** turns each raw candidate into 10 scored
   signals (below) plus a few pass-through fields used only for gating/
   explanation text (current_company, core_skill_names, etc. — not scored).
3. **`scorer.py`** percentile-normalizes the one feature that needs the
   whole population (`core_ml_infra_raw`), applies the frozen weighted
   formula, multiplies in disqualifier and honeypot penalties, then sorts by
   `(final_score desc, candidate_id asc)` — matching the tie-break rule in
   `validate_submission.py` exactly.
4. **`reasoning.py`** builds each candidate's explanation directly from
   their own score breakdown and raw profile fields — no free-text
   generation, so nothing can be hallucinated.
5. **`exporter.py`** writes the CSV and re-implements `validate_submission.py`'s
   checks inline, so a broken submission is caught locally, immediately,
   not at Stage 1 of the real competition.

## The 10 frozen scored signals

| Signal | Weight / role | Source |
|---|---|---|
| `title_tier_score` | 0.35 | `current_title` mapped to {direct / adjacent / generic_swe / unrelated} against the JD's actual ask |
| `core_ml_infra_pctile` | 0.25 | percentile rank of proficiency-weighted core ML-infra skills (embeddings, retrieval, vector DBs, ranking, LLMs — the JD's "absolutely need" list) |
| `experience_band_fit` | 0.15 | distance from the JD's stated 5–9 year sweet spot, smooth falloff outside |
| `production_eval_signal` | 0.10 | career-history text for production/scale language AND evaluation-framework language |
| `availability_score` | 0.10 | recruiter response rate + platform recency (JD: "not actually available" if inactive) |
| `location_fit_score` | 0.05 | Pune/Noida primary, Tier-1 India secondary, elsewhere lower |
| `skill_credibility` | multiplicative | self-reported proficiency vs. `skill_assessment_scores` |
| `vision_speech_only` | multiplicative (×0.6 if true) | JD's explicit down-weight for CV/speech/robotics without NLP/IR |
| disqualifier multiplier | multiplicative | consolidated in `disqualifiers.py` (see below) |
| honeypot flag | multiplicative (×0.05 if flagged) | consolidated in `honeypot.py` (see below) |

**Formula:**
```
base_fit = 0.35·title_tier_score + 0.25·core_ml_infra_pctile + 0.15·experience_band_fit
         + 0.10·production_eval_signal + 0.10·availability_score + 0.05·location_fit_score

final_score = base_fit
            × skill_credibility
            × (0.6 if vision_speech_only else 1.0)
            × disqualifier_combined_multiplier
            × (0.05 if honeypot else 1.0)
```

All constants are fixed; nothing is fit or trained. Same input → same output,
always.

## Disqualifiers (`disqualifiers.py`, consolidated)

Encoded directly from the JD's "explicitly do NOT want" / "will not move
forward" section:

| Check | JD basis | Penalty |
|---|---|---|
| `consulting_only` | "People who have only worked at consulting firms... entire career" | 0.55× if every employer is a named consulting firm |
| `title_chaser` | "optimizing for Senior → Staff → Principal by switching every 1.5 years" | 0.7× if 3+ short stints (<18mo) AND escalating seniority titles |
| `stale_on_platform` | "hasn't logged in for 6 months... not actually available" | 0.5–0.8× based on days since `last_active_date` |
| `architect_not_coding` | "moved into architecture/tech lead... hasn't written production code" | 0.75× for senior management titles at high YOE |

These are penalties, not hard exclusions — the JD itself hedges most of
these with "probably" and explicitly allows exceptions (e.g. "if you're
currently at one of these companies but have prior product-company
experience, that's fine").

## Honeypot detection (`honeypot.py`)

The spec states ~80 candidates have "subtly impossible profiles" forced to
relevance tier 0. We don't have the ground-truth list, so we derive
logical-impossibility checks directly from internal profile consistency —
things that cannot be true regardless of which candidate they belong to:

1. **Expert + zero duration**: 3+ skills claimed at "expert" with
   `duration_months == 0`.
2. **YOE/career-history mismatch**: total `career_history` duration is
   <30% or >300% of stated `years_of_experience`.
3. **Single role exceeds total YOE**: one role's duration alone is longer
   than the candidate's entire claimed career.

These three checks, run against the full dataset, flag **59 candidates with
zero overlap** between pattern 1 and pattern 2, and zero overlap between
pattern 1 and pattern 3 (2 candidates overlap between patterns 2 and 3),
suggesting genuinely distinct injected-noise patterns rather than one
mechanism double-counted. We deliberately did not chase additional patterns
to hit exactly 80 — that risks overfitting to guessed mechanics rather than
genuine logical impossibility, which is the actual point. **0 honeypots
appear in the final top-100 output.**

## Determinism & reproducibility

- No randomness anywhere (no `random`, no seeded sampling, no model
  inference).
- `core_ml_infra_pctile` uses average-rank percentiles, stable under ties
  regardless of input row order.
- The score used for sorting is rounded to the same 4 decimal places written
  to the CSV *before* sorting, so the candidate-id tie-break is applied
  consistently between what's computed and what's written (an earlier
  version of this pipeline had a bug here — full-precision floats were
  sorted but rounded values were written, occasionally producing
  out-of-order ties; caught by running the actual `validate_submission.py`
  against our own output, not just our internal reimplementation of it).
- A fixed `CORPUS_REFERENCE_DATE` (2026-05-27, the latest `last_active_date`
  observed in the released dataset) is used for all recency calculations
  instead of `datetime.now()`, so the same input always produces the same
  output regardless of when the script is run.

Verified: running the pipeline twice on the same input produces a
byte-identical `submission.csv`.

## Robustness against malformed input

Every module was stress-tested against `None` values, wrong types, missing
keys, and non-dict entries in nested lists — the realistic shapes "corrupted
JSON" actually takes once a line *does* parse as valid JSON but has a `null`
where a schema field expects a string, list, or number. Examples of bugs this
caught and fixed:

- `skill_taxonomy.categorize_skills(None)` and a `skills` list containing a
  stray non-dict entry both raised uncaught exceptions; now treated as "no
  usable skill data" rather than crashing the run.
- `disqualifiers.check_consulting_only` had a logic bug, not just a crash
  risk: a candidate with **zero** employer data was incorrectly flagged as
  "consulting-only" (an empty list makes Python's `all()` vacuously `True`).
  Fixed to return "no penalty" when there's no data to evaluate.
- `years_of_experience` as a non-numeric value (e.g. a stray string from a
  corrupted row) crashed comparisons in `feature_extractor.py`,
  `disqualifiers.py`, `honeypot.py`, and `reasoning.py` independently; all
  four now use the same safe-coercion pattern, falling back to a neutral
  default rather than raising.
- `exporter.self_validate` crashed with a raw `StopIteration` on a
  completely empty file, and was silently missing two checks
  `validate_submission.py` actually performs (the `.csv` extension
  requirement and the tie-break-on-equal-scores rule) — meaning it could
  report "valid" on a file the real grader would reject. Both gaps are
  closed.

None of these fixes changed the scoring formula, weights, thresholds, or any
value computed from well-formed input — confirmed by re-running the full
100K-candidate file before and after each fix and diffing the top-100 output.

## Validation

`rank.py` automatically re-validates its own output against the spec's rules
after writing the CSV, and exits non-zero if anything is wrong. We also ran
the organizers' own `validate_submission.py` against our output directly —
it passes cleanly.

## What we deliberately did not build

- **No sentence-transformers / semantic embeddings.** An earlier generic
  prompt for this task suggested `all-MiniLM-L6-v2` for "semantic skill
  matching." We didn't use it: the dataset's skill vocabulary is a small,
  closed set of 133 exact strings (verified by scanning the full file), so
  fuzzy/semantic matching adds complexity without adding signal, and would
  reintroduce exactly the "similar-sounding keywords look like a match"
  failure mode the JD warns against.
- **No ML model / training.** Every weight and threshold in the formula is a
  fixed constant chosen from reading the JD and validating against the data
  distribution, not fit by gradient descent or grid search. This keeps the
  system inside the "no deep learning training, explainable in 1-2 lines"
  constraint and makes it fully defensible in a Stage 5 interview.

## Known limitation, stated honestly

Within the 1,179 candidates with a direct JD-title match, the best possible
"adjacent title, hidden gem" candidate (e.g. a Data Engineer with strong
vector-search skills) still scores below the 100th-best direct-title match
(0.83 vs. 0.93). Given the JD explicitly wants someone who "owns" the
ranking/retrieval layer and writes production code under that title, and
there are more than enough direct-title matches to fill 100 slots, we
consider this the correct behavior rather than a bug — but it's a deliberate
weighting choice (title_tier_score carries the largest single weight, 0.35)
that's worth being able to defend explicitly if asked.
