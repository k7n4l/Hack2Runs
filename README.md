# Redrob Hackathon — Candidate Ranking System

A deterministic, explainable, rule-based candidate ranking system for the Redrob AI/ML hiring challenge.

## Quick Start

Run the ranking pipeline with the full candidate set:

```bash
python rank.py --candidates data/candidates.jsonl --out output/submission.csv
```

## Sandbox / Demo Mode

For a small local sample, use the bundled sample data:

```bash
python rank.py --candidates data/sample_candidates.jsonl --out output/demo.csv
```

This is intended for quick testing and small-sample validation only.

## Project Structure

```text
candidate-ranking-system/
├── rank.py
├── validate_submission.py
├── submission_metadata.yaml
├── README.md
├── data/
│   ├── (official dataset provided by organizers, not included)
│   └── sample_candidates.jsonl
├── output/
└── src/
    ├── disqualifiers.py
    ├── exporter.py
    ├── feature_extractor.py
    ├── honeypot.py
    ├── loader.py
    ├── reasoning.py
    ├── scorer.py
    └── skill_taxonomy.py
```

The official dataset is not included in this repository; only the sample dataset is provided here.

## Design Philosophy

This system avoids keyword-based ranking. It focuses on real ML engineering signals, profile evidence, and consistency checks rather than buzzwords or shallow skill lists.

## Core Idea

Candidates are ranked based on verifiable engineering evidence, not keyword frequency or self-reported skills.

## Pipeline

1. Loader: reads candidate records from JSONL input.
2. Feature extraction: converts raw profiles into structured evidence features.
3. Scoring: applies fixed weights and multipliers to compute a deterministic score.
4. Disqualifiers: applies job-description-specific penalties.
   Disqualifiers are rule-based penalties derived from job-description constraints.
5. Honeypots: flags logically inconsistent profiles.
6. Reasoning: generates concise, evidence-based explanations.
7. Exporter: writes the ranked CSV and validates the output.

## Scoring

Base score:

```text
base_fit =
0.35 * title_tier_score
+ 0.25 * core_ml_infra_pctile
+ 0.15 * experience_band_fit
+ 0.10 * production_eval_signal
+ 0.10 * availability_score
+ 0.05 * location_fit_score
```

Final score:

```text
final_score =
base_fit
* skill_credibility
* (0.6 if vision_or_speech_only else 1.0)
* disqualifier_multiplier
* honeypot_multiplier
```

## Determinism

- No randomness
- No ML models
- No external APIs
- Same input produces the same output

## Notes

- The official dataset is provided by the organizers and is not included here.
- Only the sample dataset is available in the repository.
- The pipeline is CPU-only and reproducible.

## Summary

- Explainable
- Deterministic
- Reproducible
- Lightweight
- Rule-based
=======
---
title: Redrob Ranker
emoji: 🌖
colorFrom: green
colorTo: indigo
sdk: gradio
sdk_version: 6.19.0
python_version: '3.13'
app_file: app.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
>>>>>>> 1526bdf59ca3801564b6ecdc459d7b5da1a66b15
