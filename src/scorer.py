"""
Scorer: combines extracted features into one deterministic, explainable score.

FROZEN FORMULA (see README):

    base_fit = 0.35 * title_tier_score
             + 0.25 * core_ml_infra_pctile
             + 0.15 * experience_band_fit
             + 0.10 * production_eval_signal
             + 0.10 * availability_score
             + 0.05 * location_fit_score

    final_score = base_fit
                * skill_credibility
                * (0.6 if vision_speech_only else 1.0)
                * disqualifier_combined_multiplier
                * (0.05 if honeypot else 1.0)

All weights, multipliers, and thresholds are fixed constants - no fitting, no
training, no randomness. Given the same candidates.jsonl, this always
produces the same scores in the same order.

`core_ml_infra_raw` is converted to a percentile rank (0-1) across the full
candidate pool before scoring. This is the one place we need the whole
distribution at once (see loader.py's memory note) - a percentile is
inherently a population-relative measure, robust to outliers, and avoids
having to hand-pick a min/max scale for a feature whose raw range we don't
control.
"""

from datetime import date

from disqualifiers import apply_disqualifier_penalties
from honeypot import detect_honeypot, HONEYPOT_PENALTY_MULTIPLIER

WEIGHTS = {
    "title_tier_score": 0.35,
    "core_ml_infra_pctile": 0.25,
    "experience_band_fit": 0.15,
    "production_eval_signal": 0.10,
    "availability_score": 0.10,
    "location_fit_score": 0.05,
}
VISION_ONLY_PENALTY = 0.6


def _percentile_ranks(values: list) -> list:
    """
    Deterministic percentile rank in [0,1] for each value, by position in the
    sorted order. Ties receive the same rank (average-rank method) so the
    mapping is stable regardless of input order - required for reproducibility.
    """
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [1.0]

    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0
        pct = avg_rank / (n - 1)
        for k in range(i, j + 1):
            ranks[order[k]] = pct
        i = j + 1
    return ranks


def score_all(feature_rows: list, reference_date: date) -> list:
    """
    Takes the list of per-candidate feature dicts (from feature_extractor.py),
    computes percentile ranks for core_ml_infra_raw across the full pool, then
    scores each candidate with the frozen formula. Returns the same list with
    `final_score` and `score_breakdown` added to each row, sorted by
    (final_score desc, candidate_id asc) - matching the tie-break rule in
    validate_submission.py.
    """
    raw_core_values = [r.get("core_ml_infra_raw", 0.0) for r in feature_rows]
    pctiles = _percentile_ranks(raw_core_values)

    for row, pctile in zip(feature_rows, pctiles):
        row["core_ml_infra_pctile"] = pctile

        base_fit = sum(WEIGHTS[k] * row.get(k, 0.0) for k in WEIGHTS)

        vision_penalty = VISION_ONLY_PENALTY if row.get("vision_speech_only") else 1.0

        raw_candidate = row.get("raw_candidate") or {}
        disq = apply_disqualifier_penalties(raw_candidate, reference_date)
        disq_multiplier = disq["combined_multiplier"]

        is_honeypot, honeypot_reason = detect_honeypot(raw_candidate)
        honeypot_multiplier = HONEYPOT_PENALTY_MULTIPLIER if is_honeypot else 1.0

        final_score = base_fit * row.get("skill_credibility", 1.0) * vision_penalty * disq_multiplier * honeypot_multiplier
        final_score = max(0.0, min(1.0, final_score))
        # Round to the same precision written to the CSV (4 decimal places)
        # BEFORE sorting. Sorting on full float precision while writing a
        # rounded value can silently create ties in the output that weren't
        # ties at sort time, breaking the candidate_id tie-break rule
        # required by submission_spec.docx. Round first, sort on the
        # rounded value, so what's written is exactly what was sorted.
        final_score = round(final_score, 4)

        row["final_score"] = final_score
        row["score_breakdown"] = {
            "base_fit": base_fit,
            "skill_credibility": row.get("skill_credibility", 1.0),
            "vision_penalty": vision_penalty,
            "disqualifiers": disq,
            "is_honeypot": is_honeypot,
            "honeypot_reason": honeypot_reason,
        }

    feature_rows.sort(key=lambda r: (-r["final_score"], r["candidate_id"]))
    return feature_rows
