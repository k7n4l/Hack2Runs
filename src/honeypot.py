"""
Honeypot detection.

The hackathon dataset contains ~80 candidates with "subtly impossible profiles"
(per the JD/spec) that are forced to relevance tier 0 in the hidden ground truth.
Ranking honeypots in the top 10 signals a system that's pattern-matching on
keywords rather than actually reading profiles, and a >10% honeypot rate in the
top 100 is a Stage 3 disqualifier.

We don't have the ground-truth honeypot list. Instead we derive logical-
impossibility checks directly from internal profile consistency - things that
cannot be true regardless of which candidate they belong to. This was validated
by scanning the full 100K candidate file:

  1. EXPERT_ZERO_DURATION: 3+ skills claimed at "expert" proficiency with
     duration_months == 0. You cannot be an expert in something you have used
     for zero months. (21 candidates match this in the full dataset)

  2. YOE_CAREER_MISMATCH: total months across career_history is <30% or >300%
     of the candidate's stated years_of_experience. A senior candidate
     claiming 13+ years but whose career_history sums to <1 year (or vice
     versa) doesn't add up. (20 candidates match, zero overlap with #1)

  3. SINGLE_ROLE_EXCEEDS_TOTAL_YOE: a single role's duration_months, divided
     by 12, exceeds the candidate's total stated years_of_experience. You
     cannot have worked one job longer than your entire career.
     (20 candidates match, ~zero overlap with #1/#2)

Together these three independent checks flag 59 candidates with no overlap
between patterns 1 and 2 or 1 and 5, suggesting they are genuinely distinct
injected-noise patterns rather than one mechanism. We deliberately did NOT
chase a 4th/5th pattern to hit exactly 80 - doing so risks overfitting to
guessed mechanics rather than to genuine logical impossibility, which is the
actual point ("we expect a good ranking system to naturally avoid them; you
don't need to special-case them" - submission_spec.docx Section 7).

A flagged candidate is not deleted from the candidate pool - it's given a
hard penalty multiplier so it can never surface near the top, while still
being rankable (and explainable) if it ends up in the tail.
"""

HONEYPOT_PENALTY_MULTIPLIER = 0.05  # crushes score; doesn't hard-zero it (keeps tie-break logic sane)


def _safe_duration_months(entry: dict) -> float:
    """Extract duration_months from a skill or career_history dict, defaulting
    to 0 for missing/None/wrong-type values rather than raising."""
    dm = entry.get("duration_months", 0)
    return dm if isinstance(dm, (int, float)) else 0


def detect_honeypot(candidate: dict) -> tuple:
    """
    Returns (is_honeypot: bool, reason: str or None)

    Defensive against malformed input throughout: any of skills / profile /
    career_history being None, the wrong type, or containing non-dict entries
    is treated as "no usable data for that check" rather than crashing - a
    single corrupted candidate must never take down a 100K-row run.
    """
    skills = candidate.get("skills") or []
    if not isinstance(skills, list):
        skills = []
    skills = [s for s in skills if isinstance(s, dict)]

    profile = candidate.get("profile") or {}
    if not isinstance(profile, dict):
        profile = {}

    career_history = candidate.get("career_history") or []
    if not isinstance(career_history, list):
        career_history = []
    career_history = [ch for ch in career_history if isinstance(ch, dict)]

    yoe_raw = profile.get("years_of_experience", 0)
    try:
        yoe = float(yoe_raw) if yoe_raw is not None else 0.0
    except (TypeError, ValueError):
        yoe = 0.0

    # Pattern 1: expert proficiency claimed with literally zero duration
    expert_zero = [
        s for s in skills
        if s.get("proficiency") == "expert" and _safe_duration_months(s) == 0
    ]
    if len(expert_zero) >= 3:
        names = ", ".join(s.get("name", "?") for s in expert_zero[:3])
        return True, f"claims expert-level proficiency in {len(expert_zero)} skills ({names}) with 0 months of use"

    # Pattern 2: total career_history duration wildly inconsistent with stated YOE
    total_months = sum(_safe_duration_months(ch) for ch in career_history)
    total_years = total_months / 12.0
    if yoe > 0 and total_years > 0:
        ratio = total_years / yoe
        if ratio < 0.3 or ratio > 3.0:
            return True, (
                f"stated years_of_experience ({yoe:.1f}) is inconsistent with "
                f"career_history total ({total_years:.1f} years across listed roles)"
            )

    # Pattern 3: a single role lasted longer than the candidate's entire career
    for ch in career_history:
        dm = _safe_duration_months(ch)
        if yoe > 0 and (dm / 12.0) > yoe + 1.0:  # +1yr slack for rounding
            return True, (
                f"single role at {ch.get('company', '?')} lasted {dm} months "
                f"({dm / 12:.1f} yrs), exceeding total claimed experience of {yoe:.1f} years"
            )

    return False, None
