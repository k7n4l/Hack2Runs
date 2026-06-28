"""
Feature extraction: raw candidate JSON -> flat dict of scalar features.

FROZEN at 10 scored signals (see README "Scoring formula"). Every feature
here is traceable to a specific candidate_schema.json field - nothing is
invented. Text-based features use simple keyword presence checks, not
embeddings/ML, keeping the system explainable and within the no-heavy-ML
constraint.

A few additional fields (current_company, current_industry, core_skill_names,
last_active_date, raw_candidate) are passed through for use by
disqualifiers.py / honeypot.py / reasoning.py. They are NOT scored inputs -
they support gating and explanation text only.
"""

from datetime import date

from skill_taxonomy import categorize_skills

# Title tiers, hand-classified from the dataset's closed 47-title vocabulary.
TITLE_TIER_DIRECT = {
    "AI Engineer", "Senior AI Engineer", "Lead AI Engineer",
    "ML Engineer", "Machine Learning Engineer", "Senior Machine Learning Engineer",
    "Staff Machine Learning Engineer", "Applied ML Engineer", "Junior ML Engineer",
    "NLP Engineer", "Senior NLP Engineer", "AI Research Engineer", "AI Specialist",
    "Data Scientist", "Senior Data Scientist", "Senior Applied Scientist",
    "Senior Software Engineer (ML)", "Recommendation Systems Engineer", "Search Engineer",
    "Computer Vision Engineer",
}
TITLE_TIER_ADJACENT = {
    "Senior Software Engineer", "Software Engineer", "Backend Engineer",
    "Data Engineer", "Senior Data Engineer", "Analytics Engineer", "Data Analyst",
    "Full Stack Developer",
}
TITLE_TIER_GENERIC_SWE = {
    "Cloud Engineer", "DevOps Engineer", "Java Developer", ".NET Developer",
    "Mobile Developer", "Frontend Engineer", "QA Engineer",
}
# Everything else (Business Analyst, HR Manager, Mechanical Engineer, Accountant,
# Project Manager, Customer Support, Operations Manager, Content Writer,
# Sales Executive, Civil Engineer, Graphic Designer, Marketing Manager) is
# "unrelated" by default - the JD's explicit keyword-stuffing trap.

TITLE_TIER_SCORE = {"direct": 1.0, "adjacent": 0.6, "generic_swe": 0.25, "unrelated": 0.05}

TIER1_INDIAN_CITIES = {
    "bangalore", "bengaluru", "pune", "noida", "mumbai", "delhi", "gurgaon",
    "gurugram", "hyderabad",
}
LOCATION_FIT_SCORE = {"primary": 1.0, "tier1": 0.7, "other_india": 0.4, "international": 0.2}

PRODUCTION_LANGUAGE_TERMS = (
    "production", "deployed", "real users", "scale", "shipped", "live system",
    "serving", "revenue", "latency", "throughput", "a/b test", "ab test",
    "rollout", "users", "customers",
)
EVAL_FRAMEWORK_TERMS = (
    "ndcg", "mrr", "map@", "mean average precision", "offline-to-online",
    "offline to online", "a/b test", "ab test", "evaluation framework",
    "click-through", "relevance labeling", "human judgments",
)

JD_EXPERIENCE_BAND = (5.0, 9.0)  # years; JD's stated sweet spot


def _title_tier(title: str) -> str:
    if title in TITLE_TIER_DIRECT:
        return "direct"
    if title in TITLE_TIER_ADJACENT:
        return "adjacent"
    if title in TITLE_TIER_GENERIC_SWE:
        return "generic_swe"
    return "unrelated"


def _all_career_text(candidate: dict) -> str:
    parts = [candidate.get("profile", {}).get("summary", "")]
    for ch in candidate.get("career_history", []):
        parts.append(ch.get("description", ""))
    return " ".join(parts).lower()


def _location_fit(profile: dict) -> str:
    country = (profile.get("country") or "").lower()
    location = (profile.get("location") or "").lower()
    if country != "india":
        return "international"
    if any(city in location for city in ("pune", "noida")):
        return "primary"
    if any(city in location for city in TIER1_INDIAN_CITIES):
        return "tier1"
    return "other_india"


def _experience_band_fit(yoe: float) -> float:
    """1.0 inside the JD's 5-9yr band, smooth linear falloff outside it."""
    lo, hi = JD_EXPERIENCE_BAND
    if lo <= yoe <= hi:
        return 1.0
    if yoe < lo:
        return max(0.0, 1.0 - (lo - yoe) / lo)  # falls to 0 at yoe=0
    return max(0.0, 1.0 - (yoe - hi) / 15.0)  # gentle falloff for very senior folks


def _skill_credibility(skills: list, skill_assessment_scores: dict) -> float:
    """
    [0,1] multiplier comparing self-reported proficiency against Redrob's
    platform-verified skill_assessment_scores. Primary defense against
    keyword-stuffed skill lists. Neutral (1.0) when no assessment data exists
    - never penalize for data we can't observe.
    """
    if not skill_assessment_scores:
        return 1.0

    skills_by_name = {s.get("name"): s for s in skills}
    overclaim_gaps = []
    n_checked = 0
    expected_floor = {"beginner": 0, "intermediate": 35, "advanced": 50, "expert": 65}
    for skill_name, assessed in skill_assessment_scores.items():
        sk = skills_by_name.get(skill_name)
        if not sk:
            continue
        n_checked += 1
        prof = sk.get("proficiency", "beginner")
        # .get() with a safe default: schema only allows the 4 keys above, but a
        # single malformed/unexpected proficiency string must never crash a
        # 100K-row run. Default to the most lenient floor (0) so an unrecognized
        # value can't be penalized for something we can't actually verify.
        floor = expected_floor.get(prof, 0)
        gap = assessed - floor
        if gap < 0:
            overclaim_gaps.append(gap)

    if n_checked == 0 or not overclaim_gaps:
        return 1.0

    frac_overclaimed = len(overclaim_gaps) / n_checked
    avg_overclaim_gap = sum(overclaim_gaps) / len(overclaim_gaps)  # negative
    severity = max(0.0, min(1.0, -avg_overclaim_gap / 60.0))
    multiplier = 1.0 - (severity * frac_overclaimed)
    return max(0.35, multiplier)


def _availability_score(recruiter_response_rate: float, last_active_str: str, reference_date: date) -> float:
    """Blend of responsiveness and recency, JD's explicit 'actually available' signal."""
    recency_score = 1.0
    if last_active_str:
        try:
            y, m, d = (int(x) for x in last_active_str.split("-"))
            days_inactive = (reference_date - date(y, m, d)).days
            if days_inactive >= 180:
                recency_score = 0.3
            elif days_inactive >= 90:
                recency_score = 0.6
            elif days_inactive >= 30:
                recency_score = 0.85
        except (ValueError, AttributeError):
            pass
    return 0.6 * (recruiter_response_rate or 0.0) + 0.4 * recency_score


def extract_features(candidate: dict, reference_date: date) -> dict:
    profile = candidate.get("profile", {})
    skills = candidate.get("skills", [])
    rs = candidate.get("redrob_signals", {})

    skill_cats = categorize_skills(skills)
    career_text = _all_career_text(candidate)
    title = profile.get("current_title", "")
    title_tier = _title_tier(title)
    yoe_raw = profile.get("years_of_experience", 0)
    try:
        yoe = float(yoe_raw) if yoe_raw is not None else 0.0
    except (TypeError, ValueError):
        # Schema guarantees a number, but a corrupted row could carry a string
        # or other junk. Fall back to 0 rather than crash a 100K-row run on
        # one bad value - 0 is the most neutral assumption (lowest experience
        # band fit, never silently inflates a candidate's score).
        yoe = 0.0
    has_production_language = any(t in career_text for t in PRODUCTION_LANGUAGE_TERMS)
    has_eval_framework_language = any(t in career_text for t in EVAL_FRAMEWORK_TERMS)
    location_fit = _location_fit(profile)
    credibility = _skill_credibility(skills, rs.get("skill_assessment_scores", {}))
    availability = _availability_score(rs.get("recruiter_response_rate", 0), rs.get("last_active_date"), reference_date)

    return {
        "candidate_id": candidate["candidate_id"],
        # --- 10 frozen scored signals ---
        "title_tier_score": TITLE_TIER_SCORE[title_tier],
        "core_ml_infra_raw": skill_cats["core_ml_infra_raw"],          # percentile-normalized in scorer.py
        "experience_band_fit": _experience_band_fit(yoe),
        "production_eval_signal": 1.0 if (has_production_language and has_eval_framework_language) else (
            0.5 if (has_production_language or has_eval_framework_language) else 0.0
        ),
        "availability_score": availability,
        "location_fit_score": LOCATION_FIT_SCORE[location_fit],
        "skill_credibility": credibility,                             # multiplicative
        "vision_speech_only": skill_cats["vision_speech_only"],       # multiplicative penalty trigger
        # disqualifier_multiplier and honeypot_flag are computed in scorer.py
        # via disqualifiers.py / honeypot.py, using the raw candidate below.
        # --- pass-through context for gating + reasoning text (not scored) ---
        "title": title,
        "title_tier": title_tier,
        "years_of_experience": yoe,
        "current_company": profile.get("current_company", ""),
        "current_industry": profile.get("current_industry", ""),
        "core_skill_names": skill_cats["core_skill_names"],
        "recruiter_response_rate": rs.get("recruiter_response_rate", 0) or 0,
        "last_active_date": rs.get("last_active_date"),
        "raw_candidate": candidate,
    }
