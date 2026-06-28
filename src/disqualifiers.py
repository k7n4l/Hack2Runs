"""
JD-specific disqualifiers, encoded directly from job_description.docx.

The JD is unusually explicit about who to filter out. We translate each
stated rule into a checkable signal against the candidate_schema.json fields.
Disqualifiers are NOT hard exclusions (the JD says "we will probably not move
forward," not "auto-reject" for most of these) - they're penalty multipliers,
because:
  (a) the JD itself hedges most of these with "probably" / case-by-case,
  (b) a hard exclusion can't be explained gracefully in the reasoning column,
  (c) we'd rather a borderline candidate land at rank 85 with an honest
      explanation than vanish silently and look like a bug at Stage 5.

Encoded rules:

1. CONSULTING_ONLY ("People who have only worked at consulting firms... in
   their entire career"): ALL career_history entries are at a consulting firm
   AND current_company is also a consulting firm. We use current_industry ==
   "IT Services" / "Consulting" plus a named-company list as the consulting
   signal (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, HCL, Tech
   Mahindra, Mindtree, Mphasis are present in this dataset and read as the
   Indian-IT-services / SI cohort the JD is naming).
   -> Penalty, NOT exclusion: "If you're currently at one of these companies
      but have prior product-company experience, that's fine" - so a
      candidate with ANY non-consulting employer in their history is exempt.

2. TITLE_CHASER ("optimizing for Senior -> Staff -> Principal titles by
   switching companies every 1.5 years"): 3+ roles in career_history, all with
   duration_months < 18, where the title also escalates in
   seniority-sounding words (Senior/Lead/Staff/Principal/Head). We require
   BOTH the short-tenure pattern AND title escalation, since short tenure
   alone is common and not itself disqualifying per the JD's own logic
   (it's about chasing titles specifically).

3. STALE_ON_PLATFORM ("a perfect-on-paper candidate who hasn't logged in for
   6 months... is, for hiring purposes, not actually available"): derived
   from last_active_date relative to the dataset's "today" (max last_active
   date observed across the corpus, used as a stable proxy for "now" so the
   system doesn't silently change behavior if re-run later with a stale
   real-world clock).

4. VISION_SPEECH_ONLY ("primary expertise is computer vision, speech, or
   robotics... without significant NLP/IR exposure"): from
   skill_taxonomy.categorize_skills() - candidate has vision/speech skills
   but zero NLP/IR skill signal.

5. ARCHITECT_NOT_CODING ("hasn't written production code in the last 18
   months because you've moved into architecture/tech lead roles"): current
   title contains seniority/management words (Lead/Manager/Head/Director/
   Architect/VP/Principal) AND years_of_experience is high, suggesting a
   pure people-management trajectory. This is a soft signal only - title
   text alone can't prove "hasn't written code," so it's a small penalty,
   not a hard rule.

None of these apply to candidates whose current_title doesn't even look like
an engineering/ML role in the first place - that's a separate, much larger
filter handled in scorer.py (role-fit), since the JD's central trap is
exactly the keyword-stuffed non-engineer profile.
"""

CONSULTING_FIRMS = {
    "TCS", "Infosys", "Wipro", "Accenture", "Cognizant", "Capgemini",
    "HCL", "Tech Mahindra", "Mindtree", "Mphasis",
}

SENIORITY_WORDS = ("senior", "staff", "principal", "lead", "head", "director", "vp", "architect", "manager")


def check_consulting_only(candidate: dict) -> float:
    """Returns a penalty multiplier in [0,1]. 1.0 = no penalty."""
    profile = candidate.get("profile") or {}
    career_history = candidate.get("career_history") or []
    if not isinstance(career_history, list):
        career_history = []

    companies = [profile.get("current_company", "")] if isinstance(profile, dict) else [""]
    companies += [
        ch.get("company", "") for ch in career_history if isinstance(ch, dict)
    ]
    companies = [c for c in companies if c]  # drop empty/missing entries

    if not companies:
        # No employer data observed at all - never penalize for data we can't
        # see (same principle as everywhere else in this pipeline). Without
        # this guard, an empty list made `all(...)` vacuously True and wrongly
        # flagged candidates with zero career history as "consulting-only".
        return 1.0

    all_consulting = all(c in CONSULTING_FIRMS for c in companies)
    if all_consulting:
        return 0.55  # meaningful penalty, not exclusion - JD explicitly allows exceptions
    return 1.0


def check_title_chaser(candidate: dict) -> float:
    """3+ short stints (<18mo) with escalating seniority-sounding titles."""
    career_history = candidate.get("career_history") or []
    if not isinstance(career_history, list):
        return 1.0
    career_history = [ch for ch in career_history if isinstance(ch, dict)]
    if len(career_history) < 3:
        return 1.0

    dm_values = [ch.get("duration_months") for ch in career_history]
    short_stints = [dm for dm in dm_values if isinstance(dm, (int, float)) and dm < 18]
    escalating_titles = sum(
        1 for ch in career_history
        if any(w in (ch.get("title") or "").lower() for w in SENIORITY_WORDS)
    )

    if len(short_stints) >= 3 and escalating_titles >= 2:
        return 0.7
    return 1.0


def check_stale_on_platform(candidate: dict, reference_date) -> float:
    """Penalize candidates inactive for 6+ months relative to the corpus's latest observed date."""
    from datetime import date

    rs = candidate.get("redrob_signals") or {}
    last_active_str = rs.get("last_active_date") if isinstance(rs, dict) else None
    if not last_active_str:
        return 1.0
    try:
        y, m, d = (int(x) for x in last_active_str.split("-"))
        last_active = date(y, m, d)
    except (ValueError, AttributeError, TypeError):
        return 1.0

    days_inactive = (reference_date - last_active).days
    if days_inactive >= 180:
        return 0.5
    elif days_inactive >= 90:
        return 0.8
    return 1.0


def check_architect_not_coding(candidate: dict) -> float:
    """Soft penalty for senior-management titles where hands-on coding signal is weak."""
    profile = candidate.get("profile") or {}
    title = (profile.get("current_title") or "").lower()
    yoe_raw = profile.get("years_of_experience", 0)
    try:
        yoe = float(yoe_raw) if yoe_raw is not None else 0.0
    except (TypeError, ValueError):
        yoe = 0.0

    is_management_title = any(w in title for w in ("manager", "director", "vp", "head")) and "engineering manager" not in title
    if is_management_title and yoe >= 8:
        return 0.75
    return 1.0


def apply_disqualifier_penalties(candidate: dict, reference_date) -> dict:
    """
    Returns dict of individual penalty multipliers plus a combined multiplier,
    so scorer.py / reasoning.py can both apply it and explain which ones fired.
    """
    p_consulting = check_consulting_only(candidate)
    p_title_chaser = check_title_chaser(candidate)
    p_stale = check_stale_on_platform(candidate, reference_date)
    p_architect = check_architect_not_coding(candidate)

    combined = p_consulting * p_title_chaser * p_stale * p_architect

    return {
        "consulting_only": p_consulting,
        "title_chaser": p_title_chaser,
        "stale_on_platform": p_stale,
        "architect_not_coding": p_architect,
        "combined_multiplier": combined,
    }
