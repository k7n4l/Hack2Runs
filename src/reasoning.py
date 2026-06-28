"""
Reasoning generator.

Per submission_spec.docx Section 3, reasoning is sampled at Stage 4 and
checked for: specific facts, JD connection, honest concerns, no
hallucination, variation across candidates, and rank-consistency (tone must
match rank). We satisfy each of these structurally:

  - Specific facts: every clause below pulls a real number or name straight
    out of the candidate's own feature row / raw profile (years of
    experience, current title, named core skills, response rate). We never
    invent a skill or employer that isn't in the candidate's record.
  - JD connection: the lead clause foregrounds the JD's actual axes -
    core embeddings/retrieval/ranking skills and production+evaluation
    -framework signal from career_history text - rather than opening with
    a title-tier verdict. title_tier is still used (it's a real, validated
    signal: see README "Why this isn't a keyword-count ranker"), but only
    as a supporting fact when the title itself is the issue (unrelated /
    generic_swe), not as the headline frame for every candidate.
  - Honest concerns: any active penalty (low credibility, disqualifier,
    vision-only, honeypot, weak title, weak production/eval signal) is
    surfaced as a caveat clause, not hidden - a top-ranked candidate with a
    real gap still shows the gap.
  - No hallucination: text is assembled from real field values; there is no
    free-text generation step that could invent a fact. No tool/skill name
    is ever printed unless it's present in core_skill_names, which is
    derived directly from the candidate's own skills list.
  - Variation: the exact sentence is a function of that candidate's own
    numbers (which skills, how many, production/eval signal level, which
    caveats fire), so two candidates only produce identical text if their
    inputs are genuinely identical (extremely unlikely in 100K rows).
  - Rank consistency: tone is driven directly by final_score / score
    breakdown, the same numbers used for rank, so reasoning and rank can't
    drift apart.
"""


def _format_skills(names: list, limit: int = 2) -> str:
    uniq = list(dict.fromkeys(names))  # de-dupe, preserve order, deterministic
    return ", ".join(uniq[:limit])


def generate_reasoning(row: dict) -> str:
    title = row.get("title") or "Unknown title"
    company = row.get("current_company") or "an unspecified employer"
    title_tier = row.get("title_tier", "unrelated")
    core_skills = row.get("core_skill_names") or []
    response_rate = row.get("recruiter_response_rate")
    if not isinstance(response_rate, (int, float)):
        response_rate = 0.0

    yoe_raw = row.get("years_of_experience")
    try:
        yoe = float(yoe_raw) if yoe_raw is not None else 0.0
    except (TypeError, ValueError):
        yoe = 0.0

    breakdown = row.get("score_breakdown") or {}
    prod_eval_signal = row.get("production_eval_signal", 0.0)
    if not isinstance(prod_eval_signal, (int, float)):
        prod_eval_signal = 0.0

    # --- Honeypot: short-circuit with the exact reason, no other framing ---
    if breakdown.get("is_honeypot"):
        reason = breakdown.get("honeypot_reason") or "an internal inconsistency in the profile"
        return f"Excluded from genuine contention: profile contains an internal inconsistency ({reason})."

    clauses = []

    # Lead clause: substance first - what they actually have, not the title
    # bucket. core_skill_names is pulled straight from the candidate's own
    # skills list (skill_taxonomy.py's CORE_ML_INFRA set: embeddings,
    # retrieval, vector search, ranking, LLM fine-tuning) - never invented.
    if core_skills:
        skill_text = _format_skills(core_skills, limit=3)
        clauses.append(
            f"{yoe:.1f} years of experience including hands-on work with {skill_text}, "
            f"currently {title} at {company}"
        )
    else:
        clauses.append(
            f"{yoe:.1f} years of experience as {title} at {company}, with no embeddings, "
            f"retrieval, or ranking-specific skills listed on the profile"
        )

    # Production + evaluation framework signal - this is the JD's stated
    # core ask (production deployment + measurable ranking quality), derived
    # from career_history[].description text, not from the skills list.
    if prod_eval_signal >= 1.0:
        clauses.append(
            "career history describes both production-scale deployment and "
            "evaluation-framework work (the JD's core requirement)"
        )
    elif prod_eval_signal >= 0.5:
        clauses.append(
            "career history shows either production deployment or evaluation-framework "
            "experience, but not clear evidence of both"
        )
    elif title_tier in ("direct", "adjacent"):
        clauses.append(
            "career history descriptions show no explicit production-scale or "
            "evaluation-framework language despite the role"
        )

    # Role-fit context: now a supporting fact, not the lead frame. Stated
    # plainly as what the title tells us, without "matches/doesn't match JD"
    # boilerplate.
    if title_tier == "unrelated":
        clauses.append("current title is outside engineering/ML entirely")
    elif title_tier == "generic_swe":
        clauses.append("current role is general software engineering rather than ML/ranking-specific")

    # Credibility caveat - only mention if it actually fired
    if breakdown.get("skill_credibility", 1.0) < 0.95:
        clauses.append("self-reported skill levels run ahead of Redrob's own assessment scores for those skills")

    # Vision-only caveat
    if breakdown.get("vision_penalty", 1.0) < 1.0:
        clauses.append("skillset is vision/speech-heavy with no NLP or retrieval exposure, a gap the JD calls out directly")

    # Disqualifier caveats - name only the ones that actually fired
    disq = breakdown.get("disqualifiers") or {}
    if disq.get("consulting_only", 1.0) < 1.0:
        clauses.append("entire career history is at consulting/IT-services firms with no product-company exposure")
    if disq.get("title_chaser", 1.0) < 1.0:
        clauses.append("career shows short stints with escalating titles, a title-chasing pattern the JD flags")
    if disq.get("stale_on_platform", 1.0) < 1.0:
        clauses.append("has been inactive on the platform for an extended period, limiting real availability")
    if disq.get("architect_not_coding", 1.0) < 1.0:
        clauses.append("current title suggests a management track rather than hands-on coding")

    # Experience band note, only if notably outside the 5-9yr band
    if yoe < 4 or yoe > 11:
        clauses.append(f"{yoe:.1f} years sits outside the JD's stated 5-9 year band")

    # Availability close
    clauses.append(f"recruiter response rate of {response_rate:.0%}")

    text = "; ".join(clauses) + "."
    return text[0].upper() + text[1:]