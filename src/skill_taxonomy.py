"""
Skill taxonomy for the Redrob Senior AI Engineer JD.

The candidate dataset uses a CLOSED vocabulary of ~133 skill names (verified by
scanning the full candidates.jsonl). Rather than fuzzy/semantic matching, we hand
-classify every skill name into a category. This is deliberate: the JD explicitly
warns against rewarding "AI keyword count," so we don't score on the COUNT of
AI-sounding skills. Instead we distinguish:

  CORE_ML_INFRA   - "things you absolutely need" per the JD: embeddings/retrieval,
                     vector DBs/hybrid search, ranking/eval frameworks. This is the
                     skill category that should carry real weight.
  APPLIED_ML      - general ML/DL skills (useful, but the JD says these are
                     "nice to have", not core - e.g. CV/speech/robotics without
                     NLP/IR is an explicit DOWN-weight per the JD).
  ML_ADJACENT_DATA- data engineering skills (Spark/Airflow/Kafka/dbt/etc). Useful
                     supporting signal, not a substitute for ranking/retrieval depth.
  GENERIC_SWE     - general software engineering (React, Java, Docker, etc). Needed
                     baseline but not differentiating for THIS JD.
  NON_TECHNICAL   - sales/marketing/accounting/ops skills. Presence of these
                     alongside an "AI Engineer" title is itself a minor flag (could
                     indicate a generalist or a stuffed skills list).

A skill appearing in a candidate's list contributes to whichever category it's in,
weighted by proficiency. Self-reported proficiency is cross-checked against
`skill_assessment_scores` (platform-verified) in scorer.py — that's the real
defense against keyword-stuffing, not this taxonomy alone.
"""

CORE_ML_INFRA = {
    # Embeddings / retrieval / vector search - JD's #1 "absolutely need"
    "Embeddings", "Sentence Transformers", "Vector Search", "Vector Representations",
    "Semantic Search", "FAISS", "Pinecone", "Qdrant", "Weaviate", "Milvus",
    "pgvector", "OpenSearch", "Elasticsearch", "Information Retrieval",
    "Information Retrieval Systems", "Indexing Algorithms", "Search Infrastructure",
    "Search Backend", "Search & Discovery", "Text Encoders", "BM25",
    "Haystack", "LangChain", "LlamaIndex", "RAG",
    # Ranking / eval - JD's #4 "absolutely need"
    "Recommendation Systems", "Ranking Systems", "Learning to Rank",
    # LLMs - directly named throughout JD
    "LLMs", "Fine-tuning LLMs", "Prompt Engineering", "Hugging Face Transformers",
    "LoRA", "QLoRA", "PEFT", "Model Adaptation",
}

APPLIED_ML = {
    "Machine Learning", "Deep Learning", "NLP", "Natural Language Processing",
    "Computer Vision", "Image Classification", "Object Detection", "YOLO", "OpenCV", "CNN",
    "Speech Recognition", "ASR", "TTS", "GANs", "Diffusion Models",
    "Reinforcement Learning", "Data Science", "Statistical Modeling",
    "Feature Engineering", "Time Series", "Forecasting", "scikit-learn",
    "TensorFlow", "PyTorch", "MLOps", "MLflow", "Weights & Biases", "Kubeflow",
    "BentoML", "Content Matching",
}

# CV / speech / robotics without NLP/IR exposure is an explicit JD down-weight.
# These overlap with APPLIED_ML but are tagged separately so scorer.py can detect
# "vision/speech-only, no NLP/IR" candidates.
VISION_SPEECH_ROBOTICS = {
    "Computer Vision", "Image Classification", "Object Detection", "YOLO", "OpenCV", "CNN",
    "Speech Recognition", "ASR", "TTS", "GANs", "Diffusion Models",
}
NLP_IR_SIGNAL = {
    "NLP", "Natural Language Processing", "Embeddings", "Sentence Transformers",
    "Vector Search", "Semantic Search", "Information Retrieval",
    "Information Retrieval Systems", "RAG", "LLMs", "Fine-tuning LLMs",
    "LangChain", "LlamaIndex", "Haystack", "BM25", "Recommendation Systems",
    "Ranking Systems", "Learning to Rank",
}

DATA_ENGINEERING = {
    "Data Pipelines", "Spark", "Airflow", "Apache Beam", "Apache Flink", "Kafka",
    "Hadoop", "dbt", "ETL", "Databricks", "Snowflake", "BigQuery", "Workflow Orchestration",
    "Document Processing", "Open-source ML libraries",
}

GENERIC_SWE = {
    "Python", "Java", "Go", "Rust", "JavaScript", "TypeScript", "React", "Vue.js",
    "Angular", "Redux", "Next.js", "Node.js", "HTML", "CSS", "Tailwind", "Webpack",
    "Django", "Flask", "FastAPI", "Spring Boot", "Microservices", "REST APIs", "gRPC",
    "GraphQL", "Docker", "Kubernetes", "Terraform", "CI/CD", "AWS", "GCP", "Azure",
    "SQL", "PostgreSQL", "MongoDB", "Redis", "SAP",
}

NON_TECHNICAL = {
    "Sales", "Marketing", "SEO", "Content Writing", "Accounting", "Tally", "Excel",
    "PowerPoint", "Figma", "Illustrator", "Photoshop", "Salesforce CRM", "Agile",
    "Scrum", "Project Management", "Six Sigma",
}

PROFICIENCY_WEIGHT = {
    "beginner": 0.25,
    "intermediate": 0.55,
    "advanced": 0.85,
    "expert": 1.0,
}


def categorize_skills(skills: list) -> dict:
    """
    Given a candidate's skills list (list of dicts with name/proficiency/
    endorsements), return aggregate category scores plus raw signal needed
    for the assessment-vs-self-report credibility check.

    Returns dict with:
      core_ml_infra_score, applied_ml_score, data_eng_score, generic_swe_score
      vision_speech_only: bool (has vision/speech/robotics skills but ZERO nlp/ir signal)
      core_skill_names: list of core ML infra skill names present (for reasoning text)
      n_core, n_applied

    Defensive against malformed input: a `skills` value of None (e.g. a JSON
    `null` instead of an omitted field) or a skills list containing non-dict
    entries (e.g. a stray string from a corrupted row) is treated as "no
    usable skill data" for that entry rather than crashing the whole run.
    """
    core_score = 0.0
    applied_score = 0.0
    data_eng_score = 0.0
    swe_score = 0.0
    core_names = []
    applied_names = []
    has_vision_speech = False
    has_nlp_ir = False

    for s in skills or []:
        if not isinstance(s, dict):
            continue
        name = s.get("name", "")
        prof = s.get("proficiency", "beginner")
        w = PROFICIENCY_WEIGHT.get(prof, 0.25)

        if name in CORE_ML_INFRA:
            core_score += w
            core_names.append(name)
        if name in APPLIED_ML:
            applied_score += w
            applied_names.append(name)
        if name in DATA_ENGINEERING:
            data_eng_score += w
        if name in GENERIC_SWE:
            swe_score += w
        if name in VISION_SPEECH_ROBOTICS:
            has_vision_speech = True
        if name in NLP_IR_SIGNAL:
            has_nlp_ir = True

    return {
        "core_ml_infra_raw": core_score,
        "applied_ml_raw": applied_score,
        "data_eng_raw": data_eng_score,
        "generic_swe_raw": swe_score,
        "vision_speech_only": has_vision_speech and not has_nlp_ir,
        "core_skill_names": core_names,
        "applied_skill_names": applied_names,
        "n_core": len(core_names),
        "n_applied": len(applied_names),
    }
