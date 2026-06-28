"""
Streaming loader for candidates.jsonl.

Reads the file line-by-line (never loads the raw file into memory as a single
string/list), parses each line as JSON, and yields valid candidate dicts.
Corrupted lines are counted and skipped rather than crashing the pipeline -
real-world recruiting data is never perfectly clean, and a single malformed
line shouldn't take down a 100K-candidate ranking run.

Memory note: this DOES eventually accumulate one lightweight feature-row per
candidate in scorer.py (not the raw JSON) so the ranker can compute percentile
-based normalization, which requires seeing the whole distribution. That's a
deliberate, documented exception to "streaming only" - it's ~100K small dicts
of scalars, on the order of tens of MB, nowhere near the 16GB ceiling. We do
NOT hold 100K raw nested JSON blobs in memory at once.
"""

import json
import os


def stream_candidates(path: str):
    """
    Generator. Yields (line_number, candidate_dict) for every line that:
      - parses as valid JSON
      - is a JSON object (dict), not a list/string/number/null
      - has a non-empty string candidate_id

    Skips and counts (does not raise on) lines that fail any of the above -
    a single bad line must never take down a 100K-row run.

    Duplicate candidate_id values are detected: the first occurrence is
    yielded, later occurrences are skipped and counted. This matters for
    determinism - if the same id appeared twice, scoring it twice would
    silently double-weight that candidate's influence on percentile
    normalization in scorer.py, and the run's output could change just
    because of line order in a re-exported file.

    Raises FileNotFoundError / PermissionError with a clear message rather
    than letting a raw traceback surface, since this is the first thing
    `rank.py` calls and a clear error here saves debugging time in a
    timed hackathon run.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"[loader] candidates file not found: {path}")
    if not os.path.isfile(path):
        raise ValueError(f"[loader] path exists but is not a file: {path}")

    n_ok = 0
    n_bad = 0
    n_duplicate = 0
    seen_ids = set()

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    candidate = json.loads(line)
                except json.JSONDecodeError:
                    n_bad += 1
                    continue

                if not isinstance(candidate, dict):
                    n_bad += 1
                    continue

                cid = candidate.get("candidate_id")
                if not isinstance(cid, str) or not cid.strip():
                    n_bad += 1
                    continue

                if cid in seen_ids:
                    n_duplicate += 1
                    continue
                seen_ids.add(cid)

                n_ok += 1
                yield line_no, candidate
    except UnicodeDecodeError as e:
        raise ValueError(
            f"[loader] {path} is not valid UTF-8 (per spec, candidate files must be UTF-8): {e}"
        ) from e

    total = n_ok + n_bad + n_duplicate
    if n_bad or n_duplicate:
        print(
            f"[loader] WARNING: {n_ok} valid / {n_bad} corrupted-or-invalid / "
            f"{n_duplicate} duplicate candidate_id, out of {total} non-blank line(s)"
        )


def load_candidate_index(path: str) -> dict:
    """
    Build a lightweight candidate_id -> raw dict index for O(1) lookup by id
    (e.g. validating that submitted ids exist). Intended for small samples /
    validation only, NOT the main 100K scoring pass - that pass stays
    streaming via stream_candidates() and never materializes a full id index.
    """
    index = {}
    for _, c in stream_candidates(path):
        index[c["candidate_id"]] = c
    return index
