"""
Submission exporter.

Writes the top-100 ranking in exactly the format required by
submission_spec.docx Section 2:
  - header: candidate_id,rank,score,reasoning
  - exactly 100 data rows
  - rank 1..100, each exactly once
  - score non-increasing with rank
  - tie-break: candidate_id ascending among equal scores
  - UTF-8 encoding for CSV output
  - supports .csv and .xlsx output

After writing, runs validate_submission.py's own validation logic against the
output file so a broken submission is caught immediately, locally, before
upload - not discovered at Stage 1 of the competition. self_validate()
mirrors validate_submission.py's checks line-for-line (including the
tie-break and filename-extension checks), so a file that passes here is
expected to also pass the organizers' actual validator.
"""

import csv
import re
from pathlib import Path

import pandas as pd

from reasoning import generate_reasoning

REQUIRED_HEADER = ["candidate_id", "rank", "score", "reasoning"]
CANDIDATE_ID_PATTERN = re.compile(r"^CAND_[0-9]{7}$")


def export_submission(ranked_rows: list, out_path: str, top_n: int = 100):
    """
    ranked_rows must already be sorted by (final_score desc, candidate_id asc)
    - see scorer.score_all(). Writes only the top `top_n` rows.
    """
    top = ranked_rows[:top_n]
    if len(top) < top_n:
        raise ValueError(f"Only {len(top)} candidates available; need at least {top_n} to build a valid submission.")

    rows = []
    for i, row in enumerate(top, start=1):
        rows.append({
            "candidate_id": row["candidate_id"],
            "rank": i,
            "score": f"{row['final_score']:.4f}",
            "reasoning": generate_reasoning(row),
        })

    df = pd.DataFrame(rows, columns=REQUIRED_HEADER)
    out_path_obj = Path(out_path)
    if out_path_obj.suffix.lower() == ".xlsx":
        df.to_excel(out_path_obj, index=False)
    else:
        df.to_csv(out_path_obj, index=False, encoding="utf-8", lineterminator="\n")

    return str(out_path_obj)


def self_validate(csv_path: str, expected_rows: int = 100) -> list:
    """
    Re-implements validate_submission.py's checks inline (rather than
    shelling out) so `rank.py` can run end-to-end and fail loudly if its own
    output wouldn't pass the organizers' validator. Returns a list of error
    strings; empty list means valid.

    `expected_rows` defaults to 100 (the real competition submission size).
    The sandbox/demo entrypoint passes a smaller value when running against
    a small candidate sample, per submission_spec.docx Section 10.5 ("does
    not need to handle the full 100K pool").
    """
    errors = []
    path = Path(csv_path)

    if path.suffix.lower() not in {".csv", ".xlsx"}:
        errors.append("Filename must use a .csv or .xlsx extension.")

    try:
        if path.suffix.lower() == ".xlsx":
            try:
                df = pd.read_excel(path, engine="openpyxl", header=0, dtype=str, keep_default_na=False)
            except ImportError:
                errors.append("openpyxl is required to validate .xlsx files.")
                return errors
            header = list(df.columns.astype(str))
            data_rows = [list(map(str, row)) for row in df.fillna("").values.tolist()]
        else:
            with open(path, "r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)

                try:
                    header = next(reader)
                except StopIteration:
                    errors.append("Row 1 must be the header row; file is empty.")
                    return errors

                data_rows = [row for row in reader if any(c.strip() for c in row)]
    except UnicodeDecodeError:
        errors.append("File must be UTF-8 encoded.")
        return errors
    except OSError as e:
        errors.append(f"Cannot read file: {e}")
        return errors
    except Exception as e:
        errors.append(f"Cannot read file: {e}")
        return errors

    if len(data_rows) != expected_rows:
        errors.append(f"Expected exactly {expected_rows} data rows, found {len(data_rows)}")

    seen_ids, seen_ranks, by_rank = set(), set(), []
    for i, cells in enumerate(data_rows):
        row_num = i + 2
        if len(cells) != 4:
            errors.append(f"Row {row_num}: expected 4 columns, got {len(cells)}")
            continue
        cid, rank_s, score_s, _ = cells
        if not CANDIDATE_ID_PATTERN.match(cid):
            errors.append(f"Row {row_num}: invalid candidate_id format '{cid}'")
        elif cid in seen_ids:
            errors.append(f"Row {row_num}: duplicate candidate_id '{cid}'")
        else:
            seen_ids.add(cid)

        try:
            rank = int(rank_s)
            if not 1 <= rank <= expected_rows:
                errors.append(f"Row {row_num}: rank {rank} out of 1-{expected_rows} range")
            elif rank in seen_ranks:
                errors.append(f"Row {row_num}: duplicate rank {rank}")
            else:
                seen_ranks.add(rank)
        except ValueError:
            errors.append(f"Row {row_num}: rank '{rank_s}' is not an integer")
            rank = None

        try:
            score = float(score_s)
        except ValueError:
            errors.append(f"Row {row_num}: score '{score_s}' is not a float")
            score = None

        if rank is not None and score is not None and cid:
            by_rank.append((rank, score, cid))

    missing = set(range(1, expected_rows + 1)) - seen_ranks
    if missing:
        errors.append(f"Missing ranks: {sorted(missing)}")

    by_rank.sort(key=lambda x: x[0])
    for i in range(len(by_rank) - 1):
        r1, s1, _ = by_rank[i]
        r2, s2, _ = by_rank[i + 1]
        if s1 < s2:
            errors.append(f"score not non-increasing: rank {r1} ({s1}) < rank {r2} ({s2})")

    # Tie-break check: among equal scores, candidate_id must be ascending.
    # validate_submission.py enforces this explicitly - missing it here would
    # let self_validate() pass a file the organizers' actual validator rejects.
    for i in range(len(by_rank) - 1):
        r1, s1, c1 = by_rank[i]
        r2, s2, c2 = by_rank[i + 1]
        if s1 == s2 and c1 > c2:
            errors.append(
                f"Equal scores at ranks {r1} and {r2}: tie-break requires "
                f"candidate_id ascending ({c1!r} > {c2!r})."
            )

    return errors
