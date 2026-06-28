#!/usr/bin/env python3
"""
Single entrypoint: python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Pipeline: stream candidates -> extract features -> score (percentile-normalize
+ weighted formula + disqualifier/honeypot multipliers) -> sort -> export top
100 -> self-validate against the spec.

Deterministic: no randomness anywhere in the pipeline. Re-running on the same
input file always produces byte-identical output.

Compute: streams the input file line-by-line (loader.py), holds one small
feature dict per candidate (not the raw nested JSON) for scoring, which is
on the order of tens of MB for 100K rows - well inside the 16GB/5-minute
budget required by submission_spec.docx Section 3.
"""

import argparse
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from loader import stream_candidates
from feature_extractor import extract_features
from scorer import score_all
from exporter import export_submission, self_validate

# Fixed reference "now" for recency-based features (availability, staleness),
# derived from the latest last_active_date observed in the released dataset
# at the time this pipeline was built. Using a fixed constant (rather than
# datetime.now()) is required for full determinism - otherwise the same
# input file would score differently depending on when you happen to run it.
CORPUS_REFERENCE_DATE = date(2026, 5, 27)


def main():
    parser = argparse.ArgumentParser(description="Rank candidates for the Redrob Senior AI Engineer JD.")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out", required=True, help="Path to write submission.csv")
    parser.add_argument("--top-n", type=int, default=100, help="Number of ranked candidates to output (default 100)")
    args = parser.parse_args()

    t0 = time.time()

    try:
        print(f"[rank.py] streaming candidates from {args.candidates} ...")
        feature_rows = []
        n_total = 0
        for _, candidate in stream_candidates(args.candidates):
            feature_rows.append(extract_features(candidate, CORPUS_REFERENCE_DATE))
            n_total += 1
        print(f"[rank.py] extracted features for {n_total} candidates in {time.time() - t0:.1f}s")

        print("[rank.py] scoring + ranking ...")
        ranked = score_all(feature_rows, CORPUS_REFERENCE_DATE)

        n_in_output = min(args.top_n, len(ranked))
        n_honeypots_in_pool = sum(1 for r in ranked[:n_in_output] if r["score_breakdown"]["is_honeypot"])
        if n_in_output > 0:
            print(
                f"[rank.py] honeypots in top {n_in_output}: {n_honeypots_in_pool} "
                f"({n_honeypots_in_pool / n_in_output * 100:.1f}%)"
            )

        print(f"[rank.py] writing top {args.top_n} to {args.out} ...")
        export_submission(ranked, args.out, top_n=args.top_n)

        print("[rank.py] self-validating output against submission_spec.docx rules ...")
        errors = self_validate(args.out, expected_rows=args.top_n)
        if errors:
            print(f"[rank.py] VALIDATION FAILED ({len(errors)} issue(s)):")
            for e in errors:
                print(f"  - {e}")
            sys.exit(1)

    except FileNotFoundError as e:
        # Clean one-line message rather than a raw traceback - this is the
        # most common failure mode in a timed run (wrong path / typo) and
        # deserves an immediately readable error.
        print(f"[rank.py] ERROR: {e}")
        sys.exit(1)
    except (ValueError, OSError) as e:
        # Covers exporter.py's "not enough candidates" guard, permission
        # errors writing --out, and similar expected failure modes that
        # already carry a clear message from the layer that raised them.
        print(f"[rank.py] ERROR: {e}")
        sys.exit(1)

    elapsed = time.time() - t0
    print(f"[rank.py] OK. {args.out} is valid. Total runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
