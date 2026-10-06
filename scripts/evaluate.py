#!/usr/bin/env python3
"""Score a finished (or partial) run offline, without new model calls.

    python scripts/evaluate.py --run runs/<run_id> \
        --ground-truth benchmarks/mock_mcq_v1/ground_truth.jsonl
"""

import sys

from video_report.cli import main

if __name__ == "__main__":
    sys.exit(main(["evaluate", *sys.argv[1:]]))
