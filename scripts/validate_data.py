#!/usr/bin/env python3
"""Validate a prepared benchmark directory against the data contract.

python scripts/validate_data.py --benchmark benchmarks/mock_mcq_v1 [--check-videos]
"""

import sys

from video_report.cli import main

if __name__ == "__main__":
    sys.exit(main(["validate-data", *sys.argv[1:]]))
