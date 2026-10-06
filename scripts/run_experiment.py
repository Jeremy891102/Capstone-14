#!/usr/bin/env python3
"""Run, resume, or inspect an experiment run.

python scripts/run_experiment.py run --config experiments/exp001_whole_mcq/config.yaml
python scripts/run_experiment.py resume runs/<run_id>
python scripts/run_experiment.py status runs/<run_id>
"""

import sys

from video_report.cli import main

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in {"run", "resume", "status"}:
        print("usage: run_experiment.py {run,resume,status} ...", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))
