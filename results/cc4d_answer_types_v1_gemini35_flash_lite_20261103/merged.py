"""Final field scores per run: runs_rerun/rr_X replaces the reports it re-ran in runs/fl_X."""

import glob
import json
import os

here = os.path.dirname(os.path.abspath(__file__))


def _rows(d):
    return [json.loads(x) for x in open(sorted(glob.glob(f"{d}/evaluations/2*/field_scores.jsonl"))[-1])]


def load(run):
    rows = _rows(f"{here}/runs/{run}")
    rr = f"{here}/runs_rerun/rr_{run.removeprefix('fl_')}"
    if not os.path.isdir(rr):
        return rows
    new = _rows(rr)
    redone = {r["report_id"] for r in new}
    return [r for r in rows if r["report_id"] not in redone] + new


def runs():
    return sorted(os.path.basename(d.rstrip("/")) for d in glob.glob(f"{here}/runs/*/"))
