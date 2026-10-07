"""Sampled pilot constraints and reproducible nested question sets, offline."""

from __future__ import annotations

import json
from pathlib import Path

from video_report.benchmarks.localization_pilot import check_pairs, nested_selection

SOURCE = Path(__file__).resolve().parents[1] / "data/hd_epic_q40_v1"


def sample():
    reports = list(map(json.loads, (SOURCE / "reports.jsonl").read_text().splitlines()))
    report = next(r for r in reports if r["id"] == "hdepic-dense-016")
    targets = {
        g["field_id"]: g
        for g in map(json.loads, (SOURCE / "ground_truth.jsonl").read_text().splitlines())
        if g["report_id"] == report["id"]
    }
    return report, targets


def test_nested_reproducible_and_temporally_distributed():
    report, targets = sample()
    a = nested_selection(report, targets, seed=17)
    b = nested_selection(report, targets, seed=17)
    assert a["orders"] == b["orders"]
    assert a["subsets"] == b["subsets"]
    check_pairs(a)
    original = {f["id"]: f for f in report["fields"]}
    previous = set()
    for n in [8, 16, 32]:
        fields = a["subsets"][n]
        ids = {f["id"] for f in fields}
        assert len(ids) == n and previous < ids
        assert all(f["id"].startswith("loc_") and f == original[f["id"]] for f in fields)
        assert set(a["time_thirds"][n]) == {0, 1, 2}
        previous = ids
        orders = [[q for q in o if q in ids] for o in a["orders"]]
        assert len(set(map(tuple, orders))) == 3
        for q in orders[0]:
            positions = [o.index(q) for o in orders]
            assert len(set(positions)) == 3
            assert len({int(p / n * 3) for p in positions}) >= 2


def test_insufficient_questions_do_not_get_duplicated():
    report, targets = sample()
    report["fields"] = report["fields"][:4]
    assert nested_selection(report, targets, seed=17) is None


def test_committed_sets_pair_by_stable_ids():
    root = SOURCE.parent / "hd_epic_loc_nested_v1"
    for order in range(1, 4):
        sets = {
            n: {
                r["id"]: r
                for r in map(
                    json.loads,
                    (root / f"loc_n{n:02d}_order{order}" / "reports.jsonl")
                    .read_text()
                    .splitlines(),
                )
            }
            for n in [8, 16, 32]
        }
        for rid in sets[8]:
            assert sets[8][rid]["input"] == sets[16][rid]["input"] == sets[32][rid]["input"]
            ids = {n: [f["id"] for f in sets[n][rid]["fields"]] for n in [8, 16, 32]}
            assert [q for q in ids[32] if q in ids[16]] == ids[16]
            assert [q for q in ids[16] if q in ids[8]] == ids[8]
