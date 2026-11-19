"""CC4D reference-context x video control: accuracy, context gains and D (docs/cc4d/new_experiment/reference_context_design.zh.md).

D = (video ctx7 - video single) - (novideo ctx7 - novideo single), paired by target,
bootstrap over recordings (10000 draws, seed 20261112).

    python results/cc4d_reference.py
"""

import gzip
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

D = Path(__file__).parent / "cc4d_reference_v1"
T = json.load(open(D.parent.parent / "data/cc4d_ref_v1/targets.json"))["targets"]
INFO = {t["report_id"]: (t["recording"], t["type"]) for t in T}
CELLS = [(v, c) for v in ("video", "novideo") for c in ("single", "ctx7")]


def load(model, prefix="fs"):
    out = {}
    for v, c in CELLS:
        rows = [json.loads(x) for x in gzip.open(D / f"{prefix}_{model}_{v}_{c}.jsonl.gz", "rt")]
        out[v, c] = {r["report_id"]: r for r in rows}
    return out


def cell_acc(cells, ids):
    return {k: sum(cells[k][i]["correct"] for i in ids) / len(ids) for k in CELLS}


def stats(cells, ids):
    a = cell_acc(cells, ids)
    return a, a["video", "ctx7"] - a["video", "single"], a["novideo", "ctx7"] - a["novideo", "single"]


def boot(cells, ids, n=10000, seed=20261112):
    by = defaultdict(list)
    for i in ids:
        by[INFO[i][0]].append(i)
    recs, rng, res = list(by), random.Random(seed), []
    for _ in range(n):
        s = [i for r in (rng.choice(recs) for _ in recs) for i in by[r]]
        _, gv, gn = stats(cells, s)
        res.append((gv, gn, gv - gn))
    q = lambda xs: (sorted(xs)[int(0.025 * n)], sorted(xs)[int(0.975 * n)])  # noqa: E731
    return [q([r[j] for r in res]) for j in range(3)]


def transitions(cells, ids, v):
    c = Counter((cells[v, "single"][i]["correct"], cells[v, "ctx7"][i]["correct"]) for i in ids)
    return c[True, False], c[False, True]


def const_acc(cells, ids):
    """Best fixed answer per type (oracle upper bound for a constant-answer strategy)."""
    t = Counter()
    for i in ids:
        t[repr(cells["video", "single"][i]["target"])] += 1
    return max(t.values()) / len(ids)


def answered(cells):
    """Targets where no condition answered not_visible (same questions in all four cells)."""
    return {i for i in INFO if all(cells[k][i]["predicted"] != "not_visible" for k in CELLS)}


def main():
    pp = lambda x: f"{100 * x:5.1f}"  # noqa: E731
    runs = [(m, "fs", o) for o in (False, True) for m in ("argon", "qwen")] + [("qwen", "fsnv", False)]
    for model, prefix, only_answered in runs:
        cells = load(model, prefix)
        assert all(set(cells[k]) == set(INFO) for k in CELLS), "target set mismatch"
        keep = answered(cells) if only_answered else set(INFO)
        print(f"\n=== {model}{' NO not_visible allowed' if prefix == 'fsnv' else ''}{' (only targets never answered not_visible: %d of %d)' % (len(keep), len(INFO)) if only_answered else ''} ===")
        print(f"{'subset':12} {'n':>5} | video: single ctx7  gain | novid: single ctx7  gain |   D [95% CI]   | flips video(c>w/w>c) novideo")
        for name, ids in [("all", sorted(keep))] + [(t, sorted(i for i in keep if INFO[i][1] == t)) for t in ("correct", "error_types", "duration")]:
            if not ids:
                print(f"{name:12} {0:5} | (no targets)")
                continue
            a, gv, gn = stats(cells, ids)
            (lv, hv), (ln, hn), (ld, hd) = boot(cells, ids)
            tv, tn = transitions(cells, ids, "video"), transitions(cells, ids, "novideo")
            print(
                f"{name:12} {len(ids):5} | {pp(a['video','single'])} {pp(a['video','ctx7'])} {pp(gv):>5} | "
                f"{pp(a['novideo','single'])} {pp(a['novideo','ctx7'])} {pp(gn):>5} | "
                f"{pp(gv - gn)} [{pp(ld)},{pp(hd)}] | {tv[0]}/{tv[1]}  {tn[0]}/{tn[1]}"
            )
            print(f"{'':12} {'':5}   gain CI video [{pp(lv)},{pp(hv)}]  novideo [{pp(ln)},{pp(hn)}]   fixed-answer ceiling {pp(const_acc(cells, ids))}")


if __name__ == "__main__":
    main()
