# ruff: noqa: E501
"""CaptainCook4D annotations -> benchmarks for the answer-type experiment (E1-E5).

Rules: docs/cc4d_format_questions.md. Design and call counts: docs/cc4d_answer_type_proposal.md.

Usage: python scripts/build_cc4d_answer_types.py <cc4d annotation dir> data/cc4d_answer_types_v1 \
         [--dense data/cc4d_dense_v1] [--seed 0] [--recordings 1_7 ...]

Output: one benchmark directory per group under the output dir. A group is run with a fixed
batch size (the method `batched`), so the field order of a report is its call order:

  base, base_rev        11 one-step cells x main recordings, 8 facts each. E1 k=1, E2 k=4, E3 k=8
  eight_step            the 4 "8 steps in one question" cells, one field each. E1 only
  not_done              one field per not-done step and cell. E1 k=1 (hallucination)
  h4, h8 (+ _rev)       the base reports where one fact per batch is swapped for a not-done step
  e4, e4_rev            mixed answer types: 4 targets then 4 fillers of the row's other types
  e5_loc_*, e5_dur_*    16 facts, bool / single / seconds. k = 1 / 4 / 8 / 16
                        (base, base_rev, not_done, h4, h8, h16 and their _rev)

Dense dataset (cc4d_dense_v1) is read for the step names, recipe context and video references,
so names match the question rules ("<description>", "(2nd time)" for repeats).
"""

import argparse
import collections
import json
import pathlib
import random
import re
import statistics

from video_report.benchmarks.schema import choice_letter

K5 = ["Preparation", "Measurement", "Technique", "Timing", "Temperature"]
NO_MISTAKE = "No mistake"
ERR_OPTS = [*K5, NO_MISTAKE]
CUTS = (17.1, 31.3, 56.5)  # duration cut points, so each range holds similar counts
MARGIN = 3.0  # seconds between a window / range and a step edge, so annotation noise cannot flip it
TIME = re.compile(r"(\d+(?:\.\d+)?)(?:\s*-\s*(\d+(?:\.\d+)?))?\s*(sec|second|min|minute)", re.I)
NOT_DONE = ' If the step was not done in the video, answer "not_visible".'
ONE_STEP = {
    "err": ["bool", "single", "multi", "int", "open_text"],
    "dur": ["bool", "single", "seconds"],
    "loc": ["bool", "single", "seconds"],
}
EIGHT_STEP = {"dur": ["multi", "int"], "loc": ["multi", "int"]}
E5_TYPES = ["bool", "single", "seconds"]
ANSWER_TYPE = {
    "bool": "bool",
    "single": "single_choice",
    "multi": "multi_choice",
    "int": "int",
    "seconds": "seconds",
    "open_text": "open_text",
}
# Error references too vague to judge (docs/scoring_rules.md)
VAGUE = {
    ("1_34", 9), ("2_28", 17), ("2_30", 17), ("2_46", 14), ("5_28", 71), ("5_42", 60),
    ("16_40", 176), ("16_40", 180), ("18_28", 200), ("20_26", 176), ("20_39", 176),
    ("26_24", 303), ("28_50", 330),
}  # fmt: skip

ap = argparse.ArgumentParser()
ap.add_argument("annotations")
ap.add_argument("out")
ap.add_argument("--dense", default="data/cc4d_dense_v1")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--recordings", nargs="*")
args = ap.parse_args()
A, DENSE, OUT = pathlib.Path(args.annotations), pathlib.Path(args.dense), pathlib.Path(args.out)


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def rnd(*key):
    return random.Random(":".join(map(str, (args.seed, *key))))


# ------------------------------------------------------------------ source data

raw = {r["recording_id"]: r for r in json.loads((A / "error_annotations.json").read_text())}
dense = {r["id"].removeprefix("cc4d-"): r for r in jsonl(DENSE / "reports.jsonl")}
questions = {(r["id"], f["id"]): f["question"] for r in dense.values() for f in r["fields"]}
name_of = collections.defaultdict(
    list
)  # (recording, step_id) -> model-facing step name, as in the dense benchmark
for g in jsonl(DENSE / "ground_truth.jsonl"):
    if g["field_id"].endswith("_correct"):
        m = re.fullmatch(
            r'Was the step "(.+)" done correctly\?', questions[g["report_id"], g["field_id"]]
        )
        ev = g["evidence"]  # some recordings repeat a step_id, so the start time is part of the key
        name_of[g["report_id"].removeprefix("cc4d-"), ev["step_id"], ev["start_time"]].append(
            m.group(1)
        )


def ancestors(recipe):
    """step name -> names of every step that must come before it (from the 'after' lists)."""
    after = {}
    for line in recipe.splitlines():
        if line.startswith("- "):
            name, _, rest = line[2:].partition(" (after: ")
            after[name] = [x for x in rest.removesuffix(")").split("; ") if x]
    closure = {}

    def walk(n, seen=()):
        if n not in closure:
            closure[n] = set(after.get(n, []))
            for p in after.get(n, []):
                closure[n] |= walk(p) if p not in seen else set()
        return closure[n]

    return {n: walk(n) for n in after}


def load(rid):
    r = raw[rid]
    steps, used = [], collections.Counter()
    for s in sorted(r["step_annotations"], key=lambda s: s["step_id"]):
        key = rid, s["step_id"], s["start_time"]
        used[key] += 1  # identical not-done steps share a key; they are interchangeable
        tags = {e["tag"].removesuffix(" Error") for e in s.get("errors") or []}
        filmed = s["start_time"] >= 0 and "Missing Step" not in tags
        steps.append(
            {
                "id": s["step_id"],
                "name": sorted(name_of[key])[used[key] - 1],
                "raw": s["description"],
                "s": s["start_time"],
                "e": s["end_time"],
                "tags": tags,
                "errs": s.get("errors") or [],
                "mod": s.get("modified_description"), "filmed": filmed,
                "not_done": s["start_time"] < 0 and "Missing Step" in tags,
            }  # fmt: skip
        )
    assert len({s["name"] for s in steps}) == len(steps), f"{rid}: repeated step names"
    d = dense[rid]
    return {
        "rid": rid, "steps": steps, "video": d["input"]["videos"][0], "context": d["input"]["context"],
        "end": d["input"]["videos"][0]["time_range"]["end"],
        "anc": ancestors(d["input"]["context"]["recipe"]),
    }  # fmt: skip


def dur(s):
    return s["e"] - s["s"]


def err_ok(rid, s):
    t = s["tags"]
    return (
        s["filmed"]
        and "Other" not in t
        and not (t and not t & set(K5))
        and (rid, s["id"]) not in VAGUE
    )


def dur_ok(s):
    return s["filmed"] and not TIME.search(s["raw"]) and all(abs(dur(s) - c) > 2 for c in CUTS)


def pools(rec):
    st, rid = rec["steps"], rec["rid"]
    err = [s for s in st if err_ok(rid, s)]
    return {
        "err": err,
        "err_pos": [s for s in err if s["tags"] & set(K5)],
        "err_neg": [s for s in err if not s["tags"]],
        "dur": [s for s in st if dur_ok(s)],
        "loc": [s for s in st if s["filmed"]],
    }


# ------------------------------------------------------------------ facts


def balanced(a, b, rng, block=4):
    """Interleave two equal lists so every block of `block` facts holds half of each."""
    out = []
    for i in range(0, len(a), block // 2):
        chunk = a[i : i + block // 2] + b[i : i + block // 2]
        rng.shuffle(chunk)
        out += chunk
    return out


def pick_facts(rec, content, n):
    rng = rnd(rec["rid"], "facts", content, n)
    p = pools(rec)
    if content == "loc":
        facts = rng.sample(p["loc"], n)
    else:
        a, b = (p["err_pos"], p["err_neg"]) if content == "err" else (
            [s for s in p["dur"] if dur(s) > CUTS[1]], [s for s in p["dur"] if dur(s) <= CUTS[1]]
        )  # fmt: skip
        a, b = rng.sample(a, min(n // 2, len(a))), rng.sample(b, min(n // 2, len(b)))
        if len(a) == len(b) == n // 2:
            facts = balanced(a, b, rng)
        else:  # not enough of one kind: fill from the other and shuffle
            rest = [s for s in p[content] if s not in a + b]
            facts = (a + b + rng.sample(rest, n - len(a) - len(b)))[:n]
            rng.shuffle(facts)
    facts = [dict(s) for s in facts]
    for i in range(0, n, 4):  # loc bool windows: half of each block of 4 overlap the step
        flags = [True, True, False, False]
        rng.shuffle(flags)
        for s, f in zip(facts[i : i + 4], flags, strict=False):
            s["overlap"] = f
    return facts


# ------------------------------------------------------------------ questions


def window(rng, rec, lo_hi, overlap, length):
    """An integer window (t1, t2) that overlaps lo_hi by at least MARGIN seconds, or stays
    at least MARGIN away from it. None when the video is too short for that."""
    s, e = lo_hi
    end = int(rec["end"])
    for L in (length, length // 2, length // 3):
        if overlap:
            lo, hi = max(0, int(s - L + MARGIN) + 1), min(end - L, int(e - MARGIN))
            starts = range(lo, hi + 1)
        else:
            starts = [*range(0, int(s - MARGIN - L) + 1), *range(int(e + MARGIN) + 1, end - L + 1)]
        if len(starts):
            t = rng.choice(list(starts))
            return t, t + L
    return None


def touches(win, s):  # does the window overlap step s?
    return s["s"] < win[1] and s["e"] > win[0]


def stable(win, s):  # the answer cannot flip with annotation noise
    return min(abs(s["s"] - win[1]), abs(s["e"] - win[0])) >= MARGIN or (
        touches(win, s) and min(s["e"], win[1]) - max(s["s"], win[0]) >= MARGIN
    )


def depends(rec, a, b):
    return b in rec["anc"].get(a, ()) or a in rec["anc"].get(b, ())


def evidence(s):
    return {
        "step_id": s["id"], "start_time": s["s"], "end_time": s["e"], "errors": s["errs"],
        "modified_description": s["mod"],
    }  # fmt: skip


def letters(names, picked):
    return [choice_letter(names.index(x)) for x in picked]


def one_step(rec, content, typ, s):
    """(field spec without id, target, evidence) for one question about one step."""
    rng = rnd(rec["rid"], s["id"], content, typ)
    X, done = s["name"], not s["not_done"]
    spec, ev = {"answer_type": ANSWER_TYPE[typ], "allow_not_visible": True}, evidence(s)
    if content == "err":
        tags = [t for t in K5 if t in s["tags"]]
        if typ == "bool":
            q, tgt = f'Was the step "{X}" done correctly?', not tags
        elif typ == "single":
            q = f'Which of these mistakes was made in the step "{X}"? If several were made, pick any one.'
            tgt = letters(ERR_OPTS, tags or [NO_MISTAKE])
            tgt = tgt[0] if len(tgt) == 1 else tgt
            spec["choices"] = ERR_OPTS
        elif typ == "multi":
            q, tgt = (
                f'Which of these mistakes were made in the step "{X}"?',
                letters(ERR_OPTS, tags or [NO_MISTAKE]),
            )
            spec["choices"], spec["exclusive_choice"] = ERR_OPTS, NO_MISTAKE
        elif typ == "int":
            q, tgt = f'How many different kinds of mistakes were made in the step "{X}"?', len(tags)
        else:
            q = (
                f'Compared with the recipe, what did the person do differently in the step "{X}"? '
                'Answer "none" if the step was done as written.'
            )
            tgt = [
                e["description"].strip() for e in s["errs"] if e["tag"].removesuffix(" Error") in K5
            ]
    elif content == "dur":
        c = [round(x) for x in CUTS]
        if typ == "bool":
            q, tgt = (
                f'Did the step "{X}" take longer than {c[1]} seconds?',
                done and dur(s) > CUTS[1],
            )
        elif typ == "single":
            spec["choices"] = [f"under {c[0]} seconds", f"{c[0]} to {c[1]} seconds",
                               f"{c[1]} to {c[2]} seconds", f"{c[2]} seconds or more"]  # fmt: skip
            q, tgt = (
                f'How long did the step "{X}" take?',
                choice_letter(sum(dur(s) > x for x in CUTS)) if done else None,
            )
        else:
            q, tgt = (
                f'How many seconds did the step "{X}" take?',
                round(dur(s), 1) if done else None,
            )
    else:
        end = int(rec["end"])
        if typ == "seconds":
            q = f'At what time, in seconds from the start of the video, did the step "{X}" start?'
            spec["timestamp"], tgt = True, round(s["s"], 1) if done else None
            ev["length"] = round(dur(s), 1) if done else None
        elif typ == "bool":
            w = window(rng, rec, (s["s"], s["e"]), s["overlap"], 30) if done else None
            w = w or (lambda t: (t, t + 30))(rng.randrange(0, end - 30))
            q = f'Was the step "{X}" being done at any time between {w[0]} and {w[1]} seconds from the start of the video?'
            tgt = done and touches(w, s)
        else:
            others = [o for o in rec["steps"] if o["filmed"] and o is not s and o["id"] != s["id"]]
            far = [
                o
                for o in others
                if not done or o["e"] <= s["s"] - MARGIN or o["s"] >= s["e"] + MARGIN
            ]
            free = [o for o in far if not depends(rec, X, o["name"])]
            rest = free if len(free) >= 4 else far
            pool = {f"{round(o['s'])} to {round(o['e'])} seconds": o for o in rest}
            picks = rng.sample(sorted(pool), min(len(pool), 4 if done else 5))
            true = f"{round(s['s'])} to {round(s['e'])} seconds"
            opts = [*picks, true] if done else picks
            rng.shuffle(opts)
            spec["choices"] = opts
            q, tgt = (
                f'When was the step "{X}" done?',
                choice_letter(opts.index(true)) if done else None,
            )
    spec["question"] = q + NOT_DONE
    return spec, "not_visible" if not done or tgt is None else tgt, ev


def eight_step(rec, content, typ, facts):
    """One question that lists the row's 8 facts as steps (duration or localization)."""
    rng = rnd(rec["rid"], "eight", content)
    names = [s["name"] for s in facts]
    if content == "dur":
        hit = [dur(s) > CUTS[1] for s in facts]
        what = f"took longer than {round(CUTS[1])} seconds"
    else:
        want, best = rng.randint(1, 5), None  # aim for a random count so "2" is not always right
        for _ in range(3000):
            L = rng.randrange(20, 241)
            t = rng.randrange(0, max(1, int(rec["end"]) - L))
            w = (t, t + L)
            if not all(stable(w, s) for s in facts):
                continue
            n = sum(touches(w, s) for s in facts)
            if best is None or abs(n - want) < abs(best[1] - want):
                best = (w, n)
            if n == want:
                break
        assert best, f"{rec['rid']}: no stable window for the 8 steps"
        w = best[0]
        hit = [touches(w, s) for s in facts]
        what = f"were being done at any time between {w[0]} and {w[1]} seconds from the start of the video"
    ev = {"steps": names, "hit": hit}
    if typ == "multi":
        spec = {
            "question": f"Which of these steps {what}?",
            "answer_type": "multi_choice",
            "choices": names,
        }
        return spec, [choice_letter(i) for i, h in enumerate(hit) if h], ev
    lines = "\n".join(f"- {n}" for n in names)
    return (
        {"question": f"Among these steps, how many {what}?\n{lines}", "answer_type": "int"},
        sum(hit),
        ev,
    )


# ------------------------------------------------------------------ output

GROUPS: dict[str, tuple[list, list]] = {}


def emit(group, rec, suffix, items, meta):
    """items: (field id, spec, target, evidence) in call order."""
    reports, gt = GROUPS.setdefault(group, ([], []))
    rid = f"cc4d-{rec['rid']}__{suffix}"
    reports.append(
        {
            "id": rid,
            "input": {"videos": [rec["video"]], "context": rec["context"]},
            "fields": [{"id": fid, **spec} for fid, spec, _, _ in items],
            "source": {
                "dataset": "captaincook4d",
                "split": "all",
                "original_ids": {"recording_id": rec["rid"]},
                "annotation_refs": [f"error_annotations.json#recording_id={rec['rid']}"],
            },  # fmt: skip
            "metadata": {"created_by": "build_cc4d_answer_types", "seed": args.seed, **meta},
        }
    )
    gt += [
        {"report_id": rid, "field_id": fid, "target": t, "evidence": ev} for fid, _, t, ev in items
    ]


def fact_item(rec, content, typ, s, i, role="fact", n_fact=None):
    spec, tgt, ev = one_step(rec, content, typ, s)
    fid = f"{content}{i:02d}_{typ}" if role != "not_done" else f"{content}nd{i}_{typ}"
    return (
        fid,
        spec,
        tgt,
        {**ev, "role": role, "fact": None if role == "not_done" else f"{content}{i:02d}"},
    )


def build_row_groups(prefix, recs, facts, rows, k_values):
    """Base / reversed / not-done / H-batch groups for the given rows and answer types."""
    nd_recs = [r for r in recs if any(s["not_done"] for s in r["steps"])]
    for rec in recs:
        for content, types in rows.items():
            for typ in types:
                row = facts[rec["rid"], content]
                items = [fact_item(rec, content, typ, s, i + 1) for i, s in enumerate(row)]
                meta = {"content": content, "type": typ, "n_facts": len(row)}
                emit(f"{prefix}base", rec, f"{content}_{typ}", items, meta)
                emit(f"{prefix}base_rev", rec, f"{content}_{typ}", items[::-1], meta)
    for rec in nd_recs:
        nds = [s for s in rec["steps"] if s["not_done"]]
        for content, types in rows.items():
            for c, typ in enumerate(types):
                row = facts[rec["rid"], content]
                for n, s in enumerate(nds, 1):
                    item = fact_item(rec, content, typ, s, n, "not_done")
                    emit(
                        f"{prefix}not_done",
                        rec,
                        f"nd{n}_{content}_{typ}",
                        [item],
                        {"content": content, "type": typ},
                    )
                for k in k_values:
                    items = []
                    for b in range(len(row) // k):
                        p, nd = (nd_recs.index(rec) + c + b) % k, nds[(c + b) % len(nds)]
                        for j in range(k):
                            i = b * k + j
                            items.append(
                                fact_item(rec, content, typ, nd, b + 1, "not_done")
                                if j == p else fact_item(rec, content, typ, row[i], i + 1)
                            )  # fmt: skip
                    meta = {"content": content, "type": typ, "k": k}
                    emit(f"{prefix}h{k}", rec, f"{content}_{typ}", items, meta)
                    emit(f"{prefix}h{k}_rev", rec, f"{content}_{typ}", items[::-1], meta)


def build_mixed(recs, facts):
    """E4: 4 targets of type T first, then 4 fillers of the row's other types. Run for each half."""
    for rec in recs:
        for content, types in ONE_STEP.items():
            row = facts[rec["rid"], content]
            for t_i, T in enumerate(types):
                others = [x for x in types if x != T]
                for half in (0, 1):
                    tgt_idx, fill_idx = (
                        (range(0, 4), range(4, 8)) if half == 0 else (range(4, 8), range(0, 4))
                    )
                    cyc = (
                        others if len(others) == 4 else [others[0], others[0], others[1], others[1]]
                    )
                    r = (recs.index(rec) + t_i + half) % 4
                    fill_types = [cyc[(m + r) % 4] for m in range(4)]
                    items = [fact_item(rec, content, T, row[i], i + 1, "target") for i in tgt_idx]
                    items += [
                        fact_item(rec, content, ft, row[i], i + 1, "filler")
                        for i, ft in zip(fill_idx, fill_types, strict=True)
                    ]
                    meta = {"content": content, "type": T, "half": half}
                    emit("e4", rec, f"{content}_{T}_half{half}", items, meta)
                    emit("e4_rev", rec, f"{content}_{T}_half{half}", items[::-1], meta)


def write(group, reports, gt):
    d = OUT / group
    d.mkdir(parents=True, exist_ok=True)
    for name, rows in (("reports.jsonl", reports), ("ground_truth.jsonl", gt)):
        (d / name).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
    (d / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "video_report.v1", "benchmark_id": f"{OUT.name}_{group}",
                "description": f"CaptainCook4D answer-type experiment, group {group}.",
                "reports_file": "reports.jsonl", "ground_truth_file": "ground_truth.jsonl",
                "mock_responses_file": None, "video_root": None, "source_datasets": ["captaincook4d"],
                "notes": "Video uris are <recording_id>_360p.mp4. time_range ends 5 s after the last step.",
            },
            indent=2,
        )
        + "\n"
    )  # fmt: skip


# ------------------------------------------------------------------ main

ids = args.recordings or sorted(dense, key=lambda k: [int(x) for x in k.split("_")])
recs = [load(rid) for rid in ids if rid in raw]
P = {r["rid"]: pools(r) for r in recs}
main = [
    r for r in recs
    if len(P[r["rid"]]["err"]) >= 8 and len(P[r["rid"]]["err_pos"]) >= 4
    and len(P[r["rid"]]["err_neg"]) >= 4 and len(P[r["rid"]]["dur"]) >= 8 and len(P[r["rid"]]["loc"]) >= 8
]  # fmt: skip
facts = {(r["rid"], c): pick_facts(r, c, 8) for r in main for c in ONE_STEP}
build_row_groups("", main, facts, ONE_STEP, (4, 8))
for rec in main:
    for content, types in EIGHT_STEP.items():
        for typ in types:
            spec, tgt, ev = eight_step(rec, content, typ, facts[rec["rid"], content])
            spec["allow_not_visible"] = False
            emit(
                "eight_step",
                rec,
                f"{content}_{typ}_8step",
                [(f"{content}8_{typ}", spec, tgt, ev)],
                {"content": content, "type": typ},
            )
build_mixed(main, facts)

# E5: 16 facts per report; localization samples 4 recordings per recipe (activity id)
e5 = {
    "loc": [r for r in recs if len(P[r["rid"]]["loc"]) >= 16],
    "dur": [r for r in recs if len(P[r["rid"]]["dur"]) >= 16],
}
by_recipe = {}
for r in e5["loc"]:
    by_recipe.setdefault(r["rid"].split("_")[0], []).append(r)
e5["loc"] = [
    r
    for rs in by_recipe.values()
    for r in rnd(rs[0]["rid"].split("_")[0], "e5").sample(rs, min(4, len(rs)))
]
for content, rs in e5.items():
    f16 = {(r["rid"], content): pick_facts(r, content, 16) for r in rs}
    build_row_groups(f"e5_{content}_", rs, f16, {content: E5_TYPES}, (4, 8, 16))

for group, (reports, gt) in GROUPS.items():
    write(group, reports, gt)

nd = [s for r in main for s in r["steps"] if s["not_done"]]
print(f"main recordings {len(main)} (of {len(recs)}); not-done steps {len(nd)} in "
      f"{sum(any(s['not_done'] for s in r['steps']) for r in main)} recordings")  # fmt: skip
print(f"E5 localization {len(e5['loc'])} recordings, duration {len(e5['dur'])}")
fields_per_call = {"base": (1, 4, 8), "e4": (8,)}
for group, (reports, gt) in sorted(GROUPS.items()):
    print(f"{group:22} {len(reports):6} reports {len(gt):7} fields")
print("median video minutes (main):", round(statistics.median(r["end"] for r in main) / 60, 1))
