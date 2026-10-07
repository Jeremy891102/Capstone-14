"""Offline single-type, temporally stratified nested localization pilot selection.

Targets are used only to identify annotation evidence intervals for sampling/auditing.
They are never added to model inputs. No provider calls are made here.
"""

from __future__ import annotations

import copy
import itertools
import random
import re
from collections import Counter
from typing import Any

TIME = re.compile(r"\d+:\d{2}:\d{2}(?:\.\d+)?")


def seconds(value: str) -> float:
    h, m, s = map(float, value.split(":"))
    return 3600 * h + 60 * m + s


def action(field: dict[str, Any]) -> str:
    match = re.search(r"<([^>]+)>", field["question"])
    if match is None:
        raise ValueError("localization question lacks an action label")
    return match.group(1).lower().replace("put down ", "put ").replace("leave ", "put ")


def evidence(field: dict[str, Any], target: dict[str, Any]) -> tuple[float, float]:
    text = field["choices"][ord(target["target"]) - 65]
    timestamps = TIME.findall(text)
    if len(timestamps) != 2:
        raise ValueError("localization target must contain one time interval")
    start, end = map(seconds, timestamps)
    if not 0 <= start < end:
        raise ValueError("invalid localization evidence interval")
    return start, end


def nested_selection(
    report: dict[str, Any],
    targets: dict[str, dict[str, Any]],
    *,
    seed: int,
    sizes: tuple[int, ...] = (8, 16, 32),
    orders: int = 3,
) -> dict[str, Any] | None:
    if not sizes or sorted(set(sizes)) != list(sizes) or any(n <= 0 for n in sizes):
        raise ValueError("sizes must be positive, unique and ascending")
    if orders < 3:
        raise ValueError("at least three orders are required")
    tr = report["input"]["videos"][0]["time_range"]
    duration = tr["end"] - tr["start"]
    fields = [f for f in report["fields"] if f["id"].startswith("loc_")]
    intervals = {f["id"]: evidence(f, targets[f["id"]]) for f in fields}
    for _a, b in intervals.values():
        if b > duration + 0.002:
            raise ValueError("evidence exceeds clip duration")
    names = {f["id"]: action(f) for f in fields}
    if len(set(names.values())) < max(sizes):
        return None
    rng = random.Random(f"{seed}:{report['id']}")

    def thirds(f: dict[str, Any]) -> int:
        a, b = intervals[f["id"]]
        return min(2, int((a + b) / 2 / duration * 3))

    def compatible(f: dict[str, Any], selected: list[dict[str, Any]]) -> bool:
        x = intervals[f["id"]]
        for other in selected:
            y = intervals[other["id"]]
            if names[f["id"]] == names[other["id"]] or max(x[0], y[0]) < min(x[1], y[1]):
                return False
        return True

    def select(pool: list[dict[str, Any]], n: int) -> list[dict[str, Any]] | None:
        best: tuple[float, list[dict[str, Any]]] | None = None
        for _ in range(128):
            shuffled = pool[:]
            rng.shuffle(shuffled)
            chosen: list[dict[str, Any]] = []
            for _ in range(n):
                available = [f for f in shuffled if f not in chosen and compatible(f, chosen)]
                if not available:
                    break
                counts = Counter(thirds(f) for f in chosen)
                existing = [sum(intervals[f["id"]]) / 2 for f in chosen]

                def rank(
                    f: dict[str, Any],
                    counts: Counter[int] = counts,
                    existing: list[float] = existing,
                ) -> tuple[int, float]:
                    t = sum(intervals[f["id"]]) / 2
                    distance = min((abs(t - x) for x in existing), default=duration)
                    return -counts[thirds(f)], distance

                chosen.append(max(available, key=rank))
            if len(chosen) != n:
                continue
            counts = Counter(thirds(f) for f in chosen)
            if set(counts) != {0, 1, 2}:
                continue
            points = sorted(sum(intervals[f["id"]]) / 2 / duration for f in chosen)
            penalty = sum(abs(counts[b] - n / 3) for b in range(3))
            value = points[-1] - points[0] - penalty
            if best is None or value > best[0]:
                best = value, chosen
        return sorted(best[1], key=lambda f: f["id"]) if best else None

    parent = select(fields, max(sizes))
    if parent is None:
        return None
    subsets = {max(sizes): parent}
    for n in reversed(sizes[:-1]):
        child = select(parent, n)
        if child is None:
            return None
        subsets[n] = child
        parent = child
    largest = subsets[max(sizes)]
    permutations = []
    for _ in range(10000):
        ids = [f["id"] for f in largest]
        rng.shuffle(ids)
        permutations = [
            ids[k:] + ids[:k] for k in [int(i * len(ids) / orders) for i in range(orders)]
        ]
        good = True
        for n in sizes:
            allowed = {f["id"] for f in subsets[n]}
            filtered = [[q for q in order if q in allowed] for order in permutations]
            for q in filtered[0]:
                positions = [order.index(q) for order in filtered]
                if (
                    len(set(positions)) < min(orders, n)
                    or len({int(p / n * 3) for p in positions}) < 2
                ):
                    good = False
                    break
            if not good:
                break
        if good:
            break
    else:
        raise ValueError("could not construct diverse question orders")
    # All question content and choice mappings stay exactly as supplied.
    return {
        "report": copy.deepcopy(report),
        "subsets": subsets,
        "orders": permutations,
        "intervals": intervals,
        "actions": names,
        "time_thirds": {n: dict(Counter(thirds(f) for f in fs)) for n, fs in subsets.items()},
    }


def check_pairs(selection: dict[str, Any]) -> None:
    """Auditable constraints on the largest report; does not certify semantic independence."""
    fields = selection["subsets"][max(selection["subsets"])]
    for a, b in itertools.combinations(fields, 2):
        x, y = selection["intervals"][a["id"]], selection["intervals"][b["id"]]
        assert selection["actions"][a["id"]] != selection["actions"][b["id"]]
        assert max(x[0], y[0]) >= min(x[1], y[1])
