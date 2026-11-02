"""LLM judge for open_text answers (rules: docs/scoring_rules.md, ``open_text``).

The judge sees text only: the question, the reference mistakes, any `modified_description`
notes and the answer. It never sees the video. A judge is any callable ``JudgeInput ->
JudgeVerdict``; ``build_prompt`` and ``parse_verdict`` are shared by every implementation so
the rule is written once.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from video_report.providers.base import ModelRequest, RetryableProviderError

JUDGE_PROMPT_VERSION = "open_text_v1"


@dataclass(frozen=True)
class JudgeInput:
    question: str
    reference: list[str]  # mistakes that were made; [] = the step was done as written
    answer: str
    notes: list[str]  # modified_description of the step, if any


@dataclass(frozen=True)
class JudgeVerdict:
    correct: bool
    matched: list[int]  # indices into the reference
    extra_claims: int  # mistakes the answer claims that are not in the reference


Judge = Callable[[JudgeInput], JudgeVerdict]


def build_prompt(inp: JudgeInput) -> str:
    ref = "\n".join(f"{i}. {r}" for i, r in enumerate(inp.reference)) or "none"
    notes = "\n".join(f"- {n}" for n in inp.notes) or "none"
    return (
        "You grade a short answer about a cooking step. You cannot see the video.\n\n"
        f"Question: {inp.question}\n"
        f"Reference (the mistakes that were really made, numbered; 'none' = done as written):\n"
        f"{ref}\n"
        f"Extra notes about the step:\n{notes}\n"
        f"Answer to grade: {inp.answer}\n\n"
        "Rules:\n"
        "- If the reference is 'none', the answer is correct only if it says nothing was done "
        "differently. Naming any mistake is incorrect.\n"
        "- Otherwise the answer is correct if it names at least one reference mistake by what "
        "was actually done wrong. Naming only a category (e.g. 'a timing error') or something "
        "vague ('done badly') does not count. Different wording with the same meaning counts.\n"
        "- Extra claimed mistakes do not change correctness; count them in extra_claims.\n\n"
        'Reply with JSON only: {"correct": true|false, "matched": [reference numbers], '
        '"extra_claims": integer}'
    )


def parse_verdict(text: str, n_reference: int) -> JudgeVerdict:
    """Strict parse of the judge's reply; raises ValueError on anything unexpected."""
    obj = json.loads(text.strip())
    if not isinstance(obj, dict) or not {"correct", "matched", "extra_claims"} <= obj.keys():
        raise ValueError(f"unexpected judge reply: {text[:200]!r}")
    matched, extra = obj["matched"], obj["extra_claims"]
    ok = isinstance(obj["correct"], bool) and isinstance(extra, int) and not isinstance(extra, bool)
    ok = ok and isinstance(matched, list) and extra >= 0
    ok = ok and all(
        isinstance(i, int) and not isinstance(i, bool) and 0 <= i < n_reference for i in matched
    )
    if not ok:
        raise ValueError(f"unexpected judge reply: {text[:200]!r}")
    return JudgeVerdict(obj["correct"], sorted(set(matched)), extra)


class ProviderJudge:
    """Judge backed by a text-only model call. Verdicts are cached on disk, so re-evaluating a
    run does not pay for the same answer twice. Failures after ``attempts`` tries raise."""

    def __init__(
        self,
        provider: Any,
        model: str,
        *,
        cache_path: Path | None = None,
        attempts: int = 3,
        timeout_s: float = 120.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.provider, self.model = provider, model
        self.attempts, self.timeout_s, self.sleep = attempts, timeout_s, sleep
        self.cache_path = cache_path
        self.cache: dict[str, JudgeVerdict] = {}
        self.stats = {"calls": 0, "cache_hits": 0, "input_tokens": 0, "output_tokens": 0}
        if cache_path is not None and cache_path.is_file():
            for line in cache_path.read_text().splitlines():
                row = json.loads(line)
                self.cache[row["key"]] = JudgeVerdict(**row["verdict"])

    def describe(self) -> dict[str, Any]:
        describe = getattr(self.provider, "describe", None)
        backend = describe().get("backend") if describe else None
        return {"model": self.model, "backend": backend, **self.stats}

    def __call__(self, inp: JudgeInput) -> JudgeVerdict:
        prompt = build_prompt(inp)
        key = hashlib.sha256(f"{JUDGE_PROMPT_VERSION}\n{self.model}\n{prompt}".encode()).hexdigest()
        if key in self.cache:
            self.stats["cache_hits"] += 1
            return self.cache[key]
        verdict = self._ask(prompt, len(inp.reference))
        self.cache[key] = verdict
        if self.cache_path is not None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a") as f:
                f.write(json.dumps({"key": key, "verdict": asdict(verdict)}) + "\n")
        return verdict

    def _ask(self, prompt: str, n_reference: int) -> JudgeVerdict:
        request = ModelRequest(
            call_id="judge",
            report_id="judge",
            field_ids=(),
            system=None,
            user=prompt,
            videos=(),
            model=self.model,
            # no max_output_tokens: thinking models spend it before writing the reply
            generation={"temperature": 0.0, "response_format": "json"},
        )
        last: Exception | None = None
        for attempt in range(self.attempts):
            if attempt:
                self.sleep(2.0**attempt)
            try:
                reply = self.provider.generate(request, timeout_s=self.timeout_s)
                self.stats["calls"] += 1
                if reply.usage is not None:
                    self.stats["input_tokens"] += reply.usage.input_tokens or 0
                    self.stats["output_tokens"] += (reply.usage.output_tokens or 0) + (
                        reply.usage.reasoning_tokens or 0
                    )
                return parse_verdict(reply.text or "", n_reference)
            except (RetryableProviderError, ValueError) as exc:  # bad JSON is a ValueError
                last = exc
        raise RuntimeError(f"judge failed after {self.attempts} attempts: {last}")
