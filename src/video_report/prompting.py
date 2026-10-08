"""Prompt building: render an experiment's templates for one planned call.

Templates are plain text using ``string.Template`` placeholders (``$name``; write ``$$`` for a
literal dollar sign). Unknown or missing placeholders are errors, not silent blanks.

Available placeholders:

* ``$questions``      numbered block of the call's fields with lettered choices
* ``$answer_format``  JSON skeleton with exactly the requested field ids as keys
* ``$field_ids``      comma-separated requested field ids
* ``$num_fields``     number of requested fields
* ``$context``        whitelisted context lines (``key: value``), or ``(none)``

Inputs are model-facing only: the requested ``FieldSpec``s (question + choices) and the
report's ``input.context`` filtered by ``prompt.context_keys``. Targets, evidence, ``source``
and ``metadata`` are never available here.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from string import Template

from video_report.benchmarks.schema import ContextValue, FieldSpec, choice_letter

PLACEHOLDERS = frozenset({"questions", "answer_format", "field_ids", "num_fields", "context"})


class PromptError(ValueError):
    pass


@dataclass(frozen=True)
class PromptTemplates:
    system: str | None
    user: str

    def check(self) -> None:
        for name, text in (("system", self.system), ("user", self.user)):
            if text is None:
                continue
            unknown = _identifiers(name, text) - PLACEHOLDERS
            if unknown:
                raise PromptError(
                    f"{name} template uses unknown placeholders {sorted(unknown)}; "
                    f"allowed: {sorted(PLACEHOLDERS)}"
                )
        if "questions" not in _identifiers("user", self.user):
            raise PromptError("user template must contain $questions")


def _identifiers(name: str, text: str) -> set[str]:
    """Placeholder names in a template (Template.get_identifiers needs Python 3.11+)."""
    found: set[str] = set()
    for m in Template.pattern.finditer(text):
        if m.group("invalid") is not None:
            raise PromptError(f"{name} template has an invalid '$' placeholder")
        ident = m.group("named") or m.group("braced")
        if ident:
            found.add(ident)
    return found


def render_questions(fields: Sequence[FieldSpec]) -> str:
    blocks = []
    for n, f in enumerate(fields, start=1):
        lines = [f"Q{n}. [field_id: {f.id}]", f"Question: {f.question}"]
        if f.choices:
            lines.append("Options:")
            lines += [f"{choice_letter(i)}. {c}" for i, c in enumerate(f.choices)]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _answer_hint(f: FieldSpec) -> str:
    letters = "/".join(f.letters())
    hint = {
        "single_choice": f"one letter: {letters}",
        "multi_choice": f"list of letters from {letters}, [] if none",
        "bool": "true or false",
        "int": "integer",
        "seconds": "number of seconds",
    }[f.answer_type]
    return f"<{hint}" + (' or "not_visible"' if f.allow_not_visible else "") + ">"


def render_answer_format(fields: Sequence[FieldSpec]) -> str:
    return json.dumps({f.id: _answer_hint(f) for f in fields}, indent=2)


def render_context(context: Mapping[str, ContextValue], keys: Sequence[str]) -> str:
    lines = [f"{k}: {context[k]}" for k in keys if k in context]
    return "\n".join(lines) if lines else "(none)"


def render(
    templates: PromptTemplates,
    fields: Sequence[FieldSpec],
    context: Mapping[str, ContextValue],
    context_keys: Sequence[str],
) -> tuple[str | None, str]:
    values = {
        "questions": render_questions(fields),
        "answer_format": render_answer_format(fields),
        "field_ids": ", ".join(f.id for f in fields),
        "num_fields": str(len(fields)),
        "context": render_context(context, context_keys),
    }
    system = Template(templates.system).substitute(values) if templates.system else None
    return system, Template(templates.user).substitute(values)
