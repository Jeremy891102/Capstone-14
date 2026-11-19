import importlib
from pathlib import Path


def test_answer_spans(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    spans = importlib.import_module("qwen_attention").answer_spans
    text = '{"s1_ok": true, "s1_err": ["A", "C"], "s1_sec": 12.5, "note": "a, b"}'
    got = {
        f: text[a:b]
        for f, (a, b) in spans(text, ["s1_ok", "s1_err", "s1_sec", "note", "gone"]).items()
    }
    assert got == {"s1_ok": "true", "s1_err": '["A", "C"]', "s1_sec": "12.5", "note": '"a, b"'}
