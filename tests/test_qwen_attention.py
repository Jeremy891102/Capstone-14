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


def test_query_positions_shift(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    mod = importlib.import_module("qwen_attention")
    text = '{"s1_ok": true, "s1_err": [], "s1_sec": 34}'
    # token boundaries as Qwen's pre-tokenizer splits this text (BPE never merges across them)
    toks = [
        "{",
        '"',
        "s",
        "1",
        "_ok",
        '":',
        " true",
        ",",
        ' "',
        "s",
        "1",
        "_err",
        '":',
        " [],",
        ' "',
        "s",
        "1",
        "_sec",
        '":',
        " ",
        "3",
        "4",
        "}",
    ]
    offsets, i = [], 0
    for t in toks:
        offsets.append((i, i + len(t)))
        i += len(t)
    assert "".join(toks) == text
    n_prompt = 100
    value, pred = mod.query_positions(
        offsets, mod.answer_spans(text, ["s1_ok", "s1_err", "s1_sec"]), n_prompt
    )
    tok = lambda p: toks[p - n_prompt]  # noqa: E731
    assert [tok(p) for p in value["s1_ok"]] == [" true"]
    assert [tok(p) for p in pred["s1_ok"]] == ['":']  # the token that chooses true/false
    assert [tok(p) for p in pred["s1_err"]] == ['":']  # chooses empty vs non-empty list
    assert [tok(p) for p in value["s1_sec"]] == ["3", "4"]
    assert [tok(p) for p in pred["s1_sec"]] == [" ", "3"]
