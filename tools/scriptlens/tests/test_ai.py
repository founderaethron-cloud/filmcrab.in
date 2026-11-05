import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scriptlens import ai
from scriptlens.engine import analyze
from scriptlens.extract import extract

SAMPLE = Path(__file__).resolve().parent.parent / "samples" / "sample_mixed.txt"


class FakeClient:
    def __init__(self, *a, **k):
        self.calls = []
        self.messages = self
        self.bad_once = True

    def create(self, model, max_tokens, system, messages):
        prompt = messages[0]["content"]
        self.calls.append(prompt)
        if "SCRIPT SECTION" in prompt:
            out = {"scenes": [{"scene": "1", "summary": "x"}], "character_beats": [], "dialogue_notes": [], "language_notes": []}
        elif "PRODUCER COVERAGE" in prompt:
            if self.bad_once:  # first coverage reply is malformed -> must retry
                self.bad_once = False
                return SimpleNamespace(content=[SimpleNamespace(type="text", text="Sure! {not json")])
            out = {"logline": "L", "ratings": {}, "recommendation": {"verdict": "Consider with rewrites"}}
        else:
            out = {"rewrite_plan": [{"priority": 1, "change": "c"}]}
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="```json\n" + json.dumps(out) + "\n```")])


@pytest.fixture
def script():
    ex = extract("s.txt", SAMPLE.read_bytes())
    return analyze(ex.text)


def test_chunking_respects_limit(script):
    _, scenes = script
    chunks = ai.build_chunks(scenes, max_words=60)
    assert len(chunks) > 1 and sum(len(c) for c in chunks) == len(scenes)
    assert [s.index for c in chunks for s in c] == [s.index for s in scenes]


def test_run_both_lenses_with_retry(script, monkeypatch):
    import anthropic

    holder = {}
    monkeypatch.setattr(anthropic, "Anthropic", lambda **k: holder.setdefault("c", FakeClient()))
    monkeypatch.setattr(ai, "CHUNK_WORDS", 60)
    # build_chunks default arg is bound at definition time, so pass via monkeypatching the function default
    rep, scenes = script
    msgs = []
    out = ai.run(scenes, rep, "both", lambda m, f: msgs.append((m, f)), api_key="test")
    assert out["coverage"]["logline"] == "L" and out["doctor"]["rewrite_plan"][0]["priority"] == 1
    assert msgs[-1][1] == 1.0
    assert any("not valid JSON" in c for c in holder["c"].calls)  # retry happened


def test_requires_key(script, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    rep, scenes = script
    with pytest.raises(RuntimeError):
        ai.run(scenes, rep, "both")


def test_over_limit_refused(script, monkeypatch):
    rep, scenes = script
    monkeypatch.setattr(ai, "MAX_WORDS", 10)
    with pytest.raises(ValueError):
        ai.run(scenes, rep, "coverage", api_key="k")


def test_parse_json_variants():
    assert ai._parse_json('```json\n{"a":1}\n```') == {"a": 1}
    assert ai._parse_json('Here you go: {"a": 2} thanks') == {"a": 2}
