"""LLM layer: one shared map pass over the script, then a producer-coverage view and a script-doctor view."""
from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from .engine import Scene

MODEL = os.environ.get("SCRIPTLENS_MODEL", "claude-sonnet-5-5")
CHUNK_WORDS = 3500
MAX_WORDS = 80_000

SYSTEM = """You are a senior script reader and story editor with deep working knowledge of Malayalam cinema (Mollywood):
its storytelling traditions, regional registers and dialects (Malabar, Kochi, Thiruvananthapuram, Thrissur, Kottayam and others),
the common code-switching between Malayalam, Manglish and English, and the realities of mid-budget and low-budget Kerala production.

Rules:
- Ground every claim in the script text. Cite scene labels (e.g. "sc 12") for each point. Quote at most ~10 words at a time.
- Never invent box-office figures, awards, audience data or release facts. If you name comparable films, say they come from your
  general knowledge and must be verified; prefer describing the type of film over naming titles.
- Separate what the text shows from your judgement. Mark uncertain calls as uncertain.
- On Malayalam dialect, register and idiom you are a careful reader, not a native-speaker authority: flag concerns as
  "check with a native-speaking script editor" rather than asserting errors.
- Be direct and specific. No flattery, no generic advice that could apply to any script.
- Respond with a single JSON object only. No prose outside the JSON, no markdown fences."""

MAP_PROMPT = """Below is a contiguous section of a screenplay (scenes {first} to {last}). Read it and return JSON:
{{
  "scenes": [{{"scene": "<label>", "summary": "<=25 words", "dramatic_purpose": "<=15 words", "conflict": "<=15 words or 'none'", "turn": "what changes by the end, <=15 words, or 'none'"}}],
  "character_beats": [{{"name": "...", "scene": "<label>", "beat": "<=20 words"}}],
  "dialogue_notes": [{{"scene": "<label>", "note": "on-the-nose exposition, voice sameness, strong lines, register mismatch (<=25 words)"}}],
  "language_notes": [{{"scene": "<label>", "note": "dialect/register/code-switching observation (<=25 words)"}}]
}}
Include every scene. Keep notes only where there is something worth noting.

SCRIPT SECTION:
{body}"""

COVERAGE_PROMPT = """You have per-scene reader notes for an entire screenplay, plus measured facts. Write PRODUCER COVERAGE:
a decision-support read for a Malayalam production house deciding whether to develop, option or pass.

Return JSON with exactly these keys:
{{
  "logline": "one sentence",
  "synopsis": "<=150 words",
  "genre_tone": "short phrase",
  "ratings": {{"premise": {{"score": 1-5, "why": "...", "scenes": []}}, "structure": {{}}, "character": {{}}, "dialogue": {{}}, "pacing": {{}}, "originality": {{}}, "cultural_specificity": {{}}}},
  "strengths": [{{"point": "...", "scenes": []}}],
  "weaknesses": [{{"point": "...", "scenes": []}}],
  "marketability_hypotheses": [{{"hypothesis": "who might want this and why", "basis_in_text": "...", "how_to_validate": "specific, cheap test with real buyers/audience"}}],
  "production_risks": [{{"risk": "...", "scenes": [], "mitigation": "..."}}],
  "recommendation": {{"verdict": "Pass | Consider with rewrites | Recommend | Strong recommend", "confidence": "low | medium | high", "rationale": "...", "what_would_change_my_mind": "..."}},
  "questions_for_writer": ["..."]
}}
Fill every rating (score and why). Marketability items are hypotheses, not facts; do not state market sizes or revenue.

MEASURED FACTS:
{facts}

READER NOTES (JSON):
{notes}"""

DOCTOR_PROMPT = """You have per-scene reader notes for an entire screenplay, plus measured facts. Write a SCRIPT DOCTOR read for the writer:
a craft-level diagnosis with a prioritised rewrite plan.

Return JSON with exactly these keys:
{{
  "structure_map": [{{"beat": "Setup | Inciting incident | First turn | Midpoint | Crisis | Climax | Resolution", "scene": "<label or null if absent>", "assessment": "..."}}],
  "character_arcs": [{{"name": "...", "want": "...", "need": "...", "arc": "...", "issues": "...", "scenes": []}}],
  "scene_notes": [{{"scene": "<label>", "problem": "...", "fix": "..."}}],
  "dialogue": {{"voice_distinctness": "...", "exposition_problems": [{{"scene": "<label>", "problem": "..."}}], "strong_moments": [{{"scene": "<label>", "why": "..."}}]}},
  "malayalam_craft": {{"code_switching": "...", "dialect_and_register": "...", "kinship_and_honorifics": "...", "idiom_vs_translated_feel": "...", "native_review_flags": [{{"scene": "<label>", "concern": "..."}}]}},
  "theme": "what the script is about underneath, and whether the ending earns it",
  "rewrite_plan": [{{"priority": 1, "change": "...", "why": "...", "scenes": []}}]
}}
scene_notes: the weakest scenes only (at most 8). rewrite_plan: 5 to 8 items, highest impact first.

MEASURED FACTS:
{facts}

READER NOTES (JSON):
{notes}"""


def build_chunks(scenes: list[Scene], max_words: int = CHUNK_WORDS) -> list[list[Scene]]:
    chunks: list[list[Scene]] = []
    cur: list[Scene] = []
    n = 0
    for sc in scenes:
        w = max(sc.total_words, 1)
        if cur and n + w > max_words:
            chunks.append(cur)
            cur, n = [], 0
        cur.append(sc)
        n += w
    if cur:
        chunks.append(cur)
    return chunks


def facts_digest(report: dict) -> str:
    c = report["characters"]["list"][:8]
    d = {
        "pages": report["meta"].get("pages") or report["meta"].get("pages_estimated"),
        "scenes": report["structure"]["scene_count"],
        "locations": report["structure"]["unique_locations"],
        "dialogue_share": report["meta"]["dialogue_share"],
        "dialogue_language_pct": report["language"]["dialogue_words_pct"],
        "characters": [
            {"name": x["name"], "role": x["role"], "dialogue_share": x["dialogue_share"],
             "first_appearance_pct": x["first_appearance_pct"], "lang_pct": x["lang_pct"]} for x in c
        ],
        "production_flags": {k: v["scenes"] for k, v in report["production"]["flag_counts"].items()},
        "night_exterior_scenes": report["production"]["night_exterior_scenes"],
        "pacing_by_tenth": [{"bin": b["bin"], "talk_ratio": b["talk_ratio"], "avg_scene_words": b["avg_scene_words"]} for b in report["pacing"]["bins"]],
    }
    return json.dumps(d, ensure_ascii=False)


def _parse_json(text: str) -> dict:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        a, b = t.find("{"), t.rfind("}")
        if a != -1 and b > a:
            return json.loads(t[a : b + 1])
        raise


def _call(client, prompt: str, max_tokens: int) -> dict:
    last_err: Exception | None = None
    for _ in range(2):  # one retry if the model returns malformed JSON
        resp = client.messages.create(
            model=MODEL, max_tokens=max_tokens, system=SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        try:
            return _parse_json(text)
        except json.JSONDecodeError as e:
            last_err = e
            prompt += "\n\nYour previous reply was not valid JSON. Return only the JSON object."
    raise ValueError(f"Model returned invalid JSON twice: {last_err}")


def _merge_notes(parts: list[dict]) -> dict:
    out = {"scenes": [], "character_beats": [], "dialogue_notes": [], "language_notes": []}
    for p in parts:
        for k in out:
            v = p.get(k)
            if isinstance(v, list):
                out[k].extend(v)
    return out


def total_words(scenes: list[Scene]) -> int:
    return sum(s.total_words for s in scenes)


def run(scenes: list[Scene], report: dict, lens: str, progress: Callable[[str, float], None] | None = None,
        api_key: str | None = None) -> dict:
    """lens: 'coverage' | 'doctor' | 'both'. Returns {'coverage': {...}, 'doctor': {...}} for the lenses requested."""
    import anthropic

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set on the server.")
    if total_words(scenes) > MAX_WORDS:
        raise ValueError(f"Script is over {MAX_WORDS:,} words; split it before running AI analysis.")
    client = anthropic.Anthropic(api_key=key, max_retries=3)
    say = progress or (lambda m, f: None)

    chunks = build_chunks(scenes)
    say(f"Reading {len(chunks)} section(s)", 0.02)
    parts: list[dict | None] = [None] * len(chunks)
    done = 0

    def work(i: int) -> None:
        nonlocal done
        ch = chunks[i]
        body = "\n\n".join(f"[SCENE {s.label}] {s.heading}\n{s.text}" for s in ch)
        parts[i] = _call(client, MAP_PROMPT.format(first=ch[0].label, last=ch[-1].label, body=body), 4000)
        done += 1
        say(f"Read section {done} of {len(chunks)}", 0.05 + 0.65 * done / len(chunks))

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(work, range(len(chunks))))

    notes = json.dumps(_merge_notes([p for p in parts if p]), ensure_ascii=False)
    facts = facts_digest(report)
    out: dict = {"model": MODEL}
    lenses = ["coverage", "doctor"] if lens == "both" else [lens]
    for i, ln in enumerate(lenses):
        say(f"Writing {'producer coverage' if ln == 'coverage' else 'script-doctor notes'}", 0.72 + 0.25 * i / len(lenses))
        tmpl = COVERAGE_PROMPT if ln == "coverage" else DOCTOR_PROMPT
        out[ln] = _call(client, tmpl.format(facts=facts, notes=notes), 7000)
    say("Done", 1.0)
    return out
