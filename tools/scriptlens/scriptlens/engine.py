"""Deterministic screenplay analysis: scenes, characters, language mix, production flags, pacing.

Everything here is rule-based and reproducible. Numbers are measurements of the text; anything
labelled 'heuristic' or 'estimate' is a rough aid, not a verdict.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .lexicon import (
    FLAG_KEYWORDS, FLAG_LABELS, FLAG_WEIGHTS, MANGLISH_SUFFIXES, MANGLISH_WORDS,
    RESERVED_CUES, TIME_WORDS, TRANSITION_PREFIXES, flag_pattern,
)

ML_RANGE = "ഀ-ൿ‌‍"
TOKEN_RE = re.compile(rf"[{ML_RANGE}]+|[A-Za-z][A-Za-z'’\-]*")
ML_RE = re.compile(rf"[{ML_RANGE}]")
WORDS_PER_PAGE = 180  # assumption used only when the source has no real page count

HEADING_RE = re.compile(
    r"^\s*(?:(?:scene|sc\.?|രംഗം|സീൻ)\s*[:.\-]?\s*)?(?P<num>\d{1,3}[A-Za-z]?)?\s*[.):\-]?\s*"
    r"(?P<ie>int\.?\s*/\s*ext\.?|ext\.?\s*/\s*int\.?|i\s*/\s*e\.?|int\.?|ext\.?|"
    r"ഇൻ്റീരിയർ|ഇന്റീരിയർ|എക്സ്റ്റീരിയർ|അകം|പുറം)(?P<rest>[\s.:\-–—].*)?$",
    re.I,
)
SCENE_WORD_RE = re.compile(
    r"^\s*(?:scene|sc\.?|രംഗം|സീൻ)\s*[:.\-]?\s*(?P<num>\d{1,3}[A-Za-z]?)\s*[:.\-]?\s*(?P<rest>.*)$", re.I
)
COLON_CUE_RE = re.compile(rf"^(?P<name>[^:\n]{{1,32}}?)\s*:\s*(?P<text>.*)$")
LATIN_CAPS_NAME = re.compile(r"^[A-Z][A-Z0-9 .'\-/&]{0,28}$")
EXT_RE = re.compile(r"\s*\((?:V\.?O\.?|O\.?S\.?|O\.?C\.?|CONT'?D\.?|FILTERED|ON PHONE|PHONE)\)\s*$", re.I)
ML_NAME_RE = re.compile(rf"^[{ML_RANGE}]+(?: [{ML_RANGE}]+){{0,2}}$")


@dataclass
class Scene:
    index: int
    label: str
    heading: str
    int_ext: str
    time_of_day: str
    location: str
    sublocation: str
    modifiers: list[str]
    lines: list[str] = field(default_factory=list)
    action_text: list[str] = field(default_factory=list)
    dialogue: list[tuple[str, str]] = field(default_factory=list)  # (speaker, text)
    flags: list[str] = field(default_factory=list)
    action_words: int = 0
    dialogue_words: int = 0
    characters: set[str] = field(default_factory=set)
    complexity: float = 1.0

    @property
    def total_words(self) -> int:
        return self.action_words + self.dialogue_words

    @property
    def text(self) -> str:
        return "\n".join(self.lines).strip()


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def classify_token(tok: str, names: set[str]) -> str:
    if ML_RE.match(tok):
        return "ml"
    low = tok.lower().strip("'’-")
    if tok.upper() in names:
        return "name"
    if low in MANGLISH_WORDS:
        return "manglish"
    if len(low) >= 5 and low.endswith(MANGLISH_SUFFIXES):
        return "manglish"
    return "en"


def _norm_name(raw: str) -> str:
    n = EXT.sub("", raw).strip(" .:-")
    return n.upper() if not ML_RE.search(n) else re.sub(r"\s+", " ", n)


EXT = EXT_RE


def _is_transition(s: str) -> bool:
    u = s.upper().strip(" .:")
    return u in RESERVED_CUES or u.startswith(TRANSITION_PREFIXES)


def _parse_heading(raw: str) -> dict | None:
    m = HEADING_RE.match(raw)
    num = None
    if m:
        ie_raw = m.group("ie").lower().replace(" ", "")
        rest = (m.group("rest") or "").strip(" .:-–—")
        num = m.group("num")
        if ie_raw.startswith(("int/", "i/e")) or ("int" in ie_raw and "ext" in ie_raw):
            ie = "INT/EXT"
        elif ie_raw.startswith("int") or ie_raw in ("ഇൻ്റീരിയർ", "ഇന്റീരിയർ", "അകം"):
            ie = "INT"
        else:
            ie = "EXT"
    else:
        m2 = SCENE_WORD_RE.match(raw)
        if not m2:
            return None
        ie, num, rest = "UNSPECIFIED", m2.group("num"), m2.group("rest").strip(" .:-–—")
        if ML_RE.search(rest[:6]) and rest.startswith(("അകം", "പുറം")):
            ie = "INT" if rest.startswith("അകം") else "EXT"
            rest = rest[3:].strip(" .:-")
    segs = [s.strip() for s in re.split(r"\s+[-–—]\s+", rest) if s.strip()]
    location = segs[0] if segs else "UNSPECIFIED"
    tod, modifiers = "UNSPECIFIED", []
    for seg in segs[1:]:
        found = _time_of(seg)
        if found and tod == "UNSPECIFIED":
            tod = found
        elif found is None:
            modifiers.append(seg)
    if tod == "UNSPECIFIED":  # trailing time word without a dash: "KITCHEN NIGHT"
        last = location.split()[-1].lower() if location.split() else ""
        found = _time_of(last) if len(location.split()) > 1 else None
        if found:
            tod = found
            location = " ".join(location.split()[:-1])
    main = location.split(",")[0].strip()
    sub = ", ".join(p.strip() for p in location.split(",")[1:])
    return {
        "num": num, "int_ext": ie, "tod": tod, "location": main.upper() if not ML_RE.search(main) else main,
        "sub": sub, "mods": modifiers,
    }


def _time_of(seg: str) -> str | None:
    low = seg.lower()
    words = set(re.findall(rf"[a-z]+|[{ML_RANGE}]+", low))
    for cat, ws in TIME_WORDS.items():
        for w in ws:
            if w in words or (ML_RE.search(w) and w in low):
                return cat
    return None


def _is_cue_line(s: str, prev_blank: bool, next_line: str | None) -> bool:
    if next_line is None or not next_line.strip() or _is_transition(s):
        return False
    if _parse_heading(next_line) is not None:
        return False
    core = EXT.sub("", s).strip()
    if LATIN_CAPS_NAME.match(core) and any(c.isalpha() for c in core) and len(core) >= 2:
        return True
    if prev_blank and ML_NAME_RE.match(s) and len(s) <= 20:
        return True
    return False


def split_scenes(text: str) -> tuple[list[Scene], list[str]]:
    warnings: list[str] = []
    scenes: list[Scene] = []
    cur: Scene | None = None
    preamble: list[str] = []
    for raw in text.split("\n"):
        h = _parse_heading(raw) if raw.strip() else None
        if h:
            cur = Scene(
                index=len(scenes) + 1,
                label=h["num"] or str(len(scenes) + 1),
                heading=raw.strip(),
                int_ext=h["int_ext"], time_of_day=h["tod"], location=h["location"],
                sublocation=h["sub"], modifiers=h["mods"],
            )
            scenes.append(cur)
        elif cur is None:
            preamble.append(raw)
        else:
            cur.lines.append(raw)
    if not scenes:
        warnings.append(
            "No scene headings were detected (expected lines like 'INT. LOCATION - DAY' or 'SCENE 12'). "
            "The whole text is treated as one block, so scene-level metrics are not meaningful."
        )
        scenes = [Scene(1, "1", "(unstructured text)", "UNSPECIFIED", "UNSPECIFIED", "UNSPECIFIED", "", [], lines=text.split("\n"))]
    return scenes, warnings


def _strict_cue_name(s: str) -> str | None:
    """Return normalised speaker if `s` (before the colon) looks like a character cue."""
    core = EXT.sub("", s).strip()
    if not core or _is_transition(core) or core.upper() in RESERVED_CUES:
        return None
    if ML_RE.search(core):
        return _norm_name(core) if ML_NAME_RE.match(core) and len(core) <= 24 else None
    if LATIN_CAPS_NAME.match(core) and any(c.isalpha() for c in core):
        return _norm_name(core)
    return None


def fill_scene(sc: Scene, known: set[str]) -> None:
    lines = sc.lines
    speaker: str | None = None
    in_dialogue = False
    prev_blank = True
    for i, raw in enumerate(lines):
        s = raw.strip()
        nxt = lines[i + 1] if i + 1 < len(lines) else None
        if not s:
            in_dialogue, speaker, prev_blank = False, None, True
            continue
        m = COLON_CUE_RE.match(s)
        if m:
            name = _strict_cue_name(m.group("name"))
            if name is None:
                t = m.group("name").strip()
                if t.upper() in known and t.upper() not in RESERVED_CUES and len(t.split()) <= 3:
                    name = t.upper()
            if name:
                speaker, in_dialogue = name, True
                if m.group("text").strip():
                    sc.dialogue.append((speaker, m.group("text").strip()))
                sc.characters.add(speaker)
                prev_blank = False
                continue
        if in_dialogue:
            if s.startswith("(") and s.endswith(")"):
                prev_blank = False
                continue
            sc.dialogue.append((speaker or "UNKNOWN", s))
            prev_blank = False
            continue
        if _is_cue_line(s, prev_blank, nxt):
            speaker, in_dialogue = _norm_name(s), True
            sc.characters.add(speaker)
            prev_blank = False
            continue
        sc.action_text.append(s)
        prev_blank = False


def collect_known_cues(scenes: list[Scene]) -> set[str]:
    """First pass: names that appear in unambiguous cue form (caps / Malayalam), used to accept 'Sajan:' later."""
    known: Counter = Counter()
    for sc in scenes:
        lines = sc.lines
        for i, raw in enumerate(lines):
            s = raw.strip()
            if not s:
                continue
            m = COLON_CUE_RE.match(s)
            if m:
                n = _strict_cue_name(m.group("name"))
                if n:
                    known[n] += 1
                    continue
            nxt = lines[i + 1] if i + 1 < len(lines) else None
            prev_blank = i == 0 or not lines[i - 1].strip()
            if LATIN_CAPS_NAME.match(EXT.sub("", s).strip()) and _is_cue_line(s, prev_blank, nxt):
                known[_norm_name(s)] += 1
    return {n for n, c in known.items() if c >= 1 and not ML_RE.search(n)}


def _flags_for(sc: Scene) -> list[str]:
    # Action lines and heading only: characters talking about a storm is not a storm shoot.
    blob = " ".join(sc.action_text) + " " + sc.heading
    found = []
    for flag in FLAG_KEYWORDS:
        pat, ml = flag_pattern(flag)
        if pat.search(blob) or any(w in blob for w in ml):
            found.append(flag)
    if re.search(r"^\s*(song|ഗാനം|പാട്ട്)\s*:", " ".join(sc.action_text), re.I | re.M) and "song" not in found:
        found.append("song")
    return found


def _complexity(sc: Scene) -> float:
    c = 1.0
    if sc.time_of_day == "NIGHT":
        c += 0.5
    if sc.int_ext in ("EXT", "INT/EXT"):
        c += 0.25
    c += sum(FLAG_WEIGHTS.get(f, 0.0) for f in sc.flags)
    return round(c, 2)


def _role(rank: int, share: float) -> str:
    if rank < 2 and share >= 0.12:
        return "Lead"
    if share >= 0.05:
        return "Major"
    if share >= 0.015:
        return "Supporting"
    return "Minor"


def analyze(text: str, page_count: int | None = None, filename: str = "", source_type: str = "txt",
            extraction_warnings: list[str] | None = None) -> tuple[dict, list[Scene]]:
    warnings = list(extraction_warnings or [])
    scenes, w = split_scenes(text)
    warnings += w
    known = collect_known_cues(scenes)
    for sc in scenes:
        fill_scene(sc, known)

    speakers = Counter()
    for sc in scenes:
        for sp, t in sc.dialogue:
            speakers[sp] += len(tokens(t))
    names = {n for n in speakers}
    # Merge trivially different names: "SAJAN" vs "SAJAN (V.O.)" already normalised; no fuzzy merging on purpose.

    char = defaultdict(lambda: {
        "lines": 0, "words": 0, "scenes": set(), "lang": Counter(), "first": None,
    })
    dlg_lang, act_lang = Counter(), Counter()
    cum = 0
    total_words_all = 0
    for sc in scenes:
        for a in sc.action_text:
            toks = tokens(a)
            sc.action_words += len(toks)
            for t in toks:
                k = classify_token(t, names)
                act_lang["en" if k == "name" else k] += 1
        for sp, t in sc.dialogue:
            toks = tokens(t)
            sc.dialogue_words += len(toks)
            ch = char[sp]
            ch["lines"] += 1
            ch["words"] += len(toks)
            ch["scenes"].add(sc.index)
            for tk in toks:
                k = classify_token(tk, names)
                if k != "name":
                    ch["lang"][k] += 1
                    dlg_lang[k] += 1
        # known characters named in action lines count as present
        action_blob = " ".join(sc.action_text)
        for n in names:
            if n not in sc.characters and n != "UNKNOWN":
                if (ML_RE.search(n) and n in action_blob) or re.search(rf"\b{re.escape(n)}\b", action_blob, re.I):
                    sc.characters.add(n)
        for n in sc.characters:
            char[n]["scenes"].add(sc.index)
        sc.flags = _flags_for(sc)
        sc.complexity = _complexity(sc)
        total_words_all += sc.total_words

    # position of each scene (midpoint, fraction of total words)
    pos = {}
    run = 0
    for sc in scenes:
        mid = run + sc.total_words / 2
        pos[sc.index] = mid / total_words_all if total_words_all else 0
        run += sc.total_words
    for n, ch in char.items():
        if ch["scenes"]:
            first = min(ch["scenes"])
            ch["first"] = first

    total_dlg = sum(speakers.values())
    ranked = speakers.most_common()
    characters = []
    for rank, (n, w_) in enumerate(ranked):
        ch = char[n]
        share = w_ / total_dlg if total_dlg else 0
        lang_total = sum(ch["lang"].values()) or 1
        characters.append({
            "name": n, "role": _role(rank, share), "dialogue_words": w_, "dialogue_share": round(share, 3),
            "lines": ch["lines"], "scenes": len(ch["scenes"]),
            "first_scene": ch["first"], "first_appearance_pct": round(pos.get(ch["first"], 0) * 100, 1),
            "lang_pct": {k: round(100 * ch["lang"].get(k, 0) / lang_total) for k in ("ml", "manglish", "en")},
        })

    pairs = Counter()
    for sc in scenes:
        cs = sorted(c for c in sc.characters if c != "UNKNOWN")
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                pairs[(cs[i], cs[j])] += 1
    top_pairs = [{"a": a, "b": b, "shared_scenes": c} for (a, b), c in pairs.most_common(6)]

    n_sc = len(scenes)
    loc_counter = Counter(sc.location for sc in scenes if sc.location != "UNSPECIFIED")
    flag_counts = Counter(f for sc in scenes for f in sc.flags)
    ie_counts = Counter(sc.int_ext for sc in scenes)
    tod_counts = Counter(sc.time_of_day for sc in scenes)
    night_ext = sum(1 for sc in scenes if sc.time_of_day == "NIGHT" and sc.int_ext in ("EXT", "INT/EXT"))
    avg_c = sum(sc.complexity for sc in scenes) / n_sc if n_sc else 0
    tier = "Contained" if avg_c < 1.6 else "Moderate" if avg_c < 2.4 else "Heavy"

    # pacing: 10 bins by position in the script
    bins = [{"scenes": 0, "words": 0, "dialogue": 0, "complexity": 0.0, "night": 0} for _ in range(10)]
    for sc in scenes:
        b = min(9, int(pos[sc.index] * 10))
        bins[b]["scenes"] += 1
        bins[b]["words"] += sc.total_words
        bins[b]["dialogue"] += sc.dialogue_words
        bins[b]["complexity"] += sc.complexity
        bins[b]["night"] += 1 if sc.time_of_day == "NIGHT" else 0
    pacing = []
    for i, b in enumerate(bins):
        pacing.append({
            "bin": i + 1, "scenes": b["scenes"],
            "avg_scene_words": round(b["words"] / b["scenes"]) if b["scenes"] else 0,
            "talk_ratio": round(b["dialogue"] / b["words"], 2) if b["words"] else 0,
            "avg_complexity": round(b["complexity"] / b["scenes"], 2) if b["scenes"] else 0,
            "night_scenes": b["night"],
        })

    # talk-heavy stretches: >=4 consecutive scenes with talk ratio > 0.8
    stretches, run_s = [], []
    for sc in scenes + [None]:  # type: ignore[list-item]
        ok = sc is not None and sc.total_words and sc.dialogue_words / sc.total_words > 0.8
        if ok:
            run_s.append(sc)
        else:
            if len(run_s) >= 4:
                stretches.append({"from_scene": run_s[0].label, "to_scene": run_s[-1].label, "scenes": len(run_s)})
            run_s = []

    lead = characters[0] if characters else None
    observations = []
    if lead and lead["first_appearance_pct"] > 10:
        observations.append(
            f"Top-dialogue character {lead['name']} first appears at about {lead['first_appearance_pct']}% of the script "
            "(by word position). Check this is intentional."
        )
    if characters and total_dlg and characters[0]["dialogue_share"] > 0.5:
        observations.append(f"{characters[0]['name']} carries {round(characters[0]['dialogue_share']*100)}% of all dialogue.")
    for s in stretches:
        observations.append(
            f"Scenes {s['from_scene']}-{s['to_scene']} ({s['scenes']} scenes in a row) are almost entirely dialogue."
        )
    if n_sc > 1 and ie_counts.get("UNSPECIFIED", 0) / n_sc > 0.3:
        observations.append("Over 30% of scene headings have no INT/EXT marker; location and cost metrics are less reliable.")
    if n_sc > 1 and tod_counts.get("UNSPECIFIED", 0) / n_sc > 0.3:
        observations.append("Over 30% of scene headings have no time of day; night-shoot counts may be understated.")

    words_total = total_words_all
    est_pages = page_count or round(words_total / WORDS_PER_PAGE, 1)
    dl_total = sum(dlg_lang.values()) or 1
    report = {
        "meta": {
            "filename": filename, "source_type": source_type,
            "pages": page_count, "pages_estimated": est_pages if not page_count else None,
            "words": words_total, "dialogue_words": total_dlg, "action_words": words_total - total_dlg,
            "dialogue_share": round(total_dlg / words_total, 3) if words_total else 0,
            "est_runtime_minutes": round(est_pages, 0) if est_pages else None,
            "runtime_note": "Rough: assumes about one page per minute. Songs and long silent passages are not modelled.",
        },
        "warnings": warnings,
        "structure": {
            "scene_count": n_sc,
            "int_ext": dict(ie_counts), "time_of_day": dict(tod_counts),
            "unique_locations": len(loc_counter),
            "top_locations": [{"location": l, "scenes": c} for l, c in loc_counter.most_common(8)],
            "avg_scene_words": round(words_total / n_sc) if n_sc else 0,
            "longest_scenes": [
                {"scene": sc.label, "heading": sc.heading, "words": sc.total_words}
                for sc in sorted(scenes, key=lambda s: -s.total_words)[:5]
            ],
        },
        "language": {
            "dialogue_words_pct": {k: round(100 * dlg_lang.get(k, 0) / dl_total, 1) for k in ("ml", "manglish", "en")},
            "note": "Heuristic. 'Manglish' = romanised Malayalam matched against a word list; unknown romanised words count as English.",
        },
        "characters": {
            "speaking_count": len(characters), "list": characters[:25],
            "top3_dialogue_share": round(sum(c["dialogue_share"] for c in characters[:3]), 3),
            "top_pairs": top_pairs,
        },
        "production": {
            "complexity_tier": tier, "avg_scene_complexity": round(avg_c, 2),
            "night_exterior_scenes": night_ext,
            "flag_counts": {f: {"label": FLAG_LABELS[f], "scenes": c} for f, c in flag_counts.most_common()},
            "heaviest_scenes": [
                {"scene": sc.label, "heading": sc.heading, "complexity": sc.complexity,
                 "flags": [FLAG_LABELS[f] for f in sc.flags]}
                for sc in sorted(scenes, key=lambda s: -s.complexity)[:8]
            ],
            "note": "Heuristic, uncalibrated weights from keywords in the text. Use for relative comparison between scenes, not as a budget.",
        },
        "pacing": {"bins": pacing, "talk_heavy_stretches": stretches},
        "observations": observations,
        "scenes": [
            {"index": sc.index, "label": sc.label, "heading": sc.heading, "int_ext": sc.int_ext,
             "time_of_day": sc.time_of_day, "location": sc.location, "words": sc.total_words,
             "dialogue_words": sc.dialogue_words, "speakers": sorted(c for c in sc.characters if c != "UNKNOWN"),
             "flags": sc.flags, "complexity": sc.complexity, "preview": sc.text[:160]}
            for sc in scenes
        ],
    }
    return report, scenes
