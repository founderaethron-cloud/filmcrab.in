from pathlib import Path

import pytest

from scriptlens.engine import analyze, split_scenes, tokens, classify_token
from scriptlens.extract import extract, normalize_malayalam

SAMPLE = Path(__file__).resolve().parent.parent / "samples" / "sample_mixed.txt"


@pytest.fixture(scope="module")
def sample():
    ex = extract("sample_mixed.txt", SAMPLE.read_bytes())
    return analyze(ex.text, ex.page_count, "sample_mixed.txt", ex.source_type, ex.warnings)


def test_scene_count_and_settings(sample):
    rep, _ = sample
    assert rep["structure"]["scene_count"] == 8
    assert rep["structure"]["int_ext"] == {"EXT": 4, "INT": 4}
    assert rep["structure"]["time_of_day"]["NIGHT"] == 3


def test_heading_parts():
    scenes, _ = split_scenes("12. INT. SAJAN'S HOUSE, KITCHEN - MORNING\nx\n\n13A. EXT. KAYAL - NIGHT - RAIN\ny")
    a, b = scenes
    assert (a.label, a.int_ext, a.time_of_day, a.location, a.sublocation) == ("12", "INT", "DAY", "SAJAN'S HOUSE", "KITCHEN")
    assert (b.label, b.int_ext, b.time_of_day, b.modifiers) == ("13A", "EXT", "NIGHT", ["RAIN"])


def test_interval_is_not_a_heading():
    scenes, warns = split_scenes("INTERVAL\nsome text")
    assert warns and scenes[0].heading == "(unstructured text)"


def test_malayalam_headings():
    scenes, _ = split_scenes("രംഗം 3 അകം തറവാട് - രാത്രി\nഅമ്മ ഇരിക്കുന്നു.")
    assert scenes[0].label == "3" and scenes[0].int_ext == "INT" and scenes[0].time_of_day == "NIGHT"


def test_both_cue_styles_and_speakers(sample):
    rep, _ = sample
    names = {c["name"] for c in rep["characters"]["list"]}
    assert {"SAJAN", "AMMINI", "MATHEW", "RENU", "DOCTOR", "TOUT"} <= names
    # FIGHT: / SONG: are stage directions, not characters
    assert "FIGHT" not in names and "SONG" not in names and "CUT TO BLACK" not in names


def test_renu_colon_lines_counted(sample):
    rep, _ = sample
    renu = next(c for c in rep["characters"]["list"] if c["name"] == "RENU")
    assert renu["lines"] == 3  # 'Help! Please!' plus two colon-style lines


def test_parentheticals_not_dialogue():
    text = "INT. ROOM - DAY\n\nMATHEW\n(smiling)\nStorm varatte.\n"
    rep, _ = analyze(text)
    assert rep["meta"]["dialogue_words"] == 2


def test_flags_ignore_dialogue():
    text = "INT. ROOM - DAY\n\nBOB\nThere is a storm coming.\n\nHe sits quietly.\n"
    rep, _ = analyze(text)
    assert rep["scenes"][0]["flags"] == []
    rep2, _ = analyze("EXT. ROAD - NIGHT\n\nHeavy rain. A jeep skids.\n")
    assert {"weather", "vehicle"} <= set(rep2["scenes"][0]["flags"])


def test_language_classification():
    names = {"SAJAN"}
    assert classify_token("നീ", names) == "ml"
    assert classify_token("Enthu", names) == "manglish"
    assert classify_token("odikkumo", names) == "manglish"  # suffix rule needs >=5 chars; in lexicon anyway
    assert classify_token("Storm", names) == "en"
    assert classify_token("Sajan", names) == "name"


def test_language_mix_sums_to_100(sample):
    rep, _ = sample
    assert abs(sum(rep["language"]["dialogue_words_pct"].values()) - 100) < 0.5


def test_chillu_and_normalisation():
    zwj = "ന്‍"  # old-style chillu n
    assert normalize_malayalam("അവൻ" if False else "അവന്‍") == "അവൻ"
    assert "ൻ" in normalize_malayalam(zwj)
    assert normalize_malayalam("എൻ്റെ") == "എന്റെ"


def test_legacy_font_warning():
    garbled = "".join(chr(0xE000 + (i % 200)) for i in range(600))
    ex = extract("x.txt", garbled.encode("utf-8"))
    assert any("legacy Malayalam font" in w for w in ex.warnings)


def test_empty_and_bad_types():
    with pytest.raises(ValueError):
        extract("a.txt", b"")
    with pytest.raises(ValueError):
        extract("a.exe", b"hello world" * 50)


def test_unstructured_text_warns():
    rep, _ = analyze("Just a paragraph of prose with no headings at all. " * 20)
    assert any("No scene headings" in w for w in rep["warnings"])


def test_pdf_blank_lines_recovered(tmp_path):
    """Plain PDF text extraction drops blank lines; the extractor must restore them from line gaps."""
    reportlab = pytest.importorskip("reportlab")
    from reportlab.pdfgen import canvas

    p = tmp_path / "s.pdf"
    c = canvas.Canvas(str(p))
    y = 800
    for line in SAMPLE.read_text(encoding="utf-8").split("\n"):
        if y < 40:
            c.showPage()
            y = 800
        c.drawString(40, y, line)  # Latin lines only; Malayalam needs an embedded font
        y -= 13
    c.save()
    ex = extract("s.pdf", p.read_bytes())
    rep, _ = analyze(ex.text, ex.page_count)
    names = {c["name"] for c in rep["characters"]["list"]}
    assert {"SAJAN", "MATHEW", "RENU", "TOUT", "DOCTOR"} <= names
    assert rep["structure"]["scene_count"] == 8
