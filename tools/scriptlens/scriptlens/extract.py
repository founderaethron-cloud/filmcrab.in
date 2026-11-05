"""Text extraction for screenplay files (PDF, DOCX, TXT, Fountain) with Malayalam-aware cleanup."""
from __future__ import annotations

import io
import re
import unicodedata
from dataclasses import dataclass, field

MAX_BYTES = 15 * 1024 * 1024

# Old-style chillu (consonant + virama + ZWJ) -> atomic chillu codepoints.
_CHILLU = {
    "ണ്‍": "ൺ",
    "ന്‍": "ൻ",
    "ര്‍": "ർ",
    "ല്‍": "ൽ",
    "ള്‍": "ൾ",
    "ക്‍": "ൿ",
}
_CHILLU_RE = re.compile("|".join(map(re.escape, _CHILLU)))


@dataclass
class Extracted:
    text: str
    page_count: int | None = None
    source_type: str = "txt"
    warnings: list[str] = field(default_factory=list)


def normalize_malayalam(text: str) -> str:
    """NFC-normalise, convert ZWJ chillus to atomic chillus, tidy spacing."""
    text = unicodedata.normalize("NFC", text)
    text = _CHILLU_RE.sub(lambda m: _CHILLU[m.group(0)], text)
    # Old 'ൻ്റ' spelling and new 'ന്റ' are the same sound; fold to the modern form.
    text = text.replace("ൻ്റ", "ന്റ")
    text = text.replace(" ", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text


def _quality_warnings(text: str) -> list[str]:
    warns: list[str] = []
    n = len(text)
    if n < 200:
        warns.append(
            "Very little text was extracted. If this is a scanned PDF, it needs OCR before analysis."
        )
        return warns
    pua = sum(1 for c in text if 0xE000 <= ord(c) <= 0xF8FF)
    odd = sum(1 for c in text if 0x80 <= ord(c) <= 0xFF)
    ml = sum(1 for c in text if 0x0D00 <= ord(c) <= 0x0D7F)
    letters = sum(1 for c in text if c.isalpha())
    if pua / n > 0.01 or (odd / n > 0.08 and ml / max(letters, 1) < 0.05):
        warns.append(
            "Text looks like it was typed in a legacy Malayalam font (ML-TT / Karthika style) "
            "and did not extract as Unicode. Re-export from the source as Unicode, or run OCR. "
            "Results for Malayalam dialogue will be unreliable until then."
        )
    if "�" in text:
        warns.append("Some characters could not be decoded (replacement characters found).")
    return warns


def _gap_aware_lines(rows_by_page: list[list[dict]]) -> list[str]:
    """Rebuild page text, inserting a blank line wherever the vertical gap is clearly larger than the
    document's normal line spacing. Plain text extraction drops those gaps, and the parser needs them
    to know where a dialogue block ends. Base spacing is a low percentile of all gaps in the file,
    because in a script most gaps are blank-line gaps and a median would hide them."""
    diffs = sorted(
        b["top"] - a["top"] for rows in rows_by_page for a, b in zip(rows, rows[1:]) if b["top"] - a["top"] > 0.5
    )
    base = diffs[max(0, int(len(diffs) * 0.1) - 1)] if diffs else 0
    pages: list[str] = []
    for rows in rows_by_page:
        out: list[str] = []
        for i, r in enumerate(rows):
            if i and base and r["top"] - rows[i - 1]["top"] > base * 1.45:
                out.append("")
            out.append(r["text"])
        pages.append("\n".join(out))
    return pages


def _from_pdf(data: bytes) -> Extracted:
    warnings: list[str] = []
    try:
        import pdfplumber

        rows_by_page = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                rows = [r for r in page.extract_text_lines(layout=False, strip=True) if r["text"].strip()]
                rows_by_page.append(rows)
        pages = _gap_aware_lines(rows_by_page)
    except Exception:
        pages = []
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(data)
        for i in range(len(pdf)):
            tp = pdf[i].get_textpage()
            pages.append(tp.get_text_range())
            tp.close()
        pdf.close()
        warnings.append(
            "Page layout could not be read, so blank lines between speeches were not recovered. "
            "Dialogue and action boundaries may be less accurate."
        )
    text = normalize_malayalam("\n\n".join(pages))
    return Extracted(text=text, page_count=len(pages), source_type="pdf", warnings=warnings)


def _from_docx(data: bytes) -> Extracted:
    import docx

    d = docx.Document(io.BytesIO(data))
    text = normalize_malayalam("\n".join(p.text for p in d.paragraphs))
    return Extracted(text=text, source_type="docx")


def _from_text(data: bytes, kind: str) -> Extracted:
    for enc in ("utf-8-sig", "utf-16"):
        try:
            return Extracted(text=normalize_malayalam(data.decode(enc)), source_type=kind)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return Extracted(
        text=normalize_malayalam(data.decode("utf-8", errors="replace")), source_type=kind
    )


def extract(filename: str, data: bytes) -> Extracted:
    if len(data) > MAX_BYTES:
        raise ValueError("File is larger than 15 MB.")
    if not data:
        raise ValueError("File is empty.")
    name = (filename or "").lower()
    if name.endswith(".pdf") or data[:5] == b"%PDF-":
        ex = _from_pdf(data)
    elif name.endswith(".docx") or data[:2] == b"PK":
        ex = _from_docx(data)
    elif name.endswith((".txt", ".fountain", ".md", ".text")):
        ex = _from_text(data, "fountain" if name.endswith(".fountain") else "txt")
    else:
        raise ValueError("Unsupported file type. Use PDF, DOCX, TXT or Fountain.")
    ex.warnings.extend(_quality_warnings(ex.text))
    return ex
