# ScriptLens: screenplay analysis for Mollywood

One engine, two views: producer coverage (greenlight / pass) and script doctor (rewrite plan).
Measured facts come from rule-based code; the AI reading sits on top and is clearly separated.

## Run
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=...        # optional; without it only the measured tabs work
    python -m uvicorn scriptlens.server:app --port 8765
    # open http://localhost:8765
    python -m pytest tests

## What is measured (no AI)
Scenes, INT/EXT, time of day, locations, speaking characters and dialogue share, per-character language
mix (Malayalam script / Manglish / English), production flags (fights, songs, crowds, water, weather,
vehicles, animals, children, fire, stunts, tobacco/alcohol for censor review), pacing by tenth, talk-heavy
stretches, extraction-quality warnings (legacy Malayalam fonts, scanned PDFs).

## What the AI adds (needs ANTHROPIC_API_KEY)
Map pass over scene chunks, then two reads: producer coverage and script doctor. Every claim cites scene
labels. Market items are hypotheses with a validation step, never figures. Dialect/idiom flags are marked
"confirm with a native-speaking script editor".

## Known limits (v1)
- Heuristics: Manglish word list, production-flag weights, role labels. All uncalibrated; tune on real scripts.
- Scanned PDFs and legacy-font (ML-TT/Karthika) PDFs are detected and warned about, not OCR'd.
- Characters are found from dialogue cues; silent protagonists and name aliases are not merged.
- AI layer is tested with a mocked client only. Not yet run against the live API.
- Scripts are kept in memory (last 20), never written to disk. AI use sends script text to Anthropic.

## Safe-by-default
No GitHub push or hosting deploy is configured. Do those only with explicit approval.
