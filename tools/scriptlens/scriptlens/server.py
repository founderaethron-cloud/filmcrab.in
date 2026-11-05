"""FastAPI app. Scripts are held in memory only (never written to disk) and evicted oldest-first."""
from __future__ import annotations

import os
import secrets
import threading
from collections import OrderedDict
from pathlib import Path

from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from . import ai
from .engine import Scene, analyze
from .extract import extract

STATIC = Path(__file__).resolve().parent.parent / "static"
MAX_STORED = 20

app = FastAPI(title="ScriptLens", docs_url=None, redoc_url=None)
_lock = threading.Lock()
_store: "OrderedDict[str, dict]" = OrderedDict()  # id -> {report, scenes, ai}


def _put(report: dict, scenes: list[Scene]) -> str:
    sid = secrets.token_urlsafe(12)
    with _lock:
        _store[sid] = {"report": report, "scenes": scenes, "ai": {"status": "idle"}}
        while len(_store) > MAX_STORED:
            _store.popitem(last=False)
    return sid


def _get(sid: str) -> dict:
    with _lock:
        item = _store.get(sid)
    if not item:
        raise HTTPException(404, "Report not found. It may have expired; upload the script again.")
    return item


@app.get("/api/config")
def config() -> dict:
    return {"ai_enabled": bool(os.environ.get("ANTHROPIC_API_KEY")), "model": ai.MODEL, "max_ai_words": ai.MAX_WORDS}


@app.post("/api/analyze")
async def analyze_upload(file: UploadFile = File(...)) -> JSONResponse:
    data = await file.read(15 * 1024 * 1024 + 1)
    try:
        ex = await run_in_threadpool(extract, file.filename or "", data)
        report, scenes = await run_in_threadpool(
            analyze, ex.text, ex.page_count, file.filename or "", ex.source_type, ex.warnings
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:  # corrupt PDFs, odd DOCX, etc.
        raise HTTPException(422, f"Could not read this file: {type(e).__name__}")
    sid = _put(report, scenes)
    return JSONResponse({"id": sid, "report": report})


@app.get("/api/report/{sid}")
def get_report(sid: str) -> dict:
    item = _get(sid)
    return {"id": sid, "report": item["report"], "ai": item["ai"]}


@app.post("/api/ai/{sid}/start")
def start_ai(sid: str, body: dict = Body(...)) -> dict:
    item = _get(sid)
    lens = body.get("lens", "both")
    if lens not in ("coverage", "doctor", "both"):
        raise HTTPException(400, "lens must be coverage, doctor or both")
    if body.get("consent") is not True:
        raise HTTPException(400, "Consent is required: the script text is sent to the Anthropic API for AI analysis.")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(503, "AI analysis is not configured on this server (ANTHROPIC_API_KEY missing).")
    if item["ai"].get("status") == "running":
        raise HTTPException(409, "AI analysis is already running for this script.")
    if ai.total_words(item["scenes"]) > ai.MAX_WORDS:
        raise HTTPException(400, f"Script is over {ai.MAX_WORDS:,} words; split it before running AI analysis.")
    state = {"status": "running", "message": "Starting", "progress": 0.0, "lens": lens}
    item["ai"] = state

    def progress(msg: str, frac: float) -> None:
        state["message"], state["progress"] = msg, frac

    def job() -> None:
        try:
            state["result"] = ai.run(item["scenes"], item["report"], lens, progress)
            state["status"] = "done"
        except Exception as e:
            state["status"], state["error"] = "error", f"{type(e).__name__}: {str(e)[:300]}"

    threading.Thread(target=job, daemon=True).start()
    return {"status": "running"}


@app.get("/api/ai/{sid}")
def ai_status(sid: str) -> dict:
    return _get(sid)["ai"]


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
