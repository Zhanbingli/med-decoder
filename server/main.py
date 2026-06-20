#!/usr/bin/env python3
"""
CardioVoice API — FastAPI backend
=================================

Product-grade backend for the React frontend. Reuses the whole local pipeline
(MedASR + grounded LLM note generation + correction + SQLite store + export).
Everything runs locally; this server only listens on localhost.

Run:
    conda activate medgemma
    ollama serve                       # another terminal
    uvicorn server.main:app --port 8000 --reload
"""

import json
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))

from unified_model_manager import UnifiedModelManager, create_patient_info  # noqa: E402
from store import RecordStore, NOTE_FIELDS  # noqa: E402
import export  # noqa: E402

STATE: Dict[str, object] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    mgr = UnifiedModelManager()
    mgr.load_all(verbose=False)
    STATE["manager"] = mgr
    STATE["store"] = RecordStore()
    yield


app = FastAPI(title="CardioVoice API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


def mgr() -> UnifiedModelManager:
    return STATE["manager"]  # type: ignore


def store() -> RecordStore:
    return STATE["store"]  # type: ignore


# --------------------------------------------------------------------- models
class Patient(BaseModel):
    name: str = "Patient"
    age: int = 0
    gender: str = "unknown"


class CorrectReq(BaseModel):
    text: str


class GenerateReq(BaseModel):
    transcript: str
    patient: Patient = Patient()
    template: str = "cardiology"


class SaveReq(BaseModel):
    record_id: Optional[int] = None
    encounter_id: Optional[int] = None
    patient: Patient = Patient()
    transcript: str = ""
    template: str = "cardiology"
    fields: Dict[str, str] = {}
    status: str = "draft"
    verified_by: Optional[str] = None


class ExportReq(BaseModel):
    fields: Dict[str, str] = {}
    patient: Patient = Patient()
    template: str = "cardiology"
    fmt: str = "pdf"


# ------------------------------------------------------------------ endpoints
@app.get("/api/status")
def status():
    m = mgr()
    return {
        "asr": m.medasr.is_available(),
        "llm": m.medgemma.is_available(),
        "model": m.medgemma.model_name,
        "fields": NOTE_FIELDS,
    }


@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...)):
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    data = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(data)
        path = f.name
    result = mgr().transcribe(path)
    return {
        "text": result.text,
        "confidence": result.confidence,
        "duration": result.audio_duration,
        "words": [[w, round(c, 4)] for w, c in result.word_confidences],
    }


@app.post("/api/correct")
def correct(req: CorrectReq):
    return {"text": mgr().medgemma.correct_transcription(req.text)}


@app.post("/api/generate")
def generate(req: GenerateReq):
    """Stream the structured note as Server-Sent Events (token + done)."""
    patient = create_patient_info(req.patient.name, req.patient.age, req.patient.gender)

    def sse():
        for kind, payload in mgr().medgemma.stream_structured_note(
            req.transcript, patient, req.template
        ):
            if kind == "token":
                yield f"event: token\ndata: {json.dumps({'t': payload})}\n\n"
            else:
                yield f"event: done\ndata: {json.dumps(payload)}\n\n"

    if not mgr().medgemma.is_available():
        return JSONResponse({"error": "LLM unavailable (start `ollama serve`)"}, 503)
    return StreamingResponse(sse(), media_type="text/event-stream")


@app.get("/api/encounters")
def list_encounters():
    return store().list_encounters()


@app.get("/api/encounters/{eid}")
def get_encounter(eid: int):
    enc = store().get_encounter(eid)
    if not enc:
        return JSONResponse({"error": "not found"}, 404)
    return {"encounter": enc, "record": store().get_record_for_encounter(eid)}


@app.post("/api/save")
def save(req: SaveReq):
    s = store()
    eid = req.encounter_id
    if eid is None:
        eid = s.create_encounter(
            patient=req.patient.model_dump(), transcript=req.transcript,
            template=req.template,
        )
    rid = s.save_record(
        eid, req.fields, model=mgr().medgemma.model_name,
        status=req.status, record_id=req.record_id,
    )
    if req.status == "verified":
        s.verify_record(rid, req.verified_by or "physician")
    return {"encounter_id": eid, "record_id": rid}


@app.post("/api/export")
def export_note(req: ExportReq):
    meta = {"template": req.template}
    p = req.patient.model_dump()
    if req.fmt == "txt":
        return Response(export.format_text(req.fields, p, meta),
                        media_type="text/plain")
    if req.fmt == "md":
        return Response(export.format_markdown(req.fields, p, meta),
                        media_type="text/markdown")
    return Response(export.format_pdf(req.fields, p, meta),
                    media_type="application/pdf",
                    headers={"Content-Disposition": "attachment; filename=note.pdf"})
