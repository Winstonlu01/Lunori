# backend/app.py
"""
Lunori API (local-first)

Responsibilities
- Serve the static frontend.
- Transcribe audio (upload or live recording) with Whisper.
- Compute a simple mood score from text using a 6-label emotion model.
- Save/read/delete journal entries (JSON on disk).
- Image captioning + simple tag extraction.
- Persist user ASR model preference (Whisper) across runs.

Storage layout (relative to repo root)
  /frontend/                 static assets for the UI
  /data/audio/               16 kHz mono WAVs for playback (or fallback to raw)
  /data/raw_audio/           the original container from recording (.webm/.ogg/...)
  /data/entries/             one JSON per saved journal entry
  /data/sessions/<id>/       rolling live container per active recording session
  /data/images/              uploaded images for an entry
  config.json                persisted app config (e.g., active Whisper model)

Notes
- Live transcription uses a single "rolling" container per session to avoid
  header truncation/end-clipping problems common with incremental WebM chunks.
- For playback it is prefered to use 16 kHz mono WAV (smaller, Whisper-friendly). 
  If ffmpeg fails, we fall back to the raw container.
- All data stays local to the machine.
"""

from pathlib import Path
from datetime import datetime
import json
import subprocess
import threading
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from urllib.parse import unquote

from transformers import pipeline
from PIL import Image
import whisper  # Whisper STT

app = FastAPI(title="Lunori API")

# CORS: permissive for local development. 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths and data directories
ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT / "frontend"

AUDIO_DIR = ROOT / "data" / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

RAW_AUDIO_DIR = ROOT / "data" / "raw_audio"
RAW_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

ENTRIES_DIR = ROOT / "data" / "entries"
ENTRIES_DIR.mkdir(parents=True, exist_ok=True)

SESSIONS_DIR = ROOT / "data" / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

IMAGES_DIR = ROOT / "data" / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Persisted configuration (currently: the selected Whisper model)
CONFIG_PATH = ROOT / "config.json"
DEFAULT_CONFIG = {"whisper_model": "small.en"}


def _read_config():
    # Return config dict from disk, or defaults if missing/invalid.
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return DEFAULT_CONFIG.copy()


def _write_config(obj: dict):
    # Best-effort write of config to disk (ignore failures).
    try:
        CONFIG_PATH.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


# Ensure we have a config file on first run
if not CONFIG_PATH.exists():
    _write_config(DEFAULT_CONFIG)

# Static frontend
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def root_page():
    # Serve the main UI.
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
def health():
    # Simple liveness probe.
    return {"status": "ok"}


# Audio formats we accept for upload/live
ALLOWED_EXTS = {".wav", ".mp3", ".webm", ".ogg"}

# Whisper model management (runtime switch + persistence). Lock prevents concurrent swaps.
ALLOWED_WHISPER_NAMES = {"tiny", "base.en", "small.en", "medium.en"}
WHISPER_LOCK = threading.Lock()

_cfg = _read_config()
_default_name = _cfg.get("whisper_model", DEFAULT_CONFIG["whisper_model"]) or DEFAULT_CONFIG["whisper_model"]
if _default_name not in ALLOWED_WHISPER_NAMES:
    _default_name = DEFAULT_CONFIG["whisper_model"]

WHISPER_NAME = _default_name
WHISPER_MODEL = whisper.load_model(WHISPER_NAME)


def _set_whisper_model(name: str):
    # Switch the active Whisper model and persist the choice.
    global WHISPER_MODEL, WHISPER_NAME
    if name not in ALLOWED_WHISPER_NAMES:
        raise HTTPException(status_code=400, detail=f"Unsupported model '{name}'. Allowed: {sorted(ALLOWED_WHISPER_NAMES)}")
    with WHISPER_LOCK:
        WHISPER_NAME = name
        WHISPER_MODEL = whisper.load_model(WHISPER_NAME)
    cfg = _read_config()
    cfg["whisper_model"] = name
    _write_config(cfg)


class WhisperModelIn(BaseModel):
    name: str


@app.get("/config/whisper_model")
def get_whisper_model():
    # Return the active Whisper model name.
    return {"ok": True, "name": WHISPER_NAME}


@app.post("/config/whisper_model")
def set_whisper_model(payload: WhisperModelIn):
    # Update the active Whisper model name (one of ALLOWED_WHISPER_NAMES).
    _set_whisper_model(payload.name)
    return {"ok": True, "name": WHISPER_NAME}


# Emotion model (6 labels). First call downloads weights and caches them locally.
EMO_PIPE = pipeline(
    "text-classification",
    model="bhadresh-savani/distilbert-base-uncased-emotion",
    return_all_scores=True,
)

# Map each label to a valence; final mood aggregates normalized probabilities
# into a single score on [-100, +100].
EMO_VALENCE = {
    "joy": +1.0,
    "love": +0.9,
    "surprise": +0.2,   # assume mostly non-negative
    "sadness": -1.0,
    "anger": -0.9,
    "fear": -0.8,
}


def analyze_emotions(text: str):
    # Return overall mood and per-label scores; only first ~2,000 chars considered.
    if not text.strip():
        return {"mood": 0, "top_emotions": [], "all": {}}

    scores = EMO_PIPE(text[:2000])[0]  # list of {label, score}

    total = sum(s["score"] for s in scores) or 1.0
    norm = [{"label": s["label"], "score": s["score"] / total} for s in scores]

    mood_float = sum(s["score"] * EMO_VALENCE.get(s["label"], 0.0) for s in norm)
    mood_100 = int(round(mood_float * 100))

    top3 = sorted(norm, key=lambda x: x["score"], reverse=True)[:3]
    all_map = {s["label"]: round(s["score"], 4) for s in norm}

    return {"mood": mood_100, "top_emotions": top3, "all": all_map}


@app.post("/transcribe/upload")
async def transcribe_upload(file: UploadFile = File(...)):
    # Transcribe a single audio file upload; save raw file then run Whisper.
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(status_code=400, detail="Please upload a supported audio format")

    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    safe_name = f"{ts}{ext}"
    dest_path = AUDIO_DIR / safe_name

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")
    dest_path.write_bytes(contents)

    with WHISPER_LOCK:
        result = WHISPER_MODEL.transcribe(str(dest_path), fp16=False)
        active_name = WHISPER_NAME

    transcript_text = (result.get("text") or "").strip()
    segments = result.get("segments") or []
    language = result.get("language", "unknown")

    nice_segments = [
        {"start": float(s.get("start", 0.0)), "end": float(s.get("end", 0.0)), "text": (s.get("text") or "").strip()}
        for s in segments
    ]

    return {
        "ok": True,
        "filename": safe_name,
        "path": str(dest_path),
        "note": f"Transcribed with Whisper ({active_name}).",
        "language": language,
        "transcript": transcript_text,
        "segments": nice_segments,
        "size_bytes": len(contents),
    }


class ImageMeta(BaseModel):
    filename: str
    caption: str | None = None
    tags: list[str] | None = None


class SaveEntryIn(BaseModel):
    filename: str
    transcript: str
    images: list[ImageMeta] | None = None  # list of images with captions/tags


@app.post("/entries/save")
def save_entry(payload: SaveEntryIn):
    # Persist a journal entry JSON with audio link, transcript, mood, and images.
    audio_path = (AUDIO_DIR / payload.filename).resolve()
    if not audio_path.exists():
        raise HTTPException(status_code=400, detail="Audio file not found")

    text = (payload.transcript or "").strip()
    words = len(text.split()) if text else 0

    emo = analyze_emotions(text)

    created_at = datetime.now().isoformat(timespec="seconds")
    entry_id = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    images_out = []
    if payload.images:
        for im in payload.images:
            images_out.append({
                "filename": Path(im.filename).name,
                "caption": (im.caption or "").strip() or None,
                "tags": list(dict.fromkeys([t.strip().lower() for t in (im.tags or []) if t and t.strip()])) or None,
            })

    entry = {
        "id": entry_id,
        "created_at": created_at,
        "audio_filename": payload.filename,
        "audio_path": str(audio_path),
        "transcript": text,
        "words": words,
        "mood": emo["mood"],
        "emotions_top3": emo["top_emotions"],
        "emotions_all": emo["all"],
        "images": images_out or None,
    }

    out_path = ENTRIES_DIR / f"{entry_id}.json"
    out_path.write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"ok": True, "id": entry_id, "path": str(out_path)}


@app.get("/entries")
def list_entries():
    # Return a lightweight list of saved entries (sufficient for a history view).
    items = []
    for p in ENTRIES_DIR.glob("*.json"):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            items.append(
                {
                    "id": obj.get("id"),
                    "created_at": obj.get("created_at"),
                    "audio_filename": obj.get("audio_filename"),
                    "words": obj.get("words", 0),
                    "mood": obj.get("mood"),
                    "emotions_top3": obj.get("emotions_top3"),
                    "image_count": len(obj.get("images") or []),
                }
            )
        except Exception:
            continue
    items.sort(key=lambda x: (x.get("created_at") or x.get("id") or ""), reverse=True)
    return {"ok": True, "items": items}


@app.get("/entries/{entry_id}")
def get_entry(entry_id: str):
    # Return the full JSON for a single entry.
    p = ENTRIES_DIR / f"{entry_id}.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="Entry not found")
    return json.loads(p.read_text(encoding="utf-8"))


@app.delete("/entries/{entry_id}")
def delete_entry(entry_id: str):
    # Delete an entry and associated audio. JSON first, then audio/raw by stem.
    p = ENTRIES_DIR / f"{entry_id}.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="Entry not found")

    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        obj = {}

    audio_filename = obj.get("audio_filename")
    deleted = {"json": False, "audio": False, "raw": False}

    try:
        p.unlink()
        deleted["json"] = True
    except FileNotFoundError:
        pass

    if audio_filename:
        audio_path = AUDIO_DIR / Path(audio_filename).name
        try:
            audio_path.unlink()
            deleted["audio"] = True
        except FileNotFoundError:
            pass

        stem = Path(audio_filename).stem
        for ext in (".webm", ".ogg", ".mp3", ".wav"):
            raw_candidate = RAW_AUDIO_DIR / f"{stem}{ext}"
            if raw_candidate.exists():
                try:
                    raw_candidate.unlink()
                    deleted["raw"] = True
                except FileNotFoundError:
                    pass

    return {"ok": True, "deleted": deleted}


@app.post("/transcribe/chunk")
async def transcribe_chunk(
    session_id: str = Form(...),
    index: int = Form(...),  # retained for debugging/telemetry
    file: UploadFile = File(...),
):
    # Live transcription: keep a single rolling container to avoid end clipping.
    sid = session_id.strip()
    if not sid:
        raise HTTPException(status_code=400, detail="Invalid session_id")

    ext = Path(file.filename).suffix.lower() or ".webm"
    if ext not in {".webm", ".ogg", ".wav", ".mp3"}:
        ext = ".webm"

    sdir = SESSIONS_DIR / sid
    sdir.mkdir(parents=True, exist_ok=True)

    latest_path = sdir / f"latest{ext}"

    blob = await file.read()
    if not blob:
        raise HTTPException(status_code=400, detail="Empty chunk")

    latest_path.write_bytes(blob)

    # Best-effort live preview.
    try:
        with WHISPER_LOCK:
            result = WHISPER_MODEL.transcribe(str(latest_path), fp16=False, language="en", temperature=0.0)
        text = (result.get("text") or "").strip()
        segments = result.get("segments") or []
        nice_segments = [
            {"start": float(s.get("start", 0.0)), "end": float(s.get("end", 0.0)), "text": (s.get("text") or "").strip()}
            for s in segments
        ]
    except Exception as e:
        print("live preview transcribe failed:", latest_path, e)
        text, nice_segments = "", []

    return {"ok": True, "session_id": sid, "index": index, "transcript": text, "segments": nice_segments}


class FinalizeIn(BaseModel):
    session_id: str


def _ffmpeg_to_wav(src: Path, dst: Path):
    # Convert source container to 16 kHz mono WAV (good default for playback/ASR).
    cmd = ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", "16000", str(dst)]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


@app.post("/transcribe/finalize")
async def transcribe_finalize(payload: FinalizeIn):
    # Finalize a live session using rolling container; save raw + WAV (if possible).
    sid = payload.session_id.strip()
    sdir = SESSIONS_DIR / sid
    if not sdir.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    latest = next(sdir.glob("latest.*"), None)
    if latest is None:
        chunks = sorted(sdir.glob("chunk-*"))
        if not chunks:
            raise HTTPException(status_code=400, detail="No audio found for this session")
        latest = chunks[-1]

    with WHISPER_LOCK:
        result = WHISPER_MODEL.transcribe(str(latest), fp16=False, language="en", temperature=0.0)

    final_text = (result.get("text") or "").strip()
    words = len(final_text.split())

    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    ext = latest.suffix.lower() or ".webm"
    if ext not in {".webm", ".ogg", ".mp3", ".wav"}:
        ext = ".webm"

    raw_audio_name = f"{ts}{ext}"
    raw_audio_path = RAW_AUDIO_DIR / raw_audio_name
    raw_audio_path.write_bytes(latest.read_bytes())

    audio_wav_name = f"{ts}.wav"
    audio_wav_path = AUDIO_DIR / audio_wav_name

    converted = False
    try:
        _ffmpeg_to_wav(raw_audio_path, audio_wav_path)
        converted = True
    except subprocess.CalledProcessError as e:
        print("ffmpeg to wav failed:", e)

    chosen_filename = audio_wav_name if converted else raw_audio_name
    chosen_path = str(audio_wav_path if converted else raw_audio_path)

    return {
        "ok": True,
        "session_id": sid,
        "final_transcript": final_text,
        "words": words,
        "audio_filename": chosen_filename,
        "audio_path": chosen_path,
        "raw_audio_filename": raw_audio_name,
        "raw_audio_path": str(raw_audio_path),
        "note": "Rolled up to a single container; WAV returned when available.",
    }


@app.get("/audio/{filename}")
def serve_audio(filename: str):
    # Return a saved audio file (WAV preferred; raw container as fallback).
    name = Path(unquote(filename)).name  # basic sanitization
    candidate_paths = [
        AUDIO_DIR / name,      # WAV/MP3
        RAW_AUDIO_DIR / name,  # e.g., WEBM/OGG
    ]
    for p in candidate_paths:
        if p.exists():
            return FileResponse(p)
    raise HTTPException(status_code=404, detail="Audio not found")


# Image captioning and tag extraction.
# BLIP "base" is a reasonable CPU default; upgrade to "large" for GPU setups.
try:
    CAPTION_PIPE = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
except Exception as e:
    CAPTION_PIPE = None
    print("Warning: BLIP caption model failed to load:", e)

IMG_ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _caption_image_bytes(b: bytes) -> str:
    # Return a one-line caption from image bytes; empty string if unavailable.
    if not CAPTION_PIPE:
        return ""
    try:
        img = Image.open(BytesIO(b)).convert("RGB")
        out = CAPTION_PIPE(img)
        if isinstance(out, list) and out:
            return (out[0].get("generated_text") or "").strip()
    except Exception as e:
        print("caption error:", e)
    return ""


def _tags_from_caption(caption: str) -> list[str]:
    # Extract simple lowercase keywords from a caption (stopwords removed), capped to a small set.
    import re
    cap = (caption or "").lower()
    toks = [t for t in re.split(r"[^a-z0-9]+", cap) if t]
    stop = {"a","an","the","of","on","in","and","with","to","at","for","from","is","are","this","that","it","its","by","near"}
    toks = [t for t in toks if t not in stop and len(t) >= 3]
    seen = set()
    out = []
    for t in toks:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out[:12]


@app.post("/images/upload")
async def images_upload(file: UploadFile = File(...)):
    # Upload/store one image and return caption + tags for use in an entry.
    ext = Path(file.filename).suffix.lower()
    if ext not in IMG_ALLOWED_EXTS:
        raise HTTPException(status_code=400, detail="Please upload .jpg, .jpeg, .png, or .webp")

    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    safe_name = f"{ts}{ext}"
    dest_path = IMAGES_DIR / safe_name

    blob = await file.read()
    if not blob:
        raise HTTPException(status_code=400, detail="Empty image")
    dest_path.write_bytes(blob)

    caption = _caption_image_bytes(blob)
    tags = _tags_from_caption(caption)

    return {
        "ok": True,
        "filename": safe_name,
        "path": str(dest_path),
        "url": f"/images/{safe_name}",
        "caption": caption,
        "tags": tags,
    }


@app.get("/images/{filename}")
def serve_image(filename: str):
    # Return an image previously stored in /data/images.
    name = Path(unquote(filename)).name
    p = IMAGES_DIR / name
    if p.exists():
        return FileResponse(p)
    raise HTTPException(status_code=404, detail="Image not found")
