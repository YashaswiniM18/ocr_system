import os
import io
import uuid
import time
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from image_preprocessing import load_image
from ocr_engine import run_ocr, is_ocr_ready, get_cache_stats
from doc_classifier import classify_document

from image_assets import (
    extract_aadhaar_face,
    extract_pan_face,
    extract_pan_signature,
    extract_dl_face,
    extract_dl_signature,
    extract_marksheet_face,
)
from field_extractors import aadhaar, pan, dl, marksheet

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("ocr_api")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Real-Time OCR Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For debugging, allow all while testing connectivity
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "90"))   # seconds

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# ── Simple in-process metrics ─────────────────────────────────────────────────
_metrics = {
    "requests_total": 0,
    "requests_error": 0,
    "total_latency_ms": 0.0,
}


# ── Startup warm-up ───────────────────────────────────────────────────────────
async def startup_event():
    """Warm up OCR engine. Called by FastAPI handler and directly by tests."""
    from ocr_engine import get_ocr
    loop = asyncio.get_event_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=4))
    await loop.run_in_executor(None, get_ocr)
    logger.info("OCR Engine warmed up and ready.")


@app.on_event("startup")
async def _startup_handler():
    await startup_event()


# ── Thread-pool helper ────────────────────────────────────────────────────────
def run_in_thread(func, *args):
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(None, func, *args)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    from fastapi.responses import Response
    return Response(status_code=204)


@app.get("/health")
def health():
    ready = is_ocr_ready()
    cache = get_cache_stats()
    return {
        "status": "ok",
        "ocr_ready": ready,
        "message": "OCR system is ready" if ready else "OCR system is warming up...",
        "version": os.getenv("APP_VERSION", "1.0.0"),
        "cache": cache,
    }


@app.get("/metrics")
def metrics():
    total = _metrics["requests_total"]
    avg_ms = (_metrics["total_latency_ms"] / total) if total > 0 else 0.0
    return {
        "requests_total": total,
        "requests_error": _metrics["requests_error"],
        "avg_latency_ms": round(avg_ms, 1),
        "cache": get_cache_stats(),
    }


@app.get("/")
def root():
    return {"message": "OCR API is running"}


# ── Core processing (inner function, wrapped with timeout) ────────────────────
async def _process_upload(file: UploadFile) -> dict:
    # ── Read file into memory (no disk I/O) ──────────────────────────────────
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return {"detail": "Invalid image format or unsupported file type"}

    # ── OCR (CPU-bound → thread) ──────────────────────────────────────────────
    texts, raw_ocr_data = await run_in_thread(run_ocr, image)

    # ── Classify ──────────────────────────────────────────────────────────────
    doc_type = classify_document(texts)
    job_id = uuid.uuid4().hex

    # ── Parallel: field extraction + asset extraction ─────────────────────────
    async def extract_fields_task():
        try:
            if doc_type == "Aadhaar":
                return await run_in_thread(aadhaar.extract_aadhaar_fields, texts)
            elif doc_type == "PAN":
                return await run_in_thread(pan.extract, texts, raw_ocr_data)
            elif doc_type == "Driving License":
                return await run_in_thread(dl.extract, texts, image)
            elif doc_type == "Marksheet":
                return await run_in_thread(marksheet.extract, texts, image, raw_ocr_data)
        except Exception as e:
            logger.error(f"Field extraction failed for doc_type={doc_type}: {e}", exc_info=True)
        return {}

    async def extract_assets_task():
        async def run_asset(func, *args):
            try:
                return await run_in_thread(func, *args)
            except Exception as e:
                logger.warning(f"Asset extraction skipped ({func.__name__}): {e}")
                return None

        face_out = os.path.join(OUTPUT_DIR, f"face_{job_id}.jpg")
        sig_out  = os.path.join(OUTPUT_DIR, f"signature_{job_id}.jpg")

        if doc_type == "Aadhaar":
            results = await asyncio.gather(run_asset(extract_aadhaar_face, image, face_out))
            return results[0], None
        elif doc_type == "PAN":
            results = await asyncio.gather(
                run_asset(extract_pan_face, image, face_out),
                run_asset(extract_pan_signature, image, sig_out),
            )
            return results[0], results[1]
        elif doc_type == "Driving License":
            results = await asyncio.gather(
                run_asset(extract_dl_face, image, face_out),
                run_asset(extract_dl_signature, image, sig_out),
            )
            return results[0], results[1]
        elif doc_type == "Marksheet":
            results = await asyncio.gather(run_asset(extract_marksheet_face, image, face_out))
            return results[0], None
        return None, None

    data, (face_path, signature_path) = await asyncio.gather(
        extract_fields_task(),
        extract_assets_task(),
    )

    # ── VTU marksheet: no face ────────────────────────────────────────────────
    if doc_type == "Marksheet" and isinstance(data, dict):
        if data.get("university") == "VISVESVARAYA TECHNOLOGICAL UNIVERSITY":
            if face_path and os.path.exists(face_path):
                try:
                    os.remove(face_path)
                except Exception:
                    pass
                face_path = None

    return {
        "document_type": doc_type,
        "extracted_fields": data,
        "face_image": face_path,
        "signature_image": signature_path,
        "job_id": job_id,
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    _metrics["requests_total"] += 1
    t0 = time.perf_counter()

    try:
        result = await asyncio.wait_for(_process_upload(file), timeout=REQUEST_TIMEOUT)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        _metrics["total_latency_ms"] += elapsed_ms
        logger.info(
            f"doc_type={result.get('document_type')} "
            f"job_id={result.get('job_id')} "
            f"latency_ms={elapsed_ms:.0f}"
        )
        return result

    except asyncio.TimeoutError:
        _metrics["requests_error"] += 1
        msg = f"Request timed out after {REQUEST_TIMEOUT}s"
        logger.error(msg)
        return JSONResponse(
            {"detail": msg},
            status_code=408,
        )
    except Exception as exc:
        _metrics["requests_error"] += 1
        logger.exception(f"Unhandled error during /upload: {exc}")
        return JSONResponse({"detail": "Internal server error"}, status_code=500)
