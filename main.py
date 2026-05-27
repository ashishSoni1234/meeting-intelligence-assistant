"""
main.py

FastAPI backend for the Meeting Intelligence Assistant.
Exposes REST endpoints for file processing, Q&A, summarization,
and health checks. Runs on port 8000.
"""

import os
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import torch
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from loguru import logger
from pydantic import BaseModel, Field

# Load environment variables before importing project modules
load_dotenv()

from core.omni_model import get_omni_model
from pipeline.ingestion import IngestionPipeline, IngestionResult
from pipeline.indexer import MeetingIndexer, get_indexer
from pipeline.qa_engine import QAEngine, get_qa_engine
from utils.formatter import format_processing_result


# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_AUDIO = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}
SUPPORTED_VIDEO = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
SUPPORTED_PDF = {".pdf"}

# ─────────────────────────────────────────────────────────────────
# Application state
# ─────────────────────────────────────────────────────────────────
class AppState:
    """Shared mutable application state."""

    def __init__(self):
        """Initialize with defaults."""
        self.ingestion_result: Optional[IngestionResult] = None
        self.is_processing: bool = False
        self.processing_start_time: float = 0.0
        self.pipeline: Optional[IngestionPipeline] = None
        self.indexer: Optional[MeetingIndexer] = None
        self.qa_engine: Optional[QAEngine] = None


app_state = AppState()


# ─────────────────────────────────────────────────────────────────
# Lifespan: startup / shutdown
# ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared resources on startup; clean up on shutdown."""
    logger.info("Starting Meeting Intelligence API...")

    # Initialize singletons
    app_state.indexer = get_indexer()
    app_state.pipeline = IngestionPipeline()
    app_state.qa_engine = get_qa_engine()

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
    else:
        logger.info("GPU not available — running on CPU")

    logger.info(f"API ready at http://{API_HOST}:{API_PORT}")
    yield

    # Shutdown
    logger.info("Shutting down Meeting Intelligence API")
    model = get_omni_model()
    model.unload()


# ─────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Meeting Intelligence Assistant",
    description=(
        "Cross-modal Q&A engine for meetings. "
        "Upload PDF slides, audio, and video to ask questions "
        "grounded in timestamps, slides, and speaker attribution."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────────────────────────
class AskRequest(BaseModel):
    """Request body for the /ask endpoint."""

    question: str = Field(..., min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    include_images: bool = Field(default=True)


class AskResponse(BaseModel):
    """Response body for the /ask endpoint."""

    question: str
    answer: str
    grounding: list[dict]
    slide_image_paths: list[str]
    frame_image_paths: list[str]
    model_used: str
    inference_time_s: float
    retrieval_time_s: float
    total_time_s: float
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Response body for /health endpoint."""

    status: str
    cuda_available: bool
    gpu_name: Optional[str]
    vram_total_gb: Optional[float]
    vram_free_gb: Optional[float]
    model_mode: str
    indexed_documents: int
    is_processing: bool


class SummaryResponse(BaseModel):
    """Response body for /summary endpoint."""

    summary_text: str
    action_items: list[str]
    key_decisions: list[str]
    topic_timeline: list[dict]
    model_used: str
    generation_time_s: float


# ─────────────────────────────────────────────────────────────────
# Helper: save uploaded file
# ─────────────────────────────────────────────────────────────────
async def save_upload(upload: UploadFile, allowed_exts: set[str]) -> Path:
    """
    Save an uploaded file to UPLOAD_DIR after validating its extension.

    Args:
        upload: FastAPI UploadFile object.
        allowed_exts: Set of permitted file extensions (e.g. {'.pdf'}).

    Returns:
        Path to the saved file.

    Raises:
        HTTPException 400: If file extension is not allowed.
    """
    suffix = Path(upload.filename).suffix.lower()
    if suffix not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid file type: {suffix}. "
                f"Allowed: {', '.join(sorted(allowed_exts))}"
            ),
        )

    dest = UPLOAD_DIR / upload.filename
    content = await upload.read()
    dest.write_bytes(content)
    logger.info(f"Saved upload: {dest} ({len(content) / 1e6:.1f} MB)")
    return dest


# ─────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Return API health status including GPU info and model state.

    Returns:
        HealthResponse with system and model status.
    """
    cuda_avail = torch.cuda.is_available()
    gpu_name = None
    vram_total = None
    vram_free = None

    if cuda_avail:
        gpu_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        vram_total = round(props.total_memory / 1e9, 2)
        free, _ = torch.cuda.mem_get_info()
        vram_free = round(free / 1e9, 2)

    model = get_omni_model()
    status = model.get_status()

    indexed = app_state.indexer.get_document_count() if app_state.indexer else 0

    return HealthResponse(
        status="ok",
        cuda_available=cuda_avail,
        gpu_name=gpu_name,
        vram_total_gb=vram_total,
        vram_free_gb=vram_free,
        model_mode=status["mode"],
        indexed_documents=indexed,
        is_processing=app_state.is_processing,
    )


@app.post("/process")
async def process_files(
    pdf_file: Optional[UploadFile] = File(default=None),
    audio_file: Optional[UploadFile] = File(default=None),
    video_file: Optional[UploadFile] = File(default=None),
    enable_diarization: bool = Form(default=True),
):
    """
    Upload and process PDF, audio, and/or video files.

    Runs the full ingestion pipeline in parallel across all uploaded files,
    then indexes the results into ChromaDB.

    Args:
        pdf_file: Optional PDF presentation file.
        audio_file: Optional audio file (MP3/WAV/M4A).
        video_file: Optional video file (MP4/MOV).
        enable_diarization: Whether to attempt speaker diarization.

    Returns:
        JSON summary of processing results.

    Raises:
        HTTPException 400: If no files uploaded or file format invalid.
        HTTPException 409: If processing is already in progress.
        HTTPException 500: If processing fails unexpectedly.
    """
    if not any([pdf_file, audio_file, video_file]):
        raise HTTPException(
            status_code=400,
            detail="No files uploaded. Please provide at least one file.",
        )

    if app_state.is_processing:
        raise HTTPException(
            status_code=409,
            detail="Processing already in progress. Please wait.",
        )

    app_state.is_processing = True
    app_state.processing_start_time = time.time()

    try:
        # Save uploaded files
        pdf_path = None
        audio_path = None
        video_path = None

        if pdf_file and pdf_file.filename:
            pdf_path = str(await save_upload(pdf_file, SUPPORTED_PDF))
        if audio_file and audio_file.filename:
            audio_path = str(await save_upload(audio_file, SUPPORTED_AUDIO))
        if video_file and video_file.filename:
            video_path = str(await save_upload(video_file, SUPPORTED_VIDEO))

        if not any([pdf_path, audio_path, video_path]):
            raise HTTPException(
                status_code=400,
                detail="No valid files provided.",
            )

        # Run ingestion pipeline
        logger.info(
            f"Starting ingestion | pdf={pdf_path} | "
            f"audio={audio_path} | video={video_path}"
        )
        result = app_state.pipeline.process(
            pdf_path=pdf_path,
            audio_path=audio_path,
            video_path=video_path,
            enable_diarization=enable_diarization,
        )
        app_state.ingestion_result = result

        # Index into ChromaDB
        if result.timeline:
            n_indexed = app_state.indexer.index_ingestion_result(result)
            logger.info(f"Indexed {n_indexed} documents")
        else:
            logger.warning("No timeline entries produced — nothing to index")

        elapsed = time.time() - app_state.processing_start_time
        summary = result.to_summary_dict()
        summary["status"] = "success"
        summary["processing_seconds"] = round(elapsed, 2)

        return JSONResponse(content=summary)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Processing failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)[:300]}",
        )
    finally:
        app_state.is_processing = False


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Answer a question about the meeting using cross-modal reasoning.

    Retrieves relevant content from ChromaDB, passes it with optional
    images to Qwen2.5-Omni, and returns a grounded answer.

    Args:
        request: AskRequest with question text and parameters.

    Returns:
        AskResponse with grounded answer, citations, and image paths.

    Raises:
        HTTPException 400: If question is empty.
        HTTPException 503: If no meeting data has been processed.
        HTTPException 500: If inference fails.
    """
    if app_state.indexer and app_state.indexer.is_empty():
        raise HTTPException(
            status_code=503,
            detail=(
                "No meeting data indexed. "
                "Please upload and process files first using POST /process."
            ),
        )

    try:
        result = app_state.qa_engine.ask(
            question=request.question,
            top_k=request.top_k,
            include_images=request.include_images,
        )

        return AskResponse(
            question=result.question,
            answer=result.answer,
            grounding=[g.to_dict() for g in result.grounding],
            slide_image_paths=result.slide_image_paths,
            frame_image_paths=result.frame_image_paths,
            model_used=result.model_used,
            inference_time_s=result.inference_time_s,
            retrieval_time_s=result.retrieval_time_s,
            total_time_s=result.total_time_s,
            error=result.error,
        )

    except Exception as e:
        logger.exception(f"Q&A failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Q&A failed: {str(e)[:300]}",
        )


@app.get("/summary", response_model=SummaryResponse)
async def get_summary():
    """
    Generate an auto-summary of the processed meeting.

    Produces a structured summary with action items, decisions,
    and topic timeline.

    Returns:
        SummaryResponse with all structured sections.

    Raises:
        HTTPException 503: If no meeting data has been processed.
        HTTPException 500: If summary generation fails.
    """
    if app_state.indexer and app_state.indexer.is_empty():
        raise HTTPException(
            status_code=503,
            detail="No meeting data indexed. Please process files first.",
        )

    try:
        summary = app_state.qa_engine.generate_summary()
        return SummaryResponse(
            summary_text=summary.summary_text,
            action_items=summary.action_items,
            key_decisions=summary.key_decisions,
            topic_timeline=summary.topic_timeline,
            model_used=summary.model_used,
            generation_time_s=summary.generation_time_s,
        )
    except Exception as e:
        logger.exception(f"Summary generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Summary generation failed: {str(e)[:300]}",
        )


@app.get("/transcript")
async def get_transcript():
    """
    Return the full meeting transcript with timestamps and speakers.

    Returns:
        JSON with full_text and list of segments.

    Raises:
        HTTPException 503: If no transcript has been processed.
    """
    if (
        app_state.ingestion_result is None
        or app_state.ingestion_result.transcription is None
    ):
        raise HTTPException(
            status_code=503,
            detail="No transcript available. Please process an audio/video file first.",
        )

    transcription = app_state.ingestion_result.transcription
    return JSONResponse(
        content={
            "full_text": transcription.full_text,
            "language": transcription.language,
            "duration_seconds": transcription.duration_seconds,
            "segments": transcription.segments,
            "word_count": len(transcription.full_text.split()),
        }
    )


@app.get("/timeline")
async def get_timeline():
    """
    Return the unified meeting timeline.

    Returns:
        JSON list of timeline entries with timestamps, slides, and frames.

    Raises:
        HTTPException 503: If no meeting data has been processed.
    """
    if app_state.ingestion_result is None:
        raise HTTPException(
            status_code=503,
            detail="No meeting data. Please process files first.",
        )

    timeline_data = [
        entry.to_dict()
        for entry in app_state.ingestion_result.timeline
    ]
    return JSONResponse(content={"timeline": timeline_data})


@app.get("/frame/{frame_path:path}")
async def get_frame_image(frame_path: str):
    """
    Serve a video frame image by its file path.

    Args:
        frame_path: Relative or absolute path to frame image.

    Returns:
        JPEG image response.

    Raises:
        HTTPException 404: If image file not found.
    """
    path = Path(frame_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Frame not found: {frame_path}")
    return FileResponse(str(path), media_type="image/jpeg")


@app.get("/slide/{slide_path:path}")
async def get_slide_image(slide_path: str):
    """
    Serve a slide image by its file path.

    Args:
        slide_path: Relative or absolute path to slide image.

    Returns:
        JPEG image response.

    Raises:
        HTTPException 404: If image file not found.
    """
    path = Path(slide_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Slide not found: {slide_path}")
    return FileResponse(str(path), media_type="image/jpeg")


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info(f"Starting FastAPI on {API_HOST}:{API_PORT}")
    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
        log_level="info",
        access_log=True,
    )
