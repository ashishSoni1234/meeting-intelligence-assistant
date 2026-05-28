"""
app.py

Gradio web UI for the Meeting Intelligence Assistant.
Three tabs: Upload & Process | Ask Questions | Meeting Summary
Runs on port 7860.
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from dotenv import load_dotenv
from loguru import logger

# Load environment before importing project modules
load_dotenv()

import gradio as gr

from pipeline.ingestion import IngestionPipeline, IngestionResult
from pipeline.indexer import MeetingIndexer, get_indexer
from pipeline.qa_engine import QAEngine, get_qa_engine, QAResult
from utils.formatter import (
    format_processing_result,
    format_grounded_answer,
    format_transcript_excerpt,
    format_meeting_summary,
)


# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────
GRADIO_HOST = os.getenv("GRADIO_HOST", "0.0.0.0")
GRADIO_PORT = int(os.getenv("GRADIO_PORT", "7860"))

# ─────────────────────────────────────────────────────────────────
# Shared state (Gradio is single-process)
# ─────────────────────────────────────────────────────────────────
_pipeline = IngestionPipeline()
_indexer: MeetingIndexer = get_indexer()
_qa_engine: QAEngine = get_qa_engine()
_last_ingestion: Optional[IngestionResult] = None
_last_qa_result: Optional[QAResult] = None


# ─────────────────────────────────────────────────────────────────
# Example questions (pre-loaded for quick demo)
# ─────────────────────────────────────────────────────────────────
EXAMPLE_QUESTIONS = [
    "Which slide was being shown when the recruitment program was discussed?",
    "Who were the presenters at this meeting and what roles do the slides assign them?",
    "What action items or next steps were mentioned, and at what point in the meeting?",
]


# ─────────────────────────────────────────────────────────────────
# Tab 1: Upload & Process
# ─────────────────────────────────────────────────────────────────

def process_files(
    pdf_file,
    audio_file,
    video_file,
    enable_diarization: bool,
    progress=gr.Progress(track_tqdm=True),
) -> tuple[str, str, str, str]:
    """
    Handle file upload and run the ingestion pipeline.

    Args:
        pdf_file: Gradio file object for PDF upload.
        audio_file: Gradio file object for audio upload.
        video_file: Gradio file object for video upload.
        enable_diarization: Checkbox value for speaker diarization.
        progress: Gradio progress tracker.

    Returns:
        Tuple of (status_text, transcript_preview, stats_text, error_text).
    """
    global _last_ingestion

    if not any([pdf_file, audio_file, video_file]):
        return (
            "❌ Please upload at least one file.",
            "",
            "",
            "No files provided.",
        )

    # Validate file extensions
    def get_path(f) -> Optional[str]:
        if f is None:
            return None
        if isinstance(f, str):
            return f
        if hasattr(f, "name"):
            return f.name
        return None

    pdf_path = get_path(pdf_file)
    audio_path = get_path(audio_file)
    video_path = get_path(video_file)

    # Validate formats
    if pdf_path and not pdf_path.lower().endswith(".pdf"):
        return ("❌ PDF file must be a .pdf file.", "", "", "Invalid PDF format.")

    audio_exts = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}
    if audio_path and not any(audio_path.lower().endswith(e) for e in audio_exts):
        return (
            f"❌ Audio file must be one of: {', '.join(audio_exts)}",
            "", "", "Invalid audio format.",
        )

    video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    if video_path and not any(video_path.lower().endswith(e) for e in video_exts):
        return (
            f"❌ Video file must be one of: {', '.join(video_exts)}",
            "", "", "Invalid video format.",
        )

    try:
        progress(0, desc="Starting parallel processing...")
        t_start = time.time()

        progress(0.1, desc="Processing files in parallel (audio + PDF + video)...")
        result = _pipeline.process(
            pdf_path=pdf_path,
            audio_path=audio_path,
            video_path=video_path,
            enable_diarization=enable_diarization,
        )
        _last_ingestion = result

        progress(0.7, desc="Indexing content into ChromaDB...")
        if result.timeline:
            n_indexed = _indexer.index_ingestion_result(result)
        else:
            n_indexed = 0

        progress(0.95, desc="Finalizing...")
        elapsed = time.time() - t_start

        # Build transcript preview
        transcript_preview = ""
        if result.transcription:
            preview_text = result.transcription.full_text[:800]
            first_segments = result.transcription.segments[:5]
            seg_lines = []
            for seg in first_segments:
                ts = seg.get("timestamp_fmt", "??:??")
                spk = seg.get("speaker", "")
                spk_str = f"[{spk}] " if spk else ""
                seg_lines.append(f"[{ts}] {spk_str}{seg['text']}")
            transcript_preview = "\n".join(seg_lines)

        # Stats
        stats = format_processing_result(
            slide_count=result.slide_count,
            frame_count=result.frame_count,
            transcript_preview=result.transcript_preview,
            processing_time_s=elapsed,
            errors=result.errors,
        )
        stats += f"\n\n**ChromaDB:** {n_indexed} chunks indexed"

        status = (
            f"✅ Processing complete in {elapsed:.1f}s | "
            f"{result.slide_count} slides | "
            f"{result.frame_count} frames | "
            f"{n_indexed} chunks indexed"
        )

        error_text = "\n".join(result.errors) if result.errors else "None"

        progress(1.0, desc="Done!")
        return status, transcript_preview, stats, error_text

    except Exception as e:
        logger.exception(f"Processing error: {e}")
        return (
            f"❌ Error: {str(e)[:300]}",
            "",
            "",
            str(e),
        )


# ─────────────────────────────────────────────────────────────────
# Tab 2: Ask Questions
# ─────────────────────────────────────────────────────────────────

def ask_question(
    question: str,
    include_images: bool,
    top_k: int,
) -> tuple[str, Optional[str], Optional[str], str]:
    """
    Answer a meeting question using cross-modal retrieval + Qwen-Omni.

    Args:
        question: User's question text.
        include_images: Whether to include slide/frame images in the query.
        top_k: Number of retrieval results to use.

    Returns:
        Tuple of (answer_text, slide_image_path, frame_image_path, transcript_excerpt).
    """
    global _last_qa_result

    if not question or not question.strip():
        return ("❌ Please enter a question.", None, None, "")

    if _indexer.is_empty():
        return (
            "❌ No meeting data found. Please upload and process files first (Tab 1).",
            None,
            None,
            "",
        )

    try:
        result = _qa_engine.ask(
            question=question.strip(),
            top_k=top_k,
            include_images=include_images,
        )
        _last_qa_result = result

        # Get best slide image to display
        slide_img = result.primary_slide_image
        frame_img = result.primary_frame_image

        # Build transcript excerpt
        transcript_excerpt = ""
        if result.retrieved_docs:
            # Reconstruct SearchResult-like objects for formatter
            class _SR:
                def __init__(self, d):
                    self.timestamp_fmt = d["metadata"].get("timestamp_fmt", "??:??")
                    self.speaker = d["metadata"].get("speaker")
                    self.slide_number = d["metadata"].get("slide_number")
                    self.text = d["text"]
                    self.distance = d["distance"]

            mock_results = [_SR(d) for d in result.retrieved_docs[:4]]
            transcript_excerpt = format_transcript_excerpt(mock_results)

        # Show timing in answer footer
        timing_info = (
            f"\n\n---\n*{result.model_used} | "
            f"retrieval: {result.retrieval_time_s:.2f}s | "
            f"inference: {result.inference_time_s:.2f}s | "
            f"total: {result.total_time_s:.2f}s*"
        )

        return (
            result.answer + timing_info,
            slide_img,
            frame_img,
            transcript_excerpt,
        )

    except Exception as e:
        logger.exception(f"Q&A error: {e}")
        return (f"❌ Error: {str(e)[:300]}", None, None, "")


# ─────────────────────────────────────────────────────────────────
# Tab 3: Meeting Summary
# ─────────────────────────────────────────────────────────────────

def generate_summary() -> str:
    """
    Generate an auto-summary of the processed meeting.

    Returns:
        Formatted markdown summary string.
    """
    if _indexer.is_empty():
        return "❌ No meeting data. Please upload and process files first (Tab 1)."

    try:
        summary = _qa_engine.generate_summary()
        return format_meeting_summary(
            summary_text=summary.summary_text,
            action_items=summary.action_items,
            key_decisions=summary.key_decisions,
            topic_timeline=summary.topic_timeline,
        )
    except Exception as e:
        logger.exception(f"Summary error: {e}")
        return f"❌ Summary generation failed: {str(e)[:300]}"


# ─────────────────────────────────────────────────────────────────
# GPU / system status helper
# ─────────────────────────────────────────────────────────────────

def get_system_status() -> str:
    """Return a one-line GPU/system status string for the UI header."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info()
        free_gb = free / 1e9
        total_gb = total / 1e9
        return (
            f"🟢 GPU: {name} | "
            f"VRAM: {free_gb:.1f}/{total_gb:.1f} GB free | "
            f"Indexed: {_indexer.get_document_count()} docs"
        )
    return (
        f"🟡 CPU mode (no GPU) | "
        f"Indexed: {_indexer.get_document_count()} docs"
    )


# ─────────────────────────────────────────────────────────────────
# Build Gradio UI
# ─────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    """
    Construct and return the Gradio Blocks UI.

    Returns:
        gr.Blocks application ready to launch.
    """
    with gr.Blocks(
        title="Meeting Intelligence Assistant",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
        ),
        css="""
            .grounding-badge {
                display: inline-block;
                background: #3b82f6;
                color: white;
                padding: 2px 8px;
                border-radius: 12px;
                font-size: 12px;
                margin: 2px;
            }
            .header-status {
                font-size: 13px;
                color: #64748b;
                padding: 4px 0;
            }
        """,
    ) as demo:

        # ── Header ────────────────────────────────────────────────
        gr.Markdown(
            """
            # 🎯 Meeting Intelligence Assistant
            **Upload your meeting recording, slides, and video — then ask anything.**

            Powered by **Qwen2.5-Omni-7B** + **Whisper Large v3** + **ChromaDB**
            """
        )

        # Live status bar
        with gr.Row():
            status_bar = gr.Markdown(
                value=get_system_status(),
                elem_classes=["header-status"],
            )
            refresh_btn = gr.Button("🔄 Refresh Status", size="sm", scale=0)

        refresh_btn.click(fn=get_system_status, outputs=status_bar)

        # ── Tabs ─────────────────────────────────────────────────
        with gr.Tabs():

            # ── TAB 1: Upload & Process ───────────────────────────
            with gr.Tab("📁 Upload & Process", id="upload_tab"):
                gr.Markdown(
                    "Upload your meeting files. All three are optional — "
                    "upload whichever you have."
                )

                with gr.Row():
                    pdf_input = gr.File(
                        label="📊 PDF / Slides",
                        file_types=[".pdf"],
                        type="filepath",
                    )
                    audio_input = gr.File(
                        label="🎤 Audio Recording (MP3/WAV/M4A)",
                        file_types=[".mp3", ".wav", ".m4a", ".flac"],
                        type="filepath",
                    )
                    video_input = gr.File(
                        label="🎬 Meeting Video (MP4/MOV)",
                        file_types=[".mp4", ".mov", ".avi", ".mkv"],
                        type="filepath",
                    )

                with gr.Row():
                    diarize_check = gr.Checkbox(
                        label="Enable Speaker Diarization (requires HF token)",
                        value=True,
                    )
                    process_btn = gr.Button(
                        "⚡ Process Files",
                        variant="primary",
                        size="lg",
                        scale=2,
                    )

                process_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    placeholder="Upload files and click Process...",
                )

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 📝 Transcript Preview")
                        transcript_preview_box = gr.Textbox(
                            label="",
                            interactive=False,
                            lines=8,
                            placeholder="Transcript will appear here after processing...",
                        )
                    with gr.Column():
                        gr.Markdown("### 📊 Processing Stats")
                        stats_box = gr.Markdown(
                            value="_Stats will appear after processing._"
                        )

                error_box = gr.Textbox(
                    label="⚠️ Warnings / Errors",
                    interactive=False,
                    visible=True,
                    lines=2,
                )

                process_btn.click(
                    fn=process_files,
                    inputs=[
                        pdf_input,
                        audio_input,
                        video_input,
                        diarize_check,
                    ],
                    outputs=[
                        process_status,
                        transcript_preview_box,
                        stats_box,
                        error_box,
                    ],
                    show_progress=True,
                )

            # ── TAB 2: Ask Questions ──────────────────────────────
            with gr.Tab("❓ Ask Questions", id="qa_tab"):
                gr.Markdown(
                    "Ask any question about the meeting. "
                    "Answers are grounded in timestamps, slide numbers, and speakers."
                )

                with gr.Row():
                    question_input = gr.Textbox(
                        label="Your Question",
                        placeholder="e.g. What was the final decision on the pricing change?",
                        lines=2,
                        scale=5,
                    )
                    ask_btn = gr.Button(
                        "🔍 Ask",
                        variant="primary",
                        size="lg",
                        scale=1,
                    )

                # Quick example buttons
                gr.Markdown("**Quick examples:**")
                with gr.Row():
                    for ex_q in EXAMPLE_QUESTIONS:
                        ex_btn = gr.Button(ex_q, size="sm", variant="secondary")
                        ex_btn.click(
                            fn=lambda q=ex_q: q,
                            outputs=question_input,
                        )

                with gr.Row():
                    images_check = gr.Checkbox(
                        label="Include slide/frame images in query",
                        value=True,
                    )
                    top_k_slider = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1,
                        label="Retrieval depth (top-K)",
                    )

                with gr.Row():
                    with gr.Column(scale=3):
                        answer_output = gr.Markdown(
                            label="Answer",
                            value="_Ask a question to get started._",
                        )
                    with gr.Column(scale=2):
                        gr.Markdown("**Relevant Slide:**")
                        slide_image_output = gr.Image(
                            label="",
                            type="filepath",
                            show_label=False,
                            height=300,
                        )
                        gr.Markdown("**Relevant Video Frame:**")
                        frame_image_output = gr.Image(
                            label="",
                            type="filepath",
                            show_label=False,
                            height=200,
                        )

                transcript_excerpt_output = gr.Markdown(
                    label="",
                    value="",
                )

                ask_btn.click(
                    fn=ask_question,
                    inputs=[question_input, images_check, top_k_slider],
                    outputs=[
                        answer_output,
                        slide_image_output,
                        frame_image_output,
                        transcript_excerpt_output,
                    ],
                )

                question_input.submit(
                    fn=ask_question,
                    inputs=[question_input, images_check, top_k_slider],
                    outputs=[
                        answer_output,
                        slide_image_output,
                        frame_image_output,
                        transcript_excerpt_output,
                    ],
                )

            # ── TAB 3: Meeting Summary ────────────────────────────
            with gr.Tab("📋 Meeting Summary", id="summary_tab"):
                gr.Markdown(
                    "Auto-generate a structured meeting summary with action items, "
                    "key decisions, and topic timeline."
                )

                summarize_btn = gr.Button(
                    "✨ Generate Summary",
                    variant="primary",
                    size="lg",
                )

                summary_output = gr.Markdown(
                    value="_Click 'Generate Summary' after processing your meeting files._",
                    label="",
                )

                summarize_btn.click(
                    fn=generate_summary,
                    inputs=[],
                    outputs=[summary_output],
                    show_progress=True,
                )

        # ── Footer ────────────────────────────────────────────────
        gr.Markdown(
            """
            ---
            **How it works:** Upload → Process (parallel: Whisper + PDF + Video frames)
            → ChromaDB indexing → Qwen2.5-Omni cross-modal Q&A → Grounded answers
            """
        )

    return demo


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Starting Meeting Intelligence Gradio UI...")
    logger.info(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    demo = build_ui()
    demo.launch(
        server_name=GRADIO_HOST,
        server_port=GRADIO_PORT,
        share=False,
        show_error=True,
        favicon_path=None,
        allowed_paths=["./extracted_slides", "./extracted_frames"],
    )
