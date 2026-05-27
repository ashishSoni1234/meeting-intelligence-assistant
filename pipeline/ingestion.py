"""
pipeline/ingestion.py

Multimodal ingestion pipeline: parallel processing of PDF, audio, and
video inputs followed by unified timeline construction.
"""

import os
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from loguru import logger

from core.transcriber import MeetingTranscriber, TranscriptionResult
from core.slide_processor import SlideProcessor, SlideProcessingResult
from core.video_processor import VideoProcessor, VideoProcessingResult


# ─────────────────────────────────────────────────────────────────
# Data model — Unified Timeline Entry
# ─────────────────────────────────────────────────────────────────
@dataclass
class TimelineEntry:
    """
    A single moment in the unified meeting timeline.

    Combines transcript text, slide reference, and video frame
    at (or near) a given timestamp.
    """

    timestamp_seconds: float
    timestamp_fmt: str
    transcript_text: str
    speaker: Optional[str]
    slide_number: Optional[int]
    slide_text: Optional[str]
    slide_image_path: Optional[str]
    frame_image_path: Optional[str]
    is_slide_transition: bool = False

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "timestamp_seconds": self.timestamp_seconds,
            "timestamp_fmt": self.timestamp_fmt,
            "transcript_text": self.transcript_text,
            "speaker": self.speaker,
            "slide_number": self.slide_number,
            "slide_text": self.slide_text,
            "slide_image_path": self.slide_image_path,
            "frame_image_path": self.frame_image_path,
            "is_slide_transition": self.is_slide_transition,
        }


@dataclass
class IngestionResult:
    """Complete result of a multimodal ingestion run."""

    timeline: list[TimelineEntry] = field(default_factory=list)
    transcription: Optional[TranscriptionResult] = None
    slides: Optional[SlideProcessingResult] = None
    video: Optional[VideoProcessingResult] = None
    total_processing_time_s: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def slide_count(self) -> int:
        """Total number of slides processed."""
        return self.slides.total_slides if self.slides else 0

    @property
    def frame_count(self) -> int:
        """Total number of video frames extracted."""
        return self.video.total_frames_extracted if self.video else 0

    @property
    def transcript_preview(self) -> str:
        """First 500 chars of transcript for UI display."""
        if self.transcription:
            return self.transcription.full_text[:500]
        return ""

    def to_summary_dict(self) -> dict:
        """Serialize to summary dict for API responses."""
        return {
            "timeline_entries": len(self.timeline),
            "slide_count": self.slide_count,
            "frame_count": self.frame_count,
            "transcript_length": (
                len(self.transcription.full_text) if self.transcription else 0
            ),
            "audio_duration_s": (
                self.transcription.duration_seconds if self.transcription else 0
            ),
            "video_duration_s": (
                self.video.video_duration_seconds if self.video else 0
            ),
            "total_processing_time_s": round(self.total_processing_time_s, 2),
            "errors": self.errors,
        }


# ─────────────────────────────────────────────────────────────────
# IngestionPipeline
# ─────────────────────────────────────────────────────────────────
class IngestionPipeline:
    """
    Parallel multimodal ingestion pipeline.

    Step 1: Run PDF, audio, and video processors in parallel threads.
    Step 2: Merge results into a unified timeline keyed by timestamp.
    Step 3: Return IngestionResult ready for ChromaDB indexing.
    """

    def __init__(self):
        """Initialize pipeline with default processors."""
        self.transcriber = MeetingTranscriber()
        self.slide_proc = SlideProcessor()
        self.video_proc = VideoProcessor()
        logger.info("IngestionPipeline initialized")

    def process(
        self,
        pdf_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        video_path: Optional[str] = None,
        enable_diarization: bool = True,
    ) -> IngestionResult:
        """
        Process up to three input files in parallel.

        At least one of pdf_path, audio_path, or video_path must be provided.

        Args:
            pdf_path: Path to PDF presentation file.
            audio_path: Path to audio file (MP3/WAV/M4A).
            video_path: Path to video file (MP4/MOV).
            enable_diarization: Whether to attempt speaker diarization.

        Returns:
            IngestionResult with unified timeline and all sub-results.

        Raises:
            ValueError: If no input files are provided.
        """
        if not any([pdf_path, audio_path, video_path]):
            raise ValueError(
                "At least one input file required: pdf_path, audio_path, or video_path"
            )

        t_start = time.time()
        result = IngestionResult()

        # ── Step 1: Parallel processing ───────────────────────────
        logger.info("Starting parallel processing of input files...")
        futures = {}

        # If video provided but no audio, extract audio from video for transcription
        extracted_audio_path: Optional[str] = None
        if video_path and not audio_path:
            extracted_audio_path = self._extract_audio_from_video(video_path)
            if extracted_audio_path:
                audio_path = extracted_audio_path
                logger.info(f"Audio extracted from video: {extracted_audio_path}")

        with ThreadPoolExecutor(max_workers=3) as executor:
            if pdf_path:
                futures["pdf"] = executor.submit(
                    self._process_pdf_safe, pdf_path
                )
            if audio_path:
                futures["audio"] = executor.submit(
                    self._process_audio_safe, audio_path, enable_diarization
                )
            if video_path:
                futures["video"] = executor.submit(
                    self._process_video_safe, video_path
                )

            for key, future in futures.items():
                try:
                    output = future.result()
                    if key == "pdf":
                        result.slides = output
                    elif key == "audio":
                        result.transcription = output
                    elif key == "video":
                        result.video = output
                    logger.info(f"  [{key}] processing complete")
                except Exception as e:
                    err = f"{key} processing failed: {str(e)}"
                    result.errors.append(err)
                    logger.error(err)

        # Cleanup temp audio extracted from video
        if extracted_audio_path:
            try:
                Path(extracted_audio_path).unlink(missing_ok=True)
            except Exception:
                pass

        # ── Step 2: Build unified timeline ────────────────────────
        logger.info("Building unified timeline...")
        result.timeline = self._build_timeline(
            result.transcription, result.slides, result.video
        )

        result.total_processing_time_s = time.time() - t_start
        logger.info(
            f"Ingestion complete | {len(result.timeline)} timeline entries | "
            f"{result.total_processing_time_s:.1f}s"
        )

        return result

    def _process_pdf_safe(self, pdf_path: str) -> Optional[SlideProcessingResult]:
        """Process PDF with error handling."""
        try:
            logger.info(f"Processing PDF: {pdf_path}")
            return self.slide_proc.process_pdf(pdf_path)
        except Exception as e:
            raise RuntimeError(f"PDF processing error: {e}") from e

    def _process_audio_safe(
        self, audio_path: str, enable_diarization: bool
    ) -> Optional[TranscriptionResult]:
        """Process audio with error handling."""
        try:
            logger.info(f"Processing audio: {audio_path}")
            return self.transcriber.transcribe(
                audio_path, enable_diarization=enable_diarization
            )
        except Exception as e:
            raise RuntimeError(f"Audio processing error: {e}") from e

    def _process_video_safe(self, video_path: str) -> Optional[VideoProcessingResult]:
        """Process video with error handling."""
        try:
            logger.info(f"Processing video: {video_path}")
            return self.video_proc.process_video(video_path)
        except Exception as e:
            raise RuntimeError(f"Video processing error: {e}") from e

    def _extract_audio_from_video(self, video_path: str) -> Optional[str]:
        """Extract audio track from video using ffmpeg. Returns temp WAV path or None."""
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            cmd = [
                "ffmpeg", "-y", "-i", video_path,
                "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                tmp.name,
            ]
            subprocess.run(cmd, check=True, capture_output=True, timeout=300)
            logger.info(f"Audio extracted from video -> {tmp.name}")
            return tmp.name
        except FileNotFoundError:
            logger.warning("ffmpeg not found — cannot extract audio from video")
            return None
        except subprocess.CalledProcessError as e:
            logger.warning(f"ffmpeg audio extraction failed: {e.stderr.decode()[:200]}")
            return None
        except Exception as e:
            logger.warning(f"Audio extraction error: {e}")
            return None

    def _build_timeline(
        self,
        transcription: Optional[TranscriptionResult],
        slides: Optional[SlideProcessingResult],
        video: Optional[VideoProcessingResult],
    ) -> list[TimelineEntry]:
        """
        Merge transcript segments, slide data, and video frames into a timeline.

        Strategy:
        - Primary axis: transcript segments (each has a timestamp).
        - For each segment timestamp, find:
          * Closest video frame → derive estimated slide number.
          * If slides exist: map segment to slide using frame->slide logic.
        - If no transcript: use video frames as timeline axis.
        """
        if transcription and transcription.segments:
            return self._timeline_from_transcript(transcription, slides, video)
        elif video and video.frames:
            return self._timeline_from_video(video, slides)
        elif slides:
            return self._timeline_from_slides(slides)
        else:
            logger.warning("No data to build timeline from")
            return []

    def _timeline_from_transcript(
        self,
        transcription: TranscriptionResult,
        slides: Optional[SlideProcessingResult],
        video: Optional[VideoProcessingResult],
    ) -> list[TimelineEntry]:
        """Build timeline using transcript segments as primary axis."""
        # Build a simple slide number estimator from video transitions
        slide_estimator = self._build_slide_estimator(slides, video)

        entries = []
        for seg in transcription.segments:
            ts_s = seg["start"]

            # Estimate current slide
            slide_num = slide_estimator(ts_s)
            slide_text = None
            slide_img = None

            if slides and slide_num is not None:
                slide_obj = slides.get_slide(slide_num)
                if slide_obj:
                    slide_text = slide_obj.text[:500]
                    slide_img = slide_obj.image_path

            # Find closest video frame
            frame_img = None
            is_transition = False
            if video:
                frame = video.get_frame_at_time(ts_s)
                if frame:
                    frame_img = frame.image_path
                    is_transition = frame.is_slide_transition

            entry = TimelineEntry(
                timestamp_seconds=ts_s,
                timestamp_fmt=seg.get("timestamp_fmt", self._fmt_ts(ts_s)),
                transcript_text=seg["text"],
                speaker=seg.get("speaker"),
                slide_number=slide_num,
                slide_text=slide_text,
                slide_image_path=slide_img,
                frame_image_path=frame_img,
                is_slide_transition=is_transition,
            )
            entries.append(entry)

        return entries

    def _timeline_from_video(
        self,
        video: VideoProcessingResult,
        slides: Optional[SlideProcessingResult],
    ) -> list[TimelineEntry]:
        """Build timeline using video frames as primary axis (no transcript)."""
        slide_estimator = self._build_slide_estimator(slides, video)
        entries = []

        for frame in video.frames:
            slide_num = slide_estimator(frame.timestamp_seconds)
            slide_text = None
            slide_img = None

            if slides and slide_num is not None:
                slide_obj = slides.get_slide(slide_num)
                if slide_obj:
                    slide_text = slide_obj.text[:500]
                    slide_img = slide_obj.image_path

            entry = TimelineEntry(
                timestamp_seconds=frame.timestamp_seconds,
                timestamp_fmt=frame.timestamp_fmt,
                transcript_text="[No transcript available]",
                speaker=None,
                slide_number=slide_num,
                slide_text=slide_text,
                slide_image_path=slide_img,
                frame_image_path=frame.image_path,
                is_slide_transition=frame.is_slide_transition,
            )
            entries.append(entry)

        return entries

    def _timeline_from_slides(
        self, slides: SlideProcessingResult
    ) -> list[TimelineEntry]:
        """Build timeline from slides only (no audio/video)."""
        entries = []
        for slide in slides.slides:
            entry = TimelineEntry(
                timestamp_seconds=0.0,
                timestamp_fmt="00:00",
                transcript_text="[No transcript available]",
                speaker=None,
                slide_number=slide.slide_number,
                slide_text=slide.text[:500],
                slide_image_path=slide.image_path,
                frame_image_path=None,
                is_slide_transition=False,
            )
            entries.append(entry)
        return entries

    def _build_slide_estimator(
        self,
        slides: Optional[SlideProcessingResult],
        video: Optional[VideoProcessingResult],
    ):
        """
        Return a function that estimates slide number at a given timestamp.

        Uses video transition frames to create a temporal slide map.
        Falls back to uniform distribution across video duration if needed.
        """
        if not slides:
            return lambda t: None

        total_slides = slides.total_slides

        if not video or not video.frames:
            # No video — slide number unknown, return slide 1 always
            return lambda t: 1

        # Collect transition timestamps
        transitions = [
            f.timestamp_seconds
            for f in video.frames
            if f.is_slide_transition
        ]
        transitions.sort()

        if not transitions:
            # No transitions detected: linearly distribute slides across duration
            duration = video.video_duration_seconds or 1.0

            def linear_estimator(t: float) -> int:
                frac = min(t / duration, 1.0)
                return max(1, min(total_slides, int(frac * total_slides) + 1))

            return linear_estimator

        # Map each transition to a slide number
        # Slide 1 starts at t=0; each transition advances to next slide
        def transition_estimator(t: float) -> int:
            slide = 1
            for transition_t in transitions:
                if t >= transition_t:
                    slide += 1
                else:
                    break
            return min(slide, total_slides)

        return transition_estimator

    @staticmethod
    def _fmt_ts(seconds: float) -> str:
        """Format seconds as MM:SS."""
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"


# ─────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────
_pipeline_instance: Optional[IngestionPipeline] = None


def get_ingestion_pipeline() -> IngestionPipeline:
    """Return module-level singleton ingestion pipeline."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = IngestionPipeline()
    return _pipeline_instance


# ─────────────────────────────────────────────────────────────────
# Quick standalone test
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    print("IngestionPipeline test")
    print("Usage: provide at least one of --pdf, --audio, --video")

    pdf = None
    audio = None
    video = None

    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--pdf" and i + 1 < len(args):
            pdf = args[i + 1]
        elif arg == "--audio" and i + 1 < len(args):
            audio = args[i + 1]
        elif arg == "--video" and i + 1 < len(args):
            video = args[i + 1]

    if not any([pdf, audio, video]):
        print("No inputs provided. Example:")
        print("  python pipeline/ingestion.py --audio meeting.mp3 --pdf slides.pdf")
        sys.exit(1)

    pipeline = IngestionPipeline()
    result = pipeline.process(pdf_path=pdf, audio_path=audio, video_path=video)

    print(f"\n{'─'*60}")
    print(f"Timeline entries: {len(result.timeline)}")
    print(f"Slides:           {result.slide_count}")
    print(f"Frames:           {result.frame_count}")
    print(f"Processing time:  {result.total_processing_time_s:.1f}s")
    print(f"Errors:           {result.errors}")

    print(f"\nFirst 3 timeline entries:")
    for entry in result.timeline[:3]:
        print(f"\n  [{entry.timestamp_fmt}] Slide {entry.slide_number}")
        print(f"  Speaker: {entry.speaker}")
        print(f"  Text: {entry.transcript_text[:100]}...")
