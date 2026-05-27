"""
core/transcriber.py

Whisper Large v3 GPU transcription with word-level timestamps and
optional speaker diarization using pyannote.audio.
"""

import os
import time
import tempfile
from pathlib import Path
from typing import Optional

import torch
from loguru import logger
from faster_whisper import WhisperModel


# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL", "large-v3")
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")
MODEL_DIR = os.getenv("WHISPER_MODEL_DIR", "./models/whisper")


def _get_compute_type(device: str) -> str:
    """Return optimal compute type based on device."""
    if device == "cuda":
        # float16 is fastest on A100; fall back to int8 on older GPUs
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            return "float16" if cap[0] >= 8 else "int8_float16"
    return "int8"


# ─────────────────────────────────────────────────────────────────
# TranscriptionResult dataclass
# ─────────────────────────────────────────────────────────────────
class TranscriptionResult:
    """Container for a completed transcription."""

    def __init__(
        self,
        full_text: str,
        segments: list[dict],
        language: str,
        duration_seconds: float,
        word_timestamps: list[dict],
    ):
        """
        Initialize transcription result.

        Args:
            full_text: Complete transcript as a single string.
            segments: List of segment dicts with start/end/text keys.
            language: Detected language code (e.g. 'en').
            duration_seconds: Total audio duration in seconds.
            word_timestamps: List of {word, start, end} dicts.
        """
        self.full_text = full_text
        self.segments = segments
        self.language = language
        self.duration_seconds = duration_seconds
        self.word_timestamps = word_timestamps

    def get_text_at_time(self, timestamp_seconds: float, window: float = 30.0) -> str:
        """Return transcript text within ±window seconds of timestamp."""
        relevant = [
            seg["text"]
            for seg in self.segments
            if abs(seg["start"] - timestamp_seconds) <= window
        ]
        return " ".join(relevant).strip()

    def format_timestamp(self, seconds: float) -> str:
        """Convert seconds to MM:SS string."""
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"

    def to_dict(self) -> dict:
        """Serialize to dict for JSON responses."""
        return {
            "full_text": self.full_text,
            "segments": self.segments,
            "language": self.language,
            "duration_seconds": self.duration_seconds,
            "word_timestamps": self.word_timestamps,
        }


# ─────────────────────────────────────────────────────────────────
# Main Transcriber class
# ─────────────────────────────────────────────────────────────────
class MeetingTranscriber:
    """
    GPU-accelerated transcription using Faster-Whisper Large v3.

    Provides word-level timestamps and optional speaker diarization.
    Gracefully falls back to CPU if GPU is unavailable.
    """

    def __init__(self):
        """Initialize transcriber — model is lazy-loaded on first use."""
        self.model: Optional[WhisperModel] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = _get_compute_type(self.device)
        self._diarizer = None
        logger.info(
            f"MeetingTranscriber init | device={self.device} | compute={self.compute_type}"
        )

    def _load_model(self) -> None:
        """Lazy-load Whisper model on first transcription call."""
        if self.model is not None:
            return

        logger.info(f"Loading Whisper {WHISPER_MODEL_SIZE} on {self.device}...")
        try:
            self.model = WhisperModel(
                WHISPER_MODEL_SIZE,
                device=self.device,
                compute_type=self.compute_type,
                download_root=MODEL_DIR,
                num_workers=4,
            )
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load Whisper on {self.device}: {e}")
            logger.info("Falling back to CPU with int8 quantization...")
            self.device = "cpu"
            self.compute_type = "int8"
            self.model = WhisperModel(
                WHISPER_MODEL_SIZE,
                device="cpu",
                compute_type="int8",
                download_root=MODEL_DIR,
            )
            logger.info("Whisper model loaded on CPU")

    def _load_diarizer(self) -> bool:
        """
        Attempt to load pyannote speaker diarization pipeline.

        Returns True if loaded successfully, False otherwise.
        """
        if self._diarizer is not None:
            return True
        if not HF_TOKEN or HF_TOKEN == "hf_your_token_here":
            logger.warning("No HF token — speaker diarization disabled")
            return False

        try:
            from pyannote.audio import Pipeline

            logger.info("Loading pyannote diarization pipeline...")
            self._diarizer = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=HF_TOKEN,
            )
            if torch.cuda.is_available():
                self._diarizer = self._diarizer.to(torch.device("cuda"))
            logger.info("Speaker diarization pipeline loaded")
            return True
        except Exception as e:
            logger.warning(f"Could not load diarization pipeline: {e}")
            return False

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        enable_diarization: bool = True,
    ) -> TranscriptionResult:
        """
        Transcribe an audio file with word-level timestamps.

        Args:
            audio_path: Path to audio file (MP3/WAV/M4A/FLAC).
            language: Force language code (e.g. 'en'). None = auto-detect.
            enable_diarization: Attempt speaker diarization if possible.

        Returns:
            TranscriptionResult with full text, segments, and word timestamps.

        Raises:
            FileNotFoundError: If audio_path does not exist.
            ValueError: If file format is unsupported.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        supported = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm", ".mp4"}
        if audio_path.suffix.lower() not in supported:
            raise ValueError(
                f"Unsupported audio format: {audio_path.suffix}. "
                f"Supported: {', '.join(supported)}"
            )

        self._load_model()
        t_start = time.time()
        logger.info(f"Transcribing: {audio_path.name}")

        # ── Run Whisper ───────────────────────────────────────────
        segments_gen, info = self.model.transcribe(
            str(audio_path),
            language=language,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            beam_size=5,
            best_of=5,
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            condition_on_previous_text=True,
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )

        segments: list[dict] = []
        word_timestamps: list[dict] = []
        full_text_parts: list[str] = []

        for seg in segments_gen:
            seg_dict = {
                "id": seg.id,
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": seg.text.strip(),
                "avg_logprob": round(seg.avg_logprob, 4),
                "no_speech_prob": round(seg.no_speech_prob, 4),
                "speaker": None,
                "timestamp_fmt": self._fmt_ts(seg.start),
            }
            segments.append(seg_dict)
            full_text_parts.append(seg.text.strip())

            if seg.words:
                for w in seg.words:
                    word_timestamps.append(
                        {
                            "word": w.word.strip(),
                            "start": round(w.start, 3),
                            "end": round(w.end, 3),
                            "probability": round(w.probability, 4),
                            "timestamp_fmt": self._fmt_ts(w.start),
                        }
                    )

        full_text = " ".join(full_text_parts)
        duration = info.duration if hasattr(info, "duration") else 0.0
        detected_lang = info.language if hasattr(info, "language") else "en"

        # ── Speaker Diarization (optional) ───────────────────────
        if enable_diarization and self._load_diarizer():
            segments = self._apply_diarization(str(audio_path), segments)

        elapsed = time.time() - t_start
        logger.info(
            f"Transcription done | {len(segments)} segments | "
            f"{len(word_timestamps)} words | {elapsed:.1f}s | lang={detected_lang}"
        )

        return TranscriptionResult(
            full_text=full_text,
            segments=segments,
            language=detected_lang,
            duration_seconds=duration,
            word_timestamps=word_timestamps,
        )

    def _apply_diarization(
        self, audio_path: str, segments: list[dict]
    ) -> list[dict]:
        """
        Assign speaker labels to transcript segments using pyannote.

        Args:
            audio_path: Path to audio file.
            segments: Whisper segments to annotate.

        Returns:
            Segments with 'speaker' field populated.
        """
        try:
            logger.info("Running speaker diarization...")
            diarization = self._diarizer(audio_path)

            # Build speaker timeline: list of (start, end, speaker)
            speaker_timeline = [
                (turn.start, turn.end, speaker)
                for turn, _, speaker in diarization.itertracks(yield_label=True)
            ]

            # Map generic labels to readable names
            unique_speakers = sorted(
                set(s for _, _, s in speaker_timeline)
            )
            speaker_map = {
                spk: f"Speaker {i + 1}"
                for i, spk in enumerate(unique_speakers)
            }

            # Assign speaker to each segment by majority overlap
            for seg in segments:
                seg_mid = (seg["start"] + seg["end"]) / 2
                best_speaker = None
                best_overlap = 0.0

                for s_start, s_end, spk in speaker_timeline:
                    overlap = min(seg["end"], s_end) - max(seg["start"], s_start)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_speaker = spk

                seg["speaker"] = (
                    speaker_map.get(best_speaker, "Unknown Speaker")
                    if best_speaker
                    else "Unknown Speaker"
                )

            logger.info(f"Diarization complete | {len(unique_speakers)} speakers")
        except Exception as e:
            logger.warning(f"Diarization failed: {e} — segments have no speaker labels")

        return segments

    @staticmethod
    def _fmt_ts(seconds: float) -> str:
        """Format seconds as MM:SS string."""
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"

    def unload(self) -> None:
        """Free GPU memory by unloading the model."""
        if self.model is not None:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Whisper model unloaded from memory")


# ─────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────
_transcriber_instance: Optional[MeetingTranscriber] = None


def get_transcriber() -> MeetingTranscriber:
    """Return module-level singleton transcriber."""
    global _transcriber_instance
    if _transcriber_instance is None:
        _transcriber_instance = MeetingTranscriber()
    return _transcriber_instance


# ─────────────────────────────────────────────────────────────────
# Quick standalone test
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python core/transcriber.py <audio_file>")
        sys.exit(1)

    audio_file = sys.argv[1]
    t = MeetingTranscriber()
    result = t.transcribe(audio_file)

    print(f"\n{'─'*60}")
    print(f"Language:  {result.language}")
    print(f"Duration:  {result.duration_seconds:.1f}s")
    print(f"Segments:  {len(result.segments)}")
    print(f"Words:     {len(result.word_timestamps)}")
    print(f"\nFirst 500 chars:\n{result.full_text[:500]}")
    print(f"\nFirst 3 segments:")
    for seg in result.segments[:3]:
        spk = seg.get("speaker", "?")
        print(f"  [{seg['timestamp_fmt']}] [{spk}] {seg['text']}")
