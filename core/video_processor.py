"""
core/video_processor.py

Extract key frames from video files at regular intervals and detect
slide transitions via inter-frame pixel difference scoring.
"""

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from loguru import logger


# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────
FRAMES_DIR = os.getenv("FRAMES_DIR", "./extracted_frames")
FRAME_INTERVAL = int(os.getenv("FRAME_INTERVAL", "30"))       # seconds
MAX_FRAMES = int(os.getenv("MAX_FRAMES", "50"))
TRANSITION_THRESHOLD = float(os.getenv("SLIDE_TRANSITION_THRESHOLD", "30"))
JPEG_QUALITY = int(os.getenv("FRAME_JPEG_QUALITY", "85"))


# ─────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────
@dataclass
class VideoFrame:
    """A single extracted video frame with metadata."""

    frame_number: int
    timestamp_seconds: float
    timestamp_fmt: str
    image_path: str
    width: int
    height: int
    is_slide_transition: bool = False
    diff_score: float = 0.0

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "frame_number": self.frame_number,
            "timestamp_seconds": self.timestamp_seconds,
            "timestamp_fmt": self.timestamp_fmt,
            "image_path": self.image_path,
            "width": self.width,
            "height": self.height,
            "is_slide_transition": self.is_slide_transition,
            "diff_score": self.diff_score,
        }


@dataclass
class VideoProcessingResult:
    """Result of processing an entire video file."""

    frames: list[VideoFrame] = field(default_factory=list)
    total_frames_extracted: int = 0
    transition_frames: int = 0
    video_duration_seconds: float = 0.0
    fps: float = 0.0
    width: int = 0
    height: int = 0
    processing_time_s: float = 0.0
    source_file: str = ""
    output_dir: str = ""

    def get_frame_at_time(self, timestamp_seconds: float) -> Optional[VideoFrame]:
        """Return the frame closest to the given timestamp."""
        if not self.frames:
            return None
        return min(
            self.frames,
            key=lambda f: abs(f.timestamp_seconds - timestamp_seconds),
        )

    def get_frames_in_range(
        self, start_s: float, end_s: float
    ) -> list[VideoFrame]:
        """Return all frames within the given time range."""
        return [
            f
            for f in self.frames
            if start_s <= f.timestamp_seconds <= end_s
        ]

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "frames": [f.to_dict() for f in self.frames],
            "total_frames_extracted": self.total_frames_extracted,
            "transition_frames": self.transition_frames,
            "video_duration_seconds": self.video_duration_seconds,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "processing_time_s": round(self.processing_time_s, 2),
            "source_file": self.source_file,
            "output_dir": self.output_dir,
        }


# ─────────────────────────────────────────────────────────────────
# VideoProcessor
# ─────────────────────────────────────────────────────────────────
class VideoProcessor:
    """
    Extract key frames from video at regular intervals.

    Additionally detects slide transitions using histogram-based
    frame difference scoring to identify when a new slide appears.
    """

    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize video processor.

        Args:
            output_dir: Directory to save extracted frames.
                        Defaults to FRAMES_DIR env var.
        """
        self.output_dir = Path(output_dir or FRAMES_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"VideoProcessor init | output_dir={self.output_dir}")

    def process_video(
        self,
        video_path: str,
        frame_interval: Optional[int] = None,
    ) -> VideoProcessingResult:
        """
        Extract frames from video at regular intervals.

        Args:
            video_path: Path to video file (MP4/MOV/AVI/MKV).
            frame_interval: Seconds between frame extractions.
                            Defaults to FRAME_INTERVAL env var.

        Returns:
            VideoProcessingResult with all extracted VideoFrame objects.

        Raises:
            FileNotFoundError: If video_path does not exist.
            ValueError: If file format is unsupported or video cannot be opened.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        supported = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".flv"}
        if video_path.suffix.lower() not in supported:
            raise ValueError(
                f"Unsupported video format: {video_path.suffix}. "
                f"Supported: {', '.join(supported)}"
            )

        interval = frame_interval or FRAME_INTERVAL
        t_start = time.time()
        logger.info(f"Processing video: {video_path.name} | interval={interval}s")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        # ── Video metadata ────────────────────────────────────────
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_s = total_video_frames / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(
            f"  Video info: {duration_s:.1f}s | {fps:.1f}fps | "
            f"{width}x{height} | {total_video_frames} frames"
        )

        # Per-video subdirectory
        vid_dir = self.output_dir / video_path.stem
        vid_dir.mkdir(parents=True, exist_ok=True)

        frames_to_extract = self._compute_frame_timestamps(
            duration_s, interval, fps
        )

        extracted: list[VideoFrame] = []
        prev_frame_gray: Optional[np.ndarray] = None
        frame_idx = 0

        for target_second, frame_num_in_video in frames_to_extract:
            # Seek to target frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_num_in_video))
            ret, bgr_frame = cap.read()
            if not ret:
                logger.debug(f"Could not read frame at {target_second:.1f}s")
                continue

            # ── Slide transition detection ────────────────────────
            gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
            diff_score = self._compute_frame_diff(prev_frame_gray, gray)
            is_transition = diff_score > TRANSITION_THRESHOLD
            prev_frame_gray = gray

            # ── Save frame ────────────────────────────────────────
            frame_filename = f"frame_{frame_idx:04d}_{int(target_second):05d}s.jpg"
            frame_path = vid_dir / frame_filename
            cv2.imwrite(
                str(frame_path),
                bgr_frame,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
            )

            vf = VideoFrame(
                frame_number=frame_idx,
                timestamp_seconds=round(target_second, 2),
                timestamp_fmt=self._fmt_ts(target_second),
                image_path=str(frame_path),
                width=width,
                height=height,
                is_slide_transition=is_transition,
                diff_score=round(diff_score, 2),
            )
            extracted.append(vf)
            frame_idx += 1

            logger.debug(
                f"  Frame {frame_idx} | {vf.timestamp_fmt} | "
                f"diff={diff_score:.1f} | transition={is_transition}"
            )

        cap.release()
        elapsed = time.time() - t_start

        transition_count = sum(1 for f in extracted if f.is_slide_transition)

        result = VideoProcessingResult(
            frames=extracted,
            total_frames_extracted=len(extracted),
            transition_frames=transition_count,
            video_duration_seconds=duration_s,
            fps=fps,
            width=width,
            height=height,
            processing_time_s=elapsed,
            source_file=str(video_path),
            output_dir=str(vid_dir),
        )

        logger.info(
            f"Video processed | {len(extracted)} frames | "
            f"{transition_count} transitions | {elapsed:.1f}s"
        )
        return result

    def _compute_frame_timestamps(
        self, duration_s: float, interval_s: int, fps: float
    ) -> list[tuple[float, int]]:
        """
        Compute (timestamp_seconds, frame_number) pairs to extract.

        Limits total extraction to MAX_FRAMES.
        """
        timestamps = []
        t = 0.0
        while t <= duration_s:
            frame_num = int(t * fps)
            timestamps.append((t, frame_num))
            t += interval_s

        # Enforce MAX_FRAMES cap
        if len(timestamps) > MAX_FRAMES:
            logger.warning(
                f"Video would yield {len(timestamps)} frames — "
                f"capping at {MAX_FRAMES}"
            )
            # Evenly sample MAX_FRAMES from the timeline
            step = len(timestamps) / MAX_FRAMES
            timestamps = [timestamps[int(i * step)] for i in range(MAX_FRAMES)]

        return timestamps

    def _compute_frame_diff(
        self,
        prev: Optional[np.ndarray],
        curr: np.ndarray,
    ) -> float:
        """
        Compute normalized pixel difference between two grayscale frames.

        Uses mean absolute difference on resized thumbnails for speed.
        Returns 0.0 if prev is None (first frame).
        """
        if prev is None:
            return 0.0

        # Resize to small thumbnail for fast comparison
        thumb_size = (160, 90)
        p = cv2.resize(prev, thumb_size)
        c = cv2.resize(curr, thumb_size)

        diff = np.mean(np.abs(c.astype(np.float32) - p.astype(np.float32)))
        return float(diff)

    @staticmethod
    def _fmt_ts(seconds: float) -> str:
        """Format seconds as MM:SS string."""
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"

    def get_transition_frames(
        self, result: VideoProcessingResult
    ) -> list[VideoFrame]:
        """Return only the frames identified as slide transitions."""
        return [f for f in result.frames if f.is_slide_transition]

    def get_frame_image_path(
        self, result: VideoProcessingResult, timestamp_seconds: float
    ) -> Optional[str]:
        """
        Get image path for the frame closest to a given timestamp.

        Args:
            result: Previously computed VideoProcessingResult.
            timestamp_seconds: Target time in seconds.

        Returns:
            Image path string if found and file exists, else None.
        """
        frame = result.get_frame_at_time(timestamp_seconds)
        if frame and Path(frame.image_path).exists():
            return frame.image_path
        return None


# ─────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────
_processor_instance: Optional[VideoProcessor] = None


def get_video_processor() -> VideoProcessor:
    """Return module-level singleton video processor."""
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = VideoProcessor()
    return _processor_instance


# ─────────────────────────────────────────────────────────────────
# Quick standalone test
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python core/video_processor.py <video_file>")
        sys.exit(1)

    proc = VideoProcessor()
    result = proc.process_video(sys.argv[1])

    print(f"\n{'─'*60}")
    print(f"Source:      {result.source_file}")
    print(f"Duration:    {result.video_duration_seconds:.1f}s")
    print(f"FPS:         {result.fps:.1f}")
    print(f"Resolution:  {result.width}x{result.height}")
    print(f"Frames:      {result.total_frames_extracted}")
    print(f"Transitions: {result.transition_frames}")
    print(f"Time:        {result.processing_time_s:.1f}s")
    print(f"Output dir:  {result.output_dir}")
    print(f"\nFirst 5 frames:")
    for f in result.frames[:5]:
        trans = " [TRANSITION]" if f.is_slide_transition else ""
        print(f"  [{f.timestamp_fmt}] frame_{f.frame_number} diff={f.diff_score}{trans}")
