"""
utils/grounding.py

Extract and structure grounding evidence (timestamps, slide numbers,
speakers, direct quotes) from model answers and retrieval results.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from pipeline.indexer import SearchResult


# ─────────────────────────────────────────────────────────────────
# GroundingInfo
# ─────────────────────────────────────────────────────────────────
@dataclass
class GroundingInfo:
    """
    Structured grounding evidence for a single cited moment.

    Collects all evidence that corroborates a specific claim in the answer.
    """

    timestamp_fmt: str = ""
    timestamp_seconds: float = 0.0
    slide_number: Optional[int] = None
    speaker: Optional[str] = None
    quote: Optional[str] = None
    frame_image_path: Optional[str] = None
    slide_image_path: Optional[str] = None
    relevance_score: float = 0.0

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "timestamp_fmt": self.timestamp_fmt,
            "timestamp_seconds": self.timestamp_seconds,
            "slide_number": self.slide_number,
            "speaker": self.speaker,
            "quote": self.quote,
            "frame_image_path": self.frame_image_path,
            "slide_image_path": self.slide_image_path,
            "relevance_score": round(self.relevance_score, 4),
        }

    def has_timestamp(self) -> bool:
        """Return True if this grounding has a valid timestamp."""
        return bool(self.timestamp_fmt and self.timestamp_fmt != "??:??")

    def has_slide(self) -> bool:
        """Return True if this grounding has a valid slide number."""
        return self.slide_number is not None and self.slide_number > 0

    def has_speaker(self) -> bool:
        """Return True if this grounding has a speaker."""
        return bool(
            self.speaker
            and self.speaker not in ("Unknown", "Unknown Speaker", "")
        )


# ─────────────────────────────────────────────────────────────────
# Regex patterns for parsing model-generated citations
# ─────────────────────────────────────────────────────────────────
_TS_PATTERN = re.compile(
    r"\[(?:Timestamp|Time|At):\s*(\d{1,2}:\d{2})\]",
    re.IGNORECASE,
)
_SLIDE_PATTERN = re.compile(
    r"\[Slide[:\s]+(\d+)\]",
    re.IGNORECASE,
)
_SPEAKER_PATTERN = re.compile(
    r"\[(?:Speaker|By|From):\s*([^\]]+)\]",
    re.IGNORECASE,
)
_QUOTE_PATTERN = re.compile(
    r'\[Quote:\s*["""]([^"""]+)["""]\]',
    re.IGNORECASE,
)
# Also match inline "quoted text"
_INLINE_QUOTE_PATTERN = re.compile(
    r'["""]([^"""]{15,200})["""]',
)


def extract_grounding(
    answer_text: str,
    search_results: list[SearchResult],
) -> list[GroundingInfo]:
    """
    Extract grounding evidence from model answer text and search results.

    Combines:
    1. Citations explicitly embedded in the answer (e.g. [Timestamp: 14:32]).
    2. Metadata from the top retrieved documents.

    Args:
        answer_text: Generated answer string from the model.
        search_results: Retrieved ChromaDB documents used as context.

    Returns:
        List of GroundingInfo objects, deduplicated by timestamp.
    """
    grounding_list: list[GroundingInfo] = []
    seen_timestamps: set[str] = set()

    # ── 1. Parse citations from answer text ──────────────────────
    timestamps_in_answer = _TS_PATTERN.findall(answer_text)
    slides_in_answer = _SLIDE_PATTERN.findall(answer_text)
    speakers_in_answer = _SPEAKER_PATTERN.findall(answer_text)
    quotes_in_answer = _QUOTE_PATTERN.findall(answer_text)

    # Supplement quotes from inline "..." patterns
    if not quotes_in_answer:
        inline_quotes = _INLINE_QUOTE_PATTERN.findall(answer_text)
        quotes_in_answer = inline_quotes[:2]  # Max 2 inline quotes

    # Match timestamps from the answer to search results
    for ts_str in timestamps_in_answer:
        ts_seconds = _parse_timestamp_to_seconds(ts_str)
        if ts_str in seen_timestamps:
            continue

        # Find best matching search result for this timestamp
        matching_sr = _find_closest_result(search_results, ts_seconds)

        gi = GroundingInfo(
            timestamp_fmt=ts_str,
            timestamp_seconds=ts_seconds,
            relevance_score=1.0 - (matching_sr.distance if matching_sr else 1.0),
        )

        if matching_sr:
            gi.frame_image_path = matching_sr.frame_image_path
            gi.slide_image_path = matching_sr.slide_image_path
            gi.speaker = matching_sr.speaker
            if matching_sr.slide_number and matching_sr.slide_number > 0:
                gi.slide_number = matching_sr.slide_number

        # Override with explicit citations from answer
        if slides_in_answer:
            gi.slide_number = int(slides_in_answer[0])
        if speakers_in_answer:
            gi.speaker = speakers_in_answer[0].strip()
        if quotes_in_answer:
            gi.quote = quotes_in_answer[0].strip()

        grounding_list.append(gi)
        seen_timestamps.add(ts_str)

    # ── 2. Fall back to top search results if no inline citations ─
    if not grounding_list:
        for sr in search_results[:3]:
            ts = sr.timestamp_fmt
            if ts in seen_timestamps:
                continue

            gi = GroundingInfo(
                timestamp_fmt=ts,
                timestamp_seconds=sr.timestamp_seconds,
                slide_number=sr.slide_number if sr.slide_number and sr.slide_number > 0 else None,
                speaker=sr.speaker,
                frame_image_path=sr.frame_image_path,
                slide_image_path=sr.slide_image_path,
                relevance_score=1.0 - sr.distance,
            )

            # Try to find a relevant quote from the text
            text_words = sr.text.split()
            if len(text_words) >= 5:
                gi.quote = " ".join(text_words[:20])

            grounding_list.append(gi)
            seen_timestamps.add(ts)

    return grounding_list


def _parse_timestamp_to_seconds(ts_str: str) -> float:
    """
    Convert a MM:SS string to total seconds.

    Args:
        ts_str: Timestamp string like "14:32".

    Returns:
        Total seconds as float.
    """
    try:
        parts = ts_str.strip().split(":")
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except (ValueError, IndexError):
        pass
    return 0.0


def _find_closest_result(
    results: list[SearchResult], target_seconds: float
) -> Optional[SearchResult]:
    """
    Find the search result with timestamp closest to target_seconds.

    Args:
        results: List of search results.
        target_seconds: Target timestamp in seconds.

    Returns:
        Closest SearchResult, or None if list is empty.
    """
    if not results:
        return None

    return min(
        results,
        key=lambda r: abs(r.timestamp_seconds - target_seconds),
    )


def build_grounding_from_search(
    results: list[SearchResult],
) -> list[GroundingInfo]:
    """
    Build grounding info directly from search results (no model output parsing).

    Useful when constructing citations before model inference.

    Args:
        results: Retrieved search results from ChromaDB.

    Returns:
        List of GroundingInfo objects.
    """
    grounding: list[GroundingInfo] = []
    seen: set[str] = set()

    for sr in results:
        key = f"{sr.timestamp_fmt}_{sr.slide_number}"
        if key in seen:
            continue
        seen.add(key)

        # Extract a short quote from the text
        words = sr.text.split()
        quote = " ".join(words[:15]) if words else None

        gi = GroundingInfo(
            timestamp_fmt=sr.timestamp_fmt,
            timestamp_seconds=sr.timestamp_seconds,
            slide_number=sr.slide_number if sr.slide_number and sr.slide_number > 0 else None,
            speaker=sr.speaker,
            quote=quote,
            frame_image_path=sr.frame_image_path,
            slide_image_path=sr.slide_image_path,
            relevance_score=1.0 - sr.distance,
        )
        grounding.append(gi)

    return grounding


# ─────────────────────────────────────────────────────────────────
# Quick standalone test
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_answer = (
        "The final decision was to increase pricing by 15%. "
        "[Timestamp: 14:32] [Slide: 7] [Speaker: Rahul] "
        '[Quote: "We will move to $49 per seat starting Q3"]'
    )

    class _MockResult:
        timestamp_fmt = "14:32"
        timestamp_seconds = 872.0
        slide_number = 7
        speaker = "Rahul"
        frame_image_path = None
        slide_image_path = None
        distance = 0.05
        doc_id = "test-1"
        text = "We will move to $49 per seat starting Q3."

        def to_dict(self):
            return {}

    grounding = extract_grounding(sample_answer, [_MockResult()])
    for g in grounding:
        print(g.to_dict())
