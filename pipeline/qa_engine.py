"""
pipeline/qa_engine.py

Cross-modal Q&A engine: retrieves relevant context from ChromaDB,
gathers associated images, calls Qwen2.5-Omni, and returns grounded answers.
"""

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from loguru import logger

from core.omni_model import OmniModel, OmniModelResponse, get_omni_model
from pipeline.indexer import MeetingIndexer, SearchResult, get_indexer
from utils.grounding import GroundingInfo, extract_grounding
from utils.formatter import format_grounded_answer


# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RESULTS", "5"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "3000"))
MAX_IMAGES_PER_QUERY = int(os.getenv("MAX_IMAGES_PER_QUERY", "3"))


# ─────────────────────────────────────────────────────────────────
# QAResult
# ─────────────────────────────────────────────────────────────────
@dataclass
class QAResult:
    """Complete result of a cross-modal question-answering operation."""

    question: str
    answer: str
    grounding: list[GroundingInfo] = field(default_factory=list)
    retrieved_docs: list[dict] = field(default_factory=list)
    slide_image_paths: list[str] = field(default_factory=list)
    frame_image_paths: list[str] = field(default_factory=list)
    model_used: str = ""
    inference_time_s: float = 0.0
    retrieval_time_s: float = 0.0
    total_time_s: float = 0.0
    error: Optional[str] = None

    @property
    def primary_slide_image(self) -> Optional[str]:
        """Return the most relevant slide image path, or None."""
        for p in self.slide_image_paths:
            if p and Path(p).exists():
                return p
        return None

    @property
    def primary_frame_image(self) -> Optional[str]:
        """Return the most relevant video frame image path, or None."""
        for p in self.frame_image_paths:
            if p and Path(p).exists():
                return p
        return None

    def to_dict(self) -> dict:
        """Serialize to dict for API responses."""
        return {
            "question": self.question,
            "answer": self.answer,
            "grounding": [g.to_dict() for g in self.grounding],
            "slide_image_paths": self.slide_image_paths,
            "frame_image_paths": self.frame_image_paths,
            "model_used": self.model_used,
            "inference_time_s": round(self.inference_time_s, 2),
            "retrieval_time_s": round(self.retrieval_time_s, 2),
            "total_time_s": round(self.total_time_s, 2),
            "error": self.error,
        }


# ─────────────────────────────────────────────────────────────────
# MeetingSummary
# ─────────────────────────────────────────────────────────────────
@dataclass
class MeetingSummary:
    """Auto-generated meeting summary with structured sections."""

    summary_text: str
    action_items: list[str]
    key_decisions: list[str]
    topic_timeline: list[dict]
    model_used: str
    generation_time_s: float

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "summary_text": self.summary_text,
            "action_items": self.action_items,
            "key_decisions": self.key_decisions,
            "topic_timeline": self.topic_timeline,
            "model_used": self.model_used,
            "generation_time_s": round(self.generation_time_s, 2),
        }


# ─────────────────────────────────────────────────────────────────
# QAEngine
# ─────────────────────────────────────────────────────────────────
class QAEngine:
    """
    Cross-modal Q&A engine for meeting content.

    For each question:
    1. Semantic retrieval from ChromaDB.
    2. Collect associated slide and frame images.
    3. Pass context + images to Qwen2.5-Omni.
    4. Return grounded answer with citations.
    """

    def __init__(
        self,
        indexer: Optional[MeetingIndexer] = None,
        model: Optional[OmniModel] = None,
    ):
        """
        Initialize Q&A engine with indexer and model.

        Args:
            indexer: ChromaDB indexer. Defaults to module singleton.
            model: OmniModel instance. Defaults to module singleton.
        """
        self.indexer = indexer or get_indexer()
        self.model = model or get_omni_model()
        logger.info("QAEngine initialized")

    def ask(
        self,
        question: str,
        top_k: int = TOP_K_RETRIEVAL,
        include_images: bool = True,
    ) -> QAResult:
        """
        Answer a question about the meeting using cross-modal reasoning.

        Args:
            question: Natural language question about the meeting.
            top_k: Number of documents to retrieve from ChromaDB.
            include_images: Whether to include slide/frame images in the query.

        Returns:
            QAResult with grounded answer and all supporting evidence.
        """
        t_total_start = time.time()
        logger.info(f"QA question: {question[:80]}")

        # ── Guard: check ChromaDB has content ─────────────────────
        if self.indexer.is_empty():
            return QAResult(
                question=question,
                answer="Please process meeting files first (Upload & Process tab).",
                error="ChromaDB is empty",
            )

        # ── Step 1: Semantic retrieval ────────────────────────────
        t_ret_start = time.time()
        try:
            search_results = self.indexer.search(question, top_k=top_k)
        except RuntimeError as e:
            return QAResult(
                question=question,
                answer=str(e),
                error=str(e),
            )
        retrieval_time = time.time() - t_ret_start

        if not search_results:
            return QAResult(
                question=question,
                answer="No relevant content found for your question.",
                error="No search results",
            )

        # ── Step 2: Collect image paths ───────────────────────────
        slide_images: list[str] = []
        frame_images: list[str] = []

        if include_images:
            seen_slides: set[int] = set()
            seen_frames: set[str] = set()

            for sr in search_results:
                if sr.slide_image_path and sr.slide_number not in seen_slides:
                    if Path(sr.slide_image_path).exists():
                        slide_images.append(sr.slide_image_path)
                        seen_slides.add(sr.slide_number)

                if sr.frame_image_path and sr.frame_image_path not in seen_frames:
                    if Path(sr.frame_image_path).exists():
                        frame_images.append(sr.frame_image_path)
                        seen_frames.add(sr.frame_image_path)

            # Limit images
            slide_images = slide_images[:MAX_IMAGES_PER_QUERY]
            frame_images = frame_images[:MAX_IMAGES_PER_QUERY]

        all_images = slide_images + frame_images

        # ── Step 3: Build context string ──────────────────────────
        context = self._build_context(search_results)

        # ── Step 4: Model inference ───────────────────────────────
        model_response: OmniModelResponse = self.model.generate(
            question=question,
            context_text=context,
            image_paths=all_images if include_images else None,
        )

        # ── Step 5: Extract grounding from answer ─────────────────
        grounding_info = extract_grounding(model_response.text, search_results)

        # Format the final answer with grounding badges
        formatted_answer = format_grounded_answer(
            model_response.text, grounding_info
        )

        total_time = time.time() - t_total_start

        result = QAResult(
            question=question,
            answer=formatted_answer,
            grounding=grounding_info,
            retrieved_docs=[sr.to_dict() for sr in search_results],
            slide_image_paths=slide_images,
            frame_image_paths=frame_images,
            model_used=model_response.model_used,
            inference_time_s=model_response.inference_time_s,
            retrieval_time_s=retrieval_time,
            total_time_s=total_time,
            error=model_response.error,
        )

        logger.info(
            f"QA complete | {total_time:.2f}s total | "
            f"model={model_response.model_used}"
        )
        return result

    def generate_summary(self) -> MeetingSummary:
        """
        Auto-generate a meeting summary from indexed content.

        Retrieves the most representative content from ChromaDB and
        asks the model to produce a structured summary.

        Returns:
            MeetingSummary with text, action items, decisions, and timeline.
        """
        if self.indexer.is_empty():
            return MeetingSummary(
                summary_text="Please process meeting files first.",
                action_items=[],
                key_decisions=[],
                topic_timeline=[],
                model_used="none",
                generation_time_s=0.0,
            )

        t_start = time.time()
        logger.info("Generating meeting summary...")

        # Retrieve broad sample of content for summarization
        broad_queries = [
            "main topics discussed in the meeting",
            "decisions made action items agreed",
            "problems challenges concerns raised",
        ]

        all_results: list[SearchResult] = []
        seen_ids: set[str] = set()

        for q in broad_queries:
            try:
                results = self.indexer.search(q, top_k=5)
                for r in results:
                    if r.doc_id not in seen_ids:
                        all_results.append(r)
                        seen_ids.add(r.doc_id)
            except Exception as e:
                logger.warning(f"Summary retrieval error: {e}")

        if not all_results:
            return MeetingSummary(
                summary_text="Could not retrieve meeting content for summary.",
                action_items=[],
                key_decisions=[],
                topic_timeline=[],
                model_used="none",
                generation_time_s=0.0,
            )

        context = self._build_context(all_results[:15])

        summary_prompt = (
            "Based on the meeting transcript below, provide a structured summary:\n\n"
            "1. SUMMARY: 2-3 sentence overview of the meeting.\n"
            "2. KEY DECISIONS: List each decision with [Timestamp] and [Slide] "
            "where possible.\n"
            "3. ACTION ITEMS: List each action item with owner and deadline if mentioned.\n"
            "4. TOPIC TIMELINE: List main topics in chronological order with timestamps.\n\n"
            "Be specific and cite timestamps in [MM:SS] format."
        )

        model_response = self.model.generate(
            question=summary_prompt,
            context_text=context,
            max_new_tokens=800,
        )

        # Parse structured output
        action_items = self._parse_list_section(
            model_response.text, "ACTION ITEMS"
        )
        key_decisions = self._parse_list_section(
            model_response.text, "KEY DECISIONS"
        )
        topic_timeline = self._parse_timeline_section(
            model_response.text, all_results
        )

        gen_time = time.time() - t_start

        return MeetingSummary(
            summary_text=model_response.text,
            action_items=action_items,
            key_decisions=key_decisions,
            topic_timeline=topic_timeline,
            model_used=model_response.model_used,
            generation_time_s=gen_time,
        )

    def _build_context(self, results: list[SearchResult]) -> str:
        """
        Build a context string from search results for model input.

        Includes timestamp, speaker, and slide info with each chunk.
        Truncates at MAX_CONTEXT_CHARS.
        """
        parts = []
        for r in results:
            slide_ref = f"[Slide {r.slide_number}]" if r.slide_number and r.slide_number > 0 else ""
            spk_ref = f"[{r.speaker}]" if r.speaker else ""
            header = f"[{r.timestamp_fmt}] {spk_ref} {slide_ref}".strip()
            parts.append(f"{header}\n{r.text}")

        context = "\n\n".join(parts)

        # Truncate to max context length
        if len(context) > MAX_CONTEXT_CHARS:
            context = context[:MAX_CONTEXT_CHARS] + "\n[... context truncated ...]"

        return context

    @staticmethod
    def _parse_list_section(text: str, section_name: str) -> list[str]:
        """Extract bullet-point list from a named section of model output."""
        items = []
        in_section = False
        for line in text.split("\n"):
            line = line.strip()
            if section_name in line.upper():
                in_section = True
                continue
            if in_section:
                if any(
                    kw in line.upper()
                    for kw in ["SUMMARY", "DECISIONS", "ACTION", "TOPIC", "TIMELINE"]
                ) and line and not line.startswith("-") and not line.startswith("•"):
                    break
                if line.startswith(("-", "•", "*", "–")) or (
                    len(line) > 2 and line[0].isdigit() and line[1] in ".)"
                ):
                    items.append(line.lstrip("-•*–0123456789.) ").strip())
        return items

    @staticmethod
    def _parse_timeline_section(
        text: str, results: list[SearchResult]
    ) -> list[dict]:
        """Build topic timeline from search results sorted by timestamp."""
        seen_timestamps: set[str] = set()
        timeline = []

        for r in sorted(results, key=lambda x: x.timestamp_seconds):
            ts_fmt = r.timestamp_fmt
            if ts_fmt not in seen_timestamps and r.text.strip():
                seen_timestamps.add(ts_fmt)
                timeline.append(
                    {
                        "timestamp": ts_fmt,
                        "timestamp_seconds": r.timestamp_seconds,
                        "topic": r.text[:100].strip(),
                        "slide": r.slide_number,
                        "speaker": r.speaker,
                    }
                )

        return timeline[:20]  # Max 20 timeline entries


# ─────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────
_qa_engine_instance: Optional[QAEngine] = None


def get_qa_engine() -> QAEngine:
    """Return module-level singleton QA engine."""
    global _qa_engine_instance
    if _qa_engine_instance is None:
        _qa_engine_instance = QAEngine()
    return _qa_engine_instance


# ─────────────────────────────────────────────────────────────────
# Quick standalone test
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from pipeline.indexer import MeetingIndexer

    indexer = MeetingIndexer()

    # Seed with sample data
    test_entries = [
        {
            "text": "We've finalized the pricing change. The new price will be $49 per seat, up from $39.",
            "ts": 872.0, "ts_fmt": "14:32", "speaker": "Rahul", "slide": 7,
        },
        {
            "text": "The budget slide shows we need $2M for Q3 operations.",
            "ts": 1020.0, "ts_fmt": "17:00", "speaker": "Priya", "slide": 9,
        },
        {
            "text": "I disagree with the 6-month timeline. We need at least 9 months for proper testing.",
            "ts": 1450.0, "ts_fmt": "24:10", "speaker": "Amit", "slide": 12,
        },
    ]

    for e in test_entries:
        indexer.index_additional_text(
            text=e["text"],
            metadata={
                "timestamp_seconds": e["ts"],
                "timestamp_fmt": e["ts_fmt"],
                "speaker": e["speaker"],
                "slide_number": e["slide"],
                "slide_image_path": "",
                "frame_image_path": "",
                "is_slide_transition": False,
                "chunk_idx": 0,
            },
        )

    engine = QAEngine(indexer=indexer)

    questions = [
        "What was the final decision on the pricing change?",
        "Which slide was being discussed when the budget came up?",
        "Who disagreed with the timeline and what did they propose instead?",
    ]

    for q in questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        result = engine.ask(q, include_images=False)
        print(f"A: {result.answer}")
        print(f"   Model: {result.model_used} | Time: {result.total_time_s:.2f}s")
