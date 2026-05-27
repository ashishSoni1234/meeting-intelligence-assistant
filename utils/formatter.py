"""
utils/formatter.py

Format grounded Q&A answers with citation badges, structured summaries,
and transcript excerpts for display in the Gradio UI.
"""

from typing import Optional

from utils.grounding import GroundingInfo


# ─────────────────────────────────────────────────────────────────
# Answer formatting
# ─────────────────────────────────────────────────────────────────

def format_grounded_answer(
    answer_text: str,
    grounding: list[GroundingInfo],
) -> str:
    """
    Prepend grounding badges to the answer text.

    Generates a human-readable answer with emoji citation badges such as:
      🕐 14:32  |  📊 Slide 7  |  👤 Rahul

    Args:
        answer_text: Raw generated answer from the model.
        grounding: List of GroundingInfo objects extracted from the answer.

    Returns:
        Formatted string with grounding summary followed by the answer.
    """
    if not grounding:
        return answer_text

    # Build citation badge line from all grounding entries
    citation_parts = []

    # Deduplicate: collect unique values
    timestamps = []
    slides = []
    speakers = []
    quotes = []

    seen_ts: set[str] = set()
    seen_slides: set[int] = set()
    seen_speakers: set[str] = set()

    for g in grounding:
        if g.has_timestamp() and g.timestamp_fmt not in seen_ts:
            timestamps.append(g.timestamp_fmt)
            seen_ts.add(g.timestamp_fmt)
        if g.has_slide() and g.slide_number not in seen_slides:
            slides.append(g.slide_number)
            seen_slides.add(g.slide_number)
        if g.has_speaker() and g.speaker not in seen_speakers:
            speakers.append(g.speaker)
            seen_speakers.add(g.speaker)
        if g.quote and g.quote not in quotes:
            quotes.append(g.quote)

    for ts in timestamps[:2]:
        citation_parts.append(f"🕐 {ts}")
    for sl in slides[:2]:
        citation_parts.append(f"📊 Slide {sl}")
    for sp in speakers[:2]:
        citation_parts.append(f"👤 {sp}")

    if not citation_parts:
        return answer_text

    badge_line = "  |  ".join(citation_parts)

    # Clean answer text: remove raw bracket citations to avoid duplication
    clean_answer = _strip_bracket_citations(answer_text)

    # Build final formatted answer
    lines = [
        f"**Citations:** {badge_line}",
        "",
        clean_answer,
    ]

    if quotes:
        lines.append("")
        lines.append(f"**Relevant quote:** *\"{quotes[0]}\"*")

    return "\n".join(lines)


def format_transcript_excerpt(
    search_results: list,
    max_chars: int = 600,
) -> str:
    """
    Format a readable transcript excerpt from search results.

    Args:
        search_results: List of SearchResult objects from ChromaDB.
        max_chars: Maximum characters to include.

    Returns:
        Formatted string with timestamped transcript excerpts.
    """
    if not search_results:
        return "_No transcript content available._"

    lines = ["**Relevant Transcript Excerpts:**", ""]

    total_chars = 0
    for sr in search_results:
        speaker_str = f"[**{sr.speaker}**] " if sr.speaker else ""
        slide_str = f"[Slide {sr.slide_number}] " if sr.slide_number and sr.slide_number > 0 else ""
        line = f"**[{sr.timestamp_fmt}]** {speaker_str}{slide_str}— {sr.text.strip()}"

        if total_chars + len(line) > max_chars:
            break

        lines.append(line)
        lines.append("")
        total_chars += len(line)

    return "\n".join(lines)


def format_meeting_summary(
    summary_text: str,
    action_items: list[str],
    key_decisions: list[str],
    topic_timeline: list[dict],
) -> str:
    """
    Format the complete meeting summary for display in the Summary tab.

    Args:
        summary_text: Raw summary text from model.
        action_items: List of action item strings.
        key_decisions: List of key decision strings.
        topic_timeline: List of {timestamp, topic, slide, speaker} dicts.

    Returns:
        Formatted markdown string for Gradio display.
    """
    sections = []

    # ── Overview ─────────────────────────────────────────────────
    sections.append("## 📋 Meeting Overview")
    sections.append("")
    # Extract just the summary part from full model output
    summary_clean = _extract_section(summary_text, "SUMMARY", stop_on_next_section=True)
    if not summary_clean:
        summary_clean = summary_text[:500]
    sections.append(summary_clean.strip())
    sections.append("")

    # ── Key Decisions ─────────────────────────────────────────────
    sections.append("## ✅ Key Decisions")
    sections.append("")
    if key_decisions:
        for decision in key_decisions:
            sections.append(f"- {decision}")
    else:
        # Parse from full summary text
        parsed = _parse_decisions_from_text(summary_text)
        if parsed:
            for d in parsed:
                sections.append(f"- {d}")
        else:
            sections.append("_No decisions extracted._")
    sections.append("")

    # ── Action Items ─────────────────────────────────────────────
    sections.append("## 📌 Action Items")
    sections.append("")
    if action_items:
        for item in action_items:
            sections.append(f"- [ ] {item}")
    else:
        parsed = _parse_action_items_from_text(summary_text)
        if parsed:
            for item in parsed:
                sections.append(f"- [ ] {item}")
        else:
            sections.append("_No action items extracted._")
    sections.append("")

    # ── Timeline ─────────────────────────────────────────────────
    sections.append("## 🕐 Topic Timeline")
    sections.append("")
    if topic_timeline:
        for entry in topic_timeline[:15]:
            ts = entry.get("timestamp", "??:??")
            topic = entry.get("topic", "")[:80]
            slide = entry.get("slide")
            slide_str = f" [Slide {slide}]" if slide and slide > 0 else ""
            speaker = entry.get("speaker", "")
            spk_str = f" — *{speaker}*" if speaker and speaker != "Unknown" else ""
            sections.append(f"- **[{ts}]**{slide_str}{spk_str} — {topic}")
    else:
        sections.append("_No timeline data available._")

    return "\n".join(sections)


def format_processing_result(
    slide_count: int,
    frame_count: int,
    transcript_preview: str,
    processing_time_s: float,
    errors: list[str],
) -> str:
    """
    Format the processing result summary shown after Upload & Process.

    Args:
        slide_count: Number of slides extracted from PDF.
        frame_count: Number of frames extracted from video.
        transcript_preview: First few hundred chars of transcript.
        processing_time_s: Total processing wall-clock time.
        errors: List of error messages (empty if all succeeded).

    Returns:
        Formatted markdown summary string.
    """
    lines = ["### ✅ Processing Complete", ""]

    # Stats table
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Slides extracted | {slide_count} |")
    lines.append(f"| Video frames extracted | {frame_count} |")
    lines.append(f"| Processing time | {processing_time_s:.1f}s |")
    lines.append("")

    if errors:
        lines.append("**⚠️ Warnings:**")
        for err in errors:
            lines.append(f"- {err}")
        lines.append("")

    if transcript_preview:
        lines.append("**📝 Transcript Preview:**")
        lines.append("")
        lines.append(f"_{transcript_preview[:400]}..._")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────

def _strip_bracket_citations(text: str) -> str:
    """
    Remove structured citation brackets from model output.

    Removes patterns like [Timestamp: X], [Slide: N], [Speaker: X], [Quote: "..."]
    since these are rendered as badges instead.
    """
    import re

    patterns = [
        r"\[Timestamp:\s*[^\]]+\]",
        r"\[Time:\s*[^\]]+\]",
        r"\[Slide[:\s]+\d+\]",
        r"\[Speaker:\s*[^\]]+\]",
        r"\[By:\s*[^\]]+\]",
        r"\[Quote:\s*[^\]]+\]",
    ]

    result = text
    for pattern in patterns:
        result = re.sub(pattern, "", result, flags=re.IGNORECASE)

    # Clean up double spaces left by removal
    result = re.sub(r"  +", " ", result)
    return result.strip()


def _extract_section(text: str, section_name: str, stop_on_next_section: bool = False) -> str:
    """Extract text from a named section of model output."""
    lines = text.split("\n")
    in_section = False
    collected = []

    for line in lines:
        if section_name.upper() in line.upper():
            in_section = True
            continue

        if in_section:
            if stop_on_next_section and any(
                kw in line.upper() and ":" in line
                for kw in ["DECISION", "ACTION", "TIMELINE", "TOPIC", "ITEM"]
            ):
                break
            collected.append(line)

    return "\n".join(collected).strip()


def _parse_decisions_from_text(text: str) -> list[str]:
    """Parse key decisions from summary text."""
    import re

    decisions = []
    # Find lines that look like decisions
    for line in text.split("\n"):
        line = line.strip()
        if (
            line.startswith(("-", "•", "–"))
            and any(kw in line.lower() for kw in ["decided", "approved", "will", "agreed", "chosen"])
        ):
            decisions.append(line.lstrip("-•– "))
    return decisions[:10]


def _parse_action_items_from_text(text: str) -> list[str]:
    """Parse action items from summary text."""
    items = []
    in_action_section = False

    for line in text.split("\n"):
        line = line.strip()
        if "action item" in line.lower():
            in_action_section = True
            continue

        if in_action_section:
            if any(
                kw in line.upper()
                for kw in ["DECISION", "SUMMARY", "TIMELINE", "TOPIC"]
            ) and not line.startswith("-"):
                break
            if line.startswith(("-", "•", "–")) or (
                len(line) > 2 and line[0].isdigit() and line[1] in ".)"
            ):
                items.append(line.lstrip("-•–0123456789.) ").strip())

    return items[:10]


# ─────────────────────────────────────────────────────────────────
# Quick standalone test
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from utils.grounding import GroundingInfo

    g = GroundingInfo(
        timestamp_fmt="14:32",
        timestamp_seconds=872.0,
        slide_number=7,
        speaker="Rahul",
        quote="We will move to $49 per seat starting Q3",
    )

    answer = (
        "The meeting decided to raise pricing by 15%. "
        "[Timestamp: 14:32] [Slide: 7] [Speaker: Rahul] "
        '[Quote: "We will move to $49 per seat"]'
    )

    formatted = format_grounded_answer(answer, [g])
    print("Formatted answer:")
    print(formatted)
    print()

    summary = format_meeting_summary(
        summary_text="This meeting covered pricing, budget, and Q3 timeline.",
        action_items=["Rahul to update pricing page by EOW", "Priya to send budget report"],
        key_decisions=["Pricing increased to $49/seat", "Timeline extended to 9 months"],
        topic_timeline=[
            {"timestamp": "00:05", "topic": "Welcome and agenda", "slide": 1, "speaker": "Rahul"},
            {"timestamp": "14:32", "topic": "Pricing decision", "slide": 7, "speaker": "Rahul"},
        ],
    )
    print("Meeting summary:")
    print(summary)
