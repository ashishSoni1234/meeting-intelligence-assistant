"""
core/slide_processor.py

Extract text and images from PDF/slide files.
Each page becomes a SlideData record: text content + saved image.
"""

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from loguru import logger
from PIL import Image


# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────
SLIDES_DIR = os.getenv("SLIDES_DIR", "./extracted_slides")
SLIDE_IMAGE_DPI = int(os.getenv("SLIDE_IMAGE_DPI", "150"))
MAX_SLIDES = int(os.getenv("MAX_SLIDES", "200"))


# ─────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────
@dataclass
class SlideData:
    """Represents a single extracted slide."""

    slide_number: int
    text: str
    image_path: str
    width_px: int
    height_px: int
    has_images: bool
    word_count: int

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "slide_number": self.slide_number,
            "text": self.text,
            "image_path": self.image_path,
            "width_px": self.width_px,
            "height_px": self.height_px,
            "has_images": self.has_images,
            "word_count": self.word_count,
        }


@dataclass
class SlideProcessingResult:
    """Result of processing an entire PDF file."""

    slides: list[SlideData] = field(default_factory=list)
    total_slides: int = 0
    total_words: int = 0
    processing_time_s: float = 0.0
    source_file: str = ""
    output_dir: str = ""

    def get_slide(self, number: int) -> Optional[SlideData]:
        """Return slide by 1-based slide number."""
        for s in self.slides:
            if s.slide_number == number:
                return s
        return None

    def get_text_for_slides(self, numbers: list[int]) -> str:
        """Concatenate text from multiple slides."""
        parts = []
        for n in numbers:
            s = self.get_slide(n)
            if s:
                parts.append(f"[Slide {n}]: {s.text}")
        return "\n\n".join(parts)

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "slides": [s.to_dict() for s in self.slides],
            "total_slides": self.total_slides,
            "total_words": self.total_words,
            "processing_time_s": round(self.processing_time_s, 2),
            "source_file": self.source_file,
            "output_dir": self.output_dir,
        }


# ─────────────────────────────────────────────────────────────────
# SlideProcessor
# ─────────────────────────────────────────────────────────────────
class SlideProcessor:
    """
    Extract text and rasterize images from PDF presentation files.

    Uses PyMuPDF (fitz) for both text extraction and page rendering.
    Each page is saved as a JPEG image for use in multimodal queries.
    """

    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize processor.

        Args:
            output_dir: Directory to save slide images.
                        Defaults to SLIDES_DIR env var.
        """
        self.output_dir = Path(output_dir or SLIDES_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"SlideProcessor init | output_dir={self.output_dir}")

    def process_pdf(self, pdf_path: str) -> SlideProcessingResult:
        """
        Process a PDF file: extract text and render each page as an image.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            SlideProcessingResult containing all SlideData objects.

        Raises:
            FileNotFoundError: If pdf_path does not exist.
            ValueError: If file is not a valid PDF or has no pages.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        if pdf_path.suffix.lower() not in {".pdf"}:
            raise ValueError(
                f"Expected PDF file, got: {pdf_path.suffix}. "
                "Please upload a .pdf file."
            )

        t_start = time.time()
        logger.info(f"Processing PDF: {pdf_path.name}")

        # Create per-document subdirectory to avoid name collisions
        doc_dir = self.output_dir / pdf_path.stem
        doc_dir.mkdir(parents=True, exist_ok=True)

        try:
            doc = fitz.open(str(pdf_path))
        except Exception as e:
            raise ValueError(f"Cannot open PDF: {e}")

        if doc.page_count == 0:
            raise ValueError("PDF has no pages.")

        page_count = min(doc.page_count, MAX_SLIDES)
        if doc.page_count > MAX_SLIDES:
            logger.warning(
                f"PDF has {doc.page_count} pages — processing first {MAX_SLIDES}"
            )

        slides: list[SlideData] = []
        total_words = 0

        for page_idx in range(page_count):
            page = doc.load_page(page_idx)
            slide_num = page_idx + 1

            # ── Extract text ──────────────────────────────────────
            text = self._extract_page_text(page)
            word_count = len(text.split()) if text else 0
            total_words += word_count

            # ── Render page as image ──────────────────────────────
            image_path = doc_dir / f"slide_{slide_num:04d}.jpg"
            width_px, height_px, has_images = self._render_page(
                page, image_path
            )

            slide = SlideData(
                slide_number=slide_num,
                text=text,
                image_path=str(image_path),
                width_px=width_px,
                height_px=height_px,
                has_images=has_images,
                word_count=word_count,
            )
            slides.append(slide)
            logger.debug(
                f"  Slide {slide_num}/{page_count} | "
                f"{word_count} words | images={has_images}"
            )

        doc.close()
        elapsed = time.time() - t_start

        result = SlideProcessingResult(
            slides=slides,
            total_slides=len(slides),
            total_words=total_words,
            processing_time_s=elapsed,
            source_file=str(pdf_path),
            output_dir=str(doc_dir),
        )

        logger.info(
            f"PDF processed | {len(slides)} slides | "
            f"{total_words} words | {elapsed:.1f}s"
        )
        return result

    def _extract_page_text(self, page: fitz.Page) -> str:
        """
        Extract clean text from a PDF page.

        Preserves paragraph breaks and removes excessive whitespace.
        """
        # Use layout-aware text extraction
        text = page.get_text("text", flags=fitz.TEXT_PRESERVE_LIGATURES)
        if not text:
            # Try blocks extraction as fallback
            blocks = page.get_text("blocks")
            parts = [b[4].strip() for b in blocks if b[6] == 0]  # type 0 = text
            text = "\n".join(p for p in parts if p)

        # Clean up whitespace
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        text = "\n".join(lines)
        return text

    def _render_page(
        self, page: fitz.Page, output_path: Path
    ) -> tuple[int, int, bool]:
        """
        Render a PDF page as a JPEG image.

        Args:
            page: PyMuPDF page object.
            output_path: Where to save the JPEG.

        Returns:
            Tuple of (width_px, height_px, has_images).
        """
        # Scale factor for desired DPI (PDF default is 72 dpi)
        scale = SLIDE_IMAGE_DPI / 72.0
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        # Check if page has embedded images
        has_images = len(page.get_images()) > 0

        # Save as JPEG with good quality
        pix.save(str(output_path), output="jpeg")

        return pix.width, pix.height, has_images

    def get_slide_image(
        self, result: SlideProcessingResult, slide_number: int
    ) -> Optional[str]:
        """
        Return image path for a given slide number.

        Args:
            result: Previously computed SlideProcessingResult.
            slide_number: 1-based slide number.

        Returns:
            Absolute path string if found, else None.
        """
        slide = result.get_slide(slide_number)
        if slide and Path(slide.image_path).exists():
            return slide.image_path
        return None

    def extract_text_only(self, pdf_path: str) -> list[dict]:
        """
        Quick text-only extraction without rendering images.

        Useful for fast indexing when images aren't needed immediately.

        Args:
            pdf_path: Path to PDF file.

        Returns:
            List of {slide_number, text} dicts.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc = fitz.open(str(pdf_path))
        results = []
        for i in range(doc.page_count):
            page = doc.load_page(i)
            text = self._extract_page_text(page)
            results.append({"slide_number": i + 1, "text": text})
        doc.close()
        return results


# ─────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────
_processor_instance: Optional[SlideProcessor] = None


def get_slide_processor() -> SlideProcessor:
    """Return module-level singleton slide processor."""
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = SlideProcessor()
    return _processor_instance


# ─────────────────────────────────────────────────────────────────
# Quick standalone test
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python core/slide_processor.py <pdf_file>")
        sys.exit(1)

    proc = SlideProcessor()
    result = proc.process_pdf(sys.argv[1])

    print(f"\n{'─'*60}")
    print(f"Source:     {result.source_file}")
    print(f"Slides:     {result.total_slides}")
    print(f"Words:      {result.total_words}")
    print(f"Time:       {result.processing_time_s:.1f}s")
    print(f"Output dir: {result.output_dir}")
    print(f"\nFirst 3 slides:")
    for slide in result.slides[:3]:
        print(f"\n  [Slide {slide.slide_number}]")
        print(f"  Words: {slide.word_count} | Images: {slide.has_images}")
        print(f"  Text: {slide.text[:200]}...")
        print(f"  Image: {slide.image_path}")
