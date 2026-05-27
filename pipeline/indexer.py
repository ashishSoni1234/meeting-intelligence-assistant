"""
pipeline/indexer.py

ChromaDB vector store for meeting timeline entries.
Supports semantic similarity search for cross-modal Q&A retrieval.
"""

import os
import uuid
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from loguru import logger
from sentence_transformers import SentenceTransformer

from pipeline.ingestion import IngestionResult, TimelineEntry


# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
COLLECTION_NAME = "meeting_transcripts"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))


# ─────────────────────────────────────────────────────────────────
# SearchResult
# ─────────────────────────────────────────────────────────────────
class SearchResult:
    """A single retrieved document from ChromaDB."""

    def __init__(
        self,
        text: str,
        metadata: dict,
        distance: float,
        doc_id: str,
    ):
        """
        Initialize search result.

        Args:
            text: The retrieved text chunk.
            metadata: Associated metadata (timestamp, slide, speaker, etc.).
            distance: Cosine distance from query vector (lower = more similar).
            doc_id: ChromaDB document ID.
        """
        self.text = text
        self.metadata = metadata
        self.distance = distance
        self.doc_id = doc_id

    @property
    def timestamp_fmt(self) -> str:
        """Return formatted timestamp from metadata."""
        return self.metadata.get("timestamp_fmt", "??:??")

    @property
    def timestamp_seconds(self) -> float:
        """Return timestamp in seconds from metadata."""
        return float(self.metadata.get("timestamp_seconds", 0.0))

    @property
    def slide_number(self) -> Optional[int]:
        """Return slide number from metadata."""
        val = self.metadata.get("slide_number")
        return int(val) if val is not None else None

    @property
    def speaker(self) -> Optional[str]:
        """Return speaker name from metadata."""
        return self.metadata.get("speaker")

    @property
    def frame_image_path(self) -> Optional[str]:
        """Return video frame image path from metadata."""
        path = self.metadata.get("frame_image_path")
        return path if path and Path(path).exists() else None

    @property
    def slide_image_path(self) -> Optional[str]:
        """Return slide image path from metadata."""
        path = self.metadata.get("slide_image_path")
        return path if path and Path(path).exists() else None

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "text": self.text,
            "metadata": self.metadata,
            "distance": self.distance,
            "doc_id": self.doc_id,
        }


# ─────────────────────────────────────────────────────────────────
# MeetingIndexer
# ─────────────────────────────────────────────────────────────────
class MeetingIndexer:
    """
    ChromaDB-backed semantic index for meeting content.

    Each indexed document is a chunk of transcript text with metadata
    pointing to timestamps, slide numbers, speaker names, and image paths.
    Retrieval uses sentence-transformer embeddings + cosine similarity.
    """

    def __init__(self, chroma_path: Optional[str] = None):
        """
        Initialize ChromaDB client and embedding model.

        Args:
            chroma_path: Directory for persistent ChromaDB storage.
                         Defaults to CHROMA_PATH env var.
        """
        self.chroma_path = Path(chroma_path or CHROMA_PATH)
        self.chroma_path.mkdir(parents=True, exist_ok=True)

        # Persistent ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.chroma_path),
            settings=Settings(anonymized_telemetry=False),
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        # Sentence transformer for embeddings
        self._encoder: Optional[SentenceTransformer] = None
        logger.info(
            f"MeetingIndexer init | chroma={self.chroma_path} | "
            f"collection={COLLECTION_NAME} | "
            f"existing_docs={self.collection.count()}"
        )

    def _get_encoder(self) -> SentenceTransformer:
        """Lazy-load sentence transformer encoder."""
        if self._encoder is None:
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
            logger.info("Embedding model loaded")
        return self._encoder

    def index_ingestion_result(self, result: IngestionResult) -> int:
        """
        Index all timeline entries from an IngestionResult into ChromaDB.

        Clears existing collection first to avoid stale data.

        Args:
            result: Completed IngestionResult from the ingestion pipeline.

        Returns:
            Number of documents indexed.
        """
        if not result.timeline:
            logger.warning("No timeline entries to index")
            return 0

        # Clear existing data
        self._clear_collection()

        # Prepare documents in batches
        docs_indexed = 0
        batch_size = 100
        chunks = self._prepare_chunks(result)

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            self._upsert_batch(batch)
            docs_indexed += len(batch)
            logger.debug(f"  Indexed {docs_indexed}/{len(chunks)} chunks")

        logger.info(f"Indexing complete | {docs_indexed} documents in ChromaDB")
        return docs_indexed

    def _prepare_chunks(self, result: IngestionResult) -> list[dict]:
        """
        Convert timeline entries into indexable text chunks with metadata.

        Each transcript segment becomes one or more chunks. Slide text is
        appended to the first chunk of segments that reference that slide.
        """
        chunks = []
        encoder = self._get_encoder()

        for entry in result.timeline:
            # Base text: transcript
            text_parts = [entry.transcript_text]

            # Optionally include slide text for context
            if entry.slide_text:
                text_parts.append(f"[Slide {entry.slide_number} content]: {entry.slide_text}")

            combined_text = "\n".join(text_parts).strip()
            if not combined_text:
                continue

            # Split long entries into overlapping chunks
            text_chunks = self._chunk_text(combined_text, CHUNK_SIZE, CHUNK_OVERLAP)

            for chunk_idx, chunk_text in enumerate(text_chunks):
                doc_id = str(uuid.uuid4())
                metadata = {
                    "timestamp_seconds": float(entry.timestamp_seconds),
                    "timestamp_fmt": entry.timestamp_fmt,
                    "speaker": entry.speaker or "Unknown",
                    "slide_number": entry.slide_number if entry.slide_number is not None else -1,
                    "slide_image_path": entry.slide_image_path or "",
                    "frame_image_path": entry.frame_image_path or "",
                    "is_slide_transition": bool(entry.is_slide_transition),
                    "chunk_idx": chunk_idx,
                }
                chunks.append(
                    {"id": doc_id, "text": chunk_text, "metadata": metadata}
                )

        return chunks

    def _upsert_batch(self, chunks: list[dict]) -> None:
        """Insert a batch of chunks into ChromaDB."""
        encoder = self._get_encoder()
        texts = [c["text"] for c in chunks]
        embeddings = encoder.encode(texts, batch_size=32, show_progress_bar=False)

        self.collection.upsert(
            ids=[c["id"] for c in chunks],
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=[c["metadata"] for c in chunks],
        )

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_speaker: Optional[str] = None,
        filter_slide: Optional[int] = None,
    ) -> list[SearchResult]:
        """
        Semantic search over indexed meeting content.

        Args:
            query: Natural language question or keywords.
            top_k: Number of results to return. Defaults to TOP_K_RESULTS.
            filter_speaker: Optionally filter by speaker name.
            filter_slide: Optionally filter by slide number.

        Returns:
            List of SearchResult objects sorted by relevance.

        Raises:
            RuntimeError: If ChromaDB collection is empty (no content indexed).
        """
        if self.collection.count() == 0:
            raise RuntimeError(
                "ChromaDB is empty. Please process meeting files first "
                "using the Upload & Process tab."
            )

        k = top_k or TOP_K_RESULTS
        encoder = self._get_encoder()
        query_embedding = encoder.encode([query], show_progress_bar=False)[0]

        # Build optional where clause
        where: Optional[dict] = None
        if filter_speaker:
            where = {"speaker": {"$eq": filter_speaker}}
        elif filter_slide is not None:
            where = {"slide_number": {"$eq": filter_slide}}

        kwargs = dict(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)

        search_results = []
        if results and results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                sr = SearchResult(
                    text=results["documents"][0][i],
                    metadata=results["metadatas"][0][i],
                    distance=results["distances"][0][i],
                    doc_id=results["ids"][0][i],
                )
                search_results.append(sr)

        logger.debug(f"Search '{query[:50]}' → {len(search_results)} results")
        return search_results

    def get_document_count(self) -> int:
        """Return total number of indexed documents."""
        return self.collection.count()

    def is_empty(self) -> bool:
        """Return True if no documents have been indexed."""
        return self.collection.count() == 0

    def _clear_collection(self) -> None:
        """Delete and recreate the ChromaDB collection."""
        try:
            self.client.delete_collection(COLLECTION_NAME)
            logger.info("Existing collection cleared")
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
        """
        Split text into overlapping chunks at word boundaries.

        Args:
            text: Input text to split.
            chunk_size: Maximum characters per chunk.
            overlap: Characters to overlap between consecutive chunks.

        Returns:
            List of text chunks.
        """
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            # Snap to word boundary
            if end < len(text):
                space_pos = text.rfind(" ", start, end)
                if space_pos > start:
                    end = space_pos

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - overlap

        return chunks

    def index_additional_text(
        self,
        text: str,
        metadata: dict,
    ) -> str:
        """
        Index a single text document with custom metadata.

        Useful for indexing slide text or summaries independently.

        Args:
            text: Text content to index.
            metadata: Dict of metadata to attach.

        Returns:
            Document ID.
        """
        encoder = self._get_encoder()
        embedding = encoder.encode([text], show_progress_bar=False)[0]
        doc_id = str(uuid.uuid4())

        self.collection.upsert(
            ids=[doc_id],
            documents=[text],
            embeddings=[embedding.tolist()],
            metadatas=[metadata],
        )
        return doc_id


# ─────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────
_indexer_instance: Optional[MeetingIndexer] = None


def get_indexer() -> MeetingIndexer:
    """Return module-level singleton indexer."""
    global _indexer_instance
    if _indexer_instance is None:
        _indexer_instance = MeetingIndexer()
    return _indexer_instance


# ─────────────────────────────────────────────────────────────────
# Quick standalone test
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    indexer = MeetingIndexer()
    print(f"Documents in index: {indexer.get_document_count()}")

    # Insert a test document
    doc_id = indexer.index_additional_text(
        text="We decided to increase pricing by 15% starting Q3.",
        metadata={
            "timestamp_seconds": 872.0,
            "timestamp_fmt": "14:32",
            "speaker": "Rahul",
            "slide_number": 7,
            "slide_image_path": "",
            "frame_image_path": "",
            "is_slide_transition": False,
            "chunk_idx": 0,
        },
    )
    print(f"Indexed test doc: {doc_id}")

    # Search
    results = indexer.search("pricing decision")
    print(f"\nSearch results for 'pricing decision':")
    for r in results:
        print(f"  [{r.timestamp_fmt}] [Slide {r.slide_number}] [{r.speaker}]")
        print(f"  {r.text[:100]}")
        print(f"  Distance: {r.distance:.4f}")
