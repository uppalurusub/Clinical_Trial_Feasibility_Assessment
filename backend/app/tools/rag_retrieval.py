
"""
Shared RAG retrieval utilities for the Multi-Agent Clinical Trial
Feasibility Assessment project.

Capabilities:
- Load a local JSON evidence corpus.
- Use an injected Chroma-compatible vector store when available.
- Fall back to lexical retrieval for local POC execution.
- Return normalized evidence records with source metadata and scores.
- Apply optional metadata filters.
- Avoid logging retrieved document content.

Expected JSON corpus examples:
[
    {
        "id": "EVID-001",
        "text": "The study requires 240 participants across the target region.",
        "metadata": {
            "source": "protocol_summary.pdf",
            "section": "Enrollment",
            "study_id": "STUDY-DEMO-001"
        }
    }
]
"""

from __future__ import annotations

import json

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from app.config.settings import settings
logger = settings.get_logger()

DEFAULT_CORPUS_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "evidence_corpus.json"
)

DEFAULT_TOP_K = 5
MAX_TOP_K = 20

_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+(?:[-'][a-zA-Z0-9]+)*")


class RetrievalError(RuntimeError):
    """Raised when evidence retrieval cannot be completed."""


def _tokenize(text: str) -> list[str]:
    """Normalize text into simple searchable tokens."""
    return [
        token.lower()
        for token in _TOKEN_PATTERN.findall(text or "")
        if len(token) > 1
    ]


def _as_mapping(value: Any) -> dict[str, Any]:
    """Convert supported mapping-like values to a plain dictionary."""
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _normalize_document(
    document: Mapping[str, Any],
    index: int,
) -> Optional[dict[str, Any]]:
    """Normalize a JSON corpus record into a consistent evidence shape."""
    record = dict(document)
    metadata = _as_mapping(record.get("metadata"))

    text = (
        record.get("text")
        or record.get("page_content")
        or record.get("content")
        or record.get("chunk_text")
    )

    if not isinstance(text, str) or not text.strip():
        return None

    evidence_id = (
        record.get("id")
        or record.get("evidence_id")
        or metadata.get("evidence_id")
        or f"LOCAL-EVID-{index:05d}"
    )

    source = (
        record.get("source")
        or metadata.get("source")
        or metadata.get("file_name")
        or metadata.get("document_name")
        or "unknown"
    )

    metadata.setdefault("source", source)
    metadata.setdefault("evidence_id", str(evidence_id))

    return {
        "evidence_id": str(evidence_id),
        "text": text.strip(),
        "source": str(source),
        "metadata": metadata,
    }


def load_json_corpus(
    corpus_path: str | Path = DEFAULT_CORPUS_PATH,
) -> list[dict[str, Any]]:
    """
    Load and normalize evidence records from a JSON file.

    Supported top-level formats:
      - A list of records.
      - An object containing an "evidence" or "documents" list.
    """
    path = Path(corpus_path)

    if not path.exists():
        logger.warning("RAG corpus file not found: %s", path)
        return []

    try:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise RetrievalError(
            f"Unable to load evidence corpus from {path}"
        ) from exc

    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = payload.get("evidence", payload.get("documents", []))
    else:
        raise RetrievalError(
            "Evidence corpus must be a list or an object containing "
            "'evidence' or 'documents'."
        )

    if not isinstance(records, list):
        raise RetrievalError("Evidence corpus records must be a list.")

    normalized: list[dict[str, Any]] = []

    for index, item in enumerate(records, start=1):
        if not isinstance(item, Mapping):
            logger.warning(
                "Skipping malformed evidence record at index %s",
                index,
            )
            continue

        record = _normalize_document(item, index)
        if record is not None:
            normalized.append(record)

    logger.info("Loaded %d evidence records", len(normalized))
    return normalized


def _matches_filters(
    metadata: Mapping[str, Any],
    filters: Optional[Mapping[str, Any]],
) -> bool:
    """Apply exact-match metadata filters."""
    if not filters:
        return True

    for key, expected in filters.items():
        if metadata.get(key) != expected:
            return False

    return True


def _lexical_score(
    query_tokens: Sequence[str],
    document_text: str,
) -> float:
    """
    Calculate a lightweight lexical relevance score in [0, 1].

    This is a POC ranking heuristic, not a semantic similarity score.
    """
    if not query_tokens:
        return 0.0

    document_tokens = _tokenize(document_text)
    if not document_tokens:
        return 0.0

    query_counts = Counter(query_tokens)
    document_counts = Counter(document_tokens)

    matched_terms = set(query_counts).intersection(document_counts)
    if not matched_terms:
        return 0.0

    query_coverage = (
        sum(query_counts[token] for token in matched_terms)
        / max(sum(query_counts.values()), 1)
    )

    document_coverage = (
        sum(min(query_counts[token], document_counts[token])
            for token in matched_terms)
        / max(len(document_tokens), 1)
    )

    # Favor query coverage while limiting the influence of long documents.
    score = (0.85 * query_coverage) + (0.15 * document_coverage)
    return round(min(max(score, 0.0), 1.0), 4)


class RAGRetriever:
    """
    Evidence retriever with optional vector-store integration.

    Args:
        corpus_path: Local JSON corpus path.
        documents: Optional preloaded evidence records.
        vector_store: Optional Chroma-compatible store exposing
            similarity_search_with_relevance_scores(query, k=...).

    If vector_store is omitted, lexical retrieval is used.
    """

    def __init__(
        self,
        corpus_path: str | Path = DEFAULT_CORPUS_PATH,
        documents: Optional[Sequence[Mapping[str, Any]]] = None,
        vector_store: Any = None,
    ) -> None:
        self.corpus_path = Path(corpus_path)
        self.vector_store = vector_store

        if documents is not None:
            self.documents = [
                normalized
                for index, document in enumerate(documents, start=1)
                if (
                    normalized := _normalize_document(document, index)
                ) is not None
            ]
        else:
            self.documents = load_json_corpus(self.corpus_path)

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        filters: Optional[Mapping[str, Any]] = None,
        min_score: float = 0.0,
    ) -> list[dict[str, Any]]:
        """
        Retrieve relevant evidence.

        Returns:
            A list of records containing evidence_id, text, source,
            metadata, score, and retrieval_method.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise ValueError("top_k must be an integer")

        if not 1 <= top_k <= MAX_TOP_K:
            raise ValueError(
                f"top_k must be between 1 and {MAX_TOP_K}"
            )

        if (
            isinstance(min_score, bool)
            or not isinstance(min_score, (int, float))
            or not math.isfinite(min_score)
            or not 0.0 <= min_score <= 1.0
        ):
            raise ValueError("min_score must be between 0 and 1")

        if self.vector_store is not None:
            try:
                results = self._retrieve_vector(
                    query=query.strip(),
                    top_k=top_k,
                    filters=filters,
                )
            except Exception as exc:
                # Do not silently fall back when the configured vector
                # store fails; that could hide a production outage.
                raise RetrievalError(
                    "Configured vector-store retrieval failed."
                ) from exc
        else:
            results = self._retrieve_lexical(
                query=query.strip(),
                top_k=top_k,
                filters=filters,
            )

        return [
            result
            for result in results
            if result["score"] >= float(min_score)
        ][:top_k]

    def _retrieve_lexical(
        self,
        query: str,
        top_k: int,
        filters: Optional[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        query_tokens = _tokenize(query)
        results: list[dict[str, Any]] = []

        for document in self.documents:
            metadata = document["metadata"]

            if not _matches_filters(metadata, filters):
                continue

            score = _lexical_score(query_tokens, document["text"])
            if score <= 0:
                continue

            results.append(
                {
                    **document,
                    "score": score,
                    "retrieval_method": "lexical",
                }
            )

        results.sort(
            key=lambda item: (
                item["score"],
                item["evidence_id"],
            ),
            reverse=True,
        )
        return results[:top_k]

    def _retrieve_vector(
        self,
        query: str,
        top_k: int,
        filters: Optional[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Adapt a Chroma-compatible vector store to the shared output format.

        The supplied vector store must already be configured with its
        embedding function and collection.
        """
        search_kwargs: dict[str, Any] = {"k": top_k}

        if filters:
            # Chroma expects metadata filtering under "filter" only when
            # using a wrapper that explicitly supports that argument.
            # Apply filters after retrieval for broader compatibility.
            logger.debug("Metadata filters will be applied after retrieval")

        matches = self.vector_store.similarity_search_with_relevance_scores(
            query,
            **search_kwargs,
        )

        results: list[dict[str, Any]] = []

        for index, match in enumerate(matches, start=1):
            if not isinstance(match, (tuple, list)) or len(match) != 2:
                continue

            document, raw_score = match

            metadata = _as_mapping(getattr(document, "metadata", {}))
            text = getattr(document, "page_content", "")

            if not isinstance(text, str) or not text.strip():
                continue

            if not _matches_filters(metadata, filters):
                continue

            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                continue

            if not math.isfinite(score):
                continue

            # Relevance scores are expected to be normalized by the
            # vector-store integration. Clamp defensively for consistency.
            score = min(max(score, 0.0), 1.0)

            evidence_id = (
                metadata.get("evidence_id")
                or metadata.get("chunk_id")
                or metadata.get("id")
                or f"VECTOR-EVID-{index:05d}"
            )

            source = (
                metadata.get("source")
                or metadata.get("file_name")
                or metadata.get("document_name")
                or "unknown"
            )

            metadata.setdefault("source", source)
            metadata.setdefault("evidence_id", str(evidence_id))

            results.append(
                {
                    "evidence_id": str(evidence_id),
                    "text": text.strip(),
                    "source": str(source),
                    "metadata": metadata,
                    "score": round(score, 4),
                    "retrieval_method": "vector",
                }
            )

        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:top_k]


# Module-level convenience API for specialist agents.
_default_retriever: Optional[RAGRetriever] = None


def configure_retriever(
    retriever: RAGRetriever,
) -> None:
    """Configure the shared retriever during application startup."""
    global _default_retriever
    _default_retriever = retriever


def retrieve_evidence(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    filters: Optional[Mapping[str, Any]] = None,
    min_score: float = 0.0,
) -> list[dict[str, Any]]:
    """
    Retrieve evidence using the configured retriever.

    Lazy initialization uses the local JSON corpus and lexical ranking.
    """
    global _default_retriever

    if _default_retriever is None:
        _default_retriever = RAGRetriever()

    return _default_retriever.retrieve(
        query=query,
        top_k=top_k,
        filters=filters,
        min_score=min_score,
    )
