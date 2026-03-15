"""Utility functions for Agentic RAG Assistant."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional


def generate_trace_id() -> str:
    """Generate a unique trace ID."""
    return f"trace_{uuid.uuid4().hex[:16]}"


def generate_session_id() -> str:
    """Generate a unique session ID."""
    return f"session_{uuid.uuid4().hex[:8]}"


def hash_text(text: str) -> str:
    """Generate SHA256 hash of text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def format_timestamp(dt: Optional[datetime] = None) -> str:
    """Format timestamp for logging."""
    if dt is None:
        dt = datetime.now()
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def truncate_text(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """Truncate text to max length."""
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple dictionaries."""
    result = {}
    for d in dicts:
        result.update(d)
    return result


def safe_json_serialize(obj: Any) -> str:
    """Safely serialize object to JSON."""
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return str(obj)


def calculate_latency(start_time: float) -> float:
    """Calculate latency in milliseconds."""
    return (time.monotonic() - start_time) * 1000


class Timer:
    """Context manager for timing operations."""

    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time: Optional[float] = None
        self.elapsed_ms: Optional[float] = None

    def __enter__(self) -> "Timer":
        self.start_time = time.monotonic()
        return self

    def __exit__(self, *args) -> None:
        self.elapsed_ms = calculate_latency(self.start_time)


def deduplicate_chunks(chunks: List[Any], key_field: str = "id") -> List[Any]:
    """Deduplicate chunks by a key field."""
    seen = set()
    result = []
    for chunk in chunks:
        key = getattr(chunk, key_field, None) or chunk.get(key_field)
        if key and key not in seen:
            seen.add(key)
            result.append(chunk)
    return result


def sort_chunks_by_score(chunks: List[Any], descending: bool = True) -> List[Any]:
    """Sort chunks by score."""
    return sorted(
        chunks, key=lambda c: getattr(c, "score", 0) or c.get("score", 0), reverse=descending
    )
