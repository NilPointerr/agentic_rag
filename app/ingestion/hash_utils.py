import hashlib
import re

WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Return a stable text form for exact-content deduplication.

    The ingestion pipeline uses this normalized representation before hashing
    documents and chunks so that harmless formatting differences, such as extra
    spaces, newlines, or casing changes, do not create new hashes.
    """
    return WHITESPACE_PATTERN.sub(" ", text.lower()).strip()


def generate_content_hash(text: str) -> str:
    """Return a SHA256 hash for normalized content.

    The hash is intended for exact duplicate detection, not semantic similarity.
    Two text values that differ only by whitespace or casing will produce the
    same hash because normalization happens first.
    """
    normalized = normalize_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
