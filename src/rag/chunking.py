import re
from typing import Any


def _split_into_units(text: str, chunk_size: int) -> list[str]:
    """Split text into paragraphs, then sentences when necessary."""

    paragraphs = re.split(r"\n\s*\n", text)

    units = []

    for paragraph in paragraphs:
        paragraph = re.sub(r"\s+", " ", paragraph).strip()

        if not paragraph:
            continue

        # Preserve sentence boundaries as much as possible.
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)

        for sentence in sentences:
            sentence = sentence.strip()

            if not sentence:
                continue

            # A sentence longer than chunk_size must be split by words.
            if len(sentence) > chunk_size:
                words = sentence.split()
                fragments = []
                current = ""

                for word in words:
                    candidate = f"{current} {word}".strip()

                    if len(candidate) <= chunk_size:
                        current = candidate
                    else:
                        if current:
                            fragments.append(current)

                        # Handle an exceptionally long word or URL.
                        if len(word) > chunk_size:
                            for i in range(0, len(word), chunk_size):
                                fragments.append(word[i:i + chunk_size])
                            current = ""
                        else:
                            current = word

                if current:
                    fragments.append(current)

                units.extend(fragments)
            else:
                units.append(sentence)

    return units


def split_text(
    text: str,
    chunk_size: int = 300,
    chunk_overlap: int = 50,
) -> list[str]:
    """Split text into sentence-aware chunks with overlap."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

    text = text.strip()

    if not text:
        return []

    units = _split_into_units(text, chunk_size)

    chunks = []
    current_units = []

    for unit in units:
        candidate = "\n".join(current_units + [unit])

        if len(candidate) <= chunk_size:
            current_units.append(unit)
            continue

        # Save the current chunk.
        if current_units:
            chunks.append("\n".join(current_units))

        # Keep the last complete sentences for overlap.
        overlap_units = []
        overlap_length = 0

        for previous_unit in reversed(current_units):
            additional_length = len(previous_unit)

            if overlap_units:
                additional_length += 1

            if overlap_length + additional_length > chunk_overlap:
                break

            overlap_units.insert(0, previous_unit)
            overlap_length += additional_length

        current_units = overlap_units

        # Ensure the next sentence fits with the overlap.
        while current_units and len("\n".join(current_units + [unit])) > chunk_size:
            current_units.pop(0)

        current_units.append(unit)

    if current_units:
        chunks.append("\n".join(current_units))

    return chunks


def create_document_chunks(
    text: str,
    document_id: str,
    category: str,
    chunk_size: int = 300,
    chunk_overlap: int = 50,
) -> list[dict[str, Any]]:
    """Create chunks with source metadata."""

    chunks = split_text(text, chunk_size, chunk_overlap)

    return [
        {
            "chunk_id": f"{document_id}-CHUNK-{index:04d}",
            "document_id": document_id,
            "category": category,
            "chunk_index": index,
            "text": chunk,
        }
        for index, chunk in enumerate(chunks)
    ]