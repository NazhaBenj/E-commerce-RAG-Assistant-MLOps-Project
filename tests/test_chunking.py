import pytest

from src.rag.chunking import (
    split_text,
    create_document_chunks,
)


def test_split_text_empty_input():
    assert split_text("") == []
    assert split_text("   ") == []


def test_split_text_returns_chunks():
    text = (
        "La livraison est disponible au Maroc. "
        "Le délai dépend de la destination. "
        "Le client peut suivre sa commande en ligne."
    )

    chunks = split_text(text, chunk_size=80, chunk_overlap=20)

    assert len(chunks) > 0
    assert all(isinstance(chunk, str) for chunk in chunks)
    assert all(chunk.strip() for chunk in chunks)


def test_split_text_respects_maximum_size():
    text = " ".join(
        ["livraison"] * 100
    )

    chunks = split_text(text, chunk_size=50, chunk_overlap=10)

    assert all(len(chunk) <= 50 for chunk in chunks)


def test_split_text_invalid_chunk_size():
    with pytest.raises(ValueError):
        split_text("Texte de test", chunk_size=0)


def test_split_text_invalid_overlap():
    with pytest.raises(ValueError):
        split_text("Texte de test", chunk_size=100, chunk_overlap=100)

    with pytest.raises(ValueError):
        split_text("Texte de test", chunk_size=100, chunk_overlap=-1)


def test_create_document_chunks_metadata():
    text = (
        "La commande peut être annulée. "
        "Le client doit consulter son compte."
    )

    chunks = create_document_chunks(
        text=text,
        document_id="JUMIA-001",
        category="commandes",
        chunk_size=100,
        chunk_overlap=20,
    )

    assert len(chunks) > 0

    for index, chunk in enumerate(chunks):
        assert chunk["chunk_id"] == f"JUMIA-001-CHUNK-{index:04d}"
        assert chunk["document_id"] == "JUMIA-001"
        assert chunk["category"] == "commandes"
        assert chunk["chunk_index"] == index
        assert isinstance(chunk["text"], str)
        assert chunk["text"].strip()