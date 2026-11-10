from unittest.mock import MagicMock

import pytest

from src.rag.rag_chain import RAGChain


def create_rag_chain():
    chain = RAGChain.__new__(RAGChain)

    chain.retriever = MagicMock()
    chain.model_name = "test-model"
    chain.base_url = "http://localhost:11434"
    chain.min_retrieval_score = 0.025

    chain.domain_terms = {
        "jumia",
        "commande",
        "commandes",
        "annulation",
        "annuler",
        "livraison",
        "paiement",
        "retour",
        "remboursement",
        "garantie",
        "maroc",
    }

    chain.domain_phrases = {
        "annuler mon achat",
        "suivre ma commande",
    }

    return chain


def test_invoke_rejects_empty_question():
    chain = create_rag_chain()

    with pytest.raises(ValueError, match="question"):
        chain.invoke("")


def test_invoke_returns_out_of_scope():
    chain = create_rag_chain()

    result = chain.invoke("Quelle est la capitale de la France ?")

    assert result["status"] == "out_of_scope"
    assert result["sources"] == []
    assert result["retrieved_chunks"] == 0
    chain.retriever.retrieve.assert_not_called()


def test_invoke_returns_insufficient_context():
    chain = create_rag_chain()

    chain.retriever.retrieve.return_value = [
        {
            "chunk_id": "JUMIA-001-CHUNK-0001",
            "text": "Texte sans score suffisant.",
            "metadata": {"document_id": "JUMIA-001"},
            "score": 0.01,
        }
    ]

    result = chain.invoke("Comment annuler une commande sur Jumia ?")

    assert result["status"] == "insufficient_context"
    assert result["sources"] == []
    assert result["retrieved_chunks"] == 1
    assert result["used_chunks"] == 0


def test_invoke_generates_answer_with_sources():
    chain = create_rag_chain()

    chain.retriever.retrieve.return_value = [
        {
            "chunk_id": "JUMIA-001-CHUNK-0001",
            "text": "Le client peut annuler sa commande.",
            "metadata": {
                "document_id": "JUMIA-001",
                "topic": "Annulation",
                "source_url": "https://www.jumia.ma/",
            },
            "score": 0.05,
        }
    ]

    chain._generate = MagicMock(
        return_value=(
            "Le client peut annuler sa commande. "
            "[CHUNK_ID: JUMIA-001-CHUNK-0001]"
        )
    )

    result = chain.invoke("Comment annuler une commande sur Jumia ?")

    assert result["status"] == "answered"
    assert result["answer"].startswith("Le client")
    assert len(result["sources"]) == 1
    assert result["sources"][0]["chunk_id"] == "JUMIA-001-CHUNK-0001"
    assert result["retrieved_chunks"] == 1
    assert result["used_chunks"] == 1

    chain._generate.assert_called_once()


def test_filter_chunks_removes_low_scores():
    chain = create_rag_chain()

    chunks = [
        {"chunk_id": "A", "score": 0.05},
        {"chunk_id": "B", "score": 0.01},
        {"chunk_id": "C", "score": 0.03},
    ]

    filtered = chain._filter_chunks(chunks)

    assert [chunk["chunk_id"] for chunk in filtered] == ["A", "C"]


def test_build_context_includes_chunk_references():
    chain = create_rag_chain()

    context = chain._build_context([
        {
            "chunk_id": "JUMIA-001-CHUNK-0001",
            "text": "Informations sur la commande.",
            "metadata": {
                "document_id": "JUMIA-001",
                "topic": "Commandes",
            },
        }
    ])

    assert "[CHUNK_ID: JUMIA-001-CHUNK-0001]" in context
    assert "[DOCUMENT_ID: JUMIA-001]" in context
    assert "Informations sur la commande." in context