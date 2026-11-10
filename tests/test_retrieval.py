from unittest.mock import MagicMock

import pytest

from src.rag.retriever import Retriever


def create_retriever():
    retriever = Retriever.__new__(Retriever)

    retriever.top_k = 3
    retriever.embedding_model = MagicMock()
    retriever.vector_store = MagicMock()

    return retriever


def test_retrieve_returns_formatted_results():
    retriever = create_retriever()

    retriever.embedding_model.embed_query.return_value = [0.1, 0.2]

    retriever.vector_store.search.return_value = {
        "ids": [["JUMIA-001-CHUNK-0001"]],
        "documents": [["Informations sur les commandes."]],
        "metadatas": [[{
            "document_id": "JUMIA-001",
            "topic": "commandes",
        }]],
        "distances": [[0.15]],
    }

    results = retriever.retrieve("Comment suivre ma commande ?")

    assert len(results) == 1
    assert results[0]["chunk_id"] == "JUMIA-001-CHUNK-0001"
    assert results[0]["text"] == "Informations sur les commandes."
    assert results[0]["metadata"]["document_id"] == "JUMIA-001"
    assert results[0]["distance"] == 0.15

    retriever.embedding_model.embed_query.assert_called_once()
    retriever.vector_store.search.assert_called_once()


def test_retrieve_rejects_empty_question():
    retriever = create_retriever()

    with pytest.raises(ValueError, match="question"):
        retriever.retrieve("")

    with pytest.raises(ValueError, match="question"):
        retriever.retrieve("   ")


def test_retrieve_returns_empty_list():
    retriever = create_retriever()

    retriever.embedding_model.embed_query.return_value = [0.1, 0.2]

    retriever.vector_store.search.return_value = {
        "ids": [[]],
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }

    results = retriever.retrieve("Question sans résultat")

    assert results == []


def test_retrieve_handles_multiple_results():
    retriever = create_retriever()

    retriever.embedding_model.embed_query.return_value = [0.1, 0.2]

    retriever.vector_store.search.return_value = {
        "ids": [
            [
                "JUMIA-001-CHUNK-0001",
                "JUMIA-002-CHUNK-0001",
            ]
        ],
        "documents": [[
            "Texte sur les commandes.",
            "Texte sur la livraison.",
        ]],
        "metadatas": [[
            {"document_id": "JUMIA-001"},
            {"document_id": "JUMIA-002"},
        ]],
        "distances": [[0.1, 0.2]],
    }

    results = retriever.retrieve("Question Jumia")

    assert len(results) == 2
    assert results[0]["distance"] == 0.1
    assert results[1]["distance"] == 0.2