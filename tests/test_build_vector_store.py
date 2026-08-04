from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings

from src.rag.build_vector_store import build_vector_store


# ------------------------------
# TEST DE BUILD_VECTOR_STORE()
# ------------------------------

def test_build_vector_store(tmp_path):
    # Arrange : documents fictifs
    chunks = [
        Document(
            page_content="Très bonne expérience avec cette banque.",
            metadata={
                "rating": 5,
                "bank": "Boursobank",
            },
        ),
        Document(
            page_content="Application mobile décevante.",
            metadata={
                "rating": 2,
                "bank": "Boursobank",
            },
        ),
    ]

    # Faux modèle d'embedding
    embedding = FakeEmbeddings(size=384)

    # Dossier temporaire créé automatiquement par pytest
    chroma_path = str(tmp_path)

    # Act
    vector_store = build_vector_store(
        chunks=chunks,
        embedding=embedding,
        chroma_path=chroma_path,
        collection_name="test_collection",
    )

    # Assert
    assert vector_store is not None

    # Vérifie que les documents ont bien été indexés
    results = vector_store.similarity_search(
        "application mobile",
        k=2,
    )

    assert len(results) == 2

    # Vérifie que les métadonnées sont conservées
    assert results[0].metadata["bank"] == "Boursobank"