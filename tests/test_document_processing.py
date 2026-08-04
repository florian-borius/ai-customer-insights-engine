import pandas as pd
from langchain_core.documents import Document

from src.rag.document_processing import (
    filter_long_reviews,
    create_documents,
    chunk_documents,
    prepare_chunked_documents,
)


# ------------------------------
# TEST DE FILTER_LONG_REVIEWS()
# ------------------------------

def test_filter_long_reviews():
    df = pd.DataFrame(
        {
            "review": [
                "court",
                "a" * 10,
                "b" * 20,
            ]
        }
    )

    filtered_df = filter_long_reviews(
        df=df,
        review_max_length=10,
    )

    assert len(filtered_df) == 2
    assert "court" in filtered_df["review"].values
    assert "a" * 10 in filtered_df["review"].values
    assert "b" * 20 not in filtered_df["review"].values


# ------------------------------
# TEST DE CREATE_DOCUMENTS()
# ------------------------------

def test_create_documents():
    df = pd.DataFrame(
        {
            "review_id": [1],
            "review": ["Très bonne expérience avec cette banque."],
            "title": ["Bon service"],
            "bank": ["Bank A"],
            "rating": [5],
            "publication_date": [
                pd.Timestamp("2025-01-01")
            ],
            "experience_date": [
                pd.Timestamp("2024-12-01")
            ],
        }
    )

    documents = create_documents(
        df=df,
        dataset_name="test_dataset",
    )

    assert len(documents) == 1

    document = documents[0]

    assert isinstance(document, Document)
    assert document.page_content == "Très bonne expérience avec cette banque."

    assert document.metadata["review_id"] == 1
    assert document.metadata["title"] == "Bon service"
    assert document.metadata["bank"] == "Bank A"
    assert document.metadata["rating"] == 5
    assert document.metadata["publication_date"] == "2025-01-01T00:00:00"
    assert document.metadata["experience_date"] == "2024-12-01T00:00:00"
    assert document.metadata["dataset"] == "test_dataset"


# ------------------------------
# TEST DE CHUNK_DOCUMENTS()
# ------------------------------

def test_chunk_documents():
    documents = [
        Document(
            page_content="a " * 100,
            metadata={
                "review_id": 1,
                "bank": "Bank A",
            },
        )
    ]

    chunked_documents = chunk_documents(
        documents=documents,
        separators=[" "],
        chunk_size=50,
        chunk_overlap=10,
    )

    assert len(chunked_documents) > 1

    for chunk in chunked_documents:
        assert isinstance(chunk, Document)
        assert chunk.metadata["review_id"] == 1
        assert chunk.metadata["bank"] == "Bank A"


# ------------------------------
# TEST DE PREPARE_CHUNKED_DOCUMENTS()
# (test d'intégration)
# ------------------------------

def test_prepare_chunked_documents(tmp_path):
    df = pd.DataFrame(
        {
            "review_id": [1],
            "review": [
                "Très bonne banque. " * 50
            ],
            "title": ["Service"],
            "bank": ["Bank A"],
            "rating": [5],
            "publication_date": [
                pd.Timestamp("2025-01-01")
            ],
            "experience_date": [
                pd.Timestamp("2024-12-01")
            ],
        }
    )

    input_file = tmp_path / "test_reviews.parquet"

    df.to_parquet(input_file)

    documents = prepare_chunked_documents(
        input_path=str(input_file)
    )

    assert len(documents) > 0

    assert documents[0].metadata["dataset"] == "test_reviews"