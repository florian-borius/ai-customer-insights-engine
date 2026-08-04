from pathlib import Path

import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.config import (
    REVIEW_MAX_LENGTH,
    SEPARATORS,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


# ------------------------------
# FILTRAGE DES AVIS CLIENTS AU-DELÀ D'UNE CERTAINE LONGUEUR
# ------------------------------
def filter_long_reviews(
    df: pd.DataFrame,
    review_max_length: int,
) -> pd.DataFrame:
    """Filtre les avis clients au-delà d'une certaine longueur."""

    return df[df["review"].str.len() <= review_max_length]


# ----------------------------
# CRÉATION DES DOCUMENTS LANGCHAIN
# ----------------------------
def create_documents(
    df: pd.DataFrame,
    dataset_name: str,
) -> list[Document]:
    """Crée les documents LangChain à partir des avis clients et de leurs métadonnées associées."""

    documents = []

    for row in df.itertuples():

        document = Document(
            page_content=row.review,
            metadata={
                "title": row.title,
                "bank": row.bank,
                "rating": row.rating,
                "publication_date": row.publication_date.isoformat(),
                "experience_date": row.experience_date.isoformat(),
                "review_id": row.review_id,
                "dataset": dataset_name,
            }
        )

        documents.append(document)

    return documents


# ----------------------------
# DÉCOUPAGE EN CHUNKS
# ----------------------------
def chunk_documents(
    documents: list[Document],
    separators: list[str],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Découpe le contenu des documents en chunks et conserve leurs métadonnées associées."""

    splitter = RecursiveCharacterTextSplitter(
        separators=separators,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        keep_separator="end",
    )

    chunked_documents = splitter.split_documents(documents)

    return chunked_documents


# ------------------------------
# PIPELINE DE PRÉPARATION DES DOCUMENTS CHUNKÉS
# ------------------------------
def prepare_chunked_documents(input_path: Path | str) -> list[Document]:
    """Pipeline de préparation des documents chunkés."""

    df = pd.read_parquet(input_path)

    dataset_name = Path(input_path).stem

    filtered_df = filter_long_reviews(
        df=df,
        review_max_length=REVIEW_MAX_LENGTH,
    )

    documents = create_documents(
        df=filtered_df,
        dataset_name=dataset_name,
    )

    chunked_documents = chunk_documents(
        documents=documents,
        separators=SEPARATORS,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    return chunked_documents