from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
#from langchain_openai import OpenAIEmbeddings

from src.rag.document_processing import prepare_chunked_documents

from config.config import (
    PROCESSED_DATA_PATH,
    HUGGINGFACE_EMBEDDING_MODEL,
#    OPENAI_EMBEDDING_MODEL,
#    OPENAI_API_KEY,
    CHROMA_PATH,
    COLLECTION_NAME,
)


# ------------------------------
# VECTORISATION DES CHUNKS ET CONSTRUCTION D'UN VECTOR STORE CHROMA
# ------------------------------
def build_vector_store(
    chunks: list[Document],
    embedding: Embeddings,
    chroma_path: str,
    collection_name: str,
) -> Chroma:
    """Vectorise les chunks et construit le vector store Chroma en intégrant également les métadonnées associées."""

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=chroma_path,
        collection_name=collection_name,
    )

    return vector_store


# ------------------------------
# PIPELINE DE CONSTRUCTION D'UN VECTOR STORE CHROMA
# ------------------------------
def main(
    input_path: str,
    model_name: str,
    chroma_path: str,
    collection_name: str,
):
    """Pipeline de construction d'un vector store Chroma."""

    chunked_documents = prepare_chunked_documents(input_path=input_path)
    
    embedding_function = HuggingFaceEmbeddings(
        model_name=model_name,
    )

#    embedding_function = OpenAIEmbeddings(
#        model=model_name,
#        api_key=OPENAI_API_KEY,
#    )

    build_vector_store(
        chunks=chunked_documents,
        embedding=embedding_function,
        chroma_path=chroma_path,
        collection_name=collection_name,
    )


# ------------------------------
# EXÉCUTION
# ------------------------------
if __name__ == "__main__":
    main(
        input_path=PROCESSED_DATA_PATH,
        model_name=HUGGINGFACE_EMBEDDING_MODEL,
        chroma_path=CHROMA_PATH,
        collection_name=COLLECTION_NAME,
    )