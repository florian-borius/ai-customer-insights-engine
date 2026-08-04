from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_chroma import Chroma
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_huggingface import HuggingFaceEmbeddings

from src.rag.reranker import create_reranker

from config.config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    SEARCH_TYPE,
    RETRIEVER_K,
    HUGGINGFACE_CROSSENCODER_MODEL,
    RERANKER_TOP_N,
)


# ------------------------------
# CHARGEMENT DU VECTOR STORE CHROMA ET CRÉATION D'UN RETRIEVER + RERANKER
# ------------------------------
def load_retriever(
    embedding_function: Embeddings,
    chroma_path: str,
    collection_name: str,
    search_type: str,
    retriever_k: int,
    reranker_model: str | None = None,
    reranker_top_n: int | None = None,
) -> BaseRetriever:
    """Charge le vector store Chroma et crée un retriever, avec reranker optionnel."""

    vector_store = Chroma(
        embedding_function=embedding_function,
        persist_directory=chroma_path,
        collection_name=collection_name,
    )

    # 1) Retriever initial : récupération des documents candidats
    base_retriever = vector_store.as_retriever(
        search_type=search_type,   # "similarity" : proximité des embeddings (souvent cosinus) ; "mmr" : équilibre similarité / diversité
        search_kwargs={"k": retriever_k},
    )

    # 2) Chargement du modèle de reranking
    if reranker_model is not None:
        reranker = create_reranker(
            model_name=reranker_model,
            top_n=reranker_top_n,
        )

    # 3) Retriever final : retrieval + reranking
        return ContextualCompressionRetriever(
            base_retriever=base_retriever,
            base_compressor=reranker,
        )

    return base_retriever


# ------------------------------
# PIPELINE DE CRÉATION D'UN RETRIEVER + RERANKER
# ------------------------------
def create_retriever(
    model_name: str,
    use_reranker: bool = True,
    retriever_k: int = RETRIEVER_K,
    reranker_top_n: int = RERANKER_TOP_N,
) -> BaseRetriever:
    """Pipeline de création d'un retriever avec ou sans reranker."""

    embedding_function = HuggingFaceEmbeddings(
        model_name=model_name,
    )

#    embedding_function = OpenAIEmbeddings(
#        model=model_name,
#        api_key=OPENAI_API_KEY,
#    )

    if use_reranker:
        return load_retriever(
            embedding_function=embedding_function,
            chroma_path=CHROMA_PATH,
            collection_name=COLLECTION_NAME,
            search_type=SEARCH_TYPE,
            retriever_k=retriever_k,
            reranker_model=HUGGINGFACE_CROSSENCODER_MODEL,
            reranker_top_n=reranker_top_n,
        )

    return load_retriever(
        embedding_function=embedding_function,
        chroma_path=CHROMA_PATH,
        collection_name=COLLECTION_NAME,
        search_type=SEARCH_TYPE,
        retriever_k=retriever_k,
    )