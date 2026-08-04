import pytest

from langchain_classic.retrievers import ContextualCompressionRetriever

from src.rag.retriever import create_retriever


@pytest.fixture
def retriever():
    return create_retriever(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )


# ------------------------------
# TESTS DE CREATE_RETRIEVER()
# ------------------------------

def test_create_retriever_returns_contextual_compression_retriever(retriever):
    assert isinstance(retriever, ContextualCompressionRetriever)


def test_create_retriever_has_base_retriever(retriever):
    assert retriever.base_retriever is not None


def test_create_retriever_has_reranker(retriever):
    assert retriever.base_compressor is not None


def test_retriever_returns_documents(retriever):

    docs = retriever.invoke(
        "problèmes liés à l'application mobile"
    )

    assert len(docs) > 0
    assert docs[0].page_content
    assert docs[0].metadata