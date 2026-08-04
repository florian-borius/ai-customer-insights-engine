import pytest

from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

from src.rag.reranker import create_reranker


@pytest.fixture
def reranker():
    return create_reranker(
        model_name="BAAI/bge-reranker-v2-m3",
        top_n=3,
    )


# ------------------------------
# TESTS DE CREATE_RERANKER()
# ------------------------------

def test_create_reranker_returns_cross_encoder_reranker(reranker):
    assert isinstance(reranker, CrossEncoderReranker)


def test_create_reranker_sets_top_n(reranker):
    assert reranker.top_n == 3


def test_create_reranker_loads_model(reranker):
    assert reranker.model is not None