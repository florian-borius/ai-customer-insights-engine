from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker


# ------------------------------
# CRÉATION D'UN RERANKER
# ------------------------------
def create_reranker(
    model_name: str,
    top_n: int,
) -> CrossEncoderReranker:
    """Crée un reranker."""

    crossencoder = HuggingFaceCrossEncoder(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
    )

    reranker = CrossEncoderReranker(
        model=crossencoder,
        top_n=top_n,
    )

    return reranker