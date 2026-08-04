import streamlit as st

from src.rag.retriever import create_retriever
from src.rag.rag_chain import create_rag_chain

from config.config import (
    HUGGINGFACE_EMBEDDING_MODEL,
    LLM_MODEL,
    MAX_COMPLETION_TOKENS,
)


@st.cache_resource
def initialize_rag_chain():
    """..."""

    retriever = create_retriever(
        model_name=HUGGINGFACE_EMBEDDING_MODEL,
        use_reranker=True,
    )

    rag_chain = create_rag_chain(
        llm_model=LLM_MODEL,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        retriever=retriever,
    )
                
    return rag_chain


rag_chain = initialize_rag_chain()

st.title("AI Customer Insights Engine")

question = st.text_input("Posez une question sur les avis clients :")

if question:
    if len(question) > 1000:
        st.error("Question trop longue.")
    else:
        response = rag_chain.invoke(question)
        st.write(response["answer"].content)

        for doc in response["context"]:
            st.write(doc.page_content)
            st.write(doc.metadata)



#           streamlit run src/app/app.py
# python -m streamlit run src/app/app.py