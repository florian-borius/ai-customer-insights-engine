from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config.config import OPENAI_API_KEY

if OPENAI_API_KEY is None:
    raise ValueError("OPENAI_API_KEY is missing.")


# ------------------------------
# CRÉATION D'UNE CHAÎNE RAG
# ------------------------------
def create_rag_chain(
    llm_model: str,
    max_completion_tokens: int,
    retriever: BaseRetriever,
) -> Runnable[dict, dict]:
    """Crée une chaîne RAG à partir d'un llm et d'un retriever."""

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                Tu es un assistant spécialisé dans l'analyse d'avis clients.
                Réponds à la question en utilisant uniquement le contexte fourni.
                Si l'information nécessaire pour répondre à la question n'est pas présente dans le contexte, indique-le clairement.
                """
            ),
            (
                "human",
                """
                Question :
                {question}

                Contexte :
                {context}
                """
            ),
        ]
    )

    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model=llm_model,
        max_completion_tokens=max_completion_tokens,
        temperature=0,
    )

    rag_chain = (
        {
            "question": RunnablePassthrough(),   # La question est passée au retriever (qui fait : retriever.invoke(question)) mais également au prompt template grâce à "RunnablePassthrough()" qui permet de passer la question telle quelle sans modification.
            "context": retriever,
        }
        | RunnablePassthrough.assign(
            answer = prompt | llm
        )
    )

    return rag_chain