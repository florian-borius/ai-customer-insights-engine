from pathlib import Path
import sys
from datetime import datetime

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.retriever import create_retriever
from src.rag.rag_chain import create_rag_chain

from config.config import (
    HUGGINGFACE_EMBEDDING_MODEL,
    LLM_MODEL,
    MAX_COMPLETION_TOKENS,
)


# ------------------------------
# CONFIGURATION DE LA PAGE
# ------------------------------

st.set_page_config(
    page_title="AI Customer Insights Engine",
    page_icon="🔎",
    layout="centered",
)


# ------------------------------
# CSS
# ------------------------------

st.markdown(
    """
    <style>

    /* --------------------------------
       TITRE PRINCIPAL
    -------------------------------- */

    h1 {
        margin-bottom: 1.5rem;
    }


    /* --------------------------------
       LIBELLÉ DU CHAMP DE QUESTION
    -------------------------------- */

    .question-label {
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.45rem;
    }


    /* --------------------------------
       CHAMP DE SAISIE
    -------------------------------- */

    div[data-baseweb="input"] {
        border: 1px solid #d1d5db !important;
        border-radius: 8px !important;
        box-shadow: none !important;
        transition: border-color 0.2s ease,
                    box-shadow 0.2s ease;
    }

    div[data-baseweb="input"]:focus-within {
        border: 1px solid #9ca3af !important;
        box-shadow: 0 0 0 1px #9ca3af !important;
    }

    div[data-baseweb="input"] > div {
        border: none !important;
        box-shadow: none !important;
    }


    /* --------------------------------
       FORMULAIRE
    -------------------------------- */

    div[data-testid="stForm"] {
        border: none;
        padding: 0;
    }


    /* --------------------------------
       BOUTON ANALYSER
    -------------------------------- */

    div.stFormSubmitButton > button {
        height: 42px;
        border-radius: 8px;
        font-weight: 600;
        margin-top: 0;
    }


    /* --------------------------------
       EXEMPLES DE QUESTIONS
    -------------------------------- */

    .examples {
        font-size: 0.8rem;
        color: #9ca3af;
        margin-top: 0.15rem;
        line-height: 1.5;
    }

    .examples strong {
        color: #6b7280;
        font-weight: 600;
    }


    /* --------------------------------
       TITRES DE SECTIONS
    -------------------------------- */

    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 1.2rem;
        margin-bottom: 0.6rem;
    }


    /* --------------------------------
       CADRE DE RÉPONSE
    -------------------------------- */

    .answer-box {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 0.9rem 1rem;
        margin-top: 0.6rem;
        margin-bottom: 1.2rem;
        background-color: #fafafa;
        font-size: 0.95rem;
        line-height: 1.55;
    }


    /* --------------------------------
       CONTENU DU CONTEXTE
    -------------------------------- */

    .context-content {
        font-size: 0.95rem;
        line-height: 1.5;
        margin-bottom: 0.7rem;
    }


    /* --------------------------------
       SÉPARATION MÉTADONNÉES
    -------------------------------- */

    .metadata-separator {
        border-top: 1px solid #e5e7eb;
        margin: 0.4rem 0 0.8rem 0;
    }


    /* --------------------------------
       TITRES DES BLOCS DE MÉTADONNÉES
    -------------------------------- */

    .metadata-section-title {
        font-size: 0.8rem;
        font-weight: 600;
        text-decoration: underline;
        color: #6b7280;
        margin-top: 0.7rem;
        margin-bottom: 0.3rem;
    }


    /* --------------------------------
       LISTES DE MÉTADONNÉES
    -------------------------------- */

    .metadata-list {
        font-size: 0.8rem;
        line-height: 1.45;
        margin-bottom: 0.7rem;
        color: #6b7280;
    }


    /* --------------------------------
       MÉTADONNÉES TECHNIQUES
    -------------------------------- */

    .technical-metadata {
        color: #9ca3af;
    }


    .metadata-section-title.technical {
        color: #9ca3af;
        margin-top: 0.9rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------
# INITIALISATION D'UNE CHAÎNE RAG
# ------------------------------

@st.cache_resource
def initialize_rag_chain():
    """Initialise une chaîne RAG."""

    retriever = create_retriever(
        model_name=HUGGINGFACE_EMBEDDING_MODEL,
        use_reranker=False,
    )

    rag_chain = create_rag_chain(
        llm_model=LLM_MODEL,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        retriever=retriever,
    )

    return rag_chain


rag_chain = initialize_rag_chain()


# ------------------------------
# INTERFACE
# ------------------------------

st.title("AI Customer Insights Engine")


st.markdown(
    '<div class="question-label">'
    "Posez une question sur les avis clients :"
    "</div>",
    unsafe_allow_html=True,
)


# ------------------------------
# CHAMP DE QUESTION + BOUTON
# ------------------------------

with st.form("rag_form"):

    col_question, col_button = st.columns([5, 1])

    with col_question:

        question = st.text_input(
            "Question",
            label_visibility="collapsed",
        )

    with col_button:

        analyze = st.form_submit_button(
            "Analyser",
            use_container_width=True,
        )


# ------------------------------
# EXEMPLES DE QUESTIONS
# ------------------------------

st.markdown(
    """
    <div class="examples">
        <strong><u>Exemples :</u></strong>
        Quels types de problèmes rencontrent les clients avec le service client ?
        • Comment les clients décrivent-ils leur expérience lors de l'ouverture d'un compte ?
        • Quels sont les points forts des banques selon les clients ?
    </div>
    """,
    unsafe_allow_html=True,
)


# ------------------------------
# LANCEMENT DE L'ANALYSE
# ------------------------------

st.markdown(
    '<div style="margin-top: 1rem;"></div>',
    unsafe_allow_html=True,
)


if analyze:

    question = question.strip()


    # Vérification longueur minimale
    if len(question) < 10:

        st.error(
            "Votre question doit contenir au moins 10 caractères."
        )


    # Vérification longueur maximale
    elif len(question) > 100:

        st.error(
            "Votre question est trop longue ; "
            "veuillez la reformuler de manière plus concise."
        )


    else:

        # Loader pendant le traitement
        with st.spinner(
            "Analyse des avis clients en cours..."
        ):

            response = rag_chain.invoke(question)


        # ------------------------------
        # RÉPONSE
        # ------------------------------

        st.markdown(
            '<div class="section-title">Réponse</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="answer-box">
                {response["answer"].content}
            </div>
            """,
            unsafe_allow_html=True,
        )


        # ------------------------------
        # CONTEXTES
        # ------------------------------

        st.markdown(
            '<div class="section-title">Contextes utilisés</div>',
            unsafe_allow_html=True,
        )


        for i, doc in enumerate(
            response["context"],
            start=1,
        ):

            with st.expander(
                f"Contexte {i}",
                expanded=False,
            ):

                # ------------------------------
                # CONTENU DU CONTEXTE
                # ------------------------------

                st.markdown(
                    f'<div class="context-content">'
                    f'"{doc.page_content}"'
                    f'</div>',
                    unsafe_allow_html=True,
                )


                # ------------------------------
                # SÉPARATION
                # ------------------------------

                st.markdown(
                    '<div class="metadata-separator"></div>',
                    unsafe_allow_html=True,
                )


                # ------------------------------
                # INFORMATIONS SUR L'AVIS
                # ------------------------------

                st.markdown(
                    '<div class="metadata-section-title">'
                    "Informations sur l'avis :"
                    "</div>",
                    unsafe_allow_html=True,
                )


                # Nom de la banque
                bank_names = {
                    "hellobank": "Hello bank!",
                    "fortuneo": "Fortuneo",
                    "monabanq": "Monabanq",
                    "boursobank": "BoursoBank",
                }

                bank = doc.metadata.get(
                    "bank",
                    "N/A",
                )

                bank_display = bank_names.get(
                    bank,
                    bank,
                )


                # Date de publication
                publication_date = doc.metadata.get(
                    "publication_date"
                )

                if publication_date:

                    publication_date = datetime.fromisoformat(
                        publication_date
                    ).strftime("%d/%m/%Y")

                else:

                    publication_date = "N/A"


                # Informations générales
                st.markdown(
                    f"""
<div class="metadata-list">

- **Titre :** {doc.metadata.get("title", "N/A")}
- **Note :** {doc.metadata.get("rating", "N/A")}/5
- **Date de publication :** {publication_date}
- **Banque :** {bank_display}

</div>
""",
                    unsafe_allow_html=True,
                )


                # ------------------------------
                # INFORMATIONS TECHNIQUES
                # ------------------------------

                st.markdown(
                    '<div class="metadata-section-title technical">'
                    "Informations techniques :"
                    "</div>",
                    unsafe_allow_html=True,
                )


                st.markdown(
                    f"""
<div class="metadata-list technical-metadata">

- **review_id :** {doc.metadata.get("review_id", "N/A")}
- **dataset :** {doc.metadata.get("dataset", "N/A")}

</div>
""",
                    unsafe_allow_html=True,
                )


# ------------------------------
# LANCEMENT
# ------------------------------

# streamlit run streamlit_app/app.py
# python -m streamlit run streamlit_app/app.py