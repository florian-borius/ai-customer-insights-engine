from pathlib import Path
import sys
from collections import deque
from datetime import datetime

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.security.request_guard import check_request_limit
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
# INITIALISATION DE LA PROTECTION
# ------------------------------

if "request_times" not in st.session_state:
    st.session_state.request_times = deque()

if "request_count" not in st.session_state:
    st.session_state.request_count = 0

if "blocked_until" not in st.session_state:
    st.session_state.blocked_until = 0


# ------------------------------
# CSS
# ------------------------------

st.markdown(
    """
    <style>


    /* --------------------------------
       TITRE DE L'APPLICATION
    -------------------------------- */

    h1 {
        font-size: 1rem;
        text-align: center;
    }


    /* --------------------------------
       DESCRIPTION DE L'APPLICATION
    -------------------------------- */

    .app-description {
        font-size: 1rem;
        font-weight: 300;
        line-height: 1.5;
        color: #4b5563;
        text-align: center;
        max-width: 85%;
        margin: 0 auto 2rem;
    }


    /* --------------------------------
       LIBELLÉ DU CHAMP DE SAISIE
    -------------------------------- */

    .question-label {
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }


    /* --------------------------------
       CHAMP DE SAISIE
    -------------------------------- */

    div[data-testid="InputInstructions"] {
        font-size: 0;
    }

    div[data-testid="InputInstructions"]::after {
        content: "Appuyez sur Entrée pour valider";
        font-size: 0.7rem;
    }

    /* --------------------------------
       FORMULAIRE
    -------------------------------- */

    div[data-testid="stForm"] {
        border: none;
        padding: 0;
    }


    /* --------------------------------
       EXEMPLES DE QUESTIONS
    -------------------------------- */

    .examples {
        font-size: 0.875rem;
        line-height: 1.5;
        color: #9ca3af;
        margin-bottom: 1rem;
    }

    .examples strong {
        font-weight: 600;
        color: #6b7280;
    }


    /* --------------------------------
       TITRES DES SECTIONS
    -------------------------------- */

    .section-title {
        font-size: 1rem;
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }


    /* --------------------------------
       CADRE DE RÉPONSE
    -------------------------------- */

    .answer-box {
        font-size: 1rem;
        line-height: 1.5;
        padding: 1rem;
        border: 1px solid #D1D5DB;
        border-radius: 8px;
        margin-bottom: 1rem;
    }


    /* --------------------------------
       CONTENU DU CONTEXTE
    -------------------------------- */

    .context-content {
        font-size: 1rem;
        line-height: 1.5;
        margin-bottom: 1rem;
    }


    /* --------------------------------
       SÉPARATEUR HORIZONTAL
    -------------------------------- */

    .metadata-separator {
        border-top: 1px solid #e5e7eb;
        margin: 0.4rem 0 0.875rem 0;
    }


    /* --------------------------------
       TITRES DES SOUS-SECTIONS DES MÉTADONNÉES
    -------------------------------- */

    .metadata-section-title {
        font-size: 0.875rem;
        font-weight: 600;
        text-decoration: underline;
        color: #6b7280;
        margin-top: 0.7rem;
    }


    /* --------------------------------
       LISTES DES MÉTADONNÉES
    -------------------------------- */

    .metadata-list {
        font-size: 0.875rem;
        line-height: 1.5;
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


    /* --------------------------------
       À SAVOIR
    -------------------------------- */

    .info-message {
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
        margin-top: 1rem;
        padding: 1rem;
        background-color: #eef4f8;
        border-radius: 0.5rem;
        font-size: 0.875rem;
        line-height: 1.5;
    }

    .info-icon {
        flex-shrink: 0;
        font-size: 1.1rem;
    }

    .info-content {
        flex: 1;
        color: #6f8494;
    }


    /* --------------------------------
    FOOTER
    -------------------------------- */

    .app-footer {
        font-size: 0.8rem;
        text-align: center;
        margin-top: 2rem;
        padding: 1rem 0;
        color: #9ca3af;
    }

    .app-footer a {
        color: #6b7280;
        text-decoration: none;
    }

    .app-footer a:hover {
        text-decoration: underline;
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
# TITRE + DESCRIPTION DE L'APPLICATION + LIBELLÉ DU CHAMP DE SAISIE
# ------------------------------

st.title("AI Customer Insights Engine")


st.markdown(
    """
    <div class="app-description">
        Cette application d’intelligence artificielle permet d’interroger en langage naturel
        une base de plusieurs dizaines de milliers d’avis clients issus du secteur bancaire.<br>
        Elle s’appuie sur une architecture <strong>RAG (Retrieval-Augmented Generation)</strong>
        pour rechercher les avis les plus pertinents et générer des réponses fondées sur les données disponibles.
    </div>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="question-label">
        Posez une question sur les avis clients :
    </div>
    """,
    unsafe_allow_html=True,
)


# ------------------------------
# CHAMP DE SAISIE + BOUTON
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
        • Quels sont les éléments de satisfaction des clients concernant les frais bancaires ?
    </div>
    """,
    unsafe_allow_html=True,
)


# ------------------------------
# LANCEMENT DE L'ANALYSE
# ------------------------------

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

        # ------------------------------
        # CONTRÔLE DES REQUÊTES
        # ------------------------------

        allowed, error_message, request_count, blocked_until = (
            check_request_limit(
                st.session_state.request_times,
                st.session_state.request_count,
                st.session_state.blocked_until,
            )
        )

        st.session_state.request_count = request_count
        st.session_state.blocked_until = blocked_until


        if not allowed:

            st.warning(error_message)


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
                '<div class="section-title">Réponse :</div>',
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
            # CONTEXTES UTILISÉS
            # ------------------------------

            st.markdown(
                '<div class="section-title">Contextes utilisés :</div>',
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
                    # SÉPARATEUR HORIZONTAL
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
                    <!-- - **Banque :** {bank_display} -->

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
# À SAVOIR
# ------------------------------

st.markdown(
    """
    <div class="info-message">
        <div class="info-icon">💡</div>
        <div class="info-content">
            <strong>À savoir :</strong> Les réponses sont générées à partir des avis clients récupérés et peuvent comporter des erreurs ou des inexactitudes.
            Dans sa version actuelle, le système ne permet pas d’établir des classements, comparaisons ou statistiques sur l’ensemble des avis.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ------------------------------
# FOOTER
# ------------------------------

st.markdown(
    """
    <div class="app-footer">
        <a href="https://www.linkedin.com/in/florian-borius/"
           target="_blank">
            Florian BORIUS
        </a>
        &nbsp; • &nbsp;
        <a href="https://github.com/florian-borius/ai-customer-insights-engine"
           target="_blank">
            Consulter le code source sur GitHub
        </a>
    </div>
    """,
    unsafe_allow_html=True,
)


# ------------------------------
# LANCEMENT
# ------------------------------

# streamlit run streamlit_app/app.py
# python -m streamlit run streamlit_app/app.py