from pathlib import Path
import os

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")


# ------------------------------
# DATA PATHS
# ------------------------------

RAW_DATA_PATH = PROJECT_ROOT / "data/raw/scraped_reviews_final.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data/processed/processed_reviews.parquet"


# ------------------------------
# DATA PREPROCESSING (RAG)
# ------------------------------

REVIEW_MIN_LENGTH = None
REVIEW_MAX_LENGTH = None


# ------------------------------
# CHUNKING (RAG)
# ------------------------------

SEPARATORS = [". ", "! ", "? ", ".", "!", "?", " ", ""]
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100


# ------------------------------
# EMBEDDING (RAG)
# ------------------------------

HUGGINGFACE_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# ------------------------------
# VECTOR STORE (RAG)
# ------------------------------

CHROMA_PATH = PROJECT_ROOT / "data/chroma_db"
COLLECTION_NAME = "bank_customer_reviews"


# ------------------------------
# RETRIEVER (RAG)
# ------------------------------

SEARCH_TYPE = "similarity"
RETRIEVER_K = 20


# ------------------------------
# RERANKER (RAG)
# ------------------------------

HUGGINGFACE_CROSSENCODER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANKER_TOP_N = 5


# ------------------------------
# CHAIN (RAG)
# ------------------------------

LLM_MODEL = "gpt-4o-mini"   # "gpt-4.1-mini"
MAX_COMPLETION_TOKENS = 300