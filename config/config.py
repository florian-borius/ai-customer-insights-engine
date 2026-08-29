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
RETRIEVER_K = 5


# ------------------------------
# RERANKER (RAG)
# ------------------------------

HUGGINGFACE_CROSSENCODER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANKER_TOP_N = 5


# ------------------------------
# CHAIN (RAG)
# ------------------------------

LLM_MODEL = "gpt-4o-mini"
MAX_COMPLETION_TOKENS = 300

# Tarifs gpt-4o-mini relevés le 24/08/2026 (USD / 1M de tokens)
INPUT_TOKEN_PRICE = 0.15 / 1_000_000
OUTPUT_TOKEN_PRICE = 0.60 / 1_000_000


# ------------------------------
# SECURITY
# ------------------------------

MAX_REQUESTS_PER_MINUTE = 3
MAX_REQUESTS_PER_SESSION = 5
BLOCK_DURATION = 60  # secondes