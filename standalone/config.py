# Standalone config (same values as src/config.py, duplicated for independence)
# This version has zero dependencies on LangChain — pure Python only.
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOCS_DIR = PROJECT_ROOT / "KB"
VECTORSTORE_DIR = PROJECT_ROOT / "data" / "vectorstore_standalone"  # Separate from LangChain's index

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

TOP_K = 5
SIMILARITY_THRESHOLD = 0.65

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DOMAIN_DISCLAIMER = (
    "This information is for educational purposes only and is not professional "
    "culinary advice. Always follow proper food safety guidelines and consult "
    "certified professionals for dietary or professional cooking needs."
)
