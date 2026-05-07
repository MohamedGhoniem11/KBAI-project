# =============================================================================
# CULINARY ARTS RAG SYSTEM - Configuration
# =============================================================================
# Domain: Culinary Arts
# Selected for: Rich recipe/technique content, diverse queries, disclaimer need

from pathlib import Path

# --- Project Root (derived from this file's location) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- File Paths ---
DOCS_DIR = PROJECT_ROOT / "KB"
VECTORSTORE_DIR = PROJECT_ROOT / "data" / "vectorstore"

# --- Ingestion Parameters (Stage 1) ---
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# --- Retrieval Parameters (Stage 2: FR-03) ---
TOP_K = 5
SIMILARITY_THRESHOLD = 0.65  # FR-03: Minimum cosine similarity threshold

# --- Embedding Model (FR-02) ---
# Justification: all-MiniLM-L6-v2 provides strong semantic similarity
# with fast inference, suitable for culinary terminology
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_RATIONALE = (
    "MiniLM-L6-v2 chosen for culinary domain: "
    "(1) Fast inference with 384-dim embeddings; "
    "(2) Strong semantic similarity for recipe/technique matching; "
    "(3) Proven on STS benchmarks; "
    "(4) Handles cooking terminology well."
)

# --- Domain Disclaimer (FR-07) ---
DOMAIN_DISCLAIMER = (
    "This information is for educational purposes only and is not professional "
    "culinary advice. Always follow proper food safety guidelines and consult "
    "certified professionals for dietary or professional cooking needs."
)