# =============================================================================
# CULINARY ARTS RAG SYSTEM - Configuration
# =============================================================================
# All tunable parameters live here: chunk sizes, thresholds, model names.
# Change values in ONE place instead of hunting through code.
# Domain: Culinary Arts (selected for rich recipe content & diverse queries)

from pathlib import Path  # Cross-platform path handling (windows/linux/mac)

# --- Project Root (derived from this file's location) ---
# __file__ = src/config.py, .parent = src/, .parent.parent = project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- File Paths ---
DOCS_DIR = PROJECT_ROOT / "KB"             # Where PDF cookbooks live
VECTORSTORE_DIR = PROJECT_ROOT / "data" / "vectorstore"  # Where FAISS index is saved

# --- Ingestion Parameters (Stage 1) ---
CHUNK_SIZE = 512       # Characters per chunk (big enough for a recipe step)
CHUNK_OVERLAP = 64     # Overlap between chunks (preserves context across boundaries)

# --- Retrieval Parameters (Stage 2: FR-03) ---
TOP_K = 5                              # Number of candidate chunks to fetch
SIMILARITY_THRESHOLD = 0.65            # FR-03: Minimum cosine similarity to keep a chunk

# --- Embedding Model (FR-02) ---
# Justification: all-MiniLM-L6-v2 provides strong semantic similarity
# with fast inference, suitable for culinary terminology
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_RATIONALE = (  # Stored in code so you can cite it in reports
    "MiniLM-L6-v2 chosen for culinary domain: "
    "(1) Fast inference with 384-dim embeddings; "
    "(2) Strong semantic similarity for recipe/technique matching; "
    "(3) Proven on STS benchmarks; "
    "(4) Handles cooking terminology well."
)

# --- Domain Disclaimer (FR-07) ---
# Appended to EVERY response — required for food safety / liability reasons
DOMAIN_DISCLAIMER = (
    "This information is for educational purposes only and is not professional "
    "culinary advice. Always follow proper food safety guidelines and consult "
    "certified professionals for dietary or professional cooking needs."
)
