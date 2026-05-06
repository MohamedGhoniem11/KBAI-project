# Complete rebuild and test script
import os
import sys
from pathlib import Path
import warnings

# Fix: Properly get project root directory
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

print("=" * 60)
print("COMPLETE VECTOR STORE REBUILD + TEST")
print("=" * 60)

# Step 1: Delete old vector store
vs_path = project_root / "data" / "vectorstore"
if vs_path.exists():
    import shutil
    shutil.rmtree(vs_path)
    print("✅ Deleted old vector store")

# Step 2: Build new vector store (PDF + DOCX)
print("\n[1/3] Ingesting PDF/DOCX documents...")
from src.ingestion import ingest_documents
chunks = ingest_documents()

print(f"\n[2/3] Creating vector store with normalized embeddings...")
from src.vectorstore import create_vectorstore, save_vectorstore
vectorstore = create_vectorstore(chunks)
save_vectorstore(vectorstore)
print(f"✅ Vector store saved with {len(chunks)} chunks")

# Step 3: Test retrieval
print("\n[3/3] Testing retrieval (FR-03)...")
test_queries = ["pasta", "chicken temperature", "knife skills"]

for query in test_queries:
    print(f"\n🔍 Query: '{query}'")
    docs_with_scores = vectorstore.similarity_search_with_relevance_scores(query=query, k=5)
    # Filter by 0.65 threshold (FR-03)
    filtered = [ (doc, score) for doc, score in docs_with_scores if score >= 0.65 ]
    print(f"   Found {len(filtered)} chunks with score >= 0.65")
    
    for i, (doc, score) in enumerate(filtered[:3]):
        print(f"   [{i+1}] Score: {score:.4f}")
        print(f"       Source: {doc.metadata.get('source', '?')}")
        print(f"       Content: {doc.page_content[:100]}...")

print("\n" + "=" * 60)
print("✅ REBUILD COMPLETE")
print("1. Set XAI_API_KEY in .env file")
print("2. Run 'python main.py' to test via CLI (no Streamlit needed)")
print("3. Run 'streamlit run app.py' for web UI")
print("=" * 60)
