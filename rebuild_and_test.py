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

# Step 2: Build new vector store
print("\n[1/3] Ingesting documents...")
from src.ingestion import ingest_documents
chunks = ingest_documents()

print(f"\n[2/3] Creating vector store with normalized embeddings...")
from src.vectorstore import create_vectorstore, save_vectorstore
vectorstore = create_vectorstore(chunks)
save_vectorstore(vectorstore)
print(f"✅ Vector store saved with {len(chunks)} chunks")

# Step 3: Test queries immediately
print("\n[3/3] Testing retrieval...")
test_queries = ["pasta", "chicken temperature", "knife skills"]

for query in test_queries:
    print(f"\n🔍 Query: '{query}'")
    docs_with_scores = vectorstore.similarity_search_with_relevance_scores(query=query, k=5)
    print(f"   Found {len(docs_with_scores)} results")
    
    for i, (doc, score) in enumerate(docs_with_scores[:3]):
        print(f"   [{i+1}] Score: {score:.4f}")
        print(f"       Source: {doc.metadata.get('source', '?')}")
        print(f"       Content: {doc.page_content[:100]}...")

print("\n" + "=" * 60)
print("✅ REBUILD COMPLETE - Now restart Streamlit app")
print("=" * 60)
