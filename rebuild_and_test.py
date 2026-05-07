# Complete rebuild and test script
import os
import sys
import shutil
from pathlib import Path
import warnings

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

KB_DIR = project_root / "KB"
USE_STANDALONE = "--standalone" in sys.argv

print("=" * 60)
title = "STANDALONE VECTOR STORE REBUILD + TEST" if USE_STANDALONE else "LANGCHAIN VECTOR STORE REBUILD + TEST"
print(title)
print("=" * 60)

# Step 0: Verify KB/ exists
if not KB_DIR.exists() or not any(KB_DIR.iterdir()):
    print("\n❌ KB/ directory is empty or missing.")
    print("   Download KB.zip from Google Drive:")
    print("   https://drive.google.com/uc?export=download&id=1RskXkZXqQiszdQ8QkEYySKlgToQPBZO4")
    print("   Then unzip to KB/ and run this script again.")
    sys.exit(1)

print(f"\n✅ KB/ directory found with documents")

if USE_STANDALONE:
    vs_path = project_root / "data" / "vectorstore_standalone"
else:
    vs_path = project_root / "data" / "vectorstore"

# Step 1: Delete old vector store
if vs_path.exists():
    shutil.rmtree(vs_path)
    print("\n✅ Deleted old vector store")

# Step 2: Build new vector store
print("\n[1/3] Ingesting PDF/DOCX documents...")

if USE_STANDALONE:
    from standalone.ingestion import ingest_documents
    from standalone.vectorstore import VectorStore
    from standalone.embeddings import Embeddings
    from standalone.config import EMBEDDING_MODEL

    chunks = ingest_documents()
    print(f"\n[2/3] Creating vector store with normalized embeddings...")
    embeddings = Embeddings(EMBEDDING_MODEL)
    texts = [c["content"] for c in chunks]
    vectors = embeddings.embed_documents(texts)
    vectorstore = VectorStore()
    vectorstore.add(vectors, chunks)
    vectorstore.save(vs_path)
    print(f"✅ Vector store saved with {len(chunks)} chunks")

    # Step 3: Test retrieval
    print("\n[3/3] Testing retrieval (FR-03)...")
    from standalone.config import TOP_K, SIMILARITY_THRESHOLD
    from standalone.retrieval import RetrievalEngine

    engine = RetrievalEngine(vectorstore, embeddings)
    test_queries = ["pasta", "chicken temperature", "knife skills"]

    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        results = engine.retrieve(query)
        print(f"   Found {len(results)} chunks with score >= {SIMILARITY_THRESHOLD}")
        for i, r in enumerate(results[:3]):
            print(f"   [{i+1}] Score: {r['score']:.4f}")
            print(f"       Source: {r['source']}")
            print(f"       Content: {r['content'][:100]}...")

else:
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
        from src.config import SIMILARITY_THRESHOLD
        filtered = [(doc, score) for doc, score in docs_with_scores if score >= SIMILARITY_THRESHOLD]
        print(f"   Found {len(filtered)} chunks with score >= 0.65")
        for i, (doc, score) in enumerate(filtered[:3]):
            print(f"   [{i+1}] Score: {score:.4f}")
            print(f"       Source: {doc.metadata.get('source', '?')}")
            print(f"       Content: {doc.page_content[:100]}...")

print("\n" + "=" * 60)
print("✅ REBUILD COMPLETE")
print(f"   Mode: {'Standalone' if USE_STANDALONE else 'LangChain'}")
print("1. Set XAI_API_KEY in .env file")
print("2. Run 'python main.py' (LangChain) or 'python main.py --standalone' (Standalone)")
print("3. Run 'streamlit run app.py' for web UI")
print("=" * 60)
