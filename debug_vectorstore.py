# Debug script to check what's in your vector store
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.vectorstore import load_vectorstore
from src.embeddings import create_embeddings

# Suppress pypdf warnings
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("DEBUG: Checking Vector Store Content")
print("=" * 60)

try:
    # Load vector store
    vectorstore = load_vectorstore()
    
    # Get collection info
    print(f"\n✅ Vector store loaded successfully")
    print(f"Total vectors: {vectorstore.index.ntotal}")
    
    # Test a simple query
    test_query = "pasta"
    print(f"\n🔍 Testing query: '{test_query}'")
    
    docs_with_scores = vectorstore.similarity_search_with_relevance_scores(
        query=test_query,
        k=5
    )
    
    print(f"Found {len(docs_with_scores)} results")
    
    for i, (doc, score) in enumerate(docs_with_scores):
        print(f"\n--- Result {i+1} (score: {score:.4f}) ---")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"Page: {doc.metadata.get('page', '?')}")
        print(f"Content preview: {doc.page_content[:150]}...")
    
    # Test another query
    test_query2 = "chicken temperature"
    print(f"\n\n🔍 Testing query: '{test_query2}'")
    
    docs_with_scores2 = vectorstore.similarity_search_with_relevance_scores(
        query=test_query2,
        k=5
    )
    
    print(f"Found {len(docs_with_scores2)} results")
    
    for i, (doc, score) in enumerate(docs_with_scores2):
        print(f"\n--- Result {i+1} (score: {score:.4f}) ---")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"Page: {doc.metadata.get('page', '?')}")
        print(f"Content preview: {doc.page_content[:150]}...")
    
    # Check if any scores are above 0.65 threshold
    print("\n\n📊 Score Analysis:")
    all_scores = [score for _, score in docs_with_scores]
    if all_scores:
        print(f"Max score: {max(all_scores):.4f}")
        print(f"Min score: {min(all_scores):.4f}")
        print(f"Average score: {sum(all_scores)/len(all_scores):.4f}")
        
        above_threshold = [s for s in all_scores if s >= 0.65]
        print(f"Scores >= 0.65: {len(above_threshold)} out of {len(all_scores)}")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
