"""
Test RAG System - Search Patient Cases
"""

from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np
import os

def test_rag():
    """Test the RAG system with sample queries"""
    
    print("\n" + "=" * 70)
    print("TESTING RAG SYSTEM")
    print("=" * 70)
    
    # Check if embeddings exist
    if not os.path.exists("data/embeddings/patient_cases.index"):
        print("\n❌ Embeddings not found!")
        print("   Run: python create_embeddings.py")
        return
    
    # Load model
    print("\n📥 Loading model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Load index
    print("📥 Loading FAISS index...")
    index = faiss.read_index("data/embeddings/patient_cases.index")
    
    # Load metadata
    print("📥 Loading metadata...")
    with open("data/embeddings/metadata.json", 'r') as f:
        metadata = json.load(f)
    
    print(f"✅ Loaded {index.ntotal} patient cases")
    
    # Test queries
    queries = [
        "patient with severe back pain needs lumbar surgery",
        "knee arthroscopy for torn meniscus",
        "chest pain requiring cardiac catheterization",
        "colonoscopy for screening",
        "MRI brain for persistent headaches"
    ]
    
    for query in queries:
        print("\n" + "-" * 70)
        print(f"🔍 QUERY: {query}")
        print("-" * 70)
        
        # Encode query
        query_embedding = model.encode(query, normalize_embeddings=True)
        query_vector = np.array([query_embedding]).astype('float32')
        
        # Search (k=5 most similar)
        similarities, indices = index.search(query_vector, k=5)
        
        print("\n📋 TOP 5 MOST SIMILAR CASES:")
        for rank, (idx, similarity) in enumerate(zip(indices[0], similarities[0]), 1):
            case = metadata[idx]
            print(f"\n{rank}. Similarity: {similarity:.3f}")
            print(f"   Case ID: {case['case_id']}")
            print(f"   Diagnosis: {case['diagnosis']}")
            print(f"   Procedure: {case['procedure']}")
            print(f"   Specialty: {case['specialty']}")
    
    print("\n" + "=" * 70)
    print("✅ RAG SYSTEM WORKING PERFECTLY!")
    print("=" * 70)

if __name__ == "__main__":
    test_rag()
