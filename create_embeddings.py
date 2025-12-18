"""
Create Embeddings for RAG System
Uses sentence-transformers (runs locally, FREE)
"""

from sentence_transformers import SentenceTransformer
import os
import json
import numpy as np
from tqdm import tqdm
import faiss

def create_embeddings(
    processed_dir="data/processed/cases",
    output_dir="data/embeddings"
):
    """Create embeddings for all processed cases"""
    
    print("\n" + "=" * 70)
    print("CREATING EMBEDDINGS FOR RAG SYSTEM")
    print("=" * 70)
    
    # Load model
    print("\n Loading sentence-transformer model...")
    print("(First time: downloads ~400MB)")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print(" Model loaded (384-dimensional embeddings)")
    
    # Get processed files
    if not os.path.exists(processed_dir):
        print(f" Directory not found: {processed_dir}")
        print("   Run process_cases_ollama.py first!")
        return
    
    json_files = sorted([f for f in os.listdir(processed_dir) if f.endswith('.json')])
    
    if not json_files:
        print(" No processed files found!")
        print("   Wait for process_cases_ollama.py to finish!")
        return
    
    print(f"\n Found {len(json_files)} processed cases")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create embeddings
    embeddings = []
    metadata = []
    
    print("\n Creating embeddings (FREE, runs locally)...")
    for filename in tqdm(json_files, desc="Encoding"):
        try:
            with open(os.path.join(processed_dir, filename), 'r') as f:
                case = json.load(f)
            
            # Build text representation
            text_parts = []
            
            # Add clinical information
            if case.get('clinical_information'):
                clin = case['clinical_information']
                if clin.get('diagnosis'):
                    text_parts.append(f"Diagnosis: {clin['diagnosis']}")
                if clin.get('symptoms'):
                    text_parts.append(f"Symptoms: {clin['symptoms']}")
                if clin.get('physical_exam_findings'):
                    text_parts.append(f"Findings: {clin['physical_exam_findings']}")
            
            # Add treatment info
            if case.get('treatment'):
                treat = case['treatment']
                if treat.get('procedure_performed'):
                    text_parts.append(f"Procedure: {treat['procedure_performed']}")
                if treat.get('procedure_planned'):
                    text_parts.append(f"Planned: {treat['procedure_planned']}")
            
            # Add specialty
            if case.get('meta', {}).get('original_specialty'):
                text_parts.append(f"Specialty: {case['meta']['original_specialty']}")
            
            text = " ".join(text_parts)
            
            if len(text) < 20:
                continue
            
            # Generate embedding
            embedding = model.encode(text, normalize_embeddings=True)
            
            embeddings.append(embedding)
            metadata.append({
                'case_id': case.get('meta', {}).get('case_id'),
                'diagnosis': case.get('clinical_information', {}).get('diagnosis'),
                'procedure': case.get('treatment', {}).get('procedure_performed') or case.get('treatment', {}).get('procedure_planned'),
                'specialty': case.get('meta', {}).get('original_specialty'),
                'filename': filename
            })
            
        except Exception as e:
            print(f"\n Error with {filename}: {e}")
            continue
    
    if not embeddings:
        print(" No embeddings created!")
        return
    
    # Convert to array
    embeddings_array = np.array(embeddings).astype('float32')
    
    print(f"\n Created {len(embeddings)} embeddings")
    print(f"   Dimension: {embeddings_array.shape[1]}")
    print(f"   Size: {embeddings_array.nbytes / 1024 / 1024:.1f} MB")
    
    # Create FAISS index
    print("\n Building FAISS index...")
    dimension = embeddings_array.shape[1]
    
    # Use IndexFlatIP for cosine similarity (since embeddings are normalized)
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_array)
    
    # Save index
    index_path = os.path.join(output_dir, "patient_cases.index")
    faiss.write_index(index, index_path)
    print(f" FAISS index saved: {index_path}")
    
    # Save embeddings as numpy array
    embeddings_path = os.path.join(output_dir, "embeddings.npy")
    np.save(embeddings_path, embeddings_array)
    print(f" Embeddings saved: {embeddings_path}")
    
    # Save metadata
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f" Metadata saved: {metadata_path}")
    
    print("\n" + "=" * 70)
    print("EMBEDDING CREATION COMPLETE")
    print("=" * 70)
    print(f" Output directory: {output_dir}/")
    print(f" Cost: $0 (ran locally!)")
    print(f" Search ready: <10ms per query")

if __name__ == "__main__":
    create_embeddings()
