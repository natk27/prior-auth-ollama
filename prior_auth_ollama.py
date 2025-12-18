"""
Complete Prior Authorization with Ollama
"""

import requests
from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np
import os

class PriorAuthSystemOllama:
    """Prior auth system using Ollama"""
    
    def __init__(self):
        print(" Initializing system with Ollama...")
        
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load case database
        self.case_index = faiss.read_index("data/embeddings/patient_cases.index")
        with open("data/embeddings/metadata.json", 'r') as f:
            self.case_metadata = json.load(f)
        
        print(f" Loaded {self.case_index.ntotal} patient cases")
        
        # Load policies
        policy_dir = "data/processed/policies"
        self.policies = {}
        if os.path.exists(policy_dir):
            for filename in os.listdir(policy_dir):
                if filename.endswith('.txt'):
                    with open(os.path.join(policy_dir, filename), 'r') as f:
                        procedure = filename.replace('_policy.txt', '').replace('_', ' ').title()
                        self.policies[procedure] = f.read()
        
        print(f" Loaded {len(self.policies)} policies")
        print(" Using Ollama (unlimited, FREE!)\n")
    
    def find_similar_cases(self, patient_text, k=3):
        """Find similar cases using RAG"""
        query_emb = self.embedding_model.encode(patient_text, normalize_embeddings=True)
        query_vec = np.array([query_emb]).astype('float32')
        similarities, indices = self.case_index.search(query_vec, k=k)
        
        similar_cases = []
        for idx, sim in zip(indices[0], similarities[0]):
            similar_cases.append({
                'case_id': self.case_metadata[idx]['case_id'],
                'diagnosis': self.case_metadata[idx]['diagnosis'],
                'procedure': self.case_metadata[idx]['procedure'],
                'similarity': float(sim)
            })
        
        return similar_cases
    
    def find_relevant_policy(self, procedure_name):
        """Find most relevant policy"""
        for policy_name in self.policies.keys():
            if procedure_name.lower() in policy_name.lower():
                return self.policies[policy_name]
        return "Standard prior authorization criteria apply."
    
    def make_decision_ollama(self, patient_document, procedure_requested):
        """Make decision using Ollama"""
        
        print(f"\n{'='*70}")
        print("PRIOR AUTHORIZATION REQUEST")
        print(f"{'='*70}")
        print(f"Procedure: {procedure_requested}")
        
        # Find similar cases
        print("\n Step 1: Finding similar cases...")
        similar_cases = self.find_similar_cases(patient_document, k=3)
        print(" Top 3 similar cases found")
        
        # Get policy
        print(f"\n Step 2: Retrieving policy...")
        policy = self.find_relevant_policy(procedure_requested)
        print(f" Found policy")
        
        # Make decision
        print(f"\n Step 3: Evaluating with Ollama...")
        
        prompt = f"""You are a prior authorization specialist. Review this case against the policy.

PATIENT CASE:
{patient_document}

PROCEDURE: {procedure_requested}

POLICY:
{policy[:3000]}

SIMILAR APPROVED CASES:
{json.dumps(similar_cases, indent=2)}

Evaluate if criteria are met. Return ONLY valid JSON:

{{
    "decision": "APPROVED or DENIED or ADDITIONAL_INFO_NEEDED",
    "confidence": "HIGH or MEDIUM or LOW",
    "criteria_met": [
        {{"criterion": "name", "status": "MET or NOT_MET", "evidence": "evidence from case"}}
    ],
    "reasoning": "detailed explanation",
    "missing_documentation": ["list items or empty"],
    "recommendation": "clinical recommendation"
}}"""

        try:
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': 'llama3.2',
                    'prompt': prompt,
                    'stream': False,
                    'format': 'json'
                },
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()['response']
                result = result.replace('```json', '').replace('```', '').strip()
                decision = json.loads(result)
                self.display_decision(decision)
                return decision
            else:
                print(f" Error: Status {response.status_code}")
                return None
                
        except Exception as e:
            print(f" Error: {e}")
            return None
    
    def display_decision(self, decision):
        """Display decision"""
        print(f"\n{'='*70}")
        print("AUTHORIZATION DECISION")
        print(f"{'='*70}")
        
        dec = decision['decision']
        if dec == 'APPROVED':
            print("\n APPROVED")
        elif dec == 'DENIED':
            print("\n DENIED")
        else:
            print("\n ADDITIONAL INFORMATION NEEDED")
        
        print(f"Confidence: {decision['confidence']}")
        
        print(f"\n{'-'*70}")
        print("CRITERIA EVALUATION:")
        print(f"{'-'*70}")
        for crit in decision.get('criteria_met', []):
            status = "✓" if crit['status'] == 'MET' else "✗"
            print(f"\n{status} {crit['criterion']}")
            print(f"  Status: {crit['status']}")
            print(f"  Evidence: {crit['evidence']}")
        
        print(f"\n{'-'*70}")
        print("REASONING:")
        print(f"{'-'*70}")
        print(decision['reasoning'])
        
        if decision.get('missing_documentation'):
            print(f"\n{'-'*70}")
            print("MISSING DOCUMENTATION:")
            print(f"{'-'*70}")
            for item in decision['missing_documentation']:
                print(f"  • {item}")
        
        print(f"\n{'='*70}\n")

def test_system():
    """Test system"""
    
    system = PriorAuthSystemOllama()
    
    patient_case = """
    Patient: 58-year-old male
    Chief Complaint: Chronic lower back pain radiating to left leg for 8 months
    
    History:
    - Pain rated 8/10
    - Conservative treatments:
      * Physical therapy: 12 weeks completed
      * NSAIDs: 10 weeks
      * Epidural injections: 3 injections
    - Minimal improvement
    
    Physical Exam:
    - Positive straight leg raise (left)
    - Decreased sensation L5 dermatome
    - Ankle weakness (4/5)
    
    Imaging:
    - MRI lumbar spine (11/10/2024): L4-L5 disc herniation with severe nerve compression
    
    Diagnosis: Lumbar disc herniation with radiculopathy
    Procedure Requested: Lumbar microdiscectomy
    """
    
    decision = system.make_decision_ollama(patient_case, "Lumbar Microdiscectomy")
    
    if decision:
        with open('sample_decision_ollama.json', 'w') as f:
            json.dump(decision, f, indent=2)
        print(" Decision saved to: sample_decision_ollama.json")

if __name__ == "__main__":
    test_system()
