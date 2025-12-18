"""
Test Complete End-to-End System
"""

import os
import json

def test_system():
    """Run comprehensive system test"""
    
    print("\n" + "=" * 70)
    print("COMPLETE SYSTEM TEST")
    print("=" * 70)
    
    all_good = True
    
    # Check 1: Policies
    print("\n1️⃣  CHECKING POLICIES...")
    policy_count = len([f for f in os.listdir("data/processed/policies") if f.endswith('.txt')])
    if policy_count >= 20:
        print(f"   ✅ {policy_count} policies found")
    else:
        print(f"   ❌ Only {policy_count} policies (need 20+)")
        all_good = False
    
    # Check 2: Processed cases
    print("\n2️⃣  CHECKING PROCESSED CASES...")
    case_count = len([f for f in os.listdir("data/processed/cases") if f.endswith('.json')])
    if case_count >= 4900:
        print(f"   ✅ {case_count} cases processed")
    elif case_count >= 100:
        print(f"   ⚠️  {case_count} cases processed (still processing...)")
        all_good = False
    else:
        print(f"   ❌ Only {case_count} cases (need 100+)")
        all_good = False
    
    # Check 3: Sample a processed case
    if case_count > 0:
        print("\n3️⃣  CHECKING CASE QUALITY...")
        with open("data/processed/cases/case_0000.json", 'r') as f:
            sample = json.load(f)
        
        has_clinical = 'clinical_information' in sample
        has_treatment = 'treatment' in sample
        has_meta = 'meta' in sample
        
        if has_clinical and has_treatment and has_meta:
            print(f"   ✅ Case structure looks good")
            print(f"      - Diagnosis: {sample.get('clinical_information', {}).get('diagnosis', 'N/A')[:50]}...")
            print(f"      - Specialty: {sample.get('meta', {}).get('original_specialty', 'N/A')}")
        else:
            print(f"   ❌ Case structure incomplete")
            all_good = False
    
    # Check 4: Embeddings
    print("\n4️⃣  CHECKING EMBEDDINGS...")
    if os.path.exists("data/embeddings/patient_cases.index"):
        print(f"   ✅ FAISS index exists")
        
        # Try loading it
        try:
            import faiss
            index = faiss.read_index("data/embeddings/patient_cases.index")
            print(f"   ✅ Index valid ({index.ntotal} vectors)")
        except Exception as e:
            print(f"   ❌ Index corrupted: {e}")
            all_good = False
    else:
        if case_count >= 4900:
            print(f"   ⚠️  Embeddings not created yet")
            print(f"      Run: python create_embeddings.py")
        else:
            print(f"   ⏳ Waiting for processing to finish")
        all_good = False
    
    # Check 5: Scripts exist
    print("\n5️⃣  CHECKING SCRIPTS...")
    required_scripts = [
        'process_cases_ollama.py',
        'create_embeddings.py',
        'test_rag.py',
        'prior_auth_ollama.py',
        'check_progress.py'
    ]
    
    for script in required_scripts:
        if os.path.exists(script):
            print(f"   ✅ {script}")
        else:
            print(f"   ❌ {script} missing")
            all_good = False
    
    # Final status
    print("\n" + "=" * 70)
    if all_good and case_count >= 4900:
        print("🎉 SYSTEM READY! All components working!")
        print("\n📋 NEXT STEPS:")
        if not os.path.exists("data/embeddings/patient_cases.index"):
            print("   1. python create_embeddings.py")
            print("   2. python test_rag.py")
            print("   3. python prior_auth_ollama.py")
        else:
            print("   1. python test_rag.py")
            print("   2. python prior_auth_ollama.py")
    elif case_count >= 100:
        print("⏳ SYSTEM IN PROGRESS")
        print(f"\n   Cases: {case_count}/4,966 ({(case_count/4966)*100:.1f}%)")
        print(f"   Check back later with: python check_progress.py")
    else:
        print("❌ SYSTEM NOT READY")
        print("\n   Issues found - check messages above")
    
    print("=" * 70 + "\n")

if __name__ == "__main__":
    test_system()
