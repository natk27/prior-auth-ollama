"""
Monitor Processing Progress
"""

import os
import json
from datetime import datetime

def check_progress():
    """Check current processing status"""
    
    print("\n" + "=" * 70)
    print(f"PROCESSING PROGRESS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Check policies
    policy_dir = "data/processed/policies"
    if os.path.exists(policy_dir):
        policy_count = len([f for f in os.listdir(policy_dir) if f.endswith('.txt')])
        print(f"\n POLICIES: {policy_count}/50")
        print(f"   Status: {' Complete' if policy_count >= 50 else ' In progress'}")
    else:
        print(f"\n POLICIES: 0/50")
        print(f"   Status: Not started")
    
    # Check cases
    cases_dir = "data/processed/cases"
    if os.path.exists(cases_dir):
        case_count = len([f for f in os.listdir(cases_dir) if f.endswith('.json')])
        print(f"\n PATIENT CASES: {case_count}/4,966")
        
        progress_pct = (case_count / 4966) * 100
        bar_length = 50
        filled = int(bar_length * progress_pct / 100)
        bar = ' ' * filled + ' ' * (bar_length - filled)
        
        print(f"   Progress: [{bar}] {progress_pct:.1f}%")
        
        if case_count < 4966:
            remaining = 4966 - case_count
            # Estimate: ~3.5 seconds per case
            est_minutes = (remaining * 3.5) / 60
            est_hours = est_minutes / 60
            print(f"   Remaining: {remaining} cases")
            print(f"   Est. time: {est_hours:.1f} hours")
            print(f"   Status: Processing...")
        else:
            print(f"   Status: Complete!")
    else:
        print(f"\n PATIENT CASES: 0/4,966")
        print(f"   Status: Not started")
    
    # Check embeddings
    embeddings_dir = "data/embeddings"
    if os.path.exists(embeddings_dir) and os.path.exists(f"{embeddings_dir}/patient_cases.index"):
        print(f"\n EMBEDDINGS: Created")
    else:
        print(f"\n EMBEDDINGS: Not created yet")
        if case_count >= 4966:
            print(f"   Action needed: Run 'python create_embeddings.py'")
    
    # Check summary file
    if os.path.exists("data/processing_summary.json"):
        with open("data/processing_summary.json", 'r') as f:
            summary = json.load(f)
        print(f"\n LAST SESSION:")
        print(f"   Processed: {summary.get('newly_processed', 0)} cases")
        print(f"   Errors: {summary.get('errors', 0)}")
        print(f"   Timestamp: {summary.get('timestamp', 'N/A')}")
    
    print("\n" + "=" * 70)
    
    # Next steps
    print("\n NEXT STEPS:")
    if case_count < 100:
        print("   Wait for processing to complete (check back in 30 mins)")
    elif case_count < 4966:
        print("   Processing ongoing... Check back later")
    elif not os.path.exists(f"{embeddings_dir}/patient_cases.index"):
        print("   Run: python create_embeddings.py")
    else:
        print("   Run: python test_rag.py")
        print("   Run: python prior_auth_ollama.py")
    
    print("=" * 70 + "\n")

if __name__ == "__main__":    
    check_progress()
