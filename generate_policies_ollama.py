"""
Generate Insurance Policies with Ollama (UNLIMITED, FREE)
"""

import requests
import os
from tqdm import tqdm
import time

PROCEDURES = [
    ("Lumbar Discectomy", "63030"),
    ("Lumbar Fusion", "22612"),
    ("Cervical Discectomy", "63075"),
    ("Spinal Decompression", "63047"),
    ("Knee Arthroscopy", "29881"),
    ("Total Knee Replacement", "27447"),
    ("Partial Knee Replacement", "27446"),
    ("ACL Reconstruction", "29888"),
    ("Total Hip Replacement", "27130"),
    ("Hip Arthroscopy", "29914"),
    ("Hip Resurfacing", "27132"),
    ("Rotator Cuff Repair", "29827"),
    ("Shoulder Arthroscopy", "29806"),
    ("Shoulder Replacement", "23472"),
    ("Carpal Tunnel Release", "64721"),
    ("Trigger Finger Release", "26055"),
    ("Cardiac Catheterization", "93458"),
    ("Coronary Artery Bypass", "33533"),
    ("Pacemaker Insertion", "33206"),
    ("Angioplasty with Stent", "92928"),
    ("Echocardiogram", "93306"),
    ("Stress Test", "93015"),
    ("MRI Brain", "70551"),
    ("MRI Spine Lumbar", "72148"),
    ("MRI Knee", "73721"),
    ("CT Scan Abdomen", "74177"),
    ("CT Scan Chest", "71250"),
    ("PET Scan", "78815"),
    ("Colonoscopy", "45378"),
    ("Upper Endoscopy", "43235"),
    ("Hemorrhoid Banding", "46221"),
    ("Hernia Repair Inguinal", "49505"),
    ("Gallbladder Removal", "47562"),
    ("Appendectomy", "44970"),
    ("Bariatric Surgery", "43644"),
    ("Cataract Surgery", "66984"),
    ("Sleep Study", "95810"),
    ("Physical Therapy", "97110"),
    ("Epidural Steroid Injection", "62311"),
    ("Trigger Point Injections", "20552"),
    ("Varicose Vein Treatment", "37700"),
    ("Carotid Endarterectomy", "35301"),
    ("Prostate Biopsy", "55700"),
    ("Cystoscopy", "52000"),
    ("Septoplasty", "30520"),
    ("Tonsillectomy", "42825"),
    ("Facet Joint Injection", "64493"),
    ("Radiofrequency Ablation", "64635"),
    ("Hysterectomy", "58150"),
    ("Breast Biopsy", "19083"),
]

def generate_policy_ollama(procedure_name, cpt_code):
    """Generate policy using Ollama"""
    
    prompt = f"""Generate a detailed, realistic insurance prior authorization policy.

PROCEDURE: {procedure_name}
CPT CODE: {cpt_code}

Create a comprehensive policy document including:

1. PROCEDURE OVERVIEW
   - Procedure name and CPT code
   - Clinical description

2. COVERAGE CRITERIA (Must meet ALL)
   - Conservative treatment requirements (be specific: "12 weeks of physical therapy")
   - Symptom duration requirements (e.g., "minimum 6 weeks")
   - Clinical documentation required
   - Imaging requirements (type, recency)
   - Severity criteria (pain scales, functional assessments)

3. MEDICAL NECESSITY CRITERIA
   - Patient selection criteria
   - Failed prior treatments
   - Contraindications to conservative care

4. EXCLUSION CRITERIA
   - Active infections
   - Pregnancy (if applicable)
   - Severe comorbidities
   - Lack of conservative treatment
   - Cosmetic indications

5. DOCUMENTATION REQUIREMENTS
   - History and physical examination
   - Diagnostic test results
   - Treatment logs
   - Specialist consultations

6. AUTHORIZATION DETAILS
   - Valid for: 90 days
   - Requires peer-to-peer if denied

Make it detailed and realistic like actual Blue Cross Blue Shield policies.
Use specific numbers and durations."""

    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'llama3.2',
                'prompt': prompt,
                'stream': False
            },
            timeout=120
        )
        
        if response.status_code == 200:
            return response.json()['response']
        else:
            print(f"Error: Status {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    output_dir = "data/processed/policies"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("GENERATING POLICIES WITH OLLAMA (UNLIMITED & FREE!)")
    print("=" * 70)
    
    # Check what's already done
    existing = set(f.replace('_policy.txt', '') for f in os.listdir(output_dir) if f.endswith('.txt'))
    
    todo = []
    for proc, cpt in PROCEDURES:
        safe_name = proc.lower().replace(' ', '_').replace('/', '_')
        if safe_name not in existing:
            todo.append((proc, cpt, safe_name))
    
    print(f"\n STATUS:")
    print(f"   Total procedures: {len(PROCEDURES)}")
    print(f"   Already completed: {len(existing)}")
    print(f"   To generate: {len(todo)}")
    
    if not todo:
        print("\n All policies already generated!")
        return
    
    print(f"\n Cost: $0 (runs locally!)")
    print(f" No rate limits!")
    print(f" Estimated time: ~{len(todo) * 0.5:.0f} minutes (~30 seconds per policy)")
    
    response = input("\nStart generation? (yes/no): ")
    if response.lower() != 'yes':
        print("Cancelled")
        return
    
    generated = 0
    errors = 0
    
    print("\n🔄 Generating policies...")
    for proc_name, cpt_code, safe_name in tqdm(todo, desc="Generating"):
        try:
            policy = generate_policy_ollama(proc_name, cpt_code)
            
            if policy:
                filepath = os.path.join(output_dir, f"{safe_name}_policy.txt")
                with open(filepath, 'w') as f:
                    f.write(f"PRIOR AUTHORIZATION POLICY\n")
                    f.write(f"=" * 70 + "\n\n")
                    f.write(f"Procedure: {proc_name}\n")
                    f.write(f"CPT Code: {cpt_code}\n\n")
                    f.write("=" * 70 + "\n\n")
                    f.write(policy)
                
                generated += 1
            else:
                errors += 1
            
            time.sleep(0.5)  # Small delay to not overwhelm CPU
            
        except Exception as e:
            print(f"\n Error with {proc_name}: {e}")
            errors += 1
            time.sleep(2)
    
    print("\n" + "=" * 70)
    print("POLICY GENERATION COMPLETE")
    print("=" * 70)
    print(f" Generated: {generated} policies")
    print(f" Errors: {errors}")
    print(f" Total policies: {len(os.listdir(output_dir))}")
    print(f" Cost: $0")

if __name__ == "__main__":
    main()
