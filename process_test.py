"""Quick test with 100 cases"""

import google.generativeai as genai
import os
from dotenv import load_dotenv
import json
import pandas as pd
from tqdm import tqdm
import time

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-pro')

generation_config = {
    "temperature": 0.2,
    "top_p": 0.8,
    "top_k": 40,
    "max_output_tokens": 1024,
}

def extract_clinical_info(text, case_id, specialty):
    """Extract structured clinical information"""
    
    text = str(text)[:5000]
    
    prompt = f"""Extract clinical information from this medical document as JSON.

SPECIALTY: {specialty}
DOCUMENT: {text}

Return JSON with:
- patient_demographics (age, gender, chief_complaint)
- clinical_information (diagnosis, symptoms, symptom_duration, physical_exam_findings)
- diagnostic_tests (imaging, labs, other_tests)
- treatment (procedure_performed, procedure_planned, medications, conservative_treatments)
- clinical_assessment (severity, urgency, prognosis)

Return ONLY valid JSON, no markdown."""

    try:
        response = model.generate_content(prompt, generation_config=generation_config)
        result = response.text.replace('```json', '').replace('```', '').strip()
        parsed = json.loads(result)
        parsed['meta'] = {
            'case_id': case_id,
            'original_specialty': specialty,
            'processing_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        return parsed
    except:
        return None

# Read CSV
df = pd.read_csv("data/raw/mtsamples.csv")
df = df.dropna(subset=['transcription'])
df = df.head(100)  # ONLY 100 CASES

os.makedirs("data/processed/cases", exist_ok=True)

processed = 0
print("\n PROCESSING 100 TEST CASES")
print("=" * 70)

for idx, row in tqdm(df.iterrows(), total=len(df)):
    case_id = f"case_{idx:04d}"
    
    try:
        extracted = extract_clinical_info(
            str(row['transcription']),
            case_id,
            str(row['medical_specialty'])
        )
        
        if extracted:
            extracted['original_data'] = {
                'index': int(idx),
                'specialty': str(row['medical_specialty']),
                'sample_name': str(row['sample_name'])
            }
            
            output_path = f"data/processed/cases/{case_id}.json"
            with open(output_path, 'w') as f:
                json.dump(extracted, f, indent=2)
            
            processed += 1
        
        time.sleep(2)  # Rate limiting
        
    except Exception as e:
        print(f"\n Error: {e}")
        time.sleep(5)

print(f"\n Processed {processed} cases")
print(f" Saved to: data/processed/cases/")
