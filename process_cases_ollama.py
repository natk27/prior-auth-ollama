"""
Process All MTSamples Cases with Ollama (UNLIMITED, FREE)
"""

import requests
import os
import json
import pandas as pd
from tqdm import tqdm
import time

def extract_clinical_info_ollama(text, case_id, specialty):
    """Extract structured clinical information using Ollama"""
    
    text = str(text)[:5000]
    
    prompt = f"""Extract clinical information from this medical document. Return ONLY valid JSON, no other text.

SPECIALTY: {specialty}

DOCUMENT:
{text}

Return JSON with this exact structure:
{{
    "patient_demographics": {{
        "age": "patient age or null",
        "gender": "patient gender or null",
        "chief_complaint": "main presenting problem"
    }},
    "clinical_information": {{
        "diagnosis": "primary diagnosis",
        "symptoms": "key symptoms",
        "symptom_duration": "duration or null",
        "physical_exam_findings": "key findings"
    }},
    "diagnostic_tests": {{
        "imaging": "imaging studies or null",
        "labs": "lab tests or null",
        "other_tests": "other tests or null"
    }},
    "treatment": {{
        "procedure_performed": "procedure done or null",
        "procedure_planned": "procedure planned or null",
        "medications": "medications or null",
        "conservative_treatments": "non-surgical treatments or null"
    }},
    "clinical_assessment": {{
        "severity": "mild/moderate/severe or null",
        "urgency": "elective/urgent/emergent or null",
        "prognosis": "expected outcome or null"
    }}
}}

Extract only explicitly stated information. Use null for missing data."""

    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'llama3.2',
                'prompt': prompt,
                'stream': False,
                'format': 'json'
            },
            timeout=60
        )
        
        if response.status_code != 200:
            return None
        
        result = response.json()['response']
        
        # Clean and parse
        result = result.replace('```json', '').replace('```', '').strip()
        parsed = json.loads(result)
        
        # Add metadata
        parsed['meta'] = {
            'case_id': case_id,
            'original_specialty': specialty,
            'processing_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return parsed
        
    except json.JSONDecodeError:
        return None
    except Exception as e:
        return None

def process_all_cases(
    input_file="data/raw/mtsamples.csv",
    output_dir="data/processed/cases"
):
    """Process all cases"""
    
    print("\n" + "=" * 70)
    print("PROCESSING WITH OLLAMA (UNLIMITED & FREE!)")
    print("=" * 70)
    
    # Read CSV
    print(f"\n Reading {input_file}...")
    df = pd.read_csv(input_file)
    df = df.dropna(subset=['transcription'])
    df = df[df['transcription'].str.len() >= 100]
    
    print(f" Loaded {len(df)} valid cases")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Check already processed
    already_processed = set()
    if os.path.exists(output_dir):
        already_processed = {
            f.replace('.json', '') 
            for f in os.listdir(output_dir) 
            if f.endswith('.json')
        }
    
    to_process = len(df) - len(already_processed)
    
    print(f"\n PROCESSING STATUS")
    print("-" * 70)
    print(f"Total cases: {len(df)}")
    print(f"Already processed: {len(already_processed)}")
    print(f"To process: {to_process}")
    
    # Estimate time (3-4 seconds per case with Ollama)
    estimated_hours = (to_process * 3.5) / 3600
    
    print(f"\n TIME ESTIMATE")
    print("-" * 70)
    print(f"Rate: ~3.5 seconds per case (Ollama on Mac)")
    print(f"Estimated time: {estimated_hours:.1f} hours")
    print(f" Cost: $0 (runs locally!)")
    print(f" NO RATE LIMITS!")
    
    if to_process == 0:
        print("\n All cases already processed!")
        return
    
    response = input(f"\n  Process {to_process} cases? (yes/no): ")
    if response.lower() != 'yes':
        print(" Cancelled")
        return
    
    # Process cases
    processed = 0
    skipped = 0
    errors = 0
    
    print("\n Processing cases...")
    print(" This runs UNLIMITED - no daily limits!\n")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
        case_id = f"case_{idx:04d}"
        
        # Skip if already processed
        if case_id in already_processed:
            skipped += 1
            continue
        
        try:
            text = str(row['transcription'])
            specialty = str(row['medical_specialty'])
            
            # Extract info
            extracted = extract_clinical_info_ollama(text, case_id, specialty)
            
            if extracted:
                # Add original metadata
                extracted['original_data'] = {
                    'index': int(idx),
                    'specialty': specialty,
                    'sample_name': str(row['sample_name']),
                    'description': str(row['description'])
                }
                
                # Save to file
                output_path = os.path.join(output_dir, f"{case_id}.json")
                with open(output_path, 'w') as f:
                    json.dump(extracted, f, indent=2)
                
                processed += 1
            else:
                errors += 1
            
            # Small delay to not overwhelm
            time.sleep(0.2)
            
        except Exception as e:
            errors += 1
            continue
    
    # Save summary
    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)
    print(f" Newly processed: {processed}")
    print(f" Skipped (already done): {skipped}")
    print(f" Errors: {errors}")
    print(f" Total processed: {processed + skipped}")
    print(f" Cost: $0")
    
    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'newly_processed': processed,
        'total_processed': processed + skipped,
        'errors': errors,
        'output_directory': output_dir
    }
    
    with open('data/processing_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n Summary saved to: data/processing_summary.json")

if __name__ == "__main__":
    process_all_cases()
