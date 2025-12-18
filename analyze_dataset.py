"""
Analyze MTSamples Dataset
"""

import pandas as pd
import json

def analyze_dataset(filepath="data/raw/mtsamples.csv"):
    """Analyze the CSV structure and content"""
    
    print("\n" + "=" * 70)
    print("MTSAMPLES DATASET ANALYSIS")
    print("=" * 70)
    
    # Read CSV
    df = pd.read_csv(filepath)
    
    # Remove rows with missing transcriptions
    df_clean = df.dropna(subset=['transcription'])
    
    print(f"\n DATASET SUMMARY")
    print("-" * 70)
    print(f"Total rows: {len(df)}")
    print(f"Valid transcriptions: {len(df_clean)}")
    print(f"Missing transcriptions: {len(df) - len(df_clean)}")
    
    print(f"\n COLUMNS")
    print("-" * 70)
    for col in df.columns:
        print(f"  • {col}")
    
    print(f"\n MEDICAL SPECIALTIES ({df_clean['medical_specialty'].nunique()} unique)")
    print("-" * 70)
    specialty_counts = df_clean['medical_specialty'].value_counts()
    for specialty, count in specialty_counts.head(15).items():
        print(f"  {specialty:35s}: {count:4d} cases")
    
    print(f"\n📏 TEXT LENGTH STATISTICS")
    print("-" * 70)
    text_lengths = df_clean['transcription'].str.len()
    print(f"  Average length: {text_lengths.mean():.0f} characters")
    print(f"  Median length: {text_lengths.median():.0f} characters")
    print(f"  Min length: {text_lengths.min():.0f} characters")
    print(f"  Max length: {text_lengths.max():.0f} characters")
    
    print(f"\n SAMPLE CASE")
    print("-" * 70)
    sample = df_clean.iloc[0]
    print(f"Specialty: {sample['medical_specialty']}")
    print(f"Sample Name: {sample['sample_name']}")
    print(f"Description: {sample['description'][:100]}...")
    print(f"\nTranscription (first 300 chars):")
    print(sample['transcription'][:300] + "...")
    
    # Save analysis
    analysis = {
        'total_cases': len(df_clean),
        'specialties': specialty_counts.to_dict(),
        'avg_length': int(text_lengths.mean()),
        'columns': list(df.columns)
    }
    
    with open('data/dataset_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\n Analysis saved to: data/dataset_analysis.json")
    print("=" * 70)
    
    return df_clean

if __name__ == "__main__":
    df = analyze_dataset()
    print(f"\n Ready to process {len(df)} medical cases!")
