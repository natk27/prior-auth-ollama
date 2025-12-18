# Prior Authorization AI System (Ollama-Powered)

Production-scale AI system using **Ollama (100% FREE, UNLIMITED)** for medical prior authorization.

## Key Advantages

- **$0 Cost** - Runs completely locally
- **UNLIMITED** - No rate limits
- **Private** - Data never leaves your computer
- **Fast** - Process 4,966 cases in ~5 hours
- **Offline** - Works without internet

## Project Stats

- **Patient Cases:** 4,966 (MTSamples dataset)
- **Insurance Policies:** 50 procedures
- **Vector Search:** FAISS with 384-dim embeddings
- **Processing Time:** ~5 hours for complete dataset
- **Cost:** $0

## Tech Stack

| Component | Technology | Cost |
|-----------|-----------|------|
| LLM | Ollama (Llama 3.2) | FREE |
| Embeddings | Sentence-Transformers | FREE |
| Vector DB | FAISS | FREE |
| Language | Python 3.8+ | FREE |

## Quick Start

### 1. Install Ollama
```bash
brew install ollama
ollama serve  # In separate terminal
ollama pull llama3.2
```

### 2. Setup Project
```bash
git clone https://github.com/YOUR-USERNAME/prior-auth-gemini.git
cd prior-auth-gemini
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Generate Policies (~15 mins)
```bash
python generate_policies_ollama.py
```

### 4. Process Cases (~5 hours)
```bash
python process_cases_ollama.py
# Let it run! Can leave overnight.
```

### 5. Create Embeddings (~10 mins)
```bash
python create_embeddings.py
```

### 6. Test System
```bash
python test_rag.py
python prior_auth_ollama.py
```

## Project Structure
```
prior-auth-gemini/
├── data/
│   ├── raw/mtsamples.csv              # 4,966 cases
│   ├── processed/
│   │   ├── cases/                     # 4,966 JSON files
│   │   └── policies/                  # 50 policies
│   └── embeddings/
│       ├── patient_cases.index        # FAISS index
│       └── metadata.json
├── generate_policies_ollama.py        # Generate policies
├── process_cases_ollama.py            # Extract clinical info
├── create_embeddings.py               # Create RAG index
├── test_rag.py                        # Test search
├── prior_auth_ollama.py               # End-to-end system
├── check_progress.py                  # Monitor progress
└── test_complete_system.py            # System validation
```

## Monitoring

Check progress anytime:
```bash
python check_progress.py
```

## System Requirements

- **macOS** (M1/M2/M3 recommended for speed)
- **8GB RAM minimum** (16GB+ recommended)
- **10GB disk space** (for model + data)
- **Python 3.8+**
