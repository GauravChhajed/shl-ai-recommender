# SHL Assessment Recommendation System

Recommends relevant **individual SHL assessments** for any JD or text query.

## Stack
- FastAPI backend (Python 3.11), Streamlit UI
- SentenceTransformers (MiniLM-L6-v2) + FAISS (or sklearn fallback)
- Pandas, scikit-learn

## Run API locally
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn src.api:app --host 127.0.0.1 --port 8080
