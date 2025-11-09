# 🧠 SHL Assessment Recommendation System

An intelligent recommendation engine that suggests the most relevant **individual SHL assessments** for any given **Job Description (JD)** or free-text query.

---

## 🚀 Features
- 🔍 **Semantic Search** using `SentenceTransformers (all-MiniLM-L6-v2)`
- ⚙️ **Vector Indexing** with `FAISS` for fast retrieval
- 🧾 **Re-ranking** with query–category heuristics (Knowledge / Personality balance)
- 🧩 **FastAPI** backend with `/recommend` & `/health` endpoints
- 💻 **Streamlit UI** for an interactive search experience
- 📄 **submission.csv** generated for SHL Test-Set

---

## 🧠 Tech Stack
| Component | Technology |
|------------|-------------|
| Backend | FastAPI (Python 3.11) |
| Frontend | Streamlit |
| Model | SentenceTransformer (all-MiniLM-L6-v2) |
| Indexing | FAISS |
| Libraries | pandas, numpy, scikit-learn, tqdm |
| Deployment | Render (API) + Hugging Face Spaces (UI) |

---

## ⚙️ Run Locally

### 1️⃣ Setup Environment
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

---
###2️⃣ Start FastAPI Server
python -m uvicorn src.api:app --host 127.0.0.1 --port 8080


Docs → http://127.0.0.1:8080/docs

Health Check → http://127.0.0.1:8080/health

Example Request:

curl -X POST "http://127.0.0.1:8080/recommend" ^
     -H "Content-Type: application/json" ^
     -d "{\"query\": \"Hiring a Java developer who collaborates with teams\", \"top_k\": 7}"
---
###3️⃣ Launch Streamlit UI
streamlit run src/ui_app.py


The Streamlit app provides:

Local (Python) → runs recommender directly in memory.

Remote API → connects to your deployed FastAPI endpoint (Render).

UI opens at 👉 http://localhost:8501

🌐 Deployment
🔹 FastAPI (Backend) – Render

Connect your GitHub repo → New Web Service

Environment: Python 3

Build Command:

pip install -r requirements.txt


Start Command:

uvicorn src.api:app --host 0.0.0.0 --port 8080


After deployment, test:

https://your-app.onrender.com/health

https://your-app.onrender.com/docs

🔹 Streamlit UI – Hugging Face Spaces

Create new Space → Framework: Streamlit

Upload:

src/ui_app.py
src/recommend.py
src/rerank.py
src/__init__.py
data/catalog_clean.csv
data/index.faiss
data/meta.json
requirements.txt


Set App file → src/ui_app.py

Public URL example:

https://huggingface.co/spaces/<username>/shl-assessment-recommender

###📂 Project Structure
SHL_Recommender/
│
├── src/
│   ├── api.py              # FastAPI app
│   ├── recommend.py        # Core recommender logic
│   ├── rerank.py           # Query-based reranking
│   ├── ui_app.py           # Streamlit user interface
│   └── __init__.py
│
├── data/
│   ├── catalog_clean.csv   # Cleaned SHL product catalog
│   ├── index.faiss         # FAISS vector index
│   ├── meta.json           # Embedding metadata
│   └── Gen_AI Dataset.xlsx # Original dataset (optional)
│
├── submission.csv          # Final predictions
├── requirements.txt        # Dependencies
└── README.md               # Documentation

###📊 Submission Deliverables
Deliverable	Description
🧠 submission.csv	Predictions for 90 queries (Query, Assessment_url)
🌐 API Endpoint	/recommend on Render
💻 Streamlit UI	Hosted on Hugging Face Spaces
📘 GitHub Repo	Full source code + docs
🧾 Approach PDF	2-page summary of model & evaluation
✅ Example Output
[
  {
    "assessment_name": "SHL Coding Skills Assessment and Simulations",
    "url": "https://www.shl.com/products/assessments/skills-and-simulations/coding-simulations/",
    "score": 0.2687
  },
  {
    "assessment_name": "Fast, Simple Technical Skill Assessment",
    "url": "https://www.shl.com/products/assessments/skills-and-simulations/technical-skills/",
    "score": 0.2399
  }
]

###🌐 Live API (Example)
Base URL: https://shl-ai-recommender.onrender.com
GET     /health    → Service status
POST    /recommend → Returns top assessments


Example:

curl -X POST "https://shl-ai-recommender.onrender.com/recommend" \
     -H "Content-Type: application/json" \
     -d '{"query":"Looking for mid-level Python + SQL + JS and teamwork","top_k":7}'

###👨‍💻 Author

Gaurav Chhajed
B.Tech Electrical & Electronics Engineering – NIT Andhra Pradesh
📧 gauravc3082004@gmail.com

