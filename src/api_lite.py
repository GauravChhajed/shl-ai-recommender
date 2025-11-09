import os
from typing import Optional, List
import pandas as pd
import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CAT_PATH = os.getenv("CATALOG_PATH", "data/catalog_clean.csv")

# ---------- Load catalog ----------
df = pd.read_csv(CAT_PATH)
df["name"] = df["name"].fillna("")
df["desc"] = df["desc"].fillna("")

# Weight title higher than description
weighted_texts = (df["name"].str.strip() + " " + df["name"].str.strip() + " || " + df["desc"].str.strip()).tolist()

# ---------- Vectorizer (light but decent) ----------
vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    stop_words="english",
    sublinear_tf=True,
    max_df=0.9
)
X = vectorizer.fit_transform(weighted_texts)

KEYWORD_WEIGHTS = {
    "sjt": 0.06,
    "situational judgement": 0.06,
    "situational judgment": 0.06,
    "cognitive": 0.06,
    "aptitude": 0.06,
    "numerical": 0.05,
    "verbal": 0.05,
    "personality": 0.06,
    "behavioural": 0.05,
    "behavioral": 0.05,
    "coding": 0.06,
    "programming": 0.05,
    "language": 0.04,
    "business": 0.04,
    "motivation": 0.04,
}

def keyword_boost(query: str, base_scores: np.ndarray) -> np.ndarray:
    q = query.lower()
    bonus = sum(w for k, w in KEYWORD_WEIGHTS.items() if k in q)
    return base_scores + (bonus * 0.25) if bonus else base_scores

def recommend_tfidf(query: str, top_k: int = 10):
    q_vec = vectorizer.transform([query])
    sims_full = cosine_similarity(q_vec, X)[0]
    idxs = np.argsort(-sims_full)[:max(10, top_k)]
    sims = sims_full[idxs]
    sims = keyword_boost(query, sims)
    order = np.argsort(-sims)[:min(max(top_k, 1), 10)]
    final_idxs = [idxs[i] for i in order]
    out = []
    for i in final_idxs:
        row = df.iloc[int(i)]
        out.append({
            "assessment_name": str(row["name"]),
            "url": str(row["url"]),
            "score": float(sims[order[list(final_idxs).index(i)]]) if hasattr(sims, "__len__") else float(sims),
        })
    return out

# ---------- FastAPI ----------
app = FastAPI(title="SHL Assessment Recommender (Lite)", version="1.1.0")

class RecommendRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10

@app.get("/")
def root():
    return {"status": "ok", "message": "See /docs for interactive API.", "backend": "lite"}

@app.get("/health")
def health():
    return {"status": "ok", "backend": "lite"}

@app.post("/recommend")
def recommend(req: RecommendRequest):
    k = req.top_k if req.top_k is not None else 10
    k = min(max(int(k), 1), 10)
    recs = recommend_tfidf(req.query, top_k=k)
    return {"query": req.query, "backend": "lite", "recommendations": recs}
