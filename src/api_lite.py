import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# File path
CAT_PATH = os.getenv("CATALOG_PATH", "data/catalog_clean.csv")

# Load catalog and build TF-IDF model
df = pd.read_csv(CAT_PATH)
df["name"] = df["name"].fillna("")
df["desc"] = df["desc"].fillna("")
texts = (df["name"] + " || " + df["desc"]).tolist()

vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), max_df=0.9)
X = vectorizer.fit_transform(texts)

def recommend_tfidf(query: str, top_k: int = 10):
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, X)[0]
    idxs = np.argsort(-sims)[:top_k]
    results = []
    for i in idxs:
        results.append({
            "assessment_name": df.iloc[i]["name"],
            "url": df.iloc[i]["url"],
            "score": float(sims[i])
        })
    return results

# ---- FastAPI app ----
app = FastAPI(title="SHL Assessment Recommender (Lite)", version="1.0.0")

class RecommendRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/recommend")
def recommend(req: RecommendRequest):
    k = min(max(req.top_k or 10, 1), 10)
    recs = recommend_tfidf(req.query, top_k=k)
    return {"query": req.query, "recommendations": recs}
