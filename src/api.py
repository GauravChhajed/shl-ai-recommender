from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from src.recommend import Recommender

app = FastAPI(title="SHL Assessment Recommender", version="1.0.0")
rec = Recommender()

class RecommendRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/recommend")
def recommend(req: RecommendRequest):
    top_k = min(max(req.top_k or 10, 1), 10)
    out = rec.recommend(req.query, top_k=top_k)
    return {"query": req.query, "recommendations": out}
