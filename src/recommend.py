import json, faiss, numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer
from src.rerank import rerank, detect_query_needs

CAT = "data/catalog_clean.csv"
IDX = "data/index.faiss"
META = "data/meta.json"

def diversify_by_type(df_sorted, query, top_k=10):
    """Ensure a K/P balance if the query implies both."""
    needs_k, needs_p = detect_query_needs(query)
    if not (needs_k or needs_p):
        return df_sorted.head(top_k)

    # Try to include at least one from each needed type
    picks = []
    used = set()

    def pick_of_type(t):
        for i, row in df_sorted.iterrows():
            if i in used: 
                continue
            if (row.get("type_hint") == t):
                picks.append(row)
                used.add(i)
                return True
        return False

    # If both needed, pick one of each first
    if needs_k: pick_of_type("K")
    if needs_p: pick_of_type("P")

    # Fill the rest by score
    for i, row in df_sorted.iterrows():
        if len(picks) >= top_k: break
        if i in used: 
            continue
        picks.append(row)

    return pd.DataFrame(picks).head(top_k)

class Recommender:
    def __init__(self):
        self.df = pd.read_csv(CAT)
        with open(META, "r") as f:
            meta = json.load(f)
        self.model = SentenceTransformer(meta["model"])
        self.index = faiss.read_index(IDX)

    def retrieve(self, query, topn=30):
        q = self.model.encode([query], normalize_embeddings=True).astype("float32")
        sims, idxs = self.index.search(q, topn)
        return sims[0], idxs[0]

    def recommend(self, query, top_k=10):
        sims, idxs = self.retrieve(query, topn=max(30, top_k))
        cand = self.df.iloc[idxs].copy().reset_index(drop=True)
        scores = rerank(query, cand, sims)
        cand["score"] = scores
        cand = cand.sort_values("score", ascending=False)
        cand = diversify_by_type(cand, query, top_k=top_k)
        return [
            {"assessment_name": r["name"], "url": r["url"], "score": float(r["score"])}
            for _, r in cand.iterrows()
        ]

if __name__ == "__main__":
    r = Recommender()
    print(r.recommend("mid-level python sql javascript plus teamwork and collaboration", top_k=7))
