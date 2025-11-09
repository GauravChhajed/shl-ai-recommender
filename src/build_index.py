import json, numpy as np, pandas as pd, faiss
from sentence_transformers import SentenceTransformer

CAT = "data/catalog_clean.csv"
EMB = "data/embeddings.npy"
IDX = "data/index.faiss"
META = "data/meta.json"

def main():
    df = pd.read_csv(CAT)
    df["name"] = df["name"].fillna("")
    df["desc"] = df["desc"].fillna("")
    texts = (df["name"] + " || " + df["desc"]).tolist()

    model_name = "all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    embs = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    embs = np.asarray(embs, dtype="float32")

    index = faiss.IndexFlatIP(embs.shape[1])  # cosine (since normalized)
    index.add(embs)

    np.save(EMB, embs)
    faiss.write_index(index, IDX)
    with open(META, "w") as f:
        json.dump({"model": model_name, "dim": int(embs.shape[1]), "n": int(len(texts)), "normalized": True}, f, indent=2)

    print(f"Built FAISS index with {len(texts)} items.")

if __name__ == "__main__":
    main()
