import numpy as np

def detect_query_needs(q: str):
    q = q.lower()
    needs_k = any(k in q for k in [
        "java","python","sql","c++","ml","nlp","analytics","coding","technical",
        "engineer","developer","aptitude","cognitive","data","javascript"
    ])
    needs_p = any(k in q for k in [
        "personality","behavior","behaviour","teamwork","collaboration","communication",
        "stakeholder","leadership","work style","sjt","situational"
    ])
    return needs_k, needs_p

def rerank(query, df_cand, dense_scores):
    needs_k, needs_p = detect_query_needs(query)
    # + type balance bonus from type_hint (K/P)
    type_bonus = []
    for t in df_cand["type_hint"].fillna(""):
        bonus = 0.0
        if needs_k and t == "K": bonus += 0.06
        if needs_p and t == "P": bonus += 0.06
        type_bonus.append(bonus)
    type_bonus = np.array(type_bonus)

    # light keyword overlap on name+desc
    q_terms = [w for w in query.lower().split() if len(w) > 2]
    kw_overlap = []
    for _, row in df_cand.iterrows():
        text = (str(row["name"]) + " " + str(row["desc"])).lower()
        hits = sum(1 for w in q_terms if w in text)
        kw_overlap.append(min(hits/10.0, 0.2))
    kw_overlap = np.array(kw_overlap)

    final = 0.72*dense_scores + 0.18*kw_overlap + 0.10*type_bonus
    return final
