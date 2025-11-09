import sys
from pathlib import Path
import time
import requests
import pandas as pd
import streamlit as st

# Ensure project root is on PYTHONPATH when running "streamlit run src/ui_app.py"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Local engine (FastAPI not required for local mode)
try:
    from src.recommend import Recommender
    from src.rerank import detect_query_needs
    LOCAL_ENGINE_AVAILABLE = True
except Exception:
    LOCAL_ENGINE_AVAILABLE = False

st.set_page_config(page_title="SHL Assessment Recommender", layout="centered")

st.title("🔎 SHL Assessment Recommender")
st.caption("Paste a JD / query and get the most relevant **individual test solutions**.")

# ---------- Sidebar ----------
st.sidebar.header("⚙️ Settings")

mode = st.sidebar.radio(
    "Inference Mode",
    options=["Local (Python)", "Remote API"],
    help="Local uses the Python recommender directly. Remote calls your deployed FastAPI (/recommend).",
)

top_k = st.sidebar.slider("Top K", min_value=5, max_value=10, value=7, step=1)

api_base = ""
if mode == "Remote API":
    api_base = st.sidebar.text_input(
        "Remote API Base URL",
        value="http://127.0.0.1:8080",
        help="Example: https://your-app.onrender.com",
    )

st.sidebar.markdown("---")
st.sidebar.subheader("Examples")
examples = [
    "Looking to hire mid-level professionals who are proficient in Python, SQL and JavaScript.",
    "I am hiring for Java developers who can also collaborate effectively with my business teams.",
    "Hiring an analyst; want Cognitive and Personality screening.",
]
ex_choice = st.sidebar.selectbox("Load example", ["(none)"] + examples)

# ---------- Main input ----------
query = st.text_area(
    "Paste JD / Query",
    value=("" if ex_choice == "(none)" else ex_choice),
    height=160,
    placeholder="e.g., Need a Java developer who collaborates with stakeholders and cross-functional teams…",
)

# ---------- Lazy-initialize local engine ----------
if mode == "Local (Python)":
    if not LOCAL_ENGINE_AVAILABLE:
        st.warning(
            "Local engine not available (imports failed). "
            "Ensure your project is installed and run from repo root: `streamlit run src/ui_app.py`."
        )
    else:
        if "recommender" not in st.session_state:
            with st.spinner("Loading local recommender (embeddings & index)…"):
                st.session_state.recommender = Recommender()

go = st.button("Get Recommendations", type="primary")

# ---------- Action ----------
if go and query.strip():
    start = time.time()
    try:
        if mode == "Local (Python)":
            if not LOCAL_ENGINE_AVAILABLE:
                st.error("Local engine unavailable. Switch to Remote API mode.")
            else:
                recs = st.session_state.recommender.recommend(query, top_k=top_k)
        else:
            # Remote call
            url = api_base.rstrip("/") + "/recommend"
            payload = {"query": query, "top_k": top_k}
            r = requests.post(url, json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            recs = data.get("recommendations", [])

        if not recs:
            st.info("No recommendations returned.")
        else:
            # Make a nice dataframe; keep URLs clickable
            df = pd.DataFrame(recs)
            # If assessment_name missing (shouldn’t), fallback to URL tail
            if "assessment_name" not in df.columns:
                df["assessment_name"] = df["url"].str.rsplit("/", n=2).str[-2].str.replace("-", " ").str.title()

            # Show K/P hint if available locally
            if mode == "Local (Python)":
                try:
                    needs_k, needs_p = detect_query_needs(query)
                    balance_hint = []
                    for _, row in st.session_state.recommender.df.iterrows():
                        pass
                except Exception:
                    needs_k = needs_p = False

            st.subheader("Results")
            show = df[["assessment_name", "url", "score"]] if "score" in df.columns else df[["assessment_name", "url"]]
            st.dataframe(show, use_container_width=True)

            with st.expander("Raw JSON"):
                st.write(recs)

            st.caption(f"⏱️ Completed in {time.time() - start:.2f}s")

    except requests.exceptions.RequestException as e:
        st.error(f"Remote API error: {e}")
    except Exception as e:
        st.exception(e)

st.markdown("---")
st.caption(
    "Tip: For a public demo, deploy the FastAPI backend to **Render** and (optionally) host this Streamlit app on **Hugging Face Spaces**."
)
