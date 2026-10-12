from pathlib import Path
import json
import streamlit as st
from src.rag import RegulatoryQA, list_pdf_files

st.set_page_config(page_title="Regulatory Desk", page_icon="⚖️", layout="wide")
st.title("Regulatory Desk")
st.caption("Search your own SEBI, RBI, and other regulatory PDFs with page-aware citations.")

with st.sidebar:
    st.header("Search settings")
    top_k = st.slider("Passages to retrieve", 2, 8, 4)
    min_score = st.slider("Minimum relevance score", 0.05, 0.80, 0.25, 0.05)
    show_context = st.checkbox("Show retrieved passages", True)
    if st.button("Reload PDF library"):
        st.cache_resource.clear()
        st.rerun()

DATA_DIR = Path("data/regulations")
DATA_DIR.mkdir(parents=True, exist_ok=True)
pdf_paths = tuple(str(p) for p in list_pdf_files(DATA_DIR))
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

@st.cache_resource(show_spinner="Loading embedding model and indexing PDFs...")
def get_qa(paths, model_name):
    return RegulatoryQA(pdf_paths=paths, model_name=model_name)

if not pdf_paths:
    st.warning("No PDFs found in data/regulations yet.")
    st.markdown("Add text-based PDFs to `data/regulations/`, then click **Reload PDF library**.")
    st.markdown("Download current official documents from [SEBI](https://www.sebi.gov.in/) or [RBI](https://www.rbi.org.in/).")
    st.stop()

st.success(f"Loaded {len(pdf_paths)} PDF file(s).")
question = st.text_input("Ask a question", placeholder="Example: What does this circular say about disclosure requirements?")
if st.button("Find answer", type="primary", disabled=not question.strip()):
    try:
        qa = get_qa(pdf_paths, MODEL_NAME)
        result = qa.answer(question, top_k=top_k, min_score=min_score)
        st.subheader("Answer")
        st.write(result["answer"])
        if result["sources"]:
            st.subheader("Page-level citations")
            for i, src in enumerate(result["sources"], 1):
                st.markdown(f"**[{i}] {src['filename']} — page {src['page']}** · relevance {src['score']:.2f}")
        if show_context:
            st.subheader("Retrieved passages")
            for item in result["retrieved"]:
                with st.expander(f"{item['filename']} · page {item['page']} · {item['score']:.2f}"):
                    st.write(item["text"])
        st.download_button(
            "Download result as JSON",
            data=json.dumps(result, indent=2, ensure_ascii=False),
            file_name="regulatory_answer.json",
            mime="application/json",
        )
    except Exception as exc:
        st.error(f"Could not search the library: {exc}")
        st.caption("Check that the PDFs contain selectable text and that the embedding model downloaded successfully.")
