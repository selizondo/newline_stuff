"""
Streamlit web UI for PaperSearch — P4 RAG Research Assistant.

Usage:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from rag_common.chunkers import FixedSizeChunker
from src.config import EmbedModelName
from src.embedders import SentenceTransformersEmbedder
from src.generator import generate_answer
from src.pipeline import RAGPipeline

# ── Page setup ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PaperSearch — RAG Research Assistant",
    page_icon="📄",
    layout="wide",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Configuration")

    index_dir = st.text_input(
        "Index directory",
        value="data/indices/recursive_512_ol100__minilm-l6__dense",
        help="Path to a previously saved FAISS index directory.",
    )
    embedding_model = st.selectbox(
        "Embedding model",
        options=[m.value for m in EmbedModelName],
        index=0,
        help="Must match the model used when building the index.",
    )
    retrieval_method = st.selectbox(
        "Retrieval method",
        options=["dense", "bm25", "hybrid"],
        index=0,
    )
    alpha = st.slider(
        "Hybrid alpha (dense weight)",
        min_value=0.0, max_value=1.0, value=0.6, step=0.05,
        disabled=(retrieval_method != "hybrid"),
        help="Fraction of the final score from dense retrieval (rest from BM25).",
    )
    top_k = st.slider("Top-K chunks to retrieve", min_value=1, max_value=20, value=5)

    try:
        from llm_utils.config import get_settings
        default_model = get_settings().generation_model
    except Exception:
        default_model = "llama-3.3-70b-versatile"

    llm_model = st.text_input("LLM model", value=default_model)
    load_btn = st.button("Load Index", type="primary")

# ── Index loading ──────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading index…", hash_funcs={Path: str})
def _load_pipeline(
    idx_dir: str,
    embed_model: str,
    retrieval: str,
    a: float,
) -> tuple[RAGPipeline | None, str | None]:
    p = Path(idx_dir)
    if not p.exists():
        return None, f"Index directory not found: {p}"
    embedder = SentenceTransformersEmbedder(
        model_name=embed_model,
        cache_dir=Path("data/embed_cache"),
    )
    pipeline = RAGPipeline(
        chunker=FixedSizeChunker(512, 64),  # chunker is irrelevant after load
        embedder=embedder,
        retrieval_method=retrieval,
        alpha=a,
    )
    pipeline.load(p)
    return pipeline, None


if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
    st.session_state.pipeline_err = None

if load_btn:
    pipeline, err = _load_pipeline(index_dir, embedding_model, retrieval_method, alpha)
    st.session_state.pipeline = pipeline
    st.session_state.pipeline_err = err

# ── Main area ──────────────────────────────────────────────────────────────────

st.title("📄 PaperSearch — Research Paper QA")
st.caption("Ask natural-language questions about your ingested research papers.")

if st.session_state.pipeline_err:
    st.error(st.session_state.pipeline_err)
elif st.session_state.pipeline is None:
    st.info(
        "Configure an index in the sidebar and click **Load Index** to begin. "
        "Run `python scripts/ingest.py data/papers/ -o data/indices/my_index` "
        "first if no index exists."
    )
else:
    pipeline: RAGPipeline = st.session_state.pipeline
    st.success(
        f"✅ Index loaded — **{len(pipeline.chunks):,}** chunks "
        f"from **{len(pipeline.documents)}** documents."
    )

    query = st.text_area(
        "Your question",
        height=90,
        placeholder="e.g. What attention mechanism improvements reduce transformer memory usage?",
    )
    ask_btn = st.button("🔍 Ask", type="primary", disabled=not query.strip())

    if ask_btn and query.strip():
        col_ret, col_gen = st.columns(2)

        with st.spinner("Retrieving relevant chunks…"):
            results, retrieval_s = pipeline.query_timed(query.strip(), top_k=top_k)
        col_ret.metric("Retrieval time", f"{retrieval_s * 1000:.0f} ms")

        with st.spinner("Generating answer…"):
            qa = generate_answer(query.strip(), results, model=llm_model)
        col_gen.metric("Generation time", f"{qa.generation_time_s * 1000:.0f} ms")

        # ── Answer ──────────────────────────────────────────────────────────
        st.subheader("Answer")
        st.write(qa.answer)

        # ── Citations ────────────────────────────────────────────────────────
        if qa.citations:
            st.subheader("📎 Citations")
            for c in qa.citations:
                page = f", page {c.page_number}" if c.page_number else ""
                with st.expander(f"{c.source}{page}"):
                    st.write(c.text_snippet)
        else:
            st.caption("No citations extracted — the answer may not have used [N] notation.")

        # ── Source chunks ────────────────────────────────────────────────────
        with st.expander(f"Source chunks used ({len(results)} retrieved)", expanded=False):
            for i, r in enumerate(results, 1):
                src = r.chunk.source or r.chunk.document_id or "unknown"
                score_label = f"score={r.score:.4f}"
                st.markdown(f"**[{i}] {src}** ({score_label})")
                st.text(r.chunk.content[:400])
                st.divider()
