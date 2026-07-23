import streamlit as st
import requests
import os

API_URL = os.environ.get("API_URL", "http://localhost:8000")
API_KEY = os.environ.get("API_KEY", "")

st.set_page_config(page_title="Enterprise Hybrid GraphRAG", layout="wide")
st.title("Enterprise Hybrid GraphRAG")

with st.sidebar:
    st.header("How it works")
    st.markdown("""
    1. Upload a PDF document
    2. System extracts text, chunks, embeddings
    3. Entities and relationships extracted into Neo4j graph
    4. Query routes through vector + BM25 + graph search
    5. Cross-encoder reranks results
    6. LLM generates grounded answer with citations
    """)
    st.divider()
    st.markdown("**Retrieval types:**")
    st.markdown("- **Vector** — semantic search\n- **Graph** — entity relationships\n- **Hybrid** — both combined")
    st.divider()
    st.caption("Built with FastAPI + pgvector + Neo4j + Groq")

tab_upload, tab_query, tab_docs = st.tabs(["Upload", "Query", "Documents"])

with tab_upload:
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    force = st.checkbox("Force re-upload (replace existing)")
    if uploaded_file and st.button("Ingest"):
        with st.spinner("Processing..."):
            files = {"file": (uploaded_file.name, uploaded_file.read(), "application/pdf")}
            res = requests.post(f"{API_URL}/upload?force={force}", files=files, headers={"X-API-Key": API_KEY})
            if res.status_code == 200:
                data = res.json()
                st.success(f"Done! {data['chunks_inserted']} chunks, {data['entities_extracted']} entities, {data['relationships_found']} relationships")
            else:
                st.error(res.json().get("detail", "Upload failed"))

with tab_query:
    session_id = st.text_input("Session ID (for conversation memory)", value="default")
    question = st.text_area("Ask a question", height=100)
    use_graph = st.checkbox("Use knowledge graph", value=True)
    top_k = st.slider("Top K results", 3, 10, 5)

    if st.button("Ask") and question:
        with st.spinner("Searching..."):
            res = requests.post(
                f"{API_URL}/query",
                json={"question": question, "top_k": top_k, "session_id": session_id, "use_graph": use_graph},
                headers={"X-API-Key": API_KEY},
            )
            if res.status_code == 200:
                data = res.json()
                st.markdown(f"**Retrieval type:** `{data['retrieval_type']}`")
                st.markdown(f"**Chunks used:** {data['chunks_used']}")
                if data.get("sources"):
                    st.markdown(f"**Sources:** {', '.join(data['sources'])}")
                st.divider()
                st.markdown(data["answer"])
            else:
                st.error(res.json().get("detail", "Query failed"))

with tab_docs:
    if st.button("Refresh"):
        res = requests.get(f"{API_URL}/documents", headers={"X-API-Key": API_KEY})
        if res.status_code == 200:
            docs = res.json()
            if docs:
                for doc in docs:
                    st.markdown(f"**{doc['filename']}** — {doc['chunks']} chunks, {doc['entities']} entities")
            else:
                st.info("No documents uploaded yet")
