# ============================================================
# app.py
# ============================================================

import os
import time
import shutil

import streamlit as st

os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from src.data_loader import PDFProcessor
from src.embedding   import EmbeddingManager
from src.vectorstore import VectorStore
from src.rag_system  import RAGSystem
from src.image_processor import ImageProcessor
from config import (         
    LLM_MODEL,
    EMBED_BATCH_SIZE,
    STORE_BATCH_SIZE,
    PDF_DIR,
    TOP_K_DEFAULT,
    SCORE_THRESHOLD,
)

load_dotenv()

# --- Page config ---
st.set_page_config(page_title="PDF RAG Assistant", layout="wide")
st.title("📚 Research Paper Q&A")

# --- Backend init ---
@st.cache_resource
def init_rag():
    llm        = ChatGroq(model=LLM_MODEL)          
    embeddings = EmbeddingManager(batch_size=EMBED_BATCH_SIZE)
    store      = VectorStore(batch_size=STORE_BATCH_SIZE)
    image_proc = ImageProcessor()      
    rag        = RAGSystem(store, embeddings, llm)
    return rag,image_proc

rag = init_rag()
rag, image_processor = init_rag()

#  chat history persisted across reruns
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")

    top_k     = st.slider("Top K (context chunks)", 1, 10, TOP_K_DEFAULT)
    threshold = st.slider("Similarity threshold",   0.0, 1.0, SCORE_THRESHOLD)

    st.markdown("---")

    uploaded_files = st.file_uploader(
        "Upload PDF documents", type="pdf", accept_multiple_files=True
    )

    if st.button("⚡ Process & index documents"):
        if uploaded_files:
            os.makedirs(PDF_DIR, exist_ok=True)

            for uploaded_file in uploaded_files:
                dest = os.path.join(PDF_DIR, uploaded_file.name)
                with open(dest, "wb") as f:
                    f.write(uploaded_file.getbuffer())

            with st.spinner("Ingesting…"):
                start_total = time.time()

                processor = PDFProcessor(
                    llm=rag.llm,
                    vision_func=image_processor.get_image_description,
                    process_images=True,
                )

                t1       = time.time()
                raw_docs = processor.process_pdfs(PDF_DIR)
                st.info(f"📄 Extracted {len(raw_docs)} pages in {time.time()-t1:.2f}s")

                t2     = time.time()
                chunks = processor.split_documents(raw_docs)
                st.info(f"✂️ Created {len(chunks)} chunks in {time.time()-t2:.2f}s")

                t3       = time.time()
                contents = [doc.page_content for doc in chunks]

                #  unpack tuple; filter chunks to valid indices only
                embeddings, valid_indices = rag.embedding_manager.generate_embeddings(contents)
                chunks = [chunks[i] for i in valid_indices]
                st.info(f"🧠 Embeddings generated in {time.time()-t3:.2f}s")

                # This assert will now always pass
                assert len(chunks) == len(embeddings), "Mismatch error!"

                t4 = time.time()
                rag.vector_store.add_documents(chunks, embeddings)
                st.info(f"💾 Stored in {time.time()-t4:.2f}s")

                st.success(f"✅ Done in {time.time()-start_total:.2f}s")
        else:
            st.warning("Please upload at least one PDF first.")

    st.markdown("---")

    #  clear button so stale PDFs don't pile up across sessions
    if st.button("🗑️ Clear all documents"):
        shutil.rmtree(PDF_DIR, ignore_errors=True)
        os.makedirs(PDF_DIR, exist_ok=True)
        rag.vector_store.reset()          # uses new reset() method
        st.session_state.messages = []   # also clear chat history
        st.success("Cleared. Re-upload your documents.")

# --- Chat history display ---
#  render full conversation history on every rerun
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Source chunks"):
                for idx, (source, chunk) in enumerate(
                    zip(msg["sources"], msg["chunks"])
                ):
                    st.markdown(
                        f"**Source {idx+1}** · Page {source.get('page')} "
                        f"· `{source.get('source_file')}`"
                    )
                    #  show actual text the LLM read
                    st.caption(chunk["content"][:400])
                    if idx < len(msg["sources"]) - 1:
                        st.divider()

# --- Chat input ---
#  st.chat_input replaces text_input + button pattern
prompt = st.chat_input("Ask a question about your documents…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = rag.ask(prompt, top_k=top_k, score_threshold=threshold)
            answer = result["answer"]

        st.markdown(answer)

        # Metrics row
        if result["sources"]:
            m1, m2, m3, m4 = st.columns(4)
            first = result["sources"][0]
            m1.metric("Author",          first.get("author", "N/A"))
            m2.metric("Date",            first.get("date",   "N/A"))
            m3.metric("Relevance score", result["metrics"].get("relevance_score", "0%"))
            m4.metric("Time taken",      result["metrics"].get("fetch_time",      "0s"))

        # Source chunks expander
        if result["sources"]:
            with st.expander("📚 Source chunks"):
                for idx, (source, chunk) in enumerate(
                    zip(result["sources"], result["chunks"])
                ):
                    st.markdown(
                        f"**Source {idx+1}** · Page {source.get('page')} "
                        f"· `{source.get('source_file')}`"
                    )
                    st.caption(chunk["content"][:400])  # ✅ ENHANCEMENT: actual text
                    if idx < len(result["sources"]) - 1:
                        st.divider()

    # Persist to session state (include chunks for history re-render)
    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "sources": result["sources"],
        "chunks":  result["chunks"],   # stored for history
    })