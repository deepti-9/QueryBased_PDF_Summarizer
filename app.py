import streamlit as st
import os
import time

os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]

from dotenv import load_dotenv
from langchain_groq import ChatGroq

# ✅ Correct imports
from src.data_loader import PDFProcessor
from src.embedding import EmbeddingManager
from src.vectorstore import VectorStore
from src.rag_system import RAGSystem   # ✅ fixed

load_dotenv()

# --- Page Config ---
st.set_page_config(page_title="PDF RAG Assistant", layout="wide")
st.title("📚 Dynamic PDF Summarizer & Q&A")

# --- Initialize Backend ---
@st.cache_resource
def init_rag():
    llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")
    embeddings = EmbeddingManager(batch_size=64)
    store = VectorStore(batch_size=128)
    return RAGSystem(store, embeddings, llm)

rag = init_rag()

# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")

    top_k = st.slider("Top K (Context Chunks)", 1, 10, 5)
    threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.35)

    st.markdown("---")

    uploaded_files = st.file_uploader(
        "Upload PDF Documents", type="pdf", accept_multiple_files=True
    )

    if st.button("Process & Index Documents"):
        if uploaded_files:

            os.makedirs("data/pdf_files", exist_ok=True)

            # Save files
            for uploaded_file in uploaded_files:
                with open(os.path.join("data/pdf_files", uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())

            with st.spinner("⚡ Fast ingestion in progress..."):

                start_total = time.time()

                # ✅ CLEAN (no image dependency)
                processor = PDFProcessor(
                    llm=rag.llm,
                    vision_func=None,
                    process_images=False
                )

                # Step 1: Extract
                t1 = time.time()
                raw_docs = processor.process_pdfs("data/pdf_files")
                st.info(f"📄 Extracted {len(raw_docs)} docs in {time.time() - t1:.2f}s")

                # Step 2: Split
                t2 = time.time()
                chunks = processor.split_documents(raw_docs)
                st.info(f"✂️ Created {len(chunks)} chunks in {time.time() - t2:.2f}s")

                # Step 3: Embed
                t3 = time.time()
                contents = [doc.page_content for doc in chunks]
                embeddings = rag.embedding_manager.generate_embeddings(contents)
                st.info(f"🧠 Embeddings generated in {time.time() - t3:.2f}s")

                # Validation
                assert len(chunks) == len(embeddings), "Mismatch error!"

                # Step 4: Store
                t4 = time.time()
                rag.vector_store.add_documents(chunks, embeddings)
                st.info(f"💾 Stored in {time.time() - t4:.2f}s")

                st.success(f"✅ Done in {time.time() - start_total:.2f}s")

        else:
            st.warning("Please upload a PDF first.")

# --- Chat Interface ---
query = st.text_input("Ask a question about your documents:")
submit_button = st.button("Send")

if query and submit_button:
    with st.spinner("Thinking..."):

        result = rag.ask(query, top_k=top_k, score_threshold=threshold)

        st.markdown("### 🤖 Assistant")
        st.markdown(result['answer'])

        st.markdown("---")

        m1, m2, m3, m4 = st.columns(4)

        first_source = result['sources'][0] if result['sources'] else {}

        m1.metric("Author", first_source.get("author", "N/A"))
        m2.metric("Date", first_source.get("date", "N/A"))

        # ✅ FIXED METRIC KEY
        m3.metric("Relevance Score", result['metrics'].get("relevance_score", "0%"))

        m4.metric("Time Taken", result['metrics'].get("fetch_time", "0s"))

        # Source viewer
        with st.expander("📚 View All Source Chunks"):
            for idx, source in enumerate(result['sources']):
                st.write(
                    f"📄 **Source {idx+1}** | Page {source.get('page')} | {source.get('source_file')}"
                )