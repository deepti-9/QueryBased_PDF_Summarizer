import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from src.data_loader import PDFProcessor
from src.embedding import EmbeddingManager
from src.vectorstore import VectorStore

load_dotenv()


def run_ingestion():
    start_total = time.time()

    # 1. Initialize Components
    llm = ChatGroq(model="llama-3.3-70b-versatile")

    processor = PDFProcessor(
        llm=llm,
        vision_func=None,   # ✅ cleaned
        process_images=False
    )

    embedding_manager = EmbeddingManager(batch_size=64)
    vector_store = VectorStore(batch_size=128)

    # 2. Extract
    print("\n🚀 Step 1: Extracting PDFs...")
    t1 = time.time()
    raw_docs = processor.process_pdfs("data/pdf_files")
    print(f"✅ Extracted documents: {len(raw_docs)}")
    print(f"⏱️ Time: {time.time() - t1:.2f}s")

    # 3. Split
    print("\n✂️ Step 2: Splitting documents...")
    t2 = time.time()
    chunks = processor.split_documents(raw_docs)
    print(f"✅ Total chunks: {len(chunks)}")
    print(f"⏱️ Time: {time.time() - t2:.2f}s")

    # 4. Prepare texts
    print("\n📝 Step 3: Preparing texts...")
    texts = [doc.page_content for doc in chunks]
    print(f"✅ Texts ready: {len(texts)}")

    # 5. Embed
    print("\n🧠 Step 4: Generating embeddings...")
    t3 = time.time()
    embeddings = embedding_manager.generate_embeddings(texts)
    print(f"✅ Embeddings shape: {embeddings.shape}")
    print(f"⏱️ Time: {time.time() - t3:.2f}s")

    # Validation
    assert len(chunks) == len(embeddings), "❌ Mismatch between chunks and embeddings!"

    # 6. Store
    print("\n💾 Step 5: Storing in vector DB...")
    t4 = time.time()
    vector_store.add_documents(chunks, embeddings)
    print(f"⏱️ Time: {time.time() - t4:.2f}s")

    print("\n🎉 Ingestion Complete!")
    print(f"⏱️ Total Time: {time.time() - start_total:.2f}s")


if __name__ == "__main__":
    run_ingestion()