LLM_MODEL   = "meta-llama/llama-4-scout-17b-16e-instruct"
EMBED_MODEL = "all-MiniLM-L6-v2"
 
VECTOR_STORE_DIR   = "data/vector_store"
PDF_DIR            = "data/pdf_files"
COLLECTION_NAME    = "pdf_documents"
 
CHUNK_SIZE         = 1000
CHUNK_OVERLAP      = 150
EMBED_BATCH_SIZE   = 64
STORE_BATCH_SIZE   = 128
TOP_K_DEFAULT      = 5
SCORE_THRESHOLD    = 0.35