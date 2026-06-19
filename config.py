# config.py — All settings in one place
# Change model names, chunk sizes, etc. here

OLLAMA_MODEL  = "llama3.2:3b"
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_PATH   = "./chroma_db"
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50
TOP_K         = 4
MEMORY_TURNS  = 6
