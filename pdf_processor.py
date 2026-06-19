# pdf_processor.py — Read PDF and split into chunks

import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP
from resources import get_embedding_model, get_chroma_client


def pdf_to_chunks(pdf_bytes):
    chunks_with_meta = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_num, page in enumerate(doc, start=1):
            page_text = page.get_text()
            if not page_text.strip():
                continue
            for chunk in splitter.split_text(page_text):
                chunks_with_meta.append({"text": chunk, "page": page_num})
    return chunks_with_meta


def index_pdf(pdf_bytes, collection_name):
    embed_model   = get_embedding_model()
    chroma_client = get_chroma_client()

    try:
        chroma_client.delete_collection(collection_name)
    except Exception:
        pass

    collection = chroma_client.create_collection(collection_name)
    chunks     = pdf_to_chunks(pdf_bytes)

    if not chunks:
        return 0

    texts     = [c["text"] for c in chunks]
    pages     = [c["page"] for c in chunks]
    vectors   = embed_model.embed_documents(texts)
    ids       = [f"chunk_{i}" for i in range(len(texts))]
    metadatas = [{"page": p} for p in pages]

    collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)
    return len(chunks)
