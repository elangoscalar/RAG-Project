# rag_pipeline.py — Retrieval + Answer Generation
# Searches ChromaDB and calls the LLM to produce an answer

from config import TOP_K, MEMORY_TURNS
from resources import get_embedding_model, get_llm, get_chroma_client


def build_history_text(chat_history):
    recent = chat_history[-(MEMORY_TURNS * 2):]
    lines  = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def search_and_answer(question, collection_name, chat_history):
    embed_model   = get_embedding_model()
    llm           = get_llm()
    chroma_client = get_chroma_client()
    collection    = chroma_client.get_collection(collection_name)

    q_vector = embed_model.embed_query(question)
    results  = collection.query(
        query_embeddings=[q_vector],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    docs      = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    context       = "\n\n---\n\n".join(docs)
    history_text  = build_history_text(chat_history)
    history_block = f"\nConversation so far:\n{history_text}\n" if history_text else ""

    prompt = f"""You are a helpful assistant. Answer the question using ONLY the document context provided below.
You may use the conversation history to understand follow-up questions.
If the answer is not found in the context, say: "I could not find this in the document."
Never make up information that is not in the context.

Document Context:
{context}
{history_block}
Current Question: {question}

Answer:"""

    response = llm.invoke(prompt).content
    sources  = [
        {"page": m["page"], "text": d[:200], "score": round(1 - dist, 3)}
        for d, m, dist in zip(docs, metadatas, distances)
    ]
    return response, sources
