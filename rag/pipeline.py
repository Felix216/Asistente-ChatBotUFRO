import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from providers.base import Provider

def rewrite_query(query: str) -> str:
    return query.strip().replace("\n", " ")

def retrieve(query: str, k: int, model: SentenceTransformer,
             chunks_file="data/processed/chunks.parquet",
             index_file="data/index.faiss") -> List[Dict[str, Any]]:

    df = pd.read_parquet(chunks_file)
    texts = df["text"].tolist()
    index = faiss.read_index(index_file)

    qv = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    D, I = index.search(qv, k)

    retrieved_chunks = []
    for j, i in enumerate(I[0]):
        score = float(D[0][j])
        if score > 0:
            retrieved_chunks.append({
                "chunk": texts[i],
                "score": score,
                "path": df.iloc[i].get("path","")
            })

    return retrieved_chunks

def detect_hallucination(response: str, chunks: List[Dict[str, Any]]) -> bool:
    context = " ".join(c["chunk"] for c in chunks)
    return any(word for word in response.split() if word not in context)

def synthesize(user_query: str, retrieved_chunks: List[Dict[str, Any]], provider: Provider) -> str:
    context_text = "\n\n".join([f"{c['chunk']} (Fuente: {c['path']})" for c in retrieved_chunks])
    messages = [
        {"role": "system", "content": (
            "Eres un asistente experto en reglamentos universitarios. "
            "Responde de forma breve y clara usando solo la información provista en el contexto. "
            "Si no sabes la respuesta, indica claramente que no tienes suficiente información. "
            "Cita siempre las fuentes provistas y no inventes datos."
        )},
        {"role": "user", "content": f"Contexto:\n{context_text}\n\nPregunta: {user_query}"}
    ]
    return provider.chat(messages)

def postprocess(response: str) -> str:
    return response.strip().replace("\n\n", "\n")

def run_pipeline(query: str, provider: Provider, model: SentenceTransformer, k: int = 5) -> str:
    q_clean = rewrite_query(query)
    chunks = retrieve(q_clean, k=k, model=model)

    if len(chunks) == 0 or all(c["score"] < 0.1 for c in chunks):
        return "Lo siento, no tengo información suficiente para responder esa pregunta."

    response = synthesize(q_clean, chunks, provider)
    response_clean = postprocess(response)

    if detect_hallucination(response_clean, chunks):
        response_clean += "\n Nota: Algunas partes de la respuesta podrían no estar respaldadas por los documentos."

    return response_clean
