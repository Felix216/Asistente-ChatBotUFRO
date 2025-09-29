import os
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNKS_PARQUET = "data/processed/chunks.parquet"
INDEX_FILE = "data/index.faiss"

def load_chunks(parquet_file=CHUNKS_PARQUET):
    return pd.read_parquet(parquet_file)

def embed_chunks(df, model_name=MODEL):
    model = SentenceTransformer(model_name)
    texts = df["text"].tolist()
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    return embeddings

def build_faiss_index(embeddings, use_ip=True):
    dim = embeddings.shape[1]
    if use_ip:
        index = faiss.IndexFlatIP(dim)
    else:
        index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

if __name__ == "__main__":
    os.makedirs(os.path.dirname(INDEX_FILE), exist_ok=True)

    print("Cargando chunks...")
    df_chunks = load_chunks(CHUNKS_PARQUET)

    print("Generando embeddings...")
    embeddings = embed_chunks(df_chunks)

    print("Construyendo índice FAISS...")
    index = build_faiss_index(embeddings, use_ip=True)

    faiss.write_index(index, INDEX_FILE)
    print(f"Índice FAISS guardado en {INDEX_FILE}")

    df_chunks.to_parquet(CHUNKS_PARQUET, index=False)
    print(f"Chunks guardados en {CHUNKS_PARQUET}")
    print("Embeddings y FAISS listos para RAG")
