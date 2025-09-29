#!/bin/bash
# Crear embeddings y FAISS / Parquet desde los PDFs/HTML

python rag/ingest.py --data_dir data/raw --output_file data/processed/chunks.parquet
python rag/embed.py --chunks_file data/processed/chunks.parquet --index_file data/index.faiss
