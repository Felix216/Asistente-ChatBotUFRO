import time
import csv
from typing import List, Dict, Any
from providers.base import Provider
from rag.pipeline import run_pipeline
from sentence_transformers import SentenceTransformer
import tiktoken

TOKEN_RATE = 0.002

def count_tokens(text: str, encoding_name="cl100k_base") -> int:
    """Cuenta tokens usando tiktoken."""
    import tiktoken
    tok = tiktoken.get_encoding(encoding_name)
    return len(tok.encode(text))

def evaluate_queries(queries: List[str], provider: Provider, model: SentenceTransformer, k: int = 5, csv_file: str = "eval/latency.csv"):
    """Evalúa queries, mide latencia y costo, y guarda resultados en CSV."""
    
    results = []
    
    for query in queries:
        start_total = time.time()
        
        start_retrieve = time.time()
        response = run_pipeline(query, provider, model, k=k)
        end_total = time.time()
        
        total_latency = end_total - start_total
        
        input_tokens = count_tokens(query)
        output_tokens = count_tokens(response)
        total_tokens = input_tokens + output_tokens
        estimated_cost = (total_tokens / 1000) * TOKEN_RATE
        
        results.append({
            "query": query,
            "response": response,
            "latency_s": total_latency,
            "tokens": total_tokens,
            "estimated_cost_usd": estimated_cost
        })
    
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Métricas guardadas en {csv_file}")