import json
import numpy as np
from sentence_transformers import SentenceTransformer
from providers.deepseek import DeepSeekProvider
from rag.pipeline import run_pipeline, retrieve

txt_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

providers = {
    "deepseek": DeepSeekProvider(model="deepseek-chat")
}

gold_file = "eval/gold_set.jsonl"
gold_set = [json.loads(l) for l in open(gold_file, encoding="utf-8")]

results = {prov: [] for prov in providers}

for prov_name, prov in providers.items():
    print(f"Evaluando {prov_name}...")
    for item in gold_set:
        question = item["question"]
        gold_answer = item["answer"]
        gold_refs = item.get("refs", [])

        import time
        start = time.time()
        try:
            pred_answer = run_pipeline(question, prov, txt_model, k=5)
        except Exception as e:
            pred_answer = ""
            print(f"Error al generar respuesta: {e}")
        latency = time.time() - start

        em = int(pred_answer.strip() == gold_answer.strip())
        pred_emb = txt_model.encode([pred_answer], convert_to_numpy=True, normalize_embeddings=True)[0]
        gold_emb = txt_model.encode([gold_answer], convert_to_numpy=True, normalize_embeddings=True)[0]
        cos_sim = float(np.dot(pred_emb, gold_emb))
        citations_found = sum(1 for r in gold_refs if r in pred_answer)
        citations_pct = citations_found / len(gold_refs) * 100 if gold_refs else 0

        chunks = retrieve(question, k=5, model=txt_model)
        is_od = int(len(chunks) == 0 or all(c["score"] < 0.1 for c in chunks))
        hallucination = int("Nota" in pred_answer)

        tokens = len(pred_answer.split())
        estimated_cost = tokens * 0.000002

        results[prov_name].append({
            "question": question,
            "pred": pred_answer,
            "gold": gold_answer,
            "em": em,
            "cos_sim": cos_sim,
            "citations_pct": citations_pct,
            "latency_s": latency,
            "tokens": tokens,
            "estimated_cost_usd": estimated_cost,
            "OD": is_od,
            "hallucination": hallucination
        })

import pandas as pd
for prov_name in providers:
    df = pd.DataFrame(results[prov_name])
    df.to_csv(f"eval/results_{prov_name}.csv", index=False)
    print(f"Resultados guardados en eval/results_{prov_name}.csv")
