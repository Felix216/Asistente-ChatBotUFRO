import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from rag.pipeline import run_pipeline, retrieve
from providers.chatgpt import ChatGPTProvider
from providers.deepseek import DeepSeekProvider

GOLD_FILE = "eval/gold_set.jsonl"
K = 5
TOKEN_RATE = 0.002

txt_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
providers = {
    "ChatGPT": ChatGPTProvider(model="openai/gpt-4.1-mini"),
    "DeepSeek": DeepSeekProvider(model="deepseek-chat")
}

def count_tokens(text: str, encoding_name="cl100k_base") -> int:
    import tiktoken
    tok = tiktoken.get_encoding(encoding_name)
    return len(tok.encode(text))

gold_set = [json.loads(l) for l in open(GOLD_FILE, encoding="utf-8")]

all_results = []

for prov_name, prov in providers.items():
    print(f"\nEvaluando {prov_name}...")
    results = []

    for item in gold_set:
        question = item["question"]
        gold_answer = item["answer"]
        gold_refs = item.get("refs", [])

        start = time.time()
        try:
            pred_answer = run_pipeline(question, prov, txt_model, k=K)
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

        chunks = retrieve(question, k=K, model=txt_model)
        is_od = int(len(chunks) == 0 or all(c["score"] < 0.1 for c in chunks))
        hallucination = int("Nota" in pred_answer)

        tokens = count_tokens(pred_answer) + count_tokens(question)
        estimated_cost = (tokens / 1000) * TOKEN_RATE

        results.append({
            "provider": prov_name,
            "question": question,
            "pred": pred_answer,
            "gold": gold_answer,
            "EM": em,
            "cos_sim": cos_sim,
            "citations_pct": citations_pct,
            "OD": is_od,
            "hallucination": hallucination,
            "latency_s": latency,
            "tokens": tokens,
            "estimated_cost_usd": estimated_cost
        })

    df = pd.DataFrame(results)
    csv_file = f"eval/latency_{prov_name.lower()}.csv"
    df.to_csv(csv_file, index=False)
    print(f"Resultados guardados en {csv_file}")
    all_results.append(df)

df_all = pd.concat(all_results, ignore_index=True)
summary = df_all.groupby("provider").agg({
    "latency_s": "mean",
    "tokens": "mean",
    "estimated_cost_usd": "mean",
    "EM": "mean",
    "cos_sim": "mean",
    "citations_pct": "mean"
}).reset_index()

summary_file = "eval/summary_metrics_comparison.csv"
summary.to_csv(summary_file, index=False)
print(f"\nResumen guardado en {summary_file}")
print(summary)

sns.set_theme(style="whitegrid")

plt.figure(figsize=(6,4))
sns.barplot(data=df_all, x="provider", y="latency_s", hue="provider",
            dodge=False, palette="pastel", legend=False)
plt.title("Latencia promedio por proveedor")
plt.ylabel("Segundos")
plt.xlabel("")
plt.tight_layout()
plt.savefig("eval/latency_comparison.png")
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(12,4))

sns.barplot(data=df_all, x="provider", y="tokens", hue="provider",
            dodge=False, palette="muted", legend=False, ax=ax[0])
ax[0].set_title("Tokens promedio por consulta")
ax[0].set_ylabel("Cantidad de tokens")
ax[0].set_xlabel("")

sns.barplot(data=df_all, x="provider", y="estimated_cost_usd", hue="provider",
            dodge=False, palette="muted", legend=False, ax=ax[1])
ax[1].set_title("Costo estimado por consulta (USD)")
ax[1].set_ylabel("USD")
ax[1].set_xlabel("")

plt.tight_layout()
plt.savefig("eval/cost_tokens_comparison.png")
plt.show()

print("\nObservaciones:")
print("- Comparación directa entre ChatGPT y DeepSeek.")
print("- Latencia, tokens, EM, similitud coseno, citas y costo son promedios por consulta.")