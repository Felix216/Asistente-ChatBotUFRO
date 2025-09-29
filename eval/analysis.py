import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

deepseek_csv = "eval/latency.csv"

df = pd.read_csv(deepseek_csv)
df["provider"] = "DeepSeek"

summary = df.agg({
    "latency_s": "mean",
    "tokens": "mean",
    "estimated_cost_usd": "mean"
}).to_frame().T
summary["provider"] = "DeepSeek"
summary = summary[["provider", "latency_s", "tokens", "estimated_cost_usd"]]

print("\nResumen de métricas promedio para DeepSeek:")
print(summary)

summary.to_csv("eval/summary_metrics_deepseek.csv", index=False)
print("Resumen guardado en eval/summary_metrics_deepseek.csv")

sns.set(style="whitegrid")

plt.figure(figsize=(6,4))
sns.barplot(data=df, x="provider", y="latency_s", palette="pastel")
plt.title("Latencia promedio DeepSeek")
plt.ylabel("Segundos")
plt.xlabel("")
plt.tight_layout()
plt.savefig("eval/latency_deepseek.png")
plt.show()

fig, ax = plt.subplots(1,2, figsize=(12,4))

sns.barplot(data=df, x="provider", y="tokens", palette="muted", ax=ax[0])
ax[0].set_title("Tokens promedio por consulta")
ax[0].set_ylabel("Cantidad de tokens")

sns.barplot(data=df, x="provider", y="estimated_cost_usd", palette="muted", ax=ax[1])
ax[1].set_title("Costo estimado por consulta (USD)")
ax[1].set_ylabel("USD")

plt.tight_layout()
plt.savefig("eval/cost_tokens_deepseek.png")
plt.show()

print("\nObservaciones:")
print("- Latencia promedio, tokens y costo calculados únicamente para DeepSeek.")