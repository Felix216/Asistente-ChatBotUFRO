import argparse
import json
from sentence_transformers import SentenceTransformer
from rag.pipeline import run_pipeline
from providers.deepseek import DeepSeekProvider
from providers.chatgpt import ChatGPTProvider

txt_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

PROVIDERS = {
    "deepseek": DeepSeekProvider(model="deepseek-chat"),
    "chatgpt": ChatGPTProvider(model="openai/gpt-4.1-mini")
}

def main():
    parser = argparse.ArgumentParser(description="Asistente de reglamento universitario UFRO")
    parser.add_argument("--provider", type=str, choices=PROVIDERS.keys(), default="deepseek")
    parser.add_argument("--query", type=str, help="Pregunta a realizar")
    parser.add_argument("--batch", action="store_true", help="Usar gold set desde eval/gold_set.jsonl")
    parser.add_argument("--k", type=int, default=5, help="Número de chunks a recuperar")
    args = parser.parse_args()

    provider = PROVIDERS[args.provider]

    if args.batch:
        with open("eval/gold_set.jsonl", encoding="utf-8") as f:
            gold_set = [json.loads(l) for l in f]
        for item in gold_set:
            q = item["question"]
            print(f"\nPregunta: {q}")
            resp = run_pipeline(q, provider, txt_model, k=args.k)
            print(f"Respuesta: {resp}")
            print("-"*60)
    else:
        if not args.query:
            print("Debes ingresar una pregunta con --query")
            return
        resp = run_pipeline(args.query, provider, txt_model, k=args.k)
        print(f"\nPregunta: {args.query}")
        print(f"Respuesta: {resp}")

if __name__ == "__main__":
    main()