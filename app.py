from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
from rag.pipeline import run_pipeline
from providers.deepseek import DeepSeekProvider
from providers.chatgpt import ChatGPTProvider

app = Flask(__name__)

txt_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

PROVIDERS = {
    "deepseek": DeepSeekProvider(model="deepseek-chat"),
    "chatgpt": ChatGPTProvider(model="openai/gpt-4.1-mini")
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/ask", methods=["POST"])
def api_ask():
    data = request.get_json()
    query = data.get("query")
    provider_name = data.get("provider", "deepseek")
    provider = PROVIDERS[provider_name]
    response = run_pipeline(query, provider, txt_model, k=5)
    return jsonify({"question": query, "answer": response, "provider": provider_name})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)