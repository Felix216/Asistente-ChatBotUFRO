import os
import requests
from typing import List, Dict, Any
from .base import Provider
from dotenv import load_dotenv

load_dotenv()

class ChatGPTProvider(Provider):
    def __init__(self, model: str = "openai/gpt-4.1-mini"):
        self._model = model
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("Falta OPENROUTER_API_KEY en entorno")
        self.base_url = "https://openrouter.ai/api/v1"

    @property
    def name(self) -> str:
        return "chatgpt"

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self._model,
            "messages": messages,
            **kwargs
        }
        resp = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
        if resp.status_code != 200:
            raise Exception(f"Error {resp.status_code}: {resp.text}")
        data = resp.json()
        return data["choices"][0]["message"]["content"]
