import os
from openai import OpenAI
from typing import List, Dict, Any
from .base import Provider
from dotenv import load_dotenv
load_dotenv()

class DeepSeekProvider(Provider):
    def __init__(self, model: str = "deepseek-chat"):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("Falta DEEPSEEK_API_KEY en entorno")
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self._model = model

    @property
    def name(self) -> str:
        return "deepseek"

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        resp = self.client.chat.completions.create(
            model=self._model,
            messages=messages,
            **kwargs
        )
        return resp.choices[0].message.content
