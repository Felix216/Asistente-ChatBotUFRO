import os
from openai import OpenAI
from typing import List, Dict, Any
from .base import Provider
from dotenv import load_dotenv
load_dotenv()

class ChatGPTProvider(Provider):
    def __init__(self, model: str = "openai/gpt-4.1-mini"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Falta OPENAI_API_KEY en entorno")
        self.client = OpenAI(api_key=api_key)
        self._model = model

    @property
    def name(self) -> str:
        return "chatgpt"

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        resp = self.client.chat.completions.create(
            model=self._model,
            messages=messages,
            **kwargs
        )
        return resp.choices[0].message.content
