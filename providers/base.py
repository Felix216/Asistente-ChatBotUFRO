from abc import ABC, abstractmethod
from typing import List, Dict, Any

class Provider(ABC):
    """Interfaz base para un proveedor LLM."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """Recibe mensajes [{role, content}, ...] y devuelve la respuesta del modelo."""
        ...
