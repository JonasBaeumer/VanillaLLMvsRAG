"""
Abstract base classes for embedding models and LLMs.

This module defines the interfaces for embedding models and large language models (LLMs),
which must be implemented by concrete subclasses in the system.
"""

from abc import ABC, abstractmethod
from typing import List, Dict

class EmbeddingModel(ABC):
    """
    Abstract base class for all embedding models.
    """

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """
        Converts a single string into a dense vector representation.
        """
        pass

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Optional helper to embed a list of strings. Override if batching is supported.
        """
        return [self.embed_text(t) for t in texts]


class LLM(ABC):
    """
    Abstract base class for all language generation models.
    """

    @abstractmethod
    def generate_text(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate a completion given a list of chat messages (role + content).
        """
        pass