import os
import logging
from typing import List, Dict
from openai import OpenAI
from models.base import EmbeddingModel, LLM

logger = logging.getLogger(__name__)
OPENAI_API_KEY_EMBEDDINGS = os.getenv("OPENAI_API_KEY_EMBEDDINGS")
OPENAI_API_KEY_MODEL = os.getenv("OPENAI_API_KEY_MODEL")

if not OPENAI_API_KEY_EMBEDDINGS:
    logger.warning("OPENAI_API_KEY is not set. OpenAI models will fail without it.")

if not OPENAI_API_KEY_MODEL:
    logger.warning("OPENAI_API_KEY_MODEL is not set. OpenAI LLMs will fail without it.")

# Initialize OpenAI SDK client (v1+)
openai_client_embedding = OpenAI(api_key=OPENAI_API_KEY_EMBEDDINGS)
openai_client_model = OpenAI(api_key=OPENAI_API_KEY_MODEL)

# --- Embedding Model Implementation ---
class OpenAIEmbeddingModel(EmbeddingModel):
    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.model_name = model_name
        logger.info(f"Initialized OpenAIEmbeddingModel with model: {model_name}")

    def embed_text(self, text: str) -> List[float]:
        """
        Converts a string into an embedding using OpenAI's API.
        """
        try:
            response = openai_client_embedding.embeddings.create(
                model=self.model_name,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Embedding failed for model '{self.model_name}': {e}")
            return []

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Batched embedding of multiple texts (OpenAI supports batching).
        """
        try:
            response = openai_client_embedding.embeddings.create(
                model=self.model_name,
                input=texts
            )
            return [entry.embedding for entry in response.data]
        except Exception as e:
            logger.error(f"Batched embedding failed for model '{self.model_name}': {e}")
            return [[] for _ in texts]


# --- LLM Implementation ---
class OpenAILLM(LLM):
    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name
        logger.info(f"Initialized OpenAILLM with model: {model_name}")

    def generate_text(self, messages: List[Dict[str, str]]) -> str:
        """
        Generates text using OpenAI's chat model interface.
        Expects list of {"role": ..., "content": ...} messages.
        """
        try:
            response = openai_client_model.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=1024
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Text generation failed for model '{self.model_name}': {e}")
            return ""