# test_openai_models.py

import logging
from models.openai_models import OpenAIEmbeddingModel, OpenAILLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    
    embedder = OpenAIEmbeddingModel()
    llm = OpenAILLM()

    # Test embedding a short sentence
    test_text = "Test embedding."
    print(f"\nEmbedding text: '{test_text}'")
    embedding = embedder.embed_text(test_text)
    print(f"Embedding length: {len(embedding)}")

    # Test generating text with a simple prompt (low token cost)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is RAG in one sentence?"}
    ]

    print("\nGenerating text...")
    response = llm.generate_text(messages)
    print("Generated response:", response)

if __name__ == "__main__":
    main()