import logging
import time
from typing import List

import chromadb
import ollama
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

from config import CFG

_logger = logging.getLogger("vector_store")

class OllamaEmbeddingFunction(EmbeddingFunction):
    """Adapts a local Ollama embedding model to Chroma's EmbeddingFunction interface."""

    def __init__(self, model: str = CFG.OLLAMA_EMBED_MODEL, base_url: str = CFG.OLLAMA_BASE_URL):
        self._model = model
        self._client = ollama.Client(host=base_url, timeout=CFG.OLLAMA_TIMEOUT_SECONDS)

    def __call__(self, input: Documents) -> Embeddings:
        """Takes a list of texts, queries Ollama, and returns a list of numerical vectors."""
        vectors: Embeddings = []
        for text in input:
            resp = self._client.embeddings(model=self._model, prompt=text)
            vectors.append(resp["embedding"])
        return vectors


class VectorMemoryStore:
    def __init__(self, persist_dir: str = CFG.CHROMA_PERSIST_DIR):
        # PersistentClient saves the vectors to disk so they survive server reboots
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._embedder = OllamaEmbeddingFunction()
    def _collection_name(user_id: int) -> str:
        """
        DATA ISOLATION: Generates a perfectly safe, standardized collection name.
        """
        # Single source of truth for the naming scheme, used by both the
        # collection getter and reset() so they can never drift apart again.
        return f"user_{user_id}"
    # def _collection(self, username: str):
    #     name = f"user_{username}".lower().replace(" ", "_")
    #     return self._client.get_or_create_collection(name=name, embedding_function=self._embedder)
    def _collection(self,user_id: int):
        # Automatically creates the collection if it's their first time chatting
        return self._client.get_or_create_collection(
            name=self._collection_name(user_id), embedding_function=self._embedder
        )
        #name = f"user_{user_id}"
        #return self._client.get_or_create_collection(name=name, embedding_function=self._embedder)


    def add_turn(self, user_id: int, role: str, content: str) -> None:
        if not content.strip():
            return
        try:
            #col = self._collection(username)
            col = self._collection(user_id)
            # Use a timestamp to generate a perfectly unique document ID
            doc_id = f"{role}-{time.time_ns()}"
            col.add(documents=[content], metadatas=[{"role": role}], ids=[doc_id])
        except Exception as exc:
            #If Chroma fails to save, log the error but do not crash the chat!
            _logger.warning("vector_store.add_turn failed for user_id=%s: %s", user_id, exc)


        

    def retrieve_relevant(self, user_id: int, query: str, top_k: int = CFG.RAG_TOP_K, max_chars_per_chunk: int = CFG.RAG_MAX_CHARS_PER_CHUNK) -> List[str]:
        try:
            col = self._collection(user_id)
            count = col.count()
            if count == 0:
                return []
            # Perform a semantic similarity search
            results = col.query(query_texts=[query], n_results=min(top_k, count))
            docs = results.get("documents", [[]])[0]
        except Exception as exc:
            #If the database locks or crashes, quietly return an empty list 
            # so the LLM can just answer the question normally without RAG context.
            _logger.warning("vector_store.retrieve_relevant failed for user_id=%s: %s", user_id, exc)
            #return results.get("documents", [[]])[0]
            return[]
        #SAFETY LIMIT: Truncate each retrieved document to max_chars_per_chunk.
        # This prevents a massive historical paragraph from accidentally blowing out the LLM's context window.
        return [d[:max_chars_per_chunk] for d in docs]
    
    def reset(self, user_id: int) -> None:
        """Deletes the entire Chroma collection for a user."""
        name = self._collection_name(user_id)
        #name = f"user_{username}".lower().replace(" ", "_")
        try:
            self._client.delete_collection(name)
            # self._client.delete_collection(name)
        except Exception as exc:
            _logger.info("vector_store.reset: no collection to delete for user_id=%s (%s)", user_id, exc)
            #pass
