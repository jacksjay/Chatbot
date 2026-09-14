"""
Central configuration for the chatbot application.
Tune behaviour here without touching business logic elsewhere.
"""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load variables from a local .env file 
load_dotenv()


@dataclass(frozen=True)
class Config:
    # --- Ollama ---
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_CHAT_MODEL: str = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:1b")   # light model
    OLLAMA_EMBED_MODEL: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    OLLAMA_TIMEOUT_SECONDS: float = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "60")) #don't hang forever if the local model stalls
    
    # --- Guardrails (input/output size checks) ---
    MAX_INPUT_CHARS: int = 4000          # reject user input longer than this
    MIN_INPUT_CHARS: int = 1
    MAX_OUTPUT_CHARS: int = 8000         # truncate model output longer than this
    MAX_OUTPUT_TOKENS: int = 512         # cap sent to Ollama as num_predict

    # --- Memory (bounded) ---
    MEMORY_MAX_MESSAGES: int = 20        # max turns kept in short-term memory
    MEMORY_MAX_TOKENS: int = 3000        # approx token budget for short-term memory
    MEMORY_SUMMARY_MAX_CHARS: int = 2000 # hard cap on the rolling summary of aged-out turns
    RAG_TOP_K: int = 3                   # similar past messages to retrieve from Chroma
    RAG_MAX_CHARS_PER_CHUNK: int = 500   # hard cap per retrieved chunk so can't blow the token budget

    # --- Storage ---
    SQLITE_DB_PATH: str = os.getenv("SQLITE_DB_PATH", "data/app.db")
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "data/chroma")

    # --- Auth ---
    PBKDF2_ITERATIONS: int = 260_000

    # --- Misc ---
    APP_TITLE: str = "Local Ollama Chatbot"
    SYSTEM_PROMPT: str = (
        "You are a helpful, concise AI assistant running on a local model. "
        "Use the conversation history and any retrieved context to answer accurately."
    )


CFG = Config()
