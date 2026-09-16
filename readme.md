# 🤖 Enterprise-Grade Local RAG Chatbot

An enterprise-ready, locally hosted Retrieval-Augmented Generation (RAG) chatbot built with **Streamlit, Ollama, ChromaDB, and SQLAlchemy**. 


---

## ✨ Key Features & Architecture

### 🛡️ Security & Authentication
* **Secure Auth:** Passwords are hashed using PBKDF2-HMAC-SHA256 with per-user salts. Constant-time comparisons are used to prevent timing attacks.
* **Brute-Force Protection:** Accounts are automatically locked for 15 minutes after 5 failed login attempts.
* **Prompt Injection Defense:** Uses regex heuristics to detect jailbreak attempts (e.g., *"Ignore all previous instructions"*). Malicious prompts trigger a "Hard Block"—the LLM is bypassed entirely, returning a canned refusal, and the input is prevented from entering long-term memory.

### 💾 Database (SQLAlchemy + SQLite WAL)
* **Concurrency:** SQLite is configured in **WAL (Write-Ahead Logging)** mode with a busy timeout, allowing simultaneous users to read and write without throwing `"Database is locked"` errors.
* **Connection Pooling:** Uses SQLAlchemy to maintain a pool of reusable connections, vastly improving performance over creating a new connection per query.
* **Data Normalization:** All telemetry and chat records use integer `user_id` Foreign Keys with composite indexes for lightning-fast lookups.

### 🧠 Dual-Memory System (Short & Long Term)
* **Short-Term Sliding Window:** Maintains a strict token and message-count budget. Oldest messages are dropped dynamically. Includes scaffolding for a **Rolling Summary** to compress forgotten context into the system prompt.
* **Long-Term RAG (ChromaDB):** Every user has an isolated vector collection (keyed by `user_id`). Retrieves the Top-K relevant historical messages and injects them into the current prompt.
* **RAG Budget Cap:** Retrieved chunks are strictly sliced to a maximum character limit to prevent historical data from accidentally blowing out the LLM's context window.

### 📊 Observability
* **Dual-Telemetry:** Every LLM call logs exact `latency_ms`, `prompt_tokens`, `completion_tokens`, and status. Logs are written to both a queryable SQLite table (visible in the UI) and a flat `app.log` file for server admins.
* **Fault Tolerance:** Vector database operations and telemetry writes are wrapped in `try/except` blocks. If the database locks up, the chat UI continues to function uninterrupted for the user.

---

## 📂 Project Structure

```text
chatbot_project/
├── app.py                      # Orchestrator: Streamlit UI and AI Pipeline
├── config.py                   # Centralized Configuration & Environment Variables
├── requirements.txt            # Python dependencies
├── README.md                   
├── auth/
│   └── authenticator.py        # PBKDF2 Hashing, Login, and Registration logic
├── core/
│   ├── guardrails.py           # Input/Output limits and Prompt Injection Regex
│   ├── llm_client.py           # Ollama API wrapper with timeout and token extraction
│   ├── logger.py               # Dual-logging implementation (SQLite + File)
│   ├── memory.py               # Token-budgeted sliding window & rolling summary
│   └── vector_store.py         # ChromaDB interface & Ollama embeddings adapter
└── database/
    └── db_manager.py           # SQLAlchemy ORM, Models, and Connection Pooling