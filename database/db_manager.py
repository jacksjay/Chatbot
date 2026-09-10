"""
Owns three tables:
  - users         : auth credentials (hashed passwords)
  - llm_logs      : one row per LLM call (tokens, latency, status) for observability
  - chat_messages : persisted chat history per user/session
"""
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import CFG


class DatabaseManager:
    """Thin wrapper around a local SQLite database with a module-level lock
    so it's safe to share across Streamlit's per-session reruns/threads."""

    _lock = threading.Lock()

    def __init__(self, db_path: str = CFG.SQLITE_DB_PATH):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._init_schema()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            with self._lock:
                yield conn
                conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS llm_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    model TEXT NOT NULL,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    total_tokens INTEGER,
                    latency_ms REAL,
                    input_chars INTEGER,
                    output_chars INTEGER,
                    status TEXT,
                    error TEXT,
                    timestamp TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                );
                """
            )

    # ---------------- Users ----------------
    def create_user(self, username: str, password_hash: str, salt: str) -> bool:
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO users (username, password_hash, salt, created_at) VALUES (?, ?, ?, ?)",
                    (username, password_hash, salt, datetime.utcnow().isoformat()),
                )
            return True
        except sqlite3.IntegrityError:
            return False

    def get_user(self, username: str) -> Optional[sqlite3.Row]:
        with self._connect() as conn:
            cur = conn.execute("SELECT * FROM users WHERE username = ?", (username,))
            return cur.fetchone()

    # ---------------- LLM logs ----------------
    def log_llm_call(self, **fields) -> None:
        columns = ", ".join(fields.keys())
        placeholders = ", ".join("?" * len(fields))
        with self._connect() as conn:
            conn.execute(
                f"INSERT INTO llm_logs ({columns}, timestamp) VALUES ({placeholders}, ?)",
                (*fields.values(), datetime.utcnow().isoformat()),
            )

    def get_recent_logs(self, username: str, limit: int = 50):
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT * FROM llm_logs WHERE username = ? ORDER BY id DESC LIMIT ?",
                (username, limit),
            )
            return cur.fetchall()

    def get_usage_summary(self, username: str) -> sqlite3.Row:
        with self._connect() as conn:
            cur = conn.execute(
                """
                SELECT COUNT(*) AS calls,
                       COALESCE(SUM(total_tokens), 0) AS total_tokens,
                       COALESCE(AVG(latency_ms), 0) AS avg_latency_ms
                FROM llm_logs WHERE username = ? AND status = 'success'
                """,
                (username,),
            )
            return cur.fetchone()

    # ---------------- Chat messages ----------------
    def save_message(self, username: str, session_id: str, role: str, content: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO chat_messages (username, session_id, role, content, timestamp) "
                "VALUES (?, ?, ?, ?, ?)",
                (username, session_id, role, content, datetime.utcnow().isoformat()),
            )

    def load_messages(self, username: str, session_id: str):
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT role, content, timestamp FROM chat_messages "
                "WHERE username = ? AND session_id = ? ORDER BY id ASC",
                (username, session_id),
            )
            return cur.fetchall()
