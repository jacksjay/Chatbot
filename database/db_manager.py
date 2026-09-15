
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator, Optional

# SQLAlchemy imports - 
from sqlalchemy import (
    DateTime, Float, ForeignKey, Index, Integer, String, Text,
    create_engine, event, func, select,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, Session, sessionmaker

from config import CFG


# ORM MODELS (Database Schema Definition)

class Base(DeclarativeBase):
    """The base class that all our database models will inherit from."""
    pass

class User(Base):
    __tablename__ = "users"

    # Mapped[int] provides strict type hinting. primary_key=True auto-increments the ID.
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # index=True makes searching by username much faster.
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    salt: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    # --- Brute-force protection fields ---
    failed_login_attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    locked_until: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Relationships: If a user is deleted, `cascade="all, delete-orphan"` ensures 
    # all their chat messages and logs are automatically deleted too.
    llm_logs: Mapped[list["LLMLog"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    messages: Mapped[list["ChatMessage"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class LLMLog(Base):
    __tablename__ = "llm_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # FOREIGN KEY: Instead of storing the string 'username', we link to the User table's ID. 
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    input_chars: Mapped[int] = mapped_column(Integer, default=0)
    output_chars: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    error: Mapped[str] = mapped_column(Text, default="")
    # index=True speeds up queries when we ask for "Recent logs"
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    user: Mapped["User"] = relationship(back_populates="llm_logs")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    user: Mapped["User"] = relationship(back_populates="messages")

    # Composite Index: Since we frequently search by BOTH user_id and session_id simultaneously,
    # this combined index makes that specific search lightning fast.
    __table_args__ = (Index("ix_chat_messages_user_session", "user_id", "session_id"),)


# ENGINE & CONNECTION POOL SETUP

def _build_engine(db_path: str):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    # create_engine: It manages the connection pool.
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False, "timeout": 30},
        pool_size=10,       # Keep 10 connections open in memory ready to be used
        max_overflow=5,     # Allow up to 5 extra connections during traffic spikes
        pool_pre_ping=True, # Test connections before using them to prevent random crashes
        future=True,
    )

    # EVENT LISTENER: This runs every single time a new connection is created.
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_conn, _connection_record):
        cursor = dbapi_conn.cursor()
        # WAL Mode (Write-Ahead Logging): for SQLite. 
        # Allows reading and writing at the same time without database locks.
        cursor.execute("PRAGMA journal_mode=WAL")       
        cursor.execute("PRAGMA synchronous=NORMAL")      # Speeds up disk writes safely
        cursor.execute("PRAGMA busy_timeout=30000")      # Wait 30s instead of crashing if DB is busy
        cursor.execute("PRAGMA foreign_keys=ON")         # Enforces our User -> Logs relationship
        cursor.close()

    return engine


class DatabaseManager:
    def __init__(self, db_path: str = CFG.SQLITE_DB_PATH):
        self._engine = _build_engine(db_path)
        # sessionmaker creates the "factory" for generating new database sessions
        self._SessionLocal = sessionmaker(bind=self._engine, expire_on_commit=False, future=True)
        # Automatically creates the tables if they don't exist
        Base.metadata.create_all(self._engine)
        
        # IN-MEMORY CACHE: Instead of asking the DB for a User ID every time they send a message,
        # we store it in RAM (a Python dictionary). 
        self._user_id_cache: dict[str, int] = {}
        # lock here because modifying Python dicts across multiple Streamlit threads can cause crashes.
        self._cache_lock = threading.Lock()

    def _resolve_user_id(self, session: Session, username: str) -> Optional[int]:
        """Converts a username string into an integer user_id quickly."""
        # 1. Check RAM (Cache) first (fast)
        with self._cache_lock:
            cached = self._user_id_cache.get(username)
        if cached is not None:
            return cached
            
        # 2. If not in cache, query the database
        user_id = session.scalar(select(User.id).where(User.username == username))
        if user_id is not None:
            # 3. Save it to the cache for next time
            with self._cache_lock:
                self._user_id_cache[username] = user_id
        return user_id

    @contextmanager
    def _session(self) -> Iterator[Session]:
        """Borrows a connection from the pool, does the work, and returns it."""
        session = self._SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            # If ANY error happens (like a duplicate username), cancel all changes made in this session
            session.rollback()
            raise
        finally:
            # Return the connection to the SQLAlchemy pool
            session.close() 

    # ---------------- Users ----------------
    def create_user(self, username: str, password_hash: str, salt: str) -> bool:
        with self._session() as session:
            user = User(username=username, password_hash=password_hash, salt=salt)
            session.add(user)
            try:
                # flush() pushes the data to the DB to check for errors without permanently committing it yet
                session.flush()
            except IntegrityError:
                # Triggers if the username already exists
                session.rollback()
                return False
            # Warm up the cache instantly so they can start chatting without delay
            with self._cache_lock:
                self._user_id_cache[username] = user.id  
        return True

    def get_user(self, username: str) -> Optional[User]:
        with self._session() as session:
            # session.scalar() runs the query and returns just the single Object result (or None)
            return session.scalar(select(User).where(User.username == username))

    # ---------------- Login throttling (brute-force protection) ----------------
    MAX_FAILED_ATTEMPTS = 5
    LOCKOUT_MINUTES = 15

    def is_locked_out(self, username: str) -> tuple[bool, Optional[int]]:
        """Checks if a user is currently locked out."""
        with self._session() as session:
            user = session.scalar(select(User).where(User.username == username))
            if not user or not user.locked_until:
                return False, None
                
            # If the lockout timer hasn't expired yet
            if user.locked_until > datetime.utcnow():
                remaining = int((user.locked_until - datetime.utcnow()).total_seconds())
                return True, remaining
            return False, None

    def record_failed_login(self, username: str) -> None:
        with self._session() as session:
            user = session.scalar(select(User).where(User.username == username))
            if not user:
                return  # Prevent hackers from guessing usernames by looking at error response times
            user.failed_login_attempts += 1
            # Lock the account if they cross the threshold
            if user.failed_login_attempts >= self.MAX_FAILED_ATTEMPTS:
                user.locked_until = datetime.utcnow() + timedelta(minutes=self.LOCKOUT_MINUTES)

    def record_successful_login(self, username: str) -> None:
        """Resets the failure counters on a successful login."""
        with self._session() as session:
            user = session.scalar(select(User).where(User.username == username))
            if user:
                user.failed_login_attempts = 0
                user.locked_until = None

    # ---------------- LLM logs ----------------
    def log_llm_call(self, *, username: str, model: str, prompt_tokens: int = 0, completion_tokens: int = 0, total_tokens: int = 0, latency_ms: float = 0.0, input_chars: int = 0, output_chars: int = 0, status: str = "success", error: str = "") -> None:
        with self._session() as session:
            user_id = self._resolve_user_id(session, username)
            if user_id is None: return
            
            # Create a new ORM Object and add it to the session
            session.add(
                LLMLog(user_id=user_id, model=model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total_tokens, latency_ms=latency_ms, input_chars=input_chars, output_chars=output_chars, status=status, error=error)
            )

    def get_recent_logs(self, username: str, limit: int = 50) -> list[LLMLog]:
        with self._session() as session:
            user_id = self._resolve_user_id(session, username)
            if user_id is None: return []
            
            stmt = select(LLMLog).where(LLMLog.user_id == user_id).order_by(LLMLog.id.desc()).limit(limit)
            return list(session.scalars(stmt))

    def get_usage_summary(self, username: str) -> dict:
        """Lets the database engine calculate the totals (much faster than Python doing it)."""
        with self._session() as session:
            user_id = self._resolve_user_id(session, username)
            if user_id is None:
                return {"calls": 0, "total_tokens": 0, "avg_latency_ms": 0.0}
                
            # func.coalesce ensures that if there are no logs, it returns 0 instead of 'Null'
            stmt = select(
                func.count(LLMLog.id),
                func.coalesce(func.sum(LLMLog.total_tokens), 0),
                func.coalesce(func.avg(LLMLog.latency_ms), 0.0),
            ).where(LLMLog.user_id == user_id, LLMLog.status == "success")
            
            # Execute the query and unpack the 3 values
            calls, total_tokens, avg_latency_ms = session.execute(stmt).one()
            return {"calls": calls, "total_tokens": total_tokens, "avg_latency_ms": avg_latency_ms}

    # ---------------- Chat messages ----------------
    def save_message(self, username: str, session_id: str, role: str, content: str) -> None:
        with self._session() as session:
            user_id = self._resolve_user_id(session, username)
            if user_id is None: return
            session.add(ChatMessage(user_id=user_id, session_id=session_id, role=role, content=content))

    def load_messages(self, username: str, session_id: str) -> list[ChatMessage]:
        with self._session() as session:
            user_id = self._resolve_user_id(session, username)
            if user_id is None: return []
            
            stmt = select(ChatMessage).where(ChatMessage.user_id == user_id, ChatMessage.session_id == session_id).order_by(ChatMessage.id.asc())
            return list(session.scalars(stmt))