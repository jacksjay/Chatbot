import hashlib # library for secure cryptographic hashing functions
import secrets # library for generating cryptographically strong random numbers (salts)

# Import global configurations
from config import CFG
from database.db_manager import DatabaseManager


class Authenticator:
    def __init__(self, db: DatabaseManager):
        self._db = db

    @staticmethod
    def _hash_password(password: str, salt: str) -> str:
        dk = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt),
            CFG.PBKDF2_ITERATIONS,
        )
        return dk.hex()

    def register(self, username: str, password: str) -> tuple[bool, str]:
        username = (username or "").strip() #handle None values safely and strip leading/trailing spaces
        # Check if either the username or the password is an empty string
        if not username or not password:
            return False, "Username and password cannot be empty."
        if len(password) < 6:
            return False, "Password must be at least 6 characters."
        #DB->Query to see if a user with this exact username already exists
        if self._db.get_user(username):
            return False, "Username already exists."

        salt = secrets.token_hex(16) #Cryptography: Generate a secure random 16-byte salt, returned as a 32-character hex string
        password_hash = self._hash_password(password, salt) #Generate the final hash using the plaintext password and the unique salt
        ok = self._db.create_user(username, password_hash, salt) #Attempt to save the new user record (username, hash, and salt) into SQLite
        # Return success tuple if the DB insertion was successful, otherwise return a failure tuple
        return (True, "Account created. You can log in now.") if ok else (False, "Could not create account.")

    def login(self, username: str, password: str) -> tuple[bool, str]:
        username = (username or "").strip()

        # BRUTE-FORCE CHECK: Ask the DB if this user is currently in a time-out
        locked, seconds_remaining = self._db.is_locked_out(username)
        if locked:
            minutes = max(1, seconds_remaining // 60)
            return False, f"Account temporarily locked due to repeated failed attempts. Try again in ~{minutes} min."

        user = self._db.get_user(username)
        if not user:
            return False, "Invalid username or password."
        #Re-hash the provided plaintext password using the unique salt stored in this user's DB
        expected_hash = self._hash_password(password, user.salt)

        #Security validation: Use secrets.compare_digest for constant-time comparison. 
        if secrets.compare_digest(expected_hash, user.password_hash):
            # SUCCESS: Reset the brute-force failure counter to 0
            self._db.record_successful_login(username)
            return True, "Login successful."

        #FAILURE: Increment the brute-force failure counter
        self._db.record_failed_login(username)
        return False, "Invalid username or password."