"""JWT issue/verify and password hashing.

Role scopes are checked per endpoint via app.api.deps.require_roles.
"""
from datetime import datetime, timedelta, timezone

import bcrypt
from jose import JWTError, jwt

from app.core.config import get_settings

ROLES = ("admin", "dma", "responder", "public")

# bcrypt hashes at most 72 bytes of input; longer passwords are truncated by
# the algorithm itself, so truncate explicitly rather than raising at runtime.
_MAX_BYTES = 72


def _encode(raw: str) -> bytes:
    return raw.encode("utf-8")[:_MAX_BYTES]


def hash_password(raw: str) -> str:
    return bcrypt.hashpw(_encode(raw), bcrypt.gensalt(rounds=12)).decode()


def verify_password(raw: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(_encode(raw), hashed.encode())
    except (ValueError, TypeError):
        return False


def create_access_token(subject: str, role: str) -> str:
    s = get_settings()
    now = datetime.now(timezone.utc)
    payload = {
        "sub": subject,
        "role": role,
        "iat": now,
        "exp": now + timedelta(minutes=s.access_token_minutes),
    }
    return jwt.encode(payload, s.jwt_secret, algorithm=s.jwt_algorithm)


def decode_token(token: str) -> dict | None:
    s = get_settings()
    try:
        return jwt.decode(token, s.jwt_secret, algorithms=[s.jwt_algorithm])
    except JWTError:
        return None
