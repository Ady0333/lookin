"""Lookin - JWT token handling.

Create and verify JWT tokens for authenticated sessions. Tokens contain
user_id, email, and expiry (24h). Never include password or hash.
"""

import os
from datetime import datetime, timedelta, timezone

import jwt as pyjwt
from dotenv import load_dotenv

load_dotenv()

# HS256 is sufficient for a single-server app. If you do multi-server later,
# consider RS256 (asymmetric).
ALGORITHM = "HS256"
TOKEN_EXPIRY_HOURS = 24

# Never hardcoded. Must be a long random value (64+ bytes) from .env.
JWT_SECRET = os.getenv("JWT_SECRET")
if not JWT_SECRET:
    raise RuntimeError(
        "JWT_SECRET is not set. Generate one: "
        "python -c \"import secrets; print(secrets.token_urlsafe(64))\" "
        "and add it to .env"
    )


def create_token(user_id: int, email: str) -> str:
    """Create a JWT token for an authenticated user.

    Args:
        user_id: The user's database ID.
        email: The user's email (for convenience in verifying contexts).

    Returns:
        A JWT string, signed with HS256, expiring in 24 hours.

    The token contains user_id, email, and exp. It NEVER contains the
    password, password_hash, or face_embedding -- only public/derived data
    needed for authorization.
    """
    now = datetime.now(timezone.utc)
    expires = now + timedelta(hours=TOKEN_EXPIRY_HOURS)

    payload = {
        "user_id": user_id,
        "email": email,
        "exp": expires,
        "iat": now,
    }

    token = pyjwt.encode(payload, JWT_SECRET, algorithm=ALGORITHM)
    return token


def verify_token(token: str) -> dict:
    """Decode and validate a JWT token.

    Args:
        token: The JWT string.

    Returns:
        A dict with the decoded payload (user_id, email, exp, iat, ...).

    Raises:
        pyjwt.ExpiredSignatureError: if the token has expired.
        pyjwt.InvalidTokenError: if the token is invalid or tampered.

    The caller should catch these and return a clear 401 or 403 error.
    """
    decoded = pyjwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
    return decoded
