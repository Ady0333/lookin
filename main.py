"""Lookin - FastAPI server.

Endpoints:
- POST /embed  : image -> face embedding length (Step 3).
- POST /signup : email + password -> creates a user (Step 6). No face,
                 no login, no JWT yet.

Bad input returns a clean HTTP 4xx with a JSON error message -- never a
500 crash. Passwords are bcrypt-hashed; the plaintext password and the
hash are never logged or returned.
"""

import os
import re
import tempfile

import bcrypt
import psycopg
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel

# Reuse the verified embedding logic unchanged (ArcFace + retinaface + guards).
from embed import get_embedding, NoFaceFoundError, MultipleFacesError
# Reuse the DB connection helper unchanged.
from db import get_connection

app = FastAPI(title="Lookin", version="0.6.0")

# Minimum password length (chars). Adaptive bcrypt hashing is used regardless.
MIN_PASSWORD_LENGTH = 8

# Pragmatic email-format check: non-empty local part, "@", domain with a dot.
# Not a full RFC validator -- just rejects obviously-malformed input.
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@app.post("/embed")
async def embed(file: UploadFile = File(...)):
    """Accept an uploaded image and return its face embedding length.

    Success -> 200 {"face_found": true, "embedding_length": 512}
    Bad image / no face / multiple faces -> 400 {"detail": "<clear message>"}
    """
    # Preserve the original extension so OpenCV/DeepFace decode by the right
    # format. Default to .jpg if the client sent no usable filename.
    suffix = os.path.splitext(file.filename or "")[1] or ".jpg"

    # Create a temp file we fully control. delete=False so we can hand its
    # path to get_embedding and then delete it ourselves in finally.
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = tmp.name
    try:
        # Stream the upload to disk, then close so get_embedding can read it.
        contents = await file.read()
        tmp.write(contents)
        tmp.close()

        # Run the existing, unchanged embedding logic.
        embedding = get_embedding(tmp_path)

        return {"face_found": True, "embedding_length": len(embedding)}

    except (NoFaceFoundError, MultipleFacesError, FileNotFoundError, ValueError) as exc:
        # All expected bad-input cases -> 400 with a clear message, not a 500.
        # (get_embedding wraps detection/decoding failures; ValueError catches
        #  any remaining unreadable-image case.)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    finally:
        # Always clean up the temp file, even on error.
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


class SignupRequest(BaseModel):
    """Signup payload. Kept as plain strings so WE control validation and the
    resulting status codes (Pydantic's own errors would be 422, but we want
    400). The password lives only in memory and is never logged or returned.
    """

    email: str
    password: str


@app.post("/signup", status_code=201)
def signup(payload: SignupRequest):
    """Create a new email+password user.

    201 -> {"id": <new_id>, "email": <email>}
    400 -> invalid email format or password shorter than MIN_PASSWORD_LENGTH
    409 -> email already registered
    """
    email = payload.email.strip().lower()
    password = payload.password

    # --- Validation (clear 400s, never reveal anything about the password) ---
    if not EMAIL_RE.match(email):
        raise HTTPException(status_code=400, detail="Invalid email format.")
    if len(password) < MIN_PASSWORD_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Password must be at least {MIN_PASSWORD_LENGTH} characters.",
        )

    # --- Hash with bcrypt (adaptive, salted). Never store plaintext. ---
    # bcrypt operates on bytes and caps at 72 bytes; encode explicitly.
    password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    # --- Insert. Parameterized query -- no string formatting into SQL. ---
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO users (email, password_hash) VALUES (%s, %s) RETURNING id",
                    (email, password_hash),
                )
                new_id = cur.fetchone()[0]
    except psycopg.errors.UniqueViolation as exc:
        # Email already exists -> clean 409, not a 500 crash.
        raise HTTPException(status_code=409, detail="Email already registered.") from exc

    # Never return the password or the hash.
    return {"id": new_id, "email": email}
