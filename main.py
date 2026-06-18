"""Lookin - FastAPI server.

Endpoints:
- POST /embed  : image -> face embedding length (Step 3).
- POST /signup : email + password -> creates a user (Step 6). No face,
                 no login, no JWT yet.

Bad input returns a clean HTTP 4xx with a JSON error message -- never a
500 crash. Passwords are bcrypt-hashed; the plaintext password and the
hash are never logged or returned.
"""

import json
import os
import re
import tempfile

import bcrypt
import psycopg
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Reuse the verified embedding logic unchanged (ArcFace + retinaface + guards).
from embed import get_embedding, MODEL_NAME, NoFaceFoundError, MultipleFacesError
# Reuse the DB connection helper unchanged.
from db import get_connection
# Reuse the exact match logic (DeepFace cosine distance + ArcFace threshold).
from match import find_cosine_distance, find_threshold, DISTANCE_METRIC
# JWT token creation and verification.
from jwt_token import create_token, verify_token

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


@app.post("/login")
def login(payload: SignupRequest):
    """Authenticate with email and password.

    200 -> {"email": email, "token": <jwt>}
    401 -> {"detail": "Invalid email or password."}  (same for both wrong password and missing email)

    The error message is intentionally generic: we do NOT say "email not found"
    vs "password wrong", as that leaks whether the email is registered
    (account enumeration attack).
    """
    email = payload.email.strip().lower()
    password = payload.password

    # 1. Look up the user by email.
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, password_hash FROM users WHERE email = %s", (email,))
            row = cur.fetchone()

    # 2. If email not found, or password doesn't match, return the SAME generic
    #    401 (do not leak whether the email exists).
    if row is None:
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    user_id, password_hash = row
    # bcrypt.checkpw expects bytes; both password and hash are checked against each other.
    if not bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8")):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    # 3. Password matches. Issue a JWT.
    token = create_token(user_id, email)
    return {"email": email, "token": token}


@app.post("/enroll-face")
async def enroll_face(email: str = Form(...), file: UploadFile = File(...)):
    """Enroll (or re-enroll) a face for an existing user.

    200 -> {"email": <email>, "face_enrolled": true}
    404 -> no user with that email
    400 -> no face / multiple faces / unreadable image

    DESIGN DECISION: if the user already has a face enrolled, this OVERWRITES
    it (so re-enrolling with a better photo works). Flagged for confirmation.

    The embedding itself is never returned.
    """
    email = email.strip().lower()

    # 1. Verify the user exists BEFORE doing any (expensive) face work.
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM users WHERE email = %s", (email,))
            row = cur.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="User not found.")

    # 2. Save the upload to a temp file, embed it, always clean up.
    #    (Temp-file handling identical to /embed.)
    suffix = os.path.splitext(file.filename or "")[1] or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = tmp.name
    try:
        contents = await file.read()
        tmp.write(contents)
        tmp.close()

        # Existing, unchanged embedding logic (512-d ArcFace + guards).
        embedding = get_embedding(tmp_path)

    except (NoFaceFoundError, MultipleFacesError, FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    finally:
        # Always clean up the temp file, even on error.
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # 3. Store the embedding. pgvector accepts its text form "[v1,v2,...]";
    #    we pass that as a bound parameter and cast to vector -- parameterized,
    #    no string formatting into the SQL itself.
    vector_literal = "[" + ",".join(str(v) for v in embedding) + "]"
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE users SET face_embedding = %s::vector WHERE email = %s",
                (vector_literal, email),
            )

    # Never return the embedding itself.
    return {"email": email, "face_enrolled": True}


@app.post("/login-face")
async def login_face(email: str = Form(...), file: UploadFile = File(...)):
    """Authenticate a user by email + face (1-to-1 fast path).

    200 -> {"authenticated": true,  "email": email, "distance": <rounded>}
    401 -> {"authenticated": false, "distance": <rounded>}
    404 -> no user with that email
    400 -> no face enrolled, or no face / multiple faces / unreadable image

    Embeddings are never returned. We DO return the distance for now (debugging)
    -- flagged as a possible info leak to lock down later.
    """
    email = email.strip().lower()

    # 1. Look up the user and their stored embedding.
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, face_embedding FROM users WHERE email = %s", (email,)
            )
            row = cur.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="User not found.")
    user_id, stored_vector = row
    if stored_vector is None:
        raise HTTPException(status_code=400, detail="No face enrolled for this account.")

    # 2. Embed the uploaded image (guards + temp-file cleanup, same as /embed).
    suffix = os.path.splitext(file.filename or "")[1] or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = tmp.name
    try:
        contents = await file.read()
        tmp.write(contents)
        tmp.close()
        new_embedding = get_embedding(tmp_path)
    except (NoFaceFoundError, MultipleFacesError, FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # 3. Parse the stored vector. pgvector returns its text form "[v1,v2,...]"
    #    which is valid JSON, so json.loads gives us a list of floats.
    stored_embedding = json.loads(stored_vector)

    # 4. Same cosine distance + ArcFace threshold as match.py (not reinvented).
    distance = float(find_cosine_distance(new_embedding, stored_embedding))
    threshold = find_threshold(MODEL_NAME, DISTANCE_METRIC)
    distance_rounded = round(distance, 4)

    # 5. Decide. Smaller distance = more similar; match when distance <= threshold.
    if distance <= threshold:
        token = create_token(user_id, email)
        return {
            "authenticated": True,
            "email": email,
            "distance": distance_rounded,
            "token": token,
        }
    return JSONResponse(
        status_code=401,
        content={"authenticated": False, "distance": distance_rounded},
    )
