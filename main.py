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
import numpy as np
import psycopg
from deepface import DeepFace
from fastapi import FastAPI, File, Form, Header, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import jwt as pyjwt

# Reuse the verified embedding logic unchanged (ArcFace + retinaface + guards).
from embed import get_embedding, is_live_face, MODEL_NAME, NoFaceFoundError, MultipleFacesError
# Reuse the DB connection helper unchanged.
from db import get_connection
# Reuse the exact match logic (DeepFace cosine distance + ArcFace threshold).
from match import find_cosine_distance, find_threshold, DISTANCE_METRIC
# JWT token creation and verification.
from jwt_token import create_token, verify_token

app = FastAPI(title="Lookin", version="0.6.0")

# Serve static assets (logos, icons) from lookin/static/ at /static/...
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS for the local demo frontend. The frontend can be opened directly as a
# file (Origin "null") or served from 127.0.0.1:8001. We do NOT use "*" -- this
# is an explicit allow-list. No cookies are used (JWT is sent in a header), so
# allow_credentials stays False.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8001",
        "http://localhost:8001",
        "http://172.30.244.187:8001",
        "null",  # file:// origin
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Model warmup -----------------------------------------------------------
# DeepFace lazily loads its heavy models on the first request, which makes the
# first face login painfully slow. Pre-load them at startup so they stay
# resident in memory and every request is fast from the first one onward.

@app.on_event("startup")
def warm_up_models():
    print("Models warming up...")

    # Face recognition (ArcFace) loads directly.
    DeepFace.build_model("ArcFace")

    # RetinaFace (the detector) loads automatically on the first detection call,
    # so trigger it with a dummy extract on a tiny blank image. A blank image has
    # no face, so a no-face error is expected here -- we only want the model
    # loaded into memory, so swallow it.
    blank = np.zeros((100, 100, 3), dtype=np.uint8)
    try:
        DeepFace.extract_faces(
            img_path=blank,
            detector_backend="retinaface",
            enforce_detection=True,
        )
    except Exception:
        # Expected: no face on a blank image. The detector model is now loaded.
        pass

    print("Models ready.")


# --- Frontend ---------------------------------------------------------------
# Serve the single-page demo frontend from the same app.

@app.get("/")
async def serve_frontend():
    return FileResponse("frontend/index.html")


@app.get("/index.html")
async def serve_frontend_html():
    return FileResponse("frontend/index.html")


# Minimum password length (chars). Adaptive bcrypt hashing is used regardless.
MIN_PASSWORD_LENGTH = 8

# Pragmatic email-format check: non-empty local part, "@", domain with a dot.
# Not a full RFC validator -- just rejects obviously-malformed input.
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# Stricter threshold for the 1-to-many "forgot email" search than the 1-to-1
# threshold (0.68). In 1-to-many we compare against EVERY enrolled user, so the
# chance of a false accept grows with population size; a tighter cutoff reduces
# the risk of matching the wrong person when no email is provided to anchor it.
STRICT_FACE_THRESHOLD = 0.55


def mask_email(email: str) -> str:
    """Mask an email for display so we never reveal a full address that the
    caller shouldn't already know (e.g. other people's accounts).

    Rules (mirror the frontend's masking exactly):
      local part <= 3 chars: first 1 + "***" + last 1
      local part <= 6 chars: first 2 + "***" + last 2
      otherwise:             first 2 + "***" + last 3
    """
    if "@" not in email:
        return email
    local, domain = email.split("@", 1)
    n = len(local)
    if n <= 3:
        masked = local[:1] + "***" + local[-1:]
    elif n <= 6:
        masked = local[:2] + "***" + local[-2:]
    else:
        masked = local[:2] + "***" + local[-3:]
    return masked + "@" + domain


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

        # Liveness check before storing -- reject a photo/screen spoof at
        # enrollment so a spoofed face can never be registered. Runs while
        # tmp_path still exists (before the finally deletes it).
        # is_live_face fails closed (returns not-live) on any model error.
        live, _spoof_score = is_live_face(tmp_path)
        if not live:
            raise HTTPException(
                status_code=400,
                detail="Liveness check failed. Please use a live camera, not a photo.",
            )

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

        # Liveness check FIRST -- fail fast on a spoof before any identity work.
        # is_live_face fails closed (returns not-live) on any model error.
        live, _spoof_score = is_live_face(tmp_path)
        if not live:
            raise HTTPException(status_code=401, detail="Liveness check failed")

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


@app.get("/me")
def get_user_profile(authorization: str = Header(None)):
    """Get the authenticated user's profile from a valid JWT.

    200 -> {"id": <id>, "email": <email>, "face_enrolled": <bool>}
    401 -> missing/invalid header, or invalid/expired token
    """
    # 1. Validate the Authorization header format.
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")

    token = parts[1]

    # 2. Validate and decode the token.
    try:
        decoded = verify_token(token)
    except (pyjwt.ExpiredSignatureError, pyjwt.InvalidTokenError):
        raise HTTPException(status_code=401, detail="Invalid or expired token") from None

    # 3. Extract user_id from the token and look up the user.
    user_id = decoded.get("user_id")
    if user_id is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, email, face_embedding IS NOT NULL as face_enrolled FROM users WHERE id = %s",
                (user_id,),
            )
            row = cur.fetchone()

    if row is None:
        # User ID in token doesn't exist in DB (shouldn't happen, but handle it).
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user_id, email, face_enrolled = row
    # Never return the password_hash or face_embedding itself.
    return {"id": user_id, "email": email, "face_enrolled": face_enrolled}


@app.post("/login-face-search")
async def login_face_search(file: UploadFile = File(...)):
    """Authenticate by face alone (1-to-many "forgot email" path).

    No email is provided: we embed the uploaded face and search ALL enrolled
    users for matches under the STRICTER threshold (0.55) than the 1-to-1 path.

    200 (one match)      -> {"match_count": 1, "email": <masked>, "token": <jwt>}
    200 (several matches) -> {"match_count": N,
                             "accounts": [{"masked_email": ..., "user_id": ...}]}
                             (NO token -- caller must pick one via /select-account)
    401 -> "No match found"          (nothing close enough, or no enrolled users)
    401 -> "Liveness check failed"   (spoof)
    400 -> no face / multiple faces / unreadable image
    """
    # 1. Embed the uploaded image (guards + temp-file cleanup, same as /embed).
    suffix = os.path.splitext(file.filename or "")[1] or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = tmp.name
    try:
        contents = await file.read()
        tmp.write(contents)
        tmp.close()

        # Liveness check FIRST -- fail fast on a spoof before the (expensive)
        # 1-to-many scan. is_live_face fails closed on any model error.
        live, _spoof_score = is_live_face(tmp_path)
        if not live:
            raise HTTPException(status_code=401, detail="Liveness check failed")

        new_embedding = get_embedding(tmp_path)
    except (NoFaceFoundError, MultipleFacesError, FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # 2. Pull every enrolled user. Simple linear scan in Python for now
    #    (pgvector ANN indexing is a later optimization once user count matters).
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, email, face_embedding FROM users WHERE face_embedding IS NOT NULL"
            )
            rows = cur.fetchall()

    # 3. Collect ALL users under the strict threshold (same cosine logic as
    #    match.py / /login-face), closest first.
    matches = []
    for row_id, row_email, stored_vector in rows:
        stored_embedding = json.loads(stored_vector)
        distance = float(find_cosine_distance(new_embedding, stored_embedding))
        if distance <= STRICT_FACE_THRESHOLD:
            matches.append((distance, row_id, row_email))
    matches.sort(key=lambda m: m[0])  # closest first

    # 4. No match -> generic 401 (never reveal who was near).
    if not matches:
        raise HTTPException(status_code=401, detail="No match found")

    # 5a. Exactly one match -> issue a JWT immediately (return masked email).
    if len(matches) == 1:
        _distance, user_id, email = matches[0]
        token = create_token(user_id, email)
        return {"match_count": 1, "email": mask_email(email), "token": token}

    # 5b. Several faces match -> do NOT issue a token. Return masked choices and
    #     let the caller re-verify against one specific account via /select-account.
    return {
        "match_count": len(matches),
        "accounts": [
            {"masked_email": mask_email(email), "user_id": user_id}
            for (_distance, user_id, email) in matches
        ],
    }


@app.post("/check-face-exists")
async def check_face_exists(file: UploadFile = File(...)):
    """Read-only check: is this face already enrolled on any account?

    Used during enrollment to warn about a face that's already linked elsewhere.
    Issues NO token and reveals only MASKED emails.

    200 -> {"found": bool, "accounts": [<masked email>, ...]}
    400 -> no face / multiple faces / unreadable image
    """
    # Embed the uploaded image (temp-file cleanup, same as /embed). No liveness
    # here -- this is a non-authenticating, read-only pre-check.
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

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT email, face_embedding FROM users WHERE face_embedding IS NOT NULL"
            )
            rows = cur.fetchall()

    accounts = []
    for row_email, stored_vector in rows:
        stored_embedding = json.loads(stored_vector)
        distance = float(find_cosine_distance(new_embedding, stored_embedding))
        if distance <= STRICT_FACE_THRESHOLD:
            accounts.append(mask_email(row_email))

    return {"found": len(accounts) > 0, "accounts": accounts}


@app.post("/select-account")
async def select_account(user_id: int = Form(...), file: UploadFile = File(...)):
    """Re-verify a face against ONE specific account and issue a token.

    Second step of the multi-match face-search flow: the caller picked a
    user_id from /login-face-search and now proves the face matches THAT account
    (1-to-1, threshold 0.68), with a liveness check.

    200 -> {"email": <email>, "token": <jwt>}
    401 -> "Liveness check failed" or "Face verification failed"
    404 -> user not found
    400 -> no face / multiple faces / unreadable image, or no face enrolled
    """
    # 1. Look up the chosen user and their stored embedding.
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT email, face_embedding FROM users WHERE id = %s", (user_id,)
            )
            row = cur.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="User not found.")
    email, stored_vector = row
    if stored_vector is None:
        raise HTTPException(status_code=400, detail="No face enrolled for this account.")

    # 2. Embed + liveness on the uploaded image (temp-file cleanup, same as elsewhere).
    suffix = os.path.splitext(file.filename or "")[1] or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = tmp.name
    try:
        contents = await file.read()
        tmp.write(contents)
        tmp.close()

        live, _spoof_score = is_live_face(tmp_path)
        if not live:
            raise HTTPException(status_code=401, detail="Liveness check failed")

        new_embedding = get_embedding(tmp_path)
    except (NoFaceFoundError, MultipleFacesError, FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # 3. 1-to-1 match against the chosen account (standard 0.68 threshold).
    stored_embedding = json.loads(stored_vector)
    distance = float(find_cosine_distance(new_embedding, stored_embedding))
    threshold = find_threshold(MODEL_NAME, DISTANCE_METRIC)
    if distance <= threshold:
        token = create_token(user_id, email)
        return {"email": email, "token": token}

    raise HTTPException(status_code=401, detail="Face verification failed")
