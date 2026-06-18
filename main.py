"""Lookin - Step 3: minimal FastAPI server.

ONE endpoint: POST /embed. Accepts an uploaded image, runs the existing
get_embedding() on it, and returns the embedding length. No database, no
auth, no JWT yet.

Bad input (no face / multiple faces / unreadable image) returns a clean
HTTP 400 with a JSON error message -- never a 500 crash. The temp file is
always cleaned up, success or failure.
"""

import os
import tempfile

from fastapi import FastAPI, File, UploadFile, HTTPException

# Reuse the verified embedding logic unchanged (ArcFace + retinaface + guards).
from embed import get_embedding, NoFaceFoundError, MultipleFacesError

app = FastAPI(title="Lookin", version="0.3.0")


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
