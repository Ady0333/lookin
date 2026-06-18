"""Lookin - Step 1: single-face embedding.

Produces a face embedding vector for exactly one face in an image using
DeepFace's ArcFace model. This step does NO matching, NO API, NO web --
it only turns one image into one embedding (or fails loudly).

ArcFace always produces a fixed-length 512-dimensional vector. We assert
that so any model/version drift is caught immediately.
"""

import os
import sys

from deepface import DeepFace

# --- Constants --------------------------------------------------------------

# The face-recognition model. ArcFace outputs a fixed-length embedding.
MODEL_NAME = "ArcFace"

# ArcFace embeddings are always 512-dimensional. We confirm this on every
# call so we notice immediately if the model or its version ever changes.
EXPECTED_EMBEDDING_LENGTH = 512


# --- Clear, specific errors -------------------------------------------------
# Custom exceptions so callers (and the main block) can tell exactly what
# went wrong instead of guessing from a generic crash.

class NoFaceFoundError(Exception):
    """Raised when no face can be detected in the image."""


class MultipleFacesError(Exception):
    """Raised when more than one face is detected in the image."""


# --- Core -------------------------------------------------------------------

def get_embedding(image_path):
    """Return the ArcFace embedding for the single face in `image_path`.

    Raises:
        FileNotFoundError: the path does not exist or is not a file.
        NoFaceFoundError:  no face detected in the image.
        MultipleFacesError: more than one face detected.

    Returns:
        list[float]: the face embedding (length EXPECTED_EMBEDDING_LENGTH).
    """
    # 1. Validate the path BEFORE handing anything to DeepFace, so a missing
    #    file gives a clear, immediate error rather than a confusing one.
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image path does not exist or is not a file: {image_path}")

    # 2. Ask DeepFace for the embedding(s). enforce_detection=True makes
    #    DeepFace raise instead of silently returning a bogus embedding when
    #    it cannot find a face -- exactly what we want for security.
    #    represent() returns a list with one dict per detected face.
    try:
        results = DeepFace.represent(
            img_path=image_path,
            model_name=MODEL_NAME,
            detector_backend="retinaface",  # stronger than the default opencv Haar detector
            enforce_detection=True,
        )
    except ValueError as exc:
        # DeepFace raises ValueError ("Face could not be detected...") when
        # enforce_detection is on and it finds no face. Translate that into
        # our own clear error instead of leaking the internal exception.
        raise NoFaceFoundError(
            f"No face detected in image: {image_path}"
        ) from exc

    # 3. Defensive checks on how many faces came back.
    if len(results) == 0:
        # Shouldn't happen with enforce_detection=True, but never trust it.
        raise NoFaceFoundError(f"No face detected in image: {image_path}")

    if len(results) > 1:
        raise MultipleFacesError(
            f"Multiple faces detected ({len(results)}) in image: {image_path}. "
            "This step requires exactly one face."
        )

    # 4. Exactly one face: pull out its embedding vector.
    embedding = results[0]["embedding"]

    # 5. Confirm the fixed length so model drift can never pass silently.
    if len(embedding) != EXPECTED_EMBEDDING_LENGTH:
        raise ValueError(
            f"Unexpected embedding length {len(embedding)}; "
            f"expected {EXPECTED_EMBEDDING_LENGTH} for {MODEL_NAME}."
        )

    return embedding


# --- Manual check / CLI -----------------------------------------------------

if __name__ == "__main__":
    # Expect exactly one argument: the path to an image.
    if len(sys.argv) != 2:
        print("Usage: python embed.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    try:
        embedding = get_embedding(image_path)
    except (FileNotFoundError, NoFaceFoundError, MultipleFacesError) as exc:
        # Clear, non-crashing failure with a readable message.
        print(f"ERROR: {exc}")
        sys.exit(1)

    # Success: report what we got so we can eyeball consistency.
    print(f"Face found: yes (exactly 1)")
    print(f"Embedding length: {len(embedding)}")
    print(f"Length matches expected {EXPECTED_EMBEDDING_LENGTH}: "
          f"{len(embedding) == EXPECTED_EMBEDDING_LENGTH}")
    print(f"First 5 values: {embedding[:5]}")
