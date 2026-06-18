"""Lookin - Step 2: face matching.

Given two images, decide whether they show the SAME person. This step does
NO API, NO web, NO database -- it only compares two faces.

It reuses get_embedding() from embed.py (ArcFace + retinaface + the no-face /
multiple-faces / 512-length guards) unchanged, and uses DeepFace's OWN
threshold table and cosine-distance function so we never invent a number.
"""

import sys

# Reuse the verified embedding logic -- do not duplicate or change it.
from embed import get_embedding, MODEL_NAME, NoFaceFoundError, MultipleFacesError

# Pull the threshold and distance math straight from DeepFace so they always
# match DeepFace's own tuned defaults (and track any future version change).
from deepface.modules.verification import find_cosine_distance, find_threshold

# Distance metric we use. ArcFace's pre-tuned cosine threshold (currently 0.68)
# comes from DeepFace's own config -- see find_threshold below.
DISTANCE_METRIC = "cosine"


def is_same_person(image_path_a, image_path_b):
    """Compare two images and return the match decision.

    Returns a tuple:
        (same_person: bool, distance: float, threshold: float)

    A pair is the SAME person when the cosine distance is <= the threshold
    (smaller distance = more similar), matching DeepFace's own convention.

    Propagates the same clear errors as get_embedding (FileNotFoundError,
    NoFaceFoundError, MultipleFacesError) for either image.
    """
    # Get one 512-d ArcFace embedding per image (guards run inside).
    embedding_a = get_embedding(image_path_a)
    embedding_b = get_embedding(image_path_b)

    # Cosine distance via DeepFace's own function. Cast to plain float for
    # clean printing (it returns a numpy float64).
    distance = float(find_cosine_distance(embedding_a, embedding_b))

    # DeepFace's pre-tuned threshold for (ArcFace, cosine). Looked up, not
    # invented -- changing the model or metric automatically picks the right one.
    threshold = find_threshold(MODEL_NAME, DISTANCE_METRIC)

    same_person = distance <= threshold
    return same_person, distance, threshold


if __name__ == "__main__":
    # Expect exactly two arguments: the two image paths to compare.
    if len(sys.argv) != 3:
        print("Usage: python match.py <image_path_a> <image_path_b>")
        sys.exit(1)

    image_path_a = sys.argv[1]
    image_path_b = sys.argv[2]

    try:
        same_person, distance, threshold = is_same_person(image_path_a, image_path_b)
    except (FileNotFoundError, NoFaceFoundError, MultipleFacesError) as exc:
        # Clear, non-crashing failure with a readable message.
        print(f"ERROR: {exc}")
        sys.exit(1)

    # Report the numbers so the decision is transparent and tunable later.
    verdict = "SAME PERSON" if same_person else "DIFFERENT PERSON"
    print(f"Model:     {MODEL_NAME} ({DISTANCE_METRIC} distance)")
    print(f"Distance:  {distance:.4f}")
    print(f"Threshold: {threshold}  (match when distance <= threshold)")
    print(f"Verdict:   {verdict}")
