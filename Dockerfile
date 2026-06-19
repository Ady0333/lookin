FROM python:3.10-slim

# System libraries required by OpenCV (libGL.so.1, glib), which deepface/
# opencv-python load at import time. Without these the container crashes on
# startup, since main.py -> embed.py imports cv2.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first so this layer is cached unless requirements change.
# --extra-index-url is needed because requirements.txt pins torch==2.12.1+cpu,
# a CPU build hosted on PyTorch's index rather than plain PyPI.
COPY requirements.txt .
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

# Pre-download model weights at build time so they're baked into the image and
# the first request isn't slowed by a cold download. ArcFace (recognition):
RUN python -c "from deepface import DeepFace; DeepFace.build_model('ArcFace')"

# RetinaFace (detector) downloads on first detection call; trigger it on a blank
# image. A blank image has no face, so the error is expected -- swallow it (we
# only want the weights cached). || true keeps the single-line form valid in a
# Dockerfile (a literal multi-line python -c would not parse).
RUN python -c "import numpy as np; from deepface import DeepFace; DeepFace.extract_faces(img_path=np.zeros((100,100,3), dtype=np.uint8), detector_backend='retinaface', enforce_detection=False)" || true

# Copy the rest of the project. .env is excluded via .dockerignore so secrets
# are never baked into the image; DATABASE_URL and JWT_SECRET are supplied at
# runtime (e.g. Railway's environment variables panel).
COPY . .

EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
