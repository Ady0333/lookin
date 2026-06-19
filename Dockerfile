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

# Copy the rest of the project. .env is excluded via .dockerignore so secrets
# are never baked into the image; DATABASE_URL and JWT_SECRET are supplied at
# runtime (e.g. Railway's environment variables panel).
COPY . .

EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
