"""Lookin - Step 5: Postgres connection + schema.

Connects to Postgres (pgvector image) and creates the schema. No signup/login
logic yet -- just a connection helper and init_db().

Security:
- The password is never hardcoded; it comes only from DATABASE_URL in .env.
- All SQL is static DDL or uses parameterized queries -- never string-formatted.
"""

import os
import sys

import psycopg
from dotenv import load_dotenv

# Load .env into the environment (no-op if the file is absent).
load_dotenv()


def get_connection():
    """Return a new psycopg connection built from DATABASE_URL.

    Raises a clear error if DATABASE_URL is not set, so we never silently
    fall back to insecure defaults.
    """
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is not set. Copy .env.example to .env and fill it in."
        )
    return psycopg.connect(database_url)


def init_db():
    """Create the vector extension and the users table if they don't exist.

    Idempotent: safe to run repeatedly.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            # pgvector extension (image ships it; just enable it).
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

            # Users table. face_embedding is nullable because a face is
            # optional. Embedding dimension is 512 (ArcFace).
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id            BIGSERIAL PRIMARY KEY,
                    email         TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    face_embedding vector(512),
                    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
        # `with conn` commits on success / rolls back on error.


def _print_users_columns():
    """Print the users table columns so we can confirm the schema exists."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            # Parameterized query -- table name passed as a value, not formatted.
            cur.execute(
                """
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = %s
                ORDER BY ordinal_position
                """,
                ("users",),
            )
            rows = cur.fetchall()

    print("users table columns:")
    for column_name, data_type, is_nullable in rows:
        nullable = "NULL" if is_nullable == "YES" else "NOT NULL"
        print(f"  - {column_name}: {data_type} ({nullable})")


if __name__ == "__main__":
    try:
        init_db()
    except Exception as exc:  # surface a clear message instead of a raw trace
        print(f"ERROR: {exc}")
        sys.exit(1)

    print("DB initialized")
    _print_users_columns()
