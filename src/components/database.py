import os
import sys
from dotenv import load_dotenv
from sqlalchemy import create_engine
import psycopg2

from src.logger import logging
from src.exception import CustomException

# Load the environment variables
load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

def _build_db_uri():
    missing = [name for name, val in (
        ("DB_USER", DB_USER), ("DB_PASSWORD", DB_PASSWORD),
        ("DB_HOST", DB_HOST), ("DB_PORT", DB_PORT), ("DB_NAME", DB_NAME)
    ) if not val]
    if missing:
        logging.error("Database environment variables missing: %s", missing)
        return None
    return f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

DB_URI = _build_db_uri()

engine = None
if DB_URI:
    try:
        engine = create_engine(DB_URI)
        logging.info("Created SQLAlchemy engine")
    except Exception as e:
        logging.exception("Failed to create SQLAlchemy engine for %s", DB_URI)
        engine = None
else:
    logging.warning("DB_URI not configured; engine not created")

def get_connection():
    """
    Return a psycopg2 connection to the configured Postgres DB.

    Raises:
        CustomException: if connection cannot be established or configuration missing.
    """
    if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME]):
        msg = "Database configuration incomplete: one or more DB_* env vars missing"
        logging.error(msg)
        raise CustomException(msg, sys)

    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT
        )
        logging.info("Opened psycopg2 connection")
        return conn
    except Exception as e:
        logging.exception("Failed to obtain psycopg2 connection")
        raise CustomException(e, sys)