import os
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_PUBLIC_URL")

@contextmanager
def get_db():
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    with get_db() as conn:
        cursor = conn.cursor()

        # WEATHER_SNAPSHOT table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS weather_snapshot (
            id SERIAL PRIMARY KEY,
            weather TEXT NOT NULL,
            duration INTEGER NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(weather, timestamp)
        )
        """)

        # ITEM_SNAPSHOT table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS item_snapshot (
            id SERIAL PRIMARY KEY,
            item TEXT NOT NULL,
            shop TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(item, shop, timestamp)
        )
        """)

        conn.commit()
        cursor.close()