"""Хранилище обработанных отзывов (SQLite).

Запись идемпотентна: повторная доставка одного и того же сообщения из Kafka
(семантика at-least-once) не создаёт дубликатов благодаря первичному ключу
review_id и INSERT OR IGNORE.
"""
import sqlite3
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS reviews (
    review_id     TEXT PRIMARY KEY,
    published_at  TEXT NOT NULL,   -- время публикации отзыва (ISO 8601)
    processed_at  TEXT NOT NULL,   -- время обработки системой
    score         INTEGER,         -- оценка автора, 1..5
    content       TEXT NOT NULL,
    label         TEXT NOT NULL,   -- negative / neutral / positive
    confidence    REAL NOT NULL,
    source        TEXT NOT NULL    -- live / replay
);
CREATE INDEX IF NOT EXISTS idx_reviews_published ON reviews(published_at);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")  # дашборд читает, пока консьюмер пишет
    conn.executescript(SCHEMA)
    return conn


def save_batch(conn: sqlite3.Connection, rows: list[dict]) -> int:
    """Сохраняет пачку размеченных отзывов, возвращает число новых строк."""
    before = conn.total_changes
    conn.executemany(
        """INSERT OR IGNORE INTO reviews
           (review_id, published_at, processed_at, score, content, label, confidence, source)
           VALUES (:review_id, :published_at, :processed_at, :score, :content,
                   :label, :confidence, :source)""",
        rows,
    )
    conn.commit()
    return conn.total_changes - before


def load_all(db_path: Path) -> pd.DataFrame:
    if not Path(db_path).exists():
        return pd.DataFrame()
    with sqlite3.connect(db_path, timeout=30) as conn:
        df = pd.read_sql_query("SELECT * FROM reviews", conn)
    if df.empty:
        return df
    df["published_at"] = pd.to_datetime(df["published_at"], format="ISO8601")
    df["processed_at"] = pd.to_datetime(df["processed_at"], format="ISO8601")
    return df
