"""
database.py — Async SQLite database layer using aiosqlite.

Schema design:
  - events: Raw event store (event_id is primary key for deduplication)
  - sessions: Aggregated visitor sessions (computed on ingest)
  - pos_transactions: POS data imported from pos_transactions.csv

Design decision: SQLite chosen for:
  - Zero-dependency deployment (no Postgres setup)
  - Sufficient for 40-store event volumes at this scale
  - WAL mode enables concurrent reads during write-heavy ingest
  - Can be swapped for Postgres via DATABASE_URL env var in production

For Postgres: replace aiosqlite with asyncpg and update SQL accordingly.
"""

import os
import logging
import aiosqlite
from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH = os.environ.get("DB_PATH", "/data/store_intelligence.db")


async def get_connection() -> aiosqlite.Connection:
    """Get an async DB connection. Raises if DB is unavailable."""
    try:
        conn = await aiosqlite.connect(DB_PATH)
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA synchronous=NORMAL")
        await conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
        return conn
    except Exception as e:
        logger.error(f'"DB connection failed: {e}"')
        raise


async def init_db():
    """Create all tables if they don't exist."""
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)

    async with await get_connection() as conn:
        await conn.executescript("""
            CREATE TABLE IF NOT EXISTS events (
                event_id       TEXT PRIMARY KEY,
                store_id       TEXT NOT NULL,
                camera_id      TEXT NOT NULL,
                visitor_id     TEXT NOT NULL,
                event_type     TEXT NOT NULL,
                timestamp      TEXT NOT NULL,
                zone_id        TEXT,
                dwell_ms       INTEGER DEFAULT 0,
                is_staff       INTEGER DEFAULT 0,
                confidence     REAL NOT NULL,
                queue_depth    INTEGER,
                sku_zone       TEXT,
                session_seq    INTEGER DEFAULT 0,
                ingested_at    TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_events_store_ts
                ON events(store_id, timestamp);

            CREATE INDEX IF NOT EXISTS idx_events_visitor
                ON events(visitor_id);

            CREATE INDEX IF NOT EXISTS idx_events_type
                ON events(store_id, event_type, timestamp);

            CREATE TABLE IF NOT EXISTS pos_transactions (
                transaction_id  TEXT PRIMARY KEY,
                store_id        TEXT NOT NULL,
                timestamp       TEXT NOT NULL,
                basket_value_inr REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_pos_store_ts
                ON pos_transactions(store_id, timestamp);

            CREATE TABLE IF NOT EXISTS anomaly_log (
                anomaly_id    TEXT PRIMARY KEY,
                store_id      TEXT NOT NULL,
                anomaly_type  TEXT NOT NULL,
                severity      TEXT NOT NULL,
                description   TEXT NOT NULL,
                suggested_action TEXT NOT NULL,
                detected_at   TEXT NOT NULL,
                resolved_at   TEXT,
                metadata_json TEXT DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_anomaly_store
                ON anomaly_log(store_id, detected_at);
        """)
        await conn.commit()
    logger.info('"Database schema initialised"')


async def get_db_health() -> dict:
    """Check database connectivity. Returns status dict."""
    try:
        async with await get_connection() as conn:
            async with conn.execute("SELECT COUNT(*) as cnt FROM events") as cur:
                row = await cur.fetchone()
                return {"status": "OK", "event_count": row["cnt"]}
    except Exception as e:
        return {"status": "UNAVAILABLE", "error": str(e)}
