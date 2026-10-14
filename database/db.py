"""
SQLite database layer for Melanoma CDSS.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import config
from utils.logger import setup_logger

logger = setup_logger("database")

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT NOT NULL,
    patient_name TEXT,
    age INTEGER,
    sex TEXT,
    image_path TEXT,
    image_hash TEXT,
    prediction TEXT NOT NULL,
    confidence REAL,
    malignant_prob REAL,
    benign_prob REAL,
    model_used TEXT,
    ajcc_stage TEXT,
    risk_level TEXT,
    t_category TEXT,
    n_category TEXT,
    m_category TEXT,
    staging_details TEXT,
    staging_recommendation TEXT,
    staging_follow_up TEXT,
    gradcam_path TEXT,
    report_path TEXT,
    doctor_notes TEXT,
    breslow_thickness REAL,
    ulceration INTEGER,
    mitotic_rate REAL,
    lymph_node_involvement INTEGER,
    metastasis INTEGER,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS prediction_cache (
    image_hash TEXT PRIMARY KEY,
    prediction TEXT NOT NULL,
    confidence REAL NOT NULL,
    malignant_prob REAL NOT NULL,
    benign_prob REAL NOT NULL,
    model_used TEXT,
    cache_version TEXT,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_predictions_patient ON predictions(patient_id);
CREATE INDEX IF NOT EXISTS idx_predictions_created ON predictions(created_at);
"""


class Database:
    """Lightweight SQLite wrapper for prediction history."""

    def __init__(self, db_path: Path = config.DATABASE_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialize(self) -> None:
        """Create tables if they do not exist and apply light migrations."""
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            cols = {
                r[1]
                for r in conn.execute("PRAGMA table_info(predictions)").fetchall()
            }
            migrations = {
                "image_hash": "ALTER TABLE predictions ADD COLUMN image_hash TEXT",
                "staging_details": "ALTER TABLE predictions ADD COLUMN staging_details TEXT",
                "staging_recommendation": "ALTER TABLE predictions ADD COLUMN staging_recommendation TEXT",
                "staging_follow_up": "ALTER TABLE predictions ADD COLUMN staging_follow_up TEXT",
            }
            cache_cols = {
                r[1]
                for r in conn.execute("PRAGMA table_info(prediction_cache)").fetchall()
            }
            if "cache_version" not in cache_cols:
                conn.execute(
                    "ALTER TABLE prediction_cache ADD COLUMN cache_version TEXT"
                )
            for col, sql in migrations.items():
                if col not in cols:
                    conn.execute(sql)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_predictions_hash "
                "ON predictions(image_hash)"
            )
        logger.info("Database initialized at %s", self.db_path)

    def insert_prediction(self, record: Dict[str, Any]) -> int:
        """Insert a prediction record; return row id."""
        fields = [
            "patient_id", "patient_name", "age", "sex", "image_path", "image_hash",
            "prediction", "confidence", "malignant_prob", "benign_prob",
            "model_used", "ajcc_stage", "risk_level", "t_category",
            "n_category", "m_category", "staging_details", "staging_recommendation",
            "staging_follow_up", "gradcam_path", "report_path",
            "doctor_notes", "breslow_thickness", "ulceration",
            "mitotic_rate", "lymph_node_involvement", "metastasis",
            "created_at", "owner_user_id", "assigned_doctor_id",
            "review_status", "report_code",
        ]
        record = dict(record)
        record.setdefault("created_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        values = [record.get(f) for f in fields]
        placeholders = ", ".join(["?"] * len(fields))
        cols = ", ".join(fields)
        sql = f"INSERT INTO predictions ({cols}) VALUES ({placeholders})"
        with self._connect() as conn:
            cur = conn.execute(sql, values)
            row_id = cur.lastrowid
        logger.info("Inserted prediction id=%s patient=%s", row_id, record.get("patient_id"))
        return int(row_id)

    def update_prediction(self, pred_id: int, fields: Dict[str, Any]) -> bool:
        """Update selected columns for a prediction row."""
        if not fields:
            return False
        allowed = {
            "ajcc_stage", "risk_level", "t_category", "n_category", "m_category",
            "staging_details", "staging_recommendation", "staging_follow_up",
            "report_path", "doctor_notes", "breslow_thickness", "ulceration",
            "mitotic_rate", "lymph_node_involvement", "metastasis",
            "owner_user_id", "assigned_doctor_id", "review_status", "report_code",
            "gradcam_path",
        }
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return False
        sets = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [pred_id]
        with self._connect() as conn:
            cur = conn.execute(
                f"UPDATE predictions SET {sets} WHERE id = ?",
                values,
            )
            return cur.rowcount > 0

    def get_cached_prediction(self, image_hash: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM prediction_cache WHERE image_hash = ?",
                (image_hash,),
            ).fetchone()
        if not row:
            return None
        cached = dict(row)
        if cached.get("cache_version") != config.get_prediction_cache_version():
            return None
        return cached

    def upsert_prediction_cache(self, payload: Dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO prediction_cache
                    (image_hash, prediction, confidence, malignant_prob,
                     benign_prob, model_used, cache_version, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(image_hash) DO UPDATE SET
                    prediction=excluded.prediction,
                    confidence=excluded.confidence,
                    malignant_prob=excluded.malignant_prob,
                    benign_prob=excluded.benign_prob,
                    model_used=excluded.model_used,
                    cache_version=excluded.cache_version,
                    updated_at=excluded.updated_at
                """,
                (
                    payload["image_hash"],
                    payload["prediction"],
                    payload["confidence"],
                    payload["malignant_prob"],
                    payload["benign_prob"],
                    payload.get("model_used"),
                    payload.get("cache_version", config.get_prediction_cache_version()),
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )

    def get_prediction(self, pred_id: int) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM predictions WHERE id = ?", (pred_id,)
            ).fetchone()
        return dict(row) if row else None

    def get_history(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM predictions ORDER BY datetime(created_at) DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [dict(r) for r in rows]

    def count_predictions(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0])

    def get_stats(self) -> Dict[str, Any]:
        """Aggregate stats for dashboard."""
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            malignant = conn.execute(
                "SELECT COUNT(*) FROM predictions WHERE prediction = 'Malignant'"
            ).fetchone()[0]
            benign = conn.execute(
                "SELECT COUNT(*) FROM predictions WHERE prediction = 'Benign'"
            ).fetchone()[0]
            avg_conf = conn.execute(
                "SELECT AVG(confidence) FROM predictions"
            ).fetchone()[0] or 0.0
            stages = conn.execute(
                "SELECT ajcc_stage, COUNT(*) as cnt FROM predictions "
                "WHERE ajcc_stage IS NOT NULL GROUP BY ajcc_stage"
            ).fetchall()
        return {
            "total": total,
            "malignant": malignant,
            "benign": benign,
            "avg_confidence": round(float(avg_conf) * 100, 2) if avg_conf <= 1 else round(float(avg_conf), 2),
            "stages": {r["ajcc_stage"]: r["cnt"] for r in stages},
        }

    def delete_prediction(self, pred_id: int) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM predictions WHERE id = ?", (pred_id,))
            return cur.rowcount > 0


# Singleton-style accessor
_db: Optional[Database] = None


def get_db() -> Database:
    global _db
    if _db is None:
        _db = Database()
    return _db
