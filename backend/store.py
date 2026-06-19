"""
Local-first persistence for CardioVoice
=======================================

SQLite store (stdlib, zero-config) for the single-Mac deployment. Keeps
encounters and their generated/edited outpatient notes, with a draft/verified
status and basic audit timestamps. Data never leaves the machine.

Tables:
    encounters  - one per recording session (patient + transcript + audio meta)
    records     - the structured note for an encounter (editable, versioned by
                  updated_at, with draft/verified status and who verified)

Usage:
    from store import RecordStore
    store = RecordStore()                       # ~/.cardiovoice/cardiovoice.db
    enc_id = store.create_encounter(patient={...}, transcript="...")
    rec_id = store.save_record(enc_id, fields={...}, status="draft")
    store.verify_record(rec_id, verified_by="Dr. Li")
    store.list_encounters()
"""

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_DB = Path.home() / ".cardiovoice" / "cardiovoice.db"

NOTE_FIELDS = [
    "chief_complaint",
    "present_history",
    "past_history",
    "cardiovascular_exam",
    "ecg_findings",
    "assessment",
    "plan",
]


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


class RecordStore:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: Streamlit / pipeline touch the DB from
        # different threads; guard writes with a lock.
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS encounters (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_name  TEXT,
                    patient_age   INTEGER,
                    patient_gender TEXT,
                    template      TEXT,
                    transcript    TEXT,
                    audio_seconds REAL,
                    created_at    TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS records (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    encounter_id  INTEGER NOT NULL REFERENCES encounters(id),
                    fields_json   TEXT NOT NULL,
                    raw_response  TEXT,
                    model         TEXT,
                    status        TEXT NOT NULL DEFAULT 'draft',
                    verified_by   TEXT,
                    created_at    TEXT NOT NULL,
                    updated_at    TEXT NOT NULL
                );
                """
            )

    # ---------------------------------------------------------- encounters
    def create_encounter(
        self,
        patient: Dict[str, Any],
        transcript: str = "",
        template: str = "general",
        audio_seconds: float = 0.0,
    ) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                """INSERT INTO encounters
                   (patient_name, patient_age, patient_gender, template,
                    transcript, audio_seconds, created_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    patient.get("name"),
                    patient.get("age"),
                    patient.get("gender"),
                    template,
                    transcript,
                    audio_seconds,
                    _now(),
                ),
            )
            return cur.lastrowid

    def list_encounters(self, limit: int = 50) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            """SELECT e.*,
                      (SELECT status FROM records r WHERE r.encounter_id = e.id
                       ORDER BY r.updated_at DESC LIMIT 1) AS record_status
               FROM encounters e ORDER BY e.created_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_encounter(self, encounter_id: int) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM encounters WHERE id = ?", (encounter_id,)
        ).fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------- records
    def save_record(
        self,
        encounter_id: int,
        fields: Dict[str, str],
        raw_response: str = "",
        model: str = "",
        status: str = "draft",
        record_id: Optional[int] = None,
    ) -> int:
        """Insert a new record, or update an existing one (by record_id)."""
        fields_json = json.dumps(
            {k: fields.get(k, "") for k in NOTE_FIELDS}, ensure_ascii=False
        )
        now = _now()
        with self._lock, self._conn:
            if record_id is None:
                cur = self._conn.execute(
                    """INSERT INTO records
                       (encounter_id, fields_json, raw_response, model, status,
                        created_at, updated_at)
                       VALUES (?,?,?,?,?,?,?)""",
                    (encounter_id, fields_json, raw_response, model, status, now, now),
                )
                return cur.lastrowid
            self._conn.execute(
                """UPDATE records
                   SET fields_json = ?, status = ?, updated_at = ?
                   WHERE id = ?""",
                (fields_json, status, now, record_id),
            )
            return record_id

    def verify_record(self, record_id: int, verified_by: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """UPDATE records
                   SET status = 'verified', verified_by = ?, updated_at = ?
                   WHERE id = ?""",
                (verified_by, _now(), record_id),
            )

    def get_record_for_encounter(self, encounter_id: int) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            """SELECT * FROM records WHERE encounter_id = ?
               ORDER BY updated_at DESC LIMIT 1""",
            (encounter_id,),
        ).fetchone()
        if not row:
            return None
        rec = dict(row)
        rec["fields"] = json.loads(rec.pop("fields_json"))
        return rec

    def close(self) -> None:
        self._conn.close()
