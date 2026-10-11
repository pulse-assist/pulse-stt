"""Данные плагина в SQLite (PULSE_PLUGIN_DATA/stt.sqlite): история распознаваний (только текст, N дней), расход по
месяцам и задания долгих расшифровок. Сами записи не хранятся."""

import json
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

PREVIEW_CHARS = 500


def now() -> datetime:
    return datetime.now(timezone.utc)


class Store:
    def __init__(self, data_dir: Path, history_days: int = 30):
        data_dir.mkdir(parents=True, exist_ok=True)
        self.history_days = history_days
        self._lock = threading.Lock()
        self._db = sqlite3.connect(data_dir / "stt.sqlite", check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock, self._db:
            self._db.executescript("""
                CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY, at TEXT, source TEXT, name TEXT,
                    duration REAL, engine TEXT, model TEXT, text TEXT, chars INTEGER, note TEXT, error TEXT);
                CREATE TABLE IF NOT EXISTS usage (month TEXT, engine TEXT, seconds REAL, requests INTEGER,
                    cost REAL, currency TEXT, PRIMARY KEY (month, engine));
                CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, created REAL, status TEXT, progress TEXT,
                    result TEXT, error TEXT);
            """)

    # --- история ---------------------------------------------------------------------------------------------

    def add_history(self, *, source: str, name: str | None, duration: float | None, engine: str | None = None,
                    model: str | None = None, text: str = "", note: str = "", error: str = "") -> None:
        cutoff = (now() - timedelta(days=self.history_days)).isoformat()
        with self._lock, self._db:
            self._db.execute("INSERT INTO history (at, source, name, duration, engine, model, text, chars, note, error) "
                             "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                             (now().isoformat(), source, name, duration, engine, model, text[:PREVIEW_CHARS],
                              len(text), note, error))
            self._db.execute("DELETE FROM history WHERE at < ?", (cutoff,))

    def history(self, limit: int = 20) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM history ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    # --- расход ----------------------------------------------------------------------------------------------

    def add_usage(self, engine: str, seconds: float, cost: tuple[float, str] | None) -> None:
        month = now().strftime("%Y-%m")
        amount, currency = cost if cost else (0.0, "")
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO usage (month, engine, seconds, requests, cost, currency) VALUES (?, ?, ?, 1, ?, ?) "
                "ON CONFLICT (month, engine) DO UPDATE SET seconds = seconds + excluded.seconds, "
                "requests = requests + 1, cost = cost + excluded.cost, currency = excluded.currency",
                (month, engine, seconds, amount, currency))

    def usage(self, month: str | None = None) -> list[dict]:
        month = month or now().strftime("%Y-%m")
        with self._lock:
            rows = self._db.execute("SELECT * FROM usage WHERE month = ? ORDER BY seconds DESC", (month,)).fetchall()
        return [dict(r) for r in rows]

    # --- задания ---------------------------------------------------------------------------------------------

    def new_job(self) -> str:
        job = uuid.uuid4().hex[:10]
        with self._lock, self._db:
            self._db.execute("INSERT INTO jobs (id, created, status, progress) VALUES (?, ?, 'running', '')",
                             (job, time.time()))
            self._db.execute("DELETE FROM jobs WHERE created < ?", (time.time() - 7 * 86400,))
        return job

    def update_job(self, job: str, *, status: str | None = None, progress: str | None = None,
                   result: dict | None = None, error: str | None = None) -> None:
        sets, values = [], []
        for column, value in (("status", status), ("progress", progress), ("error", error),
                              ("result", json.dumps(result, ensure_ascii=False) if result is not None else None)):
            if value is not None:
                sets.append(f"{column} = ?")
                values.append(value)
        if sets:
            with self._lock, self._db:
                self._db.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id = ?", (*values, job))  # noqa: S608

    def job(self, job: str) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM jobs WHERE id = ?", (job,)).fetchone()
        if not row:
            return None
        out = dict(row)
        out["result"] = json.loads(out["result"]) if out["result"] else None
        return out

    def interrupt_running(self) -> None:
        """Процесс перезапустился — незавершённые задания не доделаются."""
        with self._lock, self._db:
            self._db.execute("UPDATE jobs SET status = 'error', error = 'плагин перезапустился — запустите расшифровку "
                             "снова' WHERE status = 'running'")
